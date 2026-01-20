"""
This module implements an Adaptive Extended Kalman Filter (AEKF) for 3D Pen
Trajectory Reconstruction.

The AEKF estimates the full 16-dimensional IMU state—comprising position,
velocity, orientation, and sensor biases—by integrating gyroscope and
accelerometer measurements. It features adaptive noise modeling and Zero-Velocity
Update (ZUPT) capabilities to enhance accuracy and robustness in dynamic
scenarios.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rotation_utils import quaternion_multiply, quaternion_to_rotation_matrix
from zupt_detector import ZuptDetector
from config import Config


class ExtendedKalmanFilter(nn.Module):
    """Implements a batch-aware, adaptive Extended Kalman Filter (AEKF) for IMU
    state estimation.

    The filter estimates a 16-dimensional state vector, defined as:
    x = [p_w (3), v_w (3), q_b→w (4), b_g (3), b_a (3)]
      - p_w: World frame position (m)
      - v_w: World frame velocity (m/s)
      - q_b→w: Body-to-world orientation quaternion
      - b_g: Gyroscope bias in body frame (rad/s)
      - b_a: Accelerometer bias in body frame (m/s^2)

    The 6-dimensional measurement vector consists of:
    z = [accel (3), gyro (3)]
      - accel: Accelerometer readings (m/s^2)
      - gyro: Gyroscope readings (rad/s)
    """

    def __init__(
        self,
        state_dim: int = 16,
        obs_dim: int = 6,
        dt: float = Config.DT,
        device: str = "cpu",
        zupt_window_size: int = Config.ZUPT_WINDOW_SIZE,
        zupt_accel_threshold: float = Config.ZUPT_ACCEL_THRESHOLD,
        zupt_force_var_threshold: float = Config.ZUPT_FORCE_VAR_THRESHOLD,
        zupt_force_delta_threshold: float = Config.ZUPT_FORCE_DELTA_THRESHOLD,
        use_zupt: bool = True,
    ):
        """Initializes the Extended Kalman Filter (AEKF) module.

        Args:
            state_dim: Dimension of the state vector. Fixed at 16 for this
                specific state definition.
            obs_dim: Dimension of the observation vector. Fixed at 6 for
                3-axis accelerometer and 3-axis gyroscope measurements.
            dt: Time step (delta time) in seconds between successive IMU
                measurements. Crucial for integration.
            device: The computation device ('cpu', 'cuda', 'mps').
            zupt_window_size: The size of the sliding window used by the ZUPT
                detector to analyze IMU data for periods of zero velocity.
            zupt_accel_threshold: Accelerometer variance threshold for ZUPT
                detection. Below this, the acceleration is considered static.
            zupt_force_var_threshold: Force sensor variance threshold for ZUPT.
            zupt_force_delta_threshold: Force sensor delta threshold for ZUPT.
            use_zupt: Boolean flag to enable or disable Zero-Velocity Update (ZUPT).
        """
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device
        self.use_zupt = use_zupt

        # Allan Variance parameters (sensor characterization)
        arw_x, arw_y, arw_z = Config.ARW_X, Config.ARW_Y, Config.ARW_Z
        gyro_bi_x, gyro_bi_y, gyro_bi_z = Config.GYRO_BI_X, Config.GYRO_BI_Y, Config.GYRO_BI_Z
        vrw_x, vrw_y, vrw_z = Config.VRW_X, Config.VRW_Y, Config.VRW_Z
        accel_bi_x, accel_bi_y, accel_bi_z = Config.ACCEL_BI_X, Config.ACCEL_BI_Y, Config.ACCEL_BI_Z

        # Q diagonal: [Pos(3), Vel(3), Quat(4), GyBias(3), AcBias(3)]
        Q_diag_tensor = torch.zeros(state_dim, device=device)
        Q_diag_tensor[0:3] = 0.0  # Pos: driven by vel
        Q_diag_tensor[3:6] = torch.tensor([vrw_x**2, vrw_y**2, vrw_z**2], device=device)  # Vel: accel VRW
        avg_arw_sq = (arw_x**2 + arw_y**2 + arw_z**2) / 3.0
        Q_diag_tensor[6:10] = avg_arw_sq  # Quat: gyro ARW (averaged)
        Q_diag_tensor[10:13] = torch.tensor([gyro_bi_x**2, gyro_bi_y**2, gyro_bi_z**2], device=device)  # Gyro bias: BI
        Q_diag_tensor[13:16] = torch.tensor([accel_bi_x**2, accel_bi_y**2, accel_bi_z**2], device=device)  # Accel bias: BI
        self.register_buffer("Q_diag", Q_diag_tensor)

        # R diagonal (learnable): [Accel(3), Gyro(3)]
        R_diag_tensor = torch.ones(obs_dim, device=device)
        R_diag_tensor[0:3] = torch.tensor([vrw_x**2, vrw_y**2, vrw_z**2])  # Accel: VRW
        R_diag_tensor[3:6] = torch.tensor([arw_x**2, arw_y**2, arw_z**2])  # Gyro: ARW
        self.raw_R_diag = nn.Parameter(R_diag_tensor)

        self.zupt_noise_std = nn.Parameter(
            torch.tensor(Config.AEKFTCN.ZUPT_NOISE_STD_AEKF, device=device)
        )

        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, -Config.GRAVITY_MAGNITUDE], device=device))

        self.zupt_detector = ZuptDetector(
            window_size=Config.ZUPT_WINDOW_SIZE,
            accel_var_threshold=Config.ZUPT_ACCEL_THRESHOLD,
            force_var_threshold=Config.ZUPT_FORCE_VAR_THRESHOLD,
            force_delta_threshold=Config.ZUPT_FORCE_DELTA_THRESHOLD,
            device=device,
        )

        self.adaptive_R_factor = nn.Parameter(torch.tensor(Config.AEKFTCN.ADAPTIVE_R_FACTOR_AEKF, device=device))
        self.zupt_R_factor = Config.AEKFTCN.ZUPT_R_FACTOR_AEKF

    def _transform_body_to_world(
        self,
        vector_body: torch.Tensor,
        quat_body_to_world: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms a batch of vectors from the body frame to the world frame."""
        rot_mat_body_to_world = quaternion_to_rotation_matrix(quat_body_to_world)
        return (rot_mat_body_to_world @ vector_body.unsqueeze(-1)).squeeze(-1)

    def _state_transition_function(
        self,
        state: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the next state based on the current state and IMU inputs (f(x, u)).

        This function implements the discrete-time non-linear state transition model.

        Args:
            state: The current 16-dimensional state vector.
            gyro_body_raw: Raw gyroscope measurements in the body frame.
            accel_body_raw: Raw accelerometer measurements in the body frame.

        Returns:
            The predicted 16-dimensional state vector for the next time step.
        """
        # Decompose the state vector.
        pos_world, vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split([3, 3, 4, 3, 3], dim=-1)

        # Correct raw IMU measurements.
        gyro_body_corrected = gyro_body_raw - gyro_bias_body
        accel_body_corrected = accel_body_raw - accel_bias_body

        # Propagate orientation (quaternion).
        q_dot = 0.5 * quaternion_multiply(
            quat_body_to_world,
            torch.cat([torch.zeros_like(gyro_body_corrected[..., :1]), gyro_body_corrected], dim=-1),
        )
        quat_body_to_world_new = F.normalize(quat_body_to_world + q_dot * self.dt, p=2, dim=-1)

        # Transform acceleration to world frame and remove gravity.
        accel_world = self._transform_body_to_world(accel_body_corrected, quat_body_to_world) - self.gravity_w

        # Propagate position and velocity.
        pos_world_new = pos_world + vel_world * self.dt + 0.5 * accel_world * (self.dt**2)
        vel_world_new = vel_world + accel_world * self.dt

        # Biases are modeled as random walks (constant in deterministic prediction).
        gyro_bias_body_new = gyro_bias_body
        accel_bias_body_new = accel_bias_body

        # Re-compose the predicted state vector.
        return torch.cat([pos_world_new, vel_world_new, quat_body_to_world_new, gyro_bias_body_new, accel_bias_body_new], dim=-1)

    def _measurement_function(self, state: torch.Tensor) -> torch.Tensor:
        """Predicts the expected sensor measurement for a given state (h(x)).

        Args:
            state: The current 16-dimensional state vector.

        Returns:
            The predicted 6-dimensional measurement vector [accel, gyro].
        """
        # Decompose the state vector.
        _pos_world, _vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split([3, 3, 4, 3, 3], dim=-1)

        # Predicted Gyroscope Measurement is simply the bias.
        gyro_pred = gyro_bias_body

        # Predicted Accelerometer Measurement is gravity rotated into the body
        # frame plus the accelerometer bias.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_body_to_world).transpose(-2, -1)
        accel_pred = (rot_mat_world_to_body @ -self.gravity_w.unsqueeze(0).T).squeeze(-1) + accel_bias_body

        # Re-compose the predicted measurement vector.
        return torch.cat([accel_pred, gyro_pred], dim=-1)

    def _compute_jacobian_F(
        self,
        state: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the Jacobian of the state transition function (F = ∂f/∂x).

        This matrix linearizes the non-linear state transition function around
        the current state estimate.

        Args:
            state: The current 16-dimensional state vector.
            gyro_body_raw: Raw gyroscope measurements.
            accel_body_raw: Raw accelerometer measurements.

        Returns:
            The 16x16 state transition Jacobian matrix.
        """
        batch_size = state.shape[0]
        F_matrix = torch.eye(self.state_dim, device=state.device, dtype=state.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        # Decompose state components for Jacobian calculation.
        _pos_world, _vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split([3, 3, 4, 3, 3], dim=-1)
        q_w, q_x, q_y, q_z = quat_body_to_world.unbind(-1)
        accel_corrected_body = accel_body_raw - accel_bias_body
        ax, ay, az = accel_corrected_body.unbind(-1)

        # F(0:3, 3:6) = I * dt (position depends on velocity)
        F_matrix[:, 0:3, 3:6] = torch.eye(3, device=F_matrix.device, dtype=F_matrix.dtype) * self.dt

        # Jacobian of world acceleration w.r.t. quaternion (J_aq).
        J_aq = torch.zeros(batch_size, 3, 4, device=F_matrix.device, dtype=F_matrix.dtype)
        J_aq[:, 0, 0] = -2 * q_z * ay + 2 * q_y * az
        J_aq[:, 1, 0] = 2 * q_z * ax - 2 * q_x * az
        J_aq[:, 2, 0] = -2 * q_y * ax + 2 * q_x * ay
        J_aq[:, 0, 1] = 2 * q_y * ay + 2 * q_z * az
        J_aq[:, 1, 1] = 2 * q_y * ax - 4 * q_x * ay - 2 * q_w * az
        J_aq[:, 2, 1] = 2 * q_z * ax + 2 * q_w * ay - 4 * q_x * az
        J_aq[:, 0, 2] = -4 * q_y * ax + 2 * q_x * ay + 2 * q_w * az
        J_aq[:, 1, 2] = 2 * q_x * ax + 2 * q_z * az
        J_aq[:, 2, 2] = -2 * q_w * ax + 2 * q_z * ay - 4 * q_y * az
        J_aq[:, 0, 3] = -4 * q_z * ax - 2 * q_w * ay + 2 * q_x * az
        J_aq[:, 1, 3] = 2 * q_w * ax - 4 * q_z * ay + 2 * q_y * az
        J_aq[:, 2, 3] = 2 * q_x * ax + 2 * q_y * ay

        F_matrix[:, 0:3, 6:10] = J_aq * 0.5 * (self.dt**2)
        F_matrix[:, 3:6, 6:10] = J_aq * self.dt

        # Jacobian of world acceleration w.r.t. accelerometer bias.
        rot_mat_body_to_world = quaternion_to_rotation_matrix(quat_body_to_world)
        F_matrix[:, 0:3, 13:16] = -rot_mat_body_to_world * 0.5 * (self.dt**2)
        F_matrix[:, 3:6, 13:16] = -rot_mat_body_to_world * self.dt

        # Jacobian of quaternion w.r.t. quaternion and gyro bias.
        omega_corrected_body = gyro_body_raw - gyro_bias_body
        Omega = torch.zeros(batch_size, 4, 4, device=F_matrix.device, dtype=F_matrix.dtype)
        Omega[:, 0, 1:4] = -omega_corrected_body
        Omega[:, 1:4, 0] = omega_corrected_body
        Omega[:, 1, 2], Omega[:, 1, 3] = omega_corrected_body[:, 2], -omega_corrected_body[:, 1]
        Omega[:, 2, 1], Omega[:, 2, 3] = -omega_corrected_body[:, 2], omega_corrected_body[:, 0]
        Omega[:, 3, 1], Omega[:, 3, 2] = omega_corrected_body[:, 1], -omega_corrected_body[:, 0]
        F_matrix[:, 6:10, 6:10] = torch.eye(4, device=F_matrix.device, dtype=F_matrix.dtype) + 0.5 * self.dt * Omega

        Q_deriv_mat = torch.stack([-q_x, -q_y, -q_z, q_w, -q_z, q_y, q_z, q_w, -q_x, -q_y, q_x, q_w,], -1,).view(batch_size, 4, 3)
        F_matrix[:, 6:10, 10:13] = -0.5 * self.dt * Q_deriv_mat

        return F_matrix

    def _compute_jacobian_H(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the Jacobian of the measurement function (H = ∂h/∂x).

        This matrix linearizes the non-linear measurement function around the
        current state estimate.

        Args:
            state: The current 16-dimensional state vector.

        Returns:
            The 6x16 observation Jacobian matrix.
        """
        batch_size = state.shape[0]
        H_matrix = torch.zeros(batch_size, self.obs_dim, self.state_dim, device=state.device, dtype=state.dtype)

        # Decompose state components.
        quat_body_to_world = state[..., 6:10]
        q_w, q_x, q_y, q_z = quat_body_to_world.unbind(-1)
        g_magnitude = self.gravity_w[2]

        # Jacobian of predicted accel measurement w.r.t. quaternion (J_H_q).
        # This describes how orientation errors affect the predicted gravity vector.
        J_H_q = torch.zeros(batch_size, 3, 4, device=H_matrix.device, dtype=H_matrix.dtype)
        # Derivatives of a_pred = -R_wb * g_w w.r.t. q
        J_H_q[:, 0, 0] = -2 * g_magnitude * q_y
        J_H_q[:, 0, 1] = -2 * g_magnitude * q_z
        J_H_q[:, 0, 2] = -2 * g_magnitude * q_w
        J_H_q[:, 0, 3] = -2 * g_magnitude * q_x
        J_H_q[:, 1, 0] = 2 * g_magnitude * q_x
        J_H_q[:, 1, 1] = 2 * g_magnitude * q_w
        J_H_q[:, 1, 2] = -2 * g_magnitude * q_z
        J_H_q[:, 1, 3] = -2 * g_magnitude * q_y
        J_H_q[:, 2, 1] = 4 * g_magnitude * q_x
        J_H_q[:, 2, 2] = 4 * g_magnitude * q_y
        H_matrix[..., 0:3, 6:10] = J_H_q

        # Jacobian w.r.t. biases.
        H_matrix[..., 0:3, 13:16] = torch.eye(3, device=H_matrix.device, dtype=H_matrix.dtype) # Accel bias
        H_matrix[..., 3:6, 10:13] = torch.eye(3, device=H_matrix.device, dtype=H_matrix.dtype) # Gyro bias

        return H_matrix

    def predict(
        self,
        state: torch.Tensor,
        P_covariance: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the EKF prediction step.

        Args:
            state: Current state vector (x_k-1|k-1).
            P_covariance: Current error covariance matrix (P_k-1|k-1).
            gyro_body_raw: Raw gyroscope measurements.
            accel_body_raw: Raw accelerometer measurements.

        Returns:
            A tuple containing the predicted state and covariance.
        """
        # Predict next state using the non-linear state transition function.
        state_predicted = self._state_transition_function(state, gyro_body_raw, accel_body_raw)

        # Linearize the state transition function to get the Jacobian F.
        F_jacobian = self._compute_jacobian_F(state, gyro_body_raw, accel_body_raw)

        # Get the process noise covariance matrix Q.
        Q_matrix = torch.diag_embed(self.Q_diag * self.dt)

        # Predict the error covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        P_predicted = F_jacobian @ P_covariance @ F_jacobian.transpose(-2, -1) + Q_matrix

        return state_predicted, P_predicted

    def update(
        self,
        state_predicted: torch.Tensor,
        P_predicted: torch.Tensor,
        measurement: torch.Tensor,
        accel_body_raw: torch.Tensor,
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the EKF update step (measurement update).

        Args:
            state_predicted: Predicted state vector (x_k|k-1).
            P_predicted: Predicted error covariance matrix (P_k|k-1).
            measurement: Actual sensor measurement (z_k).
            accel_body_raw: Raw accelerometer data for adaptive R calculation.
            tcn_output: Optional dictionary from TCN for adaptive R.

        Returns:
            A tuple of the updated state, covariance, and the innovation.
        """
        # 1. Predicted Measurement: h(x_k|k-1)
        h_predicted = self._measurement_function(state_predicted)

        # 2. Innovation: y_k = z_k - h(x_k|k-1)
        innovation = measurement - h_predicted

        # 3. Observation Jacobian: H_k
        H_jacobian = self._compute_jacobian_H(state_predicted)

        # 4. Adaptive Measurement Noise Covariance: R_k
        if tcn_output is not None and "covariance_R" in tcn_output:
            # Use TCN-predicted noise characteristics.
            R_adaptive = torch.diag_embed(F.softplus(tcn_output["covariance_R"]) + 1e-6)
        else:
            # Adapt R based on accelerometer magnitude deviation from gravity.
            accel_norm_diff = torch.abs(torch.norm(accel_body_raw, dim=-1, keepdim=True) - torch.norm(self.gravity_w))
            scaling_factor = torch.exp(self.adaptive_R_factor * accel_norm_diff)
            R_adaptive = torch.diag_embed(F.softplus(self.raw_R_diag) + 1e-6) * scaling_factor.unsqueeze(-1)

        # 5. Innovation Covariance: S_k = H_k * P_k|k-1 * H_k^T + R_k
        S_matrix = H_jacobian @ P_predicted @ H_jacobian.transpose(-2, -1) + R_adaptive

        # 6. Kalman Gain: K_k = P_k|k-1 * H_k^T * S_k^-1
        K_gain = torch.linalg.solve(S_matrix, H_jacobian @ P_predicted.transpose(-2, -1)).transpose(-2, -1)

        # 7. State Update: x_k|k = x_k|k-1 + K_k * y_k
        state_updated = state_predicted + (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        # The quaternion must be re-normalized after the additive update.
        quat_body_to_world_normalized = F.normalize(state_updated[..., 6:10], p=2, dim=-1)
        state_updated = torch.cat([state_updated[..., :6], quat_body_to_world_normalized, state_updated[..., 10:]], dim=-1)

        # 8. Covariance Update (Joseph Form for numerical stability)
        I_matrix = torch.eye(self.state_dim, device=state_predicted.device, dtype=state_predicted.dtype)
        P_updated = (I_matrix - K_gain @ H_jacobian) @ P_predicted @ (I_matrix - K_gain @ H_jacobian).transpose(-2, -1) + K_gain @ R_adaptive @ K_gain.transpose(-2, -1)

        return state_updated, P_updated, innovation

    def _calculate_zupt_update(
        self,
        state: torch.Tensor,
        P_covariance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the state and covariance correction from a Zero-Velocity Update (ZUPT).

        Args:
            state: The current 16-dimensional state vector (x_k|k).
            P_covariance: The current 16x16 error covariance matrix (P_k|k).

        Returns:
            A tuple of the state and covariance after ZUPT correction.
        """
        batch_size = state.shape[0]
        # H_ZUPT: Measurement Jacobian for ZUPT (selects velocity components).
        H_zupt = torch.zeros(batch_size, 3, self.state_dim, device=state.device, dtype=state.dtype)
        H_zupt[:, :, 3:6] = torch.eye(3, device=state.device, dtype=state.dtype)

        # R_ZUPT: ZUPT measurement noise (high confidence in zero velocity).
        R_zupt = torch.diag(self.zupt_noise_std ** 2).unsqueeze(0).expand(batch_size, -1, -1)

        # ZUPT Innovation: y = 0 - v_predicted
        innovation_zupt = -state[..., 3:6]

        # Standard EKF update equations for the ZUPT pseudo-measurement.
        S_zupt = H_zupt @ P_covariance @ H_zupt.transpose(-2, -1) + R_zupt
        K_zupt_gain = torch.linalg.solve(S_zupt, H_zupt @ P_covariance.transpose(-2, -1)).transpose(-2, -1)
        state_updated = state + (K_zupt_gain @ innovation_zupt.unsqueeze(-1)).squeeze(-1)

        # Re-normalize quaternion.
        quat_body_to_world_normalized = F.normalize(state_updated[..., 6:10], p=2, dim=-1)
        state_updated = torch.cat([state_updated[..., :6], quat_body_to_world_normalized, state_updated[..., 10:]], dim=-1)

        # Update covariance using Joseph Form.
        I_matrix = torch.eye(self.state_dim, device=state.device, dtype=state.dtype)
        P_updated = (I_matrix - K_zupt_gain @ H_zupt) @ P_covariance @ (I_matrix - K_zupt_gain @ H_zupt).transpose(-2, -1) + K_zupt_gain @ R_zupt @ K_zupt_gain.transpose(-2, -1)

        return state_updated, P_updated

    def forward(
        self,
        state: torch.Tensor,
        P_covariance: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
        force_raw: torch.Tensor,
        measurement: torch.Tensor,
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Executes one full predict-update cycle of the EKF.

        Args:
            state (torch.Tensor): Current state vector.
                - Shape: (Batch, 16) | [pos(3), vel(3), quat(4), bg(3), ba(3)]
            P_covariance (torch.Tensor): Current error covariance matrix.
                - Shape: (Batch, 16, 16)
            gyro_body_raw (torch.Tensor): Raw gyroscope measurements.
                - Shape: (Batch, 3) | Unit: rad/s | Frame: Body
            accel_body_raw (torch.Tensor): Raw accelerometer measurements.
                - Shape: (Batch, 3) | Unit: m/s^2 | Frame: Body
            force_raw (torch.Tensor): Raw force sensor measurement.
                - Shape: (Batch, 1) | Unit: N
            measurement (torch.Tensor): The 6D sensor measurement vector.
                - Shape: (Batch, 6) | [accel, gyro]
            tcn_output (Optional[Dict[str, torch.Tensor]]): Optional dictionary from TCN.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                - final_state: Updated state vector.
                    - Shape: (Batch, 16)
                - final_P: Updated covariance matrix.
                    - Shape: (Batch, 16, 16)
                - tcn_features: Dict containing "body_velocity" (Batch, 3), "innovation" (Batch, 6), "zupt_flag" (Batch, 1)
        """
        # 1. Prediction Step
        state_predicted, P_predicted = self.predict(state, P_covariance, gyro_body_raw, accel_body_raw)

        # 2. Measurement Update Step
        state_updated, P_updated, innovation = self.update(state_predicted, P_predicted, measurement, accel_body_raw, tcn_output)

        final_state, final_P, final_innovation = state_updated, P_updated, innovation

        # 3. ZUPT Correction Step
        if self.use_zupt:
            is_zupt = self.zupt_detector(accel_body_raw, force_raw)
            if torch.any(is_zupt):
                # Apply ZUPT correction only to samples identified as static.
                state_zupt_corr, P_zupt_corr = self._calculate_zupt_update(state_updated[is_zupt], P_updated[is_zupt])
                # Use clone to avoid in-place modification issues.
                final_state, final_P, final_innovation = state_updated.clone(), P_updated.clone(), innovation.clone()
                final_state[is_zupt] = state_zupt_corr
                final_P[is_zupt] = P_zupt_corr
                final_innovation[is_zupt] = 0 # Innovation is effectively zeroed by ZUPT.
        else:
            is_zupt = torch.zeros(accel_body_raw.shape[0], dtype=torch.bool, device=self.device)


        # --- Assemble Features for TCN ---
        vel_world = final_state[..., 3:6]
        quat_body_to_world = final_state[..., 6:10]
        # Rotate world velocity to body frame for TCN features.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(
            quat_body_to_world
        ).transpose(-2, -1)
        vel_body = (rot_mat_world_to_body @ vel_world.unsqueeze(-1)).squeeze(-1)

        tcn_features: Dict[str, torch.Tensor] = {
            "body_velocity": vel_body,
            "innovation": final_innovation,
            "zupt_flag": is_zupt.float().unsqueeze(-1),
        }

        return final_state, final_P, tcn_features


if __name__ == "__main__":
    # This test case verifies the functionality and tensor shapes of the ExtendedKalmanFilter.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 4
    dt_val = Config.DT

    # Initialize the EKF.
    ekf = ExtendedKalmanFilter(dt=dt_val, device=device)

    # Initial state: zero position/velocity, identity quaternion, zero biases.
    state_initial = torch.zeros(batch_size, 16, device=device)
    state_initial[:, 6] = 1.0
    # Initial covariance with some uncertainty.
    P_initial = torch.eye(16, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

    # Create dummy IMU data for one time step.
    accel_dummy = torch.randn(batch_size, 3, device=device) * 0.1
    accel_dummy[0, 2] += Config.GRAVITY_MAGNITUDE  # Simulate near-static condition for ZUPT test
    gyro_dummy = torch.randn(batch_size, 3, device=device) * 0.01
    force_dummy = torch.rand(batch_size, 1, device=device)
    measurement_dummy = torch.cat([accel_dummy, gyro_dummy], dim=-1)

    # Run one forward pass.
    final_state, final_P_covariance, tcn_features = ekf.forward(
        state_initial, P_initial, gyro_dummy, accel_dummy, force_dummy, measurement_dummy
    )

    print(f"\nFinal State shape: {final_state.shape}")
    print(f"Final Covariance shape: {final_P_covariance.shape}")
    print("TCN Feature shapes:")
    for k, v in tcn_features.items():
        print(f"  - '{k}': {v.shape}")
    print("\nBatch-aware AEKF tested successfully.")