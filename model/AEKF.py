"""
This module implements an Adaptive Extended Kalman Filter (AEKF) for 3D Pen
Trajectory Reconstruction.

The AEKF is designed to estimate the full 16-dimensional IMU state (position,
velocity, orientation, and sensor biases) by integrating gyroscope and
accelerometer measurements. It incorporates adaptive noise modeling and
Zero-Velocity Update (ZUPT) capabilities to enhance accuracy and robustness
in real-world scenarios.
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


class ExtendedKalmanFilter(nn.Module):
    """Implements a batch-aware, adaptive Extended Kalman Filter (AEKF) for IMU
    state estimation.

    The filter estimates a 16-dimensional state vector (x), defined as:
    x = [p_x, p_y, p_z,        (3) - World frame position (m)
         v_x, v_y, v_z,        (3) - World frame velocity (m/s)
         q_w, q_x, q_y, q_z,    (4) - Body-to-world orientation quaternion
         bg_x, bg_y, bg_z,     (3) - Gyroscope bias in body frame (rad/s)
         ba_x, ba_y, ba_z]     (3) - Accelerometer bias in body frame (m/s^2)

    The measurement vector (z) is 6-dimensional, consisting of:
    z = [accel_x, accel_y, accel_z, (3) - Accelerometer readings (m/s^2)
         gyro_x, gyro_y, gyro_z]    (3) - Gyroscope readings (rad/s)
    """

    def __init__(
        self,
        state_dim: int = 16,
        obs_dim: int = 6,  # 3 accel + 3 gyro
        dt: float = 0.0025,
        device: str = "cpu",
        zupt_window_size: int = 20,
        zupt_accel_threshold: float = 0.1,
        zupt_force_var_threshold: float = 0.01,
        zupt_force_delta_threshold: float = 0.1,
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

        # --- Learnable Noise Parameters ---
        # These are registered as nn.Parameter to allow them to be optimized
        # during training, making the EKF adaptive.
        # Q_diag: Diagonal elements of the process noise covariance matrix Q.
        # It's initialized to a small value representing initial uncertainty
        # in the state propagation.
        self.Q_diag = nn.Parameter(torch.ones(state_dim, device=device) * 1e-4)
        # gyro_bias_rw_std: Standard deviation for gyroscope bias random walk,
        # modeling its slow drift over time as part of the process noise.
        self.gyro_bias_rw_std = nn.Parameter(torch.tensor(1e-5, device=device))
        # accel_bias_rw_std: Standard deviation for accelerometer bias random walk.
        self.accel_bias_rw_std = nn.Parameter(torch.tensor(1e-5, device=device))
        # raw_R_diag: Diagonal elements of the measurement noise covariance matrix R.
        # Represents the initial uncertainty in the sensor measurements.
        self.raw_R_diag = nn.Parameter(torch.ones(obs_dim, device=device) * 1e-2)

        # --- Physical Constants ---
        # Gravity vector in the world frame. Assuming Z-axis points upwards.
        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, 9.81], device=device))

        # --- ZUPT Detector ---
        # Component responsible for identifying zero-velocity intervals.
        self.zupt_detector = ZuptDetector(
            window_size=zupt_window_size,
            accel_var_threshold=zupt_accel_threshold,
            force_var_threshold=zupt_force_var_threshold,
            force_delta_threshold=zupt_force_delta_threshold,
            device=device,
        )

        # --- Tuning Factors ---
        # These factors scale the measurement noise covariance (R) to make the
        # filter adaptive to different motion conditions.
        # When `accel_norm_diff` is large, it implies motion, so `R` is increased
        # to put less trust in the measurement.
        self.adaptive_R_factor = 0.1
        # `zupt_R_factor` is a very small value, indicating high confidence in the
        # zero-velocity measurement when a ZUPT is detected.
        self.zupt_R_factor = 1e-6
        # `zupt_gravity_R_factor` can be used to specifically reduce measurement
        # uncertainty on the gravity component during ZUPT, but is not used in this
        # current implementation.

    def _transform_body_to_world(
        self, vector_body: torch.Tensor, quat_body_to_world: torch.Tensor
    ) -> torch.Tensor:
        """Transforms a batch of vectors from the body frame to the world frame.

        Args:
            vector_body: A batch of 3D vectors in the body frame (e.g., accelerometer reading).
            quat_body_to_world: A batch of quaternions representing orientation
                from body frame to world frame.

        Returns:
            A batch of 3D vectors transformed into the world frame.
        """
        # Convert quaternion to rotation matrix (body to world)
        rot_mat_body_to_world = quaternion_to_rotation_matrix(quat_body_to_world)
        # Apply rotation: world_vector = R_bw * body_vector
        return (rot_mat_body_to_world @ vector_body.unsqueeze(-1)).squeeze(-1)

    def _state_transition_function(
        self,
        state: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the next nominal state based on current nominal state and IMU inputs.

        This function implements the discrete-time non-linear state transition
        model f(x_k, u_k) for the EKF.

        Args:
            state: The current 16-dimensional nominal state vector.
            gyro_body_raw: Raw gyroscope measurements in the body frame.
            accel_body_raw: Raw accelerometer measurements in the body frame.

        Returns:
            The predicted 16-dimensional nominal state vector for the next time step.
        """
        # Decompose the state vector into its components
        pos_world, vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split(
            [3, 3, 4, 3, 3], dim=-1
        )

        # Correct raw IMU measurements by subtracting estimated biases.
        # This gives us a cleaner estimate of the true angular velocity and acceleration.
        gyro_body_corrected = gyro_body_raw - gyro_bias_body
        accel_body_corrected = accel_body_raw - accel_bias_body

        # Quaternion propagation (orientation update)
        # The quaternion derivative (q_dot) is related to angular velocity (omega).
        # q_dot = 0.5 * q * [0; omega] (quaternion multiplication)
        # Where [0; omega] is the angular velocity vector treated as a pure quaternion.
        q_dot = 0.5 * quaternion_multiply(
            quat_body_to_world,
            torch.cat(
                [torch.zeros_like(gyro_body_corrected[..., :1]), gyro_body_corrected],
                dim=-1,
            ),
        )
        # Integrate quaternion: q_new = q_old + q_dot * dt
        quat_body_to_world_new = F.normalize(
            quat_body_to_world + q_dot * self.dt, p=2, dim=-1
        )  # Normalize to maintain unit quaternion constraint

        # Accelerometer data transformation
        # Rotate the corrected body-frame acceleration to the world frame.
        # Then, subtract the gravity vector to get the *pure kinematic acceleration*
        # (i.e., acceleration due to motion, excluding gravity).
        accel_world = self._transform_body_to_world(
            accel_body_corrected, quat_body_to_world
        ) - self.gravity_w

        # Position and Velocity Integration (kinematic model)
        # Using Euler integration (first-order approximation) for simplicity.
        # p_new = p_old + v_old * dt + 0.5 * a_old * dt^2
        pos_world_new = pos_world + vel_world * self.dt + 0.5 * accel_world * (
            self.dt**2
        )
        # v_new = v_old + a_old * dt
        vel_world_new = vel_world + accel_world * self.dt

        # Gyroscope and Accelerometer biases are modeled as random walk,
        # meaning their nominal values remain constant between steps
        # for the state transition function (their uncertainty changes via Q).
        gyro_bias_body_new = gyro_bias_body
        accel_bias_body_new = accel_bias_body

        # Re-compose the predicted state vector
        return torch.cat(
            [
                pos_world_new,
                vel_world_new,
                quat_body_to_world_new,
                gyro_bias_body_new,
                accel_bias_body_new,
            ],
            dim=-1,
        )

    def _measurement_function(self, state: torch.Tensor) -> torch.Tensor:
        """Predicts the expected sensor measurement (accelerometer and gyroscope)
        for a given nominal state.

        This function implements the discrete-time non-linear observation
        model h(x_k) for the EKF.

        Args:
            state: The current 16-dimensional nominal state vector.

        Returns:
            The predicted 6-dimensional measurement vector
            [predicted_accel_x, predicted_accel_y, predicted_accel_z,
             predicted_gyro_x, predicted_gyro_y, predicted_gyro_z].
        """
        # Decompose the state vector
        _pos_world, _vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split(
            [3, 3, 4, 3, 3], dim=-1
        )

        # Predicted Gyroscope Measurement:
        # In this EKF formulation, the gyroscope measurement primarily helps
        # estimate its bias. If the true angular velocity is assumed zero in
        # the error state, then the expected raw gyro measurement would just be its bias.
        gyro_pred = gyro_bias_body

        # Predicted Accelerometer Measurement:
        # The accelerometer measures the specific force (non-gravitational acceleration).
        # When static, it measures the negative of gravity. So, to predict what the
        # accelerometer would read, we rotate the world gravity vector into the body
        # frame and add the accelerometer bias.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(
            quat_body_to_world
        ).transpose(-2, -1)  # R_wb = R_bw^T
        # Predicted specific force = -R_wb * g_w + accel_bias_body
        accel_pred = (
            rot_mat_world_to_body @ -self.gravity_w.unsqueeze(0).T
        ).squeeze(-1) + accel_bias_body
        # Note: The sign of gravity_w and the rotation applied here depend on the
        # specific sensor coordinate system and gravity vector convention.
        # Here, it aligns with a model where accel_pred = R_wb * (accel_world - g_w) + bias_a,
        # and if accel_world is zero (only gravity acting), then accel_pred = -R_wb * g_w + bias_a.

        # Re-compose the predicted measurement vector
        return torch.cat([accel_pred, gyro_pred], dim=-1)

    def _compute_jacobian_F(
        self,
        state: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the Jacobian of the state transition function (F = ∂f/∂x).

        This F matrix (state transition matrix for the error state) relates a
        small error in the current state to a small error in the next state.
        It is obtained by linearizing the _state_transition_function around the
        current nominal state.

        Args:
            state: The current 16-dimensional nominal state vector.
            gyro_body_raw: Raw gyroscope measurements.
            accel_body_raw: Raw accelerometer measurements.

        Returns:
            The 16x16 state transition Jacobian matrix for the error state.
        """
        batch_size = state.shape[0]
        # Initialize F as an identity matrix, which represents the assumption
        # that error states propagate directly if no dynamics are involved.
        F_matrix = (
            torch.eye(self.state_dim, device=state.device, dtype=state.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Decompose state components
        _pos_world, _vel_world, quat_body_to_world, gyro_bias_body, accel_bias_body = state.split(
            [3, 3, 4, 3, 3], dim=-1
        )
        q_w, q_x, q_y, q_z = quat_body_to_world.unbind(
            -1
        )  # Extract quaternion components for Jacobian calculation
        accel_corrected_body = accel_body_raw - accel_bias_body
        ax, ay, az = accel_corrected_body.unbind(-1)

        # F(0:3, 3:6) = I * dt (position depends on velocity)
        F_matrix[:, 0:3, 3:6] = torch.eye(3, device=F_matrix.device, dtype=F_matrix.dtype) * self.dt

        # F(0:3, 6:10) and F(3:6, 6:10) depend on the Jacobian of acceleration in world frame
        # with respect to quaternion (J_aq).
        # This part of the Jacobian describes how errors in orientation affect
        # the propagated position and velocity.
        J_aq = torch.zeros(batch_size, 3, 4, device=F_matrix.device, dtype=F_matrix.dtype)
        # Rows of J_aq correspond to x, y, z components of world acceleration.
        # Columns correspond to q_w, q_x, q_y, q_z.
        # These are partial derivatives of R_bw * accel_corrected_body w.r.t. quaternion components.
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

        F_matrix[:, 0:3, 6:10] = J_aq * 0.5 * (self.dt**2)  # Quaternion affects position
        F_matrix[:, 3:6, 6:10] = J_aq * self.dt  # Quaternion affects velocity

        # F(0:3, 13:16) and F(3:6, 13:16) depend on accelerometer bias
        # Errors in accelerometer bias directly affect the corrected acceleration,
        # which in turn affects position and velocity.
        rot_mat_body_to_world = quaternion_to_rotation_matrix(quat_body_to_world)
        F_matrix[:, 0:3, 13:16] = -rot_mat_body_to_world * 0.5 * (self.dt**2)
        F_matrix[:, 3:6, 13:16] = -rot_mat_body_to_world * self.dt

        # F(6:10, 6:10) and F(6:10, 10:13) depend on quaternion and gyroscope bias
        # This describes how errors in current quaternion and gyro bias affect
        # the propagated quaternion.
        omega_corrected_body = gyro_body_raw - gyro_bias_body
        # The Omega matrix for quaternion error propagation
        # (representation of angular velocity in quaternion space).
        Omega = torch.zeros(batch_size, 4, 4, device=F_matrix.device, dtype=F_matrix.dtype)
        # Skew-symmetric part of the Omega matrix
        Omega[:, 0, 1:4] = -omega_corrected_body
        Omega[:, 1, 0] = omega_corrected_body[:, 0]
        Omega[:, 2, 0] = omega_corrected_body[:, 1]
        Omega[:, 3, 0] = omega_corrected_body[:, 2]
        Omega[:, 1, 2] = omega_corrected_body[:, 2]
        Omega[:, 1, 3] = -omega_corrected_body[:, 1]
        Omega[:, 2, 1] = -omega_corrected_body[:, 2]
        Omega[:, 2, 3] = omega_corrected_body[:, 0]
        Omega[:, 3, 1] = omega_corrected_body[:, 1]
        Omega[:, 3, 2] = -omega_corrected_body[:, 0]
        # F_qq = I + 0.5 * Omega * dt (quaternion affects quaternion)
        F_matrix[:, 6:10, 6:10] = (
            torch.eye(4, device=F_matrix.device, dtype=F_matrix.dtype)
            + 0.5 * self.dt * Omega
        )

        # F_q_bg = -0.5 * Q_deriv_mat * dt (gyro bias affects quaternion)
        # Jacobian of quaternion derivative w.r.t. gyro bias.
        Q_deriv_mat = torch.stack(
            [
                -q_x,
                -q_y,
                -q_z,
                q_w,
                -q_z,
                q_y,
                q_z,
                q_w,
                -q_x,
                -q_y,
                q_x,
                q_w,
            ],
            -1,
        ).view(batch_size, 4, 3)
        F_matrix[:, 6:10, 10:13] = -0.5 * self.dt * Q_deriv_mat

        return F_matrix

    def _compute_jacobian_H(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the Jacobian of the measurement function (H = ∂h/∂x).

        This H matrix (observation matrix for the error state) relates a
        small error in the state to a small error in the predicted measurement.
        It is obtained by linearizing the _measurement_function around the
        current nominal state.

        Args:
            state: The current 16-dimensional nominal state vector.

        Returns:
            The 6x16 observation Jacobian matrix.
        """
        batch_size = state.shape[0]
        H_matrix = torch.zeros(
            batch_size, self.obs_dim, self.state_dim, device=state.device, dtype=state.dtype
        )

        # Decompose state components
        quat_body_to_world = state[..., 6:10]
        q_w, q_x, q_y, q_z = quat_body_to_world.unbind(-1)
        # Gravity magnitude in world Z-axis (from gravity_w buffer).
        g_magnitude = self.gravity_w[2]

        # H(0:3, 6:10) - Jacobian for predicted accelerometer measurement w.r.t. quaternion.
        # This describes how errors in orientation affect the predicted gravity
        # vector components in the body frame.
        J_H_q = torch.zeros(batch_size, 3, 4, device=H_matrix.device, dtype=H_matrix.dtype)
        # These are partial derivatives of -R_wb * g_w w.r.t. quaternion components.
        J_H_q[:, 0, 0] = -2 * g_magnitude * q_y
        J_H_q[:, 0, 1] = 2 * g_magnitude * q_z
        J_H_q[:, 0, 2] = -2 * g_magnitude * q_w
        J_H_q[:, 0, 3] = 2 * g_magnitude * q_x

        J_H_q[:, 1, 0] = 2 * g_magnitude * q_x
        J_H_q[:, 1, 1] = 2 * g_magnitude * q_w
        J_H_q[:, 1, 2] = 2 * g_magnitude * q_z
        J_H_q[:, 1, 3] = 2 * g_magnitude * q_y

        J_H_q[:, 2, 1] = -4 * g_magnitude * q_x
        J_H_q[:, 2, 2] = -4 * g_magnitude * q_y
        # J_H_q[:, 2, 0] and J_H_q[:, 2, 3] are zero for gravity along Z-axis.

        H_matrix[..., 0:3, 6:10] = J_H_q

        # H(0:3, 13:16) - Accelerometer bias directly affects predicted accelerometer measurement.
        H_matrix[..., 0:3, 13:16] = torch.eye(3, device=H_matrix.device, dtype=H_matrix.dtype)

        # H(3:6, 10:13) - Gyroscope bias directly affects predicted gyroscope measurement.
        H_matrix[..., 3:6, 10:13] = torch.eye(3, device=H_matrix.device, dtype=H_matrix.dtype)

        return H_matrix

    def predict(
        self,
        state: torch.Tensor,
        P_covariance: torch.Tensor,
        gyro_body_raw: torch.Tensor,
        accel_body_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the EKF prediction step (time update).

        This step propagates the state estimate and its covariance from time k-1 to k
        using the system's dynamic model.

        Args:
            state: The current 16-dimensional state vector (x_k-1).
            P_covariance: The current 16x16 error covariance matrix (P_k-1).
            gyro_body_raw: Raw gyroscope measurements at time k.
            accel_body_raw: Raw accelerometer measurements at time k.

        Returns:
            A tuple containing:
                - state_predicted (torch.Tensor): The predicted state vector (x_k|k-1).
                - P_predicted (torch.Tensor): The predicted error covariance matrix (P_k|k-1).
        """
        # 1. State Prediction: Propagate the nominal state using the non-linear function f.
        state_predicted = self._state_transition_function(
            state, gyro_body_raw, accel_body_raw
        )

        # 2. Covariance Prediction: P_k|k-1 = F * P_k-1 * F^T + Q
        # F: State transition Jacobian (F_k)
        F_jacobian = self._compute_jacobian_F(state, gyro_body_raw, accel_body_raw)

        # Q: Process noise covariance matrix. It models the uncertainty introduced
        # by the system model itself and any unmodeled disturbances (e.g., random walk).
        # We ensure Q_diag elements are positive using torch.abs().
        Q_diag = torch.abs(self.Q_diag)
        # Add random walk noise contributions for gyro and accel biases.
        # Bias random walk is typically proportional to dt.
        Q_diag[:, 10:13] += torch.square(self.gyro_bias_rw_std) * self.dt
        Q_diag[:, 13:16] += torch.square(self.accel_bias_rw_std) * self.dt
        Q_matrix = torch.diag_embed(Q_diag)  # Create a diagonal matrix from Q_diag

        # Standard covariance prediction formula
        P_predicted = (
            F_jacobian @ P_covariance @ F_jacobian.transpose(-2, -1) + Q_matrix
        )

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

        This step corrects the predicted state and covariance using the latest
        sensor measurement.

        Args:
            state_predicted: The predicted 16-dimensional state vector (x_k|k-1).
            P_predicted: The predicted 16x16 error covariance matrix (P_k|k-1).
            measurement: The actual 6-dimensional sensor measurement (z_k).
            accel_body_raw: Raw accelerometer measurements at time k, used for
                adaptive R calculation.
            tcn_output: Optional. Dictionary containing TCN predictions,
                potentially including 'covariance_R' for adaptive R.

        Returns:
            A tuple containing:
                - state_updated (torch.Tensor): The updated state vector (x_k|k).
                - P_updated (torch.Tensor): The updated error covariance matrix (P_k|k).
                - innovation (torch.Tensor): The measurement innovation (y_k).
        """
        # 1. Predicted Measurement: Calculate the expected measurement h(x_k|k-1).
        h_predicted = self._measurement_function(state_predicted)

        # 2. Innovation (Measurement Residual): y_k = z_k - h(x_k|k-1)
        # This is the difference between the actual measurement and what the filter
        # predicted it would measure based on its current state estimate.
        innovation = measurement - h_predicted

        # 3. H: Observation Jacobian (H_k)
        # Relates errors in the state to errors in the measurement.
        H_jacobian = self._compute_jacobian_H(state_predicted)

        # 4. R: Measurement Noise Covariance Matrix
        # R models the uncertainty in the sensor measurements.
        # This implementation uses an adaptive R matrix.
        if tcn_output is not None and "covariance_R" in tcn_output:
            # If TCN provides R, use it (e.g., as learned noise characteristics).
            # TCN output is often log-variance for stability, so exp() might be needed.
            # Assuming covariance_R is already diagonal variance.
            R_adaptive = torch.diag_embed(
                F.softplus(tcn_output["covariance_R"]) + 1e-6
            )  # Ensure positive and add jitter
        else:
            # Scale R based on how much the accelerometer differs from gravity.
            # If the device is moving (accel_norm_diff is large), measurements are
            # less reliable for absolute orientation/position, so R is increased.
            accel_norm_diff = torch.abs(
                torch.norm(accel_body_raw, dim=-1, keepdim=True)
                - torch.norm(self.gravity_w)
            )
            # Exponential scaling factor for R.
            scaling_factor = torch.exp(self.adaptive_R_factor * accel_norm_diff)
            # Ensure diagonal R elements are positive and add a small jitter (1e-6)
            # for numerical stability, preventing division by zero or singular matrices.
            R_adaptive = torch.diag_embed(F.softplus(self.raw_R_diag) + 1e-6) * scaling_factor.unsqueeze(-1)

        # 5. Innovation Covariance: S_k = H_k * P_k|k-1 * H_k^T + R_k
        S_matrix = H_jacobian @ P_predicted @ H_jacobian.transpose(-2, -1) + R_adaptive

        # 6. Kalman Gain: K_k = P_k|k-1 * H_k^T * S_k^-1
        # The Kalman Gain determines how much the filter "trusts" the new measurement
        # relative to its prediction.
        # `torch.linalg.solve` is numerically more stable and efficient than
        # explicitly computing the inverse of S_matrix.
        K_gain = torch.linalg.solve(
            S_matrix, H_jacobian @ P_predicted.transpose(-2, -1)
        ).transpose(-2, -1)

        # 7. State Update: x_k|k = x_k|k-1 + K_k * y_k
        state_updated = state_predicted + (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)

        # Quaternion Normalization:
        # After the state update, the quaternion component might no longer be a
        # unit quaternion due to linearization approximations and noise.
        # Re-normalizing it is crucial to maintain a valid rotation representation
        # and prevent accumulation of errors leading to drift.
        quat_body_to_world = state_updated[..., 6:10]
        quat_body_to_world_normalized = F.normalize(quat_body_to_world, p=2, dim=-1)
        # Re-assemble the state vector with the normalized quaternion.
        state_updated = torch.cat(
            [
                state_updated[..., :6],
                quat_body_to_world_normalized,
                state_updated[..., 10:],
            ],
            dim=-1,
        )

        # 8. Covariance Update (Joseph Form): P_k|k = (I - K_k * H_k) * P_k|k-1 * (I - K_k * H_k)^T + K_k * R_k * K_k^T
        # The Joseph Form of the covariance update equation is used here for
        # numerical stability. It guarantees that the updated covariance matrix
        # P_updated remains symmetric and positive semi-definite (PSD), even in
        # the presence of floating-point inaccuracies. This is crucial to prevent
        # filter divergence, which can occur if P loses its PSD property.
        I_matrix = torch.eye(
            self.state_dim, device=state_predicted.device, dtype=state_predicted.dtype
        )
        P_updated = (I_matrix - K_gain @ H_jacobian) @ P_predicted @ (
            I_matrix - K_gain @ H_jacobian
        ).transpose(-2, -1) + K_gain @ R_adaptive @ K_gain.transpose(-2, -1)

        # Symmetrization: Although Joseph form theoretically guarantees symmetry,
        # explicit symmetrization (P = (P + P^T)/2) can be performed as a
        # safeguard against numerical errors, especially in long simulations.
        # It's less critical with Joseph form but can be added if needed.
        # P_updated = (P_updated + P_updated.transpose(-2, -1)) / 2

        return state_updated, P_updated, innovation

    def _calculate_zupt_update(
        self, state: torch.Tensor, P_covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the state and covariance correction from a Zero-Velocity Update (ZUPT).

        When a ZUPT condition is detected, this acts as an additional measurement
        that the velocity in the world frame should be zero.

        Args:
            state: The current 16-dimensional state vector (x_k|k),
                potentially already updated by IMU measurements.
            P_covariance: The current 16x16 error covariance matrix (P_k|k).

        Returns:
            A tuple containing:
                - state_updated (torch.Tensor): State vector after ZUPT correction.
                - P_updated (torch.Tensor): Covariance matrix after ZUPT correction.
        """
        batch_size = state.shape[0]
        # H_ZUPT: Measurement Jacobian for ZUPT. It selects the velocity components
        # from the state vector, indicating that these are the measured quantities (zero).
        # H_ZUPT = [0 0 0 | I 0 0 0 0 | 0 0 0] - a 3x16 matrix where I is 3x3 identity
        H_zupt = torch.zeros(
            batch_size, 3, self.state_dim, device=state.device, dtype=state.dtype
        )
        H_zupt[:, :, 3:6] = torch.eye(3, device=state.device, dtype=state.dtype)

        # R_ZUPT: Measurement noise covariance for ZUPT. A very small diagonal matrix
        # implies high confidence in the zero-velocity measurement.
        R_zupt = (
            torch.eye(3, device=state.device, dtype=state.dtype).unsqueeze(0)
            * self.zupt_R_factor
        )

        # Innovation for ZUPT: The difference between the expected (zero) and
        # predicted velocity.
        vel_predicted = state[..., 3:6]
        innovation_zupt = -vel_predicted  # z_k (0) - h(x_k|k) (predicted velocity)

        # Calculate Kalman Gain and update state/covariance using standard EKF update equations.
        S_zupt = H_zupt @ P_covariance @ H_zupt.transpose(-2, -1) + R_zupt
        K_zupt_gain = torch.linalg.solve(
            S_zupt, H_zupt @ P_covariance.transpose(-2, -1)
        ).transpose(-2, -1)

        state_updated = state + (K_zupt_gain @ innovation_zupt.unsqueeze(-1)).squeeze(-1)

        # Quaternion Normalization (same as in the standard update step).
        quat_body_to_world = state_updated[..., 6:10]
        quat_body_to_world_normalized = F.normalize(quat_body_to_world, p=2, dim=-1)
        state_updated = torch.cat(
            [
                state_updated[..., :6],
                quat_body_to_world_normalized,
                state_updated[..., 10:],
            ],
            dim=-1,
        )

        # Covariance Update (Joseph Form for numerical stability).
        I_matrix = torch.eye(
            self.state_dim, device=state.device, dtype=state.dtype
        )
        P_updated = (I_matrix - K_zupt_gain @ H_zupt) @ P_covariance @ (
            I_matrix - K_zupt_gain @ H_zupt
        ).transpose(-2, -1) + K_zupt_gain @ R_zupt @ K_zupt_gain.transpose(-2, -1)

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

        This method integrates the prediction step, measurement update step,
        and optionally a ZUPT correction.

        Args:
            state: The current 16-dimensional state vector (x_k-1).
            P_covariance: The current 16x16 error covariance matrix (P_k-1).
            gyro_body_raw: Raw gyroscope measurements at time k.
            accel_body_raw: Raw accelerometer measurements at time k.
            force_raw: Raw force sensor measurement at time k, used for ZUPT.
            measurement: The actual 6-dimensional sensor measurement (z_k).
            tcn_output: Optional. Dictionary containing TCN predictions,
                potentially including 'covariance_R' for adaptive R.

        Returns:
            A tuple containing:
                - final_state (torch.Tensor): The final updated state vector (x_k|k).
                - final_P (torch.Tensor): The final updated error covariance matrix (P_k|k).
                - tcn_features (Dict[str, torch.Tensor]): A dictionary of features
                    extracted from the filter's output for potential use by a TCN.
        """
        # 1. Prediction Step: Propagate state and covariance forward in time.
        state_predicted, P_predicted = self.predict(
            state, P_covariance, gyro_body_raw, accel_body_raw
        )

        # 2. Measurement Update Step: Correct state and covariance using sensor measurements.
        state_updated, P_updated, innovation = self.update(
            state_predicted, P_predicted, measurement, accel_body_raw, tcn_output
        )

        # 3. ZUPT (Zero-Velocity Update) Correction:
        # Detect if the device is currently static.
        is_zupt = (
            self.zupt_detector(accel_body_raw, force_raw)
            if self.use_zupt
            else torch.zeros(accel_body_raw.shape[0], dtype=torch.bool, device=self.device)
        )

        # If ZUPT is detected for any sample in the batch, apply a dedicated
        # ZUPT correction to the already updated state and covariance.
        if torch.any(is_zupt):
            # Apply ZUPT correction only to the samples identified as being in ZUPT.
            state_zupt_corr, P_zupt_corr = self._calculate_zupt_update(
                state_updated[is_zupt], P_updated[is_zupt]
            )
            state_updated[is_zupt] = state_zupt_corr
            P_updated[is_zupt] = P_zupt_corr
            # For ZUPT events, the innovation is effectively forced to zero,
            # as the filter is explicitly commanded to reduce velocity to zero.
            innovation[is_zupt] = 0

        # The final state and covariance after all updates.
        final_state = state_updated
        final_P = P_updated

        # --- Assemble Features for TCN ---
        # These features are typically derived from the filter's current estimate
        # and its internal workings (e.g., innovation, velocity estimates).
        vel_world = final_state[..., 3:6]
        quat_body_to_world = final_state[..., 6:10]
        # Rotate world velocity to body frame for TCN features.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(
            quat_body_to_world
        ).transpose(-2, -1)
        vel_body = (rot_mat_world_to_body @ vel_world.unsqueeze(-1)).squeeze(-1)

        tcn_features: Dict[str, torch.Tensor] = {
            "body_velocity": vel_body,
            "innovation": innovation,
            "zupt_flag": is_zupt.float().unsqueeze(-1),  # Convert boolean to float tensor
        }

        return final_state, final_P, tcn_features


if __name__ == "__main__":
    # Simple test case to verify functionality and shapes of the ExtendedKalmanFilter.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 4
    dt_val = 0.01

    # Initialize the EKF.
    ekf = ExtendedKalmanFilter(dt=dt_val, device=device)

    # Initialize state and covariance for a batch.
    # Initial state: position (0), velocity (0), identity quaternion, zero biases.
    state_initial = torch.zeros(batch_size, 16, device=device)
    state_initial[:, 6] = 1.0  # Set quaternion 'w' component to 1.0 (identity)
    # Initial covariance: Diagonal matrix with some uncertainty.
    P_initial = (
        torch.eye(16, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1
    )

    # Create dummy IMU data and measurement for one time step.
    # Simulate accelerometer reading close to gravity for one sample to test ZUPT logic.
    accel_dummy = torch.randn(batch_size, 3, device=device) * 0.1
    accel_dummy[0, 2] += 9.81  # Add gravity to Z-axis for first sample
    gyro_dummy = torch.randn(batch_size, 3, device=device) * 0.01
    force_dummy = torch.rand(batch_size, 1, device=device)  # Random force data for ZUPT

    # Combine accel and gyro for the measurement vector.
    measurement_dummy = torch.cat([accel_dummy, gyro_dummy], dim=-1)

    # Run one forward pass of the EKF.
    final_state, final_P_covariance, tcn_features = ekf.forward(
        state_initial,
        P_initial,
        gyro_dummy,
        accel_dummy,
        force_dummy,
        measurement_dummy,
    )

    print(f"\nFinal State shape: {final_state.shape}")
    print(f"Final Covariance shape: {final_P_covariance.shape}")
    print("TCN Feature shapes:")
    for k, v in tcn_features.items():
        print(f"  - '{k}': {v.shape}")
    print("\nBatch-aware AEKF tested successfully.")
