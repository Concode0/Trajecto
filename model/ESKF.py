"""
This module implements the Error-State Kalman Filter (ESKF) for 3D Pen
Trajectory Reconstruction.

The ESKF is a robust and widely used algorithm for state estimation in
inertial navigation systems. It operates by maintaining a nominal state
(which propagates through non-linear dynamics) and a small error state
(which is estimated by a linear Kalman filter). This approach combines
the benefits of accurate non-linear propagation with the computational
efficiency and theoretical guarantees of a linear Kalman filter for
error correction.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjust sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rotation_utils import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    small_angle_to_quaternion,
    quaternion_from_two_vectors,
)
from zupt_detector import ZuptDetector
from config import Config


class ErrorStateKalmanFilter(nn.Module):
    """Implements a batch-aware Error-State Kalman Filter (ESKF) for IMU state estimation.

    The ESKF separates the state into a large, non-linear nominal state and a
    small, linear error state. This structure allows for robust and efficient
    estimation.

    - **Nominal State (x_nom)**: Propagated using non-linear dynamics.
      x_nom = [pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b]
              [  3  ,   3  ,      4     ,      3     ,      3     ] = 16 dimensions

    - **Error State (δx)**: Estimated using a linear Kalman filter. A minimal
      representation is used for orientation error.
      δx = [δpos_w, δvel_w, δtheta_b, δgyro_bias_b, δaccel_bias_b]
           [   3  ,   3  ,     3    ,      3      ,      3      ] = 15 dimensions
      where δtheta_b is a 3D rotation vector representing the small angular error.

    The filter operates in a predict-update-inject cycle:
    1.  **Predict**: The nominal state is propagated forward, and the error state
        covariance is predicted.
    2.  **Update**: The error state is corrected using sensor measurements.
    3.  **Inject**: The estimated error is injected into the nominal state, and the
        error state is reset to zero.
    """

    def __init__(
        self,
        error_state_dim: int = 15,
        obs_dim: int = 6,  # 3 accel + 3 gyro
        dt: float = Config.DT,
        device: str = "cpu",
        use_zupt: bool = True,
        use_tcn_zupt: bool = False,
    ):
        """Initializes the ESKF module.

        Args:
            error_state_dim: Dimension of the error state vector. Fixed at 15
                for this implementation (3 for position, 3 for velocity, 3 for
                orientation (small angle), 3 for gyro bias, 3 for accel bias).
            obs_dim: Dimension of the observation vector. Fixed at 6 for
                3-axis accelerometer and 3-axis gyroscope measurements.
            dt: Time step (delta time) in seconds.
            device: The compute device ('cpu', 'cuda', 'mps').
            use_zupt: Boolean flag to enable or disable traditional ZUPT detection.
            use_tcn_zupt: Boolean flag to enable ZUPT decisions based on TCN output.
        """
        super().__init__()

        self.error_state_dim = error_state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device

        # --- 1. Allan Variance Analysis Values (Fixed Physics) ---
        # These values are derived from sensor characterization (Allan Variance plots)
        # and represent the intrinsic noise properties of the IMU.
        # Gyroscope
        arw_x, arw_y, arw_z = Config.ARW_X, Config.ARW_Y, Config.ARW_Z  # Angle Random Walk (ARW)
        gyro_bi_x, gyro_bi_y, gyro_bi_z = Config.GYRO_BI_X, Config.GYRO_BI_Y, Config.GYRO_BI_Z  # Bias Instability (BI)

        # Accelerometer
        vrw_x, vrw_y, vrw_z = Config.VRW_X, Config.VRW_Y, Config.VRW_Z  # Velocity Random Walk (VRW)
        accel_bi_x, accel_bi_y, accel_bi_z = Config.ACCEL_BI_X, Config.ACCEL_BI_Y, Config.ACCEL_BI_Z # Bias Instability (BI)

        # --- 2. Build Q Matrix (Process Noise Covariance) ---
        # Q defines the uncertainty introduced during the state prediction step.
        # It models how noise in the IMU inputs (gyro, accel) affects the error state.
        # Error State Order: [Pos(3), Vel(3), Ori(3), Gyro_Bias(3), Accel_Bias(3)]
        Q_diag_tensor = torch.zeros(self.error_state_dim, device=device)
        # (1) Position Error: Assumed to be driven by velocity error, so no direct noise term.
        Q_diag_tensor[0:3] = 0.0
        # (2) Velocity Error: Driven by accelerometer white noise (VRW).
        Q_diag_tensor[3:6] = torch.tensor([vrw_x**2, vrw_y**2, vrw_z**2], device=device)
        # (3) Orientation Error: Driven by gyroscope white noise (ARW).
        Q_diag_tensor[6:9] = torch.tensor([arw_x**2, arw_y**2, arw_z**2], device=device)
        # (4) Gyro Bias Error: Modeled as a random walk, driven by gyro bias instability.
        Q_diag_tensor[9:12] = torch.tensor([gyro_bi_x**2, gyro_bi_y**2, gyro_bi_z**2], device=device)
        # (5) Accel Bias Error: Modeled as a random walk, driven by accel bias instability.
        Q_diag_tensor[12:15] = torch.tensor([accel_bi_x**2, accel_bi_y**2, accel_bi_z**2], device=device)
        self.register_buffer("Q_diag", Q_diag_tensor)

        # --- 3. Build R Matrix (Measurement Noise for ZUPT) ---
        # R defines the uncertainty in the measurements. This is for the ZUPT update.
        self.zupt_noise_std = nn.Parameter(torch.tensor(Config.ESKFTCN.ZUPT_NOISE_STD_ESKF, device=device))
        self.register_buffer("R_diag", torch.ones(self.obs_dim, device=device) * 1e-4)

        # --- Physical Constants ---
        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device))

        # --- ZUPT Detector ---
        self.zupt_detector = ZuptDetector(
            window_size=Config.ZUPT_WINDOW_SIZE,
            accel_var_threshold=Config.ZUPT_ACCEL_THRESHOLD,
            force_var_threshold=Config.ZUPT_FORCE_VAR_THRESHOLD,
            force_delta_threshold=Config.ZUPT_FORCE_DELTA_THRESHOLD,
            device=device,
        )
        self.use_zupt = use_zupt
        self.use_tcn_zupt = use_tcn_zupt

        self.adaptive_gain = Config.ESKFTCN.ADAPTIVE_GAIN_ESKF


    def get_Q(self):
        """Returns the scaled process noise covariance matrix Q."""
        return torch.diag(self.Q_diag) * self.dt

    def get_R_zupt(self):
        """Returns the measurement noise covariance R for ZUPT."""
        return torch.diag(self.zupt_noise_std ** 2)

    def _make_symmetric(self, P_covariance: torch.Tensor) -> torch.Tensor:
        """Enforces symmetry on the covariance matrix P to prevent numerical instability.

        Covariance matrices are inherently symmetric and positive semi-definite.
        Due to floating-point arithmetic inaccuracies during numerous matrix operations,
        a covariance matrix might lose its perfect symmetry. Enforcing symmetry by
        averaging with its transpose (0.5 * (P + P^T)) ensures that P remains
        mathematically valid, which is crucial for the stability and correctness
        of the Kalman filter.

        Args:
            P_covariance: The covariance matrix to symmetrize.

        Returns:
            The symmetrized covariance matrix.
        """
        return 0.5 * (P_covariance + P_covariance.transpose(-2, -1))

    def _propagate_nominal_state(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_b_raw: torch.Tensor,
        accel_b_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagates the nominal state forward in time using IMU measurements and kinematic equations.

        This function implements the non-linear dynamics model f(x_nom, u) for the ESKF.

        Args:
            pos_w: Current nominal position in world frame.
            vel_w: Current nominal velocity in world frame.
            quat_b_to_w: Current nominal body-to-world quaternion.
            gyro_bias_b: Current nominal gyroscope bias in body frame.
            accel_bias_b: Current nominal accelerometer bias in body frame.
            gyro_b_raw: Raw gyroscope measurements.
            accel_b_raw: Raw accelerometer measurements.

        Returns:
            A tuple containing the propagated nominal state components
            (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new).
        """
        # Correct raw IMU measurements by subtracting estimated biases.
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b

        # --- Quaternion propagation (orientation update) ---
        # Trapezoidal Integration (Exponential Map):
        # Instead of linearizing using q_dot, we use the exact solution for constant
        # angular velocity over the interval dt. This is equivalent to the exponential map.
        # q_new = q_old * Exp(omega * dt)
        angle_change = gyro_b_corrected * self.dt
        delta_quat = small_angle_to_quaternion(angle_change)
        quat_b_to_w_new = quaternion_multiply(quat_b_to_w, delta_quat)

        # --- Acceleration in World Frame (Old State) ---
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_w = (
            rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)
        ).squeeze(-1) - self.gravity_w

        # --- Acceleration in World Frame (New State) ---
        # We re-compute acceleration with the updated orientation to apply the Trapezoidal Rule.
        rot_mat_b_to_w_new = quaternion_to_rotation_matrix(quat_b_to_w_new)
        accel_w_new = (
            rot_mat_b_to_w_new @ accel_b_corrected.unsqueeze(-1)
        ).squeeze(-1) - self.gravity_w

        # --- Position and Velocity Integration (Trapezoidal Rule) ---
        # v_new = v_old + 0.5 * (a_old + a_new) * dt
        vel_w_new = vel_w + 0.5 * (accel_w + accel_w_new) * self.dt

        # p_new = p_old + 0.5 * (v_old + v_new) * dt
        pos_w_new = pos_w + 0.5 * (vel_w + vel_w_new) * self.dt

        # Biases are modeled as random walks, so their nominal values do not
        # change during deterministic propagation. Their uncertainty evolution is
        # handled in the error state covariance prediction (P_error).
        gyro_bias_b_new = gyro_bias_b
        accel_bias_b_new = accel_bias_b

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new)

    def predict(
        self,
        P_error_covariance: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        accel_b_raw: torch.Tensor,
        accel_bias_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the error state covariance matrix.

        This function computes the predicted error covariance P_k|k-1 using
        the linearized error state dynamics (F_error) and the process noise (Q_error).
        The equation is: P_k|k-1 = F_error * P_k-1|k-1 * F_error^T + Q_error

        Args:
            P_error_covariance: The current 15x15 error covariance matrix (P_k-1|k-1).
            quat_b_to_w: Current nominal body-to-world quaternion.
            accel_b_raw: Raw accelerometer measurements.
            accel_bias_b: Current nominal accelerometer bias.

        Returns:
            The predicted 15x15 error covariance matrix (P_k|k-1).
        """
        batch_size = P_error_covariance.shape[0]

        # --- 1. Build F_error: The Error State Transition Matrix ---
        # This matrix linearizes the error state dynamics, relating the error at
        # the previous step to the error at the current step.
        F_error_matrix = (
            torch.eye(self.error_state_dim, device=self.device, dtype=P_error_covariance.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Skew-symmetric matrix of corrected acceleration for Jacobian calculation.
        accel_ssm = torch.zeros(batch_size, 3, 3, device=self.device, dtype=P_error_covariance.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # Populate F_error based on linearized error dynamics:
        # δpos_w_new = δpos_w + dt * δvel_w
        F_error_matrix[:, 0:3, 3:6] = torch.eye(3, device=self.device) * self.dt
        # δvel_w_new = δvel_w + dt * (-R_bw * [a_b_corrected]x * δθ_b - R_bw * δa_b)
        F_error_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        # δθ_b_new = δθ_b - dt * δg_b
        F_error_matrix[:, 6:9, 9:12] = -torch.eye(3, device=self.device) * self.dt

        # --- 2. Get Q_error: The Process Noise Covariance Matrix ---
        # We use a trapezoidal integration approximation for the discrete-time Q matrix:
        # Q_k ≈ 0.5 * (F * Q_continuous * F^T + Q_continuous) * dt
        # This provides a higher-order approximation than the simple Q_continuous * dt.
        Q_continuous = torch.diag(self.Q_diag).unsqueeze(0).expand(batch_size, -1, -1)

        Q_error_matrix = 0.5 * (
            F_error_matrix @ Q_continuous @ F_error_matrix.transpose(-2, -1) + Q_continuous
        ) * self.dt

        # --- 3. Predict Covariance ---
        # P_k|k-1 = F * P_k-1|k-1 * F^T + Q_k
        P_predicted = (
            F_error_matrix @ P_error_covariance @ F_error_matrix.transpose(-2, -1)
            + Q_error_matrix
        )

        # Enforce symmetry to prevent numerical instability.
        return self._make_symmetric(P_predicted)

    def update(
        self,
        P_error_pred: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        measurement: torch.Tensor,
        R_override: Optional[torch.Tensor] = None,
        gating_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates the error state and its covariance based on a measurement.

        This function performs the measurement update (correction) step of the ESKF.

        Args:
            P_error_pred: The predicted 15x15 error covariance matrix (P_k|k-1).
            quat_b_to_w: Current nominal body-to-world quaternion.
            accel_bias_b: Current nominal accelerometer bias.
            gyro_bias_b: Current nominal gyroscope bias.
            measurement: The actual 6-dimensional sensor measurement (z_k).
            R_override: Optional override for the measurement noise covariance (R_k).
            gating_threshold: Optional Mahalanobis distance threshold for gating.

        Returns:
            A tuple containing:
                - delta_x: The estimated 15-dimensional error state (δx_k|k).
                - P_error_new: The updated 15x15 error covariance matrix (P_k|k).
                - innovation: The measurement innovation (y_k).
                - mahalanobis_sq: The squared Mahalanobis distance (d^2).
        """
        batch_size = P_error_pred.shape[0]

        # --- 1. Build H_error: The Error Observation Matrix ---
        # This matrix linearizes the measurement model, relating the error state
        # to the predicted measurement.
        H_error_matrix = torch.zeros(
            batch_size, self.obs_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype
        )

        # Predicted gravity vector in the body frame (g_b = R_wb * g_w).
        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(-1)

        # Skew-symmetric matrix of gravity_body for Jacobian calculation.
        gravity_ssm = torch.zeros(batch_size, 3, 3, device=self.device, dtype=P_error_pred.dtype)
        gravity_ssm[:, 0, 1] = -gravity_body[:, 2]
        gravity_ssm[:, 0, 2] = gravity_body[:, 1]
        gravity_ssm[:, 1, 0] = gravity_body[:, 2]
        gravity_ssm[:, 1, 2] = -gravity_body[:, 0]
        gravity_ssm[:, 2, 0] = -gravity_body[:, 1]
        gravity_ssm[:, 2, 1] = gravity_body[:, 0]

        # Populate H_error:
        # Errors in orientation (δθ_b) affect predicted accel measurement via gravity.
        H_error_matrix[:, 0:3, 6:9] = gravity_ssm
        # Errors in accel bias (δa_b) directly affect predicted accel measurement.
        H_error_matrix[:, 0:3, 12:15] = torch.eye(3, device=self.device)
        # Errors in gyro bias (δg_b) directly affect predicted gyro measurement.
        H_error_matrix[:, 3:6, 9:12] = torch.eye(3, device=self.device)

        # --- 2. Calculate Innovation ---
        # Predicted measurement h(x_nom): what the sensor should read given the nominal state.
        accel_pred = gravity_body + accel_bias_b
        gyro_pred = gyro_bias_b
        h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)

        # Innovation (residual): y = z - h(x_nom)
        innovation = measurement - h_predicted

        # --- 3. Calculate Kalman Gain ---
        # R_noise: Measurement Noise Covariance Matrix.
        if R_override is not None:
            R_noise_matrix = R_override + torch.eye(self.obs_dim, device=self.device) * 1e-6
        else:
            # Simple adaptive noise based on acceleration deviation from gravity
            accel_meas = measurement[..., 0:3]
            accel_norm_diff = torch.abs(torch.norm(accel_meas, dim=-1, keepdim=True) - torch.norm(self.gravity_w))
            scaling_factor = torch.exp(self.adaptive_gain * accel_norm_diff)
            base_R = torch.diag_embed(F.softplus(self.R_diag) + 1e-6) # (6, 6)
            R_noise_matrix = base_R.unsqueeze(0).expand(batch_size, -1, -1).clone() # (B, 6, 6)
            R_noise_matrix[..., 0:3, 0:3] *= scaling_factor.unsqueeze(-1)


        # S: Innovation Covariance Matrix: S = H * P * H^T + R
        S_matrix = H_error_matrix @ P_error_pred @ H_error_matrix.transpose(-2, -1) + R_noise_matrix
        # Add jitter for numerical stability during inversion.
        S_matrix += torch.eye(self.obs_dim, device=self.device) * 1e-6

        # K: Kalman Gain: K = P * H^T * S^-1
        # Using torch.linalg.solve is more stable than direct inversion.
        K_gain = torch.linalg.solve(
            S_matrix, H_error_matrix @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # --- 4. Update Error State and Covariance ---
        # Update error state: δx_k|k = K_k * y_k
        delta_x = (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)

        # Calculate Mahalanobis Distance (squared): d^2 = y^T * S^-1 * y
        # solve S * x = y for x.
        innovation_unsq = innovation.unsqueeze(-1)
        # Re-use S_matrix (already inverted implicitly above? No, solved against H@P.T)
        # We need to solve S * x = innovation
        sol_x = torch.linalg.solve(S_matrix, innovation_unsq)
        # d^2 = innovation^T * x
        mahalanobis_sq = (innovation.unsqueeze(1) @ sol_x).squeeze(-1).squeeze(-1) # Shape: (Batch,)

        # Update covariance using Joseph Form for numerical stability:
        # P_k|k = (I - K*H) * P_k|k-1 * (I - K*H)^T + K * R * K^T
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH = I_matrix - K_gain @ H_error_matrix
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gain @ R_noise_matrix @ K_gain.transpose(-2, -1)
        )

        # Enforce symmetry.
        P_error_new = self._make_symmetric(P_error_new)

        # --- 5. Mahalanobis Gating ---
        if gating_threshold is not None:
             reject_mask = mahalanobis_sq > gating_threshold
             # If rejected, delta_x should be 0 (no correction)
             delta_x = torch.where(reject_mask.unsqueeze(-1), torch.zeros_like(delta_x), delta_x)
             # If rejected, P should remain P_pred (no information gain)
             P_error_new = torch.where(reject_mask.unsqueeze(-1).unsqueeze(-1), P_error_pred, P_error_new)

        return delta_x, P_error_new, innovation, mahalanobis_sq

    def _calculate_zupt_update(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        tcn_zupt_prob: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the error-state correction from a Zero-Velocity Update (ZUPT) pseudo-measurement.

        When a ZUPT is detected, this acts as a measurement that world-frame
        velocity is zero (z_ZUPT = 0).

        Args:
            vel_w_pred: Predicted nominal velocity in world frame.
            P_error_pred: The predicted 15x15 error covariance matrix.
            tcn_zupt_prob: Optional TCN-predicted zero-velocity probability (0 to 1).
                           - Shape: (Batch, 1) or (Batch,)
                           - Unit: Probability | Range: 0.0 to 1.0

        Returns:
            A tuple containing:
                - delta_x_zupt: The estimated error state correction from ZUPT.
                - P_error_new: The updated error covariance matrix after ZUPT.
        """
        batch_size = P_error_pred.shape[0]

        # H_ZUPT: Measurement Jacobian for ZUPT. It selects the velocity error (δv_w).
        H_zupt = torch.zeros(batch_size, 3, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
        H_zupt[:, :, 3:6] = torch.eye(3, device=self.device)

        # R_ZUPT: ZUPT measurement noise. A small value implies high confidence
        # in the zero-velocity measurement.
        if tcn_zupt_prob is not None:
            # Scale R_zupt based on TCN probability: higher prob -> smaller R (more confident)
            # Ensure tcn_zupt_prob has shape (Batch, 1) for broadcasting
            if tcn_zupt_prob.ndim == 1:
                tcn_zupt_prob = tcn_zupt_prob.unsqueeze(-1)
            # Use a linear interpolation between min_R and max_R based on probability
            # R = R_min * prob + R_max * (1 - prob)
            # Let R_min be self.zupt_noise_std**2, and R_max be a large value for uncertainty
            min_R_val = self.zupt_noise_std**2
            # A large value for R when ZUPT prob is low (e.g., 100 times min_R_val)
            max_R_val = min_R_val * 100

            # Clamp probability to avoid extreme values and numerical instability
            # Epsilon ensures we don't divide by zero or have extremely small R
            clamped_prob = torch.clamp(tcn_zupt_prob, 0.01, 0.99)

            # R_zupt_scaled = min_R_val / (clamped_prob + 1e-6)  # Alternative scaling
            R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)

            # Ensure R_zupt_scaled_diag has shape (Batch, 3)
            if R_zupt_scaled_diag.shape[-1] == 1:
                R_zupt_scaled_diag = R_zupt_scaled_diag.repeat(1, 3)

            R_zupt_matrix = torch.diag_embed(R_zupt_scaled_diag) # (Batch, 3, 3)
        else:
            R_zupt_matrix = self.get_R_zupt().unsqueeze(0).expand(batch_size, -1, -1)

        # Innovation for ZUPT: y = z - h(x_nom) = 0 - v_w_pred
        innovation_zupt = -vel_w_pred

        # S_ZUPT: Innovation Covariance for ZUPT.
        S_zupt_matrix = H_zupt @ P_error_pred @ H_zupt.transpose(-2, -1) + R_zupt_matrix
        S_zupt_matrix += torch.eye(3, device=self.device) * 1e-6 # Jitter

        # K_ZUPT: Kalman Gain for ZUPT.
        K_zupt_gain = torch.linalg.solve(
            S_zupt_matrix, H_zupt @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # Error State Update from ZUPT.
        delta_x_zupt = (K_zupt_gain @ innovation_zupt.unsqueeze(-1)).squeeze(-1)

        # Covariance Update (Joseph Form) for ZUPT.
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH_zupt = I_matrix - K_zupt_gain @ H_zupt
        P_error_new = (
            ImKH_zupt @ P_error_pred @ ImKH_zupt.transpose(-2, -1)
            + K_zupt_gain @ R_zupt_matrix @ K_zupt_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x_zupt, P_error_new

    def _apply_tcn_velocity_correction(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        vel_corr_b: torch.Tensor,
        quat_b_to_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the error-state correction from the TCN's velocity prediction.

        This treats the TCN's velocity correction as a pseudo-measurement.

        Args:
            vel_w_pred: Predicted nominal velocity in world frame.
            P_error_pred: The predicted 15x15 error covariance matrix.
            vel_corr_b: Velocity correction predicted by the TCN in body frame.
            quat_b_to_w: Current nominal body-to-world quaternion.

        Returns:
            A tuple containing:
                - delta_x_tcn: The estimated error state from TCN correction.
                - P_error_new: The updated error covariance matrix.
        """
        batch_size = P_error_pred.shape[0]

        # H_TCN: Measurement Jacobian, selecting the velocity error.
        H_tcn = torch.zeros(batch_size, 3, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
        H_tcn[:, :, 3:6] = torch.eye(3, device=self.device)

        # Convert TCN's body-frame velocity correction to world frame.
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        vel_corr_w = (rot_mat_b_to_w @ vel_corr_b.unsqueeze(-1)).squeeze(-1)

        # Innovation is the TCN's predicted velocity correction.
        innovation_tcn = vel_corr_w

        # R_TCN: Measurement noise covariance for TCN correction.
        R_tcn_matrix = torch.eye(3, device=self.device).unsqueeze(0) * 1e-4

        # S_TCN: Innovation Covariance for TCN.
        S_tcn_matrix = H_tcn @ P_error_pred @ H_tcn.transpose(-2, -1) + R_tcn_matrix

        # K_TCN: Kalman Gain for TCN correction.
        K_tcn_gain = torch.linalg.solve(
            S_tcn_matrix, H_tcn @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # Error State Update.
        delta_x_tcn = (K_tcn_gain @ innovation_tcn.unsqueeze(-1)).squeeze(-1)

        # Covariance Update (Joseph Form).
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH_tcn = I_matrix - K_tcn_gain @ H_tcn
        P_error_new = (
            ImKH_tcn @ P_error_pred @ ImKH_tcn.transpose(-2, -1)
            + K_tcn_gain @ R_tcn_matrix @ K_tcn_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x_tcn, P_error_new

    def inject_correction(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        delta_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Injects the calculated error state (delta_x) back into the nominal state.

        This correction step is fundamental to the ESKF, preventing the error
        from growing large and ensuring the linearity assumption of the error
        state remains valid. After injection, the error state is reset to zero.

        Args:
            pos_w: Current nominal position in world frame.
            vel_w: Current nominal velocity in world frame.
            quat_b_to_w: Current nominal body-to-world quaternion.
            gyro_bias_b: Current nominal gyroscope bias.
            accel_bias_b: Current nominal accelerometer bias.
            delta_x: The estimated 15-dimensional error state.

        Returns:
            A tuple containing the corrected nominal state components.
        """
        # Decompose the estimated error state delta_x.
        d_pos_w, d_vel_w, d_theta_b, d_gyro_bias_b, d_accel_bias_b = delta_x.split([3, 3, 3, 3, 3], dim=-1)

        # Apply position and velocity corrections.
        pos_w_new = pos_w + d_pos_w
        vel_w_new = vel_w + d_vel_w

        # Apply orientation correction using a small-angle quaternion approximation.
        quat_b_to_w_new = quaternion_multiply(
            quat_b_to_w, small_angle_to_quaternion(d_theta_b)
        )
        quat_b_to_w_new = F.normalize(quat_b_to_w_new, p=2, dim=-1)

        # Apply bias corrections.
        gyro_bias_b_new = gyro_bias_b + d_gyro_bias_b
        accel_bias_b_new = accel_bias_b + d_accel_bias_b

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new)

    def forward(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        P_error: torch.Tensor,
        gyro_b_raw: torch.Tensor,
        accel_b_raw: torch.Tensor,
        force_raw: torch.Tensor,
        measurement: torch.Tensor,
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """Executes one full predict-update-inject cycle of the ESKF.

        Args:
            pos_w (torch.Tensor): Current nominal position.
                - Shape: (Batch, 3) | Unit: m | Frame: World
            vel_w (torch.Tensor): Current nominal velocity.
                - Shape: (Batch, 3) | Unit: m/s | Frame: World
            quat_b_to_w (torch.Tensor): Current nominal quaternion.
                - Shape: (Batch, 4) | Frame: Body-to-World
            gyro_bias_b (torch.Tensor): Current nominal gyroscope bias.
                - Shape: (Batch, 3) | Unit: rad/s | Frame: Body
            accel_bias_b (torch.Tensor): Current nominal accelerometer bias.
                - Shape: (Batch, 3) | Unit: m/s^2 | Frame: Body
            P_error (torch.Tensor): Current error covariance matrix.
                - Shape: (Batch, 15, 15)
            gyro_b_raw (torch.Tensor): Raw gyroscope measurements.
                - Shape: (Batch, 3) | Unit: rad/s | Frame: Body
            accel_b_raw (torch.Tensor): Raw accelerometer measurements.
                - Shape: (Batch, 3) | Unit: m/s^2 | Frame: Body
            force_raw (torch.Tensor): Raw force sensor measurement.
                - Shape: (Batch, 1) | Unit: N
            measurement (torch.Tensor): The 6D sensor measurement vector.
                - Shape: (Batch, 6) | [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            tcn_output (Optional[Dict[str, torch.Tensor]]): Optional TCN predictions.

        Returns:
            Tuple[...]: Updated state components and features:
                - pos_w_new: (Batch, 3)
                - vel_w_new: (Batch, 3)
                - quat_b_to_w_new: (Batch, 4)
                - gyro_bias_b_new: (Batch, 3)
                - accel_bias_b_new: (Batch, 3)
                - P_error_final: (Batch, 15, 15)
                - tcn_features: Dict with keys "body_velocity" (Batch, 3), "zupt_flag" (Batch, 1), "innovation" (Batch, 6), "mahalanobis" (Batch, 1)
        """
        # --- 1. Prediction Step ---
        pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred = (
            self._propagate_nominal_state(
                pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, gyro_b_raw, accel_b_raw
            )
        )
        P_error_pred = self.predict(P_error, quat_b_to_w, accel_b_raw, accel_bias_b)

        # --- 2. Update Step ---
        P_error_final = P_error_pred
        total_delta_x = torch.zeros(pos_w.shape[0], self.error_state_dim, device=self.device, dtype=pos_w.dtype)
        innovation_output = torch.zeros(pos_w.shape[0], self.obs_dim, device=self.device, dtype=pos_w.dtype)
        mahalanobis_output = torch.zeros(pos_w.shape[0], device=self.device, dtype=pos_w.dtype)

        # Determine if ZUPT should be applied.
        if self.use_tcn_zupt and tcn_output is not None:
            is_zupt = tcn_output["zupt_prob"].squeeze(-1) > 0.5
        elif self.use_zupt:
            is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        else:
            is_zupt = torch.zeros(accel_b_raw.shape[0], dtype=torch.bool, device=self.device)

        # Apply ZUPT correction where applicable.
        if torch.any(is_zupt):
            zupt_mask = is_zupt
            zupt_prob_to_pass = None
            if self.use_tcn_zupt and tcn_output is not None:
                zupt_prob_to_pass = tcn_output["zupt_prob"][zupt_mask]

            delta_x_zupt, P_after_zupt = self._calculate_zupt_update(
                vel_w_pred[zupt_mask], P_error_pred[zupt_mask], tcn_zupt_prob=zupt_prob_to_pass
            )
            total_delta_x[zupt_mask] += delta_x_zupt
            P_error_final[zupt_mask] = P_after_zupt

        # Apply TCN-based corrections or standard measurement update.
        if tcn_output is not None:
            # TCN Velocity Correction
            vel_corr_body = tcn_output["vel_corr"]
            if torch.any(is_zupt): # Don't apply TCN velocity correction during ZUPT
                vel_corr_body = torch.where(is_zupt.unsqueeze(-1), torch.zeros_like(vel_corr_body), vel_corr_body)
            delta_x_tcn, P_after_tcn = self._apply_tcn_velocity_correction(vel_w_pred, P_error_final, vel_corr_body, quat_b_to_w_pred)
            total_delta_x += delta_x_tcn
            P_error_final = P_after_tcn

            # Standard Measurement Update (with TCN-provided adaptive R)
            # TCN now predicts log(R), so use exp to get R
            tcn_cov_diag = torch.exp(tcn_output["covariance_R"])
            R_tcn_override = torch.diag_embed(tcn_cov_diag)

            # Pass gating threshold from Config
            gating_thresh = Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD

            delta_x_up, P_after_up, innovation, mahalanobis_sq = self.update(
                P_error_final,
                quat_b_to_w_pred,
                accel_bias_b_pred,
                gyro_bias_b_pred,
                measurement,
                R_override=R_tcn_override,
                gating_threshold=gating_thresh
            )
            total_delta_x += delta_x_up
            P_error_final = P_after_up
            innovation_output = innovation
            mahalanobis_output = mahalanobis_sq
        else:
            # Standard Measurement Update (without TCN)
            # Logic Change: We DO NOT perform a standard update here because the
            # assumption that accel=gravity and gyro=0 (static) is invalid during
            # motion. Applying it indiscriminately fights the integration.
            # We only compute the innovation for feature logging/TCN input.

            # Recompute necessary variables for innovation (normally done inside update)
            rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w_pred).transpose(-2, -1)
            gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(-1)

            accel_pred = gravity_body + accel_bias_b_pred
            gyro_pred = gyro_bias_b_pred
            h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)

            innovation_output = measurement - h_predicted
            # No delta_x update or P_error update in this branch.

        # --- 3. Error Injection ---
        if torch.any(total_delta_x != 0):
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = self.inject_correction(
                pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred, total_delta_x
            )
        else:
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = (pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred)

        # --- 4. Assemble Features for next TCN step ---
        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w_new).transpose(-2, -1)
        vel_body = (rot_mat_world_to_body @ vel_w_new.unsqueeze(-1)).squeeze(-1)
        tcn_features: Dict[str, torch.Tensor] = {
            "body_velocity": vel_body,
            "zupt_flag": is_zupt.float().unsqueeze(-1),
            "innovation": innovation_output,
            "mahalanobis": mahalanobis_output.unsqueeze(-1),
        }

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new, P_error_final, tcn_features)


if __name__ == "__main__":
    # This test case verifies the functionality and tensor shapes of the ErrorStateKalmanFilter.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dt_val = 0.01

    # --- Test with TCN integration (simulated TCN output) ---
    print("\n--- Testing ESKF with TCN integration ---")
    # Instantiate ESKF with TCN-based ZUPT enabled.
    eskf = ErrorStateKalmanFilter(dt=dt_val, device=device, use_tcn_zupt=True)

    batch_size = 4
    # Initial nominal state: zero position/velocity, identity quaternion, zero biases.
    pos_w_init = torch.zeros(batch_size, 3, device=device)
    vel_w_init = torch.zeros(batch_size, 3, device=device)
    quat_b_to_w_init = torch.zeros(batch_size, 4, device=device)
    quat_b_to_w_init[:, 0] = 1.0
    gyro_bias_b_init = torch.zeros(batch_size, 3, device=device)
    accel_bias_b_init = torch.zeros(batch_size, 3, device=device)
    # Initial error covariance with some uncertainty.
    P_error_init = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

    # Dummy IMU data for one time step.
    accel_dummy = torch.randn(batch_size, 3, device=device) * 0.1
    accel_dummy[0, 2] += 9.81  # Simulate near-static condition for ZUPT test
    gyro_dummy = torch.randn(batch_size, 3, device=device) * 0.01
    force_dummy = torch.rand(batch_size, 1, device=device)
    measurement_dummy = torch.cat([accel_dummy, gyro_dummy], dim=-1)

    # Dummy TCN output.
    tcn_out_dummy = {
        "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
        "covariance_R": torch.randn(batch_size, 6, device=device),
        "zupt_prob": torch.rand(batch_size, 1, device=device),
    }

    # Run one forward pass.
    (pos_w_out, vel_w_out, quat_b_to_w_out, gyro_bias_b_out, accel_bias_b_out, P_error_out, tcn_feats_out) = eskf.forward(
        pos_w_init, vel_w_init, quat_b_to_w_init, gyro_bias_b_init, accel_bias_b_init,
        P_error_init, gyro_dummy, accel_dummy, force_dummy, measurement_dummy, tcn_output=tcn_out_dummy
    )

    print(f"Updated Position (p) shape: {pos_w_out.shape}")
    print(f"Updated Velocity (v) shape: {vel_w_out.shape}")
    print(f"Updated Quaternion (q) shape: {quat_b_to_w_out.shape}")
    print(f"Updated Gyro Bias (bg) shape: {gyro_bias_b_out.shape}")
    print(f"Updated Accel Bias (ba) shape: {accel_bias_b_out.shape}")
    print(f"Final Error Covariance (P) shape: {P_error_out.shape}")
    print("TCN Features:")
    for k, v in tcn_feats_out.items():
        print(f"  - '{k}': {v.shape}")
    print("\nESKF with TCN integration tested successfully.")

    # --- Test without TCN ZUPT (using traditional ZUPT detector) ---
    print("\n--- Testing ESKF with traditional ZUPT ---")
    eskf_no_tcn_zupt = ErrorStateKalmanFilter(dt=dt_val, device=device, use_tcn_zupt=False, use_zupt=True)
    (_, _, _, _, _, P_error_out_no_tcn, _) = eskf_no_tcn_zupt.forward(
        pos_w_init, vel_w_init, quat_b_to_w_init, gyro_bias_b_init, accel_bias_b_init,
        P_error_init, gyro_dummy, accel_dummy, force_dummy, measurement_dummy, tcn_output=None
    )
    print(f"Final P_error shape (no TCN ZUPT): {P_error_out_no_tcn.shape}")
    print("ESKF with traditional ZUPT tested successfully.")
