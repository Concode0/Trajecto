"""
Error-State Kalman Filter (ESKF) for 3D pen trajectory reconstruction.

Maintains a nominal state propagated through non-linear dynamics and a
linear error state for corrections. This approach combines accurate
non-linear propagation with computationally efficient linear filtering.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

# Import rotation utilities
from rotation_utils import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    small_angle_to_quaternion,
    quaternion_from_two_vectors,
)

from zupt_detector import ZuptDetector


class ErrorStateKalmanFilter(nn.Module):
    """Batch-aware Error-State Kalman Filter for IMU state estimation.

    State Representation:
    - Nominal State (16D): [pos_w(3), vel_w(3), quat_b_to_w(4), gyro_bias_b(3), accel_bias_b(3)]
    - Error State (15D): [δpos_w(3), δvel_w(3), δtheta_b(3), δgyro_bias_b(3), δaccel_bias_b(3)]

    Filter Cycle:
    1. Predict: Propagate nominal state and error covariance
    2. Update: Correct error state using measurements
    3. Inject: Apply error correction to nominal state and reset error
    """

    def __init__(
        self,
        error_state_dim: int = 15,
        obs_dim: int = 6,  # 3 accel + 3 gyro
        dt: float = Config.DT,
        device: str = "cpu",
        use_zupt: bool = True,
        use_tcn_zupt: bool = False,
        use_virtual_measurements: bool = False,
        mahalanobis_threshold: float = Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD,
        eskf_learnable_params: bool = Config.ESKFTCN.ESKF_LEARNABLE_PARAMS,
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
            use_virtual_measurements: Boolean flag to enable virtual measurement
                updates during motion (for pure ESKF mode without TCN).
            mahalanobis_threshold: Threshold for Mahalanobis gating.
            eskf_learnable_params: If True, ESKF parameters (R_diag, zupt_noise_std,
                virtual_meas_weights) are learnable via backpropagation. If False,
                they are fixed buffers (faster training, no BPTT).
        """
        super().__init__()

        self.error_state_dim = error_state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device
        self.mahalanobis_threshold = mahalanobis_threshold

        # Allan Variance parameters (sensor characterization)
        arw_x, arw_y, arw_z = Config.ARW_X, Config.ARW_Y, Config.ARW_Z
        gyro_bi_x, gyro_bi_y, gyro_bi_z = Config.GYRO_BI_X, Config.GYRO_BI_Y, Config.GYRO_BI_Z
        vrw_x, vrw_y, vrw_z = Config.VRW_X, Config.VRW_Y, Config.VRW_Z
        accel_bi_x, accel_bi_y, accel_bi_z = Config.ACCEL_BI_X, Config.ACCEL_BI_Y, Config.ACCEL_BI_Z

        # Q diagonal: [Pos(3), Vel(3), Ori(3), GyBias(3), AcBias(3)]
        Q_diag_tensor = torch.zeros(self.error_state_dim, device=device)
        Q_diag_tensor[0:3] = 0.0  # Pos: driven by vel
        Q_diag_tensor[3:6] = torch.tensor([vrw_x**2, vrw_y**2, vrw_z**2], device=device)  # Vel: accel VRW
        Q_diag_tensor[6:9] = torch.tensor([arw_x**2, arw_y**2, arw_z**2], device=device)  # Ori: gyro ARW
        Q_diag_tensor[9:12] = torch.tensor([gyro_bi_x**2, gyro_bi_y**2, gyro_bi_z**2], device=device)  # Gyro bias: BI
        Q_diag_tensor[12:15] = torch.tensor([accel_bi_x**2, accel_bi_y**2, accel_bi_z**2], device=device)  # Accel bias: BI
        self.register_buffer("Q_diag", Q_diag_tensor)

        # ZUPT noise: learnable or fixed based on flag
        if use_zupt or use_tcn_zupt:
            zupt_noise_tensor = torch.tensor(Config.ESKFTCN.ZUPT_NOISE_STD_ESKF, device=device)
            if eskf_learnable_params:
                self.zupt_noise_std = nn.Parameter(zupt_noise_tensor)
            else:
                self.register_buffer("zupt_noise_std", zupt_noise_tensor)

        # R_diag: learnable or fixed based on flag
        R_diag_tensor = torch.ones(self.obs_dim, device=device) * 1e-4
        if eskf_learnable_params:
            self.R_diag = nn.Parameter(R_diag_tensor)
        else:
            self.register_buffer("R_diag", R_diag_tensor)

        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, -Config.GRAVITY_MAGNITUDE], device=device))

        self.zupt_detector = ZuptDetector(
            window_size=Config.ZUPT_WINDOW_SIZE,
            accel_var_threshold=Config.ZUPT_ACCEL_THRESHOLD,
            force_var_threshold=Config.ZUPT_FORCE_VAR_THRESHOLD,
            force_delta_threshold=Config.ZUPT_FORCE_DELTA_THRESHOLD,
            device=device,
        )
        self.use_zupt = use_zupt
        self.use_tcn_zupt = use_tcn_zupt
        self.use_virtual_measurements = use_virtual_measurements
        self.adaptive_gain = Config.ESKFTCN.ADAPTIVE_GAIN_ESKF

        # Virtual measurement correction weights (for blind period / pure ESKF mode)
        # Initialized from Allan variance ratios - provides principled starting point
        # Can be learnable or fixed based on eskf_learnable_params flag
        if use_virtual_measurements:
            virtual_meas_tensor = torch.tensor([
                0.0, 0.0, 0.0,  # Position (blocked - no absolute reference)
                0.0, 0.0, 0.0,  # Velocity (blocked - would cause scale drift)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,  # Orientation (from 1/ARW)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,    # Gyro bias (from 1/GYRO_BI)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,   # Accel bias (from 1/ACCEL_BI)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,
            ], device=device, dtype=torch.float32)
            if eskf_learnable_params:
                self.virtual_meas_weights = nn.Parameter(virtual_meas_tensor)
            else:
                self.register_buffer("virtual_meas_weights", virtual_meas_tensor)


    def get_Q(self):
        """Returns the scaled process noise covariance matrix Q."""
        return torch.diag(self.Q_diag) * self.dt

    def get_R_zupt(self):
        """Returns the measurement noise covariance R for ZUPT."""
        return torch.diag(self.zupt_noise_std ** 2)

    def _make_symmetric(self, P_covariance: torch.Tensor) -> torch.Tensor:
        """Enforces symmetry on covariance matrix to prevent numerical drift.

        Args:
            P_covariance: Covariance matrix to symmetrize.

        Returns:
            Symmetrized covariance matrix via (P + P^T) / 2.
        """
        return 0.5 * (P_covariance + P_covariance.transpose(-2, -1))

    def _solve_symmetric_system(
        self,
        S: torch.Tensor,
        B: torch.Tensor,
        compute_quadratic: bool = False,
        quadratic_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Solve symmetric positive definite linear system using Cholesky decomposition.

        Computes X = S^{-1} @ B using Cholesky decomposition for numerical stability.
        Optionally computes quadratic form v^T @ S^{-1} @ v for Mahalanobis distance.

        This method consolidates the repeated try-except Cholesky pattern throughout
        the codebase, providing a single point for numerical stability handling.

        Args:
            S: Symmetric positive definite matrix [batch, n, n].
            B: Right-hand side matrix [batch, n, m] or [batch, n].
            compute_quadratic: If True, compute v^T @ S^{-1} @ v.
            quadratic_vec: Vector v for quadratic form [batch, n]. Required if compute_quadratic=True.

        Returns:
            Tuple of:
                - X: Solution to S @ X = B, shape [batch, n, m] or [batch, n].
                - quadratic_result: v^T @ S^{-1} @ v if compute_quadratic, else None.
        """
        quadratic_result = None
        B_is_1d = B.dim() == 2

        if B_is_1d:
            B = B.unsqueeze(-1)

        try:
            L = torch.linalg.cholesky(S)
            X = torch.cholesky_solve(B, L)

            if compute_quadratic and quadratic_vec is not None:
                sol_v = torch.cholesky_solve(quadratic_vec.unsqueeze(-1), L)
                quadratic_result = (quadratic_vec.unsqueeze(1) @ sol_v).squeeze(-1).squeeze(-1)

        except RuntimeError as e:
            # Fallback: pseudo-inverse for non-positive-definite matrices
            # WARNING: This should be rare with proper regularization!
            import warnings
            warnings.warn(f"Cholesky decomposition failed, using pinv fallback. This indicates numerical instability! Error: {e}")
            S_inv = torch.linalg.pinv(S)
            X = S_inv @ B

            if compute_quadratic and quadratic_vec is not None:
                quadratic_result = (quadratic_vec.unsqueeze(1) @ S_inv @ quadratic_vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        if B_is_1d:
            X = X.squeeze(-1)

        return X, quadratic_result

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
        """Propagates nominal state using non-linear dynamics f(x_nom, u).

        Args:
            pos_w: Current position in world frame.
            vel_w: Current velocity in world frame.
            quat_b_to_w: Current body-to-world quaternion.
            gyro_bias_b: Current gyroscope bias in body frame.
            accel_bias_b: Current accelerometer bias in body frame.
            gyro_b_raw: Raw gyroscope measurements.
            accel_b_raw: Raw accelerometer measurements.

        Returns:
            Propagated state (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b).
        """
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Quaternion propagation via exponential map
        angle_change = gyro_b_corrected * self.dt
        delta_quat = small_angle_to_quaternion(angle_change)
        quat_b_to_w_new = quaternion_multiply(quat_b_to_w, delta_quat)

        # Trapezoidal integration for position and velocity
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_w = (rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w

        rot_mat_b_to_w_new = quaternion_to_rotation_matrix(quat_b_to_w_new)
        accel_w_new = (rot_mat_b_to_w_new @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w

        vel_w_new = vel_w + 0.5 * (accel_w + accel_w_new) * self.dt
        pos_w_new = pos_w + 0.5 * (vel_w + vel_w_new) * self.dt

        # Biases modeled as random walks
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
        """Predicts error covariance using linearized dynamics.

        Computes P_k|k-1 = F * P_k-1|k-1 * F^T + Q

        Args:
            P_error_covariance: Current 15x15 error covariance (P_k-1|k-1).
            quat_b_to_w: Current body-to-world quaternion.
            accel_b_raw: Raw accelerometer measurements.
            accel_bias_b: Current accelerometer bias.

        Returns:
            Predicted 15x15 error covariance (P_k|k-1).
        """
        batch_size = P_error_covariance.shape[0]

        F_error_matrix = (
            torch.eye(self.error_state_dim, device=self.device, dtype=P_error_covariance.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b

        # [a]_x for Jacobian
        accel_ssm = torch.zeros(batch_size, 3, 3, device=self.device, dtype=P_error_covariance.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # F: δp' = δp + δv*dt, δv' = δv - R[a]_x*δθ*dt - R*δa*dt, δθ' = δθ - δω*dt
        F_error_matrix[:, 0:3, 3:6] = torch.eye(3, device=self.device) * self.dt
        F_error_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        F_error_matrix[:, 6:9, 9:12] = -torch.eye(3, device=self.device) * self.dt

        # Q: trapezoidal integration
        Q_continuous = torch.diag(self.Q_diag).unsqueeze(0).expand(batch_size, -1, -1)
        Q_error_matrix = 0.5 * (
            F_error_matrix @ Q_continuous @ F_error_matrix.transpose(-2, -1) + Q_continuous
        ) * self.dt

        # P_k|k-1 = FPF^T + Q
        P_predicted = (
            F_error_matrix @ P_error_covariance @ F_error_matrix.transpose(-2, -1)
            + Q_error_matrix
        )

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
        """Performs bias-only measurement update (attitude handled by gravity alignment).

        This update observes accelerometer and gyroscope biases only.
        Attitude correction is handled separately by _apply_gravity_alignment_update
        to avoid double gravity observation.

        Args:
            P_error_pred: Predicted 15x15 error covariance (P_k|k-1).
            quat_b_to_w: Current body-to-world quaternion.
            accel_bias_b: Current accelerometer bias.
            gyro_bias_b: Current gyroscope bias.
            measurement: 6D sensor measurement (z_k).
            R_override: Optional measurement noise covariance override.
            gating_threshold: Optional Mahalanobis distance threshold.

        Returns:
            Tuple of (delta_x, P_error_new, innovation, mahalanobis_sq).
        """
        batch_size = P_error_pred.shape[0]

        H_error_matrix = torch.zeros(
            batch_size, self.obs_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype
        )

        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(-1)

        # H: bias-only observation (no attitude coupling)
        # Attitude correction handled by gravity alignment update
        # - Accel measures: gravity + accel_bias → observe accel_bias only
        # - Gyro measures: gyro_bias → observe gyro_bias
        H_error_matrix[:, 0:3, 12:15] = torch.eye(3, device=self.device)  # accel → accel_bias
        H_error_matrix[:, 3:6, 9:12] = torch.eye(3, device=self.device)   # gyro → gyro_bias

        # y = z - h(x)
        accel_pred = gravity_body + accel_bias_b
        gyro_pred = gyro_bias_b
        h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)
        innovation = measurement - h_predicted

        # Adaptive R: scale by |‖a‖ - g|
        if R_override is not None:
            R_noise_matrix = R_override + torch.eye(self.obs_dim, device=self.device) * 1e-6
        else:
            accel_meas = measurement[..., 0:3]
            accel_norm_diff = torch.abs(torch.norm(accel_meas, dim=-1, keepdim=True) - torch.norm(self.gravity_w))
            # CRITICAL: Clamp accel_norm_diff to prevent exponential explosion
            # Max diff of 20 m/s² (realistic for handwriting + safety margin)
            accel_norm_diff = torch.clamp(accel_norm_diff, max=20.0)
            scaling_factor = torch.exp(self.adaptive_gain * accel_norm_diff)
            # Additional safety: clamp scaling factor to [1.0, 1000.0]
            scaling_factor = torch.clamp(scaling_factor, min=1.0, max=1000.0)
            base_R = torch.diag_embed(F.softplus(self.R_diag) + 1e-6)
            R_noise_matrix = base_R.unsqueeze(0).expand(batch_size, -1, -1).clone()
            R_noise_matrix[..., 0:3, 0:3] *= scaling_factor.unsqueeze(-1)

        # S = HPH^T + R, K = PH^T S^-1
        S_matrix = H_error_matrix @ P_error_pred @ H_error_matrix.transpose(-2, -1) + R_noise_matrix
        # Add regularization to ensure S_matrix is well-conditioned for solve operations
        S_matrix = S_matrix + torch.eye(self.obs_dim, device=self.device) * 1e-5

        # Solve for Kalman gain and Mahalanobis distance using unified helper
        # K = P @ H^T @ S^{-1} = (S^{-1} @ H @ P^T)^T
        K_gain_T, mahalanobis_sq = self._solve_symmetric_system(
            S_matrix,
            H_error_matrix @ P_error_pred.transpose(-2, -1),
            compute_quadratic=True,
            quadratic_vec=innovation,
        )
        K_gain = K_gain_T.transpose(-2, -1)

        # δx = Ky
        delta_x = (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        # Clamp to reasonable range: [0, 1e6] prevents numerical issues in gating logic
        # Chi-square distribution for 6 DOF has p=0.99999 at ~27, so 1e6 is extremely conservative
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0, max=1e6)

        # P: Joseph form for numerical stability
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH = I_matrix - K_gain @ H_error_matrix
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gain @ R_noise_matrix @ K_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        # Mahalanobis gating
        if gating_threshold is not None:
             reject_mask = mahalanobis_sq > gating_threshold
             delta_x = torch.where(reject_mask.unsqueeze(-1), torch.zeros_like(delta_x), delta_x)
             P_error_new = torch.where(reject_mask.unsqueeze(-1).unsqueeze(-1), P_error_pred, P_error_new)

        return delta_x, P_error_new, innovation, mahalanobis_sq

    def _calculate_stationary_update(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        gyro_pred: torch.Tensor,
        tcn_zupt_prob: Optional[torch.Tensor] = None,
        use_zaru: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies stationary update: ZUPT (Zero-Velocity) and optionally ZARU (Zero Angular Rate).

        For ESKF-TCN: Applies both ZUPT and ZARU constraints when stationary.
        For classical ZUPT: Only applies velocity constraint.

        Args:
            vel_w_pred: Predicted velocity in world frame.
            P_error_pred: Predicted 15x15 error covariance.
            gyro_pred: Predicted gyroscope (bias estimate) in body frame.
            tcn_zupt_prob: Optional TCN-predicted zero-velocity probability [0,1].
            use_zaru: If True, also applies zero angular rate constraint (ESKF-TCN only).

        Returns:
            Tuple of (delta_x_stationary, P_error_new).
        """
        batch_size = P_error_pred.shape[0]

        if use_zaru:
            # ZUPT + ZARU: Constrain both velocity and angular rate
            meas_dim = 6
            H_stationary = torch.zeros(batch_size, meas_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
            H_stationary[:, 0:3, 3:6] = torch.eye(3, device=self.device)  # Velocity error δv
            H_stationary[:, 3:6, 9:12] = torch.eye(3, device=self.device)  # Gyro bias error δω_bias

            # Innovation: [velocity_error; gyro_error]
            innovation_stationary = torch.cat([-vel_w_pred, -gyro_pred], dim=-1)

            # R: adaptive via TCN probability for both velocity and gyro
            if tcn_zupt_prob is not None:
                if tcn_zupt_prob.ndim == 1:
                    tcn_zupt_prob = tcn_zupt_prob.unsqueeze(-1)
                min_R_val = self.zupt_noise_std**2
                max_R_val = min_R_val * 100
                clamped_prob = torch.clamp(tcn_zupt_prob, 0.01, 0.99)
                R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)
                if R_zupt_scaled_diag.shape[-1] == 1:
                    R_zupt_scaled_diag = R_zupt_scaled_diag.repeat(1, 3)

                # ZARU noise: slightly higher than ZUPT (gyro bias has more uncertainty)
                R_zaru_scaled_diag = R_zupt_scaled_diag * 2.0
                R_combined_diag = torch.cat([R_zupt_scaled_diag, R_zaru_scaled_diag], dim=-1)
                R_stationary_matrix = torch.diag_embed(R_combined_diag)
            else:
                R_zupt_base = self.get_R_zupt()
                R_zaru_base = R_zupt_base * 2.0
                R_combined = torch.cat([torch.diag(R_zupt_base), torch.diag(R_zaru_base)], dim=0)
                R_stationary_matrix = torch.diag(R_combined).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # ZUPT only: Constrain velocity
            meas_dim = 3
            H_stationary = torch.zeros(batch_size, meas_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
            H_stationary[:, :, 3:6] = torch.eye(3, device=self.device)

            innovation_stationary = -vel_w_pred

            # R_ZUPT: adaptive via TCN probability (high prob → low R)
            if tcn_zupt_prob is not None:
                if tcn_zupt_prob.ndim == 1:
                    tcn_zupt_prob = tcn_zupt_prob.unsqueeze(-1)
                min_R_val = self.zupt_noise_std**2
                max_R_val = min_R_val * 100
                clamped_prob = torch.clamp(tcn_zupt_prob, 0.01, 0.99)
                R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)
                if R_zupt_scaled_diag.shape[-1] == 1:
                    R_zupt_scaled_diag = R_zupt_scaled_diag.repeat(1, 3)
                R_stationary_matrix = torch.diag_embed(R_zupt_scaled_diag)
            else:
                R_stationary_matrix = self.get_R_zupt().unsqueeze(0).expand(batch_size, -1, -1)

        S_stationary_matrix = H_stationary @ P_error_pred @ H_stationary.transpose(-2, -1) + R_stationary_matrix
        # Regularization for numerical stability (consistent 1e-6 across all updates)
        S_stationary_matrix = S_stationary_matrix + torch.eye(meas_dim, device=self.device) * 1e-6

        # Solve for Kalman gain using unified helper
        K_stationary_gain_T, _ = self._solve_symmetric_system(
            S_stationary_matrix,
            H_stationary @ P_error_pred.transpose(-2, -1),
        )
        K_stationary_gain = K_stationary_gain_T.transpose(-2, -1)

        delta_x_stationary = (K_stationary_gain @ innovation_stationary.unsqueeze(-1)).squeeze(-1)

        # P: Joseph form
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH_stationary = I_matrix - K_stationary_gain @ H_stationary
        P_error_new = (
            ImKH_stationary @ P_error_pred @ ImKH_stationary.transpose(-2, -1)
            + K_stationary_gain @ R_stationary_matrix @ K_stationary_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x_stationary, P_error_new

    def _apply_tcn_velocity_correction(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        vel_corr_b: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        R_vel_diag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply TCN velocity correction as pseudo-measurement.

        Args:
            vel_w_pred: Predicted velocity in world frame [batch, 3].
            P_error_pred: Error covariance [batch, 15, 15].
            vel_corr_b: Velocity correction in body frame [batch, 3].
            quat_b_to_w: Body-to-world quaternion [batch, 4].
            R_vel_diag: Measurement noise diagonal [batch, 3], already processed.

        Returns:
            delta_x_tcn: State correction [batch, 15].
            P_error_new: Updated covariance [batch, 15, 15].
        """
        batch_size = P_error_pred.shape[0]
        dtype = P_error_pred.dtype

        # H matrix: velocity observation
        H_tcn = torch.zeros(batch_size, 3, self.error_state_dim, device=self.device, dtype=dtype)
        H_tcn[:, :, 3:6] = torch.eye(3, device=self.device, dtype=dtype)

        # Transform velocity correction to world frame
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        vel_corr_w = (rot_mat_b_to_w @ vel_corr_b.unsqueeze(-1)).squeeze(-1)

        # R matrix from pre-processed TCN covariance
        if R_vel_diag is not None:
            R_tcn_matrix = torch.diag_embed(R_vel_diag)
        else:
            R_tcn_matrix = torch.eye(3, device=self.device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1) * 1e-2

        # Kalman update using unified helper
        S_tcn = H_tcn @ P_error_pred @ H_tcn.transpose(-2, -1) + R_tcn_matrix
        # Add regularization for numerical stability (critical for Cholesky)
        S_tcn = S_tcn + torch.eye(3, device=self.device, dtype=dtype) * 1e-6
        K_T, _ = self._solve_symmetric_system(
            S_tcn,
            H_tcn @ P_error_pred.transpose(-2, -1),
        )
        K = K_T.transpose(-2, -1)

        delta_x_tcn = (K @ vel_corr_w.unsqueeze(-1)).squeeze(-1)

        # Joseph form covariance update
        I = torch.eye(self.error_state_dim, device=self.device, dtype=dtype)
        ImKH = I - K @ H_tcn
        P_error_new = ImKH @ P_error_pred @ ImKH.transpose(-2, -1) + K @ R_tcn_matrix @ K.transpose(-2, -1)
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x_tcn, P_error_new

    def _apply_gravity_alignment_update(
        self,
        P_error_pred: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gravity_measured_b: torch.Tensor,
        R_gravity_diag: torch.Tensor,
        accel_b_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply gravity alignment update to correct orientation error.

        Uses gravity observation to correct roll/pitch:
        - ZUPT: accelerometer as gravity measurement
        - Motion: TCN-predicted gravity direction

        Measurement model: h(q) = R(q)^T @ g_world
        Jacobian: H = -[g_body]_× (skew-symmetric)

        Args:
            P_error_pred: Error covariance [batch, 15, 15].
            quat_b_to_w: Body-to-world quaternion [batch, 4].
            gravity_measured_b: Gravity direction in body frame [batch, 3].
            R_gravity_diag: Measurement noise [batch, 3].
            accel_b_raw: Raw accelerometer for gating [batch, 3].

        Returns:
            delta_x_gravity: State correction [batch, 15].
            P_error_new: Updated covariance [batch, 15, 15].
        """
        batch_size = P_error_pred.shape[0]
        dtype = P_error_pred.dtype

        # Expected gravity in body frame from current orientation
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        rot_mat_w_to_b = rot_mat_b_to_w.transpose(-1, -2)
        gravity_expected_b = (rot_mat_w_to_b @ self.gravity_w.unsqueeze(-1)).squeeze(-1)
        gravity_expected_b_unit = F.normalize(gravity_expected_b, p=2, dim=-1, eps=1e-6)

        # Innovation: measured - expected (both unit vectors)
        innovation_gravity = gravity_measured_b - gravity_expected_b_unit

        # Gating: reject when accel magnitude far from gravity (high dynamics)
        if accel_b_raw is not None:
            accel_norm = torch.norm(accel_b_raw, dim=-1, keepdim=True)
            accel_min = Config.ESKFTCN.GRAVITY_ACCEL_MIN
            accel_max = Config.ESKFTCN.GRAVITY_ACCEL_MAX
            valid_mask = ((accel_norm > accel_min) & (accel_norm < accel_max)).squeeze(-1)
        else:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # H matrix: Jacobian of gravity w.r.t. orientation error
        # H[:, :, 6:9] = -[g_body]_× (skew-symmetric)
        H_gravity = torch.zeros(batch_size, 3, self.error_state_dim, device=self.device, dtype=dtype)
        g = gravity_expected_b_unit
        skew_g = torch.zeros(batch_size, 3, 3, device=self.device, dtype=dtype)
        skew_g[:, 0, 1] = -g[:, 2]
        skew_g[:, 0, 2] = g[:, 1]
        skew_g[:, 1, 0] = g[:, 2]
        skew_g[:, 1, 2] = -g[:, 0]
        skew_g[:, 2, 0] = -g[:, 1]
        skew_g[:, 2, 1] = g[:, 0]
        H_gravity[:, :, 6:9] = -skew_g

        # R matrix from TCN's adaptive covariance (per-sample diagonal)
        R_gravity_matrix = torch.diag_embed(R_gravity_diag)

        # Kalman update using unified helper
        S_gravity = H_gravity @ P_error_pred @ H_gravity.transpose(-2, -1) + R_gravity_matrix
        # Regularization for numerical stability (increased from 1e-8 to 1e-6)
        S_gravity = S_gravity + torch.eye(3, device=self.device, dtype=dtype) * 1e-6

        K_gravity_T, _ = self._solve_symmetric_system(
            S_gravity,
            H_gravity @ P_error_pred.transpose(-2, -1),
        )
        K_gravity = K_gravity_T.transpose(-2, -1)

        delta_x_gravity = (K_gravity @ innovation_gravity.unsqueeze(-1)).squeeze(-1)

        # Apply gating mask
        delta_x_gravity = torch.where(
            valid_mask.unsqueeze(-1), delta_x_gravity, torch.zeros_like(delta_x_gravity)
        )

        # Joseph form covariance update
        I_matrix = torch.eye(self.error_state_dim, device=self.device, dtype=dtype)
        ImKH = I_matrix - K_gravity @ H_gravity
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gravity @ R_gravity_matrix @ K_gravity.transpose(-2, -1)
        )
        P_error_new = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1), P_error_new, P_error_pred
        )
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x_gravity, P_error_new

    def inject_correction(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        delta_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Injects error state correction into nominal state.

        Args:
            pos_w: Current position in world frame.
            vel_w: Current velocity in world frame.
            quat_b_to_w: Current body-to-world quaternion.
            gyro_bias_b: Current gyroscope bias.
            accel_bias_b: Current accelerometer bias.
            delta_x: 15D error state.

        Returns:
            Corrected nominal state components.
        """
        d_pos_w, d_vel_w, d_theta_b, d_gyro_bias_b, d_accel_bias_b = delta_x.split([3, 3, 3, 3, 3], dim=-1)

        pos_w_new = pos_w + d_pos_w
        vel_w_new = vel_w + d_vel_w

        quat_b_to_w_new = quaternion_multiply(
            quat_b_to_w, small_angle_to_quaternion(d_theta_b)
        )
        quat_b_to_w_new = F.normalize(quat_b_to_w_new, p=2, dim=-1)

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
        # TCN outputs logits for BCEWithLogitsLoss compatibility; apply sigmoid for probability
        zupt_prob: Optional[torch.Tensor] = None
        if self.use_tcn_zupt and tcn_output is not None:
            zupt_prob = torch.sigmoid(tcn_output["zupt_prob"]).squeeze(-1)
            is_zupt = zupt_prob > 0.5
        elif self.use_zupt:
            is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        else:
            is_zupt = torch.zeros(accel_b_raw.shape[0], dtype=torch.bool, device=self.device)

        # Apply stationary update (ZUPT + ZARU) where applicable.
        if torch.any(is_zupt):
            zupt_mask = is_zupt
            zupt_prob_to_pass = None
            if self.use_tcn_zupt and zupt_prob is not None:
                zupt_prob_to_pass = zupt_prob[zupt_mask].unsqueeze(-1)

            # Compute gyro prediction (bias estimate)
            gyro_pred = gyro_bias_b_pred

            # Enable ZARU only for ESKF-TCN (when using TCN-based ZUPT)
            use_zaru = self.use_tcn_zupt

            delta_x_stationary, P_after_stationary = self._calculate_stationary_update(
                vel_w_pred[zupt_mask],
                P_error_pred[zupt_mask],
                gyro_pred[zupt_mask],
                tcn_zupt_prob=zupt_prob_to_pass,
                use_zaru=use_zaru
            )
            total_delta_x[zupt_mask] += delta_x_stationary
            P_error_final[zupt_mask] = P_after_stationary

        # Apply TCN-based corrections or standard measurement update.
        if tcn_output is not None:
            # Process TCN covariance once (shared by all updates)
            # covariance_R: [batch, 6] -> first 3 for accel/vel, last 3 for gyro
            tcn_cov_raw = tcn_output.get("covariance_R", None)
            if tcn_cov_raw is not None:
                tcn_cov_diag = F.softplus(tcn_cov_raw) + 1e-4
                tcn_cov_diag = torch.clamp(tcn_cov_diag, min=1e-4, max=3.0)
                R_accel = tcn_cov_diag[:, :3]  # For velocity and gravity updates
            else:
                tcn_cov_diag = None
                R_accel = None

            # 1. TCN Velocity Correction
            vel_corr_body = tcn_output["vel_corr"]
            if torch.any(is_zupt):
                vel_corr_body = torch.where(is_zupt.unsqueeze(-1), torch.zeros_like(vel_corr_body), vel_corr_body)

            delta_x_tcn, P_after_tcn = self._apply_tcn_velocity_correction(
                vel_w_pred, P_error_final, vel_corr_body, quat_b_to_w_pred,
                R_vel_diag=R_accel
            )
            total_delta_x += delta_x_tcn
            P_error_final = P_after_tcn

            # 2. Gravity Alignment Update
            # Skip during warmup period to let TCN stabilize before trusting its gravity predictions
            gravity_warmup_done = getattr(self, '_current_epoch', 0) >= Config.ESKFTCN.GRAVITY_WARMUP_EPOCHS
            if Config.ESKFTCN.USE_GRAVITY_ALIGNMENT and gravity_warmup_done and "gravity_b" in tcn_output and R_accel is not None:
                # Select gravity source: accel (ZUPT) or TCN prediction (motion)
                accel_gravity_b = F.normalize(accel_b_raw, p=2, dim=-1, eps=1e-6)
                gravity_measured_b = torch.where(
                    is_zupt.unsqueeze(-1).expand(-1, 3),
                    accel_gravity_b,
                    tcn_output["gravity_b"]
                )

                # Scale R by ZUPT state (lower R = higher trust during ZUPT)
                zupt_scale = Config.ESKFTCN.GRAVITY_R_STATIC / Config.ESKFTCN.GRAVITY_R_DYNAMIC
                R_gravity_diag = torch.where(
                    is_zupt.unsqueeze(-1).expand(-1, 3),
                    R_accel * zupt_scale,
                    R_accel
                )

                delta_x_gravity, P_after_gravity = self._apply_gravity_alignment_update(
                    P_error_final, quat_b_to_w_pred, gravity_measured_b,
                    R_gravity_diag=R_gravity_diag, accel_b_raw=accel_b_raw
                )
                total_delta_x += delta_x_gravity
                P_error_final = P_after_gravity

            # 3. Standard Measurement Update
            R_tcn_override = torch.diag_embed(tcn_cov_diag) if tcn_cov_diag is not None else None

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
            # Pure ESKF mode (without TCN)
            # Recompute necessary variables for innovation
            rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w_pred).transpose(-2, -1)
            gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(-1)

            accel_pred = gravity_body + accel_bias_b_pred
            gyro_pred = gyro_bias_b_pred
            h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)

            innovation_output = measurement - h_predicted

            # Virtual measurement update for Pure ESKF mode
            # Apply weak measurement updates during motion to reduce drift
            if self.use_virtual_measurements and not torch.all(is_zupt):
                # Compute motion level from accelerometer innovation
                accel_innovation = innovation_output[:, 0:3]
                motion_level = torch.norm(accel_innovation, dim=-1, keepdim=True)

                # Adaptive R: High-motion samples need stronger corrections (inverse relationship)
                # Analysis showed high-motion samples (40-46% time with gyro >0.5 rad/s) suffer more
                # from gyro bias drift, requiring stronger measurement updates
                # Range: [0.0001, 0.05] - low motion gets moderate R, high motion gets very low R
                motion_normalized = torch.clamp(motion_level / Config.GRAVITY_MAGNITUDE, 0.0, 1.0)
                motion_scale = 0.05 - 0.049 * motion_normalized  # Inverse: high motion → low R → strong corrections

                # Create adaptive R matrix (only for non-ZUPT samples)
                # R_diag: [6], motion_scale: [batch, 1] -> expand to [batch, 6]
                R_virtual_diag = F.softplus(self.R_diag).unsqueeze(0) * motion_scale + 1e-2  # [batch, 6]
                R_virtual = torch.diag_embed(R_virtual_diag)  # [batch, 6, 6]

                # Apply update only to non-ZUPT samples
                non_zupt_mask = ~is_zupt
                if torch.any(non_zupt_mask):
                    # Create a temporary delta_x tensor for the full batch
                    delta_x_virtual_full = torch.zeros_like(total_delta_x)
                    P_virtual_full = P_error_final.clone()

                    # Apply update to masked samples
                    delta_x_virtual, P_after_virtual, _, _ = self.update(
                        P_error_final[non_zupt_mask],
                        quat_b_to_w_pred[non_zupt_mask],
                        accel_bias_b_pred[non_zupt_mask],
                        gyro_bias_b_pred[non_zupt_mask],
                        measurement[non_zupt_mask],
                        R_override=R_virtual[non_zupt_mask],
                        gating_threshold=None  # No gating for virtual measurements
                    )

                    # Assign back to full batch tensor with learned correction weights
                    # Weights are learnable, initialized from Allan variance ratios
                    # Position/velocity weights fixed at 0 (no absolute reference)
                    # Orientation/bias weights learned to compensate for blind period drift
                    # Broadcasting: [num_masked, 15] * [15] -> [num_masked, 15]
                    delta_x_virtual_full[non_zupt_mask] = delta_x_virtual * self.virtual_meas_weights
                    P_virtual_full[non_zupt_mask] = P_after_virtual

                    # Add to total correction
                    total_delta_x += delta_x_virtual_full
                    P_error_final = P_virtual_full

        # --- 3. Error Injection ---
        if torch.any(total_delta_x != 0):
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = self.inject_correction(
                pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred, total_delta_x
            )
        else:
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = (pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred)

        # --- 3.5. Gradual Velocity Decay ---
        # Instead of hard reset at threshold, apply smooth decay proportional to zupt_prob.
        # This prevents discontinuities while still enforcing zero velocity at high confidence.
        #
        # decay_weight = clamp((zupt_prob - onset) / (1 - onset), 0, 1) ^ exponent
        # vel_new = vel * (1 - decay_weight)
        #
        # Examples (with onset=0.5, exponent=2):
        #   zupt_prob = 0.5  → decay = 0.0  → vel unchanged
        #   zupt_prob = 0.75 → decay = 0.25 → vel reduced by 25%
        #   zupt_prob = 1.0  → decay = 1.0  → vel = 0
        if self.use_tcn_zupt and zupt_prob is not None:
            onset = Config.ESKFTCN.ZUPT_DECAY_ONSET
            exponent = Config.ESKFTCN.ZUPT_DECAY_EXPONENT

            # Compute normalized probability above onset threshold
            prob_above_onset = (zupt_prob - onset) / (1.0 - onset + 1e-6)
            prob_above_onset = torch.clamp(prob_above_onset, min=0.0, max=1.0)

            # Apply exponent for smooth onset (quadratic by default)
            decay_weight = prob_above_onset ** exponent

            # Apply gradual decay: vel_new = vel * (1 - decay_weight)
            vel_w_new = vel_w_new * (1.0 - decay_weight.unsqueeze(-1))

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

    batch_size = 8192
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
    accel_dummy[0, 2] += Config.GRAVITY_MAGNITUDE  # Simulate near-static condition for ZUPT test
    gyro_dummy = torch.randn(batch_size, 3, device=device) * 0.01
    force_dummy = torch.rand(batch_size, 1, device=device)
    measurement_dummy = torch.cat([accel_dummy, gyro_dummy], dim=-1)

    # Dummy TCN output.
    tcn_out_dummy = {
        "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
        "covariance_R": torch.randn(batch_size, 6, device=device),
        "zupt_prob": torch.rand(batch_size, 1, device=device),
    }

    from cProfile import Profile

    profiler = Profile()
    profiler.run('(pos_w_out, vel_w_out, quat_b_to_w_out, gyro_bias_b_out, accel_bias_b_out, P_error_out, tcn_feats_out) = eskf.forward(pos_w_init, vel_w_init, quat_b_to_w_init, gyro_bias_b_init, accel_bias_b_init, P_error_init, gyro_dummy, accel_dummy, force_dummy, measurement_dummy, tcn_output=tcn_out_dummy)')

    from pstats import Stats

    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    # Run one forward pass.
    (pos_w_out, vel_w_out, quat_b_to_w_out,
     gyro_bias_b_out, accel_bias_b_out, P_error_out, tcn_feats_out) = eskf.forward(
        pos_w_init, vel_w_init, quat_b_to_w_init, gyro_bias_b_init, accel_bias_b_init, P_error_init,
        gyro_dummy, accel_dummy, force_dummy, measurement_dummy, tcn_output=tcn_out_dummy)

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

    # --- Test Pure ESKF mode with learnable virtual measurement weights ---
    print("\n--- Testing Pure ESKF with learnable virtual measurement weights ---")
    eskf_pure = ErrorStateKalmanFilter(
        dt=dt_val, device=device,
        use_tcn_zupt=False, use_zupt=False,  # Disable ZUPT to ensure virtual meas are used
        use_virtual_measurements=True  # Enable learnable weights
    )

    # Verify learnable weights exist and have correct shape
    print(f"Virtual measurement weights shape: {eskf_pure.virtual_meas_weights.shape}")
    print(f"Initial weights (from Allan variance):")
    print(f"  Position:    {eskf_pure.virtual_meas_weights[0:3].tolist()}")
    print(f"  Velocity:    {eskf_pure.virtual_meas_weights[3:6].tolist()}")
    print(f"  Orientation: {eskf_pure.virtual_meas_weights[6:9].tolist()}")
    print(f"  Gyro bias:   {eskf_pure.virtual_meas_weights[9:12].tolist()}")
    print(f"  Accel bias:  {eskf_pure.virtual_meas_weights[12:15].tolist()}")

    # Create motion data (not stationary) to trigger virtual measurements
    accel_motion = torch.randn(batch_size, 3, device=device) * 2.0  # High motion
    accel_motion[:, 2] += Config.GRAVITY_MAGNITUDE
    gyro_motion = torch.randn(batch_size, 3, device=device) * 0.5  # High angular rate
    measurement_motion = torch.cat([accel_motion, gyro_motion], dim=-1)

    # Test forward pass with virtual measurements
    # Note: Gradients flow through state outputs (quat, biases), NOT through P_error
    # because virtual_meas_weights multiplies delta_x which affects state, not covariance
    (pos_out, vel_out, quat_out, gyro_bias_out, accel_bias_out, P_error_pure, _) = eskf_pure.forward(
        pos_w_init, vel_w_init, quat_b_to_w_init, gyro_bias_b_init, accel_bias_b_init,
        P_error_init, gyro_motion, accel_motion, force_dummy, measurement_motion, tcn_output=None
    )
    print(f"Final P_error shape (pure ESKF): {P_error_pure.shape}")

    # Verify weights are learnable (have gradients)
    # Gradient flows: virtual_meas_weights -> delta_x -> inject_correction -> state outputs
    # Use quaternion/bias outputs since position/velocity weights are blocked (set to 0)
    loss = quat_out.sum() + gyro_bias_out.sum() + accel_bias_out.sum()
    loss.backward()
    has_grad = eskf_pure.virtual_meas_weights.grad is not None
    print(f"Weights have gradients: {has_grad}")
    if has_grad:
        grad = eskf_pure.virtual_meas_weights.grad
        print(f"Gradient norm: {grad.norm().item():.6f}")
        # Check which weight groups have non-zero gradients
        print(f"  Position grad (should be 0):    {grad[0:3].norm().item():.6f}")
        print(f"  Velocity grad (should be 0):    {grad[3:6].norm().item():.6f}")
        print(f"  Orientation grad:               {grad[6:9].norm().item():.6f}")
        print(f"  Gyro bias grad:                 {grad[9:12].norm().item():.6f}")
        print(f"  Accel bias grad:                {grad[12:15].norm().item():.6f}")
    else:
        print("No gradients (virtual measurement path not triggered)")

    print("Pure ESKF with learnable weights tested successfully.")
