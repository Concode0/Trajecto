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

from model.rotation_utils import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    small_angle_to_quaternion,
)
from model.zupt_detector import ZuptDetector


class ErrorStateKalmanFilter(nn.Module):
    """Implements a batch-aware Error-State Kalman Filter (ESKF) for IMU state estimation.

    The ESKF estimates a nominal state and a small error state.
    The **Nominal State** vector (x) is typically 16-dimensional:
    x_nom = [pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b]
            [  3  ,   3  ,      4     ,      3     ,      3     ] = 16 dimensions

    The **Error State** vector (δx) is typically 15-dimensional (using a minimal
    representation for orientation error, e.g., Euler angles or small rotation vector):
    δx = [δpos_w, δvel_w, δtheta_b, δgyro_bias_b, δaccel_bias_b]
         [   3  ,   3  ,     3    ,      3      ,      3      ] = 15 dimensions
    where δtheta_b represents the small angular error (e.g., a rotation vector).

    The filter proceeds in a predict-update-inject cycle.
    - **Prediction:** Propagates the nominal state through non-linear dynamics
      and the error state covariance through linearized dynamics.
    - **Update:** Corrects the error state and its covariance using linear
      Kalman filter equations with sensor measurements.
    - **Injection:** Injects the estimated error state back into the nominal
      state and resets the error state to zero.
    """

    def __init__(
        self,
        error_state_dim: int = 15,
        obs_dim: int = 6,  # 3 accel + 3 gyro
        dt: float = 0.0025,
        device: str = "cpu",
        zupt_window_size: int = 20,
        zupt_accel_threshold: float = 0.1,
        zupt_force_var_threshold: float = 0.01,
        zupt_force_delta_threshold: float = 0.1,
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
            zupt_window_size: Number of samples for ZUPT variance check.
            zupt_accel_threshold: Accelerometer variance threshold for ZUPT.
            zupt_force_var_threshold: Force variance threshold for ZUPT.
            zupt_force_delta_threshold: Force delta threshold for ZUPT.
            use_zupt: Boolean flag to enable or disable traditional ZUPT detection.
            use_tcn_zupt: Boolean flag to enable ZUPT decisions based on TCN output.
        """
        super().__init__()

        self.error_state_dim = error_state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device

        # --- Learnable Noise Parameters ---
        # These are registered as nn.Parameter to allow them to be optimized
        # during training, making the ESKF adaptive.
        # Q_diag: Diagonal elements of the process noise covariance matrix Q_error.
        # It represents the uncertainty introduced by the system model
        # (e.g., random walk in biases, unmodeled dynamics). `F.softplus`
        # ensures positivity, and `+ 1e-6` is a jitter for numerical stability.
        self.Q_diag = nn.Parameter(torch.ones(error_state_dim, device=device) * 1e-4)
        # R_diag: Diagonal elements of the measurement noise covariance matrix R.
        # It represents the uncertainty in the sensor measurements.
        self.R_diag = nn.Parameter(torch.ones(obs_dim, device=device) * 1e-2)

        # --- Physical Constants ---
        # Gravity vector in the world frame. Assuming Z-axis points upwards.
        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, 9.81], device=device))

        # --- ZUPT Detector ---
        # Component responsible for identifying zero-velocity intervals based on IMU data.
        self.zupt_detector = ZuptDetector(
            window_size=zupt_window_size,
            accel_var_threshold=zupt_accel_threshold,
            force_var_threshold=zupt_force_var_threshold,
            force_delta_threshold=zupt_force_delta_threshold,
            device=device,
        )
        self.use_zupt = use_zupt
        self.use_tcn_zupt = use_tcn_zupt
        # `zupt_R_factor` is a very small value, indicating high confidence in the
        # zero-velocity measurement when a ZUPT is detected.
        self.zupt_R_factor = 1e-6
        # `zupt_gravity_R_factor` can be used to specifically reduce measurement
        # uncertainty on the gravity component during ZUPT, but is not used in this
        # current implementation.
        self.zupt_gravity_R_factor = 1e-8

    def _make_symmetric(self, P_covariance: torch.Tensor) -> torch.Tensor:
        """Enforces symmetry on the covariance matrix P to prevent numerical instability.

        Covariance matrices are inherently symmetric and positive semi-definite.
        Due to floating-point arithmetic inaccuracies during numerous matrix operations,
        a covariance matrix might lose its perfect symmetry over many iterations.
        Enforcing symmetry by averaging with its transpose (0.5 * (P + P^T)) ensures
        that P remains mathematically valid, which is crucial for the stability
        and correctness of the Kalman filter, preventing filter divergence.

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
        # This yields a cleaner estimate of the true angular velocity and acceleration.
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Quaternion propagation (orientation update)
        # The quaternion derivative (q_dot) is related to angular velocity (omega).
        # q_dot = 0.5 * q * [0; omega] (quaternion multiplication)
        # This is the kinematic equation describing how the quaternion changes with angular velocity.
        q_dot = 0.5 * quaternion_multiply(
            quat_b_to_w,
            torch.cat(
                [
                    torch.zeros_like(gyro_b_corrected[..., :1]),
                    gyro_b_corrected,
                ],
                dim=-1,
            ),
        )
        # Integrate quaternion using Euler method: q_new = q_old + q_dot * dt
        quat_b_to_w_new = F.normalize(
            quat_b_to_w + q_dot * self.dt, p=2, dim=-1
        )  # Normalize to maintain unit quaternion constraint

        # Acceleration in World Frame
        # Rotate the corrected body-frame acceleration to the world frame.
        # Then, subtract the gravity vector (self.gravity_w) to get the
        # *pure kinematic acceleration* in the world frame (i.e., acceleration
        # due to motion, excluding gravity).
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_w = (
            rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)
        ).squeeze(-1) - self.gravity_w

        # Position and Velocity Integration (kinematic model)
        # Using discrete integration equations (second-order Euler/trapezoidal).
        # p_new = p_old + v_old * dt + 0.5 * a_old * dt^2
        pos_w_new = pos_w + vel_w * self.dt + 0.5 * accel_w * (self.dt**2)
        # v_new = v_old + a_old * dt
        vel_w_new = vel_w + accel_w * self.dt

        # Biases are modeled as random walk processes, so their nominal values do not
        # change based on this deterministic propagation. Their uncertainty
        # evolution is captured in the error state covariance (P_error) via process noise.
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

        # F_error: Error State Transition Matrix. This matrix linearizes the
        # error state dynamics and relates a small error in the current error
        # state to a small error in the next error state. It is derived from
        # the Jacobian of the nominal state propagation with respect to the error states.
        F_error_matrix = (
            torch.eye(self.error_state_dim, device=self.device, dtype=P_error_covariance.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Components for Jacobian calculations.
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Skew-symmetric matrix (cross-product matrix) of corrected acceleration (accel_b_corrected^x).
        # This is used in the Jacobian to represent how angular errors affect acceleration
        # measurements and thereby position/velocity errors.
        accel_ssm = torch.zeros(batch_size, 3, 3, device=self.device, dtype=P_error_covariance.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # Populate F_error_matrix based on linearized error state dynamics:
        # δpos_w depends on δvel_w: δpos_w_new = δpos_w + dt * δvel_w
        F_error_matrix[:, 0:3, 3:6] = torch.eye(3, device=self.device) * self.dt
        # δvel_w depends on δtheta_b (due to rotation of accel) and δaccel_bias_b
        # δvel_w_new = δvel_w + dt * (R_bw * (accel_b_corrected^x * δtheta_b - δaccel_bias_b))
        F_error_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        # δtheta_b depends on δgyro_bias_b
        # δtheta_b_new = δtheta_b - dt * δgyro_bias_b
        F_error_matrix[:, 6:9, 9:12] = -torch.eye(3, device=self.device) * self.dt
        # δgyro_bias_b and δaccel_bias_b are assumed to follow random walk,
        # so their errors directly propagate (identity) if no other dynamics.

        # Q_error: Process Noise Covariance Matrix for the error state.
        # It quantifies the uncertainty introduced by the system model itself,
        # including unmodeled dynamics and random walk in biases.
        # `F.softplus(self.Q_diag) + 1e-6` ensures diagonal elements are positive
        # (variances) and adds a small jitter (epsilon = 1e-6) to prevent Q_error
        # from becoming singular or zero, aiding numerical stability.
        Q_error_matrix = torch.diag_embed(F.softplus(self.Q_diag) + 1e-6)

        # Covariance Prediction: P_k|k-1 = F_error * P_k-1|k-1 * F_error^T + Q_error
        # This is the standard Kalman filter covariance prediction equation,
        # propagating the uncertainty from the previous time step.
        P_predicted = (
            F_error_matrix @ P_error_covariance @ F_error_matrix.transpose(-2, -1)
            + Q_error_matrix
        )

        # Symmetrization is applied to ensure the covariance matrix remains
        # perfectly symmetric, preventing potential numerical issues.
        return self._make_symmetric(P_predicted)

    def update(
        self,
        P_error_pred: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        measurement: torch.Tensor,
        R_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates the error state and its covariance based on the measurement.

        This function performs the measurement update (correction) step of the ESKF.

        Args:
            P_error_pred: The predicted 15x15 error covariance matrix (P_k|k-1).
            quat_b_to_w: Current nominal body-to-world quaternion (x_nom_k|k-1).
            accel_bias_b: Current nominal accelerometer bias (x_nom_k|k-1).
            gyro_bias_b: Current nominal gyroscope bias (x_nom_k|k-1).
            measurement: The actual 6-dimensional sensor measurement (z_k).
            R_override: Optional. An override for the measurement noise covariance matrix (R_k).
                If provided (e.g., from TCN), it replaces the learned R_diag.

        Returns:
            A tuple containing:
                - delta_x (torch.Tensor): The estimated 15-dimensional error state (δx_k|k).
                - P_error_new (torch.Tensor): The updated 15x15 error covariance matrix (P_k|k).
                - innovation (torch.Tensor): The measurement innovation (y_k).
        """
        batch_size = P_error_pred.shape[0]

        # H_error: Error Observation Matrix. This matrix linearizes the
        # measurement model h(x_nom) around the nominal state and relates
        # a small error in the error state to a small error in the predicted measurement.
        H_error_matrix = torch.zeros(
            batch_size, self.obs_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype
        )

        # Predicted gravity vector in body frame (g_body = R_wb * g_w).
        # This represents the expected specific force measured by an accelerometer when static.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w).transpose(
            -2, -1
        )
        gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(
            -1
        )

        # Skew-symmetric matrix of gravity_body (gravity_body^x).
        # Used in the Jacobian to represent how angular errors (δtheta_b) affect
        # the predicted accelerometer measurement.
        gravity_ssm = torch.zeros(batch_size, 3, 3, device=self.device, dtype=P_error_pred.dtype)
        gravity_ssm[:, 0, 1] = -gravity_body[:, 2]
        gravity_ssm[:, 0, 2] = gravity_body[:, 1]
        gravity_ssm[:, 1, 0] = gravity_body[:, 2]
        gravity_ssm[:, 1, 2] = -gravity_body[:, 0]
        gravity_ssm[:, 2, 0] = -gravity_body[:, 1]
        gravity_ssm[:, 2, 1] = gravity_body[:, 0]

        # Populate H_error_matrix:
        # H(0:3, 6:9) - Errors in orientation (δtheta_b) affect predicted accelerometer measurement.
        H_error_matrix[:, 0:3, 6:9] = gravity_ssm
        # H(0:3, 12:15) - Errors in accelerometer bias (δaccel_bias_b) directly affect predicted accel measurement.
        H_error_matrix[:, 0:3, 12:15] = torch.eye(3, device=self.device)
        # H(3:6, 9:12) - Errors in gyroscope bias (δgyro_bias_b) directly affect predicted gyro measurement.
        H_error_matrix[:, 3:6, 9:12] = torch.eye(3, device=self.device)

        # Predicted Measurement (h_pred): Calculate expected sensor readings from nominal state.
        # This is equivalent to self._measurement_function(nominal_state) from standard EKF,
        # predicting what the sensor should read given the current nominal state and biases.
        accel_pred = gravity_body + accel_bias_b
        gyro_pred = gyro_bias_b
        h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)

        # Innovation (Measurement Residual): y_k = z_k - h(x_nom_k|k-1)
        # This is the difference between the actual sensor measurement and what the
        # filter predicted it would measure based on its current nominal state and biases.
        innovation = measurement - h_predicted

        # R_noise: Measurement Noise Covariance Matrix.
        # It models the uncertainty in the sensor measurements.
        if R_override is not None:
            # If a TCN provides R, use it. Adding a small jitter for stability.
            R_noise_matrix = R_override + torch.eye(self.obs_dim, device=self.device) * 1e-4
        else:
            # Otherwise, use the learned R_diag. `F.softplus` ensures positivity,
            # and `+ 1e-4` is a jitter (epsilon) for numerical stability.
            R_noise_matrix = torch.diag_embed(F.softplus(self.R_diag) + 1e-4)

        # S: Innovation Covariance Matrix: S_k = H_k * P_k|k-1 * H_k^T + R_k
        # This matrix represents the covariance of the innovation.
        S_matrix = (
            H_error_matrix @ P_error_pred @ H_error_matrix.transpose(-2, -1)
            + R_noise_matrix
        )

        # Jitter: A small amount of noise (epsilon) added to S_matrix's diagonal.
        # This ensures S_matrix is well-conditioned (full rank) and numerically
        # stable for inversion, preventing potential issues if H*P*H.T becomes
        # singular or very small, which can happen in certain conditions
        # (e.g., very confident predictions, poor measurement geometry).
        # It safeguards against division by zero in Kalman gain calculation.
        jitter = torch.eye(self.obs_dim, device=self.device) * 1e-4
        S_matrix = S_matrix + jitter

        # K: Kalman Gain: K_k = P_k|k-1 * H_k^T * S_k^-1
        # The Kalman Gain determines how much the filter "trusts" the new measurement
        # innovation relative to its prediction uncertainty. A larger gain means more
        # reliance on the measurement. `torch.linalg.solve(A, B)` computes `A^-1 * B`
        # and is numerically more stable and efficient than explicitly computing the
        # inverse of S_matrix, especially for ill-conditioned matrices.
        K_gain = torch.linalg.solve(
            S_matrix, H_error_matrix @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # Error State Update: δx_k|k = K_k * y_k
        # This is the estimated correction to the error state.
        delta_x = (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)

        # I: Identity matrix for the error state dimension.
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH = I_matrix - K_gain @ H_error_matrix

        # Covariance Update (Joseph Form): P_k|k = (I - K_k*H_k) * P_k|k-1 * (I - K_k*H_k)^T + K_k * R_k * K_k^T
        # The Joseph Form is highly preferred for its numerical stability. It
        # guarantees that the updated covariance matrix P_error_new remains
        # symmetric and positive semi-definite (PSD), even in the presence of
        # floating-point inaccuracies. This is crucial to prevent filter
        # divergence, which can occur if P loses its PSD property. The
        # standard "naive" update (P = (I - KH)P_pred) does not guarantee PSD.
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gain @ R_noise_matrix @ K_gain.transpose(-2, -1)
        )

        # Explicit Symmetrization: Although Joseph form helps maintain symmetry,
        # an explicit symmetrization step ensures perfect symmetry and helps
        # correct any residual floating-point errors.
        P_error_new = self._make_symmetric(P_error_new)

        return delta_x, P_error_new, innovation

    def _calculate_zupt_update(
        self, vel_w_pred: torch.Tensor, P_error_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the error-state correction from a Zero-Velocity Update (ZUPT) pseudo-measurement.

        When a ZUPT condition is detected, this acts as an additional measurement
        that the velocity in the world frame should be zero (i.e., z_ZUPT = 0).

        Args:
            vel_w_pred: Predicted nominal velocity in world frame.
            P_error_pred: The predicted 15x15 error covariance matrix (P_k|k-1).

        Returns:
            A tuple containing:
                - delta_x_zupt (torch.Tensor): The estimated error state from ZUPT.
                - P_error_new (torch.Tensor): The updated error covariance matrix.
        """
        batch_size = P_error_pred.shape[0]

        # H_ZUPT: Measurement Jacobian for ZUPT. It selects the velocity error
        # (δvel_w) from the error state vector, indicating that these are the
        # quantities being "measured" (i.e., forced to zero).
        # H_ZUPT = [0_3x3 | I_3x3 | 0_3x9] - a 3x15 matrix where I_3x3 is the identity.
        H_zupt = torch.zeros(
            batch_size, 3, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype
        )
        H_zupt[:, :, 3:6] = torch.eye(3, device=self.device)

        # R_ZUPT: Measurement noise covariance for ZUPT. A very small diagonal
        # matrix (self.zupt_R_factor) implies high confidence in the
        # zero-velocity measurement.
        R_zupt_matrix = (
            torch.eye(3, device=self.device).unsqueeze(0) * self.zupt_R_factor
        )

        # Innovation for ZUPT: y_ZUPT = z_ZUPT - h(x_nom) = 0 - vel_w_pred
        # This is the difference between the expected (zero) velocity and the
        # filter's predicted nominal velocity.
        innovation_zupt = -vel_w_pred

        # S_ZUPT: Innovation Covariance for ZUPT.
        S_zupt_matrix = H_zupt @ P_error_pred @ H_zupt.transpose(-2, -1) + R_zupt_matrix

        # Jitter: Added to S_ZUPT for numerical stability during inversion.
        # Similar to the standard update, this prevents S_ZUPT from being singular.
        jitter = torch.eye(3, device=self.device) * 1e-4
        S_zupt_matrix = S_zupt_matrix + jitter

        # K_ZUPT: Kalman Gain for ZUPT.
        K_zupt_gain = torch.linalg.solve(
            S_zupt_matrix, H_zupt @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # Error State Update from ZUPT: δx_ZUPT = K_ZUPT * y_ZUPT
        delta_x_zupt = (K_zupt_gain @ innovation_zupt.unsqueeze(-1)).squeeze(-1)

        # Covariance Update (Joseph Form) for ZUPT.
        # Joseph form is used here for the same numerical stability reasons as
        # in the standard `update` method.
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH_zupt = I_matrix - K_zupt_gain @ H_zupt

        P_error_new = (
            ImKH_zupt @ P_error_pred @ ImKH_zupt.transpose(-2, -1)
            + K_zupt_gain @ R_zupt_matrix @ K_zupt_gain.transpose(-2, -1)
        )

        # Symmetrization.
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

        This treats the TCN's predicted velocity correction as a pseudo-measurement
        that refines the filter's velocity estimate.

        Args:
            vel_w_pred: Predicted nominal velocity in world frame.
            P_error_pred: The predicted 15x15 error covariance matrix.
            vel_corr_b: Velocity correction predicted by the TCN in the body frame.
            quat_b_to_w: Current nominal body-to-world quaternion.

        Returns:
            A tuple containing:
                - delta_x_tcn (torch.Tensor): The estimated error state from TCN correction.
                - P_error_new (torch.Tensor): The updated error covariance matrix.
        """
        batch_size = P_error_pred.shape[0]

        # H_TCN: Measurement Jacobian for TCN velocity correction.
        # Similar to H_ZUPT, it selects the velocity error (δvel_w) from the
        # error state vector, as the TCN is providing a correction related to velocity.
        H_tcn = torch.zeros(
            batch_size, 3, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype
        )
        H_tcn[:, :, 3:6] = torch.eye(3, device=self.device)

        # Convert TCN's body-frame velocity correction to world frame.
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        vel_corr_w = (rot_mat_b_to_w @ vel_corr_b.unsqueeze(-1)).squeeze(-1)

        # Innovation: This is effectively the TCN's predicted velocity correction.
        # It's treated as a measurement that `vel_w_pred` should be adjusted by `vel_corr_w`.
        innovation_tcn = vel_corr_w  # Effectively z_tcn = vel_corr_w, h(x_nom) = 0.
                                    # Or viewed as y = (vel_w_pred + vel_corr_w) - vel_w_pred
                                    # which simplifies to vel_corr_w

        # R_TCN: Measurement noise covariance for TCN correction.
        # This represents the uncertainty in the TCN's correction. It could
        # be learned by another TCN head or set as a fixed value.
        R_tcn_matrix = torch.eye(3, device=self.device).unsqueeze(0) * 1e-4

        # S_TCN: Innovation Covariance for TCN.
        S_tcn_matrix = H_tcn @ P_error_pred @ H_tcn.transpose(-2, -1) + R_tcn_matrix

        # K_TCN: Kalman Gain for TCN correction.
        K_tcn_gain = torch.linalg.solve(
            S_tcn_matrix, H_tcn @ P_error_pred.transpose(-2, -1)
        ).transpose(-2, -1)

        # Error State Update: δx_TCN = K_TCN * y_TCN
        delta_x_tcn = (K_tcn_gain @ innovation_tcn.unsqueeze(-1)).squeeze(-1)

        # Covariance Update (Joseph Form) for TCN.
        I_matrix = torch.eye(self.error_state_dim, device=self.device)
        ImKH_tcn = I_matrix - K_tcn_gain @ H_tcn

        P_error_new = (
            ImKH_tcn @ P_error_pred @ ImKH_tcn.transpose(-2, -1)
            + K_tcn_gain @ R_tcn_matrix @ K_tcn_gain.transpose(-2, -1)
        )

        # Symmetrization.
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

        After the error state δx is estimated, it is used to correct the nominal
        state components. The error state is then conceptually reset to zero
        for the next iteration. This is a fundamental step in the ESKF to
        prevent error accumulation and maintain accuracy over time.

        Args:
            pos_w: Current nominal position in world frame.
            vel_w: Current nominal velocity in world frame.
            quat_b_to_w: Current nominal body-to-world quaternion.
            gyro_bias_b: Current nominal gyroscope bias in body frame.
            accel_bias_b: Current nominal accelerometer bias in body frame.
            delta_x: The estimated 15-dimensional error state (δx_k|k).

        Returns:
            A tuple containing the corrected nominal state components
            (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new).
        """
        # Decompose the estimated error state delta_x into its components.
        d_pos_w, d_vel_w, d_theta_b, d_gyro_bias_b, d_accel_bias_b = delta_x.split(
            [3, 3, 3, 3, 3], dim=-1
        )

        # Apply position and velocity corrections by simple addition.
        pos_w_new = pos_w + d_pos_w
        vel_w_new = vel_w + d_vel_w

        # Apply orientation correction.
        # Small angular errors (d_theta_b) are typically represented as a rotation
        # vector. This is converted into a small quaternion, which is then
        # multiplicatively applied to the nominal quaternion.
        quat_b_to_w_new = quaternion_multiply(
            quat_b_to_w, small_angle_to_quaternion(d_theta_b)
        )
        # Re-normalize to ensure it remains a unit quaternion after correction.
        quat_b_to_w_new = F.normalize(quat_b_to_w_new, p=2, dim=-1)

        # Apply bias corrections by simple addition.
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
        """Executes one full predict-update-inject cycle of the ESKF, integrating TCN corrections.

        Args:
            pos_w: Current nominal position in world frame.
            vel_w: Current nominal velocity in world frame.
            quat_b_to_w: Current nominal body-to-world quaternion.
            gyro_bias_b: Current nominal gyroscope bias in body frame.
            accel_bias_b: Current nominal accelerometer bias in body frame.
            P_error: Current 15x15 error covariance matrix.
            gyro_b_raw: Raw gyroscope measurements at time k.
            accel_b_raw: Raw accelerometer measurements at time k.
            force_raw: Raw force sensor measurement at time k, used for ZUPT.
            measurement: The actual 6-dimensional sensor measurement (z_k).
            tcn_output: Optional. Dictionary containing TCN predictions,
                potentially including 'vel_corr', 'covariance_R', and 'zupt_prob'.

        Returns:
            A tuple containing:
                - pos_w_new (torch.Tensor): Updated nominal position.
                - vel_w_new (torch.Tensor): Updated nominal velocity.
                - quat_b_to_w_new (torch.Tensor): Updated nominal quaternion.
                - gyro_bias_b_new (torch.Tensor): Updated nominal gyroscope bias.
                - accel_bias_b_new (torch.Tensor): Updated nominal accelerometer bias.
                - P_error_final (torch.Tensor): The final updated error covariance matrix.
                - tcn_features (Dict[str, torch.Tensor]): A dictionary of features
                    extracted from the filter's output for potential use by a TCN.
        """
        # --- 1. Prediction Step ---
        # Propagate the nominal state forward using IMU measurements.
        pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred = (
            self._propagate_nominal_state(
                pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, gyro_b_raw, accel_b_raw
            )
        )
        # Predict the error state covariance matrix.
        P_error_pred = self.predict(
            P_error, quat_b_to_w, accel_b_raw, accel_bias_b
        )

        # --- 2. Update Step Initialization ---
        P_error_final = P_error_pred  # Start with the predicted covariance
        # Initialize containers for innovation and total error state correction.
        innovation_output = torch.zeros(
            pos_w.shape[0], self.obs_dim, device=self.device, dtype=pos_w.dtype
        )
        total_delta_x = torch.zeros(
            pos_w.shape[0], self.error_state_dim, device=self.device, dtype=pos_w.dtype
        )

        # --- 3. ZUPT (Zero-Velocity Update) Determination ---
        # Decide whether ZUPT should be applied for each sample in the batch.
        # This decision can be based on a traditional detector or TCN output.
        is_zupt: torch.Tensor  # Declare type for clarity
        if self.use_tcn_zupt and tcn_output is not None:
            # If `use_tcn_zupt` is True, ZUPT is triggered if TCN's `zupt_prob` is above a threshold.
            is_zupt = tcn_output["zupt_prob"].squeeze(-1) > 0.5
        elif self.use_zupt:
            # If `use_tcn_zupt` is False but `use_zupt` is True, use the traditional ZUPT detector.
            is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        else:
            # If ZUPT is disabled, no samples are considered ZUPT.
            is_zupt = torch.zeros(
                accel_b_raw.shape[0], dtype=torch.bool, device=self.device
            )

        # --- 4. ZUPT Correction (if applicable) ---
        if torch.any(is_zupt):
            # Apply ZUPT correction only to the samples identified as being in ZUPT.
            zupt_mask = is_zupt
            delta_x_zupt, P_after_zupt = self._calculate_zupt_update(
                vel_w_pred[zupt_mask], P_error_pred[zupt_mask]
            )
            total_delta_x[zupt_mask] += delta_x_zupt  # Accumulate error corrections
            P_error_final[zupt_mask] = P_after_zupt  # Update covariance for ZUPT samples

        # --- 5. TCN-based Corrections or Standard Measurement Update ---
        if tcn_output is not None:
            # a) TCN Velocity Correction:
            # If TCN predicts a velocity correction (`vel_corr`), apply it as a pseudo-measurement.
            vel_corr_body = tcn_output["vel_corr"]

            # If ZUPT is active for a sample, override TCN's velocity correction to zero for that sample,
            # as ZUPT already enforces zero velocity and we don't want conflicting corrections.
            if torch.any(is_zupt):
                vel_corr_body = torch.where(
                    is_zupt.unsqueeze(-1),
                    torch.zeros_like(vel_corr_body),
                    vel_corr_body,
                )

            delta_x_tcn, P_after_tcn = self._apply_tcn_velocity_correction(
                vel_w_pred, P_error_final, vel_corr_body, quat_b_to_w_pred
            )
            total_delta_x += delta_x_tcn  # Accumulate error corrections
            P_error_final = P_after_tcn  # Update covariance after TCN velocity correction

            # b) Standard Measurement Update (potentially with TCN-adapted R):
            # If TCN provides `covariance_R`, use it to override the standard measurement noise `R`.
            tcn_cov_diag = F.softplus(tcn_output["covariance_R"])
            R_tcn_override = torch.diag_embed(tcn_cov_diag)

            delta_x_up, P_after_up, innovation = self.update(
                P_error_final,
                quat_b_to_w_pred,
                accel_bias_b_pred,
                gyro_bias_b_pred,
                measurement,
                R_override=R_tcn_override,
            )

            total_delta_x += delta_x_up  # Accumulate error corrections
            P_error_final = P_after_up  # Final covariance after measurement update
            innovation_output = innovation
        else:
            # If no TCN output is provided, perform a standard measurement update with the learned R.
            delta_x_up, P_after_up, innovation = self.update(
                P_error_final,
                quat_b_to_w_pred,
                accel_bias_b_pred,
                gyro_bias_b_pred,
                measurement,
                R_override=None,
            )

            total_delta_x += delta_x_up
            P_error_final = P_after_up
            innovation_output = innovation

        # --- 6. Error Injection ---
        # Inject all accumulated error state corrections into the nominal state.
        # This corrects the nominal state and conceptually resets the error state to zero.
        if torch.any(total_delta_x != 0):  # Only inject if there's a correction
            (
                pos_w_new,
                vel_w_new,
                quat_b_to_w_new,
                gyro_bias_b_new,
                accel_bias_b_new,
            ) = self.inject_correction(
                pos_w_pred,
                vel_w_pred,
                quat_b_to_w_pred,
                gyro_bias_b_pred,
                accel_bias_b_pred,
                total_delta_x,
            )
        else:  # No correction was made, nominal state remains as predicted.
            (
                pos_w_new,
                vel_w_new,
                quat_b_to_w_new,
                gyro_bias_b_new,
                accel_bias_b_new,
            ) = (
                pos_w_pred,
                vel_w_pred,
                quat_b_to_w_pred,
                gyro_bias_b_pred,
                accel_bias_b_pred,
            )

        # --- 7. Assemble Features for TCN ---
        # Extract features from the updated state for the TCN in the next time step.
        rot_mat_world_to_body = quaternion_to_rotation_matrix(
            quat_b_to_w_new
        ).transpose(-2, -1)
        vel_body = (rot_mat_world_to_body @ vel_w_new.unsqueeze(-1)).squeeze(-1)

        tcn_features: Dict[str, torch.Tensor] = {
            "body_velocity": vel_body,
            "zupt_flag": is_zupt.float().unsqueeze(-1),  # Convert boolean to float tensor
            "innovation": innovation_output,
        }

        return (
            pos_w_new,
            vel_w_new,
            quat_b_to_w_new,
            gyro_bias_b_new,
            accel_bias_b_new,
            P_error_final,
            tcn_features,
        )


if __name__ == "__main__":
    # Test case to verify functionality and tensor shapes of the ErrorStateKalmanFilter.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dt_val = 0.01

    # Test with TCN integration (simulated TCN output)
    # Instantiate ESKF with TCN-based ZUPT enabled.
    eskf = ErrorStateKalmanFilter(dt=dt_val, device=device, use_tcn_zupt=True)

    batch_size = 4
    # Initial nominal state components: all zeros except quaternion 'w' component.
    pos_w_init = torch.zeros(batch_size, 3, device=device)
    vel_w_init = torch.zeros(batch_size, 3, device=device)
    quat_b_to_w_init = torch.zeros(batch_size, 4, device=device)
    quat_b_to_w_init[:, 0] = 1.0  # Identity quaternion (no rotation)
    gyro_bias_b_init = torch.zeros(batch_size, 3, device=device)
    accel_bias_b_init = torch.zeros(batch_size, 3, device=device)
    # Initial error covariance matrix: Diagonal with some uncertainty.
    P_error_init = (
        torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1
    )

    # Create dummy IMU data for one time step.
    # Simulate accelerometer reading close to gravity for first sample to test ZUPT.
    accel_dummy = torch.randn(batch_size, 3, device=device) * 0.1
    accel_dummy[0, 2] += 9.81  # First sample's Z-accel is near gravity
    gyro_dummy = torch.randn(batch_size, 3, device=device) * 0.01
    force_dummy = torch.rand(batch_size, 1, device=device)  # Random force data

    # Combine accel and gyro for the measurement vector.
    measurement_dummy = torch.cat([accel_dummy, gyro_dummy], dim=-1)

    # Dummy TCN output, simulating what a TCN would predict.
    tcn_out_dummy = {
        "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
        "covariance_R": torch.randn(batch_size, 6, device=device),
        "zupt_prob": torch.rand(batch_size, 1, device=device),  # ZUPT probability
    }

    # Run one forward pass of the ESKF with TCN output.
    (
        pos_w_out,
        vel_w_out,
        quat_b_to_w_out,
        gyro_bias_b_out,
        accel_bias_b_out,
        P_error_out,
        tcn_feats_out,
    ) = eskf.forward(
        pos_w_init,
        vel_w_init,
        quat_b_to_w_init,
        gyro_bias_b_init,
        accel_bias_b_init,
        P_error_init,
        gyro_dummy,
        accel_dummy,
        force_dummy,
        measurement_dummy,
        tcn_output=tcn_out_dummy,
    )

    print(f"\nUpdated Nominal Position (p) shape: {pos_w_out.shape}")
    print(f"Updated Nominal Velocity (v) shape: {vel_w_out.shape}")
    print(f"Updated Nominal Quaternion (q) shape: {quat_b_to_w_out.shape}")
    print(f"Updated Nominal Gyro Bias (bg) shape: {gyro_bias_b_out.shape}")
    print(f"Updated Nominal Accel Bias (ba) shape: {accel_bias_b_out.shape}")
    print(f"Final Error Covariance (P) shape: {P_error_out.shape}")
    print("TCN Features:")
    for k, v in tcn_feats_out.items():
        print(f"  - '{k}': {v.shape}")
    print("\nESKF with TCN integration tested successfully.")

    # Test without TCN ZUPT (using traditional ZUPT detector)
    print("\n--- Testing ESKF with traditional ZUPT ---")
    eskf_no_tcn_zupt = ErrorStateKalmanFilter(
        dt=dt_val, device=device, use_tcn_zupt=False, use_zupt=True
    )
    (
        pos_w_out_no_tcn,
        vel_w_out_no_tcn,
        quat_b_to_w_out_no_tcn,
        gyro_bias_b_out_no_tcn,
        accel_bias_b_out_no_tcn,
        P_error_out_no_tcn,
        tcn_feats_out_no_tcn,
    ) = eskf_no_tcn_zupt.forward(
        pos_w_init,
        vel_w_init,
        quat_b_to_w_init,
        gyro_bias_b_init,
        accel_bias_b_init,
        P_error_init,
        gyro_dummy,
        accel_dummy,
        force_dummy,
        measurement_dummy,
        tcn_output=None,  # No TCN output used for this test
    )
    print(f"Final P_error shape (no TCN ZUPT): {P_error_out_no_tcn.shape}")
    print("ESKF with traditional ZUPT tested successfully.")
