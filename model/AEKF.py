"""
AEKF (Adaptive Extended Kalman Filter) for 3D Pen Trajectory Reconstruction.
This module provides a batch-aware EKF with a robust, stateful Zero-Velocity Update (ZUPT)
mechanism to mitigate drift in inertial navigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from zupt_detector import ZuptDetector
from rotation_utils import quaternion_multiply, quaternion_to_rotation_matrix

class ExtendedKalmanFilter(nn.Module):
    """Implements a batch-aware, adaptive Extended Kalman Filter for IMU state estimation.

    The filter estimates a 16-dimensional state vector:
    - Position in the world frame (3)
    - Velocity in the world frame (3)
    - Orientation quaternion from body to world frame (4)
    - Gyroscope bias in the body frame (3)
    - Accelerometer bias in the body frame (3)
    """

    def __init__(self, 
                 state_dim: int = 16,
                 obs_dim: int = 6,  # 3 accel + 3 gyro
                 dt: float = 0.01,
                 device: str = 'cpu',
                 zupt_window_size: int = 20,
                 zupt_accel_threshold: float = 0.1,
                 zupt_force_var_threshold: float = 0.01,
                 zupt_force_delta_threshold: float = 0.1):
        """Initializes the EKF module.
        
        Args:
            state_dim (int): Dimension of the state vector (16).
            obs_dim (int): Dimension of the observation vector (6).
            dt (float): Time step in seconds.
            device (str): Compute device.
            zupt_window_size (int): Number of samples for ZUPT variance check.
            zupt_accel_threshold (float): Accel variance threshold for ZUPT.
            zupt_force_var_threshold (float): Force variance threshold for ZUPT.
            zupt_force_delta_threshold (float): Force delta threshold for ZUPT.
        """
        super(ExtendedKalmanFilter, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device

        # --- Learnable Noise Parameters ---
        self.register_parameter('Q_diag', nn.Parameter(torch.ones(state_dim, device=device) * 1e-4))
        self.register_parameter('gyro_bias_rw_std', nn.Parameter(torch.tensor(1e-5, device=device)))
        self.register_parameter('accel_bias_rw_std', nn.Parameter(torch.tensor(1e-5, device=device)))
        self.register_parameter('raw_R_diag', nn.Parameter(torch.ones(obs_dim, device=device) * 1e-2))

        # --- Physical Constants ---
        self.register_buffer('gravity_w', torch.tensor([0., 0., 9.81], device=device))

        # --- ZUPT Detector ---
        self.zupt_detector = ZuptDetector(
            window_size=zupt_window_size,
            accel_var_threshold=zupt_accel_threshold,
            force_var_threshold=zupt_force_var_threshold,
            force_delta_threshold=zupt_force_delta_threshold,
            device=device
        )
        
        # --- Tuning Factors ---
        self.adaptive_R_factor  = 0.1
        self.zupt_R_factor       = 1e-6
        self.zupt_gravity_R_factor = 1e-8

    def _transform_body_to_world(self, vector_b: torch.Tensor, quat_b_to_w: torch.Tensor) -> torch.Tensor:
        """Transforms a batch of vectors from the body frame to the world frame."""
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        return (rot_mat_b_to_w @ vector_b.unsqueeze(-1)).squeeze(-1)

    def _state_transition_function(self, state: torch.Tensor, gyro_b_raw: torch.Tensor, accel_b_raw: torch.Tensor) -> torch.Tensor:
        """Predicts the next state based on IMU inputs and kinematic equations."""
        pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b = state.split([3, 3, 4, 3, 3], dim=-1)
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b
        q_dot = 0.5 * quaternion_multiply(quat_b_to_w, torch.cat([torch.zeros_like(gyro_b_corrected[..., :1]), gyro_b_corrected], dim=-1))
        quat_b_to_w_new = F.normalize(quat_b_to_w + q_dot * self.dt, p=2, dim=-1)
        accel_w = self._transform_body_to_world(accel_b_corrected, quat_b_to_w) - self.gravity_w
        pos_w_new = pos_w + vel_w * self.dt + 0.5 * accel_w * self.dt**2
        vel_w_new = vel_w + accel_w * self.dt
        return torch.cat([pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b, accel_bias_b], dim=-1)

    def _measurement_function(self, state: torch.Tensor) -> torch.Tensor:
        """Predicts the expected sensor measurement (accel, gyro) for a given state."""
        _pos_w, _vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b = state.split([3, 3, 4, 3, 3], dim=-1)
        gyro_pred = gyro_bias_b
        rot_mat_w_to_b = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        accel_pred = (rot_mat_w_to_b @ self.gravity_w.unsqueeze(0).T).squeeze(-1) + accel_bias_b
        return torch.cat([accel_pred, gyro_pred], dim=-1)

    def _compute_jacobian_F(self, state: torch.Tensor, gyro_b_raw: torch.Tensor, accel_b_raw: torch.Tensor) -> torch.Tensor:
        """Computes the Jacobian of the state transition function (F = ∂f/∂x)."""
        B = state.shape[0]
        F = torch.eye(self.state_dim, device=state.device, dtype=state.dtype).unsqueeze(0).repeat(B, 1, 1)
        _pos_w, _vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b = state.split([3, 3, 4, 3, 3], dim=-1)
        q_w, q_x, q_y, q_z = quat_b_to_w.unbind(-1)
        ax, ay, az = (accel_b_raw - accel_bias_b).unbind(-1)
        F[:, 0:3, 3:6] = torch.eye(3, device=F.device, dtype=F.dtype) * self.dt
        J_aq = torch.zeros(B, 3, 4, device=F.device, dtype=F.dtype)
        J_aq[:, 0, 0] = -2*q_z*ay + 2*q_y*az; J_aq[:, 1, 0] =  2*q_z*ax - 2*q_x*az; J_aq[:, 2, 0] = -2*q_y*ax + 2*q_x*ay
        J_aq[:, 0, 1] =  2*q_y*ay + 2*q_z*az; J_aq[:, 1, 1] =  2*q_y*ax - 4*q_x*ay - 2*q_w*az; J_aq[:, 2, 1] =  2*q_z*ax + 2*q_w*ay - 4*q_x*az
        J_aq[:, 0, 2] = -4*q_y*ax + 2*q_x*ay + 2*q_w*az; J_aq[:, 1, 2] =  2*q_x*ax + 2*q_z*az; J_aq[:, 2, 2] = -2*q_w*ax + 2*q_z*ay - 4*q_y*az
        J_aq[:, 0, 3] = -4*q_z*ax - 2*q_w*ay + 2*q_x*az; J_aq[:, 1, 3] =  2*q_w*ax - 4*q_z*ay + 2*q_y*az; J_aq[:, 2, 3] =  2*q_x*ax + 2*q_y*ay
        F[:, 0:3, 6:10] = J_aq * 0.5 * self.dt**2
        F[:, 3:6, 6:10] = J_aq * self.dt
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        F[:, 0:3, 13:16] = -rot_mat_b_to_w * 0.5 * self.dt**2
        F[:, 3:6, 13:16] = -rot_mat_b_to_w * self.dt
        omega = gyro_b_raw - gyro_bias_b
        Omega = torch.zeros(B, 4, 4, device=F.device, dtype=F.dtype)
        Omega[:, 0, 1:4] = -omega; Omega[:, 1, 0] = omega[:, 0]; Omega[:, 2, 0] = omega[:, 1]; Omega[:, 3, 0] = omega[:, 2]
        Omega[:, 1, 2] = omega[:, 2]; Omega[:, 1, 3] = -omega[:, 1]; Omega[:, 2, 1] = -omega[:, 2]; Omega[:, 2, 3] = omega[:, 0]; Omega[:, 3, 1] = omega[:, 1]; Omega[:, 3, 2] = -omega[:, 0]
        F[:, 6:10, 6:10] = torch.eye(4, device=F.device, dtype=F.dtype) + 0.5 * self.dt * Omega
        Q_deriv_mat = torch.stack([-q_x, -q_y, -q_z, q_w, -q_z, q_y, q_z, q_w, -q_x, -q_y, q_x, q_w], -1).view(B, 4, 3)
        F[:, 6:10, 10:13] = -0.5 * self.dt * Q_deriv_mat
        return F

    def _compute_jacobian_H(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the Jacobian of the measurement function (H = ∂h/∂x)."""
        B = state.shape[0]
        H = torch.zeros(B, self.obs_dim, self.state_dim, device=state.device, dtype=state.dtype)
        quat_b_to_w = state[..., 6:10]
        q_w, q_x, q_y, q_z = quat_b_to_w.unbind(-1)
        g = self.gravity_w[2]
        J_H_q = torch.zeros(B, 3, 4, device=H.device, dtype=H.dtype)
        J_H_q[:, 0, 0] = -2*g*q_y; J_H_q[:, 0, 1] = 2*g*q_z; J_H_q[:, 0, 2] = -2*g*q_w; J_H_q[:, 0, 3] = 2*g*q_x
        J_H_q[:, 1, 0] = 2*g*q_x; J_H_q[:, 1, 1] = 2*g*q_w; J_H_q[:, 1, 2] = 2*g*q_z; J_H_q[:, 1, 3] = 2*g*q_y
        J_H_q[:, 2, 1] = -4*g*q_x; J_H_q[:, 2, 2] = -4*g*q_y
        H[..., 0:3, 6:10] = J_H_q
        H[..., 0:3, 13:16] = torch.eye(3, device=H.device, dtype=H.dtype)
        H[..., 3:6, 10:13] = torch.eye(3, device=H.device, dtype=H.dtype)
        return H

    def predict(self, state: torch.Tensor, P: torch.Tensor, gyro_b_raw: torch.Tensor, accel_b_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the EKF prediction step (time update)."""
        state_pred = self._state_transition_function(state, gyro_b_raw, accel_b_raw)
        F = self._compute_jacobian_F(state, gyro_b_raw, accel_b_raw)
        Q_diag = torch.abs(self.Q_diag)
        Q_diag[10:13] += torch.square(self.gyro_bias_rw_std) * self.dt
        Q_diag[13:16] += torch.square(self.accel_bias_rw_std) * self.dt
        Q = torch.diag(Q_diag)
        P_pred = F @ P @ F.transpose(-2, -1) + Q
        return state_pred, P_pred

    def update(self, state_pred: torch.Tensor, P_pred: torch.Tensor, 
               measurement: torch.Tensor, accel_b_raw: torch.Tensor,
               R_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the EKF update step (measurement update)."""
        h_pred = self._measurement_function(state_pred)
        innovation = measurement - h_pred
        H = self._compute_jacobian_H(state_pred)
        
        if R_override is not None:
            R_adaptive = R_override
        else:
            accel_norm_diff = torch.abs(torch.norm(accel_b_raw, dim=-1, keepdim=True) - torch.norm(self.gravity_w))
            scaling_factor = torch.exp(self.adaptive_R_factor * accel_norm_diff).unsqueeze(-1)
            R_adaptive = torch.diag(F.softplus(self.raw_R_diag) + 1e-6) * scaling_factor

        S = H @ P_pred @ H.transpose(-2, -1) + R_adaptive
        K = torch.linalg.solve(S, H @ P_pred.transpose(-2, -1)).transpose(-2, -1)

        state_updated = state_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        state_updated[..., 6:10] = F.normalize(state_updated[..., 6:10], p=2, dim=-1)

        I = torch.eye(self.state_dim, device=state_pred.device, dtype=state_pred.dtype)
        P_updated = (I - K @ H) @ P_pred @ (I - K @ H).transpose(-2, -1) + K @ R_adaptive @ K.transpose(-2, -1)
        return state_updated, P_updated, innovation
    
    def _calculate_zupt_update(self, state: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the state and covariance correction from a ZUPT."""
        B = state.shape[0]
        H_zupt = torch.zeros(B, 3, self.state_dim, device=state.device, dtype=state.dtype)
        H_zupt[:, :, 3:6] = torch.eye(3, device=state.device, dtype=state.dtype)
        R_zupt = torch.eye(3, device=state.device, dtype=state.dtype).unsqueeze(0) * self.zupt_R_factor

        vel_pred = state[..., 3:6]
        innov_zupt = -vel_pred

        S_zupt = H_zupt @ P @ H_zupt.transpose(-2, -1) + R_zupt
        K_zupt = torch.linalg.solve(S_zupt, H_zupt @ P.transpose(-2, -1)).transpose(-2, -1)

        state_updated = state + (K_zupt @ innov_zupt.unsqueeze(-1)).squeeze(-1)
        P_updated = (torch.eye(self.state_dim, device=state.device) - K_zupt @ H_zupt) @ P
        
        state_updated[..., 6:10] = F.normalize(state_updated[..., 6:10], p=2, dim=-1)
        return state_updated, P_updated

    def forward(self, state: torch.Tensor, P: torch.Tensor, gyro_b_raw: torch.Tensor, accel_b_raw: torch.Tensor, force_raw: torch.Tensor,
                measurement: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Executes one full predict-update cycle of the EKF."""
        state_pred, P_pred = self.predict(state, P, gyro_b_raw, accel_b_raw)
        
        is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        
        final_state = state_pred.clone()
        final_P = P_pred.clone()
        innovation = torch.zeros(state.shape[0], self.obs_dim, device=self.device, dtype=gyro_b_raw.dtype)

        # --- Update Step ---
        # Case 1: Stationary (ZUPT). Correct velocity and align with gravity.
        zupt_mask = is_zupt
        if torch.any(zupt_mask):
            state_zupt, P_zupt = self._calculate_zupt_update(state_pred[zupt_mask], P_pred[zupt_mask])
            
            # For gravity alignment, use a high-confidence (low noise) R matrix for the accelerometer
            R_gravity = torch.diag(torch.tensor([self.zupt_gravity_R_factor] * 3 + [1e3] * 3, device=self.device))
            R_gravity = R_gravity.unsqueeze(0).repeat(torch.sum(zupt_mask), 1, 1)

            state_gravity, P_gravity, innovation_gravity = self.update(
                state_zupt, P_zupt, measurement[zupt_mask], accel_b_raw[zupt_mask], R_override=R_gravity)
            
            final_state[zupt_mask] = state_gravity
            final_P[zupt_mask] = P_gravity
            innovation[zupt_mask] = innovation_gravity

        # Case 2: In Motion. Perform a standard update.
        non_zupt_mask = ~is_zupt
        if torch.any(non_zupt_mask) and measurement is not None:
            state_motion, P_motion, innovation_motion = self.update(
                state_pred[non_zupt_mask], P_pred[non_zupt_mask], 
                measurement[non_zupt_mask], accel_b_raw[non_zupt_mask])
            
            final_state[non_zupt_mask] = state_motion
            final_P[non_zupt_mask] = P_motion
            innovation[non_zupt_mask] = innovation_motion
        
        # --- Assemble Features for TCN ---
        vel_w = final_state[..., 3:6]
        quat_b_to_w = final_state[..., 6:10]
        rot_mat_w_to_b = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        vel_b = (rot_mat_w_to_b @ vel_w.unsqueeze(-1)).squeeze(-1)
        
        tcn_features = {
            'body_velocity': vel_b, 
            'innovation': innovation, 
            'zupt_flag': is_zupt.float().unsqueeze(-1)
        }
        
        return final_state, final_P, tcn_features

if __name__ == "__main__":
    print("Running tests for AEKF.py...")
    # --- Test Parameters ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 4
    state_dim = 16
    obs_dim = 6
    
    print(f"Using device: {device}")

    # --- Test 1: Instantiation ---
    try:
        ekf = ExtendedKalmanFilter(state_dim=state_dim, obs_dim=obs_dim, device=device)
        ekf.eval()
        print("Test 1 (Instantiation): PASSED")
    except Exception as e:
        print(f"Test 1 (Instantiation): FAILED - {e}")
        exit()

    # --- Test 2: Forward Pass and Shape Verification ---
    # Prepare inputs
    state = torch.zeros(batch_size, state_dim, device=device); state[:, 6] = 1.0
    P = torch.eye(state_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1
    accel = torch.randn(batch_size, 3, device=device) * 0.1; accel[:, 2] += 9.81
    gyro = torch.randn(batch_size, 3, device=device) * 0.01
    force = torch.rand(batch_size, 1, device=device)
    measurement = torch.cat([accel, gyro], dim=-1)

    try:
        state_new, P_new, tcn_features = ekf.forward(state, P, gyro, accel, force, measurement)

        # --- Shape Assertions ---
        assert state_new.shape == (batch_size, state_dim), f"State shape incorrect: {state_new.shape}"
        assert P_new.shape == (batch_size, state_dim, state_dim), f"Covariance shape incorrect: {P_new.shape}"
        assert tcn_features['body_velocity'].shape == (batch_size, 3), f"TCN body_velocity shape incorrect: {tcn_features['body_velocity'].shape}"
        assert tcn_features['innovation'].shape == (batch_size, obs_dim), f"TCN innovation shape incorrect: {tcn_features['innovation'].shape}"
        assert tcn_features['zupt_flag'].shape == (batch_size, 1), f"TCN zupt_flag shape incorrect: {tcn_features['zupt_flag'].shape}"

        # --- Stability Assertions ---
        assert not torch.any(torch.isnan(state_new)), "NaN detected in state"
        assert not torch.any(torch.isinf(state_new)), "Inf detected in state"
        assert not torch.any(torch.isnan(P_new)), "NaN detected in covariance"
        assert not torch.any(torch.isinf(P_new)), "Inf detected in covariance"

        print("Test 2 (Forward Pass & Shape Verification): PASSED")
    except Exception as e:
        print(f"Test 2 (Forward Pass & Shape Verification): FAILED - {e}")
        exit()

    print("\nAll AEKF tests passed successfully.")