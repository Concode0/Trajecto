"""
ESKF (Error-State Kalman Filter) for 3D Pen Trajectory Reconstruction.

This module implements an Error-State Kalman Filter (ESKF) for IMU state
estimation. The ESKF is a powerful alternative to the standard EKF that operates
by tracking the error of a 'nominal' state. This often leads to a more stable
and accurate filter because the error dynamics are typically more linear than
the full state dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from zupt_detector import ZuptDetector
from rotation_utils import quaternion_multiply, quaternion_to_rotation_matrix, small_angle_to_quaternion

class ErrorStateKalmanFilter(nn.Module):
    """Implements a batch-aware Error-State Kalman Filter.

    The filter estimates a nominal state and the error of that state.
    Nominal State: [pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b]
    Error State (δx): [δpos_w, δvel_w, δtheta_b, δgyro_bias_b, δaccel_bias_b] (15 dims)
    """

    def __init__(self, 
                 error_state_dim: int = 15,
                 obs_dim: int = 6,
                 dt: float = 0.01,
                 device: str = 'cpu',
                 zupt_window_size: int = 20,
                 zupt_accel_threshold: float = 0.1,
                 zupt_force_var_threshold: float = 0.01,
                 zupt_force_delta_threshold: float = 0.1):
        """Initializes the ESKF module.
        
        Args:
            error_state_dim (int): Dimension of the error state vector (15).
            obs_dim (int): Dimension of the observation vector (6).
            dt (float): Time step in seconds.
            device (str): Compute device.
            zupt_window_size (int): Number of samples for ZUPT variance check.
            zupt_accel_threshold (float): Accel variance threshold for ZUPT.
            zupt_force_var_threshold (float): Force variance threshold for ZUPT.
            zupt_force_delta_threshold (float): Force delta threshold for ZUPT.
        """
        super(ErrorStateKalmanFilter, self).__init__()

        self.error_state_dim = error_state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device

        self.register_parameter('Q_diag', nn.Parameter(torch.ones(error_state_dim, device=device) * 1e-4))
        self.register_parameter('R_diag', nn.Parameter(torch.ones(obs_dim, device=device) * 1e-2))
        self.register_buffer('gravity_w', torch.tensor([0., 0., 9.81], device=device))

        self.zupt_detector = ZuptDetector(
            window_size=zupt_window_size,
            accel_var_threshold=zupt_accel_threshold,
            force_var_threshold=zupt_force_var_threshold,
            force_delta_threshold=zupt_force_delta_threshold,
            device=device
        )
        self.zupt_R_factor = 1e-6
        self.zupt_gravity_R_factor = 1e-8

    def _propagate_nominal_state(self, pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, gyro_b_raw, accel_b_raw):
        """Propagates the nominal state forward in time using IMU measurements."""
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b
        q_dot = 0.5 * quaternion_multiply(quat_b_to_w, torch.cat([torch.zeros_like(gyro_b_corrected[...,:1]), gyro_b_corrected], dim=-1))
        quat_b_to_w_new = F.normalize(quat_b_to_w + q_dot * self.dt, p=2, dim=-1)
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_w = (rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w
        pos_w_new = pos_w + vel_w * self.dt + 0.5 * accel_w * self.dt**2
        vel_w_new = vel_w + accel_w * self.dt
        return pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b, accel_bias_b

    def predict(self, P_error: torch.Tensor, quat_b_to_w: torch.Tensor, accel_b_raw: torch.Tensor, accel_bias_b: torch.Tensor):
        """Predicts the error state covariance matrix."""
        B = P_error.shape[0]
        F_error = torch.eye(self.error_state_dim, device=self.device, dtype=P_error.dtype).unsqueeze(0).repeat(B, 1, 1)
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b
        accel_ssm = torch.zeros(B, 3, 3, device=P_error.device, dtype=P_error.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]; accel_ssm[:, 0, 2] =  accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] =  accel_b_corrected[:, 2]; accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]; accel_ssm[:, 2, 1] =  accel_b_corrected[:, 0]
        F_error[:, 0:3, 3:6] = torch.eye(3, device=self.device) * self.dt
        F_error[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        F_error[:, 6:9, 9:12] = -torch.eye(3, device=self.device) * self.dt
        Q_error = torch.diag(F.softplus(self.Q_diag) + 1e-6)
        return F_error @ P_error @ F_error.transpose(-2, -1) + Q_error

    def update(self, P_error_pred: torch.Tensor, quat_b_to_w: torch.Tensor, accel_bias_b: torch.Tensor, 
               gyro_bias_b: torch.Tensor, measurement: torch.Tensor, R_override: Optional[torch.Tensor] = None):
        """Updates the error state and its covariance based on the measurement."""
        B = P_error_pred.shape[0]
        H_error = torch.zeros(B, self.obs_dim, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
        rot_mat_w_to_b = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        g_body = rot_mat_w_to_b @ self.gravity_w
        g_ssm = torch.zeros(B, 3, 3, device=P_error_pred.device, dtype=P_error_pred.dtype)
        g_ssm[:, 0, 1] = -g_body[:, 2]; g_ssm[:, 0, 2] =  g_body[:, 1]
        g_ssm[:, 1, 0] =  g_body[:, 2]; g_ssm[:, 1, 2] = -g_body[:, 0]
        g_ssm[:, 2, 0] = -g_body[:, 1]; g_ssm[:, 2, 1] =  g_body[:, 0]
        H_error[:, 0:3, 6:9] = g_ssm
        H_error[:, 0:3, 12:15] = torch.eye(3, device=self.device)
        H_error[:, 3:6, 9:12] = torch.eye(3, device=self.device)
        
        accel_pred = g_body + accel_bias_b
        gyro_pred = gyro_bias_b
        h_pred = torch.cat([accel_pred, gyro_pred], dim=-1)
        innovation = measurement - h_pred
        
        if R_override is not None:
            R_noise = R_override
        else:
            R_noise = torch.diag(F.softplus(self.R_diag) + 1e-6)

        S = H_error @ P_error_pred @ H_error.transpose(-2, -1) + R_noise
        K = torch.linalg.solve(S, H_error @ P_error_pred.transpose(-2, -1)).transpose(-2, -1)
        
        delta_x = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        P_error_new = (torch.eye(self.error_state_dim, device=self.device) - K @ H_error) @ P_error_pred
        return delta_x, P_error_new, innovation

    def _calculate_zupt_update(self, vel_w_pred: torch.Tensor, P_error_pred: torch.Tensor):
        """Calculates the error-state correction from a ZUPT pseudo-measurement."""
        B = P_error_pred.shape[0]
        H_zupt = torch.zeros(B, 3, self.error_state_dim, device=self.device, dtype=P_error_pred.dtype)
        H_zupt[:, :, 3:6] = torch.eye(3, device=self.device)
        R_zupt = torch.eye(3, device=self.device).unsqueeze(0) * self.zupt_R_factor
        
        innov_zupt = -vel_w_pred
        S_zupt = H_zupt @ P_error_pred @ H_zupt.transpose(-2, -1) + R_zupt
        K_zupt = torch.linalg.solve(S_zupt, H_zupt @ P_error_pred.transpose(-2, -1)).transpose(-2, -1)
        
        delta_x_zupt = (K_zupt @ innov_zupt.unsqueeze(-1)).squeeze(-1)
        P_error_new = (torch.eye(self.error_state_dim, device=self.device) - K_zupt @ H_zupt) @ P_error_pred
        return delta_x_zupt, P_error_new

    def inject_correction(self, pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, delta_x):
        """Injects the calculated error state back into the nominal state."""
        d_pos_w, d_vel_w, d_theta_b, d_gyro_bias_b, d_accel_bias_b = delta_x.split([3, 3, 3, 3, 3], dim=-1)
        pos_w_new = pos_w + d_pos_w
        vel_w_new = vel_w + d_vel_w
        quat_b_to_w_new = quaternion_multiply(quat_b_to_w, small_angle_to_quaternion(d_theta_b))
        quat_b_to_w_new = F.normalize(quat_b_to_w_new, p=2, dim=-1)
        gyro_bias_b_new = gyro_bias_b + d_gyro_bias_b
        accel_bias_b_new = accel_bias_b + d_accel_bias_b
        return pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new

    def forward(self, pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error, gyro_b_raw, accel_b_raw, force_raw, measurement):
        """Executes one full predict-update cycle of the ESKF."""
        pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred = \
            self._propagate_nominal_state(pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, gyro_b_raw, accel_b_raw)
        
        P_error_pred = self.predict(P_error, quat_b_to_w, accel_b_raw, accel_bias_b)
        
        is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        
        delta_x = torch.zeros(pos_w.shape[0], self.error_state_dim, device=self.device, dtype=pos_w.dtype)
        innovation_output = torch.zeros(pos_w.shape[0], self.obs_dim, device=self.device, dtype=pos_w.dtype)
        P_error_final = P_error_pred.clone()

        # --- Update Step ---
        # Case 1: Stationary (ZUPT). Correct velocity and align with gravity.
        zupt_mask = is_zupt
        if torch.any(zupt_mask):
            # 1a. ZUPT correction for velocity
            delta_x_zupt, P_after_zupt = self._calculate_zupt_update(vel_w_pred[zupt_mask], P_error_pred[zupt_mask])
            
            # 1b. Gravity alignment correction for orientation
            R_gravity = torch.diag(torch.tensor([self.zupt_gravity_R_factor] * 3 + [1e3] * 3, device=self.device))
            R_gravity = R_gravity.unsqueeze(0).repeat(torch.sum(zupt_mask), 1, 1)
            
            delta_x_gravity, P_after_gravity, innovation_gravity = self.update(
                P_after_zupt, quat_b_to_w_pred[zupt_mask], accel_bias_b_pred[zupt_mask], 
                gyro_bias_b_pred[zupt_mask], measurement[zupt_mask], R_override=R_gravity)
            
            # Combine corrections and store results
            delta_x[zupt_mask] = delta_x_zupt + delta_x_gravity
            P_error_final[zupt_mask] = P_after_gravity
            innovation_output[zupt_mask] = innovation_gravity

        # Case 2: In Motion. Perform a standard update.
        non_zupt_mask = ~is_zupt
        if torch.any(non_zupt_mask):
            delta_x_update, P_update, innovation_update = self.update(
                P_error_pred[non_zupt_mask], quat_b_to_w_pred[non_zupt_mask], accel_bias_b_pred[non_zupt_mask], 
                gyro_bias_b_pred[non_zupt_mask], measurement[non_zupt_mask])
            
            delta_x[non_zupt_mask] = delta_x_update
            P_error_final[non_zupt_mask] = P_update
            innovation_output[non_zupt_mask] = innovation_update
        
        pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = \
            self.inject_correction(pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred, delta_x)
        
        rot_mat_w_to_b = quaternion_to_rotation_matrix(quat_b_to_w_new).transpose(-2, -1)
        vel_b = (rot_mat_w_to_b @ vel_w_new.unsqueeze(-1)).squeeze(-1)
        
        tcn_features = {
            'body_velocity': vel_b, 
            'zupt_flag': is_zupt.float().unsqueeze(-1),
            'innovation': innovation_output
        }
        
        return pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new, P_error_final, tcn_features

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    eskf = ErrorStateKalmanFilter(device=device)
    
    B=4
    p=torch.zeros(B,3,device=device); v=torch.zeros(B,3,device=device); q=torch.zeros(B,4,device=device); q[:,0]=1.0
    bg=torch.zeros(B,3,device=device); ba=torch.zeros(B,3,device=device)
    P=torch.eye(15,device=device).unsqueeze(0).repeat(B,1,1)*0.1
    
    accel=torch.randn(B,3,device=device)*0.1; accel[:,2]+=9.81
    gyro=torch.randn(B,3,device=device)*0.01
    force=torch.rand(B,1,device=device)
    meas=torch.cat([accel,gyro],dim=-1)
    
    p,v,q,bg,ba,P,tcn_feats=eskf.forward(p,v,q,bg,ba,P,gyro,accel,force,meas)

    print(f"p:{p.shape}, v:{v.shape}, q:{q.shape}, P:{P.shape}")
    print(f"TCN Features body_velocity shape: {tcn_feats['body_velocity'].shape}")
    print(f"TCN Features innovation shape: {tcn_feats['innovation'].shape}")
    print(f"TCN Features zupt_flag shape: {tcn_feats['zupt_flag'].shape}")
    print("ESKF tested successfully.")