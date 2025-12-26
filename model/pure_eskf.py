import torch
import torch.nn as nn
from typing import Dict, Optional

# Use relative imports assuming this file is in model/
from .ESKF import ErrorStateKalmanFilter
from .config import Config

class PureESKFModel(nn.Module):
    """
    A wrapper around ErrorStateKalmanFilter to process full sequences
    without TCN integration. This serves as a physics-only baseline.
    """
    def __init__(self, device: str = "cpu", dt: float = 0.01):
        super().__init__()
        self.device = device
        self.dt = dt
        
        # Initialize ESKF with traditional ZUPT, no TCN
        self.eskf = ErrorStateKalmanFilter(
            dt=dt,
            device=device,
            use_zupt=True,
            use_tcn_zupt=False
        )

    def forward(self, imu_raw: torch.Tensor, imu_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process the entire IMU sequence.

        Args:
            imu_raw (torch.Tensor): Raw IMU sequence.
                - Shape: (Batch, Seq_Len, 7) or (Batch, Seq_Len, 6)
                - Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, (optional) force]
            imu_norm (torch.Tensor): Normalized IMU sequence (unused in Pure ESKF).

        Returns:
            Dict[str, torch.Tensor]:
                - "pred_pos_w": Estimated position trajectory.
                    - Shape: (Batch, Seq_Len, 3)
        """
        batch_size, seq_len, feat_dim = imu_raw.shape
        
        # --- Initialization ---
        # 1. Nominal State
        pos_w = torch.zeros(batch_size, 3, device=self.device)
        vel_w = torch.zeros(batch_size, 3, device=self.device)
        
        # Identity quaternion (w=1, x=0, y=0, z=0)
        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device)
        quat_b_to_w[:, 0] = 1.0 
        
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device)
        
        # 2. Error Covariance
        # Initialize with small uncertainty
        P_error = torch.eye(15, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1) * 1e-4

        # Collector for outputs
        # We pre-allocate for efficiency
        pred_pos_seq = torch.zeros(batch_size, seq_len, 3, device=self.device)

        # --- Sequence Processing ---
        for t in range(seq_len):
            # Extract current measurement
            curr_meas = imu_raw[:, t, :]
            
            accel_raw = curr_meas[:, 0:3]
            gyro_raw = curr_meas[:, 3:6]
            
            if feat_dim >= 7:
                force_raw = curr_meas[:, 6:7]
            else:
                # Fallback if no force channel is present
                force_raw = torch.zeros(batch_size, 1, device=self.device)
            
            measurement = curr_meas[:, 0:6] # Acc+Gyro
            
            # Step the ESKF
            (
                pos_w, 
                vel_w, 
                quat_b_to_w, 
                gyro_bias_b, 
                accel_bias_b, 
                P_error, 
                _ # tcn_features unused
            ) = self.eskf.forward(
                pos_w, 
                vel_w, 
                quat_b_to_w, 
                gyro_bias_b, 
                accel_bias_b, 
                P_error, 
                gyro_raw, 
                accel_raw, 
                force_raw, 
                measurement, 
                tcn_output=None
            )
            
            # Store position
            pred_pos_seq[:, t, :] = pos_w

        return {"pred_pos_w": pred_pos_seq}
