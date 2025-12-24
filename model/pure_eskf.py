"""
This module implements the Pure ESKF Model wrapper for the Error-State Kalman Filter.
It serves as a physics-based baseline that does not use TCN corrections.
"""

import torch
import torch.nn as nn
from model.ESKF import ErrorStateKalmanFilter
from model.rotation_utils import quaternion_from_two_vectors
from model.config import Config

class PureESKFModel(nn.Module):
    """
    Wrapper for the pure Error-State Kalman Filter (Physics-only baseline).
    """
    def __init__(self, device: str, dt: float = Config.DT):
        super().__init__()
        self.device = device
        self.dt = dt
        # For Pure ESKF baseline, we enable traditional ZUPT and disable TCN ZUPT
        self.filter = ErrorStateKalmanFilter(device=device, dt=dt, use_zupt=True, use_tcn_zupt=False)

    def _initialize_state(self, batch_size: int, imu_data_seq: torch.Tensor):
        dtype = imu_data_seq.dtype
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)

        # Leveling logic (same as ESKFTCN_model)
        if imu_data_seq is not None:
            accel_init = imu_data_seq[:, 0, :3]
            accel_norm = torch.norm(accel_init, p=2, dim=-1)
            reliable_mask = (accel_norm > 4.9) & (accel_norm < 14.7)

            if reliable_mask.any():
                gravity_up_w = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=dtype).unsqueeze(0).expand(reliable_mask.sum(), -1)
                init_quat = quaternion_from_two_vectors(accel_init[reliable_mask], gravity_up_w)
                quat_b_to_w[reliable_mask] = init_quat

            identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype)
            quat_b_to_w[~reliable_mask] = identity_quat.expand((~reliable_mask).sum(), -1)
        else:
             quat_b_to_w[:, 0] = 1.0

        P_error = torch.eye(15, device=self.device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        return pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error

    def forward(self, imu_seq_raw, *args):
        # imu_seq_raw: (Batch, Seq, 7) [Acc(3), Gyro(3), Force(1)]
        batch_size, seq_len, _ = imu_seq_raw.shape

        pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error = self._initialize_state(batch_size, imu_seq_raw)

        preds = []

        for t in range(seq_len):
            # Extract sensor data for current step
            accel = imu_seq_raw[:, t, 0:3]
            gyro = imu_seq_raw[:, t, 3:6]
            force = imu_seq_raw[:, t, 6:7]
            measurement = torch.cat([accel, gyro], dim=-1)

            # Step the filter
            pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error, _ = self.filter(
                pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error,
                gyro, accel, force, measurement, tcn_output=None
            )
            preds.append(pos_w.unsqueeze(1))

        return torch.cat(preds, dim=1)
