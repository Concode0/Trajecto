"""
Pure Integration Baseline Model

This module implements a simple dead reckoning approach using direct double
integration of IMU measurements. It serves as a minimal baseline to demonstrate
the importance of filtering (ESKF) and data-driven corrections (TCN).

No Kalman filtering, no ZUPT detection, just raw integration with bias estimation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys, os

# Adjust sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .config import Config
    from .rotation_utils import (
        quaternion_from_two_vectors,
        quaternion_to_rotation_matrix,
        quaternion_multiply,
        small_angle_to_quaternion,
    )
except ImportError:
    # Support running as standalone script
    from config import Config
    from rotation_utils import (
        quaternion_from_two_vectors,
        quaternion_to_rotation_matrix,
        quaternion_multiply,
        small_angle_to_quaternion,
    )

# Static initialization period (must match preprocessing)
STATIC_INIT_S = 2.0


class PureIntegrationModel(nn.Module):
    """
    Dead reckoning baseline using direct IMU integration.

    Algorithm:
    1. Initialize orientation from gravity alignment
    2. Estimate gyro/accel bias from static period
    3. Integrate angular velocity for orientation (quaternion propagation)
    4. Double integrate linear acceleration for position

    This model has NO error correction mechanisms:
    - No Kalman filtering
    - No ZUPT updates
    - No TCN corrections
    - Biases drift freely after initialization

    Expected behavior: Rapid drift accumulation, demonstrating the need
    for advanced filtering and data-driven corrections.
    """

    def __init__(self, device: str = "cpu", dt: float = Config.DT):
        super().__init__()
        self.device = device
        self.dt = dt

        # Gravity vector (world frame, Z-up)
        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device))

        # Pen tip offset from IMU center
        self.pen_tip_offset = torch.tensor(Config.INITIAL_PEN_TIP_OFFSET, device=device)

    def _estimate_static_period(self, gyro_data: torch.Tensor, max_samples: int) -> int:
        """
        Detect the end of the static initialization period by monitoring gyro variance.

        Args:
            gyro_data: Gyroscope measurements (Batch, Time, 3)
            max_samples: Maximum number of samples to check

        Returns:
            Number of static samples detected
        """
        window_size = 10
        static_samples = 20  # Default minimum

        for t in range(window_size, max_samples - window_size, window_size):
            window = gyro_data[:, t:t+window_size, :]
            gyro_var = torch.var(window, dim=1).mean()

            # Motion detected if variance exceeds threshold
            if gyro_var.item() > 0.002:  # rad²/s²
                static_samples = t
                break
        else:
            static_samples = max_samples

        return max(20, min(static_samples, max_samples))

    def forward(self, imu_raw: torch.Tensor, imu_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process IMU sequence using pure dead reckoning.

        Args:
            imu_raw: Raw IMU data
                - Shape: (Batch, Seq_Len, 7) or (Batch, Seq_Len, 6)
                - Channels: [accel_xyz(3), gyro_xyz(3), force(1, optional)]
            imu_norm: Normalized IMU (unused)

        Returns:
            Dict with:
                - "pred_pos_w": Predicted position trajectory (Batch, Seq_Len, 3)
                - "filter_covariance": Dummy covariance (Batch, Seq_Len, 15, 15)
        """
        batch_size, seq_len, feat_dim = imu_raw.shape

        # ========================================
        # 1. INITIALIZATION
        # ========================================

        # Detect static period for bias estimation
        max_static_samples = min(seq_len, int(STATIC_INIT_S / self.dt))

        if max_static_samples < 10:
            # Fallback: identity orientation, zero biases
            quat_b_to_w = torch.zeros(batch_size, 4, device=self.device)
            quat_b_to_w[:, 0] = 1.0  # Identity quaternion
            gyro_bias_b = torch.zeros(batch_size, 3, device=self.device)
            accel_bias_b = torch.zeros(batch_size, 3, device=self.device)
        else:
            # Estimate static period
            gyro_data = imu_raw[:, :max_static_samples, 3:6]
            static_samples = self._estimate_static_period(gyro_data, max_static_samples)

            # Gravity alignment: align measured acceleration with gravity vector
            avg_accel_b = imu_raw[:, :static_samples, 0:3].mean(dim=1)  # (Batch, 3)

            # World "up" vector (opposite of gravity)
            world_up_w = -self.gravity_w.unsqueeze(0).repeat(batch_size, 1)

            # Quaternion: body frame's measured "up" → world "up"
            quat_b_to_w = quaternion_from_two_vectors(-avg_accel_b, world_up_w)

            # Estimate biases from static period
            gyro_bias_b = imu_raw[:, :static_samples, 3:6].mean(dim=1)

            # Accelerometer bias: difference between measured and expected gravity
            rot_mat_w_to_b = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
            expected_accel_b = (rot_mat_w_to_b @ self.gravity_w.unsqueeze(-1).unsqueeze(0)).squeeze(-1)
            accel_bias_b = avg_accel_b - expected_accel_b

        # Initialize state
        pos_w = torch.zeros(batch_size, 3, device=self.device)
        vel_w = torch.zeros(batch_size, 3, device=self.device)

        # Storage for trajectory
        pred_pos_seq = torch.zeros(batch_size, seq_len, 3, device=self.device)
        pred_cov_seq = torch.zeros(batch_size, seq_len, 15, 15, device=self.device)  # Dummy covariance

        # ========================================
        # 2. INTEGRATION LOOP
        # ========================================

        for t in range(seq_len):
            # Extract measurements
            accel_raw = imu_raw[:, t, 0:3]
            gyro_raw = imu_raw[:, t, 3:6]

            # Bias correction (static biases, NO adaptive update)
            accel_corrected = accel_raw - accel_bias_b
            gyro_corrected = gyro_raw - gyro_bias_b

            # --- Orientation Update (Quaternion Integration) ---
            # q_new = q_old * exp(ω * dt)
            angle_change = gyro_corrected * self.dt
            delta_quat = small_angle_to_quaternion(angle_change)
            quat_b_to_w = quaternion_multiply(quat_b_to_w, delta_quat)
            quat_b_to_w = torch.nn.functional.normalize(quat_b_to_w, p=2, dim=-1)

            # --- Acceleration in World Frame ---
            rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
            accel_w = (rot_mat_b_to_w @ accel_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w

            # --- Velocity Update (First Integration) ---
            # Simple forward Euler: v_new = v_old + a * dt
            vel_w = vel_w + accel_w * self.dt

            # --- Position Update (Second Integration) ---
            # Simple forward Euler: p_new = p_old + v * dt
            pos_w = pos_w + vel_w * self.dt

            # --- Apply Pen Tip Offset ---
            offset_w = (rot_mat_b_to_w @ self.pen_tip_offset.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)).squeeze(-1)
            pos_tip_w = pos_w + offset_w

            # Store results
            pred_pos_seq[:, t, :] = pos_tip_w
            # Dummy covariance (identity matrix)
            pred_cov_seq[:, t, :, :] = torch.eye(15, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        return {
            "pred_pos_w": pred_pos_seq,
            "filter_covariance": pred_cov_seq,
        }


if __name__ == "__main__":
    """Test the PureIntegrationModel with dummy data."""

    device = "cpu"
    batch_size = 2
    seq_len = 100
    dt = 0.02

    # Create dummy IMU data
    # Static period: first 50 samples with near-zero gyro and gravity-aligned accel
    imu_dummy = torch.zeros(batch_size, seq_len, 7, device=device)

    # Static period: accel = gravity, gyro = small bias
    imu_dummy[:, :50, 0:3] = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)  # Gravity
    imu_dummy[:, :50, 3:6] = torch.randn(batch_size, 50, 3, device=device) * 0.01  # Small gyro noise

    # Motion period: add some movement
    imu_dummy[:, 50:, 0:3] = torch.randn(batch_size, 50, 3, device=device) * 0.5 + torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
    imu_dummy[:, 50:, 3:6] = torch.randn(batch_size, 50, 3, device=device) * 0.1

    # Force channel (dummy)
    imu_dummy[:, :, 6] = torch.rand(batch_size, seq_len, device=device)

    # Initialize model
    model = PureIntegrationModel(device=device, dt=dt)
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(imu_dummy)

    print("Pure Integration Model Test")
    print("=" * 50)
    print(f"Input shape: {imu_dummy.shape}")
    print(f"Output position shape: {output['pred_pos_w'].shape}")
    print(f"Output covariance shape: {output['filter_covariance'].shape}")
    print(f"Final position (sample 0): {output['pred_pos_w'][0, -1, :]}")
    print("\nTest passed!")
