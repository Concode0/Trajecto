import sys, os
import torch
import torch.nn as nn
from typing import Dict, Optional

# Adjust sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use relative imports assuming this file is in model/
from .ESKF import ErrorStateKalmanFilter
from .config import Config
from .rotation_utils import quaternion_from_two_vectors, quaternion_to_rotation_matrix # Import the helper

# Define a constant for static initialization duration
# Must match STATIC_BUFFER_S from acquire.py (2 seconds)
STATIC_INIT_S = 0.5 # Seconds of static data to use for initial alignment


class PureESKFModel(nn.Module):
    """
    A wrapper around ErrorStateKalmanFilter to process full sequences
    without TCN integration. This serves as a physics-only baseline.
    """
    def __init__(self, device: str = "cpu", dt: float = Config.DT):
        super().__init__()
        self.device = device
        self.dt = dt

        # Initialize ESKF with traditional ZUPT, no TCN, but with virtual measurements
        self.eskf = ErrorStateKalmanFilter(
            dt=dt,
            device=device,
            use_zupt=True,
            use_tcn_zupt=False,
            use_virtual_measurements=False  # DISABLED: Virtual measurements have sign bug that INCREASES drift by 2x
        )

        # Pen Tip Offset
        self.pen_tip_offset = torch.tensor(Config.INITIAL_PEN_TIP_OFFSET, device=device)

    def forward(self, imu_raw: torch.Tensor, imu_norm: Optional[torch.Tensor] = None,
                seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process the entire IMU sequence with optional padding mask.

        Args:
            imu_raw (torch.Tensor): Raw IMU sequence.
                - Shape: (Batch, Seq_Len, 7) or (Batch, Seq_Len, 6)
                - Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, (optional) force]
            imu_norm (torch.Tensor): Normalized IMU sequence (unused in Pure ESKF).
            seq_lengths (torch.Tensor, optional): Actual sequence lengths for each batch.
                - Shape: (Batch,)
                - If None, assumes no padding (uses full seq_len for all samples)

        Returns:
            Dict[str, torch.Tensor]:
                - "pred_pos_w": Estimated position trajectory.
                    - Shape: (Batch, Seq_Len, 3)
                    - Padded positions are filled with last valid position
        """
        batch_size, seq_len, feat_dim = imu_raw.shape

        # If no sequence lengths provided, assume all sequences use full length
        if seq_lengths is None:
            seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=imu_raw.device)

        # --- Initialization from Static Period ---
        # Pure ESKF uses a more sophisticated initialization than hybrid models:
        # - Hybrid models (ESKF_TCN, AEKF_TCN): Use only FIRST sample for orientation
        # - Pure ESKF: Detects static period and AVERAGES over it for better accuracy
        #
        # This approach leverages the "Tap-Wait-Write-Tap" data acquisition protocol,
        # which includes ~2s of static data at the beginning for calibration.

        # 1. Nominal State
        pos_w = torch.zeros(batch_size, 3, device=self.device)
        vel_w = torch.zeros(batch_size, 3, device=self.device)

        # Detect actual static period (before motion starts)
        # Use gyroscope variance as motion detector
        max_static_samples = min(seq_len, int(STATIC_INIT_S / self.dt))

        if max_static_samples < 10:
            # Fallback to identity if no static samples available
            quat_b_to_w = torch.zeros(batch_size, 4, device=self.device)
            quat_b_to_w[:, 0] = 1.0 # Identity quaternion (w=1, x=0, y=0, z=0)
            gyro_bias_b = torch.zeros(batch_size, 3, device=self.device)
        else:
            # Detect static period by checking gyro variance in sliding windows
            window_size = 10  # 0.2s @ 50Hz
            gyro_data = imu_raw[:, :max_static_samples, 3:6]  # (batch, time, 3)

            # Compute variance over time for all batches
            # Use a conservative approach: stop at first window with high variance
            static_samples = 20  # Default minimum
            for t in range(window_size, max_static_samples - window_size, window_size):
                # Check variance across the batch
                window = gyro_data[:, t:t+window_size, :]  # (batch, window, 3)
                gyro_var = torch.var(window, dim=1).mean()  # Average across batch and axes

                # If variance exceeds threshold, motion has started
                if gyro_var.item() > 0.002:  # rad²/s² threshold
                    static_samples = t
                    break
            else:
                static_samples = max_static_samples  # All samples are static

            # Ensure we have at least 20 samples (0.4s) for stable estimate
            static_samples = max(20, min(static_samples, max_static_samples))

            # Average accelerometer reading during TRUE static period
            avg_accel_b = imu_raw[:, :static_samples, 0:3].mean(dim=1)

            # Reliability check: verify average acceleration magnitude is close to gravity
            # This filters out cases of free-fall, strong external forces, or sensor errors
            accel_norm = torch.norm(avg_accel_b, p=2, dim=-1)  # (batch_size,)
            reliable_mask = (accel_norm > 4.9) & (accel_norm < 14.7)  # ~[0.5g, 1.5g]

            # Initialize quaternion with identity (fallback for unreliable samples)
            quat_b_to_w = torch.zeros(batch_size, 4, device=self.device)
            quat_b_to_w[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)

            if reliable_mask.any():
                # The accelerometer measures -gravity when static. So the vector it measures
                # is opposite to the direction of gravity in the body frame.
                # We want to align the body frame's "up" vector (opposite of measured accel)
                # with the world frame's "up" vector (opposite of gravity_w).

                # CRITICAL FIX: Use ACTUAL measured gravity magnitude for consistency
                # If we align quaternion to measured gravity (e.g., 9.866 m/s²) but then
                # subtract fixed gravity (9.80665 m/s²), we get systematic drift!
                # Calculate actual gravity magnitude from measurements
                measured_gravity_magnitude = torch.norm(avg_accel_b[reliable_mask], p=2, dim=-1, keepdim=True)  # (num_reliable, 1)

                # World frame gravity vector (pointing down in world frame)
                # Use MEASURED magnitude instead of fixed standard gravity
                # Shape: (num_reliable, 3)
                world_gravity_down = torch.cat([
                    torch.zeros(reliable_mask.sum(), 2, device=self.device),
                    measured_gravity_magnitude
                ], dim=-1)  # [0, 0, measured_g]

                # Quaternion that rotates measured accel to align with world gravity
                # Only compute for reliable samples
                init_quat = quaternion_from_two_vectors(avg_accel_b[reliable_mask], world_gravity_down)
                quat_b_to_w[reliable_mask] = init_quat

                # Update ESKF's gravity vector to use measured magnitude for THIS batch
                # CRITICAL: Must update for EACH batch since different samples may have
                # slightly different gravity magnitudes (sensor calibration variation)
                measured_g_mean = world_gravity_down.mean(dim=0)
                self.eskf.gravity_w = measured_g_mean.detach()  # Update for current batch

            # Initialize Gyro Bias from TRUE static period
            # Average gyro readings during static period to estimate bias
            gyro_bias_b = imu_raw[:, :static_samples, 3:6].mean(dim=1)

        accel_bias_b = torch.zeros(batch_size, 3, device=self.device)

        # 2. Error Covariance
        # Initialize with small uncertainty
        P_error = torch.eye(15, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1) * 1e-4

        # Collector for outputs
        # We pre-allocate for efficiency
        pred_pos_seq = torch.zeros(batch_size, seq_len, 3, device=self.device)

        # Track last valid position for each batch element (for padding)
        last_valid_pos = torch.zeros(batch_size, 3, device=self.device)

        # --- Sequence Processing with Padding Mask ---
        for t in range(seq_len):
            # Create mask: True for batch elements where t < seq_lengths[b]
            # Shape: (batch_size,)
            active_mask = t < seq_lengths  # Boolean tensor

            # If no batch elements are active, skip to save computation
            if not active_mask.any():
                # Fill remaining timesteps with last valid position
                pred_pos_seq[:, t:, :] = last_valid_pos.unsqueeze(1)
                break

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

            # Step the ESKF (process all batch elements for simplicity)
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

            # Apply Pen Tip Offset
            # pos_tip_w = pos_w + R_bw @ offset_b
            rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
            # offset needs shape (Batch, 3, 1) for matmul
            offset_w = (rot_mat_b_to_w @ self.pen_tip_offset.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)).squeeze(-1)
            pos_tip_w = pos_w + offset_w

            # Store position based on mask
            # For active sequences: use computed position
            # For padded sequences: use last valid position
            pred_pos_seq[:, t, :] = torch.where(
                active_mask.unsqueeze(1).expand(-1, 3),
                pos_tip_w,
                last_valid_pos
            )

            # Update last valid position for active sequences
            last_valid_pos = torch.where(
                active_mask.unsqueeze(1).expand(-1, 3),
                pos_tip_w,
                last_valid_pos
            )

        return {"pred_pos_w": pred_pos_seq}
