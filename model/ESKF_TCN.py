"""
Hybrid ESKF-TCN model integrating Error-State Kalman Filter with Temporal
Convolutional Network for enhanced trajectory estimation.

The TCN processes ESKF-derived features to predict velocity corrections,
adaptive noise parameters, and ZUPT probabilities in closed-loop configuration.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.ESKF import ErrorStateKalmanFilter
from model.base_hybrid_model import BaseFilterTCNModel
from model.rotation_utils import quaternion_from_two_vectors, quaternion_to_rotation_matrix
from model.config import Config


class ESKFTCN_model(BaseFilterTCNModel):
    """Closed-loop hybrid ESKF-TCN model for trajectory estimation.

    Combines ESKF error-state formulation with TCN-predicted corrections
    for velocity, adaptive noise, and ZUPT detection in closed-loop fashion.
    """

    def __init__(
        self,
        tcn_input_size: int = Config.ESKFTCN.TCN_INPUT_SIZE,
        tcn_channels: List[int] = Config.ESKFTCN.TCN_CHANNELS,
        kernel_size: int = Config.ESKFTCN.KERNEL_SIZE,
        dropout: float = Config.ESKFTCN.DROPOUT,
        device: str = "cpu",
        tcn_dilation_factors: Optional[List[int]] = Config.ESKFTCN.TCN_DILATION_FACTORS,
        dt: float = Config.DT,
        use_zupt: bool = Config.ESKFTCN.USE_ZUPT,
        use_tcn_zupt: bool = Config.ESKFTCN.USE_TCN_ZUPT,
        separable: bool = Config.ESKFTCN.USE_SEPARABLE_CONV,
    ):
        """Initializes the ESKF-TCN hybrid model.

        Args:
            tcn_input_size: The number of features in the TCN input vector.
            tcn_channels: A list specifying the number of channels (filters)
                for each layer in the TCN.
            kernel_size: The kernel size for TCN convolutions.
            dropout: The dropout rate applied within the TCN for regularization.
            device: The computation device ('cpu', 'cuda', 'mps').
            tcn_dilation_factors: Optional list of dilation factors for each
                TCN layer. If None, default dilation factors are used.
            dt: The time step (delta time) in seconds, crucial for the filter's
                integration steps.
            use_zupt: A boolean flag indicating whether traditional ZUPT detection
                and correction should be enabled in the ESKF.
            use_tcn_zupt: A boolean flag. If True, the ZUPT decision within the
                ESKF's forward pass is made based on the TCN's output (`zupt_prob`).
                If False, the classic ZUPT detector in `ESKF` is used if `use_zupt` is True.
            separable: Whether to use Depthwise Separable Convolutions in TCN.
        """
        super().__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            device=device,
            tcn_dilation_factors=tcn_dilation_factors,
            dt=dt,
            loop_type="closed",
            separable=separable,
        )
        self.filter = ErrorStateKalmanFilter(
            device=device, dt=dt, use_zupt=use_zupt, use_tcn_zupt=use_tcn_zupt
        )

    def _initialize_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        imu_data_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Initializes ESKF nominal state and error covariance with robust estimation.

        Leverages the Tap-Wait-Write-Tap protocol's ~2s static period to:
        1. Detect actual static duration per sample (gyro variance monitoring)
        2. Average accelerometer readings for robust gravity alignment
        3. Estimate gyroscope and accelerometer biases from static data
        4. Initialize error covariance based on static period variance

        Args:
            batch_size: Number of sequences in batch.
            dtype: Data type for tensors.
            imu_data_seq: Optional IMU data [Batch, Seq, 7] for initialization.
                Channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, force]

        Returns:
            Tuple of (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error).
        """
        # Initialize position and velocity (ZUPT assumption at start)
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)

        if imu_data_seq is not None and imu_data_seq.shape[1] > 10:
            # --- Enhanced Initialization from Static Period ---
            # Static buffer duration from acquisition protocol (see CLAUDE.md)
            STATIC_INIT_S = 2.0
            seq_len = imu_data_seq.shape[1]
            max_static_samples = min(seq_len, int(STATIC_INIT_S / self.dt))

            if max_static_samples >= 20:  # Need minimum samples for reliable estimation
                # Step 1: Detect actual static period per sample
                # Use gyroscope variance as motion detector
                window_size = Config.ZUPT_WINDOW_SIZE
                gyro_data = imu_data_seq[:, :max_static_samples, 3:6]  # [batch, time, 3]

                # Initialize per-sample static counts
                static_samples = torch.full((batch_size,), max_static_samples,
                                           dtype=torch.long, device=self.device)
                still_searching = torch.ones(batch_size, dtype=torch.bool, device=self.device)

                # Detect motion onset for each sample independently
                for t in range(window_size, max_static_samples, window_size):
                    if not still_searching.any():
                        break

                    window = gyro_data[:, t:t+window_size, :]  # [batch, window, 3]
                    gyro_var_per_sample = torch.var(window, dim=1)  # [batch, 3]
                    gyro_var_mean = gyro_var_per_sample.mean(dim=1)  # [batch]

                    # Motion detected when gyro variance exceeds threshold
                    motion_detected = gyro_var_mean > 0.002  # rad²/s²
                    newly_detected = motion_detected & still_searching
                    static_samples[newly_detected] = t
                    still_searching[newly_detected] = False

                # Enforce minimum samples for numerical stability
                static_samples = torch.maximum(static_samples,
                                               torch.full_like(static_samples, 20))

                # Step 2: Average accelerometer reading during static period with outlier rejection
                avg_accel_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
                for b in range(batch_size):
                    # RANSAC-based outlier rejection using MAD (Median Absolute Deviation)
                    accel_static = imu_data_seq[b, :static_samples[b], 0:3]

                    # Compute median and MAD for robust statistics
                    median_accel = torch.median(accel_static, dim=0).values
                    mad = torch.median(torch.abs(accel_static - median_accel), dim=0).values

                    # Reject outliers: samples > 3*MAD from median (equivalent to ~3σ for Gaussian)
                    deviation = torch.abs(accel_static - median_accel)
                    inlier_mask = torch.all(deviation < 3.0 * (mad + 1e-6), dim=1)

                    # Average only inliers (fallback to all samples if too many rejected)
                    if inlier_mask.sum() > 10:  # Need minimum 10 samples
                        avg_accel_b[b] = accel_static[inlier_mask].mean(dim=0)
                    else:
                        # Fallback: use median if outlier rejection too aggressive
                        avg_accel_b[b] = median_accel

                # Step 3: Gravity alignment with averaged measurement
                accel_norm = torch.norm(avg_accel_b, p=2, dim=-1)
                reliable_mask = (accel_norm > 4.9) & (accel_norm < 14.7)  # ~[0.5g, 1.5g]

                # Initialize with identity quaternion as fallback
                quat_b_to_w[:, 0] = 1.0

                if reliable_mask.any():
                    # CRITICAL: Accelerometer measures specific force (reaction force), not gravity
                    # When stationary, it measures normal force pointing UP (opposite of gravity)
                    # Gravity in world frame points DOWN: [0, 0, -9.81]
                    # Measured accel points UP in body frame when Z-axis is vertical
                    # Therefore, align measured accel (UP) with world UP direction [0, 0, +1]
                    world_up = torch.zeros(reliable_mask.sum(), 3,
                                          device=self.device, dtype=dtype)
                    world_up[:, 2] = 1.0  # Unit vector pointing UP

                    # Normalize measured acceleration for pure orientation alignment
                    avg_accel_normalized = torch.nn.functional.normalize(
                        avg_accel_b[reliable_mask], p=2, dim=-1
                    )

                    # Quaternion aligns measured accel (specific force UP) with world UP
                    init_quat = quaternion_from_two_vectors(
                        avg_accel_normalized,
                        world_up
                    )
                    quat_b_to_w[reliable_mask] = init_quat

                # Step 4: Estimate gyroscope bias from static period with outlier rejection
                for b in range(batch_size):
                    gyro_static = imu_data_seq[b, :static_samples[b], 3:6]

                    # Robust outlier rejection for gyroscope
                    median_gyro = torch.median(gyro_static, dim=0).values
                    mad_gyro = torch.median(torch.abs(gyro_static - median_gyro), dim=0).values

                    deviation_gyro = torch.abs(gyro_static - median_gyro)
                    inlier_mask_gyro = torch.all(deviation_gyro < 3.0 * (mad_gyro + 1e-6), dim=1)

                    if inlier_mask_gyro.sum() > 10:
                        gyro_bias_b[b] = gyro_static[inlier_mask_gyro].mean(dim=0)
                    else:
                        gyro_bias_b[b] = median_gyro

                # Step 5: Estimate accelerometer bias from static period
                # Bias = measured_accel - expected_accel (gravity in body frame)
                # After leveling, expected accel in body frame ≈ R_w_to_b @ gravity_w
                rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
                rot_mat_w_to_b = rot_mat_b_to_w.transpose(-1, -2)

                # Expected gravity in body frame after alignment
                gravity_w = torch.tensor([0.0, 0.0, -Config.GRAVITY_MAGNITUDE],
                                        device=self.device, dtype=dtype)
                expected_accel_b = (rot_mat_w_to_b @ gravity_w.unsqueeze(-1)).squeeze(-1)

                # Accelerometer bias = measured - expected
                accel_bias_b = avg_accel_b - expected_accel_b

                # Step 6: Initialize error covariance based on static variance
                # Use actual variance from static period for realistic uncertainty
                P_error = torch.eye(15, device=self.device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

                for b in range(batch_size):
                    static_imu = imu_data_seq[b, :static_samples[b], :]

                    # Position uncertainty (start at origin)
                    P_error[b, 0:3, 0:3] *= 0.01  # 1cm initial position uncertainty

                    # Velocity uncertainty (ZUPT assumption)
                    P_error[b, 3:6, 3:6] *= 0.001  # 1mm/s initial velocity uncertainty

                    # Orientation uncertainty from accelerometer variance
                    accel_var = torch.var(static_imu[:, 0:3], dim=0)
                    ori_uncertainty = (accel_var / Config.GRAVITY_MAGNITUDE**2).clamp(min=1e-4, max=0.1)
                    P_error[b, 6:9, 6:9] = torch.diag(ori_uncertainty)

                    # Gyro bias uncertainty from static variance
                    gyro_var = torch.var(static_imu[:, 3:6], dim=0).clamp(min=1e-6, max=1e-3)
                    P_error[b, 9:12, 9:12] = torch.diag(gyro_var)

                    # Accel bias uncertainty from static variance
                    accel_bias_var = accel_var.clamp(min=1e-4, max=1.0)
                    P_error[b, 12:15, 12:15] = torch.diag(accel_bias_var)

            else:
                # Fallback: insufficient static data, use simple initialization
                accel_init = imu_data_seq[:, 0, :3]
                accel_norm = torch.norm(accel_init, p=2, dim=-1)
                reliable_mask = (accel_norm > 4.9) & (accel_norm < 14.7)

                quat_b_to_w[:, 0] = 1.0

                if reliable_mask.any():
                    # Accelerometer measures UP when stationary (specific force)
                    world_up = torch.zeros(reliable_mask.sum(), 3,
                                          device=self.device, dtype=dtype)
                    world_up[:, 2] = 1.0  # Align with UP, not DOWN

                    accel_normalized = torch.nn.functional.normalize(
                        accel_init[reliable_mask], p=2, dim=-1
                    )
                    init_quat = quaternion_from_two_vectors(accel_normalized, world_up)
                    quat_b_to_w[reliable_mask] = init_quat

                # Default covariance
                P_error = (
                    torch.eye(15, device=self.device, dtype=dtype)
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                    * 0.1
                )
        else:
            # No IMU data provided: use identity orientation and default covariance
            quat_b_to_w[:, 0] = 1.0
            P_error = (
                torch.eye(15, device=self.device, dtype=dtype)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
                * 0.1
            )

        return (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error)

    def _filter_step(
        self,
        state_tuple: Tuple[torch.Tensor, ...],
        imu_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Performs single ESKF predict-update cycle.

        Args:
            state_tuple: Current state (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error).
            imu_data: Sensor data (gyro_b_raw, accel_b_raw, force_raw).
            tcn_output: Optional TCN predictions (vel_corr, covariance_R, zupt_prob).

        Returns:
            Updated state tuple and features.
        """
        gyro_b_raw, accel_b_raw, force_raw = imu_data
        pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error = state_tuple

        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)

        return self.filter(
            pos_w,
            vel_w,
            quat_b_to_w,
            gyro_bias_b,
            accel_bias_b,
            P_error,
            gyro_b_raw,
            accel_b_raw,
            force_raw,
            measurement,
            tcn_output,
        )

    def _get_position_and_quaternion(
        self,
        filter_output: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts position and quaternion from ESKF output.

        Args:
            filter_output: ESKF output tuple.

        Returns:
            Tuple of (pos_w, quat_b_to_w).
        """
        pos_w = filter_output[0]
        quat_b_to_w = filter_output[2]
        return pos_w, quat_b_to_w

    def _get_gyro_bias(
        self,
        filter_output: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Extracts gyroscope bias from ESKF output.

        Args:
            filter_output: ESKF output tuple.

        Returns:
            Gyroscope bias in body frame.
        """
        gyro_bias_b = filter_output[3]
        return gyro_bias_b


if __name__ == "__main__":
    # Test case to verify functionality and tensor shapes of the ESKFTCN_model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model with default parameters, enabling TCN-based ZUPT.
    model = ESKFTCN_model(device=device).to(device)

    # Create dummy IMU data for a batch of sequences.
    batch_size, sequence_length, imu_features = 4, 100, 7
    dummy_imu_data_raw = torch.randn(
        batch_size, sequence_length, imu_features, device=device
    )
    # Simulate a realistic initial acceleration for one sample in the batch.
    # This aids the `_initialize_state` method in performing effective leveling
    # by providing a clear gravity vector.
    dummy_imu_data_raw[0, 0, :3] = torch.tensor(
        [0.5, 0.5, Config.GRAVITY_MAGNITUDE], device=device
    )  # accel_x, accel_y, accel_z near gravity magnitude

    # For the `forward` pass of BaseFilterTCNModel, `imu_data_norm` is required.
    # For this test, it's a clone of raw data, but in a real scenario, it would be
    # pre-normalized sensor data.
    dummy_imu_data_norm = dummy_imu_data_raw.clone()

    # Run the model forward pass.
    model_output = model(dummy_imu_data_raw, dummy_imu_data_norm)

    print(f"\nInput IMU sequence shape: {dummy_imu_data_raw.shape}")
    print("Output dictionary shapes:")
    for key, value in model_output.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}")
        else:
            print(f"  - {key}: {type(value)}")  # Handle non-tensor outputs if any

    # Assertions to ensure the output shapes are as expected.
    assert "pred_pos_w" in model_output
    assert model_output["pred_pos_w"].shape == (batch_size, sequence_length, 3)

    print("\nClosed-loop ESKF-TCN model created and tested successfully.")