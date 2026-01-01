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
from model.rotation_utils import quaternion_from_two_vectors
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
        """Initializes ESKF nominal state and error covariance.

        Args:
            batch_size: Number of sequences in batch.
            dtype: Data type for tensors.
            imu_data_seq: Optional IMU data for orientation initialization via leveling.

        Returns:
            Tuple of (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error).
        """
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)

        if imu_data_seq is not None:
            # Orientation initialization via gravity leveling
            accel_init = imu_data_seq[:, 0, :3]
            accel_norm = torch.norm(accel_init, p=2, dim=-1)

            reliable_mask = (accel_norm > 4.9) & (accel_norm < 14.7)

            if reliable_mask.any():
                gravity_up_w = (
                    torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(reliable_mask.sum(), -1)
                )
                init_quat = quaternion_from_two_vectors(
                    accel_init[reliable_mask], gravity_up_w
                )
                quat_b_to_w[reliable_mask] = init_quat

            identity_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            quat_b_to_w[~reliable_mask] = identity_quat.expand((~reliable_mask).sum(), -1)
        else:
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