from typing import Dict, List, Optional, Tuple
import sys, os

import torch
import torch.nn as nn

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.AEKF import ExtendedKalmanFilter
from model.base_hybrid_model import BaseFilterTCNModel
from model.config import Config


class AEKFTCN_model(BaseFilterTCNModel):
    """A hybrid model combining an Analytically-linearized Extended Kalman Filter
    (AEKF) with a Temporal Convolutional Network (TCN).

    The AEKF provides a physics-based state estimate, while the TCN learns to
    predict the residual errors in the AEKF's velocity estimates, correcting
    for unmodeled dynamics and sensor errors.
    """

    def __init__(
        self,
        device: torch.device,
        tcn_input_size: int = Config.AEKFTCN.TCN_INPUT_SIZE,
        tcn_num_channels: list = Config.AEKFTCN.TCN_NUM_CHANNELS,
        tcn_kernel_size: int = Config.AEKFTCN.TCN_KERNEL_SIZE,
        tcn_dropout: float = Config.AEKFTCN.TCN_DROPOUT,
        tcn_dilation_factors: Optional[List[int]] = Config.AEKFTCN.TCN_DILATION_FACTORS,
        dt: float = Config.DT,
        separable: bool = Config.AEKFTCN.USE_SEPARABLE_CONV,
    ) -> None:
        super().__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_num_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            device=device,
            tcn_dilation_factors=tcn_dilation_factors,
            dt=dt,
            loop_type="open",
            separable=separable,
        )

        self.filter = ExtendedKalmanFilter(device=device, dt=dt)

    def _initialize_state(self, batch_size, dtype, imu_data_seq=None):
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)

        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)
        quat_b_to_w[:, 0] = 1.0

        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)

        state = torch.cat([pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b], dim=-1)

        P_cov = torch.eye(16, device=self.device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        return (state, P_cov)

    def _filter_step(
        self,
        state_tuple: Tuple[torch.Tensor, ...],
        imu_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Performs a single predict-update step of the AEKF.

        Args:
            state_tuple (Tuple[torch.Tensor, ...]): Current state and covariance.
                - Components: (state_vector, P_covariance)
            imu_data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Sensor data for current step.
                - Components: (gyro_b_raw, accel_b_raw, force_raw)
            tcn_output (Optional[Dict[str, torch.Tensor]]): TCN predictions (unused in Open-Loop AEKF-TCN).

        Returns:
            Tuple[torch.Tensor, ...]: Updated state tuple and features.
                - Components: (state_updated, P_updated, tcn_features)
        """
        # Unpack inputs
        state, P_cov = state_tuple
        gyro_b_raw, accel_b_raw, force_raw = imu_data

        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)

        return self.filter(
            state=state,
            P_covariance=P_cov,
            gyro_body_raw=gyro_b_raw,
            accel_body_raw=accel_b_raw,
            force_raw=force_raw,
            measurement=measurement,
            tcn_output=None
        )

    def _get_position_and_quaternion(self, filter_output):
        state = filter_output[0]

        # Slicing based on AEKF state definition
        pos_w = state[..., 0:3]
        quat_b_to_w = state[..., 6:10]

        return pos_w, quat_b_to_w

    def _get_gyro_bias(self, filter_output):
        state = filter_output[0]
        # Gyro bias is at indices 10:13
        return state[..., 10:13]

if __name__ == "__main__":
    # Test case to verify functionality and tensor shapes of the AEKFTCN_model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model with default parameters.
    # AEKF-TCN operates in open-loop by default, meaning TCN output (vel_corr)
    # is a separate output, not fed back into the EKF's update.
    # The AEKF itself handles noise adaptivity.
    model = AEKFTCN_model(device=device).to(device)

    # Create dummy IMU data for a batch of sequences.
    batch_size, sequence_length, imu_features = 4, 100, 7
    dummy_imu_data_raw = torch.randn(
        batch_size, sequence_length, imu_features, device=device
    )
    # Simulate a realistic initial acceleration for one sample in the batch.
    dummy_imu_data_raw[0, 0, :3] = torch.tensor(
        [0.5, 0.5, Config.GRAVITY_MAGNITUDE], device=device
    )

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
            print(f"  - {key}: {type(value)}")

    # Assertions to ensure the output shapes are as expected.
    assert "pred_pos_w" in model_output
    assert model_output["pred_pos_w"].shape == (batch_size, sequence_length, 3)
    assert "pred_vel_resid_b" in model_output
    assert model_output["pred_vel_resid_b"].shape == (batch_size, sequence_length, 3)

    print("\nOpen-loop AEKF-TCN model created and tested successfully.")
