import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from typing import List, Tuple
from ESKF import ErrorStateKalmanFilter
from base_hybrid_model import BaseFilterTCNModel

class ESKFTCN_model(BaseFilterTCNModel):
    """A hybrid model combining an Error-State Kalman Filter (ESKF)
    with a Temporal Convolutional Network (TCN).

    This class specializes the `BaseFilterTCNModel` by implementing the
    ESKF-specific methods for state initialization and filtering.
    """
    def __init__(self, 
                 tcn_input_size: int = 20,
                 tcn_channels: List[int] = [64, 64, 64, 64], 
                 kernel_size: int = 3, 
                 dropout: float = 0.1, 
                 device: str = 'cpu',
                 tcn_dilation_factors: List[int] = None,
                 dt: float = 0.01):
        """Initializes the ESKF-TCN hybrid model.

        Args:
            tcn_input_size (int): The number of features in the TCN input.
            tcn_channels (List[int]): The number of channels in each TCN layer.
            kernel_size (int): The kernel size for TCN convolutions.
            dropout (float): The dropout rate for TCN regularization.
            device (str): The compute device ('cpu', 'cuda', 'mps').
            tcn_dilation_factors (List[int], optional): Dilation factor for each TCN layer.
            dt (float): The time delta for integration.
        """
        super(ESKFTCN_model, self).__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            device=device,
            tcn_dilation_factors=tcn_dilation_factors,
            dt=dt
        )
        self.filter = ErrorStateKalmanFilter(device=device, dt=dt)

    def _initialize_state(self, batch_size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Initializes the nominal state and error covariance for the ESKF."""
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)
        quat_b_to_w[:, 0] = 1.0  # Identity quaternion
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        P_error = torch.eye(15, device=self.device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1
        return (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error)

    def _filter_step(self, 
                     state_tuple: Tuple[torch.Tensor, ...], 
                     imu_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                     ) -> Tuple[torch.Tensor, ...]:
        """Performs a single predict-update step of the ESKF."""
        gyro_b_raw, accel_b_raw, force_raw = imu_data
        pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error = state_tuple
        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)
        return self.filter(pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error, gyro_b_raw, accel_b_raw, force_raw, measurement)

    def _get_position_and_quaternion(self, filter_output: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts world position and orientation from the ESKF output tuple."""
        pos_w = filter_output[0]
        quat_b_to_w = filter_output[2]
        return pos_w, quat_b_to_w

    def _get_gyro_bias(self, filter_output: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extracts gyroscope bias from the ESKF output tuple."""
        gyro_bias_b = filter_output[3]
        return gyro_bias_b

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = ESKFTCN_model(device=device).to(device)
    dummy_imu_data = torch.randn(4, 100, 7, device=device)
    model_output = model(dummy_imu_data)

    print(f"Input IMU sequence shape: {dummy_imu_data.shape}")
    print("Output dictionary shapes:")
    for key, value in model_output.items():
        print(f"  - {key}: {value.shape}")

    assert 'pred_pos_w' in model_output
    assert model_output['pred_pos_w'].shape == (4, 100, 3)

    print("ESKF-TCN model created and tested successfully.")