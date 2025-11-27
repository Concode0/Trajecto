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
                 tcn_input_size: int = 17,
                 tcn_channels: List[int] = [64, 64, 64, 64], 
                 kernel_size: int = 3, 
                 dropout: float = 0.1, 
                 device: str = 'cpu'):
        """Initializes the ESKF-TCN hybrid model.

        Args:
            tcn_input_size (int): The number of features in the TCN input.
            tcn_channels (List[int]): The number of channels in each TCN layer.
            kernel_size (int): The kernel size for TCN convolutions.
            dropout (float): The dropout rate for TCN regularization.
            device (str): The compute device ('cpu', 'cuda', 'mps').
        """
        super(ESKFTCN_model, self).__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            device=device
        )
        self.filter = ErrorStateKalmanFilter(device=device)

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
    print("Running tests for ESKF_TCN.py...")
    # --- Test Parameters ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 4
    seq_length = 100
    imu_features = 7
    output_dim = 3

    print(f"Using device: {device}")

    # --- Test 1: Instantiation ---
    try:
        model = ESKFTCN_model(device=device).to(device)
        model.eval()
        print("Test 1 (Instantiation): PASSED")
    except Exception as e:
        print(f"Test 1 (Instantiation): FAILED - {e}")
        exit()

    # --- Test 2: Forward Pass and Shape Verification ---
    # Prepare inputs
    dummy_imu_data = torch.randn(batch_size, seq_length, imu_features, device=device)

    try:
        # Forward pass
        final_trajectory = model(dummy_imu_data)

        # --- Shape Assertions ---
        expected_shape = (batch_size, seq_length, output_dim)
        assert final_trajectory.shape == expected_shape, f"Output trajectory shape incorrect. Expected {expected_shape}, got {final_trajectory.shape}"

        # --- Stability Assertions ---
        assert not torch.any(torch.isnan(final_trajectory)), "NaN detected in final trajectory"
        assert not torch.any(torch.isinf(final_trajectory)), "Inf detected in final trajectory"

        print(f"Input IMU sequence shape: {dummy_imu_data.shape}")
        print(f"Output trajectory shape: {final_trajectory.shape}")
        print("Test 2 (Forward Pass & Shape Verification): PASSED")
    except Exception as e:
        print(f"Test 2 (Forward Pass & Shape Verification): FAILED - {e}")
        exit()

    print("\nAll ESKF-TCN tests passed successfully.")