import torch
from typing import List, Tuple
from AEKF import ExtendedKalmanFilter
from base_hybrid_model import BaseFilterTCNModel

class AEKFTCN_model(BaseFilterTCNModel):
    """A hybrid model combining an Adaptive Extended Kalman Filter (AEKF)
    with a Temporal Convolutional Network (TCN).

    This class specializes the `BaseFilterTCNModel` by implementing the
    AEKF-specific methods for state initialization and filtering.
    """
    def __init__(self, 
                 tcn_input_size: int = 17,
                 tcn_channels: List[int] = [64, 64, 64, 64], 
                 kernel_size: int = 3, 
                 dropout: float = 0.1, 
                 device: str = 'cpu'):
        """Initializes the AEKF-TCN hybrid model.

        Args:
            tcn_input_size (int): The number of features in the TCN input.
            tcn_channels (List[int]): The number of channels in each TCN layer.
            kernel_size (int): The kernel size for TCN convolutions.
            dropout (float): The dropout rate for TCN regularization.
            device (str): The compute device ('cpu', 'cuda', 'mps').
        """
        super(AEKFTCN_model, self).__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            device=device
        )
        self.filter = ExtendedKalmanFilter(device=device)

    def _initialize_state(self, batch_size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the state and covariance for the AEKF."""
        state = torch.zeros(batch_size, 16, device=self.device, dtype=dtype)
        state[:, 6] = 1.0  # Initialize orientation to identity quaternion
        P = torch.eye(16, device=self.device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1
        return (state, P)

    def _filter_step(self, 
                     state_tuple: Tuple[torch.Tensor, torch.Tensor], 
                     imu_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                     ) -> Tuple[torch.Tensor, ...]:
        """Performs a single predict-update step of the AEKF."""
        gyro_b_raw, accel_b_raw, force_raw = imu_data
        state, P = state_tuple
        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)
        return self.filter(state, P, gyro_b_raw, accel_b_raw, force_raw, measurement)

    def _get_position_and_quaternion(self, filter_output: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts world position and orientation from the AEKF state vector."""
        state = filter_output[0]
        pos_w = state[..., :3]
        quat_b_to_w = state[..., 6:10]
        return pos_w, quat_b_to_w

    def _get_gyro_bias(self, filter_output: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extracts gyroscope bias from the AEKF state vector."""
        state = filter_output[0]
        gyro_bias_b = state[..., 10:13]
        return gyro_bias_b

if __name__ == '__main__':
    print("Running tests for AEKF_TCN.py...")
    # --- Test Parameters ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 4
    seq_length = 100
    imu_features = 7
    output_dim = 3

    print(f"Using device: {device}")

    # --- Test 1: Instantiation ---
    try:
        model = AEKFTCN_model(device=device).to(device)
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

    print("\nAll AEKF-TCN tests passed successfully.")
