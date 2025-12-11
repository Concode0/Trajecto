"""
This module defines the AEKFTCN_model, a hybrid architecture combining an Adaptive
Extended Kalman Filter (AEKF) with a Temporal Convolutional Network (TCN).

The AEKF component is responsible for robust state estimation (position, velocity,
orientation, biases) using IMU data, while the TCN component learns to correct
systematic errors or predict filter-specific measurements/uncertainties to
improve overall trajectory tracking accuracy in a closed-loop fashion.
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AEKF import ExtendedKalmanFilter
from base_hybrid_model import BaseFilterTCNModel
from rotation_utils import quaternion_from_two_vectors


class AEKFTCN_model(BaseFilterTCNModel):
    """A hybrid model combining an Adaptive Extended Kalman Filter (AEKF)
    with a Temporal Convolutional Network (TCN).

    This class specializes the `BaseFilterTCNModel` by implementing the
    AEKF-specific methods for state initialization and iterative filtering steps.
    It leverages the AEKF for core state estimation and integrates a TCN to
    enhance performance, typically by learning to refine measurements,
    predict process noise, or estimate residual errors.
    """

    def __init__(
        self,
        tcn_input_size: int = 20,
        tcn_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
        tcn_dilation_factors: Optional[List[int]] = None,
        dt: float = 0.01,
        use_zupt: bool = True,
    ):
        """Initializes the AEKF-TCN hybrid model.

        Args:
            tcn_input_size: The number of features in the TCN input vector.
            tcn_channels: A list specifying the number of channels (filters)
                for each layer in the TCN.
            kernel_size: The size of the convolutional kernel for TCN layers.
            dropout: The dropout rate applied within the TCN for regularization.
            device: The computation device ('cpu', 'cuda', 'mps').
            tcn_dilation_factors: Optional list of dilation factors for each
                TCN layer. If None, default dilation factors are used.
            dt: The time step (delta time) in seconds, crucial for the filter's
                integration steps.
            use_zupt: A boolean flag indicating whether Zero-Velocity Update
                (ZUPT) detection and correction should be enabled in the AEKF.
        """
        # Call the constructor of the base class (BaseFilterTCNModel)
        super().__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            device=device,
            tcn_dilation_factors=tcn_dilation_factors,
            dt=dt,
            loop_type="open",  # AEKF is typically used in an open-loop prediction with TCN correction
        )
        # Instantiate the core Adaptive Extended Kalman Filter.
        # This filter will manage the state propagation and measurement updates.
        self.filter = ExtendedKalmanFilter(device=device, dt=dt, use_zupt=use_zupt)

    def _initialize_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        imu_data_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the state vector and covariance matrix for the AEKF.

        The state vector is 16-dimensional and structured as:
        [pos_x, pos_y, pos_z,       (3) - World frame position
         vel_x, vel_y, vel_z,       (3) - World frame velocity
         quat_w, quat_x, quat_y, quat_z, (4) - Body-to-world orientation quaternion
         bias_g_x, bias_g_y, bias_g_z, (3) - Gyroscope bias in body frame
         bias_a_x, bias_a_y, bias_a_z] (3) - Accelerometer bias in body frame

        Args:
            batch_size: The number of sequences in the current batch.
            dtype: The desired data type for the state and covariance tensors.
            imu_data_seq: Optional. A batch of IMU sequences (e.g., first few
                samples) used to initialize the orientation through 'leveling'.

        Returns:
            A tuple containing:
                - state (torch.Tensor): The initialized state vector for the batch.
                - P (torch.Tensor): The initialized covariance matrix for the batch.
        """
        # Initialize state vector with zeros, matching the batch size and state dimension.
        state = torch.zeros(batch_size, 16, device=self.device, dtype=dtype)

        if imu_data_seq is not None:
            # Use the first accelerometer reading to determine initial orientation (leveling).
            # This assumes the device starts static or near-static, such that the dominant
            # acceleration measured is gravity. The goal is to align the body frame's
            # measured gravity vector with the world frame's known gravity vector.
            accel_init = imu_data_seq[:, 0, :3]  # First 3 components are accelerometer data
            accel_norm = torch.norm(accel_init, p=2, dim=-1)  # Magnitude of initial acceleration

            # `reliable_mask` identifies samples where the initial accelerometer reading
            # is close to the magnitude of gravity (approx 9.81 m/s^2). This helps to
            # filter out cases where the sensor might be in free-fall (low accel_norm)
            # or experiencing strong external forces (high accel_norm), which would
            # make gravity-based leveling unreliable.
            reliable_mask = (accel_norm > 4.9) & (
                accel_norm < 14.7
            )  # Thresholds [~0.5g, ~1.5g]

            quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)

            if reliable_mask.any():
                # Target vector is gravity pointing up (or down depending on convention)
                # along the Z-axis in the world frame. Here, we assume gravity is positive Z.
                gravity_up_w = (
                    torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(reliable_mask.sum(), -1)
                )

                # `quaternion_from_two_vectors` computes the quaternion required to rotate
                # the measured initial acceleration vector (which should be ~[-gx, -gy, -gz]
                # in body frame due to gravity) to align with the world's positive Z-axis.
                # This effectively "levels" the coordinate frame.
                init_quat = quaternion_from_two_vectors(
                    accel_init[reliable_mask], gravity_up_w
                )
                quat_b_to_w[reliable_mask] = init_quat

            # For unreliable measurements (where accel_norm is not near gravity),
            # fall back to an identity quaternion [1, 0, 0, 0]. This represents no
            # initial rotation and is a safe default, albeit less accurate.
            identity_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            quat_b_to_w[~reliable_mask] = identity_quat.expand((~reliable_mask).sum(), -1)

            # Assign the calculated initial quaternion to the state vector (indices 6 to 9).
            state[:, 6:10] = quat_b_to_w

        else:
            # If no IMU data is provided for initialization, default to an identity quaternion.
            state[:, 6] = 1.0  # w component of quaternion

        # Initialize the covariance matrix `P`.
        # `P` represents the uncertainty in the initial state estimate. A diagonal
        # matrix signifies uncorrelated initial errors. The `0.1` scalar
        # provides a reasonable initial uncertainty. Higher values imply less
        # confidence in the initial state and can lead to faster convergence.
        P = (
            torch.eye(16, device=self.device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            * 0.1
        )
        return (state, P)

    def _filter_step(
        self,
        state_tuple: Tuple[torch.Tensor, torch.Tensor],
        imu_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tcn_output: Optional[
            torch.Tensor
        ] = None,  # TCN output would typically be a dict
    ) -> Tuple[torch.Tensor, ...]:
        """Performs a single predict-update step of the AEKF.

        This method encapsulates the core EKF operations (prediction based on
        motion model and update based on sensor measurements) for one time step.

        Args:
            state_tuple: A tuple containing the current state vector (x) and
                its covariance matrix (P).
            imu_data: A tuple containing raw gyroscope data, raw accelerometer
                data, and raw force sensor data for the current time step.
            tcn_output: Optional. Output from the TCN, which might be used to
                adapt measurement noise, or provide additional corrections.

        Returns:
            A tuple containing the updated state vector, covariance matrix,
            and other relevant filter outputs (e.g., innovation).
        """
        gyro_b_raw, accel_b_raw, force_raw = imu_data
        state, P = state_tuple

        # Construct the measurement vector from raw IMU data.
        # In this AEKF setup, both accelerometer and gyroscope readings are
        # used as measurements.
        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)

        # Call the EKF's forward method to execute one full predict-update cycle.
        return self.filter(
            state, P, gyro_b_raw, accel_b_raw, force_raw, measurement, tcn_output
        )

    def _get_position_and_quaternion(
        self, filter_output: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts world frame position and body-to-world orientation quaternion
        from the AEKF state vector.

        Args:
            filter_output: The full output tuple from the AEKF, where the first
                element is the state vector.

        Returns:
            A tuple containing:
                - pos_w (torch.Tensor): The 3D position in the world frame.
                - quat_b_to_w (torch.Tensor): The 4-element body-to-world quaternion.
        """
        state = filter_output[0]
        # Position is the first 3 elements of the state vector.
        pos_w = state[..., :3]
        # Quaternion is elements 6 through 9 (inclusive) of the state vector.
        quat_b_to_w = state[..., 6:10]
        return pos_w, quat_b_to_w

    def _get_gyro_bias(
        self, filter_output: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Extracts gyroscope bias (in body frame) from the AEKF state vector.

        Args:
            filter_output: The full output tuple from the AEKF, where the first
                element is the state vector.

        Returns:
            gyro_bias_b (torch.Tensor): The 3D gyroscope bias in the body frame.
        """
        state = filter_output[0]
        # Gyroscope bias is elements 10 through 12 (inclusive) of the state vector.
        gyro_bias_b = state[..., 10:13]
        return gyro_bias_b


if __name__ == "__main__":
    # Test case to verify functionality and tensor shapes of the AEKFTCN_model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model with default parameters.
    model = AEKFTCN_model(device=device).to(device)

    # Create dummy IMU data for a batch of sequences.
    # The IMU data typically includes accelerometer (3), gyroscope (3), and force (1)
    # readings, totaling 7 features.
    batch_size = 4
    sequence_length = 100
    imu_features = 7
    dummy_imu_data_raw = torch.randn(
        batch_size, sequence_length, imu_features, device=device
    )

    # Simulate a realistic initial acceleration for one sample in the batch.
    # This simulates a static start where gravity is the dominant acceleration,
    # allowing the `_initialize_state` method to perform effective leveling.
    dummy_imu_data_raw[0, 0, :3] = torch.tensor(
        [0.5, 0.5, 9.8], device=device
    )  # accel_x, accel_y, accel_z near gravity magnitude

    # For the `forward` pass, `imu_data_norm` is also required, even if it's
    # just a copy of raw data for this dummy test. In a real scenario, this
    # would be pre-normalized sensor data.
    dummy_imu_data_norm = dummy_imu_data_raw.clone()

    # Run the model forward pass.
    model_output = model(dummy_imu_data_raw, dummy_imu_data_norm)

    print(f"\nInput IMU sequence shape: {dummy_imu_data_raw.shape}")
    print("Output dictionary shapes:")
    for key, value in model_output.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}")
        else:
            print(f"  - {key}: {type(value)}") # Handle non-tensor outputs if any

    # Assertions to ensure the output shapes are as expected.
    assert "pred_pos_w" in model_output
    assert model_output["pred_pos_w"].shape == (batch_size, sequence_length, 3)

    print("\nAEKF-TCN model created and tested successfully.")