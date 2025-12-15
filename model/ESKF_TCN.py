"""
This module defines the ESKFTCN_model, a hybrid architecture that integrates an
Error-State Kalman Filter (ESKF) with a Temporal Convolutional Network (TCN).

The ESKF is a robust method for estimating the full IMU state (position, velocity,
orientation, and sensor biases) by propagating a nominal state and correcting an
error state. The TCN component, in this hybrid model, processes features derived
from the ESKF's operation and predicts corrections or adaptive parameters,
particularly concerning Zero-Velocity Updates (ZUPT) and velocity corrections,
to enhance the overall trajectory estimation accuracy.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

# Adjust sys.path for relative imports to find ESKF, BaseFilterTCNModel, etc.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.ESKF import ErrorStateKalmanFilter
from model.base_hybrid_model import BaseFilterTCNModel
from model.rotation_utils import quaternion_from_two_vectors
from model.config import Config


class ESKFTCN_model(BaseFilterTCNModel):
    """A hybrid model combining a closed-loop Error-State Kalman Filter (ESKF)
    with a multi-head Temporal Convolutional Network (TCN).

    This class specializes the `BaseFilterTCNModel` by implementing the
    ESKF-specific methods for state initialization and iterative filtering steps.
    It leverages the ESKF's error-state formulation for precise state estimation
    and integrates a TCN to refine ZUPT decisions and provide additional
    velocity corrections, operating in a closed-loop fashion where TCN outputs
    directly influence the filter's updates.
    """

    def __init__(
        self,
        tcn_input_size: int = Config.ESKFTCN.TCN_INPUT_SIZE,
        tcn_channels: List[int] = Config.ESKFTCN.TCN_CHANNELS,
        kernel_size: int = Config.ESKFTCN.KERNEL_SIZE,
        dropout: float = Config.ESKFTCN.DROPOUT,
        device: str = "cpu",
        tcn_dilation_factors: Optional[List[int]] = None,
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
        # Call the constructor of the base class (BaseFilterTCNModel)
        super().__init__(
            tcn_input_size=tcn_input_size,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            device=device,
            tcn_dilation_factors=tcn_dilation_factors,
            dt=dt,
            loop_type="closed",  # ESKF-TCN typically operates in a closed-loop fashion
            separable=separable,
        )
        # Instantiate the core Error-State Kalman Filter.
        # This filter will manage the nominal state propagation, error state
        # prediction and measurement updates, and error injection.
        self.filter = ErrorStateKalmanFilter(
            device=device, dt=dt, use_zupt=use_zupt, use_tcn_zupt=use_tcn_zupt
        )

    def _initialize_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        imu_data_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Initializes the nominal state vector and error covariance matrix for the ESKF.

        For an Error-State Kalman Filter (ESKF), the filter tracks a nominal state
        (p, v, q, bg, ba) and a separate 15-dimensional error state (δp, δv, δθ, δbg, δba).
        This method initializes the *nominal* components and the initial *error covariance*.

        Nominal State Components initialized:
        - Position in the world frame (3)
        - Velocity in the world frame (3)
        - Body-to-world orientation quaternion (4)
        - Gyroscope bias in body frame (3)
        - Accelerometer bias in body frame (3)

        Args:
            batch_size: The number of sequences in the current batch.
            dtype: The desired data type for the state and covariance tensors.
            imu_data_seq: Optional. A batch of IMU sequences (e.g., first few
                samples) used to initialize the orientation through 'leveling'.

        Returns:
            A tuple containing:
                - pos_w (torch.Tensor): Initial world-frame position.
                - vel_w (torch.Tensor): Initial world-frame velocity.
                - quat_b_to_w (torch.Tensor): Initial body-to-world quaternion.
                - gyro_bias_b (torch.Tensor): Initial gyroscope bias.
                - accel_bias_b (torch.Tensor): Initial accelerometer bias.
                - P_error (torch.Tensor): The initial error covariance matrix (15x15).
        """
        # Initialize nominal state components to zeros or identity.
        pos_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        vel_w = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        gyro_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)
        accel_bias_b = torch.zeros(batch_size, 3, device=self.device, dtype=dtype)

        quat_b_to_w = torch.zeros(batch_size, 4, device=self.device, dtype=dtype)

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
        else:
            # If no IMU data is provided for initialization, default to an identity quaternion.
            quat_b_to_w[:, 0] = 1.0  # w component of quaternion

        # Initialize the error covariance matrix `P_error`.
        # The ESKF's error state is 15-dimensional (3 for δpos, 3 for δvel, 3 for δangle,
        # 3 for δgyro_bias, 3 for δaccel_bias).
        # `P_error` represents the uncertainty in the initial error state estimates.
        # A diagonal matrix signifies uncorrelated initial errors. The `0.1` scalar
        # provides a reasonable initial uncertainty.
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
        """Performs a single predict-update step of the ESKF.

        This method encapsulates the core ESKF operations (nominal state propagation,
        error state prediction and update, and error injection) for one time step.

        Args:
            state_tuple (Tuple[torch.Tensor, ...]): Current nominal state and covariance.
                - Components: (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error)
            imu_data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Sensor data for current step.
                - Components: (gyro_b_raw, accel_b_raw, force_raw)
            tcn_output (Optional[Dict[str, torch.Tensor]]): TCN predictions.
                - Keys: "vel_corr", "covariance_R", "zupt_prob"

        Returns:
            Tuple[torch.Tensor, ...]: Updated state tuple and features.
                - Components: (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new, P_error_final, tcn_features)
        """
        gyro_b_raw, accel_b_raw, force_raw = imu_data
        pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error = state_tuple

        # Construct the measurement vector from raw IMU data.
        # In this ESKF setup, both accelerometer and gyroscope readings are
        # used as measurements.
        measurement = torch.cat([accel_b_raw, gyro_b_raw], dim=-1)

        # Call the ESKF's forward method to execute one full predict-update cycle.
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
        """Extracts world frame position and body-to-world orientation quaternion
        from the ESKF's nominal state output.

        Args:
            filter_output: The full output tuple from the ESKF's step,
                which contains the updated nominal state components.

        Returns:
            A tuple containing:
                - pos_w (torch.Tensor): The 3D position in the world frame (index 0).
                - quat_b_to_w (torch.Tensor): The 4-element body-to-world quaternion (index 2).
        """
        # Nominal state components are returned as separate tensors in the tuple.
        pos_w = filter_output[0]
        quat_b_to_w = filter_output[2]
        return pos_w, quat_b_to_w

    def _get_gyro_bias(
        self,
        filter_output: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Extracts gyroscope bias (in body frame) from the ESKF's nominal state output.

        Args:
            filter_output: The full output tuple from the ESKF's step,
                which contains the updated nominal state components.

        Returns:
            gyro_bias_b (torch.Tensor): The 3D gyroscope bias in the body frame (index 3).
        """
        # Nominal state components are returned as separate tensors in the tuple.
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