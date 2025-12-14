"""
This module defines a standalone Temporal Convolutional Network (TCN) model
(`OnlyTCN`) designed for direct trajectory correction.

Instead of being integrated into a Kalman filter, this TCN model takes raw
and normalized IMU data to first compute a naive, double-integrated trajectory
and then predicts a correction to this baseline, directly outputting a
corrected 3D position trajectory.
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn

# Adjust sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

class OnlyTCN(nn.Module):
    """A standalone Temporal Convolutional Network (TCN) for direct trajectory correction.

    This model takes raw IMU data to compute a naive trajectory via double
    integration and uses normalized IMU data to predict a residual correction
    for this naive trajectory. The final output is a corrected 3D position
    trajectory.
    """

    def __init__(
        self,
        device: str = "cpu",
        input_size: int = Config.OnlyTCN.INPUT_SIZE,
        output_size: int = Config.OnlyTCN.OUTPUT_SIZE,
        tcn_channels: List[int] = Config.OnlyTCN.TCN_CHANNELS,
        kernel_size: int = Config.OnlyTCN.KERNEL_SIZE,
        dropout: float = Config.OnlyTCN.DROPOUT,
        dt: float = Config.DT,
        tcn_dilation_factors: Optional[List[int]] = None, # Add tcn_dilation_factors to __init__ for consistency
    ):
        """Initializes the OnlyTCN model.

        Args:
            device: The computation device ('cpu', 'cuda', 'mps').
            input_size: The number of features in the input IMU sequence
                (e.g., 3 for accel, 3 for gyro, 1 for force = 7).
            output_size: The number of features in the output (e.g., 3 for 3D position).
            tcn_channels: A list specifying the number of channels (filters)
                for each convolutional layer in the TCN.
            kernel_size: The size of the convolutional kernel for TCN layers.
            dropout: The dropout rate applied within the TCN for regularization.
            dt: The time step (delta time) in seconds, used for numerical
                integration of the raw IMU data.
            tcn_dilation_factors: Optional list of dilation factors for each
                TCN layer. If None, defaults to powers of 2 (1, 2, 4, 8, ...).
        """
        super().__init__()

        self.dt = dt  # Time step for numerical integration.
        self.tcn_layers = nn.ModuleList()  # Container for TCN blocks.
        in_channels = input_size

        # Default dilation factors if not provided, typically powers of 2.
        if tcn_dilation_factors is None:
            tcn_dilation_factors = [2**i for i in range(len(tcn_channels))]
        else:
            if len(tcn_dilation_factors) != len(tcn_channels):
                raise ValueError("Length of tcn_dilation_factors must match tcn_channels")

        # Build TCN layers: Convolution -> ReLU -> Dropout
        for i, out_channels in enumerate(tcn_channels):
            # Causal convolution (padding ensures output length matches input length)
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) * tcn_dilation_factors[i],
                    dilation=tcn_dilation_factors[i],
                )
            )
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        # The output layer maps the TCN's final feature representation to the
        # desired correction dimensions (e.g., 3D position correction).
        self.output_layer = nn.Linear(tcn_channels[-1], output_size)

        # Calculate receptive field for informational purposes.
        self.receptive_field = 1
        for dilation in tcn_dilation_factors:
            self.receptive_field += (kernel_size - 1) * dilation


    def forward(
        self,
        imu_sequence_raw: torch.Tensor,
        imu_sequence_norm: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the OnlyTCN model to predict a corrected trajectory.

        Args:
            imu_sequence_raw (torch.Tensor): Raw IMU data (unused).
                - Shape: (Batch, Seq_Len, Features)
            imu_sequence_norm (torch.Tensor): Normalized IMU data.
                - Shape: (Batch, Seq_Len, Features)
                - Unit: Normalized
                - Frame: Body Frame

        Returns:
            torch.Tensor: The predicted 3D position trajectory.
                - Shape: (Batch, Seq_Len, 3)
                - Unit: Meter
                - Frame: World
        """
        # Use TCN on NORMALIZED data to predict the position directly.
        # Transpose for Conv1d: [batch, sequence_length, features] -> [batch, features, sequence_length]
        tcn_input = imu_sequence_norm.transpose(1, 2)

        # Pass through TCN layers.
        for layer in self.tcn_layers:
            tcn_input = layer(tcn_input)

        # Transpose back and crop to match input sequence length
        tcn_output = tcn_input.transpose(1, 2)[:, : imu_sequence_norm.shape[1], :]
        predicted_trajectory = self.output_layer(tcn_output)

        return predicted_trajectory


if __name__ == "__main__":
    # Example usage and testing of the OnlyTCN model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model parameters from Config.
    input_size = Config.OnlyTCN.INPUT_SIZE
    output_size = Config.OnlyTCN.OUTPUT_SIZE
    sequence_length = 100
    batch_size = 32
    dt_val = Config.DT

    # Create a model instance.
    model = OnlyTCN(
        device=device,
        input_size=input_size,
        output_size=output_size,
        dt=dt_val,
    ).to(device)

    # Create some dummy data. `imu_sequence_raw` and `imu_sequence_norm`
    # can be the same for this basic test, but typically `imu_sequence_norm`
    # would be scaled/normalized version of `imu_sequence_raw`.
    dummy_imu_data_raw = torch.randn(batch_size, sequence_length, input_size).to(
        device
    )
    # Simple normalization for demonstration.
    dummy_imu_data_norm = (dummy_imu_data_raw - dummy_imu_data_raw.mean(dim=(0, 1)))
    (
        dummy_imu_data_raw.std(dim=(0, 1)) + 1e-6
    )

    # Perform a forward pass.
    predicted_trajectory = model(dummy_imu_data_raw, dummy_imu_data_norm)

    print(f"\nInput raw IMU sequence shape: {dummy_imu_data_raw.shape}")
    print(f"Input normalized IMU sequence shape: {dummy_imu_data_norm.shape}")
    print(f"Output corrected trajectory shape: {predicted_trajectory.shape}")
    print(f"Model receptive field: {model.receptive_field} steps")

    # Assertions to ensure the output shape is as expected.
    assert predicted_trajectory.shape == (batch_size, sequence_length, output_size)

    print("\nOnlyTCN model created and tested successfully.")
