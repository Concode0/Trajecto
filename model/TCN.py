"""
This module defines a Temporal Convolutional Network (TCN) model designed
for multi-head output in hybrid filter architectures.

The TCN processes a sequence of input features and, at each time step,
predicts multiple quantities such as velocity corrections, elements of
a measurement noise covariance matrix, and the probability of a
Zero-Velocity Update (ZUPT) event. This allows the TCN to provide adaptive
parameters and corrections to a Kalman filter in a closed-loop fashion.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TCN(nn.Module):
    """A multi-head Temporal Convolutional Network (TCN) for closed-loop trajectory correction.

    This network takes a sequence of rich features (e.g., filter innovations,
    velocity estimates, IMU readings) and processes them through multiple
    dilated causal convolutional layers. It produces three distinct outputs
    at each time step, designed to inform and correct a Kalman filter:
    1. `vel_corr`: A 3D velocity correction vector.
    2. `covariance_R`: A 6D vector representing parameters for the measurement
       noise covariance matrix `R` (e.g., diagonal elements or Cholesky factors).
    3. `zupt_prob`: A scalar probability indicating if the sensor is stationary
       (Zero-Velocity Update).
    """

    def __init__(
        self,
        input_size: int = 20,
        tcn_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        tcn_dilation_factors: Optional[List[int]] = None,
    ):
        """Initializes the TCN model.

        Args:
            input_size: The number of features in the input sequence. This `D` corresponds
                to the last dimension in a `[B, T, D]` input tensor.
            tcn_channels: A list where each element specifies the number of output
                channels (filters) for a corresponding TCN layer. The length of
                this list determines the number of TCN layers.
            kernel_size: The size of the convolutional kernel to be used in all TCN layers.
            dropout: The dropout rate applied after each ReLU activation for regularization.
            tcn_dilation_factors: Optional list of dilation factors for each TCN layer.
                If None, defaults to `[2**i for i in range(len(tcn_channels))]`,
                which is a common strategy for increasing the receptive field exponentially.
        """
        super().__init__()

        # Default dilation factors if not explicitly provided.
        if tcn_dilation_factors is None:
            tcn_dilation_factors = [2**i for i in range(len(tcn_channels))]
        elif len(tcn_dilation_factors) != len(tcn_channels):
            raise ValueError(
                "Length of tcn_dilation_factors must match length of tcn_channels."
            )

        self.tcn_layers = nn.ModuleList()  # Stores the sequential TCN blocks.
        in_channels = input_size
        self._receptive_field = 1  # Tracks the effective receptive field of the network.

        # Construct the TCN layers. Each layer consists of a dilated convolution,
        # followed by ReLU activation and Dropout.
        for i, out_channels in enumerate(tcn_channels):
            dilation = tcn_dilation_factors[i]
            # Padding is calculated to ensure the output sequence length after
            # convolution (and before cropping) is sufficient to match the input.
            # For a standard Conv1d with symmetric padding, (kernel_size - 1) * dilation
            # ensures that for a 'valid' part of the convolution over the receptive field,
            # the output length can be matched to the input length.
            padding = (kernel_size - 1) * dilation
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

            # The receptive field grows with each dilated convolutional layer.
            # For a TCN layer with kernel_size k and dilation d, the receptive
            # field increases by (k-1)*d.
            self._receptive_field += (kernel_size - 1) * dilation

        # Define multiple output heads, each consisting of a linear layer,
        # to predict different quantities from the TCN's final feature map.
        self.output_heads = nn.ModuleDict(
            {
                "vel_corr": nn.Linear(tcn_channels[-1], 3),  # 3D velocity correction
                "covariance_R": nn.Linear(
                    tcn_channels[-1], 6
                ),  # Parameters for 6D R matrix (e.g., diagonal)
                "zupt_prob": nn.Linear(
                    tcn_channels[-1], 1
                ),  # Scalar ZUPT probability
            }
        )

    @property
    def receptive_field(self) -> int:
        """Returns the receptive field of the TCN.

        The receptive field is the number of input time steps that influence
        a single output time step. It's an important characteristic of TCNs,
        determining how much past information the network can leverage.
        """
        return self._receptive_field

    def forward(self, feature_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Processes the sequence of input features to produce multi-head outputs.

        Args:
            feature_sequence: An input tensor of shape `[B, T, D]`, where B is
                batch size, T is sequence length, and D is the number of features.

        Returns:
            A dictionary of output tensors, where each key corresponds to an
            output head:
                - 'vel_corr': `[B, T, 3]`
                - 'covariance_R': `[B, T, 6]`
                - 'zupt_prob': `[B, T, 1]`
        """
        # Conv1d layers in PyTorch expect input of shape `[B, D, T]`.
        # Therefore, we transpose the last two dimensions of the input `feature_sequence`
        # from `[B, T, D]` to `[B, D, T]`.
        tcn_input = feature_sequence.transpose(1, 2)

        # Pass the input through all TCN layers.
        for layer in self.tcn_layers:
            tcn_input = layer(tcn_input)

        # After convolutions, transpose back to `[B, T, D']` for the linear output heads.
        # The padding strategy used in `nn.Conv1d` with `padding=(kernel_size - 1) * dilation`
        # results in an output sequence that is typically longer than the input.
        # To ensure the output `tcn_output` matches the original `feature_sequence` length,
        # we crop the output to `feature_sequence.shape[1]`. This effectively mimics
        # 'same' padding for the valid convolution region.
        tcn_output = tcn_input.transpose(1, 2)[:, : feature_sequence.shape[1], :]

        # Apply each output head (linear layer) to the TCN's final feature map.
        outputs = {
            head_name: head(tcn_output)
            for head_name, head in self.output_heads.items()
        }

        return outputs


if __name__ == "__main__":
    # Simple test case to verify functionality and shapes of the TCN model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model parameters for testing.
    input_features = 20
    seq_length = 100
    batch_size = 32
    tcn_channels_test = [64, 64, 64, 64]
    kernel_size_test = 3

    # Create a model instance.
    model = TCN(
        input_size=input_features,
        tcn_channels=tcn_channels_test,
        kernel_size=kernel_size_test,
    ).to(device)
    print(f"\nTCN Model Receptive Field: {model.receptive_field} steps")

    # Create some dummy feature data.
    dummy_feature_data = torch.randn(
        batch_size, seq_length, input_features, device=device
    )

    # Perform a forward pass.
    predictions = model(dummy_feature_data)

    print(f"\nInput feature sequence shape: {dummy_feature_data.shape}")
    print("Output shapes for each head:")
    for head, pred in predictions.items():
        print(f"  - '{head}': {pred.shape}")
        # Assert that output sequence length matches input sequence length.
        assert pred.shape[1] == seq_length, f"Output length mismatch for {head}"

    # Test the receptive field with an input sequence exactly its size.
    # The output from the TCN should be valid across this length.
    dummy_rf_feature = torch.randn(
        batch_size, model.receptive_field, input_features, device=device
    )
    predictions_rf = model(dummy_rf_feature)
    print(
        f"\nInput feature sequence shape (Receptive Field test): {dummy_rf_feature.shape}"
    )
    print("Output shapes for each head (Receptive Field test):")
    for head, pred in predictions_rf.items():
        print(f"  - '{head}': {pred.shape}")
        assert pred.shape[1] == model.receptive_field, f"RF output length mismatch for {head}"

    print("\nTCN multi-head model created and tested successfully.")