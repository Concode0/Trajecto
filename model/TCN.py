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
import torch.nn.functional as F

from model.config import Config


class CausalConv1d(nn.Module):
    """A causal 1D convolution layer (supporting Depthwise Separable).

    It pads the input on the left (past) side to ensure that the output
    at time t depends only on inputs up to time t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        separable: bool = False, # Default to False, controlled by higher-level models
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.separable = separable

        if separable and in_channels > 1:
            # Depthwise Convolution: groups = in_channels
            self.depthwise = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
                groups=in_channels,
            )
            # Pointwise Convolution: kernel_size = 1
            self.pointwise = nn.Conv1d(
                in_channels,
                out_channels,
                1,
                padding=0,
                dilation=1, # Pointwise is typically not dilated
            )
            self.conv = None # Placeholder to indicate separable mode
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,  # We handle padding manually
                dilation=dilation,
            )
            self.depthwise = None
            self.pointwise = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left side
        x = F.pad(x, (self.padding, 0))

        if self.separable and self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)

        x = self.relu(x)
        x = self.dropout(x)
        return x


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
        separable: bool = False, # Default to False, controlled by higher-level models
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
            separable: If True, uses Depthwise Separable Convolutions to reduce parameters.
        """
        super().__init__()

        # Default dilation factors if not explicitly provided.
        if tcn_dilation_factors is None:
            tcn_dilation_factors = [2**i for i in range(len(tcn_channels))]
        elif len(tcn_dilation_factors) != len(tcn_channels):
            raise ValueError(
                "Length of tcn_dilation_factors must match length of tcn_channels."
            )

        # Batch Normalization applied to the input features across the time dimension.
        # This replaces the LayerNorm that was previously in the wrapper class.
        # BatchNorm is more efficient for inference (can be fused) and treats features independently.
        self.input_bn = nn.GroupNorm(num_groups=19, num_channels=input_size, affine=True)

        self.tcn_layers = nn.ModuleList()  # Stores the sequential TCN blocks.
        in_channels = input_size
        self._receptive_field = 1  # Tracks the effective receptive field of the network.

        # Construct the TCN layers using CausalConv1d blocks.
        for i, out_channels in enumerate(tcn_channels):
            dilation = tcn_dilation_factors[i]

            self.tcn_layers.append(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    separable=separable,
                )
            )
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
                ),  # Parameters for 6D log(R) (e.g., log-diagonal)
                "zupt_prob": nn.Linear(
                    tcn_channels[-1], 1
                ),  # Scalar ZUPT probability
            }
        )

        # Register isotropic velocity correction scale as buffer
        # Using L2 norm maintains directional accuracy across all axes
        # Physical Z-axis constraints are handled by data distribution, not output scaling
        self.register_buffer(
            "vel_scale_isotropic",
            torch.tensor(Config.VEL_CORRECTION_SCALE, dtype=torch.float32)
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
            feature_sequence (torch.Tensor): Input feature sequence.
                - Shape: (Batch, Seq_Len, Input_Size)
                - Unit: Normalized/Feature-Specific
                - Frame: N/A (Abstract features)

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output tensors:
                - "vel_corr": Velocity correction.
                    - Shape: (Batch, Seq_Len, 3) | Unit: m/s | Frame: Body
                - "covariance_R": Covariance parameters.
                    - Shape: (Batch, Seq_Len, 6) | Unit: Log-Variance or similar
                - "zupt_prob": Zero-velocity probability.
                    - Shape: (Batch, Seq_Len, 1) | Range: [0, 1]
        """
        # Conv1d layers in PyTorch expect input of shape `[B, D, T]`.
        # Therefore, we transpose the last two dimensions of the input `feature_sequence`
        # from `[B, T, D]` to `[B, D, T]`.
        tcn_input = feature_sequence.transpose(1, 2)

        # Apply GroupNorm to stabilize training and normalize input feature scales
        # Different features have different scales: IMU (~1.0), velocity (~0.1), innovation (varies)
        # GroupNorm helps TCN layers learn efficiently by normalizing across feature groups
        tcn_input = self.input_bn(tcn_input)

        # Pass the input through all TCN layers.
        # Since we use CausalConv1d, the output sequence length is naturally preserved
        # and causality is maintained.
        for layer in self.tcn_layers:
            tcn_input = layer(tcn_input)

        # After convolutions, transpose back to `[B, T, D']` for the linear output heads.
        tcn_output = tcn_input.transpose(1, 2)

        # Apply each output head (linear layer) to the TCN's final feature map.
        outputs = {
            head_name: head(tcn_output)
            for head_name, head in self.output_heads.items()
        }

        # Bounded Velocity Correction: Isotropic denormalization to physical units
        # Tanh naturally maps unbounded linear output → [-1, 1]
        # Scale by isotropic L2 norm (2σ of velocity magnitude distribution)
        # Isotropic scaling is critical because:
        #   (1) Preserves directional accuracy: TCN learns true velocity vectors
        #   (2) Physical Z-axis constraint is naturally learned from data distribution
        #   (3) Avoids asymmetric regularization (all axes penalized equally)
        #   (4) Simpler architecture: network doesn't need to learn axis-specific scaling
        # Trade-off: Z-axis may saturate tanh more often, but preserves geometric correctness
        # Use cached buffer (registered in __init__) - avoids tensor creation
        outputs["vel_corr"] = torch.tanh(outputs["vel_corr"]) * self.vel_scale_isotropic

        # CRITICAL: Clip covariance output to prevent numerical explosion
        # Raw TCN output represents log-space values that will be transformed by:
        #   variance = softplus(output) + 1e-4, then clamped to [1e-4, 3.0]
        # To fully utilize the valid range without wasting network capacity:
        #   - min: softplus(-10) + 1e-4 ≈ 1.45e-4 (slightly above floor)
        #   - max: softplus(2.95) + 1e-4 ≈ 3.0 (matches ESKF/train.py upper bound)
        # This ensures TCN output directly maps to usable variance range [1e-4, 3.0]
        if "covariance_R" in outputs:
            outputs["covariance_R"] = torch.clamp(
                outputs["covariance_R"],
                min=-10.0,
                max=2.95
            )

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