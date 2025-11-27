"""
A Temporal Convolutional Network (TCN) for sequence modeling.
"""

import torch
import torch.nn as nn
from typing import List

class TCN(nn.Module):
    """A simplified TCN for learning residual trajectory corrections.

    This network takes a sequence of rich features derived from the IMU data and
    the Kalman filter's state, and outputs a sequence of 3D position corrections.
    The goal is to learn the systematic errors of the physics-based filter.
    """
    def __init__(self, 
                 input_size: int = 17, 
                 output_size: int = 3, 
                 tcn_channels: List[int] = [64, 64, 64, 64], 
                 kernel_size: int = 3, 
                 dropout: float = 0.1):
        """Initializes the TCN model.

        Args:
            input_size (int): The number of features in the input sequence.
            output_size (int): The number of output values per time step (e.g., 3 for 3D position correction).
            tcn_channels (List[int]): A list where each element is the number of output channels for a TCN layer.
            kernel_size (int): The size of the convolutional kernel.
            dropout (float): The dropout rate for regularization.
        """
        super(TCN, self).__init__()

        self.tcn_layers = nn.ModuleList()
        in_channels = input_size

        # The TCN is built as a stack of 1D convolutional layers.
        # Each layer is followed by a non-linearity (ReLU) and dropout for regularization.
        # Padding is added to maintain the sequence length through the convolutions.
        for out_channels in tcn_channels:
            self.tcn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=(kernel_size-1)//2, dilation=1)
            )
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        # A final linear layer maps the learned features to the desired output dimension (3D correction).
        self.output_layer = nn.Linear(tcn_channels[-1], output_size)

    def forward(self, feature_sequence: torch.Tensor) -> torch.Tensor:
        """Processes the sequence of input features to produce a sequence of corrections.

        Args:
            feature_sequence (torch.Tensor): An input tensor of shape `[B, T, D]`,
                where B is batch size, T is sequence length, and D is the number of features.

        Returns:
            torch.Tensor: The output tensor of position corrections, with shape `[B, T, output_size]`.
        """
        # Conv1d expects input of shape `[B, D, T]`, so we transpose the last two dimensions.
        tcn_input = feature_sequence.transpose(1, 2)

        for layer in self.tcn_layers:
            tcn_input = layer(tcn_input)

        # Transpose back to `[B, T, D]` for the final linear layer.
        tcn_output = tcn_input.transpose(1, 2)
        position_correction = self.output_layer(tcn_output)

        return position_correction

if __name__ == '__main__':
    # Simple test case to verify functionality and shapes.
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model parameters
    input_features = 17
    output_dim = 3
    seq_length = 100
    batch_size = 32

    # Create a model instance
    model = TCN(input_size=input_features, output_size=output_dim).to(device)

    # Create some dummy data
    dummy_feature_data = torch.randn(batch_size, seq_length, input_features).to(device)

    # Forward pass
    prediction = model(dummy_feature_data)

    print(f"Input shape: {dummy_feature_data.shape}")
    print(f"Output shape: {prediction.shape}")
    print("TCN model created and tested successfully.")