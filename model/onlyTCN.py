
import torch
import torch.nn as nn

class TCN(nn.Module):
    """
    Temporal Convolutional Network for residual error correction of a trajectory.
    Input: IMU data (accelerometer, gyroscope, force)
    Output: Corrected trajectory (3D position)
    """
    def __init__(self, input_size=7, output_size=3, tcn_channels=[64, 64, 64, 64], kernel_size=3, dropout=0.1, dt=0.01):
        super(TCN, self).__init__()

        self.dt = dt
        self.tcn_layers = nn.ModuleList()
        in_channels = input_size

        for out_channels in tcn_channels:
            self.tcn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=(kernel_size-1)//2, dilation=1)
            )
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.output_layer = nn.Linear(tcn_channels[-1], output_size)

    def forward(self, imu_sequence):
        """
        Forward pass for the TCN model with residual correction.

        Args:
            imu_sequence: [batch_size, sequence_length, input_size] 
                          (assuming accel is the first 3 features)

        Returns:
            corrected_trajectory: [batch_size, sequence_length, output_size]
        """
        # 1. Compute a naive base trajectory by double-integrating acceleration
        accel_data = imu_sequence[:, :, :3]
        
        # First integration (acceleration to velocity)
        velocity = torch.cumsum(accel_data * self.dt, dim=1)
        
        # Second integration (velocity to position)
        # We need to add the initial velocity contribution to each step
        base_trajectory = torch.cumsum(velocity * self.dt, dim=1)

        # 2. Use TCN to predict the correction
        # Transpose for conv1d: [batch, features, sequence]
        tcn_input = imu_sequence.transpose(1, 2)

        for layer in self.tcn_layers:
            tcn_input = layer(tcn_input)

        # Transpose back and get position corrections
        tcn_output = tcn_input.transpose(1, 2)  # [batch, seq, channels]
        correction = self.output_layer(tcn_output)

        # 3. Add the correction to the base trajectory
        corrected_trajectory = base_trajectory + correction

        return corrected_trajectory

if __name__ == '__main__':
    print("Running tests for onlyTCN.py...")
    # --- Test Parameters ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    input_size = 7  # accel(3), gyro(3), force(1)
    output_size = 3 # position(3)
    sequence_length = 100
    batch_size = 32

    print(f"Using device: {device}")

    # --- Test 1: Model Instantiation and Forward Pass ---
    try:
        model = TCN(input_size=input_size, output_size=output_size).to(device)
        model.eval() # Ensure model can be switched to evaluation mode
        print("Test 1 (Instantiation): PASSED")
    except Exception as e:
        print(f"Test 1 (Instantiation): FAILED - {e}")
        exit()

    # --- Test 2: Shape Verification ---
    # Create dummy data
    dummy_imu_data = torch.randn(batch_size, sequence_length, input_size).to(device)
    
    # Forward pass
    try:
        predicted_trajectory = model(dummy_imu_data)
        
        # Verify output shape
        expected_shape = (batch_size, sequence_length, output_size)
        assert predicted_trajectory.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {predicted_trajectory.shape}"
        
        print(f"Input shape: {dummy_imu_data.shape}")
        print(f"Output shape: {predicted_trajectory.shape}")
        print("Test 2 (Shape Verification): PASSED")
    except Exception as e:
        print(f"Test 2 (Shape Verification): FAILED - {e}")
        exit()

    print("\nAll onlyTCN tests passed successfully.")
