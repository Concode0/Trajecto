
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import argparse
from tqdm import tqdm

from model.ESKF_TCN import ESKFTCN_model
from model.AEKF_TCN import AEKFTCN_model

class TrajectoryDataset(Dataset):
    """
    Dataset for loading trajectory data from HDF5 files.
    Each item in the dataset corresponds to a sample trajectory.
    """
    def __init__(self, sensor_file, truth_file, sequence_length=300):
        self.sensor_file = sensor_file
        self.truth_file = truth_file
        self.sequence_length = sequence_length

        with h5py.File(self.sensor_file, 'r') as f:
            # Assuming the number of samples can be inferred from the top-level groups
            self.num_samples = len(f.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_key = f'Samples_{idx}'
        
        with h5py.File(self.sensor_file, 'r') as f:
            accel_x = f[f'{sample_key}/Ax'][:]
            accel_y = f[f'{sample_key}/Ay'][:]
            accel_z = f[f'{sample_key}/Az'][:]
            gyro_x = f[f'{sample_key}/Gx'][:]
            gyro_y = f[f'{sample_key}/Gy'][:]
            gyro_z = f[f'{sample_key}/Gz'][:]
            force = f[f'{sample_key}/Force'][:]
            
            # Stack the sensor data
            imu_data = np.stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, force], axis=1)

        with h5py.File(self.truth_file, 'r') as f:
            x = f[f'{sample_key}/x'][:]
            y = f[f'{sample_key}/y'][:]
            z = f[f'{sample_key}/z'][:]
            
            # Stack the ground truth data
            truth_data = np.stack([x, y, z], axis=1)

        # TODO: Implement Dynamic Time Warping (DTW) for time alignment
        # As a placeholder, we assume the data is already aligned.
        # If lengths are different, truncate to the shorter length.
        min_len = min(len(imu_data), len(truth_data))
        imu_data = imu_data[:min_len]
        truth_data = truth_data[:min_len]

        # Convert to tensors
        imu_tensor = torch.from_numpy(imu_data).float()
        truth_tensor = torch.from_numpy(truth_data).float()

        return imu_tensor, truth_tensor

def train(model, dataloader, epochs, lr, device, model_path):
    """
    Training loop for the hybrid model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Using tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imu_data, truth_data in pbar:
            imu_data, truth_data = imu_data.to(device), truth_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_trajectory = model(imu_data)
            
            loss = criterion(predicted_trajectory, truth_data)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a hybrid Kalman Filter-TCN model.")
    parser.add_argument('--model', type=str, choices=['eskf', 'aekf'], required=True,
                        help="Model to train: 'eskf' or 'aekf'.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to train on ('cpu', 'cuda', 'mps').")
    
    args = parser.parse_args()

    # File paths
    sensor_file = 'data/Sensor_Board_Data_1.h5'
    truth_file = 'data/Groud_Truth_Data_1.h5' # Corrected typo from Groud to Ground

    # Dataset and DataLoader
    dataset = TrajectoryDataset(sensor_file, truth_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model selection
    if args.model == 'eskf':
        model = ESKFTCN_model(device=args.device)
        model_path = 'eskf_tcn_model.pth'
    elif args.model == 'aekf':
        model = AEKFTCN_model(device=args.device)
        model_path = 'aekf_tcn_model.pth'
    else:
        raise ValueError("Invalid model choice. Choose 'eskf' or 'aekf'.")

    print(f"Training {args.model.upper()}-TCN model for {args.epochs} epochs...")
    
    train(model, dataloader, args.epochs, args.lr, args.device, model_path)

if __name__ == '__main__':
    main()
