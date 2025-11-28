
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import argparse
from tqdm import tqdm

from model.ESKF_TCN import ESKFTCN_model
from model.AEKF_TCN import AEKFTCN_model
from model.rotation_utils import quaternion_to_rotation_matrix

class HybridTrajectoryLoss(nn.Module):
    def __init__(self, dt=0.01):
        """
        dt: Sampling time (e.g., 100Hz -> 0.01s)
        """
        super().__init__()
        self.dt = dt
        self.criterion = nn.SmoothL1Loss()

    def forward(self,
                pred_vel_resid_b,
                filter_vel_w,
                filter_quat,
                gt_vel_w,
                gt_pos_w,
                start_pos_w,
                alpha,
                beta):

        # 1. Velocity Loss (World Frame)
        rot_mat_b_to_w = quaternion_to_rotation_matrix(filter_quat)
        pred_vel_resid_w = (rot_mat_b_to_w @ pred_vel_resid_b.unsqueeze(-1)).squeeze(-1)

        pred_final_vel_w = filter_vel_w + pred_vel_resid_w
        loss_vel = self.criterion(pred_final_vel_w, gt_vel_w)

        # 2. Position Loss (World Frame)
        vel_avg = (pred_final_vel_w[:, :-1] + pred_final_vel_w[:, 1:]) / 2.0
        vel_first = pred_final_vel_w[:, 0:1]
        vel_integrand = torch.cat([vel_first, vel_avg], dim=1)

        pred_traj_w = torch.cumsum(vel_integrand * self.dt, dim=1) + start_pos_w
        loss_pos = self.criterion(pred_traj_w, gt_pos_w)

        # 3. Total Loss
        total_loss = (alpha * loss_vel) + (beta * loss_pos)
        return total_loss, loss_vel.item(), loss_pos.item()

class TrajectoryDataset(Dataset):
    def __init__(self, sensor_file, truth_file, sequence_length=300, dt=0.01):
        self.sensor_file = sensor_file
        self.truth_file = truth_file
        self.sequence_length = sequence_length
        self.dt = dt

        with h5py.File(self.sensor_file, 'r') as f:
            self.num_samples = len(f.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_key = f'Samples_{idx}'

        with h5py.File(self.sensor_file, 'r') as f:
            imu_data = np.stack([
                f[f'{sample_key}/Ax'][:], f[f'{sample_key}/Ay'][:], f[f'{sample_key}/Az'][:],
                f[f'{sample_key}/Gx'][:], f[f'{sample_key}/Gy'][:], f[f'{sample_key}/Gz'][:],
                f[f'{sample_key}/Force'][:]
            ], axis=1)

        with h5py.File(self.truth_file, 'r') as f:
            gt_pos_w = np.stack([
                f[f'{sample_key}/x'][:], f[f'{sample_key}/y'][:], f[f'{sample_key}/z'][:]
            ], axis=1)

        min_len = min(len(imu_data), len(gt_pos_w))
        imu_data = imu_data[:min_len]
        gt_pos_w = gt_pos_w[:min_len]

        # Calculate ground truth velocity in world frame
        gt_vel_w = np.gradient(gt_pos_w, self.dt, axis=0)

        start_pos_w = gt_pos_w[0:1, :]

        return {
            "imu": torch.from_numpy(imu_data).float(),
            "gt_pos_w": torch.from_numpy(gt_pos_w).float(),
            "gt_vel_w": torch.from_numpy(gt_vel_w).float(),
            "start_pos_w": torch.from_numpy(start_pos_w).float(),
        }

def train(model, dataloader, epochs, lr, device, model_path, warmup_epochs, alpha, beta_final):
    criterion = HybridTrajectoryLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        # Linear ramp-up for beta
        if epoch < warmup_epochs:
            beta = beta_final * (epoch + 1) / warmup_epochs
        else:
            beta = beta_final

        epoch_loss, epoch_loss_vel, epoch_loss_pos = 0.0, 0.0, 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (beta: {beta:.2f})")

        for data_batch in pbar:
            imu_data = data_batch["imu"].to(device)
            gt_pos_w = data_batch["gt_pos_w"].to(device)
            gt_vel_w = data_batch["gt_vel_w"].to(device)
            start_pos_w = data_batch["start_pos_w"].to(device)

            optimizer.zero_grad()

            model_out = model(imu_data)

            loss, loss_vel, loss_pos = criterion(
                pred_vel_resid_b=model_out["pred_vel_resid_b"],
                filter_vel_w=model_out["filter_vel_w"],
                filter_quat=model_out["filter_quat"],
                gt_vel_w=gt_vel_w,
                gt_pos_w=gt_pos_w,
                start_pos_w=start_pos_w,
                alpha=alpha,
                beta=beta
            )

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional: gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_vel += loss_vel
            epoch_loss_pos += loss_pos
            pbar.set_postfix({'Loss': loss.item(), 'Vel_Loss': loss_vel, 'Pos_Loss': loss_pos})

        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_loss_vel = epoch_loss_vel / len(dataloader)
        avg_loss_pos = epoch_loss_pos / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, Vel: {avg_loss_vel:.4f}, Pos: {avg_loss_pos:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a hybrid Kalman Filter-TCN model.")
    parser.add_argument('--model', type=str, choices=['eskf', 'aekf'], required=True, help="Model to train: 'eskf' or 'aekf'.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to train on ('cpu', 'cuda', 'mps').")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warm-up epochs for the position loss.")
    parser.add_argument('--alpha', type=float, default=1.0, help="Weight for the velocity loss.")
    parser.add_argument('--beta', type=float, default=10.0, help="Final weight for the position loss.")

    args = parser.parse_args()

    sensor_file = 'data/Sensor_Board_Data_1.h5'
    truth_file = 'data/Ground_Truth_Data_1.h5'

    dataset = TrajectoryDataset(sensor_file, truth_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == 'eskf':
        model = ESKFTCN_model(device=args.device)
        model_path = 'eskf_tcn_model.pth'
    elif args.model == 'aekf':
        model = AEKFTCN_model(device=args.device)
        model_path = 'aekf_tcn_model.pth'
    else:
        raise ValueError("Invalid model choice. Choose 'eskf' or 'aekf'.")

    print(f"Training {args.model.upper()}-TCN model for {args.epochs} epochs...")
    train(model, dataloader, args.epochs, args.lr, args.device, model_path, args.warmup_epochs, args.alpha, args.beta)

if __name__ == '__main__':
    main()
