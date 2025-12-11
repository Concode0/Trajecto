"""
This script serves as the main training pipeline for various trajectory estimation
models, including hybrid Kalman filter-Temporal Convolutional Network (TCN) architectures
(ESKF-TCN, AEKF-TCN) and a standalone TCN model.

It orchestrates the data loading, model initialization, loss computation,
optimization, and evaluation. The script incorporates features like GPU-accelerated
data augmentation, adaptive learning rate scheduling, and comprehensive logging
of training history, ultimately saving the trained model and loss plots.
"""

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to sys.path for relative imports to models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.AEKF_TCN import AEKFTCN_model
from model.ESKF_TCN import ESKFTCN_model
from model.onlyTCN import OnlyTCN  # Renamed from TCN to OnlyTCN
from model.rotation_utils import quaternion_to_rotation_matrix
from model.dataset import TrajectoryDataset


class PositionLoss(nn.Module):
    """Calculates the Mean Squared Error (MSE) loss for position predictions."""

    def __init__(self):
        """Initializes the PositionLoss module."""
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(
        self, model_output: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the position loss.

        Args:
            model_output: The predicted position tensor from the model.
            batch: A dictionary containing ground truth data, including 'gt_pos_w'.

        Returns:
            A tuple containing:
                - loss (torch.Tensor): The computed MSE loss for position.
                - losses (Dict[str, float]): A dictionary containing the loss item.
        """
        loss = self.criterion(model_output, batch["gt_pos_w"])
        return loss, {"pos": loss.item()}


class HybridLoss(nn.Module):
    """Calculates a hybrid loss combining position, velocity, ZUPT, and regularization terms."""

    def __init__(self):
        """Initializes the HybridLoss module."""
        super().__init__()
        self.pos_criterion = nn.SmoothL1Loss()  # Huber Loss for position.
        self.vel_criterion = nn.SmoothL1Loss()  # Huber Loss for velocity.
        self.zupt_criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for ZUPT probability.

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        pos_weight: float,
        vel_weight: float,
        zupt_weight: float,
        reg_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the hybrid loss.

        Args:
            model_out: A dictionary containing model predictions, including 'pred_pos_w',
                'filter_vel_w', 'pred_vel_resid_b', 'filter_quat', 'pred_zupt_prob'.
            batch: A dictionary containing ground truth data, including 'gt_pos_w',
                'gt_vel_w', 'gt_zupt'.
            pos_weight: Weight for the position loss component.
            vel_weight: Weight for the velocity loss component.
            zupt_weight: Weight for the ZUPT probability loss component.
            reg_weight: Weight for the regularization loss component.

        Returns:
            A tuple containing:
                - total_loss (torch.Tensor): The weighted sum of all loss components.
                - losses (Dict[str, float]): A dictionary containing itemized loss values.
        """
        # Position Loss: Compare predicted world position with ground truth.
        pred_pos_w = model_out["pred_pos_w"]
        loss_pos = self.pos_criterion(pred_pos_w, batch["gt_pos_w"])

        # Velocity Loss:
        # The filter outputs `filter_vel_w` (world frame velocity from filter)
        # and `pred_vel_resid_b` (body frame residual velocity from TCN).
        # We need to rotate the residual velocity from body to world frame and
        # add it to the filter's velocity to get the total predicted velocity.
        filter_vel_w = model_out["filter_vel_w"]
        pred_vel_resid_b = model_out["pred_vel_resid_b"]
        filter_quat = model_out["filter_quat"]

        # Reshape quaternion for batch matrix multiplication, then convert to rotation matrix.
        rot_mat_b_to_w = quaternion_to_rotation_matrix(filter_quat.view(-1, 4)).view(
            *filter_quat.shape[:-1], 3, 3
        )
        # Rotate residual velocity from body frame to world frame.
        pred_vel_resid_w = (rot_mat_b_to_w @ pred_vel_resid_b.unsqueeze(-1)).squeeze(-1)
        # Total predicted velocity in world frame.
        pred_total_vel_w = filter_vel_w + pred_vel_resid_w
        loss_vel = self.vel_criterion(pred_total_vel_w, batch["gt_vel_w"])

        # ZUPT Loss: Binary cross-entropy for ZUPT probability prediction.
        # `pred_zupt_prob` is typically logits, so `BCEWithLogitsLoss` is appropriate.
        loss_zupt = self.zupt_criterion(model_out["pred_zupt_prob"], batch["gt_zupt"])

        # Regularization Loss: L2 norm of the predicted velocity residual.
        # This encourages the TCN to predict small corrections unless necessary.
        loss_reg = torch.norm(pred_vel_resid_b)

        # Weighted sum of all loss components.
        total_loss = (
            (pos_weight * loss_pos)
            + (vel_weight * loss_vel)
            + (zupt_weight * loss_zupt)
            + (reg_weight * loss_reg)
        )
        return total_loss, {
            "pos": loss_pos.item(),
            "vel": loss_vel.item(),
            "zupt": loss_zupt.item(),
            "reg": loss_reg.item(),
        }


class ProbabilisticHybridLoss(nn.Module):
    """Combines HybridLoss with a Negative Log Likelihood (NLL) term for probabilistic modeling."""

    def __init__(self):
        """Initializes the ProbabilisticHybridLoss module."""
        super().__init__()
        self.hybrid_loss = HybridLoss()

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        pos_weight: float,
        vel_weight: float,
        zupt_weight: float,
        reg_weight: float,
        cov_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the probabilistic hybrid loss, including a Negative Log Likelihood term.

        The NLL loss component assumes the innovation (measurement residual) follows
        a Gaussian distribution whose variance is predicted by the TCN. This term
        encourages the TCN to predict a measurement noise covariance `R` that
        accurately reflects the uncertainty in the innovations.

        Args:
            model_out: A dictionary containing model predictions, including 'filter_innovation',
                'pred_covariance_R', and 'tcn_output_mask'.
            batch: A dictionary containing ground truth data.
            pos_weight: Weight for the position loss component.
            vel_weight: Weight for the velocity loss component.
            zupt_weight: Weight for the ZUPT probability loss component.
            reg_weight: Weight for the regularization loss component.
            cov_weight: Weight for the Negative Log Likelihood (NLL) covariance loss.

        Returns:
            A tuple containing:
                - total_loss (torch.Tensor): The weighted sum of all loss components.
                - losses (Dict[str, float]): A dictionary containing itemized loss values.
        """
        hybrid_total_loss, losses = self.hybrid_loss(
            model_out, batch, pos_weight, vel_weight, zupt_weight, reg_weight
        )

        innovation = model_out["filter_innovation"]
        pred_R_diag = model_out["pred_covariance_R"]
        valid_t_mask = model_out["tcn_output_mask"]

        loss_cov = torch.tensor(0.0, device=innovation.device)
        if valid_t_mask.any():  # Only compute NLL if there are valid TCN outputs.
            innovation_valid = innovation[valid_t_mask]
            pred_R_diag_valid = pred_R_diag[valid_t_mask]

            # The TCN typically outputs log-variances or raw values that need transformation.
            # `F.softplus(pred_R_diag_valid) + 1e-4` ensures the predicted variance
            # is always positive (`softplus` for smoothness) and adds a small epsilon
            # (`1e-4`) for numerical stability, preventing division by zero or log(0).
            variance = F.softplus(pred_R_diag_valid) + 1e-4

            # Negative Log Likelihood (NLL) for a Gaussian distribution:
            # NLL = 0.5 * ( (innovation^2 / variance) + log(variance) + log(2*pi) )
            # Here, we omit the constant log(2*pi) as it doesn't affect gradients.
            nll_elementwise = 0.5 * (torch.square(innovation_valid) / variance + torch.log(variance))
            loss_cov = torch.mean(torch.sum(nll_elementwise, dim=-1))

        total_loss = hybrid_total_loss + cov_weight * loss_cov
        losses["cov"] = loss_cov.item() if not torch.isnan(loss_cov) else 0.0
        return total_loss, losses


class GPUAugmentor(nn.Module):
    """Performs data augmentation on the GPU in batches.

    This class applies random transformations (scaling, rotation, noise)
    to sensor and ground truth data directly on the GPU, significantly
    accelerating the augmentation process compared to CPU-based methods.
    """

    def __init__(self, device: str = "cuda"):
        """Initializes the GPUAugmentor.

        Args:
            device: The compute device ('cuda', 'mps', 'cpu') where augmentations will be performed.
        """
        super().__init__()
        self.device = device

    def forward(
        self, sensor: torch.Tensor, gt_pos: torch.Tensor, gt_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies data augmentation to a batch of sensor and ground truth data.

        Args:
            sensor: Batch of sensor data `[B, T, 7]` (Accel_x,y,z, Gyro_x,y,z, FSR).
            gt_pos: Batch of ground truth position data `[B, T, 3]`.
            gt_vel: Batch of ground truth velocity data `[B, T, 3]`.

        Returns:
            A tuple containing the augmented sensor, ground truth position, and
            ground truth velocity tensors.
        """
        batch_size, sequence_length, _ = sensor.shape

        # 1. Random Scale (with 50% probability):
        # Scales accelerometer, position, and velocity data by a random factor (0.8 to 1.2).
        # This helps the model generalize to different scales of motion.
        if random.random() < 0.5:
            # Generate a unique scale factor for each sample in the batch.
            # Shape [B, 1, 1] enables broadcasting across time steps and features.
            scales: torch.Tensor = (
                torch.rand(batch_size, 1, 1, device=self.device) * 0.4
            ) + 0.8  # Range [0.8, 1.2]

            sensor[..., :3] *= scales  # Accelerometer data is scaled.
            gt_pos *= scales
            gt_vel *= scales

        # 2. Random Rotation (around Z-axis, +/- 180 degrees, with 70% probability):
        # Rotates the XY components of accelerometer, gyroscope, position, and velocity.
        # This helps the model become invariant to the starting orientation in the horizontal plane.
        if random.random() < 0.7:
            # Generate a random angle for each sample in the batch.
            angles: torch.Tensor = (
                torch.rand(batch_size, device=self.device) * 2 * np.pi
            ) - np.pi  # Range [-pi, pi]
            c = torch.cos(angles)
            s = torch.sin(angles)

            # Construct 2D rotation matrix for the XY plane for each batch item.
            # R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
            rotation_matrix_2d = torch.zeros(batch_size, 2, 2, device=self.device)
            rotation_matrix_2d[:, 0, 0] = c
            rotation_matrix_2d[:, 0, 1] = -s
            rotation_matrix_2d[:, 1, 0] = s
            rotation_matrix_2d[:, 1, 1] = c

            # Apply rotation: [B, T, 2] @ [B, 2, 2].
            # Using `transpose(1, 2)` on R effectively applies R^T, rotating the data.
            # This handles batch matrix multiplication efficiently.
            sensor[..., :2] = torch.matmul(sensor[..., :2], rotation_matrix_2d.transpose(1, 2))  # Accel XY
            sensor[..., 3:5] = torch.matmul(sensor[..., 3:5], rotation_matrix_2d.transpose(1, 2))  # Gyro XY
            gt_pos[..., :2] = torch.matmul(gt_pos[..., :2], rotation_matrix_2d.transpose(1, 2))  # GT Pos XY
            gt_vel[..., :2] = torch.matmul(gt_vel[..., :2], rotation_matrix_2d.transpose(1, 2))  # GT Vel XY

        # 3. Gaussian Noise (with 30% probability):
        # Adds small Gaussian noise to all sensor readings.
        # This acts as a regularization and makes the model robust to sensor noise.
        if random.random() < 0.3:
            noise = torch.randn_like(sensor, device=self.device) * 0.01
            sensor += noise

        return sensor, gt_pos, gt_vel


def train(
    model_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_path: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    pos_weight: float = 0.0,
    vel_weight: float = 0.0,
    zupt_weight: float = 0.0,
    reg_weight: float = 0.0,
    cov_weight: float = 0.0,
) -> None:
    """Runs the training loop for the specified model.

    Args:
        model_name: The name of the model being trained ('eskf_tcn', 'aekf_tcn', 'only_tcn').
        model: The PyTorch model instance to train.
        dataloader: DataLoader providing batches of training data.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        device: The compute device ('cpu', 'cuda', 'mps').
        model_path: File path to save the trained model's state dictionary.
        mean: Mean values for sensor data normalization.
        std: Standard deviation values for sensor data normalization.
        pos_weight: Weight for the position loss.
        vel_weight: Weight for the velocity loss.
        zupt_weight: Weight for the ZUPT probability loss.
        reg_weight: Weight for the regularization loss.
        cov_weight: Weight for the covariance (NLL) loss.
    """
    # Initialize the appropriate loss criterion based on the model.
    if model_name == "eskf_tcn":
        criterion: nn.Module = ProbabilisticHybridLoss().to(device)
    elif model_name == "aekf_tcn":
        criterion = HybridLoss().to(device)
    elif model_name == "only_tcn":
        criterion = PositionLoss().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Scheduler to reduce learning rate when a metric (total loss) stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    model.to(device)  # Move model to the specified device.
    model.train()  # Set model to training mode (enables dropout, etc.).

    # History dictionary to store average losses per epoch.
    history: Dict[str, List[float]] = {k: [] for k in ["total", "pos", "vel", "zupt", "reg", "cov"]}
    print(f"Start Training for {model_name} on {device} with Augmentation...")

    augmentor = GPUAugmentor(device=device)
    # Move normalization stats to GPU once for efficiency.
    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    for epoch in range(epochs):
        epoch_losses: Dict[str, float] = {k: 0.0 for k in history.keys()}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # Move batch data to the specified device.
            sensor_raw: torch.Tensor = batch["imu_seq_raw"].to(device)
            gt_pos_w: torch.Tensor = batch["gt_pos_w"].to(device)
            gt_vel_w: torch.Tensor = batch["gt_vel_w"].to(device)

            # Apply GPU-accelerated data augmentation if in training mode.
            if model.training:
                sensor_raw, gt_pos_w, gt_vel_w = augmentor(
                    sensor_raw, gt_pos_w, gt_vel_w
                )

            # Recalculate Ground Truth ZUPT flag after augmentation.
            # ZUPT is considered true if the ground truth velocity magnitude is very low
            # (e.g., below 0.01 m/s). This threshold defines "zero velocity".
            gt_vel_norm = torch.norm(gt_vel_w, dim=-1, keepdim=True)
            gt_zupt: torch.Tensor = (gt_vel_norm < 0.01).float()
            # The 'gt_zupt' tensor is binary (0 or 1), suitable for BCEWithLogitsLoss.

            # Normalize Sensor Data: [B, T, 7] - [7].
            # This applies channel-wise normalization using pre-computed mean and std.
            # Adding a small epsilon (1e-6) to std_gpu prevents division by zero.
            sensor_norm: torch.Tensor = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            optimizer.zero_grad()  # Reset gradients from previous step.

            # Forward pass: execute the model.
            # Hybrid models (ESKF-TCN, AEKF-TCN) take both raw and normalized sensor data.
            # OnlyTCN also takes both, but ESKF/AEKF are not part of it.
            model_output: Dict[str, torch.Tensor] = model(sensor_raw, sensor_norm)

            # Re-pack ground truth data for the loss function, now on GPU.
            batch_gpu: Dict[str, torch.Tensor] = {
                "gt_pos_w": gt_pos_w,
                "gt_vel_w": gt_vel_w,
                "gt_zupt": gt_zupt,
            }

            # Calculate loss based on the model type.
            loss: torch.Tensor
            sub_losses: Dict[str, float]
            if model_name == "eskf_tcn":
                loss, sub_losses = criterion(
                    model_output,
                    batch_gpu,
                    pos_weight,
                    vel_weight,
                    zupt_weight,
                    reg_weight,
                    cov_weight,
                )
            elif model_name == "aekf_tcn":
                loss, sub_losses = criterion(
                    model_output,
                    batch_gpu,
                    pos_weight,
                    vel_weight,
                    zupt_weight,
                    reg_weight,
                )
            elif model_name == "only_tcn":
                # For OnlyTCN, `model_output` is directly the corrected position tensor.
                loss, sub_losses = criterion(model_output, batch_gpu)

            if not torch.isnan(loss):  # Check for NaN loss, which can indicate instability.
                loss.backward()  # Compute gradients.
                # Gradient clipping helps prevent exploding gradients, especially in RNNs/TCNs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Update model parameters.

            # Accumulate epoch losses and update progress bar postfix.
            epoch_losses["total"] += loss.item() if not torch.isnan(loss) else 0
            pbar_postfix: Dict[str, float] = {"L": loss.item()}
            for k, v in sub_losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v if not np.isnan(v) else 0
                pbar_postfix[f"L_{k}"] = v
            pbar.set_postfix(pbar_postfix)

        # Average epoch losses.
        avg_loss: float = epoch_losses["total"] / len(dataloader)
        scheduler.step(avg_loss)  # Update learning rate scheduler.

        # Store epoch average losses in history.
        for k in history.keys():
            history[k].append(epoch_losses[k] / len(dataloader))

        # Log epoch summary.
        log_str = f"Epoch {epoch+1}: Total={history['total'][-1]:.4f}"
        if model_name != "only_tcn":
            log_str += f" | Vel={history['vel'][-1]:.4f} | Cov={history['cov'][-1]:.4f}"
        print(log_str)

    # Save the trained model's state dictionary.
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plotting training history.
    plt.figure(figsize=(10, 6))
    for k, v_list in history.items():
        if k != "total" and any(val > 0 for val in v_list):  # Only plot non-zero sub-losses.
            plt.plot(v_list, label=f"{k.capitalize()} Loss")
    if any(val > 0 for val in history["total"]):  # Plot total loss if non-zero.
        plt.plot(history["total"], label="Total Loss", color="black", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/loss_history_{model_name}.png")
    plt.close()


def main() -> None:
    """Main function to parse arguments, set up training, and start the training process."""
    parser = argparse.ArgumentParser(description="Train various trajectory estimation models.")
    parser.add_argument(
        "--model",
        type=str,
        default="eskf_tcn",
        choices=["eskf_tcn", "aekf_tcn", "only_tcn"],
        help="Type of model to train.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cpu', 'cuda', 'mps').",
    )
    # The `dt` argument is now handled by the `TrajectoryDataset`'s internal subsampling
    # logic and directly by the models, not explicitly passed to the dataset constructor.
    # Its value is implicitly defined by the preprocessing step and model's dt.
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time delta for model integration."
    )
    parser.add_argument(
        "--augment", action="store_true", default=True, help="Enable data augmentation."
    )

    # `parse_known_args` is used here because some arguments might be specific to
    # the model (e.g., TCN parameters) and not directly handled by this main parser.
    args, _ = parser.parse_known_args()

    # Model-specific configurations and loss weights.
    model_configs: Dict[str, Dict[str, Any]] = {
        "eskf_tcn": {
            "model_params": {"tcn_input_size": 20, "use_zupt": True, "use_tcn_zupt": True},
            "loss_weights": {
                "pos_weight": 0.1,
                "vel_weight": 1.0,
                "zupt_weight": 0.1,
                "reg_weight": 1e-5,
                "cov_weight": 0.01,
            },
        },
        "aekf_tcn": {
            "model_params": {"tcn_input_size": 20, "use_zupt": True},
            "loss_weights": {
                "pos_weight": 0.1,
                "vel_weight": 1.0,
                "zupt_weight": 0.1,
                "reg_weight": 1e-5,
            },
        },
        "only_tcn": {
            "model_params": {"input_size": 7, "output_size": 3},
            "loss_weights": {},  # Only PositionLoss for OnlyTCN, no specific weights.
        },
    }

    selected_config: Dict[str, Any] = model_configs.get(args.model, {})
    model_params: Dict[str, Any] = selected_config.get("model_params", {})
    loss_weights: Dict[str, float] = selected_config.get("loss_weights", {})

    print(f"Training on device: {args.device} | Augmentation: {args.augment}")

    # Initialize Dataset and DataLoader.
    # `TrajectoryDataset` now takes only `preprocessed_file` and `augment_multiplier`.
    dataset: TrajectoryDataset = TrajectoryDataset(
        preprocessed_file="./data/dataset.h5", augment_multiplier=10
    )
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Load scaler statistics (mean and std) for sensor data normalization.
    with h5py.File("./data/scaler_stats.h5", "r") as f:
        mean: torch.Tensor = torch.from_numpy(f["mean"][:]).float()
        std: torch.Tensor = torch.from_numpy(f["std"][:]).float()

    # Model Initialization.
    model_class_map: Dict[str, Any] = {
        "eskf_tcn": ESKFTCN_model,
        "aekf_tcn": AEKFTCN_model,
        "only_tcn": OnlyTCN,  # Use the renamed OnlyTCN class.
    }
    model_class: Any = model_class_map.get(args.model)
    model: nn.Module = model_class(device=args.device, dt=args.dt, **model_params)

    model_path: str = f"{args.model}_model.pth"

    # Start Training.
    train(args.model, model, dataloader, args.epochs, args.lr, args.device, model_path, mean, std, **loss_weights)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)  # Exit with a non-zero code to indicate an error.