"""
Minimal ESKF-TCN Training Script (Refactored)

Features:
1. Dynamic Weight Averaging (DWA) for automated loss balancing.
2. Gyro-based Context-Aware Weighting for sharp cornering.
3. Physics-informed constraints kept separate from DWA.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.ESKF_TCN import ESKFTCN_model
from model.dataset import TrajectoryDataset
from model.rotation_utils import quaternion_to_rotation_matrix
from model.dwa import DWALossUpdater
from model.losses import (
    magnitude_loss,
    context_aware_direction_loss,
    zupt_loss,
    covariance_nll_loss,
    regularization_loss,
    fft_loss,
    sliding_window_delta_loss,
    integrate_velocity_to_position
)
from model.config import Config
from model.validation import validate_batch_tensors, validate_loss_dict
from model.qat_tcn import QATScheduler, PT2E_AVAILABLE


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    dataset_path: str = "data/dataset.h5"
    val_dataset_path: str = "data/validation_dataset.h5"
    scaler_path: str = "data/scaler_stats.h5"
    augment_multiplier: int = 1
    do_augment: bool = True
    yaw_angle: Tuple[float, float] = Config.YAW_RANGE
    sigma_tilt: float = Config.SIGMA_TILT

    # Training
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Initial Loss weights (used as fallback or initial values)
    w_mag: float = 1.0
    w_cos: float = 1.0
    w_zupt: float = 0.5
    w_cov: float = 0.1
    w_fft: float = 0.5  # FFT Loss Weight
    w_reg: float = Config.LOSS.REG_WEIGHT_ESKF_TCN     # Fixed regularization (Not touched by DWA)

    # Delta loss (semi-loop closure) - Fixed weight, not in DWA
    use_delta_loss: bool = Config.DELTA_LOSS.ENABLED
    w_delta: float = Config.DELTA_LOSS.WEIGHT
    delta_window_sizes: List[int] = None  # Defaults to Config.DELTA_LOSS.WINDOW_SIZES
    delta_window_weights: List[float] = None  # Defaults to Config.DELTA_LOSS.WINDOW_WEIGHTS
    delta_stride: int = Config.DELTA_LOSS.STRIDE

    # ZUPT threshold
    zupt_vel_threshold: float = 0.005

    # Model Hyperparameters (HPO)
    tcn_channels: List[int] = None  # Defaults to Config.ESKFTCN.TCN_CHANNELS if None
    tcn_kernel_size: int = Config.ESKFTCN.KERNEL_SIZE
    tcn_dropout: float = Config.ESKFTCN.DROPOUT
    mahalanobis_threshold: float = Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD

    # Output
    checkpoint_dir: str = "checkpoints"
    model_name: str = "eskf_tcn_dwa_gyro"
    resume_path: Optional[str] = None

    # Quantization-Aware Training (QAT)
    qat_enabled: bool = Config.ESKFTCN.QAT_ENABLED
    qat_start_epoch: int = Config.ESKFTCN.QAT_START_EPOCH

    # Device (auto-detect: CUDA > MPS > CPU)
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def __post_init__(self):
        if self.tcn_channels is None:
            self.tcn_channels = Config.ESKFTCN.TCN_CHANNELS
        if self.delta_window_sizes is None:
            self.delta_window_sizes = Config.DELTA_LOSS.WINDOW_SIZES
        if self.delta_window_weights is None:
            self.delta_window_weights = Config.DELTA_LOSS.WINDOW_WEIGHTS


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    config: TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    task_weights: Optional[torch.Tensor] = None, # Received from Train Loop
    epoch: int = 0,
    batch_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """Single training step with dynamic weighting."""

    # Validate input batch before moving to device
    validate_batch_tensors(batch, expected_sensor_channels=7, expected_spatial_dims=3)

    device = config.device
    sensor_raw = batch["imu_seq_raw"].to(device)
    gt_vel = batch["gt_vel_w"].to(device)
    gt_pos = batch["gt_pos_w"].to(device)
    seq_lens = batch["len"].to(device)

    B, T = sensor_raw.shape[:2]
    mask = torch.arange(T, device=device)[None, :] < seq_lens[:, None]

    gt_vel_norm = torch.norm(gt_vel, dim=-1, keepdim=True)
    gt_zupt = (gt_vel_norm < config.zupt_vel_threshold).float()

    # Use pre-normalized data from dataset if available, otherwise compute
    if "imu_seq_norm" in batch:
        sensor_norm = batch["imu_seq_norm"].to(device)
    else:
        sensor_norm = (sensor_raw - mean) / (std + 1e-6)

    # [Important] Extract Gyro for Context-Aware Loss (Assumes indices 3:6)
    gyro_raw = sensor_raw[:, :, 3:6]

    # Forward
    out = model(sensor_raw, sensor_norm)

    filter_vel_w = out["filter_vel_w"]
    vel_correction_b = out["pred_vel_resid_b"]
    filter_quat = out["filter_quat"]

    R_b_to_w = quaternion_to_rotation_matrix(filter_quat.view(-1, 4))
    R_b_to_w = R_b_to_w.view(B, T, 3, 3)
    vel_correction_w = (R_b_to_w @ vel_correction_b.unsqueeze(-1)).squeeze(-1)
    pred_vel_w = filter_vel_w + vel_correction_w

    losses = {}

    # 1. Compute Individual Losses
    losses["mag"] = magnitude_loss(pred_vel_w, gt_vel, mask)

    # [Gyro-Weighting Applied Here]
    losses["cos"] = context_aware_direction_loss(pred_vel_w, gt_vel, mask, gt_zupt, gyro_raw)

    losses["reg"] = regularization_loss(vel_correction_b, mask)

    losses["fft"] = fft_loss(pred_vel_w, gt_vel, mask)

    if out.get("pred_zupt_prob") is not None:
        losses["zupt"] = zupt_loss(out["pred_zupt_prob"], gt_zupt, mask)
    else:
        losses["zupt"] = torch.tensor(0.0, device=device)

    if out.get("filter_innovation") is not None and out.get("pred_covariance_R") is not None:
        tcn_mask = out.get("tcn_output_mask", mask)
        losses["cov"] = covariance_nll_loss(out["filter_innovation"], out["pred_covariance_R"], mask, tcn_mask)
    else:
        losses["cov"] = torch.tensor(0.0, device=device)

    # Delta loss (semi-loop closure)
    if config.use_delta_loss:
        initial_pos_w = gt_pos[:, 0, :]
        pred_pos_w = integrate_velocity_to_position(pred_vel_w, initial_pos_w, Config.DT)
        losses["delta"] = sliding_window_delta_loss(
            pred_pos_w, gt_pos, mask,
            window_sizes=config.delta_window_sizes,
            window_weights=config.delta_window_weights,
            stride=config.delta_stride
        )
    else:
        losses["delta"] = torch.tensor(0.0, device=device)

    # 2. Apply Weights (DWA or Config)
    if task_weights is not None:
        # DWA applies to learning tasks: [mag, cos, zupt, cov, fft]
        # 'reg' and 'delta' are excluded (Physics Constraints)
        losses["total"] = (
            task_weights[0] * losses["mag"] +
            task_weights[1] * losses["cos"] +
            task_weights[2] * losses["zupt"] +
            task_weights[3] * losses["cov"] +
            task_weights[4] * losses["fft"] +
            config.w_reg * losses["reg"] +
            config.w_delta * losses["delta"]
        )
    else:
        # Fallback
        losses["total"] = (
            config.w_mag * losses["mag"] +
            config.w_cos * losses["cos"] +
            config.w_zupt * losses["zupt"] +
            config.w_cov * losses["cov"] +
            config.w_fft * losses["fft"] +
            config.w_reg * losses["reg"] +
            config.w_delta * losses["delta"]
        )

    return losses


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_losses = {k: 0.0 for k in ["total", "mag", "cos", "zupt", "cov", "fft", "reg", "delta"]}
    num_batches = 0

    for batch in dataloader:
        # Validation uses Fixed Weights (Standard Metric) to keep comparison fair
        losses = train_step(model, batch, config, mean, std, task_weights=None)
        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

    return {k: v / num_batches for k, v in total_losses.items()}


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig, check_grads: bool = False, model: Optional[nn.Module] = None) -> float:
    print(f"Training ESKF-TCN on {config.device} with DWA + Gyro-Context + FFT")

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Check file existence before loading
    if not os.path.exists(config.scaler_path):
        raise FileNotFoundError(
            f"Scaler stats not found at {config.scaler_path}\n"
            f"Run data preprocessing first: python utils/acquire.py --reprocess"
        )

    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(
            f"Training dataset not found at {config.dataset_path}\n"
            f"Run data preprocessing first: python utils/acquire.py"
        )

    if not os.path.exists(config.val_dataset_path):
        raise FileNotFoundError(
            f"Validation dataset not found at {config.val_dataset_path}\n"
            f"Run data preprocessing first: python utils/acquire.py"
        )

    with h5py.File(config.scaler_path, "r") as f:
        mean = torch.tensor(f["mean"][:], dtype=torch.float32).to(config.device)
        std = torch.tensor(f["std"][:], dtype=torch.float32).to(config.device)

    train_dataset = TrajectoryDataset(
        config.dataset_path,
        do_augment=config.do_augment,
        augment_multiplier=config.augment_multiplier,
        yaw_range=config.yaw_angle,
        sigma_tilt=config.sigma_tilt,
        scaler_stats_path=config.scaler_path,
        return_normalized=True
    )
    val_dataset = TrajectoryDataset(
        config.val_dataset_path,
        do_augment=False,
        scaler_stats_path=config.scaler_path,
        return_normalized=True
    )

    # Data loader optimization: use multiple workers for parallel data loading
    # pin_memory speeds up GPU transfers (only useful for CUDA)
    num_workers = 4 if config.device != "mps" else 0  # MPS has issues with multiprocessing
    pin_memory = config.device == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    if model is None:
        model = ESKFTCN_model(
            device=config.device,
            tcn_channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.tcn_dropout,
            mahalanobis_threshold=config.mahalanobis_threshold
        )

    # Ensure model is on the correct device
    model = model.to(config.device)

    # Compile ESKF for faster execution (PyTorch 2.0+)
    # Skip if on MPS (not fully supported) or if torch.compile unavailable
    if hasattr(torch, 'compile') and config.device != "mps":
        try:
            model.filter = torch.compile(model.filter, mode="reduce-overhead")
            print("ESKF compiled with torch.compile for faster execution")
        except Exception as e:
            print(f"torch.compile failed (using eager mode): {e}")

    # Initialize QAT Scheduler
    qat_scheduler = None
    if config.qat_enabled:
        if PT2E_AVAILABLE:
            qat_scheduler = QATScheduler(
                start_epoch=config.qat_start_epoch,
                enabled=True
            )
            # Create example input for QAT export (B, T, D)
            example_input = torch.randn(1, 100, model.tcn_input_size).to(config.device)
            qat_scheduler.set_example_input(example_input)
            print(f"QAT enabled: will activate at epoch {config.qat_start_epoch}")
        else:
            print("Warning: QAT requested but PT2E not available (requires PyTorch >= 2.1)")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # [DWA Setup]
    dwa_updater = DWALossUpdater(num_tasks=5, temp=2.0)
    # Start with equal weights (1.0)
    current_task_weights = torch.ones(5).to(config.device)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume Logic
    if config.resume_path:
        print(f"Resuming from: {config.resume_path}")
        checkpoint = torch.load(config.resume_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for epoch in range(start_epoch, config.epochs):
        # QAT: Check if we should activate quantization-aware training at this epoch
        if qat_scheduler is not None:
            model = qat_scheduler.step(model, epoch)

        model.train()

        # Accumulators for DWA calculation
        epoch_accum_losses = {k: 0.0 for k in ["total", "mag", "cos", "zupt", "cov", "fft", "reg", "delta"]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for i_batch, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Pass current DWA weights to train_step
            losses = train_step(model, batch, config, mean, std,
                              task_weights=current_task_weights,
                              epoch=epoch, batch_idx=i_batch)

            # Validate losses and print if NaN/Inf detected
            is_valid, error_msg = validate_loss_dict(losses, epoch, i_batch)
            if not is_valid:
                print(f"\n{error_msg}")
                continue

            # (Optional) Gradient Check
            if check_grads and i_batch == 0:
                 # Check grads logic can be inserted here if needed
                 pass

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            for k, v in losses.items():
                epoch_accum_losses[k] += v.item()

            pbar.set_postfix({
                "Loss": f"{losses['total'].item():.2f}",
                "MagW": f"{current_task_weights[0]:.2f}",
                "CosW": f"{current_task_weights[1]:.2f}",
                "FFTW": f"{current_task_weights[4]:.2f}"
            })

        # Calculate Average Losses for this Epoch
        num_batches = len(train_loader)
        epoch_avg_losses = {k: v / num_batches for k, v in epoch_accum_losses.items()}

        # [DWA Update] Calculate weights for NEXT epoch
        # List order: [mag, cos, zupt, cov, fft]
        loss_list_for_dwa = [
            epoch_avg_losses['mag'],
            epoch_avg_losses['cos'],
            epoch_avg_losses['zupt'],
            epoch_avg_losses['cov'],
            epoch_avg_losses['fft']
        ]

        new_weights = dwa_updater.get_weights(loss_list_for_dwa)
        current_task_weights = new_weights.to(config.device)

        # Validation & Logging
        val_losses = validate(model, val_loader, config, mean, std)
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train: {epoch_avg_losses['total']:.4f} | Val: {val_losses['total']:.4f} | Delta: {epoch_avg_losses['delta']:.4f}")
        print(f"  [DWA Weights] Mag: {current_task_weights[0]:.2f}, Cos: {current_task_weights[1]:.2f}, Zupt: {current_task_weights[2]:.2f}, Cov: {current_task_weights[3]:.2f}, FFT: {current_task_weights[4]:.2f}")

        # Save Best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_path = Path(config.checkpoint_dir) / f"{config.model_name}_best.pth"
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_losses["total"],
                "config": config,
            }
            # Save QAT state if active
            if qat_scheduler is not None:
                checkpoint_data["qat_active"] = qat_scheduler.is_active()
            torch.save(checkpoint_data, save_path)
            print(f"  Saved best model." + (" [QAT]" if qat_scheduler and qat_scheduler.is_active() else ""))

    qat_status = " (QAT trained)" if (qat_scheduler and qat_scheduler.is_active()) else ""
    print(f"Training complete{qat_status}. Best val_loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", type=str, default="data/dataset.h5")
    parser.add_argument("--val-dataset", type=str, default="data/dataset.h5")
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    # Loss weights (Initial/Fallback)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-zupt", type=float, default=0.5)
    parser.add_argument("--w-cov", type=float, default=0.1)
    parser.add_argument("--w-fft", type=float, default=0.5)
    parser.add_argument("--w-reg", type=float, default=1e-2)
    # Delta loss
    parser.add_argument("--no-delta-loss", action="store_true", help="Disable delta loss")
    parser.add_argument("--w-delta", type=float, default=Config.DELTA_LOSS.WEIGHT)
    parser.add_argument("--check-grads", action="store_true")
    # QAT (Quantization-Aware Training)
    parser.add_argument("--qat", action="store_true", help="Enable Quantization-Aware Training")
    parser.add_argument("--qat-start-epoch", type=int, default=Config.ESKFTCN.QAT_START_EPOCH,
                        help=f"Epoch to start QAT (default: {Config.ESKFTCN.QAT_START_EPOCH})")

    args = parser.parse_args()

    config = TrainConfig(
        dataset_path=args.dataset,
        val_dataset_path=args.val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_mag=args.w_mag,
        w_cos=args.w_cos,
        w_zupt=args.w_zupt,
        w_cov=args.w_cov,
        w_fft=args.w_fft,
        w_reg=args.w_reg,
        use_delta_loss=not args.no_delta_loss,
        w_delta=args.w_delta,
        resume_path=args.resume,
        qat_enabled=args.qat,
        qat_start_epoch=args.qat_start_epoch
    )

    train(config, check_grads=args.check_grads)

if __name__ == "__main__":
    main()