"""
Two-Stage Sim2Real Training for ESKF-TCN.

Stage 1: Pretrain on simulated data with ESKF parameters frozen
Stage 2: Fine-tune on mixed sim+real data with ESKF unfrozen

Usage:
    python train_two_stage.py --sim-dataset data/simulated_dataset.h5
    python train_two_stage.py --resume-stage2 checkpoints/stage1_final.pth
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.ESKF_TCN import ESKFTCN_model
from model.dataset import TrajectoryDataset
from model.sim_dataset import SimulatedDataset, MixedDataset
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


@dataclass
class TwoStageConfig:
    """Configuration for two-stage training."""

    # Data paths
    sim_dataset_path: str = Config.SIMDATA.DEFAULT_PATH
    real_dataset_path: str = "data/dataset.h5"
    val_dataset_path: str = "data/validation_dataset.h5"
    scaler_path: str = "data/scaler_stats.h5"

    # Stage 1: Sim pretrain
    stage1_epochs: int = Config.TWO_STAGE.STAGE1_EPOCHS
    stage1_lr: float = Config.TWO_STAGE.STAGE1_LR
    stage1_batch_size: int = Config.TWO_STAGE.STAGE1_BATCH_SIZE
    stage1_eskf_learnable: bool = Config.TWO_STAGE.STAGE1_ESKF_LEARNABLE
    stage1_augment_multiplier: int = 5

    # Stage 2: Mixed fine-tune
    stage2_epochs: int = Config.TWO_STAGE.STAGE2_EPOCHS
    stage2_tcn_lr: float = Config.TWO_STAGE.STAGE2_TCN_LR
    stage2_eskf_lr: float = Config.TWO_STAGE.STAGE2_ESKF_LR
    stage2_batch_size: int = Config.TWO_STAGE.STAGE2_BATCH_SIZE
    stage2_mix_ratio: float = Config.TWO_STAGE.STAGE2_MIX_RATIO
    stage2_augment_multiplier: int = 3

    # Delta loss
    use_delta_loss: bool = Config.DELTA_LOSS.ENABLED
    delta_weight: float = Config.DELTA_LOSS.WEIGHT
    delta_window_sizes: List[int] = field(default_factory=lambda: Config.DELTA_LOSS.WINDOW_SIZES)
    delta_window_weights: List[float] = field(default_factory=lambda: Config.DELTA_LOSS.WINDOW_WEIGHTS)
    delta_stride: int = Config.DELTA_LOSS.STRIDE

    # Common training params
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    zupt_vel_threshold: float = 0.005
    w_reg: float = Config.LOSS.REG_WEIGHT_ESKF_TCN

    # Augmentation
    yaw_range: tuple = Config.YAW_RANGE
    sigma_tilt: float = Config.SIGMA_TILT

    # Output
    checkpoint_dir: str = "checkpoints"
    model_name: str = "eskf_tcn_two_stage"

    # Device (auto-detect: CUDA > MPS > CPU)
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Resume
    resume_stage1: Optional[str] = None
    resume_stage2: Optional[str] = None


def freeze_eskf_params(model: nn.Module) -> None:
    """Freeze ESKF learnable parameters (R_diag, zupt_noise_std, etc.)."""
    for name, param in model.named_parameters():
        if 'filter.R_diag' in name or 'filter.zupt_noise' in name or 'filter.virtual_meas' in name:
            param.requires_grad = False
            print(f"  Frozen: {name}")


def unfreeze_eskf_params(model: nn.Module) -> None:
    """Unfreeze ESKF parameters for fine-tuning."""
    for name, param in model.named_parameters():
        if 'filter.R_diag' in name or 'filter.zupt_noise' in name or 'filter.virtual_meas' in name:
            param.requires_grad = True
            print(f"  Unfrozen: {name}")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    config: TwoStageConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    task_weights: Optional[torch.Tensor] = None,
    epoch: int = 0,
    batch_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """Single training step with optional delta loss."""

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

    gyro_raw = sensor_raw[:, :, 3:6]

    # Forward pass
    out = model(sensor_raw, sensor_norm)

    filter_vel_w = out["filter_vel_w"]
    vel_correction_b = out["pred_vel_resid_b"]  # TCN velocity correction in body frame
    filter_quat = out["filter_quat"]

    # Rotation matrix: body-to-world transformation
    R_b_to_w = quaternion_to_rotation_matrix(filter_quat.view(-1, 4))
    R_b_to_w = R_b_to_w.view(B, T, 3, 3)

    # Transform velocity correction from body to world frame
    vel_correction_w = (R_b_to_w @ vel_correction_b.unsqueeze(-1)).squeeze(-1)
    pred_vel_w = filter_vel_w + vel_correction_w

    losses = {}

    # Standard losses
    losses["mag"] = magnitude_loss(pred_vel_w, gt_vel, mask)
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

    # Combine losses
    if task_weights is not None:
        losses["total"] = (
            task_weights[0] * losses["mag"] +
            task_weights[1] * losses["cos"] +
            task_weights[2] * losses["zupt"] +
            task_weights[3] * losses["cov"] +
            task_weights[4] * losses["fft"] +
            config.w_reg * losses["reg"] +
            config.delta_weight * losses["delta"]
        )
    else:
        losses["total"] = (
            losses["mag"] +
            losses["cos"] +
            0.5 * losses["zupt"] +
            0.1 * losses["cov"] +
            0.5 * losses["fft"] +
            config.w_reg * losses["reg"] +
            config.delta_weight * losses["delta"]
        )

    return losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: TwoStageConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, float]:
    """Validation step."""
    model.eval()
    total_losses = {k: 0.0 for k in ["total", "mag", "cos", "zupt", "cov", "fft", "reg", "delta"]}
    num_batches = 0

    for batch in dataloader:
        losses = train_step(model, batch, config, mean, std, task_weights=None)
        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def train_stage1(config: TwoStageConfig) -> nn.Module:
    """Stage 1: Pretrain on simulated data with frozen ESKF."""
    print("=" * 60)
    print("STAGE 1: Sim Pretrain (ESKF Frozen)")
    print("=" * 60)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load scaler with file existence check
    if not os.path.exists(config.scaler_path):
        raise FileNotFoundError(
            f"Scaler stats not found at {config.scaler_path}\n"
            f"Run data preprocessing first: python utils/acquire.py --reprocess"
        )

    with h5py.File(config.scaler_path, "r") as f:
        mean = torch.tensor(f["mean"][:], dtype=torch.float32).to(config.device)
        std = torch.tensor(f["std"][:], dtype=torch.float32).to(config.device)

    # Create model with ESKF learnable=False
    model = ESKFTCN_model(
        device=config.device,
        eskf_learnable_params=config.stage1_eskf_learnable
    )
    model = model.to(config.device)

    # Freeze ESKF params explicitly
    print("Freezing ESKF parameters:")
    freeze_eskf_params(model)

    # Check dataset files exist
    if not os.path.exists(config.sim_dataset_path):
        raise FileNotFoundError(
            f"Simulated dataset not found at {config.sim_dataset_path}\n"
            f"Generate simulated data first: python utils/generate_sim_data.py"
        )

    if not os.path.exists(config.val_dataset_path):
        raise FileNotFoundError(
            f"Validation dataset not found at {config.val_dataset_path}\n"
            f"Run data preprocessing: python utils/acquire.py --reprocess"
        )

    # Load simulated dataset
    sim_dataset = SimulatedDataset(
        config.sim_dataset_path,
        do_augment=True,
        augment_multiplier=config.stage1_augment_multiplier,
        yaw_range=config.yaw_range,
        sigma_tilt=config.sigma_tilt
    )

    # Validation on real data
    val_dataset = TrajectoryDataset(config.val_dataset_path, do_augment=False)

    train_loader = DataLoader(sim_dataset, batch_size=config.stage1_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.stage1_batch_size, shuffle=False, num_workers=0)

    # Only optimize non-frozen params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(trainable_params)} parameter groups (TCN only)")

    optimizer = AdamW(trainable_params, lr=config.stage1_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.stage1_epochs, eta_min=1e-6)

    # DWA setup
    dwa_updater = DWALossUpdater(num_tasks=5, temp=2.0)
    current_task_weights = torch.ones(5).to(config.device)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume logic
    if config.resume_stage1:
        print(f"Resuming Stage 1 from: {config.resume_stage1}")
        checkpoint = torch.load(config.resume_stage1, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        freeze_eskf_params(model)  # Re-freeze after loading

    for epoch in range(start_epoch, config.stage1_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in ["total", "mag", "cos", "zupt", "cov", "fft", "reg", "delta"]}

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{config.stage1_epochs}")
        for i_batch, batch in enumerate(pbar):
            optimizer.zero_grad()
            losses = train_step(model, batch, config, mean, std,
                              task_weights=current_task_weights,
                              epoch=epoch, batch_idx=i_batch)

            # Validate losses and print if NaN/Inf detected
            is_valid, error_msg = validate_loss_dict(losses, epoch, i_batch)
            if not is_valid:
                print(f"\n⚠️  {error_msg}")
                continue

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

            pbar.set_postfix({"Loss": f"{losses['total'].item():.3f}"})

        num_batches = len(train_loader)
        epoch_avg = {k: v / num_batches for k, v in epoch_losses.items()}

        # DWA update
        loss_list = [epoch_avg['mag'], epoch_avg['cos'], epoch_avg['zupt'], epoch_avg['cov'], epoch_avg['fft']]
        current_task_weights = dwa_updater.get_weights(loss_list).to(config.device)

        # Validation
        val_losses = validate(model, val_loader, config, mean, std)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train={epoch_avg['total']:.4f}, Val={val_losses['total']:.4f}, Delta={epoch_avg['delta']:.4f}")

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_losses["total"],
                "stage": 1,
            }, Path(config.checkpoint_dir) / f"{config.model_name}_stage1_best.pth")
            print("  Saved best Stage 1 model.")

    # Save final stage 1 model
    torch.save({
        "epoch": config.stage1_epochs - 1,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "stage": 1,
    }, Path(config.checkpoint_dir) / f"{config.model_name}_stage1_final.pth")

    print(f"Stage 1 complete. Best val_loss: {best_val_loss:.4f}")
    return model


def train_stage2(config: TwoStageConfig, model: Optional[nn.Module] = None) -> nn.Module:
    """Stage 2: Fine-tune on mixed data with unfrozen ESKF."""
    print("=" * 60)
    print("STAGE 2: Mixed Fine-tune (ESKF Unfrozen)")
    print("=" * 60)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load scaler with file existence check
    if not os.path.exists(config.scaler_path):
        raise FileNotFoundError(
            f"Scaler stats not found at {config.scaler_path}\n"
            f"Run data preprocessing first: python utils/acquire.py --reprocess"
        )

    with h5py.File(config.scaler_path, "r") as f:
        mean = torch.tensor(f["mean"][:], dtype=torch.float32).to(config.device)
        std = torch.tensor(f["std"][:], dtype=torch.float32).to(config.device)

    # Load or create model
    if model is None:
        if config.resume_stage2:
            checkpoint_path = config.resume_stage2
        else:
            checkpoint_path = Path(config.checkpoint_dir) / f"{config.model_name}_stage1_final.pth"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}\n"
                f"Run Stage 1 training first: python train_two_stage.py (without --skip-stage1)"
            )

        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        model = ESKFTCN_model(device=config.device, eskf_learnable_params=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)

    # Unfreeze ESKF params
    print("Unfreezing ESKF parameters:")
    unfreeze_eskf_params(model)

    # Check dataset files exist
    if not os.path.exists(config.sim_dataset_path):
        raise FileNotFoundError(
            f"Simulated dataset not found at {config.sim_dataset_path}\n"
            f"Generate simulated data first: python utils/generate_sim_data.py"
        )

    if not os.path.exists(config.real_dataset_path):
        raise FileNotFoundError(
            f"Real dataset not found at {config.real_dataset_path}\n"
            f"Run data preprocessing: python utils/acquire.py --reprocess"
        )

    if not os.path.exists(config.val_dataset_path):
        raise FileNotFoundError(
            f"Validation dataset not found at {config.val_dataset_path}\n"
            f"Run data preprocessing: python utils/acquire.py --reprocess"
        )

    # Create mixed dataset
    sim_dataset = SimulatedDataset(
        config.sim_dataset_path,
        do_augment=True,
        augment_multiplier=config.stage2_augment_multiplier,
        yaw_range=config.yaw_range,
        sigma_tilt=config.sigma_tilt
    )
    real_dataset = TrajectoryDataset(
        config.real_dataset_path,
        do_augment=True,
        augment_multiplier=config.stage2_augment_multiplier,
        yaw_range=config.yaw_range,
        sigma_tilt=config.sigma_tilt
    )
    mixed_dataset = MixedDataset(sim_dataset, real_dataset, sim_ratio=config.stage2_mix_ratio)
    val_dataset = TrajectoryDataset(config.val_dataset_path, do_augment=False)

    train_loader = DataLoader(mixed_dataset, batch_size=config.stage2_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.stage2_batch_size, shuffle=False, num_workers=0)

    # Separate parameter groups with different LRs
    tcn_params = []
    eskf_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'filter.R_diag' in name or 'filter.zupt_noise' in name or 'filter.virtual_meas' in name:
            eskf_params.append(param)
        else:
            tcn_params.append(param)

    print(f"TCN params: {len(tcn_params)}, ESKF params: {len(eskf_params)}")

    optimizer = AdamW([
        {'params': tcn_params, 'lr': config.stage2_tcn_lr},
        {'params': eskf_params, 'lr': config.stage2_eskf_lr}
    ], weight_decay=config.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.stage2_epochs, eta_min=1e-7)

    # DWA setup
    dwa_updater = DWALossUpdater(num_tasks=5, temp=2.0)
    current_task_weights = torch.ones(5).to(config.device)

    best_val_loss = float("inf")

    for epoch in range(config.stage2_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in ["total", "mag", "cos", "zupt", "cov", "fft", "reg", "delta"]}

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{config.stage2_epochs}")
        for i_batch, batch in enumerate(pbar):
            optimizer.zero_grad()
            losses = train_step(model, batch, config, mean, std,
                              task_weights=current_task_weights,
                              epoch=epoch, batch_idx=i_batch)

            # Validate losses and print if NaN/Inf detected
            is_valid, error_msg = validate_loss_dict(losses, epoch, i_batch)
            if not is_valid:
                print(f"\n⚠️  {error_msg}")
                continue

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

            pbar.set_postfix({"Loss": f"{losses['total'].item():.3f}"})

        num_batches = len(train_loader)
        epoch_avg = {k: v / num_batches for k, v in epoch_losses.items()}

        # DWA update
        loss_list = [epoch_avg['mag'], epoch_avg['cos'], epoch_avg['zupt'], epoch_avg['cov'], epoch_avg['fft']]
        current_task_weights = dwa_updater.get_weights(loss_list).to(config.device)

        # Validation
        val_losses = validate(model, val_loader, config, mean, std)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train={epoch_avg['total']:.4f}, Val={val_losses['total']:.4f}, Delta={epoch_avg['delta']:.4f}")

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_losses["total"],
                "stage": 2,
            }, Path(config.checkpoint_dir) / f"{config.model_name}_best.pth")
            print("  Saved best model.")

    print(f"Stage 2 complete. Best val_loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Sim2Real Training")

    # Data paths
    parser.add_argument("--sim-dataset", type=str, default=Config.SIMDATA.DEFAULT_PATH)
    parser.add_argument("--real-dataset", type=str, default="data/dataset.h5")
    parser.add_argument("--val-dataset", type=str, default="data/validation_dataset.h5")

    # Stage 1
    parser.add_argument("--stage1-epochs", type=int, default=Config.TWO_STAGE.STAGE1_EPOCHS)
    parser.add_argument("--stage1-lr", type=float, default=Config.TWO_STAGE.STAGE1_LR)
    parser.add_argument("--stage1-batch-size", type=int, default=Config.TWO_STAGE.STAGE1_BATCH_SIZE)

    # Stage 2
    parser.add_argument("--stage2-epochs", type=int, default=Config.TWO_STAGE.STAGE2_EPOCHS)
    parser.add_argument("--stage2-tcn-lr", type=float, default=Config.TWO_STAGE.STAGE2_TCN_LR)
    parser.add_argument("--stage2-eskf-lr", type=float, default=Config.TWO_STAGE.STAGE2_ESKF_LR)
    parser.add_argument("--stage2-batch-size", type=int, default=Config.TWO_STAGE.STAGE2_BATCH_SIZE)
    parser.add_argument("--stage2-mix-ratio", type=float, default=Config.TWO_STAGE.STAGE2_MIX_RATIO)

    # Delta loss
    parser.add_argument("--no-delta-loss", action="store_true", help="Disable delta loss")
    parser.add_argument("--delta-weight", type=float, default=Config.DELTA_LOSS.WEIGHT)

    # Resume
    parser.add_argument("--resume-stage1", type=str, default=None, help="Resume Stage 1 from checkpoint")
    parser.add_argument("--resume-stage2", type=str, default=None, help="Resume Stage 2 from checkpoint")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1, load from stage1_final")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--model-name", type=str, default="eskf_tcn_two_stage")

    args = parser.parse_args()

    config = TwoStageConfig(
        sim_dataset_path=args.sim_dataset,
        real_dataset_path=args.real_dataset,
        val_dataset_path=args.val_dataset,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage1_batch_size=args.stage1_batch_size,
        stage2_epochs=args.stage2_epochs,
        stage2_tcn_lr=args.stage2_tcn_lr,
        stage2_eskf_lr=args.stage2_eskf_lr,
        stage2_batch_size=args.stage2_batch_size,
        stage2_mix_ratio=args.stage2_mix_ratio,
        use_delta_loss=not args.no_delta_loss,
        delta_weight=args.delta_weight,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        resume_stage1=args.resume_stage1,
        resume_stage2=args.resume_stage2,
    )

    if args.skip_stage1 or args.resume_stage2:
        # Skip Stage 1, go directly to Stage 2
        model = train_stage2(config)
    else:
        # Full two-stage training
        model = train_stage1(config)
        model = train_stage2(config, model)

    print("Two-stage training complete!")


if __name__ == "__main__":
    main()
