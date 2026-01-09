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
import sys
from typing import Any, Dict, List, Tuple
from datetime import datetime
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Suppress common noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress FutureWarning (numpy, keras, etc.)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')  # Suppress PyTorch UserWarnings
warnings.filterwarnings('ignore', message='.*MPS.*')  # Suppress MPS-related warnings

# Add parent directory to sys.path for relative imports to models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.AEKF_TCN import AEKFTCN_model
from model.ESKF_TCN import ESKFTCN_model
from model.onlyTCN import OnlyTCN
from model.rotation_utils import quaternion_to_rotation_matrix
from model.dataset import TrajectoryDataset
from model.config import Config


class UncertaintyLoss(nn.Module):
    """
    Calculates a hybrid loss by automatically weighting tasks based on homoscedastic
    uncertainty, as described in "Multi-Task Learning Using Uncertainty to Weigh
    Losses..." by Kendall, Gal, and Cipolla.

    This approach avoids manual tuning of loss weights by making them learnable
    parameters of the model.
    """

    def __init__(self, device: str):
        """Initializes the UncertaintyLoss module."""
        super().__init__()
        # Use 'none' reduction to get per-element losses for manual masking and weighting.
        self.pos_criterion = nn.SmoothL1Loss(reduction="none")
        self.vel_criterion = nn.SmoothL1Loss(reduction="none")
        self.zupt_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Create learnable parameters for the log variance of each task.
        # Initializing with zeros is a common starting point.
        self.log_var_pos = nn.Parameter(torch.tensor(0.0, device=device)) # Position Loss weight
        self.log_var_vel = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_var_cos = nn.Parameter(torch.tensor(0.0, device=device)) # Cosine Similarity Loss weight
        self.log_var_zupt = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_var_cov = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        reg_weight: float,
        model_name: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss as a sum of uncertainty-weighted task losses.
        Each task loss is formulated as: loss = exp(-log_var) * task_error + log_var
        """
        # --- Common Preparations ---
        mask_3d = mask.unsqueeze(-1)
        valid_element_count = mask.sum() + 1e-8
        losses = {}
        total_loss = torch.tensor(0.0, device=mask.device)

        # --- Position Loss ---
        # Only applied for 'only_tcn' as it predicts position directly.
        # For hybrid models, position drift makes this loss counter-productive.
        if model_name == "only_tcn":
             pred_pos_w = model_out
             per_element_pos_loss = self.pos_criterion(pred_pos_w, batch["gt_pos_w"])
             mse_pos = (per_element_pos_loss * mask_3d).sum() / valid_element_count

             # Apply uncertainty weighting
             precision_pos = torch.exp(-self.log_var_pos)
             loss_pos = precision_pos * mse_pos + self.log_var_pos
             total_loss += loss_pos
             # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
             losses["pos"] = mse_pos.detach()
             losses["w_pos"] = loss_pos.detach()

        # --- Hybrid Model Losses (ESKF-TCN, AEKF-TCN) ---
        if model_name != "only_tcn":
            # --- Velocity Loss ---
            filter_vel_w = model_out["filter_vel_w"]
            pred_vel_resid_b = model_out["pred_vel_resid_b"]
            filter_quat = model_out["filter_quat"]
            rot_mat_b_to_w = quaternion_to_rotation_matrix(filter_quat.view(-1, 4)).view(*filter_quat.shape[:-1], 3, 3)
            pred_vel_resid_w = (rot_mat_b_to_w @ pred_vel_resid_b.unsqueeze(-1)).squeeze(-1)
            pred_total_vel_w = filter_vel_w + pred_vel_resid_w

            # 1. SmoothL1 Loss for Velocity Magnitude/Values
            per_element_vel_loss = self.vel_criterion(pred_total_vel_w, batch["gt_vel_w"])
            mse_vel = (per_element_vel_loss * mask_3d).sum() / valid_element_count
            precision_vel = torch.exp(-self.log_var_vel)
            loss_vel = precision_vel * mse_vel + self.log_var_vel
            total_loss += loss_vel
            # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
            losses["vel"] = mse_vel.detach()
            losses["w_vel"] = loss_vel.detach()

            # 2. Cosine Similarity Loss for Velocity Direction
            # Loss = 1 - cosine_similarity. Range [0, 2].
            cos_sim = F.cosine_similarity(pred_total_vel_w, batch["gt_vel_w"], dim=-1, eps=1e-6)
            per_element_cos_loss = 1.0 - cos_sim
            moving_mask = mask * (1.0 - batch["gt_zupt"].squeeze(-1)) # Adopt Cos Loss when Moving
            mean_cos_loss = (per_element_cos_loss * moving_mask).sum() / (moving_mask.sum() + 1e-8)
            # mean_cos_loss = (per_element_cos_loss * mask).sum() / valid_element_count
            precision_cos = torch.exp(-self.log_var_cos)
            loss_cos = precision_cos * mean_cos_loss + self.log_var_cos
            total_loss += loss_cos
            # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
            losses["cos"] = mean_cos_loss.detach()
            losses["w_cos"] = loss_cos.detach()

            # --- ZUPT Loss (Conditional) ---
            if "pred_zupt_prob" in model_out and model_out["pred_zupt_prob"] is not None:
                per_element_zupt_loss = self.zupt_criterion(model_out["pred_zupt_prob"], batch["gt_zupt"])
                bce_zupt = (per_element_zupt_loss * mask_3d).sum() / valid_element_count
                precision_zupt = torch.exp(-self.log_var_zupt)
                loss_zupt = precision_zupt * bce_zupt + self.log_var_zupt
                total_loss += loss_zupt
                # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
                losses["zupt"] = bce_zupt.detach()
                losses["w_zupt"] = loss_zupt.detach()

            # --- Regularization Loss (fixed weight) ---
            # Compute mean squared error instead of total norm to avoid scaling with sequence length
            masked_vel_resid = pred_vel_resid_b * mask_3d
            loss_reg = (masked_vel_resid ** 2).sum() / (mask_3d.sum() + 1e-8)
            weighted_reg_loss = reg_weight * loss_reg
            total_loss += weighted_reg_loss
            # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
            losses["reg"] = loss_reg.detach()
            losses["w_reg"] = weighted_reg_loss.detach()

        # --- Probabilistic Model Loss (ESKF-TCN) ---
        if model_name == "eskf_tcn":
            innovation = model_out["filter_innovation"]
            pred_R_diag = model_out["pred_covariance_R"]
            tcn_output_mask = model_out["tcn_output_mask"]
            final_cov_mask = tcn_output_mask & mask

            nll_cov = torch.tensor(0.0, device=innovation.device)
            if final_cov_mask.any():
                innovation_valid = innovation[final_cov_mask]
                pred_R_diag_valid = pred_R_diag[final_cov_mask]
                # Note: Input is already clamped in TCN.py to [-10, 5]
                # Apply softplus for positive variance
                variance = F.softplus(pred_R_diag_valid) + 1e-4
                # Clamp output variance to match ESKF measurement update behavior
                variance = torch.clamp(variance, min=1e-4, max=3.0)
                nll_elementwise = 0.5 * (torch.square(innovation_valid) / variance + torch.log(variance))
                nll_cov = torch.mean(torch.sum(nll_elementwise, dim=-1))

            precision_cov = torch.exp(-self.log_var_cov)
            loss_cov = precision_cov * nll_cov + self.log_var_cov
            total_loss += loss_cov
            # OPTIMIZATION: Return detached tensors instead of .item() to avoid GPU sync
            losses["cov"] = nll_cov.detach()
            losses["w_cov"] = loss_cov.detach()

        return total_loss, losses


def validate(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: UncertaintyLoss,
    device: str,
    reg_weight: float,
    model_name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, float]:
    """Evaluates the model on the validation set."""
    model.eval()
    val_losses = {}
    total_val_loss = 0.0
    num_batches = 0

    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    with torch.no_grad():
        for batch in val_dataloader:
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            gt_vel_w = batch["gt_vel_w"].to(device)
            seq_lens = batch["len"].to(device)

            max_len = sensor_raw.shape[1]
            mask = torch.arange(max_len, device=device)[None, :] < seq_lens[:, None]

            gt_vel_norm = torch.norm(gt_vel_w, dim=-1, keepdim=True)
            gt_zupt = (gt_vel_norm < 0.005).float()

            sensor_norm = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            model_output = model(sensor_raw, sensor_norm)

            batch_gpu = {
                "gt_pos_w": gt_pos_w,
                "gt_vel_w": gt_vel_w,
                "gt_zupt": gt_zupt,
            }

            loss, sub_losses = criterion(
                model_out=model_output,
                batch=batch_gpu,
                mask=mask,
                reg_weight=reg_weight,
                model_name=model_name,
            )

            # Add Pen Tip Regularization Loss if available
            if hasattr(model, "get_pen_tip_regularization_loss"):
                pen_tip_loss = model.get_pen_tip_regularization_loss()
                weighted_pen_tip = reg_weight * pen_tip_loss
                loss += weighted_pen_tip
                # OPTIMIZATION: Accumulate detached tensors
                prev_reg = sub_losses.get("reg", torch.tensor(0.0, device=pen_tip_loss.device))
                prev_w_reg = sub_losses.get("w_reg", torch.tensor(0.0, device=pen_tip_loss.device))
                sub_losses["reg"] = prev_reg + pen_tip_loss.detach() if torch.is_tensor(prev_reg) else torch.tensor(prev_reg, device=pen_tip_loss.device) + pen_tip_loss.detach()
                sub_losses["w_reg"] = prev_w_reg + weighted_pen_tip.detach() if torch.is_tensor(prev_w_reg) else torch.tensor(prev_w_reg, device=pen_tip_loss.device) + weighted_pen_tip.detach()

            # OPTIMIZATION: Convert tensors to scalars only once per batch
            total_val_loss += loss.detach().item()
            for k, v in sub_losses.items():
                v_scalar = v.item() if torch.is_tensor(v) else v
                val_losses[k] = val_losses.get(k, 0.0) + v_scalar

            num_batches += 1

    avg_losses = {"total": total_val_loss / num_batches}
    for k, v in val_losses.items():
        avg_losses[k] = v / num_batches

    return avg_losses


def train(
    model_name: str,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_path: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    reg_weight: float = 0.0,
    reg_warmup_epochs: int = 50,
    patience: int = 20,
    min_delta: float = 1e-4,
) -> float:
    """Runs the training loop for the specified model with early stopping support.

    Args:
        reg_warmup_epochs: Number of epochs to warm up regularization from 0 to reg_weight.
                          Set to 0 to disable warmup (use full reg_weight from start).
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum change in validation loss to qualify as improvement.
    """

    # Initialize TensorBoard SummaryWriter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"{model_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging enabled: {log_dir}")

    criterion: nn.Module = UncertaintyLoss(device=device)

    # Ensure the learnable loss parameters are included in the optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    # ReduceLROnPlateau: Reduce LR when validation loss plateaus
    # Called once per epoch with validation loss (not per-batch)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    # Check for checkpoint to resume training
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_best.pth")
    start_epoch = 0

    # Initialize history dictionaries for both training and validation
    train_history: Dict[str, List[float]] = {"total": [], "pos": [], "vel": [], "cos": [], "zupt": [], "reg": [], "cov": [], "lr": []}

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} to resume training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Restore training history if available
        if 'train_history' in checkpoint:
            train_history = checkpoint['train_history']
            print(f"Loaded training history with {len(train_history['total'])} epochs")
        # Restore early stopping state if available
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"Restored early stopping state: best_val_loss={best_val_loss:.4f}, epochs_without_improvement={epochs_without_improvement}")
        print(f"Resumed from epoch {start_epoch}")
    elif os.path.exists(model_path):
        print(f"Loading saved model from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    print(f"Start Training for {model_name} on {device}...")

    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    # Store target regularization weight for warmup
    target_reg_weight = reg_weight

    for epoch in range(start_epoch, epochs):
        # --- Calculate Regularization Warm-up Weight ---
        if reg_warmup_epochs > 0 and epoch < 20:
            current_reg_weight = 0
        elif reg_warmup_epochs > 0 and 20 <= epoch < reg_warmup_epochs:
            # Linear ramp from 0.0 to target_reg_weight
            # Allows model to learn main task before applying regularization
            progress = epoch / float(reg_warmup_epochs)
            current_reg_weight = progress * target_reg_weight
        else:
            # Use full regularization weight
            current_reg_weight = target_reg_weight

        # --- Training Loop ---
        model.train() # Set model to training mode
        epoch_train_losses: Dict[str, float] = {k: 0.0 for k in train_history.keys() if k != "lr"}
        # Add weighted loss keys for tracking
        for k in ["pos", "vel", "cos", "zupt", "reg", "cov"]:
            epoch_train_losses[f"w_{k}"] = 0.0

        # Track gradient norms for monitoring
        epoch_grad_norms: List[float] = []

        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar_train:
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            gt_vel_w = batch["gt_vel_w"].to(device)
            seq_lens = batch["len"].to(device)

            max_len = sensor_raw.shape[1]
            mask = torch.arange(max_len, device=device)[None, :] < seq_lens[:, None]

            gt_vel_norm = torch.norm(gt_vel_w, dim=-1, keepdim=True)
            gt_zupt = (gt_vel_norm < 0.005).float()  # Tightened from 0.01 to 0.005 for better static/slow-motion separation

            sensor_norm = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            optimizer.zero_grad()

            model_output = model(sensor_raw, sensor_norm)

            batch_gpu = {
                "gt_pos_w": gt_pos_w,
                "gt_vel_w": gt_vel_w,
                "gt_zupt": gt_zupt,
            }

            loss, sub_losses = criterion(
                model_out=model_output,
                batch=batch_gpu,
                mask=mask,
                reg_weight=current_reg_weight,
                model_name=model_name,
            )

            # Add Pen Tip Regularization Loss if available
            if hasattr(model, "get_pen_tip_regularization_loss"):
                pen_tip_loss = model.get_pen_tip_regularization_loss()
                # Weight it? Let's treat it as part of 'reg' loss or add it to total.
                # Since 'reg_weight' is passed to criterion for velocity reg,
                # we can use a small weight or just add it.
                # Let's add it to total loss and log it under 'reg' (accumulating).
                # Assuming reg_weight is appropriate for this too (it's usually small, e.g., 1e-4).
                weighted_pen_tip = current_reg_weight * pen_tip_loss
                loss += weighted_pen_tip
                # OPTIMIZATION: Accumulate detached tensors instead of calling .item() here
                prev_reg = sub_losses.get("reg", torch.tensor(0.0, device=pen_tip_loss.device))
                prev_w_reg = sub_losses.get("w_reg", torch.tensor(0.0, device=pen_tip_loss.device))
                sub_losses["reg"] = prev_reg + pen_tip_loss.detach() if torch.is_tensor(prev_reg) else torch.tensor(prev_reg, device=pen_tip_loss.device) + pen_tip_loss.detach()
                sub_losses["w_reg"] = prev_w_reg + weighted_pen_tip.detach() if torch.is_tensor(prev_w_reg) else torch.tensor(prev_w_reg, device=pen_tip_loss.device) + weighted_pen_tip.detach()


            # CRITICAL: Check for NaN/Inf before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN/Inf loss detected! Skipping batch. Loss: {loss.item()}")
                # Print sub-losses for debugging
                for k, v in sub_losses.items():
                    v_check = v if not torch.is_tensor(v) else v.item()
                    if np.isnan(v_check) or np.isinf(v_check):
                        print(f"  - Problematic loss component '{k}': {v_check}")
                continue  # Skip this batch

            loss.backward()

            # Gradient clipping to prevent explosion (critical post-GroupNorm removal)
            # Returns the total norm of gradients before clipping for monitoring
            grad_clip_threshold = 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_threshold
            )
            # Track gradient norm for epoch average
            epoch_grad_norms.append(grad_norm.item())

            optimizer.step()

            # OPTIMIZATION: Detach loss before .item() to avoid holding computation graph
            loss_value = loss.detach().item()
            epoch_train_losses["total"] += loss_value
            pbar_train_postfix: Dict[str, float] = {"L": loss_value}  # Reuse value
            # OPTIMIZATION: Convert tensor losses to scalars only once for display
            for k, v in sub_losses.items():
                # Convert tensor to scalar (sub_losses now contains detached tensors)
                v_scalar = v.item() if torch.is_tensor(v) else v
                if k in epoch_train_losses:
                    epoch_train_losses[k] += v_scalar if not (np.isnan(v_scalar) or np.isinf(v_scalar)) else 0
                if not k.startswith("w_"): # Only show raw losses in progress bar
                    pbar_train_postfix[f"L_{k}"] = v_scalar
            pbar_train.set_postfix(pbar_train_postfix)

        for k in train_history.keys():
            if k in epoch_train_losses:
                train_history[k].append(epoch_train_losses[k] / len(train_dataloader))

        # Log Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history["lr"].append(current_lr)

        # --- Validation Loop ---
        # Use target_reg_weight for validation (full weight, not warmed-up)
        val_metrics = validate(model, val_dataloader, criterion, device, target_reg_weight, model_name, mean, std)

        # Step the scheduler with validation loss (ReduceLROnPlateau)
        scheduler.step(val_metrics['total'])

        # --- TensorBoard Logging ---
        writer.add_scalar("Loss/Train_Total", train_history["total"][-1], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics["total"], epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Log gradient norm (average over epoch)
        if epoch_grad_norms:
            avg_grad_norm = np.mean(epoch_grad_norms)
            max_grad_norm_epoch = np.max(epoch_grad_norms)
            writer.add_scalar("Gradients/Average_Norm", avg_grad_norm, epoch)
            writer.add_scalar("Gradients/Max_Norm", max_grad_norm_epoch, epoch)

        # Log sub-losses (Train)
        for k in ["pos", "vel", "cos", "zupt", "reg", "cov"]:
            if k in epoch_train_losses:
                val = epoch_train_losses[k] / len(train_dataloader)
                writer.add_scalar(f"Loss_Components/Train_{k.capitalize()}", val, epoch)

        # Log weighted sub-losses (Train)
        for k in ["pos", "vel", "cos", "zupt", "reg", "cov"]:
             key = f"w_{k}"
             if key in epoch_train_losses:
                 val = epoch_train_losses[key] / len(train_dataloader)
                 writer.add_scalar(f"Loss_Weighted/Train_{k.capitalize()}", val, epoch)

        # Log sub-losses (Val)
        for k, v in val_metrics.items():
            if k != "total" and not k.startswith("w_"):
                writer.add_scalar(f"Loss_Components/Val_{k.capitalize()}", v, epoch)

        # Calculate weights for logging
        with torch.no_grad():
            w_pos = torch.exp(-criterion.log_var_pos).item()
            w_vel = torch.exp(-criterion.log_var_vel).item()
            w_cos = torch.exp(-criterion.log_var_cos).item()
            w_zupt = torch.exp(-criterion.log_var_zupt).item()
            w_cov = torch.exp(-criterion.log_var_cov).item()

            writer.add_scalars("Loss_Weights", {
                "pos": w_pos,
                "vel": w_vel,
                "cos": w_cos,
                "zupt": w_zupt,
                "cov": w_cov
            }, epoch)

        # Log regularization weight (warmup progress)
        writer.add_scalar("Regularization/weight", current_reg_weight, epoch)

        log_str = f"Epoch {epoch+1}: Train_Total={train_history['total'][-1]:.4f} | Val_Total={val_metrics['total']:.4f} | LR={current_lr:.2e} | RegW={current_reg_weight:.2e}"

        total_sum = epoch_train_losses["total"]
        def get_pct(key):
            val = epoch_train_losses.get(f"w_{key}", 0.0)
            return (val / total_sum) * 100.0 if total_sum > 0 else 0.0

        if model_name != "only_tcn":
            log_str += f" | Train_Vel={train_history['vel'][-1]:.4f} ({get_pct('vel'):.1f}%)"
            log_str += f" | Train_Cos={train_history['cos'][-1]:.4f} ({get_pct('cos'):.1f}%)"
            log_str += f"\n    Weights: Vel={w_vel:.3f} | Cos={w_cos:.3f} | ZUPT={w_zupt:.3f} ({get_pct('zupt'):.1f}%) | Cov={w_cov:.3f} ({get_pct('cov'):.1f}%) | Reg=({get_pct('reg'):.1f}%)"
        else:
            log_str += f" | Weight_Pos={w_pos:.3f} ({get_pct('pos'):.1f}%)"
        print(log_str)

        # Early stopping logic
        current_val_loss = val_metrics['total']
        if current_val_loss < best_val_loss - min_delta:
            # Improvement detected
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best validation loss: {best_val_loss:.4f} - Model saved to {best_model_path}")
        else:
            # No improvement
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s). Best: {best_val_loss:.4f}")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1 - patience}")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'train_history': train_history,
                'best_val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Restore best model if it exists
    if os.path.exists(best_model_path):
        print(f"\nRestoring best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model state dict for inference/compatibility
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

    # Save final complete checkpoint
    final_checkpoint = {
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'train_history': train_history,
        'best_val_loss': best_val_loss,
        'epochs_without_improvement': epochs_without_improvement,
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")

    writer.close()

    # Plotting training history
    plt.figure(figsize=(12, 8))

    # Define a consistent order for plotting metrics
    plot_order = ["total", "pos", "vel", "cos", "zupt", "reg", "cov"]

    for metric in plot_order:
        if metric in train_history and train_history[metric]:
            plt.plot(train_history[metric], label=f"Train {metric.capitalize()} Loss", marker='.')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss History ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/loss_history_{model_name}.png")
    plt.close()

    return best_val_loss


def main() -> None:
    """Main function to parse arguments, set up training, and start the training process."""
    parser = argparse.ArgumentParser(description="Train various trajectory estimation models.")
    parser.add_argument("--model", type=str, default="eskf_tcn", choices=["eskf_tcn", "aekf_tcn", "only_tcn"], help="Type of model to train.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--reg_warmup_epochs", type=int, default=50, help="Number of epochs for regularization weight warmup (0 -> target). Set to 0 to disable.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="mps" if torch.mps.is_available() else "cpu", help="Computation device ('cpu', 'cuda', 'mps').")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum change in validation loss to qualify as improvement.")
    # HPO-specific arguments
    parser.add_argument("--mahalanobis_threshold", type=float, default=None, help="Mahalanobis distance threshold for measurement gating (HPO).")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate for TCN layers (HPO).")
    parser.add_argument("--reg_weight", type=float, default=None, help="Regularization weight override (HPO).")
    parser.add_argument("--kernel_size", type=int, default=None, help="TCN kernel size (HPO).")
    parser.add_argument("--tcn_channel_size", type=int, default=None, help="TCN channel size per layer (HPO).")
    parser.add_argument("--num_tcn_layers", type=int, default=None, help="Number of TCN layers (HPO).")
    parser.add_argument("--hpo_mode", action="store_true", help="Enable HPO mode (outputs FINAL_LOSS for agent parsing).")
    parser.add_argument("--hpo_trial_id", type=str, default=None, help="HPO trial ID for checkpoint isolation.")
    args, _ = parser.parse_known_args()

    # Note: Most loss weights are now learned automatically. Only `reg_weight` remains.
    # Apply HPO overrides if specified (modify Config at runtime for ESKF consumption)
    dropout_rate = args.dropout if args.dropout is not None else Config.ESKFTCN.DROPOUT
    kernel_size = args.kernel_size if args.kernel_size is not None else Config.ESKFTCN.KERNEL_SIZE
    if args.mahalanobis_threshold is not None:
        Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD = args.mahalanobis_threshold

    # Build TCN channels list from HPO params or use defaults
    num_layers = args.num_tcn_layers if args.num_tcn_layers is not None else len(Config.ESKFTCN.TCN_CHANNELS)
    channel_size = args.tcn_channel_size if args.tcn_channel_size is not None else Config.ESKFTCN.TCN_CHANNELS[0]
    tcn_channels = [channel_size] * num_layers

    # Build dilation factors matching num_layers (using base pattern from config)
    base_dilations = Config.ESKFTCN.TCN_DILATION_FACTORS
    if num_layers <= len(base_dilations):
        tcn_dilations = base_dilations[:num_layers]
    else:
        # Extend with doubling pattern
        tcn_dilations = base_dilations + [base_dilations[-1] * (2 ** i) for i in range(1, num_layers - len(base_dilations) + 1)]

    model_configs: Dict[str, Dict[str, Any]] = {
        "eskf_tcn": {
            "model_params": {
                "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
                "use_zupt": Config.ESKFTCN.USE_ZUPT,
                "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT,
                "dropout": dropout_rate,
                "kernel_size": kernel_size,
                "tcn_channels": tcn_channels,
                "tcn_dilation_factors": tcn_dilations,
            },
            "loss_weights": {"reg_weight": args.reg_weight if args.reg_weight is not None else Config.LOSS.REG_WEIGHT_ESKF_TCN},
        },
        "aekf_tcn": {
            "model_params": {
                "tcn_input_size": Config.AEKFTCN.TCN_INPUT_SIZE,
                "dropout": dropout_rate,
            },
            "loss_weights": {"reg_weight": args.reg_weight if args.reg_weight is not None else Config.LOSS.REG_WEIGHT_AEKF_TCN},
        },
        "only_tcn": {
            "model_params": {"input_size": Config.OnlyTCN.INPUT_SIZE, "output_size": Config.OnlyTCN.OUTPUT_SIZE},
            "loss_weights": {},
        },
    }

    selected_config = model_configs.get(args.model, {})
    model_params = selected_config.get("model_params", {})
    loss_weights = selected_config.get("loss_weights", {})

    print(f"Training on device: {args.device}")

    train_dataset = TrajectoryDataset(
        preprocessed_file=Config.DATASET_H5_PATH,
        augment_multiplier=Config.AUGMENT_MULTIPLIER,
        subsample_step=Config.SUBSAMPLE_STEP,
        do_augment=Config.DO_AUGMENT,
        yaw_range=Config.YAW_ANGLE,
        sigma_tilt=Config.SIGMA_TILT
    )

    # Load Validation Dataset (No augmentation)
    # NOTE: Set Config.VALIDATION_DATASET_H5_PATH to a separate validation file
    # to avoid training/validation data leak
    val_dataset = TrajectoryDataset(
        preprocessed_file=Config.VALIDATION_DATASET_H5_PATH,
        augment_multiplier=1,
        subsample_step=Config.SUBSAMPLE_STEP,
        do_augment=False,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)

    with h5py.File(Config.SCALER_STATS_H5_PATH, "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float()
        std = torch.from_numpy(f["std"][:]).float()

    model_class_map = {
        "eskf_tcn": ESKFTCN_model,
        "aekf_tcn": AEKFTCN_model,
        "only_tcn": OnlyTCN,
    }
    model_class = model_class_map.get(args.model)
    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model}. Available: {list(model_class_map.keys())}")
    model = model_class(device=args.device, dt=Config.DT, **model_params)

    # HPO-specific model path to avoid checkpoint conflicts between trials
    if args.hpo_mode and args.hpo_trial_id:
        hpo_checkpoint_dir = os.path.join("checkpoints", "hpo")
        os.makedirs(hpo_checkpoint_dir, exist_ok=True)
        model_path = os.path.join(hpo_checkpoint_dir, f"{args.model}_{args.hpo_trial_id}.pth")
    else:
        model_path = f"{args.model}_model.pth"

    best_val_loss = train(
        args.model,
        model,
        train_dataloader,
        val_dataloader,
        args.epochs,
        args.lr,
        args.device,
        model_path,
        mean,
        std,
        **loss_weights,
        reg_warmup_epochs=args.reg_warmup_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    # HPO Mode: Output final loss for agent parsing
    if args.hpo_mode:
        print(f"FINAL_LOSS: {best_val_loss:.6f}")

    return best_val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)