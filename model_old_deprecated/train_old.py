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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from model.qat_tcn import QATScheduler, get_qat_state, estimate_quantization_error


# =============================================================================
# PCGrad: Projecting Conflicting Gradients (Yu et al., 2020)
# https://arxiv.org/abs/2001.06782
# =============================================================================

def pcgrad_project(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Apply PCGrad to a list of task gradients.

    When two task gradients conflict (negative cosine similarity), project one
    onto the normal plane of the other to remove the conflicting component.

    Args:
        grads: List of flattened gradient tensors, one per task

    Returns:
        List of projected gradient tensors (same shape as input)
    """
    num_tasks = len(grads)
    if num_tasks == 0:
        return grads

    projected = [g.clone() for g in grads]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                # Compute projection coefficient
                g_j_norm_sq = grads[j].norm() ** 2 + 1e-8
                dot_product = projected[i] @ grads[j]

                # Only project if conflicting (negative cosine similarity)
                if dot_product < 0:
                    # Remove the conflicting component
                    projected[i] = projected[i] - (dot_product / g_j_norm_sq) * grads[j]

    return projected


def apply_pcgrad(
    model: nn.Module,
    losses: Dict[str, torch.Tensor],
    loss_keys: List[str],
    normalize_grads: bool = True,
) -> torch.Tensor:
    """
    Compute PCGrad-adjusted gradients for multiple task losses.

    Args:
        model: The neural network model
        losses: Dictionary containing loss tensors (keys prefixed with '_' for grad-enabled losses)
        loss_keys: List of loss keys to include in PCGrad (e.g., ['vel', 'cos', 'zupt', 'cov'])
        normalize_grads: If True, normalize gradient magnitudes before PCGrad to balance task contributions

    Returns:
        Combined gradient vector after PCGrad projection
    """
    grad_vectors = []
    grad_norms = []
    valid_keys = []

    # Collect gradients for each loss
    for key in loss_keys:
        grad_key = f"_{key}_grad"
        if grad_key in losses and losses[grad_key] is not None:
            loss_tensor = losses[grad_key]
            if not loss_tensor.requires_grad:
                continue

            model.zero_grad()
            loss_tensor.backward(retain_graph=True)

            # Flatten all parameter gradients into a single vector
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())

            if grads:
                grad_vec = torch.cat(grads)
                grad_vectors.append(grad_vec)
                grad_norms.append(grad_vec.norm().item())
                valid_keys.append(key)

    if not grad_vectors:
        return None

    # Optional: Normalize gradients to balance task contributions
    if normalize_grads and len(grad_vectors) > 1:
        target_norm = sum(grad_norms) / len(grad_norms)  # Average norm
        for i in range(len(grad_vectors)):
            if grad_norms[i] > 1e-8:
                scale = target_norm / grad_norms[i]
                # Clamp scale to avoid extreme adjustments
                scale = max(0.1, min(10.0, scale))
                grad_vectors[i] = grad_vectors[i] * scale

    # Apply PCGrad projection
    projected_grads = pcgrad_project(grad_vectors)

    # Sum projected gradients
    final_grad = sum(projected_grads)

    return final_grad


def set_model_gradients(model: nn.Module, flat_grad: torch.Tensor) -> None:
    """
    Set model parameter gradients from a flattened gradient vector.

    Args:
        model: The neural network model
        flat_grad: Flattened gradient vector
    """
    idx = 0
    for p in model.parameters():
        if p.grad is not None:
            numel = p.grad.numel()
            p.grad.copy_(flat_grad[idx:idx + numel].view_as(p.grad))
            idx += numel


class UncertaintyLoss(nn.Module):
    """
    Calculates a hybrid loss by automatically weighting tasks based on homoscedastic
    uncertainty, as described in "Multi-Task Learning Using Uncertainty to Weigh
    Losses..." by Kendall, Gal, and Cipolla.

    This approach avoids manual tuning of loss weights by making them learnable
    parameters of the model.

    Supports warmup: In early epochs, uncertainty weighting is disabled (fixed equal weights).
    The warmup factor linearly increases from 0 to 1 over `warmup_epochs`.
    """

    def __init__(self, device: str, warmup_epochs: int = 0):
        """Initializes the UncertaintyLoss module.

        Args:
            device: Computation device.
            warmup_epochs: Number of epochs to warm up uncertainty weighting.
                          During warmup, uses fixed equal weights. Set to 0 to disable.
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs

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
        self.log_var_gravity = nn.Parameter(torch.tensor(0.0, device=device))  # Gravity direction loss weight

    def get_warmup_factor(self, epoch: int) -> float:
        """Compute the warmup factor for uncertainty weighting.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Warmup factor in [0, 1]. 0 = fixed weights, 1 = full uncertainty weighting.
        """
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    def _compute_context_aware_precision(
        self,
        base_log_var: torch.Tensor,
        gate: torch.Tensor,
        min_scale: float = 0.3,
    ) -> torch.Tensor:
        """Compute context-aware per-timestep precision.

        Args:
            base_log_var: Learnable log-variance (scalar, clamped).
            gate: Motion gate from adapter_weight [B, T]. High=stable, Low=agile.
            min_scale: Minimum precision scale for agile motion (default 0.3).

        Returns:
            precision_t: Per-timestep precision [B, T].
                - Stable (gate≈1): base_precision * 1.0 (high confidence, strict)
                - Agile (gate≈0): base_precision * min_scale (low confidence, tolerant)
        """
        base_precision = torch.exp(-base_log_var)  # scalar
        precision_scale = min_scale + (1.0 - min_scale) * gate  # [B, T], range [min_scale, 1.0]
        return base_precision * precision_scale  # [B, T]

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        reg_weight: float,
        model_name: str,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss with context-aware per-timestep precision weighting.

        Instead of global uncertainty weighting (same weight for all timesteps),
        this uses motion-adaptive precision based on adapter_weight (jerk-based gate):
        - Stable motion (gate≈1): High precision → strict enforcement
        - Agile motion (gate≈0): Low precision → tolerant enforcement

        The Bayesian interpretation: uncertainty should be higher during rapid motion.

        Args:
            epoch: Current epoch (0-indexed) for warmup scheduling.
        """
        # --- Common Preparations ---
        mask_3d = mask.unsqueeze(-1)
        valid_element_count = mask.sum() + 1e-8
        losses = {}
        total_loss = torch.tensor(0.0, device=mask.device)

        # Compute warmup factor for uncertainty weighting
        warmup_factor = self.get_warmup_factor(epoch)

        # Extract motion gate once (if available)
        has_gate = "adapter_weight" in model_out
        gate = model_out["adapter_weight"] if has_gate else None  # [B, T]

        # Clamp log_var to prevent precision explosion (min) and vanishing gradients (max)
        # min=-4.0 limits precision to exp(4)≈54.6, max=2.0 limits precision to exp(-2)≈0.14
        limit_var_pos = torch.clamp(self.log_var_pos, min=-4.0, max=2.0)
        limit_var_vel = torch.clamp(self.log_var_vel, min=-4.0, max=2.0)
        limit_var_cos = torch.clamp(self.log_var_cos, min=-4.0, max=2.0)
        limit_var_zupt = torch.clamp(self.log_var_zupt, min=-4.0, max=2.0)
        limit_var_cov = torch.clamp(self.log_var_cov, min=-4.0, max=2.0)
        limit_var_gravity = torch.clamp(self.log_var_gravity, min=-4.0, max=2.0)

        # --- Position Loss (only_tcn) ---
        if model_name == "only_tcn":
            pred_pos_w = model_out
            per_element_pos_loss = self.pos_criterion(pred_pos_w, batch["gt_pos_w"])
            raw_pos_loss = (per_element_pos_loss * mask_3d).sum() / valid_element_count

            precision_pos = torch.exp(-limit_var_pos)
            loss_pos = (1 - warmup_factor) * raw_pos_loss + warmup_factor * (precision_pos * raw_pos_loss + limit_var_pos)
            total_loss += loss_pos
            losses["pos"] = raw_pos_loss.detach()
            losses["w_pos"] = loss_pos.detach()

        # --- Hybrid Model Losses (ESKF-TCN, AEKF-TCN) ---
        if model_name != "only_tcn":
            filter_vel_w = model_out["filter_vel_w"]
            pred_vel_resid_b = model_out["pred_vel_resid_b"]
            filter_quat = model_out["filter_quat"]
            rot_mat_b_to_w = quaternion_to_rotation_matrix(filter_quat.view(-1, 4)).view(*filter_quat.shape[:-1], 3, 3)
            pred_vel_resid_w = (rot_mat_b_to_w @ pred_vel_resid_b.unsqueeze(-1)).squeeze(-1)
            pred_total_vel_w = filter_vel_w + pred_vel_resid_w

            # ===== 1. Velocity Loss (Context-Aware Precision) =====
            # Mutual exclusivity: Apply vel loss only during clear motion to avoid conflict with ZUPT
            gt_vel_norm = torch.norm(batch["gt_vel_w"], dim=-1, keepdim=True)  # [B, T, 1]
            vel_moving_threshold = 0.01  # 1 cm/s - clear motion threshold
            vel_moving_mask = (gt_vel_norm > vel_moving_threshold).float()  # [B, T, 1]
            vel_mask_3d = mask_3d * vel_moving_mask
            vel_valid_count = vel_mask_3d.sum() + 1e-8

            per_element_vel_loss = self.vel_criterion(pred_total_vel_w, batch["gt_vel_w"])  # [B, T, 3]
            raw_vel_loss = (per_element_vel_loss * vel_mask_3d).sum() / vel_valid_count

            if has_gate:
                # Context-aware: per-timestep precision weighted loss (with mutual exclusivity mask)
                precision_vel_t = self._compute_context_aware_precision(limit_var_vel, gate, min_scale=0.3)  # [B, T]
                weighted_vel_loss = (precision_vel_t.unsqueeze(-1) * per_element_vel_loss * vel_mask_3d).sum() / vel_valid_count
                loss_vel = (1 - warmup_factor) * raw_vel_loss + warmup_factor * (weighted_vel_loss + limit_var_vel)
            else:
                # Fallback: global precision
                precision_vel = torch.exp(-limit_var_vel)
                loss_vel = (1 - warmup_factor) * raw_vel_loss + warmup_factor * (precision_vel * raw_vel_loss + limit_var_vel)

            total_loss += loss_vel
            losses["vel"] = raw_vel_loss.detach()
            losses["w_vel"] = loss_vel.detach()
            # Store non-detached for PCGrad
            losses["_vel_grad"] = loss_vel

            # ===== 2. Cosine Similarity Loss (Context-Aware Precision) =====
            cos_sim = F.cosine_similarity(pred_total_vel_w, batch["gt_vel_w"], dim=-1, eps=1e-6)
            per_element_cos_loss = 1.0 - cos_sim  # [B, T]
            moving_mask = mask * (1.0 - batch["gt_zupt"].squeeze(-1))
            moving_count = moving_mask.sum() + 1e-8
            raw_cos_loss = (per_element_cos_loss * moving_mask).sum() / moving_count

            if has_gate:
                # Context-aware: stricter direction accuracy during stable motion
                precision_cos_t = self._compute_context_aware_precision(limit_var_cos, gate, min_scale=0.3)  # [B, T]
                weighted_cos_loss = (precision_cos_t * per_element_cos_loss * moving_mask).sum() / moving_count
                loss_cos = (1 - warmup_factor) * raw_cos_loss + warmup_factor * (weighted_cos_loss + limit_var_cos)
            else:
                precision_cos = torch.exp(-limit_var_cos)
                loss_cos = (1 - warmup_factor) * raw_cos_loss + warmup_factor * (precision_cos * raw_cos_loss + limit_var_cos)

            total_loss += loss_cos
            losses["cos"] = raw_cos_loss.detach()
            losses["w_cos"] = loss_cos.detach()
            # Store non-detached for PCGrad
            losses["_cos_grad"] = loss_cos

            # ===== 3. ZUPT Loss (Context-Aware Precision + Transition Weighting) =====
            if "pred_zupt_prob" in model_out and model_out["pred_zupt_prob"] is not None:
                per_element_zupt_loss = self.zupt_criterion(model_out["pred_zupt_prob"], batch["gt_zupt"])  # [B, T, 1]

                # Transition weighting: emphasize state changes
                gt_zupt_squeezed = batch["gt_zupt"].squeeze(-1)
                zupt_diff = torch.abs(gt_zupt_squeezed[:, 1:] - gt_zupt_squeezed[:, :-1])
                transition_mask = F.pad(zupt_diff, (1, 0), value=0.0)
                transition_weight = 5.0
                sample_weights = (1.0 + transition_weight * transition_mask).unsqueeze(-1)  # [B, T, 1]

                weighted_mask = mask_3d * sample_weights
                weighted_count = weighted_mask.sum() + 1e-8
                raw_zupt_loss = (per_element_zupt_loss * weighted_mask).sum() / weighted_count

                if has_gate:
                    # Context-aware: ZUPT more important during stable motion
                    precision_zupt_t = self._compute_context_aware_precision(limit_var_zupt, gate, min_scale=0.5)  # [B, T]
                    precision_zupt_3d = precision_zupt_t.unsqueeze(-1)  # [B, T, 1]
                    weighted_zupt_loss = (precision_zupt_3d * per_element_zupt_loss * weighted_mask).sum() / weighted_count
                    loss_zupt = (1 - warmup_factor) * raw_zupt_loss + warmup_factor * (weighted_zupt_loss + limit_var_zupt)
                else:
                    precision_zupt = torch.exp(-limit_var_zupt)
                    loss_zupt = (1 - warmup_factor) * raw_zupt_loss + warmup_factor * (precision_zupt * raw_zupt_loss + limit_var_zupt)

                total_loss += loss_zupt
                losses["zupt"] = raw_zupt_loss.detach()
                losses["w_zupt"] = loss_zupt.detach()
                # Store non-detached for PCGrad
                losses["_zupt_grad"] = loss_zupt

            # ===== 4. Regularization Loss (Lyapunov-style, already context-aware) =====
            vel_corr_mag = torch.norm(pred_vel_resid_b, dim=-1, keepdim=True)  # [B, T, 1]

            if has_gate:
                gate_3d = gate.unsqueeze(-1)  # [B, T, 1]

                # Lyapunov thresholds (m/s)
                eps_strict = 0.02   # 2 cm/s for stable motion
                eps_relaxed = 0.15  # 15 cm/s for agile motion

                reg_strict = F.relu(vel_corr_mag - eps_strict) ** 2
                reg_relaxed = F.relu(vel_corr_mag - eps_relaxed) ** 2

                # Lyapunov blend: strict when stable (gate≈1), relaxed when agile (gate≈0)
                loss_reg_elementwise = gate_3d * reg_strict + (1 - gate_3d) * reg_relaxed
                loss_reg = (loss_reg_elementwise * mask_3d).sum() / (mask_3d.sum() + 1e-8)
            else:
                masked_vel_resid = pred_vel_resid_b * mask_3d
                loss_reg = (masked_vel_resid ** 2).sum() / (mask_3d.sum() + 1e-8)

            weighted_reg_loss = reg_weight * loss_reg
            total_loss += weighted_reg_loss
            losses["reg"] = loss_reg.detach()
            losses["w_reg"] = weighted_reg_loss.detach()
            # Store non-detached for PCGrad
            losses["_reg_grad"] = weighted_reg_loss

        # --- Probabilistic Model Loss (ESKF-TCN) ---
        if model_name == "eskf_tcn":
            innovation = model_out["filter_innovation"]
            pred_R_diag = model_out["pred_covariance_R"]
            tcn_output_mask = model_out["tcn_output_mask"]
            final_cov_mask = tcn_output_mask & mask

            # ===== 5. Covariance NLL Loss (Context-Aware Precision) =====
            nll_cov = torch.tensor(0.0, device=innovation.device)
            weighted_nll_cov = torch.tensor(0.0, device=innovation.device)

            if final_cov_mask.any():
                innovation_valid = innovation[final_cov_mask]  # [N, 6]
                pred_R_diag_valid = pred_R_diag[final_cov_mask]  # [N, 6]

                variance = F.softplus(pred_R_diag_valid) + 1e-4
                variance = torch.clamp(variance, min=1e-4, max=3.0)
                nll_elementwise = 0.5 * (torch.square(innovation_valid) / variance + torch.log(variance))
                nll_per_sample = torch.sum(nll_elementwise, dim=-1)  # [N]
                nll_cov = nll_per_sample.mean()

                if has_gate:
                    # Context-aware: extract gate values for valid positions
                    gate_valid = gate[final_cov_mask]  # [N]
                    precision_cov_t = self._compute_context_aware_precision(limit_var_cov, gate_valid, min_scale=0.3)  # [N]
                    weighted_nll_cov = (precision_cov_t * nll_per_sample).mean()

            if has_gate and final_cov_mask.any():
                loss_cov = (1 - warmup_factor) * nll_cov + warmup_factor * (weighted_nll_cov + limit_var_cov)
            else:
                precision_cov = torch.exp(-limit_var_cov)
                loss_cov = (1 - warmup_factor) * nll_cov + warmup_factor * (precision_cov * nll_cov + limit_var_cov)

            total_loss += loss_cov
            losses["cov"] = nll_cov.detach()
            losses["w_cov"] = loss_cov.detach()
            # Store non-detached for PCGrad
            losses["_cov_grad"] = loss_cov

            # ===== 6. Gravity Direction Loss (Context-Aware Precision) =====
            if "pred_gravity_b" in model_out:
                pred_gravity_b = model_out["pred_gravity_b"]

                if "gt_gravity_b" in batch:
                    gt_gravity_b = batch["gt_gravity_b"]
                    gravity_mask = final_cov_mask
                else:
                    imu_raw = batch["imu_seq_raw"]
                    accel_raw = imu_raw[:, :, :3]
                    accel_norm = torch.norm(accel_raw, dim=-1, keepdim=True)
                    gt_gravity_b = accel_raw / (accel_norm + 1e-6)
                    accel_valid = (accel_norm.squeeze(-1) > 8.0) & (accel_norm.squeeze(-1) < 12.0)
                    gravity_mask = final_cov_mask & accel_valid

                gravity_loss = torch.tensor(0.0, device=pred_gravity_b.device)
                weighted_gravity_loss = torch.tensor(0.0, device=pred_gravity_b.device)

                if gravity_mask.any():
                    pred_g_valid = pred_gravity_b[gravity_mask]
                    gt_g_valid = gt_gravity_b[gravity_mask]
                    cos_sim_gravity = F.cosine_similarity(pred_g_valid, gt_g_valid, dim=-1, eps=1e-6)
                    per_element_gravity_loss = 1.0 - cos_sim_gravity  # [N]
                    gravity_loss = per_element_gravity_loss.mean()

                    if has_gate:
                        gate_valid = gate[gravity_mask]  # [N]
                        precision_gravity_t = self._compute_context_aware_precision(limit_var_gravity, gate_valid, min_scale=0.3)
                        weighted_gravity_loss = (precision_gravity_t * per_element_gravity_loss).mean()

                if has_gate and gravity_mask.any():
                    loss_gravity = (1 - warmup_factor) * gravity_loss + warmup_factor * (weighted_gravity_loss + limit_var_gravity)
                else:
                    precision_gravity = torch.exp(-limit_var_gravity)
                    loss_gravity = (1 - warmup_factor) * gravity_loss + warmup_factor * (precision_gravity * gravity_loss + limit_var_gravity)

                total_loss += loss_gravity
                losses["gravity"] = gravity_loss.detach()
                losses["w_gravity"] = loss_gravity.detach()

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
    epoch: int = 999999,
) -> Dict[str, float]:
    """Evaluates the model on the validation set.

    Args:
        epoch: Current epoch for uncertainty warmup. Defaults to large value
               to always use full uncertainty weighting during validation.
    """
    model.eval()
    # Set epoch for warmup scheduling (validation uses full model behavior)
    if hasattr(model, 'set_current_epoch'):
        model.set_current_epoch(epoch)
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
                "imu_seq_raw": sensor_raw,  # For gravity loss fallback (accelerometer)
            }

            # Add pencil-derived gravity GT if available in dataset
            if "gt_gravity_b" in batch:
                batch_gpu["gt_gravity_b"] = batch["gt_gravity_b"].to(device)

            loss, sub_losses = criterion(
                model_out=model_output,
                batch=batch_gpu,
                mask=mask,
                reg_weight=reg_weight,
                model_name=model_name,
                epoch=epoch,
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
    uncertainty_warmup_epochs: int = 0,
    patience: int = 20,
    min_delta: float = 1e-4,
    qat_enabled: bool = False,
    qat_start_epoch: int = 10,
    qat_backend: str = "qnnpack",
) -> float:
    """Runs the training loop for the specified model with early stopping support.

    Args:
        uncertainty_warmup_epochs: Number of epochs to warm up uncertainty weighting.
                                   During warmup, uses fixed equal weights and gradually
                                   transitions to learned uncertainty weights.
                                   Set to 0 to disable warmup (use full uncertainty from start).
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum change in validation loss to qualify as improvement.
        qat_enabled: Enable Quantization-Aware Training for INT8 deployment.
        qat_start_epoch: Epoch to start QAT (after FP32 warmup).
        qat_backend: QAT backend ("qnnpack" for ARM/ESP32, "fbgemm" for x86).
    """

    # Initialize TensorBoard SummaryWriter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"{model_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging enabled: {log_dir}")

    criterion: nn.Module = UncertaintyLoss(device=device, warmup_epochs=uncertainty_warmup_epochs)

    # Ensure the learnable loss parameters are included in the optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    # CosineAnnealingLR: Smoothly decays learning rate following cosine curve
    # T_max = epochs for full cosine cycle, eta_min = minimum LR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,  # Minimum LR = 1% of initial LR
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

    # Initialize QAT scheduler
    qat_scheduler = QATScheduler(
        start_epoch=qat_start_epoch,
        backend=qat_backend,
        enabled=qat_enabled,
    )
    if qat_enabled:
        print(f"[QAT] Enabled - will activate at epoch {qat_start_epoch} with backend={qat_backend}")

    print(f"Start Training for {model_name} on {device}...")

    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    for epoch in range(start_epoch, epochs):
        # Check if QAT should be activated at this epoch
        model = qat_scheduler.step(model, epoch)
        # --- Training Loop ---
        model.train() # Set model to training mode
        # Set current epoch for warmup scheduling (e.g., gravity alignment warmup)
        if hasattr(model, 'set_current_epoch'):
            model.set_current_epoch(epoch)
        epoch_train_losses: Dict[str, float] = {k: 0.0 for k in train_history.keys() if k != "lr"}
        # Add weighted loss keys for tracking
        for k in ["pos", "vel", "cos", "zupt", "reg", "cov", "gravity"]:
            epoch_train_losses[f"w_{k}"] = 0.0

        # Track gradient norms for monitoring
        epoch_grad_norms: List[float] = []

        # [TEMP DEBUG] Track gradient conflicts between losses
        epoch_grad_conflicts: Dict[str, List[float]] = {
            "cos_vel_cos": [], "cos_vel_zupt": [], "cos_vel_reg": [], "cos_vel_cov": [],
            "cos_cos_zupt": [], "cos_cos_reg": [], "cos_cos_cov": [],
            "cos_zupt_reg": [], "cos_zupt_cov": [], "cos_reg_cov": [],
            "norm_vel": [], "norm_cos": [], "norm_zupt": [], "norm_reg": [], "norm_cov": [],
        }
        DEBUG_GRAD_CONFLICT = True  # Set to False to disable
        USE_PCGRAD = True  # Set to False to disable PCGrad
        PCGRAD_LOSS_KEYS = ["vel", "cos", "zupt", "cov", "reg"]  # Losses to include in PCGrad

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
                "imu_seq_raw": sensor_raw,  # For gravity loss fallback (accelerometer)
            }

            # Add pencil-derived gravity GT if available in dataset
            if "gt_gravity_b" in batch:
                batch_gpu["gt_gravity_b"] = batch["gt_gravity_b"].to(device)

            loss, sub_losses = criterion(
                model_out=model_output,
                batch=batch_gpu,
                mask=mask,
                reg_weight=reg_weight,
                model_name=model_name,
                epoch=epoch,
            )

            # Add Pen Tip Regularization Loss if available
            if hasattr(model, "get_pen_tip_regularization_loss"):
                pen_tip_loss = model.get_pen_tip_regularization_loss()
                # Weight it? Let's treat it as part of 'reg' loss or add it to total.
                # Since 'reg_weight' is passed to criterion for velocity reg,
                # we can use a small weight or just add it.
                # Let's add it to total loss and log it under 'reg' (accumulating).
                # Assuming reg_weight is appropriate for this too (it's usually small, e.g., 1e-4).
                weighted_pen_tip = reg_weight * pen_tip_loss
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

            # [TEMP DEBUG] Gradient conflict analysis between losses
            if DEBUG_GRAD_CONFLICT and model_name != "only_tcn":
                def get_grad_vector(loss_tensor: torch.Tensor) -> torch.Tensor:
                    """Compute flattened gradient vector for a loss."""
                    model.zero_grad()
                    loss_tensor.backward(retain_graph=True)
                    grads = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grads.append(p.grad.detach().flatten())
                    return torch.cat(grads) if grads else torch.tensor([0.0], device=device)

                def cosine_sim(g1: torch.Tensor, g2: torch.Tensor) -> float:
                    """Compute cosine similarity between two gradient vectors."""
                    norm1, norm2 = g1.norm(), g2.norm()
                    if norm1 < 1e-8 or norm2 < 1e-8:
                        return 0.0
                    return (g1 @ g2 / (norm1 * norm2)).item()

                try:
                    # Recompute individual losses with graph for gradient analysis
                    filter_vel_w = model_output["filter_vel_w"]
                    pred_vel_resid_b = model_output["pred_vel_resid_b"]
                    filter_quat = model_output["filter_quat"]
                    rot_mat = quaternion_to_rotation_matrix(filter_quat.view(-1, 4)).view(*filter_quat.shape[:-1], 3, 3)
                    pred_vel_resid_w = (rot_mat @ pred_vel_resid_b.unsqueeze(-1)).squeeze(-1)
                    pred_total_vel_w = filter_vel_w + pred_vel_resid_w

                    # Velocity loss
                    loss_vel_debug = F.smooth_l1_loss(pred_total_vel_w * mask.unsqueeze(-1), gt_vel_w * mask.unsqueeze(-1))

                    # Cosine loss
                    cos_sim_val = F.cosine_similarity(pred_total_vel_w, gt_vel_w, dim=-1, eps=1e-6)
                    loss_cos_debug = ((1.0 - cos_sim_val) * mask).sum() / (mask.sum() + 1e-8)

                    # ZUPT loss
                    loss_zupt_debug = None
                    if "pred_zupt_prob" in model_output and model_output["pred_zupt_prob"] is not None:
                        loss_zupt_debug = F.binary_cross_entropy_with_logits(
                            model_output["pred_zupt_prob"] * mask.unsqueeze(-1),
                            gt_zupt * mask.unsqueeze(-1),
                            reduction='mean'
                        )

                    # Reg loss
                    loss_reg_debug = (pred_vel_resid_b ** 2 * mask.unsqueeze(-1)).mean()

                    # Cov loss (NLL)
                    loss_cov_debug = None
                    if "filter_innovation" in model_output and "pred_covariance_R" in model_output:
                        innovation = model_output["filter_innovation"]
                        pred_R_diag = model_output["pred_covariance_R"]
                        tcn_mask = model_output.get("tcn_output_mask", mask)
                        cov_mask = tcn_mask & mask
                        if cov_mask.any():
                            inn_valid = innovation[cov_mask]
                            R_valid = pred_R_diag[cov_mask]
                            variance = F.softplus(R_valid) + 1e-4
                            variance = torch.clamp(variance, min=1e-4, max=3.0)
                            nll = 0.5 * (torch.square(inn_valid) / variance + torch.log(variance))
                            loss_cov_debug = nll.mean()

                    # Compute gradients
                    grad_vel = get_grad_vector(loss_vel_debug)
                    grad_cos = get_grad_vector(loss_cos_debug)
                    grad_zupt = get_grad_vector(loss_zupt_debug) if loss_zupt_debug is not None else None
                    grad_reg = get_grad_vector(loss_reg_debug)
                    grad_cov = get_grad_vector(loss_cov_debug) if loss_cov_debug is not None else None

                    # Compute cosine similarities
                    epoch_grad_conflicts["cos_vel_cos"].append(cosine_sim(grad_vel, grad_cos))
                    epoch_grad_conflicts["cos_vel_reg"].append(cosine_sim(grad_vel, grad_reg))
                    epoch_grad_conflicts["cos_cos_reg"].append(cosine_sim(grad_cos, grad_reg))
                    if grad_zupt is not None:
                        epoch_grad_conflicts["cos_vel_zupt"].append(cosine_sim(grad_vel, grad_zupt))
                        epoch_grad_conflicts["cos_cos_zupt"].append(cosine_sim(grad_cos, grad_zupt))
                        epoch_grad_conflicts["cos_zupt_reg"].append(cosine_sim(grad_zupt, grad_reg))
                    if grad_cov is not None:
                        epoch_grad_conflicts["cos_vel_cov"].append(cosine_sim(grad_vel, grad_cov))
                        epoch_grad_conflicts["cos_cos_cov"].append(cosine_sim(grad_cos, grad_cov))
                        epoch_grad_conflicts["cos_reg_cov"].append(cosine_sim(grad_reg, grad_cov))
                        if grad_zupt is not None:
                            epoch_grad_conflicts["cos_zupt_cov"].append(cosine_sim(grad_zupt, grad_cov))

                    # Compute norms
                    epoch_grad_conflicts["norm_vel"].append(grad_vel.norm().item())
                    epoch_grad_conflicts["norm_cos"].append(grad_cos.norm().item())
                    if grad_zupt is not None:
                        epoch_grad_conflicts["norm_zupt"].append(grad_zupt.norm().item())
                    epoch_grad_conflicts["norm_reg"].append(grad_reg.norm().item())
                    if grad_cov is not None:
                        epoch_grad_conflicts["norm_cov"].append(grad_cov.norm().item())

                except RuntimeError as e:
                    pass  # Skip if gradient computation fails

                # Clear grads before actual backward
                model.zero_grad()

            # ===== PCGrad or Standard Backward =====
            if USE_PCGRAD and model_name != "only_tcn":
                # Apply PCGrad: project conflicting gradients with gradient normalization
                pcgrad_result = apply_pcgrad(model, sub_losses, PCGRAD_LOSS_KEYS, normalize_grads=True)

                if pcgrad_result is not None:
                    # Set the PCGrad-adjusted gradients
                    model.zero_grad()
                    # Need to do one backward to initialize grad shapes
                    loss.backward(retain_graph=True)
                    # Then overwrite with PCGrad result
                    set_model_gradients(model, pcgrad_result)
                else:
                    # Fallback to standard backward if PCGrad fails
                    loss.backward()
            else:
                loss.backward()

            # Gradient clipping to prevent explosion (critical post-GroupNorm removal)
            # Returns the total norm of gradients before clipping for monitoring
            grad_clip_threshold = 3
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
        val_metrics = validate(model, val_dataloader, criterion, device, reg_weight, model_name, mean, std, epoch=epoch)

        # Step the scheduler (CosineAnnealingLR - epoch-based, no val_loss needed)
        scheduler.step()

        # --- TensorBoard Logging ---
        writer.add_scalar("Loss/Train_Total", train_history["total"][-1], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics["total"], epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Log uncertainty warmup factor
        uncertainty_warmup_factor = criterion.get_warmup_factor(epoch)
        writer.add_scalar("Uncertainty_Warmup_Factor", uncertainty_warmup_factor, epoch)

        # Log gradient norm (average over epoch)
        if epoch_grad_norms:
            avg_grad_norm = np.mean(epoch_grad_norms)
            max_grad_norm_epoch = np.max(epoch_grad_norms)
            writer.add_scalar("Gradients/Average_Norm", avg_grad_norm, epoch)
            writer.add_scalar("Gradients/Max_Norm", max_grad_norm_epoch, epoch)

        # [TEMP DEBUG] Print gradient conflicts
        if DEBUG_GRAD_CONFLICT and epoch_grad_conflicts["cos_vel_cos"]:
            avg_conflicts = {k: np.mean(v) if v else 0.0 for k, v in epoch_grad_conflicts.items()}

            # Print cosine similarities (negative = conflict)
            print(f"  [Grad Conflict] vel↔cos: {avg_conflicts['cos_vel_cos']:+.3f} | "
                  f"vel↔zupt: {avg_conflicts['cos_vel_zupt']:+.3f} | "
                  f"vel↔cov: {avg_conflicts['cos_vel_cov']:+.3f} | "
                  f"cos↔cov: {avg_conflicts['cos_cos_cov']:+.3f}")

            # Print norms
            print(f"  [Grad Norms] vel: {avg_conflicts['norm_vel']:.2e} | "
                  f"cos: {avg_conflicts['norm_cos']:.2e} | "
                  f"zupt: {avg_conflicts['norm_zupt']:.2e} | "
                  f"cov: {avg_conflicts['norm_cov']:.2e}")

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

        # Log QAT status and quantization error if active
        if qat_scheduler.is_active():
            writer.add_scalar("QAT/Active", 1.0, epoch)
            # Estimate quantization error periodically (every 5 epochs)
            if epoch % 5 == 0:
                try:
                    # Get a sample batch for error estimation
                    sample_batch = next(iter(train_dataloader))
                    sample_input = sample_batch["imu_seq_raw"][:1].to(device)
                    sample_norm = (sample_input - mean_gpu) / (std_gpu + 1e-6)
                    model.eval()
                    with torch.no_grad():
                        mean_err, per_output_err = estimate_quantization_error(model, sample_norm)
                    model.train()
                    writer.add_scalar("QAT/Mean_Quantization_Error", mean_err, epoch)
                    for output_name, err in per_output_err.items():
                        writer.add_scalar(f"QAT/Error_{output_name}", err, epoch)
                except Exception as e:
                    print(f"[QAT] Warning: Could not estimate quantization error: {e}")
        else:
            writer.add_scalar("QAT/Active", 0.0, epoch)

        log_str = f"Epoch {epoch+1}: Train_Total={train_history['total'][-1]:.4f} | Val_Total={val_metrics['total']:.4f} | LR={current_lr:.2e}"
        if uncertainty_warmup_factor < 1.0:
            log_str += f" | UncWarmup={uncertainty_warmup_factor:.2f}"

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
    parser.add_argument("--uncertainty_warmup_epochs", type=int, default=0, help="Number of epochs to warm up uncertainty weighting. During warmup, uses fixed equal weights and gradually transitions to learned uncertainty weights. Set to 0 to disable.")
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
    # QAT arguments
    parser.add_argument("--qat", action="store_true", help="Enable Quantization-Aware Training for INT8 deployment.")
    parser.add_argument("--qat_start_epoch", type=int, default=None, help="Epoch to start QAT (default: from Config).")
    parser.add_argument("--qat_backend", type=str, default=None, choices=["qnnpack", "fbgemm"], help="QAT backend (qnnpack for ARM/ESP32, fbgemm for x86).")
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

    model_configs: Dict[str, Dict[str, Any]] = {
        "eskf_tcn": {
            "model_params": {
                "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
                "use_zupt": Config.ESKFTCN.USE_ZUPT,
                "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT,
                "dropout": dropout_rate,
                "kernel_size": kernel_size,
                "tcn_channels": tcn_channels,
                "tcn_backbone_dilations": Config.ESKFTCN.TCN_BACKBONE_DILATIONS,
                "tcn_dynamic_dilations": Config.ESKFTCN.TCN_DYNAMIC_DILATIONS,
                "tcn_static_dilations": Config.ESKFTCN.TCN_STATIC_DILATIONS,
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

    # QAT parameters: use args if specified, else fall back to Config
    qat_enabled = args.qat if args.qat else Config.ESKFTCN.QAT_ENABLED
    qat_start_epoch = args.qat_start_epoch if args.qat_start_epoch is not None else Config.ESKFTCN.QAT_START_EPOCH
    qat_backend = args.qat_backend if args.qat_backend is not None else Config.ESKFTCN.QAT_BACKEND

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
        uncertainty_warmup_epochs=args.uncertainty_warmup_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        qat_enabled=qat_enabled,
        qat_start_epoch=qat_start_epoch,
        qat_backend=qat_backend,
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

    """
    QAT 진행해야함. ( 우선순위는 낮음 )
    Gravity_body의 GT라벨 생성 방법에 관해서, Apple Pencil Pro의 Pose (Azimuth/Altitude/Pose) 적극 사용 !
    전처리 및 회전 행렬 구현 필요함. ( 초기 정렬 !)
    """