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
import math
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split # Add random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from scipy.spatial.distance import cdist

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
        # self.log_var_phy = nn.Parameter(torch.tensor(0.0, device=device)) # Physics Consistency Loss weight (commented out)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        reg_weight: float,
        model_name: str,
        target_phy_weight: Optional[float] = None,
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
             losses["pos"] = mse_pos.item()
             losses["w_pos"] = loss_pos.item()

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
            losses["vel"] = mse_vel.item()
            losses["w_vel"] = loss_vel.item()

            # 2. Cosine Similarity Loss for Velocity Direction
            # Loss = 1 - cosine_similarity. Range [0, 2].
            cos_sim = F.cosine_similarity(pred_total_vel_w, batch["gt_vel_w"], dim=-1, eps=1e-6)
            per_element_cos_loss = 1.0 - cos_sim
            mean_cos_loss = (per_element_cos_loss * mask).sum() / valid_element_count
            precision_cos = torch.exp(-self.log_var_cos)
            loss_cos = precision_cos * mean_cos_loss + self.log_var_cos
            total_loss += loss_cos
            losses["cos"] = mean_cos_loss.item()
            losses["w_cos"] = loss_cos.item()

            # --- Kinematic Consistency Loss (Physics Loss) ---
            # This loss term was found to be detrimental, as it penalizes acceleration
            # which is critical for handwriting strokes. It also effectively fights
            # the ESKF's corrections.
            # The kinematic consistency is already implicitly handled by the ESKF's
            # integration step.
            # dt = Config.DT
            # Calculate numerical derivative of position (batch, seq-1, 3)
            # pred_vel_from_pos = (pred_pos_w[:, 1:] - pred_pos_w[:, :-1]) / dt

            # Clip the numerical derivative to prevent massive spikes/exploding gradients
            # from 1/dt scaling on noisy predictions, especially early in training.
            # +/- 20.0 m/s is a safe upper bound for handwriting.
            # pred_vel_from_pos = torch.clamp(pred_vel_from_pos, -20.0, 20.0)

            # Align predicted velocity (batch, seq-1, 3)
            # Euler: p_{t+1} = p_t + v_t * dt. So v_t ~ (p_{t+1}-p_t)/dt.
            # We take 0..L-2 to match the diff length.
            # pred_vel_ref = pred_total_vel_w[:, :-1]

            # Adjust mask for the shortened sequence
            # mask_phy = mask_3d[:, :-1]
            # valid_phy_count = mask_phy.sum() + 1e-8

            # per_element_phy_loss = self.vel_criterion(pred_vel_from_pos, pred_vel_ref)
            # mse_phy = (per_element_phy_loss * mask_phy).sum() / valid_phy_count

            # --- Warm-up / Forced Schedule Logic ---
            # if target_phy_weight is not None:
            #     # Force the weight (precision) to the target value.
            #     # Update the learnable parameter so the optimizer picks up from here.
            #     # precision = exp(-log_var)  =>  log_var = -log(precision)
            #     forced_log_var = -math.log(target_phy_weight + 1e-8)
            #     self.log_var_phy.data.fill_(forced_log_var)

            #     # Use the forced values for calculation
            #     precision_phy = torch.tensor(target_phy_weight, device=self.log_var_phy.device)
            #     log_var_phy_val = torch.tensor(forced_log_var, device=self.log_var_phy.device)

            #     loss_phy = precision_phy * mse_phy + log_var_phy_val
            # else:
            #     # Standard learned uncertainty weighting
            #     precision_phy = torch.exp(-self.log_var_phy)
            #     loss_phy = precision_phy * mse_phy + self.log_var_phy

            # total_loss += loss_phy
            # losses["phy"] = mse_phy.item() # Keep for logging historical values if desired, otherwise comment out.


            # --- ZUPT Loss (Conditional) ---
            if "pred_zupt_prob" in model_out and model_out["pred_zupt_prob"] is not None:
                per_element_zupt_loss = self.zupt_criterion(model_out["pred_zupt_prob"], batch["gt_zupt"])
                bce_zupt = (per_element_zupt_loss * mask_3d).sum() / valid_element_count
                precision_zupt = torch.exp(-self.log_var_zupt)
                loss_zupt = precision_zupt * bce_zupt + self.log_var_zupt
                total_loss += loss_zupt
                losses["zupt"] = bce_zupt.item()
                losses["w_zupt"] = loss_zupt.item()

            # --- Regularization Loss (fixed weight) ---
            masked_vel_resid = pred_vel_resid_b * mask_3d
            loss_reg = torch.norm(masked_vel_resid)
            weighted_reg_loss = reg_weight * loss_reg
            total_loss += weighted_reg_loss
            losses["reg"] = loss_reg.item()
            losses["w_reg"] = weighted_reg_loss.item()

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
                variance = F.softplus(pred_R_diag_valid) + 1e-4
                nll_elementwise = 0.5 * (torch.square(innovation_valid) / variance + torch.log(variance))
                nll_cov = torch.mean(torch.sum(nll_elementwise, dim=-1))

            precision_cov = torch.exp(-self.log_var_cov)
            loss_cov = precision_cov * nll_cov + self.log_var_cov
            total_loss += loss_cov
            losses["cov"] = nll_cov.item()
            losses["w_cov"] = loss_cov.item()

        return total_loss, losses


def calculate_metrics(gt_pos: np.ndarray, pred_pos: np.ndarray) -> Tuple[float, float]:
    """Calculates ATE (RMSE after alignment) and DTW distance."""
    # Alignment (Umeyama)
    gt_mean = np.mean(gt_pos, axis=0)
    pred_mean = np.mean(pred_pos, axis=0)
    gt_centered = gt_pos - gt_mean
    pred_centered = pred_pos - pred_mean

    gt_std = np.linalg.norm(gt_centered)
    pred_std = np.linalg.norm(pred_centered)
    scale = gt_std / pred_std if pred_std > 1e-6 else 1.0

    pred_scaled = pred_centered * scale
    H = np.dot(pred_scaled.T, gt_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    pred_aligned = np.dot(pred_scaled, R.T)

    # ATE (RMSE)
    error = np.linalg.norm(gt_centered - pred_aligned, axis=1)
    ate = np.sqrt(np.mean(error**2))

    # DTW
    # Use cdist for pairwise distances
    dist_matrix = cdist(gt_centered, pred_aligned, metric='euclidean')
    n, m = len(gt_centered), len(pred_aligned)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Dynamic programming for DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    normalized_dtw = dtw[n, m] / (n + m)

    return float(ate) * 100, float(normalized_dtw)


def train(
    model_name: str,
    model: nn.Module,
    train_dataloader: DataLoader, # Modified to train_dataloader
    val_dataloader: DataLoader,   # Added val_dataloader
    epochs: int,
    lr: float,
    device: str,
    model_path: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    reg_weight: float = 0.0,
    warmup_epochs: int = 10, # Added warmup_epochs argument
) -> None:
    """Runs the training loop for the specified model."""

    criterion: nn.Module = UncertaintyLoss(device=device)

    # Ensure the learnable loss parameters are included in the optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    # Calculate total_steps for OneCycleLR
    total_steps = epochs * len(train_dataloader)

    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, div_factor=25)

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path))

    model.to(device)

    # Initialize history dictionaries for both training and validation
    train_history: Dict[str, List[float]] = {"total": [], "pos": [], "vel": [], "cos": [], "zupt": [], "reg": [], "cov": [], "lr": []}


    print(f"Start Training for {model_name} on {device}...")

    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    for epoch in range(epochs):
        # --- Calculate Physics Warm-up Weight ---
        target_phy_weight: Optional[float] = None
        if epoch < warmup_epochs:
            # Linear ramp from 1e-4 to 1.0
            # Start small to avoid shock, end at 1.0 (strong enforcement)
            progress = epoch / float(warmup_epochs)
            target_phy_weight = 1e-4 + (1.0 - 1e-4) * progress
        else:
            # Let the model learn the weight via uncertainty
            target_phy_weight = None

        # --- Training Loop ---
        model.train() # Set model to training mode
        epoch_train_losses: Dict[str, float] = {k: 0.0 for k in train_history.keys() if k != "lr"}
        # Add weighted loss keys for tracking
        for k in ["pos", "vel", "cos", "zupt", "reg", "cov"]:
            epoch_train_losses[f"w_{k}"] = 0.0

        pbar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar_train:
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            gt_vel_w = batch["gt_vel_w"].to(device)
            seq_lens = batch["len"].to(device)

            max_len = sensor_raw.shape[1]
            mask = torch.arange(max_len, device=device)[None, :] < seq_lens[:, None]

            gt_vel_norm = torch.norm(gt_vel_w, dim=-1, keepdim=True)
            gt_zupt = (gt_vel_norm < 0.01).float()

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
                reg_weight=reg_weight,
                model_name=model_name,
                target_phy_weight=target_phy_weight # Pass target weight
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
                sub_losses["reg"] = sub_losses.get("reg", 0.0) + pen_tip_loss.item()
                sub_losses["w_reg"] = sub_losses.get("w_reg", 0.0) + weighted_pen_tip.item()


            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() # Step the scheduler after each batch

            epoch_train_losses["total"] += loss.item() if not torch.isnan(loss) else 0
            pbar_train_postfix: Dict[str, float] = {"L": loss.item()}
            for k, v in sub_losses.items():
                if k in epoch_train_losses:
                    epoch_train_losses[k] += v if not np.isnan(v) else 0
                if not k.startswith("w_"): # Only show raw losses in progress bar
                    pbar_train_postfix[f"L_{k}"] = v
            pbar_train.set_postfix(pbar_train_postfix)

        for k in train_history.keys():
            if k in epoch_train_losses:
                train_history[k].append(epoch_train_losses[k] / len(train_dataloader))

        # Log Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history["lr"].append(current_lr)

        # Calculate weights for logging
        with torch.no_grad():
            w_pos = torch.exp(-criterion.log_var_pos).item()
            w_vel = torch.exp(-criterion.log_var_vel).item()
            w_cos = torch.exp(-criterion.log_var_cos).item()
            # w_phy = torch.exp(-criterion.log_var_phy).item() # Commented out
            w_zupt = torch.exp(-criterion.log_var_zupt).item()
            w_cov = torch.exp(-criterion.log_var_cov).item()

        log_str = f"Epoch {epoch+1}: Train_Total={train_history['total'][-1]:.4f} | LR={current_lr:.2e}"

        total_sum = epoch_train_losses["total"]
        def get_pct(key):
            val = epoch_train_losses.get(f"w_{key}", 0.0)
            return (val / total_sum) * 100.0 if total_sum > 0 else 0.0

        if model_name != "only_tcn":
            # log_str += f" | Train_Pos={train_history['pos'][-1]:.4f}" # Commented out as pos_loss is removed
            log_str += f" | Train_Vel={train_history['vel'][-1]:.4f} ({get_pct('vel'):.1f}%)"
            log_str += f" | Train_Cos={train_history['cos'][-1]:.4f} ({get_pct('cos'):.1f}%)"
            # log_str += f" | Train_Phy={train_history['phy'][-1]:.4f}" # Commented out as phy_loss is removed
            log_str += f"\n    Weights: Vel={w_vel:.3f} | Cos={w_cos:.3f} | ZUPT={w_zupt:.3f} ({get_pct('zupt'):.1f}%) | Cov={w_cov:.3f} ({get_pct('cov'):.1f}%) | Reg=({get_pct('reg'):.1f}%)"
        else:
            log_str += f" | Weight_Pos={w_pos:.3f} ({get_pct('pos'):.1f}%)"
        print(log_str)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

    # --- Final Metric Evaluation (ATE, DTW) ---
    print("\n--- Computing Final Metrics on Validation Set ---")
    model.eval()
    ate_list = []
    dtw_list = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating Metrics"):
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            seq_lens = batch["len"].to(device)
            sensor_norm = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            model_output = model(sensor_raw, sensor_norm)

            if model_name == "only_tcn":
                pred_pos_w = model_output
            else:
                pred_pos_w = model_output["pred_pos_w"]

            # Calculate metrics for each sample in the batch
            for i in range(len(seq_lens)):
                length = int(seq_lens[i].item())
                gt_traj = gt_pos_w[i, :length].cpu().numpy()
                pred_traj = pred_pos_w[i, :length].cpu().numpy()

                ate, dtw = calculate_metrics(gt_traj, pred_traj)
                ate_list.append(ate)
                dtw_list.append(dtw)

    print(f"\nFinal Validation Results:")
    print(f"Mean ATE (RMSE): {np.mean(ate_list):.4f} m")
    print(f"Mean DTW: {np.mean(dtw_list):.4f}")


def main() -> None:
    """Main function to parse arguments, set up training, and start the training process."""
    parser = argparse.ArgumentParser(description="Train various trajectory estimation models.")
    parser.add_argument("--model", type=str, default="eskf_tcn", choices=["eskf_tcn", "aekf_tcn", "only_tcn"], help="Type of model to train.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of epochs for physics loss warmup.") # Added arg
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="mps" if torch.cuda.is_available() else "cpu", help="Computation device ('cpu', 'cuda', 'mps').")
    args, _ = parser.parse_known_args()

    # Note: Most loss weights are now learned automatically. Only `reg_weight` remains.
    model_configs: Dict[str, Dict[str, Any]] = {
        "eskf_tcn": {
            "model_params": {
                "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
                "use_zupt": Config.ESKFTCN.USE_ZUPT,
                "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT
            },
            "loss_weights": {"reg_weight": Config.LOSS.REG_WEIGHT_ESKF_TCN},
        },
        "aekf_tcn": {
            "model_params": {"tcn_input_size": Config.AEKFTCN.TCN_INPUT_SIZE},
            "loss_weights": {"reg_weight": Config.LOSS.REG_WEIGHT_AEKF_TCN},
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

    # Load Training Dataset
    train_dataset = TrajectoryDataset(
        preprocessed_file=Config.DATASET_H5_PATH,
        augment_multiplier=Config.AUGMENT_MULTIPLIER,
        subsample_step=Config.SUBSAMPLE_STEP,
        do_augment=Config.DO_AUGMENT,
    )

    # Load Validation Dataset (No augmentation)
    val_dataset = TrajectoryDataset(
        preprocessed_file=Config.VALIDATION_DATASET_H5_PATH,
        augment_multiplier=1,
        subsample_step=Config.SUBSAMPLE_STEP,
        do_augment=False,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    with h5py.File(Config.SCALER_STATS_H5_PATH, "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float()
        std = torch.from_numpy(f["std"][:]).float()

    model_class_map = {
        "eskf_tcn": ESKFTCN_model,
        "aekf_tcn": AEKFTCN_model,
        "only_tcn": OnlyTCN,
    }
    model_class = model_class_map.get(args.model)
    model = model_class(device=args.device, dt=Config.DT, **model_params)

    model_path = f"{args.model}_model.pth"

    train(args.model, model, train_dataloader, val_dataloader, args.epochs, args.lr, args.device, model_path, mean, std, **loss_weights, warmup_epochs=args.warmup_epochs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)
