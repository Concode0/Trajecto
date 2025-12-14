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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split # Add random_split
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
        self.log_var_pos = nn.Parameter(torch.tensor(0.0, device=device))
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

        # --- Position Loss (for all models) ---
        if model_name == "only_tcn":
             pred_pos_w = model_out
        else:
             pred_pos_w = model_out["pred_pos_w"]

        per_element_pos_loss = self.pos_criterion(pred_pos_w, batch["gt_pos_w"])
        mse_pos = (per_element_pos_loss * mask_3d).sum() / valid_element_count

        # Apply uncertainty weighting
        precision_pos = torch.exp(-self.log_var_pos)
        loss_pos = precision_pos * mse_pos + self.log_var_pos
        total_loss += loss_pos
        losses["pos"] = mse_pos.item() # Log the unweighted MSE for monitoring

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

            # 2. Cosine Similarity Loss for Velocity Direction
            # Loss = 1 - cosine_similarity. Range [0, 2].
            cos_sim = F.cosine_similarity(pred_total_vel_w, batch["gt_vel_w"], dim=-1, eps=1e-6)
            per_element_cos_loss = 1.0 - cos_sim
            mean_cos_loss = (per_element_cos_loss * mask).sum() / valid_element_count
            precision_cos = torch.exp(-self.log_var_cos)
            loss_cos = precision_cos * mean_cos_loss + self.log_var_cos
            total_loss += loss_cos
            losses["cos"] = mean_cos_loss.item()

            # --- ZUPT Loss (Conditional) ---
            if "pred_zupt_prob" in model_out and model_out["pred_zupt_prob"] is not None:
                per_element_zupt_loss = self.zupt_criterion(model_out["pred_zupt_prob"], batch["gt_zupt"])
                bce_zupt = (per_element_zupt_loss * mask_3d).sum() / valid_element_count
                precision_zupt = torch.exp(-self.log_var_zupt)
                loss_zupt = precision_zupt * bce_zupt + self.log_var_zupt
                total_loss += loss_zupt
                losses["zupt"] = bce_zupt.item()

            # --- Regularization Loss (fixed weight) ---
            masked_vel_resid = pred_vel_resid_b * mask_3d
            loss_reg = torch.norm(masked_vel_resid)
            total_loss += reg_weight * loss_reg
            losses["reg"] = loss_reg.item()

            # --- Pen Tip Offset Regularization ---
            # If the model has a learnable pen tip offset, add its regularization loss.
            # We need to access the model instance. Ideally, this should be passed or handled outside,
            # but currently loss function doesn't see the model instance directly.
            # However, train loop has access.
            # For now, let's keep it simple. If we want to add it to UncertaintyLoss,
            # we'd need to pass the model or the loss value.
            # Given the current structure, let's assume pen tip reg is handled
            # via `reg_weight` or similar if needed, but the prompt specifically asked
            # to implement `get_pen_tip_regularization_loss` in the model.
            # So, we should add it in the training loop, NOT here, OR pass it here.
            # The prompt for *this* step was just about Cosine Similarity.
            # The previous step handled Pen Tip implementation.
            # I will calculate pen tip loss in the train loop and add it to `loss_reg` or a new key.

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

    return float(ate), float(dtw[n, m])


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
) -> None:
    """Runs the training loop for the specified model."""

    criterion: nn.Module = UncertaintyLoss(device=device)

    # Ensure the learnable loss parameters are included in the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr)
    # Scheduler now monitors validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path))

    model.to(device)

    # Initialize history dictionaries for both training and validation
    train_history: Dict[str, List[float]] = {"total": [], "pos": [], "vel": [], "cos": [], "zupt": [], "reg": [], "cov": []}
    val_history: Dict[str, List[float]] = {"total": [], "pos": [], "vel": [], "cos": [], "zupt": [], "reg": [], "cov": []}

    print(f"Start Training for {model_name} on {device}...")

    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    for epoch in range(epochs):
        # --- Training Loop ---
        model.train() # Set model to training mode
        epoch_train_losses: Dict[str, float] = {k: 0.0 for k in train_history.keys()}
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
                model_name=model_name
            )

            # Add Pen Tip Regularization Loss if available
            if hasattr(model, "get_pen_tip_regularization_loss"):
                pen_tip_loss = model.get_pen_tip_regularization_loss()
                # Weight it? Let's treat it as part of 'reg' loss or add it to total.
                # Since 'reg_weight' is passed to criterion for velocity reg,
                # we can use a small weight or just add it.
                # Let's add it to total loss and log it under 'reg' (accumulating).
                # Assuming reg_weight is appropriate for this too (it's usually small, e.g., 1e-4).
                loss += reg_weight * pen_tip_loss
                sub_losses["reg"] = sub_losses.get("reg", 0.0) + pen_tip_loss.item()


            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_losses["total"] += loss.item() if not torch.isnan(loss) else 0
            pbar_train_postfix: Dict[str, float] = {"L": loss.item()}
            for k, v in sub_losses.items():
                if k in epoch_train_losses:
                    epoch_train_losses[k] += v if not np.isnan(v) else 0
                pbar_train_postfix[f"L_{k}"] = v
            pbar_train.set_postfix(pbar_train_postfix)

        for k in train_history.keys():
            if k in epoch_train_losses:
                train_history[k].append(epoch_train_losses[k] / len(train_dataloader))

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        epoch_val_losses: Dict[str, float] = {k: 0.0 for k in val_history.keys()}
        val_total_loss_sum = 0.0
        val_batch_count = 0

        with torch.no_grad(): # Disable gradient calculations for validation
            pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            for batch in pbar_val:
                sensor_raw = batch["imu_seq_raw"].to(device)
                gt_pos_w = batch["gt_pos_w"].to(device)
                gt_vel_w = batch["gt_vel_w"].to(device)
                seq_lens = batch["len"].to(device)

                max_len = sensor_raw.shape[1]
                mask = torch.arange(max_len, device=device)[None, :] < seq_lens[:, None]

                gt_vel_norm = torch.norm(gt_vel_w, dim=-1, keepdim=True)
                gt_zupt = (gt_vel_norm < 0.01).float()

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
                    model_name=model_name
                )

                # Add Pen Tip Regularization Loss if available (for tracking)
                if hasattr(model, "get_pen_tip_regularization_loss"):
                    pen_tip_loss = model.get_pen_tip_regularization_loss()
                    loss += reg_weight * pen_tip_loss
                    sub_losses["reg"] = sub_losses.get("reg", 0.0) + pen_tip_loss.item()

                if not torch.isnan(loss):
                    val_total_loss_sum += loss.item()
                    epoch_val_losses["total"] += loss.item()
                    for k, v in sub_losses.items():
                        if k in epoch_val_losses:
                            epoch_val_losses[k] += v if not np.isnan(v) else 0
                val_batch_count += 1
                pbar_val.set_postfix({"L_val": loss.item()})

        avg_val_loss = val_total_loss_sum / val_batch_count
        scheduler.step(avg_val_loss) # Step scheduler with validation loss

        for k in val_history.keys():
            if k in epoch_val_losses:
                val_history[k].append(epoch_val_losses[k] / val_batch_count)

        log_str = f"Epoch {epoch+1}: Train_Total={train_history['total'][-1]:.4f} | Val_Total={val_history['total'][-1]:.4f}"
        if model_name != "only_tcn":
            log_str += f" | Train_Pos={train_history['pos'][-1]:.4f} | Val_Pos={val_history['pos'][-1]:.4f}"
            log_str += f" | Train_Vel={train_history['vel'][-1]:.4f} | Val_Vel={val_history['vel'][-1]:.4f}"
            log_str += f" | Train_Cos={train_history['cos'][-1]:.4f} | Val_Cos={val_history['cos'][-1]:.4f}"
        print(log_str)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plotting training and validation history
    plt.figure(figsize=(12, 8))

    # Define a consistent order for plotting metrics
    plot_order = ["total", "pos", "vel", "cos", "zupt", "reg", "cov"]

    for metric in plot_order:
        if metric in train_history and train_history[metric]:
            plt.plot(train_history[metric], label=f"Train {metric.capitalize()} Loss", marker='.')
        if metric in val_history and val_history[metric]:
            plt.plot(val_history[metric], label=f"Val {metric.capitalize()} Loss", linestyle='--', marker='x')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss History ({model_name})")
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
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device ('cpu', 'cuda', 'mps').")
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

    train(args.model, model, train_dataloader, val_dataloader, args.epochs, args.lr, args.device, model_path, mean, std, **loss_weights)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)
