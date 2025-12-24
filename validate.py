"""Validation script for Trajecto using 'evo' library for trajectory evaluation.

This script loads a trained model, runs inference on a validation dataset,
and uses the 'evo' library to compute metrics (APE/RPE) by comparing
the predicted trajectories against the ground truth.

Metrics computed:
- APE (Absolute Pose Error) - Translation Part (RMSE)
"""

import argparse
import os
import sys
import numpy as np
import torch
import h5py
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Evo imports
from evo.core import metrics, trajectory, sync
from evo.core.metrics import PoseRelation

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import TrajectoryDataset
from model.config import Config
from model.ESKF_TCN import ESKFTCN_model
from model.AEKF_TCN import AEKFTCN_model
from model.onlyTCN import OnlyTCN
from model.pure_eskf import PureESKFModel


def load_model(model_type: str, model_path: str, device: str, mean: torch.Tensor, std: torch.Tensor) -> torch.nn.Module:
    """Initializes and loads the trained model weights."""

    if model_type == "pure_eskf":
        print("Initializing Pure ESKF (Physics Baseline)...")
        return PureESKFModel(device=device, dt=Config.DT)

    model_params: Dict[str, Any] = {}

    if model_type == "eskf_tcn":
        model_params = {
            "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
            "use_zupt": Config.ESKFTCN.USE_ZUPT,
            "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT,
            "dt": Config.DT
        }
        model_class = ESKFTCN_model
    elif model_type == "aekf_tcn":
         model_params = {
            "tcn_input_size": Config.AEKFTCN.TCN_INPUT_SIZE,
            "dt": Config.DT
         }
         model_class = AEKFTCN_model
    elif model_type == "only_tcn":
        model_params = {
            "input_size": Config.OnlyTCN.INPUT_SIZE,
            "output_size": Config.OnlyTCN.OUTPUT_SIZE,
            "dt": Config.DT
        }
        model_class = OnlyTCN
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_class(device=device, **model_params)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def create_evo_trajectory(positions: np.ndarray, dt: float = 0.02) -> trajectory.PoseTrajectory3D:
    """
    Creates an evo PoseTrajectory3D from a sequence of positions.
    Since orientation is not tracked in dataset for evaluation, we use Identity.
    """
    n = positions.shape[0]
    timestamps = np.arange(n) * dt

    # Identity orientation (w, x, y, z) -> (1, 0, 0, 0)
    # Evo internally uses scalar-last or scalar-first depending on context,
    # but constructing with (N, 4) usually implies (x, y, z, w) or (w, x, y, z).
    # We stick to scalar-last (x, y, z, w) which is common in ROS.
    qs = np.zeros((n, 4))
    qs[:, 3] = 1.0

    return trajectory.PoseTrajectory3D(positions_xyz=positions, orientations_quat_wxyz=qs, timestamps=timestamps)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    output_dir: str
) -> None:
    """
    Runs inference and computes metrics using evo.
    """
    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    ape_rmse_stats = []

    # New metrics containers
    error_over_dist_stats = []
    error_over_time_stats = []
    rmse_x_stats = []
    rmse_y_stats = []
    rmse_z_stats = []

    print("Starting evaluation...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Eval")):
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            seq_lens = batch["len"] # keep on CPU for indexing

            sensor_norm = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            model_output = model(sensor_raw, sensor_norm)

            # Extract position based on model type
            if isinstance(model_output, dict):
                pred_pos_w = model_output["pred_pos_w"]
            else:
                pred_pos_w = model_output

            # Process each sequence in the batch
            for b in range(sensor_raw.shape[0]):
                length = int(seq_lens[b].item())

                # Get numpy arrays
                gt_xyz = gt_pos_w[b, :length].cpu().numpy()
                pred_xyz = pred_pos_w[b, :length].cpu().numpy()

                # Create evo trajectories
                traj_ref = create_evo_trajectory(gt_xyz, dt=Config.DT)
                traj_est = create_evo_trajectory(pred_xyz, dt=Config.DT)

                # Align (Umeyama alignment - Scale, Rotation, Translation)
                # For handwriting, scale might be ambiguous if IMU is not perfect,
                # but typically we want 1:1 scale if we trust our physics.
                # However, for pure shape matching, we align with scale.
                # 'train.py' does: `scale = gt_std / pred_std` (Sim(3) alignment)
                # We will perform Sim(3) alignment using evo.

                traj_est.align(traj_ref, correct_scale=True)
                traj_est_aligned = traj_est

                # Compute APE (Translation)
                data = (traj_ref, traj_est_aligned)
                ape_metric = metrics.APE(PoseRelation.translation_part)
                ape_metric.process_data(data)

                ape_stats = ape_metric.get_all_statistics()
                rmse = ape_stats["rmse"]
                ape_rmse_stats.append(rmse)

                # --- New Metrics Calculation ---

                # 1. Path Length & Duration
                gt_diff = np.diff(gt_xyz, axis=0)
                path_len = np.sum(np.linalg.norm(gt_diff, axis=1))
                duration = length * Config.DT

                # Avoid division by zero
                safe_path_len = max(path_len, 1e-3)
                safe_duration = max(duration, 1e-3)

                error_over_dist_stats.append(rmse / safe_path_len)
                error_over_time_stats.append(rmse / safe_duration)

                # 2. Individual Axis Analysis
                # Compare aligned prediction with ground truth
                est_aligned_xyz = traj_est_aligned.positions_xyz
                error_xyz = est_aligned_xyz - gt_xyz

                # RMSE per axis
                axis_rmse = np.sqrt(np.mean(error_xyz**2, axis=0))
                rmse_x_stats.append(axis_rmse[0])
                rmse_y_stats.append(axis_rmse[1])
                rmse_z_stats.append(axis_rmse[2])

    # Aggregated Results
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Num Samples: {len(ape_rmse_stats)}")

    # Helper to print stats
    def print_stats(name, data, unit="m"):
        print(f"{name}:")
        print(f"  Mean:   {np.mean(data):.4f} {unit}")
        print(f"  Median: {np.median(data):.4f} {unit}")
        print(f"  Std:    {np.std(data):.4f} {unit}")
        if unit == "m": # Only for absolute errors
            print(f"  Min:    {np.min(data):.4f} {unit}")
            print(f"  Max:    {np.max(data):.4f} {unit}")

    print_stats("APE (RMSE) - Overall", ape_rmse_stats)
    print("-" * 20)
    print_stats("Error / Distance", error_over_dist_stats, unit="m/m")
    print("-" * 20)
    print_stats("Error / Time", error_over_time_stats, unit="m/s")
    print("-" * 20)
    print("Individual Axis RMSE:")
    print(f"  X: {np.mean(rmse_x_stats):.4f} m")
    print(f"  Y: {np.mean(rmse_y_stats):.4f} m")
    print(f"  Z: {np.mean(rmse_z_stats):.4f} m")
    print("="*40)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "validation_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Mean APE (RMSE): {np.mean(ape_rmse_stats):.6f}\n")
        f.write(f"Median APE (RMSE): {np.median(ape_rmse_stats):.6f}\n")
        f.write(f"Std APE (RMSE): {np.std(ape_rmse_stats):.6f}\n")

        f.write(f"\nMean Error/Distance: {np.mean(error_over_dist_stats):.6f}\n")
        f.write(f"Mean Error/Time: {np.mean(error_over_time_stats):.6f}\n")

        f.write(f"\nAxis RMSE:\n")
        f.write(f"  X: {np.mean(rmse_x_stats):.6f}\n")
        f.write(f"  Y: {np.mean(rmse_y_stats):.6f}\n")
        f.write(f"  Z: {np.mean(rmse_z_stats):.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Trajecto Validation Tool")
    parser.add_argument("--model_type", type=str, required=True, choices=["eskf_tcn", "aekf_tcn", "only_tcn", "pure_eskf"], help="Model architecture")
    parser.add_argument("--model_path", type=str, required=False, help="Path to .pth model file (not required for pure_eskf)")
    parser.add_argument("--dataset", type=str, default=Config.VALIDATION_DATASET_H5_PATH, help="Path to validation HDF5 dataset")
    parser.add_argument("--device", type=str, default="mps" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--output_dir", type=str, default="results/validation", help="Directory to save results")

    args = parser.parse_args()

    # Validation for model path
    if args.model_type != "pure_eskf" and not args.model_path:
        parser.error("--model_path is required for model_type other than 'pure_eskf'")

    print(f"Validating {args.model_type} model")
    if args.model_path:
        print(f"From {args.model_path}")
    print(f"Using dataset: {args.dataset}")
    print(f"Device: {args.device}")

    # Load Scaler Stats
    print(f"Loading scaler stats from {Config.SCALER_STATS_H5_PATH}...")
    with h5py.File(Config.SCALER_STATS_H5_PATH, "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float()
        std = torch.from_numpy(f["std"][:]).float()

    # Load Dataset
    val_dataset = TrajectoryDataset(
        preprocessed_file=args.dataset,
        augment_multiplier=1,
        subsample_step=Config.SUBSAMPLE_STEP,
        do_augment=False
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load Model
    model = load_model(args.model_type, args.model_path, args.device, mean, std)

    # Run Evaluation
    evaluate(model, val_dataloader, args.device, mean, std, args.output_dir)

if __name__ == "__main__":
    main()