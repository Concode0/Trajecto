# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: This software is protected under the following ROK Patent Applications:
# 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
# 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
#
# Commercial use or redistribution of the core logic requires a separate license.
# For inquiries, contact: nemonanconcode@gmail.com

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Evo imports
from evo.core import metrics, trajectory, sync
from evo.core.metrics import PoseRelation

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import TrajectoryDataset
from model.config import Config
from model.ESKF_TCN import ESKFTCN_model
from train_eskf import TrainConfig # Import TrainConfig

# Add TrainConfig to safe globals for checkpoint loading
torch.serialization.add_safe_globals([TrainConfig])

def load_model(model_type: str, model_path: str, device: str, mean: torch.Tensor, std: torch.Tensor) -> torch.nn.Module:
    """Initializes and loads the trained model weights."""

    model_params: Dict[str, Any] = {}

    if model_type == "eskf_tcn":
        # Updated parameters for the Y-branch ESKF-TCN architecture
        model_params = {
            "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
            "tcn_channels": Config.ESKFTCN.TCN_CHANNELS,
            "kernel_size": Config.ESKFTCN.KERNEL_SIZE,
            "dropout": Config.ESKFTCN.DROPOUT,
            "tcn_backbone_dilations": Config.ESKFTCN.TCN_BACKBONE_DILATIONS,
            "tcn_dynamic_dilations": Config.ESKFTCN.TCN_DYNAMIC_DILATIONS,
            "tcn_static_dilations": Config.ESKFTCN.TCN_STATIC_DILATIONS,
            "use_zupt": Config.ESKFTCN.USE_ZUPT,
            "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT,
            "dt": Config.DT,
            "separable": Config.ESKFTCN.USE_SEPARABLE_CONV,
        }
        model_class = ESKFTCN_model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'eskf_tcn' is supported in this minimal version.")

    model = model_class(device=device, **model_params)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load weights
    # Set weights_only=False to allow loading custom classes like TrainConfig from the checkpoint
    # This is safe here as we are loading locally generated trusted checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
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

    # Identity orientation in wxyz format (scalar-first): [w, x, y, z] = [1, 0, 0, 0]
    # The evo library expects quaternions in wxyz (scalar-first) format
    qs = np.zeros((n, 4))
    qs[:, 0] = 1.0  # w component (scalar part)

    return trajectory.PoseTrajectory3D(positions_xyz=positions, orientations_quat_wxyz=qs, timestamps=timestamps)


def visualize_sample(
    sample_idx: int,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    pred_xyz_aligned: np.ndarray,
    metrics_dict: Dict[str, float],
    output_dir: str,
    dt: float = 0.02
) -> None:
    """
    Creates a comprehensive visualization for a single sample with its own scale.

    Args:
        sample_idx: Sample index for labeling
        gt_xyz: Ground truth positions [N, 3]
        pred_xyz: Raw predicted positions [N, 3]
        pred_xyz_aligned: Aligned predicted positions [N, 3]
        metrics_dict: Dictionary containing sample metrics
        output_dir: Directory to save plots
        dt: Time step
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    time = np.arange(len(gt_xyz)) * dt

    # Calculate unified axis limits for consistent scaling across all spatial plots
    # Combine GT and aligned prediction for bounds calculation
    all_data = np.vstack([gt_xyz, pred_xyz_aligned])
    margin = 0.05  # 5% margin for better visualization

    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    z_min, z_max = all_data[:, 2].min(), all_data[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_lim = [x_min - margin * x_range, x_max + margin * x_range]
    y_lim = [y_min - margin * y_range, y_max + margin * y_range]
    z_lim = [z_min - margin * z_range, z_max + margin * z_range]

    # Plot 1: 3D Trajectory (before alignment)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], 'r--', linewidth=1.5, label='Prediction (raw)', alpha=0.6)
    ax1.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_title(f'3D Trajectory (Before Alignment)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: 3D Trajectory (after alignment)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 1], pred_xyz_aligned[:, 2], 'b-', linewidth=1.5, label='Prediction (aligned)', alpha=0.7)
    ax2.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='green', s=100, marker='o')
    ax2.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='red', s=100, marker='x')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)
    ax2.set_title(f'3D Trajectory (After Alignment)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: XY Projection
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax3.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 1], 'b-', linewidth=1.5, label='Prediction', alpha=0.7)
    ax3.scatter(gt_xyz[0, 0], gt_xyz[0, 1], c='green', s=100, marker='o')
    ax3.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], c='red', s=100, marker='x')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    ax3.set_title('XY Projection')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Plot 4: XZ Projection
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax4.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 2], 'b-', linewidth=1.5, label='Prediction', alpha=0.7)
    ax4.scatter(gt_xyz[0, 0], gt_xyz[0, 2], c='green', s=100, marker='o')
    ax4.scatter(gt_xyz[-1, 0], gt_xyz[-1, 2], c='red', s=100, marker='x')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_xlim(x_lim)
    ax4.set_ylim(z_lim)
    ax4.set_title('XZ Projection')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')

    # Plot 5: Per-axis errors over time
    error_xyz = pred_xyz_aligned - gt_xyz
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.plot(time, error_xyz[:, 0], 'r-', linewidth=1.5, label='X error', alpha=0.7)
    ax5.plot(time, error_xyz[:, 1], 'g-', linewidth=1.5, label='Y error', alpha=0.7)
    ax5.plot(time, error_xyz[:, 2], 'b-', linewidth=1.5, label='Z error', alpha=0.7)
    ax5.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Error (m)')
    ax5.set_title('Per-Axis Position Error Over Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Euclidean error over time
    euclidean_error = np.linalg.norm(error_xyz, axis=1)
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.plot(time, euclidean_error, 'purple', linewidth=2, alpha=0.8)
    ax6.axhline(metrics_dict['ape_rmse'], color='red', linestyle='--', linewidth=2, label=f'RMSE: {metrics_dict["ape_rmse"]:.4f}m')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Euclidean Error (m)')
    ax6.set_title('Euclidean Position Error Over Time')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Position components over time (GT vs Pred)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(time, gt_xyz[:, 0], 'g-', linewidth=2, label='GT X', alpha=0.8)
    ax7.plot(time, pred_xyz_aligned[:, 0], 'r--', linewidth=1.5, label='Pred X', alpha=0.7)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('X Position (m)')
    ax7.set_title('X Component')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(time, gt_xyz[:, 1], 'g-', linewidth=2, label='GT Y', alpha=0.8)
    ax8.plot(time, pred_xyz_aligned[:, 1], 'r--', linewidth=1.5, label='Pred Y', alpha=0.7)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Y Position (m)')
    ax8.set_title('Y Component')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(time, gt_xyz[:, 2], 'g-', linewidth=2, label='GT Z', alpha=0.8)
    ax9.plot(time, pred_xyz_aligned[:, 2], 'r--', linewidth=1.5, label='Pred Z', alpha=0.7)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Z Position (m)')
    ax9.set_title('Z Component')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # Plot 10: Metrics summary
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')

    metrics_text = f"""
SAMPLE {sample_idx} METRICS

APE RMSE: {metrics_dict['ape_rmse']:.4f} m
Error/Distance: {metrics_dict['error_over_dist']:.4f}
Error/Time: {metrics_dict['error_over_time']:.4f} m/s

Axis RMSE:
  X: {metrics_dict['rmse_x']:.4f} m
  Y: {metrics_dict['rmse_y']:.4f} m
  Z: {metrics_dict['rmse_z']:.4f} m

Scale ratio: {metrics_dict['scale_ratio']:.4f}
Path length: {metrics_dict['path_length']:.4f} m
Duration: {metrics_dict['duration']:.2f} s
Sequence length: {metrics_dict['seq_length']} steps
"""

    ax10.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
              verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Main title
    fig.suptitle(f'Sample {sample_idx} - Detailed Analysis', fontsize=16, fontweight='bold')

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    output_dir: str,
    correct_scale: bool = True,
    model_type: str = None
) -> None:
    """
    Runs inference and computes metrics using evo.

    Args:
        correct_scale: If True, uses Sim(3) alignment (with scale).
                       If False, uses SE(3) alignment (without scale).
        model_type: Type of model being evaluated (used to determine if seq_lengths should be passed).
    """
    mean_gpu = mean.to(device)
    std_gpu = std.to(device)

    ape_rmse_stats = []
    scale_ratio_stats = []

    # New metrics containers
    error_over_dist_stats = []
    error_over_time_stats = []
    rmse_x_stats = []
    rmse_y_stats = []
    rmse_z_stats = []

    # Per-sample results storage
    sample_results = []
    global_sample_idx = 0

    print(f"Starting evaluation (scale alignment: {'ON' if correct_scale else 'OFF'})...")
    print("Generating individual sample visualizations...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Eval")):
            sensor_raw = batch["imu_seq_raw"].to(device)
            gt_pos_w = batch["gt_pos_w"].to(device)
            seq_lens = batch["len"] # keep on CPU for indexing
            seq_lens_device = seq_lens.to(device) # For model input

            sensor_norm = (sensor_raw - mean_gpu) / (std_gpu + 1e-6)

            # Pass sequence lengths to model for padding mask (only for models that support it)
            if model_type == "pure_eskf":
                model_output = model(sensor_raw, sensor_norm, seq_lengths=seq_lens_device)
            else:
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

                # Compute trajectory statistics BEFORE alignment for diagnostics
                # CRITICAL: Center trajectories before computing std to get scale of MOTION, not absolute position
                # GT data doesn't start at origin (typically ~0.11m offset), but ESKF does
                # Without centering, std() measures spread around origin, not spread around trajectory center
                pred_xyz_centered = pred_xyz - pred_xyz.mean(axis=0)
                gt_xyz_centered = gt_xyz - gt_xyz.mean(axis=0)

                pred_scale = np.std(pred_xyz_centered)
                gt_scale = np.std(gt_xyz_centered)
                scale_ratio = pred_scale / gt_scale if gt_scale > 1e-6 else 0.0
                scale_ratio_stats.append(scale_ratio)

                # Compute path length and duration for diagnostics
                gt_diff = np.diff(gt_xyz, axis=0)
                path_length_diag = np.sum(np.linalg.norm(gt_diff, axis=1))
                duration_diag = length * Config.DT

                # Align trajectories (Sim(3) or SE(3) depending on correct_scale flag)
                traj_est.align(traj_ref, correct_scale=correct_scale)
                traj_est_aligned = traj_est

                # Print diagnostic info for first few samples to debug scale issues
                if i < 3:  # Only print for first 3 batches
                    print(f"\n--- Sample {i*sensor_raw.shape[0] + b} Diagnostics ---")
                    print(f"  GT trajectory std:   {gt_scale:.4f} m")
                    print(f"  Pred trajectory std: {pred_scale:.4f} m")
                    print(f"  Scale ratio (pred/gt): {scale_ratio:.4f}")
                    print(f"  Path length: {path_length_diag:.4f} m")
                    print(f"  Duration: {duration_diag:.2f} s")

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

                # Store sample metrics
                sample_metrics = {
                    'sample_idx': global_sample_idx,
                    'ape_rmse': rmse,
                    'error_over_dist': rmse / safe_path_len,
                    'error_over_time': rmse / safe_duration,
                    'rmse_x': axis_rmse[0],
                    'rmse_y': axis_rmse[1],
                    'rmse_z': axis_rmse[2],
                    'scale_ratio': scale_ratio,
                    'path_length': path_len,
                    'duration': duration,
                    'seq_length': length
                }
                sample_results.append(sample_metrics)

                # Create visualization for this sample
                visualize_sample(
                    sample_idx=global_sample_idx,
                    gt_xyz=gt_xyz,
                    pred_xyz=pred_xyz,
                    pred_xyz_aligned=est_aligned_xyz,
                    metrics_dict=sample_metrics,
                    output_dir=os.path.join(output_dir, "individual_samples"),
                    dt=Config.DT
                )

                global_sample_idx += 1

    # Print per-sample results table
    print("\n" + "="*120)
    print("PER-SAMPLE RESULTS")
    print("="*120)

    header = f"{'Sample':<8} | {'APE RMSE':<10} | {'Err/Dist':<10} | {'Err/Time':<10} | {'RMSE_X':<8} | {'RMSE_Y':<8} | {'RMSE_Z':<8} | {'Scale':<8} | {'Path(m)':<8} | {'Dur(s)':<7}"
    print(header)
    print("-" * 120)

    for result in sample_results:
        row = (f"{result['sample_idx']:<8} | "
               f"{result['ape_rmse']:<10.4f} | "
               f"{result['error_over_dist']:<10.4f} | "
               f"{result['error_over_time']:<10.4f} | "
               f"{result['rmse_x']:<8.4f} | "
               f"{result['rmse_y']:<8.4f} | "
               f"{result['rmse_z']:<8.4f} | "
               f"{result['scale_ratio']:<8.4f} | "
               f"{result['path_length']:<8.4f} | "
               f"{result['duration']:<7.2f}")
        print(row)

    print("="*120)

    # Aggregated Results
    print("\n" + "="*40)
    print("AGGREGATED RESULTS")
    print("="*40)
    print(f"Num Samples: {len(ape_rmse_stats)}")
    print(f"Alignment: {'Sim(3) with scale' if correct_scale else 'SE(3) without scale'}")

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
    print_stats("Scale Ratio (pred/gt)", scale_ratio_stats, unit="")
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

    # Save aggregated results
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

    # Save per-sample results to CSV
    csv_path = os.path.join(output_dir, "per_sample_results.csv")
    with open(csv_path, "w") as f:
        f.write("sample_idx,ape_rmse,error_over_dist,error_over_time,rmse_x,rmse_y,rmse_z,scale_ratio,path_length,duration,seq_length\n")
        for result in sample_results:
            f.write(f"{result['sample_idx']},{result['ape_rmse']:.6f},{result['error_over_dist']:.6f},"
                   f"{result['error_over_time']:.6f},{result['rmse_x']:.6f},{result['rmse_y']:.6f},"
                   f"{result['rmse_z']:.6f},{result['scale_ratio']:.6f},{result['path_length']:.6f},"
                   f"{result['duration']:.6f},{result['seq_length']}\n")

    print(f"\nSaved results to:")
    print(f"  - {results_path}")
    print(f"  - {csv_path}")
    print(f"  - Individual sample visualizations in: {os.path.join(output_dir, 'individual_samples')}")


def main():
    parser = argparse.ArgumentParser(description="Trajecto Validation Tool")
    parser.add_argument("--model_type", type=str, default="eskf_tcn", choices=["eskf_tcn"], help="Model architecture (only eskf_tcn supported)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--dataset", type=str, default="./data/dataset.h5", help="Path to validation HDF5 dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), help="Device to run inference on (cuda/mps/cpu)")
    parser.add_argument("--output_dir", type=str, default="results/validation", help="Directory to save results")
    parser.add_argument("--no_scale", action="store_true", help="Disable scale correction in alignment (use SE(3) instead of Sim(3))")

    args = parser.parse_args()

    # Validation for model path
    if not args.model_path:
        parser.error("--model_path is required")

    print(f"Validating {args.model_type} model")
    if args.model_path:
        print(f"From {args.model_path}")
    print(f"Using dataset: {args.dataset}")
    print(f"Device: {args.device}")

    # Load Scaler Stats
    print(f"Loading scaler stats from...")
    with h5py.File("./data/scaler_stats.h5", "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float()
        std = torch.from_numpy(f["std"][:]).float()

    # Load Dataset
    val_dataset = TrajectoryDataset(
        preprocessed_file=args.dataset,
        augment_multiplier=1,
        subsample_step=1,
        do_augment=False
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load Model
    model = load_model(args.model_type, args.model_path, args.device, mean, std)

    # Run Evaluation
    correct_scale = not args.no_scale  # Default is True (Sim(3)), --no_scale makes it False (SE(3))
    evaluate(model, val_dataloader, args.device, mean, std, args.output_dir, correct_scale=correct_scale, model_type=args.model_type)

if __name__ == "__main__":
    main()