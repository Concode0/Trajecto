"""
Test pure_eskf with strengthened virtual measurements on short sequences.

Split long sequences into shorter chunks (3s, 5s, 7s) to see if drift is manageable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.pure_eskf import PureESKFModel
from model.config import Config

print("="*60)
print("Short Sequence Test - Optimized pure_eskf")
print("="*60)

# Load data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nOriginal sample: {sample_key}")
print(f"Total length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# Test different sequence lengths
sequence_durations = [3.0, 5.0, 7.0, 10.0, 15.0]  # seconds

print("Testing different sequence lengths:")
print("-"*80)
print(f"{'Duration':>10} {'Samples':>10} {'Vel Error':>12} {'Pos Error':>12} {'Scale Ratio':>12}")
print("-"*80)

# Initialize model
model = PureESKFModel(device="cpu", dt=Config.DT)

for duration_s in sequence_durations:
    # Calculate number of samples
    num_samples = int(duration_s / Config.DT)

    # Skip if not enough data
    if num_samples > seq_len:
        continue

    # Slice data (start from beginning which has static period)
    sensor_slice = sensor_data[:num_samples]
    gt_pos_slice = gt_pos[:num_samples]
    gt_vel_slice = gt_vel[:num_samples]

    # Convert to torch tensors
    sensor_torch = torch.from_numpy(sensor_slice).float().unsqueeze(0)  # (1, T, 7)
    seq_lengths = torch.tensor([num_samples], dtype=torch.long)

    # Run model
    with torch.no_grad():
        output = model(
            imu_raw=sensor_torch,
            imu_norm=sensor_torch,  # Not used by pure_eskf
            seq_lengths=seq_lengths
        )

    pred_pos = output["pred_pos_w"][0, :num_samples].cpu().numpy()

    # Calculate errors
    # Final velocity
    pred_vel_final = np.linalg.norm(np.diff(pred_pos[-2:], axis=0)[0]) / Config.DT
    gt_vel_final = np.linalg.norm(gt_vel_slice[-1])
    vel_error_ratio = pred_vel_final / (gt_vel_final + 1e-9)

    # Position scale (relative to first position)
    pred_disp = np.linalg.norm(pred_pos[-1] - pred_pos[0])
    gt_disp = np.linalg.norm(gt_pos_slice[-1] - gt_pos_slice[0])
    pos_error_ratio = pred_disp / (gt_disp + 1e-9)

    # Trajectory scale (std of centered positions)
    pred_centered = pred_pos - pred_pos.mean(axis=0)
    gt_centered = gt_pos_slice - gt_pos_slice.mean(axis=0)
    pred_scale = np.std(pred_centered)
    gt_scale = np.std(gt_centered)
    scale_ratio = pred_scale / (gt_scale + 1e-9)

    print(f"{duration_s:10.1f} {num_samples:10d} {vel_error_ratio:12.2f}x {pos_error_ratio:12.2f}x {scale_ratio:12.2f}x")

print("-"*80)
print()

print("="*60)
print("Analysis")
print("="*60)
print("Shorter sequences should show:")
print("  1. Lower velocity error (less time for drift to accumulate)")
print("  2. Lower position error (integrated velocity error is smaller)")
print("  3. Lower scale ratio (shape matches better)")
print()
print("If errors remain high even at 3s, virtual measurements need more tuning.")
print("If errors are acceptable at 5s, can use sequence segmentation.")
print("="*60)
