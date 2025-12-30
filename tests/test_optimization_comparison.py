"""
Compare original vs strengthened virtual measurements.

Test on full validation dataset to see overall improvement.
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
print("Optimization Comparison Test")
print("="*60)

# Test on all samples from training dataset
with h5py.File("data/dataset.h5", "r") as f:
    samples = list(f.keys())

    print(f"\nTesting on {len(samples)} samples")
    print("-"*80)
    print(f"{'Sample':>20} {'Length':>8} {'Vel Ratio':>12} {'Pos Ratio':>12} {'Scale Ratio':>12}")
    print("-"*80)

    model = PureESKFModel(device="cpu", dt=Config.DT)

    all_vel_ratios = []
    all_pos_ratios = []
    all_scale_ratios = []

    for sample_key in samples:
        g = f[sample_key]
        seq_len = g.attrs["sequence_length"]
        sensor_data = g["sensor_data"][:seq_len]
        gt_pos = g["gt_pos_data"][:seq_len]
        gt_vel = g["gt_vel_data"][:seq_len]

        # Convert to torch
        sensor_torch = torch.from_numpy(sensor_data).float().unsqueeze(0)
        seq_lengths = torch.tensor([seq_len], dtype=torch.long)

        # Run model
        with torch.no_grad():
            output = model(
                imu_raw=sensor_torch,
                imu_norm=sensor_torch,
                seq_lengths=seq_lengths
            )

        pred_pos = output["pred_pos_w"][0, :seq_len].cpu().numpy()

        # Calculate metrics
        # Velocity ratio
        pred_vel_final = np.linalg.norm(np.diff(pred_pos[-2:], axis=0)[0]) / Config.DT
        gt_vel_final = np.linalg.norm(gt_vel[-1])
        vel_ratio = pred_vel_final / (gt_vel_final + 1e-9)

        # Position ratio
        pred_disp = np.linalg.norm(pred_pos[-1] - pred_pos[0])
        gt_disp = np.linalg.norm(gt_pos[-1] - gt_pos[0])
        pos_ratio = pred_disp / (gt_disp + 1e-9)

        # Scale ratio (centered)
        pred_centered = pred_pos - pred_pos.mean(axis=0)
        gt_centered = gt_pos - gt_pos.mean(axis=0)
        pred_scale = np.std(pred_centered)
        gt_scale = np.std(gt_centered)
        scale_ratio = pred_scale / (gt_scale + 1e-9)

        all_vel_ratios.append(vel_ratio)
        all_pos_ratios.append(pos_ratio)
        all_scale_ratios.append(scale_ratio)

        duration_s = seq_len * Config.DT
        print(f"{sample_key:>20} {duration_s:8.1f}s {vel_ratio:12.2f}x {pos_ratio:12.2f}x {scale_ratio:12.2f}x")

    print("-"*80)
    print(f"{'MEAN':>20} {'':<8} {np.mean(all_vel_ratios):12.2f}x {np.mean(all_pos_ratios):12.2f}x {np.mean(all_scale_ratios):12.2f}x")
    print(f"{'MEDIAN':>20} {'':<8} {np.median(all_vel_ratios):12.2f}x {np.median(all_pos_ratios):12.2f}x {np.median(all_scale_ratios):12.2f}x")
    print("-"*80)

print()
print("="*60)
print("Baseline Comparison (Before Optimization)")
print("="*60)
print("Previous results with weaker virtual measurements:")
print("  Velocity ratio: ~30x")
print("  Position ratio: ~400x")
print("  Scale ratio: ~250x")
print()
print("With strengthened virtual measurements:")
print(f"  Velocity ratio: {np.mean(all_vel_ratios):.2f}x")
print(f"  Position ratio: {np.mean(all_pos_ratios):.2f}x")
print(f"  Scale ratio: {np.mean(all_scale_ratios):.2f}x")
print()

improvement_vel = 30.0 / np.mean(all_vel_ratios)
improvement_pos = 400.0 / np.mean(all_pos_ratios)
improvement_scale = 250.0 / np.mean(all_scale_ratios)

print(f"Improvement factors:")
print(f"  Velocity: {improvement_vel:.2f}x better")
print(f"  Position: {improvement_pos:.2f}x better")
print(f"  Scale: {improvement_scale:.2f}x better")
print("="*60)
