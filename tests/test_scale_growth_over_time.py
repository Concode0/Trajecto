"""
Test how scale error grows over time to identify logical bugs.

If scale grows:
- Linearly: Suggests bias/drift accumulation (physics limitation)
- Exponentially/compounding: Suggests LOGICAL ERROR in corrections
- Step-wise: Suggests periodic bug triggering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.pure_eskf import PureESKFModel
from model.config import Config

print("="*80)
print("SCALE ERROR GROWTH OVER TIME")
print("="*80)

# Load full validation sample
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Full length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# Initialize model
model = PureESKFModel(device="cpu", dt=Config.DT)
model.eval()

# Test at different time windows
test_durations = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, seq_len * Config.DT]

print("="*80)
print("Scale Error vs Sequence Length")
print("="*80)
print(f"{'Duration (s)':>12} {'Samples':>8} {'Scale Ratio':>12} {'Velocity Ratio':>15} {'Under/Over':>12}")
print("-"*80)

scale_ratios = []
durations_sec = []

for duration_s in test_durations:
    # Get segment
    n_samples = min(int(duration_s / Config.DT), seq_len)

    sensor_seg = sensor_data[:n_samples]
    gt_pos_seg = gt_pos[:n_samples]
    gt_vel_seg = gt_vel[:n_samples]

    # Run model
    sensor_torch = torch.from_numpy(sensor_seg).float().unsqueeze(0)
    seq_lengths = torch.tensor([n_samples], dtype=torch.long)

    with torch.no_grad():
        output = model(
            imu_raw=sensor_torch,
            imu_norm=sensor_torch,
            seq_lengths=seq_lengths
        )

    pred_pos = output["pred_pos_w"][0, :n_samples].cpu().numpy()

    # Calculate scale ratio (centered)
    pred_centered = pred_pos - pred_pos.mean(axis=0)
    gt_centered = gt_pos_seg - gt_pos_seg.mean(axis=0)

    pred_scale = np.std(pred_centered)
    gt_scale = np.std(gt_centered)
    scale_ratio = pred_scale / (gt_scale + 1e-9)

    # Calculate velocity ratio
    pred_vel_mag = np.linalg.norm(np.diff(pred_pos[-2:], axis=0)[0]) / Config.DT
    gt_vel_mag = np.linalg.norm(gt_vel_seg[-1])
    vel_ratio = pred_vel_mag / (gt_vel_mag + 1e-9)

    under_over = "UNDER" if scale_ratio < 1.0 else "OVER"

    print(f"{duration_s:12.1f} {n_samples:8d} {scale_ratio:12.2f}x {vel_ratio:15.2f}x {under_over:>12}")

    scale_ratios.append(scale_ratio)
    durations_sec.append(duration_s)

print("-"*80)

# Analyze growth pattern
print()
print("="*80)
print("GROWTH PATTERN ANALYSIS")
print("="*80)

# Check if linear or exponential
durations_arr = np.array(durations_sec)
scale_arr = np.array(scale_ratios)

# Fit linear: scale = a*t + b
from numpy.polynomial import Polynomial
p_linear = Polynomial.fit(durations_arr, scale_arr, 1)
residuals_linear = np.sum((scale_arr - p_linear(durations_arr))**2)

# Fit exponential: scale = a*exp(b*t)
# Take log: log(scale) = log(a) + b*t
# Only fit where scale > 0
valid = scale_arr > 0
if np.sum(valid) > 2:
    p_exp = Polynomial.fit(durations_arr[valid], np.log(np.abs(scale_arr[valid])), 1)
    residuals_exp = np.sum((np.log(np.abs(scale_arr[valid])) - p_exp(durations_arr[valid]))**2)
else:
    residuals_exp = np.inf

print(f"\nLinear fit: scale = {p_linear.convert().coef[1]:.4f}*t + {p_linear.convert().coef[0]:.4f}")
print(f"  Residual sum: {residuals_linear:.6f}")

if residuals_exp < np.inf:
    a = np.exp(p_exp.convert().coef[0])
    b = p_exp.convert().coef[1]
    print(f"\nExponential fit: scale = {a:.4f}*exp({b:.4f}*t)")
    print(f"  Residual sum: {residuals_exp:.6f}")

if residuals_linear < residuals_exp * 0.8:
    print("\n✅ Growth is primarily LINEAR")
    print("   → Suggests constant bias/drift (physics limitation)")
elif residuals_exp < residuals_linear * 0.8:
    print("\n⚠️  Growth is EXPONENTIAL/COMPOUNDING")
    print("   → Suggests LOGICAL ERROR in correction injection")
    print("   → Corrections may be ADDING to error instead of SUBTRACTING")
else:
    print("\n⚠️  Growth is NEITHER linear nor exponential")
    print("   → May have step-wise or complex behavior")

# Check for sign flip at certain duration
print()
print("="*80)
print("CRITICAL OBSERVATIONS")
print("="*80)

under_scaled = [d for d, s in zip(durations_sec, scale_ratios) if s < 1.0]
over_scaled = [d for d, s in zip(durations_sec, scale_ratios) if s > 1.0]

if under_scaled and over_scaled:
    transition_time = max(under_scaled)
    print(f"\n⚠️  CRITICAL: Scale transitions from UNDER to OVER at ~{transition_time:.1f}s")
    print(f"   This suggests corrections are TOO AGGRESSIVE initially,")
    print(f"   then become INSUFFICIENT as error accumulates.")
    print()
    print(f"   Possible causes:")
    print(f"   1. Virtual measurement corrections overshoot initially")
    print(f"   2. Correction sign error that compounds over time")
    print(f"   3. Bias correction in wrong direction")
elif under_scaled:
    print(f"\n✅ Always UNDER-scaled (corrections too strong)")
elif over_scaled:
    print(f"\n⚠️  Always OVER-scaled (corrections too weak or wrong direction)")

print("="*80)
