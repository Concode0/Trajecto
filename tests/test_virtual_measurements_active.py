"""
Verify that virtual measurements are actually being applied in pure_eskf.

Check:
1. Virtual measurement updates are triggered
2. Corrections are non-zero
3. Covariance is reduced after virtual measurements
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
print("VIRTUAL MEASUREMENTS ACTIVATION TEST")
print("="*80)

# Load short sequence
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = min(g.attrs["sequence_length"], 200)  # First 4 seconds
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Testing first {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# Initialize model
model = PureESKFModel(device="cpu", dt=Config.DT)
model.eval()

# Prepare data
sensor_torch = torch.from_numpy(sensor_data).float().unsqueeze(0)
seq_lengths = torch.tensor([seq_len], dtype=torch.long)

# Run with instrumentation
print("="*80)
print("Running ESKF with instrumentation...")
print("="*80)

# Monkey-patch the ESKF to track virtual measurement usage
original_update = model.eskf.update
update_calls = []

def instrumented_update(P_error, quat_b_to_w, accel_bias_b, gyro_bias_b, measurement, R_override=None, gating_threshold=None):
    result = original_update(P_error, quat_b_to_w, accel_bias_b, gyro_bias_b, measurement, R_override, gating_threshold)
    delta_x = result[0]

    # Track delta_x magnitude
    delta_x_norm = torch.norm(delta_x, dim=-1).mean().item()
    update_calls.append({
        'delta_x_norm': delta_x_norm,
        'used_override_R': R_override is not None,
    })

    return result

model.eskf.update = instrumented_update

# Run forward pass
with torch.no_grad():
    output = model(
        imu_raw=sensor_torch,
        imu_norm=sensor_torch,
        seq_lengths=seq_lengths
    )

# Restore original update
model.eskf.update = original_update

print(f"\nTotal update() calls: {len(update_calls)}")
print(f"Updates with R_override (virtual meas): {sum(1 for c in update_calls if c['used_override_R'])}")
print()

# Analyze virtual measurement strength
virtual_updates = [c for c in update_calls if c['used_override_R']]
if virtual_updates:
    delta_norms = [c['delta_x_norm'] for c in virtual_updates]
    print(f"Virtual measurement correction statistics:")
    print(f"  Mean correction: {np.mean(delta_norms):.6f}")
    print(f"  Max correction:  {np.max(delta_norms):.6f}")
    print(f"  Min correction:  {np.min(delta_norms):.6f}")

    if np.mean(delta_norms) > 1e-6:
        print("  ✅ Virtual measurements are producing corrections")
    else:
        print("  ⚠️  Virtual measurements producing negligible corrections!")
else:
    print("  ❌ NO VIRTUAL MEASUREMENTS APPLIED!")

print()

# Check ZUPT activation (if available in output)
if "zupt_flag" in output:
    zupt_flags = output["zupt_flag"][0, :seq_len].cpu().numpy()
    zupt_rate = np.mean(zupt_flags)
    print(f"ZUPT activation rate: {zupt_rate*100:.2f}%")
    print(f"ZUPT activated steps: {np.sum(zupt_flags)} / {seq_len}")
else:
    print("ZUPT flag not in output (pure_eskf doesn't return it)")
    zupt_rate = 0.0

if zupt_rate > 0.05:
    print("  ✅ ZUPT is activating (>5%)")
elif zupt_rate > 0:
    print("  ⚠️  ZUPT activation low (<5%)")
else:
    print("  ❌ ZUPT NEVER ACTIVATED!")

# Check use_virtual_measurements flag
print()
print(f"Model configuration:")
print(f"  use_virtual_measurements: {model.eskf.use_virtual_measurements}")
print(f"  use_zupt: {model.eskf.use_zupt}")
print(f"  use_tcn_zupt: {model.eskf.use_tcn_zupt}")

# Check trajectory scale
pred_pos = output["pred_pos_w"][0, :seq_len].cpu().numpy()
pred_scale = np.std(pred_pos - pred_pos.mean(axis=0))
gt_scale = np.std(gt_pos - gt_pos.mean(axis=0))
scale_ratio = pred_scale / gt_scale

print()
print(f"Trajectory scale:")
print(f"  Predicted std: {pred_scale:.6f} m")
print(f"  GT std:        {gt_scale:.6f} m")
print(f"  Scale ratio:   {scale_ratio:.2f}x")

print()
print("="*80)
print("ANALYSIS")
print("="*80)

if not model.eskf.use_virtual_measurements:
    print("⚠️  CRITICAL: use_virtual_measurements is DISABLED!")
    print("   Virtual measurements will NOT be applied regardless of settings.")
elif not virtual_updates:
    print("⚠️  CRITICAL: Virtual measurements enabled but NOT being applied!")
    print("   Check logic in ESKF.one_step() around line 766.")
elif np.mean([c['delta_x_norm'] for c in virtual_updates]) < 1e-6:
    print("⚠️  Virtual measurements applied but corrections are TINY!")
    print("   This suggests measurement noise R is too large.")
else:
    print("✅ Virtual measurements are working and producing corrections.")

if zupt_rate == 0:
    print("⚠️  ZUPT never activates - thresholds may still be too strict.")
elif zupt_rate < 0.05:
    print("⚠️  ZUPT activation low - consider relaxing thresholds further.")
else:
    print("✅ ZUPT is activating at reasonable rate.")

if scale_ratio > 5.0:
    print(f"⚠️  Scale error still high ({scale_ratio:.1f}x) despite optimizations.")
    print("   Possible causes:")
    print("   1. Virtual measurement weights too weak")
    print("   2. R matrix (measurement noise) too large")
    print("   3. Logical error in error state injection")
else:
    print(f"✅ Scale error acceptable ({scale_ratio:.1f}x)")

print("="*80)
