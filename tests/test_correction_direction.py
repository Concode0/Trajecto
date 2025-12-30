"""
Check if virtual measurement corrections are accumulating in the wrong direction.

Key question: Do virtual measurement corrections REDUCE or INCREASE drift?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.pure_eskf import PureESKFModel
from model.config import Config
import copy

print("="*80)
print("VIRTUAL MEASUREMENT CORRECTION DIRECTION TEST")
print("="*80)

# Load validation sample
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# Test with and without virtual measurements
results = {}

for use_virtual in [False, True]:
    print(f"\n{'='*80}")
    print(f"Running with use_virtual_measurements = {use_virtual}")
    print(f"{'='*80}")

    # Create model
    model = PureESKFModel(device="cpu", dt=Config.DT)
    # Override virtual measurement setting
    model.eskf.use_virtual_measurements = use_virtual
    model.eval()

    # Run full sequence
    sensor_torch = torch.from_numpy(sensor_data).float().unsqueeze(0)
    seq_lengths = torch.tensor([seq_len], dtype=torch.long)

    with torch.no_grad():
        output = model(
            imu_raw=sensor_torch,
            imu_norm=sensor_torch,
            seq_lengths=seq_lengths
        )

    pred_pos = output["pred_pos_w"][0, :seq_len].cpu().numpy()

    # Calculate scale ratio
    pred_centered = pred_pos - pred_pos.mean(axis=0)
    gt_centered = gt_pos - gt_pos.mean(axis=0)

    pred_scale = np.std(pred_centered)
    gt_scale = np.std(gt_centered)
    scale_ratio = pred_scale / gt_scale

    results[use_virtual] = {
        'scale_ratio': scale_ratio,
        'pred_scale': pred_scale,
        'gt_scale': gt_scale
    }

    print(f"\nResults:")
    print(f"  Predicted std: {pred_scale:.6f} m")
    print(f"  GT std:        {gt_scale:.6f} m")
    print(f"  Scale ratio:   {scale_ratio:.2f}x")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

without_virtual = results[False]
with_virtual = results[True]

print(f"\nScale ratio:")
print(f"  Without virtual measurements: {without_virtual['scale_ratio']:.2f}x")
print(f"  With virtual measurements:    {with_virtual['scale_ratio']:.2f}x")

improvement = without_virtual['scale_ratio'] / with_virtual['scale_ratio']

if with_virtual['scale_ratio'] < without_virtual['scale_ratio']:
    print(f"  ✅ Virtual measurements REDUCE drift by {improvement:.2f}x")
elif with_virtual['scale_ratio'] > without_virtual['scale_ratio']:
    print(f"  ❌ ERROR: Virtual measurements INCREASE drift by {1/improvement:.2f}x!")
    print(f"     This suggests corrections are applied in the WRONG direction!")
else:
    print(f"  No effect")

print()
print(f"{'='*80}")
print("DIAGNOSIS")
print(f"{'='*80}")

if with_virtual['scale_ratio'] > without_virtual['scale_ratio']:
    print("\n❌ CRITICAL BUG FOUND!")
    print("   Virtual measurements are making things WORSE.")
    print()
    print("   Possible causes:")
    print("   1. Correction sign error in inject_correction()")
    print("   2. H_error matrix has wrong sign")
    print("   3. Innovation computed backwards (h_pred - measurement instead of measurement - h_pred)")
    print("   4. Bias correction adding instead of subtracting")
elif with_virtual['scale_ratio'] < 1.0:
    print("\n✅ Virtual measurements work but are TOO AGGRESSIVE")
    print(f"   Scale ratio {with_virtual['scale_ratio']:.2f}x means over-correction.")
    print("   This explains why short sequences under-scale but long sequences compound.")
elif with_virtual['scale_ratio'] > 5.0:
    print("\n⚠️  Virtual measurements work but are TOO WEAK")
    print(f"   Scale ratio {with_virtual['scale_ratio']:.2f}x is still large.")
    print("   Corrections are in right direction but insufficient magnitude.")
else:
    print("\n✅ Virtual measurements are working correctly!")
    print(f"   Scale ratio {with_virtual['scale_ratio']:.2f}x is reasonable.")

print(f"{'='*80}")
