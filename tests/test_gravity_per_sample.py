"""
Check gravity magnitude across all samples to see if we should
calibrate a single global value instead of per-sample estimation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

print("="*60)
print("Gravity Magnitude Across All Samples")
print("="*60)

# Check both datasets
for dataset_path in ["data/dataset.h5", "data/validation_dataset.h5"]:
    if not os.path.exists(dataset_path):
        continue

    print(f"\nDataset: {dataset_path}")
    print("-"*60)

    with h5py.File(dataset_path, "r") as f:
        gravity_mags = []

        for sample_key in f.keys():
            g = f[sample_key]
            seq_len = g.attrs["sequence_length"]
            sensor_data = g["sensor_data"][:seq_len]

            # Use first 50 samples (static period)
            static_samples = min(50, seq_len)
            static_accel = sensor_data[:static_samples, :3]

            # Calculate average gravity magnitude
            avg_accel = static_accel.mean(axis=0)
            gravity_mag = np.linalg.norm(avg_accel)
            gravity_mags.append(gravity_mag)

            print(f"  {sample_key}: {gravity_mag:.6f} m/s²")

        gravity_mags = np.array(gravity_mags)
        print()
        print(f"Statistics across {len(gravity_mags)} samples:")
        print(f"  Mean:   {gravity_mags.mean():.6f} m/s²")
        print(f"  Median: {np.median(gravity_mags):.6f} m/s²")
        print(f"  Std:    {gravity_mags.std():.6f} m/s²")
        print(f"  Min:    {gravity_mags.min():.6f} m/s²")
        print(f"  Max:    {gravity_mags.max():.6f} m/s²")
        print(f"  Range:  {gravity_mags.max() - gravity_mags.min():.6f} m/s²")

print()
print("="*60)
print("Recommendation")
print("="*60)
print("If std is small (<0.1 m/s²), use mean value in Config.GRAVITY_MAGNITUDE")
print("If std is large (>0.1 m/s²), must use per-sample estimation")
print("="*60)
