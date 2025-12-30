"""
Check force sensor (FSR) noise characteristics.

High noise could affect ZUPT detection and cause drift.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*60)
print("Force Sensor Noise Analysis")
print("="*60)

# Analyze all samples
with h5py.File("data/dataset.h5", "r") as f:
    all_force_data = []

    for sample_key in f.keys():
        g = f[sample_key]
        seq_len = g.attrs["sequence_length"]
        sensor_data = g["sensor_data"][:seq_len]
        force = sensor_data[:, 6]  # FSR is 7th column

        all_force_data.append(force)

        print(f"\n{sample_key}:")
        print(f"  Length: {seq_len} samples ({seq_len * 0.02:.1f}s)")
        print(f"  Force range: [{force.min():.2f}, {force.max():.2f}]")
        print(f"  Force mean: {force.mean():.2f}")
        print(f"  Force std: {force.std():.2f}")
        print(f"  SNR: {force.mean() / force.std():.2f}")

        # Check static period (first 50 samples)
        static_force = force[:50]
        print(f"  Static period force std: {static_force.std():.2f}")

        # Calculate sample-to-sample differences (high-frequency noise)
        force_diff = np.abs(np.diff(force))
        print(f"  Sample-to-sample change mean: {force_diff.mean():.2f}")
        print(f"  Sample-to-sample change max: {force_diff.max():.2f}")

        # Check for outliers/spikes
        outliers = force_diff > (force_diff.mean() + 3 * force_diff.std())
        print(f"  Outlier spikes (>3σ): {outliers.sum()} ({outliers.sum()/len(outliers)*100:.1f}%)")

print()
print("="*60)
print("Overall Statistics")
print("="*60)

all_force = np.concatenate(all_force_data)
print(f"Total samples: {len(all_force)}")
print(f"Global force range: [{all_force.min():.2f}, {all_force.max():.2f}]")
print(f"Global force mean: {all_force.mean():.2f}")
print(f"Global force std: {all_force.std():.2f}")

# Plot force histogram
plt.figure(figsize=(10, 6))
plt.hist(all_force, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Force (FSR units)')
plt.ylabel('Count')
plt.title('Force Sensor Distribution Across All Samples')
plt.grid(True, alpha=0.3)
plt.savefig('tests/force_histogram.png', dpi=150, bbox_inches='tight')
print(f"\nSaved histogram to tests/force_histogram.png")

print()
print("="*60)
print("Analysis")
print("="*60)
print("Force sensor noise affects ZUPT detection:")
print("  - High noise → large force variance → ZUPT fails to trigger")
print("  - Spikes → false ZUPT detections or misses")
print()
print("Expected for good FSR:")
print("  - SNR > 10 (signal-to-noise ratio)")
print("  - Static period std < 10% of mean")
print("  - Sample-to-sample change < 5% of mean")
print("="*60)
