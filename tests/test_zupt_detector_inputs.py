"""
Debug ZUPT detector to see why it's not triggering.

Check actual force variance and delta values in the data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*60)
print("ZUPT Detector Input Analysis")
print("="*60)

# Load data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Sequence length: {seq_len}")
print()

# Extract force (FSR) signal
force = sensor_data[:, 6]  # 7th column is FSR
accel = sensor_data[:, :3]

print("=== Force (FSR) Signal Statistics ===")
print(f"  Mean: {force.mean():.2f}")
print(f"  Std:  {force.std():.2f}")
print(f"  Min:  {force.min():.2f}")
print(f"  Max:  {force.max():.2f}")
print()

# Calculate force variance and delta over sliding windows
window_size = Config.ZUPT_WINDOW_SIZE
force_vars = []
force_deltas = []

for i in range(len(force) - window_size):
    window = force[i:i+window_size]
    force_var = np.var(window)
    force_delta = window.max() - window.min()

    force_vars.append(force_var)
    force_deltas.append(force_delta)

force_vars = np.array(force_vars)
force_deltas = np.array(force_deltas)

print(f"=== Force Variance (window={window_size}) ===")
print(f"  Mean:   {force_vars.mean():.2f}")
print(f"  Median: {np.median(force_vars):.2f}")
print(f"  Min:    {force_vars.min():.2f}")
print(f"  Max:    {force_vars.max():.2f}")
print(f"  5th percentile:  {np.percentile(force_vars, 5):.2f}")
print(f"  95th percentile: {np.percentile(force_vars, 95):.2f}")
print()

print(f"=== Force Delta (window={window_size}) ===")
print(f"  Mean:   {force_deltas.mean():.2f}")
print(f"  Median: {np.median(force_deltas):.2f}")
print(f"  Min:    {force_deltas.min():.2f}")
print(f"  Max:    {force_deltas.max():.2f}")
print(f"  5th percentile:  {np.percentile(force_deltas, 5):.2f}")
print(f"  95th percentile: {np.percentile(force_deltas, 95):.2f}")
print()

# Calculate accelerometer variance
accel_vars = []
for i in range(len(accel) - window_size):
    window = accel[i:i+window_size]
    accel_var = np.var(window, axis=0).mean()  # Mean of variances across 3 axes
    accel_vars.append(accel_var)

accel_vars = np.array(accel_vars)

print(f"=== Accelerometer Variance (window={window_size}) ===")
print(f"  Mean:   {accel_vars.mean():.2f}")
print(f"  Median: {np.median(accel_vars):.2f}")
print(f"  Min:    {accel_vars.min():.2f}")
print(f"  Max:    {accel_vars.max():.2f}")
print(f"  5th percentile:  {np.percentile(accel_vars, 5):.2f}")
print(f"  95th percentile: {np.percentile(accel_vars, 95):.2f}")
print()

# Compare with thresholds
print("=== ZUPT Threshold Comparison ===")
print(f"Config.ZUPT_ACCEL_THRESHOLD = {Config.ZUPT_ACCEL_THRESHOLD}")
print(f"  → {(accel_vars < Config.ZUPT_ACCEL_THRESHOLD).sum()}/{len(accel_vars)} windows pass")
print()

print(f"Config.ZUPT_FORCE_VAR_THRESHOLD = {Config.ZUPT_FORCE_VAR_THRESHOLD}")
print(f"  → {(force_vars < Config.ZUPT_FORCE_VAR_THRESHOLD).sum()}/{len(force_vars)} windows pass")
print()

print(f"Config.ZUPT_FORCE_DELTA_THRESHOLD = {Config.ZUPT_FORCE_DELTA_THRESHOLD}")
print(f"  → {(force_deltas < Config.ZUPT_FORCE_DELTA_THRESHOLD).sum()}/{len(force_deltas)} windows pass")
print()

# Find windows that would pass ALL thresholds
zupt_candidates = (
    (accel_vars < Config.ZUPT_ACCEL_THRESHOLD) &
    (force_vars < Config.ZUPT_FORCE_VAR_THRESHOLD) &
    (force_deltas < Config.ZUPT_FORCE_DELTA_THRESHOLD)
)

print(f"Windows passing ALL thresholds: {zupt_candidates.sum()}/{len(zupt_candidates)}")
print()

if zupt_candidates.sum() == 0:
    print("❌ NO windows pass all thresholds!")
    print()
    print("Failure analysis:")

    # Check each threshold separately
    accel_pass = accel_vars < Config.ZUPT_ACCEL_THRESHOLD
    force_var_pass = force_vars < Config.ZUPT_FORCE_VAR_THRESHOLD
    force_delta_pass = force_deltas < Config.ZUPT_FORCE_DELTA_THRESHOLD

    print(f"  Accel threshold:       {accel_pass.sum()}/{len(accel_pass)} pass ({accel_pass.mean()*100:.1f}%)")
    print(f"  Force var threshold:   {force_var_pass.sum()}/{len(force_var_pass)} pass ({force_var_pass.mean()*100:.1f}%)")
    print(f"  Force delta threshold: {force_delta_pass.sum()}/{len(force_delta_pass)} pass ({force_delta_pass.mean()*100:.1f}%)")
    print()

    # Suggest better thresholds
    print("Suggested thresholds (using 10th percentile as target):")
    print(f"  ZUPT_ACCEL_THRESHOLD: {np.percentile(accel_vars, 10):.2f} (currently {Config.ZUPT_ACCEL_THRESHOLD})")
    print(f"  ZUPT_FORCE_VAR_THRESHOLD: {np.percentile(force_vars, 10):.0f} (currently {Config.ZUPT_FORCE_VAR_THRESHOLD})")
    print(f"  ZUPT_FORCE_DELTA_THRESHOLD: {np.percentile(force_deltas, 10):.0f} (currently {Config.ZUPT_FORCE_DELTA_THRESHOLD})")

else:
    print(f"✓ {zupt_candidates.sum()} windows would trigger ZUPT")
    # Show first few ZUPT candidates
    zupt_indices = np.where(zupt_candidates)[0]
    print(f"\nFirst 10 ZUPT candidate windows:")
    for idx in zupt_indices[:10]:
        time_s = idx * Config.DT
        print(f"  Window starting at step {idx} ({time_s:.2f}s)")

print()
print("="*60)
