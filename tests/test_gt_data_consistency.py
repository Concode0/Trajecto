"""
Test ground truth data consistency.

Check if GT positions and velocities are consistent.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*60)
print("Ground Truth Data Consistency Test")
print("="*60)

# Load data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]

    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len] if "gt_vel_data" in g else None

print(f"\nSample: {sample_key}")
print(f"Sequence length: {seq_len}")
print(f"DT from Config: {Config.DT:.6f} s ({1/Config.DT:.2f} Hz)")
print()

# Check 1: Verify GT velocity matches position derivative
print("=== Check 1: GT velocity consistency ===")

if gt_vel is not None:
    # Calculate velocity from position diff
    dt = Config.DT
    pos_diff = np.diff(gt_pos, axis=0)
    vel_from_diff = pos_diff / dt

    # Compare with stored GT velocity (skip first sample which has no derivative)
    vel_error = vel_from_diff - gt_vel[1:]
    vel_error_mag = np.linalg.norm(vel_error, axis=1)

    print(f"First 10 samples:")
    print(f"  GT velocity (stored):    {gt_vel[1]}")
    print(f"  Velocity from diff:      {vel_from_diff[0]}")
    print(f"  Error:                   {vel_error[0]}")
    print(f"  Error magnitude:         {vel_error_mag[0]:.6f} m/s")
    print()

    print(f"Overall statistics:")
    print(f"  Mean velocity error: {vel_error_mag.mean():.6f} m/s")
    print(f"  Max velocity error:  {vel_error_mag.max():.6f} m/s")
    print(f"  Median error:        {np.median(vel_error_mag):.6f} m/s")

    if vel_error_mag.mean() > 0.01:
        print(f"  ❌ INCONSISTENT: GT velocity doesn't match position derivative!")
    else:
        print(f"  ✓ CONSISTENT: GT velocity matches position derivative")
else:
    print("  No GT velocity data found")

print()

# Check 2: Verify sensor data units
print("=== Check 2: Sensor data units ===")

# Check gravity magnitude in static period
static_period = sensor_data[:50, :3]
gravity_mags = np.linalg.norm(static_period, axis=1)

print(f"Static period (first 50 samples):")
print(f"  Gravity magnitude mean: {gravity_mags.mean():.6f} m/s²")
print(f"  Gravity magnitude std:  {gravity_mags.std():.6f} m/s²")
print(f"  Expected: 9.81 m/s²")
print(f"  Mismatch: {gravity_mags.mean() - 9.81:.6f} m/s²")

if abs(gravity_mags.mean() - 9.81) < 0.1:
    print(f"  ✓ Sensor units appear correct (m/s²)")
else:
    print(f"  ❌ WARNING: Gravity mismatch - sensor units may be wrong!")

print()

# Check 3: Verify GT position units
print("=== Check 3: GT position units and scale ===")

gt_range_x = gt_pos[:, 0].max() - gt_pos[:, 0].min()
gt_range_y = gt_pos[:, 1].max() - gt_pos[:, 1].min()
gt_range_z = gt_pos[:, 2].max() - gt_pos[:, 2].min()

print(f"GT position range:")
print(f"  X: {gt_range_x:.6f} m ({gt_range_x*100:.2f} cm)")
print(f"  Y: {gt_range_y:.6f} m ({gt_range_y*100:.2f} cm)")
print(f"  Z: {gt_range_z:.6f} m ({gt_range_z*100:.2f} cm)")
print()

# Typical handwriting is 5-20cm
if 0.05 < gt_range_x < 0.3 and 0.05 < gt_range_y < 0.3:
    print(f"  ✓ GT position range is reasonable for handwriting (5-30cm)")
else:
    print(f"  ❌ WARNING: GT position range seems unusual!")
    if gt_range_x < 0.01:
        print(f"     Could GT be in different units (pixels, mm, etc.)?")

print()

# Check 4: Calculate theoretical position from IMU integration
print("=== Check 4: Compare IMU integration vs GT ===")

# Simple forward integration for first 100 samples
accel_data = sensor_data[:100, :3]
dt = Config.DT

# Assume gravity is along Z and subtract it
accel_corrected = accel_data.copy()
accel_corrected[:, 2] -= 9.81

# Integrate
vel_integrated = np.zeros(3)
pos_integrated = np.zeros(3)
positions_integrated = []

for t in range(100):
    vel_integrated += accel_corrected[t] * dt
    pos_integrated += vel_integrated * dt
    positions_integrated.append(pos_integrated.copy())

positions_integrated = np.array(positions_integrated)

print(f"After 100 steps (2 seconds):")
print(f"  GT position:         {gt_pos[99]}")
print(f"  Integrated position: {positions_integrated[99]}")
print(f"  Ratio: {np.linalg.norm(positions_integrated[99]) / np.linalg.norm(gt_pos[99]):.2f}x")

print()
print("="*60)
print("Summary")
print("="*60)
print(f"If velocity is consistent and sensor units are correct,")
print(f"but integrated position is >>100x larger than GT,")
print(f"then the issue is likely:")
print(f"  1. Missing ZUPT corrections")
print(f"  2. Incorrect quaternion/gravity removal")
print(f"  3. DT mismatch")
print("="*60)
