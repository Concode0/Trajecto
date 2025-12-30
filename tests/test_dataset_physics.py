"""
Verify dataset physics consistency - check if GT data is physically plausible.

This test validates:
1. GT velocities match position derivatives
2. GT accelerations match velocity derivatives
3. Sensor readings are in expected ranges
4. No unit conversion errors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*80)
print("DATASET PHYSICS CONSISTENCY CHECK")
print("="*80)

# Load validation sample
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]  # [accel_xyz, gyro_xyz, fsr]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# --- Test 1: GT Velocity vs Position Derivative ---
print("="*80)
print("TEST 1: GT Velocity Consistency")
print("="*80)

# Compute velocity from position using finite difference
gt_vel_from_pos = np.diff(gt_pos, axis=0) / Config.DT
gt_vel_provided = gt_vel[:-1]  # Match length

# Compare
vel_diff = gt_vel_from_pos - gt_vel_provided
vel_error_norm = np.linalg.norm(vel_diff, axis=1)
mean_vel_error = np.mean(vel_error_norm)
max_vel_error = np.max(vel_error_norm)

print(f"\nVelocity derivative check:")
print(f"  Mean error: {mean_vel_error:.6f} m/s")
print(f"  Max error:  {max_vel_error:.6f} m/s")
print(f"  GT vel range: {np.linalg.norm(gt_vel, axis=1).min():.4f} to {np.linalg.norm(gt_vel, axis=1).max():.4f} m/s")

if mean_vel_error < 0.01:
    print("  ✅ GT velocity matches position derivative")
else:
    print(f"  ⚠️  WARNING: GT velocity mismatch (error: {mean_vel_error:.6f} m/s)")

# --- Test 2: Check if GT velocity is reasonable ---
print("\n" + "="*80)
print("TEST 2: GT Velocity Magnitude")
print("="*80)

gt_vel_mag = np.linalg.norm(gt_vel, axis=1)
print(f"\nGT Velocity statistics:")
print(f"  Mean: {np.mean(gt_vel_mag):.4f} m/s")
print(f"  Max:  {np.max(gt_vel_mag):.4f} m/s")
print(f"  Min:  {np.min(gt_vel_mag):.4f} m/s")
print(f"  Std:  {np.std(gt_vel_mag):.4f} m/s")

# Handwriting typically moves at 0.01-0.5 m/s
if np.max(gt_vel_mag) > 1.0:
    print("  ⚠️  WARNING: Unrealistically high velocity for handwriting")
else:
    print("  ✅ Velocity range is reasonable for handwriting")

# --- Test 3: GT Acceleration from Velocity ---
print("\n" + "="*80)
print("TEST 3: GT Acceleration from Velocity")
print("="*80)

gt_accel_from_vel = np.diff(gt_vel, axis=0) / Config.DT
gt_accel_mag = np.linalg.norm(gt_accel_from_vel, axis=1)

print(f"\nGT Acceleration (from velocity):")
print(f"  Mean: {np.mean(gt_accel_mag):.4f} m/s²")
print(f"  Max:  {np.max(gt_accel_mag):.4f} m/s²")
print(f"  Min:  {np.min(gt_accel_mag):.4f} m/s²")
print(f"  Std:  {np.std(gt_accel_mag):.4f} m/s²")

# --- Test 4: Sensor Data Ranges ---
print("\n" + "="*80)
print("TEST 4: Sensor Data Ranges")
print("="*80)

accel_b = sensor_data[:, 0:3]  # m/s²
gyro_b = sensor_data[:, 3:6]   # rad/s
fsr = sensor_data[:, 6]        # force

accel_mag = np.linalg.norm(accel_b, axis=1)
gyro_mag = np.linalg.norm(gyro_b, axis=1)

print(f"\nAccelerometer (body frame):")
print(f"  Mean magnitude: {np.mean(accel_mag):.4f} m/s²")
print(f"  Max magnitude:  {np.max(accel_mag):.4f} m/s²")
print(f"  Min magnitude:  {np.min(accel_mag):.4f} m/s²")
print(f"  Expected: ~9.81 m/s² when static (gravity)")

if np.abs(np.mean(accel_mag) - 9.81) < 1.0:
    print("  ✅ Accelerometer magnitude reasonable (near gravity)")
else:
    print(f"  ⚠️  WARNING: Accelerometer magnitude off by {np.abs(np.mean(accel_mag) - 9.81):.2f} m/s²")

print(f"\nGyroscope (body frame):")
print(f"  Mean magnitude: {np.mean(gyro_mag):.4f} rad/s")
print(f"  Max magnitude:  {np.max(gyro_mag):.4f} rad/s")
print(f"  Min magnitude:  {np.min(gyro_mag):.4f} rad/s")
print(f"  Expected: <10 rad/s for handwriting")

if np.max(gyro_mag) < 10.0:
    print("  ✅ Gyroscope magnitude reasonable")
else:
    print("  ⚠️  WARNING: Very high angular velocity detected")

print(f"\nForce Sensor:")
print(f"  Mean: {np.mean(fsr):.2f}")
print(f"  Max:  {np.max(fsr):.2f}")
print(f"  Min:  {np.min(fsr):.2f}")

# --- Test 5: Check for unrealistic GT trajectory scale ---
print("\n" + "="*80)
print("TEST 5: GT Trajectory Scale")
print("="*80)

gt_displacement = np.linalg.norm(gt_pos[-1] - gt_pos[0])
gt_path_length = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
gt_std = np.std(gt_pos - gt_pos.mean(axis=0))

print(f"\nGT Position statistics:")
print(f"  Start position: {gt_pos[0]}")
print(f"  End position:   {gt_pos[-1]}")
print(f"  Displacement:   {gt_displacement:.4f} m")
print(f"  Path length:    {gt_path_length:.4f} m")
print(f"  Std (centered): {gt_std:.4f} m")

# Handwriting typically spans 0.05-0.15m (5-15cm)
if gt_std > 0.5:
    print("  ⚠️  WARNING: Trajectory very large for handwriting (>50cm spread)")
elif gt_std < 0.01:
    print("  ⚠️  WARNING: Trajectory very small (<1cm spread)")
else:
    print("  ✅ Trajectory scale reasonable for handwriting")

# --- Test 6: Check timestep used for GT velocity ---
print("\n" + "="*80)
print("TEST 6: GT Velocity Timestep Check")
print("="*80)

# The mismatch in Test 1 suggests GT velocity might use different dt
# Try different timesteps to see which one matches best
print(f"\nTesting different timesteps for GT velocity computation:")
print(f"Config.DT = {Config.DT:.9f} s ({1/Config.DT:.2f} Hz)\n")

test_dts = [Config.DT, 0.020, 1/50.0, 1/60.0, 1/240.0]
for test_dt in test_dts:
    vel_test = np.diff(gt_pos, axis=0) / test_dt
    error = np.mean(np.linalg.norm(vel_test - gt_vel[:-1], axis=1))
    print(f"  dt={test_dt:.9f} ({1/test_dt:6.1f} Hz): error = {error:.6f} m/s")

print("\nIf error is lowest at Config.DT, then GT velocity is consistent.")
print("If error is lower at a different dt, GT velocity uses wrong timestep!")

# --- Summary ---
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nDataset validation complete. Check for any ⚠️ warnings above.")
print()
