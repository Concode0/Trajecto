"""
Check if accelerometer bias estimation during static period would help.

During static period, after gravity removal, any residual acceleration
could be accelerometer bias that should be subtracted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.config import Config
from model.rotation_utils import quaternion_from_two_vectors, quaternion_to_rotation_matrix

print("="*60)
print("Accelerometer Bias Estimation Test")
print("="*60)

# Load real data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    sensor_data = g["sensor_data"][:100]

print(f"\nSample: {sample_key}")
print("Analyzing first 100 samples (2 seconds @ 50Hz)")
print()

# Initialize like pure_eskf
static_samples = 50
avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)

print("=== Initial State ===")
print(f"Averaged accel (body frame): {avg_accel_b[0].numpy()}")
print(f"Magnitude: {torch.norm(avg_accel_b).item():.6f} m/s²")
print(f"Gyro bias: {gyro_bias_b[0].numpy()}")
print()

# Use measured gravity magnitude
measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
world_gravity = torch.cat([torch.zeros(1, 2), measured_gravity_magnitude], dim=-1)

# Get quaternion
quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)
rot_mat = quaternion_to_rotation_matrix(quat)

print("=== Static Period Analysis ===")

# For each sample in static period, calculate residual acceleration
residuals = []
for t in range(static_samples):
    accel_b = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)

    # Rotate to world frame and remove gravity
    accel_w = (rot_mat @ accel_b.unsqueeze(-1)).squeeze(-1) - world_gravity
    residuals.append(accel_w[0].numpy())

residuals = np.array(residuals)

print(f"Residual acceleration after gravity removal:")
print(f"  Mean:     {residuals.mean(axis=0)}")
print(f"  Std:      {residuals.std(axis=0)}")
print(f"  Magnitude mean: {np.linalg.norm(residuals, axis=1).mean():.6f} m/s²")
print(f"  Magnitude std:  {np.linalg.norm(residuals, axis=1).std():.6f} m/s²")
print()

# Estimate accelerometer bias as the mean residual during static
estimated_accel_bias_w = residuals.mean(axis=0)
print(f"Estimated accelerometer bias (world frame): {estimated_accel_bias_w}")
print()

# Calculate theoretical drift reduction
dt = Config.DT
t = 1.0  # 1 second
drift_without_bias = 0.5 * np.linalg.norm(estimated_accel_bias_w) * t**2
print(f"Theoretical drift from bias over 1s: {drift_without_bias*1000:.2f} mm")
print()

# Rotate bias back to body frame
rot_mat_w_to_b = rot_mat.transpose(-2, -1)
estimated_accel_bias_b = (rot_mat_w_to_b @ torch.from_numpy(estimated_accel_bias_w).float().unsqueeze(-1)).squeeze(-1)
print(f"Estimated accelerometer bias (body frame): {estimated_accel_bias_b.numpy()}")
print()

# Test: Apply bias correction
print("=== With Bias Correction ===")
residuals_corrected = []
for t in range(static_samples):
    accel_b = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)

    # Apply bias correction in body frame BEFORE rotation
    accel_b_corrected = accel_b - estimated_accel_bias_b.unsqueeze(0)

    # Rotate to world frame and remove gravity
    accel_w = (rot_mat @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - world_gravity
    residuals_corrected.append(accel_w[0].numpy())

residuals_corrected = np.array(residuals_corrected)

print(f"Residual acceleration after bias correction:")
print(f"  Mean:     {residuals_corrected.mean(axis=0)}")
print(f"  Std:      {residuals_corrected.std(axis=0)}")
print(f"  Magnitude mean: {np.linalg.norm(residuals_corrected, axis=1).mean():.6f} m/s²")
print(f"  Magnitude std:  {np.linalg.norm(residuals_corrected, axis=1).std():.6f} m/s²")
print()

# Compare
improvement = (np.linalg.norm(residuals, axis=1).mean() -
               np.linalg.norm(residuals_corrected, axis=1).mean())
print(f"Improvement in residual magnitude: {improvement:.6f} m/s²")
print(f"Drift reduction over 1s: {0.5 * improvement * t**2 * 1000:.2f} mm")

print()
print("="*60)
print("Conclusion")
print("="*60)
print("If bias correction significantly reduces residual acceleration,")
print("we should initialize accelerometer bias from static period.")
print("This would reduce the 8.66mm static drift even further.")
print("="*60)
