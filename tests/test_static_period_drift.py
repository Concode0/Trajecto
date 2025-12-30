"""
Investigate drift during static period.

The scale error peaks at step 50 (~1s) then decreases. This corresponds to
the end of the static initialization period. This test checks:
1. Is the static period truly static in GT data?
2. Is ESKF accumulating drift during static period?
3. Why does scale error peak then decrease?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.ESKF import ErrorStateKalmanFilter
from model.config import Config
from model.rotation_utils import quaternion_from_two_vectors, quaternion_to_rotation_matrix

print("="*60)
print("Static Period Drift Analysis")
print("="*60)

# Load real data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    sensor_data = g["sensor_data"][:200]  # Get more data
    gt_pos = g["gt_pos_data"][:200]
    gt_vel = g["gt_vel_data"][:200]

print(f"\nSample: {sample_key}")
print(f"Analyzing first 200 samples (4 seconds @ 50Hz)")
print()

# Check GT movement during static period
print("=== GT Movement Analysis ===")
print("First 50 samples (static initialization period):")
gt_pos_static = gt_pos[:50]
gt_vel_static = gt_vel[:50]

# Calculate total displacement
disp_static = np.linalg.norm(gt_pos_static[-1] - gt_pos_static[0])
print(f"  Total displacement: {disp_static:.6f} m ({disp_static*1000:.2f} mm)")

# Calculate velocity magnitude
vel_mag_static = np.linalg.norm(gt_vel_static, axis=1)
print(f"  Mean velocity magnitude: {vel_mag_static.mean():.6f} m/s")
print(f"  Max velocity magnitude:  {vel_mag_static.max():.6f} m/s")

# Check if truly static
if disp_static < 0.001 and vel_mag_static.mean() < 0.005:
    print("  ✓ GT appears static during initialization period")
else:
    print(f"  ⚠ WARNING: GT shows movement during 'static' period!")

print()

print("Samples 50-100 (start of writing):")
gt_pos_writing = gt_pos[50:100]
disp_writing = np.linalg.norm(gt_pos_writing[-1] - gt_pos_writing[0])
vel_mag_writing = np.linalg.norm(gt_vel[50:100], axis=1)
print(f"  Total displacement: {disp_writing:.6f} m ({disp_writing*100:.2f} cm)")
print(f"  Mean velocity magnitude: {vel_mag_writing.mean():.6f} m/s")
print()

# Initialize ESKF exactly like pure_eskf
print("=== ESKF Initialization (like pure_eskf) ===")
static_samples = 50
avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)
accel_bias = torch.zeros(1, 3)

print(f"Averaged accel (body frame): {avg_accel_b[0].numpy()}")
print(f"Magnitude: {torch.norm(avg_accel_b).item():.6f} m/s²")
print(f"Gyro bias: {gyro_bias_b[0].numpy()}")
print()

# Check gravity alignment - USE MEASURED MAGNITUDE
measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
print(f"Measured gravity magnitude: {measured_gravity_magnitude.item():.6f} m/s²")

world_gravity = torch.cat([
    torch.zeros(1, 2),
    measured_gravity_magnitude
], dim=-1)
print(f"World gravity vector: {world_gravity[0].numpy()}")

quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)
print(f"Initial quaternion: {quat[0].numpy()}")

# Verify rotation
rot_mat = quaternion_to_rotation_matrix(quat)
accel_world = (rot_mat @ avg_accel_b.unsqueeze(-1)).squeeze(-1)
print(f"Accel rotated to world frame: {accel_world[0].numpy()}")
print(f"After gravity removal: {(accel_world - world_gravity)[0].numpy()}")
print()

# Initialize ESKF
eskf = ErrorStateKalmanFilter(
    dt=Config.DT,
    device="cpu",
    use_zupt=False,
    use_tcn_zupt=False,
    use_virtual_measurements=False
)

# CRITICAL: Update ESKF's gravity vector to use measured magnitude
eskf.gravity_w = world_gravity[0]
print(f"ESKF gravity_w updated to: {eskf.gravity_w.numpy()}")
print()

pos = torch.zeros(1, 3)
vel = torch.zeros(1, 3)
P = torch.eye(15).unsqueeze(0) * 1e-4

# Run integration with detailed logging
print("=== ESKF Integration ===")
print("During static period (samples 0-50):")

positions = []
velocities = []

for t in range(200):
    accel_raw = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)
    gyro_raw = torch.from_numpy(sensor_data[t, 3:6]).float().unsqueeze(0)
    force_raw = torch.from_numpy(sensor_data[t, 6:7]).float().unsqueeze(0)
    measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

    pos, vel, quat, gyro_bias_b, accel_bias, P, _ = eskf.forward(
        pos, vel, quat, gyro_bias_b, accel_bias, P,
        gyro_raw, accel_raw, force_raw, measurement
    )

    positions.append(pos[0].detach().numpy().copy())
    velocities.append(vel[0].detach().numpy().copy())

    # Log at key points
    if t in [0, 10, 20, 30, 40, 49]:
        gt_disp = np.linalg.norm(gt_pos[t] - gt_pos[0])
        pred_disp = np.linalg.norm(positions[t] - positions[0])
        scale = pred_disp / (gt_disp + 1e-9)
        print(f"  Step {t:2d}: GT_disp={gt_disp*1000:.2f}mm, Pred_disp={pred_disp*1000:.2f}mm, "
              f"Vel_mag={np.linalg.norm(velocities[t]):.4f}m/s, Scale={scale:.1f}x")

print()
print("During writing (samples 50-100):")

for t in [50, 60, 70, 80, 90, 99]:
    gt_disp = np.linalg.norm(gt_pos[t] - gt_pos[0])
    pred_disp = np.linalg.norm(positions[t] - positions[0])
    scale = pred_disp / (gt_disp + 1e-9)
    print(f"  Step {t:2d}: GT_disp={gt_disp*1000:.2f}mm, Pred_disp={pred_disp*1000:.2f}mm, Scale={scale:.1f}x")

print()
print("Extended period (samples 100-200):")

for t in [100, 120, 150, 199]:
    gt_disp = np.linalg.norm(gt_pos[t] - gt_pos[0])
    pred_disp = np.linalg.norm(positions[t] - positions[0])
    scale = pred_disp / (gt_disp + 1e-9)
    print(f"  Step {t:3d}: GT_disp={gt_disp*1000:.2f}mm, Pred_disp={pred_disp*1000:.2f}mm, Scale={scale:.1f}x")

positions = np.array(positions)
velocities = np.array(velocities)

print()
print("=== Drift Accumulation During Static ===")
static_drift = np.linalg.norm(positions[49] - positions[0])
print(f"ESKF drift after 50 steps (1s): {static_drift:.6f} m ({static_drift*1000:.2f} mm)")
print(f"GT movement after 50 steps: {disp_static:.6f} m ({disp_static*1000:.2f} mm)")
print(f"Excess drift: {(static_drift - disp_static)*1000:.2f} mm")

# Velocity during static
vel_mag_eskf_static = np.linalg.norm(velocities[:50], axis=1)
print(f"\nESKF velocity during static:")
print(f"  Mean: {vel_mag_eskf_static.mean():.6f} m/s")
print(f"  Max:  {vel_mag_eskf_static.max():.6f} m/s")
print(f"  Final (step 49): {vel_mag_eskf_static[49]:.6f} m/s")

print()
print("="*60)
print("Analysis")
print("="*60)
print("If ESKF accumulates significant drift (>5mm) during static period,")
print("while GT barely moves (<1mm), this causes the scale error peak.")
print()
print("Possible causes:")
print("1. Accelerometer bias not fully removed")
print("2. Small residual gravity misalignment accumulating")
print("3. IMU noise integration (should be small with Q matrix)")
print("4. Static period has actual small movements in real data")
print("="*60)
