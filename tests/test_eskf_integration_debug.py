"""
Deep dive into ESKF integration to find why velocity explodes.

Instrument ESKF to log every step of the integration process.
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
print("ESKF Integration Debug Trace")
print("="*60)

# Load data - full sequence
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Analyzing full sequence: {seq_len} steps ({seq_len * Config.DT:.1f}s)")
print()

# Initialize exactly like pure_eskf
static_samples = 50
avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
world_gravity = torch.cat([torch.zeros(1, 2), measured_gravity_magnitude], dim=-1)
gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)

quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)

# Create ESKF
eskf = ErrorStateKalmanFilter(
    dt=Config.DT,
    device="cpu",
    use_zupt=False,  # Disable to isolate integration
    use_tcn_zupt=False,
    use_virtual_measurements=False
)
eskf.gravity_w = world_gravity[0]

pos = torch.zeros(1, 3)
vel = torch.zeros(1, 3)
accel_bias = torch.zeros(1, 3)
P = torch.eye(15).unsqueeze(0) * 1e-4

print("Initial state:")
print(f"  Measured gravity: {measured_gravity_magnitude.item():.6f} m/s²")
print(f"  Gyro bias: {gyro_bias_b[0].numpy()}")
print(f"  Initial quat: {quat[0].numpy()}")
print()

# Track suspicious growth
print("Integration trace (showing every 10th step):")
print(f"{'Step':>4} {'GT_vel':>15} {'Pred_vel':>15} {'Vel_ratio':>10} {'GT_pos':>15} {'Pred_pos':>15} {'Pos_ratio':>10}")
print("-"*100)

for t in range(seq_len):
    accel_raw = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)
    gyro_raw = torch.from_numpy(sensor_data[t, 3:6]).float().unsqueeze(0)
    force_raw = torch.from_numpy(sensor_data[t, 6:7]).float().unsqueeze(0)
    measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

    # Get state BEFORE update
    pos_before = pos.clone()
    vel_before = vel.clone()
    quat_before = quat.clone()

    # Forward pass
    pos, vel, quat, gyro_bias_b, accel_bias, P, tcn_features = eskf.forward(
        pos, vel, quat, gyro_bias_b, accel_bias, P,
        gyro_raw, accel_raw, force_raw, measurement
    )

    # Calculate changes
    pos_delta = pos - pos_before
    vel_delta = vel - vel_before

    # Compare with GT
    gt_vel_mag = np.linalg.norm(gt_vel[t])
    pred_vel_mag = torch.norm(vel).item()
    vel_ratio = pred_vel_mag / (gt_vel_mag + 1e-9)

    gt_pos_mag = np.linalg.norm(gt_pos[t] - gt_pos[0])
    pred_pos_mag = torch.norm(pos).item()
    pos_ratio = pred_pos_mag / (gt_pos_mag + 1e-9)

    if t % 10 == 0 or t < 5:
        print(f"{t:4d} {gt_vel_mag:15.6f} {pred_vel_mag:15.6f} {vel_ratio:10.2f} {gt_pos_mag:15.6f} {pred_pos_mag:15.6f} {pos_ratio:10.2f}")

    # Check for anomalies
    if pred_vel_mag > 1.0:  # Velocity exceeds 1 m/s (unreasonable for handwriting)
        print(f"\n⚠ ANOMALY at step {t}: Velocity {pred_vel_mag:.3f} m/s (should be <1.0)")
        print(f"  Accel (body frame): {accel_raw[0].numpy()}")
        print(f"  Gyro (body frame): {gyro_raw[0].numpy()}")
        print(f"  Quaternion: {quat[0].numpy()}")

        # Calculate what acceleration in world frame is
        rot_mat = quaternion_to_rotation_matrix(quat)
        accel_world = (rot_mat @ accel_raw.unsqueeze(-1)).squeeze(-1) - world_gravity.unsqueeze(0)
        print(f"  Accel (world frame, after gravity removal): {accel_world[0].numpy()}")
        print(f"  Velocity delta this step: {vel_delta[0].numpy()}")
        print(f"  Expected delta (dt * accel): {(Config.DT * accel_world[0]).numpy()}")

        # Check if quaternion is still normalized
        quat_norm = torch.norm(quat)
        print(f"  Quaternion norm: {quat_norm.item():.6f} (should be 1.0)")

        if abs(quat_norm.item() - 1.0) > 0.01:
            print(f"  ❌ QUATERNION NOT NORMALIZED!")

        print()

print("\n" + "="*60)
print("Analysis Summary")
print("="*60)

# Final statistics
final_vel_mag = torch.norm(vel).item()
final_pos_mag = torch.norm(pos).item()
final_gt_vel_mag = np.linalg.norm(gt_vel[seq_len-1])
final_gt_pos_mag = np.linalg.norm(gt_pos[seq_len-1] - gt_pos[0])

print(f"\nFinal state (step {seq_len-1}):")
print(f"  Predicted velocity: {final_vel_mag:.3f} m/s")
print(f"  GT velocity: {final_gt_vel_mag:.3f} m/s")
print(f"  Velocity ratio: {final_vel_mag / (final_gt_vel_mag + 1e-9):.1f}x")
print()
print(f"  Predicted position: {final_pos_mag:.3f} m")
print(f"  GT position: {final_gt_pos_mag:.3f} m")
print(f"  Position ratio: {final_pos_mag / (final_gt_pos_mag + 1e-9):.1f}x")

print()
print("Key Observations:")
print("  1. If velocity ratio stays close to 1.0, integration is correct")
print("  2. If velocity ratio grows exponentially, there's a bug in integration")
print("  3. If quaternion norm != 1.0, quaternion update has a bug")
print("  4. If accel_world has large values, gravity removal is wrong")
print("="*60)
