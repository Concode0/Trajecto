"""
Test ESKF with virtual measurements enabled to see if they prevent quaternion drift.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.ESKF import ErrorStateKalmanFilter
from model.config import Config
from model.rotation_utils import quaternion_from_two_vectors

print("="*60)
print("ESKF with Virtual Measurements Test")
print("="*60)

# Load full sequence
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Sequence length: {seq_len} steps ({seq_len * Config.DT:.1f}s)")
print()

# Initialize
static_samples = 50
avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
world_gravity = torch.cat([torch.zeros(1, 2), measured_gravity_magnitude], dim=-1)
gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)
quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)

print("Configuration comparison:")
print("-"*60)

results = {}

for config_name, use_virtual, use_zupt in [
    ("ZUPT only", False, True),
    ("Virtual only", True, False),
    ("ZUPT + Virtual", True, True),
]:
    # Create ESKF
    eskf = ErrorStateKalmanFilter(
        dt=Config.DT,
        device="cpu",
        use_zupt=use_zupt,
        use_tcn_zupt=False,
        use_virtual_measurements=use_virtual
    )
    eskf.gravity_w = world_gravity[0]

    pos = torch.zeros(1, 3)
    vel = torch.zeros(1, 3)
    accel_bias = torch.zeros(1, 3)
    P = torch.eye(15).unsqueeze(0) * 1e-4
    quat_local = quat.clone()
    gyro_bias_local = gyro_bias_b.clone()

    # Run full sequence
    for t in range(seq_len):
        accel_raw = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)
        gyro_raw = torch.from_numpy(sensor_data[t, 3:6]).float().unsqueeze(0)
        force_raw = torch.from_numpy(sensor_data[t, 6:7]).float().unsqueeze(0)
        measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

        pos, vel, quat_local, gyro_bias_local, accel_bias, P, tcn_features = eskf.forward(
            pos, vel, quat_local, gyro_bias_local, accel_bias, P,
            gyro_raw, accel_raw, force_raw, measurement
        )

    # Final statistics
    final_vel_mag = torch.norm(vel).item()
    final_pos_mag = torch.norm(pos).item()
    final_gt_vel_mag = np.linalg.norm(gt_vel[seq_len-1])
    final_gt_pos_mag = np.linalg.norm(gt_pos[seq_len-1] - gt_pos[0])

    vel_ratio = final_vel_mag / (final_gt_vel_mag + 1e-9)
    pos_ratio = final_pos_mag / (final_gt_pos_mag + 1e-9)

    results[config_name] = {
        'vel': final_vel_mag,
        'pos': final_pos_mag,
        'vel_ratio': vel_ratio,
        'pos_ratio': pos_ratio
    }

    print(f"\n{config_name}:")
    print(f"  Final velocity:     {final_vel_mag:.3f} m/s (GT: {final_gt_vel_mag:.3f}, ratio: {vel_ratio:.1f}x)")
    print(f"  Final position:     {final_pos_mag:.3f} m (GT: {final_gt_pos_mag:.3f}, ratio: {pos_ratio:.1f}x)")

print()
print("="*60)
print("Summary")
print("="*60)

best_config = min(results.items(), key=lambda x: x[1]['vel_ratio'])
print(f"\nBest configuration: {best_config[0]}")
print(f"  Velocity ratio: {best_config[1]['vel_ratio']:.1f}x")
print(f"  Position ratio: {best_config[1]['pos_ratio']:.1f}x")

print()
print("Conclusion:")
print("  - If virtual measurements help: quaternion drift is being corrected")
print("  - If ZUPT helps: velocity corrections during static periods work")
print("  - If both needed: drift requires multiple correction strategies")
print("="*60)
