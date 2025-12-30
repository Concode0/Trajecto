"""
Check if ZUPT is actually triggering during trajectory.

The scale error grows from 1.7x at 2s to 10.7x at 4s, suggesting
ZUPT is not working properly during writing motion.
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
print("ZUPT Activity Test")
print("="*60)

# Load full sample
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Total sequence length: {seq_len} ({seq_len * Config.DT:.1f}s)")
print()

# Initialize ESKF with ZUPT enabled
static_samples = 50
avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
world_gravity = torch.cat([torch.zeros(1, 2), measured_gravity_magnitude], dim=-1)

gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)
quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)

# Create ESKF with ZUPT ENABLED
eskf = ErrorStateKalmanFilter(
    dt=Config.DT,
    device="cpu",
    use_zupt=True,  # ← ENABLED
    use_tcn_zupt=False,
    use_virtual_measurements=False
)
eskf.gravity_w = world_gravity[0]

pos = torch.zeros(1, 3)
vel = torch.zeros(1, 3)
accel_bias = torch.zeros(1, 3)
P = torch.eye(15).unsqueeze(0) * 1e-4

# Track ZUPT activity
zupt_activations = []
velocities = []
positions = []

print("=== ESKF Integration with ZUPT ===")
print(f"ZUPT Configuration:")
print(f"  ZUPT_WINDOW_SIZE: {Config.ZUPT_WINDOW_SIZE}")
print(f"  ZUPT_ACCEL_THRESHOLD: {Config.ZUPT_ACCEL_THRESHOLD}")
print(f"  ZUPT_FORCE_VAR_THRESHOLD: {Config.ZUPT_FORCE_VAR_THRESHOLD}")
print(f"  ZUPT_FORCE_DELTA_THRESHOLD: {Config.ZUPT_FORCE_DELTA_THRESHOLD}")
print()

for t in range(seq_len):
    accel_raw = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)
    gyro_raw = torch.from_numpy(sensor_data[t, 3:6]).float().unsqueeze(0)
    force_raw = torch.from_numpy(sensor_data[t, 6:7]).float().unsqueeze(0)
    measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

    # Forward pass
    pos, vel, quat, gyro_bias_b, accel_bias, P, tcn_features = eskf.forward(
        pos, vel, quat, gyro_bias_b, accel_bias, P,
        gyro_raw, accel_raw, force_raw, measurement
    )

    # Check if ZUPT was applied (tcn_features["zupt_flag"] is shape (batch, 1))
    zupt_flag = tcn_features.get("zupt_flag", torch.tensor([[0.0]]))
    zupt_applied = (zupt_flag[0, 0].item() > 0.5)  # Convert to boolean
    zupt_activations.append(zupt_applied)
    velocities.append(vel[0].detach().numpy().copy())
    positions.append(pos[0].detach().numpy().copy())

zupt_activations = np.array(zupt_activations)
velocities = np.array(velocities)
positions = np.array(positions)

# Analysis
zupt_rate = zupt_activations.sum() / len(zupt_activations) * 100
print(f"ZUPT Activation Rate: {zupt_rate:.2f}% ({zupt_activations.sum()}/{len(zupt_activations)} steps)")
print()

if zupt_rate < 1.0:
    print("❌ WARNING: ZUPT activation rate is very low!")
    print("This means drift is not being corrected.")
    print()

# Find ZUPT regions
zupt_regions = []
in_region = False
start = 0
for i, active in enumerate(zupt_activations):
    if active and not in_region:
        start = i
        in_region = True
    elif not active and in_region:
        zupt_regions.append((start, i-1))
        in_region = False
if in_region:
    zupt_regions.append((start, len(zupt_activations)-1))

print(f"Number of ZUPT regions: {len(zupt_regions)}")
if zupt_regions:
    print("First 10 ZUPT regions:")
    for i, (start, end) in enumerate(zupt_regions[:10]):
        duration = (end - start + 1) * Config.DT
        print(f"  Region {i+1}: steps {start}-{end} ({duration:.2f}s)")
print()

# Check velocity during ZUPT vs non-ZUPT
vel_mag = np.linalg.norm(velocities, axis=1)
vel_during_zupt = vel_mag[zupt_activations]
vel_no_zupt = vel_mag[~zupt_activations]

if len(vel_during_zupt) > 0:
    print(f"Velocity magnitude during ZUPT:")
    print(f"  Mean: {vel_during_zupt.mean():.6f} m/s")
    print(f"  Max:  {vel_during_zupt.max():.6f} m/s")
    print()

if len(vel_no_zupt) > 0:
    print(f"Velocity magnitude during motion (no ZUPT):")
    print(f"  Mean: {vel_no_zupt.mean():.6f} m/s")
    print(f"  Max:  {vel_no_zupt.max():.6f} m/s")
    print()

# Check scale at different time points
checkpoints = [100, 200, 400, seq_len-1]
print("Scale progression:")
for cp in checkpoints:
    if cp >= seq_len:
        continue
    gt_disp = np.linalg.norm(gt_pos[cp] - gt_pos[0])
    pred_disp = np.linalg.norm(positions[cp] - positions[0])
    scale = pred_disp / (gt_disp + 1e-9)
    time_s = cp * Config.DT
    print(f"  t={time_s:4.1f}s (step {cp:3d}): GT={gt_disp*100:6.2f}cm, "
          f"Pred={pred_disp*100:6.2f}cm, Scale={scale:.2f}x")

print()
print("="*60)
print("Analysis")
print("="*60)
print("If ZUPT activation rate is low (<10%), the thresholds are too strict.")
print("Common issues:")
print("  - ZUPT_FORCE_VAR_THRESHOLD too small (should be ~1500-5000)")
print("  - ZUPT_FORCE_DELTA_THRESHOLD too small (should be ~100-200)")
print("  - ZUPT_ACCEL_THRESHOLD too strict")
print("="*60)
