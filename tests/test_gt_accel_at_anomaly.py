"""
Check what the GT acceleration should be at the anomaly points.

If ESKF predicts 3.6 m/s² but GT shows 0.1 m/s², then there's a problem.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*60)
print("GT Acceleration Analysis at Anomaly Points")
print("="*60)

# Load data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]

    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

print(f"\nSample: {sample_key}")
print()

# Calculate GT acceleration from velocity
dt = Config.DT
gt_accel = np.gradient(gt_vel, dt, axis=0)

# Check critical points where ESKF showed large accelerations
critical_steps = [91, 92, 93, 94, 95, 100, 200, 400, 600, 770]

print(f"{'Step':>4} {'GT_vel':>25} {'GT_vel_mag':>12} {'GT_accel':>25} {'GT_accel_mag':>14}")
print("-"*110)

for step in critical_steps:
    if step >= seq_len:
        continue

    vel = gt_vel[step]
    vel_mag = np.linalg.norm(vel)

    accel = gt_accel[step]
    accel_mag = np.linalg.norm(accel)

    print(f"{step:4d} [{vel[0]:7.4f}, {vel[1]:7.4f}, {vel[2]:7.4f}] {vel_mag:12.4f} "
          f"[{accel[0]:7.4f}, {accel[1]:7.4f}, {accel[2]:7.4f}] {accel_mag:14.4f}")

print()
print("="*60)
print("Analysis")
print("="*60)
print("If GT acceleration is small (< 0.5 m/s²) but ESKF predicts large")
print("acceleration (> 2 m/s²), then:")
print("  1. Quaternion orientation is wrong (gravity not removed correctly)")
print("  2. Accelerometer bias is wrong")
print("  3. Sensor calibration is wrong")
print("="*60)

# Compare with ESKF predictions from earlier test
print("\nESKF predicted accelerations (world frame) at these steps:")
print("  Step 91: [1.12, -1.69, -0.61] → mag = 2.13 m/s²")
print("  Step 93: [2.35, -0.76, -0.58] → mag = 2.56 m/s²")
print("  Step 94: [3.37, 0.01, -0.12] → mag = 3.37 m/s²")
print("  Step 95: [3.62, 0.63, 0.19] → mag = 3.68 m/s²")
print()
print("If GT shows much lower accelerations, the quaternion is drifting!")
