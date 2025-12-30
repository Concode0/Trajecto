"""
Check if ZUPT corrections are coupling to orientation/bias incorrectly.

ZUPT should primarily correct velocity, but through Kalman gain coupling,
it can also affect other states. If this coupling is wrong, it causes drift.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import numpy as np
from model.pure_eskf import PureESKFModel
from model.config import Config

print("="*80)
print("ZUPT COUPLING ANALYSIS")
print("="*80)

# Load short sequence
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    sensor_data = g["sensor_data"][:200]  # 4 seconds

# Create model with ZUPT only
model = PureESKFModel(device="cpu", dt=Config.DT)
model.eskf.use_zupt = True
model.eskf.use_virtual_measurements = False
model.eval()

# Instrument ESKF to track ZUPT corrections
zupt_corrections = []
original_calculate_zupt = model.eskf._calculate_zupt_update

def instrumented_zupt(vel_w_pred, P_error_pred, tcn_zupt_prob=None):
    delta_x, P_new = original_calculate_zupt(vel_w_pred, P_error_pred, tcn_zupt_prob)
    zupt_corrections.append(delta_x.cpu().numpy())
    return delta_x, P_new

model.eskf._calculate_zupt_update = instrumented_zupt

# Run model
sensor_torch = torch.from_numpy(sensor_data).float().unsqueeze(0)
with torch.no_grad():
    output = model(sensor_torch, sensor_torch, torch.tensor([200]))

# Restore
model.eskf._calculate_zupt_update = original_calculate_zupt

print(f"\nCollected {len(zupt_corrections)} ZUPT corrections")

if zupt_corrections:
    # Stack all corrections
    all_corrections = np.stack(zupt_corrections)  # (N, batch, 15)

    # Analyze each component
    # delta_x layout: [pos(3), vel(3), orientation(3), gyro_bias(3), accel_bias(3)]
    print()
    print("Average magnitude of ZUPT corrections by state component:")
    print("-"*80)

    labels = [
        "Position X", "Position Y", "Position Z",
        "Velocity X", "Velocity Y", "Velocity Z",
        "Orientation X", "Orientation Y", "Orientation Z",
        "Gyro Bias X", "Gyro Bias Y", "Gyro Bias Z",
        "Accel Bias X", "Accel Bias Y", "Accel Bias Z"
    ]

    for i, label in enumerate(labels):
        avg_mag = np.mean(np.abs(all_corrections[:, 0, i]))
        print(f"  {label:18s}: {avg_mag:.6e}")

    # Check ratios
    vel_correction = np.mean(np.abs(all_corrections[:, 0, 3:6]))
    orient_correction = np.mean(np.abs(all_corrections[:, 0, 6:9]))
    gyro_bias_correction = np.mean(np.abs(all_corrections[:, 0, 9:12]))
    accel_bias_correction = np.mean(np.abs(all_corrections[:, 0, 12:15]))

    print()
    print("Component group averages:")
    print(f"  Velocity:    {vel_correction:.6e}")
    print(f"  Orientation: {orient_correction:.6e}")
    print(f"  Gyro bias:   {gyro_bias_correction:.6e}")
    print(f"  Accel bias:  {accel_bias_correction:.6e}")

    print()
    print("Coupling ratios (relative to velocity correction):")
    print(f"  Orientation / Velocity: {orient_correction/vel_correction:.4f}x")
    print(f"  Gyro bias / Velocity:   {gyro_bias_correction/vel_correction:.4f}x")
    print(f"  Accel bias / Velocity:  {accel_bias_correction/vel_correction:.4f}x")

    print()
    if orient_correction / vel_correction > 0.1:
        print("⚠️  WARNING: ZUPT is significantly affecting orientation!")
        print("   This coupling can cause quaternion drift to compound.")
    else:
        print("✅ Orientation coupling is small (<10% of velocity correction)")

    if gyro_bias_correction / vel_correction > 0.1:
        print("⚠️  WARNING: ZUPT is significantly affecting gyro bias!")
        print("   This coupling can cause bias estimates to drift incorrectly.")
    else:
        print("✅ Gyro bias coupling is small (<10% of velocity correction)")

else:
    print("\n❌ No ZUPT corrections were applied!")

print("="*80)
