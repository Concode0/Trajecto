"""
Check if there's a gravity sign error in the measurement model.

When stationary:
- Accelerometer should measure +gravity (upward, fighting gravity)
- Predicted measurement should also be +gravity
- Innovation should be ~0
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
print("GRAVITY SIGN CHECK")
print("="*80)

# Load validation sample
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    sensor_data = g["sensor_data"][:100]  # First 100 samples (static period)

print(f"\nUsing first 100 samples (static period)")

# Check actual accelerometer measurements during static period
accel_b = sensor_data[:, 0:3]
gyro_b = sensor_data[:, 3:6]

accel_mag = np.linalg.norm(accel_b, axis=1)
gyro_mag = np.linalg.norm(gyro_b, axis=1)

print(f"\nActual sensor data (first 100 samples):")
print(f"  Accel magnitude:")
print(f"    Mean: {np.mean(accel_mag):.4f} m/s²")
print(f"    Std:  {np.std(accel_mag):.4f} m/s²")
print(f"  Gyro magnitude:")
print(f"    Mean: {np.mean(gyro_mag):.6f} rad/s")
print(f"    Std:  {np.std(gyro_mag):.6f} rad/s")

print(f"\n  Accel vector (mean):")
print(f"    X: {np.mean(accel_b[:, 0]):.4f} m/s²")
print(f"    Y: {np.mean(accel_b[:, 1]):.4f} m/s²")
print(f"    Z: {np.mean(accel_b[:, 2]):.4f} m/s²")

# Expected: Z ≈ +9.81 (upward), X and Y ≈ 0
if np.abs(np.mean(accel_b[:, 2]) - 9.81) < 1.0:
    print("  ✅ Accel Z is positive (measures upward acceleration)")
else:
    print(f"  ⚠️  Accel Z is {np.mean(accel_b[:, 2]):.4f}, expected ~9.81")

# Now check what ESKF predicts during static initialization
print()
print("="*80)
print("ESKF Measurement Prediction")
print("="*80)

# Initialize ESKF
model = PureESKFModel(device="cpu", dt=Config.DT)
model.eval()

# Get initial state after gravity alignment
sensor_torch = torch.from_numpy(sensor_data).float().unsqueeze(0)

# Access ESKF's initial state (after initialization)
with torch.no_grad():
    # Initialize like pure_eskf does
    batch_size = 1
    device = "cpu"

    # Static period detection
    static_buffer_samples = int(2.0 / Config.DT)  # 2 seconds
    avg_accel_b = sensor_torch[:, :static_buffer_samples, 0:3].mean(dim=1)

    # Gravity alignment
    measured_gravity_magnitude = torch.norm(avg_accel_b, p=2, dim=-1, keepdim=True)
    world_gravity_down = torch.cat([
        torch.zeros(batch_size, 2, device=device),
        measured_gravity_magnitude
    ], dim=-1)

    print(f"\nGravity alignment:")
    print(f"  Measured accel (avg first 2s): {avg_accel_b[0].numpy()}")
    print(f"  Measured magnitude: {measured_gravity_magnitude[0].item():.4f} m/s²")
    print(f"  World gravity vector: {world_gravity_down[0].numpy()}")

    # After alignment, body frame accel should equal world gravity
    # because quaternion rotates body to align with world

    # Now check measurement prediction
    # In ESKF, accel_pred = gravity_body + accel_bias_b
    # If bias is initially zero, accel_pred = gravity_body

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
    from quaternion_ops import quaternion_from_two_vectors, quaternion_to_rotation_matrix

    quat = quaternion_from_two_vectors(avg_accel_b, world_gravity_down)
    rot_mat_world_to_body = quaternion_to_rotation_matrix(quat).transpose(-2, -1)
    gravity_body = (rot_mat_world_to_body @ world_gravity_down.unsqueeze(-1)).squeeze(-1)

    print(f"\nAfter gravity alignment:")
    print(f"  Quaternion: {quat[0].numpy()}")
    print(f"  Gravity in body frame: {gravity_body[0].numpy()}")
    print(f"  Expected: Should match measured accel ~{avg_accel_b[0].numpy()}")

    # Check if they match
    diff = torch.abs(gravity_body - avg_accel_b)
    print(f"  Difference: {diff[0].numpy()}")

    if torch.max(diff) < 0.5:
        print("  ✅ Gravity body matches measured accel")
    else:
        print("  ❌ MISMATCH! Gravity body != measured accel")

    # Now check the measurement model
    # accel_pred = gravity_body + accel_bias_b (initially 0)
    accel_bias_b = torch.zeros(batch_size, 3, device=device)
    accel_pred = gravity_body + accel_bias_b

    # Actual measurement
    accel_meas = avg_accel_b

    # Innovation
    innovation = accel_meas - accel_pred

    print(f"\nMeasurement model check:")
    print(f"  Predicted measurement: {accel_pred[0].numpy()}")
    print(f"  Actual measurement:    {accel_meas[0].numpy()}")
    print(f"  Innovation:            {innovation[0].numpy()}")
    print(f"  Innovation magnitude:  {torch.norm(innovation, dim=-1)[0].item():.6f} m/s²")

    if torch.norm(innovation) < 0.1:
        print("  ✅ Innovation is small (model matches reality)")
    else:
        print("  ⚠️  Large innovation (model mismatch)")

print()
print("="*80)
print("PROPAGATION CHECK")
print("="*80)

# Check what happens when we propagate
# accel_w = R * accel_b_corrected - gravity_w
# If stationary, accel_b_corrected ≈ gravity_body (after bias subtraction)
# So accel_w = R * gravity_body - gravity_w

print("\nWhen stationary, propagation computes:")
print("  accel_w = R_b2w * accel_b_corrected - gravity_w")
print("  accel_b_corrected = accel_raw - bias ≈ gravity_body (when bias ≈ 0)")

# After gravity alignment, R should rotate gravity_body to gravity_world
rot_mat_b_to_w = quaternion_to_rotation_matrix(quat)
accel_b_corrected = avg_accel_b  # ≈ gravity_body
accel_w = (rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - world_gravity_down

print(f"\n  R_b2w * accel_b: {(rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1)[0].numpy()}")
print(f"  gravity_w:       {world_gravity_down[0].numpy()}")
print(f"  accel_w:         {accel_w[0].numpy()}")
print(f"  accel_w magnitude: {torch.norm(accel_w)[0].item():.6f} m/s²")

if torch.norm(accel_w) < 0.5:
    print("  ✅ Stationary accel_w ≈ 0 (correct)")
else:
    print(f"  ❌ ERROR: Stationary accel_w = {torch.norm(accel_w)[0].item():.4f} m/s² (should be ~0)")
    print("     This means there's a SIGN ERROR in gravity handling!")

print("="*80)
