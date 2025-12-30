"""
Investigate GT velocity calculation method discrepancy.

np.gradient() uses central difference, which is different from simple forward difference.
This test checks if the stored GT velocity matches np.gradient() behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*60)
print("GT Velocity Calculation Method Test")
print("="*60)

# Load data
with h5py.File("data/dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]

    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel_stored = g["gt_vel_data"][:seq_len]

dt = Config.DT
print(f"\nSample: {sample_key}")
print(f"Sequence length: {seq_len}")
print(f"DT: {dt:.6f} s ({1/dt:.2f} Hz)")
print()

# Method 1: Simple forward difference (what test_gt_data_consistency used)
print("=== Method 1: Forward Difference ===")
vel_forward = np.diff(gt_pos, axis=0) / dt
print(f"Shape: {vel_forward.shape}")
print(f"First 3 values:")
for i in range(3):
    print(f"  [{i}]: {vel_forward[i]}")
print()

# Method 2: np.gradient (what acquire.py uses)
print("=== Method 2: np.gradient ===")
vel_gradient = np.gradient(gt_pos, dt, axis=0)
print(f"Shape: {vel_gradient.shape}")
print(f"First 3 values:")
for i in range(3):
    print(f"  [{i}]: {vel_gradient[i]}")
print()

# Method 3: Stored GT velocity
print("=== Method 3: Stored GT velocity ===")
print(f"Shape: {gt_vel_stored.shape}")
print(f"First 3 values:")
for i in range(3):
    print(f"  [{i}]: {gt_vel_stored[i]}")
print()

# Compare stored with gradient
print("=== Comparison: Stored vs np.gradient ===")
error_gradient = gt_vel_stored - vel_gradient
error_mag_gradient = np.linalg.norm(error_gradient, axis=1)

print(f"Mean error magnitude: {error_mag_gradient.mean():.9f} m/s")
print(f"Max error magnitude:  {error_mag_gradient.max():.9f} m/s")
print(f"Median error:         {np.median(error_mag_gradient):.9f} m/s")

if error_mag_gradient.mean() < 1e-6:
    print("✓ Stored GT velocity MATCHES np.gradient()")
else:
    print("❌ Stored GT velocity DOES NOT match np.gradient()!")
    print(f"\nFirst 10 differences:")
    for i in range(10):
        print(f"  [{i}]: stored={gt_vel_stored[i]}, gradient={vel_gradient[i]}, error={error_gradient[i]}")

print()

# Compare stored with forward difference (offset by 1)
print("=== Comparison: Stored vs Forward Difference ===")
# vel_forward[i] corresponds to velocity at time i+1 (between pos[i] and pos[i+1])
# So compare vel_forward[i] with gt_vel_stored[i+1]
error_forward = vel_forward - gt_vel_stored[1:]
error_mag_forward = np.linalg.norm(error_forward, axis=1)

print(f"Mean error magnitude: {error_mag_forward.mean():.9f} m/s")
print(f"Max error magnitude:  {error_mag_forward.max():.9f} m/s")

print()

# Verify np.gradient calculation manually
print("=== Manual Verification of np.gradient ===")
print("np.gradient uses:")
print("  - Forward diff at i=0:  (pos[1] - pos[0]) / dt")
print("  - Central diff at i=1:  (pos[2] - pos[0]) / (2*dt)")
print("  - Backward diff at i=N-1: (pos[N-1] - pos[N-2]) / dt")
print()

# Check first point (forward difference)
manual_0 = (gt_pos[1] - gt_pos[0]) / dt
print(f"Manual calc at [0]: {manual_0}")
print(f"np.gradient at [0]: {vel_gradient[0]}")
print(f"Match: {np.allclose(manual_0, vel_gradient[0])}")
print()

# Check middle point (central difference)
manual_50 = (gt_pos[51] - gt_pos[49]) / (2 * dt)
print(f"Manual calc at [50]: {manual_50}")
print(f"np.gradient at [50]: {vel_gradient[50]}")
print(f"Match: {np.allclose(manual_50, vel_gradient[50])}")
print()

# Check last point (backward difference)
manual_last = (gt_pos[-1] - gt_pos[-2]) / dt
print(f"Manual calc at [-1]: {manual_last}")
print(f"np.gradient at [-1]: {vel_gradient[-1]}")
print(f"Match: {np.allclose(manual_last, vel_gradient[-1])}")

print()
print("="*60)
print("Conclusion")
print("="*60)
print("If stored velocity matches np.gradient, then GT velocity is correct.")
print("The error in test_gt_data_consistency was due to comparing with")
print("forward difference instead of central difference.")
print("="*60)
