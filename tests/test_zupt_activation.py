"""
Check if ZUPT is actually activating in pure_eskf.
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
print("ZUPT ACTIVATION CHECK")
print("="*80)

# Load validation sample
with h5py.File("data/validation_dataset.h5", "r") as f:
    sample_key = list(f.keys())[0]
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]
    sensor_data = g["sensor_data"][:seq_len]

print(f"\nSample: {sample_key}")
print(f"Length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
print()

# Manually compute ZUPT conditions to check activation potential
# (Don't actually run the ZUPT detector, just check the criteria)

accel_b = sensor_data[:, 0:3]
force = sensor_data[:, 6]

# Compute windowed statistics
window_size = Config.ZUPT_WINDOW_SIZE

# Accel criterion: deviation from gravity magnitude
accel_mag = np.linalg.norm(accel_b, axis=1)
accel_dev = np.abs(accel_mag - 9.81)
accel_zupt_ok = accel_dev < Config.ZUPT_ACCEL_THRESHOLD

# Force variance criterion
from scipy.ndimage import uniform_filter1d
force_mean = uniform_filter1d(force, window_size)
force_var = uniform_filter1d(np.square(force - force_mean), window_size)
force_var_ok = force_var < Config.ZUPT_FORCE_VAR_THRESHOLD

# Force delta criterion
force_delta = np.abs(np.diff(force, prepend=force[0]))
force_delta_ok = force_delta < Config.ZUPT_FORCE_DELTA_THRESHOLD

# All three must be true for ZUPT
would_zupt = accel_zupt_ok & force_var_ok & force_delta_ok

zupt_rate = np.mean(would_zupt)
zupt_count = np.sum(would_zupt)

print(f"ZUPT Activation Analysis (manual check):")
print(f"  Total steps: {len(would_zupt)}")
print(f"  Would ZUPT: {zupt_count} steps")
print(f"  ZUPT rate: {zupt_rate*100:.2f}%")
print()
print(f"Individual criteria pass rates:")
print(f"  Accel deviation OK: {np.mean(accel_zupt_ok)*100:.1f}%")
print(f"  Force variance OK:  {np.mean(force_var_ok)*100:.1f}%")
print(f"  Force delta OK:     {np.mean(force_delta_ok)*100:.1f}%")
print()

if zupt_rate > 0.1:
    print("  ✅ ZUPT would activate frequently (>10%)")
elif zupt_rate > 0.01:
    print("  ⚠️  ZUPT activation low (1-10%)")
elif zupt_rate > 0:
    print("  ⚠️  ZUPT activation very low (<1%)")
else:
    print("  ❌ ZUPT WOULD NEVER ACTIVATE!")
    print()
    print("  Bottleneck criterion:")
    if np.mean(accel_zupt_ok) < 0.1:
        print("    Accel deviation is the limiting factor")
    elif np.mean(force_var_ok) < 0.1:
        print("    Force variance is the limiting factor")
    elif np.mean(force_delta_ok) < 0.1:
        print("    Force delta is the limiting factor")

# Check thresholds
print()
print(f"Current ZUPT thresholds:")
print(f"  ZUPT_ACCEL_THRESHOLD: {Config.ZUPT_ACCEL_THRESHOLD}")
print(f"  ZUPT_FORCE_VAR_THRESHOLD: {Config.ZUPT_FORCE_VAR_THRESHOLD}")
print(f"  ZUPT_FORCE_DELTA_THRESHOLD: {Config.ZUPT_FORCE_DELTA_THRESHOLD}")
print(f"  ZUPT_WINDOW_SIZE: {Config.ZUPT_WINDOW_SIZE}")

# Analyze force and accel data to see what values we actually have
accel_b = sensor_data[:, 0:3]
force = sensor_data[:, 6]

# Compute windowed variance of force
from scipy.ndimage import uniform_filter1d
window_size = Config.ZUPT_WINDOW_SIZE
force_var = uniform_filter1d(np.square(force - uniform_filter1d(force, window_size)), window_size)
force_delta = np.abs(np.diff(force, prepend=force[0]))

# Compute accel variance (deviation from gravity)
accel_mag = np.linalg.norm(accel_b, axis=1)
accel_dev = np.abs(accel_mag - 9.81)

print()
print(f"Actual data statistics:")
print(f"  Accel deviation from 9.81:")
print(f"    Min: {np.min(accel_dev):.4f}")
print(f"    Mean: {np.mean(accel_dev):.4f}")
print(f"    Max: {np.max(accel_dev):.4f}")
print(f"    Threshold: {Config.ZUPT_ACCEL_THRESHOLD}")
print(f"    % below threshold: {np.mean(accel_dev < Config.ZUPT_ACCEL_THRESHOLD)*100:.1f}%")

print(f"\n  Force variance:")
print(f"    Min: {np.min(force_var):.2f}")
print(f"    Mean: {np.mean(force_var):.2f}")
print(f"    Max: {np.max(force_var):.2f}")
print(f"    Threshold: {Config.ZUPT_FORCE_VAR_THRESHOLD}")
print(f"    % below threshold: {np.mean(force_var < Config.ZUPT_FORCE_VAR_THRESHOLD)*100:.1f}%")

print(f"\n  Force delta:")
print(f"    Min: {np.min(force_delta):.2f}")
print(f"    Mean: {np.mean(force_delta):.2f}")
print(f"    Max: {np.max(force_delta):.2f}")
print(f"    Threshold: {Config.ZUPT_FORCE_DELTA_THRESHOLD}")
print(f"    % below threshold: {np.mean(force_delta < Config.ZUPT_FORCE_DELTA_THRESHOLD)*100:.1f}%")

print("="*80)
