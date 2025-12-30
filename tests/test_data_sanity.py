"""
Comprehensive data sanity check to detect if preprocessed data is garbage.

Check for:
1. NaN/Inf values
2. Unreasonable magnitudes
3. Discontinuities/jumps
4. Unit consistency
5. GT trajectory reasonableness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from model.config import Config

print("="*60)
print("Preprocessed Data Sanity Check")
print("="*60)

def check_sample(f, sample_key):
    """Check one sample for data integrity issues."""
    g = f[sample_key]
    seq_len = g.attrs["sequence_length"]

    sensor_data = g["sensor_data"][:seq_len]
    gt_pos = g["gt_pos_data"][:seq_len]
    gt_vel = g["gt_vel_data"][:seq_len]

    print(f"\n{'='*60}")
    print(f"Sample: {sample_key}")
    print(f"Length: {seq_len} samples ({seq_len * Config.DT:.1f}s)")
    print(f"{'='*60}")

    issues = []

    # Check 1: NaN/Inf values
    print("\n1. NaN/Inf Check:")
    if np.any(np.isnan(sensor_data)) or np.any(np.isinf(sensor_data)):
        issues.append("❌ Sensor data contains NaN/Inf")
        print("  ❌ Sensor data contains NaN/Inf!")
    else:
        print("  ✓ Sensor data clean (no NaN/Inf)")

    if np.any(np.isnan(gt_pos)) or np.any(np.isinf(gt_pos)):
        issues.append("❌ GT position contains NaN/Inf")
        print("  ❌ GT position contains NaN/Inf!")
    else:
        print("  ✓ GT position clean (no NaN/Inf)")

    if np.any(np.isnan(gt_vel)) or np.any(np.isinf(gt_vel)):
        issues.append("❌ GT velocity contains NaN/Inf")
        print("  ❌ GT velocity contains NaN/Inf!")
    else:
        print("  ✓ GT velocity clean (no NaN/Inf)")

    # Check 2: Sensor data ranges
    print("\n2. Sensor Data Range Check:")
    accel = sensor_data[:, :3]
    gyro = sensor_data[:, 3:6]
    force = sensor_data[:, 6]

    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    print(f"  Accelerometer magnitude: [{accel_mag.min():.2f}, {accel_mag.max():.2f}] m/s²")
    if accel_mag.min() < 0.5 or accel_mag.max() > 50.0:
        issues.append(f"❌ Accelerometer magnitude out of range: [{accel_mag.min():.2f}, {accel_mag.max():.2f}]")
        print("    ❌ Out of reasonable range [0.5, 50] m/s²")
    else:
        print("    ✓ Within reasonable range")

    print(f"  Gyroscope magnitude: [{gyro_mag.min():.4f}, {gyro_mag.max():.4f}] rad/s")
    if gyro_mag.max() > 50.0:
        issues.append(f"❌ Gyroscope magnitude too high: {gyro_mag.max():.2f}")
        print("    ❌ Exceeds reasonable range (>50 rad/s)")
    else:
        print("    ✓ Within reasonable range")

    print(f"  Force sensor: [{force.min():.2f}, {force.max():.2f}]")
    if force.min() < 0:
        issues.append(f"❌ Force sensor has negative values: {force.min():.2f}")
        print("    ❌ Contains negative values!")
    else:
        print("    ✓ All positive")

    # Check 3: Discontinuities (large jumps between samples)
    print("\n3. Discontinuity Check:")
    accel_diff = np.abs(np.diff(accel, axis=0))
    accel_max_jump = np.max(accel_diff)

    gyro_diff = np.abs(np.diff(gyro, axis=0))
    gyro_max_jump = np.max(gyro_diff)

    pos_diff = np.abs(np.diff(gt_pos, axis=0))
    pos_max_jump = np.linalg.norm(pos_diff, axis=1).max()

    print(f"  Max accel jump: {accel_max_jump:.2f} m/s² (should be <20)")
    if accel_max_jump > 20.0:
        issues.append(f"❌ Large accel jump: {accel_max_jump:.2f}")
        print("    ❌ Suspiciously large!")

    print(f"  Max gyro jump: {gyro_max_jump:.4f} rad/s (should be <10)")
    if gyro_max_jump > 10.0:
        issues.append(f"❌ Large gyro jump: {gyro_max_jump:.4f}")
        print("    ❌ Suspiciously large!")

    print(f"  Max position jump: {pos_max_jump*1000:.2f} mm (should be <10mm)")
    if pos_max_jump > 0.01:  # 10mm
        issues.append(f"❌ Large position jump: {pos_max_jump*1000:.2f} mm")
        print("    ❌ Suspiciously large!")
        # Show where the jump occurs
        jump_idx = np.argmax(np.linalg.norm(pos_diff, axis=1))
        print(f"      Occurs at step {jump_idx} → {jump_idx+1}")
        print(f"      Before: {gt_pos[jump_idx]}")
        print(f"      After:  {gt_pos[jump_idx+1]}")
        print(f"      Jump:   {pos_diff[jump_idx]}")

    # Check 4: GT trajectory reasonableness
    print("\n4. GT Trajectory Check:")
    trajectory_range = gt_pos.max(axis=0) - gt_pos.min(axis=0)
    print(f"  Trajectory range (XYZ): {trajectory_range*100} cm")

    if trajectory_range[0] < 0.01 or trajectory_range[1] < 0.01:
        issues.append(f"❌ Trajectory too small: X={trajectory_range[0]*100:.1f}cm, Y={trajectory_range[1]*100:.1f}cm")
        print("    ❌ Trajectory range too small for handwriting")
    elif trajectory_range[0] > 0.5 or trajectory_range[1] > 0.5:
        issues.append(f"❌ Trajectory too large: X={trajectory_range[0]*100:.1f}cm, Y={trajectory_range[1]*100:.1f}cm")
        print("    ❌ Trajectory range too large for handwriting")
    else:
        print("    ✓ Reasonable handwriting range (5-50cm)")

    # Check 5: Velocity reasonableness
    print("\n5. GT Velocity Check:")
    vel_mag = np.linalg.norm(gt_vel, axis=1)
    print(f"  Velocity magnitude: mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f} m/s")

    if vel_mag.max() > 2.0:
        issues.append(f"❌ GT velocity too high: {vel_mag.max():.2f} m/s")
        print("    ❌ Too fast for handwriting (>2 m/s)")
        # Find where
        fast_idx = np.argmax(vel_mag)
        print(f"      Occurs at step {fast_idx}: vel={gt_vel[fast_idx]}")
    else:
        print("    ✓ Reasonable handwriting velocity")

    # Check 6: Static period validation
    print("\n6. Static Period Check:")
    static_end = min(50, seq_len)
    static_pos = gt_pos[:static_end]
    static_movement = np.linalg.norm(static_pos[-1] - static_pos[0])

    print(f"  Movement in first {static_end} steps: {static_movement*1000:.2f} mm")
    if static_movement > 0.005:  # 5mm
        issues.append(f"❌ Static period has movement: {static_movement*1000:.2f} mm")
        print("    ⚠ WARNING: Static period has significant movement (>5mm)")
        print("      This violates the 'static initialization' assumption")
    else:
        print("    ✓ Truly static (<5mm movement)")

    # Check 7: Gravity during static period
    print("\n7. Gravity Calibration Check:")
    static_accel = sensor_data[:static_end, :3]
    gravity_mags = np.linalg.norm(static_accel, axis=1)

    print(f"  Gravity magnitude: mean={gravity_mags.mean():.4f}, std={gravity_mags.std():.4f} m/s²")
    if abs(gravity_mags.mean() - 9.81) > 0.5:
        issues.append(f"❌ Gravity mean far from 9.81: {gravity_mags.mean():.4f}")
        print("    ❌ Mean far from 9.81 m/s² - possible wrong units or calibration")
    elif gravity_mags.std() > 0.5:
        issues.append(f"❌ Gravity variance too high: std={gravity_mags.std():.4f}")
        print("    ❌ High variance during static - sensor unstable")
    else:
        print("    ✓ Gravity measurement looks good")

    # Summary
    print(f"\n{'='*60}")
    if issues:
        print(f"❌ ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ ALL CHECKS PASSED - Data looks clean")
    print(f"{'='*60}")

    return issues

# Check all samples
all_issues = []
with h5py.File("data/dataset.h5", "r") as f:
    for sample_key in f.keys():
        issues = check_sample(f, sample_key)
        all_issues.extend(issues)

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
if all_issues:
    print(f"❌ Total issues found: {len(all_issues)}")
    print("\nMost critical issues:")
    for issue in all_issues[:10]:
        print(f"  • {issue}")

    print("\n⚠ DATA MAY BE CORRUPTED - Check acquire.py preprocessing!")
else:
    print("✓ All data passed sanity checks")
    print("  Data is clean - issue is likely in ESKF logic")
print(f"{'='*60}")
