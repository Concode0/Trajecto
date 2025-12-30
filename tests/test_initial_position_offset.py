"""
Check if GT data starts at origin or has an offset.

If GT starts at [x0, y0, z0] != [0, 0, 0], but ESKF starts at origin,
this creates an immediate position error that compounds over time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np

print("="*60)
print("Initial Position Offset Analysis")
print("="*60)

# Check all samples
with h5py.File("data/dataset.h5", "r") as f:
    print("\nTraining Dataset:")
    print("-"*60)

    for sample_key in f.keys():
        g = f[sample_key]
        seq_len = g.attrs["sequence_length"]
        gt_pos = g["gt_pos_data"][:seq_len]

        initial_pos = gt_pos[0]
        print(f"{sample_key}:")
        print(f"  Initial GT position: {initial_pos}")
        print(f"  Distance from origin: {np.linalg.norm(initial_pos):.6f} m")

        # Check if static period has movement
        static_end = min(50, seq_len)
        static_movement = np.linalg.norm(gt_pos[static_end-1] - gt_pos[0])
        print(f"  Movement during first {static_end} steps: {static_movement*1000:.2f} mm")
        print()

with h5py.File("data/validation_dataset.h5", "r") as f:
    print("\nValidation Dataset:")
    print("-"*60)

    for sample_key in f.keys():
        g = f[sample_key]
        seq_len = g.attrs["sequence_length"]
        gt_pos = g["gt_pos_data"][:seq_len]

        initial_pos = gt_pos[0]
        print(f"{sample_key}:")
        print(f"  Initial GT position: {initial_pos}")
        print(f"  Distance from origin: {np.linalg.norm(initial_pos):.6f} m")

        static_end = min(50, seq_len)
        static_movement = np.linalg.norm(gt_pos[static_end-1] - gt_pos[0])
        print(f"  Movement during first {static_end} steps: {static_movement*1000:.2f} mm")
        print()

print("="*60)
print("Analysis")
print("="*60)
print("If GT data starts at non-zero position, but pure_eskf starts at [0,0,0],")
print("this creates an initial offset error that affects scale calculation.")
print()
print("However, for trajectory comparison, we use Sim(3) alignment which")
print("handles translation offset. So this should NOT cause scale error.")
print()
print("But if GT positions are RELATIVE to first position (already centered),")
print("then ESKF should also output relative positions!")
print("="*60)
