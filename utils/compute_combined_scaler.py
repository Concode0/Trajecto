# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: This software is protected under the following ROK Patent Applications:
# 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
# 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
#
# Commercial use or redistribution of the core logic requires a separate license.
# For inquiries, contact: nemonanconcode@gmail.com

"""
Compute combined scaler statistics from simulated + real data.

This script computes weighted mean and std from both datasets
for proper normalization during two-stage training.

Usage:
    python utils/compute_combined_scaler.py
    python utils/compute_combined_scaler.py --sim-weight 0.7
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def load_dataset_stats(file_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load sensor data from HDF5 and compute running statistics.

    Args:
        file_path: Path to HDF5 dataset file

    Returns:
        Tuple of (sum, sum_sq, count) for online mean/var computation
    """
    total_sum = None
    total_sum_sq = None
    total_count = 0

    with h5py.File(file_path, "r") as f:
        keys = [k for k in f.keys() if "sample" in k.lower()]
        print(f"Processing {len(keys)} samples from {file_path}")

        for key in tqdm(keys, desc=f"Loading {Path(file_path).name}"):
            sensor = f[key]["sensor_data"][:]

            # Get sequence length
            try:
                seq_len = f[key].attrs["sequence_length"]
            except KeyError:
                # Fallback: find last non-padded row
                diff = np.diff(sensor, axis=0)
                try:
                    seq_len = np.where(np.any(diff != 0, axis=1))[0][-1] + 2
                except IndexError:
                    seq_len = sensor.shape[0]

            # Use only valid timesteps
            valid_data = sensor[:seq_len]

            if total_sum is None:
                total_sum = np.zeros(valid_data.shape[1])
                total_sum_sq = np.zeros(valid_data.shape[1])

            total_sum += valid_data.sum(axis=0)
            total_sum_sq += (valid_data ** 2).sum(axis=0)
            total_count += seq_len

    return total_sum, total_sum_sq, total_count


def compute_combined_scaler_stats(
    sim_file: str,
    real_file: str,
    output_file: str,
    sim_weight: float = 0.5
) -> Dict[str, np.ndarray]:
    """Compute weighted mean/std from combined datasets.

    Uses Welford's online algorithm for stable computation.

    Args:
        sim_file: Path to simulated dataset HDF5
        real_file: Path to real dataset HDF5
        output_file: Path to output scaler stats HDF5
        sim_weight: Weight for sim data (0.0-1.0). Real weight = 1 - sim_weight

    Returns:
        Dict with 'mean' and 'std' arrays
    """
    print(f"Computing combined scaler stats (sim_weight={sim_weight})")

    # Load stats from both datasets
    sim_sum, sim_sum_sq, sim_count = load_dataset_stats(sim_file)
    real_sum, real_sum_sq, real_count = load_dataset_stats(real_file)

    # Weighted combination
    # For weighted average: mean = (w1 * sum1 + w2 * sum2) / (w1 * count1 + w2 * count2)
    real_weight = 1.0 - sim_weight

    # Normalize weights by counts to get per-sample weights
    total_weighted_count = sim_weight * sim_count + real_weight * real_count

    # Combined mean
    combined_mean = (sim_weight * sim_sum + real_weight * real_sum) / total_weighted_count

    # For variance, we need to be more careful with weighted pooled variance
    # Using the formula: var = E[X^2] - E[X]^2
    sim_mean = sim_sum / sim_count
    real_mean = real_sum / real_count

    sim_var = sim_sum_sq / sim_count - sim_mean ** 2
    real_var = real_sum_sq / real_count - real_mean ** 2

    # Pooled weighted variance (accounting for different means)
    # Var_combined = w1 * (var1 + (mean1 - combined_mean)^2) + w2 * (var2 + (mean2 - combined_mean)^2)
    combined_var = (
        sim_weight * (sim_var + (sim_mean - combined_mean) ** 2) +
        real_weight * (real_var + (real_mean - combined_mean) ** 2)
    )

    combined_std = np.sqrt(combined_var)

    # Ensure no zero std (add small epsilon)
    combined_std = np.maximum(combined_std, 1e-6)

    print(f"\nCombined Statistics:")
    print(f"  Sim samples: {sim_count} timesteps")
    print(f"  Real samples: {real_count} timesteps")
    print(f"  Mean: {combined_mean}")
    print(f"  Std:  {combined_std}")

    # Save to HDF5
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        f.create_dataset("mean", data=combined_mean.astype(np.float32))
        f.create_dataset("std", data=combined_std.astype(np.float32))

        # Store source info as attributes
        f.attrs["sim_file"] = sim_file
        f.attrs["real_file"] = real_file
        f.attrs["sim_weight"] = sim_weight
        f.attrs["sim_count"] = sim_count
        f.attrs["real_count"] = real_count

    print(f"\nSaved to: {output_file}")

    return {"mean": combined_mean, "std": combined_std}


def main():
    parser = argparse.ArgumentParser(description="Compute combined scaler statistics")

    parser.add_argument(
        "--sim-file",
        type=str,
        default="data/simulated_dataset.h5",
        help="Path to simulated dataset"
    )
    parser.add_argument(
        "--real-file",
        type=str,
        default="data/dataset.h5",
        help="Path to real dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/combined_scaler_stats.h5",
        help="Output path for combined scaler stats"
    )
    parser.add_argument(
        "--sim-weight",
        type=float,
        default=0.5,
        help="Weight for sim data (0.0-1.0)"
    )

    args = parser.parse_args()

    compute_combined_scaler_stats(
        sim_file=args.sim_file,
        real_file=args.real_file,
        output_file=args.output,
        sim_weight=args.sim_weight
    )


if __name__ == "__main__":
    main()
