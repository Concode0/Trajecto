# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
data_visualizer.py

A utility script to visualize raw and processed data from Trajecto's HDF5 files.

This script allows loading and plotting:
- Raw IMU (accelerometer, gyroscope) data.
- Raw Ground Truth (position, force, isHovering) data.
- Processed sensor data (accelerometer, gyroscope, FSR).
- Processed ground truth position and velocity data.

Usage:
    python utils/data_visualizer.py --list_raw_samples
    python utils/data_visualizer.py --raw_sample_id sample_001
    python utils/data_visualizer.py --list_processed_segments
    python utils/data_visualizer.py --processed_segment_id sample_001_seg0
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from typing import Dict, Any

# Define HDF5 file paths based on acquire.py
RAW_HDF5_PATH = "acquired_data/raw_acquired_data.h5"
PROCESSED_DATASET_PATH = "data/dataset.h5"


def load_raw_sample(sample_name: str) -> Dict[str, Any]:
    """
    Loads a raw sample from the raw_acquired_data.h5 file.

    Args:
        sample_name (str): The ID of the raw sample (e.g., 'sample_001').

    Returns:
        Dict[str, Any]: A dictionary containing 'pen_data' (DataFrame),
                        'gt_data' (DataFrame), 'label' (str), and 'ipad_idx' (int).
                        Returns an empty dictionary if the sample is not found.
    """
    data = {}
    if not os.path.exists(RAW_HDF5_PATH):
        print(f"Error: Raw HDF5 file not found at '{RAW_HDF5_PATH}'.")
        return data

    try:
        with h5py.File(RAW_HDF5_PATH, "r") as f:
            grp_path = f"raw_data/{sample_name}"
            if grp_path in f:
                grp = f[grp_path]
                if "pen_data" in grp:
                    # Convert h5py structured array to pandas DataFrame
                    pen_array = grp["pen_data"][:]
                    data["pen_data"] = pd.DataFrame(pen_array.tolist(), columns=pen_array.dtype.names)
                if "gt_data" in grp:
                    gt_array = grp["gt_data"][:]
                    data["gt_data"] = pd.DataFrame(gt_array.tolist(), columns=gt_array.dtype.names)
                data["label"] = grp.attrs.get("original_label", "N/A")
                data["ipad_idx"] = grp.attrs.get("ipad_file_index", "N/A")
            else:
                print(f"Sample '{sample_name}' not found in raw HDF5 at '{grp_path}'.")
    except Exception as e:
        print(f"Error loading raw sample '{sample_name}': {e}")
    return data


def load_processed_segment(segment_name: str) -> Dict[str, Any]:
    """
    Loads a processed segment from the dataset.h5 file.

    Args:
        segment_name (str): The ID of the processed segment (e.g., 'sample_001_seg0').

    Returns:
        Dict[str, Any]: A dictionary containing 'sensor_data' (ndarray),
                        'gt_pos_data' (ndarray), 'gt_vel_data' (ndarray),
                        'label' (str), and 'seq_len' (int).
                        Returns an empty dictionary if the segment is not found.
    """
    data = {}
    if not os.path.exists(PROCESSED_DATASET_PATH):
        print(f"Error: Processed HDF5 file not found at '{PROCESSED_DATASET_PATH}'.")
        return data

    try:
        with h5py.File(PROCESSED_DATASET_PATH, "r") as f:
            if segment_name in f:
                grp = f[segment_name]
                data["sensor_data"] = grp["sensor_data"][:]
                data["gt_pos_data"] = grp["gt_pos_data"][:]
                if "gt_vel_data" in grp:
                    data["gt_vel_data"] = grp["gt_vel_data"][:]
                else:
                    data["gt_vel_data"] = None
                data["label"] = grp.attrs.get("original_label", "N/A")
                data["seq_len"] = grp.attrs.get("sequence_length", 0)
            else:
                print(f"Segment '{segment_name}' not found in processed HDF5.")
    except Exception as e:
        print(f"Error loading processed segment '{segment_name}': {e}")
    return data


def plot_raw_imu(df_pen_data: pd.DataFrame, title: str):
    """
    Plots raw accelerometer and gyroscope data.

    Args:
        df_pen_data (pd.DataFrame): DataFrame containing raw IMU data.
        title (str): Title for the plots.
    """
    if df_pen_data.empty:
        print("No raw IMU data to plot.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Raw IMU Data: {title}")

    # Accelerometer
    accel_cols = ['accel_x', 'accel_y', 'accel_z']
    if all(col in df_pen_data.columns for col in accel_cols):
        df_pen_data[accel_cols].plot(ax=axes[0])
        axes[0].set_title("Accelerometer (m/s^2)")
        axes[0].set_ylabel("Acceleration")
        axes[0].legend(loc='upper right')
        axes[0].grid(True)
    else:
        axes[0].set_title("Accelerometer data not available")

    # Gyroscope
    gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
    if all(col in df_pen_data.columns for col in gyro_cols):
        df_pen_data[gyro_cols].plot(ax=axes[1])
        axes[1].set_title("Gyroscope (rad/s)")
        axes[1].set_ylabel("Angular Velocity")
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
    else:
        axes[1].set_title("Gyroscope data not available")

    axes[1].set_xlabel("Sample Index")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_raw_gt(df_gt_data: pd.DataFrame, title: str):
    """
    Plots raw ground truth data (position, force, hovering).

    Args:
        df_gt_data (pd.DataFrame): DataFrame containing raw ground truth data.
        title (str): Title for the plots.
    """
    if df_gt_data.empty:
        print("No raw GT data to plot.")
        return

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Raw Ground Truth Data: {title}")

    # Position
    pos_cols = ['x', 'y', 'z']
    if all(col in df_gt_data.columns for col in pos_cols):
        df_gt_data[pos_cols].plot(ax=axes[0])
        axes[0].set_title("Position (pixels/mm)")
        axes[0].set_ylabel("Value")
        axes[0].legend(loc='upper right')
        axes[0].grid(True)
    else:
        axes[0].set_title("Position data not available")

    # Force
    if 'force' in df_gt_data.columns:
        df_gt_data['force'].plot(ax=axes[1], color='red')
        axes[1].set_title("Force")
        axes[1].set_ylabel("Force Value")
        axes[1].grid(True)
    else:
        axes[1].set_title("Force data not available")

    # isHovering
    if 'isHovering' in df_gt_data.columns:
        df_gt_data['isHovering'].plot(ax=axes[2], color='green', drawstyle='steps-post')
        axes[2].set_title("isHovering")
        axes[2].set_ylabel("Binary (0/1)")
        axes[2].grid(True)
    else:
        axes[2].set_title("isHovering data not available")

    axes[2].set_xlabel("Sample Index")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_processed_data(data: Dict[str, Any], title: str):
    """
    Plots processed sensor and ground truth data.

    Args:
        data (Dict[str, Any]): Dictionary containing processed data arrays.
        title (str): Title for the plots.
    """
    seq_len = data.get("seq_len", 0)
    if seq_len == 0:
        print("No valid sequence length for processed data to plot.")
        return

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Processed Data: {title}", fontsize=14)

    # Sensor Data - Accelerometer
    ax1 = plt.subplot(3, 2, 1)
    sensor_data = data.get("sensor_data")
    if sensor_data is not None and sensor_data.shape[1] >= 6:
        ax1.plot(sensor_data[:seq_len, 0], label='accel_x')
        ax1.plot(sensor_data[:seq_len, 1], label='accel_y')
        ax1.plot(sensor_data[:seq_len, 2], label='accel_z')
        ax1.set_title("Sensor Accelerometer (m/s²)")
        ax1.set_ylabel("Acceleration")
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax1.set_xlabel("Time (samples)")
    else:
        ax1.set_title("Accelerometer data not available")

    # Sensor Data - Gyroscope
    ax2 = plt.subplot(3, 2, 3)
    if sensor_data is not None and sensor_data.shape[1] >= 6:
        ax2.plot(sensor_data[:seq_len, 3], label='gyro_x')
        ax2.plot(sensor_data[:seq_len, 4], label='gyro_y')
        ax2.plot(sensor_data[:seq_len, 5], label='gyro_z')
        ax2.set_title("Sensor Gyroscope (rad/s)")
        ax2.set_ylabel("Angular Velocity")
        ax2.legend(loc='upper right')
        ax2.grid(True)
        ax2.set_xlabel("Time (samples)")
    else:
        ax2.set_title("Gyroscope data not available")

    # GT Position Time Series
    ax3 = plt.subplot(3, 2, 5)
    gt_pos_data = data.get("gt_pos_data")
    if gt_pos_data is not None and gt_pos_data.shape[1] >= 3:
        ax3.plot(gt_pos_data[:seq_len, 0], label='X')
        ax3.plot(gt_pos_data[:seq_len, 1], label='Y')
        ax3.plot(gt_pos_data[:seq_len, 2], label='Z')
        ax3.set_title("Ground Truth Position vs Time")
        ax3.set_ylabel("Position (m)")
        ax3.set_xlabel("Time (samples)")
        ax3.legend(loc='upper right')
        ax3.grid(True)
    else:
        ax3.set_title("Ground Truth Position data not available")

    # 2D Trajectory (X-Y plane)
    ax4 = plt.subplot(3, 2, 2)
    if gt_pos_data is not None and gt_pos_data.shape[1] >= 3:
        ax4.plot(gt_pos_data[:seq_len, 0], gt_pos_data[:seq_len, 1], 'b-', linewidth=1.5)
        ax4.plot(gt_pos_data[0, 0], gt_pos_data[0, 1], 'go', markersize=10, label='Start')
        ax4.plot(gt_pos_data[seq_len-1, 0], gt_pos_data[seq_len-1, 1], 'ro', markersize=10, label='End')
        ax4.set_title("2D Trajectory (Top View - X-Y Plane)")
        ax4.set_xlabel("X Position (m)")
        ax4.set_ylabel("Y Position (m)")
        ax4.legend(loc='upper right')
        ax4.grid(True)
        ax4.axis('equal')
    else:
        ax4.set_title("2D Trajectory not available")

    # 3D Trajectory
    ax5 = fig.add_subplot(3, 2, (4, 6), projection='3d')
    if gt_pos_data is not None and gt_pos_data.shape[1] >= 3:
        ax5.plot(gt_pos_data[:seq_len, 0], gt_pos_data[:seq_len, 1], gt_pos_data[:seq_len, 2], 'b-', linewidth=1.5)
        ax5.scatter(gt_pos_data[0, 0], gt_pos_data[0, 1], gt_pos_data[0, 2], c='green', s=100, marker='o', label='Start')
        ax5.scatter(gt_pos_data[seq_len-1, 0], gt_pos_data[seq_len-1, 1], gt_pos_data[seq_len-1, 2], c='red', s=100, marker='o', label='End')
        ax5.set_title("3D Trajectory")
        ax5.set_xlabel("X Position (m)")
        ax5.set_ylabel("Y Position (m)")
        ax5.set_zlabel("Z Position (m)")
        ax5.legend(loc='upper right')

        # Set equal aspect ratio for better visualization
        max_range = np.array([
            gt_pos_data[:seq_len, 0].max() - gt_pos_data[:seq_len, 0].min(),
            gt_pos_data[:seq_len, 1].max() - gt_pos_data[:seq_len, 1].min(),
            gt_pos_data[:seq_len, 2].max() - gt_pos_data[:seq_len, 2].min()
        ]).max() / 2.0

        mid_x = (gt_pos_data[:seq_len, 0].max() + gt_pos_data[:seq_len, 0].min()) * 0.5
        mid_y = (gt_pos_data[:seq_len, 1].max() + gt_pos_data[:seq_len, 1].min()) * 0.5
        mid_z = (gt_pos_data[:seq_len, 2].max() + gt_pos_data[:seq_len, 2].min()) * 0.5
        ax5.set_xlim(mid_x - max_range, mid_x + max_range)
        ax5.set_ylim(mid_y - max_range, mid_y + max_range)
        ax5.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax5.set_title("3D Trajectory not available")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Trajecto data from HDF5 files.")
    parser.add_argument("--raw_sample_id", type=str, help="ID of the raw sample to visualize (e.g., 'sample_001')")
    parser.add_argument("--processed_segment_id", type=str, help="ID of the processed segment to visualize (e.g., 'sample_001_seg0')")
    parser.add_argument("--list_raw_samples", action="store_true", help="List all available raw sample IDs")
    parser.add_argument("--list_processed_segments", action="store_true", help="List all available processed segment IDs")
    args = parser.parse_args()

    if args.list_raw_samples:
        if not os.path.exists(RAW_HDF5_PATH):
            print(f"Raw HDF5 file not found at {RAW_HDF5_PATH}")
            return
        with h5py.File(RAW_HDF5_PATH, "r") as f:
            if "raw_data" in f:
                print("Available Raw Sample IDs:")
                for key in f["raw_data"].keys():
                    print(f"- {key}")
            else:
                print("No 'raw_data' group found in raw HDF5.")
        return

    if args.list_processed_segments:
        if not os.path.exists(PROCESSED_DATASET_PATH):
            print(f"Processed HDF5 file not found at {PROCESSED_DATASET_PATH}")
            return
        with h5py.File(PROCESSED_DATASET_PATH, "r") as f:
            print("Available Processed Segment IDs:")
            for key in f.keys():
                print(f"- {key}")
        return

    if args.raw_sample_id:
        sample_data = load_raw_sample(args.raw_sample_id)
        if sample_data:
            print(f"Visualizing Raw Sample: {args.raw_sample_id} (Label: {sample_data.get('label')}, iPad Index: {sample_data.get('ipad_idx')})")
            if "pen_data" in sample_data and not sample_data["pen_data"].empty:
                plot_raw_imu(sample_data["pen_data"], f"Raw IMU - {args.raw_sample_id} ({sample_data['label']})")
            if "gt_data" in sample_data and not sample_data["gt_data"].empty:
                plot_raw_gt(sample_data["gt_data"], f"Raw GT - {args.raw_sample_id} ({sample_data['label']})")
        else:
            print(f"Could not load raw sample '{args.raw_sample_id}'.")

    if args.processed_segment_id:
        segment_data = load_processed_segment(args.processed_segment_id)
        if segment_data:
            print(f"Visualizing Processed Segment: {args.processed_segment_id} (Label: {segment_data.get('label')})")
            plot_processed_data(segment_data, f"Processed - {args.processed_segment_id} ({segment_data['label']})")
        else:
            print(f"Could not load processed segment '{args.processed_segment_id}'.")

    if not args.raw_sample_id and not args.processed_segment_id and not args.list_raw_samples and not args.list_processed_segments:
        parser.print_help()

if __name__ == "__main__":
    main()
