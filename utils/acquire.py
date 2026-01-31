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

"""Unified data acquisition and preprocessing pipeline for Trajecto.

Integrates BLE data collection, two-tap synchronization, segmentation, and visualization.

Usage:
    python utils/acquire.py             # Interactive acquisition
    python utils/acquire.py --reprocess # Reprocess existing raw data
"""

import argparse
import asyncio
import os
import sys
import random
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, bessel, filtfilt, correlate, correlation_lags
from scipy.ndimage import gaussian_filter1d

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from receive import TrajectoDriver, RawImuPacket
except ImportError:
    print("Error: Could not import 'receive.py'. Ensure it is in the 'utils/' directory.")
    sys.exit(1)

from model.config import Config


DATA_DIR = "acquired_data"
RAW_HDF5_PATH = os.path.join(DATA_DIR, "raw_acquired_data.h5")
PROCESSED_DATASET_PATH = "data/dataset.h5"
VALIDATION_DATASET_PATH = "data/validation_dataset.h5"
SCALER_STATS_PATH = "data/scaler_stats.h5"

NUM_SESSIONS = 5
LABELS_PER_SESSION = 10
CONTINUOUS_SHAPES = [
    "Infinity Loop (∞)", "Spiral (In-Out)", "Spiral (Out-In)",
    "Zigzag", "Random Scribble (Fast)", "Random Scribble (Slow)",
    "Figure Eight (∞) - continuous", "Waves (horizontal) - continuous",
    "Waves (vertical) - continuous", "Circles (connected) - continuous",
    "Squares (connected) - continuous"
]
WORD_LIST = [
    "apple", "banana", "cat", "dog", "elephant", "fish", "grape", "house",
    "ice", "juice", "kite", "lion", "moon", "nest", "orange", "pen", "queen",
    "robot", "sun", "tree", "umbrella", "violet", "water", "xylophone",
    "yellow", "zebra", "writing", "draw", "circle", "square", "line",
    "computer", "science", "algorithm", "data", "system", "project", "research",
    "experiment", "solution", "analysis", "technology", "development",
    "engineer", "software", "hardware", "network", "cloud", "internet",
    "mobile", "application", "database", "security", "privacy", "innovation",
    "creative", "design", "geometric", "pattern", "number", "equation",
    "formula", "graph", "sketch"
]

TARGET_SAMPLING_RATE_HZ = Config.TARGET_SAMPLING_RATE_HZ
GRAVITY = Config.GRAVITY_MAGNITUDE  # Standard gravity (m/s²) - CODATA 2018
CUTOFF_FREQ_HZ = 5.0  # Reduced from 20.0 for better noise reduction
FILTER_ORDER = 4
SEGMENTATION_THRESHOLD = 0.01  # GT iPad screen force (0-1 normalized): exclude true zeros, keep actual writing (was: 0)
SEGMENTATION_MARGIN = 30
PIXEL_TO_METER = 0.0254 / 132.0  # iPad Retina display: 264 PPI ( 132 )
MAX_SEQUENCE_LENGTH = int(TARGET_SAMPLING_RATE_HZ * 35.0)
TRAIN_VAL_SPLIT = 1.0

SYNC_WINDOW_S = 3.0  # Window for correlation in  (reduced from 5.0 to support sequences ≥6s)
ROI_TAP_SEARCH_WINDOW_S = 2.0  # Initial search window for taps to define ROI in preprocess_single
ROI_MARGIN_S = 0.5  # Margin around taps to define writing ROI
MIN_SEGMENT_LENGTH_S = 1.0 # Minimum length for a valid segment (after margin)
STATIC_BUFFER_S = 2.5 # Static buffer duration to include before the segmeestimate_time_alignment_two_tapsnt

DIGITIZER_JUMP_THRESHOLD_M = 0.020  # 10mm - Maximum plausible single-frame position jump
DIGITIZER_JUMP_MIN_VELOCITY = 0.5  # m/s - Minimum velocity threshold to trigger jump detection (ignore static regions)

# Gravity vector in iPad frame (iPad lying flat on table, gravity points into table)
GRAVITY_IPAD_FRAME = np.array([0.0, 0.0, -GRAVITY])


def pencil_angles_to_rotation_matrix(azimuth: np.ndarray, altitude: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Convert Apple Pencil pose angles to rotation matrices (pencil body → iPad world).

    Apple Pencil coordinate convention:
    - Azimuth: Angle in iPad's x-y plane from +x axis (counterclockwise when viewed from +z)
    - Altitude: Angle above the x-y plane (0 = parallel to screen, π/2 = perpendicular)
    - Roll: Rotation around pencil's longitudinal axis

    The rotation sequence is: Rz(azimuth) @ Ry(altitude) @ Rx(roll)
    This gives R_pencil_to_ipad: transforms vectors from pencil frame to iPad frame.

    Args:
        azimuth: [N,] array of azimuth angles (radians)
        altitude: [N,] array of altitude angles (radians)
        roll: [N,] array of roll angles (radians)

    Returns:
        R: [N, 3, 3] rotation matrices (pencil → iPad)
    """
    N = len(azimuth)
    cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
    cos_alt, sin_alt = np.cos(altitude), np.sin(altitude)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    # Rz(azimuth) - rotation around iPad's Z axis
    Rz = np.zeros((N, 3, 3))
    Rz[:, 0, 0] = cos_az
    Rz[:, 0, 1] = -sin_az
    Rz[:, 1, 0] = sin_az
    Rz[:, 1, 1] = cos_az
    Rz[:, 2, 2] = 1.0

    # Ry(altitude) - rotation around Y axis (elevation)
    Ry = np.zeros((N, 3, 3))
    Ry[:, 0, 0] = cos_alt
    Ry[:, 0, 2] = sin_alt
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sin_alt
    Ry[:, 2, 2] = cos_alt

    # Rx(roll) - rotation around X axis (pencil axis)
    Rx = np.zeros((N, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cos_roll
    Rx[:, 1, 2] = -sin_roll
    Rx[:, 2, 1] = sin_roll
    Rx[:, 2, 2] = cos_roll

    # Combined: R = Rz @ Ry @ Rx
    R = np.einsum('nij,njk->nik', Rz, Ry)
    R = np.einsum('nij,njk->nik', R, Rx)

    return R


def compute_gravity_body_from_pencil_pose(
    azimuth: np.ndarray,
    altitude: np.ndarray,
    roll: np.ndarray,
    R_calibration: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute gravity vector in pencil body frame from Apple Pencil pose angles.

    Args:
        azimuth: [N,] array of azimuth angles (radians)
        altitude: [N,] array of altitude angles (radians)
        roll: [N,] array of roll angles (radians)
        R_calibration: [3, 3] optional rotation to align pencil frame with IMU frame

    Returns:
        gravity_body: [N, 3] unit gravity vectors in body frame
    """
    # Get rotation matrices: pencil → iPad
    R_pencil_to_ipad = pencil_angles_to_rotation_matrix(azimuth, altitude, roll)

    # Transpose to get iPad → pencil
    R_ipad_to_pencil = np.transpose(R_pencil_to_ipad, (0, 2, 1))

    # Transform gravity from iPad frame to pencil frame
    # gravity_pencil = R_ipad_to_pencil @ gravity_ipad
    gravity_body = np.einsum('nij,j->ni', R_ipad_to_pencil, GRAVITY_IPAD_FRAME)

    # Apply calibration rotation if provided (align pencil frame with IMU frame)
    if R_calibration is not None:
        gravity_body = np.einsum('ij,nj->ni', R_calibration, gravity_body)

    # Normalize to unit vectors
    norms = np.linalg.norm(gravity_body, axis=1, keepdims=True)
    gravity_body_unit = gravity_body / (norms + 1e-8)

    return gravity_body_unit


def calibrate_imu_pencil_alignment(
    accel_static: np.ndarray,
    azimuth_static: np.ndarray,
    altitude_static: np.ndarray,
    roll_static: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Calibrate the rotation offset between Apple Pencil pose frame and IMU body frame.

    During static periods, accelerometer measures gravity. We compare this with
    the gravity vector computed from pencil pose to find the alignment rotation.

    Uses Kabsch algorithm (SVD) to find optimal rotation.

    Args:
        accel_static: [N, 3] accelerometer readings during static period (m/s²)
        azimuth_static: [N,] azimuth angles during static period
        altitude_static: [N,] altitude angles during static period
        roll_static: [N,] roll angles during static period

    Returns:
        R_calibration: [3, 3] rotation matrix to apply to pencil-derived gravity
        alignment_error: Mean angular error after calibration (radians)
    """
    # Get gravity from accelerometer (normalize since we want direction only)
    accel_norms = np.linalg.norm(accel_static, axis=1, keepdims=True)
    gravity_from_accel = accel_static / (accel_norms + 1e-8)

    # Get gravity from pencil pose (without calibration)
    gravity_from_pencil = compute_gravity_body_from_pencil_pose(
        azimuth_static, altitude_static, roll_static, R_calibration=None
    )

    # Use Kabsch algorithm to find optimal rotation R such that:
    # gravity_from_accel ≈ R @ gravity_from_pencil
    #
    # Center the point clouds (already centered since unit vectors sum to ~0 over time)
    H = gravity_from_pencil.T @ gravity_from_accel  # [3, 3] cross-covariance

    U, S, Vt = np.linalg.svd(H)

    # Handle reflection case
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])

    R_calibration = Vt.T @ D @ U.T

    # Compute alignment error
    gravity_calibrated = np.einsum('ij,nj->ni', R_calibration, gravity_from_pencil)
    cos_angles = np.sum(gravity_calibrated * gravity_from_accel, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    alignment_errors = np.arccos(cos_angles)
    mean_error = np.mean(alignment_errors)

    return R_calibration, mean_error


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter. Preserves temporal alignment critical for training."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def bessel_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """Zero-phase Bessel lowpass filter. Better phase response than Butterworth for IMU data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = bessel(order, normal_cutoff, btype='low', analog=False, norm='phase')
    return filtfilt(b, a, data)

def pad_sequence(data: np.ndarray, max_len: int, is_velocity: bool = False) -> np.ndarray:
    """Pads or truncates sequence to fixed length.

    Args:
        data: Input sequence [seq_len, features]
        max_len: Target length
        is_velocity: If True, zero-pad; else edge-pad (repeat last value)

    Returns:
        Padded/truncated array [max_len, features]
    """
    seq_len = min(len(data), max_len)
    padded = np.zeros((max_len, data.shape[1]))
    padded[:seq_len, :] = data[:seq_len, :]

    if not is_velocity and seq_len < max_len:
        padded[seq_len:, :] = data[seq_len - 1, :]
    return padded

def detect_digitizer_jumps(pos_xyz: np.ndarray, force: np.ndarray, fs: float) -> Dict[str, Any]:
    """Detects unrealistic position jumps from digitizer errors.

    Args:
        pos_xyz: Position [N, 3] in meters
        force: Force [N], normalized 0-1
        fs: Sampling frequency (Hz)

    Returns:
        Dict with has_jumps, jump_indices, jump_magnitudes, max_jump, num_jumps
    """
    if len(pos_xyz) < 2:
        return {
            "has_jumps": False,
            "jump_indices": [],
            "jump_magnitudes": [],
            "max_jump": 0.0,
            "num_jumps": 0
        }

    pos_diff = np.diff(pos_xyz, axis=0)
    jump_distances = np.linalg.norm(pos_diff, axis=1)

    dt = 1.0 / fs
    velocities = jump_distances / dt

    is_moving = velocities > DIGITIZER_JUMP_MIN_VELOCITY
    is_jump = (jump_distances > DIGITIZER_JUMP_THRESHOLD_M) & is_moving

    jump_indices = np.where(is_jump)[0] + 1  # +1 because diff shifts indices
    jump_magnitudes = jump_distances[is_jump].tolist()

    result = {
        "has_jumps": bool(np.any(is_jump)),
        "jump_indices": jump_indices.tolist(),
        "jump_magnitudes": jump_magnitudes,
        "max_jump": float(np.max(jump_distances)) if len(jump_distances) > 0 else 0.0,
        "num_jumps": int(np.sum(is_jump))
    }

    return result

def preprocess_gt_data(gt_data_dict: Dict[str, np.ndarray], target_fs: float) -> pd.DataFrame:
    """Preprocesses iPad ground truth: unit conversion, outlier filtering, smoothing, resampling.

    Pipeline: pixels→meters, remove jumps (>100px), Gaussian smoothing (σ=1), PCHIP resample.
    Also preserves and interpolates Apple Pencil pose angles (azimuth, altitude, rollAngle).

    Args:
        gt_data_dict: Raw iPad data (timestamp, x, y, hoverDistance, force, azimuth, altitude, rollAngle)
        target_fs: Target frequency (Hz)

    Returns:
        Preprocessed DataFrame with x,y,z in meters plus pose angles, resampled to target_fs
    """
    df = pd.DataFrame(gt_data_dict)
    if "timestamp" not in df.columns:
        return df

    df["x"] = df["x"] * PIXEL_TO_METER
    df["y"] = df["y"] * PIXEL_TO_METER

    if "hoverDistance" in df.columns:
        df = df.rename(columns={"hoverDistance": "zOffset"})
        df["z"] = 12.49 * df["zOffset"].pow(0.78) / 1000.0

    pos_diff = np.diff(df[["x", "y"]].to_numpy(), axis=0)
    dist = np.linalg.norm(pos_diff, axis=1)

    valid_mask = np.insert(dist < (100.0 * PIXEL_TO_METER), 0, True)
    df = df[valid_mask].reset_index(drop=True)

    for col in ["x", "y", "z", "force"]:
        if col in df.columns:
            df[col] = gaussian_filter1d(df[col].to_numpy(), sigma=1.0)

    # Unwrap azimuth to handle ±π discontinuity before interpolation
    if "azimuth" in df.columns:
        df["azimuth"] = np.unwrap(df["azimuth"].to_numpy())

    original_time = df["timestamp"].to_numpy()
    sort_idx = np.argsort(original_time)
    original_time = original_time[sort_idx]
    unique_time, unique_idx = np.unique(original_time, return_index=True)
    original_time = unique_time

    if len(original_time) < 2:
        return df

    new_time = np.arange(
        original_time[0],
        original_time[-1] + (1.0 / target_fs) * 0.5,
        1.0 / target_fs,
    )
    upsampled_df = pd.DataFrame({"timestamp": new_time})

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != "timestamp":
            original_data = df[col].to_numpy()[sort_idx][unique_idx]
            if len(original_data) > 1:
                interp_func = PchipInterpolator(original_time, original_data)
                upsampled_df[col] = interp_func(new_time)
            else:
                 upsampled_df[col] = original_data[0]

    # Re-wrap azimuth back to [-π, π] after interpolation
    if "azimuth" in upsampled_df.columns:
        upsampled_df["azimuth"] = np.arctan2(
            np.sin(upsampled_df["azimuth"]),
            np.cos(upsampled_df["azimuth"])
        )

    return upsampled_df

def estimate_time_alignment_two_taps(sig_ref: np.ndarray, sig_target: np.ndarray, fs: float) -> Tuple[float, float, bool]:
    """Estimates clock drift and offset using two-tap correlation.

    Correlates start/end tap windows to calculate linear drift model:
    target_time[i] = source_time[i] * (1 + slope) + intercept

    Args:
        sig_ref: ESP32 acceleration (normalized)
        sig_target: iPad force (normalized)
        fs: Sampling frequency (Hz)

    Returns:
        (slope, intercept, success) where slope is dimensionless drift coefficient
    """
    n = min(len(sig_ref), len(sig_target))
    window = int(SYNC_WINDOW_S * fs)
    if n < 2 * window:
        return 0.0, 0.0, False

    corr_start = correlate(sig_ref[:window] - np.mean(sig_ref[:window]),
                           sig_target[:window] - np.mean(sig_target[:window]), mode='full')
    lag_start = correlation_lags(window, window, mode='full')[np.argmax(corr_start)]

    corr_end = correlate(sig_ref[-window:] - np.mean(sig_ref[-window:]),
                         sig_target[-window:] - np.mean(sig_target[-window:]), mode='full')
    lag_end = correlation_lags(window, window, mode='full')[np.argmax(corr_end)]

    dist_target = n - window
    dist_ref = dist_target + lag_end - lag_start

    if dist_ref == 0: return 0.0, 0.0, False

    slope = (dist_ref / dist_target) - 1.0
    intercept = lag_start

    return slope, intercept, True

def find_force_segments(df_gt: pd.DataFrame, threshold: int, margin: int) -> List[Tuple[int, int]]:
    """Detects writing segments where force exceeds threshold.

    Args:
        df_gt: DataFrame with 'force' column (0-1 normalized)
        threshold: Force threshold for active detection
        margin: Samples to add before/after each segment

    Returns:
        List of (start_idx, end_idx) tuples. Returns [(0, len)] if no segments found.
    """
    if "force" not in df_gt.columns: return [(0, len(df_gt))]
    force = df_gt["force"].to_numpy()
    active = force > threshold
    if not np.any(active): return [(0, len(force))]

    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if active[0]: starts = np.insert(starts, 0, 0)
    if active[-1]: ends = np.append(ends, len(force) - 1)

    segments = []
    for s, e in zip(starts, ends):
        segments.append((max(0, s - margin), min(len(force), e + margin + 1)))
    return segments

class AcquisitionManager:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        self.driver = None
        self.pen_buffer = []  # List[Dict]: IMU data in SI units (m/s², rad/s)
        self.global_counter = 0
        self.ipad_counter = 1

        self._init_h5()

    def _init_h5(self):
        """Initializes HDF5 file and resumes global counter."""
        with h5py.File(RAW_HDF5_PATH, "a") as f:
            grp = f.require_group("raw_data")
            if len(grp.keys()) > 0:
                indices = []
                for k in grp.keys():
                    try:
                        indices.append(int(k.split("_")[-1]))
                    except:
                        pass
                if indices:
                    self.global_counter = max(indices)
                    print(f"[Init] Resuming Global ID from {self.global_counter}")
            else:
                print("[Init] Starting Global ID from 0")

    async def connect(self):
        print("Connecting to Device...")
        self.driver = TrajectoDriver(raw_callback=self._on_data, verbose=True)
        if await self.driver.connect():
            print("Connected!")
            return True
        return False

    async def disconnect(self):
        if self.driver:
            await self.driver.disconnect()
            print("Disconnected.")

    def _on_data(self, packet: RawImuPacket):
        """Callback for raw IMU packets. Stores data in SI units (m/s², rad/s)."""
        data = {
            "time": packet.timestamp_us / 1_000_000.0,  # microseconds → seconds
            "accel_x": packet.accel[0],  # m/s² (from firmware)
            "accel_y": packet.accel[1],  # m/s²
            "accel_z": packet.accel[2],  # m/s²
            "gyro_x": packet.gyro[0],    # rad/s (from firmware)
            "gyro_y": packet.gyro[1],    # rad/s
            "gyro_z": packet.gyro[2],    # rad/s
            "fsr": packet.force          # raw ADC value
        }
        self.pen_buffer.append(data)

    async def setup_counters(self):
        print(f"\nNext Global ID: {self.global_counter + 1}")
        try:
            val = input(f"Enter iPad Start Number (Default {self.ipad_counter}): ").strip()
            if val.isdigit():
                self.ipad_counter = int(val)
        except EOFError:
            pass
        print(f"Mapping: iPad 'Sample_{self.ipad_counter}.csv' -> Global 'sample_{self.global_counter + 1:03d}'")

    async def acquire_sequence(self, label: str):
        """Executes Tap-Wait-Write-Tap acquisition protocol."""
        self.pen_buffer.clear()
        if not self.driver:
            return False

        print(f"\n--- ACQUISITION: {(self.global_counter + 1):03d} ---")
        input("Press Enter to START recording...")

        # Start streaming in RAW mode (mode=0)
        if not await self.driver.start_streaming(mode=0):
            print("Failed to start streaming!")
            return False

        print("1. TAP (Start Sync)")
        os.system("afplay /System/Library/Sounds/Tink.aiff &")
        await asyncio.sleep(1.5)

        print("2. WAIT (Calib)...")
        for i in range(2, 0, -1):
            print(f"         {i}...")
            await asyncio.sleep(1.0)

        print(f"3. WRITE NOW! >>> {label}")
        os.system("afplay /System/Library/Sounds/Glass.aiff &")
        input("   Press Enter when FINISHED writing...")

        print("4. STOP PEN (Wait)")
        os.system("afplay /System/Library/Sounds/Purr.aiff &")
        await asyncio.sleep(1.0)

        print("5. TAP (End Sync)")
        os.system("afplay /System/Library/Sounds/Tink.aiff &")
        await asyncio.sleep(1.5)

        await self.driver.stop_streaming()
        print(f"Captured {len(self.pen_buffer)} samples.")
        return True

    def load_gt_file(self, index: int) -> Optional[pd.DataFrame]:
        """Loads iPad CSV by index."""
        fname = f"Sample_{index}.csv"
        path = os.path.join(self.data_dir, fname)

        if not os.path.exists(path):
            input(f"Please transfer {fname} to {self.data_dir} and press Enter...")

        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"Error loading CSV: {e}")
                return None
        return None

    def preprocess_single(self, pen_data: List[Dict], df_gt_raw: pd.DataFrame, sample_name: str = "temp") -> Tuple[Optional[List[Dict]], Dict]:
        """Preprocesses raw acquisition: two-tap sync, jump detection, segmentation, filtering.

        Pipeline: sync IMU/GT via correlation → detect digitizer jumps → extract ROI from taps
        → filter FSR → format output with static buffer for ESKF init.

        Args:
            pen_data: IMU dicts (time, accel_xyz, gyro_xyz, fsr) in SI units
            df_gt_raw: iPad CSV (timestamp, x, y, hoverDistance, force)
            sample_name: Identifier for debugging

        Returns:
            (segments, debug_info) where segments is list of dicts with sensor/gt_pos/gt_vel
        """
        debug_info = {}

        if not pen_data:
            return None, {"error": "No pen data"}
        df_sensor = pd.DataFrame(pen_data)

        df_gt_proc = preprocess_gt_data(df_gt_raw.to_dict(orient='list'), TARGET_SAMPLING_RATE_HZ)

        acc_norm = np.sqrt(df_sensor["accel_x"]**2 + df_sensor["accel_y"]**2 + df_sensor["accel_z"]**2)
        sig_sensor = (acc_norm - GRAVITY) / (acc_norm.std() + 1e-6)

        gt_force = df_gt_proc["force"] if "force" in df_gt_proc.columns else np.zeros(len(df_gt_proc))
        sig_gt = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)

        slope, intercept, success = estimate_time_alignment_two_taps(sig_sensor.to_numpy(), sig_gt.to_numpy(), TARGET_SAMPLING_RATE_HZ)

        debug_info["sync_success"] = success
        debug_info["slope"] = slope
        debug_info["intercept"] = intercept
        duration_s = len(df_gt_proc) / TARGET_SAMPLING_RATE_HZ
        time_offset_ms = (intercept / TARGET_SAMPLING_RATE_HZ) * 1000.0  # samples → ms
        drift_ppm = slope * 1e6  # dimensionless → parts per million
        total_drift_ms = (slope * len(df_gt_proc) / TARGET_SAMPLING_RATE_HZ) * 1000.0

        print(f"\n{'='*60}")
        print(f"TIME-LAG COMPENSATION: {sample_name}")
        print(f"{'='*60}")
        print(f"Status:   {'Two-Tap Sync' if success else 'Fallback Correlation'}")
        print(f"Duration: {duration_s:.1f}s ({len(df_gt_proc)} samples @ {TARGET_SAMPLING_RATE_HZ}Hz)")
        print(f"\nCompensation Values:")
        print(f"  • Intercept: {intercept:+8.1f} samples = {time_offset_ms:+7.2f} ms")
        print(f"  • Slope:     {slope:+.6f}         = {drift_ppm:+7.1f} ppm")
        print(f"  • Total Drift: {total_drift_ms:+.2f} ms over {duration_s:.1f}s")
        print(f"\nInterpretation:")
        print(f"  ESP32 clock was {abs(time_offset_ms):.1f}ms {'ahead' if time_offset_ms > 0 else 'behind'} at start")
        print(f"  ESP32 clock runs {abs(drift_ppm):.1f}ppm {'faster' if drift_ppm > 0 else 'slower'} than iPad")
        print(f"{'='*60}")

        if success:
            target_indices = np.arange(len(df_gt_proc))
            source_indices = target_indices * (1.0 + slope) + intercept

            new_sensor = {}
            for col in df_sensor.columns:
                new_sensor[col] = np.interp(source_indices, np.arange(len(df_sensor)), df_sensor[col], left=np.nan, right=np.nan)
            df_sensor_aligned = pd.DataFrame(new_sensor)

            valid = ~df_sensor_aligned.isna().any(axis=1)
            if valid.sum() == 0:
                return None, {"error": "Sync resulted in empty data"}

            first, last = valid.idxmax(), valid[::-1].idxmax()
            df_sensor_aligned = df_sensor_aligned.iloc[first:last+1].reset_index(drop=True)
            df_gt_aligned = df_gt_proc.iloc[first:last+1].reset_index(drop=True)
        else:
            corr = correlate(sig_sensor, sig_gt, mode='full')
            lag = correlation_lags(len(sig_sensor), len(sig_gt), mode='full')[np.argmax(corr)]
            debug_info["lag"] = lag

            lag_ms = (lag / TARGET_SAMPLING_RATE_HZ) * 1000.0
            print(f"\nFallback: Simple cross-correlation")
            print(f"  • Lag: {lag:+d} samples = {lag_ms:+.2f} ms")
            print(f"  (No clock drift correction - assumes constant rate)")

            if lag > 0:
                df_sensor_aligned = df_sensor.iloc[lag:].reset_index(drop=True)
                df_gt_aligned = df_gt_proc
            else:
                df_sensor_aligned = df_sensor
                df_gt_aligned = df_gt_proc.iloc[abs(lag):].reset_index(drop=True)

            min_len = min(len(df_sensor_aligned), len(df_gt_aligned))
            df_sensor_aligned = df_sensor_aligned.iloc[:min_len]
            df_gt_aligned = df_gt_aligned.iloc[:min_len]

        debug_info["sensor_aligned"] = df_sensor_aligned
        debug_info["gt_aligned"] = df_gt_aligned
        gt_pos = df_gt_aligned[["x", "y", "z"]].to_numpy()
        gt_force_arr = df_gt_aligned["force"].to_numpy() if "force" in df_gt_aligned.columns else np.zeros(len(df_gt_aligned))

        jump_info = detect_digitizer_jumps(gt_pos, gt_force_arr, TARGET_SAMPLING_RATE_HZ)
        debug_info["digitizer_jumps"] = jump_info

        if jump_info["has_jumps"]:
            print(f"\n{'='*60}")
            print(f"DIGITIZER JUMP DETECTION: {sample_name}")
            print(f"{'='*60}")
            print(f"Detected {jump_info['num_jumps']} position jump(s) in ground truth data:")
            for idx, mag in zip(jump_info["jump_indices"], jump_info["jump_magnitudes"]):
                time_s = idx / TARGET_SAMPLING_RATE_HZ
                print(f"  • Sample {idx} (t={time_s:.2f}s): {mag*1000:.1f} mm jump")
            print(f"\nMax jump: {jump_info['max_jump']*1000:.1f} mm (Threshold: {DIGITIZER_JUMP_THRESHOLD_M*1000:.1f} mm)")
            print(f"\nPossible causes:")
            print(f"  • iPad digitizer glitches during data collection")
            print(f"  • Pencil tracking errors (occlusion, edge effects)")
            print(f"  • Rapid movements exceeding digitizer sample rate")
            print(f"\nNote: Review visualization carefully before saving.")
            print(f"{'='*60}")

        gt_f = df_gt_aligned["force"].to_numpy() if "force" in df_gt_aligned.columns else np.zeros(len(df_gt_aligned))
        roi_start, roi_end = 0, len(gt_f)

        window = int(ROI_TAP_SEARCH_WINDOW_S * TARGET_SAMPLING_RATE_HZ)
        margin_samples = int(ROI_MARGIN_S * TARGET_SAMPLING_RATE_HZ)

        if len(gt_f) > 2 * window:
             start_search_region = gt_f[:window]
             start_tap_idx = np.argmax(start_search_region)

             end_search_region = gt_f[-window:]
             end_tap_local = np.argmax(end_search_region)
             end_tap_idx = len(gt_f) - window + end_tap_local

             roi_start = start_tap_idx + margin_samples
             roi_end = end_tap_idx - margin_samples

             if roi_end <= roi_start:
                 roi_start = 0
                 roi_end = len(gt_f)
             print(f"\nTap Detection Debug:")
             print(f"  Start tap at: {start_tap_idx} samples ({start_tap_idx/TARGET_SAMPLING_RATE_HZ:.2f}s), force={start_search_region[start_tap_idx]:.3f}")
             print(f"  End tap at:   {end_tap_idx} samples ({end_tap_idx/TARGET_SAMPLING_RATE_HZ:.2f}s), force={gt_f[end_tap_idx]:.3f}")
             print(f"  ROI: [{roi_start}:{roi_end}] = [{roi_start/TARGET_SAMPLING_RATE_HZ:.2f}s:{roi_end/TARGET_SAMPLING_RATE_HZ:.2f}s]")

        segments = []
        raw_segs = find_force_segments(df_gt_aligned, SEGMENTATION_THRESHOLD, SEGMENTATION_MARGIN)

        for s, e in raw_segs:
            s_ = max(s, roi_start)
            e_ = min(e, roi_end)
            if e_ - s_ > int(MIN_SEGMENT_LENGTH_S * TARGET_SAMPLING_RATE_HZ):
                segments.append((s_, e_))

        if not segments:
             if roi_end - roi_start > int(MIN_SEGMENT_LENGTH_S * TARGET_SAMPLING_RATE_HZ):
                 segments.append((roi_start, roi_end))
             else:
                 return None, {"error": "No valid segments found"}

        final_start, final_end = segments[0][0], segments[-1][1]

        static_buffer_samples = int(STATIC_BUFFER_S * TARGET_SAMPLING_RATE_HZ)
        final_start_before_buffer = final_start
        final_start = max(0, final_start - static_buffer_samples)
        print(f"\nSegmentation Debug:")
        print(f"  Found {len(raw_segs)} raw segments, {len(segments)} after ROI clipping")
        print(f"  Final segment before static buffer: [{final_start_before_buffer}:{final_end}] = [{final_start_before_buffer/TARGET_SAMPLING_RATE_HZ:.2f}s:{final_end/TARGET_SAMPLING_RATE_HZ:.2f}s]")
        print(f"  Static buffer: {static_buffer_samples} samples ({STATIC_BUFFER_S}s)")
        print(f"  Final segment after static buffer:  [{final_start}:{final_end}] = [{final_start/TARGET_SAMPLING_RATE_HZ:.2f}s:{final_end/TARGET_SAMPLING_RATE_HZ:.2f}s]")
        print(f"  Total duration: {(final_end - final_start)/TARGET_SAMPLING_RATE_HZ:.2f}s")

        debug_info["segment"] = (final_start, final_end)

        df_s_seg = df_sensor_aligned.iloc[final_start:final_end].reset_index(drop=True)
        df_g_seg = df_gt_aligned.iloc[final_start:final_end].reset_index(drop=True)

        # Apply Bessel lowpass filter to FSR at 5Hz
        if "fsr" in df_s_seg.columns:
            df_s_seg["fsr"] = bessel_lowpass_filter(df_s_seg["fsr"], CUTOFF_FREQ_HZ, TARGET_SAMPLING_RATE_HZ, FILTER_ORDER)

        processed_segments = []

        gt_pos = df_g_seg[["x", "y", "z"]].to_numpy()
        gt_pos[:, 2] = np.maximum(gt_pos[:, 2], 1e-7)

        gt_vel = np.gradient(gt_pos, 1.0/TARGET_SAMPLING_RATE_HZ, axis=0)

        fsr = df_s_seg["fsr"].to_numpy().reshape(-1, 1) if "fsr" in df_s_seg.columns else np.zeros((len(df_s_seg), 1))
        sensor_final = np.hstack([
            df_s_seg[["accel_x", "accel_y", "accel_z"]].to_numpy(),
            df_s_seg[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(),
            fsr
        ])

        # Compute gravity GT from Apple Pencil pose angles
        gt_gravity_b = None
        pose_cols = ["azimuth", "altitude", "rollAngle"]
        if all(col in df_g_seg.columns for col in pose_cols):
            azimuth = df_g_seg["azimuth"].to_numpy()
            altitude = df_g_seg["altitude"].to_numpy()
            roll = df_g_seg["rollAngle"].to_numpy()

            # Use the static buffer period (first STATIC_BUFFER_S seconds) for calibration
            # The static buffer is at the beginning of the segment before writing starts
            static_samples = min(static_buffer_samples, len(df_s_seg))
            if static_samples > 10:  # Need enough samples for robust calibration
                accel_static = df_s_seg[["accel_x", "accel_y", "accel_z"]].to_numpy()[:static_samples]

                # Filter out hover/invalid pose data during calibration
                # Check if pencil was in contact (force > 0) or just use accel magnitude
                accel_norms = np.linalg.norm(accel_static, axis=1)
                static_mask = (accel_norms > 8.0) & (accel_norms < 12.0)  # Near gravity

                if static_mask.sum() > 10:
                    R_calibration, alignment_error = calibrate_imu_pencil_alignment(
                        accel_static[static_mask],
                        azimuth[:static_samples][static_mask],
                        altitude[:static_samples][static_mask],
                        roll[:static_samples][static_mask]
                    )
                    print(f"\nGravity Calibration:")
                    print(f"  Static samples used: {static_mask.sum()}/{static_samples}")
                    print(f"  Alignment error: {np.degrees(alignment_error):.2f}°")

                    # Compute gravity for entire sequence using calibration
                    gt_gravity_b = compute_gravity_body_from_pencil_pose(
                        azimuth, altitude, roll, R_calibration
                    )
                else:
                    print(f"\nWarning: Insufficient static samples for gravity calibration")
                    print(f"  Valid static samples: {static_mask.sum()}, required: >10")
                    # Fallback: compute without calibration
                    gt_gravity_b = compute_gravity_body_from_pencil_pose(
                        azimuth, altitude, roll, R_calibration=None
                    )
            else:
                print(f"\nWarning: Static buffer too short for calibration ({static_samples} samples)")
                gt_gravity_b = compute_gravity_body_from_pencil_pose(
                    azimuth, altitude, roll, R_calibration=None
                )
        else:
            print(f"\nWarning: Pose columns not found in GT data. Skipping gravity GT.")
            print(f"  Available columns: {list(df_g_seg.columns)}")

        processed_segments.append({
            "name": f"{sample_name}_seg0",
            "sensor": sensor_final,
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_gravity_b": gt_gravity_b
        })

        return processed_segments, debug_info

    def visualize_sync(self, debug_info: Dict, label: str):
        """Displays comprehensive data integrity plots (sync, trajectory, jumps)."""
        if "sensor_aligned" not in debug_info:
            print("Cannot visualize: No aligned data.")
            return

        df_s = debug_info["sensor_aligned"]
        df_g = debug_info["gt_aligned"]
        seg_start, seg_end = debug_info.get("segment", (0, 0))
        jump_info = debug_info.get("digitizer_jumps", {})

        time_s = np.arange(len(df_s)) / TARGET_SAMPLING_RATE_HZ

        acc_norm = np.sqrt(df_s["accel_x"]**2 + df_s["accel_y"]**2 + df_s["accel_z"]**2)
        gt_force = df_g["force"] if "force" in df_g.columns else np.zeros(len(df_g))
        fsr = df_s["fsr"].to_numpy() if "fsr" in df_s.columns else np.zeros(len(df_s))

        gt_x = df_g["x"].to_numpy() if "x" in df_g.columns else np.zeros(len(df_g))
        gt_y = df_g["y"].to_numpy() if "y" in df_g.columns else np.zeros(len(df_g))
        gt_z = df_g["z"].to_numpy() if "z" in df_g.columns else np.zeros(len(df_g))

        gyro_norm = np.sqrt(df_s["gyro_x"]**2 + df_s["gyro_y"]**2 + df_s["gyro_z"]**2)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        acc_vis = (acc_norm - acc_norm.mean()) / (acc_norm.std() + 1e-6)
        force_vis = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)
        ax1.plot(time_s, acc_vis, label="Sensor Accel (Norm)", alpha=0.7, linewidth=1)
        ax1.plot(time_s, force_vis, label="GT Force (Norm)", alpha=0.7, linewidth=1)
        ax1.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2, label="Selected Segment")

        if jump_info.get("has_jumps"):
            for idx in jump_info["jump_indices"]:
                if idx < len(time_s):
                    ax1.axvline(time_s[idx], color='red', linestyle='--', alpha=0.6, linewidth=2)

        ax1.set_title("Synchronization Check")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normalized Signal")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(gt_x * 1000, gt_y * 1000, 'b-', linewidth=1.5, alpha=0.7)
        ax2.plot(gt_x[0] * 1000, gt_y[0] * 1000, 'go', markersize=8, label='Start')
        ax2.plot(gt_x[-1] * 1000, gt_y[-1] * 1000, 'ro', markersize=8, label='End')

        if jump_info.get("has_jumps"):
            for idx in jump_info["jump_indices"]:
                if idx < len(gt_x):
                    ax2.plot(gt_x[idx] * 1000, gt_y[idx] * 1000, 'rx', markersize=12,
                            markeredgewidth=3, label='Jump' if idx == jump_info["jump_indices"][0] else '')

        ax2.set_title("GT Trajectory (X-Y)" + (f" [{jump_info['num_jumps']} jump(s)]" if jump_info.get("has_jumps") else ""))
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Y (mm)")
        ax2.axis('equal')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_s, gt_x * 1000, label='X', alpha=0.7, linewidth=1)
        ax3.plot(time_s, gt_y * 1000, label='Y', alpha=0.7, linewidth=1)
        ax3.plot(time_s, gt_z * 1000, label='Z', alpha=0.7, linewidth=1)
        ax3.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)

        if jump_info.get("has_jumps"):
            for idx in jump_info["jump_indices"]:
                if idx < len(time_s):
                    ax3.axvline(time_s[idx], color='red', linestyle='--', alpha=0.6, linewidth=2)

        ax3.set_title("GT Position vs Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position (mm)")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_s, fsr, 'purple', alpha=0.7, linewidth=1)
        ax4.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)
        ax4.set_title("FSR Signal (Sensor)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("FSR (ADC)")
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time_s, df_s["accel_x"], label='X', alpha=0.7, linewidth=1)
        ax5.plot(time_s, df_s["accel_y"], label='Y', alpha=0.7, linewidth=1)
        ax5.plot(time_s, df_s["accel_z"], label='Z', alpha=0.7, linewidth=1)
        ax5.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)
        ax5.set_title("Acceleration (Sensor)")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Accel (m/s²)")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time_s, df_s["gyro_x"], label='X', alpha=0.7, linewidth=1)
        ax6.plot(time_s, df_s["gyro_y"], label='Y', alpha=0.7, linewidth=1)
        ax6.plot(time_s, df_s["gyro_z"], label='Z', alpha=0.7, linewidth=1)
        ax6.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)
        ax6.set_title("Gyroscope (Sensor)")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Gyro (rad/s)")
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        duration = len(df_s) / TARGET_SAMPLING_RATE_HZ
        seg_duration = (seg_end - seg_start) / TARGET_SAMPLING_RATE_HZ
        path_length_2d = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)) * 1000  # mm
        path_length_3d = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2 + np.diff(gt_z)**2)) * 1000

        gt_vel = np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2 + np.diff(gt_z)**2) * TARGET_SAMPLING_RATE_HZ
        avg_velocity = np.mean(gt_vel) * 1000
        max_velocity = np.max(gt_vel) * 1000

        acc_mean = np.mean(acc_norm)
        acc_std = np.std(acc_norm)
        digitizer_status = ""
        if jump_info.get("has_jumps"):
            digitizer_status = f"\nDigitizer Jumps: {jump_info['num_jumps']} detected, max={jump_info['max_jump']*1000:.1f}mm (red markers on plots)"

        stats_text = f"""
                DATA INTEGRITY SUMMARY - {label}{digitizer_status}
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Total Duration: {duration:.2f}s ({len(df_s)} samples)  |  Segment Duration: {seg_duration:.2f}s ({seg_end-seg_start} samples)
                Path Length: 2D={path_length_2d:.1f}mm, 3D={path_length_3d:.1f}mm  |  Velocity: avg={avg_velocity:.1f}mm/s, max={max_velocity:.1f}mm/s
                Accel Magnitude: mean={acc_mean:.2f}m/s², std={acc_std:.2f}m/s²  |  FSR Range: [{np.min(fsr):.1f}, {np.max(fsr):.1f}]
                GT Position Range: X=[{np.min(gt_x)*1000:.1f}, {np.max(gt_x)*1000:.1f}]mm, Y=[{np.min(gt_y)*1000:.1f}, {np.max(gt_y)*1000:.1f}]mm, Z=[{np.min(gt_z)*1000:.1f}, {np.max(gt_z)*1000:.1f}]mm
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Close window to continue...
                        """

        text_color = 'red' if jump_info.get("has_jumps") else 'black'
        bg_color = 'lightyellow' if jump_info.get("has_jumps") else 'wheat'

        ax7.text(0.5, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 color=text_color,
                 bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5, edgecolor='red' if jump_info.get("has_jumps") else 'gray'))

        title_suffix = f" (Digitizer Jumps: {jump_info['num_jumps']})" if jump_info.get("has_jumps") else ""
        fig.suptitle(f"Data Integrity Check: {label}{title_suffix}", fontsize=14, fontweight='bold')
        plt.show()

    def save_data(self, pen_data, df_gt, processed_segs, label, ipad_idx: int):
        """Saves raw and processed data to HDF5."""
        global_idx = self.global_counter + 1
        name = f"sample_{global_idx:03d}"

        with h5py.File(RAW_HDF5_PATH, "a") as f:
            grp = f.require_group("raw_data").create_group(name)

            # Save Pen
            pen_keys = list(pen_data[0].keys())
            dtype = [(k, 'f8') if isinstance(pen_data[0][k], (int, float)) else (k, h5py.string_dtype()) for k in pen_keys]
            arr = np.array([tuple(d.get(k) for k in pen_keys) for d in pen_data], dtype=dtype)
            grp.create_dataset("pen_data", data=arr)

            # Save GT
            rec = df_gt.to_records(index=False)
            grp.create_dataset("gt_data", data=rec)

            grp.attrs["original_label"] = label
            grp.attrs["ipad_file_index"] = ipad_idx
            grp.attrs["user_approved"] = True

        with h5py.File(PROCESSED_DATASET_PATH, "a") as f:
            for i, seg in enumerate(processed_segs):
                seg_name = f"{name}_seg{i}"
                if seg_name in f:
                    del f[seg_name]
                g = f.create_group(seg_name)

                g.create_dataset("sensor_data", data=pad_sequence(seg["sensor"], MAX_SEQUENCE_LENGTH))
                g.create_dataset("gt_pos_data", data=pad_sequence(seg["gt_pos"], MAX_SEQUENCE_LENGTH))
                g.create_dataset("gt_vel_data", data=pad_sequence(seg["gt_vel"], MAX_SEQUENCE_LENGTH))

                # Save gravity GT from Apple Pencil pose if available
                if seg.get("gt_gravity_b") is not None:
                    g.create_dataset("gt_gravity_b_data", data=pad_sequence(seg["gt_gravity_b"], MAX_SEQUENCE_LENGTH))

                g.attrs["original_label"] = label
                g.attrs["sequence_length"] = len(seg["sensor"])

        self.global_counter += 1
        print(f"[Saved] {name} (Raw + {len(processed_segs)} Segments)")

    def update_scaler_stats(self):
        """Recalculates normalization statistics from processed dataset."""
        print("\nUpdating Scaler Statistics...")
        if not os.path.exists(PROCESSED_DATASET_PATH):
            print("  No dataset found at", PROCESSED_DATASET_PATH)
            return

        all_sensor_data = []
        try:
            with h5py.File(PROCESSED_DATASET_PATH, "r") as f:
                for key in f.keys():
                    grp = f[key]
                    if "sensor_data" in grp and "sequence_length" in grp.attrs:
                        data = grp["sensor_data"][:]
                        seq_len = int(grp.attrs["sequence_length"])
                        real_data = data[:seq_len]
                        all_sensor_data.append(real_data)

            if not all_sensor_data:
                print("  No valid sensor data found to calculate stats.")
                return

            all_sensor_stacked = np.vstack(all_sensor_data)
            mean = np.mean(all_sensor_stacked, axis=0)
            std = np.std(all_sensor_stacked, axis=0)
            std[std == 0] = 1.0

            with h5py.File(SCALER_STATS_PATH, "w") as f:
                f.create_dataset("mean", data=mean)
                f.create_dataset("std", data=std)

            print(f"  [Updated] Scaler stats saved to {SCALER_STATS_PATH}")
            print(f"  Total samples used for stats: {len(all_sensor_data)}")

        except Exception as e:
            print(f"  [Error] Failed to update scaler stats: {e}")
            traceback.print_exc()

    async def run_interactive(self):
        await self.connect()
        await self.setup_counters()

        try:
            for session_num in range(NUM_SESSIONS):
                labels = list(CONTINUOUS_SHAPES)
                # Add random words
                for _ in range(LABELS_PER_SESSION - len(labels)):
                        labels.append(" ".join(random.choices(WORD_LIST, k=random.randint(10, 20))))

                labels = labels[:LABELS_PER_SESSION]
                random.shuffle(labels)

                session_buffer = []
                print(f"\n=== SESSION {session_num + 1}/{NUM_SESSIONS} ===")
                for i, label in enumerate(labels):
                    print(f"\n>>> Task ({i+1}/{len(labels)}): {label}")

                    while True:
                        if await self.acquire_sequence(label):
                            num_samples = len(self.pen_buffer)
                            duration_s = num_samples / 50.0
                            print(f"    -> Captured {num_samples} samples (~{duration_s:.1f}s)")
                            session_buffer.append({
                                "label": label,
                                "pen_data": list(self.pen_buffer),
                                "ipad_idx": self.ipad_counter
                            })

                            self.ipad_counter += 1
                            break
                        else:
                            print("    [!] Acquisition Failed.")
                            if input("    Retry this label? (y/n): ").lower() != 'y':
                                break

                print(f"\n=== SESSION {session_num + 1} ACQUISITION COMPLETE ===")
                if session_buffer:
                    print(f"Expected files: Sample_{session_buffer[0]['ipad_idx']}.csv to Sample_{session_buffer[-1]['ipad_idx']}.csv")
                    input(">>> Please transfer these files to 'acquired_data/' folder. Press Enter when ready...")

                    print("\n--- Verifying Data ---")

                    any_saved = False
                    for item in session_buffer:
                        label = item["label"]
                        pen_data = item["pen_data"]
                        ipad_idx = item["ipad_idx"]

                        print(f"\nProcessing: '{label}' (iPad File: Sample_{ipad_idx}.csv)")

                        df_gt = self.load_gt_file(ipad_idx)

                        if df_gt is None:
                            print(f"  [Skipped] GT file missing for '{label}'.")
                            continue

                        temp_name = f"sample_{self.global_counter + 1:03d}"
                        proc_segs, debug = self.preprocess_single(pen_data, df_gt, temp_name)

                        if proc_segs:
                            self.visualize_sync(debug, label)

                            choice = input(f"  [{label}] Save (s) / Discard (d)? ").lower()
                            if choice == 's':
                                self.save_data(pen_data, df_gt, proc_segs, label, ipad_idx)
                                any_saved = True
                            else:
                                print("  [Discarded]")
                        else:
                            print(f"  [Failed] Preprocessing Error: {debug.get('error')}")

                    if any_saved:
                        self.update_scaler_stats()
                else:
                    print("No samples collected in this session.")

        finally:
            await self.disconnect()

    def run_reprocess(self, approve_all: bool = False, no_viz: bool = False):
        """Reprocesses existing raw data with optional interactive review.

        Args:
            approve_all: If True, automatically approve all successfully processed samples.
            no_viz: If True, skip visualization plots (useful with approve_all for batch processing).
        """
        print(f"Reprocessing raw data from {RAW_HDF5_PATH}...")
        if approve_all:
            print("  Mode: Auto-approve all valid samples")
        if no_viz:
            print("  Mode: Visualization disabled")

        if not os.path.exists(RAW_HDF5_PATH):
            print("Raw data file not found.")
            return

        samples = []
        with h5py.File(RAW_HDF5_PATH, "r") as f:
            if "raw_data" not in f:
                print("No 'raw_data' group found.")
                return

            for k in f["raw_data"]:
                grp = f["raw_data"][k]
                samples.append({
                    "name": k,
                    "pen_data": pd.DataFrame(grp["pen_data"][:]).to_dict('records'),
                    "gt_data": pd.DataFrame(grp["gt_data"][:]),
                    "label": grp.attrs.get("original_label", "unknown")
                })

        samples.sort(key=lambda x: x["name"])
        mode_str = "auto-approve" if approve_all else "interactive review"
        print(f"Found {len(samples)} samples. Starting {mode_str}...")

        all_segments = []
        failed_samples = []

        for i, s in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Processing: {s['name']} (Label: {s['label']})")

            proc_segs, debug = self.preprocess_single(s["pen_data"], s["gt_data"], s["name"])

            if approve_all:
                # Auto-approve mode
                if proc_segs:
                    if not no_viz:
                        self.visualize_sync(debug, s['label'])
                    for seg in proc_segs:
                        seg["original_label"] = s["label"]
                        all_segments.append(seg)
                    print(f"  -> Auto-approved. ({len(proc_segs)} segments)")
                else:
                    failed_samples.append(s['name'])
                    print(f"  [Error] Preprocessing failed: {debug.get('error')}")
                    print(f"  -> Skipped (failed).")
            else:
                # Interactive mode
                if proc_segs:
                    if not no_viz:
                        self.visualize_sync(debug, s['label'])
                    choice = input(f"  Action for {s['name']}? [ (A)pprove / (S)kip / (Q)uit ]: ").lower().strip()
                else:
                    print(f"  [Error] Preprocessing failed: {debug.get('error')}")
                    if not no_viz and "sensor_aligned" in debug:
                        print("  Showing debug plot...")
                        self.visualize_sync(debug, s['label'])
                    choice = input(f"  Action for {s['name']} (Failed)? [ (S)kip / (Q)uit ]: ").lower().strip()

                if choice in ['a', 'approve', '']:
                    if proc_segs:
                        for seg in proc_segs:
                            seg["original_label"] = s["label"]
                            all_segments.append(seg)
                        print(f"  -> Approved. ({len(proc_segs)} segments)")
                    else:
                        print("  -> Cannot approve failed sample.")
                elif choice in ['q', 'quit', 'exit']:
                    print("Stopping review.")
                    if all_segments:
                         if input("Save currently approved samples? (y/n): ").lower() == 'y':
                             break
                         else:
                             return
                    else:
                        return
                else:
                    print("  -> Skipped.")

        if failed_samples:
            print(f"\n[Warning] {len(failed_samples)} samples failed preprocessing:")
            for name in failed_samples:
                print(f"  - {name}")

        if all_segments:
            print(f"\nReprocessing Complete. {len(all_segments)} valid segments collected.")
            self._finalize_dataset(all_segments)
            self.update_scaler_stats()
        else:
            print("No segments were approved. Dataset not updated.")

    def _finalize_dataset(self, segments: List[Dict]):
        """Splits train/val, calculates normalization stats, saves datasets."""
        print("\nFinalizing Datasets (Split & Stats)...")

        groups = {}
        for s in segments:
            source = s["name"].split("_seg")[0]
            if source not in groups: groups[source] = []
            groups[source].append(s)

        unique_samples = list(groups.keys())
        random.shuffle(unique_samples)

        split = int(len(unique_samples) * TRAIN_VAL_SPLIT)
        train_keys = unique_samples[:split]
        val_keys = unique_samples[split:]

        train_segs = [s for k in train_keys for s in groups[k]]
        val_segs = [s for k in val_keys for s in groups[k]]

        print(f"Train: {len(train_keys)} samples ({len(train_segs)} segs)")
        print(f"Val:   {len(val_keys)} samples ({len(val_segs)} segs)")

        if train_segs:
            all_sensor = np.vstack([s["sensor"] for s in train_segs])
            mean = np.mean(all_sensor, axis=0)
            std = np.std(all_sensor, axis=0)
            std[std == 0] = 1.0

            with h5py.File(SCALER_STATS_PATH, "w") as f:
                f.create_dataset("mean", data=mean)
                f.create_dataset("std", data=std)
            print(f"Stats saved to {SCALER_STATS_PATH}")

        def save_h5(path, segs):
            with h5py.File(path, "w") as f:
                for s in segs:
                    g = f.create_group(s["name"])
                    g.create_dataset("sensor_data", data=pad_sequence(s["sensor"], MAX_SEQUENCE_LENGTH))
                    g.create_dataset("gt_pos_data", data=pad_sequence(s["gt_pos"], MAX_SEQUENCE_LENGTH))
                    g.create_dataset("gt_vel_data", data=pad_sequence(s["gt_vel"], MAX_SEQUENCE_LENGTH, True))
                    # Save gravity GT from Apple Pencil pose if available
                    if s.get("gt_gravity_b") is not None:
                        g.create_dataset("gt_gravity_b_data", data=pad_sequence(s["gt_gravity_b"], MAX_SEQUENCE_LENGTH))
                    g.attrs["original_label"] = s["original_label"]
                    g.attrs["sequence_length"] = len(s["sensor"])

        save_h5(PROCESSED_DATASET_PATH, train_segs)
        save_h5(VALIDATION_DATASET_PATH, val_segs)
        print("Datasets saved.")


async def main():
    parser = argparse.ArgumentParser(
        description="Trajecto data acquisition and preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/acquire.py                      # Interactive acquisition mode
  python utils/acquire.py --reprocess          # Reprocess with interactive review
  python utils/acquire.py --reprocess --approve-all          # Auto-approve all valid samples
  python utils/acquire.py --reprocess --approve-all --no-viz # Batch mode (no plots)
"""
    )
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-process existing raw data from raw_acquired_data.h5")
    parser.add_argument("--approve-all", "-y", action="store_true",
                        help="Auto-approve all successfully processed samples (skip interactive prompts)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization plots (useful for batch processing)")
    args = parser.parse_args()

    manager = AcquisitionManager()

    if args.reprocess:
        manager.run_reprocess(approve_all=args.approve_all, no_viz=args.no_viz)
    else:
        if args.approve_all or args.no_viz:
            print("Warning: --approve-all and --no-viz only apply to --reprocess mode.")
        await manager.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
