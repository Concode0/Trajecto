"""Unified Data Acquisition & Preprocessing Tool for Trajecto.

This script integrates:
1.  Data Acquisition (from 'data_acquire.py'): Connecting to pen, collecting data.
2.  Preprocessing (from 'preprocess.py'): Synchronization, Segmentation, Filtering.
3.  Visualization: Interactive plot to verify data quality before saving.

Usage:
    python utils/acquire.py             # Run interactive acquisition
    python utils/acquire.py --reprocess # Re-preprocess all raw data & generate datasets
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
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from scipy.ndimage import gaussian_filter1d

# Ensure we can import 'receive.py' from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from receive import TrajectoDriver, RawImuPacket
except ImportError:
    print("Error: Could not import 'receive.py'. Ensure it is in the 'utils/' directory.")
    sys.exit(1)


# --- Constants & Configuration ---
DATA_DIR = "acquired_data"
RAW_HDF5_PATH = os.path.join(DATA_DIR, "raw_acquired_data.h5")
PROCESSED_DATASET_PATH = "data/dataset.h5"
VALIDATION_DATASET_PATH = "data/validation_dataset.h5"
SCALER_STATS_PATH = "data/scaler_stats.h5"

# Acquisition Config
NUM_SESSIONS = 3
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

# Preprocessing Config
TARGET_SAMPLING_RATE_HZ = 50.107
GRAVITY = 9.80665  # Standard gravity (m/s²) - CODATA 2018
CUTOFF_FREQ_HZ = 5.0  # Reduced from 20.0 for better noise reduction (see optimize_fsr_zero_phase.py)
FILTER_ORDER = 4
SEGMENTATION_THRESHOLD = 0.01  # GT iPad screen force (0-1 normalized): exclude true zeros, keep actual writing (was: 0)
SEGMENTATION_MARGIN = 30
PIXEL_TO_METER = 0.0254 / 132.0  # iPad Retina display: 264 PPI ( 132 )
MAX_SEQUENCE_LENGTH = int(TARGET_SAMPLING_RATE_HZ * 35.0)
TRAIN_VAL_SPLIT = 1.0

# Parameters for synchronization and segmentation
SYNC_WINDOW_S = 5.0  # Window for correlation in estimate_time_alignment_two_taps
ROI_TAP_SEARCH_WINDOW_S = 5.0  # Initial search window for taps to define ROI in preprocess_single
ROI_MARGIN_S = 0.5  # Margin around taps to define writing ROI
MIN_SEGMENT_LENGTH_S = 1.0 # Minimum length for a valid segment (after margin)
STATIC_BUFFER_S = 2 # Static buffer duration to include before the segment

# Digitizer Error Detection
DIGITIZER_JUMP_THRESHOLD_M = 0.010  # 10mm - Maximum plausible single-frame position jump
DIGITIZER_JUMP_MIN_VELOCITY = 0.5  # m/s - Minimum velocity threshold to trigger jump detection (ignore static regions)


# --- Helper Functions (Preprocessing) ---

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter using filtfilt.

    Unlike lfilter (causal), filtfilt processes the signal forward and backward,
    resulting in zero phase distortion - critical for preserving temporal alignment
    between FSR and velocity features in training data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def pad_sequence(data: np.ndarray, max_len: int, is_velocity: bool = False) -> np.ndarray:
    """Pads or truncates sequence to fixed length with appropriate padding strategy.

    Implements two padding strategies based on data type:
    - **Position/sensor data**: Edge padding (repeat last value) to maintain continuity
    - **Velocity data**: Zero padding to avoid artificial motion

    This function is used to normalize all sequences to MAX_SEQUENCE_LENGTH for
    batched training, as PyTorch requires fixed-size tensors.

    Args:
        data: Input sequence array of shape [seq_len, features]
        max_len: Target sequence length (usually MAX_SEQUENCE_LENGTH=1750)
        is_velocity: If True, use zero padding; if False, use edge padding.
            Default: False

    Returns:
        np.ndarray: Padded/truncated array of shape [max_len, features]

    Example:
        >>> position = np.random.randn(1000, 3)  # Short sequence
        >>> padded_pos = pad_sequence(position, 1750, is_velocity=False)
        >>> padded_pos.shape
        (1750, 3)
        >>> # Last 750 samples are copies of position[999]

        >>> velocity = np.random.randn(2000, 3)  # Long sequence
        >>> padded_vel = pad_sequence(velocity, 1750, is_velocity=True)
        >>> padded_vel.shape
        (1750, 3)
        >>> # Truncated to first 1750 samples
    """
    seq_len = min(len(data), max_len)
    padded = np.zeros((max_len, data.shape[1]))
    padded[:seq_len, :] = data[:seq_len, :]

    if not is_velocity and seq_len < max_len:
        # Edge padding: Repeat last value for position/sensor data
        padded[seq_len:, :] = data[seq_len - 1, :]
    # Velocity uses zero padding (default from np.zeros initialization)
    return padded

def detect_digitizer_jumps(pos_xyz: np.ndarray, force: np.ndarray, fs: float) -> Dict[str, Any]:
    """Detect unrealistic position jumps caused by digitizer errors.

    Args:
        pos_xyz: Position array [N, 3] in meters (x, y, z)
        force: Force array [N] normalized 0-1 (or raw FSR values)
        fs: Sampling frequency in Hz

    Returns:
        Dictionary containing:
            - has_jumps: bool
            - jump_indices: List of sample indices where jumps occur
            - jump_magnitudes: List of jump distances in meters
            - max_jump: Maximum jump distance in meters
            - num_jumps: Total number of jumps detected
    """
    if len(pos_xyz) < 2:
        return {
            "has_jumps": False,
            "jump_indices": [],
            "jump_magnitudes": [],
            "max_jump": 0.0,
            "num_jumps": 0
        }

    # Calculate frame-to-frame position differences
    pos_diff = np.diff(pos_xyz, axis=0)
    jump_distances = np.linalg.norm(pos_diff, axis=1)

    # Calculate instantaneous velocity (m/s)
    dt = 1.0 / fs
    velocities = jump_distances / dt

    # Detect jumps: position change exceeds threshold AND velocity is above minimum
    # (ignore static regions where small noise can look like relative jumps)
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
    """Preprocesses ground truth data from iPad: unit conversion, filtering, resampling.

    This function performs a complete preprocessing pipeline on raw iPad Apple Pencil data:

    **Step 1: Unit Conversion**
    - X, Y: pixels → meters (using PIXEL_TO_METER = 0.0254/132 for iPad Retina 264 PPI)
    - Z: Calculated from hoverDistance (mm) using power law: z = 12.49 * hover^0.78 / 1000

    **Step 2: Outlier Filtering**
    - Detects unrealistic position jumps (>100 pixels ≈ 9.6mm in single frame)
    - Removes outlier frames caused by digitizer glitches

    **Step 3: Gaussian Smoothing**
    - Applies Gaussian filter (sigma=1.0) to x, y, z, force
    - Reduces high-frequency noise from 240Hz iPad sampling

    **Step 4: Resampling**
    - Resamples to target_fs (50.107 Hz) using PCHIP interpolation
    - PCHIP (Piecewise Cubic Hermite) preserves monotonicity and avoids overshooting
    - Handles duplicate timestamps and sorts data temporally

    Args:
        gt_data_dict: Dictionary with keys:
            - 'timestamp': Unix timestamps in seconds (float)
            - 'x', 'y': Position in pixels (float)
            - 'hoverDistance': Pencil hover distance in mm (float)
            - 'force': Normalized pressure 0-1 (float)
            - Optional: 'azimuthAngle', 'altitudeAngle'
        target_fs: Target sampling frequency in Hz (typically TARGET_SAMPLING_RATE_HZ=50.107)

    Returns:
        pd.DataFrame: Preprocessed ground truth data with columns:
            - 'timestamp': Resampled timestamps at target_fs
            - 'x', 'y', 'z': Position in meters (float)
            - 'force': Smoothed force 0-1 (float)
            - Other numeric columns from input (resampled)

    Note:
        - All position outputs are in SI units (meters)
        - Z-axis accuracy is limited by Apple Pencil hover distance noise
        - Short sequences (<2 samples) are returned without resampling

    Example:
        >>> raw_gt = {
        ...     'timestamp': np.array([1.0, 1.004, 1.008]),
        ...     'x': np.array([512.0, 513.0, 514.0]),  # pixels
        ...     'y': np.array([768.0, 769.0, 770.0]),
        ...     'hoverDistance': np.array([5.0, 5.1, 5.2]),  # mm
        ...     'force': np.array([0.5, 0.6, 0.7])
        ... }
        >>> df = preprocess_gt_data(raw_gt, 50.107)
        >>> df['x'].values  # Now in meters
        array([0.098..., 0.099..., 0.100...])
    """
    df = pd.DataFrame(gt_data_dict)
    if "timestamp" not in df.columns:
        return df

    # STEP 1: Convert all units FIRST (pixels → meters, mm → meters)
    # Convert X, Y from pixels to meters
    df["x"] = df["x"] * PIXEL_TO_METER
    df["y"] = df["y"] * PIXEL_TO_METER

    # Calculate and convert Z from hoverDistance (mm → meters)
    if "hoverDistance" in df.columns:
        df = df.rename(columns={"hoverDistance": "zOffset"})
        df["z"] = 12.49 * df["zOffset"].pow(0.78) / 1000.0

    # STEP 2: Outlier filtering (now in meters)
    pos_diff = np.diff(df[["x", "y"]].to_numpy(), axis=0)
    dist = np.linalg.norm(pos_diff, axis=1)

    # Threshold in meters: 100 pixels * PIXEL_TO_METER ≈ 0.0096 m
    valid_mask = np.insert(dist < (100.0 * PIXEL_TO_METER), 0, True)
    df = df[valid_mask].reset_index(drop=True)

    # STEP 3: Gaussian smoothing (sigma is in sample units, not data units)
    for col in ["x", "y", "z", "force"]:
        if col in df.columns:
            df[col] = gaussian_filter1d(df[col].to_numpy(), sigma=1.0)

    # Sort and unique
    original_time = df["timestamp"].to_numpy()
    sort_idx = np.argsort(original_time)
    original_time = original_time[sort_idx]
    unique_time, unique_idx = np.unique(original_time, return_index=True)
    original_time = unique_time

    # Resample
    if len(original_time) < 2:
        return df # Too short to resample

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

    return upsampled_df

def estimate_time_alignment_two_taps(sig_ref: np.ndarray, sig_target: np.ndarray, fs: float) -> Tuple[float, float, bool]:
    """Estimates clock drift and offset using two-tap correlation synchronization.

    This is the core synchronization algorithm for aligning ESP32 IMU data with
    iPad ground truth. It exploits the "Tap-Wait-Write-Tap" acquisition protocol:

    **Protocol Overview:**
    1. Start Tap: Sharp acceleration spike at sequence beginning
    2. Writing: Actual handwriting motion
    3. End Tap: Sharp acceleration spike at sequence end

    **Algorithm:**
    1. Correlates start window (first SYNC_WINDOW_S seconds) to find lag_start
    2. Correlates end window (last SYNC_WINDOW_S seconds) to find lag_end
    3. Calculates linear clock drift:
       - slope = (distance_ratio - 1.0) where ratio accounts for lag differences
       - intercept = lag_start (initial time offset)

    **Clock Drift Model:**
    ```
    target_time[i] = source_time[i] * (1 + slope) + intercept
    ```
    - slope: Clock rate mismatch (dimensionless, typically ±0.001 = ±1000 ppm)
    - intercept: Initial time offset in samples

    Args:
        sig_ref: Reference signal (ESP32 acceleration norm - GRAVITY), normalized
        sig_target: Target signal (iPad force), normalized
        fs: Sampling frequency in Hz (must be same for both signals)

    Returns:
        Tuple[float, float, bool]:
            - slope: Clock drift coefficient (dimensionless)
            - intercept: Initial offset in samples
            - success: True if sync succeeded, False if sequence too short

    Example:
        >>> # ESP32 clock runs 0.1% faster than iPad
        >>> slope, intercept, ok = estimate_time_alignment_two_taps(
        ...     sig_sensor, sig_gt, 50.107
        ... )
        >>> print(f"Drift: {slope*1e6:.1f} ppm, Offset: {intercept:.1f} samples")
        Drift: +1000.0 ppm, Offset: -5.2 samples

    Note:
        - Requires sequences longer than 2*SYNC_WINDOW_S (default: 10s total)
        - Both signals must be normalized (zero mean, unit variance)
        - Accuracy degrades if taps are not sharp/distinct
        - Correlation assumes taps are the dominant signal features

    See Also:
        - preprocess_single(): Applies the computed alignment to data
        - SYNC_WINDOW_S: Correlation window size constant
    """
    # Simplified version of the one in preprocess.py
    n = min(len(sig_ref), len(sig_target))
    window = int(SYNC_WINDOW_S * fs)
    if n < 2 * window:
        return 0.0, 0.0, False

    # Start tap correlation
    corr_start = correlate(sig_ref[:window] - np.mean(sig_ref[:window]),
                           sig_target[:window] - np.mean(sig_target[:window]), mode='full')
    lag_start = correlation_lags(window, window, mode='full')[np.argmax(corr_start)]

    # End
    corr_end = correlate(sig_ref[-window:] - np.mean(sig_ref[-window:]),
                         sig_target[-window:] - np.mean(sig_target[-window:]), mode='full')
    lag_end = correlation_lags(window, window, mode='full')[np.argmax(corr_end)]

    # Distances
    dist_target = n - window
    #dist_ref = dist_target - lag_end + lag_start
    dist_ref = dist_target + lag_end - lag_start   # check which is correct...

    if dist_ref == 0: return 0.0, 0.0, False

    slope = (dist_ref / dist_target) - 1.0
    intercept = lag_start # Simplified

    return slope, intercept, True

def find_force_segments(df_gt: pd.DataFrame, threshold: int, margin: int) -> List[Tuple[int, int]]:
    """Detects active writing segments from iPad force signal using threshold detection.

    Identifies continuous regions where Apple Pencil force exceeds a threshold,
    indicating actual writing (as opposed to hover or static periods).

    **Algorithm:**
    1. Threshold force signal to create binary active mask
    2. Detect rising edges (start of writing) and falling edges (end of writing)
    3. Add margin samples around each segment to avoid truncating stroke boundaries
    4. Handle edge cases (active at start/end of sequence)

    Args:
        df_gt: Ground truth DataFrame containing 'force' column (0-1 normalized)
        threshold: Force threshold for active writing detection (typically SEGMENTATION_THRESHOLD=0.01)
        margin: Number of samples to add before/after each segment (typically SEGMENTATION_MARGIN=30)

    Returns:
        List[Tuple[int, int]]: List of (start_idx, end_idx) tuples for each writing segment.
            Returns [(0, len(df_gt))] if no force data or no active segments detected.

    Example:
        >>> df = pd.DataFrame({'force': [0, 0, 0.5, 0.6, 0.7, 0, 0]})
        >>> segments = find_force_segments(df, threshold=0.01, margin=1)
        >>> segments
        [(1, 6)]  # Segment from index 1 to 6 (includes margin)

    Note:
        - Margin prevents cutting off stroke starts/ends due to force ramping
        - Multiple segments may be returned for discontinuous writing
        - Segment boundaries are clipped to [0, len(force)]

    See Also:
        - SEGMENTATION_THRESHOLD: Global force threshold constant
        - SEGMENTATION_MARGIN: Global margin constant
    """
    if "force" not in df_gt.columns: return [(0, len(df_gt))]
    force = df_gt["force"].to_numpy()
    active = force > threshold
    if not np.any(active): return [(0, len(force))]

    # Detect segment boundaries using diff on binary mask
    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0] + 1  # Rising edges (+1 to get first active sample)
    ends = np.where(diff == -1)[0]       # Falling edges (last active sample)

    # Handle edge cases
    if active[0]: starts = np.insert(starts, 0, 0)
    if active[-1]: ends = np.append(ends, len(force) - 1)

    # Apply margin and clip to valid range
    segments = []
    for s, e in zip(starts, ends):
        segments.append((max(0, s - margin), min(len(force), e + margin + 1)))
    return segments

# --- Acquisition & Processing Class ---

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
        """Initialize or check HDF5 file."""
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
        """Callback for raw IMU packets from BLE

        Stores data in SI units (m/s², rad/s) as received from firmware.
        No conversion needed - units are already correct for preprocessing.
        """
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
        """Runs the Tap-Wait-Write-Tap sequence."""
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
        """Loads the expected iPad CSV."""
        fname = f"Sample_{index}.csv"
        path = os.path.join(self.data_dir, fname)
        # print(f"Looking for {fname}...")

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
        """Preprocesses a single raw acquisition: synchronization, segmentation, filtering.

        This is the core preprocessing pipeline that transforms raw sensor + iPad data
        into training-ready samples. The pipeline consists of:

        **Step 1: Data Preparation**
        - Converts pen_data list to DataFrame (already in SI units: m/s², rad/s)
        - Preprocesses ground truth via preprocess_gt_data() (unit conversion, resampling)

        **Step 2: Two-Tap Synchronization**
        - Normalizes acceleration magnitude and iPad force signals
        - Runs estimate_time_alignment_two_taps() to get drift parameters
        - Applies linear time warping: target_idx = source_idx * (1 + slope) + intercept
        - Trims NaN values from interpolation boundaries
        - Prints detailed time-lag compensation report

        **Step 3: Digitizer Error Detection**
        - Detects unrealistic position jumps in ground truth (>10mm single frame)
        - Prints warning if jumps detected (possible iPad digitizer glitches)
        - See detect_digitizer_jumps() for details

        **Step 4: Segmentation (ROI Extraction)**
        - Finds start/end taps in force signal to define Region of Interest
        - Applies ROI_MARGIN_S to exclude tap artifacts
        - Uses find_force_segments() to detect writing regions
        - Adds STATIC_BUFFER_S (2s) before segment for ESKF initialization

        **Step 5: FSR Filtering**
        - Applies zero-phase Butterworth lowpass filter to FSR signal
        - Cutoff: CUTOFF_FREQ_HZ (5 Hz), Order: 4

        **Step 6: Output Formatting**
        - Stacks sensor channels: [accel_xyz(3), gyro_xyz(3), fsr(1)] → (N, 7)
        - Calculates ground truth velocity via np.gradient()
        - Ensures Z position has minimum value (1e-7) to avoid numerical issues
        - Returns segments with metadata

        Args:
            pen_data: List of IMU data dicts from BLE, keys:
                - 'time': timestamp in seconds
                - 'accel_x/y/z': acceleration in m/s²
                - 'gyro_x/y/z': angular velocity in rad/s
                - 'fsr': force sensor raw ADC value
            df_gt_raw: Raw iPad DataFrame from CSV with columns:
                - 'timestamp': Unix time in seconds
                - 'x', 'y': position in pixels
                - 'hoverDistance': hover in mm
                - 'force': pressure 0-1
            sample_name: Identifier for debug/visualization (e.g., "sample_042")

        Returns:
            Tuple[Optional[List[Dict]], Dict]:
                - segments: List of processed segment dicts (or None on failure):
                    - 'name': Segment identifier (e.g., "sample_042_seg0")
                    - 'sensor': Sensor data [N, 7] (accel, gyro, fsr) in SI units
                    - 'gt_pos': Ground truth position [N, 3] in meters
                    - 'gt_vel': Ground truth velocity [N, 3] in m/s
                - debug_info: Dictionary for visualization:
                    - 'sync_success': bool
                    - 'slope', 'intercept': Sync parameters
                    - 'sensor_aligned', 'gt_aligned': Aligned DataFrames
                    - 'segment': (start_idx, end_idx) tuple
                    - 'digitizer_jumps': Jump detection results
                    - 'error': Error message (if failed)

        Example:
            >>> manager = AcquisitionManager()
            >>> pen_data = [...]  # From BLE acquisition
            >>> df_gt = pd.read_csv("Sample_1.csv")
            >>> segments, debug = manager.preprocess_single(pen_data, df_gt, "sample_001")
            >>> if segments:
            ...     print(f"Success! {len(segments)} segments")
            ...     print(f"Sensor shape: {segments[0]['sensor'].shape}")

        Note:
            - Prints verbose synchronization diagnostics to console
            - All units are SI (meters, m/s, m/s², rad/s)
            - Digitizer jumps are detected but NOT corrected (manual review required)
            - Z-axis uses power-law hover distance estimate (less accurate than X/Y)

        See Also:
            - estimate_time_alignment_two_taps(): Synchronization algorithm
            - detect_digitizer_jumps(): Jump detection
            - find_force_segments(): Segmentation
            - visualize_sync(): Debug visualization
        """
        debug_info = {}

        # 1. Prepare Sensor Data
        if not pen_data:
            return None, {"error": "No pen data"}
        df_sensor = pd.DataFrame(pen_data)
        # Data is already in SI units (m/s², rad/s) from firmware via BLE
        # No conversion needed - ready for processing

        # 2. Prepare GT Data
        df_gt_proc = preprocess_gt_data(df_gt_raw.to_dict(orient='list'), TARGET_SAMPLING_RATE_HZ)

        # 3. Synchronization (Two-Tap / Correlation)
        # Calculate acceleration magnitude (in m/s²)
        # Subtract GRAVITY (9.80665 m/s²) to remove DC offset and detect taps
        acc_norm = np.sqrt(df_sensor["accel_x"]**2 + df_sensor["accel_y"]**2 + df_sensor["accel_z"]**2)
        sig_sensor = (acc_norm - GRAVITY) / (acc_norm.std() + 1e-6)

        gt_force = df_gt_proc["force"] if "force" in df_gt_proc.columns else np.zeros(len(df_gt_proc))
        sig_gt = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)

        slope, intercept, success = estimate_time_alignment_two_taps(sig_sensor.to_numpy(), sig_gt.to_numpy(), TARGET_SAMPLING_RATE_HZ)

        debug_info["sync_success"] = success
        debug_info["slope"] = slope
        debug_info["intercept"] = intercept

        # === TIME-LAG COMPENSATION REPORT ===
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
            # Apply Drift Correction
            target_indices = np.arange(len(df_gt_proc))
            #source_indices = target_indices * (1.0 - slope) + intercept
            source_indices = target_indices * (1.0 + slope) + intercept     # check which is correct?

            new_sensor = {}
            for col in df_sensor.columns:
                new_sensor[col] = np.interp(source_indices, np.arange(len(df_sensor)), df_sensor[col], left=np.nan, right=np.nan)
            df_sensor_aligned = pd.DataFrame(new_sensor)

            # Trim
            valid = ~df_sensor_aligned.isna().any(axis=1)
            if valid.sum() == 0:
                return None, {"error": "Sync resulted in empty data"}

            first, last = valid.idxmax(), valid[::-1].idxmax()
            df_sensor_aligned = df_sensor_aligned.iloc[first:last+1].reset_index(drop=True)
            df_gt_aligned = df_gt_proc.iloc[first:last+1].reset_index(drop=True)
        else:
            # Fallback Correlation
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

        # Save for visualization
        debug_info["sensor_aligned"] = df_sensor_aligned
        debug_info["gt_aligned"] = df_gt_aligned

        # === DIGITIZER ERROR DETECTION ===
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

        # 4. Segmentation
        # Find start/end taps in GT Force to define ROI
        gt_f = df_gt_aligned["force"].to_numpy() if "force" in df_gt_aligned.columns else np.zeros(len(df_gt_aligned))
        roi_start, roi_end = 0, len(gt_f)

        # Simple heuristic: Taps are high peaks at start/end
        window = int(ROI_TAP_SEARCH_WINDOW_S * TARGET_SAMPLING_RATE_HZ)
        if len(gt_f) > 2 * window:
             start_tap = np.argmax(gt_f[:window])
             end_tap = len(gt_f) - window + np.argmax(gt_f[-window:])
             roi_start = start_tap + int(ROI_MARGIN_S * TARGET_SAMPLING_RATE_HZ)
             roi_end = end_tap - int(ROI_MARGIN_S * TARGET_SAMPLING_RATE_HZ)

        segments = []
        raw_segs = find_force_segments(df_gt_aligned, SEGMENTATION_THRESHOLD, SEGMENTATION_MARGIN)

        for s, e in raw_segs:
            s_ = max(s, roi_start)
            e_ = min(e, roi_end)
            if e_ - s_ > int(MIN_SEGMENT_LENGTH_S * TARGET_SAMPLING_RATE_HZ):
                segments.append((s_, e_))

        if not segments:
             # Fallback: just use the whole ROI if valid
             if roi_end - roi_start > int(MIN_SEGMENT_LENGTH_S * TARGET_SAMPLING_RATE_HZ):
                 segments.append((roi_start, roi_end))
             else:
                 return None, {"error": "No valid segments found"}

        # Merge segments into one block for Trajecto usually
        final_start, final_end = segments[0][0], segments[-1][1]

        static_buffer_samples = int(STATIC_BUFFER_S * TARGET_SAMPLING_RATE_HZ)
        final_start = max(0, final_start - static_buffer_samples)

        debug_info["segment"] = (final_start, final_end)

        # Extract Final Segment
        df_s_seg = df_sensor_aligned.iloc[final_start:final_end].reset_index(drop=True)
        df_g_seg = df_gt_aligned.iloc[final_start:final_end].reset_index(drop=True)

        # Filter FSR
        if "fsr" in df_s_seg.columns:
            df_s_seg["fsr"] = butter_lowpass_filter(df_s_seg["fsr"], CUTOFF_FREQ_HZ, TARGET_SAMPLING_RATE_HZ, FILTER_ORDER)

        # Format Output
        processed_segments = []

        # All coordinates are already in meters from preprocess_gt_data
        gt_pos = df_g_seg[["x", "y", "z"]].to_numpy()
        # Ensure Z has minimum value to avoid numerical issues
        gt_pos[:, 2] = np.maximum(gt_pos[:, 2], 1e-7)

        gt_vel = np.gradient(gt_pos, 1.0/TARGET_SAMPLING_RATE_HZ, axis=0)

        fsr = df_s_seg["fsr"].to_numpy().reshape(-1, 1) if "fsr" in df_s_seg.columns else np.zeros((len(df_s_seg), 1))
        sensor_final = np.hstack([
            df_s_seg[["accel_x", "accel_y", "accel_z"]].to_numpy(),
            df_s_seg[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(),
            fsr
        ])

        processed_segments.append({
            "name": f"{sample_name}_seg0",
            "sensor": sensor_final,
            "gt_pos": gt_pos,
            "gt_vel": gt_vel
        })

        return processed_segments, debug_info

    def visualize_sync(self, debug_info: Dict, label: str):
        """Shows comprehensive data integrity visualization."""
        if "sensor_aligned" not in debug_info:
            print("Cannot visualize: No aligned data.")
            return

        df_s = debug_info["sensor_aligned"]
        df_g = debug_info["gt_aligned"]
        seg_start, seg_end = debug_info.get("segment", (0, 0))
        jump_info = debug_info.get("digitizer_jumps", {})

        # Create time axis in seconds
        time_s = np.arange(len(df_s)) / TARGET_SAMPLING_RATE_HZ

        # Prepare data
        acc_norm = np.sqrt(df_s["accel_x"]**2 + df_s["accel_y"]**2 + df_s["accel_z"]**2)
        gt_force = df_g["force"] if "force" in df_g.columns else np.zeros(len(df_g))
        fsr = df_s["fsr"].to_numpy() if "fsr" in df_s.columns else np.zeros(len(df_s))

        # GT position (already in meters)
        gt_x = df_g["x"].to_numpy() if "x" in df_g.columns else np.zeros(len(df_g))
        gt_y = df_g["y"].to_numpy() if "y" in df_g.columns else np.zeros(len(df_g))
        gt_z = df_g["z"].to_numpy() if "z" in df_g.columns else np.zeros(len(df_g))

        # Gyro data
        gyro_norm = np.sqrt(df_s["gyro_x"]**2 + df_s["gyro_y"]**2 + df_s["gyro_z"]**2)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. Sync Check (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        acc_vis = (acc_norm - acc_norm.mean()) / (acc_norm.std() + 1e-6)
        force_vis = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)
        ax1.plot(time_s, acc_vis, label="Sensor Accel (Norm)", alpha=0.7, linewidth=1)
        ax1.plot(time_s, force_vis, label="GT Force (Norm)", alpha=0.7, linewidth=1)
        ax1.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2, label="Selected Segment")

        # Mark digitizer jumps
        if jump_info.get("has_jumps"):
            for idx in jump_info["jump_indices"]:
                if idx < len(time_s):
                    ax1.axvline(time_s[idx], color='red', linestyle='--', alpha=0.6, linewidth=2)

        ax1.set_title("Synchronization Check")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normalized Signal")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. GT Position XY (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(gt_x * 1000, gt_y * 1000, 'b-', linewidth=1.5, alpha=0.7)
        ax2.plot(gt_x[0] * 1000, gt_y[0] * 1000, 'go', markersize=8, label='Start')
        ax2.plot(gt_x[-1] * 1000, gt_y[-1] * 1000, 'ro', markersize=8, label='End')

        # Mark digitizer jump locations
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

        # 3. GT Position Time Series (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_s, gt_x * 1000, label='X', alpha=0.7, linewidth=1)
        ax3.plot(time_s, gt_y * 1000, label='Y', alpha=0.7, linewidth=1)
        ax3.plot(time_s, gt_z * 1000, label='Z', alpha=0.7, linewidth=1)
        ax3.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)

        # Mark digitizer jumps
        if jump_info.get("has_jumps"):
            for idx in jump_info["jump_indices"]:
                if idx < len(time_s):
                    ax3.axvline(time_s[idx], color='red', linestyle='--', alpha=0.6, linewidth=2)

        ax3.set_title("GT Position vs Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position (mm)")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. FSR Signal (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_s, fsr, 'purple', alpha=0.7, linewidth=1)
        ax4.axvspan(seg_start/TARGET_SAMPLING_RATE_HZ, seg_end/TARGET_SAMPLING_RATE_HZ,
                    color='green', alpha=0.2)
        ax4.set_title("FSR Signal (Sensor)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("FSR (ADC)")
        ax4.grid(True, alpha=0.3)

        # 5. Acceleration Components (bottom left)
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

        # 6. Gyro Components (bottom right)
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

        # 7. Data Integrity Stats (bottom span)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')

        # Calculate statistics
        duration = len(df_s) / TARGET_SAMPLING_RATE_HZ
        seg_duration = (seg_end - seg_start) / TARGET_SAMPLING_RATE_HZ
        path_length_2d = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)) * 1000  # mm
        path_length_3d = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2 + np.diff(gt_z)**2)) * 1000  # mm

        # GT velocity
        gt_vel = np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2 + np.diff(gt_z)**2) * TARGET_SAMPLING_RATE_HZ
        avg_velocity = np.mean(gt_vel) * 1000  # mm/s
        max_velocity = np.max(gt_vel) * 1000  # mm/s

        # Acceleration stats
        acc_mean = np.mean(acc_norm)
        acc_std = np.std(acc_norm)

        # Build status text with digitizer warning
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
        """Saves raw and processed data (Interactive Mode)."""
        # 1. Save RAW
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

        # 2. Save PROCESSED (Append to dataset, no split/stats for simplicity in interactive)
        with h5py.File(PROCESSED_DATASET_PATH, "a") as f:
            for i, seg in enumerate(processed_segs):
                seg_name = f"{name}_seg{i}"
                if seg_name in f:
                    del f[seg_name]
                g = f.create_group(seg_name)

                g.create_dataset("sensor_data", data=pad_sequence(seg["sensor"], MAX_SEQUENCE_LENGTH))
                g.create_dataset("gt_pos_data", data=pad_sequence(seg["gt_pos"], MAX_SEQUENCE_LENGTH))
                g.create_dataset("gt_vel_data", data=pad_sequence(seg["gt_vel"], MAX_SEQUENCE_LENGTH))
                g.attrs["original_label"] = label
                g.attrs["sequence_length"] = len(seg["sensor"])

        # Update Counters
        self.global_counter += 1
        print(f"[Saved] {name} (Raw + {len(processed_segs)} Segments)")

    def update_scaler_stats(self):
        """Updates scaler statistics using all data in the processed dataset (Training set)."""
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
                        # Truncate padding (pad_sequence uses edge padding)
                        real_data = data[:seq_len]
                        all_sensor_data.append(real_data)

            if not all_sensor_data:
                print("  No valid sensor data found to calculate stats.")
                return

            all_sensor_stacked = np.vstack(all_sensor_data)
            mean = np.mean(all_sensor_stacked, axis=0)
            std = np.std(all_sensor_stacked, axis=0)
            std[std == 0] = 1.0 # Prevent division by zero

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

                # --- 1. ACQUISITION PHASE ---
                for i, label in enumerate(labels):
                    print(f"\n>>> Task ({i+1}/{len(labels)}): {label}")

                    while True: # Retry loop for acquisition
                        if await self.acquire_sequence(label):
                            # Quick Check
                            num_samples = len(self.pen_buffer)
                            duration_s = num_samples / 50.0 # Approx 50Hz
                            print(f"    -> Captured {num_samples} samples (~{duration_s:.1f}s)")

                            # Buffer Data
                            session_buffer.append({
                                "label": label,
                                "pen_data": list(self.pen_buffer),
                                "ipad_idx": self.ipad_counter
                            })

                            # Advance iPad counter for next file
                            self.ipad_counter += 1
                            break
                        else:
                            # Connection failed or user aborted in driver?
                            print("    [!] Acquisition Failed.")
                            if input("    Retry this label? (y/n): ").lower() != 'y':
                                break

                # --- 2. PROCESSING PHASE ---
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

                        # Preprocess & Visualize
                        # Predict next global ID for visualization purposes (actual ID committed on save)
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
                            # Could ask to save Raw only? For now just skip.

                    if any_saved:
                        self.update_scaler_stats()
                else:
                    print("No samples collected in this session.")

        finally:
            await self.disconnect()

    def run_reprocess(self):
        """Interactive re-preprocessing of existing raw data."""
        print(f"Reprocessing raw data from {RAW_HDF5_PATH}...")

        if not os.path.exists(RAW_HDF5_PATH):
            print("Raw data file not found.")
            return

        # 1. Load all raw samples
        samples = []
        with h5py.File(RAW_HDF5_PATH, "r") as f:
            if "raw_data" not in f:
                print("No 'raw_data' group found.")
                return

            for k in f["raw_data"]:
                grp = f["raw_data"][k]
                # Load data into memory
                samples.append({
                    "name": k,
                    "pen_data": pd.DataFrame(grp["pen_data"][:]).to_dict('records'),
                    "gt_data": pd.DataFrame(grp["gt_data"][:]),
                    "label": grp.attrs.get("original_label", "unknown")
                })

        # Sort by name
        samples.sort(key=lambda x: x["name"])
        print(f"Found {len(samples)} samples. Starting interactive review...")

        all_segments = []

        for i, s in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Reviewing: {s['name']} (Label: {s['label']})")

            # Run Preprocessing
            proc_segs, debug = self.preprocess_single(s["pen_data"], s["gt_data"], s["name"])

            if proc_segs:
                self.visualize_sync(debug, s['label'])
                choice = input(f"  Action for {s['name']}? [ (A)pprove / (S)kip / (Q)uit ]: ").lower().strip()
            else:
                print(f"  [Error] Preprocessing failed: {debug.get('error')}")
                if "sensor_aligned" in debug:
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

        # Finalize
        if all_segments:
            print(f"\nReprocessing Complete. {len(all_segments)} valid segments collected.")
            self._finalize_dataset(all_segments)
            self.update_scaler_stats()
        else:
            print("No segments were approved. Dataset not updated.")

    def _finalize_dataset(self, segments: List[Dict]):
        """Splits data, calcs stats, and saves final datasets."""
        print("\nFinalizing Datasets (Split & Stats)...")

        # Group by source sample
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

        # Calc Stats
        if train_segs:
            all_sensor = np.vstack([s["sensor"] for s in train_segs])
            mean = np.mean(all_sensor, axis=0)
            std = np.std(all_sensor, axis=0)
            std[std == 0] = 1.0

            with h5py.File(SCALER_STATS_PATH, "w") as f:
                f.create_dataset("mean", data=mean)
                f.create_dataset("std", data=std)
            print(f"Stats saved to {SCALER_STATS_PATH}")

        # Save Files
        def save_h5(path, segs):
            with h5py.File(path, "w") as f:
                for s in segs:
                    g = f.create_group(s["name"])
                    g.create_dataset("sensor_data", data=pad_sequence(s["sensor"], MAX_SEQUENCE_LENGTH))
                    g.create_dataset("gt_pos_data", data=pad_sequence(s["gt_pos"], MAX_SEQUENCE_LENGTH))
                    g.create_dataset("gt_vel_data", data=pad_sequence(s["gt_vel"], MAX_SEQUENCE_LENGTH, True))
                    g.attrs["original_label"] = s["original_label"]
                    g.attrs["sequence_length"] = len(s["sensor"])

        save_h5(PROCESSED_DATASET_PATH, train_segs)
        save_h5(VALIDATION_DATASET_PATH, val_segs)
        print("Datasets saved.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reprocess", action="store_true", help="Re-process existing raw data")
    args = parser.parse_args()

    manager = AcquisitionManager()

    if args.reprocess:
        manager.run_reprocess()
    else:
        await manager.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
