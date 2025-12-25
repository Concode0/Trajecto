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
from scipy.signal import iirfilter, lfilter, correlate, correlation_lags

# Ensure we can import 'receive.py' from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from receive import TrajectoDriver
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
NUM_SESSIONS = 1
LABELS_PER_SESSION = 30
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
TARGET_SAMPLING_RATE_HZ = 50.0
GRAVITY = 9.81
CUTOFF_FREQ_HZ = 20.0
FILTER_ORDER = 4
SEGMENTATION_THRESHOLD = 0
SEGMENTATION_MARGIN = 15
PIXEL_TO_METER = 2.1277e-4
MAX_SEQUENCE_LENGTH = int(TARGET_SAMPLING_RATE_HZ * 35.0)
TRAIN_VAL_SPLIT = 0.9

# Parameters for synchronization and segmentation
SYNC_WINDOW_S = 5.0  # Window for correlation in estimate_time_alignment_two_taps
ROI_TAP_SEARCH_WINDOW_S = 5.0  # Initial search window for taps to define ROI in preprocess_single
ROI_MARGIN_S = 0.5  # Margin around taps to define writing ROI
MIN_SEGMENT_LENGTH_S = 1.0 # Minimum length for a valid segment (after margin)


# --- Helper Functions (Preprocessing) ---

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = iirfilter(order, normal_cutoff, btype='low', ftype='butter', analog=False)
    return lfilter(b, a, data)

def pad_sequence(data: np.ndarray, max_len: int, is_velocity: bool = False) -> np.ndarray:
    seq_len = min(len(data), max_len)
    padded = np.zeros((max_len, data.shape[1]))
    padded[:seq_len, :] = data[:seq_len, :]

    if not is_velocity and seq_len < max_len:
        padded[seq_len:, :] = data[seq_len - 1, :]
    return padded

def preprocess_gt_data(gt_data_dict: Dict[str, np.ndarray], target_fs: float) -> pd.DataFrame:
    df = pd.DataFrame(gt_data_dict)
    if "timestamp" not in df.columns:
        return df

    if "hoverDistance" in df.columns:
        df = df.rename(columns={"hoverDistance": "zOffset"})
        df["z"] = 12.49 * df["zOffset"].pow(0.78)

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
    # Simplified version of the one in preprocess.py
    n = min(len(sig_ref), len(sig_target))
    window = int(SYNC_WINDOW_S * fs)
    if n < 2 * window:
        return 0.0, 0.0, False

    # Start
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
        segments.append((max(0, s - margin), min(len(force), e + margin)))
    return segments

def check_and_fix_gt_jump(self, df: pd.DataFrame, threshold_pt: float = 0.5) -> pd.DataFrame:
    touch_starts = df.index[(df['isHovering'].shift(1) == 0) & (df['isHovering'] > 0)].tolist()

    if not touch_starts:
        return df

    corrected_df = df.copy()
    was_any_fixed = False

    for idx in touch_starts:
        if idx < 1: continue

        prev_avg = df.loc[idx-2:idx-1, ['x', 'y']].mean()
        curr_avg = df.loc[idx:idx+1, ['x', 'y']].mean()

        diff = curr_avg - prev_avg
        jump_dist = np.sqrt(diff['x']**2 + diff['y']**2)

        if jump_dist > threshold_pt:
            print(f"\n[Segment Jump at Index {idx}]")
            print(f"  - Jump: {jump_dist:.2f} pts")

            choice = input(f"  - Align hover segment ending at {idx}? (y/n): ").lower()
            if choice == 'y':
                last_touch_end = df[:idx].index[df['force'].shift(-1) > 0].tolist()
                hover_start = last_touch_end[-2] + 1 if len(last_touch_end) > 1 else 0

                corrected_df.loc[hover_start:idx-1, 'x'] += diff['x']
                corrected_df.loc[hover_start:idx-1, 'y'] += diff['y']
                was_any_fixed = True

    return corrected_df, was_any_fixed

# --- Acquisition & Processing Class ---

class AcquisitionManager:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)

        self.driver = None
        self.pen_buffer = []
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
        self.driver = TrajectoDriver(data_callback=self._on_data)
        if await self.driver.connect():
            print("Connected!")
            return True
        return False

    async def disconnect(self):
        if self.driver:
            await self.driver.disconnect()
            print("Disconnected.")

    def _on_data(self, data):
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

        print(f"\n--- ACQUISITION: {self.global_counter + 1:03d} ---")
        input("Press Enter to START recording...")
        await self.driver.start_data_collection()

        print("1. TAP (Start Sync)")
        os.system("afplay /System/Library/Sounds/Tink.aiff &")
        await asyncio.sleep(1.5)

        print("2. WAIT (Calib)...")
        for i in range(2, 0, -1):
            print(f"         {i}...")
            await asyncio.sleep(1.0)

        print("3. WRITE NOW! >>> {label}")
        os.system("afplay /System/Library/Sounds/Glass.aiff &")
        input("   Press Enter when FINISHED writing...")

        print("4. STOP PEN (Wait)")
        os.system("afplay /System/Library/Sounds/Purr.aiff &")
        await asyncio.sleep(1.0)

        print("5. TAP (End Sync)")
        os.system("afplay /System/Library/Sounds/Tink.aiff &")
        await asyncio.sleep(1.5)

        await self.driver.stop_data_collection()
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
        """
        Runs the preprocessing logic on a single in-memory sample.
        Returns: (List of Processed Segments, Debug Info Dict)
        """
        debug_info = {}

        # 1. Prepare Sensor Data
        if not pen_data:
            return None, {"error": "No pen data"}
        df_sensor = pd.DataFrame(pen_data)
        # Convert Accel to m/s^2 (assuming raw is in g)
        for axis in ["x", "y", "z"]:
            if f"accel_{axis}" in df_sensor.columns:
                 df_sensor[f"accel_{axis}"] *= GRAVITY

        # 2. Prepare GT Data
        df_gt_proc = preprocess_gt_data(df_gt_raw.to_dict(orient='list'), TARGET_SAMPLING_RATE_HZ)

        # 3. Synchronization (Two-Tap / Correlation)
        acc_norm = np.sqrt(df_sensor["accel_x"]**2 + df_sensor["accel_y"]**2 + df_sensor["accel_z"]**2)
        sig_sensor = (acc_norm - GRAVITY) / (acc_norm.std() + 1e-6)

        gt_force = df_gt_proc["force"] if "force" in df_gt_proc.columns else np.zeros(len(df_gt_proc))
        sig_gt = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)

        slope, intercept, success = estimate_time_alignment_two_taps(sig_sensor.to_numpy(), sig_gt.to_numpy(), TARGET_SAMPLING_RATE_HZ)

        debug_info["sync_success"] = success
        debug_info["slope"] = slope
        debug_info["intercept"] = intercept

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

        # Check point jumps in gt_data
        df_gt_aligned, was_corrected = self.check_and_fix_gt_jump(df_gt_aligned)
        debug_info["jump_corrected"] = was_corrected

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

        debug_info["segment"] = (final_start, final_end)

        # Extract Final Segment
        df_s_seg = df_sensor_aligned.iloc[final_start:final_end].reset_index(drop=True)
        df_g_seg = df_gt_aligned.iloc[final_start:final_end].reset_index(drop=True)

        # Filter FSR
        if "fsr" in df_s_seg.columns:
            df_s_seg["fsr"] = butter_lowpass_filter(df_s_seg["fsr"], CUTOFF_FREQ_HZ, TARGET_SAMPLING_RATE_HZ, FILTER_ORDER)

        # Format Output
        processed_segments = []

        gt_pos = df_g_seg[["x", "y", "z"]].to_numpy()
        gt_pos[:, 0] *= PIXEL_TO_METER
        gt_pos[:, 1] *= PIXEL_TO_METER
        gt_pos[:, 2] = np.maximum(gt_pos[:, 2] * 0.001, 1e-7) # mm -> m

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
        """Shows synchronization plot to the user."""
        if "sensor_aligned" not in debug_info:
            print("Cannot visualize: No aligned data.")
            return

        df_s = debug_info["sensor_aligned"]
        df_g = debug_info["gt_aligned"]
        seg_start, seg_end = debug_info.get("segment", (0, 0))

        acc_norm = np.sqrt(df_s["accel_x"]**2 + df_s["accel_y"]**2 + df_s["accel_z"]**2)
        gt_force = df_g["force"] if "force" in df_g.columns else np.zeros(len(df_g))

        # Normalize for plotting
        acc_vis = (acc_norm - acc_norm.mean()) / (acc_norm.std() + 1e-6)
        force_vis = (gt_force - gt_force.mean()) / (gt_force.std() + 1e-6)

        plt.figure(figsize=(12, 6))
        plt.plot(acc_vis, label="Sensor Accel (Norm)", alpha=0.7)
        plt.plot(force_vis, label="GT Force (Norm)", alpha=0.7)

        # Highlight Segment
        plt.axvspan(seg_start, seg_end, color='green', alpha=0.2, label="Selected Segment")

        plt.title(f"Sync Check: {label}\n(Close window to continue)")
        plt.legend()
        plt.grid(True)
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
        """Batch re-preprocessing of existing raw data."""
        print(f"Reprocessing raw data from {RAW_HDF5_PATH}...")

        if not os.path.exists(RAW_HDF5_PATH):
            print("Raw data file not found.")
            return

        with h5py.File(RAW_HDF5_PATH, "r") as f:
            if "raw_data" not in f:
                return
            samples = []
            for k in f["raw_data"]:
                grp = f["raw_data"][k]

                # Check for approval - Default to True for old data without attribute
                if not grp.attrs.get("user_approved", True):
                    # print(f"Skipping {k}: Not approved.")
                    continue

                samples.append({
                    "name": k,
                    "pen_data": grp["pen_data"][:],
                    "gt_data": grp["gt_data"][:],
                    "label": grp.attrs.get("original_label", "unknown")
                })

        # Sort by name
        samples.sort(key=lambda x: x["name"])
        print(f"Found {len(samples)} approved samples. Processing...")

        all_segments = []
        for s in samples:
            pen_list = pd.DataFrame(s["pen_data"]).to_dict('records')
            gt_df = pd.DataFrame(s["gt_data"])

            proc_segs, debug = self.preprocess_single(pen_list, gt_df, s["name"])

            if proc_segs:
                for seg in proc_segs:
                    seg["original_label"] = s["label"]
                    all_segments.append(seg)
                print(f"  Processed {s['name']}")
            else:
                print(f"  Failed {s['name']}: {debug.get('error')}")

        self._finalize_dataset(all_segments)
        self.update_scaler_stats()

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