"""This script preprocesses raw trajectory data into a clean, segmented, and
normalized dataset.

The preprocessing pipeline includes the following steps:
1.  **Data Loading:** Load raw sensor and ground truth data from an HDF5 file.
2.  **Bias Correction:** Calculate and remove gyroscope bias from a static
    initial segment.
3.  **Unit Conversion:** Convert accelerometer data to m/s^2.
4.  **Resampling:** Resample ground truth data to a target frequency.
5.  **Synchronization:** Align sensor and ground truth data using
    cross-correlation.
6.  **Segmentation:** Identify and segment the data into meaningful chunks based on
    force sensor readings.
7.  **Filtering:** Apply a low-pass filter to the sensor data.
8.  **Normalization:** Calculate and save global mean and standard deviation for
    sensor data.
9.  **Saving:** Save the preprocessed data to an HDF5 file.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, correlation_lags

# --- Parameters ---
DATA_DIR: str = "acquired_data"
TARGET_SAMPLING_RATE_HZ: float = 400.0
BIAS_SEARCH_DURATION_S: float = 1.5
STATIC_VARIANCE_THRESHOLD: float = 0.005
SEGMENTATION_THRESHOLD: int = 0
SEGMENTATION_MARGIN: int = 15

PIXEL_TO_METER: float = 0.0254 / 264  # iPad PPI

# Output paths
FINAL_DATASET_PATH: str = "data/dataset.h5"
SCALER_STATS_PATH: str = "data/scaler_stats.h5"
HDF5_RAW_DATA_PATH: str = os.path.join(DATA_DIR, "raw_acquired_data.h5")

# Filters & Physics
CUTOFF_FREQ_HZ: float = 20.0
FILTER_ORDER: int = 4
GRAVITY: float = 9.81
MAX_SEQUENCE_LENGTH: int = int(TARGET_SAMPLING_RATE_HZ * 14.0)


# --- Helper Functions ---
def find_initial_static_bias(
    sensor_data_dict: Dict[str, np.ndarray], fs: float
) -> Optional[Dict[str, float]]:
    """Finds the most stable static segment to calculate gyro bias.

    This function analyzes the initial segment of sensor data to find the most
    stable period, which is then used to calculate the gyroscope bias.

    Args:
        sensor_data_dict (Dict[str, np.ndarray]): Dictionary of raw sensor data.
            - Keys: 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
            - Values Shape: (N,)
            - Values Unit: g (Accel), rad/s (Gyro)
            - Values Frame: Body
        fs (float): The sampling frequency of the sensor data in Hz.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing:
            - 'gyro_x', 'gyro_y', 'gyro_z': Mean bias (rad/s)
            - 'accel_norm_static': Mean accelerometer norm (g)
            Returns None if 'accel_x' is not present.
    """
    if "accel_x" not in sensor_data_dict:
        print(
            "  [Warning] 'accel_x' not in sensor data dict. Bias"
            " calculation skipped."
        )
        return None

    df_temp = pd.DataFrame(sensor_data_dict)
    limit_idx = int(BIAS_SEARCH_DURATION_S * fs)
    df_head = df_temp.iloc[:limit_idx].copy()

    accel_norm = np.sqrt(
        df_head["accel_x"] ** 2
        + df_head["accel_y"] ** 2
        + df_head["accel_z"] ** 2
    )
    rolling_var = accel_norm.rolling(window=50, min_periods=50).var()

    stable_indices = np.where(rolling_var < STATIC_VARIANCE_THRESHOLD)[0]

    if len(stable_indices) > 0:
        best_end_idx = rolling_var.idxmin()
        if pd.isna(best_end_idx):
            best_end_idx = 50
        else:
            best_end_idx = int(best_end_idx)
        start_static = max(0, best_end_idx - 50)
        end_static = best_end_idx
        static_segment = df_head.iloc[start_static:end_static]
        return {
            "gyro_x": static_segment["gyro_x"].mean(),
            "gyro_y": static_segment["gyro_y"].mean(),
            "gyro_z": static_segment["gyro_z"].mean(),
            "accel_norm_static": accel_norm.iloc[
                start_static:end_static
            ].mean(),
        }

    print(
        "  [Warning] No stable static segment found. Using mean of first 1s for"
        " bias."
    )
    return {
        "gyro_x": df_head["gyro_x"].mean(),
        "gyro_y": df_head["gyro_y"].mean(),
        "gyro_z": df_head["gyro_z"].mean(),
        "accel_norm_static": accel_norm.mean(),
    }


def find_force_segments(
    df_gt: pd.DataFrame, threshold: int, margin: int
) -> List[Tuple[int, int]]:
    """Finds writing segments based on GT Force values from a DataFrame.

    This function identifies segments where the 'force' column exceeds a given
    threshold and adds a margin to the start and end of each segment.

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame.
            - Column 'force': Ground truth force data.
                - Shape: (N,)
                - Unit: Normalized/Arbitrary
                - Frame: N/A
        threshold (int): The force threshold to determine active segments.
        margin (int): The number of samples to add to the beginning and end of
            each segment.

    Returns:
        List[Tuple[int, int]]: A list of (start_idx, end_idx) tuples.
    """
    if "force" not in df_gt.columns:
        print(
            "  [Warning] No 'force' data in GT DataFrame. Using full length"
            " for segmentation."
        )
        return [(0, len(df_gt))]

    force_data = df_gt["force"].to_numpy()
    above_threshold = force_data > threshold

    if not np.any(above_threshold):
        print(
            "  [Warning] Force never exceeded threshold. Using full length for"
            " segmentation."
        )
        return [(0, len(force_data))]

    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(force_data) - 1)

    segments = [
        (max(0, s - margin), min(len(force_data), e + margin))
        for s, e in zip(starts, ends)
    ]
    return segments


def preprocess_gt_data(
    gt_data_dict: Dict[str, np.ndarray], target_fs: float
) -> pd.DataFrame:
    """Preprocesses the ground truth data by resampling and interpolating.

    This function takes a dictionary of ground truth data, converts it to a
    DataFrame, and then resamples it to a target frequency. It also handles
    the conversion of 'hoverDistance' to a 'z' coordinate.

    Args:
        gt_data_dict (Dict[str, np.ndarray]): Dictionary of raw ground truth data.
            - Keys: 'timestamp', 'x', 'y', 'hoverDistance' (optional), 'force' (optional)
            - Values Shape: (M,)
            - Values Unit: s (timestamp), points (x, y), unitless (force)
            - Frame: World (Screen)
        target_fs (float): The target sampling frequency in Hz.

    Returns:
        pd.DataFrame: A DataFrame containing resampled and interpolated GT data.
    """
    if "timestamp" not in gt_data_dict:
        print(
            "  [Warning] 'timestamp' not in GT data. Skipping GT"
            " preprocessing."
        )
        return pd.DataFrame(gt_data_dict)

    df = pd.DataFrame(gt_data_dict)
    if "hoverDistance" in df.columns:
        df = df.rename(columns={"hoverDistance": "zOffset"})
        df["z"] = 12.49 * df["zOffset"].pow(0.78)

    original_time = df["timestamp"].to_numpy()
    sort_idx = np.argsort(original_time)
    original_time = original_time[sort_idx]
    unique_time, unique_idx = np.unique(original_time, return_index=True)
    original_time = unique_time

    new_time = np.arange(
        original_time[0],
        original_time[-1] + (1.0 / target_fs) * 0.5,
        1.0 / target_fs,
    )
    upsampled_df = pd.DataFrame({"timestamp": new_time})

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != "timestamp":
            original_data = df[col].to_numpy()[sort_idx][unique_idx]
            upsampled_df[col] = np.interp(
                new_time, original_time, original_data
            )

    return upsampled_df


def butter_lowpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int
) -> np.ndarray:
    """Applies a Butterworth low-pass filter to the data.

    Args:
        data (np.ndarray): The input data to filter.
            - Shape: (N,)
            - Unit: Any
            - Frame: Any
        cutoff (float): The cutoff frequency of the filter in Hz.
        fs (float): The sampling frequency of the data in Hz.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered data.
            - Shape: (N,)
            - Unit: Same as input
            - Frame: Same as input
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def pad_sequence(data: np.ndarray, max_len: int) -> np.ndarray:
    """Pads a sequence to a maximum length.

    If the sequence is shorter than the maximum length, it is padded by
    replicating the last value.

    Args:
        data (np.ndarray): The input sequence to pad.
            - Shape: (Seq_Len, Features)
            - Unit: Any
            - Frame: Any
        max_len (int): The maximum length to pad to.

    Returns:
        np.ndarray: The padded sequence.
            - Shape: (max_len, Features)
            - Unit: Same as input
            - Frame: Same as input
    """
    seq_len = min(len(data), max_len)
    padded = np.zeros((max_len, data.shape[1]))
    padded[:seq_len, :] = data[:seq_len, :]

    if seq_len < max_len:
        last_val = data[seq_len - 1, :]
        padded[seq_len:, :] = last_val

    return padded


def load_h5_samples(h5_path: str) -> List[Dict[str, Any]]:
    """Loads samples from the HDF5 raw data file.

    This function reads from separate pen and gt datasets in the HDF5 file and
    returns a list of dictionaries, where each dictionary represents a sample.

    Args:
        h5_path: The path to the HDF5 raw data file.

    Returns:
        A list of dictionaries, where each dictionary
            contains the sample name, original label, sensor data, and ground
            truth data.
    """
    samples: List[Dict[str, Any]] = []
    if not os.path.exists(h5_path):
        print(f"Error: Raw HDF5 data file not found at {h5_path}")
        return samples

    with h5py.File(h5_path, "r") as hf:
        if "raw_data" in hf:
            for sample_name, sample_group in hf["raw_data"].items():
                if "pen_data" not in sample_group or "gt_data" not in sample_group:
                    print(
                        f"  [Warning] Skipping sample '{sample_name}' because it's"
                        " missing 'pen_data' or 'gt_data'."
                    )
                    continue

                # Load sensor data
                pen_data_array = sample_group["pen_data"][:]
                sensor_data_dict = {
                    name: pen_data_array[name]
                    for name in pen_data_array.dtype.names
                }

                # Load ground truth data
                gt_data_array = sample_group["gt_data"][:]
                gt_data_dict = {
                    name: gt_data_array[name]
                    for name in gt_data_array.dtype.names
                }

                # Extract label from attributes
                original_label = sample_group.attrs.get(
                    "original_label", "unknown_label"
                )

                samples.append(
                    {
                        "name": sample_name,
                        "original_label": original_label,
                        "sensor_data_dict": sensor_data_dict,
                        "gt_data_dict": gt_data_dict,
                    }
                )
    return samples


def main() -> None:
    """Main function to preprocess the data."""
    print("--- Unified Preprocessing: Unit Conv + Force Segmentation ---")
    os.makedirs(os.path.dirname(FINAL_DATASET_PATH), exist_ok=True)

    raw_samples = load_h5_samples(HDF5_RAW_DATA_PATH)
    if not raw_samples:
        print("No raw samples found in HDF5 file. Exiting.")
        return

    processed_samples_list: List[Dict[str, Any]] = []
    failed_samples: List[str] = []

    total_samples = len(raw_samples)

    for sample in raw_samples:
        try:
            processed_sample = process_sample(sample)
            if processed_sample:
                processed_samples_list.extend(processed_sample)
            else:
                failed_samples.append(sample["name"])
        except Exception as e:
            print(f"  Error processing {sample['name']}: {e}")
            traceback.print_exc()
            failed_samples.append(sample["name"])

    if not processed_samples_list:
        print("No samples processed successfully.")
    else:
        save_processed_data(processed_samples_list)

    # Enhanced Summary
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total Raw Samples : {total_samples}")
    print(
        f"Successfully Saved: {len(processed_samples_list)} segments (from"
        f" {total_samples - len(failed_samples)} samples)"
    )
    print(f"Failed Samples    : {len(failed_samples)}")

    if failed_samples:
        print("-" * 20)
        print("List of Failed Samples:")
        for name in failed_samples:
            print(f" - {name}")
    print("=" * 50)

    print(f"\n--- Completed. Dataset saved to {FINAL_DATASET_PATH} ---")


def process_sample(sample: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Processes a single raw sample.

    This function takes a single raw sample and performs the following steps:
    1.  Bias Correction, Unit Conversion, and GT Upsampling
    2.  Synchronization
    3.  Segmentation
    4.  Filtering
    5.  GT Final Processing

    Args:
        sample: A dictionary containing the raw sample data.

    Returns:
        A list of dictionaries, where each
            dictionary represents a processed segment.
    """
    sample_name = sample["name"]
    original_label = sample["original_label"]
    sensor_data_dict = sample["sensor_data_dict"]
    gt_data_dict = sample["gt_data_dict"]

    print(f"\nProcessing: {sample_name} (Label: {original_label})")

    # 1. Bias Correction, Unit Conversion, and GT Upsampling
    bias = find_initial_static_bias(sensor_data_dict, TARGET_SAMPLING_RATE_HZ)
    df_sensor_orig = pd.DataFrame(sensor_data_dict)
    df_gt_proc = preprocess_gt_data(gt_data_dict, TARGET_SAMPLING_RATE_HZ)

    if bias:
        for axis in ["x", "y", "z"]:
            df_sensor_orig[f"gyro_{axis}"] -= bias[f"gyro_{axis}"]
            df_sensor_orig[f"accel_{axis}"] *= GRAVITY
        static_acc_ref = bias["accel_norm_static"] * GRAVITY
    else:
        print("  [Warning] Bias calculation failed. Converting units anyway.")
        for axis in ["x", "y", "z"]:
            df_sensor_orig[f"accel_{axis}"] *= GRAVITY
        static_acc_ref = GRAVITY

    # 2. Synchronization
    acc_norm = np.sqrt(
        df_sensor_orig["accel_x"] ** 2
        + df_sensor_orig["accel_y"] ** 2
        + df_sensor_orig["accel_z"] ** 2
    )
    # Center the sensor signal by removing the gravity component
    sig_sensor = (acc_norm - static_acc_ref) / (acc_norm.std() + 1e-6)

    gt_sync = (
        df_gt_proc["force"]
        if "force" in df_gt_proc.columns
        else np.zeros(len(df_gt_proc))
    )
    # Center the ground truth signal by removing the mean
    sig_gt = (gt_sync - gt_sync.mean()) / (gt_sync.std() + 1e-6)

    # Limit synchronization window to the first 5 seconds to focus on the Tap event
    sync_limit = int(5.0 * TARGET_SAMPLING_RATE_HZ)
    sig_sensor_sync = sig_sensor.iloc[:sync_limit] if len(sig_sensor) > sync_limit else sig_sensor
    sig_gt_sync = sig_gt.iloc[:sync_limit] if len(sig_gt) > sync_limit else sig_gt

    correlation = correlate(sig_sensor_sync, sig_gt_sync, mode="full")
    lag = (
        correlation_lags(len(sig_sensor_sync), len(sig_gt_sync), mode="full")[
            np.argmax(correlation)
        ]
        if len(correlation) > 0
        else 0
    )

    if lag > 0:
        df_sensor_orig = df_sensor_orig.iloc[lag:].reset_index(drop=True)
    elif lag < 0:
        df_gt_proc = df_gt_proc.iloc[abs(lag) :].reset_index(drop=True)

    min_len = min(len(df_sensor_orig), len(df_gt_proc))
    df_sensor = df_sensor_orig.iloc[:min_len]
    df_gt = df_gt_proc.iloc[:min_len]

    # 3. Segmentation (Performed AFTER synchronization)
    # Strategy: Find Tap (Accel Peak) -> Wait (Gap) -> Write (GT Force)

    # A. Detect Tap Event (Max Acceleration) within first 2 seconds
    acc_norm_aligned = np.sqrt(
        df_sensor["accel_x"] ** 2
        + df_sensor["accel_y"] ** 2
        + df_sensor["accel_z"] ** 2
    )
    
    # Limit search to the first 2 seconds (or full length if shorter)
    search_limit = int(2.0 * TARGET_SAMPLING_RATE_HZ)
    search_region = acc_norm_aligned.iloc[:search_limit] if len(acc_norm_aligned) > search_limit else acc_norm_aligned
    
    tap_idx = int(np.argmax(search_region))

    # B. Detect Writing Segments (GT Force > 0)
    # SEGMENTATION_MARGIN=15 ensures we include ~15 static samples before writing.
    raw_segments = find_force_segments(
        df_gt, SEGMENTATION_THRESHOLD, SEGMENTATION_MARGIN
    )

    write_start_idx = None
    min_gap_samples = int(0.3 * TARGET_SAMPLING_RATE_HZ)  # Min 0.3s gap after Tap

    # Find the first segment that starts cleanly after the Tap
    for start, end in raw_segments:
        if start > (tap_idx + min_gap_samples):
            write_start_idx = start
            break

    if write_start_idx is None:
        print(f"  [Warning] Could not distinguish 'Write' from 'Tap' (Peak: {tap_idx}).")
        
        # Fallback 1: Use last segment if it starts after tap + 50 (existing)
        if raw_segments and raw_segments[-1][0] > tap_idx + 50:
            print("  [Info] Fallback: Using the last detected segment.")
            write_start_idx = raw_segments[-1][0]
        # Fallback 2: Force-start after tap if we have a valid tap
        elif tap_idx + min_gap_samples < min_len:
             print("  [Warning] Fallback: Forcing write start 0.3s after Tap.")
             write_start_idx = tap_idx + min_gap_samples
        else:
            print("  [Error] No valid writing segment found. Skipping sample.")
            return None

    # Safety: Ensure we don't include the Tap or its ringing
    # Even if valid_write_start is found, clamp it to be at least some distance from Tap
    safe_start = max(write_start_idx, tap_idx + 100)  # 100 samples = 0.25s buffer min

    max_end = raw_segments[-1][1]
    segments = [(safe_start, max_end)]

    print(
        f"  [Info] Tap: {tap_idx}, Write Start: {write_start_idx} -> Final Start: {safe_start} (Gap: {safe_start - tap_idx})"
    )

    processed_segments: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(segments):
        if end - start < 50:
            continue  # Skip very short segments

        df_sensor_segment = df_sensor.iloc[start:end].reset_index(drop=True)
        df_gt_segment = df_gt.iloc[start:end].reset_index(drop=True)
        segment_name = f"{sample_name}_seg{i}"
        print(
            f"  -> Processing Segment: {start} ~ {end} (Length: {len(df_sensor_segment)}) as {segment_name}"
        )

        # 4. Filtering
        # We ONLY filter FSR data. We do NOT filter Accel/Gyro data because:
        # 1. The Kalman Filter (ESKF/AEKF) assumes white noise. Low-pass filtering
        #    colors the noise, violating this assumption.
        # 2. The Q matrix is tuned based on the raw Allan Variance parameters
        #    (ARW, VRW). Altering the noise profile ruins this tuning.
        df_sensor_filt = df_sensor_segment.copy()
        for col in ["fsr"]:
            if col in df_sensor_filt.columns:
                df_sensor_filt[col] = butter_lowpass_filter(
                    df_sensor_filt[col],
                    CUTOFF_FREQ_HZ,
                    TARGET_SAMPLING_RATE_HZ,
                    FILTER_ORDER,
                )

        # 5. GT Final Processing
        if not all(c in df_gt_segment.columns for c in ["x", "y", "z"]):
            print(
                f"  [Warning] Missing position columns in GT data for {segment_name}. Skipping segment."
            )
            continue

        gt_pos = df_gt_segment[["x", "y", "z"]].to_numpy()
        gt_pos_final = np.zeros_like(gt_pos)
        gt_pos_final[:, 0] = gt_pos[:, 0] * PIXEL_TO_METER
        gt_pos_final[:, 1] = gt_pos[:, 1] * PIXEL_TO_METER
        gt_pos_final[:, 2] = np.maximum(gt_pos[:, 2] * 0.001, 1e-7)

        # Calculate GT Velocity from gt_pos_final
        gt_vel_final = np.gradient(
            gt_pos_final, (1.0 / TARGET_SAMPLING_RATE_HZ), axis=0
        )  # Use dt directly

        fsr_data = (
            df_sensor_filt[["fsr"]].to_numpy()
            if "fsr" in df_sensor_filt.columns
            else np.zeros((len(df_sensor_filt), 1))
        )
        sensor_data_final = np.hstack(
            [
                df_sensor_filt[["accel_x", "accel_y", "accel_z"]].to_numpy(),
                df_sensor_filt[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(),
                fsr_data,
            ]
        )

        if (
            np.isnan(sensor_data_final).any()
            or np.isnan(gt_pos_final).any()
            or np.isnan(gt_vel_final).any()
        ):
            print(f"🚨 SKIP: NaN detected in {segment_name}")
            continue

        processed_segments.append(
            {
                "name": segment_name,
                "sensor": sensor_data_final,
                "gt_pos": gt_pos_final,
                "gt_vel": gt_vel_final,
                "original_label": original_label,
            }
        )
    return processed_segments


def save_processed_data(processed_samples_list: List[Dict[str, Any]]) -> None:
    """Saves the processed data and scaler statistics.

    Args:
        processed_samples_list: A list of
            dictionaries, where each dictionary represents a processed
            segment.
    """
    print("\nCalculating Global Statistics...")
    all_sensor_data = np.vstack([s["sensor"] for s in processed_samples_list])
    if np.isnan(all_sensor_data).any():
        all_sensor_data = np.nan_to_num(all_sensor_data)
    sensor_mean, sensor_std = np.mean(all_sensor_data, axis=0), np.std(
        all_sensor_data, axis=0
    )
    sensor_std[sensor_std == 0] = 1.0

    with h5py.File(SCALER_STATS_PATH, "w") as f:
        f.create_dataset("mean", data=sensor_mean)
        f.create_dataset("std", data=sensor_std)
    print(f"  Global Scaler stats saved to {SCALER_STATS_PATH}")

    with h5py.File(FINAL_DATASET_PATH, "w") as hf_out:
        for sample in processed_samples_list:
            if len(sample["sensor"]) == 0:
                continue
            sensor_padded = pad_sequence(sample["sensor"], MAX_SEQUENCE_LENGTH)
            gt_pos_padded = pad_sequence(sample["gt_pos"], MAX_SEQUENCE_LENGTH)
            gt_vel_padded = pad_sequence(sample["gt_vel"], MAX_SEQUENCE_LENGTH)
            g = hf_out.create_group(sample["name"])
            g.create_dataset("sensor_data", data=sensor_padded)
            g.create_dataset("gt_pos_data", data=gt_pos_padded)
            g.create_dataset("gt_vel_data", data=gt_vel_padded)
            g.attrs["original_label"] = sample["original_label"]
            g.attrs["sequence_length"] = len(sample["sensor"])  # Store true length before padding
            print(f"  Saved {sample['name']} to dataset.")


if __name__ == "__main__":
    main()