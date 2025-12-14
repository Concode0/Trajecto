"""
This script performs an Allan Variance analysis on a stationary IMU data recording
to characterize sensor noise and stability.

It reads raw IMU data from a CSV file, calculates the Allan Deviation for both
accelerometer and gyroscope data on all three axes, and generates corresponding
log-log plots.

Finally, it computes and prints the key noise parameters required for tuning the
process noise covariance matrix (Q) of a Kalman filter:
- Angle Random Walk (ARW) for the gyroscope.
- Velocity Random Walk (VRW) for the accelerometer.
- Bias Instability for both sensors.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools

# Standard gravity constant for unit conversion
GRAVITY = 9.81

def analyze_sensor(data: np.ndarray, rate: float, sensor_name: str, unit: str) -> None:
    """
    Performs and plots Allan Variance analysis for a single sensor axis.

    Args:
        data: 1D array of sensor data.
        rate: The sampling rate of the sensor in Hz.
        sensor_name: The name of the sensor for plot titles (e.g., "Gyro X").
        unit: The unit of the sensor data for plot labels.
    """
    # --- Perform Allan Variance Calculation ---
    # Compute taus, allan deviation, and confidence intervals
    (taus, adev, adeverr, ns) = allantools.oadev(data, rate=rate, data_type="freq")

    # --- Plot the Allan Deviation Curve ---
    plt.figure(figsize=(10, 6))
    plt.loglog(taus, adev, label='Allan Deviation')
    plt.title(f'Allan Deviation for {sensor_name}')
    plt.xlabel('Averaging Time τ (s)')
    plt.ylabel(f'Allan Deviation ({unit})')
    plt.grid(True, which="both", ls="-")

    # --- Calculate and Print Noise Parameters ---
    print(f"\n--- {sensor_name} Noise Analysis ---")

    # --- Angle/Velocity Random Walk ---
    # Find the index where tau is closest to 1
    tau1_index = np.argmin(np.abs(taus - 1.0))
    random_walk = adev[tau1_index]

    # --- Bias Instability ---
    # Find the minimum point of the curve
    min_adev_index = np.argmin(adev)
    bias_instability = adev[min_adev_index]
    tau_bias = taus[min_adev_index]

    if 'Gyro' in sensor_name:
        print(f"Angle Random Walk (ARW): {random_walk:.4e} {unit}√s")
        print(f"Bias Instability: {bias_instability:.4e} {unit}")

        # Overlay parameters on the plot
        plt.loglog(1, random_walk, 'o', label=f'Angle Random Walk (τ=1)')
        plt.loglog(tau_bias, bias_instability, 'o', label=f'Bias Instability')

    elif 'Accel' in sensor_name:
        print(f"Velocity Random Walk (VRW): {random_walk:.4e} {unit}√s")
        print(f"Bias Instability: {bias_instability:.4e} {unit}")

        # Overlay parameters on the plot
        plt.loglog(1, random_walk, 'o', label=f'Velocity Random Walk (τ=1)')
        plt.loglog(tau_bias, bias_instability, 'o', label=f'Bias Instability')

    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Allan Variance Analysis Tool for IMU Data.")
    parser.add_argument("filepath", type=str, help="Path to the CSV file containing stationary IMU data.")
    parser.add_argument("--rate", type=float, required=True, help="Sampling rate of the IMU in Hz.")
    parser.add_argument("--slice-start", type=float, default=0, help="Number of seconds to remove from the beginning of the recording.")
    parser.add_argument("--slice-end", type=float, default=0, help="Number of seconds to remove from the end of the recording.")

    args = parser.parse_args()

    # --- Load Data from CSV File ---
    try:
        df = pd.read_csv(args.filepath)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Ensure required columns exist
        expected_cols = ['accel_x_g', 'accel_y_g', 'accel_z_g', 'gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']
        if not all(col in df.columns for col in expected_cols):
            print(f"Error: CSV file must contain the following columns: {expected_cols}")
            return

        # --- Slice Data ---
        num_samples_start = int(args.slice_start * args.rate)
        num_samples_end = int(args.slice_end * args.rate)

        if num_samples_start > 0:
            print(f"Slicing {num_samples_start} samples from the beginning ({args.slice_start} seconds).")
            df = df.iloc[num_samples_start:]

        if num_samples_end > 0:
            print(f"Slicing {num_samples_end} samples from the end ({args.slice_end} seconds).")
            df = df.iloc[:-num_samples_end]

        # Extract data into numpy arrays
        accel_data_g = df[['accel_x_g', 'accel_y_g', 'accel_z_g']].to_numpy()
        gyro_data_rad = df[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].to_numpy()

        # Convert accelerometer from g's to m/s^2 for analysis
        accel_data_ms2 = accel_data_g * GRAVITY

    except FileNotFoundError:
        print(f"Error: File not found at '{args.filepath}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print(f"Successfully loaded and processed {len(df)} samples from '{args.filepath}' at {args.rate} Hz.")

    # --- Analyze each sensor axis ---
    analyze_sensor(gyro_data_rad[:, 0], args.rate, "Gyro X", "rad/s")
    analyze_sensor(gyro_data_rad[:, 1], args.rate, "Gyro Y", "rad/s")
    analyze_sensor(gyro_data_rad[:, 2], args.rate, "Gyro Z", "rad/s")

    analyze_sensor(accel_data_ms2[:, 0], args.rate, "Accel X", "m/s²")
    analyze_sensor(accel_data_ms2[:, 1], args.rate, "Accel Y", "m/s²")
    analyze_sensor(accel_data_ms2[:, 2], args.rate, "Accel Z", "m/s²")

if __name__ == "__main__":
    main()