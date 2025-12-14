"""
This script generates synthetic ground truth and sensor data for trajectory
estimation.

The generated data simulates IMU (accelerometer, gyroscope) and FSR (Force
Sensitive Resistor) readings along a predefined trajectory. It creates separate
CSV files for ground truth and sensor data, which can then be used by the
'preprocess.py' script.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd


# --- Helper Functions ---
def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions.

    Args:
        q1 (np.ndarray): The first quaternion (w, x, y, z).
            - Shape: (4,)
            - Frame: Body/World (depends on usage)
        q2 (np.ndarray): The second quaternion (w, x, y, z).
            - Shape: (4,)
            - Frame: Body/World (depends on usage)

    Returns:
        np.ndarray: The resulting quaternion from the multiplication.
            - Shape: (4,)
            - Frame: Body/World (depends on usage)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion to a 3x3 rotation matrix.

    Args:
        q (np.ndarray): The quaternion (w, x, y, z).
            - Shape: (4,)
            - Frame: Body-to-World (typically)

    Returns:
        np.ndarray: A 3x3 rotation matrix.
            - Shape: (3, 3)
            - Frame: Body-to-World (typically)
    """
    q_norm = q / (np.linalg.norm(q) + 1e-8)  # Normalize quaternion to prevent NaNs
    w, x, y, z = q_norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def generate_trajectory_data(
    session_id: int,
    number_of_samples: int,
    points_per_sample: int = 1000,
    dt: float = 0.01,
    output_dir: str = "./acquired_data",
) -> None:
    """Generates synthetic ground truth and sensor data based on a predefined trajectory.

    This function simulates IMU (accelerometer, gyroscope) and FSR readings,
    along with ground truth position, velocity, and force. The data is saved
    into separate CSV files for each sample.

    Args:
        session_id: An identifier for the current session, used in naming output files.
        number_of_samples: The total number of individual data samples to generate.
        points_per_sample: The number of data points (time steps) in each sample.
            Defaults to 1000.
        dt: The time step duration in seconds. Defaults to 0.01.
        output_dir: The directory where the generated CSV files will be saved.
            Defaults to './acquired_data'.
    """
    os.makedirs(output_dir, exist_ok=True)

    GRAVITY: float = 9.81  # Magnitude of gravity
    GRAVITY_VECTOR: np.ndarray = np.array([0.0, 0.0, -GRAVITY])

    for i in range(number_of_samples):
        # Naming convention for the generated files.
        session_name_base = f"Session_{session_id:02d}_{i:03d}_A"

        # Generate time vector for the current sample.
        t = np.arange(0, points_per_sample * dt, dt)

        # --- 1. Generate Predefined Ground Truth Trajectory ---
        # Introduce random frequencies and phases for varied sinusoidal trajectories.
        f_x, f_y, f_z = np.random.uniform(0.1, 0.3, 3)
        p_x, p_y, p_z = np.random.uniform(0, np.pi, 3)
        pos_x = 0.5 * np.sin(2 * np.pi * f_x * t + p_x)
        pos_y = 0.5 * np.cos(2 * np.pi * f_y * t + p_y)
        pos_z = 0.2 * np.sin(2 * np.pi * f_z * t + p_z)
        pos: np.ndarray = np.vstack([pos_x, pos_y, pos_z]).T

        # --- 2. Calculate Derivatives (for simulating sensor data) ---
        # Velocity and acceleration are calculated by numerical differentiation
        # of the ground truth position.
        vel: np.ndarray = np.gradient(pos, dt, axis=0)
        accel_world: np.ndarray = np.gradient(vel, dt, axis=0)

        # --- 3. Generate Synthetic Orientation (Quaternion) ---
        quat: np.ndarray = np.zeros((len(t), 4))
        quat[0] = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation (identity quaternion)

        # Generate synthetic gyroscope readings for rotation.
        w_freq = np.random.uniform(0.05, 0.2)
        gyro_true: np.ndarray = np.vstack(
            [
                0.1 * np.sin(2 * np.pi * w_freq * t),
                0.1 * np.cos(2 * np.pi * w_freq * t),
                np.zeros_like(t),  # No rotation around Z-axis for simplicity
            ]
        ).T

        # Integrate gyroscope data to simulate quaternion orientation over time.
        for k in range(len(t) - 1):
            omega = gyro_true[k]
            omega_norm = np.linalg.norm(omega)
            if omega_norm > 1e-8:  # Avoid division by zero
                angle = omega_norm * dt
                axis = omega / omega_norm
                dq_w = np.cos(angle / 2)
                dq_xyz = np.sin(angle / 2) * axis
                dq = np.concatenate(([dq_w], dq_xyz))
                quat[k + 1] = _quaternion_multiply(quat[k], dq)
            else:
                quat[k + 1] = quat[k]

        # --- 4. Calculate "True" Body-Frame Sensor Data ---
        accel_body_true: np.ndarray = np.zeros_like(accel_world)
        for k in range(len(t)):
            # Rotate world-frame acceleration (minus gravity) into body frame.
            rotation_matrix = _quaternion_to_rotation_matrix(quat[k])
            accel_body_true[k] = rotation_matrix.T @ (accel_world[k] - GRAVITY_VECTOR)

        # --- 5. Add Noise to Create Final Sensor Data ---
        NOISE_LEVEL_ACCEL: float = 0.25
        NOISE_LEVEL_GYRO: float = 0.15
        NOISE_LEVEL_FSR: float = 0.5

        accel_noisy: np.ndarray = accel_body_true + np.random.normal(
            0, NOISE_LEVEL_ACCEL, size=accel_body_true.shape
        )
        gyro_noisy: np.ndarray = gyro_true + np.random.normal(
            0, NOISE_LEVEL_GYRO, size=gyro_true.shape
        )

        # FSR data is simulated based on the magnitude of body-frame acceleration
        # plus some noise, ensuring non-negative values.
        force_noisy: np.ndarray = np.linalg.norm(accel_body_true, axis=1) + np.random.normal(
            0, NOISE_LEVEL_FSR, size=len(t)
        )
        fsr_data: np.ndarray = np.maximum(0, force_noisy)

        # --- 6. Write Ground Truth Data to CSV ---
        gt_df = pd.DataFrame(
            {
                "time_delta": t,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
                "force": np.linalg.norm(accel_body_true, axis=1),  # True force for GT
            }
        )
        gt_filepath = os.path.join(output_dir, f"{session_name_base}_gt.csv")
        gt_df.to_csv(gt_filepath, index=False)

        # --- 7. Write Sensor Board Data to CSV ---
        sensor_df = pd.DataFrame(
            {
                "time": t,
                "accel_x": accel_noisy[:, 0],
                "accel_y": accel_noisy[:, 1],
                "accel_z": accel_noisy[:, 2],
                "gyro_x": gyro_noisy[:, 0],
                "gyro_y": gyro_noisy[:, 1],
                "gyro_z": gyro_noisy[:, 2],
                "fsr": fsr_data,  # Include FSR data
            }
        )
        sensor_filepath = os.path.join(output_dir, f"{session_name_base}.csv")
        sensor_df.to_csv(sensor_filepath, index=False)


if __name__ == "__main__":
    # Clean up only the dummy CSV files generated by this script.
    ACQUIRED_DATA_DIR = "./acquired_data"
    if os.path.exists(ACQUIRED_DATA_DIR):
        print(f"Cleaning up old CSV files in {ACQUIRED_DATA_DIR}...")
        for f in os.listdir(ACQUIRED_DATA_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(ACQUIRED_DATA_DIR, f))

    # Generate 5 samples for session 1.
    generate_trajectory_data(session_id=1, number_of_samples=5)
    print("Done generating new trajectory-based dummy data in CSV format!")