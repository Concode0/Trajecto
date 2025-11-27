"""
Generate dummy data -> Ground Truth / Sensor Board Data in h5 format.

1. Make Fake Trajectory
2. Convert Trajectory to IMU data.
"""

import numpy as np
import h5py
import os

# Helper for quaternion multiplication
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

# Helper to get rotation matrix from quaternion
def quaternion_to_rotation_matrix(q):
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q_norm
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def generate_trajectory_data(name_number, number_of_samples, points_per_sample=300, dt=0.02):
    """
    Generates and saves ground truth and sensor data from a predefined trajectory.
    """
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    gt_filepath = os.path.join(data_dir, f'Groud_Truth_Data_{name_number}.h5')
    sb_filepath = os.path.join(data_dir, f'Sensor_Board_Data_{name_number}.h5')

    gravity = np.array([0., 0., -9.81])

    with h5py.File(gt_filepath, 'w') as f_gt, h5py.File(sb_filepath, 'w') as f_sb:
        f_gt.attrs['descr'] = "Predefined 3D Trajectory (Ground Truth)"
        f_sb.attrs['descr'] = "Simulated IMU & Force Sensor Data from Trajectory"

        for i in range(number_of_samples):
            t = np.arange(0, points_per_sample * dt, dt)

            # --- 1. Generate Predefined Ground Truth Trajectory ---
            # Lissajous curve for a smooth, interesting path
            f_x, f_y, f_z = np.random.uniform(0.1, 0.3, 3)
            p_x, p_y, p_z = np.random.uniform(0, np.pi, 3)
            pos_x = 0.5 * np.sin(2 * np.pi * f_x * t + p_x)
            pos_y = 0.5 * np.cos(2 * np.pi * f_y * t + p_y)
            pos_z = 0.2 * np.sin(2 * np.pi * f_z * t + p_z)
            pos = np.vstack([pos_x, pos_y, pos_z]).T

            # --- 2. Calculate Derivatives to get Velocity and Acceleration ---
            vel = np.gradient(pos, dt, axis=0)
            accel_world = np.gradient(vel, dt, axis=0)

            # --- 3. Generate Synthetic Orientation (Quaternion) ---
            quat = np.zeros((len(t), 4))
            quat[0] = [1, 0, 0, 0] # Initial orientation
            
            # Simple, slow rotation to simulate hand movement
            w_freq = np.random.uniform(0.05, 0.2)
            gyro_true = np.vstack([
                0.1 * np.sin(2 * np.pi * w_freq * t),
                0.1 * np.cos(2 * np.pi * w_freq * t),
                np.zeros_like(t)
            ]).T

            for k in range(len(t) - 1):
                omega = gyro_true[k]
                omega_norm = np.linalg.norm(omega)
                if omega_norm > 1e-8:
                    angle = omega_norm * dt
                    axis = omega / omega_norm
                    dq_w = np.cos(angle / 2)
                    dq_xyz = np.sin(angle / 2) * axis
                    dq = np.concatenate(([dq_w], dq_xyz))
                    quat[k+1] = quaternion_multiply(quat[k], dq)
                else:
                    quat[k+1] = quat[k]

            # --- 4. Calculate "True" Body-Frame Sensor Data ---
            accel_body_true = np.zeros_like(accel_world)
            for k in range(len(t)):
                R = quaternion_to_rotation_matrix(quat[k])
                accel_body_true[k] = R.T @ (accel_world[k] - gravity)

            # --- 5. Add Noise to Create Final Sensor Data ---
            noise_level_accel = 0.25
            noise_level_gyro = 0.15
            
            accel_noisy = accel_body_true + np.random.normal(0, noise_level_accel, size=accel_body_true.shape)
            gyro_noisy = gyro_true + np.random.normal(0, noise_level_gyro, size=gyro_true.shape)

            force_true = np.linalg.norm(accel_body_true, axis=1)
            force_noisy = force_true + np.random.normal(0, 0.5, size=force_true.shape)
            force_noisy = np.maximum(0, force_noisy) # Force can't be negative

            # --- 6. Write Ground Truth Data ---
            gt_group = f_gt.create_group(f'/Samples_{i}')
            gt_group.attrs['start_time'] = 200.0 * i
            gt_group.attrs['number_of_points'] = points_per_sample
            gt_group['x'], gt_group['y'], gt_group['z'] = pos.T
            gt_group['force'] = force_true
            gt_group['Time_delta'] = t

            # --- 7. Write Sensor Board Data ---
            sb_group = f_sb.create_group(f'/Samples_{i}')
            sb_group.attrs['start_time'] = 200.0 * i
            sb_group.attrs['number_of_points'] = points_per_sample
            sb_group['Ax'], sb_group['Ay'], sb_group['Az'] = accel_noisy.T
            sb_group['Gx'], sb_group['Gy'], sb_group['Gz'] = gyro_noisy.T
            sb_group['Force'] = force_noisy
            sb_group['Time_delta'] = t

if __name__ == "__main__":
    generate_trajectory_data(name_number=1, number_of_samples=100)
    print("Done generating new trajectory-based dummy data!")