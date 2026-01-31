# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.


import os
import sys
import argparse
import numpy as np
import h5py
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.config import Config

class SimTrajectoryGenerator:
    """Generates synthetic trajectory and IMU data for Trajecto training."""

    def __init__(self, fs: float = Config.TARGET_SAMPLING_RATE_HZ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.gravity = np.array([0.0, 0.0, -Config.GRAVITY_MAGNITUDE])

    def generate_lissajous_trajectory(self, duration: float, 
                                      scale: Tuple[float, float, float] = (0.2, 0.2, 0.05),
                                      freq: Tuple[float, float, float] = (0.5, 0.5, 1.0),
                                      phase: Tuple[float, float, float] = (0.0, 1.57, 0.0)) -> Dict[str, np.ndarray]:
        """Generates a 3D Lissajous trajectory."""
        t = np.arange(0, duration, self.dt)
        
        # Position
        x = scale[0] * np.sin(2 * np.pi * freq[0] * t + phase[0])
        y = scale[1] * np.sin(2 * np.pi * freq[1] * t + phase[1])
        z = scale[2] * np.sin(2 * np.pi * freq[2] * t + phase[2])
        pos_w = np.stack([x, y, z], axis=1)

        # Velocity (analytical derivative)
        vx = scale[0] * 2 * np.pi * freq[0] * np.cos(2 * np.pi * freq[0] * t + phase[0])
        vy = scale[1] * 2 * np.pi * freq[1] * np.cos(2 * np.pi * freq[1] * t + phase[1])
        vz = scale[2] * 2 * np.pi * freq[2] * np.cos(2 * np.pi * freq[2] * t + phase[2])
        vel_w = np.stack([vx, vy, vz], axis=1)

        # Acceleration (analytical derivative)
        ax = -scale[0] * (2 * np.pi * freq[0])**2 * np.sin(2 * np.pi * freq[0] * t + phase[0])
        ay = -scale[1] * (2 * np.pi * freq[1])**2 * np.sin(2 * np.pi * freq[1] * t + phase[1])
        az = -scale[2] * (2 * np.pi * freq[2])**2 * np.sin(2 * np.pi * freq[2] * t + phase[2])
        accel_w = np.stack([ax, ay, az], axis=1)

        return {
            "pos_w": pos_w,
            "vel_w": vel_w,
            "accel_w": accel_w,
            "time": t
        }

    def generate_imu_data(self, trajectory: Dict[str, np.ndarray],
                          static_duration: float = 2.5,
                          noise_std_acc: float = 0.01,
                          noise_std_gyro: float = 0.001,
                          bias_acc: Optional[np.ndarray] = None,
                          bias_gyro: Optional[np.ndarray] = None,
                          pen_tilt: Tuple[float, float] = (0.5, 0.0)) -> Dict[str, np.ndarray]:
        """
        Generates IMU data from World Frame trajectory.
        Includes a static buffer at the start.
        
        Args:
            trajectory: Dict with pos_w, vel_w, accel_w, time.
            static_duration: Seconds of static data to prepend.
            noise_std_acc: Std dev of accelerometer noise.
            noise_std_gyro: Std dev of gyroscope noise.
            bias_acc: Constant accelerometer bias [3].
            bias_gyro: Constant gyroscope bias [3].
            pen_tilt: (pitch, roll) in radians for static orientation base.
        """
        
        # 1. Prepare Static Buffer
        num_static = int(static_duration * self.fs)
        t_static = np.arange(0, num_static) * self.dt
        
        pos_static = np.tile(trajectory["pos_w"][0], (num_static, 1))
        vel_static = np.zeros((num_static, 3))
        accel_w_static = np.zeros((num_static, 3)) # No kinematic accel, just gravity later
        
        # 2. Combine Static + Dynamic
        pos_full = np.vstack([pos_static, trajectory["pos_w"]])
        vel_full = np.vstack([vel_static, trajectory["vel_w"]])
        accel_w_full = np.vstack([accel_w_static, trajectory["accel_w"]])
        
        num_samples = len(pos_full)
        
        # 3. Orientation Simulation (Simple: Fixed Tilt + some sway correlated with velocity)
        # Base orientation: Pen held at an angle
        # Pitch ~ 45 deg, Yaw arbitrary.
        
        # Let's define a Body-to-World orientation.
        # Ideally, we want World-to-Body to project gravity.
        # But R in scipy is usually specified as "rotation vector" or "euler".
        
        # Let's assume a slightly dynamic orientation based on velocity direction or just random sway
        # For simplicity: Constant base orientation + small random walk
        
        # Base rotation (Euler: ZYX sequence)
        # Yaw=0, Pitch=pen_tilt[0], Roll=pen_tilt[1]
        base_r = R.from_euler('yxz', [pen_tilt[0], pen_tilt[1], 0], degrees=False)
        base_quat = base_r.as_quat() # (x, y, z, w)
        
        # Replicate base orientation
        quats = np.tile(base_quat, (num_samples, 1))
        
        # Add some orientation dynamics (wobble)
        # Simple sinusoidal wobble based on time
        t_full = np.arange(num_samples) * self.dt
        wobble_pitch = 0.1 * np.sin(2 * np.pi * 0.5 * t_full)
        wobble_roll = 0.1 * np.cos(2 * np.pi * 0.3 * t_full)
        
        # Apply wobble
        # Convert to Rotation objects
        r_base = R.from_quat(quats)
        r_wobble = R.from_euler('yxz', np.stack([wobble_pitch, wobble_roll, np.zeros_like(t_full)], axis=1), degrees=False)
        
        # Combined rotation: R_wb = R_base * R_wobble
        r_wb = r_base * r_wobble
        
        # 4. Compute Angular Velocity (Gyro)
        # omega_b approx (rotation difference) / dt
        # R[t+1] = R[t] * R_delta
        # R_delta = R[t]^T * R[t+1]
        # R_delta approx I + [omega * dt]_x
        
        # Using scipy:
        # rot_vec = (r[t].inv() * r[t+1]).as_rotvec()
        # omega = rot_vec / dt
        
        r_inv = r_wb.inv()
        # Shifted multiplication for diff
        r_next = r_wb[1:]
        r_curr = r_inv[:-1]
        r_diff = r_curr * r_next
        rot_vecs = r_diff.as_rotvec()
        gyro_clean = rot_vecs / self.dt
        # Pad last sample
        gyro_clean = np.vstack([gyro_clean, gyro_clean[-1]])
        
        # 5. Compute Accelerometer (Body Frame)
        # a_meas = R_wb^T * (a_w - g_w)
        # g_w = [0, 0, -9.81]
        
        g_w_vec = np.tile(self.gravity, (num_samples, 1))
        accel_true_w = accel_w_full - g_w_vec # Specific force in world frame
        
        # Rotate to body frame
        accel_clean = r_wb.inv().apply(accel_true_w)
        
        # 6. Add Biases and Noise
        if bias_acc is None: bias_acc = np.array([0.1, -0.1, 0.05])
        if bias_gyro is None: bias_gyro = np.array([0.01, 0.01, -0.01])
        
        accel_noisy = accel_clean + bias_acc + np.random.normal(0, noise_std_acc, accel_clean.shape)
        gyro_noisy = gyro_clean + bias_gyro + np.random.normal(0, noise_std_gyro, gyro_clean.shape)
        
        # 7. FSR (Force Sensitive Resistor)
        # Static: 0 (or noise)
        # Dynamic: High
        fsr = np.zeros(num_samples)
        fsr[num_static:] = 1.0 + np.random.normal(0, 0.05, num_samples - num_static) # Normalized force > 0
        fsr = np.clip(fsr, 0, None)
        
        # 8. Gravity GT (Body Frame)
        # g_b = R_wb^T * g_w
        # Wait, usually gravity GT is unit vector pointing roughly DOWN in world, but rotated to Body
        # g_b_unit = R_wb^T * [0, 0, -1]
        g_w_unit = np.array([0.0, 0.0, -1.0])
        gravity_b = r_wb.inv().apply(g_w_unit)
        
        # 9. Pack Data
        # Sensor: [Acc(3), Gyro(3), FSR(1)]
        sensor_data = np.hstack([accel_noisy, gyro_noisy, fsr.reshape(-1, 1)])
        
        return {
            "sensor_data": sensor_data,
            "gt_pos_data": pos_full,
            "gt_vel_data": vel_full,
            "gt_gravity_b_data": gravity_b,
            "sequence_length": num_samples
        }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Trajecto data.")
    parser.add_argument("--output", type=str, default="data/simulated_dataset.h5", help="Output HDF5 file.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of writing (excluding static buffer).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    gen = SimTrajectoryGenerator()
    
    print(f"Generating {args.samples} samples into {args.output}...")
    
    with h5py.File(args.output, "w") as f:
        for i in range(args.samples):
            # Randomize parameters for diversity
            scale = (
                np.random.uniform(0.1, 0.3),
                np.random.uniform(0.1, 0.3),
                np.random.uniform(0.0, 0.1)
            )
            freq = (
                np.random.uniform(0.3, 0.8),
                np.random.uniform(0.3, 0.8),
                np.random.uniform(0.8, 1.5)
            )
            phase = (
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 2*np.pi)
            )
            
            traj = gen.generate_lissajous_trajectory(args.duration, scale, freq, phase)
            
            # Randomize Bias
            bias_acc = np.random.uniform(-0.2, 0.2, 3)
            bias_gyro = np.random.uniform(-0.02, 0.02, 3)
            
            data = gen.generate_imu_data(traj, 
                                         static_duration=2.5, 
                                         bias_acc=bias_acc, 
                                         bias_gyro=bias_gyro)
            
            # Save to HDF5
            grp_name = f"sim_sample_{i:03d}"
            grp = f.create_group(grp_name)
            
            grp.create_dataset("sensor_data", data=data["sensor_data"])
            grp.create_dataset("gt_pos_data", data=data["gt_pos_data"])
            grp.create_dataset("gt_vel_data", data=data["gt_vel_data"])
            grp.create_dataset("gt_gravity_b_data", data=data["gt_gravity_b_data"])
            
            grp.attrs["sequence_length"] = data["sequence_length"]
            grp.attrs["original_label"] = "simulation"
            
    print("Done.")

if __name__ == "__main__":
    main()
