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
Calibration Data Collector for TFLite Quantization

This script collects calibration data from the Trajecto device for use with
TensorFlow Lite's dynamic range quantization. It records real IMU input sequences
and the corresponding TCN internal states to ensure accurate INT8 quantization.
"""

import asyncio
import torch
import numpy as np
import os
import h5py
from tqdm import tqdm
from receive import TrajectoDriver, RawImuPacket
from model.ESKF_TCN import ESKFTCN_model
from model.stateful_tcn import StatefulTCNExport
from model.config import Config

# Save path (matching convert_tflite.py)
CALIB_DIR = "onnx_export/calib_data"
os.makedirs(CALIB_DIR, exist_ok=True)


class CalibDataCollector:
    """Collects calibration data for TFLite quantization"""

    def __init__(self, model_path="eskf_tcn_model.pth"):
        self.device = "cpu"

        # Load base model
        base_model = ESKFTCN_model(device=self.device, dt=Config.DT)
        base_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Extract stateful TCN
        self.stateful_tcn = StatefulTCNExport(base_model.tcn)
        self.stateful_tcn.eval()

        # Initialize state buffers
        self.state_buffers = []
        for layer in base_model.tcn.tcn_layers:
            k = layer.conv.kernel_size[0]
            d = layer.conv.dilation[0]
            hist_len = (k - 1) * d
            in_ch = layer.conv.in_channels
            self.state_buffers.append(torch.zeros(1, hist_len, in_ch))

        # Normalization stats
        with h5py.File(Config.SCALER_STATS_H5_PATH, "r") as f:
            self.mean = torch.from_numpy(f["mean"][:]).float()
            self.std = torch.from_numpy(f["std"][:]).float()

        self.sample_count = 0

    def process_imu_packet(self, packet: RawImuPacket):
        """
        Process one IMU packet and save calibration data.

        Args:
            packet: RawImuPacket from BLE driver
        """
        # Convert to raw sensor vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, fsr]
        raw_imu = torch.tensor([
            packet.accel[0],  # m/s^2
            packet.accel[1],
            packet.accel[2],
            packet.gyro[0],   # rad/s
            packet.gyro[1],
            packet.gyro[2],
            float(packet.force)
        ], dtype=torch.float32)

        # Normalize
        norm_imu = (raw_imu - self.mean) / (self.std + 1e-6)

        # TODO: Build full 19D feature vector for TCN
        # This would require maintaining ESKF state, which is complex
        # For basic calibration, we can use a simplified approach or just raw IMU

        # For now, save raw normalized IMU
        # In production, you'd compute the full feature vector
        norm_imu_seq = norm_imu.unsqueeze(0).unsqueeze(0)  # [1, 1, 7]

        # Save input and states
        with torch.no_grad():
            np.save(f"{CALIB_DIR}/input_raw_{self.sample_count}.npy", norm_imu_seq.numpy())

            for i, state in enumerate(self.state_buffers):
                np.save(f"{CALIB_DIR}/state_in_{i}_{self.sample_count}.npy", state.numpy())

            # Note: You'd need to pass the full 19D feature vector here
            # outputs = self.stateful_tcn(full_feature_vector, *self.state_buffers)
            # self.state_buffers = list(outputs[1:])

        self.sample_count += 1


async def main():
    """Main calibration data collection routine"""
    print("=" * 60)
    print("Trajecto Calibration Data Collector")
    print("=" * 60)
    print(f"Saving calibration data to: {CALIB_DIR}")

    num_samples = 200
    print(f"Target samples: {num_samples}")

    collector = CalibDataCollector()
    collected = [0]  # Use list to allow modification in closure

    pbar = tqdm(total=num_samples, desc="Collecting Samples")

    def on_imu_received(packet: RawImuPacket):
        """Callback for each IMU packet"""
        if collected[0] < num_samples:
            collector.process_imu_packet(packet)
            collected[0] += 1
            pbar.update(1)

    # Create driver with callback
    driver = TrajectoDriver(raw_callback=on_imu_received, verbose=True)

    try:
        # Connect to device
        if not await driver.connect():
            print("Failed to connect to device!")
            return

        print("\nInstructions:")
        print("1. Hold the device and move it naturally")
        print("2. Include various motions (rotations, translations, writing)")
        print("3. Data will be collected automatically")
        print("\nPress Enter to start...")
        input()

        # Start streaming
        if not await driver.start_streaming(mode=0):  # Raw mode
            print("Failed to start streaming!")
            return

        # Wait until enough samples collected
        while collected[0] < num_samples:
            await asyncio.sleep(0.1)

        # Stop streaming
        await driver.stop_streaming()
        pbar.close()

        print(f"\n✅ Collected {collected[0]} samples!")
        print(f"Calibration data saved to: {CALIB_DIR}")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    finally:
        await driver.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
