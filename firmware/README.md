# Trajecto Firmware (ESP32)

This directory contains the embedded C++ firmware for the Trajecto system, designed for the ESP32C3 microcontroller.

It implements a **Sim2Real** pipeline, running the full hybrid ESKF-TCN trajectory estimation model on-device in real-time.

## Architecture

The firmware is built on **ESP-IDF** and utilizes a layered architecture:

1.  **Hardware Layer**:
    *   **IMU**: Bosch BMI270 (via I2C) @ 400Hz.
    *   **FSR**: Pressure sensor for ZUPT detection.
    *   **BLE**: Custom GATT service for data streaming.

2.  **Inference Engine**:
    *   **TFLite Micro**: Runs the quantized (INT8) TCN model to predict velocity residuals and zero-velocity probability.
    *   **Eigen**: Performs the Error-State Kalman Filter (ESKF) updates for 6-DOF tracking.

3.  **Application Layer**:
    *   **Real-time Tracking**: Fuses IMU and TCN outputs.
    *   **Data Logger**: Buffers and streams raw/processed data for debugging.

## Features

- **On-Device Inference**: Runs the full TCN+ESKF model at ~50Hz (inference) / 400Hz (integration).
- **Quantization**: Uses INT8 TFLite models for efficient execution on ESP32C3.
- **Stateful Buffer**: Manages TCN causal history for continuous streaming.
- **BLE Service**: Streams real-time trajectory and raw sensor data.

## Hardware Setup

*   **MCU**: ESP32C3 (Required for adequate AI instruction support).
*   **IMU**: BMI270 connected to I2C.
    *   SDA: GPIO 10
    *   SCL: GPIO 4
    *   INT: GPIO 1
*   **FSR**: Analog input on GPIO 3.

## Build & Flash

This project uses **ESP-IDF**.

### Prerequisites
*   ESP-IDF v5.0+
*   Model file: Ensure `tcn_model.cc` (generated from `utils/convert_tflite.py`) is present in `main/`.

### Commands

1.  **Navigate to firmware directory:**
    ```bash
    cd firmware
    ```

2.  **Set Target:**
    ```bash
    idf.py set-target esp32s3
    ```

3.  **Build:**
    ```bash
    idf.py build
    ```

4.  **Flash & Monitor:**
    ```bash
    idf.py flash monitor
    ```

## BLE Protocol

The device exposes a custom service `AD43...` with characteristics for:
- **Command**: Write `"strt"`/`"stop"` to control execution.
- **Data**: Notifications containing packed trajectory data (Pos, Vel, Quat).

## Development

*   **Main Loop**: `main/main.cpp` - Orchestrates sensor reading and model inference.
*   **TCN Wrapper**: `main/tcn_wrapper.cpp` - Handles TFLite Micro interpreter and tensor buffers.
*   **ESKF**: `main/eskf.cpp` - C++ implementation of the Kalman Filter.

## License

This work is licensed under the **GNU General Public License v3.0**.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
