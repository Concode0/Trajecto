# Trajecto: 3D Pen Trajectory Reconstruction

## Status: Under Active Development

**Trajecto** is a high-precision handwriting trajectory estimation system that reconstructs 3D pen movements using a single 6-axis IMU (BMI270).

It utilizes a **Hybrid Architecture** combining Deep Learning (Temporal Convolutional Networks) with Physics-based Filtering (ESKF/AEKF) to overcome the inherent drift of low-cost inertial sensors. The system is designed for **Sim2Real** deployment, featuring a complete pipeline from PyTorch training to optimized C++ inference on ESP32 microcontrollers.

## Features

- **Hybrid Physics+AI Models**: Integrates Error-State Kalman Filters (ESKF) with TCNs to correct velocity and orientation errors in real-time.
- **Robust Data Protocol**: Implements a strict "Tap-Wait-Write" acquisition protocol for reliable synchronization and gravity alignment.
- **Embedded Inference**: Fully optimized C++ port for ESP32C3 using TFLite Micro with INT8 quantization and stateful buffering.
- **Sim2Real Pipeline**: Automated tools for exporting PyTorch models to ONNX, TFLite, and C++ arrays.
- **Advanced ZUPT**: Zero-Velocity Update detection to minimize drift during stationary periods.

## System Architecture

The system estimates a 15-dimensional Error State to correct the inertial integration:

1.  **Input**: 6-axis IMU data (Accel, Gyro) in the Body Frame.
2.  **TCN Engine**: Extracts features and predicts corrections (velocity residuals, zero-velocity probability).
3.  **Filter (ESKF/AEKF)**: Fuses raw IMU physics with TCN predictions.
    -   **State**: Position, Velocity, Orientation (Quaternion), Accel Bias, Gyro Bias.
    -   **Output**: Trajectory in the World Frame (NED).

## Data Acquisition Protocol (CRITICAL)

All data collection must strictly follow the **"Double-Tap-Wait-Write"** protocol to ensure valid synchronization and leveling:

1.  **Initial Tap (Start Sync)**: A sharp acceleration spike to mark the beginning of data capture.
2.  **Calib/Static Wait**: ~2 seconds of static data collection to initialize gravity alignment and sensor biases. This also serves as a buffer for discarding initial impact noise.
3.  **Write**: The actual handwriting motion.
4.  **Stop Wait**: A brief period of static data after writing is complete.
5.  **Final Tap (End Sync)**: A sharp acceleration spike to mark the end of the writing segment.

## Directory Structure

```text
Trajecto/
├── acquired_data/      # Raw CSV data from sensors
├── analyzer/           # Julia-based offline analysis tools
├── data/               # Training datasets and stats
├── docs/               # Documentation
├── firmware/           # ESP32 C++ Firmware (ESP-IDF)
├── model/              # PyTorch Model definitions (ESKF, TCN, etc.)
├── onnx_export/        # Exported ONNX models
├── utils/              # Utility scripts (Acquisition, Export, Visualization)
├── Embedded_Porting.md # Detailed guide for ESP32 deployment
├── GEMINI.md           # Project Context & Standards
├── train.py            # Main training script
└── pyproject.toml      # Project dependencies (managed by uv)
```

## Getting Started

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- ESP-IDF (for firmware development)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Trajecto.git
    cd Trajecto
    ```

2.  **Sync dependencies:**
    Trajecto uses `uv` for fast dependency management.
    ```bash
    uv sync
    source .venv/bin/activate
    ```

## Workflow

### 1. Data Acquisition
Use the utility scripts to collect training data from the device over BLE/Serial.
- `utils/acquire.py`: Main script for data collection.
- `utils/receive.py`: Passive receiver script.

### 2. Training
Train the hybrid models using `train.py`. The script handles data loading and augmentation.

```bash
# Train ESKF-TCN (Standard Hybrid)
python train.py --model eskf_tcn --epochs 200 --lr 1e-4

# Train Standalone TCN
python train.py --model only_tcn --epochs 200
```

### 3. Visualization
Visualize the reconstructed 3D trajectories:
- `utils/h5_viewer.py`: Interactive TUI for viewing HDF5 datasets.
- `utils/check_data.py`: Quick validation of data integrity.

### 4. Embedded Deployment (Sim2Real)
To deploy the trained model to the ESP32:

1.  **Train**: Generate `results/eskf_tcn_model.pth`.
2.  **Export**: `python utils/export_onnx.py` -> `onnx_export/tcn_model.onnx`.
3.  **Convert**: `python utils/convert_tflite.py` -> `firmware/main/tcn_model_dynamic_range_quant.tflite`.
4.  **Flash**: Build and flash the `firmware/` project.

*See [Embedded_Porting.md](Embedded_Porting.md) for the complete step-by-step guide.*

## Hardware

The system uses a custom PCB with an ESP32C3 and BMI270 IMU.
Design files are available in the `hardware/` directory.

| Top View | Bottom View |
| :---: | :---: |
| ![Top](hardware/images/Trajecto-Board-Top.png) | ![Bottom](hardware/images/Trajecto-Board-Bottom.png) |

## Firmware

The `firmware/` directory contains the C++ implementation for the ESP32C3.
- **Framework**: ESP-IDF
- **Inference**: TFLite Micro (INT8/Dynamic Quantization)
- **Math**: Eigen (for Kalman Filtering)

## License

This work is licensed under the **GNU General Public License v3.0**.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)