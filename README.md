# 🖊️ Trajecto: 3D Pen Trajectory Reconstruction

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Trajecto is a comprehensive system for robust 3D pen trajectory reconstruction from a single IMU sensor. It combines classic state estimation techniques with deep learning to overcome the inherent drift problem in inertial sensors, enabling accurate tracking of movement in 3D space.

## ✨ Features

- **Hybrid Models**: Implements hybrid architectures combining Kalman Filters (ESKF, AEKF) with a Temporal Convolutional Network (TCN) for state-of-the-art drift correction.
- **Standalone Model**: Includes a pure TCN model (`OnlyTCN`) for a direct deep learning-based approach.
- **Data Pipeline**: Full pipeline from data acquisition, preprocessing, training, to visualization.
- **Embedded Firmware**: Contains the firmware for the ESP32-based data acquisition hardware.
- **Pressure-based ZUPT**: Utilizes a pressure sensor for robust Zero-Velocity Updates, significantly improving tracking accuracy during pauses.

## 🏛️ System Architecture

The core of this project lies in its hybrid modeling approach to trajectory reconstruction. By combining physical models (Kalman Filters) with data-driven deep learning models (TCN), Trajecto achieves high-precision 3D trajectories.

### Models
- **ESKF-TCN**: An Error-State Kalman Filter provides a robust trajectory baseline, which is then corrected by a TCN that learns to compensate for non-linear errors.
- **AEKF-TCN**: An Adaptive Extended Kalman Filter-based hybrid model.
- **OnlyTCN**: A standalone TCN that directly predicts the 3D trajectory from IMU data.

## 📂 Repository Structure

```
Trajecto/
├── TrajectoFW/         # Firmware for the ESP32-based hardware
├── acquired_data/      # Raw data from the data acquisition scripts
├── collection/         # Scripts for data acquisition
├── data/               # Processed and training-ready data
├── model/              # Model implementations (ESKF, AEKF, TCNs)
├── plots/              # Saved plots from training and visualization
├── utils/              # Utility scripts for preprocessing and visualization
├── train.py            # Main script for training the models
├── README.md           # This file
└── pyproject.toml      # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `uv` package manager (`pip install uv`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Trajecto.git
    cd Trajecto
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *(Note: If a `requirements.txt` is not present, you may need to generate it from `pyproject.toml` or install dependencies from it directly if your toolchain supports it.)*

## Workflow

The project follows a standard machine learning workflow.

### 1. Data Acquisition
Raw trajectory data can be collected using the scripts in the `collection/` directory. These scripts interface with the hardware to gather IMU and pressure sensor data.
- `collection/data_acquire.py`: Main script to start data acquisition.

### 2. Preprocessing
Once raw data is collected, it needs to be preprocessed into a format suitable for training.
- `utils/preprocess.py`: This script cleans the raw data, computes features, and creates the final HDF5 dataset (`dataset.h5`) used for training.

### 3. Training
Train the models using the `train.py` script. You can specify which model to train using command-line arguments.

**Train the ESKF-TCN model:**
```bash
python train.py --model eskf_tcn --epochs 200 --lr 1e-4
```

**Train the AEKF-TCN model:**
```bash
python train.py --model aekf_tcn --epochs 200 --lr 1e-4
```

**Train the OnlyTCN model:**
```bash
python train.py --model only_tcn --epochs 200 --lr 1e-4
```
The trained model weights (`.pth` file) and loss history plots will be saved.

### 4. Visualization
After training, you can visualize the performance of your model on test data.
- `utils/visualize.py`: This script loads a trained model and a data sample, runs inference, and generates 3D plots comparing the predicted trajectory to the ground truth.

## Firmware

The `TrajectoFW/` directory contains the C++ firmware for the ESP32-C3 based custom hardware. It uses the ESP-IDF framework. For more details on building and flashing the firmware, see the `README.md` inside the `TrajectoFW` directory.

## 📄 License

This work is licensed under a **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
</a>
