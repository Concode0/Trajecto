# Trajecto: AI-Enhanced 3D Pen Tracking

<img src="logo.png" width="200" alt="Trajecto Logo">

**Trajecto** is a centimeter-level 3D handwriting reconstruction system that fuses **Deep Learning (TCN)** with **Physics-Based Filtering (ESKF)**. It uses a single low-cost 6-axis IMU (BMI270) to track pen trajectories in real-time on an **ESP32-C3** microcontroller.

---

## 🚧 Project Status: Active Development
This project is **currently under intensive development**. We are actively refining the hybrid neuro-physical models, optimizing embedded inference, and expanding the ground-truth dataset.

📄 **[Detailed Technical Specification (PDF)](technical_spec.pdf)**
*(Note: Documentation is currently incomplete and in active progress.)*

---

## 🔬 Technical Breakthroughs

Trajecto bridges the gap between purely data-driven black boxes and rigid physical models through a novel **Closed-Loop Hybrid Architecture**:

### 1. Hybrid Neuro-Physical Core
Instead of predicting positions directly (which drift unboundedly), Trajecto uses a **Temporal Convolutional Network (TCN)** to correct the error states of an **Error-State Kalman Filter (ESKF)**.
*   **Physics Backbone**: The ESKF integrates raw IMU dynamics at **400Hz**, guaranteeing physical consistency (Newtonian mechanics) and low-latency tracking.
*   **AI Correction**: The TCN observes a window of raw sensor data to predict **Velocity Residuals** and **ZUPT (Zero-Velocity Update)** probabilities, dynamically tuning the Kalman Gain ($K$) and Measurement Noise ($R$).

### 2. Parallel Scan Optimization
To enable efficient long-sequence training, we implemented a **Parallel Scan (Associative Scan)** algorithm for the Kalman Filter.
*   **O(log N) Complexity**: Unlike traditional sequential filters ($O(N)$), our parallel formulation allows computing covariance propagation across thousands of timesteps in logarithmic time on GPUs.
*   **Differentiable Filter**: The entire ESKF-TCN pipeline is end-to-end differentiable, allowing gradients to flow from the position loss back to the neural network weights.

### 3. Sim2Real & Quantization
The model is trained in PyTorch and deployed to embedded hardware via a rigorous **Sim2Real** pipeline:
*   **Quantization-Aware Training (QAT)**: The model is trained with simulated quantization noise to ensure fidelity when converted to **INT8** for the ESP32-C3.
*   **On-Device Inference**: The firmware runs TFLite Micro for the TCN and a custom C++ Eigen implementation for the ESKF, achieving a loop time of **<30ms** on a single-core RISC-V processor.

---

## 🛠️ System Components

### 🧠 Firmware (ESP32-C3)
*   **Real-Time Fusion**: Runs the hybrid model at 50Hz (inference) / 400Hz (integration).
*   **Stateful Buffer**: Manages TCN causal history to handle continuous streams without resetting state.
*   **BLE Service**: Streams packed trajectory data (Pos, Vel, Quat) via a custom GATT service.

### 📱 Data Acquisition (iPad Pro)
*   **Ground Truth**: A custom iPadOS app (**Trajectory Recorder**) captures Apple Pencil Pro data at **240Hz+** using coalesced touches.
*   **6-DoF Logging**: Records 3D position (x, y, hover-z), Azimuth, Altitude, and Roll for full pose supervision.
*   **Passive Logger**: Adheres to Apple SDK policies by offloading 3D coordinate transformations to an offline pipeline.

### 🧪 Analyzer (Julia)
*   **Plug-and-Play**: A high-performance offline analysis suite for verifying algorithms.
*   **Interactive Dashboard**: 3D Makie visualization with time-scrubbing and error metric analysis.
*   **Modular Estimation**: Hot-swap prediction models (`ESKF-TCN`, `AEKF`, `Pure TCN`) for comparison.

---

## ⚡ Workflow

### 1. Data Collection ("Tap-Wait-Write")
Precise synchronization is achieved via a physical protocol:
1.  **Tap 1**: Sharp acceleration spike synchronizes start time.
2.  **Wait (2s)**: **CRITICAL**. Static period for gravity alignment and bias estimation.
3.  **Write**: Perform the motion task.
4.  **Tap 2**: End sync spike for clock drift correction.

```bash
# Capture data from ESP32 (BLE) and iPad
python utils/acquire.py
```

### 2. Training
The training loop utilizes **Dynamic Weight Averaging (DWA)** to balance conflicting loss terms (Position vs. Velocity vs. ZUPT).

```bash
python train_eskf.py --epochs 200 --batch-size 16 --parallel-scan
```

### 3. Deployment
Compile the model for the ESP32-C3.

```bash
# 1. Export PyTorch -> ONNX -> TFLite (INT8)
python utils/convert_tflite.py --model_path checkpoints/best_model.pth

# 2. Flash Firmware
cd firmware
idf.py set-target esp32c3
idf.py build flash monitor
```

---

## 🔧 Hardware Specs

*   **MCU**: Espressif **ESP32-C3-MINI-1-N4** (RISC-V 32-bit, 160MHz)
*   **IMU**: Bosch **BMI270** (16-bit Accel/Gyro, Low-Noise)
*   **Power**: Active filtering and isolation for analog rails to ensure high signal integrity ($R^2 = 0.9991$).
*   **Input**: Force Sensitive Resistor (FSR) with active op-amp conditioning.

---

## 📂 Project Structure

```
Trajecto/
├── model/                  # PyTorch Models (TCN, ESKF, DWA)
│   ├── parallel_scan_ops.py # Custom Parallel Kalman Filter
│   └── ESKF_TCN.py         # Main Hybrid Architecture
├── firmware/               # ESP32-C3 Firmware (C++17)
│   ├── main/               # Application & Inference Logic
│   └── components/         # BMI270, TFLite Micro, Eigen
├── analyzer/               # Julia Analysis Suite
├── TrajectoryRecorder/     # iPad Data Acquisition App (Swift)
├── hardware/               # PCB Design (Kicad/Gerber)
└── utils/                  # Helper Scripts (Acquire, Convert)
```

## 📄 License

*   **Software**: Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).
*   **Hardware**: Licensed under the **CERN Open Hardware Licence Version 2 - Permissive (CERN-OHL-P)**. See [LICENSE_HARDWARE](LICENSE_HARDWARE).

Notice: This project includes technologies for which patents are currently pending (Application related to Hybrid ESKF-TCN and Data Acquisition Methodology).