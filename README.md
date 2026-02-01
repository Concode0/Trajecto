# Trajecto: Vertical-Integration of AI-Aided 3D Pen Tracking

[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE) [![Hardware: CERN OHL-S](https://img.shields.io/badge/Hardware-CERN%20OHL--S-orange.svg)](LICENSE_HARDWARE)

<img src="docs/assets/logo.png" width="200" alt="Trajecto Logo">

**Trajecto** is a centimeter-level 3D handwriting reconstruction system that fuses **Deep Learning (TCN)** with **Physics-Based Filtering (ESKF)**. It uses a single low-cost 6-axis IMU (BMI270) to track pen trajectories in real-time on an **ESP32-C3** microcontroller.

---

## 🚧 Project Status: Active Development
This project is **currently under intensive development**. We are actively refining the hybrid neuro-physical models, optimizing embedded inference, and expanding the ground-truth dataset.

📄 **[Detailed Technical Specification (PDF)](docs/technical_spec.pdf)**
*(Note: Documentation is currently incomplete and in active progress.)*

---

## 🚀 Future Milestones & Ongoing Research

While **Trajecto** has achieved significant precision, we are exploring the following frontiers to push the boundaries of 3D reconstruction:

* **Hardware-In-The-Loop (HITL) HPO:** Developing an automated pipeline for tuning the ESKF-TCN hybrid parameters directly on the ESP32-C3 hardware to maximize on-device efficiency.
* **Comprehensive Documentation:** Expanding the `technical_spec.pdf` with detailed hardware schematics and signal processing flowcharts to bridge the gap between abstract math and physical implementation.

---

### 📈 Performance Metrics

The $7.7\text{mm}$ precision is not a static value but a result of sustained stability over dynamic movements.

* **Test Duration:** 5-second continuous trajectory segments.
* **Metric:** Root Mean Square Error (RMSE) between TCN-ESKF estimate and iPad Pro Ground Truth.
* **Result:** Current SOTA achieved at **$7.7\text{mm}$** within the 5s window, effectively suppressing the inherent cubic drift of the IMU.

### ⚖️ Long-term Scale Stability & Consistency

One of the most critical challenges in IMU-based reconstruction is **Scale Drift**. Trajecto overcomes this by maintaining a near-perfect scale factor across extensive datasets.

* **Dataset Scale:** Evaluated over **40 independent sequences**.
* **Average Sequence Length:** ~15 seconds per trial.
* **Scale Consistency:** Maintained **Scale Standard Deviation ($SD_{scale}$) < 1.0**.

> **Insight:** The fact that the scale standard deviation remains below 1.0 across 40 distinct 15-second trials proves the effectiveness of the **Hybrid ESKF-TCN** in suppressing scale explosion. This empirical consistency is a direct result of the **UUB (Uniformly Ultimately Bounded)** stability proven in our theoretical analysis.


## 📐 Theoretical Grounding & Stability Proof

This project goes beyond empirical results. The **Trajecto** architecture is built upon a rigorous mathematical foundation that guarantees system stability and convergence.

We provide two versions of the mathematical proof demonstrating the **Uniform Ultimate Boundedness (UUB)** of the error states and the reduction of the **Cramér-Rao Lower Bound (CRLB)** via the Hybrid ESKF-TCN injection.

| Version | Description | Link |
| :--- | :--- | :--- |
| **Formal Proof** | Rigorous derivation in LaTeX, confirming Lyapunov stability and rank deficiency resolution. | [📄 **Read the Paper (PDF)**](./docs/Proof_of_Hybrid_Model.pdf) |
| **Handwriting Log** | The original handwritten derivation notes, capturing the initial intuition and raw logic. | [✍️ **View Original Notes**](./docs/Handwritten_Proof.pdf) |

### Key Theoretical Contributions
* **Divergence of Pure Integration:** Mathematically proves that open-loop IMU integration leads to unbounded drift due to the rank deficiency of the Fisher Information Matrix (FIM).
* **CRLB Reduction:** Demonstrates that the TCN-based pseudo-measurement injection strictly reduces the theoretical lower bound of the estimation error ($CRLB_{hybrid} \le CRLB_{base}$).
* **Lyapunov Stability:** Establishes that the system is Uniformly Ultimately Bounded (UUB) by ensuring the energy dissipation rate ($\beta$) overpowers noise entropy.

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

## 🛡️ License & Commercial Inquiry

### Open Source License
This project is licensed under the **GNU AGPLv3**.
> [!IMPORTANT]
> **Copyleft Requirement**: If you modify or run this software on a network (Server/IoT), you **MUST** release your entire source code under the same license.

### Commercial Dual-Licensing
The AGPLv3 is intended for open-source collaboration. For entities that wish to:
1. Use Trajecto in **proprietary/closed-source** products.
2. Avoid the copyleft obligations of AGPLv3.
3. License the underlying **patented technologies** (KR 10-2025-0201093 / 093).

**A separate Commercial License is required.** We offer flexible licensing terms for startups and research institutions.

📧 **Contact for Licensing:** `nemonanconcode@gmail.com`

* **Patent Pending (KR 10-2025-0201092 / 093)**: Hybrid ESKF-TCN & Hovering Signal De-normalization.

### How to Cite
If you use this architecture or code in your research, please cite it as follows:

> **Kim, E. (2026).** *Trajecto: Robust 3D Pen Tracking with Neural Lyapunov-Certified ESKF.* GitHub Repository. https://github.com/concode0/trajecto

```bibtex
@software{Kim_Trajecto_2026,
  author = {Kim, Eunkyum},
  title = {Trajecto: Robust 3D Pen Tracking with Neural Lyapunov-Certified ESKF},
  url = {[https://github.com/concode0/trajecto](https://github.com/concode0/trajecto)},
  year = {2026}
}
```

<p align="right">
  <a href="https://concode0.goatcounter.com">
    <img src="https://concode0.goatcounter.com/count?p=/README&title=Trajecto_README" alt="GoatCounter">
  </a>
</p>