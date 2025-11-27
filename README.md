# 🖊️ Trajecto: Robust 3D Pen Trajectory Reconstruction

> **A Robust 3D Pen Trajectory Reconstruction from a Single IMU using a Hybrid AEKF-TCN Architecture and Pressure-based Zero-Velocity Updates**

![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

## 📖 Abstract
**Trajecto** is a surface-free 3D pen interface project that allows users to write freely in mid-air.

To overcome the inherent drift problem of single IMU sensors, this study proposes an **ESKF-TCN Hybrid Architecture** that combines physical models with data-driven models. The system utilizes a pressure-sensor-based ZUPT (Zero-Velocity Update) algorithm and an ESKF (Error-State Kalman Filter) to form a robust trajectory skeleton, while a TCN (Temporal Convolutional Network) precisely corrects non-linear errors to reconstruct high-precision 3D trajectories.

---

## 🚀 Key Features

**Hybrid Architecture (ESKF + TCN)**: Combines Physical Constraints and Data-driven Patterns to achieve drastic error reduction compared to raw integration.
**Pressure-based ZUPT**: Detects pressure changes on the pen grip to precisely identify 'Zero Velocity' states and forcibly correct drift.
**Custom Hardware**: Designed a custom 4-layer PCB integrating ESP32-C3, BMI270, and high-precision Op-Amp circuits to replicate the usability of a real pen.
**Embedded Optimization**: Implemented low-power/high-efficiency firmware using TensorFlow Lite Quantization (8-bit) and the SDA algorithm.
**Virtual Plane Tracking**: Equipped with a UX algorithm that tracks changes in the pen's attitude (Quaternion) to detect the user's intent to "switch virtual planes".

---

## 🛠️ System Architecture

### 1. The Paradigm Shift: From AEKF to ESKF
While the initial model used **AEKF (Adaptive EKF)**, it was upgraded to **ESKF (Error-State Kalman Filter)** to address quaternion normalization issues and convergence instability occurring during the rapid rotational movements of 3D handwriting .

* **Nominal State:** Generates the 'nominal trajectory' by integrating high-frequency IMU data.
* **Error State:** Estimates low-frequency errors (Bias, Drift) through linearization.
* **Advantage:** Prevents Gimbal Lock and maximizes computational efficiency.

### 2. Deep Learning Correction (TCN)
Corrects the residual errors of the trajectory estimated by the ESKF
* **Input Feature Optimization:** Improves learning efficiency by processing ESKF residuals and Rotation Invariants as inputs, rather than simple sensor values.
* **Batch Aware Optimization:** Minimizes the loss of Temporal Context during training by configuring batches that account for variable stroke lengths.

---

## 💻 Hardware & Firmware

| Component | Description |
| :--- | :--- |
| **MCU** | Espressif **ESP32-C3** (RISC-V, BLE 5.0) |
| **IMU** | Bosch **BMI270** (6-DoF, Low Power) |
| **Pressure** | FSR with **Op-Amp + Analog Switch** (Dynamic Sensitivity Control) |
| **PCB** | Custom **4-Layer** Design for Signal Integrity |

* **Firmware:** FreeRTOS based Realtime Inference and Data Acquisitio
* **Optimization:** TFLite Micro (Int8 Quantization), Trajectory Compression (SDA + Hermite Interpolation).

---

## 📊 Performance Evaluation

| Model | ATE (m) | RPE (m/s) | Note |
| :--- | :---: | :---: | :--- |
| Raw Integration | High | High | Diverges rapidly |
| ZUPT Only | Medium | Medium | Corrects stationary drift only |
| **ESKF + TCN** | **Low (Best)** | **Low (Best)** | **Robust to dynamic movements** |

*(Detailed quantitative results are available in the technical report.)*

## Docs

Detailed reports, project decision-making processes, and technical explanations are documented separately in the docs/ directory.

## License

This work is licensed under a **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
</a>

You are free to:
* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material

Under the following terms:
* **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **NonCommercial** — You may not use the material for commercial purposes.
* **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/