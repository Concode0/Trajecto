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
초기 모델은 AEKF(Adaptive EKF)를 사용했으나, 3차원 필기의 급격한 회전 운동에서 발생하는 쿼터니언 정규화 문제와 수렴 불안정성을 해결하기 위해 **ESKF(Error-State Kalman Filter)**로 고도화했습니다.

* **Nominal State:** 빠른 주파수의 IMU 데이터를 적분하여 '명목 궤적' 생성.
* **Error State:** 느린 주파수의 오차(Bias, Drift)를 선형화하여 추정.
* **Advantage:** 짐벌 락(Gimbal Lock) 방지 및 연산 효율성 극대화.

### 2. Deep Learning Correction (TCN)
ESKF가 추정한 궤적의 잔여 오차(Residual)를 보정합니다.
* **Input Feature Optimization:** 단순 센서 값이 아닌, ESKF 잔차 및 회전 불변량(Rotation Invariants)을 입력으로 가공하여 학습 효율 증대.
* **Batch Aware Optimization:** 가변적인 필기 획 길이를 고려한 배치 구성을 통해 학습 시 시간적 맥락(Temporal Context) 손실 최소화.

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