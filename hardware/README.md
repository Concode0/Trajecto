# Trajecto Hardware

This directory contains the complete hardware design assets for the Trajecto Stylus Sensor Node, a custom-engineered device optimized for high-precision 3D trajectory reconstruction. The design prioritizes signal integrity, low-latency data acquisition, and edge-computing efficiency.

## Board Visualization

| Top View | Bottom View |
| :---: | :---: |
| ![Top View](images/Trajecto-Board-Top.png) | ![Bottom View](images/Trajecto-Board-Bottom.png) |

## Technical Specifications

* **Main Processing Unit**: Espressif ESP32-C3-MINI-1-N4 (RISC-V 32-bit Core)
* **Motion Tracking**: Bosch BMI270 (6-Axis IMU)
* **Force Sensing**: Force Sensitive Resistor (FSR) Interface with Active Conditioning
* **Power Management**:
  * DC/DC Buck Converter: Silergy SY8088IAAC for stable $3.3\text{V}$ rails
  * Li-Ion Charger: Microchip MCP73831T-2ACI/OT
* **Form Factor**: Ultra-slim $13\text{mm} \times 60\text{mm}$ PCB for stylus integration

## Design Principles

### 1. Signal Integrity via Power Isolation
To achieve the high precision required for $R^2 = 0.9991$ verification, the design implements strict analog/digital isolation:

* **Isolated Analog Rail**: The 3V3_ANA rail for the IMU and FSR is isolated from digital switching noise using a $600\Omega$ ferrite bead (L2: GZ1608D601TF).
* **Multi-stage Filtering**: A dedicated decoupling network (C20: $1\mu\text{F}$, C18: $100\text{nF}$) ensures the sensors receive ultra-clean power, minimizing jitter in the high-frequency 6-DoF data.

### 2. Advanced FSR Conditioning
Instead of a simple voltage divider, this system uses an active front-end to ensure reliable pressure data:
* **Buffering & Filtering**: The MCP6002-E/MS Op-Amp (U10) provides stable signal conditioning and impedance matching for the FSR output.
* **State Control**: An NC7SB3157P6X Analog Switch (U9) manages signal routing to differentiate between hovering and active writing states.

## Directory Structure

*   `schematic/`: Circuit diagrams (PDF/EDA files).
*   `gerber/`: PCB manufacturing files (Gerber/Drill).
*   `bom/`: Bill of Materials.
*   `cpl/`: Component Placement List (Pick & Place).
*   `images/`: High-resolution board renders.

## License

Hardware designs are licensed under the **CERN Open Hardware Licence Version 2 - Strongly Reciprocal**.

[![License: CERN OHL P v2](https://img.shields.io/badge/License-CERN_OHL_P_v2-blue.svg)](https://opensource.org/licenses/cern-ohl-s)

Designed by **Eunkyum Kim**
Part of the **Trajecto Adaptive Hybrid ESKF-Stateful TCN System**