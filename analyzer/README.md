# Trajecto Analyzer (Julia)

The **Trajecto Analyzer** is a high-performance offline analysis suite written in **Julia**. It provides a flexible environment for testing, visualizing, and verifying trajectory estimation algorithms.

## Architecture

The analyzer follows a layered "Plug-and-Play" architecture described in `Plan.md`:

1.  **Perception Layer (`Input Plugins`)**:
    *   Handles data ingestion from HDF5 datasets or raw CSV streams.
    *   Standardizes IMU and pressure data for the estimation core.

2.  **Estimation Core Layer (`Algorithm Interface`)**:
    *   Abstract interface for swapping prediction models.
    *   Supports: `ESKF-TCN`, `AEKF-TCN`, and `Only-TCN`.
    *   Interacts with PyTorch models (via `PythonCall`) or pure Julia implementations.

3.  **Application Layer (`Output Plugins`)**:
    *   **MakieDashboard**: Interactive 3D visualization of the reconstructed path.
    *   Analysis tools for velocity profiles, drift measurement, and error metrics.

## Getting Started

### Prerequisites
*   **Julia 1.9+**
*   **Python Environment**: The analyzer uses `PythonCall.jl` to interface with the project's Python virtual environment (for loading PyTorch models).

### Setup

1.  **Activate Project**:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

2.  **Configure Environment**:
    Ensure the `VENV_PATH` in `main.jl` points to your project's Python virtual environment (e.g., `../.venv/bin/python`).

## Usage

Run the main analysis script to process a sample and launch the dashboard:

```bash
julia main.jl
```

### Configuration (`main.jl`)

You can modify `main.jl` to select the model and sample:

```julia
# Select Model Type
const MODEL_TYPE = "eskf" # Options: "eskf", "aekf", "tcn"

# Select Sample
run_analysis("sample_001_seg0")
```

## Dashboard Controls

The **MakieDashboard** provides interactive controls:
- **3D View**: Rotate, zoom, and pan the trajectory plot.
- **Time Scrubber**: Scroll through the timeline to see instantaneous velocity and orientation.
- **Layer Toggle**: Show/hide Ground Truth vs. Prediction.

## License

This work is licensed under the **GNU General Public License v3.0**.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
