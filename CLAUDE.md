# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Trajecto** is a high-precision 3D pen trajectory estimation system that reconstructs handwriting movements using a hybrid architecture combining Deep Learning (Temporal Convolutional Networks) with physics-based filtering (Error-State Kalman Filter). The project implements a complete **Sim2Real** pipeline from PyTorch training to optimized C++ inference on ESP32 microcontrollers.

**Core Innovation**: Fusing physics-based inertial integration (ESKF) with data-driven TCN corrections to overcome IMU drift from a single BMI270 6-axis sensor.

## Repository Structure

```
Trajecto/
├── firmware/              # ESP32 embedded C++ (ESP-IDF)
├── model/                 # PyTorch model definitions
├── utils/                 # Data acquisition & preprocessing
├── TrajectoryRecorder/    # iOS/iPadOS ground truth collection app
├── analyzer/              # Julia-based offline analysis
├── acquired_data/         # Raw sensor + iPad CSV data
├── data/                  # Processed HDF5 datasets
├── train.py              # Main training script
└── validate.py           # Validation with evo metrics
```

## Build & Run Commands

### Python Environment (Model Training)

This project uses `uv` for dependency management:

```bash
# Setup environment
uv sync
source .venv/bin/activate  # or `.venv/bin/activate` on Unix

# Train ESKF-TCN hybrid model
python train.py --model eskf_tcn --epochs 200 --lr 1e-4 --batch_size 4

# Train AEKF-TCN (Adaptive EKF variant)
python train.py --model aekf_tcn --epochs 200 --lr 1e-4

# Train TCN-only baseline
python train.py --model only_tcn --epochs 200

# Validate trained model
python validate.py --model_type eskf_tcn --model_path eskf_tcn_model.pth

# Validate physics baseline (no ML)
python validate.py --model_type pure_eskf
```

### Firmware (ESP32)

See `firmware/CLAUDE.md` for detailed ESP-IDF commands. Quick reference:

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

### iOS Ground Truth App

Open `TrajectoryRecorder/TrajectoryRecorder.xcodeproj` in Xcode and build for iPad.

## Architecture Overview

### 1. Data Flow Pipeline

```
iPad (Apple Pencil)  →  CSV Files (x, y, force, timestamp)
      ↓
ESP32 (BMI270)       →  Raw IMU @ 50Hz (accel, gyro, FSR)
      ↓
utils/acquire.py     →  Synchronization (Two-Tap Correlation)
      ↓
Preprocessing        →  Segmentation, Filtering, Alignment
      ↓
HDF5 Datasets        →  data/dataset.h5 (train), data/validation_dataset.h5
      ↓
train.py             →  PyTorch Training (ESKF-TCN)
      ↓
Model Export         →  .pth → ONNX → TFLite (INT8)
      ↓
firmware/            →  ESP32 Real-time Inference
```

### 2. Model Architecture (ESKF-TCN)

**Hybrid Closed-Loop Design**:

1. **ESKF Predict Step**: Integrates raw IMU (accel, gyro) using physics equations
   - State: [position, velocity, quaternion, accel_bias, gyro_bias] (15D error state)
   - Process Noise: Diagonal Q matrix tuned in `model/ESKF_TCN.py`

2. **TCN Inference**: Multi-head network processes rich feature sequence
   - Input Features (20D): [filter_vel_b(3), filter_innovation_b(3), accel_b(3), gyro_b(3), vel_norm(1), fsr(1), prev_vel_resid(3), prev_zupt_prob(1), heading_change(1), centripetal_acc(1)]
   - Outputs:
     - `vel_corr` (3D): Velocity correction in body frame
     - `covariance_R` (6D): Log-variance for adaptive measurement noise
     - `zupt_prob` (1D): Zero-velocity probability [0, 1]

3. **ESKF Update Steps**:
   - **TCN Velocity Correction Update**: Uses predicted `vel_corr` as pseudo-measurement
   - **ZUPT Update**: If `zupt_prob > threshold`, enforce zero velocity constraint
   - **Standard IMU Update**: Propagates uncertainty (P matrix)

4. **Integration**: Corrected nominal state used for next prediction cycle

**Key Files**:
- `model/ESKF_TCN.py`: Full hybrid model definition
- `model/TCN.py`: Multi-head TCN architecture with causal convolutions
- `model/config.py`: Hyperparameters and constants

### 3. Data Acquisition Protocol (CRITICAL)

All data collection MUST follow the **"Tap-Wait-Write-Tap"** protocol:

1. **First Tap** (Start Sync): Sharp acceleration spike to mark beginning
2. **Static Wait** (~2s): Gravity calibration + bias initialization
3. **Write**: Actual handwriting motion
4. **Stop Wait** (~1s): Brief static period
5. **Second Tap** (End Sync): Acceleration spike to mark end

This protocol enables:
- **Two-tap synchronization** between iPad (ground truth) and ESP32 (IMU) via correlation
- **Gravity alignment** for ESKF initialization
- **Static buffer** for bias estimation

**Implementation**: `utils/acquire.py:acquire_sequence()`

### 4. Preprocessing Details

**Synchronization** (`estimate_time_alignment_two_taps`):
- Correlates normalized acceleration signal with iPad force signal
- Finds lag at start/end taps to calculate clock drift (slope) and offset (intercept)
- Resamples IMU data to align with 50Hz iPad timestamps

**Segmentation**:
- Uses force threshold to detect writing regions
- ROI (Region of Interest) defined between taps with `ROI_MARGIN_S` buffer
- Includes `STATIC_BUFFER_S` (2s) of static data before writing for ESKF initialization

**Coordinate Frames**:
- iPad: (x, y) in pixels → converted to meters via `PIXEL_TO_METER = 0.0254/264.0` (iPad Retina: 264 PPI)
- Z-axis: Estimated from Apple Pencil hover distance using power law `z = 12.49 * hoverDistance^0.78`
- IMU: Body frame (accelerometer/gyro axes), rotated to align with writing surface normal

### 5. Training Pipeline

**Loss Function** (`UncertaintyLoss` in `train.py`):

Multi-task learning with automatic uncertainty weighting (Kendall et al.):
- `loss_task = exp(-log_var) * mse_task + log_var`

Tasks:
1. **Velocity Loss** (SmoothL1): Magnitude of predicted velocity vs ground truth
2. **Cosine Similarity Loss**: Direction of velocity vector (only when moving)
3. **ZUPT Loss** (BCE): Binary cross-entropy for zero-velocity classification
4. **Covariance Loss** (NLL): Negative log-likelihood for adaptive R matrix
5. **Regularization**: L2 penalty on TCN velocity corrections

**Data Augmentation** (`model/dataset.py`):
- Yaw rotation: Random rotation around z-axis (±45°)
- Tilt error: Small Lie group perturbations (simulates grip variation)
- Gaussian noise injection on IMU channels

**Metrics** (`validate.py`):
- **APE (RMSE)**: Absolute Pose Error after Sim(3) alignment
- **Error/Distance**: Normalized by path length
- **Axis-wise RMSE**: Individual X, Y, Z error analysis

### 6. Embedded Deployment (Sim2Real)

**Export Pipeline**:

```bash
# 1. Train PyTorch model
python train.py --model eskf_tcn --epochs 200

# 2. Export to ONNX (optional, for debugging)
python utils/export_onnx.py

# 3. Convert to TFLite with INT8 quantization
python utils/convert_tflite.py
# Generates: firmware/main/tcn_model_dynamic_range_quant.tflite

# 4. Build firmware (auto-converts .tflite → C array)
cd firmware
idf.py build
```

**Firmware Architecture** (see `firmware/CLAUDE.md`):

- **Components**:
  - `trajecto_core`: ESKF + TCN wrapper (C++ ports of Python models)
  - `trajecto_protocol`: BLE GATT service definitions
  - `BMI270_SensorAPI`: Sensor driver

- **Main Loop** (`firmware/main/main.cpp`):
  1. Read IMU @ 50Hz (interrupt-driven)
  2. Build feature vector (20D)
  3. TCN inference via TFLite Micro
  4. ESKF predict + TCN update + ZUPT update
  5. Stream via BLE (position, velocity, quaternion)

- **Critical Optimizations**:
  - INT8 quantization: 4x memory reduction, ~2x speedup
  - Stateful buffering: Maintains TCN causal history across inference calls
  - Fast math LUTs: Pre-computed sin/cos tables for quaternion ops

## Configuration & Hyperparameters

All constants centralized in `model/config.py`:

**Core Parameters**:
- `DT = 0.02` (50 Hz sampling rate)
- `SUBSAMPLE_STEP = 1` (no subsampling)
- `MAX_SEQUENCE_LENGTH = 1750` (35 seconds @ 50Hz)

**ESKF-TCN**:
- `TCN_INPUT_SIZE = 20` (feature dimension)
- `TCN_CHANNELS = [64, 64, 64, 64]` (4 layers)
- `KERNEL_SIZE = 3`
- `USE_ZUPT = True`
- `USE_TCN_ZUPT = True` (use TCN's zupt prediction vs threshold-based)

**Loss Weights** (learnable via uncertainty):
- `REG_WEIGHT_ESKF_TCN = 1e-4` (fixed L2 regularization)

**Data Augmentation**:
- `AUGMENT_MULTIPLIER = 10`
- `YAW_ANGLE = (-0.78, 0.78)` rad (±45°)
- `SIGMA_TILT = 0.03` rad (~1.7°)

## Common Development Patterns

### Adding New Features to TCN

1. Modify feature construction in `model/ESKF_TCN.py:forward()`
2. Update `TCN_INPUT_SIZE` in `model/config.py`
3. Adjust `input_bn` in `model/TCN.py` (GroupNorm num_groups)
4. Retrain model
5. Port feature logic to `firmware/components/trajecto_core/trajecto_system.hpp`

### Tuning ESKF Process Noise

Edit `model/ESKF_TCN.py:predict()`:
```python
Q = torch.diag(torch.tensor([
    0.01, 0.01, 0.01,  # Position noise
    0.1, 0.1, 0.1,     # Velocity noise
    0.01, 0.01, 0.01,  # Orientation noise
    1e-4, 1e-4, 1e-4,  # Accel bias drift
    1e-5, 1e-5, 1e-5   # Gyro bias drift
]))
```

### Debugging Synchronization Issues

1. Set `do_augment=False` in dataset to preserve raw alignment
2. Use `utils/acquire.py --reprocess` to re-sync existing raw data
3. Check visualization plot (acceleration vs force correlation)
4. Verify tap peaks occur at expected times (~1s and end-1s)

### Analyzing Training Loss Components

Check `plots/loss_history_eskf_tcn.png` after training:
- High `vel` loss → Increase TCN capacity or check feature scaling
- High `zupt` loss → Adjust ZUPT threshold or use `USE_TCN_ZUPT`
- High `cov` loss → TCN's uncertainty calibration may be off (check NLL term)
- High `reg` loss → Velocity corrections too large (potential overfitting)

## Unit Conversion Policy

**CRITICAL**: The pipeline uses **SI units** throughout:

**Firmware → Python Flow**:
```
Firmware IMU Driver:    g, rad/s
         ↓
Firmware BLE Packet:    m/s², rad/s  (×9.81 for accel)
         ↓
Python receive.py:      m/s², rad/s  (RawImuPacket)
         ↓
Python acquire.py:      m/s², rad/s  (pen_buffer, no conversion!)
         ↓
Preprocessing:          m/s², rad/s  (df_sensor, no conversion!)
         ↓
HDF5 Dataset:          m/s², rad/s  (sensor_data)
         ↓
Training:              normalized   (z-score from m/s², rad/s)
```

**Rule**: Never convert units between stages. Data flows as **m/s²** and **rad/s** from firmware to training.

**Firmware CSV Logging** (UART): Uses `g` and `rad/s` (different from BLE!) for human readability.

## Data Files & Formats

**HDF5 Structure** (`data/dataset.h5`):
```
sample_001_seg0/
  ├── sensor_data: [seq_len, 7] (accel_xyz, gyro_xyz, fsr)
  ├── gt_pos_data: [seq_len, 3] (meters, world frame)
  ├── gt_vel_data: [seq_len, 3] (m/s, world frame)
  └── attrs:
      ├── sequence_length: int
      └── original_label: str
```

**Scaler Stats** (`data/scaler_stats.h5`):
- `mean`: [7,] array for z-score normalization
- `std`: [7,] array

**Raw Data** (`acquired_data/raw_acquired_data.h5`):
- Stores unprocessed pen + iPad CSV for reproducibility
- Structure: `raw_data/sample_NNN/{pen_data, gt_data, attrs}`

## iOS Ground Truth App (TrajectoryRecorder)

**Purpose**: Collect synchronized ground truth trajectories using Apple Pencil on iPad.

**Key Files**:
- `ContentView.swift`: Main UI with record/save controls
- `EnhancedCanvasView.swift`: PKCanvas wrapper capturing pencil data @ 240Hz
- `ApplePencilProExtensions.swift`: Extracts force, azimuth, altitude
- `DataModels.swift`: CSV export logic

**Output Format** (Sample_N.csv):
```csv
timestamp,x,y,force,hoverDistance,azimuthAngle,altitudeAngle
1234567890.123,512.5,768.2,0.85,5.2,0.0,1.047
...
```

**Usage**:
1. Launch app on iPad
2. Tap "Start" → Draw on canvas
3. Tap "Save & Next" → Exports to Files app
4. Transfer CSV to `acquired_data/` folder
5. Run `python utils/acquire.py` to sync with ESP32 IMU data

## Testing & Validation

**Unit Tests**: None currently (opportunity for contribution)

**Validation Script**:
```bash
python validate.py --model_type eskf_tcn --model_path eskf_tcn_model.pth
```

Outputs:
- `results/validation/validation_results.txt`: Summary metrics
- Terminal: Mean APE, Error/Distance, Axis RMSE

**Baselines**:
- `pure_eskf`: Physics-only (no ML correction)
- `only_tcn`: Direct position regression (no physics)

## Known Issues & Limitations

1. **BMI270 Sensitivity Drift**: Requires CRT (Component Retrim) calibration at first boot to correct 17m position error. See `firmware/main/main.cpp:ensure_calibration()`.

2. **Clock Drift**: iPad and ESP32 clocks drift ~0.1-0.5% over 30s. Two-tap sync compensates via linear interpolation.

3. **Z-Axis Ground Truth**: Apple Pencil hover distance is noisy and has limited range (~15mm). Z estimates are less accurate than X/Y.

4. **TCN Receptive Field**: 4-layer TCN with kernel=3, dilations=[1,2,4,8] → receptive field = 22 steps (0.44s). Cannot model long-term dependencies beyond this window.

5. **Quantization Accuracy**: INT8 quantization introduces ~1-2% error vs FP32. Validate on-device performance matches PyTorch.

## Analyzer (Julia)

**Purpose**: High-performance offline analysis using symbolic regression and CRLB (Cramer-Rao Lower Bound) estimation.

**Setup**:
```bash
cd analyzer
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia main.jl
```

**Key Modules**:
- `TrajectoLab.jl`: Main analysis framework
- `Analysis/Metrics.jl`: Custom trajectory metrics
- `Analysis/CRLB.jl`: Information-theoretic bounds

## Utilities

**acquire.py**: Unified data acquisition + preprocessing
- Interactive mode: `python utils/acquire.py`
- Reprocess all raw: `python utils/acquire.py --reprocess`

**receive.py**: BLE/Serial receiver (standalone passive logging)

**convert_tflite.py**: PyTorch → TFLite pipeline with INT8 quantization

**data_visualizer.py**: Plot trajectories from HDF5 (2D/3D)

**h5_viewer.py**: Interactive TUI for browsing datasets

## Performance Targets

**PyTorch (MacBook M1)**:
- Training: ~2min/epoch (200 samples, batch=4)
- Inference: <1ms/sequence (RTX 3090)

**ESP32S3**:
- TCN Inference: ~20ms @ 240MHz
- ESKF Update: <5ms
- Total Latency: <30ms (50Hz real-time capable)

**Accuracy** (Validation Set):
- ESKF-TCN: APE RMSE ~0.8-1.2 cm
- Pure ESKF: APE RMSE ~3-5 cm
- Only TCN: APE RMSE ~2-4 cm

## License

GNU General Public License v3.0 - See LICENSE file.
