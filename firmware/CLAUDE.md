# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Trajecto Firmware** for ESP32-based trajectory tracking using a hybrid ESKF-TCN (Error-State Kalman Filter + Temporal Convolutional Network) pipeline. The system runs real-time inference on-device, fusing IMU data with a quantized (INT8) deep learning model to predict 6-DOF trajectory.

**Target Hardware**: ESP32C3 or ESP32S3 (requires AI instruction support for TFLite Micro)

## Build Commands

This project uses **ESP-IDF v5.0+** (Espressif IoT Development Framework).

### Set Target
```bash
idf.py set-target esp32s3
```

### Build
```bash
idf.py build
```

### Flash and Monitor
```bash
idf.py flash monitor
```

### Flash Only
```bash
idf.py flash
```

### Monitor Only
```bash
idf.py monitor
```

### Clean Build
```bash
idf.py fullclean
idf.py build
```

## Architecture

### Layered Structure

1. **Hardware Layer** (`main/main.cpp`)
   - **IMU**: Bosch BMI270 sensor via I2C @ 400kHz (accelerometer + gyroscope)
   - **FSR**: Force-sensitive resistor for zero-velocity (ZUPT) detection
   - **BLE**: NimBLE stack with custom GATT service for data streaming

2. **Core Components** (`components/trajecto_core/`)
   - **ESKF** (`eskf.hpp/.cpp`): Error-State Kalman Filter for 6-DOF state estimation
   - **TCN Wrapper** (`tcn_wrapper.hpp/.cpp`): TFLite Micro interpreter for velocity residual prediction
   - **Trajecto System** (`trajecto_system.hpp`): Orchestrates ESKF-TCN fusion pipeline
   - **Fast Math** (`fast_math.hpp`, `fast_math_lut.hpp/.cpp`): Optimized math operations with lookup tables

3. **Protocol** (`components/trajecto_protocol/`)
   - **BLE Protocol** (`trajecto_protocol.h`): Packet definitions for command/response and data streaming

### Data Flow

```
IMU @ 50Hz → ESKF Predict → TCN Inference →
  ├─ Velocity Correction Update
  ├─ ZUPT Update (if detected)
  └─ Standard IMU Update → State Output → BLE Notify
```

### Key Integration Points

- **main/main.cpp**: Application entry point, BLE setup, sensor initialization, main control loop
- **trajecto_system.hpp**: High-level API that ties together ESKF and TCN inference
- **eskf.cpp**: Implements prediction, ZUPT updates, TCN velocity correction updates, and standard IMU updates
- **tcn_wrapper.cpp**: Manages TFLite interpreter, input tensor preparation, and output parsing

## Important Implementation Details

### Calibration (CRT & FOC)

The BMI270 requires **Component Retrim (CRT)** and **Gyroscope Fast Offset Compensation (FOC)** to correct sensitivity drift. This is handled in `main/main.cpp:ensure_calibration()`:

- **First Boot**: Executes CRT + FOC (device must be stationary), saves offsets to NVS (Non-Volatile Storage)
- **Subsequent Boots**: Loads saved calibration from NVS
- **CRT Requirement**: Critical for correcting 17m position error caused by factory sensitivity drift

### BLE Protocol

Custom GATT service UUID: `AD43434E-C549-4594-B474-5431535445`

**Characteristics**:
- **Command (Write)**: `AD43434D-C549-4594-B474-5431535445` - Accepts `CMD_START_STREAM`, `CMD_STOP_STREAM`, `CMD_SET_CONFIG`
- **Data (Notify)**: `AD43434F-C549-4594-B474-5431535445` - Streams `DATA_RAW_IMU` or `DATA_TRAJECTORY` packets

**Streaming Modes**:
- **Raw Mode**: Streams 6-axis IMU data (accel + gyro)
- **Trajectory Mode**: Streams processed 6-DOF pose (position, velocity, quaternion) + ZUPT probability

### Model File Requirements

The TCN model must be compiled into the firmware:
- **Source**: `main/tcn_model_dynamic_range_quant.tflite` (INT8 quantized)
- **Build Integration**: Converted to `tcn_model.cc` and compiled into `trajecto_core` component
- **Generation**: Use `../utils/convert_tflite.py` to generate C array from `.tflite` file

### Component Dependencies

The project uses ESP-IDF managed components (via `idf_component.yml`):
- `espressif__eigen`: Linear algebra (Kalman filter matrix operations)
- `espressif__esp-tflite-micro`: TensorFlow Lite Micro for inference
- `espp__bmi270`: BMI270 sensor driver wrapper
- `espp__i2c`: I2C communication abstraction
- `espp__task`, `espp__logger`, `espp__format`: ESP++ utilities

## Common Development Patterns

### Adding New Sensor Features

1. Add I2C device in `main/main.cpp` using `espp::I2c` class
2. Initialize in `app_main()` before task creation
3. Read sensor data inside `task_fn` lambda (runs at IMU interrupt rate)

### Modifying ESKF Behavior

- **State Definition**: `components/trajecto_core/include/eskf.hpp` (NominalState, ErrorState)
- **Process Noise**: Tuned in `eskf.cpp:ESKF::predict()` - Q matrix diagonal values
- **Measurement Noise**: TCN outputs adaptive R parameters, see `trajecto_system.hpp:step()`

### Debugging Real-Time Performance

- **CSV Logging**: Raw IMU data is printed to UART in CSV format (line 564 in main.cpp)
- **LED Indicator**: GPIO 7 toggles when BLE packets are sent
- **ESP-IDF Monitor**: Use `idf.py monitor` to view logs with automatic decoding

### Working with TFLite Models

When updating the TCN model:
1. Train new model in `../model/TCN.py`
2. Export to `.tflite` using `../utils/convert_tflite.py`
3. Copy `.tflite` file to `main/` directory
4. Rebuild firmware (build system auto-converts to C array)
5. Verify tensor shapes match `model_params.hpp` definitions

## Pin Configuration

**ESP32S3 Default Pinout** (sdkconfig.defaults):
- **I2C SDA**: GPIO 10
- **I2C SCL**: GPIO 4
- **IMU INT**: GPIO 1
- **FSR ADC**: GPIO 3 (ADC1 Channel 3)
- **FSR Enable**: GPIO 6 (controls resistor divider range)
- **Data LED**: GPIO 7

## Performance Characteristics

- **IMU Rate**: 50 Hz (fixed, required for TCN model)
- **Inference Latency**: ~20ms per TCN forward pass
- **BLE Packet Rate**: ~50 Hz (matches sensor rate)
- **Stack Usage**: 12KB for IMU task (TFLite needs substantial stack)

## License

GNU General Public License v3.0
