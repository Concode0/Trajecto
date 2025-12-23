# Trajecto Embedded Implementation Plan (ESP32)

This document outlines the plan to port the Trajecto ESKF-TCN closed-loop system to the ESP32 S3 (TrajectoFW).

## 1. System Architecture

The embedded system mimics the Python `ESKFTCN_model` architecture, split into two main components running on the ESP32:

1.  **Physics Engine (ESKF)**: A high-frequency (e.g., 400Hz) C++ implementation of the Error-State Kalman Filter. It integrates IMU data to estimate Position, Velocity, and Orientation.
2.  **AI Engine (Stateful TCN)**: A neural network inference engine (TFLite Micro) that runs at 50Hz.
    *   **Optimization 1**: **Stateful Causal TCN**. Uses persistent buffers for dilated convolutions, reducing complexity to O(1) per step.
    *   **Optimization 2**: **INT8 Quantization**. Uses integer arithmetic for inference.
    *   **Optimization 3**: **Depthwise Separable Convolutions**. Reduces parameters and MACs by splitting standard convolutions.

## 2. Dependencies

To implement this on the ESP32, we need the following libraries in `TrajectoFW`:

1.  **Eigen**: For high-performance matrix operations (Kalman Filter).
2.  **TFLite Micro (esp-tflite-micro)**: For running the TCN model.

## 3. Implementation Steps

### Step 1: Model Export & Conversion

1.  **Export ONNX**:
    Run `python3 utils/export_onnx.py`. This generates `onnx_export/tcn_model.onnx` with stateful inputs/outputs and `model_params.hpp`.
    *   **Note**: The Python model now defaults to `separable=True` for Depthwise Separable convolutions. You must retrain the model with this architecture before exporting for optimal performance.

2.  **Convert to INT8 TFLite**:
    Run `python3 utils/convert_tflite.py`.
    *   **Requires**: `onnx`, `onnx2tf`, `tensorflow`, `tf-keras`, `onnx-graphsurgeon`.
    *   **Action**: Converts ONNX to TensorFlow SavedModel using `onnx2tf` (with `--not_use_onnxsim` flag due to `onnxsim` instability).
    *   **Outcome**:
        *   **Float32 TFLite**: Successfully generated `tcn_model_float32.tflite`.
        *   **Dynamic Range Quantized TFLite**: Successfully generated `tcn_model_dynamic_range_quant.tflite`.
        *   **Full Integer Quantized TFLite**: Failed due to persistent `concatenation` dimension mismatch errors (e.g., `(20 != 64) Node number 0 (CONCATENATION) failed to prepare`), even after various attempts to modify converter settings (e.g., `supported_ops`, `inference_input/output_type`). This indicates a deep-seated incompatibility or structural issue with the model and TFLite's full integer quantization process at this time.
    *   **Recommendation**: Due to the persistent issues with full integer quantization, it is currently recommended to proceed with the **Dynamic Range Quantized TFLite model** (`tcn_model_dynamic_range_quant.tflite`). While not fully integer, it offers a good balance of size reduction and performance improvement compared to Float32, without the current quantization graph issues.
    *   **Regarding more data for embedded training**: Acquiring more data for model training would generally improve model accuracy and robustness. However, it is unlikely to directly resolve the observed `concatenation` error during TFLite full integer quantization, as this appears to be a structural conversion problem rather than a data distribution issue. More data might be beneficial if future attempts involve quantization-aware training (QAT) or if model architecture changes are considered to make it more quantization-friendly.

3.  **Generate C Array**:
    Run `xxd -i TrajectoFW/main/tcn_model.tflite > TrajectoFW/main/tcn_model.cc`.
    *   **IMPORTANT**: Ensure the generated `tcn_model.cc` defines the array as `const unsigned char tcn_model_tflite[]` and `const unsigned int tcn_model_tflite_len`. The default `xxd` behavior matches this if the file is named `tcn_model.tflite`.
    *   **Alignment**: TFLite Micro requires 16-byte alignment. You may need to manually add `alignas(16)` (C++) or `__attribute__((aligned(16)))` (GCC) to the array definition in `tcn_model.cc` if you experience crashes. `xxd` does *not* do this automatically.

### Step 2: C++ ESKF Port

Implemented in `eskf.hpp`/`eskf.cpp`. Use `Eigen::Matrix` for all linear algebra.

### Step 3: Stateful TCN Wrapper

Implemented in `tcn_wrapper.hpp`/`tcn_wrapper.cpp`.
*   Allocates persistent buffers for `state_in` tensors.
*   Enables `AddDepthwiseConv2D` in the TFLite resolver.
*   In `process_step`:
    1. Copy current feature to Input 0.
    2. Copy persistent buffers to Inputs 1..N.
    3. Invoke TFLite (Integer inference).
    4. Read Outputs 0..2 (Predictions).
    5. Copy Outputs 3..M (New States) back to persistent buffers.

### Step 4: Integration Loop (Main Entry Point)

The ESKF-TCN integration logic is now located in `TrajectoFW/main/main.cpp`. This file contains the `app_main` function which instantiates `TrajectoSystem`, handles IMU data acquisition, decimation, and calls `sys.step()`.

The original data acquisition code (simple BLE logging) has been preserved in `TrajectoFW/main/data_acquire.cpp` for reference. To use it instead, you must modify `CMakeLists.txt`.

## 4. Build Instructions

1.  **Environment**: Ensure ESP-IDF is set up.
2.  **Dependencies**: `idf_component.yml` is configured.
3.  **Build**:
    ```bash
    cd TrajectoFW
    idf.py build flash monitor
    ```

**Important**: Because the model architecture changed (Depthwise Separable), you **MUST** retrain the model in Python before exporting it for the final application. The current `export_onnx.py` uses random weights if no checkpoint is found, which is fine for testing the pipeline but useless for actual tracking.

## 5. Guidelines for Embedded Porting

When maintaining or extending this embedded port, follow these critical guidelines to ensure performance and correctness.

### 5.1. Consistency is King
*   **Feature Extraction**: The feature vector logic in C++ (`tcn_wrapper.cpp`) MUST be mathematically identical to the Python implementation (`model/base_hybrid_model.py`).
    *   *Check*: Normalization constants (`IMU_MEAN`, `IMU_STD`) must be synced via `model_params.hpp`.
    *   *Check*: Coordinate frame rotations (World-to-Body for gravity) must match.
    *   *Check*: Tanh squashing must be applied to the same fields.
*   **Stateful Logic**: The TFLite export relies on strict ordering of state inputs/outputs.
    *   *Rule*: Never manually reorder inputs in `export_onnx.py` without updating `tcn_wrapper.cpp`.
    *   *Tip*: Use `StateDim` in `model_params.hpp` to verify buffer sizes at runtime.

### 5.2. Memory Management
*   **Tensor Arena**: TFLite Micro requires a static `tensor_arena`.
    *   *Sizing*: Start with a generous size (e.g., 60KB). If `AllocateTensors()` fails, increase it. If it succeeds, check the logs for "Arena used bytes" and reduce it to save RAM.
    *   *Placement*: On ESP32-S3, place the arena in internal SRAM (default) for speed. If the model is huge, use PSRAM (`heap_caps_malloc(size, MALLOC_CAP_SPIRAM)`), but expect higher latency.
*   **Stack Usage**:
    *   Eigen objects can be large. Avoid allocating large matrices on the stack in recursive functions.
    *   Increase the FreeRTOS task stack size (default 4KB is often too small for TFLite+Eigen). We use **8KB** or **16KB**.

### 5.3. Performance Optimization
*   **Quantization**: Always use **INT8** models for the ESP32-S3. The ESP-NN library accelerates integer vector instructions.
*   **Op Resolver**: Use `MicroMutableOpResolver` instead of `AllOpsResolver`. Only add the specific operations your model uses (e.g., `AddConv2D`, `AddReshape`, `AddFullyConnected`). This saves ~200KB of flash.
*   **Clock Speed**: Ensure the ESP32 is running at 240MHz (`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`).

## 6. Testing & Debugging

Given the complexity of the hybrid physics+AI system, debugging on hardware can be challenging. Use these methods to isolate issues.

### 6.1. Golden Vector Test (Unit Test)
This verifies that the TCN inference on the ESP32 matches the Python model, ruling out quantization or conversion errors.

1.  **Generate Golden Vector (Python)**:
    *   Create a script `utils/generate_golden_vector.py`.
    *   Instantiate `ESKFTCN_model`.
    *   Run one forward pass with a fixed input (e.g., all ones or a specific pattern).
    *   Print the input feature vector (20 floats) and the output predictions (vel_corr, zupt_prob).
2.  **Verify on ESP32 (C++)**:
    *   In `tcn_wrapper.cpp`, temporarily replace `extract_features` output with the hardcoded input vector from Python.
    *   Print the TFLite output to the console.
    *   **Pass Criteria**: Outputs should match within ~1-5% error (due to INT8 quantization). If completely different, check input ordering or normalization constants.

### 6.2. BLE Live Debugging
This allows real-time visualization of the system's internal state without slowing down the control loop.

1.  **Use `h5_viewer.py` or a BLE Plotter**:
    *   The firmware sends `TrajectoryData` packets (Time, Pos, Vel, Quat, FSR).
    *   Connect via BLE and plot `Vel[0], Vel[1], Vel[2]` in real-time.
2.  **Stationary Test**:
    *   Place the sensor flat on a table.
    *   **Expected**: Velocity should be near zero. Position should be constant. ZUPT flag (if exposed) should be active.
    *   **Failure**: If velocity drifts linearly, ZUPT is not triggering. If velocity is erratic, TCN might be outputting noise.
3.  **Tap Test**:
    *   Tap the sensor.
    *   **Expected**: Spike in acceleration, momentary velocity change, then return to zero (ZUPT).
    *   **Failure**: If velocity explodes after a tap, the ESKF integration might be unstable (check `dt`) or TCN corrections are too aggressive.

### 6.3. Serial Logging (Post-Mortem)
For deep analysis when BLE bandwidth is insufficient.

1.  **Circular Buffer Logging**:
    *   Do not print in the 400Hz/50Hz loop! It will cause timing jitter.
    *   Write debug values (e.g., innovation, raw TCN output) to a RAM buffer.
    *   Dump the buffer to UART only when the buffer is full or triggered by a button press.
2.  **Analyze**:
    *   Copy the dumped CSV data.
    *   Plot in Python to see internal variables that aren't sent over BLE.

## 7. Critical Checklist (MUST DO)

Before deploying the firmware for actual usage, complete this checklist:

*   [ ] **Retrain Model**: Run `python3 train.py` to generate a new `eskf_tcn_model.pth` with the Depthwise Separable architecture.
*   [ ] **Update Stats**: Ensure `data/scaler_stats.h5` reflects your training data distribution.
*   [ ] **Export & Convert**: Run `python3 utils/export_onnx.py` then `python3 utils/convert_tflite.py` to generate the new `tcn_model.tflite`.
*   [ ] **Generate C Array**: Run `xxd -i TrajectoFW/main/tcn_model.tflite > TrajectoFW/main/tcn_model.cc`.
*   [ ] **Align Model**: **CRITICAL**: Edit `TrajectoFW/main/tcn_model.cc` manually to add `alignas(16)` to the `tcn_model_tflite` array definition. Without this, TFLite Micro may crash.
    *   *Example*: `alignas(16) const unsigned char tcn_model_tflite[] = { ... };`
*   [ ] **Flash**: Build and flash the firmware to the ESP32.