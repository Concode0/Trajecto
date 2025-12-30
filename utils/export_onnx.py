"""PyTorch ESKF-TCN Model Export to ONNX for Trajecto Firmware Deployment.

This script exports the trained PyTorch ESKF-TCN model to ONNX format with
stateful TCN buffer management for real-time inference on ESP32. The export
process handles:

1. **Stateful Buffer Export**: TCN layers require causal history buffers to
   maintain temporal context across single-timestep inference calls. This script
   wraps the TCN in StatefulTCNExport to expose buffers as explicit inputs/outputs.

2. **Parameter Export**: Generates C++ header file (model_params.hpp) containing:
   - TCN architecture constants (input size, layer count)
   - State buffer dimensions for each layer
   - IMU normalization parameters (mean/std from scaler_stats.h5)
   - Pen tip offset in body frame

3. **ONNX Graph Construction**: Creates ONNX graph with:
   - Input: [Batch=1, Seq=1, Features] current timestep features
   - State Inputs: [Batch=1, History, Channels] per-layer history buffers
   - Outputs: velocity correction, covariance R, ZUPT probability
   - State Outputs: Updated history buffers for next timestep

The exported ONNX model serves as input to convert_tflite.py for quantization
and deployment to ESP32 via TFLite Micro.

Dependencies:
    - torch: PyTorch model loading and ONNX export
    - onnx: ONNX format (implicit via torch.onnx)
    - h5py: Loading scaler statistics from HDF5
    - numpy: Array manipulation

Usage:
    python utils/export_onnx.py

    Or import and call:
    from utils.export_onnx import export_onnx
    export_onnx(model_path="custom_model.pth", output_dir="custom_export")

Prerequisites:
    - Trained model file (default: eskf_tcn_model.pth)
    - Scaler statistics (data/scaler_stats.h5) from preprocessing
    - Model configuration in model/config.py

Output:
    - onnx_export/tcn_model.onnx: ONNX graph with stateful buffers
    - onnx_export/model_params.hpp: C++ header with constants

Architecture Notes:
    The stateful export is critical for embedded real-time inference:
    - Desktop inference: Process full sequences [Batch, SeqLen, Feat] in one call
    - Embedded inference: Process single timesteps [1, 1, Feat] with persistent buffers
    - Buffer management: Firmware maintains state between calls (ring buffer pattern)

See Also:
    - model/stateful_tcn.py: StatefulTCNExport wrapper implementation
    - firmware/components/trajecto_core/tcn_wrapper.cpp: C++ buffer management
    - utils/convert_tflite.py: Next step in deployment pipeline
"""

import torch
import torch.nn as nn
import sys
import os
import json
import numpy as np
import h5py

# Determine project root dynamically
# Assuming this script is in `utils/` and project root is one level up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to sys.path to ensure modules like 'model.config' are found
sys.path.insert(0, PROJECT_ROOT)

from model.ESKF_TCN import ESKFTCN_model
from model.stateful_tcn import StatefulTCNExport
from model.config import Config

def export_onnx(model_path="eskf_tcn_model.pth", output_dir="onnx_export"):
    """Exports trained ESKF-TCN model to ONNX with stateful buffers and C++ parameters.

    This function performs a complete export pipeline for embedded deployment:

    **Step 1: Model Loading**
    - Initializes ESKF-TCN model architecture from config
    - Loads trained weights from .pth file
    - Falls back to random weights if file missing (with warning)

    **Step 2: Stateful TCN Wrapping**
    - Wraps TCN layers in StatefulTCNExport for explicit buffer management
    - Calculates state buffer dimensions based on kernel size and dilation:
      - For layer i with kernel k, dilation d: history_length = (k-1) * d
      - Supports both standard and depthwise separable convolutions
    - Creates dummy inputs for ONNX tracing (input + per-layer state buffers)

    **Step 3: ONNX Export**
    - Exports to ONNX opset 13 with constant folding enabled
    - Input/Output naming convention:
      - Inputs: 'input_feature', 'state_in_0', 'state_in_1', ...
      - Outputs: 'vel_corr', 'cov_R', 'zupt_prob', 'state_out_0', 'state_out_1', ...
    - State tensors use NLC (Batch, History, Channels) layout

    **Step 4: C++ Header Generation**
    - Exports model_params.hpp with compile-time constants:
      - TCN_INPUT_SIZE: Feature dimension (typically 20)
      - TCN_NUM_LAYERS: Number of TCN layers (typically 4)
      - TCN_STATE_DIMS[]: Array of {channels, history} per layer
      - IMU_MEAN[], IMU_STD[]: Normalization parameters (7D: accel, gyro, force)
      - PEN_TIP_OFFSET[]: Offset in body frame (3D: x, y, z)

    Args:
        model_path: Path to trained PyTorch model (.pth) relative to project root.
            Default: "eskf_tcn_model.pth"
        output_dir: Directory for ONNX and header files, relative to project root.
            Default: "onnx_export"

    Raises:
        FileNotFoundError: If scaler_stats.h5 is missing (loads defaults with warning)
        RuntimeError: If ONNX export fails during torch.onnx.export()

    Side Effects:
        Creates/overwrites files:
        - {output_dir}/tcn_model.onnx
        - {output_dir}/model_params.hpp
        Prints verbose status messages during export process

    Example:
        >>> # Export with default paths
        >>> export_onnx()

        >>> # Export custom model
        >>> export_onnx(
        ...     model_path="experiments/best_model.pth",
        ...     output_dir="deployment"
        ... )

    Implementation Notes:
        - State buffer layout is critical for firmware compatibility
        - Separable convolutions have different channel progression than standard
        - ONNX constant folding reduces model size by pre-computing static ops
        - Header file must be #include'd in firmware before TFLite conversion

    See Also:
        - model/stateful_tcn.py: StatefulTCNExport wrapper class
        - firmware/components/trajecto_core/include/model_params.hpp: Template
        - utils/convert_tflite.py: Next step (ONNX → TFLite)
    """
    # Resolve paths relative to PROJECT_ROOT
    model_path_abs = os.path.join(PROJECT_ROOT, model_path)
    output_dir_abs = os.path.join(PROJECT_ROOT, output_dir)
    scaler_stats_path_abs = os.path.join(PROJECT_ROOT, Config.SCALER_STATS_H5_PATH)

    os.makedirs(output_dir_abs, exist_ok=True)

    device = "cpu"

    # 1. Initialize Model
    print("Initializing ESKFTCN_model...")
    # Pass separable argument explicitly from config
    model = ESKFTCN_model(device=device, dt=Config.DT, separable=Config.ESKFTCN.USE_SEPARABLE_CONV)
    model.eval()

    # 2. Load Weights
    if os.path.exists(model_path_abs):
        print(f"Loading weights from {model_path_abs}...")
        try:
            model.load_state_dict(torch.load(model_path_abs, map_location=device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random weights (WARNING: Model output will be garbage).")
    else:
        print(f"Warning: {model_path_abs} not found. Using random weights.")

    # 3. Prepare Stateful Export
    print("Wrapping TCN in StatefulTCNExport...")
    stateful_model = StatefulTCNExport(model.tcn)
    stateful_model.eval()
    
    input_size = model.tcn_input_size
    print(f"TCN Input Size: {input_size}")

    # 4. Construct Dummy Inputs (Input + States)
    # Main Input: [Batch=1, Seq=1, Feat=InputSize] (NLC)
    # Note: StatefulTCNExport.forward() expects x_t as NLC
    x_t = torch.randn(1, 1, input_size)
    
    state_inputs = []
    input_names = ['input_feature']
    output_names = ['vel_corr', 'cov_R', 'zupt_prob']
    
    # Iterate layers to determine state shapes
    in_ch = input_size
    state_shapes = []
    
    for i, layer in enumerate(model.tcn.tcn_layers):
        # Determine actual input channels for the layer based on separable flag
        # If separable is True, CausalConv1d has `in_channels` for depthwise and `in_channels` for pointwise (from previous layer output)
        # If separable is False, CausalConv1d has `in_channels` for standard conv.
        # This logic is correct as `layer` itself holds the `in_channels` from its constructor.

        # Access kernel_size and dilation based on whether it's separable or not
        if layer.separable and layer.depthwise is not None:
             k = layer.depthwise.kernel_size[0]
             d = layer.depthwise.dilation[0]
        else:
             k = layer.conv.kernel_size[0]
             d = layer.conv.dilation[0]

        hist_len = (k - 1) * d
        
        # State: [Batch=1, History, Channels] (NLC layout)
        state = torch.zeros(1, hist_len, in_ch)
        state_inputs.append(state)
        state_shapes.append((hist_len, in_ch))
        
        input_names.append(f"state_in_{i}")
        output_names.append(f"state_out_{i}")
        
        # Get output channels for the next layer's input
        if layer.separable and layer.pointwise is not None:
            in_ch = layer.pointwise.out_channels
        else:
            in_ch = layer.conv.out_channels


    print(f"State Shapes: {state_shapes}")

    onnx_path = os.path.join(output_dir_abs, "tcn_model.onnx")
    
    print(f"Exporting Stateful TCN to {onnx_path}...")
    torch.onnx.export(
        stateful_model,
        (x_t, *state_inputs),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        # No dynamic axes for buffers in embedded usually to keep it static/fast
    )
    print("ONNX export complete.")

    # 5. Export Parameters to C++ Header
    header_path = os.path.join(output_dir_abs, "model_params.hpp")
    print(f"Exporting parameters to {header_path}...")
    
    # Load scaler stats - use absolute path from Config
    
    # Load scaler stats
    imu_mean = np.zeros(7) # Default to 7 dimensions (accel_x,y,z, gyro_x,y,z, force)
    imu_std = np.ones(7)
    
    if os.path.exists(scaler_stats_path_abs):
        try:
            with h5py.File(scaler_stats_path_abs, "r") as f:
                imu_mean = f["mean"][:]
                imu_std = f["std"][:]
                print(f"Loaded scaler stats from {scaler_stats_path_abs}: {imu_mean.shape}")
        except Exception as e:
            print(f"Error loading scaler stats from {scaler_stats_path_abs}: {e}")
    else:
        print(f"Scaler stats file not found at {scaler_stats_path_abs}. Using defaults.")


    pen_offset = model.pen_tip_offset_b.detach().cpu().numpy()
    
    with open(header_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("// Auto-generated by export_onnx.py\n")
        f.write("#include <vector>\n\n")
        f.write("namespace trajecto {\n\n")
        
        f.write(f"    constexpr int TCN_INPUT_SIZE = {input_size};\n")
        f.write(f"    constexpr int TCN_NUM_LAYERS = {len(state_inputs)};\n")
        f.write(f"    constexpr float DT = {Config.DT};\n\n")
        
        f.write("    // State Buffer Dimensions {Channels, HistoryLength}\n")
        f.write("    struct StateDim { int channels; int history; };\n")
        f.write("    constexpr StateDim TCN_STATE_DIMS[] = {\n")
        for s in state_shapes:
            # s is (History, Channels) -> we write {channels, history}
            f.write(f"        {{ {s[1]}, {s[0]} }},\n")
        f.write("    };\n\n")

        # IMU Normalization Params
        f.write("    // IMU Normalization Parameters (AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, Force)\n")
        f.write("    constexpr float IMU_MEAN[] = {")
        f.write(", ".join([f"{x:.8f}f" for x in imu_mean]))
        f.write("};\n")
        
        f.write("    constexpr float IMU_STD[] = {")
        f.write(", ".join([f"{x:.8f}f" for x in imu_std]))
        f.write("};\n\n")
        
        # Pen Tip Offset
        f.write("    // Pen Tip Offset in Body Frame (x, y, z)\n")
        f.write("    constexpr float PEN_TIP_OFFSET[] = {")
        f.write(", ".join([f"{x:.8f}f" for x in pen_offset]))
        f.write("};\n\n")
        
        f.write("} // namespace trajecto\n")
    
    print("Export finished.")

if __name__ == "__main__":
    export_onnx()
