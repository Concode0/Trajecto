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
