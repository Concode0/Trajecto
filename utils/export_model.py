"""Unified Model Export Pipeline for Trajecto ESP32 Deployment.

This script provides a complete export pipeline from PyTorch to TFLite INT8,
with full support for Quantization-Aware Training (QAT) models.

Pipeline:
    PyTorch Model (FP32 or QAT) → ONNX → TensorFlow → TFLite INT8

Features:
    - QAT-trained model support (learns INT8-robust weights during training)
    - Stateful TCN buffer management for real-time embedded inference
    - Full INT8 quantization with representative dataset calibration
    - C++ header generation with model parameters

Usage:
    # Export with default settings (uses checkpoint best model)
    python utils/export_model.py

    # Export specific model with QAT
    python utils/export_model.py --model checkpoints/eskf_tcn_model_best.pth --qat

    # Full pipeline with custom calibration samples
    python utils/export_model.py --calib-samples 100

Output:
    - firmware/main/tcn_model.tflite: INT8 quantized model for ESP32
    - firmware/main/model_params.hpp: C++ header with constants
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.ESKF_TCN import ESKFTCN_model
from model.stateful_tcn import StatefulTCNExport
from model.config import Config
from model.qat_tcn import convert_to_quantized


# Output paths
DEFAULT_MODEL_PATH = "checkpoints/eskf_tcn_model_best.pth"
OUTPUT_DIR = "firmware/main"
ONNX_TEMP_DIR = "build/onnx_export"
SCALER_STATS_PATH = Config.SCALER_STATS_H5_PATH


def load_model(model_path: str, device: str = "cpu", is_qat: bool = False) -> nn.Module:
    """Load trained ESKF-TCN model from checkpoint.

    Args:
        model_path: Path to .pth model file
        device: Device to load model on
        is_qat: If True, convert QAT model to quantized form

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {model_path}...")

    model = ESKFTCN_model(
        device=device,
        dt=Config.DT,
        separable=Config.ESKFTCN.USE_SEPARABLE_CONV
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights successfully")
    else:
        print(f"  WARNING: Model file not found, using random weights!")

    model.eval()

    # Convert QAT model if needed
    if is_qat:
        print("  Converting QAT model to quantized form...")
        model = convert_to_quantized(model, inplace=True)

    return model


def get_state_shapes(model: nn.Module) -> List[Tuple[int, int]]:
    """Extract state buffer shapes from TCN layers.

    Returns:
        List of (history_length, channels) tuples for each layer
    """
    state_shapes = []
    in_ch = model.tcn_input_size

    for layer in model.tcn.tcn_layers:
        if layer.separable and layer.depthwise is not None:
            k = layer.depthwise.kernel_size[0]
            d = layer.depthwise.dilation[0]
        else:
            k = layer.conv.kernel_size[0]
            d = layer.conv.dilation[0]

        hist_len = (k - 1) * d
        state_shapes.append((hist_len, in_ch))

        if layer.separable and layer.pointwise is not None:
            in_ch = layer.pointwise.out_channels
        else:
            in_ch = layer.conv.out_channels

    return state_shapes


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    state_shapes: List[Tuple[int, int]]
) -> List[Dict]:
    """Export model to ONNX with stateful buffers.

    Args:
        model: PyTorch model
        output_path: Path for ONNX file
        state_shapes: List of (history, channels) for each layer

    Returns:
        List of input info dicts with 'name' and 'shape' keys
    """
    print(f"Exporting to ONNX: {output_path}")

    stateful_model = StatefulTCNExport(model.tcn)
    stateful_model.eval()

    input_size = model.tcn_input_size

    # Build inputs
    x_t = torch.randn(1, 1, input_size)
    state_inputs = []
    input_names = ['input_feature']
    output_names = ['vel_corr', 'cov_R', 'zupt_prob', 'gravity_b']

    for i, (hist_len, channels) in enumerate(state_shapes):
        state = torch.zeros(1, hist_len, channels)
        state_inputs.append(state)
        input_names.append(f"state_in_{i}")
        output_names.append(f"state_out_{i}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        stateful_model,
        (x_t, *state_inputs),
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

    # Build input info for TFLite conversion
    input_info = [{'name': 'input_feature', 'shape': (1, 1, input_size)}]
    for i, (hist_len, channels) in enumerate(state_shapes):
        input_info.append({'name': f'state_in_{i}', 'shape': (1, hist_len, channels)})

    print(f"  ONNX export complete: {len(input_names)} inputs, {len(output_names)} outputs")
    return input_info


def generate_calibration_data(
    input_info: List[Dict],
    output_dir: str,
    num_samples: int = 100,
    dataset_path: Optional[str] = None
) -> None:
    """Generate calibration data for TFLite quantization.

    Args:
        input_info: List of input metadata dicts
        output_dir: Directory to save .npy files
        num_samples: Number of calibration samples
        dataset_path: Optional path to real dataset for calibration
    """
    print(f"Generating calibration data ({num_samples} samples)...")
    os.makedirs(output_dir, exist_ok=True)

    # Try to use real data if available
    real_data = None
    if dataset_path and os.path.exists(dataset_path):
        try:
            with h5py.File(dataset_path, 'r') as f:
                keys = list(f.keys())[:num_samples]
                real_data = []
                for key in keys:
                    sensor = f[key]['sensor_data'][:]
                    real_data.append(sensor)
            print(f"  Using real data from {dataset_path}")
        except Exception as e:
            print(f"  Could not load real data: {e}")
            real_data = None

    # Load scaler stats for normalization
    scaler_path = os.path.join(PROJECT_ROOT, SCALER_STATS_PATH)
    if os.path.exists(scaler_path):
        with h5py.File(scaler_path, 'r') as f:
            imu_mean = f['mean'][:]
            imu_std = f['std'][:]
    else:
        imu_mean = np.zeros(7)
        imu_std = np.ones(7)

    for info in input_info:
        name = info['name']
        shape = info['shape']
        calib_shape = (num_samples,) + shape

        if name == 'input_feature' and real_data is not None:
            # Use real sensor data (normalized)
            data = np.zeros(calib_shape, dtype=np.float32)
            for i, sensor in enumerate(real_data[:num_samples]):
                # Normalize
                normalized = (sensor - imu_mean) / (imu_std + 1e-6)
                # Take a random timestep
                t = np.random.randint(0, len(normalized))
                data[i, 0, :7] = normalized[t, :7]
                # Fill remaining features with reasonable values
                if shape[-1] > 7:
                    data[i, 0, 7:] = np.random.randn(shape[-1] - 7) * 0.1
        else:
            # Generate synthetic data
            data = np.random.randn(*calib_shape).astype(np.float32) * 0.5

        npy_path = os.path.join(output_dir, f"{name}.npy")
        np.save(npy_path, data)

    print(f"  Saved calibration data to {output_dir}")


def convert_to_tflite(
    onnx_path: str,
    tflite_path: str,
    calib_dir: str,
    input_info: List[Dict]
) -> None:
    """Convert ONNX model to TFLite with INT8 quantization.

    Args:
        onnx_path: Path to ONNX model
        tflite_path: Output path for TFLite model
        calib_dir: Directory with calibration .npy files
        input_info: Input metadata for calibration
    """
    print("Converting to TFLite INT8...")

    # Check dependencies
    try:
        import onnx
        import tensorflow as tf
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install onnx tensorflow onnx2tf")
        return

    # Step 1: ONNX → TensorFlow SavedModel
    tf_model_dir = os.path.join(os.path.dirname(onnx_path), "tf_model")

    input_names = [info['name'] for info in input_info]
    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", tf_model_dir,
        "-osd",
        "-kat", *input_names,
        "--not_use_onnxsim",
    ]

    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: onnx2tf failed: {e.stderr.decode()}")
        return
    except FileNotFoundError:
        print("  ERROR: onnx2tf not found. Install with: pip install onnx2tf")
        return

    # Step 2: TensorFlow → TFLite with INT8 quantization
    def representative_dataset_gen():
        num_samples = len([f for f in os.listdir(calib_dir) if f.endswith('.npy')])
        num_samples = min(100, num_samples) if num_samples > 0 else 1

        for i in range(num_samples):
            tensors = []
            for info in input_info:
                npy_path = os.path.join(calib_dir, f"{info['name']}.npy")
                if os.path.exists(npy_path):
                    data = np.load(npy_path)
                    if i < len(data):
                        tensors.append(data[i:i+1].astype(np.float32))
                    else:
                        tensors.append(data[0:1].astype(np.float32))
                else:
                    tensors.append(np.random.randn(1, *info['shape'][1:]).astype(np.float32))
            yield tensors

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Use INT8 for internal ops but keep FP32 for I/O (firmware uses float tensors)
    # This provides INT8 speedup while avoiding quantization/dequantization in firmware
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # For FP32 I/O ops
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8  # For INT8 internal ops
    ]
    # Keep FP32 I/O - firmware copies float data directly to input tensors
    # (see tcn_wrapper.cpp: std::memcpy(input_feat->data.f, features.data(), ...))
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")


def generate_cpp_header(
    output_path: str,
    input_size: int,
    state_shapes: List[Tuple[int, int]],
    scaler_path: str,
    pen_offset: np.ndarray
) -> None:
    """Generate C++ header with model parameters.

    Args:
        output_path: Path for .hpp file
        input_size: TCN input feature dimension
        state_shapes: List of (history, channels) per layer
        scaler_path: Path to scaler_stats.h5
        pen_offset: Pen tip offset in body frame [3]
    """
    print(f"Generating C++ header: {output_path}")

    # Load scaler stats
    imu_mean = np.zeros(7)
    imu_std = np.ones(7)
    if os.path.exists(scaler_path):
        with h5py.File(scaler_path, 'r') as f:
            imu_mean = f['mean'][:]
            imu_std = f['std'][:]

    with open(output_path, 'w') as f:
        f.write("#pragma once\n\n")
        f.write("// Auto-generated by utils/export_model.py\n")
        f.write("// DO NOT EDIT - regenerate with: python utils/export_model.py\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("namespace trajecto {\n\n")

        # Model architecture
        f.write("// TCN Architecture\n")
        f.write(f"constexpr int TCN_INPUT_SIZE = {input_size};\n")
        f.write(f"constexpr int TCN_NUM_LAYERS = {len(state_shapes)};\n")
        f.write(f"constexpr float DT = {Config.DT:.10f}f;\n")
        f.write(f"constexpr int TCN_UPDATE_STRIDE = {Config.ESKFTCN.TCN_UPDATE_STRIDE};\n\n")

        # State buffer dimensions
        f.write("// State Buffer Dimensions {channels, history}\n")
        f.write("struct StateDim { int channels; int history; };\n")
        f.write("constexpr StateDim TCN_STATE_DIMS[] = {\n")
        for hist, ch in state_shapes:
            f.write(f"    {{{ch}, {hist}}},\n")
        f.write("};\n\n")

        # IMU normalization
        f.write("// IMU Normalization (AccelXYZ, GyroXYZ, Force)\n")
        f.write("constexpr float IMU_MEAN[] = {")
        f.write(", ".join(f"{x:.8f}f" for x in imu_mean))
        f.write("};\n")
        f.write("constexpr float IMU_STD[] = {")
        f.write(", ".join(f"{x:.8f}f" for x in imu_std))
        f.write("};\n\n")

        # Pen tip offset
        f.write("// Pen Tip Offset in Body Frame (x, y, z) meters\n")
        f.write("constexpr float PEN_TIP_OFFSET[] = {")
        f.write(", ".join(f"{x:.8f}f" for x in pen_offset))
        f.write("};\n\n")

        # Velocity normalization
        f.write("// Velocity Normalization (isotropic, from Python config)\n")
        f.write(f"constexpr float VEL_STD_L2 = {Config.VEL_STD_L2:.15f}f;\n")
        f.write(f"constexpr float VEL_CORRECTION_SCALE = {Config.VEL_CORRECTION_SCALE:.8f}f;\n\n")

        # Innovation normalization (Allan variance based)
        f.write("// Innovation Normalization (Allan variance based)\n")
        max_vrw = max(Config.VRW_X, Config.VRW_Y, Config.VRW_Z)
        max_arw = max(Config.ARW_X, Config.ARW_Y, Config.ARW_Z)
        f.write(f"constexpr float MAX_VRW = {max_vrw:.4e}f;  // max(VRW_X, VRW_Y, VRW_Z)\n")
        f.write(f"constexpr float MAX_ARW = {max_arw:.4e}f;  // max(ARW_X, ARW_Y, ARW_Z)\n")
        f.write(f"constexpr float INNOVATION_CLAMP_RANGE = {Config.INNOVATION_CLAMP_RANGE:.1f}f;\n\n")

        # Physical constants
        f.write("// Physical Constants\n")
        f.write(f"constexpr float GRAVITY_MAGNITUDE = {Config.GRAVITY_MAGNITUDE:.6f}f;\n")
        f.write(f"constexpr float GRAVITY_NORM_SCALE = {Config.GRAVITY_NORM_SCALE:.1f}f;\n\n")

        # ESKF Parameters (centralized - previously hardcoded in eskf.cpp)
        f.write("// ESKF Parameters (from Config.ESKFTCN)\n")
        f.write(f"constexpr float ZUPT_NOISE_STD = {Config.ESKFTCN.ZUPT_NOISE_STD:.6f}f;\n")
        f.write(f"constexpr float TCN_VEL_NOISE_STD = {Config.ESKFTCN.TCN_VEL_NOISE_STD:.6f}f;\n")
        f.write(f"constexpr float MAHALANOBIS_GATE_THRESHOLD = {Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD_IMU:.1f}f;\n")
        f.write(f"constexpr float ZUPT_HARD_RESET_THRESHOLD = {Config.ESKFTCN.ZUPT_HARD_RESET_THRESHOLD:.2f}f;\n\n")

        # Allan Variance noise parameters (for ESKF process noise)
        f.write("// Allan Variance Noise Parameters (for ESKF Q matrix)\n")
        f.write(f"constexpr float VRW_X = {Config.VRW_X:.4e}f, VRW_Y = {Config.VRW_Y:.4e}f, VRW_Z = {Config.VRW_Z:.4e}f;\n")
        f.write(f"constexpr float ARW_X = {Config.ARW_X:.4e}f, ARW_Y = {Config.ARW_Y:.4e}f, ARW_Z = {Config.ARW_Z:.4e}f;\n")
        f.write(f"constexpr float GYRO_BI_X = {Config.GYRO_BI_X:.4e}f, GYRO_BI_Y = {Config.GYRO_BI_Y:.4e}f, GYRO_BI_Z = {Config.GYRO_BI_Z:.4e}f;\n")
        f.write(f"constexpr float ACCEL_BI_X = {Config.ACCEL_BI_X:.4e}f, ACCEL_BI_Y = {Config.ACCEL_BI_Y:.4e}f, ACCEL_BI_Z = {Config.ACCEL_BI_Z:.4e}f;\n\n")

        f.write("} // namespace trajecto\n")

    print(f"  Header generated with {len(state_shapes)} layer configs")


def main():
    parser = argparse.ArgumentParser(
        description="Export ESKF-TCN model to TFLite INT8 for ESP32 deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/export_model.py                           # Export default model
  python utils/export_model.py --model my_model.pth      # Export specific model
  python utils/export_model.py --qat                     # Export QAT-trained model
  python utils/export_model.py --calib-samples 200       # More calibration samples
"""
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to PyTorch model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--qat", action="store_true",
                        help="Model was trained with QAT (convert to quantized form)")
    parser.add_argument("--calib-samples", type=int, default=100,
                        help="Number of calibration samples for quantization (default: 100)")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="Skip TFLite conversion (ONNX + header only)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary ONNX files")
    args = parser.parse_args()

    # Resolve paths
    model_path = os.path.join(PROJECT_ROOT, args.model)
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    temp_dir = os.path.join(PROJECT_ROOT, ONNX_TEMP_DIR)
    scaler_path = os.path.join(PROJECT_ROOT, SCALER_STATS_PATH)
    dataset_path = os.path.join(PROJECT_ROOT, Config.DATASET_H5_PATH)

    print("=" * 60)
    print("Trajecto Model Export Pipeline")
    print("=" * 60)
    print(f"Model:       {model_path}")
    print(f"Output:      {output_dir}")
    print(f"QAT Mode:    {args.qat}")
    print("=" * 60)

    # Step 1: Load model
    model = load_model(model_path, device="cpu", is_qat=args.qat)
    state_shapes = get_state_shapes(model)
    print(f"  State shapes: {state_shapes}")

    # Step 2: Export to ONNX
    onnx_path = os.path.join(temp_dir, "tcn_model.onnx")
    input_info = export_to_onnx(model, onnx_path, state_shapes)

    # Step 3: Generate calibration data
    calib_dir = os.path.join(temp_dir, "calib_data")
    generate_calibration_data(
        input_info, calib_dir,
        num_samples=args.calib_samples,
        dataset_path=dataset_path
    )

    # Step 4: Convert to TFLite
    if not args.skip_tflite:
        tflite_path = os.path.join(output_dir, "tcn_model.tflite")
        convert_to_tflite(onnx_path, tflite_path, calib_dir, input_info)

    # Step 5: Generate C++ header
    header_path = os.path.join(output_dir, "model_params.hpp")
    pen_offset = model.pen_tip_offset_b.detach().cpu().numpy()
    generate_cpp_header(
        header_path,
        model.tcn_input_size,
        state_shapes,
        scaler_path,
        pen_offset
    )

    # Cleanup
    if not args.keep_temp and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary files: {temp_dir}")

    print("=" * 60)
    print("Export complete!")
    print(f"  TFLite: {os.path.join(output_dir, 'tcn_model.tflite')}")
    print(f"  Header: {header_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
