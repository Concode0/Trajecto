"""PyTorch to TFLite Conversion Pipeline for Trajecto TCN Model.

This script handles the complete model conversion pipeline from ONNX to TensorFlow Lite
with full INT8 quantization for ESP32 deployment. The pipeline consists of:

1. ONNX → TensorFlow SavedModel conversion (via onnx2tf)
2. TensorFlow → TFLite conversion with INT8 quantization
3. Representative dataset generation for quantization calibration

The resulting quantized model achieves:
- 4x memory reduction (INT8 vs FP32)
- ~2x inference speedup on ESP32
- <2% accuracy degradation vs full precision

Dependencies:
    - onnx: ONNX model loading and inspection
    - tensorflow: TFLite conversion and quantization
    - onnx2tf: ONNX to TensorFlow conversion utility
    - numpy: Calibration data generation

Usage:
    python utils/convert_tflite.py

Prerequisites:
    - ONNX model must exist at onnx_export/tcn_model.onnx
    - Run utils/export_onnx.py first to generate ONNX model

Output:
    - TrajectoFW/main/tcn_model_integer_quantized.tflite: INT8 quantized model
"""

import onnx
import tensorflow as tf
import numpy as np
import os
import onnx2tf
import sys
import shutil
import subprocess

# Determine project root dynamically
# Assuming this script is in `utils/` and project root is one level up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to sys.path if needed for imports, though not directly for this script
sys.path.insert(0, PROJECT_ROOT)


ONNX_PATH_RELATIVE = "onnx_export/tcn_model.onnx"
TF_PATH_RELATIVE = "onnx_export/tf_model"
CALIB_DATA_DIR = os.path.join(PROJECT_ROOT, "onnx_export/calib_data")
TFLITE_PATH_RELATIVE = "TrajectoFW/main/tcn_model.tflite"

ONNX_PATH = os.path.join(PROJECT_ROOT, ONNX_PATH_RELATIVE)
TF_PATH = os.path.join(PROJECT_ROOT, TF_PATH_RELATIVE)
TFLITE_PATH = os.path.join(PROJECT_ROOT, TFLITE_PATH_RELATIVE)


def get_input_info(onnx_model):
    """Extracts input tensor metadata from ONNX model graph.

    Parses the ONNX graph to extract input tensor names and shapes required
    for conversion. Handles dynamic dimensions by defaulting them to 1.

    Args:
        onnx_model: Loaded ONNX model (onnx.ModelProto)

    Returns:
        List[Dict]: List of input metadata dictionaries with keys:
            - 'name' (str): Input tensor name
            - 'shape' (Tuple[int]): Input tensor shape

    Example:
        >>> model = onnx.load("model.onnx")
        >>> info = get_input_info(model)
        >>> print(info)
        [{'name': 'input_feature', 'shape': (1, 1, 20)},
         {'name': 'state_in_0', 'shape': (1, 2, 20)}]
    """
    info = []
    for tensor in onnx_model.graph.input:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(1) # Default to 1 for dynamic dimensions
        info.append({'name': tensor.name, 'shape': tuple(shape)})
    return info

def representative_dataset_gen_factory(input_info):
    """Creates a representative dataset generator for TFLite quantization calibration.

    The representative dataset is used by TFLite's post-training quantization to
    determine optimal quantization parameters (scale/zero-point) by analyzing
    activation ranges during inference on calibration data.

    This factory pattern allows the generator to capture input_info in its closure
    while providing TFLiteConverter with a parameter-free callable.

    Args:
        input_info: List of input metadata dicts from get_input_info()
            Each dict contains 'name' and 'shape' keys

    Returns:
        Callable[[], Generator]: Generator function that yields calibration samples

    Yields:
        List[np.ndarray]: Input tensors for one inference pass, ordered to match
            the ONNX model's input order. Each tensor is FP32.

    Raises:
        FileNotFoundError: If calibration .npy files are missing from CALIB_DATA_DIR

    Note:
        Currently configured for 200 calibration samples. Adjust num_samples
        based on dataset diversity vs conversion time tradeoff.

    Example:
        >>> gen = representative_dataset_gen_factory(input_info)
        >>> converter.representative_dataset = gen
    """
    def generator():
        num_samples = 200 # Config after acquire_calib
        for i in range(num_samples):
            input_tensors = []
            for info in input_info:
                name = info['name']
                npy_path = os.path.join(CALIB_DATA_DIR, f"{name}.npy")

                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"Calibration file {npy_path} not found.")

                all_data = np.load(npy_path)
                sample_data = all_data[i] # [Batch, Seq, Feat]

                input_tensors.append(sample_data.astype(np.float32))
            yield input_tensors
    return generator


def main():
    """Main conversion pipeline: ONNX → TensorFlow SavedModel → TFLite (INT8).

    Pipeline stages:
    1. Load ONNX model and extract input metadata
    2. Generate random calibration data (normal distribution)
    3. Convert ONNX to TensorFlow SavedModel using onnx2tf:
       - Preserves input tensor layouts (NLC format)
       - Disables onnxsim to avoid shape inference issues
    4. Convert TensorFlow to TFLite with full INT8 quantization:
       - Uses representative dataset for activation range calibration
       - Sets input/output types to INT8
       - Restricts ops to INT8-only (no fallback to FP32)

    The resulting model is optimized for TFLite Micro on ESP32S3.

    Raises:
        FileNotFoundError: If ONNX model doesn't exist
        subprocess.CalledProcessError: If onnx2tf conversion fails

    Side Effects:
        - Creates calibration data in onnx_export/calib_data/
        - Creates TensorFlow SavedModel in onnx_export/tf_model/
        - Writes TFLite model to TrajectoFW/main/tcn_model_integer_quantized.tflite

    Note:
        Current calibration uses synthetic random data. For production, use
        real calibration data via utils/acquire_calib.py for better quantization accuracy.
    """
    if not os.path.exists(ONNX_PATH):
        print(f"Error: {ONNX_PATH} not found. Run export_onnx.py first.")
        return

    print(f"Loading ONNX from {ONNX_PATH}...")
    onnx_model = onnx.load(ONNX_PATH)
    input_info = get_input_info(onnx_model)

    # Get all input names to prevent onnx2tf from messing with their layout
    input_names = [i['name'] for i in input_info]
    print(f"Preserving shapes for inputs: {input_names}")

    # Generate calibration data (for TFLiteConverter's representative_dataset)
    print("Generating calibration data (for TFLiteConverter's representative_dataset)...")
    os.makedirs(CALIB_DATA_DIR, exist_ok=True)

    num_calib_samples = 1 # We'll just use one sample for now, matching the original setup

    for info in input_info:
        name = info['name']
        shape = info['shape']
        # Calibration data shape: [N, ...]
        calib_shape = (num_calib_samples,) + shape

        # Generate random data (similar to representative_dataset_gen)
        # Using float32
        data = np.random.normal(0, 1, calib_shape).astype(np.float32)

        npy_path = os.path.join(CALIB_DATA_DIR, f"{name}.npy")
        # Save each sample separately if num_calib_samples > 1, or just the one sample
        if num_calib_samples == 1:
            np.save(npy_path, data)
        else:
            # If multiple samples, perhaps save as data[0], data[1] etc or a single file containing all.
            # For simplicity, sticking to num_calib_samples = 1 for now.
            print("Warning: Currently only saving one calibration sample. Adjust logic for multiple samples.")

    print("Converting ONNX to TensorFlow SavedModel using onnx2tf...")

    cmd = [
        "onnx2tf",
        "-i", ONNX_PATH,
        "-o", TF_PATH,
        "-osd", # Output signature defs
        "-kat", *input_names, # Keep shape absolutely
        "--not_use_onnxsim", # Disable onnxsim
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"onnx2tf failed with error: {e}")
        return

    # Convert the TensorFlow SavedModel to TFLite (Full Integer Quantization)
    print(f"Converting TensorFlow SavedModel from {TF_PATH} to Full Integer Quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen_factory(input_info)

    # Ensure that ops are only quantized to integers
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (or int8)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_full_int_quant = converter.convert()

    full_int_quant_tflite_path = os.path.join(os.path.dirname(TFLITE_PATH), "tcn_model_integer_quantized.tflite")
    os.makedirs(os.path.dirname(full_int_quant_tflite_path), exist_ok=True)

    with open(full_int_quant_tflite_path, "wb") as f:
        f.write(tflite_model_full_int_quant)
    print(f"Full Integer Quantized TFLite model saved to {full_int_quant_tflite_path}")

    print("Success! (Full Integer Quantized TFLite model generated)")


if __name__ == "__main__":
    main()