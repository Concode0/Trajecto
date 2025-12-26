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
    info = []
    for tensor in onnx_model.graph.input:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(1) # Default to 1
        info.append({'name': tensor.name, 'shape': tuple(shape)})
    return info

def representative_dataset_gen_factory(input_info):
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