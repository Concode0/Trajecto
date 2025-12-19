import tensorflow as tf
import numpy as np
import os

TFLITE_PATH = "onnx_export/tf_model/tcn_model_float32.tflite"

def main():
    if not os.path.exists(TFLITE_PATH):
        print(f"Error: {TFLITE_PATH} not found.")
        return

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Inputs:")
    for i in input_details:
        print(f"  {i['name']}: {i['shape']}")

    print("Outputs:")
    for i in output_details:
        print(f"  {i['name']}: {i['shape']}")

    # Create dummy inputs
    for i, detail in enumerate(input_details):
        shape = detail['shape']
        dtype = detail['dtype']
        data = np.random.randn(*shape).astype(dtype)
        interpreter.set_tensor(detail['index'], data)

    print("Invoking interpreter...")
    interpreter.invoke()
    print("Invoke successful!")

if __name__ == "__main__":
    main()