import torch
import numpy as np
import os
import h5py
from tqdm import tqdm
from receive import BleReceiver
from model.ESKF_TCN import ESKFTCN_model
from model.stateful_tcn import StatefulTCNExport
from model.config import Config

# 저장 경로 설정 (convert_tflite.py와 일치시킴)
CALIB_DIR = "onnx_export/calib_data"
os.makedirs(CALIB_DIR, exist_ok=True)

class CalibDataCollector:
    def __init__(self, model_path="eskf_tcn_model.pth"):
        self.device = "cpu"
        base_model = ESKFTCN_model(device=self.device, dt=Config.DT)
        base_model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.stateful_tcn = StatefulTCNExport(base_model.tcn)
        self.stateful_tcn.eval()

        self.state_buffers = []
        for layer in base_model.tcn.tcn_layers:
            k = layer.conv.kernel_size[0]
            d = layer.conv.dilation[0]
            hist_len = (k - 1) * d
            in_ch = layer.conv.in_channels
            self.state_buffers.append(torch.zeros(1, hist_len, in_ch))

    def collect_step(self, norm_imu, sample_idx):
        """
        norm_imu: 전처리가 완료된 [1, 1, 20] 형태의 텐서 (TCN 입력 규격)
        """
        with torch.no_grad():
            np.save(f"{CALIB_DIR}/input_feature_{sample_idx}.npy", norm_imu.numpy())
            for i, state in enumerate(self.state_buffers):
                np.save(f"{CALIB_DIR}/state_in_{i}_{sample_idx}.npy", state.numpy())

            outputs = self.stateful_tcn(norm_imu, *self.state_buffers)

            self.state_buffers = list(outputs[1:])

def main():
    collector = CalibDataCollector()
    receiver = BleReceiver()

    num_samples = 200
    collected = 0
    pbar = tqdm(total=num_samples, desc="Collecting Calibration Data")

    def on_imu_received(imu_data):
        nonlocal collected
        if collected < num_samples:
            # imu_data: [ax, ay, az, gx, gy, gz, force]
            collector.collect_step(imu_data, collected)
            collected += 1
            pbar.update(1)
        else:
            receiver.stop()

    receiver.set_callback(on_imu_received)
    print("Acquire the Calibration Data")
    receiver.start()

if __name__ == "__main__":
    main()