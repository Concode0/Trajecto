"""Dataset classes for simulated data and mixed sim+real training.

This module provides:
- SimulatedDataset: Loads SimGenerator HDF5 output
- MixedDataset: Combines simulated and real data with configurable ratio
"""

from typing import Dict, List, Optional

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SimulatedDataset(Dataset):
    """Dataset for loading SimGenerator HDF5 output.

    The SimGenerator outputs data in the same format as TrajectoryDataset:
    - sensor_data: [T, 7] (accel, gyro, fsr)
    - gt_pos_data: [T, 3]
    - gt_vel_data: [T, 3]
    - gt_gravity_b_data: [T, 3]
    - pen_down: [T] (optional)
    """

    def __init__(
        self,
        sim_file: str,
        augment_multiplier: int = 1,
        subsample_step: int = 1,
        do_augment: bool = True,
        noise_std: float = 0.01,
        yaw_range: tuple = (-0.78, 0.78),
        sigma_tilt: float = 0.02
    ) -> None:
        """Initialize SimulatedDataset.

        Args:
            sim_file: Path to HDF5 file from SimGenerator
            augment_multiplier: Multiplier for virtual dataset size
            subsample_step: Step size for subsampling
            do_augment: Whether to apply augmentation
            noise_std: Standard deviation for noise injection
            yaw_range: Range for yaw rotation augmentation (radians)
            sigma_tilt: Standard deviation for tilt perturbation
        """
        self.augment_multiplier = augment_multiplier
        self.cached_data: List[Dict[str, torch.Tensor]] = []
        self.do_augment = do_augment
        self.noise_std = noise_std
        self.yaw_range = yaw_range
        self.sigma_tilt = sigma_tilt

        print(f"Loading simulated dataset from {sim_file}...")
        with h5py.File(sim_file, "r") as f:
            keys = [k for k in f.keys() if k.startswith("sample_")]
            for key in tqdm(keys, desc="Caching sim dataset"):
                sensor = f[key]["sensor_data"][:]
                pos = f[key]["gt_pos_data"][:]
                vel = f[key]["gt_vel_data"][:]

                gravity_b = None
                if "gt_gravity_b_data" in f[key]:
                    gravity_b = f[key]["gt_gravity_b_data"][:]

                pen_down = None
                if "pen_down" in f[key]:
                    pen_down = f[key]["pen_down"][:]

                try:
                    seq_len = f[key].attrs["sequence_length"]
                except KeyError:
                    import numpy as np
                    diff = np.diff(sensor, axis=0)
                    try:
                        seq_len = np.where(np.any(diff != 0, axis=1))[0][-1] + 2
                    except IndexError:
                        seq_len = sensor.shape[0]

                data_dict = {
                    "sensor": torch.from_numpy(sensor[::subsample_step]).float(),
                    "pos": torch.from_numpy(pos[::subsample_step]).float(),
                    "vel": torch.from_numpy(vel[::subsample_step]).float(),
                    "len": seq_len // subsample_step,
                }
                if gravity_b is not None:
                    data_dict["gravity_b"] = torch.from_numpy(gravity_b[::subsample_step]).float()
                if pen_down is not None:
                    data_dict["pen_down"] = torch.from_numpy(pen_down[::subsample_step].astype(bool))

                self.cached_data.append(data_dict)

        self.num_original_samples = len(self.cached_data)
        print(f"Loaded {self.num_original_samples} simulated samples.")

    def __len__(self) -> int:
        return self.num_original_samples * self.augment_multiplier

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve a sample from the dataset.

        Returns:
            Dict with keys: imu_seq_raw, gt_pos_w, gt_vel_w, gt_gravity_b, pen_down, len
        """
        real_idx = idx % self.num_original_samples
        data = self.cached_data[real_idx]

        sensor_data = data["sensor"].clone()
        pos_data = data["pos"].clone()
        vel_data = data["vel"].clone()
        gravity_b_data = data["gravity_b"].clone() if "gravity_b" in data else None
        pen_down_data = data["pen_down"].clone() if "pen_down" in data else None

        if self.do_augment and idx >= self.num_original_samples:
            # Yaw rotation augmentation
            yaw = torch.rand(1) * (self.yaw_range[1] - self.yaw_range[0]) + self.yaw_range[0]
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            rot_yaw = torch.tensor([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ]).float()

            pos_data = (rot_yaw @ pos_data.T).T
            vel_data = (rot_yaw @ vel_data.T).T
            sensor_data[:, :3] = (rot_yaw @ sensor_data[:, :3].T).T
            sensor_data[:, 3:6] = (rot_yaw @ sensor_data[:, 3:6].T).T

            # Local grip error (Lie group perturbation)
            delta_theta = torch.randn(3) * self.sigma_tilt
            angle = torch.norm(delta_theta)
            if angle > 1e-8:
                axis = delta_theta / angle
                K = torch.tensor([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                I = torch.eye(3)
                rot_local = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

                sensor_data[:, :3] = (rot_local @ sensor_data[:, :3].T).T
                sensor_data[:, 3:6] = (rot_local @ sensor_data[:, 3:6].T).T

                if gravity_b_data is not None:
                    gravity_b_data = (rot_local @ gravity_b_data.T).T

            # Noise injection based on Allan variance
            noise = torch.zeros_like(sensor_data)
            accel_noise_std = torch.tensor([8.33e-3, 6.72e-3, 9.33e-3], dtype=sensor_data.dtype)
            noise[:, :3] = torch.randn(sensor_data.shape[0], 3) * accel_noise_std * 1.5

            gyro_noise_std = torch.tensor([7.17e-4, 7.93e-4, 7.53e-4], dtype=sensor_data.dtype)
            noise[:, 3:6] = torch.randn(sensor_data.shape[0], 3) * gyro_noise_std * 1.5

            noise[:, 6:] = torch.randn(sensor_data.shape[0], sensor_data.shape[1] - 6) * 0.005

            sensor_data += noise

        result = {
            "imu_seq_raw": sensor_data,
            "gt_pos_w": pos_data,
            "gt_vel_w": vel_data,
            "len": data["len"],
        }
        if gravity_b_data is not None:
            result["gt_gravity_b"] = gravity_b_data
        if pen_down_data is not None:
            result["pen_down"] = pen_down_data

        return result


class MixedDataset(Dataset):
    """Dataset mixing simulated and real data.

    During training, samples are drawn from either sim or real dataset
    based on the configured ratio.
    """

    def __init__(
        self,
        sim_dataset: SimulatedDataset,
        real_dataset: "Dataset",
        sim_ratio: float = 0.8,
        seed: int = 42
    ) -> None:
        """Initialize MixedDataset.

        Args:
            sim_dataset: SimulatedDataset instance
            real_dataset: TrajectoryDataset instance
            sim_ratio: Probability of sampling from sim_dataset (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.sim_dataset = sim_dataset
        self.real_dataset = real_dataset
        self.sim_ratio = sim_ratio
        self.rng = torch.Generator().manual_seed(seed)

        # Total size is max of both (with ratio applied)
        self.total_samples = max(len(sim_dataset), len(real_dataset))

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample from either sim or real dataset based on ratio."""
        if torch.rand(1, generator=self.rng).item() < self.sim_ratio:
            return self.sim_dataset[idx % len(self.sim_dataset)]
        else:
            return self.real_dataset[idx % len(self.real_dataset)]

    def get_source(self, idx: int) -> str:
        """Get the source dataset for a given index (for debugging)."""
        # Re-seed to match the same random draw
        temp_rng = torch.Generator().manual_seed(42)
        for _ in range(idx):
            torch.rand(1, generator=temp_rng)
        if torch.rand(1, generator=temp_rng).item() < self.sim_ratio:
            return "sim"
        return "real"
