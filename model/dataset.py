"""This module defines a custom dataset for trajectory data.

This module defines a custom dataset for trajectory data, designed for use with
PyTorch. It loads, processes, and serves trajectory data from HDF5 files, making
it suitable for training machine learning models. The dataset supports subsampling
and data augmentation to enhance training efficiency and performance.
"""

from typing import Dict, List

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TrajectoryDataset(Dataset[Dict[str, torch.Tensor]]):
    """A PyTorch dataset for handling trajectory data from HDF5 files.

    This dataset loads all data from the specified HDF5 file into memory upon
    initialization to accelerate the training process. It also supports data
-   augmentation by multiplying the dataset size and subsampling to reduce
    sequence length.

    Attributes:
        augment_multiplier (int): The factor by which to augment the dataset size.
        cached_data (List[Dict[str, torch.Tensor]]): A list of dictionaries
            containing the cached data. Each dictionary holds 'sensor', 'pos',
            and 'vel' tensors for a single sample.
        num_original_samples (int): The number of unique samples loaded from
            the HDF5 file before augmentation.
    """

    def __init__(
        self,
        preprocessed_file: str,
        augment_multiplier: int = 1,
        subsample_step: int = 1,
        do_augment: bool = False,
        noise_std: float = 0.01,
        scale_range: tuple = (0.9, 1.1),
        yaw_range: tuple = (-0.78, 0.78), # Control Rotation
        sigma_tilt: float = 0.00
    ) -> None:
        """Initializes the TrajectoryDataset.

        Args:
            preprocessed_file: The path to the HDF5 file containing the
                preprocessed data.
            augment_multiplier: A multiplier for data augmentation. The total
                dataset size will be `num_original_samples * augment_multiplier`.
                Defaults to 1.
            subsample_step: The step size for subsampling the data to reduce
                sequence length. Defaults to 4.
            do_augment: Whether to perform data augmentation. Defaults to False.
            noise_std: The standard deviation of the noise to add to the sensor data.
            scale_range: A tuple representing the range of the random scaling factor.
            yaw_range: A tuple representing the range of the random yaw angle in radians.
        """
        self.augment_multiplier = augment_multiplier
        self.cached_data: List[Dict[str, torch.Tensor]] = []
        self.do_augment = do_augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.yaw_range = yaw_range
        self.sigma_tilt = sigma_tilt

        print("Loading dataset into RAM for high-speed training...")
        with h5py.File(preprocessed_file, "r") as f:
            keys = list(f.keys())
            for key in tqdm(keys, desc="Caching dataset"):
                # Load sensor data, ground truth position, and velocity from HDF5.
                sensor = f[key]["sensor_data"][:]
                pos = f[key]["gt_pos_data"][:]
                vel = f[key]["gt_vel_data"][:]
                try:
                    seq_len = f[key].attrs["sequence_length"]
                except KeyError:
                    # Fallback for older datasets
                    import numpy as np
                    diff = np.diff(sensor, axis=0)
                    try:
                        seq_len = np.where(np.any(diff != 0, axis=1))[0][-1] + 2
                    except IndexError:
                        seq_len = sensor.shape[0]


                # Subsample to reduce sequence length and convert to PyTorch tensors.
                # This is a crucial step for managing memory and computational load
                # during training, especially with high-frequency data.
                self.cached_data.append(
                    {
                        "sensor": torch.from_numpy(sensor[::subsample_step]).float(),
                        "pos": torch.from_numpy(pos[::subsample_step]).float(),
                        "vel": torch.from_numpy(vel[::subsample_step]).float(),
                        "len": seq_len // subsample_step,
                    }
                )

        self.num_original_samples = len(self.cached_data)
        print(f"Loaded {self.num_original_samples} original samples into RAM.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset, including augmentations.

        Returns:
            The total number of samples.
        """

        return self.num_original_samples * self.augment_multiplier

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves a sample from the dataset by index.

        The modulo operator on the index allows for augmenting the dataset
        by creating a larger virtual dataset that wraps around the original
        samples.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing sample data:
                - "imu_seq_raw": Raw IMU sequence.
                    - Shape: (Seq_Len, 6) | [accel(3), gyro(3)]
                    - Unit: m/s^2, rad/s
                    - Frame: Body
                - "gt_pos_w": Ground truth position.
                    - Shape: (Seq_Len, 3) | Unit: m | Frame: World
                - "gt_vel_w": Ground truth velocity.
                    - Shape: (Seq_Len, 3) | Unit: m/s | Frame: World
                - "len": Sequence length.
                    - Shape: Scalar | Type: Int
        """
        # The modulo operator allows for augmenting the dataset by wrapping around
        # the original number of samples.
        real_idx = idx % self.num_original_samples
        data = self.cached_data[real_idx]

        sensor_data = data["sensor"].clone()  # Shape: (Seq, 7) [Acc(3), Gyro(3), FSR(1)]
        pos_data = data["pos"].clone()        # Shape: (Seq, 3)
        vel_data = data["vel"].clone()        # Shape: (Seq, 3)

        if self.do_augment and idx >= self.num_original_samples:
            yaw = torch.rand(1) * (self.yaw_range[1] - self.yaw_range[0]) + self.yaw_range[0]
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            rot_yaw = torch.tensor([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ]).float()

            # Rotate sensor_data and gt_data
            pos_data = (rot_yaw @ pos_data.T).T
            vel_data = (rot_yaw @ vel_data.T).T
            sensor_data[:, :3] = (rot_yaw @ sensor_data[:, :3].T).T
            sensor_data[:, 3:6] = (rot_yaw @ sensor_data[:, 3:6].T).T

            # Set sigmal_tilt as (0.017~0.052 rad)
            # Simulate Local Grip Error based on Lie Group
            delta_theta = torch.randn(3) * self.sigma_tilt

            # Exponential Map (Rodrigues' rotation formula)
            angle = torch.norm(delta_theta)
            if angle > 1e-8:
                axis = delta_theta / angle
                # Generate Skew-symmetric matrix
                K = torch.tensor([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                # Rodrigues Formula: R = I + sin(theta)K + (1-cos(theta))K^2
                I = torch.eye(3)
                rot_local = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

                # Adopt local grip error in sensor data only
                sensor_data[:, :3] = (rot_local @ sensor_data[:, :3].T).T
                sensor_data[:, 3:6] = (rot_local @ sensor_data[:, 3:6].T).T

            # Inject Noise in all channels
            noise = torch.randn_like(sensor_data) * self.noise_std
            sensor_data += noise

        return {
            "imu_seq_raw": sensor_data,
            "gt_pos_w": pos_data,
            "gt_vel_w": vel_data,
            "len": data["len"],
        }