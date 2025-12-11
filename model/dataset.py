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
        subsample_step: int = 4,
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
        """
        self.augment_multiplier = augment_multiplier
        self.cached_data: List[Dict[str, torch.Tensor]] = []

        print("Loading dataset into RAM for high-speed training...")
        with h5py.File(preprocessed_file, "r") as f:
            keys = list(f.keys())
            for key in tqdm(keys, desc="Caching dataset"):
                # Load sensor data, ground truth position, and velocity from HDF5.
                sensor = f[key]["sensor_data"][:]
                pos = f[key]["gt_pos_data"][:]
                vel = f[key]["gt_vel_data"][:]

                # Subsample to reduce sequence length and convert to PyTorch tensors.
                # This is a crucial step for managing memory and computational load
                # during training, especially with high-frequency data.
                self.cached_data.append(
                    {
                        "sensor": torch.from_numpy(sensor[::subsample_step]).float(),
                        "pos": torch.from_numpy(pos[::subsample_step]).float(),
                        "vel": torch.from_numpy(vel[::subsample_step]).float(),
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
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the sensor data ('imu_seq_raw'),
            ground truth position ('gt_pos_w'), and ground truth
            velocity ('gt_vel_w') as PyTorch tensors.
        """
        # The modulo operator allows for augmenting the dataset by wrapping around
        # the original number of samples.
        real_idx = idx % self.num_original_samples
        data = self.cached_data[real_idx]

        # Cloning the tensors ensures that any modifications to the returned data
        # will not affect the cached data, which is important during training
        # when data augmentations or other transformations might be applied.
        return {
            "imu_seq_raw": data["sensor"].clone(),
            "gt_pos_w": data["pos"].clone(),
            "gt_vel_w": data["vel"].clone(),
        }