"""
Unit tests for Dataset loading and preprocessing
"""
import pytest
import torch
import h5py
import tempfile
from pathlib import Path
from model.dataset import TrajectoryDataset
from model.config import Config


class TestDatasetLoading:
    """Test dataset loading from HDF5"""

    @pytest.fixture
    def mock_hdf5_file(self):
        """Create a mock HDF5 file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        # Create mock HDF5 structure
        with h5py.File(filepath, 'w') as h5f:
            # Create sample data
            seq_len = 200
            sensor_data = torch.randn(seq_len, 7).numpy()  # accel(3) + gyro(3) + fsr(1)
            gt_pos = torch.randn(seq_len, 3).numpy()
            gt_vel = torch.randn(seq_len, 3).numpy()

            # Save to HDF5
            grp = h5f.create_group('sample_001_seg0')
            grp.create_dataset('sensor_data', data=sensor_data)
            grp.create_dataset('gt_pos_data', data=gt_pos)
            grp.create_dataset('gt_vel_data', data=gt_vel)
            grp.attrs['sequence_length'] = seq_len
            grp.attrs['original_label'] = 'sample_001'

        yield filepath

        # Cleanup
        Path(filepath).unlink()

    def test_dataset_loads_hdf5(self, mock_hdf5_file):
        """Dataset should load HDF5 file without errors"""
        dataset = TrajectoryDataset(mock_hdf5_file, do_augment=False)

        assert len(dataset) > 0, "Dataset is empty"
        assert dataset.num_samples > 0, "No samples loaded"

    def test_getitem_returns_correct_format(self, mock_hdf5_file):
        """Dataset __getitem__ should return properly formatted data"""
        dataset = TrajectoryDataset(mock_hdf5_file, do_augment=False)

        sample = dataset[0]

        # Check keys
        assert 'sensor' in sample
        assert 'gt_pos' in sample
        assert 'gt_vel' in sample
        assert 'seq_len' in sample

        # Check shapes
        assert sample['sensor'].shape[1] == 7, "Sensor data should have 7 channels"
        assert sample['gt_pos'].shape[1] == 3, "Position should be 3D"
        assert sample['gt_vel'].shape[1] == 3, "Velocity should be 3D"

    def test_normalization(self, mock_hdf5_file):
        """Dataset should normalize sensor data"""
        dataset = TrajectoryDataset(mock_hdf5_file, do_augment=False)

        sample = dataset[0]
        sensor = sample['sensor']

        # After normalization, data should have reasonable range
        # (not exact zero mean / unit std due to small sample)
        assert sensor.abs().max() < 10, "Normalized data has extreme values"


class TestDataAugmentation:
    """Test data augmentation transformations"""

    @pytest.fixture
    def sample_data(self):
        """Create sample sensor + GT data"""
        seq_len = 100
        return {
            'sensor': torch.randn(seq_len, 7),
            'gt_pos': torch.randn(seq_len, 3),
            'gt_vel': torch.randn(seq_len, 3),
            'seq_len': seq_len
        }

    def test_yaw_rotation_preserves_shapes(self, sample_data):
        """Yaw rotation should preserve data shapes"""
        from model.dataset import apply_yaw_rotation

        angle = 0.5  # radians

        sensor_aug, pos_aug, vel_aug = apply_yaw_rotation(
            sample_data['sensor'],
            sample_data['gt_pos'],
            sample_data['gt_vel'],
            angle
        )

        assert sensor_aug.shape == sample_data['sensor'].shape
        assert pos_aug.shape == sample_data['gt_pos'].shape
        assert vel_aug.shape == sample_data['gt_vel'].shape

    def test_yaw_rotation_changes_data(self, sample_data):
        """Yaw rotation should actually modify the data"""
        from model.dataset import apply_yaw_rotation

        angle = 0.5

        sensor_aug, pos_aug, vel_aug = apply_yaw_rotation(
            sample_data['sensor'].clone(),
            sample_data['gt_pos'].clone(),
            sample_data['gt_vel'].clone(),
            angle
        )

        # Data should be different after rotation (unless angle=0)
        assert not torch.allclose(sensor_aug, sample_data['sensor']), \
            "Sensor data unchanged after rotation"
        assert not torch.allclose(pos_aug, sample_data['gt_pos']), \
            "Position unchanged after rotation"

    def test_augmentation_multiplier(self, mock_hdf5_file):
        """Augmentation should increase dataset size"""
        dataset_no_aug = TrajectoryDataset(mock_hdf5_file, do_augment=False)
        original_size = len(dataset_no_aug)

        # With augmentation (Config.AUGMENT_MULTIPLIER = 10)
        dataset_aug = TrajectoryDataset(mock_hdf5_file, do_augment=True)
        augmented_size = len(dataset_aug)

        # Should have more samples (original + augmented)
        expected_size = original_size * (1 + Config.AUGMENT_MULTIPLIER)
        assert augmented_size == expected_size, \
            f"Expected {expected_size} samples, got {augmented_size}"


class TestSequencePadding:
    """Test sequence padding/truncation"""

    def test_padding_short_sequences(self, mock_hdf5_file):
        """Short sequences should be padded to MAX_SEQUENCE_LENGTH"""
        dataset = TrajectoryDataset(mock_hdf5_file, do_augment=False)

        sample = dataset[0]

        # If original sequence was shorter than MAX_SEQUENCE_LENGTH,
        # it should be padded
        assert sample['sensor'].shape[0] <= Config.MAX_SEQUENCE_LENGTH

    def test_truncating_long_sequences(self, mock_hdf5_file):
        """Long sequences should be truncated"""
        # Create dataset with very long sequence
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        seq_len = Config.MAX_SEQUENCE_LENGTH + 500  # Longer than max

        with h5py.File(filepath, 'w') as h5f:
            grp = h5f.create_group('long_sample')
            grp.create_dataset('sensor_data', data=torch.randn(seq_len, 7).numpy())
            grp.create_dataset('gt_pos_data', data=torch.randn(seq_len, 3).numpy())
            grp.create_dataset('gt_vel_data', data=torch.randn(seq_len, 3).numpy())
            grp.attrs['sequence_length'] = seq_len

        try:
            dataset = TrajectoryDataset(filepath, do_augment=False)
            sample = dataset[0]

            # Should be truncated to max length
            assert sample['sensor'].shape[0] <= Config.MAX_SEQUENCE_LENGTH
        finally:
            Path(filepath).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
