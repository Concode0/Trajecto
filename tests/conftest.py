"""
Pytest configuration and shared fixtures for Trajecto tests
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_imu_data():
    """Generate synthetic IMU data for testing"""
    n_steps = 100
    np.random.seed(42)

    # Stationary: gravity only
    accel = np.zeros((n_steps, 3))
    accel[:, 2] = 9.81  # Gravity in Z

    # Small random noise
    accel += np.random.randn(n_steps, 3) * 0.01
    gyro = np.random.randn(n_steps, 3) * 0.001

    return {
        'accel': torch.tensor(accel, dtype=torch.float32),
        'gyro': torch.tensor(gyro, dtype=torch.float32),
        'n_steps': n_steps
    }


@pytest.fixture
def moving_imu_data():
    """Generate IMU data with motion"""
    n_steps = 100
    t = np.arange(n_steps) * 0.02  # 50 Hz

    # Sinusoidal motion in X
    accel_x = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation
    accel = np.zeros((n_steps, 3))
    accel[:, 0] = accel_x
    accel[:, 2] = 9.81  # Gravity

    gyro = np.zeros((n_steps, 3))

    return {
        'accel': torch.tensor(accel, dtype=torch.float32),
        'gyro': torch.tensor(gyro, dtype=torch.float32),
        'time': t
    }
