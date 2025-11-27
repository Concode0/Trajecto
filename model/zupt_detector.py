import torch
import torch.nn as nn
from typing import Tuple

class ZuptDetector(nn.Module):
    """A stateful detector for Zero-Velocity Updates (ZUPT).

    This module implements a robust ZUPT detection algorithm, crucial for mitigating
    drift in Inertial Navigation Systems (INS). It operates on a rolling window
    of IMU data and uses a combination of a force sensor trigger and IMU motion
    variance to determine if the device is stationary.
    """
    def __init__(self,
                 window_size: int = 20,
                 accel_var_threshold: float = 0.1,
                 force_var_threshold: float = 0.01,
                 force_delta_threshold: float = 0.1,
                 device: str = 'cpu'):
        """Initializes the ZUPT detector module.

        Args:
            window_size (int): The number of IMU samples in the rolling window used
                for variance calculation.
            accel_var_threshold (float): The variance threshold for accelerometer data.
                If the mean variance is below this, the device is considered to have low linear motion.
            force_var_threshold (float): The variance threshold for force data.
            force_delta_threshold (float): The delta threshold for force data.
            device (str): The compute device ('cpu', 'cuda', 'mps').
        """
        super(ZuptDetector, self).__init__()
        self.window_size = window_size
        self.accel_var_threshold = accel_var_threshold
        self.force_var_threshold = force_var_threshold
        self.force_delta_threshold = force_delta_threshold
        self.device = device

        self.accel_body_buffer = None
        self.force_buffer = None
        self.buffer_idx = 0
        self.is_buffer_full = False
        self.is_enabled = True

    def _init_buffers(self, batch_size: int, dtype: torch.dtype):
        """Initializes the ring buffers for gyroscope and accelerometer data."""
        self.accel_body_buffer = torch.zeros(batch_size, self.window_size, 3, device=self.device, dtype=dtype)
        self.force_buffer = torch.zeros(batch_size, self.window_size, 1, device=self.device, dtype=dtype)
        self.buffer_idx = 0
        self.is_buffer_full = False

    def update(self, accel_body_raw: torch.Tensor, force_raw: torch.Tensor):
        """Updates the internal ring buffers with new IMU data.

        Args:
            accel_body_raw (torch.Tensor): A tensor of raw accelerometer readings in the
                body frame, with shape `[B, 3]`.
            force_raw (torch.Tensor): A tensor of raw force readings, with shape `[B, 1]`.
        """
        if self.accel_body_buffer is None or self.accel_body_buffer.shape[0] != accel_body_raw.shape[0]:
            self._init_buffers(accel_body_raw.shape[0], accel_body_raw.dtype)

        self.accel_body_buffer[:, self.buffer_idx, :] = accel_body_raw
        self.force_buffer[:, self.buffer_idx, :] = force_raw
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size

        if not self.is_buffer_full and self.buffer_idx == 0:
            self.is_buffer_full = True

    def detect(self) -> torch.Tensor:
        """Detects if the device is stationary based on buffered data and force.

        The core ZUPT logic combines two conditions:
        1.  Stance Detection: The force sensor reading must be above a threshold,
            indicating the pen is pressed against a surface.
        2.  Low Motion Detection: The variance of both gyroscope and accelerometer
            readings over a time window must be below their respective thresholds,
            indicating minimal movement.

        Returns:
            torch.Tensor: A boolean tensor of shape `[B]` indicating if a ZUPT
            is detected for each item in the batch.
        """
        if not self.is_enabled or not self.is_buffer_full:
            if self.accel_body_buffer is None:
                return torch.tensor([False], device=self.device).expand(self.force_buffer.shape[0])
            return torch.zeros(self.accel_body_buffer.shape[0], device=self.device, dtype=torch.bool)

        # Condition 1: Low motion is detected by checking signal variance.
        accel_variance = torch.var(self.accel_body_buffer, dim=1)
        is_low_motion = torch.mean(accel_variance, dim=-1) < self.accel_var_threshold

        # Condition 2: Force is stable
        force_variance = torch.var(self.force_buffer, dim=1)
        force_delta = torch.max(self.force_buffer, dim=1).values - torch.min(self.force_buffer, dim=1).values
        is_force_stable = (torch.mean(force_variance, dim=-1) < self.force_var_threshold) & \
                          (force_delta.squeeze(-1) < self.force_delta_threshold)

        # ZUPT is triggered only when both conditions are met.
        is_static = is_low_motion & is_force_stable
        return is_static

    def forward(self, accel_body_raw: torch.Tensor, force_raw: torch.Tensor) -> torch.Tensor:
        """A convenience method that combines updating the buffer and running detection.

        Args:
            accel_body_raw (torch.Tensor): A tensor of raw accelerometer readings in the
                body frame, with shape `[B, 3]`.
            force_raw (torch.Tensor): A tensor of raw force readings, with shape `[B, 1]`.

        Returns:
            torch.Tensor: A boolean tensor of shape `[B]` indicating if a ZUPT is detected.
        """
        self.update(accel_body_raw, force_raw)
        return self.detect()

if __name__ == '__main__':
    # Example usage and test case
    device = 'cpu'
    batch_size = 4
    window_size = 10

    detector = ZuptDetector(window_size=window_size, device=device)

    # Simulate some data
    # Stationary data
    accel_stationary = torch.randn(batch_size, window_size, 3, device=device) * 0.01
    accel_stationary[:, :, 2] += 9.81
    force_stationary = torch.ones(batch_size, 1, device=device) * 1.0

    # Moving data
    accel_moving = torch.randn(batch_size, window_size, 3, device=device) * 2.0
    force_moving = torch.zeros(batch_size, 1, device=device)

    # Test stationary case
    for i in range(window_size):
        detector.update(accel_stationary[:, i, :], force_stationary)
    
    zupt_detected = detector.detect()
    print(f"Stationary test: ZUPT detected? -> {zupt_detected}")
    assert torch.all(zupt_detected), "ZUPT should be detected for stationary data"

    # Test moving case
    for i in range(window_size):
        detector.update(accel_moving[:, i, :], force_moving)

    zupt_detected = detector.detect()
    print(f"Moving test: ZUPT detected? -> {zupt_detected}")
    assert not torch.any(zupt_detected), "ZUPT should not be detected for moving data"
    
    print("\nZuptDetector tested successfully.")
