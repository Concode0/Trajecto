"""
This module defines a `ZuptDetector` class, a crucial component for IMU-based
navigation systems that implements Zero-Velocity Update (ZUPT) detection.

ZUPT is a technique used in inertial navigation to reduce drift errors by
exploiting periods when the sensor is stationary. During these periods,
the velocity of the sensor is known to be zero, and this information can be
fed into a Kalman filter as a highly accurate pseudo-measurement to
constrain state estimation. This detector uses a combination of accelerometer
variance and force sensor readings (variance and delta) over a sliding window
to robustly identify stationary periods.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ZuptDetector(nn.Module):
    """A stateful detector for Zero-Velocity Updates (ZUPT) based on IMU and force sensor data.

    This detector buffers incoming sensor data (accelerometer and force) over a
    sliding window. It identifies stationary periods by checking if the variance
    of accelerometer readings and the stability of force sensor readings fall
    below predefined thresholds.

    Conditions for ZUPT detection:
    1. **Low Motion:** The variance of accelerometer readings over a window
       must be below a threshold, indicating minimal linear motion.
    2. **Stable Force:** The variance and the range (max-min) of force sensor
       readings over a window must be below respective thresholds, indicating
       the sensor is stably pressed (or not moving significantly) on a surface.
    """

    def __init__(
        self,
        window_size: int = 20,
        accel_var_threshold: float = 0.1,
        force_var_threshold: float = 0.01,
        force_delta_threshold: float = 0.1,
        device: str = "cpu",
    ):
        """Initializes the ZUPT detector module.

        Args:
            window_size: The number of IMU samples in the rolling window used
                for variance and delta calculations.
            accel_var_threshold: The variance threshold for accelerometer data.
                If the mean variance across all axes is below this value, the
                device is considered to have low linear motion.
            force_var_threshold: The variance threshold for force data. If the
                variance is below this, the force reading is considered stable.
            force_delta_threshold: The peak-to-peak (max-min) delta threshold
                for force data. If the range is below this, the force reading
                is considered stable.
            device: The compute device ('cpu', 'cuda', 'mps').
        """
        super().__init__()
        self.window_size = window_size
        self.accel_var_threshold = accel_var_threshold
        self.force_var_threshold = force_var_threshold
        self.force_delta_threshold = force_delta_threshold
        self.device = device

        # Ring buffers to store historical IMU data for variance calculation.
        # These are dynamically initialized based on the batch size of the first input.
        self.accel_body_buffer: Optional[torch.Tensor] = None
        self.force_buffer: Optional[torch.Tensor] = None
        self.buffer_idx: int = 0  # Current insertion point in the circular buffer.
        self.is_buffer_full: bool = False  # Flag to indicate if the buffer has been filled once.
        self.is_enabled: bool = True  # Allows enabling/disabling the detector.

    def _init_buffers(self, batch_size: int, dtype: torch.dtype) -> None:
        """Initializes the ring buffers for accelerometer and force data.

        This method is called once when the first batch of data is received
        to set up the buffers with the correct batch size and data type.

        Args:
            batch_size: The batch size of the incoming IMU data.
            dtype: The data type (e.g., torch.float32) for the buffers.
        """
        # Accelerometer buffer: [batch_size, window_size, 3] for 3-axis accel.
        self.accel_body_buffer = torch.zeros(
            batch_size, self.window_size, 3, device=self.device, dtype=dtype
        )
        # Force buffer: [batch_size, window_size, 1] for scalar force.
        self.force_buffer = torch.zeros(
            batch_size, self.window_size, 1, device=self.device, dtype=dtype
        )
        self.buffer_idx = 0
        self.is_buffer_full = False

    def update(self, accel_body_raw: torch.Tensor, force_raw: torch.Tensor) -> None:
        """Updates the internal ring buffers with new IMU data for each batch element.

        Args:
            accel_body_raw (torch.Tensor): Raw accelerometer readings.
                - Shape: (Batch, 3)
                - Unit: m/s^2
                - Frame: Body
            force_raw (torch.Tensor): Raw force readings.
                - Shape: (Batch, 1)
                - Unit: N
        """
        # Initialize buffers if they haven't been or if batch size changes.
        if (
            self.accel_body_buffer is None
            or self.accel_body_buffer.shape[0] != accel_body_raw.shape[0]
        ):
            self._init_buffers(accel_body_raw.shape[0], accel_body_raw.dtype)

        # Update the circular buffers with the latest data.
        # `buffer_idx` points to the oldest data point, which is overwritten.
        # This effectively implements a sliding window.
        if self.accel_body_buffer is not None and self.force_buffer is not None:
            self.accel_body_buffer[:, self.buffer_idx, :] = accel_body_raw
            self.force_buffer[:, self.buffer_idx, :] = force_raw
        else:
            raise RuntimeError("Buffers were not initialized correctly.")

        # Increment buffer index, wrapping around at `window_size`.
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size

        # Set `is_buffer_full` flag once the buffer has been filled for the first time.
        if not self.is_buffer_full and self.buffer_idx == 0:
            self.is_buffer_full = True

    def detect(self) -> torch.Tensor:
        """Detects if the device is stationary (ZUPT condition) based on buffered data.

        The ZUPT logic combines two primary conditions:
        1.  **Low Motion Detection:** Assessed by the variance of accelerometer data.
        2.  **Stable Contact (for pen-based systems):** Assessed by the variance
            and maximum delta of force sensor readings.

        Returns:
            torch.Tensor: A boolean tensor indicating if ZUPT is detected.
                - Shape: (Batch,)
                - Type: Bool
        """
        # ZUPT cannot be detected if the detector is disabled or the buffer
        # has not yet been filled with enough data for a full window.
        if not self.is_enabled or not self.is_buffer_full:
            if self.accel_body_buffer is None:
                # If buffers are not even initialized, return False for all.
                return torch.tensor([False], device=self.device).expand(
                    1
                )  # Expand to a default batch size of 1
            # If buffer is not full but initialized, return False for the batch size.
            return torch.zeros(
                self.accel_body_buffer.shape[0], device=self.device, dtype=torch.bool
            )

        # Condition 1: Low motion detection using accelerometer variance.
        # Calculate variance across the time window (dim=1) for each accelerometer axis (dim=2).
        # Then, take the mean variance across the three axes.
        if self.accel_body_buffer is None:
            raise RuntimeError("Accelerometer buffer is None.")
        accel_variance = torch.var(self.accel_body_buffer, dim=1)  # [B, 3]
        # Check if the mean variance across all 3 accelerometer axes is below the threshold.
        is_low_motion = (
            torch.mean(accel_variance, dim=-1) < self.accel_var_threshold
        )  # [B]

        # Condition 2: Stable force detection using force variance and delta.
        if self.force_buffer is None:
            raise RuntimeError("Force buffer is None.")
        force_variance = torch.var(self.force_buffer, dim=1)  # [B, 1]
        # Calculate the peak-to-peak change (max - min) in force over the window.
        force_delta = (
            torch.max(self.force_buffer, dim=1).values
            - torch.min(self.force_buffer, dim=1).values
        )  # [B, 1]
        # Check if both force variance and force delta are below their respective thresholds.
        is_force_stable = (
            force_variance.squeeze(-1) < self.force_var_threshold
        ) & (  # [B]
            force_delta.squeeze(-1) < self.force_delta_threshold
        )  # [B]

        # A ZUPT is detected only when both low motion and stable force conditions are met.
        is_static = is_low_motion & is_force_stable  # [B]
        return is_static

    def forward(
        self, accel_body_raw: torch.Tensor, force_raw: torch.Tensor
    ) -> torch.Tensor:
        """A convenience method that combines updating the buffer and running detection.

        Args:
            accel_body_raw (torch.Tensor): Raw accelerometer readings.
                - Shape: (Batch, 3)
                - Unit: m/s^2
                - Frame: Body
            force_raw (torch.Tensor): Raw force readings.
                - Shape: (Batch, 1)
                - Unit: N

        Returns:
            torch.Tensor: A boolean tensor indicating if ZUPT is detected.
                - Shape: (Batch,)
                - Type: Bool
        """
        self.update(accel_body_raw, force_raw)
        return self.detect()
