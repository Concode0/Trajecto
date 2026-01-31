# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: This software is protected under the following ROK Patent Applications:
# 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
# 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
#
# Commercial use or redistribution of the core logic requires a separate license.
# For inquiries, contact: nemonanconcode@gmail.com

"""
Tests for Zero-velocity Update (ZUPT) detector.

Covers:
- Stationary detection (ZUPT should trigger)
- Motion rejection (ZUPT should NOT trigger)
- Force instability detection
- Window buffer functionality

Run with: pytest tests/test_zupt_detector.py -v
"""

import torch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.zupt_detector import ZuptDetector
from model.config import Config


class TestZuptDetectorStationary:
    """Tests for stationary scenario detection."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @pytest.fixture
    def detector(self, device):
        """Create ZUPT detector with test thresholds."""
        return ZuptDetector(
            window_size=10,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

    def test_stationary_detection(self, detector, device):
        """Test that stationary data triggers ZUPT."""
        batch_size = 4
        window_size = 10

        # Static accelerometer data close to gravity
        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = (
            accel_static.unsqueeze(0).unsqueeze(0).repeat(batch_size, window_size, 1)
            + torch.randn(batch_size, window_size, 3, device=device) * 0.001
        )

        # Stable force data
        force_data = (
            torch.ones(batch_size, window_size, 1, device=device) * 10.0
            + torch.randn(batch_size, window_size, 1, device=device) * 0.0001
        )

        # Update detector with each timestep
        for i in range(window_size):
            detector.update(accel_data[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()

        assert torch.all(zupt_detected), \
            f"Expected ZUPT for static data, got: {zupt_detected}"

    def test_stationary_with_minimal_noise(self, detector, device):
        """Test ZUPT detection with minimal sensor noise."""
        batch_size = 2
        window_size = 10

        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = accel_static.unsqueeze(0).unsqueeze(0).repeat(batch_size, window_size, 1)

        force_data = torch.ones(batch_size, window_size, 1, device=device) * 5.0

        for i in range(window_size):
            detector.update(accel_data[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()
        assert torch.all(zupt_detected), "Perfect static data should trigger ZUPT"


class TestZuptDetectorMotion:
    """Tests for motion scenario rejection."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @pytest.fixture
    def detector(self, device):
        return ZuptDetector(
            window_size=10,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

    def test_motion_rejection(self, detector, device):
        """Test that motion data does NOT trigger ZUPT."""
        batch_size = 4
        window_size = 10

        # High variance accelerometer data (motion)
        accel_motion = torch.randn(batch_size, window_size, 3, device=device) * 1.0

        # Stable force
        force_data = torch.ones(batch_size, window_size, 1, device=device) * 10.0

        for i in range(window_size):
            detector.update(accel_motion[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()

        assert torch.all(~zupt_detected), \
            f"Expected NO ZUPT for motion data, got: {zupt_detected}"

    def test_high_angular_rate_rejection(self, detector, device):
        """Test that high angular rates don't affect accelerometer-based ZUPT."""
        batch_size = 2
        window_size = 10

        # Static accelerometer
        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = (
            accel_static.unsqueeze(0).unsqueeze(0).repeat(batch_size, window_size, 1)
            + torch.randn(batch_size, window_size, 3, device=device) * 0.001
        )

        # Stable force
        force_data = torch.ones(batch_size, window_size, 1, device=device) * 10.0

        for i in range(window_size):
            detector.update(accel_data[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()
        # Accelerometer-based ZUPT should still trigger
        assert torch.all(zupt_detected)


class TestZuptDetectorForceInstability:
    """Tests for force instability detection."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @pytest.fixture
    def detector(self, device):
        return ZuptDetector(
            window_size=10,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

    def test_unstable_force_rejection(self, detector, device):
        """Test that unstable force data does NOT trigger ZUPT."""
        batch_size = 4
        window_size = 10

        # Static accelerometer
        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = (
            accel_static.unsqueeze(0).unsqueeze(0).repeat(batch_size, window_size, 1)
            + torch.randn(batch_size, window_size, 3, device=device) * 0.001
        )

        # Very unstable force data
        force_data = torch.randn(batch_size, window_size, 1, device=device) * 5.0 + 10.0

        for i in range(window_size):
            detector.update(accel_data[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()

        assert torch.all(~zupt_detected), \
            f"Expected NO ZUPT for unstable force, got: {zupt_detected}"

    def test_force_spike_rejection(self, detector, device):
        """Test that force spikes prevent ZUPT detection."""
        batch_size = 2
        window_size = 10

        # Static accelerometer
        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = (
            accel_static.unsqueeze(0).unsqueeze(0).repeat(batch_size, window_size, 1)
            + torch.randn(batch_size, window_size, 3, device=device) * 0.001
        )

        # Force with large spike
        force_data = torch.ones(batch_size, window_size, 1, device=device) * 10.0
        force_data[:, window_size // 2, :] = 20.0  # Spike in the middle

        for i in range(window_size):
            detector.update(accel_data[:, i, :], force_data[:, i, :])

        zupt_detected = detector.detect()
        # Large delta should prevent ZUPT
        assert torch.all(~zupt_detected), "Force spike should prevent ZUPT"


class TestZuptDetectorBuffer:
    """Tests for ZUPT detector buffer mechanics."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_buffer_filling(self, device):
        """Test that detector requires full buffer before detection."""
        window_size = 10
        batch_size = 2

        detector = ZuptDetector(
            window_size=window_size,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        force_stable = torch.ones(batch_size, 1, device=device) * 10.0

        # Feed data but not enough to fill buffer
        for i in range(window_size - 1):
            accel = accel_static.unsqueeze(0).repeat(batch_size, 1)
            detector.update(accel, force_stable)

        # Detection before buffer is full
        zupt_before = detector.detect()

        # Fill the rest of the buffer
        accel = accel_static.unsqueeze(0).repeat(batch_size, 1)
        detector.update(accel, force_stable)

        # Detection after buffer is full
        zupt_after = detector.detect()

        assert torch.all(zupt_after), "ZUPT should trigger after buffer is full"

    def test_reset_detection(self, device):
        """Test detector reset functionality."""
        window_size = 5
        batch_size = 2

        detector = ZuptDetector(
            window_size=window_size,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        force_stable = torch.ones(batch_size, 1, device=device) * 10.0

        # Fill buffer with static data
        for i in range(window_size):
            accel = accel_static.unsqueeze(0).repeat(batch_size, 1)
            detector.update(accel, force_stable)

        zupt_before_reset = detector.detect()
        assert torch.all(zupt_before_reset), "Should detect ZUPT before reset"


class TestZuptDetectorEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_single_batch(self, device):
        """Test with batch size of 1."""
        detector = ZuptDetector(
            window_size=5,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

        batch_size = 1
        window_size = 5

        accel_static = torch.tensor([[0.0, 0.0, Config.GRAVITY_MAGNITUDE]], device=device)
        force_stable = torch.ones(batch_size, 1, device=device) * 10.0

        for i in range(window_size):
            detector.update(accel_static, force_stable)

        zupt_detected = detector.detect()
        assert zupt_detected.shape == (batch_size,)

    def test_large_batch(self, device):
        """Test with large batch size."""
        detector = ZuptDetector(
            window_size=5,
            accel_var_threshold=0.01,
            force_var_threshold=0.005,
            force_delta_threshold=0.05,
            device=device,
        )

        batch_size = 128
        window_size = 5

        accel_static = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)
        accel_data = accel_static.unsqueeze(0).repeat(batch_size, 1)
        force_stable = torch.ones(batch_size, 1, device=device) * 10.0

        for i in range(window_size):
            detector.update(accel_data, force_stable)

        zupt_detected = detector.detect()
        assert zupt_detected.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
