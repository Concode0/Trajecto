# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
Tests for ESKF-TCN hybrid model.

Covers:
- Model instantiation
- Forward pass with closed-loop integration
- Output dictionary structure
- Gravity initialization

Run with: pytest tests/test_eskf_tcn.py -v
"""

import torch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.ESKF_TCN import ESKFTCN_model
from model.config import Config


class TestESKFTCNInstantiation:
    """Tests for ESKF-TCN model instantiation."""

    @pytest.fixture
    def device(self):
        """Get available device. Use CPU due to MPS cholesky_solve limitation."""
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_model_creation(self, device):
        """Test that model can be created."""
        model = ESKFTCN_model(device=device).to(device)
        assert model is not None

    def test_model_has_eskf(self, device):
        """Test that model has ESKF component."""
        model = ESKFTCN_model(device=device).to(device)
        # The model should have filter attribute (inherited from BaseFilterTCNModel)
        assert hasattr(model, 'filter') or hasattr(model, 'eskf')

    def test_model_has_tcn(self, device):
        """Test that model has TCN component."""
        model = ESKFTCN_model(device=device).to(device)
        assert hasattr(model, 'tcn')


class TestESKFTCNForward:
    """Tests for ESKF-TCN forward pass."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    @pytest.fixture
    def model(self, device):
        """Create model instance."""
        return ESKFTCN_model(device=device).to(device)

    def test_forward_pass(self, model, device):
        """Test basic forward pass."""
        batch_size = 4
        sequence_length = 100
        imu_features = 7  # accel(3) + gyro(3) + force(1)

        # Raw IMU data
        dummy_imu_raw = torch.randn(batch_size, sequence_length, imu_features, device=device)
        # Simulate realistic initial acceleration (gravity-aligned)
        dummy_imu_raw[0, 0, :3] = torch.tensor([0.5, 0.5, Config.GRAVITY_MAGNITUDE], device=device)

        # Normalized IMU data (in real scenario, pre-normalized)
        dummy_imu_norm = dummy_imu_raw.clone()

        output = model(dummy_imu_raw, dummy_imu_norm)

        assert isinstance(output, dict), "Output should be a dictionary"

    def test_forward_output_keys(self, model, device):
        """Test that forward pass returns expected keys."""
        batch_size = 4
        sequence_length = 100
        imu_features = 7

        dummy_imu_raw = torch.randn(batch_size, sequence_length, imu_features, device=device)
        dummy_imu_raw[:, 0, 2] = Config.GRAVITY_MAGNITUDE  # Initial gravity
        dummy_imu_norm = dummy_imu_raw.clone()

        output = model(dummy_imu_raw, dummy_imu_norm)

        # Check for position prediction (primary output)
        assert "pred_pos_w" in output, "Output should contain 'pred_pos_w'"

    def test_forward_position_shape(self, model, device):
        """Test position output shape."""
        batch_size = 4
        sequence_length = 100
        imu_features = 7

        dummy_imu_raw = torch.randn(batch_size, sequence_length, imu_features, device=device)
        dummy_imu_raw[:, 0, 2] = Config.GRAVITY_MAGNITUDE
        dummy_imu_norm = dummy_imu_raw.clone()

        output = model(dummy_imu_raw, dummy_imu_norm)

        assert output["pred_pos_w"].shape == (batch_size, sequence_length, 3), \
            f"Position shape mismatch: {output['pred_pos_w'].shape}"

    def test_forward_tensor_outputs(self, model, device):
        """Test that all tensor outputs have correct batch dimension."""
        batch_size = 4
        sequence_length = 100
        imu_features = 7

        dummy_imu_raw = torch.randn(batch_size, sequence_length, imu_features, device=device)
        dummy_imu_raw[:, 0, 2] = Config.GRAVITY_MAGNITUDE
        dummy_imu_norm = dummy_imu_raw.clone()

        output = model(dummy_imu_raw, dummy_imu_norm)

        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == batch_size, \
                    f"Batch size mismatch for '{key}': {value.shape[0]} != {batch_size}"


class TestESKFTCNGravityInit:
    """Tests for gravity-based initialization."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_gravity_aligned_initialization(self, device):
        """Test initialization with gravity-aligned acceleration."""
        model = ESKFTCN_model(device=device).to(device)

        batch_size = 2
        sequence_length = 50
        imu_features = 7

        # Create data with clear gravity signal at start
        dummy_imu_raw = torch.randn(batch_size, sequence_length, imu_features, device=device) * 0.1

        # First sample: gravity along z-axis (upright)
        dummy_imu_raw[0, 0, :3] = torch.tensor([0.0, 0.0, Config.GRAVITY_MAGNITUDE], device=device)

        # Second sample: tilted (gravity has x,y components)
        dummy_imu_raw[1, 0, :3] = torch.tensor([0.5, 0.5, Config.GRAVITY_MAGNITUDE], device=device)

        dummy_imu_norm = dummy_imu_raw.clone()

        output = model(dummy_imu_raw, dummy_imu_norm)

        # Should complete without error
        assert "pred_pos_w" in output


class TestESKFTCNGradients:
    """Tests for gradient flow through ESKF-TCN."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_gradient_flow(self, device):
        """Test that gradients flow through the hybrid model."""
        model = ESKFTCN_model(device=device).to(device)

        batch_size = 2
        sequence_length = 50
        imu_features = 7

        dummy_imu_raw = torch.randn(
            batch_size, sequence_length, imu_features,
            device=device, requires_grad=True
        )
        dummy_imu_raw_detached = dummy_imu_raw.detach().clone()
        dummy_imu_raw_detached[:, 0, 2] = Config.GRAVITY_MAGNITUDE

        # Need to re-enable grad after clone
        dummy_imu_input = dummy_imu_raw_detached.requires_grad_(True)
        dummy_imu_norm = dummy_imu_input.clone()

        output = model(dummy_imu_input, dummy_imu_norm)

        loss = output["pred_pos_w"].sum()
        loss.backward()

        # Check gradient exists
        assert dummy_imu_input.grad is not None, "No gradient for input"

    def test_model_parameters_trainable(self, device):
        """Test that model has trainable parameters."""
        model = ESKFTCN_model(device=device).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0, "Model should have trainable parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
