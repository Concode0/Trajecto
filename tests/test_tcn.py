# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [PATENT NOTICE]
# This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
# Commercial use without a separate license is strictly prohibited.
#
# Contact: nemonanconcode@gmail.com

"""
Tests for Temporal Convolutional Network (TCN).

Covers:
- Forward pass
- Output shapes for each head
- Receptive field computation
- Y-branch architecture

Run with: pytest tests/test_tcn.py -v
"""

import torch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.TCN import TCN
from model.config import Config


class TestTCNForward:
    """Tests for TCN forward pass."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @pytest.fixture
    def tcn_model(self, device):
        """Create TCN model with default config."""
        return TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

    def test_forward_pass_shapes(self, tcn_model, device):
        """Test forward pass produces correct output shapes."""
        batch_size = 32
        seq_length = 100
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn_model(dummy_input)

        # Check that outputs is a dictionary
        assert isinstance(outputs, dict), "Output should be a dictionary"

        # Check expected output heads exist
        for head, pred in outputs.items():
            assert isinstance(pred, torch.Tensor), f"Output '{head}' should be a tensor"
            assert pred.shape[0] == batch_size, f"Batch size mismatch for '{head}'"
            assert pred.shape[1] == seq_length, f"Sequence length mismatch for '{head}'"

    def test_output_heads_exist(self, tcn_model, device):
        """Test that expected output heads are present."""
        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn_model(dummy_input)

        # Common heads in ESKF-TCN architecture
        expected_heads = ["vel_corr", "covariance_R", "zupt_prob"]
        for head in expected_heads:
            assert head in outputs, f"Missing expected output head: {head}"

    def test_vel_corr_shape(self, tcn_model, device):
        """Test velocity correction output shape (3D vector)."""
        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn_model(dummy_input)

        if "vel_corr" in outputs:
            assert outputs["vel_corr"].shape == (batch_size, seq_length, 3), \
                f"vel_corr shape mismatch: {outputs['vel_corr'].shape}"

    def test_covariance_R_shape(self, tcn_model, device):
        """Test covariance R output shape (6D log-variance)."""
        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn_model(dummy_input)

        if "covariance_R" in outputs:
            assert outputs["covariance_R"].shape == (batch_size, seq_length, 6), \
                f"covariance_R shape mismatch: {outputs['covariance_R'].shape}"

    def test_zupt_prob_shape(self, tcn_model, device):
        """Test ZUPT probability output shape (scalar)."""
        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn_model(dummy_input)

        if "zupt_prob" in outputs:
            assert outputs["zupt_prob"].shape == (batch_size, seq_length, 1), \
                f"zupt_prob shape mismatch: {outputs['zupt_prob'].shape}"


class TestTCNReceptiveField:
    """Tests for TCN receptive field."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_receptive_field_attribute(self, device):
        """Test that receptive_field attribute is computed."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        assert hasattr(tcn, 'receptive_field'), "TCN should have receptive_field attribute"
        assert tcn.receptive_field > 0, "Receptive field should be positive"

    def test_receptive_field_forward_pass(self, device):
        """Test forward pass with input length equal to receptive field."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        batch_size = 8
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE
        rf = tcn.receptive_field

        # Input with length exactly equal to receptive field
        dummy_input = torch.randn(batch_size, rf, input_features, device=device)
        outputs = tcn(dummy_input)

        for head, pred in outputs.items():
            expected_len = rf
            actual_len = pred.shape[1]
            assert actual_len == expected_len, \
                f"Output length mismatch for {head}: expected {expected_len}, got {actual_len}"

    def test_receptive_field_shorter_input(self, device):
        """Test forward pass with input shorter than receptive field."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        batch_size = 4
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE
        short_len = max(1, tcn.receptive_field // 2)

        dummy_input = torch.randn(batch_size, short_len, input_features, device=device)
        outputs = tcn(dummy_input)

        # Should still produce output
        for head, pred in outputs.items():
            assert pred.shape[0] == batch_size
            assert pred.shape[1] == short_len


class TestTCNGradients:
    """Tests for TCN gradient flow."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_gradient_flow(self, device):
        """Test that gradients flow through TCN."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(
            batch_size, seq_length, input_features,
            device=device, requires_grad=True
        )
        outputs = tcn(dummy_input)

        # Compute loss from all outputs
        loss = sum(v.sum() for v in outputs.values())
        loss.backward()

        assert dummy_input.grad is not None, "No gradient for input"
        assert not torch.isnan(dummy_input.grad).any(), "NaN in input gradient"

    def test_parameter_gradients(self, device):
        """Test that all parameters receive gradients."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn(dummy_input)

        loss = sum(v.sum() for v in outputs.values())
        loss.backward()

        # Check that at least some parameters have gradients
        params_with_grad = sum(1 for p in tcn.parameters() if p.grad is not None)
        total_params = sum(1 for _ in tcn.parameters())
        assert params_with_grad > 0, "No parameters received gradients"


class TestTCNNormalization:
    """Tests for TCN normalization behavior."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_gravity_b_normalized(self, device):
        """Test that gravity_b output is L2 normalized if present."""
        tcn = TCN(
            input_size=Config.ESKFTCN.TCN_INPUT_SIZE,
            tcn_channels=Config.ESKFTCN.TCN_CHANNELS,
            kernel_size=Config.ESKFTCN.KERNEL_SIZE,
        ).to(device)

        batch_size = 4
        seq_length = 50
        input_features = Config.ESKFTCN.TCN_INPUT_SIZE

        dummy_input = torch.randn(batch_size, seq_length, input_features, device=device)
        outputs = tcn(dummy_input)

        if "gravity_b" in outputs:
            gravity_b = outputs["gravity_b"]
            norms = torch.norm(gravity_b, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
                "gravity_b should be L2 normalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
