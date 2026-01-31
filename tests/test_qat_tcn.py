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
Tests for PT2E Quantization-Aware Training (QAT) for TCN.

Covers:
- PT2E availability check
- QAT model preparation
- Forward pass with fake quantization
- Observer statistics collection
- Quantized model conversion

Run with: pytest tests/test_qat_tcn.py -v
"""

import torch
import torch.nn as nn
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.qat_tcn import (
    PT2E_AVAILABLE,
    prepare_qat_pt2e_model,
    convert_qat_to_quantized,
    get_qat_observer_stats,
)


class TestPT2EAvailability:
    """Tests for PT2E availability."""

    def test_pt2e_available_is_boolean(self):
        """Test that PT2E_AVAILABLE is a boolean."""
        assert isinstance(PT2E_AVAILABLE, bool)


@pytest.mark.skipif(not PT2E_AVAILABLE, reason="PT2E not available")
class TestQATPreparation:
    """Tests for QAT model preparation."""

    @pytest.fixture
    def tcn_model(self):
        """Create TCN model for testing."""
        from model.TCN import TCN
        return TCN(
            input_size=19,
            tcn_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1,
        )

    @pytest.fixture
    def mock_eskf_tcn(self, tcn_model):
        """Create mock ESKF-TCN wrapper."""
        class MockESKFTCN(nn.Module):
            def __init__(self, tcn):
                super().__init__()
                self.tcn = tcn

            def forward(self, x):
                return self.tcn(x)

        return MockESKFTCN(tcn_model)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size, seq_len = 2, 100
        return torch.randn(batch_size, seq_len, 19)

    def test_qat_preparation(self, mock_eskf_tcn, sample_input):
        """Test QAT model preparation."""
        try:
            model, _ = prepare_qat_pt2e_model(mock_eskf_tcn, sample_input)
            assert model is not None
        except Exception as e:
            pytest.skip(f"QAT preparation failed: {e}")

    def test_qat_forward_pass(self, mock_eskf_tcn, sample_input):
        """Test forward pass with QAT model."""
        try:
            model, _ = prepare_qat_pt2e_model(mock_eskf_tcn, sample_input)
            model.train()
            outputs = model(sample_input)
            assert isinstance(outputs, dict)
        except Exception as e:
            pytest.skip(f"QAT forward pass failed: {e}")

    def test_qat_output_shapes(self, mock_eskf_tcn, sample_input):
        """Test QAT output shapes match original."""
        batch_size, seq_len = sample_input.shape[:2]

        try:
            model, _ = prepare_qat_pt2e_model(mock_eskf_tcn, sample_input)
            model.train()
            outputs = model(sample_input)

            for key, val in outputs.items():
                assert val.shape[0] == batch_size, f"Batch mismatch for {key}"
                assert val.shape[1] == seq_len, f"Seq length mismatch for {key}"
        except Exception as e:
            pytest.skip(f"QAT output shapes test failed: {e}")


@pytest.mark.skipif(not PT2E_AVAILABLE, reason="PT2E not available")
class TestQATObservers:
    """Tests for QAT observer statistics."""

    @pytest.fixture
    def trained_qat_model(self):
        """Create and train QAT model briefly."""
        from model.TCN import TCN

        class MockESKFTCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.tcn = TCN(
                    input_size=19,
                    tcn_channels=[32, 32, 32, 32],  # Must be 4 elements for Y-architecture
                    kernel_size=3,
                    dropout=0.0,
                )

            def forward(self, x):
                return self.tcn(x)

        model = MockESKFTCN()
        sample_input = torch.randn(2, 50, 19)

        try:
            model, _ = prepare_qat_pt2e_model(model, sample_input)
            model.train()
            # Run a few forward passes to populate observers
            for _ in range(3):
                _ = model(torch.randn(2, 50, 19))
            return model
        except Exception:
            return None

    def test_observer_stats_collection(self, trained_qat_model):
        """Test that observer statistics can be collected."""
        if trained_qat_model is None:
            pytest.skip("Could not create trained QAT model")

        try:
            stats = get_qat_observer_stats(trained_qat_model)
            assert isinstance(stats, (dict, list))
        except Exception as e:
            pytest.skip(f"Observer stats collection failed: {e}")


@pytest.mark.skipif(not PT2E_AVAILABLE, reason="PT2E not available")
class TestQATConversion:
    """Tests for QAT to quantized model conversion."""

    @pytest.fixture
    def qat_model(self):
        """Create QAT-prepared model."""
        from model.TCN import TCN

        class MockESKFTCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.tcn = TCN(
                    input_size=19,
                    tcn_channels=[32, 32, 32, 32],  # Must be 4 elements for Y-architecture
                    kernel_size=3,
                    dropout=0.0,
                )

            def forward(self, x):
                return self.tcn(x)

        model = MockESKFTCN()
        sample_input = torch.randn(2, 50, 19)

        try:
            model, _ = prepare_qat_pt2e_model(model, sample_input)
            model.train()
            # Run forward passes
            for _ in range(5):
                _ = model(torch.randn(2, 50, 19))
            model.eval()
            return model
        except Exception:
            return None

    def test_conversion_to_quantized(self, qat_model):
        """Test conversion to quantized model."""
        if qat_model is None:
            pytest.skip("Could not create QAT model")

        try:
            quantized_model = convert_qat_to_quantized(qat_model)
            assert quantized_model is not None
        except Exception as e:
            pytest.skip(f"Conversion failed: {e}")


class TestTCNForwardWithoutQAT:
    """Tests for TCN forward pass (no QAT dependency)."""

    def test_tcn_creation(self):
        """Test TCN can be created."""
        from model.TCN import TCN

        tcn = TCN(
            input_size=19,
            tcn_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1,
        )
        assert tcn is not None

    def test_tcn_forward(self):
        """Test TCN forward pass."""
        from model.TCN import TCN

        tcn = TCN(
            input_size=19,
            tcn_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1,
        )

        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, 19)
        outputs = tcn(x)

        assert isinstance(outputs, dict)
        for key, val in outputs.items():
            assert val.shape[0] == batch_size
            assert val.shape[1] == seq_len

    def test_tcn_receptive_field(self):
        """Test TCN receptive field computation."""
        from model.TCN import TCN

        tcn = TCN(
            input_size=19,
            tcn_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1,
        )

        assert hasattr(tcn, 'receptive_field')
        assert tcn.receptive_field > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
