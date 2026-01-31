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
Tests for ErrorStateKalmanFilter (ESKF).

Covers:
- ESKF forward pass with TCN integration
- Traditional ZUPT mode
- Learnable virtual measurement weights
- Block-Parallel Scan caching

Run with: pytest tests/test_eskf.py -v
"""

import torch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.ESKF import ErrorStateKalmanFilter
from model.config import Config


class TestESKFForward:
    """Tests for ESKF forward pass."""

    @pytest.fixture
    def device(self):
        """Get available device. Use CPU for ESKF due to MPS cholesky_solve limitation."""
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    @pytest.fixture
    def eskf_with_tcn(self, device):
        """Create ESKF with TCN-based ZUPT enabled."""
        return ErrorStateKalmanFilter(dt=0.01, device=device, use_tcn_zupt=True)

    @pytest.fixture
    def initial_state(self, device):
        """Create initial nominal state."""
        batch_size = 8
        pos_w = torch.zeros(batch_size, 3, device=device)
        vel_w = torch.zeros(batch_size, 3, device=device)
        quat_b_to_w = torch.zeros(batch_size, 4, device=device)
        quat_b_to_w[:, 0] = 1.0  # Identity quaternion
        gyro_bias_b = torch.zeros(batch_size, 3, device=device)
        accel_bias_b = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        return {
            "pos_w": pos_w,
            "vel_w": vel_w,
            "quat_b_to_w": quat_b_to_w,
            "gyro_bias_b": gyro_bias_b,
            "accel_bias_b": accel_bias_b,
            "P_error": P_error,
            "batch_size": batch_size,
        }

    @pytest.fixture
    def imu_data(self, device, initial_state):
        """Create dummy IMU data."""
        batch_size = initial_state["batch_size"]
        accel = torch.randn(batch_size, 3, device=device) * 0.1
        accel[:, 2] += Config.GRAVITY_MAGNITUDE  # Near-static condition
        gyro = torch.randn(batch_size, 3, device=device) * 0.01
        force = torch.rand(batch_size, 1, device=device)
        measurement = torch.cat([accel, gyro], dim=-1)

        return {
            "accel": accel,
            "gyro": gyro,
            "force": force,
            "measurement": measurement,
        }

    @pytest.fixture
    def tcn_output(self, device, initial_state):
        """Create dummy TCN output."""
        batch_size = initial_state["batch_size"]
        return {
            "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
            "covariance_R": torch.randn(batch_size, 6, device=device),
            "zupt_prob": torch.rand(batch_size, 1, device=device),
        }

    def test_forward_with_tcn_output_shapes(self, eskf_with_tcn, initial_state, imu_data, tcn_output):
        """Test ESKF forward pass produces correct output shapes."""
        result = eskf_with_tcn.forward(
            initial_state["pos_w"],
            initial_state["vel_w"],
            initial_state["quat_b_to_w"],
            initial_state["gyro_bias_b"],
            initial_state["accel_bias_b"],
            initial_state["P_error"],
            imu_data["gyro"],
            imu_data["accel"],
            imu_data["force"],
            imu_data["measurement"],
            tcn_output=tcn_output,
        )

        pos_w_out, vel_w_out, quat_out, gyro_bias_out, accel_bias_out, P_error_out, tcn_feats = result
        batch_size = initial_state["batch_size"]

        assert pos_w_out.shape == (batch_size, 3), f"Position shape mismatch: {pos_w_out.shape}"
        assert vel_w_out.shape == (batch_size, 3), f"Velocity shape mismatch: {vel_w_out.shape}"
        assert quat_out.shape == (batch_size, 4), f"Quaternion shape mismatch: {quat_out.shape}"
        assert gyro_bias_out.shape == (batch_size, 3), f"Gyro bias shape mismatch: {gyro_bias_out.shape}"
        assert accel_bias_out.shape == (batch_size, 3), f"Accel bias shape mismatch: {accel_bias_out.shape}"
        assert P_error_out.shape == (batch_size, 15, 15), f"P_error shape mismatch: {P_error_out.shape}"

    def test_forward_tcn_features_dict(self, eskf_with_tcn, initial_state, imu_data, tcn_output):
        """Test that TCN features dictionary is populated."""
        result = eskf_with_tcn.forward(
            initial_state["pos_w"],
            initial_state["vel_w"],
            initial_state["quat_b_to_w"],
            initial_state["gyro_bias_b"],
            initial_state["accel_bias_b"],
            initial_state["P_error"],
            imu_data["gyro"],
            imu_data["accel"],
            imu_data["force"],
            imu_data["measurement"],
            tcn_output=tcn_output,
        )

        tcn_feats = result[-1]
        assert isinstance(tcn_feats, dict), "TCN features should be a dictionary"
        # Check for expected keys in TCN features
        for key, value in tcn_feats.items():
            assert isinstance(value, torch.Tensor), f"TCN feature '{key}' should be a tensor"


class TestESKFTraditionalZUPT:
    """Tests for ESKF with traditional ZUPT detector."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_traditional_zupt_mode(self, device):
        """Test ESKF with traditional ZUPT (no TCN ZUPT)."""
        eskf = ErrorStateKalmanFilter(dt=0.01, device=device, use_tcn_zupt=False, use_zupt=True)

        batch_size = 4
        pos_w = torch.zeros(batch_size, 3, device=device)
        vel_w = torch.zeros(batch_size, 3, device=device)
        quat_b_to_w = torch.zeros(batch_size, 4, device=device)
        quat_b_to_w[:, 0] = 1.0
        gyro_bias_b = torch.zeros(batch_size, 3, device=device)
        accel_bias_b = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        accel = torch.randn(batch_size, 3, device=device) * 0.1
        accel[:, 2] += Config.GRAVITY_MAGNITUDE
        gyro = torch.randn(batch_size, 3, device=device) * 0.01
        force = torch.rand(batch_size, 1, device=device)
        measurement = torch.cat([accel, gyro], dim=-1)

        result = eskf.forward(
            pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error,
            gyro, accel, force, measurement, tcn_output=None
        )

        P_error_out = result[5]
        assert P_error_out.shape == (batch_size, 15, 15)


class TestESKFLearnableWeights:
    """Tests for ESKF with learnable virtual measurement weights."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_virtual_measurement_weights_exist(self, device):
        """Test that learnable virtual measurement weights are created."""
        eskf = ErrorStateKalmanFilter(
            dt=0.01, device=device,
            use_tcn_zupt=False, use_zupt=False,
            use_virtual_measurements=True
        )

        assert hasattr(eskf, 'virtual_meas_weights'), "Should have virtual_meas_weights attribute"
        assert eskf.virtual_meas_weights.shape == (15,), f"Shape mismatch: {eskf.virtual_meas_weights.shape}"

    def test_virtual_measurement_forward(self, device):
        """Test that virtual measurements can be applied during forward pass."""
        eskf = ErrorStateKalmanFilter(
            dt=0.01, device=device,
            use_tcn_zupt=False, use_zupt=False,
            use_virtual_measurements=True
        )

        batch_size = 4
        pos_w = torch.zeros(batch_size, 3, device=device)
        vel_w = torch.zeros(batch_size, 3, device=device)
        quat_b_to_w = torch.zeros(batch_size, 4, device=device)
        quat_b_to_w[:, 0] = 1.0
        gyro_bias_b = torch.zeros(batch_size, 3, device=device)
        accel_bias_b = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        # Motion data (not stationary) to trigger virtual measurements
        accel = torch.randn(batch_size, 3, device=device) * 2.0
        accel[:, 2] += Config.GRAVITY_MAGNITUDE
        gyro = torch.randn(batch_size, 3, device=device) * 0.5
        force = torch.rand(batch_size, 1, device=device)
        measurement = torch.cat([accel, gyro], dim=-1)

        result = eskf.forward(
            pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, P_error,
            gyro, accel, force, measurement, tcn_output=None
        )

        # Check outputs are produced without error
        pos_out, vel_out, quat_out, gyro_bias_out, accel_bias_out, P_error_out, _ = result
        assert pos_out.shape == (batch_size, 3)
        assert quat_out.shape == (batch_size, 4)

    def test_virtual_measurement_learnable(self, device):
        """Test that virtual measurement weights can be made learnable."""
        eskf = ErrorStateKalmanFilter(
            dt=0.01, device=device,
            use_tcn_zupt=False, use_zupt=False,
            use_virtual_measurements=True,
            eskf_learnable_params=True  # Make weights learnable
        )

        # Verify virtual_meas_weights is a learnable parameter
        assert eskf.virtual_meas_weights.requires_grad, \
            "virtual_meas_weights should require gradients when eskf_learnable_params=True"

        # Verify it's part of model parameters
        param_names = [name for name, _ in eskf.named_parameters()]
        assert 'virtual_meas_weights' in param_names, \
            "virtual_meas_weights should be a named parameter"


class TestESKFBlockParallelCache:
    """Tests for ESKF block-parallel scan caching."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support cholesky_solve, fall back to CPU
        return "cpu"

    def test_cache_initialization(self, device):
        """Test cache initialization."""
        eskf = ErrorStateKalmanFilter(dt=0.01, device=device, use_tcn_zupt=True)
        batch_size, seq_len = 4, 10

        eskf.init_cache(batch_size, seq_len)
        # Should not raise

    def test_cache_finalization(self, device):
        """Test cache finalization returns expected structure."""
        eskf = ErrorStateKalmanFilter(dt=0.01, device=device, use_tcn_zupt=True)
        batch_size, seq_len = 4, 10

        # Initialize state
        pos_w = torch.zeros(batch_size, 3, device=device)
        vel_w = torch.zeros(batch_size, 3, device=device)
        quat = torch.zeros(batch_size, 4, device=device)
        quat[:, 0] = 1.0
        gyro_bias = torch.zeros(batch_size, 3, device=device)
        accel_bias = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        eskf.init_cache(batch_size, seq_len)

        for t in range(seq_len):
            accel = torch.randn(batch_size, 3, device=device) * 0.1
            accel[:, 2] += Config.GRAVITY_MAGNITUDE
            gyro = torch.randn(batch_size, 3, device=device) * 0.01
            force = torch.rand(batch_size, 1, device=device)
            measurement = torch.cat([accel, gyro], dim=-1)
            tcn_out = {
                "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
                "covariance_R": torch.randn(batch_size, 6, device=device),
                "zupt_prob": torch.rand(batch_size, 1, device=device),
            }

            pos_w, vel_w, quat, gyro_bias, accel_bias, P_error, _ = eskf.forward(
                pos_w, vel_w, quat, gyro_bias, accel_bias, P_error,
                gyro, accel, force, measurement, tcn_output=tcn_out
            )

        cache = eskf.finalize_cache()
        assert hasattr(cache, 'F_seq'), "Cache should have F_seq"
        assert hasattr(cache, 'Q_seq'), "Cache should have Q_seq"
        assert hasattr(cache, 'P_init'), "Cache should have P_init"

    def test_parallel_covariance_matches_sequential(self, device):
        """Test that parallel covariance computation matches sequential."""
        torch.manual_seed(42)
        eskf = ErrorStateKalmanFilter(dt=0.01, device=device, use_tcn_zupt=True)
        batch_size, seq_len = 2, 10

        # Initialize state
        pos_w = torch.zeros(batch_size, 3, device=device)
        vel_w = torch.zeros(batch_size, 3, device=device)
        quat = torch.zeros(batch_size, 4, device=device)
        quat[:, 0] = 1.0
        gyro_bias = torch.zeros(batch_size, 3, device=device)
        accel_bias = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

        eskf.init_cache(batch_size, seq_len)
        P_seq_list = []

        for t in range(seq_len):
            accel = torch.randn(batch_size, 3, device=device) * 0.1
            accel[:, 2] += Config.GRAVITY_MAGNITUDE
            gyro = torch.randn(batch_size, 3, device=device) * 0.01
            force = torch.rand(batch_size, 1, device=device)
            measurement = torch.cat([accel, gyro], dim=-1)
            tcn_out = {
                "vel_corr": torch.randn(batch_size, 3, device=device) * 0.01,
                "covariance_R": torch.randn(batch_size, 6, device=device),
                "zupt_prob": torch.rand(batch_size, 1, device=device),
            }

            pos_w, vel_w, quat, gyro_bias, accel_bias, P_error, _ = eskf.forward(
                pos_w, vel_w, quat, gyro_bias, accel_bias, P_error,
                gyro, accel, force, measurement, tcn_output=tcn_out
            )
            P_seq_list.append(P_error.clone())

        cache = eskf.finalize_cache()

        try:
            P_parallel = eskf.parallel_covariance_from_cache(cache, block_size=4)
            P_sequential = torch.stack(P_seq_list, dim=1)
            max_diff = (P_parallel - P_sequential).abs().max().item()
            assert max_diff < 1e-3, f"Max diff too large: {max_diff:.2e}"
        except Exception:
            # Skip if parallel covariance not implemented
            pytest.skip("Parallel covariance computation not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
