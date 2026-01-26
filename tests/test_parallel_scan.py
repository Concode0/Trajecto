"""
Comprehensive tests for Block-Parallel Scan implementation.

These tests verify:
1. Mathematical correctness (parallel matches sequential)
2. Associativity of the combine operator
3. Gradient flow and backpropagation (BPTT)
4. Numerical stability
5. Memory contiguity

Run with: pytest tests/test_parallel_scan.py -v
Or: python tests/test_parallel_scan.py
"""

import torch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.parallel_scan_ops import (
    sequential_covariance_propagation,
    parallel_covariance_scan,
    parallel_covariance_scan_blocked,
    verify_associativity,
    sanity_check_parallel_vs_sequential,
)


class TestCovarianceScanPrimitives:
    """Tests for the core covariance scan operations."""

    @pytest.fixture
    def setup_data(self):
        """Create test data for covariance tests."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        batch_size = 4
        seq_len = 100
        state_dim = 15

        # F matrices close to identity for numerical stability
        F_seq = torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0)
        F_seq = F_seq.repeat(batch_size, seq_len, 1, 1)
        F_seq += 0.01 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)

        # Q matrices positive semi-definite
        Q_raw = torch.randn(batch_size, seq_len, state_dim, state_dim, device=device) * 0.1
        Q_seq = torch.einsum("btij,btkj->btik", Q_raw, Q_raw)

        # P_init positive definite
        P_init_raw = torch.randn(batch_size, state_dim, state_dim, device=device) * 0.1
        P_init = torch.einsum("bij,bkj->bik", P_init_raw, P_init_raw)
        P_init += 0.01 * torch.eye(state_dim, device=device)

        return {
            "F_seq": F_seq,
            "Q_seq": Q_seq,
            "P_init": P_init,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "state_dim": state_dim,
            "device": device,
        }

    def test_associativity(self, setup_data):
        """Test that the combine operator is associative: (a⊗b)⊗c = a⊗(b⊗c)"""
        F_seq = setup_data["F_seq"]
        Q_seq = setup_data["Q_seq"]

        # Take three random elements
        F1, Q1 = F_seq[:, 0], Q_seq[:, 0]
        F2, Q2 = F_seq[:, 1], Q_seq[:, 1]
        F3, Q3 = F_seq[:, 2], Q_seq[:, 2]

        assert verify_associativity(F1, Q1, F2, Q2, F3, Q3, rtol=1e-5, atol=1e-7), \
            "Associativity property violated"

    def test_parallel_matches_sequential(self, setup_data):
        """Test that parallel scan produces same results as sequential."""
        F_seq = setup_data["F_seq"]
        Q_seq = setup_data["Q_seq"]
        P_init = setup_data["P_init"]

        P_sequential = sequential_covariance_propagation(F_seq, Q_seq, P_init)
        P_parallel = parallel_covariance_scan(F_seq, Q_seq, P_init)

        assert torch.allclose(P_parallel, P_sequential, rtol=1e-4, atol=1e-6), \
            f"Max diff: {(P_parallel - P_sequential).abs().max().item():.2e}"

    def test_blocked_matches_sequential(self, setup_data):
        """Test that blocked parallel scan matches sequential."""
        F_seq = setup_data["F_seq"]
        Q_seq = setup_data["Q_seq"]
        P_init = setup_data["P_init"]

        P_sequential = sequential_covariance_propagation(F_seq, Q_seq, P_init)

        # Relaxed tolerance: blocked scan has more FP operations due to block boundaries
        for block_size in [16, 32, 64]:
            P_blocked = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size)
            assert torch.allclose(P_blocked, P_sequential, rtol=1e-4, atol=1e-5), \
                f"Block size {block_size}: max diff: {(P_blocked - P_sequential).abs().max().item():.2e}"

    def test_gradient_flow(self, setup_data):
        """Test that gradients flow correctly through parallel scan."""
        F_seq = setup_data["F_seq"].clone().requires_grad_(True)
        Q_seq = setup_data["Q_seq"].clone().requires_grad_(True)
        P_init = setup_data["P_init"].clone().requires_grad_(True)

        P_out = parallel_covariance_scan(F_seq, Q_seq, P_init)
        loss = P_out.sum()
        loss.backward()

        assert F_seq.grad is not None, "No gradient for F_seq"
        assert Q_seq.grad is not None, "No gradient for Q_seq"
        assert P_init.grad is not None, "No gradient for P_init"

        assert not torch.isnan(F_seq.grad).any(), "NaN in F_seq gradient"
        assert not torch.isnan(Q_seq.grad).any(), "NaN in Q_seq gradient"
        assert not torch.isnan(P_init.grad).any(), "NaN in P_init gradient"

    def test_gradient_matches_sequential(self, setup_data):
        """Test that gradients match between parallel and sequential."""
        # Sequential
        F_seq_s = setup_data["F_seq"].clone().requires_grad_(True)
        Q_seq_s = setup_data["Q_seq"].clone().requires_grad_(True)
        P_init_s = setup_data["P_init"].clone().requires_grad_(True)

        P_sequential = sequential_covariance_propagation(F_seq_s, Q_seq_s, P_init_s)
        loss_s = P_sequential.sum()
        loss_s.backward()

        # Parallel
        F_seq_p = setup_data["F_seq"].clone().requires_grad_(True)
        Q_seq_p = setup_data["Q_seq"].clone().requires_grad_(True)
        P_init_p = setup_data["P_init"].clone().requires_grad_(True)

        P_parallel = parallel_covariance_scan(F_seq_p, Q_seq_p, P_init_p)
        loss_p = P_parallel.sum()
        loss_p.backward()

        # Compare gradients (relaxed tolerance for numerical differences)
        rtol, atol = 1e-3, 1e-5
        assert torch.allclose(F_seq_s.grad, F_seq_p.grad, rtol=rtol, atol=atol), \
            f"F gradient mismatch: max diff {(F_seq_s.grad - F_seq_p.grad).abs().max().item():.2e}"
        assert torch.allclose(Q_seq_s.grad, Q_seq_p.grad, rtol=rtol, atol=atol), \
            f"Q gradient mismatch: max diff {(Q_seq_s.grad - Q_seq_p.grad).abs().max().item():.2e}"
        assert torch.allclose(P_init_s.grad, P_init_p.grad, rtol=rtol, atol=atol), \
            f"P_init gradient mismatch: max diff {(P_init_s.grad - P_init_p.grad).abs().max().item():.2e}"

    def test_bptt_gradient_from_final_timestep(self, setup_data):
        """Test BPTT: gradient from final timestep reaches early timesteps.

        This verifies that the scatter-based implementation correctly propagates
        gradients through all iterations of the parallel scan.
        """
        F_seq = setup_data["F_seq"].clone().requires_grad_(True)
        Q_seq = setup_data["Q_seq"].clone().requires_grad_(True)
        P_init = setup_data["P_init"].clone().requires_grad_(True)
        seq_len = setup_data["seq_len"]

        P_out = parallel_covariance_scan(F_seq, Q_seq, P_init)

        # Loss on ONLY the final timestep - this must propagate to early timesteps
        loss = P_out[:, -1].sum()
        loss.backward()

        # Verify early timesteps have non-zero gradients
        early_F_grad = F_seq.grad[:, 0].abs().max().item()
        early_Q_grad = Q_seq.grad[:, 0].abs().max().item()

        assert early_F_grad > 1e-10, \
            f"Early F gradient too small: {early_F_grad:.2e}, BPTT may be broken"
        assert early_Q_grad > 1e-10, \
            f"Early Q gradient too small: {early_Q_grad:.2e}, BPTT may be broken"

        # Verify gradient pattern: later timesteps should have similar or larger gradients
        early_norm = F_seq.grad[:, :seq_len//4].abs().mean().item()
        late_norm = F_seq.grad[:, -seq_len//4:].abs().mean().item()

        # Late gradients should be at least 10% of early (cumulative effect)
        assert late_norm >= early_norm * 0.1, \
            f"Gradient pattern unexpected: early={early_norm:.2e}, late={late_norm:.2e}"

    def test_output_contiguity(self, setup_data):
        """Test that output tensors are contiguous for efficient downstream ops."""
        F_seq = setup_data["F_seq"]
        Q_seq = setup_data["Q_seq"]
        P_init = setup_data["P_init"]

        P_parallel = parallel_covariance_scan(F_seq, Q_seq, P_init)
        assert P_parallel.is_contiguous(), "Parallel scan output not contiguous"

        P_blocked = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size=32)
        assert P_blocked.is_contiguous(), "Blocked scan output not contiguous"

    def test_numerical_stability_long_sequence(self):
        """Test numerical stability for long sequences."""
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        batch_size = 2
        seq_len = 500  # Long sequence
        state_dim = 15

        # Well-conditioned F matrices
        F_seq = torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0)
        F_seq = F_seq.repeat(batch_size, seq_len, 1, 1)
        F_seq += 0.005 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)

        # Small positive Q
        Q_seq = 0.001 * torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0)
        Q_seq = Q_seq.repeat(batch_size, seq_len, 1, 1)

        P_init = 0.01 * torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        P_out = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size=64)

        assert not torch.isnan(P_out).any(), "NaN in output"
        assert not torch.isinf(P_out).any(), "Inf in output"
        assert (P_out > 0).any(), "All zeros in output"


class TestESKFCacheIntegration:
    """Tests for ESKF cache-based parallel covariance computation."""

    def test_eskf_cache_parallel_matches_sequential(self):
        """Test that ESKF cache-based parallel scan matches sequential."""
        from model.ESKF import ErrorStateKalmanFilter

        torch.manual_seed(42)
        device = "cpu"
        batch_size, seq_len = 2, 50

        eskf = ErrorStateKalmanFilter(dt=0.02, device=device, use_tcn_zupt=True)
        eskf.init_cache(batch_size, seq_len)

        # Initialize state
        pos = torch.zeros(batch_size, 3, device=device)
        vel = torch.zeros(batch_size, 3, device=device)
        quat = torch.tensor([[1., 0., 0., 0.]], device=device).expand(batch_size, 4).contiguous()
        gyro_bias = torch.zeros(batch_size, 3, device=device)
        accel_bias = torch.zeros(batch_size, 3, device=device)
        P_error = torch.eye(15, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous() * 0.1

        # IMU data
        accel = torch.randn(batch_size, 3, device=device) * 0.1
        accel[:, 2] += 9.81
        gyro = torch.randn(batch_size, 3, device=device) * 0.01
        force = torch.rand(batch_size, 1, device=device)
        meas = torch.cat([accel, gyro], dim=-1)

        tcn_out = {
            'vel_corr': torch.randn(batch_size, 3, device=device) * 0.01,
            'covariance_R': torch.randn(batch_size, 6, device=device),
            'zupt_prob': torch.randn(batch_size, 1, device=device),
            'gravity_b': torch.tensor([[0., 0., -1.]], device=device).expand(batch_size, 3).contiguous(),
        }

        # Run sequential and collect P
        P_seq_list = []
        for t in range(seq_len):
            pos, vel, quat, gyro_bias, accel_bias, P_error, _ = eskf.forward(
                pos, vel, quat, gyro_bias, accel_bias, P_error,
                gyro, accel, force, meas, tcn_output=tcn_out
            )
            P_seq_list.append(P_error.clone())

        # Get parallel result
        cache = eskf.finalize_cache()
        P_parallel = eskf.parallel_covariance_from_cache(cache, block_size=16)
        P_sequential = torch.stack(P_seq_list, dim=1)

        diff = (P_parallel - P_sequential).abs().max().item()
        assert diff < 1e-4, f"ESKF cache parallel vs sequential diff too large: {diff:.2e}"


def run_all_sanity_checks():
    """Run comprehensive sanity checks (can be called independently)."""
    print("=" * 60)
    print("Running all sanity checks for Block-Parallel Scan")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Test 1: Parallel scan primitives
    print("\n1. Testing parallel scan primitives...")
    results = sanity_check_parallel_vs_sequential(device=device)
    for key, value in results.items():
        status = "PASS" if value == True else ("FAIL" if value == False else f"{value}")
        print(f"   {key}: {status}")

    # Test 2: Long sequence stability
    print("\n2. Testing numerical stability (seq_len=500)...")
    torch.manual_seed(42)
    try:
        batch_size, seq_len, state_dim = 2, 500, 15
        F_seq = torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0)
        F_seq = F_seq.repeat(batch_size, seq_len, 1, 1)
        F_seq += 0.005 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)
        Q_seq = 0.001 * torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        P_init = 0.01 * torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        P_out = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size=64)
        stable = not torch.isnan(P_out).any() and not torch.isinf(P_out).any()
        print(f"   stability: {'PASS' if stable else 'FAIL'}")
    except Exception as e:
        print(f"   stability: FAIL ({e})")

    # Test 3: ESKF integration
    print("\n3. Testing ESKF cache integration...")
    try:
        from model.ESKF import ErrorStateKalmanFilter
        eskf = ErrorStateKalmanFilter(dt=0.02, device=device, use_tcn_zupt=True)
        eskf.init_cache(2, 20)
        print("   cache_init: PASS")
    except Exception as e:
        print(f"   cache_init: FAIL ({e})")

    print("\n" + "=" * 60)
    print("All sanity checks completed!")


if __name__ == "__main__":
    run_all_sanity_checks()
