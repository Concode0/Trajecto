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
Parallel Scan Operations for ESKF Covariance Propagation

This module implements efficient parallel scan (prefix sum) operations specifically
for Kalman filter covariance propagation: P[t+1] = F[t] @ P[t] @ F[t]^T + Q[t]

Key insight: The covariance update can be expressed as an associative operation:
    (F1, Q1) ⊗ (F2, Q2) = (F2 @ F1, F2 @ Q1 @ F2^T + Q2)

This allows computing all P matrices in O(log T) parallel depth instead of O(T) sequential.

References:
- Blelloch, G.E. (1990). "Prefix Sums and Their Applications"
- Sarkka & Garcia-Fernandez (2021). "Temporal Parallelization of Bayesian Smoothers"
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


def _combine_covariance(
    elem1: Tuple[torch.Tensor, torch.Tensor],
    elem2: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative operator for covariance propagation: P[t+1] = F @ P @ F^T + Q

    The key insight is that we can represent the affine transformation as a tuple
    (F, Q) and combine them associatively:

    (F2, Q2) ⊗ (F1, Q1) = (F2 @ F1, F2 @ Q1 @ F2^T + Q2)

    This allows computing prefix products in parallel.

    Args:
        elem1: (F_1, Q_1) - earlier element
        elem2: (F_2, Q_2) - later element

    Returns:
        Combined element (F_combined, Q_combined)
    """
    F1, Q1 = elem1
    F2, Q2 = elem2

    # F_combined = F_2 @ F_1
    F_combined = torch.einsum("...ij,...jk->...ik", F2, F1)

    # Q_combined = F_2 @ Q_1 @ F_2^T + Q_2
    Q_combined = torch.einsum("...ij,...jk,...lk->...il", F2, Q1, F2) + Q2

    return (F_combined, Q_combined)


def sequential_covariance_propagation(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    P_init: torch.Tensor,
) -> torch.Tensor:
    """
    Sequential covariance propagation (baseline for comparison).

    Computes P[t+1] = F[t] @ P[t] @ F[t]^T + Q[t] for t = 0, ..., T-1

    Args:
        F_seq: Sequence of transition Jacobians [batch, seq_len, state_dim, state_dim]
        Q_seq: Sequence of process noise matrices [batch, seq_len, state_dim, state_dim]
        P_init: Initial covariance [batch, state_dim, state_dim]

    Returns:
        P_seq: All covariance matrices [batch, seq_len, state_dim, state_dim]
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype

    P_seq = torch.zeros(batch_size, seq_len, state_dim, state_dim, device=device, dtype=dtype)
    P = P_init.clone()

    for t in range(seq_len):
        # P[t+1] = F @ P @ F^T + Q
        F_t = F_seq[:, t]  # [batch, state_dim, state_dim]
        Q_t = Q_seq[:, t]
        P = torch.einsum("bij,bjk,blk->bil", F_t, P, F_t) + Q_t
        P_seq[:, t] = P

    return P_seq


def parallel_covariance_scan(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    P_init: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scan for covariance propagation: P[t+1] = F[t] @ P[t] @ F[t]^T + Q[t]

    Uses the work-efficient iterative doubling algorithm adapted for covariance matrices.

    Args:
        F_seq: Sequence of transition Jacobians [batch, seq_len, state_dim, state_dim]
        Q_seq: Sequence of process noise matrices [batch, seq_len, state_dim, state_dim]
        P_init: Initial covariance [batch, state_dim, state_dim]

    Returns:
        P_seq: All covariance matrices [batch, seq_len, state_dim, state_dim]

    Complexity:
        Work: O(T * state_dim^3) - same as sequential
        Depth: O(log(T) * state_dim^3) - parallelizable
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype

    # For short sequences or CPU, fall back to sequential
    if seq_len <= 32 or not F_seq.is_cuda:
        return sequential_covariance_propagation(F_seq, Q_seq, P_init)

    # Adjust first element to include initial covariance contribution
    # P_1 = F_0 @ P_0 @ F_0^T + Q_0
    # So Q_0_adjusted = F_0 @ P_init @ F_0^T + Q_0
    Q_adjusted = Q_seq.clone()
    F_0 = F_seq[:, 0]
    Q_adjusted[:, 0] = torch.einsum("bij,bjk,blk->bil", F_0, P_init, F_0) + Q_seq[:, 0]

    return _parallel_covariance_scan_gpu(F_seq, Q_adjusted)


def _parallel_covariance_scan_gpu(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
) -> torch.Tensor:
    """
    GPU-optimized parallel scan using iterative doubling.

    This implements the parallel prefix algorithm where in each iteration,
    element i is combined with element i - 2^k for increasing k.

    The combination operator is:
        (F_target, Q_target) ⊗ (F_source, Q_source) =
            (F_target @ F_source, F_target @ Q_source @ F_target^T + Q_target)

    BPTT Note: Uses torch.scatter for gradient-safe indexed assignment.
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype

    # Working copies
    F_work = F_seq.clone()
    Q_work = Q_seq.clone()

    # Number of iterations = ceil(log2(seq_len))
    num_iters = int(math.ceil(math.log2(seq_len)))

    for d in range(num_iters):
        stride = 2 ** d

        if stride >= seq_len:
            break

        # Indices that get updated
        update_indices = torch.arange(stride, seq_len, device=device)
        source_indices = update_indices - stride
        num_updates = update_indices.shape[0]

        # Get elements to combine (ensure contiguity for einsum performance)
        F_source = F_work[:, source_indices].contiguous()  # [B, num_updates, D, D]
        Q_source = Q_work[:, source_indices].contiguous()  # [B, num_updates, D, D]
        F_target = F_work[:, update_indices].contiguous()  # [B, num_updates, D, D]
        Q_target = Q_work[:, update_indices].contiguous()  # [B, num_updates, D, D]

        # Combine: (F_target, Q_target) ⊗ (F_source, Q_source)
        # F_new = F_target @ F_source
        F_new = torch.einsum("bnij,bnjk->bnik", F_target, F_source)

        # Q_new = F_target @ Q_source @ F_target^T + Q_target
        Q_new = torch.einsum("bnij,bnjk,bnlk->bnil", F_target, Q_source, F_target) + Q_target

        # Update working arrays using scatter for gradient-safe assignment
        # This preserves autograd graph unlike in-place slice assignment
        indices_expanded = update_indices.view(1, num_updates, 1, 1).expand(
            batch_size, num_updates, state_dim, state_dim
        )
        F_work = F_work.scatter(1, indices_expanded, F_new)
        Q_work = Q_work.scatter(1, indices_expanded, Q_new)

    # Q_work now contains all covariance matrices P[1], P[2], ..., P[T]
    return Q_work


def parallel_covariance_scan_blocked(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    P_init: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Block-wise parallel covariance scan for memory efficiency.

    Divides the sequence into blocks, processes each block with parallel scan,
    and combines block results using the associative operator.

    This reduces memory usage from O(T * D^2) to O(block_size * D^2) for intermediate
    results while maintaining O(log T) parallel depth within blocks.

    Args:
        F_seq: Sequence of transition Jacobians [batch, seq_len, state_dim, state_dim]
        Q_seq: Sequence of process noise matrices [batch, seq_len, state_dim, state_dim]
        P_init: Initial covariance [batch, state_dim, state_dim]
        block_size: Size of each processing block (default 64)

    Returns:
        P_seq: All covariance matrices [batch, seq_len, state_dim, state_dim]
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype

    # Calculate number of blocks
    num_blocks = (seq_len + block_size - 1) // block_size

    # Output tensor
    P_seq = torch.zeros(batch_size, seq_len, state_dim, state_dim, device=device, dtype=dtype)

    # Current covariance (updated after each block)
    P_current = P_init.clone()

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, seq_len)
        actual_block_size = block_end - block_start

        # Extract block data
        F_block = F_seq[:, block_start:block_end]
        Q_block = Q_seq[:, block_start:block_end]

        # Compute covariances for this block
        if actual_block_size <= 32 or not F_seq.is_cuda:
            # Sequential for small blocks
            P_block = sequential_covariance_propagation(F_block, Q_block, P_current)
        else:
            # Parallel scan for larger blocks
            P_block = parallel_covariance_scan(F_block, Q_block, P_current)

        # Store results
        P_seq[:, block_start:block_end] = P_block

        # Update current covariance for next block
        P_current = P_block[:, -1]

    return P_seq


class ParallelCovarianceModule(nn.Module):
    """
    PyTorch module wrapper for parallel covariance propagation.

    Provides a differentiable interface for use in training pipelines.
    """

    def __init__(
        self,
        state_dim: int = 15,
        block_size: int = 64,
        use_double_precision: bool = False,
    ):
        """
        Args:
            state_dim: Dimension of the error state (default 15 for ESKF)
            block_size: Block size for blocked scan (default 64)
            use_double_precision: If True, compute in float64 for better precision
        """
        super().__init__()
        self.state_dim = state_dim
        self.block_size = block_size
        self.use_double_precision = use_double_precision

    def forward(
        self,
        F_seq: torch.Tensor,
        Q_seq: torch.Tensor,
        P_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute all covariance matrices using parallel scan.

        Args:
            F_seq: Transition Jacobians [batch, seq_len, state_dim, state_dim]
            Q_seq: Process noise matrices [batch, seq_len, state_dim, state_dim]
            P_init: Initial covariance [batch, state_dim, state_dim]

        Returns:
            P_seq: All covariances [batch, seq_len, state_dim, state_dim]
        """
        original_dtype = F_seq.dtype

        if self.use_double_precision:
            F_seq = F_seq.double()
            Q_seq = Q_seq.double()
            P_init = P_init.double()

        P_seq = parallel_covariance_scan_blocked(
            F_seq, Q_seq, P_init, self.block_size
        )

        if self.use_double_precision:
            P_seq = P_seq.to(original_dtype)

        return P_seq


def verify_associativity(
    F1: torch.Tensor,
    Q1: torch.Tensor,
    F2: torch.Tensor,
    Q2: torch.Tensor,
    F3: torch.Tensor,
    Q3: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> bool:
    """
    Verify that the covariance operator is associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)

    This is a sanity check to confirm mathematical correctness.

    Args:
        F1, Q1: First element
        F2, Q2: Second element
        F3, Q3: Third element
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if associative property holds within tolerance
    """
    # (a ⊗ b) ⊗ c
    ab = _combine_covariance((F1, Q1), (F2, Q2))
    ab_c = _combine_covariance(ab, (F3, Q3))

    # a ⊗ (b ⊗ c)
    bc = _combine_covariance((F2, Q2), (F3, Q3))
    a_bc = _combine_covariance((F1, Q1), bc)

    F_match = torch.allclose(ab_c[0], a_bc[0], rtol=rtol, atol=atol)
    Q_match = torch.allclose(ab_c[1], a_bc[1], rtol=rtol, atol=atol)

    return F_match and Q_match


def sanity_check_parallel_vs_sequential(
    batch_size: int = 4,
    seq_len: int = 100,
    state_dim: int = 15,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Comprehensive sanity check comparing parallel and sequential implementations.

    Args:
        batch_size: Batch size for test
        seq_len: Sequence length for test
        state_dim: State dimension
        rtol: Relative tolerance for allclose
        atol: Absolute tolerance for allclose
        device: Device to run on

    Returns:
        Dictionary with test results
    """
    torch.manual_seed(42)

    # Generate random but well-conditioned test data
    # F matrices should be close to identity for numerical stability
    F_seq = torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    F_seq += 0.01 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)

    # Q matrices should be positive semi-definite
    Q_raw = torch.randn(batch_size, seq_len, state_dim, state_dim, device=device) * 0.1
    Q_seq = torch.einsum("btij,btkj->btik", Q_raw, Q_raw)  # Q = Q_raw @ Q_raw^T

    # Initial covariance (positive definite)
    P_init_raw = torch.randn(batch_size, state_dim, state_dim, device=device) * 0.1
    P_init = torch.einsum("bij,bkj->bik", P_init_raw, P_init_raw) + 0.01 * torch.eye(state_dim, device=device)

    results = {}

    # Test 1: Sequential computation
    P_sequential = sequential_covariance_propagation(F_seq, Q_seq, P_init)

    # Test 2: Parallel computation
    P_parallel = parallel_covariance_scan(F_seq, Q_seq, P_init)

    # Test 3: Blocked parallel computation
    P_blocked = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size=32)

    # Compare results
    results["parallel_vs_sequential"] = torch.allclose(P_parallel, P_sequential, rtol=rtol, atol=atol)
    results["blocked_vs_sequential"] = torch.allclose(P_blocked, P_sequential, rtol=rtol, atol=atol)

    if not results["parallel_vs_sequential"]:
        diff = (P_parallel - P_sequential).abs()
        results["parallel_max_diff"] = diff.max().item()
        results["parallel_mean_diff"] = diff.mean().item()

    if not results["blocked_vs_sequential"]:
        diff = (P_blocked - P_sequential).abs()
        results["blocked_max_diff"] = diff.max().item()
        results["blocked_mean_diff"] = diff.mean().item()

    # Test 4: Verify associativity
    F1, F2, F3 = F_seq[:, 0], F_seq[:, 1], F_seq[:, 2]
    Q1, Q2, Q3 = Q_seq[:, 0], Q_seq[:, 1], Q_seq[:, 2]
    results["associativity"] = verify_associativity(F1, Q1, F2, Q2, F3, Q3)

    # Test 5: Gradient flow (basic)
    F_seq_grad = F_seq.clone().requires_grad_(True)
    Q_seq_grad = Q_seq.clone().requires_grad_(True)
    P_init_grad = P_init.clone().requires_grad_(True)

    P_out = parallel_covariance_scan(F_seq_grad, Q_seq_grad, P_init_grad)
    loss = P_out.sum()
    loss.backward()

    results["gradient_flow_F"] = F_seq_grad.grad is not None and not torch.isnan(F_seq_grad.grad).any()
    results["gradient_flow_Q"] = Q_seq_grad.grad is not None and not torch.isnan(Q_seq_grad.grad).any()
    results["gradient_flow_P_init"] = P_init_grad.grad is not None and not torch.isnan(P_init_grad.grad).any()

    # Test 6: BPTT gradient correctness - verify gradients from late timesteps reach early timesteps
    # This is the critical test for scatter-based implementation
    F_seq_bptt = F_seq.clone().requires_grad_(True)
    Q_seq_bptt = Q_seq.clone().requires_grad_(True)
    P_init_bptt = P_init.clone().requires_grad_(True)

    P_out_bptt = parallel_covariance_scan(F_seq_bptt, Q_seq_bptt, P_init_bptt)
    # Loss on LAST timestep only - this must propagate back to ALL early timesteps
    loss_final = P_out_bptt[:, -1].sum()
    loss_final.backward()

    # Gradient from final output should reach early F and Q matrices
    results["bptt_gradient_F_early"] = (
        F_seq_bptt.grad is not None and
        F_seq_bptt.grad[:, 0].abs().max() > 1e-10  # Early F should have non-zero gradient
    )
    results["bptt_gradient_Q_early"] = (
        Q_seq_bptt.grad is not None and
        Q_seq_bptt.grad[:, 0].abs().max() > 1e-10  # Early Q should have non-zero gradient
    )

    # Verify gradient magnitude increases towards later timesteps (expected for cumulative scan)
    if F_seq_bptt.grad is not None:
        early_grad_norm = F_seq_bptt.grad[:, :seq_len//4].abs().mean().item()
        late_grad_norm = F_seq_bptt.grad[:, -seq_len//4:].abs().mean().item()
        results["bptt_gradient_pattern"] = late_grad_norm >= early_grad_norm * 0.1  # Late should be larger or similar
    else:
        results["bptt_gradient_pattern"] = False

    # Summary
    results["all_passed"] = all([
        results["parallel_vs_sequential"],
        results["blocked_vs_sequential"],
        results["associativity"],
        results["gradient_flow_F"],
        results["gradient_flow_Q"],
        results["gradient_flow_P_init"],
        results["bptt_gradient_F_early"],
        results["bptt_gradient_Q_early"],
        results["bptt_gradient_pattern"],
    ])

    return results


def test_memory_contiguity(
    batch_size: int = 4,
    seq_len: int = 100,
    state_dim: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Test that output tensors remain contiguous through operations."""
    torch.manual_seed(42)

    F_seq = torch.eye(state_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    F_seq += 0.01 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)

    Q_raw = torch.randn(batch_size, seq_len, state_dim, state_dim, device=device) * 0.1
    Q_seq = torch.einsum("btij,btkj->btik", Q_raw, Q_raw)

    P_init_raw = torch.randn(batch_size, state_dim, state_dim, device=device) * 0.1
    P_init = torch.einsum("bij,bkj->bik", P_init_raw, P_init_raw) + 0.01 * torch.eye(state_dim, device=device)

    results = {}

    # Test parallel scan output contiguity
    P_parallel = parallel_covariance_scan(F_seq, Q_seq, P_init)
    results["parallel_output_contiguous"] = P_parallel.is_contiguous()

    # Test blocked scan output contiguity
    P_blocked = parallel_covariance_scan_blocked(F_seq, Q_seq, P_init, block_size=32)
    results["blocked_output_contiguous"] = P_blocked.is_contiguous()

    # Test ParallelCovarianceModule output
    module = ParallelCovarianceModule(state_dim=state_dim, block_size=32)
    P_module = module(F_seq, Q_seq, P_init)
    results["module_output_contiguous"] = P_module.is_contiguous()

    results["all_passed"] = all([
        results["parallel_output_contiguous"],
        results["blocked_output_contiguous"],
        results["module_output_contiguous"],
    ])

    return results
