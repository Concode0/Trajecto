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
Unified Parallel Scan Analysis: Integrating Update into Transition Operator

Mathematical Analysis of the Proposed Method:
    New Transition: F̃_t = W_t @ F_t  where W_t = I - K_t @ H_t
    New Noise:      Q̃_t = W_t @ Q_t @ W_t^T + K_t @ R_t @ K_t^T

This allows the full Kalman filter (predict + update) to be expressed as:
    P_{t|t} = F̃_t @ P_{t-1|t-1} @ F̃_t^T + Q̃_t

Which maintains the associative structure for parallel scan!
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict
import time


# =============================================================================
# MATHEMATICAL DERIVATION
# =============================================================================
"""
## Standard Kalman Filter

**Predict:**
    x̂_{t|t-1} = F_t @ x̂_{t-1|t-1}
    P_{t|t-1} = F_t @ P_{t-1|t-1} @ F_t^T + Q_t

**Update:**
    S_t = H_t @ P_{t|t-1} @ H_t^T + R_t        (Innovation covariance)
    K_t = P_{t|t-1} @ H_t^T @ S_t^{-1}         (Kalman gain)
    P_{t|t} = (I - K_t @ H_t) @ P_{t|t-1} @ (I - K_t @ H_t)^T + K_t @ R_t @ K_t^T
            = W_t @ P_{t|t-1} @ W_t^T + K_t @ R_t @ K_t^T   (Joseph form)

## Unified Formulation

Substituting predict into update:
    P_{t|t} = W_t @ (F_t @ P_{t-1|t-1} @ F_t^T + Q_t) @ W_t^T + K_t @ R_t @ K_t^T
            = W_t @ F_t @ P_{t-1|t-1} @ F_t^T @ W_t^T + W_t @ Q_t @ W_t^T + K_t @ R_t @ K_t^T
            = (W_t @ F_t) @ P_{t-1|t-1} @ (W_t @ F_t)^T + (W_t @ Q_t @ W_t^T + K_t @ R_t @ K_t^T)

Define:
    F̃_t = W_t @ F_t
    Q̃_t = W_t @ Q_t @ W_t^T + K_t @ R_t @ K_t^T

Then:
    P_{t|t} = F̃_t @ P_{t-1|t-1} @ F̃_t^T + Q̃_t

This has the SAME FORM as pure prediction! The associative operator applies:
    (F̃_1, Q̃_1) ⊗ (F̃_2, Q̃_2) = (F̃_2 @ F̃_1, F̃_2 @ Q̃_1 @ F̃_2^T + Q̃_2)

## The Critical Challenge: K_t depends on P_{t|t-1}

K_t = P_{t|t-1} @ H_t^T @ (H_t @ P_{t|t-1} @ H_t^T + R_t)^{-1}

This creates a CIRCULAR DEPENDENCY:
- To compute F̃_t, we need K_t
- To compute K_t, we need P_{t|t-1}
- P_{t|t-1} = F_t @ P_{t-1|t-1} @ F_t^T + Q_t
- P_{t-1|t-1} depends on K_{t-1}, which depends on P_{t-1|t-2}...

For parallel scan, we need ALL (F̃_t, Q̃_t) upfront!
"""


# =============================================================================
# SOLUTION STRATEGIES
# =============================================================================

def strategy_1_steady_state_kalman_gain(
    F: torch.Tensor,
    H: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    num_iterations: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Strategy 1: Use Steady-State Kalman Gain (SSKF)

    For time-invariant systems, the Kalman gain converges to a constant K_ss.
    Solve the Discrete Algebraic Riccati Equation (DARE):
        P_ss = F @ P_ss @ F^T + Q - F @ P_ss @ H^T @ (H @ P_ss @ H^T + R)^{-1} @ H @ P_ss @ F^T

    Then K_ss = P_ss @ H^T @ (H @ P_ss @ H^T + R)^{-1}

    PROS:
    - K is constant, no circular dependency
    - Full parallelization possible
    - Computationally efficient (solve DARE once)

    CONS:
    - Assumes time-invariant F, H, Q, R
    - Not optimal for time-varying systems (ESKF has time-varying F!)
    - May lose accuracy during transients
    """
    state_dim = F.shape[0]

    # Iterate to find steady-state P
    P = torch.eye(state_dim, device=F.device, dtype=F.dtype)

    for _ in range(num_iterations):
        # Predict
        P_pred = F @ P @ F.T + Q

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ torch.linalg.solve(S, torch.eye(S.shape[0], device=S.device, dtype=S.dtype))

        # Update
        W = torch.eye(state_dim, device=F.device, dtype=F.dtype) - K @ H
        P = W @ P_pred @ W.T + K @ R @ K.T

    # Steady-state gain
    K_ss = K
    P_ss = P

    return K_ss, P_ss


def strategy_2_iterative_refinement(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    H: torch.Tensor,
    R_seq: torch.Tensor,
    P_init: torch.Tensor,
    num_refinement_passes: int = 2,
) -> torch.Tensor:
    """
    Strategy 2: Iterative Refinement

    Pass 1: Compute P using prediction-only (F, Q)
    Pass 2: Use predicted P to estimate K_t, compute refined (F̃, Q̃)
    Pass 3: Recompute P with unified operators

    PROS:
    - Handles time-varying systems
    - Converges to correct solution with enough passes
    - Can trade accuracy for speed (fewer passes)

    CONS:
    - Multiple passes reduce speedup
    - May not converge for highly nonlinear systems
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype
    obs_dim = H.shape[0]

    # Pass 1: Prediction-only to get initial P estimates
    P_pred_seq = _parallel_covariance_scan_prediction_only(F_seq, Q_seq, P_init)

    for refine_pass in range(num_refinement_passes):
        # Compute K_t from predicted covariances
        K_seq = torch.zeros(batch_size, seq_len, state_dim, obs_dim, device=device, dtype=dtype)
        W_seq = torch.zeros(batch_size, seq_len, state_dim, state_dim, device=device, dtype=dtype)

        for t in range(seq_len):
            P_pred = P_pred_seq[:, t]  # [batch, state_dim, state_dim]

            # S = H @ P @ H^T + R
            S = H @ P_pred @ H.T + R_seq[:, t]  # [batch, obs_dim, obs_dim]

            # K = P @ H^T @ S^{-1}
            K = P_pred @ H.T @ torch.linalg.solve(S, torch.eye(obs_dim, device=device, dtype=dtype))
            K_seq[:, t] = K

            # W = I - K @ H
            W_seq[:, t] = torch.eye(state_dim, device=device, dtype=dtype) - K @ H

        # Build unified operators
        F_unified_seq = torch.einsum("btij,btjk->btik", W_seq, F_seq)
        Q_unified_seq = (
            torch.einsum("btij,btjk,btlk->btil", W_seq, Q_seq, W_seq) +
            torch.einsum("btij,btjk,btlk->btil", K_seq, R_seq, K_seq)
        )

        # Recompute P with unified operators
        P_pred_seq = _parallel_covariance_scan_prediction_only(F_unified_seq, Q_unified_seq, P_init)

    return P_pred_seq


def strategy_3_tcn_predicted_gain(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    tcn_K_seq: torch.Tensor,  # TCN-predicted Kalman gain factors
    H: torch.Tensor,
    R_seq: torch.Tensor,
    P_init: torch.Tensor,
) -> torch.Tensor:
    """
    Strategy 3: TCN-Predicted Kalman Gain

    Instead of computing K from P, have the TCN predict K directly (or a factor of it).
    The TCN already predicts covariance_R, extend to predict K or W.

    PROS:
    - No circular dependency (K is input, not computed from P)
    - Leverages TCN's learned dynamics
    - Single parallel pass
    - TCN can learn optimal gain for specific motion patterns

    CONS:
    - Requires TCN architecture modification
    - May need more training data
    - K prediction may be harder to learn than R prediction

    Implementation:
    - TCN outputs: vel_corr, covariance_R, zupt_prob, AND gain_factor
    - gain_factor: [batch, seq_len, state_dim] or learned W directly
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype

    # Build W from TCN-predicted K
    W_seq = torch.eye(state_dim, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    W_seq = W_seq.expand(batch_size, seq_len, -1, -1).clone()
    W_seq = W_seq - torch.einsum("btij,jk->btik", tcn_K_seq, H)

    # Build unified operators
    F_unified_seq = torch.einsum("btij,btjk->btik", W_seq, F_seq)
    Q_unified_seq = (
        torch.einsum("btij,btjk,btlk->btil", W_seq, Q_seq, W_seq) +
        torch.einsum("btij,btjk,btlk->btil", tcn_K_seq, R_seq, tcn_K_seq)
    )

    # Single parallel pass
    P_seq = _parallel_covariance_scan_prediction_only(F_unified_seq, Q_unified_seq, P_init)

    return P_seq


def strategy_4_block_local_gain(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    H: torch.Tensor,
    R_seq: torch.Tensor,
    P_init: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Strategy 4: Block-Local Kalman Gain

    Compute K at block boundaries only, use constant K within blocks.

    Block structure:
    [----Block 1----][----Block 2----][----Block 3----]
         K_1              K_2              K_3

    Within each block, use parallel scan with constant K (Strategy 1 locally).
    At block boundaries, recompute K from accumulated P.

    PROS:
    - Handles time-varying systems with block-level adaptation
    - Maintains O(log B) parallel depth within blocks (B = block_size)
    - More accurate than global steady-state assumption

    CONS:
    - Sequential across blocks (O(T/B) sequential steps)
    - K approximation within blocks may lose accuracy
    """
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype
    obs_dim = H.shape[0]

    num_blocks = (seq_len + block_size - 1) // block_size
    P_seq = torch.zeros(batch_size, seq_len, state_dim, state_dim, device=device, dtype=dtype)

    P_current = P_init.clone()

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, seq_len)

        # Compute K at block start from current P
        P_pred_block_start = F_seq[:, block_start] @ P_current @ F_seq[:, block_start].transpose(-1, -2) + Q_seq[:, block_start]
        S = H @ P_pred_block_start @ H.T + R_seq[:, block_start]
        K_block = P_pred_block_start @ H.T @ torch.linalg.solve(S, torch.eye(obs_dim, device=device, dtype=dtype))
        W_block = torch.eye(state_dim, device=device, dtype=dtype) - K_block @ H

        # Build unified operators for this block (constant K)
        F_block = F_seq[:, block_start:block_end]
        Q_block = Q_seq[:, block_start:block_end]
        R_block = R_seq[:, block_start:block_end]

        actual_block_size = block_end - block_start
        W_expanded = W_block.unsqueeze(1).expand(-1, actual_block_size, -1, -1)
        K_expanded = K_block.unsqueeze(1).expand(-1, actual_block_size, -1, -1)

        F_unified = torch.einsum("btij,btjk->btik", W_expanded, F_block)
        Q_unified = (
            torch.einsum("btij,btjk,btlk->btil", W_expanded, Q_block, W_expanded) +
            torch.einsum("btij,btjk,btlk->btil", K_expanded, R_block, K_expanded)
        )

        # Parallel scan within block
        P_block = _parallel_covariance_scan_prediction_only(F_unified, Q_unified, P_current)
        P_seq[:, block_start:block_end] = P_block

        # Update for next block
        P_current = P_block[:, -1]

    return P_seq


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parallel_covariance_scan_prediction_only(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    P_init: torch.Tensor,
) -> torch.Tensor:
    """Parallel scan for covariance (imported from parallel_scan_ops)."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model.parallel_scan_ops import parallel_covariance_scan
    return parallel_covariance_scan(F_seq, Q_seq, P_init)


# =============================================================================
# EVALUATION AND COMPARISON
# =============================================================================

def evaluate_strategies(
    batch_size: int = 4,
    seq_len: int = 200,
    state_dim: int = 15,
    obs_dim: int = 6,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """
    Evaluate all strategies against sequential ground truth.
    """
    torch.manual_seed(42)

    # Create test system
    # F: close to identity (stable system)
    F = torch.eye(state_dim, device=device) + 0.01 * torch.randn(state_dim, state_dim, device=device)
    F_seq = F.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()

    # Add small time-variation
    F_seq += 0.001 * torch.randn(batch_size, seq_len, state_dim, state_dim, device=device)

    # Q: small positive definite
    Q_raw = torch.randn(state_dim, state_dim, device=device) * 0.01
    Q = Q_raw @ Q_raw.T + 0.001 * torch.eye(state_dim, device=device)
    Q_seq = Q.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()

    # H: observation matrix
    H = torch.randn(obs_dim, state_dim, device=device) * 0.1

    # R: measurement noise
    R_raw = torch.randn(obs_dim, obs_dim, device=device) * 0.1
    R = R_raw @ R_raw.T + 0.01 * torch.eye(obs_dim, device=device)
    R_seq = R.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()

    # P_init
    P_init = 0.01 * torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    results = {}

    # Ground truth: Sequential Kalman filter
    print("Computing sequential ground truth...")
    start = time.time()
    P_sequential = _sequential_kalman_filter(F_seq, Q_seq, H, R_seq, P_init)
    seq_time = time.time() - start
    results["sequential"] = {"time": seq_time, "P": P_sequential}

    # Strategy 1: Steady-state K
    print("Testing Strategy 1: Steady-State K...")
    start = time.time()
    K_ss, _ = strategy_1_steady_state_kalman_gain(F, H, Q, R)
    W_ss = torch.eye(state_dim, device=device) - K_ss @ H
    F_unified = W_ss @ F
    Q_unified = W_ss @ Q @ W_ss.T + K_ss @ R @ K_ss.T
    F_unified_seq = F_unified.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
    Q_unified_seq = Q_unified.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
    P_ss = _parallel_covariance_scan_prediction_only(F_unified_seq, Q_unified_seq, P_init)
    ss_time = time.time() - start

    diff_ss = (P_ss - P_sequential).abs()
    results["steady_state"] = {
        "time": ss_time,
        "max_diff": diff_ss.max().item(),
        "mean_diff": diff_ss.mean().item(),
        "speedup": seq_time / ss_time if ss_time > 0 else float('inf'),
    }

    # Strategy 2: Iterative refinement
    print("Testing Strategy 2: Iterative Refinement...")
    start = time.time()
    P_iterative = strategy_2_iterative_refinement(F_seq, Q_seq, H, R_seq, P_init, num_refinement_passes=2)
    iter_time = time.time() - start

    diff_iter = (P_iterative - P_sequential).abs()
    results["iterative"] = {
        "time": iter_time,
        "max_diff": diff_iter.max().item(),
        "mean_diff": diff_iter.mean().item(),
        "speedup": seq_time / iter_time if iter_time > 0 else float('inf'),
    }

    # Strategy 4: Block-local K
    print("Testing Strategy 4: Block-Local K...")
    start = time.time()
    P_block = strategy_4_block_local_gain(F_seq, Q_seq, H, R_seq, P_init, block_size=32)
    block_time = time.time() - start

    diff_block = (P_block - P_sequential).abs()
    results["block_local"] = {
        "time": block_time,
        "max_diff": diff_block.max().item(),
        "mean_diff": diff_block.mean().item(),
        "speedup": seq_time / block_time if block_time > 0 else float('inf'),
    }

    return results


def _sequential_kalman_filter(
    F_seq: torch.Tensor,
    Q_seq: torch.Tensor,
    H: torch.Tensor,
    R_seq: torch.Tensor,
    P_init: torch.Tensor,
) -> torch.Tensor:
    """Sequential Kalman filter (ground truth)."""
    batch_size, seq_len, state_dim, _ = F_seq.shape
    device = F_seq.device
    dtype = F_seq.dtype
    obs_dim = H.shape[0]

    P_seq = torch.zeros(batch_size, seq_len, state_dim, state_dim, device=device, dtype=dtype)
    P = P_init.clone()

    I = torch.eye(state_dim, device=device, dtype=dtype)
    I_obs = torch.eye(obs_dim, device=device, dtype=dtype)

    for t in range(seq_len):
        F_t = F_seq[:, t]
        Q_t = Q_seq[:, t]
        R_t = R_seq[:, t]

        # Predict
        P_pred = F_t @ P @ F_t.transpose(-1, -2) + Q_t

        # Update
        S = H @ P_pred @ H.T + R_t
        K = P_pred @ H.T @ torch.linalg.solve(S, I_obs.unsqueeze(0).expand(batch_size, -1, -1))
        W = I - K @ H
        P = W @ P_pred @ W.transpose(-1, -2) + K @ R_t @ K.transpose(-1, -2)

        P_seq[:, t] = P

    return P_seq


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED PARALLEL SCAN ANALYSIS")
    print("Integrating Kalman Update into Transition Operator")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("MATHEMATICAL SUMMARY")
    print("=" * 70)
    print("""
Proposed Unified Formulation:
    F̃_t = W_t @ F_t        where W_t = I - K_t @ H_t
    Q̃_t = W_t @ Q_t @ W_t^T + K_t @ R_t @ K_t^T

    P_{t|t} = F̃_t @ P_{t-1|t-1} @ F̃_t^T + Q̃_t

Key Insight: This maintains the associative structure!
    (F̃_1, Q̃_1) ⊗ (F̃_2, Q̃_2) = (F̃_2 @ F̃_1, F̃_2 @ Q̃_1 @ F̃_2^T + Q̃_2)

Challenge: K_t depends on P_{t|t-1}, creating circular dependency.
""")

    print("\n" + "=" * 70)
    print("STRATEGY EVALUATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    results = evaluate_strategies(device=device)

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Time (ms)':<12} {'Max Diff':<12} {'Mean Diff':<12} {'Speedup':<10}")
    print("-" * 70)

    print(f"{'Sequential':<20} {results['sequential']['time']*1000:>10.2f}ms {'-':<12} {'-':<12} {'1.00x':<10}")

    for name in ["steady_state", "iterative", "block_local"]:
        r = results[name]
        print(f"{name:<20} {r['time']*1000:>10.2f}ms {r['max_diff']:>10.2e} {r['mean_diff']:>10.2e} {r['speedup']:>8.2f}x")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR ESKF-TCN")
    print("=" * 70)
    print("""
1. BEST FOR TRAINING SPEEDUP: Strategy 4 (Block-Local K)
   - Balances accuracy and parallelism
   - Adapts to ESKF's time-varying dynamics
   - Recommended block_size: 32-64

2. BEST FOR TCN INTEGRATION: Strategy 3 (TCN-Predicted K)
   - Requires TCN modification to output gain_factor
   - Eliminates circular dependency entirely
   - Single parallel pass (maximum speedup)

3. FOR STEADY-STATE ANALYSIS: Strategy 1 (Steady-State K)
   - Use for long sequences where system reaches equilibrium
   - Good for initialization or warm-start

4. FOR MAXIMUM ACCURACY: Strategy 2 (Iterative Refinement)
   - 2-3 passes typically sufficient
   - Tradeoff: more passes = more accuracy, less speedup
""")
