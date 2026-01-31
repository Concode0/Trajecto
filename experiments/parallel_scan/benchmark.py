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
Benchmark Suite for Parallel Scan Experiments

This module provides comprehensive benchmarks comparing:
1. Sequential vs parallel scan implementations
2. SSM Filter vs ESKF training speed
3. Different sequence lengths and batch sizes
4. CPU vs CUDA vs MPS backends

Usage:
    python -m experiments.parallel_scan.benchmark

    # Or specific benchmarks:
    python -m experiments.parallel_scan.benchmark --scan-only
    python -m experiments.parallel_scan.benchmark --filter-only
    python -m experiments.parallel_scan.benchmark --sequence-lengths 100 500 1000 2000
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .scan_ops import sequential_scan, parallel_scan, _parallel_scan_gpu
from .linear_ssm import S4DKernel, LinearSSM, S5Layer
from .ssm_filter import SSMFilter, HybridSSMESKF, ParallelIntegrator


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def warmup_device(device: torch.device, iterations: int = 10) -> None:
    """Warm up the device to get accurate timings."""
    x = torch.randn(32, 100, 64, device=device)
    for _ in range(iterations):
        _ = x @ x.transpose(-1, -2)
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, device: torch.device, sync: bool = True):
        self.device = device
        self.sync = sync
        self.elapsed = 0.0

    def __enter__(self):
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync and self.device.type == "cuda":
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start


def benchmark_scan_operations(
    batch_sizes: List[int] = [1, 4, 8, 16],
    seq_lengths: List[int] = [100, 500, 1000, 2000],
    state_dim: int = 16,
    input_dim: int = 19,
    num_runs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark sequential vs parallel scan operations.

    Args:
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        state_dim: State space dimension
        input_dim: Input dimension
        num_runs: Number of runs for averaging
        device: Device to run on

    Returns:
        Dictionary of benchmark results
    """
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Scan Operation Benchmarks (device: {device})")
    print(f"{'='*60}")

    results = {}
    warmup_device(device)

    # Fixed A matrix
    A = torch.eye(state_dim, device=device) * 0.99  # Stable dynamics
    B = torch.randn(state_dim, input_dim, device=device) * 0.1

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"batch={batch_size}, seq={seq_len}"
            print(f"\n{key}")
            print("-" * 40)

            # Generate test data
            inputs = torch.randn(batch_size, seq_len, input_dim, device=device)
            initial_state = torch.zeros(batch_size, state_dim, device=device)

            # Build A_seq and b_seq for parallel scan
            A_seq = A.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()
            b_seq = torch.einsum("ij,btj->bti", B, inputs)

            # Benchmark sequential scan
            seq_times = []
            for _ in range(num_runs):
                with Timer(device) as t:
                    _ = sequential_scan(inputs, initial_state, A, B)
                seq_times.append(t.elapsed)
            seq_mean = sum(seq_times) / len(seq_times)
            seq_std = (sum((x - seq_mean) ** 2 for x in seq_times) / len(seq_times)) ** 0.5

            # Benchmark parallel scan (GPU version if available)
            par_times = []
            if device.type == "cuda" and seq_len >= 32:
                for _ in range(num_runs):
                    with Timer(device) as t:
                        _ = _parallel_scan_gpu(A_seq, b_seq)
                    par_times.append(t.elapsed)
            else:
                for _ in range(num_runs):
                    with Timer(device) as t:
                        _ = parallel_scan(A_seq, b_seq, initial_state)
                    par_times.append(t.elapsed)
            par_mean = sum(par_times) / len(par_times)
            par_std = (sum((x - par_mean) ** 2 for x in par_times) / len(par_times)) ** 0.5

            speedup = seq_mean / par_mean if par_mean > 0 else float("inf")

            print(f"  Sequential: {seq_mean*1000:.3f} ± {seq_std*1000:.3f} ms")
            print(f"  Parallel:   {par_mean*1000:.3f} ± {par_std*1000:.3f} ms")
            print(f"  Speedup:    {speedup:.2f}x")

            results[key] = {
                "sequential_ms": seq_mean * 1000,
                "parallel_ms": par_mean * 1000,
                "speedup": speedup,
            }

    return results


def benchmark_ssm_layers(
    batch_sizes: List[int] = [4, 8],
    seq_lengths: List[int] = [100, 500, 1000, 1750],
    hidden_dim: int = 64,
    state_dim: int = 64,
    num_runs: int = 10,
    include_backward: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark SSM layers (S4D, S5).

    Args:
        batch_sizes: List of batch sizes
        seq_lengths: List of sequence lengths
        hidden_dim: Hidden dimension
        state_dim: State dimension
        num_runs: Number of runs
        include_backward: Whether to include backward pass timing
        device: Device to run on

    Returns:
        Benchmark results
    """
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"SSM Layer Benchmarks (device: {device})")
    print(f"{'='*60}")

    results = {}
    warmup_device(device)

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"batch={batch_size}, seq={seq_len}"
            print(f"\n{key}")
            print("-" * 40)

            # Create layers
            s4d = S4DKernel(state_dim, hidden_dim).to(device)
            s5 = S5Layer(hidden_dim, state_dim).to(device)

            # Test data
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=include_backward)

            # Benchmark S4D
            s4d_fwd_times = []
            s4d_bwd_times = []
            for _ in range(num_runs):
                x_in = x.detach().clone().requires_grad_(include_backward)

                with Timer(device) as t:
                    y = s4d(x_in)
                s4d_fwd_times.append(t.elapsed)

                if include_backward:
                    grad_out = torch.randn_like(y)
                    with Timer(device) as t:
                        y.backward(grad_out)
                    s4d_bwd_times.append(t.elapsed)

            s4d_fwd_mean = sum(s4d_fwd_times) / len(s4d_fwd_times)

            # Benchmark S5
            s5_fwd_times = []
            s5_bwd_times = []
            for _ in range(num_runs):
                x_in = x.detach().clone().requires_grad_(include_backward)

                with Timer(device) as t:
                    y = s5(x_in)
                s5_fwd_times.append(t.elapsed)

                if include_backward:
                    grad_out = torch.randn_like(y)
                    with Timer(device) as t:
                        y.backward(grad_out)
                    s5_bwd_times.append(t.elapsed)

            s5_fwd_mean = sum(s5_fwd_times) / len(s5_fwd_times)

            print(f"  S4D Forward:  {s4d_fwd_mean*1000:.3f} ms")
            print(f"  S5 Forward:   {s5_fwd_mean*1000:.3f} ms")

            if include_backward:
                s4d_bwd_mean = sum(s4d_bwd_times) / len(s4d_bwd_times)
                s5_bwd_mean = sum(s5_bwd_times) / len(s5_bwd_times)
                print(f"  S4D Backward: {s4d_bwd_mean*1000:.3f} ms")
                print(f"  S5 Backward:  {s5_bwd_mean*1000:.3f} ms")

            results[key] = {
                "s4d_forward_ms": s4d_fwd_mean * 1000,
                "s5_forward_ms": s5_fwd_mean * 1000,
            }

    return results


def benchmark_ssm_filter(
    batch_sizes: List[int] = [4, 8],
    seq_lengths: List[int] = [500, 1000, 1750],
    input_dim: int = 19,
    hidden_dim: int = 64,
    state_dim: int = 64,
    num_layers: int = 4,
    num_runs: int = 10,
    include_backward: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark SSMFilter against baseline models.

    Args:
        batch_sizes: Batch sizes to test
        seq_lengths: Sequence lengths to test
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        state_dim: State dimension
        num_layers: Number of layers
        num_runs: Number of benchmark runs
        include_backward: Include backward pass
        device: Device to run on

    Returns:
        Benchmark results
    """
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"SSM Filter Benchmarks (device: {device})")
    print(f"{'='*60}")

    results = {}
    warmup_device(device)

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"batch={batch_size}, seq={seq_len}"
            print(f"\n{key}")
            print("-" * 40)

            # Create SSM filter
            ssm_filter = SSMFilter(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_layers=num_layers,
            ).to(device)

            # Test data
            x = torch.randn(batch_size, seq_len, input_dim, device=device, requires_grad=include_backward)

            # Benchmark forward pass
            fwd_times = []
            bwd_times = []

            for _ in range(num_runs):
                x_in = x.detach().clone().requires_grad_(include_backward)

                with Timer(device) as t:
                    outputs = ssm_filter(x_in)
                fwd_times.append(t.elapsed)

                if include_backward:
                    # Backward on position output
                    loss = outputs["position"].sum()
                    with Timer(device) as t:
                        loss.backward()
                    bwd_times.append(t.elapsed)

            fwd_mean = sum(fwd_times) / len(fwd_times)
            print(f"  Forward:  {fwd_mean*1000:.3f} ms")

            result = {"forward_ms": fwd_mean * 1000}

            if include_backward:
                bwd_mean = sum(bwd_times) / len(bwd_times)
                print(f"  Backward: {bwd_mean*1000:.3f} ms")
                print(f"  Total:    {(fwd_mean + bwd_mean)*1000:.3f} ms")
                result["backward_ms"] = bwd_mean * 1000
                result["total_ms"] = (fwd_mean + bwd_mean) * 1000

            results[key] = result

    return results


def benchmark_integrator(
    batch_sizes: List[int] = [4, 8, 16],
    seq_lengths: List[int] = [500, 1000, 1750],
    num_runs: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark parallel vs sequential integration.

    Args:
        batch_sizes: Batch sizes to test
        seq_lengths: Sequence lengths to test
        num_runs: Number of runs
        device: Device to run on

    Returns:
        Benchmark results
    """
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Integrator Benchmarks (device: {device})")
    print(f"{'='*60}")

    results = {}
    warmup_device(device)

    integrator = ParallelIntegrator(dt=0.02)

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"batch={batch_size}, seq={seq_len}"
            print(f"\n{key}")
            print("-" * 40)

            # Test data
            accel = torch.randn(batch_size, seq_len, 3, device=device)
            init_pos = torch.zeros(batch_size, 3, device=device)
            init_vel = torch.zeros(batch_size, 3, device=device)

            # Benchmark cumsum-based integration
            cumsum_times = []
            for _ in range(num_runs):
                with Timer(device) as t:
                    _, _ = integrator.forward(accel, init_pos, init_vel)
                cumsum_times.append(t.elapsed)

            cumsum_mean = sum(cumsum_times) / len(cumsum_times)
            print(f"  Cumsum-based: {cumsum_mean*1000:.3f} ms")

            results[key] = {"cumsum_ms": cumsum_mean * 1000}

    return results


def verify_correctness(device: Optional[torch.device] = None) -> bool:
    """
    Verify that parallel scan produces same results as sequential.

    Args:
        device: Device to run on

    Returns:
        True if all tests pass
    """
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Correctness Verification (device: {device})")
    print(f"{'='*60}")

    all_passed = True
    tolerance = 1e-5

    # Test 1: Basic scan correctness
    print("\n1. Basic scan correctness...")
    batch_size, seq_len, state_dim, input_dim = 4, 100, 8, 16

    A = torch.eye(state_dim, device=device) * 0.9
    B = torch.randn(state_dim, input_dim, device=device) * 0.1
    inputs = torch.randn(batch_size, seq_len, input_dim, device=device)
    initial_state = torch.zeros(batch_size, state_dim, device=device)

    # Sequential
    seq_states = sequential_scan(inputs, initial_state, A, B)

    # Parallel
    A_seq = A.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).clone()
    b_seq = torch.einsum("ij,btj->bti", B, inputs)
    par_states = parallel_scan(A_seq, b_seq, initial_state)

    diff = (seq_states - par_states).abs().max().item()
    passed = diff < tolerance
    print(f"   Max difference: {diff:.2e} ({'PASS' if passed else 'FAIL'})")
    all_passed = all_passed and passed

    # Test 2: S5 layer forward/backward
    print("\n2. S5 layer gradients...")
    s5 = S5Layer(32, 16).to(device)
    x = torch.randn(2, 50, 32, device=device, requires_grad=True)
    y = s5(x)
    loss = y.sum()

    try:
        loss.backward()
        grad_norm = x.grad.norm().item()
        passed = grad_norm > 0 and not torch.isnan(x.grad).any()
        print(f"   Gradient norm: {grad_norm:.4f} ({'PASS' if passed else 'FAIL'})")
    except Exception as e:
        print(f"   Backward failed: {e} (FAIL)")
        passed = False
    all_passed = all_passed and passed

    # Test 3: SSMFilter output shapes
    print("\n3. SSMFilter output shapes...")
    ssm_filter = SSMFilter(input_dim=19, hidden_dim=32, state_dim=32, num_layers=2).to(device)
    x = torch.randn(2, 100, 19, device=device)
    outputs = ssm_filter(x)

    expected_shapes = {
        "position": (2, 100, 3),
        "velocity": (2, 100, 3),
        "log_variance": (2, 100, 6),
    }

    for name, expected_shape in expected_shapes.items():
        if name in outputs:
            actual_shape = tuple(outputs[name].shape)
            passed = actual_shape == expected_shape
            print(f"   {name}: {actual_shape} ({'PASS' if passed else 'FAIL'})")
            all_passed = all_passed and passed

    # Test 4: Integrator correctness
    print("\n4. Integrator correctness...")
    integrator = ParallelIntegrator(dt=0.02)
    accel = torch.zeros(2, 100, 3, device=device)
    accel[:, :, 0] = 1.0  # Constant acceleration in x

    init_pos = torch.zeros(2, 3, device=device)
    init_vel = torch.zeros(2, 3, device=device)

    pos, vel = integrator.forward(accel, init_pos, init_vel)

    # After 100 steps with a=1 and dt=0.02:
    # v_100 ≈ 1 * 0.02 * 100 = 2.0 m/s
    # p_100 ≈ 0.5 * 1 * (0.02 * 100)^2 = 2.0 m
    expected_vel_x = 2.0
    expected_pos_x = 2.0

    actual_vel_x = vel[:, -1, 0].mean().item()
    actual_pos_x = pos[:, -1, 0].mean().item()

    vel_passed = abs(actual_vel_x - expected_vel_x) < 0.1
    pos_passed = abs(actual_pos_x - expected_pos_x) < 0.2

    print(f"   Final velocity x: {actual_vel_x:.3f} (expected ~{expected_vel_x}) {'PASS' if vel_passed else 'FAIL'}")
    print(f"   Final position x: {actual_pos_x:.3f} (expected ~{expected_pos_x}) {'PASS' if pos_passed else 'FAIL'}")
    all_passed = all_passed and vel_passed and pos_passed

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")

    return all_passed


def run_all_benchmarks(args: argparse.Namespace) -> None:
    """Run all benchmarks based on command line arguments."""
    device = get_device()
    print(f"\nRunning benchmarks on: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if args.verify:
        verify_correctness(device)
        return

    if args.scan_only:
        benchmark_scan_operations(
            seq_lengths=args.sequence_lengths,
            num_runs=args.num_runs,
            device=device,
        )
        return

    if args.filter_only:
        benchmark_ssm_filter(
            seq_lengths=args.sequence_lengths,
            num_runs=args.num_runs,
            device=device,
        )
        return

    # Run all benchmarks
    verify_correctness(device)
    benchmark_scan_operations(
        seq_lengths=args.sequence_lengths,
        num_runs=args.num_runs,
        device=device,
    )
    benchmark_ssm_layers(
        seq_lengths=args.sequence_lengths,
        num_runs=args.num_runs,
        device=device,
    )
    benchmark_ssm_filter(
        seq_lengths=args.sequence_lengths,
        num_runs=args.num_runs,
        device=device,
    )
    benchmark_integrator(
        seq_lengths=args.sequence_lengths,
        num_runs=args.num_runs,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Parallel Scan Benchmarks")
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 1750],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs for averaging",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only run scan operation benchmarks",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only run filter benchmarks",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only run correctness verification",
    )

    args = parser.parse_args()
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
