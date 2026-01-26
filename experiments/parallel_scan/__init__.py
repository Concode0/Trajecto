"""
Parallel Scan Experiment for Trajecto

This module explores parallel scan methods as potential alternatives to
sequential ESKF filtering for faster training.

Key Components:
- scan_ops: Core parallel scan primitives
- linear_ssm: Parallelizable linear state space model
- ssm_filter: SSM-based filter (drop-in replacement for ESKF)
- benchmark: Speed comparison utilities
"""

from .scan_ops import parallel_scan, sequential_scan, associative_scan
from .linear_ssm import LinearSSM, S4DKernel
from .ssm_filter import SSMFilter

__all__ = [
    "parallel_scan",
    "sequential_scan",
    "associative_scan",
    "LinearSSM",
    "S4DKernel",
    "SSMFilter",
]
