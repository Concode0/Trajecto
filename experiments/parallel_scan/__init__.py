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
