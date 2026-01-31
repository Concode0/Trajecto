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
