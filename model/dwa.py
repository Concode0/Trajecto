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

import torch
import numpy as np
from typing import List

class DWALossUpdater:
    """
    Dynamic Weight Averaging (CVPR 2019)
    Balances learning speed across different tasks.
    """
    def __init__(self, num_tasks: int = 4, temp: float = 2.0):
        self.num_tasks = num_tasks
        self.temp = temp # Temperature: Higher = smoother weights
        self.avg_losses = np.zeros((2, num_tasks)) # Buffer for [t-1, t-2]
        self.epoch_count = 0

    def get_weights(self, current_epoch_losses: List[float]) -> torch.Tensor:
        # current_epoch_losses: [mag, cos, zupt, cov]

        # 1. First 2 epochs: use equal weights (need history)
        if self.epoch_count < 5:
            self.avg_losses[1] = self.avg_losses[0]
            self.avg_losses[0] = current_epoch_losses
            self.epoch_count += 1
            return torch.ones(self.num_tasks)

        # 2. Update history
        self.avg_losses[1] = self.avg_losses[0]
        self.avg_losses[0] = current_epoch_losses

        # 3. Calculate Relative Learning Rate
        # r_k = L(t-1) / L(t-2)
        # Small r_k = Loss dropped significantly = Task is "easy" or learning fast
        # Large r_k = Loss didn't drop = Task is "hard" -> Increase weight
        r = self.avg_losses[0] / (self.avg_losses[1] + 1e-8)

        # Clamp r to prevent exponential overflow
        r = np.clip(r, 0.0, 10.0)

        # 4. Softmax Normalization
        w = np.exp(r / self.temp)
        w_sum = np.sum(w)
        if w_sum == 0:
            return torch.ones(self.num_tasks)

        w = self.num_tasks * w / w_sum # Scale so sum equals num_tasks

        self.epoch_count += 1
        return torch.tensor(w, dtype=torch.float32)
