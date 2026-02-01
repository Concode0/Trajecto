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

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TCN import TCN
from model.config import Config

class StatefulTCNExport(nn.Module):
    """
    Wrapper around a trained TCN model to support stateful (step-by-step) execution.
    Designed primarily for ONNX export to embedded systems.
    
    Instead of recomputing the entire receptive field at each step, this model
    takes the current input and a list of state buffers (previous inputs to each layer),
    and returns the prediction and the updated state buffers.
    
    Layout: Standard PyTorch NCL [Batch, Channels, Length].
    """
    def __init__(self, tcn_model: TCN):
        super().__init__()

        # Y-shaped architecture: backbone + (dynamic_branch & static_branch)
        self.backbone = tcn_model.backbone
        self.dynamic_branch = tcn_model.dynamic_branch
        self.static_branch = tcn_model.static_branch
        self.dynamic_heads = tcn_model.dynamic_heads
        self.static_heads = tcn_model.static_heads
        self.gravity_head = tcn_model.gravity_head

        # No input normalization - features are pre-normalized upstream
        # This preserves physical relationships between feature magnitudes

        # Precompute buffer sizes for each layer (all 3 branches)
        self.buffer_sizes = []
        all_layers = list(tcn_model.backbone) + list(tcn_model.dynamic_branch) + list(tcn_model.static_branch)

        for layer in all_layers:
            # layer is CausalConv1d
            if layer.separable and layer.depthwise is not None:
                # Use depthwise conv params
                k = layer.depthwise.kernel_size[0]
                d = layer.depthwise.dilation[0]
            else:
                # Use standard conv params
                k = layer.conv.kernel_size[0]
                d = layer.conv.dilation[0]

            # History needed is (Kernel-1) * Dilation
            self.buffer_sizes.append((k - 1) * d)

    def forward(self, x_t, *states):
        """
        Args:
            x_t: Current input feature [Batch, 1, InputSize] -> Transpose to [Batch, InputSize, 1]
            *states: Variable number of state tensors. 
                     Expected len(states) == len(layers).
                     State i shape: [Batch, InChannels_i, BufferSize_i] (NCL)
        
        Returns:
            outputs (tuple): (vel_corr, covariance_R, zupt_prob) each [Batch, 1, OutDim]
            new_states (tuple): Updated state tensors.
        """
        # Input x_t is [Batch, 1, C] (NLC) from external world.
        # Transpose to [Batch, C, 1] (NCL)
        # Features are already normalized upstream (see base_hybrid_model.py)
        current_input = x_t.transpose(1, 2)
        
        new_states = []
        
        state_idx = 0

        # === Backbone (shared) ===
        for layer in self.backbone:
            buffer = states[state_idx]  # [Batch, C, Hist]

            # Concatenate history with current input on Time (dim 2)
            input_window = torch.cat([buffer, current_input], dim=2)

            if layer.separable and layer.depthwise is not None:
                out = layer.depthwise(input_window)  # [Batch, C, 1]
                out = layer.pointwise(out)  # [Batch, OutC, 1]
            else:
                out = layer.conv(input_window)  # [Batch, OutC, 1]

            out = layer.relu(out)
            out = layer.dropout(out)

            new_buffer = input_window[:, :, 1:]
            new_states.append(new_buffer)
            state_idx += 1

            current_input = out

        backbone_output = current_input  # [Batch, C, 1]

        # === Dynamic Branch ===
        current_input = backbone_output
        for layer in self.dynamic_branch:
            buffer = states[state_idx]

            input_window = torch.cat([buffer, current_input], dim=2)

            if layer.separable and layer.depthwise is not None:
                out = layer.depthwise(input_window)
                out = layer.pointwise(out)
            else:
                out = layer.conv(input_window)

            out = layer.relu(out)
            out = layer.dropout(out)

            new_buffer = input_window[:, :, 1:]
            new_states.append(new_buffer)
            state_idx += 1

            current_input = out

        dynamic_output = current_input  # [Batch, C, 1]

        # === Static Branch ===
        current_input = backbone_output
        for layer in self.static_branch:
            buffer = states[state_idx]

            input_window = torch.cat([buffer, current_input], dim=2)

            if layer.separable and layer.depthwise is not None:
                out = layer.depthwise(input_window)
                out = layer.pointwise(out)
            else:
                out = layer.conv(input_window)

            out = layer.relu(out)
            out = layer.dropout(out)

            new_buffer = input_window[:, :, 1:]
            new_states.append(new_buffer)
            state_idx += 1

            current_input = out

        static_output = current_input  # [Batch, C, 1]

        # === Output Heads ===
        # Convert to [Batch, 1, C] for linear layers
        dynamic_feat = dynamic_output.transpose(1, 2)  # [Batch, 1, C]
        static_feat = static_output.transpose(1, 2)   # [Batch, 1, C]

        # Dynamic branch outputs
        vel_corr_raw = self.dynamic_heads["vel_corr"](dynamic_feat)

        vel_log_scale = torch.sign(vel_corr_raw) * torch.log1p(torch.abs(vel_corr_raw))
        vel_corr = torch.clamp(
            vel_log_scale * Config.VEL_CORRECTION_SCALE,
            min=-Config.ESKFTCN.VEL_CORR_CLIP_RANGE,
            max=Config.ESKFTCN.VEL_CORR_CLIP_RANGE
        )

        cov_R = self.dynamic_heads["covariance_R"](dynamic_feat)
        cov_R = torch.clamp(
            cov_R,
            min=-10.0,
            max=Config.ESKFTCN.MAX_COVARIANCE_VAL
        )

        # Static branch outputs
        zupt_p = self.static_heads["zupt_prob"](static_feat)
        zupt_p = torch.sigmoid(zupt_p)

        # Gravity Head: Fuse both branches
        feat_gravity = torch.cat([dynamic_feat, static_feat], dim=-1)
        gravity_b = self.gravity_head(feat_gravity)
        gravity_b = F.normalize(gravity_b, p=2, dim=-1, eps=1e-6)

        return (vel_corr, cov_R, zupt_p, gravity_b), tuple(new_states)