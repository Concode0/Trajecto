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
This module defines a Temporal Convolutional Network (TCN) model designed
for multi-head output in hybrid filter architectures.

The TCN uses a Y-shaped architecture with:
- Shared backbone (2 layers): Common feature extraction
- Dynamic branch (2 layers): Fast motion with smaller receptive field
- Static branch (2 layers): Static/ZUPT with larger receptive field

This design allows the network to capture both fast-changing dynamics and
slow/static patterns with appropriate temporal contexts.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import Config


class CausalConv1d(nn.Module):
    """A causal 1D convolution layer (supporting Depthwise Separable).

    It pads the input on the left (past) side to ensure that the output
    at time t depends only on inputs up to time t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        separable: bool = False, # Default to False, controlled by higher-level models
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.separable = separable

        if separable and in_channels > 1:
            # Depthwise Convolution: groups = in_channels
            self.depthwise = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=0,
                dilation=dilation,
                groups=in_channels,
            )
            # Pointwise Convolution: kernel_size = 1
            self.pointwise = nn.Conv1d(
                in_channels,
                out_channels,
                1,
                padding=0,
                dilation=1, # Pointwise is typically not dilated
            )
            self.conv = None # Placeholder to indicate separable mode
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,  # We handle padding manually
                dilation=dilation,
            )
            self.depthwise = None
            self.pointwise = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left side
        x = F.pad(x, (self.padding, 0))

        if self.separable and self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)

        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCN(nn.Module):
    """A Y-shaped multi-head TCN for closed-loop trajectory correction.

    Architecture:
        Input → [Shared Backbone: 2 layers] → Split
                                              ├→ [Dynamic Branch: 2 layers] → vel_corr, covariance_R
                                              └→ [Static Branch: 2 layers]  → zupt_prob, gravity_b

    The Y-shaped design uses different receptive fields for different outputs:
    - Dynamic branch: Smaller dilations for fast-changing motion signals
    - Static branch: Larger dilations for static/ZUPT detection

    Outputs:
    1. `vel_corr`: A 3D velocity correction vector (from dynamic branch).
    2. `covariance_R`: A 6D vector for adaptive measurement noise (from dynamic branch).
    3. `zupt_prob`: A scalar probability indicating stationary state (from static branch).
    4. `gravity_b`: A 3D unit vector for gravity direction (from static branch).
    """

    def __init__(
        self,
        input_size: int = 7,
        tcn_channels: List[int] = [96, 96, 96, 96],
        kernel_size: int = 3,
        dropout: float = 0.1,
        tcn_dilation_factors: Optional[List[int]] = None,
        separable: bool = False,
        # Y-branch specific parameters
        dynamic_dilations: Optional[List[int]] = None,
        static_dilations: Optional[List[int]] = None,
    ):
        """Initializes the Y-shaped TCN model.

        Args:
            input_size: The number of features in the input sequence (default 7:
                gyro[3] + accel[3] + force[1]).
            tcn_channels: A list of 4 channel sizes:
                [backbone_1, backbone_2, branch_1, branch_2]
                First 2 for shared backbone, last 2 for each branch.
            kernel_size: The kernel size for all convolutional layers.
            dropout: Dropout rate for regularization.
            tcn_dilation_factors: Dilation factors for shared backbone (2 values).
                Defaults to [1, 2] if None.
            separable: If True, uses Depthwise Separable Convolutions.
            dynamic_dilations: Dilation factors for dynamic branch (2 values).
                Defaults to [1, 2] for smaller RF (fast response).
            static_dilations: Dilation factors for static branch (2 values).
                Defaults to [4, 8] for larger RF (slow/static patterns).
        """
        super().__init__()

        # Validate channel configuration (need 4 values for backbone + branches)
        if len(tcn_channels) != 4:
            raise ValueError(
                f"tcn_channels must have exactly 4 elements for Y-architecture, got {len(tcn_channels)}"
            )

        # Default dilation factors for each component
        if tcn_dilation_factors is None:
            tcn_dilation_factors = [1, 2]  # Shared backbone dilations
        if dynamic_dilations is None:
            dynamic_dilations = [1, 2]  # Smaller RF for fast motion
        if static_dilations is None:
            static_dilations = [4, 8]  # Larger RF for static/ZUPT

        # Validate dilation lists
        if len(tcn_dilation_factors) != 2:
            raise ValueError("tcn_dilation_factors must have 2 elements for backbone")
        if len(dynamic_dilations) != 2:
            raise ValueError("dynamic_dilations must have 2 elements")
        if len(static_dilations) != 2:
            raise ValueError("static_dilations must have 2 elements")

        # === Shared Backbone (2 layers) ===
        self.backbone = nn.ModuleList()
        in_channels = input_size
        self._backbone_rf = 1

        for i in range(2):
            dilation = tcn_dilation_factors[i]
            out_channels = tcn_channels[i]
            self.backbone.append(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    separable=separable,
                )
            )
            in_channels = out_channels
            self._backbone_rf += (kernel_size - 1) * dilation

        backbone_out_channels = tcn_channels[1]

        # === Dynamic Branch (2 layers) - for motion-related outputs ===
        self.dynamic_branch = nn.ModuleList()
        in_channels = backbone_out_channels
        self._dynamic_rf = 0

        for i in range(2):
            dilation = dynamic_dilations[i]
            out_channels = tcn_channels[2 + i]
            self.dynamic_branch.append(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    separable=separable,
                )
            )
            in_channels = out_channels
            self._dynamic_rf += (kernel_size - 1) * dilation

        dynamic_out_channels = tcn_channels[3]

        # === Static Branch (2 layers) - for static/ZUPT-related outputs ===
        self.static_branch = nn.ModuleList()
        in_channels = backbone_out_channels
        self._static_rf = 0

        for i in range(2):
            dilation = static_dilations[i]
            out_channels = tcn_channels[2 + i]
            self.static_branch.append(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    separable=separable,
                )
            )
            in_channels = out_channels
            self._static_rf += (kernel_size - 1) * dilation

        static_out_channels = tcn_channels[3]

        # Total receptive fields
        self._receptive_field = self._backbone_rf + max(self._dynamic_rf, self._static_rf)
        self._dynamic_total_rf = self._backbone_rf + self._dynamic_rf
        self._static_total_rf = self._backbone_rf + self._static_rf

        # === Output Heads ===
        # Dynamic branch outputs: motion-related predictions
        self.dynamic_heads = nn.ModuleDict(
            {
                "vel_corr": nn.Linear(dynamic_out_channels, 3),
                "covariance_R": nn.Linear(dynamic_out_channels, 6),
            }
        )

        # Static branch outputs: static/ZUPT-related predictions
        self.static_heads = nn.ModuleDict(
            {
                "zupt_prob": nn.Linear(static_out_channels, 1),
            }
        )

        # Gravity Head: Fuses Static (Stable) + Dynamic (Responsive) features
        # "Gravity is static, but its observation in Body Frame is dynamic."
        fused_dim = dynamic_out_channels + static_out_channels
        hidden_dim = fused_dim // 2
        self.gravity_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        # Register isotropic velocity correction scale as buffer
        self.register_buffer(
            "vel_scale_isotropic",
            torch.tensor(Config.VEL_CORRECTION_SCALE, dtype=torch.float32)
        )

    @property
    def receptive_field(self) -> int:
        """Returns the receptive field of the TCN.

        The receptive field is the number of input time steps that influence
        a single output time step. It's an important characteristic of TCNs,
        determining how much past information the network can leverage.
        """
        return self._receptive_field

    def forward(
        self,
        feature_sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Processes the sequence of input features to produce multi-head outputs.

        Args:
            feature_sequence (torch.Tensor): Input feature sequence.
                - Shape: (Batch, Seq_Len, Input_Size)
                - Unit: Normalized/Feature-Specific
                - Frame: N/A (Abstract features)

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output tensors:
                - "vel_corr": Velocity correction.
                    - Shape: (Batch, Seq_Len, 3) | Unit: m/s | Frame: Body
                - "covariance_R": Covariance parameters.
                    - Shape: (Batch, Seq_Len, 6) | Unit: Log-Variance or similar
                - "zupt_prob": Zero-velocity probability.
                    - Shape: (Batch, Seq_Len, 1) | Range: [0, 1]
                - "gravity_b": Estimated gravity vector in body frame.
                    - Shape: (Batch, Seq_Len, 3) | Unit: normalized vector
        """
        # Conv1d layers in PyTorch expect input of shape `[B, D, T]`.
        # Therefore, we transpose the last two dimensions of the input `feature_sequence`
        # from `[B, T, D]` to `[B, D, T]`.
        tcn_input = feature_sequence.transpose(1, 2)

        # Input features already z-score normalized upstream - preserving relative
        # magnitudes allows TCN to learn from physical relationships between features
        # (e.g., innovation magnitude vs velocity, force vs acceleration correlation)

        # Pass the input through shared backbone
        x = tcn_input
        for layer in self.backbone:
            x = layer(x)

        # --- Dynamic Branch ---
        x_dynamic = x
        for layer in self.dynamic_branch:
            x_dynamic = layer(x_dynamic)

        # --- Static Branch ---
        x_static = x
        for layer in self.static_branch:
            x_static = layer(x_static)

        # After convolutions, transpose back to `[B, T, D']` for the linear output heads.
        tcn_output_dynamic = x_dynamic.transpose(1, 2)
        tcn_output_static = x_static.transpose(1, 2)

        outputs = {}

        # Apply dynamic heads
        for head_name, head in self.dynamic_heads.items():
            outputs[head_name] = head(tcn_output_dynamic)

        # Apply static heads
        for head_name, head in self.static_heads.items():
            outputs[head_name] = head(tcn_output_static)

        # Gravity Head Fusion: Static (Stable) + Dynamic (Responsive)
        # Concatenate features from both branches: [B, T, Dynamic_Dim + Static_Dim]
        feat_gravity = torch.cat([tcn_output_dynamic, tcn_output_static], dim=-1)
        outputs["gravity_b"] = self.gravity_head(feat_gravity)

        # Log-Scale Velocity Correction: signed_log1p for better gradient flow
        # Problem with tanh: gradients vanish near saturation (±1), crushing details
        # Solution: sign(x) * log1p(|x|) has:
        #   (1) Linear behavior near 0: good for small corrections
        #   (2) Logarithmic compression for large values: prevents explosion
        #   (3) Non-vanishing gradients: gradient = 1/(1+|x|) never goes to 0
        #   (4) Preserves sign: can represent positive/negative corrections per axis
        # Scale by isotropic factor and clip to physical limits
        raw_vel = outputs["vel_corr"]
        # Signed log1p: preserves sign, compresses magnitude logarithmically
        vel_log_scale = torch.sign(raw_vel) * torch.log1p(torch.abs(raw_vel))
        # Scale to physical units and clip to reasonable range (5 m/s max)
        outputs["vel_corr"] = torch.clamp(
            vel_log_scale * self.vel_scale_isotropic,
            min=-Config.ESKFTCN.VEL_CORR_CLIP_RANGE,
            max=Config.ESKFTCN.VEL_CORR_CLIP_RANGE
        )

        # CRITICAL: Clip covariance output to prevent numerical explosion
        if "covariance_R" in outputs:
            outputs["covariance_R"] = torch.clamp(
                outputs["covariance_R"],
                min=-10.0,
                max=Config.ESKFTCN.MAX_COVARIANCE_VAL
            )

        # Gravity direction in body frame: normalize to unit vector
        # TCN predicts the expected gravity direction, used for attitude correction
        # L2 normalization ensures output is always a valid direction vector
        if "gravity_b" in outputs:
            outputs["gravity_b"] = F.normalize(outputs["gravity_b"], p=2, dim=-1, eps=1e-6)

        return outputs
