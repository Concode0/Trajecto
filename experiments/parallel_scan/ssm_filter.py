"""
SSM-Based Filter for Trajectory Estimation

This module implements a learnable state space model that serves as a
drop-in replacement for the ESKF in the Trajecto pipeline.

Key differences from ESKF:
1. Linear dynamics (learned A, B matrices vs physics-based nonlinear)
2. Parallel scan training (O(log T) depth vs O(T) sequential)
3. No explicit quaternion handling (orientation in vector space)

Trade-offs:
- Faster training due to parallelization
- May need larger state dimension to match ESKF accuracy
- Less physically interpretable
- No hard constraints on quaternion normalization

Usage:
    # Replace ESKF in training loop
    filter = SSMFilter(input_dim=19, state_dim=64, output_dim=10)
    outputs = filter(features)  # Parallel during training

    # Or use recurrent mode for inference
    for t in range(seq_len):
        output, state = filter.step(features[:, t], state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .linear_ssm import S5Layer, LinearSSM


class SSMFilter(nn.Module):
    """
    State Space Model based filter for trajectory estimation.

    Replaces the nonlinear ESKF with a learned linear SSM that can be
    trained efficiently using parallel scan.

    Architecture:
        Input Features (19D) → SSM Stack → Output Heads
        - Position (3D)
        - Velocity (3D)
        - Orientation (4D quaternion or 3D rotation vector)
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 64,
        state_dim: int = 64,
        num_layers: int = 4,
        output_pos: bool = True,
        output_vel: bool = True,
        output_ori: bool = False,
        use_quaternion: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension (default 19 for Trajecto)
            hidden_dim: Hidden/channel dimension
            state_dim: State space dimension per layer
            num_layers: Number of SSM layers
            output_pos: Whether to output position
            output_vel: Whether to output velocity
            output_ori: Whether to output orientation
            use_quaternion: If True, output 4D quaternion; else 3D rotation vector
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.output_pos = output_pos
        self.output_vel = output_vel
        self.output_ori = output_ori
        self.use_quaternion = use_quaternion

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # SSM backbone
        self.ssm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.ssm_layers.append(S5Layer(hidden_dim, state_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        # Output heads
        self.output_heads = nn.ModuleDict()

        if output_pos:
            self.output_heads["position"] = nn.Linear(hidden_dim, 3)

        if output_vel:
            self.output_heads["velocity"] = nn.Linear(hidden_dim, 3)

        if output_ori:
            ori_dim = 4 if use_quaternion else 3
            self.output_heads["orientation"] = nn.Linear(hidden_dim, ori_dim)

        # Optional: uncertainty estimation (like TCN covariance output)
        self.output_heads["log_variance"] = nn.Linear(hidden_dim, 6)  # pos + vel variance

        # Store states for recurrent mode
        self._states = None

    def forward(
        self,
        features: torch.Tensor,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using parallel scan (efficient for training).

        Args:
            features: Input features [batch, seq_len, input_dim]
            return_hidden: If True, also return hidden states

        Returns:
            outputs: Dictionary of output tensors
        """
        batch_size, seq_len, _ = features.shape

        # Input projection
        h = self.input_proj(features)  # [batch, seq_len, hidden_dim]

        # SSM layers with residual connections
        for i, (ssm, norm, drop) in enumerate(zip(
            self.ssm_layers, self.layer_norms, self.dropouts
        )):
            residual = h
            h = ssm(h)  # Parallel scan inside
            h = drop(h)
            h = norm(h + residual)

        # Output heads
        outputs = {}

        if "position" in self.output_heads:
            outputs["position"] = self.output_heads["position"](h)

        if "velocity" in self.output_heads:
            outputs["velocity"] = self.output_heads["velocity"](h)

        if "orientation" in self.output_heads:
            ori = self.output_heads["orientation"](h)
            if self.use_quaternion:
                # Normalize to unit quaternion
                ori = F.normalize(ori, dim=-1)
            outputs["orientation"] = ori

        # Log variance for uncertainty
        outputs["log_variance"] = self.output_heads["log_variance"](h)

        if return_hidden:
            outputs["hidden"] = h

        return outputs

    def init_state(self, batch_size: int, device: torch.device) -> None:
        """
        Initialize states for recurrent mode.

        Args:
            batch_size: Batch size
            device: Device to create tensors on
        """
        self._states = []
        for ssm in self.ssm_layers:
            state = torch.zeros(batch_size, ssm.state_dim, device=device)
            self._states.append(state)

    def step(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single step for recurrent inference.

        Must call init_state() before first step.

        Args:
            features: Input features [batch, input_dim]

        Returns:
            outputs: Dictionary of output tensors
        """
        if self._states is None:
            raise RuntimeError("Must call init_state() before step()")

        # Input projection
        h = self.input_proj(features)  # [batch, hidden_dim]

        # SSM layers (recurrent mode)
        new_states = []
        for i, (ssm, norm) in enumerate(zip(self.ssm_layers, self.layer_norms)):
            residual = h
            h, new_state = ssm.step(h, self._states[i])
            new_states.append(new_state)
            h = norm(h + residual)

        self._states = new_states

        # Output heads
        outputs = {}

        if "position" in self.output_heads:
            outputs["position"] = self.output_heads["position"](h)

        if "velocity" in self.output_heads:
            outputs["velocity"] = self.output_heads["velocity"](h)

        if "orientation" in self.output_heads:
            ori = self.output_heads["orientation"](h)
            if self.use_quaternion:
                ori = F.normalize(ori, dim=-1)
            outputs["orientation"] = ori

        outputs["log_variance"] = self.output_heads["log_variance"](h)

        return outputs


class HybridSSMESKF(nn.Module):
    """
    Hybrid model combining SSM for fast feature extraction with
    physics-based integration for position/velocity.

    Architecture:
        Features → SSM → Velocity Correction
                      → ZUPT Probability
                      → Covariance

        Physics Integration (sequential):
            vel += vel_correction
            pos += vel * dt
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dim: int = 64,
        state_dim: int = 64,
        num_layers: int = 4,
        dt: float = 0.02,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            state_dim: SSM state dimension
            num_layers: Number of SSM layers
            dt: Time step for integration
        """
        super().__init__()

        self.dt = dt

        # SSM for feature extraction (parallelizable)
        self.ssm = SSMFilter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_layers=num_layers,
            output_pos=False,
            output_vel=False,
            output_ori=False,
        )

        # Additional output heads for ESKF-style outputs
        self.vel_correction_head = nn.Linear(hidden_dim, 3)
        self.zupt_head = nn.Linear(hidden_dim, 1)
        self.covariance_head = nn.Linear(hidden_dim, 6)

    def forward(
        self,
        features: torch.Tensor,
        initial_pos: Optional[torch.Tensor] = None,
        initial_vel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hybrid SSM + physics integration.

        Args:
            features: Input features [batch, seq_len, input_dim]
            initial_pos: Initial position [batch, 3] (default: zeros)
            initial_vel: Initial velocity [batch, 3] (default: zeros)

        Returns:
            outputs: Dictionary with positions, velocities, corrections, etc.
        """
        batch_size, seq_len, _ = features.shape
        device = features.device
        dtype = features.dtype

        # Initialize
        if initial_pos is None:
            initial_pos = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        if initial_vel is None:
            initial_vel = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        # SSM forward (parallel)
        ssm_out = self.ssm(features, return_hidden=True)
        hidden = ssm_out["hidden"]  # [batch, seq_len, hidden_dim]

        # Compute corrections and ZUPT (parallel)
        vel_corrections = self.vel_correction_head(hidden)  # [batch, seq_len, 3]
        zupt_logits = self.zupt_head(hidden)  # [batch, seq_len, 1]
        zupt_prob = torch.sigmoid(zupt_logits).squeeze(-1)  # [batch, seq_len]
        log_covariance = self.covariance_head(hidden)  # [batch, seq_len, 6]

        # Physics integration (sequential - this part cannot be parallelized)
        # But it's simple scalar operations, so it's fast
        positions = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        velocities = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

        pos = initial_pos.clone()
        vel = initial_vel.clone()

        for t in range(seq_len):
            # Apply velocity correction
            vel = vel + vel_corrections[:, t]

            # Apply ZUPT (zero velocity update)
            zupt_mask = zupt_prob[:, t] > 0.5
            vel = torch.where(zupt_mask.unsqueeze(-1), torch.zeros_like(vel), vel)

            # Integrate position
            pos = pos + vel * self.dt

            positions[:, t] = pos
            velocities[:, t] = vel

        return {
            "position": positions,
            "velocity": velocities,
            "vel_correction": vel_corrections,
            "zupt_prob": zupt_prob,
            "log_covariance": log_covariance,
        }


class ParallelIntegrator(nn.Module):
    """
    Parallel position integration using scan operations.

    For simple kinematic integration:
        v_t = v_{t-1} + a_t * dt
        p_t = p_{t-1} + v_t * dt

    This can be reformulated as a linear recurrence and solved with parallel scan.
    """

    def __init__(self, dt: float = 0.02):
        super().__init__()
        self.dt = dt

    def forward(
        self,
        accelerations: torch.Tensor,
        initial_pos: Optional[torch.Tensor] = None,
        initial_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel integration of accelerations to positions.

        The state is [p, v] with dynamics:
            [p_{t+1}]   [1  dt] [p_t]   [0.5*dt^2]
            [v_{t+1}] = [0   1] [v_t] + [dt      ] * a_t

        This is a linear recurrence that can be computed with parallel scan!

        Args:
            accelerations: Acceleration sequence [batch, seq_len, 3]
            initial_pos: Initial position [batch, 3]
            initial_vel: Initial velocity [batch, 3]

        Returns:
            positions: Position sequence [batch, seq_len, 3]
            velocities: Velocity sequence [batch, seq_len, 3]
        """
        batch_size, seq_len, dim = accelerations.shape
        device = accelerations.device
        dtype = accelerations.dtype

        if initial_pos is None:
            initial_pos = torch.zeros(batch_size, dim, device=device, dtype=dtype)
        if initial_vel is None:
            initial_vel = torch.zeros(batch_size, dim, device=device, dtype=dtype)

        # State transition matrix (same for all timesteps)
        # A = [[1, dt], [0, 1]]
        # But we process each dimension independently, so A is scalar-like

        # Input matrix
        # B = [[0.5*dt^2], [dt]]

        dt = self.dt
        dt2_half = 0.5 * dt * dt

        # Compute prefix products of A and weighted sums
        # Since A is constant and simple, we can use cumsum tricks

        # Velocity: v_t = v_0 + dt * sum_{k=0}^{t-1} a_k
        # This is just cumsum of accelerations * dt
        accel_cumsum = torch.cumsum(accelerations, dim=1)
        velocities = initial_vel.unsqueeze(1) + dt * accel_cumsum

        # Position: p_t = p_0 + sum_{k=0}^{t-1} v_k * dt
        #               = p_0 + v_0 * t * dt + dt^2 * sum_{k=0}^{t-1} sum_{j=0}^{k-1} a_j
        # This is cumsum of cumsum

        # Simpler formulation using velocity integral:
        # p_t = p_0 + dt * sum_{k=0}^{t-1} v_k
        vel_cumsum = torch.cumsum(velocities, dim=1) - velocities  # Exclusive cumsum
        positions = initial_pos.unsqueeze(1) + dt * vel_cumsum

        # Shift to get correct timing (v_t is velocity at time t, used for p_{t+1})
        velocities_shifted = F.pad(velocities[:, :-1], (0, 0, 1, 0))
        velocities_shifted[:, 0] = initial_vel

        return positions, velocities_shifted

    def forward_parallel_scan(
        self,
        accelerations: torch.Tensor,
        initial_pos: Optional[torch.Tensor] = None,
        initial_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alternative implementation using explicit parallel scan.

        This demonstrates the general pattern that works for any linear system.
        """
        from .scan_ops import parallel_scan

        batch_size, seq_len, dim = accelerations.shape
        device = accelerations.device
        dtype = accelerations.dtype

        if initial_pos is None:
            initial_pos = torch.zeros(batch_size, dim, device=device, dtype=dtype)
        if initial_vel is None:
            initial_vel = torch.zeros(batch_size, dim, device=device, dtype=dtype)

        dt = self.dt

        # Stack state: [p, v] for each dimension
        # State dim = 2, but we process 3 dimensions independently

        # For each dimension d:
        # State: [p_d, v_d]
        # A = [[1, dt], [0, 1]]
        # B = [[0.5*dt^2], [dt]]

        A = torch.tensor([[1, dt], [0, 1]], device=device, dtype=dtype)
        B = torch.tensor([[0.5 * dt * dt], [dt]], device=device, dtype=dtype)

        # Process each dimension
        positions_list = []
        velocities_list = []

        for d in range(dim):
            accel_d = accelerations[..., d:d+1]  # [batch, seq_len, 1]
            init_state = torch.stack([initial_pos[:, d], initial_vel[:, d]], dim=-1)  # [batch, 2]

            # Build A_seq and b_seq for parallel scan
            A_seq = A.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            b_seq = torch.einsum("ij,btj->bti", B, accel_d)  # [batch, seq_len, 2]

            # Run parallel scan
            states = parallel_scan(A_seq, b_seq, init_state)  # [batch, seq_len, 2]

            positions_list.append(states[..., 0])
            velocities_list.append(states[..., 1])

        positions = torch.stack(positions_list, dim=-1)
        velocities = torch.stack(velocities_list, dim=-1)

        return positions, velocities


def create_ssm_filter_from_config(config: dict) -> SSMFilter:
    """
    Factory function to create SSMFilter from configuration dictionary.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        SSMFilter instance
    """
    return SSMFilter(
        input_dim=config.get("input_dim", 19),
        hidden_dim=config.get("hidden_dim", 64),
        state_dim=config.get("state_dim", 64),
        num_layers=config.get("num_layers", 4),
        output_pos=config.get("output_pos", True),
        output_vel=config.get("output_vel", True),
        output_ori=config.get("output_ori", False),
        use_quaternion=config.get("use_quaternion", False),
        dropout=config.get("dropout", 0.1),
    )
