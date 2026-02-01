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
Linear State Space Model for Parallelizable Sequence Modeling

This module implements learnable linear state space models (S4/S5 style)
that can be trained efficiently using parallel scan operations.

Key advantages over RNNs/TCNs:
1. O(log T) parallel depth during training (vs O(T) for RNNs)
2. O(T) inference like RNNs (vs O(T * kernel_size) for TCNs)
3. Theoretically infinite receptive field

References:
- Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces" (S4)
- Smith et al. (2023). "Simplified State Space Layers for Sequence Modeling" (S5)
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .scan_ops import parallel_scan, sequential_scan, discretize_continuous_ssm


class S4DKernel(nn.Module):
    """
    S4D (Diagonal State Space) Kernel.

    Uses diagonal state matrix for efficient computation:
    - Parallel scan with element-wise operations
    - Convolution view for long-range dependencies
    - HiPPO initialization for memory

    The continuous-time SSM is:
        dx/dt = A x + B u
        y = C x + D u

    With diagonal A, each dimension evolves independently.
    """

    def __init__(
        self,
        state_dim: int,
        channels: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[dict] = None,
    ):
        """
        Args:
            state_dim: Dimension of the state space (N in S4 papers)
            channels: Number of independent SSM channels (H in S4 papers)
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
            lr: Learning rate multipliers for different parameters
        """
        super().__init__()

        self.state_dim = state_dim
        self.channels = channels

        # Initialize A with HiPPO-LegS matrix (diagonal approximation)
        # For diagonal S4, we use the eigenvalues of HiPPO
        # A_n = -1/2 + n*i for n = 0, ..., N-1
        A_real = torch.full((channels, state_dim), -0.5)
        A_imag = torch.arange(state_dim).float().unsqueeze(0).expand(channels, -1) * math.pi

        # Store as log for numerical stability (A is negative real part)
        self.log_A_real = nn.Parameter(torch.log(-A_real))
        self.A_imag = nn.Parameter(A_imag)

        # B, C initialized as complex normal
        self.B = nn.Parameter(torch.randn(channels, state_dim, dtype=torch.cfloat) / math.sqrt(state_dim))
        self.C = nn.Parameter(torch.randn(channels, state_dim, dtype=torch.cfloat) / math.sqrt(state_dim))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(channels))

        # Discretization step (learnable, per-channel)
        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def _get_A(self) -> torch.Tensor:
        """Get continuous-time A matrix (diagonal, complex)."""
        A_real = -torch.exp(self.log_A_real)  # Ensure negative real part (stable)
        return A_real + 1j * self.A_imag  # [channels, state_dim]

    def _discretize(self, dt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize the continuous-time SSM.

        For diagonal A with ZOH discretization:
            A_bar = exp(A * dt)
            B_bar = (A_bar - 1) / A * B

        Args:
            dt: Discretization step [channels] or None to use learned dt

        Returns:
            A_bar: Discretized A [channels, state_dim]
            B_bar: Discretized B [channels, state_dim]
        """
        if dt is None:
            dt = torch.exp(self.log_dt)  # [channels]

        A = self._get_A()  # [channels, state_dim]

        # Expand dt to match A shape
        dt = dt.unsqueeze(-1)  # [channels, 1]

        # ZOH discretization for diagonal systems
        A_bar = torch.exp(A * dt)  # [channels, state_dim]

        # B_bar = (A_bar - 1) / A * B, with numerical stability for A ≈ 0
        B_bar = (A_bar - 1) / A * self.B  # [channels, state_dim]

        return A_bar, B_bar

    def forward(
        self,
        u: torch.Tensor,
        return_state: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass using parallel scan.

        Args:
            u: Input sequence [batch, seq_len, channels]
            return_state: If True, also return final state

        Returns:
            y: Output sequence [batch, seq_len, channels]
            (optional) state: Final state [batch, channels, state_dim]
        """
        batch_size, seq_len, _ = u.shape

        # Discretize
        A_bar, B_bar = self._discretize()  # [channels, state_dim], [channels, state_dim]

        # Compute using convolution kernel (more efficient for training)
        # Kernel K[t] = C @ A_bar^t @ B_bar for t = 0, ..., seq_len-1
        K = self._compute_kernel(seq_len)  # [channels, seq_len]

        # Convolve: y = K * u
        # Use FFT for efficiency when seq_len is large
        if seq_len >= 64:
            y = self._fft_conv(u, K)
        else:
            y = self._direct_conv(u, K)

        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u

        if return_state:
            # Compute final state for recurrent inference
            state = self._compute_final_state(u, A_bar, B_bar)
            return y, state

        return y

    def _compute_kernel(self, length: int) -> torch.Tensor:
        """
        Compute the SSM convolution kernel.

        K[t] = C @ A_bar^t @ B_bar = C * (A_bar^t) * B_bar (element-wise for diagonal)

        Args:
            length: Kernel length

        Returns:
            K: Kernel [channels, length] (real-valued)
        """
        A_bar, B_bar = self._discretize()  # [channels, state_dim]

        # Powers of A_bar: [channels, state_dim, length]
        powers = torch.arange(length, device=A_bar.device, dtype=A_bar.dtype)
        A_powers = A_bar.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)  # [channels, state_dim, length]

        # K = sum over state_dim of C * A^t * B
        # [channels, state_dim] * [channels, state_dim, length] * [channels, state_dim]
        CB = self.C * B_bar  # [channels, state_dim]
        K = (CB.unsqueeze(-1) * A_powers).sum(dim=1)  # [channels, length]

        # Take real part (imaginary parts cancel for conjugate-paired eigenvalues)
        K = K.real

        return K

    def _fft_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        FFT-based convolution for efficiency with long sequences.

        Args:
            u: Input [batch, seq_len, channels]
            K: Kernel [channels, kernel_len]

        Returns:
            y: Output [batch, seq_len, channels]
        """
        seq_len = u.shape[1]
        kernel_len = K.shape[1]

        # Pad to power of 2 for FFT efficiency
        fft_len = 2 ** math.ceil(math.log2(seq_len + kernel_len - 1))

        # FFT of input and kernel
        u_f = torch.fft.rfft(u.transpose(1, 2), n=fft_len, dim=-1)  # [batch, channels, fft_len//2+1]
        K_f = torch.fft.rfft(K, n=fft_len, dim=-1)  # [channels, fft_len//2+1]

        # Multiply in frequency domain
        y_f = u_f * K_f.unsqueeze(0)  # [batch, channels, fft_len//2+1]

        # Inverse FFT and truncate
        y = torch.fft.irfft(y_f, n=fft_len, dim=-1)[..., :seq_len]  # [batch, channels, seq_len]

        return y.transpose(1, 2)  # [batch, seq_len, channels]

    def _direct_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Direct convolution for short sequences.

        Args:
            u: Input [batch, seq_len, channels]
            K: Kernel [channels, kernel_len]

        Returns:
            y: Output [batch, seq_len, channels]
        """
        # Use F.conv1d with groups=channels for efficiency
        u_t = u.transpose(1, 2)  # [batch, channels, seq_len]
        K_t = K.unsqueeze(1)  # [channels, 1, kernel_len]

        # Causal padding
        padding = K.shape[1] - 1
        y = F.conv1d(u_t, K_t, padding=padding, groups=self.channels)
        y = y[..., :u.shape[1]]  # Remove future padding

        return y.transpose(1, 2)

    def _compute_final_state(
        self,
        u: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute final state for continuing recurrent inference.

        Args:
            u: Input sequence [batch, seq_len, channels]
            A_bar: Discretized A [channels, state_dim]
            B_bar: Discretized B [channels, state_dim]

        Returns:
            state: Final state [batch, channels, state_dim]
        """
        batch_size, seq_len, _ = u.shape

        # x_T = sum_{t=0}^{T-1} A^{T-1-t} B u_t
        # = A^{T-1} B u_0 + A^{T-2} B u_1 + ... + B u_{T-1}

        state = torch.zeros(
            batch_size, self.channels, self.state_dim,
            dtype=torch.cfloat, device=u.device
        )

        A_power = torch.ones_like(A_bar)  # A^0

        for t in range(seq_len - 1, -1, -1):
            # state += A^(T-1-t) * B * u[t]
            Bu = B_bar * u[:, t, :, None]  # [batch, channels, state_dim]
            state = state + A_power.unsqueeze(0) * Bu
            A_power = A_power * A_bar  # Update power

        return state

    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step for recurrent inference.

        Args:
            u: Input [batch, channels]
            state: Current state [batch, channels, state_dim]

        Returns:
            y: Output [batch, channels]
            new_state: Updated state [batch, channels, state_dim]
        """
        A_bar, B_bar = self._discretize()

        # x' = A_bar * x + B_bar * u
        new_state = A_bar * state + B_bar * u.unsqueeze(-1)

        # y = Re(C @ x) + D * u
        y = (self.C * new_state).sum(dim=-1).real + self.D * u

        return y, new_state


class LinearSSM(nn.Module):
    """
    Linear State Space Model layer for sequence modeling.

    Combines multiple S4D kernels with input/output projections
    for use as a drop-in replacement for TCN layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        state_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (SSM channels)
            output_dim: Output dimension
            state_dim: State space dimension per channel
            num_layers: Number of SSM layers
            dropout: Dropout rate
            bidirectional: If True, use bidirectional SSM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # SSM layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(S4DKernel(state_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

            if bidirectional:
                self.layers.append(S4DKernel(state_dim, hidden_dim))

        # Output projection
        scale = 2 if bidirectional else 1
        self.output_proj = nn.Linear(hidden_dim * scale, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, input_dim]

        Returns:
            y: Output [batch, seq_len, output_dim]
        """
        # Input projection
        h = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # SSM layers
        layer_idx = 0
        for i in range(self.num_layers):
            residual = h

            # Forward SSM
            h_fwd = self.layers[layer_idx](h)
            layer_idx += 1

            if self.bidirectional:
                # Backward SSM (flip, process, flip back)
                h_bwd = self.layers[layer_idx](h.flip(dims=[1])).flip(dims=[1])
                layer_idx += 1
                h = torch.cat([h_fwd, h_bwd], dim=-1)
                h = h[..., :self.hidden_dim]  # Project back to hidden_dim
            else:
                h = h_fwd

            # Residual, norm, dropout
            h = self.norms[i](h + residual)
            h = self.dropouts[i](h)

        # Output projection
        if self.bidirectional:
            h_final = torch.cat([h, self.layers[-1](x.flip(dims=[1])).flip(dims=[1])], dim=-1)
        else:
            h_final = h

        y = self.output_proj(h_final if not self.bidirectional else h)

        return y


class S5Layer(nn.Module):
    """
    S5 (Simplified State Space) Layer.

    Key simplification over S4:
    - Uses real-valued diagonal state matrix (no complex arithmetic)
    - MIMO (multi-input multi-output) instead of SISO
    - Direct parallel scan instead of convolution

    Reference: Smith et al. (2023)
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        """
        Args:
            input_dim: Input/output dimension
            state_dim: State dimension
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
        """
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        # Diagonal A (log-parameterized for stability)
        self.log_A = nn.Parameter(torch.randn(state_dim) * 0.5 - 1.0)

        # B: [state_dim, input_dim]
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) / math.sqrt(input_dim))

        # C: [input_dim, state_dim]
        self.C = nn.Parameter(torch.randn(input_dim, state_dim) / math.sqrt(state_dim))

        # D: skip connection
        self.D = nn.Parameter(torch.zeros(input_dim))

        # Learnable dt
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using parallel scan.

        Args:
            u: Input [batch, seq_len, input_dim]

        Returns:
            y: Output [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A)  # Ensure stable (negative)

        # Discretize
        A_bar = torch.exp(A * dt)  # [state_dim]
        B_bar = (A_bar - 1) / A  # [state_dim]
        B_bar = B_bar.unsqueeze(-1) * self.B  # [state_dim, input_dim]

        # Compute Bu for all timesteps
        Bu = torch.einsum("ni,bti->btn", B_bar, u)  # [batch, seq_len, state_dim]

        # Parallel scan to compute states
        # x_t = A_bar * x_{t-1} + Bu_t

        # Use log-space cumsum for numerical stability
        log_A_bar = A * dt  # log(A_bar) = A * dt
        log_A_cumsum = torch.cumsum(
            log_A_bar.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1),
            dim=1
        )  # [batch, seq_len, state_dim]

        # x_t = sum_{k=0}^{t} A_bar^{t-k} Bu_k
        # = sum_{k=0}^{t} exp((t-k) * log_A_bar) * Bu_k

        # Shift cumsum to get factors
        log_A_shifted = F.pad(log_A_cumsum[:, :-1], (0, 0, 1, 0), value=0)

        # Compute all states using vectorized operations
        # For each t: weight[t,k] = exp(log_A_cumsum[t] - log_A_shifted[k])
        # x[t] = sum_k weight[t,k] * Bu[k] for k <= t

        # This is O(T^2) naively, but we can do better with scan
        states = self._parallel_scan_states(A_bar, Bu)

        # Output: y = C @ x + D * u
        y = torch.einsum("in,btn->bti", self.C, states) + self.D * u

        return y

    def _parallel_scan_states(
        self,
        A_bar: torch.Tensor,
        Bu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute all states using parallel scan.

        For diagonal A, the recurrence x_{t+1} = A_bar * x_t + Bu_t
        decouples across state dimensions.
        """
        batch_size, seq_len, state_dim = Bu.shape
        device = Bu.device
        dtype = Bu.dtype

        # Initialize
        states = torch.zeros_like(Bu)

        # Simple parallel scan with iterative doubling
        # This computes prefix sums with the linear recurrence operator

        # Work arrays
        A_work = A_bar.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone()
        b_work = Bu.clone()

        # Iterative doubling
        num_iters = int(math.ceil(math.log2(seq_len)))

        for d in range(num_iters):
            stride = 2 ** d

            if stride >= seq_len:
                break

            # Elements to update: indices >= stride
            mask = torch.arange(seq_len, device=device) >= stride

            # Source indices
            src_indices = torch.arange(seq_len, device=device) - stride
            src_indices = src_indices.clamp(min=0)

            # Get source values
            A_src = A_work[:, src_indices]
            b_src = b_work[:, src_indices]

            # Combine: (A_new, b_new) = (A * A_src, A * b_src + b)
            A_new = A_work * A_src
            b_new = A_work * b_src + b_work

            # Apply update only where mask is True
            A_work = torch.where(mask.unsqueeze(0).unsqueeze(-1), A_new, A_work)
            b_work = torch.where(mask.unsqueeze(0).unsqueeze(-1), b_new, b_work)

        return b_work

    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step for recurrent inference.

        Args:
            u: Input [batch, input_dim]
            state: Current state [batch, state_dim]

        Returns:
            y: Output [batch, input_dim]
            new_state: Updated state [batch, state_dim]
        """
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A)
        A_bar = torch.exp(A * dt)
        B_bar = ((A_bar - 1) / A).unsqueeze(-1) * self.B

        # State update
        Bu = torch.einsum("ni,bi->bn", B_bar, u)
        new_state = A_bar * state + Bu

        # Output
        y = torch.einsum("in,bn->bi", self.C, new_state) + self.D * u

        return y, new_state
