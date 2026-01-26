"""
Parallel Scan Primitives for State Space Models

This module implements efficient parallel scan (prefix sum) operations that can
be used to parallelize sequential computations like Kalman filtering.

Key insight: For linear dynamics x_{k+1} = A x_k + B u_k, we can reformulate
as an associative scan operation and compute all states in O(log T) parallel depth.

References:
- Blelloch, G.E. (1990). "Prefix Sums and Their Applications"
- Smith et al. (2023). "Simplified State Space Layers for Sequence Modeling"
- Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces"
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Optional
import math


def sequential_scan(
    inputs: torch.Tensor,
    initial_state: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Sequential state space computation (baseline for comparison).

    Computes x_{k+1} = A @ x_k + B @ u_k for k = 0, ..., T-1

    Args:
        inputs: Input sequence [batch, seq_len, input_dim]
        initial_state: Initial state [batch, state_dim]
        A: State transition matrix [state_dim, state_dim]
        B: Input matrix [state_dim, input_dim]

    Returns:
        states: All states [batch, seq_len, state_dim]
    """
    batch_size, seq_len, _ = inputs.shape
    state_dim = initial_state.shape[-1]
    device = inputs.device
    dtype = inputs.dtype

    states = torch.zeros(batch_size, seq_len, state_dim, device=device, dtype=dtype)
    state = initial_state

    for t in range(seq_len):
        # x_{k+1} = A @ x_k + B @ u_k
        state = torch.einsum("ij,bj->bi", A, state) + torch.einsum("ij,bj->bi", B, inputs[:, t])
        states[:, t] = state

    return states


def parallel_scan_naive(
    elements: torch.Tensor,
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    """
    Naive parallel scan implementation using recursion (for educational purposes).

    This implements the Blelloch scan algorithm conceptually but executes
    sequentially in Python. Use `parallel_scan` for the optimized version.

    Args:
        elements: Input tensor to scan
        operator: Associative binary operator
        dim: Dimension along which to scan

    Returns:
        Scanned tensor with same shape as input
    """
    n = elements.shape[dim]

    if n == 1:
        return elements

    # Up-sweep (reduce) phase
    result = elements.clone()
    d = 1
    while d < n:
        step = d * 2
        indices = torch.arange(step - 1, n, step)
        for i in indices:
            left_idx = i.item() - d
            result_slice_left = result.select(dim, int(left_idx))
            result_slice_right = result.select(dim, int(i))
            combined = operator(result_slice_left, result_slice_right)
            # In-place update
            result.select(dim, int(i)).copy_(combined)
        d *= 2

    return result


def _combine_linear_recurrence(
    element1: Tuple[torch.Tensor, torch.Tensor],
    element2: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative operator for linear recurrence: x_{k+1} = A @ x_k + b_k

    The key insight is that we can represent the affine transformation as a tuple
    (A, b) and combine them associatively:

    (A_2, b_2) * (A_1, b_1) = (A_2 @ A_1, A_2 @ b_1 + b_2)

    This allows us to compute prefix products in parallel!

    Args:
        element1: (A_1, b_1) - earlier element
        element2: (A_2, b_2) - later element

    Returns:
        Combined element (A_combined, b_combined)
    """
    A1, b1 = element1
    A2, b2 = element2

    # A_combined = A_2 @ A_1
    A_combined = torch.einsum("...ij,...jk->...ik", A2, A1)

    # b_combined = A_2 @ b_1 + b_2
    b_combined = torch.einsum("...ij,...j->...i", A2, b1) + b2

    return (A_combined, b_combined)


def parallel_scan(
    A_seq: torch.Tensor,
    b_seq: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scan for linear recurrence x_{k+1} = A_k @ x_k + b_k

    Uses the work-efficient Blelloch algorithm adapted for linear recurrences.

    For time-invariant systems (A_k = A for all k), the A matrices can be
    precomputed as powers of A, enabling further optimization.

    Args:
        A_seq: Sequence of transition matrices [batch, seq_len, state_dim, state_dim]
               For time-invariant: can broadcast from [state_dim, state_dim]
        b_seq: Sequence of bias terms [batch, seq_len, state_dim]
        initial_state: Initial state [batch, state_dim]

    Returns:
        states: All states [batch, seq_len, state_dim]

    Complexity:
        Work: O(T * state_dim^2)
        Depth: O(log(T) * state_dim^3) for matrix multiply
    """
    batch_size, seq_len, state_dim = b_seq.shape
    device = b_seq.device
    dtype = b_seq.dtype

    # Handle time-invariant A (broadcast from [d,d] to [B, T, d, d])
    if A_seq.dim() == 2:
        A_seq = A_seq.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
    elif A_seq.dim() == 3:
        A_seq = A_seq.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Initialize elements with (A_k, b_k) tuples
    # For the first element, we need to include the initial state contribution
    # x_1 = A @ x_0 + b_0, so b_0_adjusted = A @ x_0 + b_0

    # Create adjusted first bias
    b_adjusted = b_seq.clone()
    b_adjusted[:, 0] = torch.einsum("bij,bj->bi", A_seq[:, 0], initial_state) + b_seq[:, 0]

    # Work-efficient parallel scan (Blelloch algorithm)
    # This implementation uses a simple iterative approach that can be parallelized

    # For small sequences or when not on GPU, fall back to sequential
    if seq_len <= 32 or not b_seq.is_cuda:
        return _parallel_scan_sequential_fallback(A_seq, b_adjusted, initial_state)

    return _parallel_scan_gpu(A_seq, b_adjusted)


def _parallel_scan_sequential_fallback(
    A_seq: torch.Tensor,
    b_seq: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """
    Sequential fallback for parallel scan (used for small sequences or CPU).
    """
    batch_size, seq_len, state_dim = b_seq.shape
    device = b_seq.device
    dtype = b_seq.dtype

    states = torch.zeros(batch_size, seq_len, state_dim, device=device, dtype=dtype)

    # First state is just b_adjusted[0] (which includes A @ x_0)
    states[:, 0] = b_seq[:, 0]

    for t in range(1, seq_len):
        states[:, t] = torch.einsum("bij,bj->bi", A_seq[:, t], states[:, t-1]) + b_seq[:, t]

    return states


def _parallel_scan_gpu(
    A_seq: torch.Tensor,
    b_seq: torch.Tensor,
) -> torch.Tensor:
    """
    GPU-optimized parallel scan using iterative doubling.

    This implements the parallel prefix algorithm where in each iteration,
    element i is combined with element i - 2^k for increasing k.
    """
    batch_size, seq_len, state_dim = b_seq.shape
    device = b_seq.device
    dtype = b_seq.dtype

    # Working copies
    A_work = A_seq.clone()
    b_work = b_seq.clone()

    # Number of iterations = ceil(log2(seq_len))
    num_iters = int(math.ceil(math.log2(seq_len)))

    for d in range(num_iters):
        stride = 2 ** d

        if stride >= seq_len:
            break

        # Indices that get updated
        # At iteration d, element i gets combined with element i - stride
        update_indices = torch.arange(stride, seq_len, device=device)
        source_indices = update_indices - stride

        # Get elements to combine
        A_source = A_work[:, source_indices]  # [B, num_updates, D, D]
        b_source = b_work[:, source_indices]  # [B, num_updates, D]
        A_target = A_work[:, update_indices]  # [B, num_updates, D, D]
        b_target = b_work[:, update_indices]  # [B, num_updates, D]

        # Combine: (A_target, b_target) * (A_source, b_source)
        # = (A_target @ A_source, A_target @ b_source + b_target)
        A_new = torch.einsum("bnij,bnjk->bnik", A_target, A_source)
        b_new = torch.einsum("bnij,bnj->bni", A_target, b_source) + b_target

        # Update working arrays
        A_work[:, update_indices] = A_new
        b_work[:, update_indices] = b_new

    # The b_work now contains all states
    return b_work


def associative_scan(
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    elements: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Generic associative scan for arbitrary associative operators.

    This is a more general version that works with any associative binary operator,
    not just linear recurrences.

    Args:
        operator: Associative binary operator f(a, b) -> c
        elements: Input tensor [batch, seq_len, ...]
        dim: Dimension along which to scan

    Returns:
        Scanned tensor with prefix reductions

    Example:
        >>> # Cumulative sum using associative scan
        >>> x = torch.tensor([[1, 2, 3, 4, 5]])
        >>> associative_scan(torch.add, x, dim=1)
        tensor([[ 1,  3,  6, 10, 15]])
    """
    seq_len = elements.shape[dim]

    if seq_len <= 1:
        return elements

    result = elements.clone()

    # Iterative doubling
    stride = 1
    while stride < seq_len:
        # Get indices for elements to update
        indices_to_update = list(range(stride, seq_len))
        indices_sources = list(range(0, seq_len - stride))

        # Combine elements
        # result[i] = operator(result[i - stride], result[i])
        for i, src_i in zip(indices_to_update, indices_sources):
            left = result.select(dim, src_i)
            right = result.select(dim, i)
            combined = operator(left, right)
            result.select(dim, i).copy_(combined)

        stride *= 2

    return result


def make_diagonal_A(
    eigenvalues: torch.Tensor,
) -> torch.Tensor:
    """
    Create a diagonal state matrix from eigenvalues.

    Diagonal matrices enable efficient parallel scan because:
    A^k is just element-wise power of diagonal entries.

    Args:
        eigenvalues: Diagonal entries [state_dim] (can be complex)

    Returns:
        A: Diagonal matrix [state_dim, state_dim]
    """
    return torch.diag(eigenvalues)


def discretize_continuous_ssm(
    A_continuous: torch.Tensor,
    B_continuous: torch.Tensor,
    dt: float,
    method: str = "zoh",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize continuous-time state space model.

    Continuous: dx/dt = A_c x + B_c u
    Discrete: x_{k+1} = A_d x_k + B_d u_k

    Args:
        A_continuous: Continuous state matrix [state_dim, state_dim]
        B_continuous: Continuous input matrix [state_dim, input_dim]
        dt: Sampling period
        method: Discretization method ("zoh" for zero-order hold, "euler" for forward Euler)

    Returns:
        A_discrete: Discretized state matrix
        B_discrete: Discretized input matrix
    """
    if method == "euler":
        # Forward Euler: simple but less accurate
        A_d = torch.eye(A_continuous.shape[0], device=A_continuous.device) + dt * A_continuous
        B_d = dt * B_continuous
    elif method == "zoh":
        # Zero-order hold: exact for constant input over dt
        # A_d = exp(A_c * dt)
        # B_d = A_c^{-1} (A_d - I) B_c

        # Use matrix exponential
        A_d = torch.matrix_exp(A_continuous * dt)

        # For B_d, we need to handle potential singularity
        I = torch.eye(A_continuous.shape[0], device=A_continuous.device, dtype=A_continuous.dtype)

        # Use series approximation for numerical stability
        # B_d = dt * (I + A_c*dt/2 + (A_c*dt)^2/6 + ...) B_c
        B_d = dt * B_continuous
        term = dt * B_continuous
        for k in range(2, 10):  # 10 terms usually sufficient
            term = (dt / k) * torch.einsum("ij,jk->ik", A_continuous, term)
            B_d = B_d + term
    else:
        raise ValueError(f"Unknown discretization method: {method}")

    return A_d, B_d


# ============================================================================
# Selective Scan Operations (Mamba-style)
# ============================================================================

def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Selective scan operation as in Mamba architecture.

    Key difference from standard SSM: A, B, C are input-dependent, allowing
    the model to selectively propagate or forget information.

    Args:
        u: Input sequence [batch, seq_len, input_dim]
        delta: Time step (input-dependent) [batch, seq_len, state_dim]
        A: State matrix (input-dependent) [batch, seq_len, state_dim]
        B: Input projection [batch, seq_len, state_dim]
        C: Output projection [batch, seq_len, state_dim]
        D: Skip connection [input_dim] (optional)

    Returns:
        y: Output sequence [batch, seq_len, input_dim]
    """
    batch_size, seq_len, state_dim = delta.shape
    input_dim = u.shape[-1]
    device = u.device
    dtype = u.dtype

    # Discretize A and B using delta (zero-order hold)
    # A_bar = exp(delta * A)
    # B_bar = (A_bar - I) / A * B ≈ delta * B (when delta*A is small)

    delta_A = delta * A  # [batch, seq_len, state_dim]
    A_bar = torch.exp(delta_A)  # [batch, seq_len, state_dim]

    # Simplified B discretization (Euler approximation)
    B_bar = delta * B  # [batch, seq_len, state_dim]

    # Sequential scan (could be replaced with parallel scan)
    x = torch.zeros(batch_size, state_dim, device=device, dtype=dtype)
    outputs = []

    for t in range(seq_len):
        # x = A_bar * x + B_bar * u
        x = A_bar[:, t] * x + B_bar[:, t] * u[:, t, :state_dim] if input_dim >= state_dim else A_bar[:, t] * x + B_bar[:, t].unsqueeze(-1) * u[:, t, :1]

        # y = C * x
        y_t = (C[:, t] * x).sum(dim=-1, keepdim=True)

        if D is not None:
            y_t = y_t + D * u[:, t]

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


def parallel_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parallel version of selective scan for training.

    Uses the same associative scan trick, but now A_bar varies with time.

    The key insight is that even with time-varying A, we can still use
    parallel scan because the operator remains associative.
    """
    batch_size, seq_len, state_dim = delta.shape
    device = u.device
    dtype = u.dtype

    # Discretize
    delta_A = delta * A
    A_bar = torch.exp(delta_A)  # [batch, seq_len, state_dim]
    B_bar = delta * B

    # Compute Bu term
    if u.shape[-1] >= state_dim:
        Bu = B_bar * u[:, :, :state_dim]
    else:
        Bu = B_bar * u[:, :, :1]  # Broadcast if input_dim < state_dim

    # For diagonal A, parallel scan simplifies significantly
    # x_t = A_bar_t * x_{t-1} + Bu_t
    # This is element-wise, so we can scan each dimension independently

    # Initialize with zeros
    x = torch.zeros_like(Bu)

    # Use cumulative product for A_bar (since it's diagonal)
    # log(A_bar) cumsum then exp gives us product of A_bar from t to T
    log_A_bar = delta_A  # log(exp(delta_A)) = delta_A
    log_A_bar_cumsum = torch.cumsum(log_A_bar, dim=1)  # [batch, seq_len, state_dim]

    # For each position t, we need sum_{k=0}^{t} (prod_{j=k+1}^{t} A_bar_j) * Bu_k
    # = sum_{k=0}^{t} exp(sum_{j=k+1}^{t} log A_bar_j) * Bu_k
    # = sum_{k=0}^{t} exp(cumsum[t] - cumsum[k]) * Bu_k

    # Efficient computation using exp-log trick
    # Shift cumsum to get cumsum[k-1] for computing products from k to t
    log_A_bar_cumsum_shifted = F.pad(log_A_bar_cumsum[:, :-1], (0, 0, 1, 0), value=0)

    # For each output position t, sum over all input positions k <= t
    # x_t = sum_{k=0}^{t} exp(cumsum[t] - cumsum[k]) * Bu_k
    # This is a convolution-like operation

    # Reshape for batched computation
    # [batch, t, k, state_dim] would be too large, so we compute iteratively
    # but in a vectorized way

    for t in range(seq_len):
        if t == 0:
            x[:, t] = Bu[:, t]
        else:
            # Product of A_bar from k+1 to t for all k < t
            # = exp(cumsum[t] - cumsum[k]) for k = 0, ..., t-1
            log_products = log_A_bar_cumsum[:, t:t+1] - log_A_bar_cumsum_shifted[:, :t+1]  # [batch, t+1, state_dim]
            products = torch.exp(log_products)  # [batch, t+1, state_dim]

            # Weighted sum of Bu
            x[:, t] = (products * Bu[:, :t+1]).sum(dim=1)

    # Output: y = C * x + D * u
    y = (C * x).sum(dim=-1, keepdim=True)

    if D is not None:
        y = y + D * u

    return y
