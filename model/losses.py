from typing import List

import torch
import torch.nn.functional as F

def magnitude_loss(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    mask: torch.Tensor,
    use_mutual_exclusivity: bool = True,
    moving_threshold: float = 0.01  # 10 mm/s - clearly moving
) -> torch.Tensor:
    """
    Velocity magnitude loss with optional mutual exclusivity masking.

    Args:
        pred_vel: Predicted velocity [B, T, 3]
        gt_vel: Ground truth velocity [B, T, 3]
        mask: Valid timestep mask [B, T]
        use_mutual_exclusivity: If True, only apply loss during clear motion
        moving_threshold: Minimum velocity to be considered "clearly moving" (m/s)

    Returns:
        Scalar loss value
    """
    mask_3d = mask.unsqueeze(-1)
    pred_mag = torch.norm(pred_vel, dim=-1, keepdim=True)
    gt_mag = torch.norm(gt_vel, dim=-1, keepdim=True)
    loss = F.l1_loss(pred_mag, gt_mag, reduction='none')

    # Structural gradient conflict resolution: only apply during clear motion
    if use_mutual_exclusivity:
        moving_mask = (gt_mag > moving_threshold).float()
        mask_3d = mask_3d * moving_mask

    return (loss * mask_3d).sum() / (mask_3d.sum() + 1e-8)


def context_aware_direction_loss(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    mask: torch.Tensor,
    gt_zupt: torch.Tensor,
    gyro_raw: torch.Tensor,
    base_weight: float = 1.0,
    gyro_sensitivity: float = 2.0,
    max_context_weight: float = 3.0
) -> torch.Tensor:
    """
    Gyro-based Context-Aware Cosine Loss.
    Increases penalty during fast rotations (high gyro magnitude).

    Args:
        pred_vel: Predicted velocity [B, T, 3]
        gt_vel: Ground truth velocity [B, T, 3]
        mask: Valid timestep mask [B, T]
        gt_zupt: Ground truth ZUPT label [B, T, 1]
        gyro_raw: Raw gyroscope readings [B, T, 3] in rad/s
        base_weight: Baseline weight for all timesteps (default: 1.0)
        gyro_sensitivity: Scaling factor for gyro magnitude (default: 2.0)
        max_context_weight: Maximum context weight to prevent dominance (default: 3.0)

    Returns:
        Scalar loss value

    Note:
        Context weight is capped at max_context_weight to prevent extreme amplification
        during very fast rotations, which can destabilize DWA balancing.
    """
    # 1. Base Cosine Loss (0.0 ~ 2.0)
    cos_sim = F.cosine_similarity(pred_vel, gt_vel, dim=-1, eps=1e-6)
    base_loss = 1.0 - cos_sim

    # 2. Extract Context (Gyro Magnitude)
    # gyro_raw: [Batch, Time, 3] -> Norm -> [Batch, Time]
    gyro_mag = torch.norm(gyro_raw, dim=-1)  # rad/s

    # 3. Dynamic Weighting with Cap
    # e.g., if gyro is 3 rad/s, weight = 1.0 + 2.0*3 = 7.0 → clamped to 3.0
    context_weight = base_weight + gyro_sensitivity * gyro_mag
    context_weight = torch.clamp(context_weight, min=base_weight, max=max_context_weight)

    # 4. Apply Weight
    weighted_loss = base_loss * context_weight

    # Masking (Ignore direction during ZUPT)
    moving_mask = mask * (1.0 - gt_zupt.squeeze(-1))

    return (weighted_loss * moving_mask).sum() / (moving_mask.sum() + 1e-8)


def zupt_loss(
    pred_zupt: torch.Tensor,
    gt_zupt: torch.Tensor,
    mask: torch.Tensor,
    use_mutual_exclusivity: bool = True
) -> torch.Tensor:
    """
    ZUPT (Zero-velocity update) loss with optional mutual exclusivity masking.

    Args:
        pred_zupt: Predicted ZUPT probability [B, T, 1]
        gt_zupt: Ground truth ZUPT label [B, T, 1] (derived from velocity threshold)
        mask: Valid timestep mask [B, T]
        use_mutual_exclusivity: If True, only apply loss during clear static periods

    Returns:
        Scalar loss value

    Note:
        Creates a dead zone (0.005 < vel < 0.01) where neither magnitude nor ZUPT
        loss applies strongly, reducing gradient conflicts. The threshold is applied
        when computing gt_zupt, not here.
    """
    mask_3d = mask.unsqueeze(-1)
    loss = F.binary_cross_entropy_with_logits(pred_zupt, gt_zupt, reduction='none')

    # Structural gradient conflict resolution: only apply during clear static periods
    # gt_zupt is already 1.0 when vel < threshold, but we add extra gating here
    if use_mutual_exclusivity:
        # Only apply ZUPT loss when gt says we're static (gt_zupt=1)
        # This prevents conflict with magnitude loss which applies during motion
        static_mask = gt_zupt  # Already binary: 1 when static, 0 when moving
        mask_3d = mask_3d * static_mask

    return (loss * mask_3d).sum() / (mask_3d.sum() + 1e-8)


def covariance_nll_loss(
    innovation: torch.Tensor,
    pred_R: torch.Tensor,
    mask: torch.Tensor,
    tcn_mask: torch.Tensor
) -> torch.Tensor:
    final_mask = tcn_mask & mask
    if not final_mask.any():
        return torch.tensor(0.0, device=innovation.device)

    inn_valid = innovation[final_mask]
    R_valid = pred_R[final_mask]

    variance = F.softplus(R_valid) + 1e-4
    variance = torch.clamp(variance, min=1e-4, max=3.0)
    nll = 0.5 * (torch.square(inn_valid) / variance + torch.log(variance))
    return nll.mean()


def regularization_loss(vel_correction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_3d = mask.unsqueeze(-1)
    loss = vel_correction ** 2
    return (loss * mask_3d).sum() / (mask_3d.sum() + 1e-8)


def fft_loss(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    mask: torch.Tensor,
    use_windowing: bool = True
) -> torch.Tensor:
    """
    Computes FFT loss between predicted and ground truth velocities.
    Uses L1 loss on the magnitude of the frequency components.

    Args:
        pred_vel: Predicted velocity [B, T, 3]
        gt_vel: Ground truth velocity [B, T, 3]
        mask: Valid timestep mask [B, T]
        use_windowing: If True, apply Hann window to reduce edge artifacts

    Returns:
        Scalar loss value

    Note:
        Windowing reduces spectral leakage from mask edges (valid→zero transitions).
    """
    _, T, _ = pred_vel.shape
    device = pred_vel.device

    # mask: (B, T) -> (B, T, 1)
    mask_3d = mask.unsqueeze(-1).float()

    # Apply Hann window to reduce edge artifacts from masking
    if use_windowing:
        # Create Hann window: w(n) = 0.5 * (1 - cos(2π*n/(T-1)))
        n = torch.arange(T, device=device, dtype=torch.float32)
        hann_window = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * n / (T - 1)))
        # Shape: [1, T, 1] for broadcasting
        hann_window = hann_window.view(1, T, 1)

        # Apply window to mask (smooth transitions at edges)
        # Combine mask and window: only window where mask is valid
        window_mask = mask_3d * hann_window
    else:
        window_mask = mask_3d

    # Apply windowing/masking
    pred_windowed = pred_vel * window_mask
    gt_windowed = gt_vel * window_mask

    # Compute Real FFT along the time dimension (dim=1)
    # Output shape: (B, T//2 + 1, 3)
    pred_fft = torch.fft.rfft(pred_windowed, dim=1)
    gt_fft = torch.fft.rfft(gt_windowed, dim=1)

    # Compute Magnitude
    pred_mag = torch.abs(pred_fft)
    gt_mag = torch.abs(gt_fft)

    eps = 1e-8
    loss = F.l1_loss(torch.log(pred_mag + eps), torch.log(gt_mag + eps))

    return loss


def sliding_window_delta_loss(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    mask: torch.Tensor,
    window_sizes: List[int] = [25, 100, 250],
    window_weights: List[float] = [0.3, 0.4, 0.3],
    stride: int = 10
) -> torch.Tensor:
    """Semi-loop closure loss via sliding window delta position comparison.

    Vectorized implementation using advanced indexing instead of nested loops.

    For each window [t, t+W]:
        delta_pred = pred_pos[t+W] - pred_pos[t]
        delta_gt = gt_pos[t+W] - gt_pos[t]
        loss += |delta_pred - delta_gt|

    This enforces that predicted displacement over any window matches GT,
    providing drift correction without actual loop closure.

    Args:
        pred_pos: Predicted positions [B, T, 3]
        gt_pos: Ground truth positions [B, T, 3]
        mask: Valid timestep mask [B, T]
        window_sizes: Window sizes in timesteps (default: 0.5s, 2s, 5s @ 50Hz)
        window_weights: Weight for each window size (must sum to ~1.0)
        stride: Sliding window stride for efficiency

    Returns:
        Scalar loss value
    """
    B, T, _ = pred_pos.shape
    device = pred_pos.device
    total_loss = torch.tensor(0.0, device=device)

    for w_size, w_weight in zip(window_sizes, window_weights):
        if T <= w_size:
            continue

        # Generate all window start indices with stride
        start_indices = torch.arange(0, T - w_size, stride, device=device)
        end_indices = start_indices + w_size
        num_windows = len(start_indices)

        if num_windows == 0:
            continue

        # Vectorized validity check: [B, num_windows]
        valid_start = mask[:, start_indices]  # [B, num_windows]
        valid_end = mask[:, end_indices]      # [B, num_windows]
        valid_windows = valid_start & valid_end

        # Vectorized delta computation: [B, num_windows, 3]
        delta_pred = pred_pos[:, end_indices] - pred_pos[:, start_indices]
        delta_gt = gt_pos[:, end_indices] - gt_pos[:, start_indices]

        # L1 loss per window per batch: [B, num_windows]
        window_losses = F.l1_loss(delta_pred, delta_gt, reduction='none').mean(dim=-1)

        # Apply mask and average over valid windows
        masked_losses = window_losses * valid_windows.float()
        valid_count = valid_windows.float().sum()

        if valid_count > 0:
            window_loss = masked_losses.sum() / valid_count
            total_loss = total_loss + w_weight * window_loss

    return total_loss


def integrate_velocity_to_position(
    velocity: torch.Tensor,
    initial_pos: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """Integrate velocity to get position using cumulative sum.

    Args:
        velocity: Velocity tensor [B, T, 3] in m/s
        initial_pos: Initial position [B, 3] or [B, 1, 3]
        dt: Time step in seconds

    Returns:
        Position tensor [B, T, 3] in meters
    """
    # Cumulative integration: pos[t] = pos[0] + sum(vel[0:t] * dt)
    displacement = torch.cumsum(velocity * dt, dim=1)

    if initial_pos.dim() == 2:
        initial_pos = initial_pos.unsqueeze(1)

    return initial_pos + displacement