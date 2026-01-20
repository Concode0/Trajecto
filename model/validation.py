"""Input validation utilities for training pipeline."""

import torch
import numpy as np
from typing import Dict, Tuple

from model.config import Config


def validate_batch_tensors(
    batch: Dict[str, torch.Tensor],
    expected_sensor_channels: int = Config.VALIDATION.EXPECTED_SENSOR_CHANNELS,
    expected_spatial_dims: int = Config.VALIDATION.EXPECTED_SPATIAL_DIMS
) -> None:
    """Validate training batch tensor shapes and values.

    Args:
        batch: Batch dictionary with keys [imu_seq_raw, gt_vel_w, gt_pos_w, len]
        expected_sensor_channels: Expected number of sensor channels (default: 7)
        expected_spatial_dims: Expected spatial dimensions for vel/pos (default: 3)

    Raises:
        ValueError: If any validation fails
    """
    # Check required keys
    required_keys = ["imu_seq_raw", "gt_vel_w", "gt_pos_w", "len"]
    missing_keys = [k for k in required_keys if k not in batch]
    if missing_keys:
        raise ValueError(f"Batch missing required keys: {missing_keys}")

    sensor_raw = batch["imu_seq_raw"]
    gt_vel = batch["gt_vel_w"]
    gt_pos = batch["gt_pos_w"]
    seq_lens = batch["len"]

    # Validate sensor tensor
    if sensor_raw.ndim != 3:
        raise ValueError(
            f"sensor_raw must be 3D [B, T, C], got shape {sensor_raw.shape}"
        )

    B, T, C = sensor_raw.shape
    if C != expected_sensor_channels:
        raise ValueError(
            f"Expected {expected_sensor_channels} sensor channels, got {C}"
        )

    # Validate ground truth tensors
    if gt_vel.shape != (B, T, expected_spatial_dims):
        raise ValueError(
            f"gt_vel shape mismatch: expected [{B}, {T}, {expected_spatial_dims}], "
            f"got {gt_vel.shape}"
        )

    if gt_pos.shape != (B, T, expected_spatial_dims):
        raise ValueError(
            f"gt_pos shape mismatch: expected [{B}, {T}, {expected_spatial_dims}], "
            f"got {gt_pos.shape}"
        )

    # Validate sequence lengths
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape {seq_lens.shape}")

    if seq_lens.shape[0] != B:
        raise ValueError(
            f"seq_lens batch size mismatch: expected {B}, got {seq_lens.shape[0]}"
        )

    # Check for NaN/Inf in inputs
    if torch.isnan(sensor_raw).any():
        raise ValueError("sensor_raw contains NaN values")

    if torch.isinf(sensor_raw).any():
        raise ValueError("sensor_raw contains Inf values")

    if torch.isnan(gt_vel).any():
        raise ValueError("gt_vel contains NaN values")

    if torch.isnan(gt_pos).any():
        raise ValueError("gt_pos contains NaN values")


def validate_loss_dict(
    losses: Dict[str, torch.Tensor],
    epoch: int,
    batch_idx: int
) -> Tuple[bool, str]:
    """Validate loss dictionary and detect issues.

    Args:
        losses: Dictionary of loss tensors
        epoch: Current epoch number (for logging)
        batch_idx: Current batch index (for logging)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: False if any loss is NaN/Inf
        - error_message: Detailed error description if invalid
    """
    invalid_losses = []

    for name, loss in losses.items():
        if not isinstance(loss, torch.Tensor):
            invalid_losses.append(f"{name}: not a tensor (type={type(loss)})")
            continue

        if torch.isnan(loss):
            invalid_losses.append(f"{name}={loss.item():.6f} (NaN)")
        elif torch.isinf(loss):
            invalid_losses.append(f"{name}={loss.item():.6f} (Inf)")

    if invalid_losses:
        error_msg = (
            f"Invalid losses at epoch {epoch}, batch {batch_idx}:\n"
            f"  {', '.join(invalid_losses)}\n"
            f"  All losses: {', '.join([f'{k}={v.item():.6f}' for k, v in losses.items()])}"
        )
        return False, error_msg

    return True, ""


def validate_normalization_stats(
    mean: torch.Tensor,
    std: torch.Tensor,
    expected_channels: int = 7
) -> None:
    """Validate normalization statistics.

    Args:
        mean: Normalization mean tensor
        std: Normalization std tensor
        expected_channels: Expected number of channels

    Raises:
        ValueError: If validation fails
    """
    if mean.shape != (expected_channels,):
        raise ValueError(
            f"mean shape mismatch: expected [{expected_channels}], got {mean.shape}"
        )

    if std.shape != (expected_channels,):
        raise ValueError(
            f"std shape mismatch: expected [{expected_channels}], got {std.shape}"
        )

    if torch.isnan(mean).any():
        raise ValueError("Normalization mean contains NaN")

    if torch.isnan(std).any():
        raise ValueError("Normalization std contains NaN")

    if (std <= 0).any():
        raise ValueError(f"Normalization std contains non-positive values: {std}")
