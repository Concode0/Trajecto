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
This module provides fundamental utility functions for quaternion and rotation
matrix manipulations, which are crucial for IMU-based state estimation algorithms
like the Extended Kalman Filter (EKF) and Error-State Kalman Filter (ESKF).

It includes functions for:
- Calculating a quaternion from two vectors (useful for initial alignment).
- Performing quaternion multiplication (Hamilton product).
- Converting quaternions to 3x3 rotation matrices.
- Converting small angle error vectors to quaternions for error injection.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def quaternion_from_two_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Calculates the quaternion that rotates vector `v1` to vector `v2`.

    This function is robust, handling parallel and anti-parallel input vectors.
    It's commonly used for initial attitude alignment, where `v1` might be an
    IMU-measured gravity vector and `v2` the world-frame gravity vector.

    Args:
        v1 (torch.Tensor): A batch of 3D vectors (source vectors).
            - Shape: (Batch, 3)
            - Unit: Arbitrary (normalized internally)
            - Frame: Source Frame
        v2 (torch.Tensor): A batch of 3D vectors (target vectors).
            - Shape: (Batch, 3)
            - Unit: Arbitrary (normalized internally)
            - Frame: Target Frame

    Returns:
        torch.Tensor: A batch of quaternions (w, x, y, z).
            - Shape: (Batch, 4)
            - Frame: Rotation from Source to Target
    """
    # Normalize input vectors to unit length. This is crucial for geometric
    # calculations and for `torch.cross` and `torch.sum` to yield correct angles.
    v1_norm = F.normalize(v1, p=2, dim=-1)
    v2_norm = F.normalize(v2, p=2, dim=-1)

    # The dot product (v1_norm . v2_norm) gives cos(theta), where theta is
    # the angle between the two vectors.
    dot = torch.sum(v1_norm * v2_norm, dim=-1)

    # Initialize a batch of identity quaternions [1, 0, 0, 0].
    # This is returned if v1 and v2 are nearly parallel (no rotation needed).
    identity_quat = torch.zeros_like(v1.new_empty(v1.shape[0], 4))
    identity_quat[:, 0] = 1.0

    # Handle anti-parallel vectors (vectors pointing in opposite directions, angle = 180 degrees).
    # In this case, the cross product is zero, making `axis` undefined.
    # We find an arbitrary perpendicular axis to `v1_norm` for the 180-degree rotation.
    # First attempt: cross with [1,0,0]. If v1 is parallel to [1,0,0], cross with [0,1,0].
    axis = torch.linalg.cross(
        v1_norm,
        torch.tensor([1.0, 0.0, 0.0], device=v1.device, dtype=v1.dtype).expand_as(
            v1_norm
        ),
    )
    # Check for cases where `axis` is still zero (i.e., `v1_norm` was parallel to [1,0,0]).
    mask_parallel_x = torch.all(
        torch.isclose(axis, torch.zeros_like(axis)), dim=-1
    )
    if mask_parallel_x.any():
        axis[mask_parallel_x] = torch.cross(
            v1_norm[mask_parallel_x],
            torch.tensor([0.0, 1.0, 0.0], device=v1.device, dtype=v1.dtype).expand_as(
                v1_norm[mask_parallel_x]
            ),
        )

    # Normalize the chosen axis vector.
    axis = F.normalize(axis, p=2, dim=-1, eps=1e-8)
    # The quaternion for 180-degree rotation around `axis` is [0, axis_x, axis_y, axis_z].
    anti_parallel_quat = torch.cat([torch.zeros_like(dot).unsqueeze(-1), axis], dim=-1)

    # General case: Calculate rotation axis and angle.
    # The cross product (v1_norm x v2_norm) gives a vector perpendicular to both,
    # whose magnitude is |v1_norm||v2_norm|sin(theta) = sin(theta).
    cross_prod = torch.cross(v1_norm, v2_norm, dim=-1)
    cross_prod_norm = torch.norm(cross_prod, p=2, dim=-1)

    # Angle of rotation is found using atan2(sin(theta), cos(theta)).
    # This is more robust than `acos(dot)` as it covers the full 0-180 degree range.
    angle = torch.atan2(cross_prod_norm, dot)
    half_angle = angle / 2.0

    # Quaternion components: q_w = cos(theta/2), q_xyz = axis * sin(theta/2)
    q_w = torch.cos(half_angle)
    sin_half_angle = torch.sin(half_angle)

    # The rotation axis is the normalized cross product vector.
    # `eps=1e-8` is a small jitter value to prevent division by zero if `cross_prod_norm` is very small
    # (e.g., when vectors are nearly parallel).
    rotation_axis = F.normalize(cross_prod, p=2, dim=-1, eps=1e-8)
    q_xyz = rotation_axis * sin_half_angle.unsqueeze(-1)

    # Combine into a general quaternion.
    general_quat = torch.cat([q_w.unsqueeze(-1), q_xyz], dim=-1)

    # Select the appropriate quaternion based on the angle between v1 and v2:
    # - If dot_product > 0.99999 (almost parallel), use identity.
    # - If dot_product < -0.99999 (almost anti-parallel), use anti_parallel_quat.
    # - Otherwise, use the general_quat.
    quat = torch.where(
        dot.unsqueeze(-1) > 0.99999,
        identity_quat,
        torch.where(
            dot.unsqueeze(-1) < -0.99999, anti_parallel_quat, general_quat
        ),
    )

    # Final normalization to ensure unit quaternion property, correcting any
    # minor floating-point errors accumulated during calculations.
    return F.normalize(quat, p=2, dim=-1)


def quaternion_multiply(quat_1: torch.Tensor, quat_2: torch.Tensor) -> torch.Tensor:
    """Performs batch-aware multiplication of two quaternions (Hamilton product).
    `q_new = q1 * q2`. This corresponds to applying the rotation `q2` followed
    by the rotation `q1`.

    Args:
        quat_1 (torch.Tensor): The first quaternion (w, x, y, z).
            - Shape: (Batch, 4)
        quat_2 (torch.Tensor): The second quaternion (w, x, y, z).
            - Shape: (Batch, 4)

    Returns:
        torch.Tensor: The resulting quaternion (w, x, y, z).
            - Shape: (Batch, 4)
    """
    # Decompose quaternions into scalar (w) and vector (x,y,z) parts.
    w1, x1, y1, z1 = quat_1.unbind(-1)
    w2, x2, y2, z2 = quat_2.unbind(-1)

    # Hamilton product formula:
    # q_new.w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    # q_new.x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # q_new.y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    # q_new.z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)


def quaternion_to_rotation_matrix(quat_b_to_w: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions to a batch of 3x3 rotation matrices.

    The resulting matrix `R_bw` rotates a vector from the body frame (`b`)
    to the world frame (`w`). `v_w = R_bw * v_b`.

    Args:
        quat_b_to_w (torch.Tensor): Body-to-world quaternion (w, x, y, z).
            - Shape: (Batch, 4)
            - Frame: Body-to-World

    Returns:
        torch.Tensor: Batch of 3x3 rotation matrices.
            - Shape: (Batch, 3, 3)
            - Frame: Body-to-World
    """
    # Normalize the quaternion to ensure it's a unit quaternion.
    # This is essential for the conversion formula to produce a valid rotation matrix.
    quat_norm = F.normalize(quat_b_to_w, p=2, dim=-1)
    w, x, y, z = quat_norm.unbind(-1)

    # Pre-compute all squared and product terms to ensure device consistency
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Standard conversion formula from unit quaternion to rotation matrix:
    # Row 0:
    #   R_00 = 1 - 2y^2 - 2z^2 = w^2 + x^2 - y^2 - z^2
    #   R_01 = 2xy - 2wz
    #   R_02 = 2xz + 2wy
    # Row 1:
    #   R_10 = 2xy + 2wz
    #   R_11 = 1 - 2x^2 - 2z^2 = w^2 - x^2 + y^2 - z^2
    #   R_12 = 2yz - 2wx
    # Row 2:
    #   R_20 = 2xz - 2wy
    #   R_21 = 2yz + 2wx
    #   R_22 = 1 - 2x^2 - 2y^2 = w^2 - x^2 - y^2 + z^2
    # Build rotation matrix using only tensor operations (no Python scalars)
    two = xx.new_tensor(2.0)  # Create scalar 2.0 on same device/dtype as input
    rot_mat_b_to_w = torch.stack(
        [
            torch.stack([ww + xx - yy - zz, two*(xy - wz), two*(xz + wy)], -1),
            torch.stack([two*(xy + wz), ww - xx + yy - zz, two*(yz - wx)], -1),
            torch.stack([two*(xz - wy), two*(yz + wx), ww - xx - yy + zz], -1),
        ],
        -2,
    )
    return rot_mat_b_to_w


def small_angle_to_quaternion(small_angle_vec: torch.Tensor) -> torch.Tensor:
    """Converts a small 3D angle vector (rotation vector) to a quaternion.

    This function is particularly useful in Error-State Kalman Filters (ESKFs)
    where a small angular error `δθ` (represented as a 3D vector where magnitude
    is the angle and direction is the axis) is estimated. This `δθ` is then
    converted to a quaternion and multiplicatively applied to the nominal
    quaternion to correct the orientation.

    Args:
        small_angle_vec (torch.Tensor): Small rotation vector.
            - Shape: (Batch, 3)
            - Unit: rad
            - Frame: Body

    Returns:
        torch.Tensor: Correction quaternion (w, x, y, z).
            - Shape: (Batch, 4)
            - Frame: Body
    """
    # Calculate the magnitude of the rotation vector, which represents the angle of rotation.
    angle_sq_norm = torch.sum(small_angle_vec * small_angle_vec, dim=-1, keepdim=True)
    angle_norm = torch.sqrt(angle_sq_norm)  # Angle θ

    # Calculate half the angle for quaternion components.
    half_angle = angle_norm / 2.0  # θ/2

    # The scalar (w) part of the quaternion is cos(θ/2).
    q_w = torch.cos(half_angle)

    # The vector (xyz) part of the quaternion is (rotation_axis * sin(θ/2)).
    sin_half_angle = torch.sin(half_angle)

    # Normalize the small_angle_vec to get the rotation axis.
    # `eps=1e-8` is a jitter value to prevent division by zero if `angle_norm` is
    # extremely small (i.e., near-zero rotation).
    rotation_axis = F.normalize(small_angle_vec, p=2, dim=-1, eps=1e-8)
    q_xyz = rotation_axis * sin_half_angle

    # Combine scalar and vector parts and normalize to ensure it's a unit quaternion.
    return F.normalize(torch.cat([q_w, q_xyz], dim=-1), p=2, dim=-1)