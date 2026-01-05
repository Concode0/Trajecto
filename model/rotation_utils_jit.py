"""
JIT-optimized rotation utilities for faster inference and training.

This module provides TorchScript-compiled versions of rotation operations
that are heavily used in the ESKF-TCN forward loop.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def quaternion_multiply_jit(quat_1: torch.Tensor, quat_2: torch.Tensor) -> torch.Tensor:
    """JIT-compiled batch quaternion multiplication (Hamilton product).

    Args:
        quat_1: First quaternion (w, x, y, z) - Shape: (Batch, 4)
        quat_2: Second quaternion (w, x, y, z) - Shape: (Batch, 4)

    Returns:
        Resulting quaternion (w, x, y, z) - Shape: (Batch, 4)
    """
    w1, x1, y1, z1 = quat_1.unbind(-1)
    w2, x2, y2, z2 = quat_2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), -1)


@torch.jit.script
def quaternion_to_rotation_matrix_jit(quat_b_to_w: torch.Tensor) -> torch.Tensor:
    """JIT-compiled conversion from quaternion to rotation matrix.

    Args:
        quat_b_to_w: Body-to-world quaternion (w, x, y, z) - Shape: (Batch, 4)

    Returns:
        Batch of 3x3 rotation matrices - Shape: (Batch, 3, 3)
    """
    # Normalize quaternion
    quat_norm = F.normalize(quat_b_to_w, p=2, dim=-1)
    w, x, y, z = quat_norm.unbind(-1)

    # Compute rotation matrix elements
    rot_mat_b_to_w = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    return rot_mat_b_to_w


@torch.jit.script
def small_angle_to_quaternion_jit(small_angle_vec: torch.Tensor) -> torch.Tensor:
    """JIT-compiled conversion from small angle vector to quaternion.

    Args:
        small_angle_vec: Small rotation vector - Shape: (Batch, 3)

    Returns:
        Correction quaternion (w, x, y, z) - Shape: (Batch, 4)
    """
    angle_sq_norm = torch.sum(small_angle_vec * small_angle_vec, dim=-1, keepdim=True)
    angle_norm = torch.sqrt(angle_sq_norm)
    half_angle = angle_norm / 2.0

    q_w = torch.cos(half_angle)
    sin_half_angle = torch.sin(half_angle)

    rotation_axis = F.normalize(small_angle_vec, p=2, dim=-1, eps=1e-8)
    q_xyz = rotation_axis * sin_half_angle

    return F.normalize(torch.cat([q_w, q_xyz], dim=-1), p=2, dim=-1)


@torch.jit.script
def batch_quaternion_rotate_vector_jit(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """JIT-compiled efficient vector rotation using quaternion.

    Faster than converting to rotation matrix for single vector operations.

    Args:
        quat: Quaternion (w, x, y, z) - Shape: (Batch, 4)
        vec: Vector to rotate - Shape: (Batch, 3)

    Returns:
        Rotated vector - Shape: (Batch, 3)
    """
    # Normalize quaternion
    quat_norm = F.normalize(quat, p=2, dim=-1)
    w, x, y, z = quat_norm.unbind(-1)

    # Efficient quaternion-vector product
    # v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    q_xyz = torch.stack([x, y, z], dim=-1)

    # First cross product: q_xyz x v
    uv = torch.cross(q_xyz, vec, dim=-1)

    # Second cross product: q_xyz x (q_xyz x v)
    uuv = torch.cross(q_xyz, uv, dim=-1)

    # Final result
    return vec + 2.0 * w.unsqueeze(-1) * uv + 2.0 * uuv


if __name__ == "__main__":
    # Test JIT-compiled functions
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing JIT-compiled rotation utilities on {device}")

    batch_size = 32

    # Test quaternion_to_rotation_matrix_jit
    quat = F.normalize(torch.randn(batch_size, 4, device=device), dim=-1)
    rot_mat = quaternion_to_rotation_matrix_jit(quat)
    print(f"✓ quaternion_to_rotation_matrix_jit: {rot_mat.shape}")

    # Test quaternion_multiply_jit
    quat2 = F.normalize(torch.randn(batch_size, 4, device=device), dim=-1)
    quat_result = quaternion_multiply_jit(quat, quat2)
    print(f"✓ quaternion_multiply_jit: {quat_result.shape}")

    # Test small_angle_to_quaternion_jit
    small_angle = torch.randn(batch_size, 3, device=device) * 0.01
    quat_small = small_angle_to_quaternion_jit(small_angle)
    print(f"✓ small_angle_to_quaternion_jit: {quat_small.shape}")

    # Test batch_quaternion_rotate_vector_jit
    vec = torch.randn(batch_size, 3, device=device)
    vec_rotated = batch_quaternion_rotate_vector_jit(quat, vec)
    print(f"✓ batch_quaternion_rotate_vector_jit: {vec_rotated.shape}")

    print("\nAll JIT functions compiled and tested successfully!")
