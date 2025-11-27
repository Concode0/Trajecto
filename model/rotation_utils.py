import torch
import torch.nn.functional as F
from typing import Tuple

def quaternion_multiply(quat_1: torch.Tensor, quat_2: torch.Tensor) -> torch.Tensor:
    """Performs batch-aware multiplication of two quaternions, q_new = q1 * q2.
    This corresponds to applying rotation q2 followed by rotation q1.

    Args:
        quat_1 (torch.Tensor): A tensor of shape `[B, 4]` representing the first quaternion (w, x, y, z).
        quat_2 (torch.Tensor): A tensor of shape `[B, 4]` representing the second quaternion (w, x, y, z).

    Returns:
        torch.Tensor: The resulting quaternion of shape `[B, 4]`.
    """
    # Hamilton product for quaternion multiplication
    w1, x1, y1, z1 = quat_1.unbind(-1)
    w2, x2, y2, z2 = quat_2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)

def quaternion_to_rotation_matrix(quat_b_to_w: torch.Tensor) -> torch.Tensor:
    """Converts a batch of quaternions to a batch of rotation matrices.
    The resulting matrix rotates a vector from the body frame to the world frame.

    Args:
        quat_b_to_w (torch.Tensor): A tensor of shape `[B, 4]` representing the body-to-world quaternion (w, x, y, z).

    Returns:
        torch.Tensor: The corresponding rotation matrix of shape `[B, 3, 3]`.
    """
    # Ensure the quaternion is normalized to prevent scaling errors
    quat_norm = F.normalize(quat_b_to_w, p=2, dim=-1)
    w, x, y, z = quat_norm.unbind(-1)

    # Conversion formula from quaternion to rotation matrix
    # See: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    rot_mat_b_to_w = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)
    ], -2)
    return rot_mat_b_to_w

def small_angle_to_quaternion(small_angle_vec: torch.Tensor) -> torch.Tensor:
    """Converts a small 3D angle vector (e.g., from gyroscope error) to a quaternion.
    This is used to update the orientation from a small angular correction.

    Args:
        small_angle_vec (torch.Tensor): A tensor of shape `[B, 3]` representing the small rotation vector.

    Returns:
        torch.Tensor: The corresponding correction quaternion of shape `[B, 4]`.
    """
    angle_sq_norm = torch.sum(small_angle_vec * small_angle_vec, dim=-1, keepdim=True)
    angle_norm = torch.sqrt(angle_sq_norm)
    half_angle = angle_norm / 2.0
    
    # The real part of the quaternion is cos(theta/2)
    w = torch.cos(half_angle)
    
    # The vector part is sin(theta/2) * rotation_axis
    sin_half_angle = torch.sin(half_angle)
    
    # Normalize the angle vector to get the rotation axis
    # Use a small epsilon to avoid division by zero for near-zero rotations
    axis = F.normalize(small_angle_vec, p=2, dim=-1, eps=1e-8)
    xyz = axis * sin_half_angle
    
    # The final quaternion must be normalized to be a valid rotation
    return F.normalize(torch.cat([w, xyz], dim=-1), p=2, dim=-1)

if __name__ == '__main__':
    # Simple test case to verify functionality and shapes.
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test quaternion_multiply
    q1 = torch.tensor([[0.7071, 0, 0, 0.7071]], device=device)  # 90 deg around z
    q2 = torch.tensor([[0.7071, 0.7071, 0, 0]], device=device)  # 90 deg around x
    q_mult = quaternion_multiply(q1, q2)
    print(f"Quaternion multiplication result shape: {q_mult.shape}")

    # Test quaternion_to_rotation_matrix
    q_z_90 = torch.tensor([[0.7071, 0, 0, 0.7071]], device=device) # 90 deg around z
    rot_mat = quaternion_to_rotation_matrix(q_z_90)
    print(f"Quaternion to rotation matrix result shape: {rot_mat.shape}")

    # Test small_angle_to_quaternion
    small_angle = torch.tensor([[0.01, 0.02, 0.03]], device=device)
    q_small = small_angle_to_quaternion(small_angle)
    print(f"Small angle to quaternion result shape: {q_small.shape}")
    
    print("Rotation utils tested successfully.")