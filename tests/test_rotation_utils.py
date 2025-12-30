"""
Unit tests for rotation utilities and quaternion operations

These are CRITICAL to test thoroughly as quaternion math errors
can cause subtle bugs in trajectory estimation.
"""
import pytest
import torch
import numpy as np
from model.ESKF import ErrorStateKalmanFilter as ESKF


class TestQuaternionOperations:
    """Test quaternion arithmetic and conversions"""

    def test_identity_quaternion(self):
        """Identity quaternion should not rotate vectors"""

        eskf = ESKF()
        # Identity quaternion (w=1, x=y=z=0)
        eskf.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Rotate a vector
        vec = torch.tensor([[1.0, 0.0, 0.0]])
        rotated = eskf.quat_rotate(vec)

        # Should be unchanged
        assert torch.allclose(rotated, vec, atol=1e-6)

    def test_90_degree_rotation_x(self):
        """90° rotation around X-axis"""

        eskf = ESKF()

        # 90° rotation around X-axis: q = [cos(45°), sin(45°), 0, 0]
        angle = np.pi / 2
        eskf.quat = torch.tensor([[
            np.cos(angle/2),
            np.sin(angle/2),
            0.0,
            0.0
        ]])

        # Rotate Y-axis vector [0, 1, 0]
        vec = torch.tensor([[0.0, 1.0, 0.0]])
        rotated = eskf.quat_rotate(vec)

        # Should become [0, 0, 1] after 90° rotation around X
        expected = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_quaternion_normalization(self):
        """Quaternion should remain normalized"""

        eskf = ESKF()

        # Start with non-normalized quaternion
        eskf.quat = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

        # After normalization (should happen internally)
        # We'll test via prediction
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.zeros(1, 3)

        eskf.predict(accel, gyro)

        # Check normalization
        quat_norm = torch.norm(eskf.quat, dim=1)
        assert torch.allclose(quat_norm, torch.ones(1), atol=1e-5)

    def test_quaternion_inverse(self):
        """Quaternion inverse should undo rotation"""

        eskf = ESKF()

        # Random rotation
        eskf.quat = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])  # 90° around X

        vec_original = torch.tensor([[1.0, 1.0, 1.0]])

        # Rotate
        vec_rotated = eskf.quat_rotate(vec_original)

        # Inverse quaternion (conjugate for unit quaternion)
        quat_inv = eskf.quat.clone()
        quat_inv[:, 1:] *= -1  # Negate x, y, z components

        eskf.quat = quat_inv
        vec_back = eskf.quat_rotate(vec_rotated)

        # Should recover original vector
        assert torch.allclose(vec_back, vec_original, atol=1e-5)

    def test_180_degree_rotation(self):
        """180° rotation should flip vector"""

        eskf = ESKF()

        # 180° rotation around Z-axis
        angle = np.pi
        eskf.quat = torch.tensor([[
            np.cos(angle/2),
            0.0,
            0.0,
            np.sin(angle/2)
        ]])

        # Rotate X-axis vector
        vec = torch.tensor([[1.0, 0.0, 0.0]])
        rotated = eskf.quat_rotate(vec)

        # Should become [-1, 0, 0]
        expected = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(rotated, expected, atol=1e-5)


class TestCoordinateTransforms:
    """Test coordinate frame transformations"""

    def test_body_to_world_gravity(self):
        """Gravity in body frame should transform to world frame"""

        eskf = ESKF()

        # Identity orientation (body = world)
        eskf.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Gravity in body frame (pointing down in Z)
        accel_body = torch.tensor([[0.0, 0.0, 9.81]])

        # Transform to world frame
        accel_world = eskf.quat_rotate(accel_body)

        # Should be same in world frame (identity rotation)
        assert torch.allclose(accel_world, accel_body, atol=1e-5)

    def test_body_to_world_tilted(self):
        """Test transformation with tilted sensor"""

        eskf = ESKF()

        # 90° tilt around Y-axis
        angle = np.pi / 2
        eskf.quat = torch.tensor([[
            np.cos(angle/2),
            0.0,
            np.sin(angle/2),
            0.0
        ]])

        # Forward acceleration in body frame
        accel_body = torch.tensor([[1.0, 0.0, 0.0]])

        # Transform to world
        accel_world = eskf.quat_rotate(accel_body)

        # Should point upward in world frame
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(accel_world, expected, atol=1e-5)


class TestGyroscopeIntegration:
    """Test angular velocity integration to quaternion"""

    def test_constant_rotation_rate(self):
        """Constant gyro should produce continuous rotation"""
        from model.config import Config

        eskf = ESKF()

        # Constant rotation around Z-axis: 1 rad/s
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.tensor([[0.0, 0.0, 1.0]])  # 1 rad/s around Z

        dt = Config.DT
        n_steps = int(np.pi / 2 / dt)  # Time for 90° rotation

        for _ in range(n_steps):
            eskf.predict(accel, gyro)

        # After π/2 time at 1 rad/s, should have rotated ~90°
        # Check quaternion represents ~90° rotation
        angle = 2 * np.arccos(eskf.quat[0, 0].item())
        expected_angle = np.pi / 2

        assert abs(angle - expected_angle) < 0.2, f"Expected {expected_angle:.2f}, got {angle:.2f}"

    def test_zero_gyro_no_rotation(self):
        """Zero gyro should not change orientation"""

        eskf = ESKF()

        initial_quat = eskf.quat.clone()

        # No rotation
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.zeros(1, 3)

        for _ in range(100):
            eskf.predict(accel, gyro)

        # Orientation should not change significantly
        quat_diff = torch.norm(eskf.quat - initial_quat)
        assert quat_diff < 0.01, "Orientation changed without gyro input"


class TestRotationMatrixConversion:
    """Test quaternion to rotation matrix conversion"""

    def test_identity_rotation_matrix(self):
        """Identity quaternion should give identity rotation matrix"""
        # Manual calculation
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

        # Build rotation matrix from quaternion
        w, x, y, z = q
        R = torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        # Should be identity matrix
        I = torch.eye(3)
        assert torch.allclose(R, I, atol=1e-6)

    def test_rotation_matrix_properties(self):
        """Rotation matrix should be orthogonal with det=1"""
        # Random quaternion (normalized)
        q = torch.tensor([0.7071, 0.7071, 0.0, 0.0])

        w, x, y, z = q
        R = torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        # R^T * R should be identity (orthogonality)
        I = torch.eye(3)
        assert torch.allclose(R.T @ R, I, atol=1e-5)

        # Determinant should be 1
        det = torch.det(R)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)


class TestLieGroupOperations:
    """Test Lie algebra operations for SO(3)"""

    def test_small_angle_approximation(self):
        """For small angles, rotation ≈ identity + skew-symmetric"""

        eskf = ESKF()

        # Very small rotation
        small_angle = 0.01  # radians
        gyro = torch.tensor([[small_angle, 0.0, 0.0]])

        # One prediction step
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        eskf.predict(accel, gyro)

        # Quaternion should be close to identity
        quat_error = torch.norm(eskf.quat - torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        assert quat_error < 0.1, "Small angle approximation failed"


class TestNumericalStability:
    """Test numerical stability of rotation operations"""

    def test_repeated_normalization(self):
        """Repeated normalization should not accumulate error"""

        eskf = ESKF()

        # Many prediction steps
        accel = torch.randn(1, 3) * 0.1
        accel[0, 2] += 9.81
        gyro = torch.randn(1, 3) * 0.1

        for _ in range(10000):
            eskf.predict(accel, gyro)

            # Check normalization after each step
            quat_norm = torch.norm(eskf.quat)
            assert abs(quat_norm - 1.0) < 1e-4, f"Quaternion denormalized: {quat_norm}"

    def test_gimbal_lock_avoidance(self):
        """Quaternions should avoid gimbal lock"""

        eskf = ESKF()

        # Rotate to potentially problematic orientation
        # (90° pitch in Euler angles causes gimbal lock)
        angle = np.pi / 2
        eskf.quat = torch.tensor([[
            np.cos(angle/2),
            0.0,
            np.sin(angle/2),
            0.0
        ]])

        # Should still be able to rotate around other axes
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.tensor([[0.0, 0.0, 1.0]])  # Yaw rotation

        for _ in range(10):
            eskf.predict(accel, gyro)

        # Quaternion should still be valid and normalized
        assert torch.isfinite(eskf.quat).all()
        quat_norm = torch.norm(eskf.quat)
        assert torch.allclose(quat_norm, torch.ones(1), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
