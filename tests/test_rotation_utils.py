# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
Tests for rotation utility functions.

Covers:
- quaternion_from_two_vectors (parallel, anti-parallel, general)
- quaternion_multiply
- quaternion_to_rotation_matrix
- small_angle_to_quaternion

Run with: pytest tests/test_rotation_utils.py -v
"""

import torch
import pytest
import math

# Add parent directory to path for imports
import sys
sys.path.insert(0, "/Users/haro/works/Trajecto")

from model.rotation_utils import (
    quaternion_from_two_vectors,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    small_angle_to_quaternion,
)


class TestQuaternionFromTwoVectors:
    """Tests for quaternion_from_two_vectors function."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_z_to_y_rotation(self, device):
        """Test rotation from +Z to +Y (90 degrees around X-axis)."""
        v1 = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        v2 = torch.tensor([[0.0, 1.0, 0.0]], device=device)

        q = quaternion_from_two_vectors(v1, v2)

        # Quaternion should be unit quaternion
        norm = torch.norm(q, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
            f"Quaternion norm should be 1, got {norm}"

        # Verify rotation: apply quaternion to v1 should give v2
        R = quaternion_to_rotation_matrix(q)
        v1_rotated = torch.einsum('bij,bj->bi', R, v1)
        assert torch.allclose(v1_rotated, v2, atol=1e-4), \
            f"Rotated v1 should equal v2, got {v1_rotated}"

    def test_parallel_vectors(self, device):
        """Test with parallel vectors (no rotation needed)."""
        v1 = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        v2 = torch.tensor([[0.0, 0.0, 2.0]], device=device)  # Same direction, different magnitude

        q = quaternion_from_two_vectors(v1, v2)

        # Should be identity quaternion [1, 0, 0, 0]
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        assert torch.allclose(q, identity, atol=1e-5), \
            f"Parallel vectors should give identity quaternion, got {q}"

    def test_anti_parallel_vectors(self, device):
        """Test with anti-parallel vectors (180-degree rotation)."""
        v1 = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        v2 = torch.tensor([[0.0, 0.0, -1.0]], device=device)

        q = quaternion_from_two_vectors(v1, v2)

        # Quaternion should be unit quaternion
        norm = torch.norm(q, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
            f"Quaternion norm should be 1, got {norm}"

        # w should be ~0 for 180-degree rotation
        assert abs(q[0, 0].item()) < 0.1, \
            f"180-degree rotation should have w~0, got {q[0, 0]}"

        # Verify rotation: apply quaternion to v1 should give v2
        R = quaternion_to_rotation_matrix(q)
        v1_rotated = torch.einsum('bij,bj->bi', R, v1)
        assert torch.allclose(v1_rotated, v2 / torch.norm(v2), atol=1e-4), \
            f"Rotated v1 should equal normalized v2, got {v1_rotated}"

    def test_batch_vectors(self, device):
        """Test with batched vectors."""
        batch_size = 4
        v1 = torch.randn(batch_size, 3, device=device)
        v2 = torch.randn(batch_size, 3, device=device)

        q = quaternion_from_two_vectors(v1, v2)

        assert q.shape == (batch_size, 4), f"Shape mismatch: {q.shape}"

        # All quaternions should be unit quaternions
        norms = torch.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size, device=device), atol=1e-5)

    def test_x_to_y_rotation(self, device):
        """Test rotation from +X to +Y (90 degrees around Z-axis)."""
        v1 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        v2 = torch.tensor([[0.0, 1.0, 0.0]], device=device)

        q = quaternion_from_two_vectors(v1, v2)
        R = quaternion_to_rotation_matrix(q)
        v1_rotated = torch.einsum('bij,bj->bi', R, v1)

        assert torch.allclose(v1_rotated, v2, atol=1e-4)


class TestQuaternionMultiply:
    """Tests for quaternion_multiply function."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_identity_multiplication(self, device):
        """Test multiplication with identity quaternion."""
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        q_x_90 = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=device)

        result = quaternion_multiply(q_identity, q_x_90)
        assert torch.allclose(result, q_x_90, atol=1e-5), \
            f"Identity * q should equal q, got {result}"

        result2 = quaternion_multiply(q_x_90, q_identity)
        assert torch.allclose(result2, q_x_90, atol=1e-5), \
            f"q * Identity should equal q, got {result2}"

    def test_x_then_y_rotation(self, device):
        """Test sequential X then Y rotation."""
        # 90 degrees around X
        q_x_90 = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=device)
        # 90 degrees around Y
        q_y_90 = torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]], device=device)

        q_combined = quaternion_multiply(q_x_90, q_y_90)

        # Result should be unit quaternion
        norm = torch.norm(q_combined, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_inverse_multiplication(self, device):
        """Test q * q^-1 = identity."""
        q = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=device)
        q_conj = torch.tensor([[0.7071068, -0.7071068, 0.0, 0.0]], device=device)  # Conjugate

        result = quaternion_multiply(q, q_conj)

        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        assert torch.allclose(result, identity, atol=1e-5)

    def test_batch_multiplication(self, device):
        """Test batched quaternion multiplication."""
        batch_size = 8
        q1 = torch.randn(batch_size, 4, device=device)
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)  # Normalize
        q2 = torch.randn(batch_size, 4, device=device)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

        result = quaternion_multiply(q1, q2)

        assert result.shape == (batch_size, 4)
        # Result should be unit quaternions
        norms = torch.norm(result, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size, device=device), atol=1e-5)


class TestQuaternionToRotationMatrix:
    """Tests for quaternion_to_rotation_matrix function."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_identity_quaternion(self, device):
        """Test identity quaternion gives identity matrix."""
        q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        R = quaternion_to_rotation_matrix(q_identity)

        I = torch.eye(3, device=device).unsqueeze(0)
        assert torch.allclose(R, I, atol=1e-5), \
            f"Identity quaternion should give identity matrix, got {R}"

    def test_90_deg_z_rotation(self, device):
        """Test 90-degree rotation around Z-axis."""
        # Quaternion for 90 deg around Z: [cos(45), 0, 0, sin(45)]
        q_z_90 = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]], device=device)
        R = quaternion_to_rotation_matrix(q_z_90)

        # Expected rotation matrix for 90 deg around Z
        expected = torch.tensor([[[0.0, -1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]]], device=device)

        assert torch.allclose(R, expected, atol=1e-4), \
            f"90-deg Z rotation mismatch:\n{R}\nvs expected:\n{expected}"

    def test_rotation_matrix_orthogonality(self, device):
        """Test that rotation matrix is orthogonal (R^T R = I)."""
        q = torch.randn(1, 4, device=device)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        R = quaternion_to_rotation_matrix(q)
        RtR = torch.einsum('bij,bik->bjk', R, R)

        I = torch.eye(3, device=device).unsqueeze(0)
        assert torch.allclose(RtR, I, atol=1e-5), \
            f"R^T R should be identity, got {RtR}"

    def test_rotation_matrix_determinant(self, device):
        """Test that rotation matrix has determinant 1."""
        q = torch.randn(1, 4, device=device)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        R = quaternion_to_rotation_matrix(q)
        det = torch.linalg.det(R)

        assert torch.allclose(det, torch.ones(1, device=device), atol=1e-5), \
            f"Determinant should be 1, got {det}"

    def test_batch_rotation_matrix(self, device):
        """Test batched rotation matrix computation."""
        batch_size = 16
        q = torch.randn(batch_size, 4, device=device)
        q = q / torch.norm(q, dim=-1, keepdim=True)

        R = quaternion_to_rotation_matrix(q)

        assert R.shape == (batch_size, 3, 3)


class TestSmallAngleToQuaternion:
    """Tests for small_angle_to_quaternion function."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_zero_angle(self, device):
        """Test zero rotation vector gives identity quaternion."""
        small_angle = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        q = small_angle_to_quaternion(small_angle)

        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        assert torch.allclose(q, identity, atol=1e-5), \
            f"Zero angle should give identity, got {q}"

    def test_small_angle_unit_quaternion(self, device):
        """Test that output is always a unit quaternion."""
        small_angle = torch.tensor([[0.01, 0.02, 0.03]], device=device)
        q = small_angle_to_quaternion(small_angle)

        norm = torch.norm(q, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
            f"Quaternion should be unit, got norm {norm}"

    def test_small_angle_w_near_one(self, device):
        """Test that small angles have w component near 1."""
        small_angle = torch.tensor([[0.01, 0.02, 0.03]], device=device)
        q = small_angle_to_quaternion(small_angle)

        # For small angles, w should be close to 1
        assert q[0, 0].item() > 0.99, f"w should be close to 1, got {q[0, 0]}"

    def test_small_angle_xyz_proportional(self, device):
        """Test that xyz components are proportional to input."""
        small_angle = torch.tensor([[0.01, 0.02, 0.03]], device=device)
        q = small_angle_to_quaternion(small_angle)

        # For small angles, xyz should be approximately proportional to input/2
        xyz = q[0, 1:4]
        expected_ratio = small_angle[0] / 2

        # Ratio should be close to 1 for small angles
        ratios = xyz / expected_ratio
        assert torch.allclose(ratios, torch.ones(3, device=device), atol=0.1), \
            f"xyz/angle ratio unexpected: {ratios}"

    def test_batch_small_angles(self, device):
        """Test batched small angle computation."""
        batch_size = 8
        small_angles = torch.randn(batch_size, 3, device=device) * 0.05

        q = small_angle_to_quaternion(small_angles)

        assert q.shape == (batch_size, 4)

        # All should be unit quaternions
        norms = torch.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size, device=device), atol=1e-5)

    def test_larger_angle_still_valid(self, device):
        """Test that larger angles still produce valid quaternions."""
        # Not really "small" but should still work
        angle = torch.tensor([[0.5, 0.3, 0.4]], device=device)
        q = small_angle_to_quaternion(angle)

        norm = torch.norm(q, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
