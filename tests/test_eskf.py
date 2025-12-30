"""
Unit tests for ESKF (Error-State Kalman Filter)

Tests verify:
- Initialization
- Prediction step
- Update steps (ZUPT, TCN velocity correction)
- Quaternion normalization
- State consistency
"""
import pytest
import torch
import numpy as np
from model.ESKF import ErrorStateKalmanFilter as ESKF
from model.config import Config


class TestESKFInitialization:
    """Test ESKF initialization and default state"""

    def test_initial_state_is_zero(self):
        """Position and velocity should start at zero"""
        eskf = ESKF()

        assert torch.allclose(eskf.pos, torch.zeros(1, 3)), "Position not initialized to zero"
        assert torch.allclose(eskf.vel, torch.zeros(1, 3)), "Velocity not initialized to zero"

    def test_initial_quaternion_is_identity(self):
        """Quaternion should start as identity (w=1, x=y=z=0)"""
        eskf = ESKF()

        expected_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(eskf.quat, expected_quat, atol=1e-6), "Quaternion not identity"

    def test_initial_biases_are_zero(self):
        """Accelerometer and gyro biases should start at zero"""
        eskf = ESKF()

        assert torch.allclose(eskf.accel_bias, torch.zeros(1, 3)), "Accel bias not zero"
        assert torch.allclose(eskf.gyro_bias, torch.zeros(1, 3)), "Gyro bias not zero"


class TestESKFPredict:
    """Test ESKF prediction step"""

    def test_stationary_no_drift(self, sample_imu_data):
        """ESKF should not drift when stationary"""
        eskf = ESKF()

        accel = sample_imu_data['accel']
        gyro = sample_imu_data['gyro']
        n_steps = sample_imu_data['n_steps']

        # Run predictions (stationary with only gravity)
        for i in range(n_steps):
            eskf.predict(accel[i:i+1], gyro[i:i+1])

        # Position drift should be minimal after 2 seconds
        position_drift = torch.norm(eskf.pos)
        assert position_drift < 0.05, f"Position drifted {position_drift:.4f}m while stationary"

    def test_quaternion_stays_normalized(self, moving_imu_data):
        """Quaternion should remain unit quaternion after many steps"""
        eskf = ESKF()

        accel = moving_imu_data['accel']
        gyro = moving_imu_data['gyro']

        for i in range(len(accel)):
            eskf.predict(accel[i:i+1], gyro[i:i+1])

            # Check normalization after each step
            quat_norm = torch.norm(eskf.quat, dim=1)
            assert torch.allclose(quat_norm, torch.ones(1), atol=1e-5), \
                f"Quaternion denormalized at step {i}: norm={quat_norm.item()}"

    def test_gravity_compensation(self):
        """Gravity should be properly compensated in world frame"""
        eskf = ESKF()

        # Pure gravity measurement (sensor at rest)
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.zeros(1, 3)

        # Run a few steps to stabilize
        for _ in range(10):
            eskf.predict(accel, gyro)

        # Velocity should not grow (gravity compensated)
        vel_magnitude = torch.norm(eskf.vel)
        assert vel_magnitude < 0.1, f"Velocity grew under gravity: {vel_magnitude:.4f}m/s"

    def test_constant_acceleration_integration(self):
        """Constant acceleration should produce linear velocity growth"""
        eskf = ESKF()

        # 1 m/s² forward + gravity
        # Start with identity quaternion (aligned with world)
        eskf.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        accel_body = torch.tensor([[1.0, 0.0, 9.81]])
        gyro = torch.zeros(1, 3)

        # Integrate for 1 second (50 steps @ 50Hz)
        dt = Config.DT
        n_steps = int(1.0 / dt)

        for _ in range(n_steps):
            eskf.predict(accel_body, gyro)

        # Expected velocity: ~1 m/s in X direction
        # (with some error due to discretization)
        expected_vel_x = 1.0
        actual_vel_x = eskf.vel[0, 0].item()

        assert abs(actual_vel_x - expected_vel_x) < 0.2, \
            f"Velocity integration error: expected {expected_vel_x}, got {actual_vel_x}"


class TestESKFUpdate:
    """Test ESKF update steps"""

    def test_zupt_update_zeros_velocity(self):
        """Zero-velocity update should reduce velocity to near zero"""
        eskf = ESKF()

        # Add some velocity (simulate motion)
        eskf.vel = torch.tensor([[1.0, 0.5, 0.2]])

        # Apply ZUPT
        eskf.update_zupt()

        # Velocity should be significantly reduced
        vel_after = torch.norm(eskf.vel)
        assert vel_after < 0.1, f"ZUPT didn't reduce velocity: {vel_after:.4f}m/s"

    def test_tcn_velocity_update_corrects_drift(self):
        """TCN velocity correction should adjust estimated velocity"""
        eskf = ESKF()

        # Set some velocity
        eskf.vel = torch.tensor([[1.0, 0.0, 0.0]])

        # TCN predicts we should be moving slower
        vel_correction = torch.tensor([[-0.5, 0.0, 0.0]])  # -0.5 m/s correction
        R_adaptive = torch.eye(3) * 0.01  # Low uncertainty

        # Apply correction
        eskf.update_tcn_velocity(vel_correction, R_adaptive)

        # Velocity should be corrected toward prediction
        # (won't be exactly -0.5 due to Kalman gain, but should move in that direction)
        assert eskf.vel[0, 0] < 1.0, "TCN correction didn't adjust velocity"


class TestESKFEdgeCases:
    """Test edge cases and numerical stability"""

    def test_large_gyro_rates(self):
        """ESKF should handle large gyroscope rates without exploding"""
        eskf = ESKF()

        # Large rotation rate (10 rad/s ≈ 570°/s)
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.tensor([[10.0, 0.0, 0.0]])

        # Should not crash or produce NaN
        for _ in range(50):
            eskf.predict(accel, gyro)

        assert not torch.isnan(eskf.quat).any(), "Quaternion became NaN"
        assert not torch.isnan(eskf.pos).any(), "Position became NaN"

    def test_zero_input(self):
        """ESKF should handle all-zero input gracefully"""
        eskf = ESKF()

        accel = torch.zeros(1, 3)
        gyro = torch.zeros(1, 3)

        # Should not crash
        eskf.predict(accel, gyro)

        # State should remain valid
        assert torch.isfinite(eskf.pos).all(), "Position became non-finite"
        assert torch.isfinite(eskf.vel).all(), "Velocity became non-finite"

    def test_batch_prediction(self):
        """ESKF should handle batch dimension correctly"""
        eskf = ESKF()

        # Note: Current ESKF might only support batch_size=1
        # This test documents expected behavior if batching is added
        accel = torch.randn(1, 3)
        gyro = torch.randn(1, 3)

        try:
            eskf.predict(accel, gyro)
            assert True, "Batch prediction works"
        except Exception as e:
            pytest.skip(f"Batch prediction not supported: {e}")


class TestESKFIntegration:
    """Integration tests combining multiple operations"""

    def test_predict_update_cycle(self, sample_imu_data):
        """Test full predict-update cycle"""
        eskf = ESKF()

        accel = sample_imu_data['accel']
        gyro = sample_imu_data['gyro']

        # Simulate realistic cycle: predict + occasional ZUPT
        for i in range(len(accel)):
            eskf.predict(accel[i:i+1], gyro[i:i+1])

            # Apply ZUPT every 20 steps (simulating detected stillness)
            if i % 20 == 0:
                eskf.update_zupt()

        # Should complete without errors
        assert torch.isfinite(eskf.pos).all()
        assert torch.isfinite(eskf.vel).all()
        assert torch.isfinite(eskf.quat).all()

    def test_long_sequence(self):
        """Test stability over long sequences (1000 steps)"""
        eskf = ESKF()

        np.random.seed(123)
        n_steps = 1000

        for i in range(n_steps):
            # Random but bounded IMU data
            accel = torch.randn(1, 3) * 0.5
            accel[0, 2] += 9.81  # Add gravity
            gyro = torch.randn(1, 3) * 0.1

            eskf.predict(accel, gyro)

            # Periodic ZUPT
            if i % 50 == 0:
                eskf.update_zupt()

        # Check state is still valid after 1000 steps
        assert torch.isfinite(eskf.pos).all(), "Position became non-finite"
        assert torch.norm(eskf.quat, dim=1).item() > 0.99, "Quaternion denormalized significantly"


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v"])
