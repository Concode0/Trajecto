"""
Unit tests for ESKF components to isolate the scale error.

Run with: python -m pytest tests/test_eskf_components.py -v
Or directly: python tests/test_eskf_components.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from model.ESKF import ErrorStateKalmanFilter
from model.config import Config
from model.rotation_utils import quaternion_from_two_vectors, quaternion_to_rotation_matrix


class TestESKFBasicIntegration:
    """Test basic ESKF integration without any updates."""

    def setup_method(self):
        """Initialize ESKF before each test."""
        self.dt = 0.02  # 50 Hz
        self.eskf = ErrorStateKalmanFilter(
            dt=self.dt,
            device="cpu",
            use_zupt=False,
            use_tcn_zupt=False,
            use_virtual_measurements=False
        )

    def test_static_gravity_aligned(self):
        """Test 1: Static pen with gravity perfectly aligned to Z-axis.

        Expected: Position and velocity should remain at zero.
        """
        print("\n=== Test 1: Static with perfect gravity alignment ===")

        # Initialize state
        pos = torch.zeros(1, 3)
        vel = torch.zeros(1, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity (no rotation)
        gyro_bias = torch.zeros(1, 3)
        accel_bias = torch.zeros(1, 3)
        P = torch.eye(15).unsqueeze(0) * 1e-4

        # Static measurements: gravity in Z
        accel_raw = torch.tensor([[0.0, 0.0, 9.81]])
        gyro_raw = torch.zeros(1, 3)
        force_raw = torch.zeros(1, 1)
        measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

        # Run 100 steps
        for i in range(100):
            pos, vel, quat, gyro_bias, accel_bias, P, _ = self.eskf.forward(
                pos, vel, quat, gyro_bias, accel_bias, P,
                gyro_raw, accel_raw, force_raw, measurement
            )

        print(f"Final position: {pos[0].detach().numpy()}")
        print(f"Final velocity: {vel[0].detach().numpy()}")

        # Assert position and velocity are near zero
        assert torch.allclose(pos, torch.zeros_like(pos), atol=1e-6), \
            f"Position should be zero, got {pos[0].detach().numpy()}"
        assert torch.allclose(vel, torch.zeros_like(vel), atol=1e-6), \
            f"Velocity should be zero, got {vel[0].detach().numpy()}"

        print("✓ PASSED: Static gravity-aligned test")

    def test_constant_acceleration(self):
        """Test 2: Constant acceleration in X direction.

        Expected: Position should follow s = 0.5 * a * t^2
        """
        print("\n=== Test 2: Constant acceleration ===")

        # Initialize
        pos = torch.zeros(1, 3)
        vel = torch.zeros(1, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        gyro_bias = torch.zeros(1, 3)
        accel_bias = torch.zeros(1, 3)
        P = torch.eye(15).unsqueeze(0) * 1e-4

        # Constant acceleration: 1 m/s² in X, gravity in Z
        accel_raw = torch.tensor([[1.0, 0.0, 9.81]])
        gyro_raw = torch.zeros(1, 3)
        force_raw = torch.zeros(1, 1)
        measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

        # Run for 2 seconds (100 steps @ 50Hz)
        num_steps = 100
        for i in range(num_steps):
            pos, vel, quat, gyro_bias, accel_bias, P, _ = self.eskf.forward(
                pos, vel, quat, gyro_bias, accel_bias, P,
                gyro_raw, accel_raw, force_raw, measurement
            )

        t = num_steps * self.dt  # Total time
        expected_x = 0.5 * 1.0 * t**2  # s = 0.5 * a * t^2
        expected_vx = 1.0 * t  # v = a * t

        print(f"Final position: {pos[0].detach().numpy()}")
        print(f"Expected position: [{expected_x:.4f}, 0, 0]")
        print(f"Final velocity: {vel[0].detach().numpy()}")
        print(f"Expected velocity: [{expected_vx:.4f}, 0, 0]")

        # Check position (allow 5% error)
        assert abs(pos[0, 0].item() - expected_x) / expected_x < 0.05, \
            f"X position error too large: got {pos[0, 0].item():.4f}, expected {expected_x:.4f}"
        assert abs(pos[0, 1].item()) < 1e-3, \
            f"Y position should be zero, got {pos[0, 1].item()}"
        assert abs(pos[0, 2].item()) < 1e-3, \
            f"Z position should be zero, got {pos[0, 2].item()}"

        # Check velocity
        assert abs(vel[0, 0].item() - expected_vx) / expected_vx < 0.05, \
            f"X velocity error too large: got {vel[0, 0].item():.4f}, expected {expected_vx:.4f}"

        print("✓ PASSED: Constant acceleration test")

    def test_gravity_removal_tilted(self):
        """Test 3: Static pen tilted 45 degrees.

        Expected: After gravity removal, acceleration should be zero.
        """
        print("\n=== Test 3: Gravity removal with tilt ===")

        # Simulate pen tilted 45° in X-Z plane
        # Gravity vector in body frame: [sin(45°)*g, 0, cos(45°)*g]
        angle = np.pi / 4  # 45 degrees
        g = 9.81
        accel_body = torch.tensor([[g * np.sin(angle), 0.0, g * np.cos(angle)]], dtype=torch.float32)

        # Create quaternion that aligns this gravity vector to world Z
        world_gravity = torch.tensor([[0.0, 0.0, g]], dtype=torch.float32)
        quat = quaternion_from_two_vectors(accel_body, world_gravity)

        print(f"Accel in body frame: {accel_body[0].numpy()}")
        print(f"Quaternion: {quat[0].numpy()}")

        # Rotate to world frame
        rot_mat = quaternion_to_rotation_matrix(quat)
        accel_world = (rot_mat @ accel_body.unsqueeze(-1)).squeeze(-1)

        print(f"Accel in world frame: {accel_world[0].numpy()}")
        print(f"After gravity removal: {(accel_world - world_gravity)[0].numpy()}")

        # After gravity removal, should be nearly zero
        accel_corrected = accel_world - world_gravity
        assert torch.allclose(accel_corrected, torch.zeros_like(accel_corrected), atol=1e-5), \
            f"Gravity removal failed: {accel_corrected[0].numpy()}"

        print("✓ PASSED: Gravity removal test")


class TestESKFWithRealData:
    """Test ESKF with actual sensor data."""

    def test_first_100_samples_scale(self):
        """Test 4: Run ESKF on first 100 samples and check scale.

        This test uses real data to verify that scale doesn't explode early.
        """
        import h5py

        print("\n=== Test 4: Real data first 100 samples ===")

        # Load real data
        with h5py.File("data/dataset.h5", "r") as f:
            sample_key = list(f.keys())[0]
            g = f[sample_key]
            sensor_data = g["sensor_data"][:100]
            gt_pos = g["gt_pos_data"][:100]

        # Initialize ESKF
        eskf = ErrorStateKalmanFilter(
            dt=Config.DT,
            device="cpu",
            use_zupt=False,  # Disable all corrections for this test
            use_tcn_zupt=False,
            use_virtual_measurements=False
        )

        # Initialize state like pure_eskf
        static_samples = 50
        avg_accel_b = torch.from_numpy(sensor_data[:static_samples, :3].mean(axis=0)).float().unsqueeze(0)
        gyro_bias_b = torch.from_numpy(sensor_data[:static_samples, 3:6].mean(axis=0)).float().unsqueeze(0)
        world_gravity = torch.tensor([[0.0, 0.0, 9.81]])
        quat = quaternion_from_two_vectors(avg_accel_b, world_gravity)

        pos = torch.zeros(1, 3)
        vel = torch.zeros(1, 3)
        accel_bias = torch.zeros(1, 3)
        P = torch.eye(15).unsqueeze(0) * 1e-4

        print(f"Initial gravity in body frame: {avg_accel_b[0].numpy()}")
        print(f"Initial quaternion: {quat[0].numpy()}")

        # Run integration with detailed logging
        positions = []
        velocities = []
        accelerations_world = []

        for t in range(100):
            accel_raw = torch.from_numpy(sensor_data[t, :3]).float().unsqueeze(0)
            gyro_raw = torch.from_numpy(sensor_data[t, 3:6]).float().unsqueeze(0)
            force_raw = torch.from_numpy(sensor_data[t, 6:7]).float().unsqueeze(0)
            measurement = torch.cat([accel_raw, gyro_raw], dim=-1)

            # Calculate expected acceleration in world frame
            rot_mat = quaternion_to_rotation_matrix(quat)
            accel_world_expected = (rot_mat @ accel_raw.unsqueeze(-1)).squeeze(-1) - torch.tensor([[0.0, 0.0, 9.81]])

            pos, vel, quat, gyro_bias, accel_bias, P, _ = eskf.forward(
                pos, vel, quat, gyro_bias_b, accel_bias, P,
                gyro_raw, accel_raw, force_raw, measurement
            )

            positions.append(pos[0].detach().numpy().copy())
            velocities.append(vel[0].detach().numpy().copy())
            accelerations_world.append(accel_world_expected[0].detach().numpy().copy())

            # Debug first few steps
            if t < 5:
                print(f"  Step {t}: accel_world={accel_world_expected[0].detach().numpy()}, "
                      f"vel={vel[0].detach().numpy()}, pos={pos[0].detach().numpy()}")

        positions = np.array(positions)
        velocities = np.array(velocities)

        # Check scale at different points
        checkpoints = [20, 50, 99]
        print("\nScale check:")
        for cp in checkpoints:
            gt_dist = np.linalg.norm(gt_pos[cp] - gt_pos[0])
            pred_dist = np.linalg.norm(positions[cp] - positions[0])
            scale = pred_dist / (gt_dist + 1e-9)

            print(f"  Step {cp}: GT={gt_dist:.4f}m, Pred={pred_dist:.4f}m, Scale={scale:.2f}x")

            # Assert scale is not crazy (allow up to 20x for this diagnostic test)
            if scale > 20.0:
                print(f"  WARNING: Scale too large at step {cp}: {scale:.2f}x")

        # Final position check
        final_gt = gt_pos[99]
        final_pred = positions[99]
        final_scale = np.linalg.norm(final_pred) / np.linalg.norm(final_gt)

        print(f"\nFinal position:")
        print(f"  GT:   {final_gt}")
        print(f"  Pred: {final_pred}")
        print(f"  Scale: {final_scale:.2f}x")

        if final_scale < 5.0:
            print("✓ PASSED: Real data test (scale < 5x)")
        else:
            print(f"❌ FAILED: Scale too large: {final_scale:.2f}x")
            print("\n🔍 Scale grows rapidly from start - issue is in early integration!")


class TestGravityCalibration:
    """Test gravity magnitude calibration."""

    def test_gravity_magnitude_mismatch(self):
        """Test 5: Check if gravity mismatch causes scale error."""
        import h5py

        print("\n=== Test 5: Gravity calibration ===")

        with h5py.File("data/dataset.h5", "r") as f:
            all_mags = []
            for key in list(f.keys())[:10]:
                g = f[key]
                sensor = g["sensor_data"][:50, :3]
                mag = np.linalg.norm(sensor.mean(axis=0))
                all_mags.append(mag)

        measured_gravity = np.mean(all_mags)
        config_gravity = Config.GRAVITY_MAGNITUDE

        print(f"Measured gravity: {measured_gravity:.6f} m/s²")
        print(f"Config gravity: {config_gravity:.6f} m/s²")
        print(f"Mismatch: {measured_gravity - config_gravity:.6f} m/s²")
        print(f"Ratio: {measured_gravity / config_gravity:.6f}")

        # Calculate theoretical position error from gravity mismatch
        dt = Config.DT
        t = 16.0  # 16 seconds
        gravity_error = measured_gravity - config_gravity

        # Position error from constant acceleration error: s = 0.5 * a * t^2
        position_error = 0.5 * gravity_error * t**2

        print(f"\nTheoretical position error after {t}s:")
        print(f"  From gravity mismatch: {position_error:.4f} m")
        print(f"  If GT trajectory is ~0.1m, scale error would be: {position_error / 0.1:.2f}x")

        # Assert gravity mismatch is small
        assert abs(measured_gravity - config_gravity) < 0.1, \
            f"Gravity mismatch too large: {measured_gravity - config_gravity:.6f} m/s²"

        print("✓ PASSED: Gravity calibration check")


if __name__ == "__main__":
    # Run tests manually
    print("="*60)
    print("ESKF Component Unit Tests")
    print("="*60)

    # Test 1: Basic integration
    test1 = TestESKFBasicIntegration()
    test1.setup_method()
    test1.test_static_gravity_aligned()

    test1.setup_method()
    test1.test_constant_acceleration()

    test1.test_gravity_removal_tilted()

    # Test 2: Real data
    test2 = TestESKFWithRealData()
    test2.test_first_100_samples_scale()

    # Test 3: Gravity calibration
    test3 = TestGravityCalibration()
    test3.test_gravity_magnitude_mismatch()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
