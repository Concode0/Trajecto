/**
 * Unit tests for ESKF (firmware implementation)
 *
 * These tests verify the C++ ESKF implementation matches expected behavior.
 * Run with: idf.py build flash monitor
 */

#include <cmath>
#include <cstring>
#include "unity.h"
#include "eskf.hpp"

// Test setup/teardown
void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

// ============================================================================
// Initialization Tests
// ============================================================================

void test_eskf_initialization(void) {
    trajecto::ESKF eskf;
    const auto& state = eskf.get_state();

    // Position should be zero
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.z());

    // Velocity should be zero
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.z());

    // Quaternion should be identity (w=1, x=y=z=0)
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, state.quat.w());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.z());
}

void test_initial_biases_are_zero(void) {
    trajecto::ESKF eskf;
    const auto& state = eskf.get_state();

    // Gyro bias should be zero
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.z());

    // Accel bias should be zero
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.z());
}

// ============================================================================
// Prediction Tests
// ============================================================================

void test_quaternion_normalization(void) {
    trajecto::ESKF eskf;

    // Simulate many prediction steps with random motion
    Eigen::Vector3f accel(0.1f, 0.1f, 9.81f);
    Eigen::Vector3f gyro(0.05f, 0.05f, 0.05f);

    for (int i = 0; i < 1000; i++) {
        eskf.predict(accel, gyro);
    }

    // Check quaternion is still normalized
    const auto& quat = eskf.get_state().quat;
    float norm = std::sqrt(quat.w()*quat.w() + quat.x()*quat.x() +
                          quat.y()*quat.y() + quat.z()*quat.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, norm);
}

void test_stationary_no_drift(void) {
    trajecto::ESKF eskf;

    // Pure gravity, no rotation
    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    // Run for 5 seconds (250 steps @ 50Hz)
    for (int i = 0; i < 250; i++) {
        eskf.predict(accel, gyro);
    }

    // Position drift should be minimal
    const auto& pos = eskf.get_state().pos;
    float drift = std::sqrt(pos.x()*pos.x() + pos.y()*pos.y() + pos.z()*pos.z());

    TEST_ASSERT_LESS_THAN_FLOAT(0.05f, drift);  // Less than 5cm drift
}

void test_predict_with_small_noise(void) {
    trajecto::ESKF eskf;

    // Gravity with small noise
    for (int i = 0; i < 100; i++) {
        Eigen::Vector3f accel(0.01f, -0.01f, 9.81f);  // Small noise
        Eigen::Vector3f gyro(0.001f, 0.001f, 0.001f);

        eskf.predict(accel, gyro);
    }

    // Should complete without NaN or extreme values
    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.quat.w()));
}

// ============================================================================
// Update Tests
// ============================================================================

void test_zupt_reduces_velocity(void) {
    trajecto::ESKF eskf;

    // Add some velocity first
    // Note: In real implementation, you might need a setter method
    // For now, we'll test via prediction then ZUPT

    // Apply acceleration to build up velocity
    Eigen::Vector3f accel(2.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 10; i++) {
        eskf.predict(accel, gyro);
    }

    // Check we have some velocity
    float vel_before = eskf.get_state().vel.norm();
    TEST_ASSERT_GREATER_THAN_FLOAT(0.1f, vel_before);

    // Apply ZUPT
    eskf.update_zupt();

    // Velocity should be reduced
    float vel_after = eskf.get_state().vel.norm();
    TEST_ASSERT_LESS_THAN_FLOAT(vel_before, vel_after);
}

void test_tcn_velocity_update(void) {
    trajecto::ESKF eskf;

    // Set some velocity via prediction
    Eigen::Vector3f accel(1.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    float vel_before = eskf.get_state().vel.x();

    // TCN correction suggesting slower velocity
    Eigen::Vector3f vel_correction(-0.5f, 0.0f, 0.0f);
    Eigen::Matrix3f R_adaptive = Eigen::Matrix3f::Identity() * 0.01f;

    eskf.update_tcn_velocity(vel_correction, R_adaptive);

    // Velocity should be adjusted
    float vel_after = eskf.get_state().vel.x();
    TEST_ASSERT_NOT_EQUAL_FLOAT(vel_before, vel_after);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_large_gyro_rates(void) {
    trajecto::ESKF eskf;

    // Very large rotation rate (10 rad/s)
    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(10.0f, 0.0f, 0.0f);

    for (int i = 0; i < 50; i++) {
        eskf.predict(accel, gyro);
    }

    // Should not produce NaN or infinite values
    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.quat.w()));
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
}

void test_zero_input(void) {
    trajecto::ESKF eskf;

    // All zeros (edge case)
    Eigen::Vector3f accel(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    eskf.predict(accel, gyro);

    // Should not crash and state should be valid
    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.x()));
}

// ============================================================================
// Integration Test
// ============================================================================

void test_predict_update_cycle(void) {
    trajecto::ESKF eskf;

    // Simulate realistic predict-update cycle
    for (int i = 0; i < 200; i++) {
        // Prediction
        Eigen::Vector3f accel(
            0.01f * std::sin(i * 0.1f),
            0.01f * std::cos(i * 0.1f),
            9.81f
        );
        Eigen::Vector3f gyro(0.005f, 0.005f, 0.005f);

        eskf.predict(accel, gyro);

        // Apply ZUPT every 20 steps
        if (i % 20 == 0) {
            eskf.update_zupt();
        }

        // Apply TCN correction every 10 steps
        if (i % 10 == 0) {
            Eigen::Vector3f vel_corr(0.01f, 0.01f, 0.0f);
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity() * 0.01f;
            eskf.update_tcn_velocity(vel_corr, R);
        }
    }

    // After 200 steps, state should still be valid
    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.norm()));

    // Quaternion should still be normalized
    float quat_norm = state.quat.norm();
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, quat_norm);
}

// ============================================================================
// Test Runner
// ============================================================================

extern "C" void app_main(void) {
    // Wait a moment for serial connection
    vTaskDelay(pdMS_TO_TICKS(2000));

    printf("\n");
    printf("========================================\n");
    printf("  ESKF Unit Tests (Firmware)\n");
    printf("========================================\n");
    printf("\n");

    UNITY_BEGIN();

    // Initialization tests
    RUN_TEST(test_eskf_initialization);
    RUN_TEST(test_initial_biases_are_zero);

    // Prediction tests
    RUN_TEST(test_quaternion_normalization);
    RUN_TEST(test_stationary_no_drift);
    RUN_TEST(test_predict_with_small_noise);

    // Update tests
    RUN_TEST(test_zupt_reduces_velocity);
    RUN_TEST(test_tcn_velocity_update);

    // Edge cases
    RUN_TEST(test_large_gyro_rates);
    RUN_TEST(test_zero_input);

    // Integration
    RUN_TEST(test_predict_update_cycle);

    UNITY_END();

    printf("\n");
    printf("========================================\n");
    printf("  Tests Complete!\n");
    printf("========================================\n");
}
