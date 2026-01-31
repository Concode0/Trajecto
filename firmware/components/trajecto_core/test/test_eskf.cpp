/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System
 * Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * NOTICE: This software is protected under the following ROK Patent Applications:
 * 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
 * 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
 *
 * Commercial use or redistribution of the core logic requires a separate license.
 * For inquiries, contact: nemonanconcode@gmail.com
 */

#include <cmath>
#include <cstring>
#include "unity.h"
#include "eskf.hpp"
#include "model_params.hpp"

using namespace trajecto;

// Helper: compute expected R value matching Python's log-space soft-thresholding
static float compute_expected_R(float prob) {
    float cp = std::max(1e-4f, std::min(prob, 1.0f));
    float above_onset = std::max(0.0f, std::min(
        (cp - ZUPT_DECAY_ONSET) / (1.0f - ZUPT_DECAY_ONSET + 1e-6f), 1.0f));
    float alpha = above_onset * above_onset;
    float log_R = (1.0f - alpha) * LOG_R_MAX + alpha * LOG_R_MIN;
    return std::exp(log_R);
}

// Test setup/teardown
void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// Initialization Tests
// ============================================================================

void test_eskf_initialization(void) {
    ESKF eskf(DT);
    const auto& state = eskf.get_state();

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.pos.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.vel.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, state.quat.w());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.quat.z());
}

void test_initial_biases_are_zero(void) {
    ESKF eskf(DT);
    const auto& state = eskf.get_state();

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.gyro_bias.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, state.accel_bias.z());
}

void test_gravity_alignment(void) {
    ESKF eskf(DT);

    // Device flat: accel reads [0, 0, +g] (reaction to gravity)
    Eigen::Vector3f accel_flat(0.0f, 0.0f, GRAVITY_MAGNITUDE);
    eskf.initialize(accel_flat);

    const auto& state = eskf.get_state();
    // After alignment, gravity in world should be [0, 0, -g]
    // Quaternion should be identity (or close) for flat device
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, state.quat.w());
}

// ============================================================================
// Prediction Tests
// ============================================================================

void test_quaternion_normalization(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(0.1f, 0.1f, 9.81f);
    Eigen::Vector3f gyro(0.05f, 0.05f, 0.05f);

    for (int i = 0; i < 1000; i++) {
        eskf.predict(accel, gyro);
    }

    const auto& quat = eskf.get_state().quat;
    float norm = std::sqrt(quat.w()*quat.w() + quat.x()*quat.x() +
                          quat.y()*quat.y() + quat.z()*quat.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, norm);
}

void test_stationary_no_drift(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 250; i++) {
        eskf.predict(accel, gyro);
    }

    const auto& pos = eskf.get_state().pos;
    float drift = std::sqrt(pos.x()*pos.x() + pos.y()*pos.y() + pos.z()*pos.z());

    TEST_ASSERT_LESS_THAN_FLOAT(0.05f, drift);
}

void test_predict_with_small_noise(void) {
    ESKF eskf(DT);

    for (int i = 0; i < 100; i++) {
        Eigen::Vector3f accel(0.01f, -0.01f, 9.81f);
        Eigen::Vector3f gyro(0.001f, 0.001f, 0.001f);
        eskf.predict(accel, gyro);
    }

    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.quat.w()));
}

void test_covariance_symmetry_after_predict(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, 0.5f, 9.5f);
    Eigen::Vector3f gyro(0.02f, -0.01f, 0.03f);

    for (int i = 0; i < 50; i++) {
        eskf.predict(accel, gyro);
    }

    const auto& P = eskf.get_covariance();
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = i + 1; j < STATE_DIM; j++) {
            TEST_ASSERT_FLOAT_WITHIN(1e-6f, P(i, j), P(j, i));
        }
    }
}

// ============================================================================
// Stationary Update Tests (ZUPT + ZARU, log-space R)
// ============================================================================

void test_stationary_reduces_velocity(void) {
    ESKF eskf(DT);

    // Build up velocity
    Eigen::Vector3f accel(2.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 10; i++) {
        eskf.predict(accel, gyro);
    }

    float vel_before = eskf.get_state().vel.norm();
    TEST_ASSERT_GREATER_THAN_FLOAT(0.1f, vel_before);

    // Apply stationary update with no TCN prob (tight constraint)
    eskf.update_stationary(gyro);

    float vel_after = eskf.get_state().vel.norm();
    TEST_ASSERT_LESS_THAN_FLOAT(vel_before, vel_after);
}

void test_stationary_with_high_prob_tight_constraint(void) {
    // High prob (1.0) should give R = min_R = 1e-4 (tight constraint, strongly zeros velocity)
    ESKF eskf(DT);

    Eigen::Vector3f accel(3.0f, 2.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    float vel_before = eskf.get_state().vel.norm();
    TEST_ASSERT_GREATER_THAN_FLOAT(0.5f, vel_before);

    // prob=1.0: alpha=1, R=exp(LOG_R_MIN)=1e-4 (very tight)
    eskf.update_stationary(gyro, 1.0f);

    float vel_after = eskf.get_state().vel.norm();
    // With R=1e-4 (tight), velocity should be nearly zeroed
    TEST_ASSERT_LESS_THAN_FLOAT(0.1f, vel_after);
}

void test_stationary_with_low_prob_weak_constraint(void) {
    // Low prob (<= 0.5) should give R = max_R = 100 (alpha=0, weak constraint)
    ESKF eskf(DT);

    Eigen::Vector3f accel(3.0f, 2.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    float vel_before = eskf.get_state().vel.norm();

    // prob=0.3 (below onset 0.5): alpha=0, R=exp(LOG_R_MAX)=100 (very weak)
    eskf.update_stationary(gyro, 0.3f);

    float vel_after = eskf.get_state().vel.norm();
    // With R=100 (weak), velocity should barely change
    float reduction_ratio = vel_after / vel_before;
    TEST_ASSERT_GREATER_THAN_FLOAT(0.8f, reduction_ratio); // Less than 20% reduction
}

void test_stationary_onset_dead_zone(void) {
    // At exactly onset (0.5), alpha should be 0, same as below onset
    float R_at_onset = compute_expected_R(0.5f);
    float R_below_onset = compute_expected_R(0.3f);

    // Both should give max_R = 100
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 100.0f, R_at_onset);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 100.0f, R_below_onset);
}

void test_stationary_r_values_at_key_points(void) {
    // Verify R values at key probability points (from plan verification section)
    float eps = 0.5f; // generous tolerance for exp/log

    // prob=0.0: R = max_R = 100
    float R_0 = compute_expected_R(0.0f);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 100.0f, R_0);

    // prob=0.5: onset threshold, alpha=0, R = max_R = 100
    float R_05 = compute_expected_R(0.5f);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 100.0f, R_05);

    // prob=0.75: alpha = ((0.75-0.5)/0.5)^2 = 0.25
    // log_R = 0.75*4.605 + 0.25*(-9.210) = 1.15
    // R = exp(1.15) ≈ 3.16
    float R_075 = compute_expected_R(0.75f);
    TEST_ASSERT_FLOAT_WITHIN(eps, 3.16f, R_075);

    // prob=1.0: alpha=1, R = min_R = 1e-4
    float R_1 = compute_expected_R(1.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1e-4f, R_1);
}

void test_stationary_r_monotonic_decrease(void) {
    // R should decrease monotonically as prob increases above onset
    float probs[] = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float prev_R = compute_expected_R(probs[0]);

    for (int i = 1; i < 6; i++) {
        float R = compute_expected_R(probs[i]);
        TEST_ASSERT_LESS_THAN_FLOAT(prev_R, R);
        prev_R = R;
    }
}

void test_stationary_zaru_reduces_gyro_bias(void) {
    ESKF eskf(DT);

    // Initialize and set up some gyro bias via prediction with gyro input
    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.1f, -0.05f, 0.02f); // non-zero gyro

    for (int i = 0; i < 50; i++) {
        eskf.predict(accel, gyro);
    }

    // After predictions with non-zero gyro, IMU update would adjust bias
    // Apply IMU update to let bias estimator work
    Eigen::Matrix<float, 6, 1> R_diag;
    R_diag.setConstant(0.01f);
    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
        eskf.update_imu(accel, gyro, R_diag);
    }

    float bias_before = eskf.get_state().gyro_bias.norm();

    // Now apply stationary update with high confidence — ZARU should push bias toward zero
    Eigen::Vector3f gyro_zero(0.0f, 0.0f, 0.0f);
    eskf.update_stationary(gyro_zero, 1.0f);

    float bias_after = eskf.get_state().gyro_bias.norm();

    // ZARU with high confidence should reduce gyro bias
    TEST_ASSERT_LESS_THAN_FLOAT(bias_before, bias_after);
}

void test_stationary_6d_observation_dimensions(void) {
    // Verify 6D update doesn't crash and produces valid state
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, -0.5f, 9.5f);
    Eigen::Vector3f gyro(0.01f, 0.02f, -0.01f);

    for (int i = 0; i < 30; i++) {
        eskf.predict(accel, gyro);
    }

    // Apply with various prob values
    eskf.update_stationary(gyro, 0.6f);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));

    eskf.update_stationary(gyro, 0.9f);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));

    eskf.update_stationary(gyro, -1.0f); // no prob
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));
}

void test_stationary_covariance_symmetry(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, 0.5f, 9.5f);
    Eigen::Vector3f gyro(0.02f, -0.01f, 0.03f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    // Apply stationary update (Joseph form should maintain symmetry)
    eskf.update_stationary(gyro, 0.8f);

    const auto& P = eskf.get_covariance();
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = i + 1; j < STATE_DIM; j++) {
            TEST_ASSERT_FLOAT_WITHIN(1e-6f, P(i, j), P(j, i));
        }
    }
}

void test_stationary_covariance_positive_diagonal(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, 0.5f, 9.5f);
    Eigen::Vector3f gyro(0.02f, -0.01f, 0.03f);

    for (int i = 0; i < 30; i++) {
        eskf.predict(accel, gyro);
    }

    // Apply multiple stationary updates
    for (int i = 0; i < 5; i++) {
        eskf.update_stationary(gyro, 0.9f);
    }

    const auto& P = eskf.get_covariance();
    for (int i = 0; i < STATE_DIM; i++) {
        TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, P(i, i));
    }
}

void test_stationary_multiple_updates_convergence(void) {
    // Multiple high-confidence stationary updates should converge velocity to zero
    ESKF eskf(DT);

    Eigen::Vector3f accel(5.0f, 3.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 50; i++) {
        eskf.predict(accel, gyro);
    }

    float initial_vel = eskf.get_state().vel.norm();
    TEST_ASSERT_GREATER_THAN_FLOAT(1.0f, initial_vel);

    // Apply many high-confidence stationary updates
    for (int i = 0; i < 10; i++) {
        eskf.update_stationary(gyro, 0.99f);
    }

    float final_vel = eskf.get_state().vel.norm();
    TEST_ASSERT_LESS_THAN_FLOAT(0.01f, final_vel);
}

// ============================================================================
// TCN Velocity Update Tests
// ============================================================================

void test_tcn_velocity_update(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    float vel_before = eskf.get_state().vel.x();

    // TCN correction suggesting slower velocity
    Eigen::Vector3f vel_correction(-0.5f, 0.0f, 0.0f);
    // R_params: raw values before softplus (6 values: 3 vel + 3 gyro)
    Eigen::Matrix<float, 6, 1> R_params;
    R_params << -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f; // softplus(-2) ≈ 0.127

    eskf.update_tcn_vel(vel_correction, R_params);

    float vel_after = eskf.get_state().vel.x();
    // Velocity should change in the correction direction
    TEST_ASSERT_TRUE(vel_after < vel_before);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_large_gyro_rates(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(10.0f, 0.0f, 0.0f);

    for (int i = 0; i < 50; i++) {
        eskf.predict(accel, gyro);
    }

    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.quat.w()));
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
}

void test_zero_input(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    eskf.predict(accel, gyro);

    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.x()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.x()));
}

void test_stationary_with_extreme_prob_values(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(1.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 10; i++) {
        eskf.predict(accel, gyro);
    }

    // Edge case: prob exactly at boundaries
    eskf.update_stationary(gyro, 0.0f);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));

    eskf.update_stationary(gyro, 1.0f);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));

    // Edge case: very small prob
    eskf.update_stationary(gyro, 1e-6f);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));

    // Edge case: no prob (default path)
    eskf.update_stationary(gyro);
    TEST_ASSERT_TRUE(std::isfinite(eskf.get_state().vel.norm()));
}

void test_hard_reset_velocity(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel(5.0f, 3.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 20; i++) {
        eskf.predict(accel, gyro);
    }

    TEST_ASSERT_GREATER_THAN_FLOAT(0.5f, eskf.get_state().vel.norm());

    eskf.hard_reset_velocity();

    TEST_ASSERT_FLOAT_WITHIN(1e-8f, 0.0f, eskf.get_state().vel.norm());
}

// ============================================================================
// Integration Test
// ============================================================================

void test_predict_update_cycle(void) {
    ESKF eskf(DT);

    for (int i = 0; i < 200; i++) {
        Eigen::Vector3f accel(
            0.01f * std::sin(i * 0.1f),
            0.01f * std::cos(i * 0.1f),
            9.81f
        );
        Eigen::Vector3f gyro(0.005f, 0.005f, 0.005f);

        eskf.predict(accel, gyro);

        // Apply stationary update every 20 steps (simulating ZUPT)
        if (i % 20 == 0) {
            eskf.update_stationary(gyro, 0.8f);
        }

        // Apply TCN correction every 10 steps
        if (i % 10 == 0) {
            Eigen::Vector3f vel_corr(0.01f, 0.01f, 0.0f);
            Eigen::Matrix<float, 6, 1> R_params;
            R_params.setConstant(0.0f); // softplus(0) ≈ 0.693
            eskf.update_tcn_vel(vel_corr, R_params);
        }
    }

    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(state.vel.norm()));

    float quat_norm = state.quat.norm();
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, quat_norm);
}

void test_full_pipeline_stationary_then_motion(void) {
    ESKF eskf(DT);

    Eigen::Vector3f accel_static(0.0f, 0.0f, GRAVITY_MAGNITUDE);
    Eigen::Vector3f gyro_zero(0.0f, 0.0f, 0.0f);

    eskf.initialize(accel_static);

    // Phase 1: Stationary (2s = 100 steps @ 50Hz)
    for (int i = 0; i < 100; i++) {
        eskf.predict(accel_static, gyro_zero);
        eskf.update_stationary(gyro_zero, 0.95f);
    }

    // After stationary period, velocity should be near zero
    TEST_ASSERT_LESS_THAN_FLOAT(0.01f, eskf.get_state().vel.norm());

    // Phase 2: Motion (1s = 50 steps)
    Eigen::Vector3f accel_move(2.0f, 1.0f, GRAVITY_MAGNITUDE);
    for (int i = 0; i < 50; i++) {
        eskf.predict(accel_move, gyro_zero);
    }

    // Should have accumulated velocity
    TEST_ASSERT_GREATER_THAN_FLOAT(0.5f, eskf.get_state().vel.norm());

    // Phase 3: Return to stationary
    for (int i = 0; i < 50; i++) {
        eskf.predict(accel_static, gyro_zero);
        eskf.update_stationary(gyro_zero, 0.99f);
    }

    // Velocity should converge back toward zero
    TEST_ASSERT_LESS_THAN_FLOAT(0.1f, eskf.get_state().vel.norm());

    // All state should be finite
    const auto& state = eskf.get_state();
    TEST_ASSERT_TRUE(std::isfinite(state.pos.norm()));
    TEST_ASSERT_TRUE(std::isfinite(state.quat.norm()));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, state.quat.norm());
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
    RUN_TEST(test_gravity_alignment);

    // Prediction tests
    RUN_TEST(test_quaternion_normalization);
    RUN_TEST(test_stationary_no_drift);
    RUN_TEST(test_predict_with_small_noise);
    RUN_TEST(test_covariance_symmetry_after_predict);

    // Stationary update tests (ZUPT + ZARU, log-space R)
    RUN_TEST(test_stationary_reduces_velocity);
    RUN_TEST(test_stationary_with_high_prob_tight_constraint);
    RUN_TEST(test_stationary_with_low_prob_weak_constraint);
    RUN_TEST(test_stationary_onset_dead_zone);
    RUN_TEST(test_stationary_r_values_at_key_points);
    RUN_TEST(test_stationary_r_monotonic_decrease);
    RUN_TEST(test_stationary_zaru_reduces_gyro_bias);
    RUN_TEST(test_stationary_6d_observation_dimensions);
    RUN_TEST(test_stationary_covariance_symmetry);
    RUN_TEST(test_stationary_covariance_positive_diagonal);
    RUN_TEST(test_stationary_multiple_updates_convergence);

    // TCN velocity update tests
    RUN_TEST(test_tcn_velocity_update);

    // Edge cases
    RUN_TEST(test_large_gyro_rates);
    RUN_TEST(test_zero_input);
    RUN_TEST(test_stationary_with_extreme_prob_values);
    RUN_TEST(test_hard_reset_velocity);

    // Integration
    RUN_TEST(test_predict_update_cycle);
    RUN_TEST(test_full_pipeline_stationary_then_motion);

    UNITY_END();

    printf("\n");
    printf("========================================\n");
    printf("  Tests Complete!\n");
    printf("========================================\n");
}
