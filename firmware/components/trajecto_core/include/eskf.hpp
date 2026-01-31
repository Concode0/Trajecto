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

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include "model_params.hpp"

namespace trajecto {

constexpr int STATE_DIM = 15;

struct NominalState {
    Eigen::Vector3f pos;
    Eigen::Vector3f vel;
    Eigen::Quaternionf quat;
    Eigen::Vector3f gyro_bias;
    Eigen::Vector3f accel_bias;

    NominalState() {
        pos.setZero();
        vel.setZero();
        quat.setIdentity();
        gyro_bias.setZero();
        accel_bias.setZero();
    }
};

class ESKF {
public:
    ESKF(float dt);

    /** @brief Initialize via gravity alignment */
    void initialize(const Eigen::Vector3f& accel_init);

    /** @brief Propagate nominal state and error covariance */
    void predict(const Eigen::Vector3f& gyro_raw, const Eigen::Vector3f& accel_raw);

    /** @brief Stationary update: ZUPT (zero velocity) + ZARU (zero angular rate)
     *  Uses log-space soft-thresholding for R computation.
     *  6D observation: vel=0, gyro_bias=0 when stationary. */
    void update_stationary(const Eigen::Vector3f& gyro_raw, float prob = -1.0f);

    /** @brief Apply TCN velocity correction in body frame */
    void update_tcn_vel(const Eigen::Vector3f& vel_corr_body, const Eigen::Matrix<float, 6, 1>& R_params);

    /** @brief Standard IMU update for bias estimation with Mahalanobis gating */
    Eigen::Matrix<float, 6, 1> update_imu(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        const Eigen::Matrix<float, 6, 1>& R_diag,
        float* out_mahalanobis = nullptr
    );

    /** @brief Simple threshold-based ZUPT check */
    bool check_zupt(const Eigen::Vector3f& accel_raw);

    /** @brief Hard reset velocity to zero (for high-confidence ZUPT) */
    void hard_reset_velocity();

    // Getters
    const NominalState& get_state() const { return state_; }
    const Eigen::Matrix<float, STATE_DIM, STATE_DIM>& get_covariance() const { return P_; }

private:
    void inject_error(const Eigen::Matrix<float, STATE_DIM, 1>& delta_x);
    void enforce_symmetry();

    float dt_;
    NominalState state_;
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> P_;
    Eigen::Matrix<float, STATE_DIM, 1> Q_diag_;

    float zupt_noise_std_;
    float tcn_vel_noise_std_;

    Eigen::Vector3f gravity_w_;

    float mahalanobis_gate_threshold_;
};

} // namespace trajecto
