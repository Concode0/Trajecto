/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System
 * Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * [PATENT NOTICE]
 * This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
 * Commercial use without a separate license is strictly prohibited.
 *
 * Contact: nemonanconcode@gmail.com
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

    /** @brief Reset covariance to initial values (divergence recovery) */
    void reset_covariance();

    /** @brief Reset covariance + zero velocity + tighten velocity P (divergence ZUPT) */
    void reset_covariance_with_zupt();

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
