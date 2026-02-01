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

#include "eskf.hpp"
#include "fixed_point.hpp"
#include "fixed_ops.hpp"
#include "fixed_point_constants.hpp"

namespace trajecto {

// ---------------------------------------------------------------------------
// Fixed-Point Nominal State
// ---------------------------------------------------------------------------
struct NominalStateFixed {
    fp::Vec3_Q4_27  pos;        // position in meters (Q5.27)
    fp::Vec3_Q4_27  vel;        // velocity in m/s (Q5.27)
    fp::QuatQ       quat;       // unit quaternion (Q2.30)
    fp::Vec3_Q1_30  gyro_bias;  // gyro bias in rad/s (Q2.30)
    fp::Vec3_Q4_27  accel_bias; // accel bias in m/s² (Q5.27)

    NominalStateFixed() {
        quat = fp::QuatQ::identity();
    }

    // Convert to float NominalState for output / compatibility
    NominalState to_float() const;
};

// ---------------------------------------------------------------------------
// Fixed-Point ESKF
// ---------------------------------------------------------------------------
class ESKFFixed {
public:
    ESKFFixed(float dt);

    void initialize(const float accel_init[3]);

    void predict(const float gyro_raw[3], const float accel_raw[3]);

    void update_stationary(const float gyro_raw[3], float prob = -1.0f);

    void update_tcn_vel(const float vel_corr_body[3], const float R_params[6]);

    // Returns innovation as float[6] for compatibility with TCN feature extraction
    void update_imu(
        const float accel_raw[3],
        const float gyro_raw[3],
        const float R_diag[6],
        float out_innovation[6],
        float* out_mahalanobis = nullptr
    );

    bool check_zupt(const float accel_raw[3]);

    void hard_reset_velocity();

    /** @brief Reset covariance to initial values (divergence recovery) */
    void reset_covariance();

    /** @brief Reset covariance + zero velocity + tighten velocity P (divergence ZUPT) */
    void reset_covariance_with_zupt();

    // Getters
    const NominalStateFixed& get_state_fixed() const { return state_; }
    NominalState get_state() const { return state_.to_float(); }

    // Get P diagonal for monitoring (returns float for diagnostics)
    void get_P_diagonal(float out[15]) const;

private:
    void inject_error(
        fp::Vec3_Q4_27 d_pos,
        fp::Vec3_Q4_27 d_vel,
        fp::Vec3_Q1_30 d_theta,
        fp::Vec3_Q1_30 d_bg,
        fp::Vec3_Q4_27 d_ba
    );

    void enforce_symmetry();

    // Internal helpers to convert float[3] to fixed-point
    static fp::Vec3_Q16_15 float3_to_q16_15(const float v[3]);
    static fp::Vec3_Q4_27  float3_to_q4_27(const float v[3]);

    // State
    NominalStateFixed state_;

    // Covariance: 5x5 block matrix of ScaledBlock3x3
    fp::BlockCovMatrix P_;

    // Process noise Q diagonal blocks (set at construction from Allan variance)
    fp::ScaledBlock3x3 Q_blocks_[5]; // diagonal blocks only

    // Gravity in world frame
    fp::Vec3_Q4_27 gravity_w_;

    // Cached constants
    fp::Q4_27 dt_;
    fp::Q4_27 half_dt2_;
};

} // namespace trajecto
