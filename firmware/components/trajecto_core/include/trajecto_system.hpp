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

#include "eskf.hpp"
#include "eskf_fixed.hpp"
#include "tcn_wrapper.hpp"
#include "fast_math_lut.hpp"
#include <cmath>

namespace trajecto {

// Define TRAJECTO_USE_FIXED_POINT to use fixed-point ESKF (for ESP32C3 without FPU).
// When not defined, uses the original floating-point Eigen-based ESKF.
// This can be set via CMakeLists.txt: target_compile_definitions(... -DTRAJECTO_USE_FIXED_POINT)

class TrajectoSystem {
public:
    TrajectoSystem() :
#ifdef TRAJECTO_USE_FIXED_POINT
        eskf_fixed_(DT),
#endif
        eskf_(DT)
    {
        last_innovation_.setZero();
    }

    bool setup() {
        return tcn_.setup();
    }

    void initialize(const Eigen::Vector3f& accel_init) {
#ifdef TRAJECTO_USE_FIXED_POINT
        float accel_arr[3] = { accel_init.x(), accel_init.y(), accel_init.z() };
        eskf_fixed_.initialize(accel_arr);
#endif
        eskf_.initialize(accel_init);
        last_innovation_.setZero();
        initialized_ = true;

        // Reset stride-based TCN control
        step_counter_ = 0;
        tcn_output_valid_ = false;
    }

    void step(const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro, float force) {
        if (!initialized_) {
            initialize(accel);
            return;
        }

#ifdef TRAJECTO_USE_FIXED_POINT
        step_fixed(accel, gyro, force);
#else
        step_float(accel, gyro, force);
#endif
    }

    const NominalState& get_state() const {
#ifdef TRAJECTO_USE_FIXED_POINT
        // Cache the float conversion
        cached_float_state_ = eskf_fixed_.get_state();
        return cached_float_state_;
#else
        return eskf_.get_state();
#endif
    }

    const TCNWrapper& get_tcn() const { return tcn_; }
    float get_zupt_prob() const { return zupt_prob_; }

#ifdef TRAJECTO_USE_FIXED_POINT
    const ESKFFixed& get_eskf_fixed() const { return eskf_fixed_; }
#endif

private:
    // Float-point path (original, uses Eigen)
    void step_float(const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro, float force) {
        bool is_zupt_heuristic = eskf_.check_zupt(accel);
        bool is_zupt = is_zupt_heuristic;
        float current_step_prob = 0.0f;

        eskf_.predict(gyro, accel);

        // Stride-based TCN: run only every N timesteps (zero-order hold, matches Python training)
        TCNOutput tcn_out;
        if (step_counter_ % TCN_UPDATE_STRIDE == 0) {
            tcn_out = tcn_.process_step(accel, gyro, force, eskf_, last_innovation_, is_zupt_heuristic);
            cached_tcn_out_ = tcn_out;  // Cache for zero-order hold
            tcn_output_valid_ = true;
        } else if (tcn_output_valid_) {
            tcn_out = cached_tcn_out_;  // Reuse cached output
        } else {
            tcn_out.valid = false;
        }
        step_counter_++;

        Eigen::Matrix<float, 6, 1> R_diag;
        R_diag.setConstant(R_MIN);

        if (tcn_out.valid) {
            if (tcn_out.zupt_prob > ZUPT_HARD_RESET_THRESHOLD) {
                is_zupt = true;
            }
            current_step_prob = tcn_out.zupt_prob;

            if (!is_zupt) {
                eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params);
            }

            for(int i=0; i<6; i++) {
                 float val = fast_softplus(tcn_out.R_params[i]);
                 val = std::max(R_MIN, std::min(R_MAX, val + R_MIN));
                 R_diag[i] = val;
            }
        } else {
             current_step_prob = is_zupt ? 1.0f : 0.0f;
        }

        zupt_prob_ = current_step_prob;

        if (is_zupt) {
            float prob_to_pass = (tcn_out.valid && tcn_out.zupt_prob > 0.5f) ? tcn_out.zupt_prob : -1.0f;
            eskf_.update_stationary(gyro, prob_to_pass);
        }

        float mahalanobis_sq = 0.0f;
        last_innovation_ = eskf_.update_imu(accel, gyro, R_diag, &mahalanobis_sq);

        if (tcn_out.valid && tcn_out.zupt_prob >= ZUPT_HARD_RESET_THRESHOLD) {
            eskf_.hard_reset_velocity();
        }
    }

#ifdef TRAJECTO_USE_FIXED_POINT
    // Fixed-point path (for ESP32C3 without FPU)
    void step_fixed(const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro, float force) {
        float accel_f[3] = { accel.x(), accel.y(), accel.z() };
        float gyro_f[3] = { gyro.x(), gyro.y(), gyro.z() };

        bool is_zupt_heuristic = eskf_fixed_.check_zupt(accel_f);
        bool is_zupt = is_zupt_heuristic;
        float current_step_prob = 0.0f;

        eskf_fixed_.predict(gyro_f, accel_f);

        // Stride-based TCN: same logic, but use float ESKF for feature extraction
        // (TCN wrapper still uses float Eigen interface)
        // We maintain a shadow float ESKF in sync for TCN feature extraction
        eskf_.predict(gyro, accel);

        TCNOutput tcn_out;
        if (step_counter_ % TCN_UPDATE_STRIDE == 0) {
            tcn_out = tcn_.process_step(accel, gyro, force, eskf_, last_innovation_, is_zupt_heuristic);
            cached_tcn_out_ = tcn_out;
            tcn_output_valid_ = true;
        } else if (tcn_output_valid_) {
            tcn_out = cached_tcn_out_;
        } else {
            tcn_out.valid = false;
        }
        step_counter_++;

        float R_diag_arr[6];
        for (int i = 0; i < 6; ++i) R_diag_arr[i] = R_MIN;

        if (tcn_out.valid) {
            if (tcn_out.zupt_prob > ZUPT_HARD_RESET_THRESHOLD) {
                is_zupt = true;
            }
            current_step_prob = tcn_out.zupt_prob;

            if (!is_zupt) {
                float vc[3] = { tcn_out.vel_corr.x(), tcn_out.vel_corr.y(), tcn_out.vel_corr.z() };
                float rp[6];
                for (int i = 0; i < 6; ++i) rp[i] = tcn_out.R_params[i];
                eskf_fixed_.update_tcn_vel(vc, rp);
            }

            for (int i = 0; i < 6; i++) {
                float val = fast_softplus(tcn_out.R_params[i]);
                val = std::max(R_MIN, std::min(R_MAX, val + R_MIN));
                R_diag_arr[i] = val;
            }
        } else {
            current_step_prob = is_zupt ? 1.0f : 0.0f;
        }

        zupt_prob_ = current_step_prob;

        if (is_zupt) {
            float prob_to_pass = (tcn_out.valid && tcn_out.zupt_prob > 0.5f) ? tcn_out.zupt_prob : -1.0f;
            eskf_fixed_.update_stationary(gyro_f, prob_to_pass);
        }

        float innovation_f[6];
        float mahalanobis_sq = 0.0f;
        eskf_fixed_.update_imu(accel_f, gyro_f, R_diag_arr, innovation_f, &mahalanobis_sq);

        // Update last_innovation_ for TCN feature extraction (float path)
        for (int i = 0; i < 6; ++i) last_innovation_[i] = innovation_f[i];

        // Also update float ESKF shadow for TCN (sync bias/quat state)
        // This is approximate but sufficient for feature extraction
        Eigen::Matrix<float, 6, 1> R_diag_eigen;
        for (int i = 0; i < 6; ++i) R_diag_eigen[i] = R_diag_arr[i];
        eskf_.update_imu(accel, gyro, R_diag_eigen);

        if (tcn_out.valid && tcn_out.zupt_prob >= ZUPT_HARD_RESET_THRESHOLD) {
            eskf_fixed_.hard_reset_velocity();
        }
    }
#endif

    ESKF eskf_;
#ifdef TRAJECTO_USE_FIXED_POINT
    ESKFFixed eskf_fixed_;
    mutable NominalState cached_float_state_;
#endif
    TCNWrapper tcn_;
    Eigen::Matrix<float, 6, 1> last_innovation_;
    bool initialized_ = false;
    float zupt_prob_ = 0.0f;

    // Stride-based TCN control (matches Python training: TCN runs every 4 timesteps)
    int step_counter_ = 0;           // Counts timesteps for stride control
    TCNOutput cached_tcn_out_;       // Zero-order hold: cached TCN output
    bool tcn_output_valid_ = false;  // Whether we have a valid cached output
};

} // namespace trajecto
