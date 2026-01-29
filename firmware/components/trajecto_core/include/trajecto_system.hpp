#pragma once

#include "eskf.hpp"
#include "tcn_wrapper.hpp"
#include "fast_math_lut.hpp"
#include <iostream>
#include <cmath>

namespace trajecto {

class TrajectoSystem {
public:
    TrajectoSystem() : eskf_(DT) {
        last_innovation_.setZero();
    }

    bool setup() {
        return tcn_.setup();
    }

    void initialize(const Eigen::Vector3f& accel_init) {
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
            // No valid TCN output yet (initial timesteps before first TCN run)
            tcn_out.valid = false;
        }
        step_counter_++;

        Eigen::Matrix<float, 6, 1> R_diag;
        R_diag.setConstant(R_MIN);

        if (tcn_out.valid) {
            if (tcn_out.zupt_prob > ZUPT_PROB_THRESHOLD) {
                is_zupt = true;
            }
            current_step_prob = tcn_out.zupt_prob;

            // Apply TCN velocity correction only when NOT in ZUPT (matches Python)
            if (!is_zupt) {
                eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params);
            }

            // Process R_params for IMU update
            for(int i=0; i<6; i++) {
                 float val = fast_softplus(tcn_out.R_params[i]);
                 // Clamp to valid range [R_MIN, R_MAX] (matches Python)
                 val = std::max(R_MIN, std::min(R_MAX, val + R_MIN));
                 R_diag[i] = val;
            }
        } else {
             current_step_prob = is_zupt ? 1.0f : 0.0f;
        }

        zupt_prob_ = current_step_prob;

        if (is_zupt) {
            float prob_to_pass = (tcn_out.valid && tcn_out.zupt_prob > 0.5f) ? tcn_out.zupt_prob : -1.0f;
            eskf_.update_zupt(prob_to_pass);
        }

        float mahalanobis_sq = 0.0f;
        last_innovation_ = eskf_.update_imu(accel, gyro, R_diag, &mahalanobis_sq);

        // ZUPT Hard Reset: When TCN is very confident about stationary state,
        // directly reset velocity to zero (matches Python ESKF.py:901-908)
        if (tcn_out.valid && tcn_out.zupt_prob >= ZUPT_HARD_RESET_THRESHOLD) {
            eskf_.hard_reset_velocity();
        }
    }

    const NominalState& get_state() const { return eskf_.get_state(); }
    const TCNWrapper& get_tcn() const { return tcn_; }
    float get_zupt_prob() const { return zupt_prob_; }

private:
    ESKF eskf_;
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
