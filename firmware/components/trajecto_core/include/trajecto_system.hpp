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

        TCNOutput tcn_out = tcn_.process_step(accel, gyro, force, eskf_, last_innovation_, is_zupt_heuristic);

        Eigen::Matrix<float, 6, 1> R_diag;
        R_diag.setConstant(1e-4f);

        if (tcn_out.valid) {
            if (tcn_out.zupt_prob > 0.5f) {
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
                 // Clamp to valid range [1e-4, 3.0] (matches Python)
                 val = std::max(1e-4f, std::min(3.0f, val + 1e-4f));
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
};

} // namespace trajecto
