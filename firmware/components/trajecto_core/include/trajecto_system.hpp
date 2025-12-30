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

        // 1. ZUPT Detection (Heuristic)
        bool is_zupt_heuristic = eskf_.check_zupt(accel);
        bool is_zupt = is_zupt_heuristic;
        float current_step_prob = 0.0f; // Default for visualization

        // 2. Predict (Propagate State)
        eskf_.predict(gyro, accel);

        // 3. TCN Inference
        // TCN needs features which depend on current state and *last* innovation
        TCNOutput tcn_out = tcn_.process_step(accel, gyro, force, eskf_, last_innovation_, is_zupt_heuristic);

        // Prepare Measurement Noise R
        Eigen::Matrix<float, 6, 1> R_diag;
        R_diag.setConstant(1e-4f); // Default baseline

        if (tcn_out.valid) {
            // TCN Override for ZUPT
            if (tcn_out.zupt_prob > 0.5f) {
                is_zupt = true;
            }
            current_step_prob = tcn_out.zupt_prob;

            // Apply TCN Velocity Correction (if not ZUPT)
            if (!is_zupt) {
                eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params);
            }

            // Parse R from TCN for IMU Update
            for(int i=0; i<6; i++) {
                 // Softplus approximation using LUT
                 float val = fast_softplus(tcn_out.R_params[i]);
                 R_diag[i] = val + 1e-6f;
            }
        } else {
             // If TCN not running (e.g. buffer filling), rely on heuristic
             current_step_prob = is_zupt ? 1.0f : 0.0f;
        }
        
        // Store for Getter
        zupt_prob_ = current_step_prob;

        // 4. ZUPT Update
        // Note: ESKF::update_zupt expects a probability if available, or uses default high confidence if -1
        // We pass -1 if we just want "hard" ZUPT, or pass the probability if we have one.
        // If it was TCN ZUPT, we have a prob. If heuristic, we treat it as 1.0.
        if (is_zupt) {
            float prob_to_pass = (tcn_out.valid && tcn_out.zupt_prob > 0.5f) ? tcn_out.zupt_prob : -1.0f;
            eskf_.update_zupt(prob_to_pass);
        }

        // 5. Standard IMU Update
        // This estimates biases and generates innovation for the next TCN step.
        // It now includes Mahalanobis Gating logic inside.
        float mahalanobis_sq = 0.0f;
        last_innovation_ = eskf_.update_imu(accel, gyro, R_diag, &mahalanobis_sq);
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
