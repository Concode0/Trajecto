#pragma once

#include "eskf.hpp"
#include "tcn_wrapper.hpp"
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
        float zupt_prob = -1.0f;

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
            zupt_prob = tcn_out.zupt_prob;

            // Apply TCN Velocity Correction (if not ZUPT)
            if (!is_zupt) {
                eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params);
            }

            // Parse R from TCN for IMU Update
            for(int i=0; i<6; i++) {
                 float x = tcn_out.R_params[i];
                 // Softplus approximation: log(1 + exp(x))
                 // Use simple check to avoid overflow for large x
                 float val = (x > 20.0f) ? x : std::log(1.0f + std::exp(x));
                 R_diag[i] = val + 1e-6f;
            }
        }

        // 4. ZUPT Update
        if (is_zupt) {
            eskf_.update_zupt(zupt_prob);
        }

        // 5. Standard IMU Update
        // This estimates biases and generates innovation for the next TCN step.
        // It now includes Mahalanobis Gating logic inside.
        float mahalanobis_sq = 0.0f;
        last_innovation_ = eskf_.update_imu(accel, gyro, R_diag, &mahalanobis_sq);
    }

    const NominalState& get_state() const { return eskf_.get_state(); }
    const TCNWrapper& get_tcn() const { return tcn_; }

private:
    ESKF eskf_;
    TCNWrapper tcn_;
    Eigen::Matrix<float, 6, 1> last_innovation_;
    bool initialized_ = false;
};

} // namespace trajecto
