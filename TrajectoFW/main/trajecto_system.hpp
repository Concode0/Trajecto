#pragma once

#include "eskf.hpp"
#include "tcn_wrapper.hpp"
#include <iostream>

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
        bool is_zupt = eskf_.check_zupt(accel);

        // 2. Predict (Propagate State)
        eskf_.predict(gyro, accel);

        // 3. TCN Inference
        // TCN needs features which depend on current state and *last* innovation
        TCNOutput tcn_out = tcn_.process_step(accel, gyro, force, eskf_, last_innovation_, is_zupt);

        // 4. Update
        if (tcn_out.valid) {
            // TCN Override for ZUPT?
            // Python: is_zupt = tcn_output["zupt_prob"] > 0.5
            if (tcn_out.zupt_prob > 0.5f) {
                is_zupt = true;
            }

            if (is_zupt) {
                eskf_.update_zupt();
                // ZUPT update doesn't produce standard innovation for next step features?
                // Actually, Python code: "innovation_output = innovation" (from update_imu or 0 if only ZUPT?)
                // If only ZUPT is applied, innovation_output is usually zeroed or specific to ZUPT?
                // Python: if ZUPT, apply ZUPT. If TCN, also apply TCN vel corr (masked if ZUPT).
                // AND "Standard Measurement Update (with TCN-provided adaptive R)" is ALWAYS applied in Python loop
                // UNLESS we are in a simplified mode.
                // In `ESKF_TCN.py`:
                //   if torch.any(is_zupt): apply ZUPT
                //   if tcn_output: apply TCN Vel Corr (masked) AND apply Update (with R override)
                
                // So we should ALWAYS do update_imu for feature generation purposes at least.
                // But if we are stationary, IMU update helps converge biases.
                
                // Let's replicate Python logic:
                // 1. ZUPT Update (if needed)
                // 2. TCN Velocity Correction (if valid and not stationary?)
                // 3. IMU Update (Always, maybe with TCN R)
                
                // Correction: Python applies TCN Vel Corr *unless* ZUPT is active.
                // "vel_corr_body = torch.where(is_zupt..., zeros, vel_corr_body)"
                if (!is_zupt) {
                   eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params); 
                }
                
                // 4. IMU Measurement Update (For Biases + Innovation Feature)
                // TCN provides R params (covariance_R).
                // Python: R = softplus(covariance_R).
                // We need to process R_params.
                Eigen::Matrix<float, 6, 1> R_diag;
                for(int i=0; i<6; i++) {
                     // Softplus approximation: log(1 + exp(x))
                     // TFLite doesn't have softplus op usually enabled?
                     // We can do it here.
                     float x = tcn_out.R_params[i];
                     R_diag[i] = std::log(1.0f + std::exp(x));
                }
                
                last_innovation_ = eskf_.update_imu(accel, gyro, R_diag);
                
            } else {
                // TCN Valid but not ZUPT
                // Apply TCN Vel
                eskf_.update_tcn_vel(tcn_out.vel_corr, tcn_out.R_params);
                
                // Apply IMU Update
                Eigen::Matrix<float, 6, 1> R_diag;
                for(int i=0; i<6; i++) {
                     float x = tcn_out.R_params[i];
                     R_diag[i] = std::log(1.0f + std::exp(x));
                }
                last_innovation_ = eskf_.update_imu(accel, gyro, R_diag);
            }
            
        } else {
            // TCN Not Ready (Startup)
            if (is_zupt) {
                eskf_.update_zupt();
            }
            
            // Standard IMU Update with default R
            Eigen::Matrix<float, 6, 1> R_default;
            R_default.setConstant(1e-4f); // Default
            last_innovation_ = eskf_.update_imu(accel, gyro, R_default);
        }
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
