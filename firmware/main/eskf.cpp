#include "eskf.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace trajecto {

ESKF::ESKF(float dt) : dt_(dt) {
    // Initialize Covariance P with small uncertainty
    P_.setIdentity();
    P_ *= 0.1f;

    // Initialize Process Noise Q (Diagonal)
    // VRW (Velocity Random Walk) -> Accel Noise
    float vrw = 1.0e-3f; 
    // ARW (Angle Random Walk) -> Gyro Noise
    float arw = 4.0e-3f;
    // Bias Instability
    float accel_bi = 4.0e-4f;
    float gyro_bi = 2.0e-3f;

    Q_diag_.setZero();
    // Velocity Error driven by VRW
    Q_diag_.segment<3>(3).setConstant(vrw * vrw);
    // Orientation Error driven by ARW
    Q_diag_.segment<3>(6).setConstant(arw * arw);
    // Bias Errors driven by Bias Instability
    Q_diag_.segment<3>(9).setConstant(gyro_bi * gyro_bi);
    Q_diag_.segment<3>(12).setConstant(accel_bi * accel_bi);

    // Measurement Noise Defaults
    zupt_noise_std_ = 0.01f; // 1 cm/s confidence
    tcn_vel_noise_std_ = 0.05f; 

    // Gravity (Z-up assumption for World Frame)
    gravity_w_ << 0.0f, 0.0f, 9.81f;
    
    // Gating
    mahalanobis_gate_threshold_ = 20.0f;
}

void ESKF::initialize(const Eigen::Vector3f& accel_init) {
    // Initialize orientation by aligning gravity
    // Assume stationary: accel_init should be ~ [0, 0, g] in World
    // In Body frame it is R_bw^T * g_w.
    // We want to find q_bw such that R_bw * accel_init = [0, 0, g]
    // Or simpler: Rotation from accel_init to [0, 0, 1] is q_wb (inverse)
    
    Eigen::Vector3f accel_norm = accel_init.normalized();
    Eigen::Vector3f target_up = Eigen::Vector3f::UnitZ();

    // Rotation that aligns measured accel to Z-up
    Eigen::Quaternionf q_init = Eigen::Quaternionf::FromTwoVectors(accel_norm, target_up);
    
    // Set state
    state_.quat = q_init;
    state_.pos.setZero();
    state_.vel.setZero();
    state_.gyro_bias.setZero();
    state_.accel_bias.setZero();
    
    // Reset Covariance
    P_.setIdentity();
    P_ *= 0.1f;
}

void ESKF::predict(const Eigen::Vector3f& gyro_raw, const Eigen::Vector3f& accel_raw) {
    // 1. Nominal State Propagation
    Eigen::Vector3f gyro_corrected = gyro_raw - state_.gyro_bias;
    Eigen::Vector3f accel_corrected = accel_raw - state_.accel_bias;

    // Rotation Matrix
    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();

    // Accel in World Frame
    Eigen::Vector3f accel_w = R_bw * accel_corrected - gravity_w_;

    // Position Update
    state_.pos += state_.vel * dt_ + 0.5f * accel_w * dt_ * dt_;
    
    // Velocity Update
    state_.vel += accel_w * dt_;

    // Orientation Update (0th order integration)
    // q_new = q_old * q_delta
    // q_delta is from angular velocity vector
    Eigen::Vector3f omega = gyro_corrected * dt_;
    float angle = omega.norm();
    Eigen::Quaternionf q_delta;
    if (angle > 1e-8f) {
        q_delta = Eigen::Quaternionf(Eigen::AngleAxisf(angle, omega / angle));
    } else {
        q_delta = Eigen::Quaternionf::Identity();
    }
    state_.quat = state_.quat * q_delta;
    state_.quat.normalize();

    // 2. Error Covariance Prediction
    // F Matrix (15x15) linearization
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> F;
    F.setIdentity();

    // Pos deriv wrt Vel
    F.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dt_;

    // Vel deriv wrt Ori ( -R * [a_b]_x * dt )
    Eigen::Matrix3f accel_skew;
    accel_skew << 0, -accel_corrected.z(), accel_corrected.y(),
                  accel_corrected.z(), 0, -accel_corrected.x(),
                  -accel_corrected.y(), accel_corrected.x(), 0;
    F.block<3, 3>(3, 6) = -R_bw * accel_skew * dt_;
    
    // Vel deriv wrt Accel Bias ( -R * dt )
    F.block<3, 3>(3, 12) = -R_bw * dt_;

    // Ori deriv wrt Ori (Identity approximation)
    // Ori deriv wrt Gyro Bias ( -I * dt )
    F.block<3, 3>(6, 9) = -Eigen::Matrix3f::Identity() * dt_;

    // Q Matrix
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> Q;
    Q.setZero();
    Q.diagonal() = Q_diag_ * dt_; // Integrate noise over dt

    // P = F * P * F^T + Q
    P_ = F * P_ * F.transpose() + Q;
    
    enforce_symmetry();
}

void ESKF::update_zupt(float prob) {
    // Measurement: Zero Velocity
    // H matrix selects Velocity Error (3x15)
    Eigen::Matrix<float, 3, STATE_DIM> H;
    H.setZero();
    H.block<3, 3>(0, 3).setIdentity();

    // Measurement Noise R
    Eigen::Matrix3f R;
    R.setIdentity();

    if (prob >= 0.0f) {
        // Scale R based on probability (matches Python _calculate_zupt_update logic)
        // R = R_min * prob + R_max * (1 - prob)
        // This logic in Python seems inverted or handled via linear interpolation.
        // Actually Python says: R_zupt_scaled_diag = min_R_val * prob + max_R_val * (1 - prob)
        // Wait, if prob is high (near 1.0, confident ZUPT), R should be small (min_R).
        // If prob is low (near 0.0, moving), R should be huge (max_R).
        // Let's check Python code again carefully.
        
        // Python:
        // clamped_prob = torch.clamp(tcn_zupt_prob, 0.01, 0.99)
        // R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)
        
        // If clamped_prob is 0.99 (confident): R = min * 0.99 + max * 0.01. This is WRONG.
        // If prob is high, we want MIN noise.
        // The Python code actually says:
        // R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)
        // This effectively makes R large when prob is high?
        // Let's re-read Python code carefully.
        
        /*
            # Let R_min be self.zupt_noise_std**2, and R_max be a large value for uncertainty
            min_R_val = self.zupt_noise_std**2
            # A large value for R when ZUPT prob is low (e.g., 100 times min_R_val)
            max_R_val = min_R_val * 100

            # Clamp probability to avoid extreme values and numerical instability
            # Epsilon ensures we don't divide by zero or have extremely small R
            clamped_prob = torch.clamp(tcn_zupt_prob, 0.01, 0.99)

            # R_zupt_scaled_diag = min_R_val * clamped_prob + max_R_val * (1 - clamped_prob)
        */
        
        // If prob = 0.99 (Static): R = min * 0.99 + max * 0.01.
        // If prob = 0.01 (Moving): R = min * 0.01 + max * 0.99.
        // R is dominated by max_R when Moving. (Correct)
        // R is dominated by min_R when Static? No.
        // If max_R = 100, min_R = 1.
        // Prob 0.99: R = 1*0.99 + 100*0.01 = 0.99 + 1 = 1.99. (Close to min)
        // Prob 0.01: R = 1*0.01 + 100*0.99 = 0.01 + 99 = 99.01. (Close to max)
        // So the interpolation logic effectively works, but it's mixing them.
        // It's a linear blend.
        // Okay, I will implement exactly this.
        
        float min_R_val = zupt_noise_std_ * zupt_noise_std_;
        float max_R_val = min_R_val * 100.0f;
        float clamped_prob = std::max(0.01f, std::min(prob, 0.99f));
        
        float r_val = min_R_val * clamped_prob + max_R_val * (1.0f - clamped_prob);
        R.diagonal().setConstant(r_val);
    } else {
        // Standard fixed ZUPT noise
        R *= (zupt_noise_std_ * zupt_noise_std_);
    }

    // Innovation (0 - vel_pred)
    Eigen::Vector3f innovation = -state_.vel;

    // Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
    Eigen::Matrix<float, STATE_DIM, 3> PHt = P_ * H.transpose();
    Eigen::Matrix3f S = H * PHt + R;
    Eigen::Matrix<float, STATE_DIM, 3> K = PHt * S.inverse();

    // Delta X
    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;

    // Covariance Update P = (I - KH)P
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
    P_ = (I - K * H) * P_; // Simple form. Joseph form is better but this is cheaper.
    
    enforce_symmetry();
    inject_error(delta_x);
}

void ESKF::update_tcn_vel(const Eigen::Vector3f& vel_corr_body, const Eigen::Matrix<float, 6, 1>& R_params) {
    // Treat TCN velocity correction (body frame) as a direct measurement of velocity error.
    
    // Rotate correction to world frame
    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();
    Eigen::Vector3f vel_corr_w = R_bw * vel_corr_body;

    // H matrix (3x15) selects Velocity Error
    Eigen::Matrix<float, 3, STATE_DIM> H;
    H.setZero();
    H.block<3, 3>(0, 3).setIdentity();

    // R Matrix (Measurement Noise)
    // Use the params if needed, or fixed. Python uses fixed 1e-4 diag.
    Eigen::Matrix3f R;
    R.setIdentity();
    R *= (tcn_vel_noise_std_ * tcn_vel_noise_std_);

    // Innovation is the correction itself
    Eigen::Vector3f innovation = vel_corr_w;

    // Kalman Gain
    Eigen::Matrix<float, STATE_DIM, 3> PHt = P_ * H.transpose();
    Eigen::Matrix3f S = H * PHt + R;
    Eigen::Matrix<float, STATE_DIM, 3> K = PHt * S.inverse();

    // Delta X
    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;

    // Update P
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
    P_ = (I - K * H) * P_;

    enforce_symmetry();
    inject_error(delta_x);
}

Eigen::Matrix<float, 6, 1> ESKF::update_imu(
    const Eigen::Vector3f& accel_raw,
    const Eigen::Vector3f& gyro_raw,
    const Eigen::Matrix<float, 6, 1>& R_diag,
    float* out_mahalanobis
) {
    // 1. Build H Matrix (6x15)
    Eigen::Matrix<float, 6, STATE_DIM> H;
    H.setZero();

    // Gravity in Body Frame
    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();
    Eigen::Matrix3f R_wb = R_bw.transpose();
    Eigen::Vector3f g_body = R_wb * gravity_w_;

    // Skew Symmetric of g_body
    Eigen::Matrix3f g_skew;
    g_skew << 0, -g_body.z(), g_body.y(),
              g_body.z(), 0, -g_body.x(),
              -g_body.y(), g_body.x(), 0;

    // H structure:
    // Accel part (rows 0-2):
    // d(accel)/d(theta) = [g_b]_x
    H.block<3, 3>(0, 6) = g_skew;
    // d(accel)/d(b_a) = I
    H.block<3, 3>(0, 12).setIdentity();

    // Gyro part (rows 3-5):
    // d(gyro)/d(b_g) = I
    H.block<3, 3>(3, 9).setIdentity();

    // 2. Predicted Measurement
    // Accel: g_b + b_a
    Eigen::Vector3f accel_pred = g_body + state_.accel_bias;
    // Gyro: b_g
    Eigen::Vector3f gyro_pred = state_.gyro_bias;

    // 3. Innovation
    Eigen::Matrix<float, 6, 1> innovation;
    innovation.segment<3>(0) = accel_raw - accel_pred;
    innovation.segment<3>(3) = gyro_raw - gyro_pred;

    // 4. Kalman Gain
    Eigen::Matrix<float, 6, 6> R;
    R.setZero();
    R.diagonal() = R_diag;
    // Add small epsilon
    R.diagonal().array() += 1e-6f;

    Eigen::Matrix<float, STATE_DIM, 6> PHt = P_ * H.transpose();
    Eigen::Matrix<float, 6, 6> S = H * PHt + R;
    
    // Calculate Mahalanobis Distance for Gating
    // d^2 = y^T * S^-1 * y
    // Use LDLT for solving S * x = y
    Eigen::Matrix<float, 6, 1> S_inv_y = S.ldlt().solve(innovation);
    float mahalanobis_sq = innovation.dot(S_inv_y);

    if (out_mahalanobis) {
        *out_mahalanobis = mahalanobis_sq;
    }

    if (mahalanobis_sq > mahalanobis_gate_threshold_) {
        // Gating: Reject update
        // We do NOT update State or P
        // But we return the innovation for logging/TCN
        return innovation; 
    }
    
    // Continue with Update if passed gating
    
    // S inverse (6x6) - we can reuse decomposition or just invert
    Eigen::Matrix<float, STATE_DIM, 6> K = PHt * S.inverse();

    // 5. Update State
    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;
    
    // 6. Update Covariance
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
    P_ = (I - K * H) * P_;

    enforce_symmetry();
    inject_error(delta_x);

    return innovation;
}

bool ESKF::check_zupt(const Eigen::Vector3f& accel_raw) {
    // Simple magnitude check
    // Gravity is ~9.81.
    float norm = accel_raw.norm();
    // Threshold ~0.3 m/s^2 deviation
    return std::abs(norm - 9.81f) < 0.3f; 
}

void ESKF::inject_error(const Eigen::Matrix<float, STATE_DIM, 1>& delta_x) {
    // Split
    Eigen::Vector3f d_pos = delta_x.segment<3>(0);
    Eigen::Vector3f d_vel = delta_x.segment<3>(3);
    Eigen::Vector3f d_theta = delta_x.segment<3>(6);
    Eigen::Vector3f d_bg = delta_x.segment<3>(9);
    Eigen::Vector3f d_ba = delta_x.segment<3>(12);

    // Apply
    state_.pos += d_pos;
    state_.vel += d_vel;
    state_.gyro_bias += d_bg;
    state_.accel_bias += d_ba;

    // Orientation correction
    // q_new = q_old * q_delta
    // q_delta from small angle d_theta
    // q_delta = [1, 0.5*dx, 0.5*dy, 0.5*dz]
    Eigen::Quaternionf q_delta;
    q_delta.w() = 1.0f;
    q_delta.vec() = 0.5f * d_theta;
    q_delta.normalize();

    state_.quat = state_.quat * q_delta;
    state_.quat.normalize();
}

void ESKF::enforce_symmetry() {
    P_ = 0.5f * (P_ + P_.transpose());
}

} // namespace trajecto
