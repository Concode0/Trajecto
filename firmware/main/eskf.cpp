#include "eskf.hpp"
#include <iostream>

namespace trajecto {

ESKF::ESKF(float dt) : dt_(dt) {
    // Initialize Covariance P with small uncertainty
    P_.setIdentity();
    P_ *= 0.1f;

    // Initialize Process Noise Q (Diagonal)
    // Values derived from Config.py (approximate)
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

    // Ori deriv wrt Ori (Identity for small angles in body frame usually, but here standard approx)
    // Using simple error state dynamics: delta_theta_new = delta_theta - R_bw * gyro_bias * dt?
    // Actually: delta_theta_dot = -[w]_x * delta_theta - delta_bias_g
    // So F_theta_theta is I - [w]_x * dt. But often approximated as I if w is small or incorporated.
    // Standard ESKF: F_theta_theta = R{w*dt}^T.
    // Let's stick to the Python implementation:
    // F_error_matrix[:, 6:9, 9:12] = -torch.eye(3) * self.dt
    // It seems Python neglected the [w]_x term or assumed it's small/handled in integration.
    F.block<3, 3>(6, 9) = -Eigen::Matrix3f::Identity() * dt_;

    // Q Matrix
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> Q;
    Q.setZero();
    Q.diagonal() = Q_diag_ * dt_; // Integrate noise over dt

    // P = F * P * F^T + Q
    P_ = F * P_ * F.transpose() + Q;
    
    enforce_symmetry();
}

void ESKF::update_zupt() {
    // Measurement: Zero Velocity
    // H matrix selects Velocity Error (3x15)
    Eigen::Matrix<float, 3, STATE_DIM> H;
    H.setZero();
    H.block<3, 3>(0, 3).setIdentity();

    // Measurement Noise R
    Eigen::Matrix3f R;
    R.setIdentity();
    R *= (zupt_noise_std_ * zupt_noise_std_);

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
    // TCN predicts velocity correction in BODY frame.
    // We treat this as a measurement of velocity error? 
    // Or as a measurement of velocity: z = v_body_pred + vel_corr_body?
    
    // In Python `_apply_tcn_velocity_correction`:
    // It treats "vel_corr_w" as the innovation directly.
    // Wait, the TCN output is `vel_corr` (body frame).
    // The Python code rotates it to World: `vel_corr_w = R * vel_corr_b`.
    // Then `innovation = vel_corr_w`.
    // This implies the measurement `z` was "Velocity should be adjusted by X".
    // Effectively, it's treating the TCN output as an observation of the Velocity Error itself.
    // Measurement model: z = delta_v + noise.
    // So H maps state to delta_v. H for delta_v is Identity on indices 3-5.
    
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
    const Eigen::Matrix<float, 6, 1>& R_diag
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
    
    // S inverse (6x6)
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
