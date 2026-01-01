#include "eskf.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace trajecto {

ESKF::ESKF(float dt) : dt_(dt) {
    P_.setIdentity();
    P_ *= 0.1f;

    // Process noise from Allan Variance analysis (2025-12-29)
    float vrw = 8.1255e-4f;
    float arw = 7.5427e-5f;
    float accel_bi = 2.9840e-4f;
    float gyro_bi = 1.8947e-5f;

    Q_diag_.setZero();
    Q_diag_.segment<3>(3).setConstant(vrw * vrw);
    Q_diag_.segment<3>(6).setConstant(arw * arw);
    Q_diag_.segment<3>(9).setConstant(gyro_bi * gyro_bi);
    Q_diag_.segment<3>(12).setConstant(accel_bi * accel_bi);

    zupt_noise_std_ = 0.01f;
    tcn_vel_noise_std_ = 0.05f;

    gravity_w_ << 0.0f, 0.0f, 9.81f;

    mahalanobis_gate_threshold_ = 20.0f;
}

void ESKF::initialize(const Eigen::Vector3f& accel_init) {
    Eigen::Vector3f accel_norm = accel_init.normalized();
    Eigen::Vector3f target_up = Eigen::Vector3f::UnitZ();

    Eigen::Quaternionf q_init = Eigen::Quaternionf::FromTwoVectors(accel_norm, target_up);

    state_.quat = q_init;
    state_.pos.setZero();
    state_.vel.setZero();
    state_.gyro_bias.setZero();
    state_.accel_bias.setZero();

    P_.setIdentity();
    P_ *= 0.1f;
}

void ESKF::predict(const Eigen::Vector3f& gyro_raw, const Eigen::Vector3f& accel_raw) {
    Eigen::Vector3f gyro_corrected = gyro_raw - state_.gyro_bias;
    Eigen::Vector3f accel_corrected = accel_raw - state_.accel_bias;

    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();

    Eigen::Vector3f accel_w = R_bw * accel_corrected - gravity_w_;

    state_.pos += state_.vel * dt_ + 0.5f * accel_w * dt_ * dt_;

    state_.vel += accel_w * dt_;

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

    Eigen::Matrix<float, STATE_DIM, STATE_DIM> F;
    F.setIdentity();

    F.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dt_;

    Eigen::Matrix3f accel_skew;
    accel_skew << 0, -accel_corrected.z(), accel_corrected.y(),
                  accel_corrected.z(), 0, -accel_corrected.x(),
                  -accel_corrected.y(), accel_corrected.x(), 0;
    F.block<3, 3>(3, 6) = -R_bw * accel_skew * dt_;

    F.block<3, 3>(3, 12) = -R_bw * dt_;

    F.block<3, 3>(6, 9) = -Eigen::Matrix3f::Identity() * dt_;

    Eigen::Matrix<float, STATE_DIM, STATE_DIM> Q;
    Q.setZero();
    Q.diagonal() = Q_diag_ * dt_;

    P_ = F * P_ * F.transpose() + Q;
    
    enforce_symmetry();
}

void ESKF::update_zupt(float prob) {
    Eigen::Matrix<float, 3, STATE_DIM> H;
    H.setZero();
    H.block<3, 3>(0, 3).setIdentity();

    Eigen::Matrix3f R;
    R.setIdentity();

    if (prob >= 0.0f) {
        float min_R_val = zupt_noise_std_ * zupt_noise_std_;
        float max_R_val = min_R_val * 100.0f;
        float clamped_prob = std::max(0.01f, std::min(prob, 0.99f));
        
        float r_val = min_R_val * clamped_prob + max_R_val * (1.0f - clamped_prob);
        R.diagonal().setConstant(r_val);
    } else {
        R *= (zupt_noise_std_ * zupt_noise_std_);
    }

    Eigen::Vector3f innovation = -state_.vel;

    Eigen::Matrix<float, STATE_DIM, 3> PHt = P_ * H.transpose();
    Eigen::Matrix3f S = H * PHt + R;
    Eigen::Matrix<float, STATE_DIM, 3> K = PHt * S.inverse();

    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;

    Eigen::Matrix<float, STATE_DIM, STATE_DIM> I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
    P_ = (I - K * H) * P_;
    
    enforce_symmetry();
    inject_error(delta_x);
}

void ESKF::update_tcn_vel(const Eigen::Vector3f& vel_corr_body, const Eigen::Matrix<float, 6, 1>& R_params) {
    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();
    Eigen::Vector3f vel_corr_w = R_bw * vel_corr_body;

    Eigen::Matrix<float, 3, STATE_DIM> H;
    H.setZero();
    H.block<3, 3>(0, 3).setIdentity();

    Eigen::Matrix3f R;
    R.setIdentity();
    R *= (tcn_vel_noise_std_ * tcn_vel_noise_std_);

    Eigen::Vector3f innovation = vel_corr_w;

    Eigen::Matrix<float, STATE_DIM, 3> PHt = P_ * H.transpose();
    Eigen::Matrix3f S = H * PHt + R;
    Eigen::Matrix<float, STATE_DIM, 3> K = PHt * S.inverse();

    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;

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
    Eigen::Matrix<float, 6, STATE_DIM> H;
    H.setZero();

    Eigen::Matrix3f R_bw = state_.quat.toRotationMatrix();
    Eigen::Matrix3f R_wb = R_bw.transpose();
    Eigen::Vector3f g_body = R_wb * gravity_w_;

    Eigen::Matrix3f g_skew;
    g_skew << 0, -g_body.z(), g_body.y(),
              g_body.z(), 0, -g_body.x(),
              -g_body.y(), g_body.x(), 0;

    H.block<3, 3>(0, 6) = g_skew;
    H.block<3, 3>(0, 12).setIdentity();

    H.block<3, 3>(3, 9).setIdentity();

    Eigen::Vector3f accel_pred = g_body + state_.accel_bias;
    Eigen::Vector3f gyro_pred = state_.gyro_bias;

    Eigen::Matrix<float, 6, 1> innovation;
    innovation.segment<3>(0) = accel_raw - accel_pred;
    innovation.segment<3>(3) = gyro_raw - gyro_pred;

    Eigen::Matrix<float, 6, 6> R;
    R.setZero();
    R.diagonal() = R_diag;
    R.diagonal().array() += 1e-6f;

    Eigen::Matrix<float, STATE_DIM, 6> PHt = P_ * H.transpose();
    Eigen::Matrix<float, 6, 6> S = H * PHt + R;

    Eigen::Matrix<float, 6, 1> S_inv_y = S.ldlt().solve(innovation);
    float mahalanobis_sq = innovation.dot(S_inv_y);

    if (out_mahalanobis) {
        *out_mahalanobis = mahalanobis_sq;
    }

    if (mahalanobis_sq > mahalanobis_gate_threshold_) {
        return innovation;
    }

    Eigen::Matrix<float, STATE_DIM, 6> K = PHt * S.inverse();

    Eigen::Matrix<float, STATE_DIM, 1> delta_x = K * innovation;

    Eigen::Matrix<float, STATE_DIM, STATE_DIM> I = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
    P_ = (I - K * H) * P_;

    enforce_symmetry();
    inject_error(delta_x);

    return innovation;
}

bool ESKF::check_zupt(const Eigen::Vector3f& accel_raw) {
    float norm = accel_raw.norm();
    return std::abs(norm - 9.81f) < 0.3f; 
}

void ESKF::inject_error(const Eigen::Matrix<float, STATE_DIM, 1>& delta_x) {
    Eigen::Vector3f d_pos = delta_x.segment<3>(0);
    Eigen::Vector3f d_vel = delta_x.segment<3>(3);
    Eigen::Vector3f d_theta = delta_x.segment<3>(6);
    Eigen::Vector3f d_bg = delta_x.segment<3>(9);
    Eigen::Vector3f d_ba = delta_x.segment<3>(12);

    state_.pos += d_pos;
    state_.vel += d_vel;
    state_.gyro_bias += d_bg;
    state_.accel_bias += d_ba;

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
