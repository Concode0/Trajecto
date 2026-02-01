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

#include "eskf_fixed.hpp"
#include "model_params.hpp"
#include <cmath>
#include <algorithm>

namespace trajecto {

using namespace fp;

// ---------------------------------------------------------------------------
// NominalStateFixed -> NominalState conversion
// ---------------------------------------------------------------------------
NominalState NominalStateFixed::to_float() const {
    NominalState s;
    s.pos << pos.x().to_float(), pos.y().to_float(), pos.z().to_float();
    s.vel << vel.x().to_float(), vel.y().to_float(), vel.z().to_float();
    s.quat.w() = quat.w.to_float();
    s.quat.x() = quat.x.to_float();
    s.quat.y() = quat.y.to_float();
    s.quat.z() = quat.z.to_float();
    s.gyro_bias << gyro_bias.x().to_float(), gyro_bias.y().to_float(), gyro_bias.z().to_float();
    s.accel_bias << accel_bias.x().to_float(), accel_bias.y().to_float(), accel_bias.z().to_float();
    return s;
}

// ---------------------------------------------------------------------------
// Helper: float[3] -> Q16.15
// ---------------------------------------------------------------------------
Vec3_Q16_15 ESKFFixed::float3_to_q16_15(const float v[3]) {
    return {
        Q16_15::from_float(v[0]),
        Q16_15::from_float(v[1]),
        Q16_15::from_float(v[2])
    };
}

Vec3_Q4_27 ESKFFixed::float3_to_q4_27(const float v[3]) {
    return {
        Q4_27::from_float(v[0]),
        Q4_27::from_float(v[1]),
        Q4_27::from_float(v[2])
    };
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ESKFFixed::ESKFFixed(float dt)
    : dt_(Q4_27::from_float(dt))
    , half_dt2_(Q4_27::from_float(0.5f * dt * dt))
{
    // Initialize P as scaled identity (0.1 on diagonal)
    P_.set_scaled_identity(0.1f);

    // Gravity world frame: (0, 0, -9.80665)
    gravity_w_ = GRAVITY_W_Q;

    // Process noise Q diagonal blocks (variance * dt)
    // Block 0: position (zero process noise)
    Q_blocks_[0] = ScaledBlock3x3();

    // Block 1: velocity (VRW^2 * dt)
    Q_blocks_[1] = ScaledBlock3x3::from_float_diag(
        VRW_X * VRW_X * DT,
        VRW_Y * VRW_Y * DT,
        VRW_Z * VRW_Z * DT
    );

    // Block 2: orientation (ARW^2 * dt)
    Q_blocks_[2] = ScaledBlock3x3::from_float_diag(
        ARW_X * ARW_X * DT,
        ARW_Y * ARW_Y * DT,
        ARW_Z * ARW_Z * DT
    );

    // Block 3: gyro bias drift (BI^2 * dt)
    Q_blocks_[3] = ScaledBlock3x3::from_float_diag(
        GYRO_BI_X * GYRO_BI_X * DT,
        GYRO_BI_Y * GYRO_BI_Y * DT,
        GYRO_BI_Z * GYRO_BI_Z * DT
    );

    // Block 4: accel bias drift (BI^2 * dt)
    Q_blocks_[4] = ScaledBlock3x3::from_float_diag(
        ACCEL_BI_X * ACCEL_BI_X * DT,
        ACCEL_BI_Y * ACCEL_BI_Y * DT,
        ACCEL_BI_Z * ACCEL_BI_Z * DT
    );
}

// ---------------------------------------------------------------------------
// Initialize via gravity alignment
// ---------------------------------------------------------------------------
void ESKFFixed::initialize(const float accel_init[3]) {
    // Normalize accel vector
    float ax = accel_init[0], ay = accel_init[1], az = accel_init[2];
    float norm = std::sqrt(ax * ax + ay * ay + az * az);
    if (norm < 1e-6f) norm = 1.0f;
    float inv_norm = 1.0f / norm;
    float anx = -ax * inv_norm; // negate: accel measures -gravity
    float any = -ay * inv_norm;
    float anz = -az * inv_norm;

    // Target: gravity direction = (0, 0, -1)
    float gx = 0.0f, gy = 0.0f, gz = -1.0f;

    // Quaternion from two vectors (Rodrigues)
    // axis = cross(from, to), angle via dot product
    float cx = any * gz - anz * gy;
    float cy = anz * gx - anx * gz;
    float cz = anx * gy - any * gx;
    float d = anx * gx + any * gy + anz * gz; // dot product

    // q = [1 + d, cross] normalized (shortcut for FromTwoVectors)
    float qw = 1.0f + d;
    float qnorm = std::sqrt(qw * qw + cx * cx + cy * cy + cz * cz);
    if (qnorm < 1e-8f) {
        // Vectors are opposite — use 180-degree rotation about any perpendicular axis
        state_.quat = QuatQ::identity();
    } else {
        float inv_qn = 1.0f / qnorm;
        state_.quat.w = Q1_30::from_float(qw * inv_qn);
        state_.quat.x = Q1_30::from_float(cx * inv_qn);
        state_.quat.y = Q1_30::from_float(cy * inv_qn);
        state_.quat.z = Q1_30::from_float(cz * inv_qn);
    }

    state_.pos = Vec3_Q4_27::zero();
    state_.vel = Vec3_Q4_27::zero();
    state_.gyro_bias = Vec3_Q1_30::zero();
    state_.accel_bias = Vec3_Q4_27::zero();

    P_.set_scaled_identity(0.1f);
}

// ---------------------------------------------------------------------------
// Predict: propagate nominal state and error covariance
// ---------------------------------------------------------------------------
void ESKFFixed::predict(const float gyro_raw[3], const float accel_raw[3]) {
    // Convert raw IMU to fixed-point Q16.15
    Vec3_Q16_15 gyro_q = float3_to_q16_15(gyro_raw);
    Vec3_Q16_15 accel_q = float3_to_q16_15(accel_raw);

    // Bias correction
    // gyro_corrected = gyro_raw - gyro_bias
    // gyro_bias is Q2.30, gyro_raw is Q17.15. Convert bias to Q17.15.
    Vec3_Q16_15 gyro_bias_q15 = vec3_convert<17, 15>(state_.gyro_bias);
    Vec3_Q16_15 gyro_corr = vec3_sub(gyro_q, gyro_bias_q15);

    // accel_corrected = accel_raw - accel_bias
    // accel_bias is Q5.27, accel_raw is Q17.15. Convert bias to Q17.15.
    Vec3_Q16_15 accel_bias_q15 = vec3_convert<17, 15>(state_.accel_bias);
    Vec3_Q16_15 accel_corr = vec3_sub(accel_q, accel_bias_q15);

    // Rotation matrix from quaternion (body->world)
    Mat3Q R_bw = quat_to_rotmat(state_.quat);

    // World-frame acceleration: R_bw * accel_corrected - gravity_w
    // R_bw is Q2.30, accel_corr is Q17.15 -> result in Q5.27
    Vec3_Q4_27 accel_w = mat3_vec3_mul<5, 27>(R_bw, accel_corr);
    accel_w = vec3_sub(accel_w, gravity_w_);

    // State propagation
    // pos += vel * dt + 0.5 * accel_w * dt^2
    // vel * dt: Q5.27 * Q5.27 -> Q5.27
    Vec3_Q4_27 vel_dt = {
        mul(state_.vel.x(), dt_),
        mul(state_.vel.y(), dt_),
        mul(state_.vel.z(), dt_)
    };
    Vec3_Q4_27 accel_dt2 = {
        mul(accel_w.x(), half_dt2_),
        mul(accel_w.y(), half_dt2_),
        mul(accel_w.z(), half_dt2_)
    };
    state_.pos = vec3_add(state_.pos, vec3_add(vel_dt, accel_dt2));

    // vel += accel_w * dt
    Vec3_Q4_27 accel_dt = {
        mul(accel_w.x(), dt_),
        mul(accel_w.y(), dt_),
        mul(accel_w.z(), dt_)
    };
    state_.vel = vec3_add(state_.vel, accel_dt);

    // Quaternion integration
    // omega = gyro_corrected * dt (small angle in radians)
    // Convert gyro_corr (Q17.15) * dt (Q5.27) -> Q5.27 -> then to Q2.30 for angle
    Vec3_Q4_27 omega_q27 = {
        mul_cross<5, 27>(gyro_corr.x(), dt_),
        mul_cross<5, 27>(gyro_corr.y(), dt_),
        mul_cross<5, 27>(gyro_corr.z(), dt_)
    };
    Vec3_Q1_30 omega = vec3_convert<2, 30>(omega_q27);

    // Angle = ||omega|| via dot product + sqrt
    Q1_30 angle_sq = vec3_dot(omega, omega);
    // For small angles (typical), use first-order approximation:
    // q_delta = [cos(angle/2), sin(angle/2) * axis]
    // ~= [1 - angle^2/8, omega/2] for small angles
    // Since dt is small (~0.02s) and gyro is moderate, angle is typically < 0.01 rad

    if (angle_sq.raw > 100) { // angle^2 > ~1e-7 rad^2 (non-trivial rotation)
        // angle = sqrt(angle_sq) in Q1.30 (pure integer Newton-Raphson)
        int32_t angle_raw = isqrt_fixed(angle_sq.raw, 30);
        int32_t half_angle_raw = angle_raw >> 1; // angle/2 in Q1.30

        int32_t sin_ha, cos_ha;
        cordic_sincos(half_angle_raw, sin_ha, cos_ha);

        // axis = omega / angle (unit vector in Q1.30)
        // axis[i] = (omega[i].raw << 30) / angle_raw via int64 division
        Q1_30 ax, ay, az;
        if (angle_raw > 0) {
            ax = Q1_30(static_cast<int32_t>((static_cast<int64_t>(omega.x().raw) << 30) / angle_raw));
            ay = Q1_30(static_cast<int32_t>((static_cast<int64_t>(omega.y().raw) << 30) / angle_raw));
            az = Q1_30(static_cast<int32_t>((static_cast<int64_t>(omega.z().raw) << 30) / angle_raw));
        } else {
            ax = ay = az = Q1_30(0);
        }

        QuatQ q_delta;
        q_delta.w = Q1_30(cos_ha);
        q_delta.x = mul(Q1_30(sin_ha), ax);
        q_delta.y = mul(Q1_30(sin_ha), ay);
        q_delta.z = mul(Q1_30(sin_ha), az);

        state_.quat = quat_mul(state_.quat, q_delta);
    } else if (angle_sq.raw > 0) {
        // Small angle approximation: q_delta = [1, omega/2]
        QuatQ q_delta;
        q_delta.w = Q1_30::one();
        // omega/2 in Q2.30: shift right by 1
        q_delta.x = Q1_30(omega.x().raw >> 1);
        q_delta.y = Q1_30(omega.y().raw >> 1);
        q_delta.z = Q1_30(omega.z().raw >> 1);

        state_.quat = quat_mul(state_.quat, q_delta);
    }
    // else: no rotation (angle = 0)

    state_.quat = quat_normalize(state_.quat);

    // --- Covariance propagation: P = F * P * F^T + Q ---
    // F is sparse identity + off-diagonal blocks:
    //   F01 = I*dt        (pos <- vel)
    //   F12 = -R*[a]x*dt  (vel <- orient)
    //   F14 = -R*dt       (vel <- accel_bias)
    //   F23 = -I*dt       (orient <- gyro_bias)
    //
    // Exploit: (I + F_off) * P * (I + F_off)^T
    // = P + F_off*P + P*F_off^T + F_off*P*F_off^T + Q
    // For first-order approximation (dt is small, F_off*P*F_off^T ~ dt^2, negligible):
    // P' ~= P + F_off*P + (F_off*P)^T + Q
    //
    // But for accuracy, let's do the full sparse multiplication.
    // P' = F * P * F^T + Q where F = I + dF
    //
    // Computing F*P block by block:
    // (F*P)[i][j] = sum_k F[i][k] * P[k][j]
    // Since F is mostly identity, (F*P)[i][j] = P[i][j] + sum_{k!=i} F[i][k]*P[k][j]
    // Non-zero off-diag F blocks: F01, F12, F14, F23

    // Precompute F blocks as ScaledBlock3x3 (pure integer)
    {
        // F01 = I * dt (diagonal)
        ScaledBlock3x3 F01 = ScaledBlock3x3::diagonal(DT_SCALAR, DT_SCALAR, DT_SCALAR, DT_SCALAR_EXP);
        block_renormalize(F01);

        // F12 = -R_bw * [accel_corr]x * dt (pure integer)
        ScaledBlock3x3 R_block = mat3q_to_block(R_bw);
        ScaledBlock3x3 skew_block = vec3q15_to_skew_block(accel_corr);
        ScaledBlock3x3 R_skew = block_mul(R_block, skew_block);
        ScaledBlock3x3 F12 = block_scale_int(R_skew, NEG_DT_SCALAR, DT_SCALAR_EXP);

        // F14 = -R_bw * dt (pure integer)
        ScaledBlock3x3 F14 = block_scale_int(R_block, NEG_DT_SCALAR, DT_SCALAR_EXP);

        // F23 = -I * dt (diagonal, pure integer)
        ScaledBlock3x3 F23 = ScaledBlock3x3::diagonal(NEG_DT_SCALAR, NEG_DT_SCALAR, NEG_DT_SCALAR, DT_SCALAR_EXP);
        block_renormalize(F23);

        // --- Compute P' = F * P * F^T + Q using sparse structure ---
        // FP[i][j] = P[i][j] + F_off[i][k] * P[k][j] for non-zero off-diag F blocks
        // Then P' = FP * F^T -> P'[i][j] = FP[i][j] + FP[i][k] * F_off[j][k]^T

        // Step 1: Compute FP = F * P
        // FP[0][j] = P[0][j] + F01 * P[1][j]    for j=0..4
        // FP[1][j] = P[1][j] + F12 * P[2][j] + F14 * P[4][j]
        // FP[2][j] = P[2][j] + F23 * P[3][j]
        // FP[3][j] = P[3][j]  (no off-diagonal F in row 3)
        // FP[4][j] = P[4][j]  (no off-diagonal F in row 4)

        BlockCovMatrix FP;
        for (int j = 0; j < 5; ++j) {
            FP.at(0, j) = block_add(P_.at(0, j), block_mul(F01, P_.at(1, j)));
            FP.at(1, j) = block_add(block_add(P_.at(1, j), block_mul(F12, P_.at(2, j))),
                                     block_mul(F14, P_.at(4, j)));
            FP.at(2, j) = block_add(P_.at(2, j), block_mul(F23, P_.at(3, j)));
            FP.at(3, j) = P_.at(3, j);
            FP.at(4, j) = P_.at(4, j);
        }

        // Step 2: P' = FP * F^T
        // P'[i][0] = FP[i][0] + FP[i][1] * F01^T
        // P'[i][1] = FP[i][1] + FP[i][2] * F12^T + FP[i][4] * F14^T
        // P'[i][2] = FP[i][2] + FP[i][3] * F23^T
        // P'[i][3] = FP[i][3]
        // P'[i][4] = FP[i][4]

        ScaledBlock3x3 F01T = block_transpose(F01);
        ScaledBlock3x3 F12T = block_transpose(F12);
        ScaledBlock3x3 F14T = block_transpose(F14);
        ScaledBlock3x3 F23T = block_transpose(F23);

        for (int i = 0; i < 5; ++i) {
            P_.at(i, 0) = block_add(FP.at(i, 0), block_mul(FP.at(i, 1), F01T));
            P_.at(i, 1) = block_add(block_add(FP.at(i, 1), block_mul(FP.at(i, 2), F12T)),
                                     block_mul(FP.at(i, 4), F14T));
            P_.at(i, 2) = block_add(FP.at(i, 2), block_mul(FP.at(i, 3), F23T));
            P_.at(i, 3) = FP.at(i, 3);
            P_.at(i, 4) = FP.at(i, 4);
        }

        // Step 3: Add Q (diagonal only)
        for (int i = 0; i < 5; ++i) {
            P_.at(i, i) = block_add(P_.at(i, i), Q_blocks_[i]);
        }
    }

    enforce_symmetry();
}

// ---------------------------------------------------------------------------
// Stationary Update: ZUPT + ZARU with log-space soft-thresholding R
// ---------------------------------------------------------------------------
void ESKFFixed::update_stationary(const float gyro_raw[3], float prob) {
    // Log-space R computation (matches Python ESKF.py:_calculate_stationary_update)
    // Uses float boundary for R computation (only 2 float ops, negligible vs block algebra)
    float R_zupt_val, R_zaru_val;
    if (prob >= 0.0f) {
        float cp = prob < 1e-4f ? 1e-4f : (prob > 1.0f ? 1.0f : prob);
        float above_onset = (cp - ZUPT_DECAY_ONSET) / (1.0f - ZUPT_DECAY_ONSET + 1e-6f);
        above_onset = above_onset < 0.0f ? 0.0f : (above_onset > 1.0f ? 1.0f : above_onset);
        float alpha = above_onset * above_onset; // exponent = 2

        float log_R = (1.0f - alpha) * LOG_R_MAX + alpha * LOG_R_MIN;
        R_zupt_val = std::exp(log_R);
    } else {
        R_zupt_val = ZUPT_R_MIN_VAL; // tight constraint
    }
    R_zaru_val = R_zupt_val * ZARU_R_SCALE;

    ScaledBlock3x3 R_zupt = ScaledBlock3x3::from_float_diag(R_zupt_val, R_zupt_val, R_zupt_val);
    ScaledBlock3x3 R_zaru = ScaledBlock3x3::from_float_diag(R_zaru_val, R_zaru_val, R_zaru_val);

    // 6D observation: H = [0 I 0 0 0; 0 0 0 I 0]
    // Row block 0 selects velocity (block col 1)
    // Row block 1 selects gyro_bias (block col 3)

    // Innovation: [-vel (Q4.27), -gyro_bias (Q1.30→Q4.27)]
    Vec3_Q4_27 innov_vel = vec3_neg(state_.vel);
    // ZARU innovation: 0 - gyro_bias = -gyro_bias
    Vec3_Q4_27 innov_gyro = {
        Q4_27::from_float(-state_.gyro_bias.x().to_float()),
        Q4_27::from_float(-state_.gyro_bias.y().to_float()),
        Q4_27::from_float(-state_.gyro_bias.z().to_float())
    };

    // S (6x6 as Block6x6):
    // S = H * P * H^T + R
    // S[0][0] = P[1][1] + R_zupt    (vel-vel)
    // S[0][1] = P[1][3]              (vel-gyro_bias)
    // S[1][0] = P[3][1]              (gyro_bias-vel)
    // S[1][1] = P[3][3] + R_zaru    (gyro_bias-gyro_bias)
    Block6x6 S;
    S.blocks[0][0] = block_add(P_.at(1, 1), R_zupt);
    S.blocks[0][1] = P_.at(1, 3);
    S.blocks[1][0] = P_.at(3, 1);
    S.blocks[1][1] = block_add(P_.at(3, 3), R_zaru);

    Block6x6 Sinv;
    if (!block_inverse_6x6(S, Sinv)) return;

    // K[i] (5 blocks, each 3×6 = two 3×3 sub-blocks):
    // K[i][a] = P[i][1]*Sinv[0][a] + P[i][3]*Sinv[1][a]   for a=0,1
    ScaledBlock3x3 K[5][2];
    for (int i = 0; i < 5; ++i) {
        K[i][0] = block_add(block_mul(P_.at(i, 1), Sinv.blocks[0][0]),
                             block_mul(P_.at(i, 3), Sinv.blocks[1][0]));
        K[i][1] = block_add(block_mul(P_.at(i, 1), Sinv.blocks[0][1]),
                             block_mul(P_.at(i, 3), Sinv.blocks[1][1]));
    }

    // delta_x[i] = K[i][0]*innov_vel + K[i][1]*innov_gyro
    Vec3_Q4_27 d_pos = vec3_add(block_mul_vec3_q427(K[0][0], innov_vel),
                                 block_mul_vec3_q427(K[0][1], innov_gyro));
    Vec3_Q4_27 d_vel = vec3_add(block_mul_vec3_q427(K[1][0], innov_vel),
                                 block_mul_vec3_q427(K[1][1], innov_gyro));
    Vec3_Q1_30 d_theta = vec3_add(block_mul_vec3_q130(K[2][0], innov_vel),
                                   block_mul_vec3_q130(K[2][1], innov_gyro));
    Vec3_Q1_30 d_bg = vec3_add(block_mul_vec3_q130(K[3][0], innov_vel),
                                block_mul_vec3_q130(K[3][1], innov_gyro));
    Vec3_Q4_27 d_ba = vec3_add(block_mul_vec3_q427(K[4][0], innov_vel),
                                 block_mul_vec3_q427(K[4][1], innov_gyro));

    // P update: Joseph form for 6D
    // (I-KH)*P: P'_temp[i][j] = P[i][j] - K[i][0]*P[1][j] - K[i][1]*P[3][j]
    BlockCovMatrix ImKH_P;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ScaledBlock3x3 t0 = block_mul(K[i][0], P_.at(1, j));
            ScaledBlock3x3 t1 = block_mul(K[i][1], P_.at(3, j));
            ImKH_P.at(i, j) = block_sub(block_sub(P_.at(i, j), t0), t1);
        }
    }

    // Right multiply by (I-KH)^T:
    // P''[i][j] = ImKH_P[i][j] - ImKH_P[i][1]*K[j][0]^T - ImKH_P[i][3]*K[j][1]^T
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ScaledBlock3x3 t0 = block_mul(ImKH_P.at(i, 1), block_transpose(K[j][0]));
            ScaledBlock3x3 t1 = block_mul(ImKH_P.at(i, 3), block_transpose(K[j][1]));
            P_.at(i, j) = block_sub(block_sub(ImKH_P.at(i, j), t0), t1);
        }
    }

    // Add K*R*K^T:
    // KRKT[i][j] = K[i][0]*R_zupt*K[j][0]^T + K[i][1]*R_zaru*K[j][1]^T
    for (int i = 0; i < 5; ++i) {
        ScaledBlock3x3 KR0i = block_mul(K[i][0], R_zupt);
        ScaledBlock3x3 KR1i = block_mul(K[i][1], R_zaru);
        for (int j = 0; j < 5; ++j) {
            ScaledBlock3x3 t0 = block_mul(KR0i, block_transpose(K[j][0]));
            ScaledBlock3x3 t1 = block_mul(KR1i, block_transpose(K[j][1]));
            P_.at(i, j) = block_add(P_.at(i, j), block_add(t0, t1));
        }
    }

    enforce_symmetry();

    inject_error(d_pos, d_vel, d_theta, d_bg, d_ba);
}

// ---------------------------------------------------------------------------
// TCN Velocity Correction Update (Joseph form)
// ---------------------------------------------------------------------------
void ESKFFixed::update_tcn_vel(const float vel_corr_body[3], const float R_params_f[6]) {
    // Rotate vel correction body -> world
    Mat3Q R_bw = quat_to_rotmat(state_.quat);
    Vec3_Q4_27 vel_body = float3_to_q4_27(vel_corr_body);
    Vec3_Q4_27 vel_corr_w = mat3_vec3_mul<5, 27>(R_bw, vel_body);

    // H selects velocity block (same as ZUPT)
    // Innovation = vel_corr_w (the TCN correction IS the innovation, in Q4.27)

    // Build R from TCN parameters (first 3 of R_params)
    // Softplus stays float: only 3 values at 12.5Hz (TCN stride)
    float R_diag_f[3];
    for (int i = 0; i < 3; ++i) {
        float val = R_params_f[i];
        float sp = (val > 20.0f) ? val : std::log1p(std::exp(val));
        float r = sp + 1e-4f;
        r = std::max(1e-4f, std::min(3.0f, r));
        R_diag_f[i] = r;
    }
    ScaledBlock3x3 R = ScaledBlock3x3::from_float_diag(R_diag_f[0], R_diag_f[1], R_diag_f[2]);

    // S = P[1][1] + R
    ScaledBlock3x3 S = block_add(P_.at(1, 1), R);

    // S inverse
    ScaledBlock3x3 Sinv;
    if (!block_inverse_3x3(S, Sinv)) return;

    // K[i] = P[i][1] * Sinv
    ScaledBlock3x3 K[5];
    for (int i = 0; i < 5; ++i) {
        K[i] = block_mul(P_.at(i, 1), Sinv);
    }

    // delta_x = K * innovation (pure integer)
    // vel_corr_w is Vec3_Q4_27 — use directly as innovation
    Vec3_Q4_27 d_pos = block_mul_vec3_q427(K[0], vel_corr_w);
    Vec3_Q4_27 d_vel = block_mul_vec3_q427(K[1], vel_corr_w);
    Vec3_Q1_30 d_theta = block_mul_vec3_q130(K[2], vel_corr_w);
    Vec3_Q1_30 d_bg = block_mul_vec3_q130(K[3], vel_corr_w);
    Vec3_Q4_27 d_ba = block_mul_vec3_q427(K[4], vel_corr_w);

    // Joseph form: P = (I-KH)*P*(I-KH)^T + K*R*K^T
    // (I-KH)[i][j] = delta_ij - K[i]*delta_j1
    // ImKH * P: same as P[i][j] - K[i]*P[1][j]
    BlockCovMatrix ImKH_P;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ImKH_P.at(i, j) = block_sub(P_.at(i, j), block_mul(K[i], P_.at(1, j)));
        }
    }

    // ImKH_P * (I-KH)^T: P'[i][j] = ImKH_P[i][j] - ImKH_P[i][1]*K[j]^T
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ScaledBlock3x3 term = block_mul(ImKH_P.at(i, 1), block_transpose(K[j]));
            P_.at(i, j) = block_sub(ImKH_P.at(i, j), term);
        }
    }

    // Add K*R*K^T
    // KR[i] = K[i] * R
    // KRKT[i][j] = KR[i] * K[j]^T
    for (int i = 0; i < 5; ++i) {
        ScaledBlock3x3 KRi = block_mul(K[i], R);
        for (int j = 0; j < 5; ++j) {
            ScaledBlock3x3 KRiKjT = block_mul(KRi, block_transpose(K[j]));
            P_.at(i, j) = block_add(P_.at(i, j), KRiKjT);
        }
    }

    enforce_symmetry();

    inject_error(d_pos, d_vel, d_theta, d_bg, d_ba);
}

// ---------------------------------------------------------------------------
// IMU Update (6D observation: accel + gyro for bias estimation)
// ---------------------------------------------------------------------------
void ESKFFixed::update_imu(
    const float accel_raw[3],
    const float gyro_raw[3],
    const float R_diag_f[6],
    float out_innovation[6],
    float* out_mahalanobis
) {
    // Build H matrix (6x15 = 2x5 blocks of 3x3)
    // H = [0 0 g_skew 0 I; 0 0 0 I 0]
    // Row block 0 (accel): H[0][2] = [g_body]x, H[0][4] = I
    // Row block 1 (gyro):  H[1][3] = I

    Mat3Q R_bw = quat_to_rotmat(state_.quat);
    Mat3Q R_wb = R_bw.transpose();

    // g_body = R_wb * gravity_w (Q5.27)
    Vec3_Q4_27 g_body = mat3_vec3_mul<5, 27>(R_wb, gravity_w_);

    // Predicted measurements
    // accel_pred = g_body + accel_bias
    // gyro_pred = gyro_bias
    float accel_pred[3] = {
        g_body.x().to_float() + state_.accel_bias.x().to_float(),
        g_body.y().to_float() + state_.accel_bias.y().to_float(),
        g_body.z().to_float() + state_.accel_bias.z().to_float()
    };
    float gyro_pred[3] = {
        state_.gyro_bias.x().to_float(),
        state_.gyro_bias.y().to_float(),
        state_.gyro_bias.z().to_float()
    };

    // Innovation
    float innov[6];
    for (int i = 0; i < 3; ++i) innov[i] = accel_raw[i] - accel_pred[i];
    for (int i = 0; i < 3; ++i) innov[3 + i] = gyro_raw[i] - gyro_pred[i];

    if (out_innovation) {
        for (int i = 0; i < 6; ++i) out_innovation[i] = innov[i];
    }

    // Build H blocks as ScaledBlock3x3 from float
    float gb[3] = { g_body.x().to_float(), g_body.y().to_float(), g_body.z().to_float() };
    float g_skew_f[9] = {
        0, -gb[2], gb[1],
        gb[2], 0, -gb[0],
        -gb[1], gb[0], 0
    };

    // Convert g_skew to ScaledBlock3x3
    float max_gs = 0;
    for (int i = 0; i < 9; ++i) {
        float av = g_skew_f[i] < 0 ? -g_skew_f[i] : g_skew_f[i];
        if (av > max_gs) max_gs = av;
    }
    ScaledBlock3x3 H_02; // g_skew
    if (max_gs > 0) {
        int exp_bits = 0;
        float tmp = max_gs;
        while (tmp >= 2.0f) { tmp *= 0.5f; exp_bits++; }
        while (tmp < 1.0f) { tmp *= 2.0f; exp_bits--; }
        H_02.exponent = static_cast<int8_t>(exp_bits - 30);
        float scale = 1.0f;
        int8_t e = H_02.exponent;
        if (e > 0) for (int i = 0; i < e; ++i) scale *= 2.0f;
        else for (int i = 0; i < -e; ++i) scale *= 0.5f;
        float inv_s = 1.0f / scale;
        for (int i = 0; i < 9; ++i)
            H_02.m[i] = static_cast<int32_t>(g_skew_f[i] * inv_s);
    }

    // Compute S = H * P * H^T + R (6x6)
    // S is 2x2 blocks of 3x3
    // S[a][b] = sum_i sum_j H_row[a][i] * P[i][j] * H_row[b][j]^T + R[a][b]
    //
    // H_row[0] has blocks at positions 2 (g_skew) and 4 (I)
    // H_row[1] has block at position 3 (I)

    // PHt_col[i][a] = sum_j P[i][j] * H_row[a][j]^T
    // For a=0: PHt_col[i][0] = P[i][2]*H_02^T + P[i][4]*I
    // For a=1: PHt_col[i][1] = P[i][3]*I = P[i][3]

    // We need S and K. Let's compute via float for the 6x6 system to avoid complexity.
    // The block inverse is complex, and IMU update is only ~500us in float.

    // Convert P to float for this update (pragmatic approach for 6x6)
    float P_f[15][15];
    for (int bi = 0; bi < 5; ++bi) {
        for (int bj = 0; bj < 5; ++bj) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    P_f[bi * 3 + r][bj * 3 + c] = P_.at(bi, bj).to_float(r, c);
                }
            }
        }
    }

    // Build H (6x15) float
    float H_f[6][15] = {};
    for (int i = 0; i < 9; ++i) H_f[i / 3][6 + i % 3] = g_skew_f[(i / 3) * 3 + (i % 3)]; // H[0:3][6:9]
    H_f[0][12] = 1; H_f[1][13] = 1; H_f[2][14] = 1; // H[0:3][12:15] = I
    H_f[3][9] = 1; H_f[4][10] = 1; H_f[5][11] = 1;  // H[3:6][9:12] = I

    // PHt (15x6)
    float PHt[15][6] = {};
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 6; ++j)
            for (int k = 0; k < 15; ++k)
                PHt[i][j] += P_f[i][k] * H_f[j][k]; // H^T col j = H row j

    // S = H * PHt + R (6x6)
    float S_f[6][6] = {};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 15; ++k)
                S_f[i][j] += H_f[i][k] * PHt[k][j];
            if (i == j) S_f[i][j] += R_diag_f[i] + 1e-6f;
        }

    // Mahalanobis: y^T * S^{-1} * y via solving Sx = y
    // Use simple Cholesky or LU. For 6x6, Gaussian elimination is fine.
    // Copy S for solving
    float S_copy[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            S_copy[i][j] = S_f[i][j];

    // LU decomposition in-place (partial pivoting)
    int piv[6] = {0, 1, 2, 3, 4, 5};
    for (int k = 0; k < 6; ++k) {
        // Find pivot
        float max_val = 0;
        int max_row = k;
        for (int i = k; i < 6; ++i) {
            float av = S_copy[i][k] < 0 ? -S_copy[i][k] : S_copy[i][k];
            if (av > max_val) { max_val = av; max_row = i; }
        }
        if (max_val < 1e-12f) {
            // Near singular
            if (out_mahalanobis) *out_mahalanobis = 1e10f;
            for (int i = 0; i < 6; ++i) out_innovation[i] = innov[i];
            return;
        }
        if (max_row != k) {
            for (int j = 0; j < 6; ++j) {
                float tmp = S_copy[k][j];
                S_copy[k][j] = S_copy[max_row][j];
                S_copy[max_row][j] = tmp;
            }
            int tmp = piv[k]; piv[k] = piv[max_row]; piv[max_row] = tmp;
        }
        for (int i = k + 1; i < 6; ++i) {
            S_copy[i][k] /= S_copy[k][k];
            for (int j = k + 1; j < 6; ++j)
                S_copy[i][j] -= S_copy[i][k] * S_copy[k][j];
        }
    }

    // Solve Sx = innov (with pivoting)
    float y_piv[6];
    for (int i = 0; i < 6; ++i) y_piv[i] = innov[piv[i]];

    // Forward substitution (L)
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < i; ++j)
            y_piv[i] -= S_copy[i][j] * y_piv[j];

    // Back substitution (U)
    for (int i = 5; i >= 0; --i) {
        for (int j = i + 1; j < 6; ++j)
            y_piv[i] -= S_copy[i][j] * y_piv[j];
        y_piv[i] /= S_copy[i][i];
    }

    // Mahalanobis = innov . (S^{-1} * innov) = innov . y_piv
    float mahal_sq = 0;
    for (int i = 0; i < 6; ++i) mahal_sq += innov[i] * y_piv[i];

    if (out_mahalanobis) *out_mahalanobis = mahal_sq;

    if (mahal_sq > MAHALANOBIS_GATE_THRESHOLD) {
        return;
    }

    // Compute S^{-1} (6x6) by solving S * X = I column by column
    float Sinv_f[6][6] = {};
    for (int col = 0; col < 6; ++col) {
        float rhs[6] = {};
        rhs[col] = 1.0f;
        float rhs_piv[6];
        for (int i = 0; i < 6; ++i) rhs_piv[i] = rhs[piv[i]];
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < i; ++j)
                rhs_piv[i] -= S_copy[i][j] * rhs_piv[j];
        for (int i = 5; i >= 0; --i) {
            for (int j = i + 1; j < 6; ++j)
                rhs_piv[i] -= S_copy[i][j] * rhs_piv[j];
            rhs_piv[i] /= S_copy[i][i];
        }
        for (int i = 0; i < 6; ++i) Sinv_f[i][col] = rhs_piv[i];
    }

    // K = PHt * Sinv (15x6)
    float K_f[15][6] = {};
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 6; ++j)
            for (int k = 0; k < 6; ++k)
                K_f[i][j] += PHt[i][k] * Sinv_f[k][j];

    // delta_x = K * innovation (15x1)
    float dx_f[15] = {};
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 6; ++j)
            dx_f[i] += K_f[i][j] * innov[j];

    // P update: P = (I - K*H) * P
    // KH (15x15)
    float KH[15][15] = {};
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 15; ++j)
            for (int k = 0; k < 6; ++k)
                KH[i][j] += K_f[i][k] * H_f[k][j];

    // (I-KH) * P
    float P_new[15][15] = {};
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 15; ++j) {
            float ImKH_ij = ((i == j) ? 1.0f : 0.0f) - KH[i][j];
            for (int k = 0; k < 15; ++k)
                P_new[i][j] += ImKH_ij * P_f[i][k]; // Hmm, this isn't right
            // Actually: P_new[i][j] = sum_k (I-KH)[i][k] * P[k][j]
        }

    // Redo P_new correctly
    for (int i = 0; i < 15; ++i)
        for (int j = 0; j < 15; ++j) {
            P_new[i][j] = 0;
            for (int k = 0; k < 15; ++k) {
                float ImKH_ik = ((i == k) ? 1.0f : 0.0f) - KH[i][k];
                P_new[i][j] += ImKH_ik * P_f[k][j];
            }
        }

    // Convert P_new back to BlockCovMatrix
    for (int bi = 0; bi < 5; ++bi) {
        for (int bj = 0; bj < 5; ++bj) {
            float vals[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    vals[r * 3 + c] = P_new[bi * 3 + r][bj * 3 + c];

            float max_v = 0;
            for (int i = 0; i < 9; ++i) {
                float av = vals[i] < 0 ? -vals[i] : vals[i];
                if (av > max_v) max_v = av;
            }
            ScaledBlock3x3& blk = P_.at(bi, bj);
            if (max_v < 1e-30f) {
                blk = ScaledBlock3x3();
                continue;
            }
            int exp_bits = 0;
            float tmp = max_v;
            while (tmp >= 2.0f) { tmp *= 0.5f; exp_bits++; }
            while (tmp < 1.0f) { tmp *= 2.0f; exp_bits--; }
            blk.exponent = static_cast<int8_t>(exp_bits - 30);
            float scale = 1.0f;
            int8_t e = blk.exponent;
            if (e > 0) for (int i = 0; i < e; ++i) scale *= 2.0f;
            else for (int i = 0; i < -e; ++i) scale *= 0.5f;
            float inv_s = 1.0f / scale;
            for (int i = 0; i < 9; ++i)
                blk.m[i] = static_cast<int32_t>(vals[i] * inv_s);
        }
    }

    enforce_symmetry();

    inject_error(
        float3_to_q4_27(dx_f),
        float3_to_q4_27(dx_f + 3),
        { Q1_30::from_float(dx_f[6]), Q1_30::from_float(dx_f[7]), Q1_30::from_float(dx_f[8]) },
        { Q1_30::from_float(dx_f[9]), Q1_30::from_float(dx_f[10]), Q1_30::from_float(dx_f[11]) },
        float3_to_q4_27(dx_f + 12)
    );
}

// ---------------------------------------------------------------------------
// Check ZUPT
// ---------------------------------------------------------------------------
bool ESKFFixed::check_zupt(const float accel_raw[3]) {
    float norm_sq = accel_raw[0] * accel_raw[0] + accel_raw[1] * accel_raw[1] + accel_raw[2] * accel_raw[2];
    float norm = std::sqrt(norm_sq);
    float diff = norm - GRAVITY_MAGNITUDE;
    return (diff < 0 ? -diff : diff) < 0.3f;
}

// ---------------------------------------------------------------------------
// Hard reset velocity
// ---------------------------------------------------------------------------
void ESKFFixed::hard_reset_velocity() {
    state_.vel = Vec3_Q4_27::zero();
}

// ---------------------------------------------------------------------------
// Inject error state corrections
// ---------------------------------------------------------------------------
void ESKFFixed::inject_error(
    Vec3_Q4_27 d_pos,
    Vec3_Q4_27 d_vel,
    Vec3_Q1_30 d_theta,
    Vec3_Q1_30 d_bg,
    Vec3_Q4_27 d_ba
) {
    state_.pos = vec3_add(state_.pos, d_pos);
    state_.vel = vec3_add(state_.vel, d_vel);
    state_.gyro_bias = vec3_add(state_.gyro_bias, d_bg);
    state_.accel_bias = vec3_add(state_.accel_bias, d_ba);

    // Quaternion update: q = q * [1, 0.5*d_theta]
    QuatQ q_delta;
    q_delta.w = Q1_30::one();
    q_delta.x = Q1_30(d_theta.x().raw >> 1); // 0.5 * d_theta
    q_delta.y = Q1_30(d_theta.y().raw >> 1);
    q_delta.z = Q1_30(d_theta.z().raw >> 1);
    q_delta = quat_normalize(q_delta);

    state_.quat = quat_mul(state_.quat, q_delta);
    state_.quat = quat_normalize(state_.quat);
}

// ---------------------------------------------------------------------------
// Enforce P symmetry
// ---------------------------------------------------------------------------
void ESKFFixed::enforce_symmetry() {
    P_.enforce_symmetry();
}

// ---------------------------------------------------------------------------
// Get P diagonal for monitoring
// ---------------------------------------------------------------------------
void ESKFFixed::get_P_diagonal(float out[15]) const {
    for (int b = 0; b < 5; ++b) {
        for (int i = 0; i < 3; ++i) {
            out[b * 3 + i] = P_.at(b, b).to_float(i, i);
        }
    }
}

} // namespace trajecto
