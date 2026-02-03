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

#include "tcn_features_fixed.hpp"
#include "tcn_feature_constants.hpp"

namespace trajecto {

using namespace fp;
using namespace tcn_features;

void extract_features_fixed(
    const float accel_raw[3],
    const float gyro_raw[3],
    float force_raw,
    const ESKFFixed& eskf,
    const float last_innovation[6],
    float out_features[16]
) {
    // ---------------------------------------------------------------------------
    // 1. Z-score normalization for IMU data
    // ---------------------------------------------------------------------------
    // Convert raw floats to Q16.15
    Vec3_Q16_15 accel_q{
        Q16_15::from_float(accel_raw[0]),
        Q16_15::from_float(accel_raw[1]),
        Q16_15::from_float(accel_raw[2])
    };
    Vec3_Q16_15 gyro_q{
        Q16_15::from_float(gyro_raw[0]),
        Q16_15::from_float(gyro_raw[1]),
        Q16_15::from_float(gyro_raw[2])
    };

    // Mean-center
    Vec3_Q16_15 accel_centered = vec3_sub(accel_q, IMU_MEAN_ACCEL);
    Vec3_Q16_15 gyro_centered = vec3_sub(gyro_q, IMU_MEAN_GYRO);

    // Divide by std (multiply by reciprocal)
    Vec3_Q16_15 accel_norm{
        mul_cross<17, 15>(accel_centered.x(), INV_IMU_STD[0]),
        mul_cross<17, 15>(accel_centered.y(), INV_IMU_STD[1]),
        mul_cross<17, 15>(accel_centered.z(), INV_IMU_STD[2])
    };

    Vec3_Q16_15 gyro_norm{
        mul_cross<17, 15>(gyro_centered.x(), INV_IMU_STD[3]),
        mul_cross<17, 15>(gyro_centered.y(), INV_IMU_STD[4]),
        mul_cross<17, 15>(gyro_centered.z(), INV_IMU_STD[5])
    };

    // Force normalization
    Q16_15 force_q = Q16_15::from_float(force_raw);
    Q16_15 force_norm = mul_cross<17, 15>(
        sat_sub(force_q, IMU_MEAN_FORCE),
        INV_IMU_STD[6]
    );

    // ---------------------------------------------------------------------------
    // 2. Gravity in body frame (unit-normalized, scaled by 2.0)
    // ---------------------------------------------------------------------------
    const auto& state = eskf.get_state_fixed();
    Mat3Q R_bw = quat_to_rotmat(state.quat);
    Mat3Q R_wb = R_bw.transpose();

    // Gravity points down in world frame
    Vec3_Q4_27 gravity_w{
        Q4_27::zero(),
        Q4_27::zero(),
        Q4_27::from_float(-9.806650f)
    };
    Vec3_Q4_27 gravity_b_raw = mat3_vec3_mul<5, 27>(R_wb, gravity_w);

    // Unit normalize: (g_b / g_mag) * 2.0
    // First divide by magnitude (Q1.30 reciprocal), then scale by 2.0
    Vec3_Q4_27 gravity_b_unit = vec3_scale_cross<5, 27>(gravity_b_raw, INV_GRAVITY_MAG);
    Vec3_Q4_27 gravity_b_norm = vec3_scale_cross<5, 27>(gravity_b_unit, GRAVITY_SCALE);

    // ---------------------------------------------------------------------------
    // 3. Innovation normalization (Allan variance based, clamped ±10)
    // ---------------------------------------------------------------------------
    float innovation_norm[6];
    for (int i = 0; i < 3; ++i) {  // Accel channels (0-2)
        Q16_15 innov = Q16_15::from_float(last_innovation[i]);
        Q16_15 normalized = mul_cross<17, 15>(innov, INV_MAX_VRW);
        normalized = clamp(normalized, INNOVATION_CLAMP_MIN, INNOVATION_CLAMP_MAX);
        innovation_norm[i] = normalized.to_float();
    }
    for (int i = 3; i < 6; ++i) {  // Gyro channels (3-5)
        Q16_15 innov = Q16_15::from_float(last_innovation[i]);
        Q16_15 normalized = mul_cross<17, 15>(innov, INV_MAX_ARW);
        normalized = clamp(normalized, INNOVATION_CLAMP_MIN, INNOVATION_CLAMP_MAX);
        innovation_norm[i] = normalized.to_float();
    }

    // ---------------------------------------------------------------------------
    // 4. Pack output (16D total, matches Python)
    // ---------------------------------------------------------------------------
    int idx = 0;

    // gyro_b_norm [3]
    out_features[idx++] = gyro_norm.x().to_float();
    out_features[idx++] = gyro_norm.y().to_float();
    out_features[idx++] = gyro_norm.z().to_float();

    // accel_b_norm [3]
    out_features[idx++] = accel_norm.x().to_float();
    out_features[idx++] = accel_norm.y().to_float();
    out_features[idx++] = accel_norm.z().to_float();

    // force_norm [1]
    out_features[idx++] = force_norm.to_float();

    // gravity_b_norm [3]
    out_features[idx++] = gravity_b_norm.x().to_float();
    out_features[idx++] = gravity_b_norm.y().to_float();
    out_features[idx++] = gravity_b_norm.z().to_float();

    // innovation_norm [6]
    for (int i = 0; i < 6; ++i) {
        out_features[idx++] = innovation_norm[i];
    }
}

} // namespace trajecto
