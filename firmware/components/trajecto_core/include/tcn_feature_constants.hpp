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

#pragma once

#include "fixed_point.hpp"
#include "model_params.hpp"

namespace trajecto {
namespace tcn_features {

// ---------------------------------------------------------------------------
// Pre-computed reciprocals for z-score normalization (Q16.15)
// Derived from model_params.hpp IMU_STD values
// ---------------------------------------------------------------------------
constexpr fp::Q16_15 INV_IMU_STD[7] = {
    fp::Q16_15::from_float(1.0f / 1.99539632f),   // accel_x
    fp::Q16_15::from_float(1.0f / 0.98817613f),   // accel_y
    fp::Q16_15::from_float(1.0f / 1.47147918f),   // accel_z
    fp::Q16_15::from_float(1.0f / 0.35205848f),   // gyro_x
    fp::Q16_15::from_float(1.0f / 0.50353031f),   // gyro_y
    fp::Q16_15::from_float(1.0f / 0.34035876f),   // gyro_z
    fp::Q16_15::from_float(1.0f / 409.52650890f)  // force
};

// IMU mean values (Q16.15 for accel/gyro, Q16.15 for force)
constexpr fp::Vec3_Q16_15 IMU_MEAN_ACCEL{
    fp::Q16_15::from_float(2.91851636f),
    fp::Q16_15::from_float(-8.09016939f),
    fp::Q16_15::from_float(4.23098800f)
};

constexpr fp::Vec3_Q16_15 IMU_MEAN_GYRO{
    fp::Q16_15::from_float(-0.01135616f),
    fp::Q16_15::from_float(0.00813827f),
    fp::Q16_15::from_float(-0.00344270f)
};

constexpr fp::Q16_15 IMU_MEAN_FORCE = fp::Q16_15::from_float(956.41780222f);

// ---------------------------------------------------------------------------
// Gravity normalization (Q1.30 for high precision unit norm operations)
// ---------------------------------------------------------------------------
constexpr fp::Q1_30 INV_GRAVITY_MAG = fp::Q1_30::from_float(1.0f / 9.806650f);
constexpr fp::Q1_30 GRAVITY_SCALE = fp::Q1_30::from_float(2.0f);

// ---------------------------------------------------------------------------
// Innovation normalization (Q16.15)
// Uses Allan variance noise parameters (max VRW/ARW + safety epsilon)
// ---------------------------------------------------------------------------
constexpr fp::Q16_15 INV_MAX_VRW = fp::Q16_15::from_float(1.0f / (9.3271e-04f + 1e-3f));
constexpr fp::Q16_15 INV_MAX_ARW = fp::Q16_15::from_float(1.0f / (7.9283e-05f + 1e-3f));
constexpr fp::Q16_15 INNOVATION_CLAMP_MIN = fp::Q16_15::from_float(-10.0f);
constexpr fp::Q16_15 INNOVATION_CLAMP_MAX = fp::Q16_15::from_float(10.0f);

} // namespace tcn_features
} // namespace trajecto
