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

#include "eskf_fixed.hpp"

namespace trajecto {

/**
 * @brief Extract TCN input features using fixed-point arithmetic
 *
 * Produces 16D feature vector matching Python base_hybrid_model.py:743-751:
 *   1. gyro_b_norm [3]     - Z-score normalized
 *   2. accel_b_norm [3]    - Z-score normalized
 *   3. force_norm [1]      - Z-score normalized
 *   4. gravity_b_norm [3]  - Unit-normalized, scaled by 2.0
 *   5. innovation_norm [6] - Allan variance normalized, clamped ±10
 *
 * @param accel_raw         Raw accelerometer [m/s²]
 * @param gyro_raw          Raw gyroscope [rad/s]
 * @param force_raw         Raw force sensor reading
 * @param eskf              Fixed-point ESKF instance
 * @param last_innovation   Last measurement innovation [6]
 * @param out_features      Output buffer [16] (float32)
 */
void extract_features_fixed(
    const float accel_raw[3],
    const float gyro_raw[3],
    float force_raw,
    const ESKFFixed& eskf,
    const float last_innovation[6],
    float out_features[16]
);

} // namespace trajecto
