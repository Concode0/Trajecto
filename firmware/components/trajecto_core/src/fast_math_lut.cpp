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

#include "fast_math_lut.hpp"
#include <cmath>

namespace trajecto {

static LutTables build_lut_tables() {
    LutTables tables;

    for (int i = 0; i < EXP_LUT_SIZE; i++) {
        float x = EXP_MIN + i * EXP_STEP;
        tables.exp_lut[i] = std::exp(x);
    }

    for (int i = 0; i < SIGMOID_LUT_SIZE; i++) {
        float x = SIGMOID_MIN + i * SIGMOID_STEP;
        tables.sigmoid_lut[i] = 1.0f / (1.0f + std::exp(-x));
    }

    return tables;
}

const LutTables& get_lut_tables() {
    static const LutTables tables = build_lut_tables();
    return tables;
}

} // namespace trajecto
