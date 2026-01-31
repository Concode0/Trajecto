/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System
 * Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * NOTICE: This software is protected under the following ROK Patent Applications:
 * 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
 * 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
 *
 * Commercial use or redistribution of the core logic requires a separate license.
 * For inquiries, contact: nemonanconcode@gmail.com
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
