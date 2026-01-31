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

#pragma once

#include <array>
#include <cmath>

namespace trajecto {

// ----------------------------------------------------------------------------
// LUT Generation Constants
// ----------------------------------------------------------------------------
constexpr float EXP_MIN = -10.0f;
constexpr float EXP_MAX = 10.0f;
constexpr int EXP_LUT_SIZE = 256;
constexpr float EXP_STEP = (EXP_MAX - EXP_MIN) / static_cast<float>(EXP_LUT_SIZE - 1);
constexpr float EXP_INV_STEP = 1.0f / EXP_STEP;

constexpr float SIGMOID_MIN = -6.0f;
constexpr float SIGMOID_MAX = 6.0f;
constexpr int SIGMOID_LUT_SIZE = 256;
constexpr float SIGMOID_STEP = (SIGMOID_MAX - SIGMOID_MIN) / static_cast<float>(SIGMOID_LUT_SIZE - 1);
constexpr float SIGMOID_INV_STEP = 1.0f / SIGMOID_STEP;

constexpr float SOFTPLUS_LARGE_THRESHOLD = 20.0f;

// ----------------------------------------------------------------------------
// LUT Storage
// ----------------------------------------------------------------------------
struct LutTables {
    std::array<float, EXP_LUT_SIZE> exp_lut;
    std::array<float, SIGMOID_LUT_SIZE> sigmoid_lut;
};

// Returns a reference to the singleton LUT tables.
// Uses Meyers' singleton (function-local static): thread-safe in C++11+, no SIOF.
const LutTables& get_lut_tables();

/**
 * @brief Fast Exponential approximation using LUT with Linear Interpolation.
 */
inline float fast_exp(float x) {
    const auto& lut = get_lut_tables().exp_lut;
    if (x <= EXP_MIN) return 0.0f;
    if (x >= EXP_MAX) return lut[EXP_LUT_SIZE - 1];

    float pos = (x - EXP_MIN) * EXP_INV_STEP;
    int index = static_cast<int>(pos);
    float frac = pos - index;

    return lut[index] + frac * (lut[index + 1] - lut[index]);
}

/**
 * @brief Fast Sigmoid approximation using LUT with Linear Interpolation.
 *        sigmoid(x) = 1 / (1 + exp(-x))
 */
inline float fast_sigmoid(float x) {
    const auto& lut = get_lut_tables().sigmoid_lut;
    if (x <= SIGMOID_MIN) return 0.0f;
    if (x >= SIGMOID_MAX) return 1.0f;

    float pos = (x - SIGMOID_MIN) * SIGMOID_INV_STEP;
    int index = static_cast<int>(pos);
    float frac = pos - index;

    return lut[index] + frac * (lut[index + 1] - lut[index]);
}

/**
 * @brief Fast Softplus: log(1 + exp(x))
 *        Uses fast_exp for range inside LUT, linear approximation for large x.
 */
inline float fast_softplus(float x) {
    if (x > SOFTPLUS_LARGE_THRESHOLD) return x;
    if (x < -SOFTPLUS_LARGE_THRESHOLD) return 0.0f;
    return std::log(1.0f + fast_exp(x));
}

} // namespace trajecto
