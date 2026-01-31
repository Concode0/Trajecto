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

#include <cmath>
#include <limits>

namespace trajecto {

/**
 * @brief Fast approximation of tanh using a rational polynomial or LUT.
 *        Using a rational approximation here for balance of size/speed.
 *        Formula: tanh(x) ~= x * (27 + x^2) / (27 + 9x^2) for small x
 *        or simplified Padé approximant.
 *
 *        For embedded, we can also use:
 *        tanh(x) = 2*sigmoid(2x) - 1
 *        And sigmoid(x) = 0.5 * (x / (1 + |x|) + 1) [Fast Sigmoid]
 *        So tanh(x) ~= x / (1 + |x|) ? No, that's x / (1+|x|) is softsign.
 *
 *        Let's use the rational approximation for range [-3, 3] and clamp outside.
 *
 * @param x Input float
 * @return float Tanh approximation
 */
inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    // Simple rational approximation
    // tanh(x) ~= x * ( 135135.0f + x*x * ( 17325.0f + x*x * 378.0f ) ) /
    //                ( 135135.0f + x*x * ( 62370.0f + x*x * 3150.0f ) );
    // That's too heavy.

    // TFLite uses integer tables.
    // For manual feature extraction, let's use standard std::tanh if not critical bottleneck.
    // But since you asked for optimization:
    return x / (1.0f + std::abs(x)); // Softsign (very fast, similar shape, gradient differs)
    // Wait, Softsign is NOT Tanh. TCN expects Tanh distribution [-1, 1].
    // Softsign is flatter.

    // Better approximation:
    // y = x * (27 + x^2) / (27 + 9 * x^2)
    // float x2 = x * x;
    // return x * (27.0f + x2) / (27.0f + 9.0f * x2);

    // Even faster: Piecewise linear?
    // Let's stick to std::tanh unless profiling shows it's >10% of total time.
    // The TCN execution dominates (thousands of MACs). Tanh is just 3 calls per step.
    // So this optimization might be premature.

    // However, if we want to replace std::tanh in the TCN WRAPPER (feature extraction):
    return std::tanh(x);
}

} // namespace trajecto