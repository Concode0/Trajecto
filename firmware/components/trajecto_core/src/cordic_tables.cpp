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

#include "fixed_ops.hpp"
#include <cstdint>

namespace trajecto {
namespace fp {

// ---------------------------------------------------------------------------
// CORDIC Sine/Cosine Implementation
// ---------------------------------------------------------------------------
// Uses 24-iteration CORDIC in rotation mode for ~24-bit accuracy.
// Input: angle in Q2.30 radians
// Output: sin, cos in Q2.30
//
// CORDIC gain K = product of cos(atan(2^-i)) for i=0..23
// K ~ 0.60725293... We pre-compensate by scaling the initial vector.
// 1/K ~ 1.64676025... in Q2.30 = 1768084846 (0x69618AA1)
// ---------------------------------------------------------------------------

// atan(2^-i) in Q2.30 radians, for i = 0..23
// atan(1) = pi/4 = 0.7853981634... -> Q2.30: 843314857
// atan(0.5) = 0.4636476090... -> Q2.30: 497837829
// etc.
static const int32_t CORDIC_ATAN_TABLE[24] = {
    843314857,   // atan(2^0)  = 0.785398163
    497837829,   // atan(2^-1) = 0.463647609
    263043837,   // atan(2^-2) = 0.244978663
    133525159,   // atan(2^-3) = 0.124354995
    67021687,    // atan(2^-4) = 0.062418810
    33524879,    // atan(2^-5) = 0.031239833
    16764568,    // atan(2^-6) = 0.015623729
    8382495,     // atan(2^-7) = 0.007812341
    4191276,     // atan(2^-8) = 0.003906230
    2095639,     // atan(2^-9) = 0.001953123
    1047820,     // atan(2^-10)= 0.000976562
    523910,      // atan(2^-11)= 0.000488281
    261955,      // atan(2^-12)= 0.000244141
    130978,      // atan(2^-13)= 0.000122070
    65489,       // atan(2^-14)= 0.000061035
    32744,       // atan(2^-15)= 0.000030518
    16372,       // atan(2^-16)= 0.000015259
    8186,        // atan(2^-17)= 0.000007629
    4093,        // atan(2^-18)= 0.000003815
    2047,        // atan(2^-19)= 0.000001907
    1023,        // atan(2^-20)= 0.000000954
    512,         // atan(2^-21)= 0.000000477
    256,         // atan(2^-22)= 0.000000238
    128,         // atan(2^-23)= 0.000000119
};

// 1/K (CORDIC gain compensation) in Q2.30
// 1/K = 1.6467602581... * 2^30 = 1768084846
static constexpr int32_t CORDIC_INV_K = 1768084846;

// pi in Q2.30: 3.14159265 * 2^30 = 3373259426 -> exceeds int32! Use two halves.
// pi/2 in Q2.30: 1.5707963 * 2^30 = 1686629713
static constexpr int32_t PI_HALF_Q30 = 1686629713;
// pi in Q2.30 would be ~3.37e9 which overflows int32. We handle via PI_HALF_Q30 only.

void cordic_sincos(int32_t angle_q30, int32_t& sin_out, int32_t& cos_out) {
    // Range reduction to [-pi/2, pi/2]
    // Input is in Q2.30 radians (range +-2 rad covers +-pi easily)
    int32_t a = angle_q30;
    bool negate_sin = false;
    bool negate_cos = false;

    // Reduce to [-pi, pi] first (for angles > pi)
    // Since Q2.30 range is +-2, and pi < 2, we might still get angles near +-pi
    // from quaternion operations. Full reduction:

    // Reduce to [-pi/2, pi/2]
    if (a > PI_HALF_Q30) {
        // a in (pi/2, pi]: sin(a) = sin(pi - a), cos(a) = -cos(pi - a)
        // Approximate: a = pi - a
        a = static_cast<int32_t>(static_cast<int64_t>(PI_HALF_Q30) * 2 - a);
        negate_cos = true;
    } else if (a < -PI_HALF_Q30) {
        // a in [-pi, -pi/2): sin(a) = sin(-pi - a) = -sin(pi+a), cos = -cos(pi+a)
        a = static_cast<int32_t>(-static_cast<int64_t>(PI_HALF_Q30) * 2 - a);
        negate_cos = true;
    }

    // CORDIC rotation mode
    // Start with vector (1/K, 0) and rotate by angle a
    // After N iterations: x ~ cos(a), y ~ sin(a)

    // Initial vector scaled by 1/K. Use half the range to avoid overflow.
    // x0 = CORDIC_INV_K / 2, to leave headroom. We'll scale output by 2.
    // Actually, CORDIC_INV_K ~ 1.647 in Q2.30. During iterations, values grow
    // by at most sqrt(2) per step, but the gain K compensates. With 1/K init,
    // final values are in [-1, 1] range which fits Q2.30.
    // But intermediate values may exceed Q2.30 range slightly. Use int64 to be safe.

    int64_t x = CORDIC_INV_K;  // ~1.647 in Q2.30 — fits int32 (< 2.0)
    int64_t y = 0;
    int64_t z = a; // remaining angle

    for (int i = 0; i < 24; ++i) {
        int64_t x_new, y_new;
        if (z >= 0) {
            x_new = x - (y >> i);
            y_new = y + (x >> i);
            z -= CORDIC_ATAN_TABLE[i];
        } else {
            x_new = x + (y >> i);
            y_new = y - (x >> i);
            z += CORDIC_ATAN_TABLE[i];
        }
        x = x_new;
        y = y_new;
    }

    // Clamp to int32 range
    if (x > INT32_MAX) x = INT32_MAX;
    if (x < INT32_MIN) x = INT32_MIN;
    if (y > INT32_MAX) y = INT32_MAX;
    if (y < INT32_MIN) y = INT32_MIN;

    cos_out = static_cast<int32_t>(negate_cos ? -x : x);
    sin_out = static_cast<int32_t>(negate_sin ? -y : y);
    // negate_sin is always false in current reduction — sin sign is handled by angle sign
    static_cast<void>(negate_sin);
}

} // namespace fp
} // namespace trajecto
