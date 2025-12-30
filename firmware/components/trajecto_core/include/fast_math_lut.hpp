#pragma once

#include <cmath>
#include <algorithm>

namespace trajecto {

// ----------------------------------------------------------------------------
// LUT Generation Constants
// ----------------------------------------------------------------------------
constexpr float EXP_MIN = -10.0f;
constexpr float EXP_MAX = 10.0f;
constexpr int EXP_LUT_SIZE = 256;
constexpr float EXP_STEP = (EXP_MAX - EXP_MIN) / (float)(EXP_LUT_SIZE - 1);
constexpr float EXP_INV_STEP = 1.0f / EXP_STEP;

constexpr float SIGMOID_MIN = -6.0f;
constexpr float SIGMOID_MAX = 6.0f;
constexpr int SIGMOID_LUT_SIZE = 256;
constexpr float SIGMOID_STEP = (SIGMOID_MAX - SIGMOID_MIN) / (float)(SIGMOID_LUT_SIZE - 1);
constexpr float SIGMOID_INV_STEP = 1.0f / SIGMOID_STEP;

// ----------------------------------------------------------------------------
// Precomputed Tables (Declarations)
// ----------------------------------------------------------------------------
extern const float* const exp_lut;
extern const float* const sigmoid_lut;

/**
 * @brief Fast Exponential approximation using LUT with Linear Interpolation.
 */
inline float fast_exp(float x) {
    if (x <= EXP_MIN) return 0.0f;
    if (x >= EXP_MAX) return exp_lut[EXP_LUT_SIZE - 1]; // Or std::exp(x) if precision needed at high range

    float pos = (x - EXP_MIN) * EXP_INV_STEP;
    int index = (int)pos;
    float frac = pos - index;

    // Linear Interpolation
    return exp_lut[index] + frac * (exp_lut[index + 1] - exp_lut[index]);
}

/**
 * @brief Fast Sigmoid approximation using LUT with Linear Interpolation.
 *        sigmoid(x) = 1 / (1 + exp(-x))
 */
inline float fast_sigmoid(float x) {
    if (x <= SIGMOID_MIN) return 0.0f;
    if (x >= SIGMOID_MAX) return 1.0f;

    float pos = (x - SIGMOID_MIN) * SIGMOID_INV_STEP;
    int index = (int)pos;
    float frac = pos - index;

    return sigmoid_lut[index] + frac * (sigmoid_lut[index + 1] - sigmoid_lut[index]);
}

/**
 * @brief Fast Softplus: log(1 + exp(x))
 *        Uses fast_exp for range inside LUT, linear approximation for large x.
 */
inline float fast_softplus(float x) {
    if (x > 20.0f) return x; // log(1+exp(x)) -> x for large x
    if (x < -20.0f) return 0.0f; // log(1+0) -> 0
    return std::log(1.0f + fast_exp(x));
}

} // namespace trajecto
