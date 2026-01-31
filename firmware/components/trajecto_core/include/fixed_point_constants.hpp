#pragma once

#include "fixed_point.hpp"

// Converts model_params.hpp float constants to Q-format compile-time constants
// for the fixed-point ESKF implementation.

namespace trajecto {
namespace fp {

// ---------------------------------------------------------------------------
// Time Step: DT in Q4.27 (0.019957245 s)
// Q5.27 value = 0.019957245 * 2^27 = 2676671
// ---------------------------------------------------------------------------
static constexpr Q4_27 DT_Q = Q4_27(2676671);

// DT^2 / 2 = 0.5 * 0.019957245^2 = 1.991e-4
// In Q5.27: 1.991e-4 * 2^27 = 26713
static constexpr Q4_27 HALF_DT2_Q = Q4_27(26713);

// ---------------------------------------------------------------------------
// Gravity: 9.80665 m/s^2 in Q5.27
// 9.80665 * 2^27 = 1316055961
// ---------------------------------------------------------------------------
static constexpr Q4_27 GRAVITY_Q = Q4_27(1316055961);

// Gravity world vector: (0, 0, -9.80665) in Q5.27
static constexpr Vec3_Q4_27 GRAVITY_W_Q = {
    Q4_27(0), Q4_27(0), Q4_27(-1316055961)
};

// ---------------------------------------------------------------------------
// Allan Variance Parameters (Process Noise Q diagonal)
// These are variance values (squared): very small numbers.
// Store as float and convert at init time since they're used once.
// ---------------------------------------------------------------------------

// VRW^2 values (m^2/s^4 * dt):
// VRW_X^2 * dt = (8.3297e-04)^2 * 0.019957 = 1.384e-08
// In Q5.27: too small for Q5.27 directly. We store the Q diagonal as ScaledBlock3x3.
// These are only used at initialization, so float conversion is acceptable.

// ---------------------------------------------------------------------------
// ZUPT noise: ZUPT_NOISE_STD^2 in Q5.27
// 0.01^2 = 1e-4, in Q5.27: 1e-4 * 2^27 = 13422
// ---------------------------------------------------------------------------
static constexpr Q4_27 ZUPT_R_MIN_Q = Q4_27(13422);

// ZUPT max R: 100 * min = 1e-2, in Q5.27: 1e-2 * 2^27 = 1342177
static constexpr Q4_27 ZUPT_R_MAX_Q = Q4_27(1342177);

// ---------------------------------------------------------------------------
// Mahalanobis gate threshold: 8.0 in Q5.27
// 8.0 * 2^27 = 1073741824
// ---------------------------------------------------------------------------
static constexpr Q4_27 MAHAL_GATE_Q = Q4_27(1073741824);

// ---------------------------------------------------------------------------
// ZUPT hard reset threshold: 0.98 in Q2.30
// 0.98 * 2^30 = 1052266987
// ---------------------------------------------------------------------------
static constexpr Q1_30 ZUPT_HARD_RESET_Q = Q1_30(1052266987);

// ---------------------------------------------------------------------------
// TCN velocity noise std: 0.05 in Q5.27
// 0.05 * 2^27 = 6710886
// ---------------------------------------------------------------------------
static constexpr Q4_27 TCN_VEL_NOISE_Q = Q4_27(6710886);

// ---------------------------------------------------------------------------
// Feature extraction constants in Q formats
// ---------------------------------------------------------------------------

// IMU_MEAN[7] in Q17.15
// accel mean: ~2.78, -2.68, 4.20 -> *32768
// gyro mean: ~-0.168, 0.330, 0.238
// force mean: 3830.2 (clamped to Q17.15 range)
static constexpr int32_t IMU_MEAN_Q15[7] = {
    91160,    // 2.78230069 * 32768
    -87884,   // -2.68209409 * 32768
    137562,   // 4.19848411 * 32768
    -5513,    // -0.16821001 * 32768
    10820,    // 0.33020652 * 32768
    7802,     // 0.23807174 * 32768
    125506437 // 3830.24641663 * 32768 (fits in int32)
};

// 1/IMU_STD[7] in Q17.15 (reciprocal for multiplication instead of division)
// 1/5.8165 = 0.1719 -> *32768 = 5634
// etc.
static constexpr int32_t INV_IMU_STD_Q15[7] = {
    5634,   // 1/5.81654358 * 32768
    5746,   // 1/5.70299290 * 32768
    19152,  // 1/1.71116575 * 32768
    871,    // 1/37.63273751 * 32768
    840,    // 1/39.02173354 * 32768
    1189,   // 1/27.56065179 * 32768
    55,     // 1/597.15314685 * 32768
};

// Velocity normalization: 1/(VEL_STD_L2 + 1e-3) in Q17.15
// 1/0.153274 = 6.524 -> *32768 = 213842
static constexpr int32_t INV_VEL_STD_Q15 = 213842;

// Gravity norm scale: 2.0/9.80665 in Q2.30
// = 0.20394 -> *2^30 = 219020119
static constexpr Q1_30 GRAVITY_NORM_FACTOR_Q = Q1_30(219020119);

// Innovation normalization: 1/(MAX_VRW + 1e-3) and 1/(MAX_ARW + 1e-3)
// 1/(9.3271e-04 + 1e-3) = 1/(1.932e-3) = 517.5 in Q17.15 = 16957440
// 1/(7.9283e-05 + 1e-3) = 1/(1.079e-3) = 926.9 in Q17.15 = 30369177
static constexpr int32_t INV_VRW_SCALE_Q15 = 16957440;
static constexpr int32_t INV_ARW_SCALE_Q15 = 30369177;

// Innovation clamp range: 10.0 in Q17.15 = 327680
static constexpr int32_t INNOV_CLAMP_Q15 = 327680;

// Pen tip offset in Q5.27: (0, -0.06, 0)
// -0.06 * 2^27 = -8053064
static constexpr Vec3_Q4_27 PEN_TIP_OFFSET_Q = {
    Q4_27(0), Q4_27(-8053064), Q4_27(0)
};

// Softplus R_MIN: 1e-4 in Q5.27 = 13422
static constexpr Q4_27 R_MIN_FP = Q4_27(13422);
// Softplus R_MAX: 3.0 in Q5.27 = 402653184
static constexpr Q4_27 R_MAX_FP = Q4_27(402653184);

// 0.5 in Q2.30 for quaternion small-angle approximation
static constexpr Q1_30 HALF_Q30 = Q1_30(1 << 29);

// 0.01 clamped prob min: 0.01 in Q2.30
static constexpr Q1_30 PROB_MIN_Q = Q1_30(10737418);   // 0.01 * 2^30
// 0.99 clamped prob max: 0.99 in Q2.30
static constexpr Q1_30 PROB_MAX_Q = Q1_30(1063004406);  // 0.99 * 2^30

// Initial P diagonal: 0.1 in Q5.27 = 13421773
static constexpr Q4_27 P_INIT_DIAG_Q = Q4_27(13421773);

// ---------------------------------------------------------------------------
// Precomputed scalar for block_scale_int(-dt)
// ---------------------------------------------------------------------------
// -dt as Q5.27 raw value for use as scalar in block_scale_int
// -0.019957245 * 2^27 = -2676671
static constexpr int32_t NEG_DT_SCALAR = -2676671;
static constexpr int8_t  DT_SCALAR_EXP = -27;

// dt as Q5.27 raw value
static constexpr int32_t DT_SCALAR = 2676671;

// ---------------------------------------------------------------------------
// ZUPT/ZARU R computation in fixed-point (log-space soft-thresholding)
// ---------------------------------------------------------------------------
// R range: 1e-4 (tight constraint) to 100.0 (very uncertain) — 6 orders of magnitude
// Computed via geometric interpolation in log-space:
//   log(1e-4) = -9.21034, log(100) = 4.60517
// These float values are used at the boundary (only 2 float ops per update)

// ZUPT onset threshold: 0.5 in Q1.30
static constexpr int32_t ONSET_Q30 = (1 << 29);  // 0.5 * 2^30

// Legacy R values (still used by other code paths)
static constexpr int32_t ZUPT_R_MIN_RAW = 13422;     // 1e-4 * 2^27
static constexpr int32_t ZUPT_R_MAX_RAW = 1342177;   // 1e-2 * 2^27
// These have implicit exponent -27

} // namespace fp
} // namespace trajecto
