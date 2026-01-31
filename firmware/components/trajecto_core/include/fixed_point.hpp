#pragma once

#include <array>
#include <cstdint>

namespace trajecto {
namespace fp {

// ---------------------------------------------------------------------------
// Q-Format Fixed Point Type
// ---------------------------------------------------------------------------
// QFixed<I, F> represents a signed fixed-point number with I integer bits
// (including sign) and F fractional bits. Total = I + F = 32 bits always.
// Value = raw / 2^F
// Range = [-2^(I-1), 2^(I-1) - 2^(-F)]
// ---------------------------------------------------------------------------

template <int IntBits, int FracBits>
struct QFixed {
    static_assert(IntBits + FracBits == 32, "IntBits + FracBits must equal 32");
    static_assert(FracBits >= 0 && IntBits >= 1, "Need at least 1 sign bit");

    static constexpr int kIntBits = IntBits;
    static constexpr int kFracBits = FracBits;
    static constexpr int64_t kOne = static_cast<int64_t>(1) << FracBits;
    static constexpr int32_t kMax = INT32_MAX;
    static constexpr int32_t kMin = INT32_MIN;

    int32_t raw;

    constexpr QFixed() : raw(0) {}
    explicit constexpr QFixed(int32_t r) : raw(r) {}

    // Convert from float (compile-time or runtime)
    static constexpr QFixed from_float(float v) {
        // Clamp to representable range
        float scaled = v * static_cast<float>(kOne);
        if (scaled >= static_cast<float>(INT32_MAX)) return QFixed(INT32_MAX);
        if (scaled <= static_cast<float>(INT32_MIN)) return QFixed(INT32_MIN);
        return QFixed(static_cast<int32_t>(scaled + (scaled >= 0 ? 0.5f : -0.5f)));
    }

    // Convert to float
    float to_float() const {
        return static_cast<float>(raw) / static_cast<float>(kOne);
    }

    // Zero
    static constexpr QFixed zero() { return QFixed(0); }

    // One (1.0)
    static constexpr QFixed one() { return QFixed(static_cast<int32_t>(kOne)); }
};

// Common type aliases
// Convention: IntBits includes sign bit, so QFixed<N,F> has range +-2^(N-1)
// "Q4.27" means 1 sign + 3 integer + 27 frac = QFixed<5,27> = 32 bits
// "Q1.30" means 1 sign + 1 integer + 30 frac = QFixed<2,30> = 32 bits
using Q4_27  = QFixed<5, 27>;   // position, velocity, accel_bias: range +-16
using Q1_30  = QFixed<2, 30>;   // quaternion, gyro_bias, angles: range +-2
using Q16_15 = QFixed<17, 15>;  // raw IMU, intermediates: range +-65536
using Q8_23  = QFixed<9, 23>;   // intermediate computations: range +-256

// ---------------------------------------------------------------------------
// Saturating Arithmetic
// ---------------------------------------------------------------------------

// Saturating add (same format)
template <int I, int F>
inline QFixed<I, F> sat_add(QFixed<I, F> a, QFixed<I, F> b) {
    int64_t sum = static_cast<int64_t>(a.raw) + static_cast<int64_t>(b.raw);
    if (sum > INT32_MAX) return QFixed<I, F>(INT32_MAX);
    if (sum < INT32_MIN) return QFixed<I, F>(INT32_MIN);
    return QFixed<I, F>(static_cast<int32_t>(sum));
}

// Saturating subtract (same format)
template <int I, int F>
inline QFixed<I, F> sat_sub(QFixed<I, F> a, QFixed<I, F> b) {
    int64_t diff = static_cast<int64_t>(a.raw) - static_cast<int64_t>(b.raw);
    if (diff > INT32_MAX) return QFixed<I, F>(INT32_MAX);
    if (diff < INT32_MIN) return QFixed<I, F>(INT32_MIN);
    return QFixed<I, F>(static_cast<int32_t>(diff));
}

// Saturating negate
template <int I, int F>
inline QFixed<I, F> sat_neg(QFixed<I, F> a) {
    if (a.raw == INT32_MIN) return QFixed<I, F>(INT32_MAX);
    return QFixed<I, F>(-a.raw);
}

// ---------------------------------------------------------------------------
// Fixed-Point Multiply
// ---------------------------------------------------------------------------
// Multiply two Q-format numbers, result in specified output format.
// Uses int64_t intermediate to prevent overflow.
// Result format: caller specifies output template params.

// Same-format multiply: Q<I,F> * Q<I,F> -> Q<I,F>
template <int I, int F>
inline QFixed<I, F> mul(QFixed<I, F> a, QFixed<I, F> b) {
    int64_t prod = static_cast<int64_t>(a.raw) * static_cast<int64_t>(b.raw);
    // Round: add 0.5 ULP before shift
    prod += (static_cast<int64_t>(1) << (F - 1));
    int64_t result = prod >> F;
    // Saturate
    if (result > INT32_MAX) return QFixed<I, F>(INT32_MAX);
    if (result < INT32_MIN) return QFixed<I, F>(INT32_MIN);
    return QFixed<I, F>(static_cast<int32_t>(result));
}

// Cross-format multiply: Q<IA,FA> * Q<IB,FB> -> Q<IO,FO>
// The product has FA+FB fractional bits. We need to shift to get FO frac bits.
template <int IO, int FO, int IA, int FA, int IB, int FB>
inline QFixed<IO, FO> mul_cross(QFixed<IA, FA> a, QFixed<IB, FB> b) {
    int64_t prod = static_cast<int64_t>(a.raw) * static_cast<int64_t>(b.raw);
    // Product has (FA + FB) fractional bits, we need FO
    constexpr int shift = FA + FB - FO;
    if constexpr (shift > 0) {
        prod += (static_cast<int64_t>(1) << (shift - 1)); // round
        prod >>= shift;
    } else if constexpr (shift < 0) {
        prod <<= (-shift);
    }
    if (prod > INT32_MAX) return QFixed<IO, FO>(INT32_MAX);
    if (prod < INT32_MIN) return QFixed<IO, FO>(INT32_MIN);
    return QFixed<IO, FO>(static_cast<int32_t>(prod));
}

// Multiply-accumulate: acc += a * b (int64 accumulator)
inline int64_t mac(int64_t acc, int32_t a, int32_t b) {
    return acc + static_cast<int64_t>(a) * static_cast<int64_t>(b);
}

// ---------------------------------------------------------------------------
// Format Conversion (shift between Q formats)
// ---------------------------------------------------------------------------

template <int IO, int FO, int II, int FI>
inline QFixed<IO, FO> convert(QFixed<II, FI> x) {
    constexpr int shift = FO - FI;
    int64_t val = static_cast<int64_t>(x.raw);
    if constexpr (shift > 0) {
        val <<= shift;
    } else if constexpr (shift < 0) {
        val += (static_cast<int64_t>(1) << ((-shift) - 1)); // round
        val >>= (-shift);
    }
    if (val > INT32_MAX) return QFixed<IO, FO>(INT32_MAX);
    if (val < INT32_MIN) return QFixed<IO, FO>(INT32_MIN);
    return QFixed<IO, FO>(static_cast<int32_t>(val));
}

// ---------------------------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------------------------

// Shift right with rounding
inline int32_t rshift_round(int32_t x, int bits) {
    if (bits <= 0) return x;
    return (x + (1 << (bits - 1))) >> bits;
}

// Absolute value (saturating)
template <int I, int F>
inline QFixed<I, F> abs_val(QFixed<I, F> a) {
    if (a.raw == INT32_MIN) return QFixed<I, F>(INT32_MAX);
    return QFixed<I, F>(a.raw < 0 ? -a.raw : a.raw);
}

// Max of two values
template <int I, int F>
inline QFixed<I, F> max_val(QFixed<I, F> a, QFixed<I, F> b) {
    return QFixed<I, F>(a.raw > b.raw ? a.raw : b.raw);
}

// Min of two values
template <int I, int F>
inline QFixed<I, F> min_val(QFixed<I, F> a, QFixed<I, F> b) {
    return QFixed<I, F>(a.raw < b.raw ? a.raw : b.raw);
}

// Clamp
template <int I, int F>
inline QFixed<I, F> clamp(QFixed<I, F> x, QFixed<I, F> lo, QFixed<I, F> hi) {
    return max_val(lo, min_val(x, hi));
}

// ---------------------------------------------------------------------------
// 3-Element Vector (same Q format)
// ---------------------------------------------------------------------------

template <int I, int F>
struct Vec3Q {
    std::array<QFixed<I, F>, 3> data;

    constexpr Vec3Q() : data{QFixed<I,F>(), QFixed<I,F>(), QFixed<I,F>()} {}
    constexpr Vec3Q(QFixed<I, F> x_, QFixed<I, F> y_, QFixed<I, F> z_) : data{x_, y_, z_} {}

    // Named accessors for readability
    constexpr QFixed<I, F>& x() { return data[0]; }
    constexpr const QFixed<I, F>& x() const { return data[0]; }
    constexpr QFixed<I, F>& y() { return data[1]; }
    constexpr const QFixed<I, F>& y() const { return data[1]; }
    constexpr QFixed<I, F>& z() { return data[2]; }
    constexpr const QFixed<I, F>& z() const { return data[2]; }

    static Vec3Q zero() { return Vec3Q(); }

    QFixed<I, F>& operator[](int i) { return data[i]; }
    const QFixed<I, F>& operator[](int i) const { return data[i]; }
};

using Vec3_Q4_27  = Vec3Q<5, 27>;
using Vec3_Q1_30  = Vec3Q<2, 30>;
using Vec3_Q16_15 = Vec3Q<17, 15>;
using Vec3_Q8_23  = Vec3Q<9, 23>;

// Vector add (same format)
template <int I, int F>
inline Vec3Q<I, F> vec3_add(Vec3Q<I, F> a, Vec3Q<I, F> b) {
    return { sat_add(a.x(), b.x()), sat_add(a.y(), b.y()), sat_add(a.z(), b.z()) };
}

// Vector subtract (same format)
template <int I, int F>
inline Vec3Q<I, F> vec3_sub(Vec3Q<I, F> a, Vec3Q<I, F> b) {
    return { sat_sub(a.x(), b.x()), sat_sub(a.y(), b.y()), sat_sub(a.z(), b.z()) };
}

// Vector negate
template <int I, int F>
inline Vec3Q<I, F> vec3_neg(Vec3Q<I, F> a) {
    return { sat_neg(a.x()), sat_neg(a.y()), sat_neg(a.z()) };
}

// Vector scale (multiply each element by scalar, same format)
template <int I, int F>
inline Vec3Q<I, F> vec3_scale(Vec3Q<I, F> v, QFixed<I, F> s) {
    return { mul(v.x(), s), mul(v.y(), s), mul(v.z(), s) };
}

// Cross-format vector scale: Vec3<IA,FA> * Q<IB,FB> -> Vec3<IO,FO>
template <int IO, int FO, int IA, int FA, int IB, int FB>
inline Vec3Q<IO, FO> vec3_scale_cross(Vec3Q<IA, FA> v, QFixed<IB, FB> s) {
    return {
        mul_cross<IO, FO>(v.x(), s),
        mul_cross<IO, FO>(v.y(), s),
        mul_cross<IO, FO>(v.z(), s)
    };
}

// Dot product: Vec3<I,F> . Vec3<I,F> -> Q<I,F>
template <int I, int F>
inline QFixed<I, F> vec3_dot(Vec3Q<I, F> a, Vec3Q<I, F> b) {
    int64_t acc = 0;
    acc = mac(acc, a.x().raw, b.x().raw);
    acc = mac(acc, a.y().raw, b.y().raw);
    acc = mac(acc, a.z().raw, b.z().raw);
    acc += (static_cast<int64_t>(1) << (F - 1));
    acc >>= F;
    if (acc > INT32_MAX) return QFixed<I, F>(INT32_MAX);
    if (acc < INT32_MIN) return QFixed<I, F>(INT32_MIN);
    return QFixed<I, F>(static_cast<int32_t>(acc));
}

// Cross product: Vec3<I,F> x Vec3<I,F> -> Vec3<I,F>
template <int I, int F>
inline Vec3Q<I, F> vec3_cross(Vec3Q<I, F> a, Vec3Q<I, F> b) {
    return {
        sat_sub(mul(a.y(), b.z()), mul(a.z(), b.y())),
        sat_sub(mul(a.z(), b.x()), mul(a.x(), b.z())),
        sat_sub(mul(a.x(), b.y()), mul(a.y(), b.x()))
    };
}

// Convert Vec3 between formats
template <int IO, int FO, int II, int FI>
inline Vec3Q<IO, FO> vec3_convert(Vec3Q<II, FI> v) {
    return {
        convert<IO, FO>(v.x()),
        convert<IO, FO>(v.y()),
        convert<IO, FO>(v.z())
    };
}

// ---------------------------------------------------------------------------
// 4-Element Quaternion in Q1.30
// Layout: w, x, y, z (scalar-first)
// ---------------------------------------------------------------------------

struct QuatQ {
    Q1_30 w, x, y, z;

    QuatQ() : w(Q1_30::one()), x(), y(), z() {}
    QuatQ(Q1_30 w_, Q1_30 x_, Q1_30 y_, Q1_30 z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    static QuatQ identity() {
        QuatQ q;
        q.w = Q1_30::one();
        q.x = Q1_30::zero();
        q.y = Q1_30::zero();
        q.z = Q1_30::zero();
        return q;
    }

    // Access vec part as Vec3
    Vec3_Q1_30 vec() const { return Vec3_Q1_30(x, y, z); }
};

// Quaternion multiply: q1 * q2 (Hamilton product)
// Uses int64 intermediates to avoid overflow
inline QuatQ quat_mul(QuatQ p, QuatQ q) {
    // w = pw*qw - px*qx - py*qy - pz*qz
    // x = pw*qx + px*qw + py*qz - pz*qy
    // y = pw*qy - px*qz + py*qw + pz*qx
    // z = pw*qz + px*qy - py*qx + pz*qw
    auto compute = [](int32_t a0, int32_t b0,
                      int32_t a1, int32_t b1,
                      int32_t a2, int32_t b2,
                      int32_t a3, int32_t b3) -> Q1_30 {
        int64_t acc = 0;
        acc = mac(acc, a0, b0);
        acc = mac(acc, a1, b1);
        acc = mac(acc, a2, b2);
        acc = mac(acc, a3, b3);
        acc += (1LL << 29); // round
        acc >>= 30;
        if (acc > INT32_MAX) return Q1_30(INT32_MAX);
        if (acc < INT32_MIN) return Q1_30(INT32_MIN);
        return Q1_30(static_cast<int32_t>(acc));
    };

    QuatQ r;
    r.w = compute(p.w.raw, q.w.raw, -p.x.raw, q.x.raw, -p.y.raw, q.y.raw, -p.z.raw, q.z.raw);
    r.x = compute(p.w.raw, q.x.raw,  p.x.raw, q.w.raw,  p.y.raw, q.z.raw, -p.z.raw, q.y.raw);
    r.y = compute(p.w.raw, q.y.raw, -p.x.raw, q.z.raw,  p.y.raw, q.w.raw,  p.z.raw, q.x.raw);
    r.z = compute(p.w.raw, q.z.raw,  p.x.raw, q.y.raw, -p.y.raw, q.x.raw,  p.z.raw, q.w.raw);
    return r;
}

// Quaternion normalize using Newton-Raphson reciprocal sqrt
// ||q||^2 = w^2 + x^2 + y^2 + z^2 in Q1.30
// We need the inverse sqrt and multiply each component
inline QuatQ quat_normalize(QuatQ q) {
    // Compute norm^2 in int64 (Q2.60 before shift)
    int64_t norm2 = 0;
    norm2 = mac(norm2, q.w.raw, q.w.raw);
    norm2 = mac(norm2, q.x.raw, q.x.raw);
    norm2 = mac(norm2, q.y.raw, q.y.raw);
    norm2 = mac(norm2, q.z.raw, q.z.raw);
    // norm2 is in Q2.60 format. Shift to Q2.30 for further processing.
    int32_t n2 = static_cast<int32_t>((norm2 + (1LL << 29)) >> 30); // Q2.30

    if (n2 == 0) return QuatQ::identity();

    // Newton-Raphson for 1/sqrt(n2) in Q1.30
    // Initial estimate using bit tricks: count leading zeros
    int lz = __builtin_clz(static_cast<uint32_t>(n2));
    // n2 is Q2.30; if n2 ~ 1.0 (=2^30), lz=1
    // 1/sqrt(x) for x near 1.0 is near 1.0
    // Rough: 1/sqrt(2^k) = 2^(-k/2)
    // n2 bits: 32 - lz significant bits, so n2 ~ 2^(31-lz)
    // In Q2.30, value = n2 / 2^30, so actual = 2^(31-lz) / 2^30 = 2^(1-lz)
    // 1/sqrt(actual) = 2^((lz-1)/2)
    // In Q1.30: raw = 2^((lz-1)/2 + 30)
    // For lz=1 (n2=1.0): raw = 2^30 = Q1.30 one. Correct!
    int half_exp = (lz - 1) / 2;
    int32_t inv_sqrt = static_cast<int32_t>(1) << (30 + half_exp > 31 ? 31 : 30 + half_exp);
    if (30 + half_exp > 31) inv_sqrt = INT32_MAX;

    // 3 iterations of Newton-Raphson: y' = y * (3 - x*y^2) / 2
    // Working in Q1.30 arithmetic
    for (int iter = 0; iter < 4; ++iter) {
        // y^2 in Q2.60, shift to Q2.30
        int64_t y2 = static_cast<int64_t>(inv_sqrt) * inv_sqrt;
        int32_t y2_30 = static_cast<int32_t>((y2 + (1LL << 29)) >> 30);
        // x*y^2 in Q4.60, shift to Q2.30
        int64_t xy2 = static_cast<int64_t>(n2) * y2_30;
        int32_t xy2_30 = static_cast<int32_t>((xy2 + (1LL << 29)) >> 30);
        // 3 - x*y^2 in Q2.30 (3.0 = 3 << 30, but that overflows int32!)
        // Use int64: three_q = 3 * (1 << 30)
        int64_t three_q = 3LL << 30;
        int64_t factor = three_q - xy2_30; // Q2.30 in int64
        // y * factor / 2 in Q3.60, shift to Q1.30
        int64_t new_y = static_cast<int64_t>(inv_sqrt) * factor;
        // new_y is Q3.60, divide by 2 = shift right 1 more = Q3.59
        // then shift to Q1.30 means >>29 total from Q3.60 → >>30 from Q3.60/2
        new_y >>= 31; // Q3.60 / 2^31 = Q1.30 (roughly)
        if (new_y > INT32_MAX) new_y = INT32_MAX;
        if (new_y < 0) new_y = 1 << 30; // fallback to 1.0
        inv_sqrt = static_cast<int32_t>(new_y);
    }

    // Multiply each component: q_i * inv_sqrt, both Q1.30
    auto scale = [inv_sqrt](int32_t comp) -> Q1_30 {
        int64_t r = static_cast<int64_t>(comp) * inv_sqrt;
        r += (1LL << 29);
        r >>= 30;
        if (r > INT32_MAX) return Q1_30(INT32_MAX);
        if (r < INT32_MIN) return Q1_30(INT32_MIN);
        return Q1_30(static_cast<int32_t>(r));
    };

    QuatQ out;
    out.w = scale(q.w.raw);
    out.x = scale(q.x.raw);
    out.y = scale(q.y.raw);
    out.z = scale(q.z.raw);
    return out;
}

// Quaternion to 3x3 rotation matrix in Q1.30
// R_bw = rotation from body to world
// Returns row-major [9] array of Q1.30
struct Mat3Q {
    Q1_30 m[9]; // row-major: m[row*3+col]

    Q1_30& operator()(int r, int c) { return m[r * 3 + c]; }
    const Q1_30& operator()(int r, int c) const { return m[r * 3 + c]; }

    // Transpose
    Mat3Q transpose() const {
        Mat3Q t;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                t(r, c) = (*this)(c, r);
        return t;
    }
};

inline Mat3Q quat_to_rotmat(QuatQ q) {
    // Standard formula: R = I + 2w*[v]x + 2*[v]x*[v]x
    // Or direct:
    // R00 = 1 - 2(y^2 + z^2)
    // R01 = 2(xy - wz)
    // R02 = 2(xz + wy)  etc.

    // All products via int64, result in Q1.30
    auto prod2 = [](int32_t a, int32_t b) -> Q1_30 {
        int64_t p = static_cast<int64_t>(a) * b;
        // p is Q2.60. We need 2*q_i*q_j which is Q2.60 * 2 = shift one less
        p >>= 29; // Q2.60 >> 29 = Q1.31, but we want Q1.30... do >>30 then <<1
        // Actually: 2*a*b in Q1.30 = (a*b) >> 29
        if (p > INT32_MAX) return Q1_30(INT32_MAX);
        if (p < INT32_MIN) return Q1_30(INT32_MIN);
        return Q1_30(static_cast<int32_t>(p));
    };

    // q component products (2*qi*qj)
    Q1_30 xx2 = prod2(q.x.raw, q.x.raw);
    Q1_30 yy2 = prod2(q.y.raw, q.y.raw);
    Q1_30 zz2 = prod2(q.z.raw, q.z.raw);
    Q1_30 xy2 = prod2(q.x.raw, q.y.raw);
    Q1_30 xz2 = prod2(q.x.raw, q.z.raw);
    Q1_30 yz2 = prod2(q.y.raw, q.z.raw);
    Q1_30 wx2 = prod2(q.w.raw, q.x.raw);
    Q1_30 wy2 = prod2(q.w.raw, q.y.raw);
    Q1_30 wz2 = prod2(q.w.raw, q.z.raw);

    Q1_30 one = Q1_30::one();
    Mat3Q R;
    R(0, 0) = sat_sub(sat_sub(one, yy2), zz2);
    R(0, 1) = sat_sub(xy2, wz2);
    R(0, 2) = sat_add(xz2, wy2);
    R(1, 0) = sat_add(xy2, wz2);
    R(1, 1) = sat_sub(sat_sub(one, xx2), zz2);
    R(1, 2) = sat_sub(yz2, wx2);
    R(2, 0) = sat_sub(xz2, wy2);
    R(2, 1) = sat_add(yz2, wx2);
    R(2, 2) = sat_sub(sat_sub(one, xx2), yy2);
    return R;
}

// Matrix-vector multiply: Mat3(Q1.30) * Vec3<IA,FA> -> Vec3<IO,FO>
template <int IO, int FO, int IA, int FA>
inline Vec3Q<IO, FO> mat3_vec3_mul(const Mat3Q& M, Vec3Q<IA, FA> v) {
    Vec3Q<IO, FO> out;
    for (int r = 0; r < 3; ++r) {
        int64_t acc = 0;
        for (int c = 0; c < 3; ++c) {
            // M(r,c) is Q1.30, v[c] is Q<IA,FA>
            acc = mac(acc, M(r, c).raw, v[c].raw);
        }
        // acc is Q(1+IA).(30+FA), need Q<IO,FO>
        constexpr int shift = 30 + FA - FO;
        if constexpr (shift > 0) {
            acc += (1LL << (shift - 1));
            acc >>= shift;
        } else if constexpr (shift < 0) {
            acc <<= (-shift);
        }
        if (acc > INT32_MAX) acc = INT32_MAX;
        if (acc < INT32_MIN) acc = INT32_MIN;
        out[r] = QFixed<IO, FO>(static_cast<int32_t>(acc));
    }
    return out;
}

// Mat3 * Mat3 multiply (both Q1.30 -> Q1.30)
inline Mat3Q mat3_mul(const Mat3Q& A, const Mat3Q& B) {
    Mat3Q C;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            int64_t acc = 0;
            for (int k = 0; k < 3; ++k) {
                acc = mac(acc, A(r, k).raw, B(k, c).raw);
            }
            acc += (1LL << 29);
            acc >>= 30;
            if (acc > INT32_MAX) acc = INT32_MAX;
            if (acc < INT32_MIN) acc = INT32_MIN;
            C(r, c) = Q1_30(static_cast<int32_t>(acc));
        }
    }
    return C;
}

// Skew-symmetric matrix from vector [v]x in Q1.30
inline Mat3Q skew_symmetric(Vec3_Q1_30 v) {
    Mat3Q S;
    S(0, 0) = Q1_30::zero(); S(0, 1) = sat_neg(v.z()); S(0, 2) = v.y();
    S(1, 0) = v.z();         S(1, 1) = Q1_30::zero();   S(1, 2) = sat_neg(v.x());
    S(2, 0) = sat_neg(v.y()); S(2, 1) = v.x();          S(2, 2) = Q1_30::zero();
    return S;
}

} // namespace fp
} // namespace trajecto
