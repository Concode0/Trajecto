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

#include "fixed_point.hpp"
#include <cstring>

namespace trajecto {
namespace fp {

// ---------------------------------------------------------------------------
// ScaledBlock3x3: Block floating-point 3x3 matrix
// ---------------------------------------------------------------------------
// Stores 9 int32_t mantissa values with a shared power-of-2 exponent.
// actual_value[i] = m[i] * 2^exponent
// This allows the same int32 dynamic range to adapt to different scales
// across the covariance matrix.
// ---------------------------------------------------------------------------

struct ScaledBlock3x3 {
    int32_t m[9];      // row-major mantissa values
    int8_t  exponent;  // shared 2^exponent scale

    ScaledBlock3x3() : exponent(0) {
        for (int i = 0; i < 9; ++i) m[i] = 0;
    }

    // Access element
    int32_t& operator()(int r, int c) { return m[r * 3 + c]; }
    const int32_t& operator()(int r, int c) const { return m[r * 3 + c]; }

    // Create identity block scaled to given exponent
    static ScaledBlock3x3 identity(int8_t exp) {
        ScaledBlock3x3 b;
        b.exponent = exp;
        b(0, 0) = INT32_MAX;
        b(1, 1) = INT32_MAX;
        b(2, 2) = INT32_MAX;
        return b;
    }

    // Create diagonal block
    static ScaledBlock3x3 diagonal(int32_t d0, int32_t d1, int32_t d2, int8_t exp) {
        ScaledBlock3x3 b;
        b.exponent = exp;
        b(0, 0) = d0;
        b(1, 1) = d1;
        b(2, 2) = d2;
        return b;
    }

    // Create from float values (for initialization)
    static ScaledBlock3x3 from_float_diag(float d0, float d1, float d2) {
        // Find the max magnitude to determine exponent
        float max_val = 0.0f;
        float vals[3] = {d0, d1, d2};
        for (int i = 0; i < 3; ++i) {
            float av = vals[i] < 0 ? -vals[i] : vals[i];
            if (av > max_val) max_val = av;
        }

        ScaledBlock3x3 b;
        if (max_val == 0.0f) return b;

        // Choose exponent so max_val maps near INT32_MAX/2
        // max_val = mantissa * 2^exp, mantissa near 2^30
        // exp = log2(max_val) - 30
        int exp_bits = 0;
        float tmp = max_val;
        while (tmp >= 2.0f) { tmp *= 0.5f; exp_bits++; }
        while (tmp < 1.0f) { tmp *= 2.0f; exp_bits--; }
        b.exponent = static_cast<int8_t>(exp_bits - 30);

        float scale = 1.0f;
        int8_t e = b.exponent;
        if (e > 0) {
            for (int i = 0; i < e; ++i) scale *= 2.0f;
        } else {
            for (int i = 0; i < -e; ++i) scale *= 0.5f;
        }
        float inv_scale = 1.0f / scale;

        for (int i = 0; i < 9; ++i) b.m[i] = 0;
        b(0, 0) = static_cast<int32_t>(d0 * inv_scale + 0.5f);
        b(1, 1) = static_cast<int32_t>(d1 * inv_scale + 0.5f);
        b(2, 2) = static_cast<int32_t>(d2 * inv_scale + 0.5f);
        return b;
    }

    // Convert element to float
    float to_float(int r, int c) const {
        float scale = 1.0f;
        int8_t e = exponent;
        if (e > 0) {
            for (int i = 0; i < e; ++i) scale *= 2.0f;
        } else {
            for (int i = 0; i < -e; ++i) scale *= 0.5f;
        }
        return static_cast<float>(m[r * 3 + c]) * scale;
    }
};

// ---------------------------------------------------------------------------
// Block Renormalization
// ---------------------------------------------------------------------------
// Shift mantissas to maximize dynamic range (fill int32 range)
// This prevents precision loss after operations.

inline void block_renormalize(ScaledBlock3x3& b) {
    // Find max absolute mantissa value
    int32_t max_abs = 0;
    for (int i = 0; i < 9; ++i) {
        int32_t av = b.m[i] < 0 ? -b.m[i] : b.m[i];
        if (b.m[i] == INT32_MIN) av = INT32_MAX; // handle overflow
        if (av > max_abs) max_abs = av;
    }

    if (max_abs == 0) return; // all zeros

    // Count leading zeros to find how much we can shift left
    int lz = __builtin_clz(static_cast<uint32_t>(max_abs));
    // lz=1 means max_abs uses 31 bits (full). lz=2 means 30 bits. etc.
    // We want max_abs to use ~30 bits (leave 1 sign bit + 1 guard bit)
    int shift = lz - 2;  // positive = shift left, negative = shift right

    if (shift > 0) {
        for (int i = 0; i < 9; ++i) b.m[i] <<= shift;
        b.exponent -= static_cast<int8_t>(shift);
    } else if (shift < 0) {
        shift = -shift;
        for (int i = 0; i < 9; ++i) {
            b.m[i] = (b.m[i] + (1 << (shift - 1))) >> shift;
        }
        b.exponent += static_cast<int8_t>(shift);
    }
}

// ---------------------------------------------------------------------------
// Block Arithmetic
// ---------------------------------------------------------------------------

// Block add: C = A + B (align exponents first)
inline ScaledBlock3x3 block_add(const ScaledBlock3x3& A, const ScaledBlock3x3& B) {
    ScaledBlock3x3 C;

    // Align to the larger exponent (less precision but no overflow)
    if (A.exponent >= B.exponent) {
        int shift = A.exponent - B.exponent;
        C.exponent = A.exponent;
        for (int i = 0; i < 9; ++i) {
            int32_t b_shifted = (shift < 32) ? (B.m[i] >> shift) : 0;
            int64_t sum = static_cast<int64_t>(A.m[i]) + b_shifted;
            if (sum > INT32_MAX) sum = INT32_MAX;
            if (sum < INT32_MIN) sum = INT32_MIN;
            C.m[i] = static_cast<int32_t>(sum);
        }
    } else {
        int shift = B.exponent - A.exponent;
        C.exponent = B.exponent;
        for (int i = 0; i < 9; ++i) {
            int32_t a_shifted = (shift < 32) ? (A.m[i] >> shift) : 0;
            int64_t sum = static_cast<int64_t>(a_shifted) + B.m[i];
            if (sum > INT32_MAX) sum = INT32_MAX;
            if (sum < INT32_MIN) sum = INT32_MIN;
            C.m[i] = static_cast<int32_t>(sum);
        }
    }

    block_renormalize(C);
    return C;
}

// Block subtract: C = A - B
inline ScaledBlock3x3 block_sub(const ScaledBlock3x3& A, const ScaledBlock3x3& B) {
    ScaledBlock3x3 negB;
    negB.exponent = B.exponent;
    for (int i = 0; i < 9; ++i) {
        negB.m[i] = (B.m[i] == INT32_MIN) ? INT32_MAX : -B.m[i];
    }
    return block_add(A, negB);
}

// Block multiply: C = A * B (3x3 matrix multiply)
// A mantissa has exponent eA, B has eB. Result exponent = eA + eB + (shift adjustment)
inline ScaledBlock3x3 block_mul(const ScaledBlock3x3& A, const ScaledBlock3x3& B) {
    ScaledBlock3x3 C;

    // Each element: C(r,c) = sum_k A(r,k)*B(k,c)
    // Products are int64. We need to find result exponent.
    // A values ~ 2^30 * 2^eA, B values ~ 2^30 * 2^eB
    // Product ~ 2^60 * 2^(eA+eB)
    // We want result mantissa ~ 2^30, so shift down by 30
    // Result exponent = eA + eB + 30

    int64_t tmp[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            int64_t acc = 0;
            for (int k = 0; k < 3; ++k) {
                acc += static_cast<int64_t>(A.m[r * 3 + k]) * B.m[k * 3 + c];
            }
            tmp[r * 3 + c] = acc;
        }
    }

    // Find max |tmp| to determine shift
    int64_t max_abs = 0;
    for (int i = 0; i < 9; ++i) {
        int64_t av = tmp[i] < 0 ? -tmp[i] : tmp[i];
        if (av > max_abs) max_abs = av;
    }

    if (max_abs == 0) {
        C.exponent = 0;
        return C;
    }

    // Count leading zeros of max_abs in 64 bits
    int lz64 = __builtin_clzll(static_cast<uint64_t>(max_abs));
    // We want mantissa to use ~30 bits. max_abs has (64-lz64) significant bits.
    // Shift right by (64-lz64-30) = (34-lz64) to get ~30-bit mantissa
    int shift_right = 34 - lz64;
    if (shift_right < 0) shift_right = 0;

    for (int i = 0; i < 9; ++i) {
        if (shift_right > 0) {
            C.m[i] = static_cast<int32_t>((tmp[i] + (1LL << (shift_right - 1))) >> shift_right);
        } else {
            int64_t v = tmp[i] << (-shift_right);
            if (v > INT32_MAX) v = INT32_MAX;
            if (v < INT32_MIN) v = INT32_MIN;
            C.m[i] = static_cast<int32_t>(v);
        }
    }

    // Exponent: product is in 2^(eA + eB) units for int64 mantissa product
    // After shifting right by shift_right, exponent increases by shift_right
    C.exponent = static_cast<int8_t>(
        static_cast<int>(A.exponent) + static_cast<int>(B.exponent) + shift_right
    );

    block_renormalize(C);
    return C;
}

// Block transpose
inline ScaledBlock3x3 block_transpose(const ScaledBlock3x3& A) {
    ScaledBlock3x3 T;
    T.exponent = A.exponent;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T.m[r * 3 + c] = A.m[c * 3 + r];
    return T;
}

// Block scale by scalar (with exponent adjustment)
// Multiply all mantissas by a scalar int32 and adjust exponent
inline ScaledBlock3x3 block_scale_int(const ScaledBlock3x3& A, int32_t scalar, int8_t scalar_exp) {
    ScaledBlock3x3 C;
    int64_t tmp[9];
    int64_t max_abs = 0;

    for (int i = 0; i < 9; ++i) {
        tmp[i] = static_cast<int64_t>(A.m[i]) * scalar;
        int64_t av = tmp[i] < 0 ? -tmp[i] : tmp[i];
        if (av > max_abs) max_abs = av;
    }

    if (max_abs == 0) {
        C.exponent = 0;
        return C;
    }

    int lz64 = __builtin_clzll(static_cast<uint64_t>(max_abs));
    int shift_right = 34 - lz64;
    if (shift_right < 0) shift_right = 0;

    for (int i = 0; i < 9; ++i) {
        if (shift_right > 0)
            C.m[i] = static_cast<int32_t>((tmp[i] + (1LL << (shift_right - 1))) >> shift_right);
        else
            C.m[i] = static_cast<int32_t>(tmp[i] << (-shift_right));
    }

    C.exponent = static_cast<int8_t>(
        static_cast<int>(A.exponent) + static_cast<int>(scalar_exp) + shift_right
    );
    block_renormalize(C);
    return C;
}

// ---------------------------------------------------------------------------
// 3x3 Inverse via Cramer's Rule (Adjugate / Determinant)
// ---------------------------------------------------------------------------
// Works on ScaledBlock3x3. Returns success flag.
// If near-singular, returns false.

inline bool block_inverse_3x3(const ScaledBlock3x3& A, ScaledBlock3x3& out) {
    // Compute cofactors using int64 arithmetic
    // All values in A have exponent A.exponent

    auto cofactor = [&](int r0, int c0, int r1, int c1) -> int64_t {
        return static_cast<int64_t>(A.m[r0 * 3 + c0]) * A.m[r1 * 3 + c1]
             - static_cast<int64_t>(A.m[r0 * 3 + c1]) * A.m[r1 * 3 + c0];
    };

    // Cofactor matrix (adjugate transposed)
    int64_t cof[9];
    cof[0] = cofactor(1, 1, 2, 2);  // C00
    cof[1] = -cofactor(1, 0, 2, 2); // C01 (transposed: C10)
    cof[2] = cofactor(1, 0, 2, 1);  // C02 (transposed: C20)
    cof[3] = -cofactor(0, 1, 2, 2); // C10 (transposed: C01)
    cof[4] = cofactor(0, 0, 2, 2);  // C11
    cof[5] = -cofactor(0, 0, 2, 1); // C12 (transposed: C21)
    cof[6] = cofactor(0, 1, 1, 2);  // C20 (transposed: C02)
    cof[7] = -cofactor(0, 0, 1, 2); // C21 (transposed: C12)
    cof[8] = cofactor(0, 0, 1, 1);  // C22

    // Determinant = A(0,0)*cof[0] + A(0,1)*cof[3] + A(0,2)*cof[6]
    // Wait - the adjugate is the transpose of the cofactor matrix.
    // adj(A)(r,c) = cofactor(c,r)
    // det = sum_j A(0,j) * cofactor(0,j)
    // Let me recalculate using the proper cofactor layout:

    // Cofactor C(i,j) = (-1)^(i+j) * M(i,j) where M is minor
    // C(0,0) = +(A11*A22 - A12*A21) = cofactor(1,1,2,2)
    // C(0,1) = -(A10*A22 - A12*A20)
    // C(0,2) = +(A10*A21 - A11*A20)
    // C(1,0) = -(A01*A22 - A02*A21)
    // C(1,1) = +(A00*A22 - A02*A20)
    // C(1,2) = -(A00*A21 - A01*A20)
    // C(2,0) = +(A01*A12 - A02*A11)
    // C(2,1) = -(A00*A12 - A02*A10)
    // C(2,2) = +(A00*A11 - A01*A10)

    // adj(A)(r,c) = C(c,r) - so adj is transpose of cofactor matrix
    // A^{-1} = adj(A) / det

    // Let's recompute properly:
    int64_t C00 =  cofactor(1, 1, 2, 2);
    int64_t C01 = -cofactor(1, 0, 2, 2);
    int64_t C02 =  cofactor(1, 0, 2, 1);
    int64_t C10 = -cofactor(0, 1, 2, 2);
    int64_t C11 =  cofactor(0, 0, 2, 2);
    int64_t C12 = -cofactor(0, 0, 2, 1);
    int64_t C20 =  cofactor(0, 1, 1, 2);
    int64_t C21 = -cofactor(0, 0, 1, 2);
    int64_t C22 =  cofactor(0, 0, 1, 1);

    // Determinant
    // det = A00*C00 + A01*C01 + A02*C02
    // This would be int64 * int64 = int128... too wide.
    // Instead, scale cofactors down first.

    // Cofactors are products of two A mantissas (~ 2^30 each), so ~ 2^60 range
    // Shift cofactors to ~30-bit range
    int64_t cof_max = 0;
    int64_t cofs[9] = {C00, C01, C02, C10, C11, C12, C20, C21, C22};
    for (int i = 0; i < 9; ++i) {
        int64_t av = cofs[i] < 0 ? -cofs[i] : cofs[i];
        if (av > cof_max) cof_max = av;
    }

    if (cof_max == 0) return false;

    int lz = __builtin_clzll(static_cast<uint64_t>(cof_max));
    int cof_shift = 34 - lz; // shift to get ~30-bit mantissas
    if (cof_shift < 0) cof_shift = 0;

    int32_t adj[9]; // adjugate = transpose of cofactor
    for (int i = 0; i < 9; ++i) {
        if (cof_shift > 0)
            adj[i] = static_cast<int32_t>((cofs[i] + (1LL << (cof_shift - 1))) >> cof_shift);
        else
            adj[i] = static_cast<int32_t>(cofs[i]);
    }

    // Transpose to get adjugate: adj(A)(r,c) = C(c,r)
    // cofs are already in cofactor order [C00,C01,C02,C10,...,C22]
    // adj[r*3+c] should be C(c,r)
    int32_t adj_t[9];
    adj_t[0] = adj[0]; adj_t[1] = adj[3]; adj_t[2] = adj[6];
    adj_t[3] = adj[1]; adj_t[4] = adj[4]; adj_t[5] = adj[7];
    adj_t[6] = adj[2]; adj_t[7] = adj[5]; adj_t[8] = adj[8];

    // Determinant from scaled cofactors: det_scaled = A00*adj[0] + A01*adj[1] + A02*adj[2]
    // Wait, using original cofactor order: det = A(0,0)*C00 + A(0,1)*C01 + A(0,2)*C02
    // In adj[] (before transpose): adj[0]=C00_scaled, adj[1]=C01_scaled, adj[2]=C02_scaled
    int64_t det_scaled = static_cast<int64_t>(A.m[0]) * adj[0]
                       + static_cast<int64_t>(A.m[1]) * adj[1]
                       + static_cast<int64_t>(A.m[2]) * adj[2];

    if (det_scaled == 0) return false;

    // Now we need A^{-1} = adj_t / det
    // adj_t elements are ~30-bit, det_scaled is ~60-bit (30-bit A * 30-bit adj)
    // Result should have ~30-bit mantissas

    // A^{-1}(r,c) = adj_t[r*3+c] / det_scaled
    // = adj_t[r*3+c] * (2^K / det_scaled) >> K for some K
    // This is integer division. For best precision:
    // Shift adj_t left by as much as possible before dividing

    // Find max |adj_t|
    int32_t adj_max = 0;
    for (int i = 0; i < 9; ++i) {
        int32_t av = adj_t[i] < 0 ? -adj_t[i] : adj_t[i];
        if (av > adj_max) adj_max = av;
    }

    if (adj_max == 0) return false;

    // Shift adj_t left to ~60 bits for division
    int adj_lz = __builtin_clz(static_cast<uint32_t>(adj_max));
    int adj_upshift = adj_lz - 2; // leave 2 guard bits
    if (adj_upshift < 0) adj_upshift = 0;

    for (int i = 0; i < 9; ++i) {
        // adj_t[i] << adj_upshift, then divide by det_scaled (which is ~60-bit)
        int64_t numerator = static_cast<int64_t>(adj_t[i]) << (adj_upshift + 30);
        out.m[i] = static_cast<int32_t>(numerator / det_scaled);
    }

    // Exponent: A has exponent eA. Cofactors have exponent 2*eA (product of two A elements).
    // After cof_shift, scaled cofactors have exponent 2*eA + cof_shift.
    // adj_t has same exponent as cofactors (just transposed): 2*eA + cof_shift.
    // det = A * cofactor, so det has exponent eA + (2*eA + cof_shift) = 3*eA + cof_shift.
    // A^{-1} = adj_t / det
    //        exponent = adj_t_exp - det_exp + adjustment_for_shifts
    //        = (2*eA + cof_shift) - (3*eA + cof_shift) + (adj_upshift + 30) [from << in numerator]
    //        = -eA + adj_upshift + 30
    // But wait, we shifted adj_t left by (adj_upshift + 30) before dividing.
    // Result mantissa ~ adj_t * 2^(adj_upshift+30) / det
    // adj_t physical = adj_t_mantissa * 2^(2*eA + cof_shift)
    // det physical = det_mantissa * 2^(3*eA + cof_shift) [but det_mantissa here is det_scaled which is in mixed units]
    //
    // Let me think more carefully:
    // Physical cofactor = cof_int64 * 2^(2*eA) [product of two A-mantissas, each with eA exponent]
    // After cof_shift: adj[] = cof_int64 >> cof_shift, so physical adj[] = adj[] * 2^(2*eA + cof_shift)
    // Physical det = A_mantissa * adj_mantissa * 2^(eA + 2*eA + cof_shift) = det_scaled * 2^(3*eA + cof_shift)
    // Physical inverse element = (adj_t * 2^(adj_upshift+30) / det_scaled) * 2^(2*eA+cof_shift) / 2^(3*eA+cof_shift) / 2^(adj_upshift+30)
    //                          = out.m * 2^(2*eA+cof_shift - 3*eA - cof_shift - adj_upshift - 30)
    //                          = out.m * 2^(-eA - adj_upshift - 30)
    out.exponent = static_cast<int8_t>(
        -static_cast<int>(A.exponent) - adj_upshift - 30
    );

    block_renormalize(out);
    return true;
}

// ---------------------------------------------------------------------------
// Block Covariance Matrix (5x5 grid of 3x3 blocks = 15x15)
// ---------------------------------------------------------------------------
// Stores upper triangle; lower is transpose.
// Block indices: 0=pos, 1=vel, 2=orient, 3=gyro_bias, 4=accel_bias

struct BlockCovMatrix {
    static constexpr int N_BLOCKS = 5;
    ScaledBlock3x3 blocks[N_BLOCKS][N_BLOCKS]; // [row_block][col_block]

    BlockCovMatrix() = default;

    // Access block (returns reference for upper triangle, copy for lower)
    ScaledBlock3x3& at(int r, int c) { return blocks[r][c]; }
    const ScaledBlock3x3& at(int r, int c) const { return blocks[r][c]; }

    // Initialize as scaled identity
    void set_scaled_identity(float scale) {
        for (int i = 0; i < N_BLOCKS; ++i) {
            for (int j = 0; j < N_BLOCKS; ++j) {
                blocks[i][j] = ScaledBlock3x3();
            }
            blocks[i][i] = ScaledBlock3x3::from_float_diag(scale, scale, scale);
        }
    }

    // Enforce symmetry: copy upper triangle to lower (transposed)
    void enforce_symmetry() {
        for (int i = 0; i < N_BLOCKS; ++i) {
            for (int j = i + 1; j < N_BLOCKS; ++j) {
                blocks[j][i] = block_transpose(blocks[i][j]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// BlockCovMatrix operations for ESKF
// ---------------------------------------------------------------------------

// P = F * P * F^T + Q
// F for ESKF is sparse: only specific off-diagonal blocks are non-identity.
// F block structure (15x15 = 5x5 blocks of 3x3):
//   Block(0,1) = I*dt          (pos <- vel)
//   Block(1,2) = -R*[a]x*dt   (vel <- orient)
//   Block(1,4) = -R*dt         (vel <- accel_bias)
//   Block(2,3) = -I*dt         (orient <- gyro_bias)
// All diagonal blocks are I. All other off-diagonal blocks are 0.
//
// Rather than doing full 5x5 block multiply, we exploit this sparsity.
// This is implemented in eskf_fixed.cpp as part of the predict function.

// ---------------------------------------------------------------------------
// 6x6 Block Inverse via Schur Complement
// ---------------------------------------------------------------------------
// For S = [A B; C D] where A,B,C,D are 3x3 blocks:
// S^{-1} = [A^{-1} + A^{-1}*B*Sinv*C*A^{-1},  -A^{-1}*B*Sinv;
//           -Sinv*C*A^{-1},                      Sinv]
// where Sinv = (D - C*A^{-1}*B)^{-1} (Schur complement of A)

struct Block6x6 {
    ScaledBlock3x3 blocks[2][2]; // [row][col]
};

inline bool block_inverse_6x6(const Block6x6& S, Block6x6& out) {
    // A = S[0][0], B = S[0][1], C = S[1][0], D = S[1][1]
    ScaledBlock3x3 Ainv;
    if (!block_inverse_3x3(S.blocks[0][0], Ainv)) return false;

    // Schur = D - C * Ainv * B
    ScaledBlock3x3 AinvB = block_mul(Ainv, S.blocks[0][1]);
    ScaledBlock3x3 CAinvB = block_mul(S.blocks[1][0], AinvB);
    ScaledBlock3x3 Schur = block_sub(S.blocks[1][1], CAinvB);

    ScaledBlock3x3 Sinv;
    if (!block_inverse_3x3(Schur, Sinv)) return false;

    // CAinv = C * Ainv
    ScaledBlock3x3 CAinv = block_mul(S.blocks[1][0], Ainv);

    // out[0][0] = Ainv + AinvB * Sinv * CAinv
    ScaledBlock3x3 SinvCAinv = block_mul(Sinv, CAinv);
    out.blocks[0][0] = block_add(Ainv, block_mul(AinvB, SinvCAinv));

    // out[0][1] = -AinvB * Sinv
    ScaledBlock3x3 AinvBSinv = block_mul(AinvB, Sinv);
    out.blocks[0][1] = AinvBSinv;
    for (int i = 0; i < 9; ++i) {
        out.blocks[0][1].m[i] = (out.blocks[0][1].m[i] == INT32_MIN)
            ? INT32_MAX : -out.blocks[0][1].m[i];
    }

    // out[1][0] = -Sinv * CAinv
    out.blocks[1][0] = SinvCAinv;
    for (int i = 0; i < 9; ++i) {
        out.blocks[1][0].m[i] = (out.blocks[1][0].m[i] == INT32_MIN)
            ? INT32_MAX : -out.blocks[1][0].m[i];
    }

    // out[1][1] = Sinv
    out.blocks[1][1] = Sinv;

    return true;
}

// ---------------------------------------------------------------------------
// CORDIC sin/cos (declared, implemented in cordic_tables.cpp)
// ---------------------------------------------------------------------------
void cordic_sincos(int32_t angle_q30, int32_t& sin_out, int32_t& cos_out);

// ---------------------------------------------------------------------------
// Mat3Q <-> ScaledBlock3x3 conversion (pure integer)
// ---------------------------------------------------------------------------

// Convert Mat3Q (Q2.30 elements) to ScaledBlock3x3
// Mat3Q element physical = raw * 2^(-30), so exponent = -30
inline ScaledBlock3x3 mat3q_to_block(const Mat3Q& M) {
    ScaledBlock3x3 b;
    for (int i = 0; i < 9; ++i) b.m[i] = M.m[i].raw;
    b.exponent = -30;
    block_renormalize(b);
    return b;
}

// Build skew-symmetric ScaledBlock3x3 from Vec3_Q16_15
// [  0  -vz  vy ]
// [ vz   0  -vx ]
// [-vy  vx   0  ]
inline ScaledBlock3x3 vec3q15_to_skew_block(const Vec3Q<17, 15>& v) {
    ScaledBlock3x3 b;
    b.m[0] = 0;          b.m[1] = -v.z().raw; b.m[2] = v.y().raw;
    b.m[3] = v.z().raw;  b.m[4] = 0;          b.m[5] = -v.x().raw;
    b.m[6] = -v.y().raw; b.m[7] = v.x().raw;  b.m[8] = 0;
    // Handle INT32_MIN edge case for negations
    if (v.z().raw == INT32_MIN) { b.m[1] = INT32_MAX; }
    if (v.x().raw == INT32_MIN) { b.m[5] = INT32_MAX; }
    if (v.y().raw == INT32_MIN) { b.m[6] = INT32_MAX; }
    b.exponent = -15; // Q17.15 implicit exponent
    block_renormalize(b);
    return b;
}

// ---------------------------------------------------------------------------
// ScaledBlock3x3 * Vec3 multiplication (pure integer)
// ---------------------------------------------------------------------------
// Multiplies a ScaledBlock3x3 by a vector whose raw values have a known
// fractional bit count, producing output in a target Q format.
//
// Physics:
//   M element physical  = M.m[i] * 2^(M.exponent)
//   vec element physical = v_raw[c] * 2^(-v_frac)
//   result physical      = (sum M.m[r*3+c] * v_raw[c]) * 2^(M.exponent - v_frac)
//   target output        = out_raw * 2^(-out_frac)
//   => out_raw           = acc * 2^(M.exponent - v_frac + out_frac)
//   => shift             = M.exponent + (out_frac - v_frac)

inline void block_vec3_mul_raw(
    const ScaledBlock3x3& M,
    const int32_t v_raw[3], int v_frac,
    int32_t out_raw[3], int out_frac
) {
    int shift = static_cast<int>(M.exponent) + (out_frac - v_frac);

    for (int r = 0; r < 3; ++r) {
        int64_t acc = 0;
        for (int c = 0; c < 3; ++c) {
            acc += static_cast<int64_t>(M.m[r * 3 + c]) * v_raw[c];
        }

        if (shift > 0) {
            // Check for overflow before shifting
            int64_t limit = static_cast<int64_t>(INT32_MAX) >> shift;
            if (acc > limit) { out_raw[r] = INT32_MAX; continue; }
            if (acc < -limit) { out_raw[r] = INT32_MIN; continue; }
            acc <<= shift;
        } else if (shift < 0) {
            int rs = -shift;
            acc = (acc + (1LL << (rs - 1))) >> rs;
        }

        if (acc > INT32_MAX) acc = INT32_MAX;
        if (acc < INT32_MIN) acc = INT32_MIN;
        out_raw[r] = static_cast<int32_t>(acc);
    }
}

// ScaledBlock3x3 * Vec3_Q4_27 -> Vec3_Q4_27 (positions, velocities, accel_bias)
inline Vec3Q<5, 27> block_mul_vec3_q427(const ScaledBlock3x3& M, Vec3Q<5, 27> v) {
    int32_t v_raw[3] = {v.x().raw, v.y().raw, v.z().raw};
    int32_t o_raw[3];
    block_vec3_mul_raw(M, v_raw, 27, o_raw, 27);
    return { QFixed<5,27>(o_raw[0]), QFixed<5,27>(o_raw[1]), QFixed<5,27>(o_raw[2]) };
}

// ScaledBlock3x3 * Vec3_Q4_27 -> Vec3_Q1_30 (orientation, gyro_bias)
inline Vec3Q<2, 30> block_mul_vec3_q130(const ScaledBlock3x3& M, Vec3Q<5, 27> v) {
    int32_t v_raw[3] = {v.x().raw, v.y().raw, v.z().raw};
    int32_t o_raw[3];
    block_vec3_mul_raw(M, v_raw, 27, o_raw, 30);
    return { QFixed<2,30>(o_raw[0]), QFixed<2,30>(o_raw[1]), QFixed<2,30>(o_raw[2]) };
}

// ---------------------------------------------------------------------------
// Fixed-point Newton-Raphson integer square root
// ---------------------------------------------------------------------------
// Returns sqrt(x) where x is a positive int32 with `frac` fractional bits.
// Result has `frac` fractional bits. Uses 5 iterations.
inline int32_t isqrt_fixed(int32_t x, int frac) {
    if (x <= 0) return 0;

    // Initial estimate via bit-shift: sqrt(2^k) = 2^(k/2)
    int bits = 32 - __builtin_clz(static_cast<uint32_t>(x));
    int32_t y = 1 << ((bits + frac) / 2); // rough initial guess

    // Newton-Raphson: y = (y + x/y) / 2
    for (int i = 0; i < 5; ++i) {
        if (y == 0) return 0;
        // x/y in fixed-point: (x << frac) / y, then result has frac bits
        int64_t x_shifted = static_cast<int64_t>(x) << frac;
        int32_t div = static_cast<int32_t>(x_shifted / y);
        y = (y + div) >> 1;
    }
    return y;
}

} // namespace fp
} // namespace trajecto
