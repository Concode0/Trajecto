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
