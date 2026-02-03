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

#pragma once

#include <array>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "eskf.hpp"
#include "eskf_fixed.hpp"
#include "model_params.hpp"

namespace tflite {
    class MicroInterpreter;
    class Model;
}

namespace trajecto {

struct TCNOutput {
    Eigen::Vector3f vel_corr;
    Eigen::Matrix<float, 6, 1> R_params;
    float zupt_prob;
    bool valid;
};

class TCNWrapper {
public:
    TCNWrapper();
    ~TCNWrapper();

    // Non-copyable, non-movable (owns TFLite interpreter lifecycle)
    TCNWrapper(const TCNWrapper&) = delete;
    TCNWrapper& operator=(const TCNWrapper&) = delete;
    TCNWrapper(TCNWrapper&&) = delete;
    TCNWrapper& operator=(TCNWrapper&&) = delete;

    bool setup();

    TCNOutput process_step(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        float force_raw,
        const ESKF& eskf,
        const Eigen::Matrix<float, 6, 1>& last_innovation,
        bool is_zupt
    );

    TCNOutput process_step_fixed(
        const float accel_raw[3],
        const float gyro_raw[3],
        float force_raw,
        const ESKFFixed& eskf_fixed,
        const float last_innovation[6]
    );

private:
    void extract_features(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        float force_raw,
        const ESKF& eskf,
        const Eigen::Matrix<float, 6, 1>& last_innovation,
        bool is_zupt
    );

    const tflite::Model* model_ = nullptr;                // Non-owning: points to flash data
    tflite::MicroInterpreter* interpreter_ = nullptr;     // Non-owning: points to static local
    std::unique_ptr<uint8_t[]> tensor_arena_;

    std::vector<std::vector<float>> state_buffers_;
    std::array<float, TCN_INPUT_SIZE> features_;           // Pre-allocated feature buffer

    static constexpr int kTensorArenaSize = 120 * 1024;
    static constexpr int kInputSize = TCN_INPUT_SIZE;
};

} // namespace trajecto
