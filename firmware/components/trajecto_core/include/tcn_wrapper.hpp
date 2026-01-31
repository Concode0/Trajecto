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
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "eskf.hpp"
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
