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

#include "tcn_wrapper.hpp"
#include "fast_math_lut.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#include "esp_log.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "TCNWrapper";

extern const unsigned char tcn_model_tflite[];
extern const unsigned int tcn_model_tflite_len;

namespace trajecto {

TCNWrapper::TCNWrapper() {
    features_.fill(0.0f);
}

TCNWrapper::~TCNWrapper() = default;

bool TCNWrapper::setup() {
    model_ = tflite::GetModel(tcn_model_tflite);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        return false;
    }

    static tflite::MicroMutableOpResolver<32> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddRelu();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddDiv();
    resolver.AddConcatenation();
    resolver.AddMinimum();
    resolver.AddMaximum();
    resolver.AddAbs();
    resolver.AddNeg();
    resolver.AddCos();
    resolver.AddSqrt();
    resolver.AddRsqrt();
    resolver.AddSquare();
    resolver.AddReduceMax();
    resolver.AddMean();
    resolver.AddPack();
    resolver.AddUnpack();
    resolver.AddSplit();
    resolver.AddQuantize();
    resolver.AddDequantize();

    tensor_arena_ = std::make_unique<uint8_t[]>(kTensorArenaSize);

    static tflite::MicroInterpreter static_interpreter(
        model_, resolver, tensor_arena_.get(), kTensorArenaSize);
    interpreter_ = &static_interpreter;

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!");
        return false;
    }

    state_buffers_.resize(TCN_NUM_LAYERS);
    for (int i = 0; i < TCN_NUM_LAYERS; i++) {
        int size = TCN_STATE_DIMS[i].channels * TCN_STATE_DIMS[i].history;
        state_buffers_[i].assign(size, 0.0f);
    }

    return true;
}

void TCNWrapper::extract_features(
    const Eigen::Vector3f& accel_raw,
    const Eigen::Vector3f& gyro_raw,
    float force_raw,
    const ESKF& eskf,
    const Eigen::Matrix<float, 6, 1>& last_innovation,
    bool is_zupt
) {
    const auto& state = eskf.get_state();

    // Z-score normalization for IMU data (matches Python)
    float norm_accel[3];
    float norm_gyro[3];
    float norm_force;

    for (int i = 0; i < 3; i++) norm_accel[i] = (accel_raw[i] - IMU_MEAN[i]) / IMU_STD[i];
    for (int i = 0; i < 3; i++) norm_gyro[i] = (gyro_raw[i] - IMU_MEAN[i + 3]) / IMU_STD[i + 3];
    norm_force = (force_raw - IMU_MEAN[6]) / IMU_STD[6];

    Eigen::Matrix3f R_bw = state.quat.toRotationMatrix();
    Eigen::Matrix3f R_wb = R_bw.transpose();

    // Gravity in body frame: use world gravity vector (negative Z) rotated to body
    // Python: gravity_b_raw = rot_mat_w_to_b @ gravity_w, then scale by GRAVITY_NORM_SCALE
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -GRAVITY_MAGNITUDE);  // Gravity points down in world
    Eigen::Vector3f gravity_b_raw = R_wb * gravity_w;
    Eigen::Vector3f gravity_b_norm = (gravity_b_raw / GRAVITY_MAGNITUDE) * GRAVITY_NORM_SCALE;

    // Pen tip velocity in body frame
    Eigen::Vector3f vel_b_linear = R_wb * state.vel;
    Eigen::Vector3f gyro_corr = gyro_raw - state.gyro_bias;
    Eigen::Vector3f offset;
    offset << PEN_TIP_OFFSET[0], PEN_TIP_OFFSET[1], PEN_TIP_OFFSET[2];
    Eigen::Vector3f vel_tangential = gyro_corr.cross(offset);
    Eigen::Vector3f pen_tip_vel_b = vel_b_linear + vel_tangential;

    // ISOTROPIC velocity normalization (matches Python)
    // Python uses: pen_tip_vel_b / (VEL_STD_L2 + 1e-3)
    Eigen::Vector3f pen_tip_vel_b_norm = pen_tip_vel_b / (VEL_STD_L2 + 1e-3f);

    // Innovation normalization using Allan variance (matches Python)
    // Accel channels (0-2): normalize by max VRW
    // Gyro channels (3-5): normalize by max ARW

    Eigen::Matrix<float, 6, 1> innovation_norm;
    for (int i = 0; i < 3; i++) {
        float val = last_innovation(i) / (MAX_VRW + 1e-3f);
        innovation_norm(i) = std::max(-INNOVATION_CLAMP_RANGE, std::min(INNOVATION_CLAMP_RANGE, val));
    }
    for (int i = 3; i < 6; i++) {
        float val = last_innovation(i) / (MAX_ARW + 1e-3f);
        innovation_norm(i) = std::max(-INNOVATION_CLAMP_RANGE, std::min(INNOVATION_CLAMP_RANGE, val));
    }

    // Build feature vector (order matches Python: base_hybrid_model.py:447-456)
    int idx = 0;
    // 1. gyro_b_norm [3]
    features_[idx++] = norm_gyro[0];
    features_[idx++] = norm_gyro[1];
    features_[idx++] = norm_gyro[2];
    // 2. accel_b_norm [3]
    features_[idx++] = norm_accel[0];
    features_[idx++] = norm_accel[1];
    features_[idx++] = norm_accel[2];
    // 3. force_norm [1]
    features_[idx++] = norm_force;
    // 4. pen_tip_vel_b_norm [3] - isotropic normalized
    features_[idx++] = pen_tip_vel_b_norm(0);
    features_[idx++] = pen_tip_vel_b_norm(1);
    features_[idx++] = pen_tip_vel_b_norm(2);
    // 5. gravity_b_norm [3] - scaled unit vector
    features_[idx++] = gravity_b_norm(0);
    features_[idx++] = gravity_b_norm(1);
    features_[idx++] = gravity_b_norm(2);
    // 6. innovation_norm [6] - Allan variance normalized, clamped
    for (int i = 0; i < 6; i++) features_[idx++] = innovation_norm(i);
}

TCNOutput TCNWrapper::process_step(
    const Eigen::Vector3f& accel_raw,
    const Eigen::Vector3f& gyro_raw,
    float force_raw,
    const ESKF& eskf,
    const Eigen::Matrix<float, 6, 1>& last_innovation,
    bool is_zupt
) {
    TCNOutput result;
    result.valid = true;

    extract_features(accel_raw, gyro_raw, force_raw, eskf, last_innovation, is_zupt);

    TfLiteTensor* input_feat = interpreter_->input(0);
    std::memcpy(input_feat->data.f, features_.data(), features_.size() * sizeof(float));

    for (int i = 0; i < TCN_NUM_LAYERS; i++) {
        TfLiteTensor* input_state = interpreter_->input(1 + i);
        std::memcpy(input_state->data.f, state_buffers_[i].data(), state_buffers_[i].size() * sizeof(float));
    }

    if (!interpreter_) {
        result.valid = false;
        return result;
    }

    if (interpreter_->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed!");
        result.valid = false;
        return result;
    }

    TfLiteTensor* out_vel = interpreter_->output(0);
    TfLiteTensor* out_cov = interpreter_->output(1);
    TfLiteTensor* out_zupt = interpreter_->output(2);

    result.vel_corr[0] = out_vel->data.f[0];
    result.vel_corr[1] = out_vel->data.f[1];
    result.vel_corr[2] = out_vel->data.f[2];

    for(int i=0; i<6; i++) result.R_params[i] = out_cov->data.f[i];

    float zupt_logit = out_zupt->data.f[0];
    result.zupt_prob = fast_sigmoid(zupt_logit);

    for (int i = 0; i < TCN_NUM_LAYERS; i++) {
        TfLiteTensor* out_state = interpreter_->output(3 + i);
        std::memcpy(state_buffers_[i].data(), out_state->data.f, state_buffers_[i].size() * sizeof(float));
    }

    return result;
}

} // namespace trajecto
