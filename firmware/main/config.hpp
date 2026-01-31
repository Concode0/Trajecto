/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
 * Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
 * protected under ROK Patent Application No. 10-2025-YYYYYYY.
 * Commercial use requires a separate license from the author.
 */

#pragma once

namespace trajecto {

constexpr int SENSOR_ODR_HZ = 50;
constexpr int BLE_BATCH_SIZE = 3;

constexpr int IMU_TASK_STACK_SIZE = 12 * 1024;
constexpr int IMU_TASK_PRIORITY = 10;
constexpr int IMU_TASK_CORE_ID = 0;

} // namespace trajecto
