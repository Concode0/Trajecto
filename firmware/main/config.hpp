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

namespace trajecto {

constexpr int SENSOR_ODR_HZ = 50;
constexpr int BLE_BATCH_SIZE = 3;

constexpr int IMU_TASK_STACK_SIZE = 12 * 1024;
constexpr int IMU_TASK_PRIORITY = 10;
constexpr int IMU_TASK_CORE_ID = 0;

} // namespace trajecto
