#pragma once

namespace trajecto {

constexpr int SENSOR_ODR_HZ = 50;
constexpr int BLE_BATCH_SIZE = 3;

constexpr int IMU_TASK_STACK_SIZE = 12 * 1024;
constexpr int IMU_TASK_PRIORITY = 10;
constexpr int IMU_TASK_CORE_ID = 0;

} // namespace trajecto
