#pragma once

namespace trajecto {

// System Configuration
constexpr int SENSOR_ODR_HZ = 50; // Fixed for TCN compatibility
constexpr int BLE_BATCH_SIZE = 3;  // Number of samples per BLE packet

// Task Configuration
constexpr int IMU_TASK_STACK_SIZE = 12 * 1024; // 12KB stack for TFLite
constexpr int IMU_TASK_PRIORITY = 10;
constexpr int IMU_TASK_CORE_ID = 0;

} // namespace trajecto
