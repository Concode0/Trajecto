#ifndef BLE_TASK_HPP
#define BLE_TASK_HPP

#include "ring_buffer.hpp"
#include <esp_err.h>

// BLE Task
#define BLE_TASK_PRIORITY           (configMAX_PRIORITIES - 10) // Low priority
#define BLE_TASK_STACK_SIZE         4096

// BLE Service and Characteristic UUIDs (Generated UUIDs)
// Base UUID: 0000xxxx-0000-1000-8000-00805F9B34FB
#define BLE_SVC_UUID_SENSOR_DATA    0xABCD // Custom Service UUID for Sensor Data
#define BLE_CHR_UUID_SENSOR_DATA    0xABCE // Custom Characteristic UUID for Sensor Data

// Number of sensor samples to batch per BLE notification
// sensor_sample_t is 15 bytes.
// With an MTU of 247, and ATT header of 3 bytes, max payload is 244 bytes.
// 244 / 15 = 16 samples per notification (240 bytes)
#define BLE_SAMPLES_PER_NOTIFICATION (244 / sizeof(sensor_sample_t))

/**
 * @brief Initializes the BLE stack and starts the BLE advertising and streaming task.
 *
 * @param buffer Pointer to the RingBuffer instance to read sensor data from.
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t ble_task_init(RingBuffer *buffer);

#endif // BLE_TASK_HPP
