#include <iostream>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"

#include "ring_buffer.hpp"
#include "imu_task.hpp"
#include "ble_task.hpp"

extern "C" {
    void app_main();
}

static const char *TAG = "APP_MAIN";

// Create a global ring buffer instance
// Capacity: 400 samples/second * 5 seconds (arbitrary, adjust as needed)
// 400 * 5 = 2000 samples.
// Each sample is 15 bytes. Total size = 2000 * 15 = 30000 bytes.
// This should be sufficient for temporary buffering before BLE transmission.
RingBuffer sensor_ring_buffer(2000);

void app_main(void) {
    ESP_LOGI(TAG, "TrajectoFW Application Starting...");

    // Initialize IMU and FSR tasks (Producer)
    esp_err_t ret = imu_task_init(&sensor_ring_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize IMU Task, exiting!");
        return;
    }

    // Initialize BLE streaming task (Consumer)
    ret = ble_task_init(&sensor_ring_buffer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize BLE Task, exiting!");
        return;
    }

    ESP_LOGI(TAG, "All tasks initialized successfully. Entering main loop.");

    // Main loop (can be empty or used for other high-level logic)
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000)); // Sleep for 5 seconds
        ESP_LOGI(TAG, "Main loop heartbeat. Buffer items: %zu, Free space: %zu",
                 sensor_ring_buffer.available_items(), sensor_ring_buffer.free_space());
    }
}
