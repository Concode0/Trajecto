#ifndef IMU_TASK_HPP
#define IMU_TASK_HPP

#include "sensor_data.hpp"
#include "ring_buffer.hpp"
#include "bmi270.h" // Bosch BMI270 driver
#include "driver/i2c.h" // ESP-IDF I2C driver
#include "esp_timer.h" // High-precision timer

// I2C defines for BMI270
#define I2C_MASTER_SCL_IO           GPIO_NUM_10  // SCL pin
#define I2C_MASTER_SDA_IO           GPIO_NUM_8   // SDA pin
#define I2C_MASTER_NUM              I2C_NUM_0    // I2C port number
#define I2C_MASTER_FREQ_HZ          400000       // I2C clock frequency (400kHz Fast Mode)
#define BMI270_SENSOR_ADDR          BMI270_I2C_ADDRESS // BMI270 I2C address

// IMU Task
#define IMU_TASK_PRIORITY           (configMAX_PRIORITIES - 5) // High priority
#define IMU_TASK_STACK_SIZE         4096
#define IMU_TASK_PERIOD_US          (1000000 / 400) // 400Hz update rate

/**
 * @brief Initializes the IMU sensor (BMI270) and starts the data acquisition task.
 *
 * @param buffer Pointer to the RingBuffer instance to write sensor data to.
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t imu_task_init(RingBuffer *buffer);

#endif // IMU_TASK_HPP
