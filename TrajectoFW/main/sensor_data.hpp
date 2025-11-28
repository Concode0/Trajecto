#ifndef SENSOR_DATA_HPP
#define SENSOR_DATA_HPP

#include <stdint.h>

/**
 * @brief Structure to hold a single sensor data sample.
 */
typedef struct __attribute__((packed)) {
    uint16_t timestamp_delta; /**< Time difference from previous sample, in microseconds */
    int16_t accel_x;          /**< Raw accelerometer X-axis value */
    int16_t accel_y;          /**< Raw accelerometer Y-axis value */
    int16_t accel_z;          /**< Raw accelerometer Z-axis value */
    int16_t gyro_x;           /**< Raw gyroscope X-axis value */
    int16_t gyro_y;           /**< Raw gyroscope Y-axis value */
    int16_t gyro_z;           /**< Raw gyroscope Z-axis value */
    uint16_t fsr_val;         /**< Raw FSR ADC value */
    uint8_t gain_state;       /**< FSR gain state (0 or 1) */
} sensor_sample_t;

#endif // SENSOR_DATA_HPP
