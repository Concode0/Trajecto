#include "imu_task.hpp"
#include "fsr_task.hpp" // For FSR reading
#include "esp_log.h"
#include "esp_err.h"
#include "bmi2.h" // Bosch BMI2 driver for common structures/functions

static const char *TAG = "IMU_TASK";

static bmi270_handle_t bmi270_dev_handle = NULL;
static RingBuffer *sensor_data_buffer = NULL;
static esp_timer_handle_t imu_timer_handle;
static uint64_t prev_timestamp_us = 0;

// Internal function prototypes
static int8_t bmi2_i2c_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr);
static int8_t bmi2_i2c_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr);
static void bmi2_delay_us(uint32_t period, void *intf_ptr);
static void imu_timer_callback(void *arg);

esp_err_t imu_task_init(RingBuffer *buffer) {
    esp_err_t ret;
    sensor_data_buffer = buffer; // Store the buffer reference

    // 1. Initialize I2C bus
    i2c_config_t i2c_conf;
    memset(&i2c_conf, 0, sizeof(i2c_config_t));
    i2c_conf.mode = I2C_MODE_MASTER;
    i2c_conf.sda_io_num = I2C_MASTER_SDA_IO;
    i2c_conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    i2c_conf.scl_io_num = I2C_MASTER_SCL_IO;
    i2c_conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    i2c_conf.master.clk_speed = I2C_MASTER_FREQ_HZ;
    i2c_conf.clk_flags = 0; // Optional: you can add I2C_SCLK_SRC_FLAG_*

    ret = i2c_param_config(I2C_MASTER_NUM, &i2c_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = i2c_driver_install(I2C_MASTER_NUM, I2C_MODE_MASTER, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "I2C master initialized on port %d, SDA: %d, SCL: %d", I2C_MASTER_NUM, I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);

    // 2. Initialize FSR sensor
    ret = fsr_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FSR sensor initialization failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // 3. Initialize BMI270
    bmi270_i2c_config_t i2c_conf_bmi = {
        .i2c_handle = I2C_MASTER_NUM, // i2c_bus_handle_t is defined as i2c_port_t in esp-idf i2c_bus component
        .i2c_addr = BMI270_SENSOR_ADDR
    };

    ret = bmi270_sensor_create(&i2c_conf_bmi, &bmi270_dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "BMI270 sensor create failed: %s", esp_err_to_name(ret));
        return ret;
    }

    bmi270_dev_handle->read = bmi2_i2c_read;
    bmi270_dev_handle->write = bmi2_i2c_write;
    bmi270_dev_handle->delay_us = bmi2_delay_us;
    bmi270_dev_handle->intf_ptr = (void*)(size_t)I2C_MASTER_NUM; // Pass I2C port as interface pointer
    bmi270_dev_handle->intf = BMI2_I2C_INTF;
    bmi270_dev_handle->dev_id = BMI270_SENSOR_ADDR;

    // The bmi270_init function in the component internally calls bmi2_init, which writes config file
    int8_t rslt = bmi270_init(bmi270_dev_handle);
    if (rslt != BMI2_OK) {
        ESP_LOGE(TAG, "BMI270 initialization failed: %d", rslt);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "BMI270 sensor initialized.");

    // Configure Accelerometer and Gyroscope
    struct bmi2_sens_config sens_cfg[2];
    uint8_t n_sens = 2;

    sens_cfg[0].type = BMI2_ACCEL;
    sens_cfg[0].cfg.acc.odr = BMI2_ACC_ODR_400HZ;
    sens_cfg[0].cfg.acc.range = BMI2_ACC_RANGE_4G; // Adjust range as needed
    sens_cfg[0].cfg.acc.bwp = BMI2_ACC_BW_NORMAL_AVG4;
    sens_cfg[0].cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;

    sens_cfg[1].type = BMI2_GYRO;
    sens_cfg[1].cfg.gyr.odr = BMI2_GYR_ODR_400HZ;
    sens_cfg[1].cfg.gyr.range = BMI2_GYR_RANGE_2000DPS; // Adjust range as needed
    sens_cfg[1].cfg.gyr.bwp = BMI2_GYR_BW_NORMAL_MODE;
    sens_cfg[1].cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;

    rslt = bmi2_set_sensor_config(sens_cfg, n_sens, bmi270_dev_handle);
    if (rslt != BMI2_OK) {
        ESP_LOGE(TAG, "BMI270 sensor config failed: %d", rslt);
        return ESP_FAIL;
    }

    // Enable Accelerometer and Gyroscope
    uint8_t sens_list[2] = {BMI2_ACCEL, BMI2_GYRO};
    rslt = bmi2_sensor_enable(sens_list, n_sens, bmi270_dev_handle);
    if (rslt != BMI2_OK) {
        ESP_LOGE(TAG, "BMI270 sensor enable failed: %d", rslt);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "BMI270 Accelerometer and Gyroscope enabled at 400Hz.");

    // 4. Create and start high-precision timer for 400Hz acquisition
    const esp_timer_create_args_t imu_timer_args = {
        .callback = &imu_timer_callback,
        .name = "imu_acquisition_timer"
    };
    ret = esp_timer_create(&imu_timer_args, &imu_timer_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create IMU timer: %s", esp_err_to_name(ret));
        return ret;
    }
    ret = esp_timer_start_periodic(imu_timer_handle, IMU_TASK_PERIOD_US);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start IMU timer: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "IMU acquisition timer started at %d Hz.", 1000000 / IMU_TASK_PERIOD_US);

    return ESP_OK;
}

static void imu_timer_callback(void *arg) {
    struct bmi2_sens_data sensor_data;
    uint16_t fsr_val;
    uint8_t gain_state;
    esp_err_t ret;

    // Calculate timestamp delta
    uint64_t current_timestamp_us = esp_timer_get_time();
    uint16_t timestamp_delta = 0;
    if (prev_timestamp_us != 0) {
        timestamp_delta = (uint16_t)(current_timestamp_us - prev_timestamp_us);
    }
    prev_timestamp_us = current_timestamp_us;

    // Read BMI270 data
    // bmi2_get_sensor_data reads accel and gyro data if enabled
    int8_t rslt = bmi2_get_sensor_data(&sensor_data, bmi270_dev_handle);
    if (rslt != BMI2_OK) {
        ESP_LOGW(TAG, "Failed to read BMI270 data: %d", rslt);
        // On error, use previous values or zeros
        sensor_data.acc.x = 0;
        sensor_data.acc.y = 0;
        sensor_data.acc.z = 0;
        sensor_data.gyr.x = 0;
        sensor_data.gyr.y = 0;
        sensor_data.gyr.z = 0;
    }

    // Read FSR data
    ret = fsr_read(&fsr_val, &gain_state);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Failed to read FSR data: %s", esp_err_to_name(ret));
        fsr_val = 0;
        gain_state = 0; // Default or previous state
    }

    sensor_sample_t sample = {
        .timestamp_delta = timestamp_delta,
        .accel_x = sensor_data.acc.x,
        .accel_y = sensor_data.acc.y,
        .accel_z = sensor_data.acc.z,
        .gyro_x = sensor_data.gyr.x,
        .gyro_y = sensor_data.gyr.y,
        .gyro_z = sensor_data.gyr.z,
        .fsr_val = fsr_val,
        .gain_state = gain_state,
    };

    // Write to ring buffer
    if (!sensor_data_buffer->write(sample)) {
        ESP_LOGW(TAG, "Ring buffer full, dropped a sample!");
    }
}

// BMI2 I2C read function for the Bosch driver
static int8_t bmi2_i2c_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
    i2c_port_t i2c_num = (i2c_port_t)(size_t)intf_ptr;
    esp_err_t ret = i2c_master_write_read_device(i2c_num, BMI270_SENSOR_ADDR, &reg_addr, 1, data, len, 1000 / portTICK_PERIOD_MS);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C read failed at addr 0x%x, reg 0x%x, len %d: %s", BMI270_SENSOR_ADDR, reg_addr, len, esp_err_to_name(ret));
        return BMI2_E_COM_FAIL;
    }
    return BMI2_OK;
}

// BMI2 I2C write function for the Bosch driver
static int8_t bmi2_i2c_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
    i2c_port_t i2c_num = (i2c_port_t)(size_t)intf_ptr;
    uint8_t write_buf[len + 1];
    write_buf[0] = reg_addr;
    memcpy(&write_buf[1], data, len);

    esp_err_t ret = i2c_master_write_to_device(i2c_num, BMI270_SENSOR_ADDR, write_buf, len + 1, 1000 / portTICK_PERIOD_MS);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C write failed at addr 0x%x, reg 0x%x, len %d: %s", BMI270_SENSOR_ADDR, reg_addr, len, esp_err_to_name(ret));
        return BMI2_E_COM_FAIL;
    }
    return BMI2_OK;
}

// BMI2 delay function for the Bosch driver
static void bmi2_delay_us(uint32_t period, void *intf_ptr) {
    (void)intf_ptr; // Unused
    ets_delay_us(period);
}
