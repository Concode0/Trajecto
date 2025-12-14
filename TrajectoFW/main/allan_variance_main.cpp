#include <chrono>
#include <vector>

#include "bmi270.hpp"
#include "i2c.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/gpio.h"
#include "task.hpp"

// Use espp::Logger for consistent logging
#include "logger.hpp"

// Use std::bind for callbacks
#include <functional>

using namespace std::chrono_literals;

// Semaphore to signal when IMU data is ready
static SemaphoreHandle_t imu_sem = nullptr;

/**
 * @brief Interrupt Service Routine (ISR) called by the BMI270's DRDY pin.
 * @param arg Unused.
 * @note This function is designed to be as fast as possible. It only gives a
 *       semaphore and yields to a higher priority task if one is waiting.
 */
static void IRAM_ATTR isr_handler(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(imu_sem, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

extern "C" void app_main(void) {
    espp::Logger logger({.tag = "AllanVariance", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting Allan Variance data acquisition firmware...");

    // Create a binary semaphore to be used by the ISR
    imu_sem = xSemaphoreCreateBinary();

    // --- I2C and BMI270 Sensor Initialization ---
    // This section is kept similar to your original main.cpp for consistency.

    // 1. Configure I2C peripheral
    static constexpr auto i2c_port = I2C_NUM_0;
    espp::I2c i2c({.port = i2c_port,
                    .sda_io_num = (gpio_num_t)10,
                    .scl_io_num = (gpio_num_t)4,
                    .sda_pullup_en = GPIO_PULLUP_ENABLE,
                    .scl_pullup_en = GPIO_PULLUP_ENABLE,
                    .timeout_ms = 200,
                    .clk_speed = 400 * 1000});

    // 2. Probe for the BMI270 sensor address
    uint8_t bmi270_address = espp::Bmi270<>::DEFAULT_ADDRESS;
    if (!i2c.probe_device(bmi270_address)) {
        logger.warn("No BMI270 found at default address, trying alternative...");
        bmi270_address = espp::Bmi270<>::DEFAULT_ADDRESS_SDO_HIGH;
        if (!i2c.probe_device(bmi270_address)) {
            logger.error("BMI270 not found at any address. Halting.");
            return;
        }
    }
    logger.info("Found BMI270 at address: 0x{:02X}", bmi270_address);

    // 3. Define BMI270 configuration
    using Imu = espp::Bmi270<espp::bmi270::Interface::I2C>;
    Imu::Config config{
        .device_address = bmi270_address,
        .write = std::bind(&espp::I2c::write, &i2c, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        .read = std::bind(&espp::I2c::read, &i2c, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        .imu_config = {
            // NOTE: For best Allan Variance results, use the widest range possible
            // to avoid clipping, even though the device is stationary.
            .accelerometer_range = Imu::AccelerometerRange::RANGE_16G,
            .accelerometer_odr = Imu::AccelerometerODR::ODR_400_HZ,
            .accelerometer_bandwidth = Imu::AccelerometerBandwidth::NORMAL_AVG4,
            .gyroscope_range = Imu::GyroscopeRange::RANGE_2000DPS,
            .gyroscope_odr = Imu::GyroscopeODR::ODR_400_HZ,
            .gyroscope_bandwidth = Imu::GyroscopeBandwidth::NORMAL_MODE,
            .gyroscope_performance_mode = Imu::GyroscopePerformanceMode::PERFORMANCE_OPTIMIZED,
            .enable_advanced_features = true,
        },
        .burst_write_size = 128,
        .auto_init = true,
        .log_level = espp::Logger::Verbosity::INFO,
    };

    // 4. Create IMU object
    Imu imu(config);

    // 5. Configure the data ready interrupt
    espp::Bmi270<>::InterruptConfig int_config{
        .pin = espp::Bmi270<>::InterruptPin::INT1,
        .output_type = espp::Bmi270<>::InterruptOutput::OPEN_DRAIN,
        .active_level = espp::Bmi270<>::InterruptLevel::ACTIVE_LOW,
        .enable_data_ready = true,
    };
    std::error_code ec;
    imu.configure_interrupts(int_config, ec);
    if (ec) {
        logger.error("Failed to configure interrupts: {}", ec.message());
    }

    // 6. Configure the GPIO pin for the interrupt
    static constexpr gpio_num_t interrupt_pin = GPIO_NUM_1;
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_NEGEDGE; // BMI270 DRDY is Active Low
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << interrupt_pin);
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(interrupt_pin, isr_handler, nullptr);

    logger.info("IMU initialized and interrupt configured. Starting data stream...");

    // Print the CSV header for easy data logging
    printf("accel_x_g,accel_y_g,accel_z_g,gyro_x_rads,gyro_y_rads,gyro_z_rads\n");

    // --- Main Loop (in a task) ---
    auto task_fn = [&]() -> bool {
        static auto start_time = esp_timer_get_time();
        static auto t0 = start_time;
        // Wait indefinitely for the semaphore from the ISR
        if (xSemaphoreTake(imu_sem, portMAX_DELAY) == pdTRUE) {
            auto now = esp_timer_get_time();
            float dt = (now - t0) * 1e-6f;
            t0 = now;
            std::error_code ec;
            if (!imu.update(dt, ec)) {
                logger.error("Failed to update IMU: {}", ec.message());
                return false; // stop the task
            }

            auto accel = imu.get_accelerometer(); // In g's
            auto gyro = imu.get_gyroscope();       // In rad/s

            // Print the raw data in CSV format to the serial port
            printf("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                   accel.x, accel.y, accel.z,
                   gyro.x, gyro.y, gyro.z);
        }
        return false; // loop forever
    };

    espp::Task imu_task({
        .callback = task_fn,
        .task_config = {
            .name = "IMU Task",
            .stack_size_bytes = 6 * 1024,
            .priority = 10,
        }
    });
    imu_task.start();

    // The rest of app_main can be used for other things, or just loop
    while (true) {
        std::this_thread::sleep_for(1s);
    }
}
