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

#include <atomic>
#include <chrono>
#include <cstring> // for memcpy
#include <functional> // for std::bind

#include "bmi270.hpp"
#include "i2c.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "esp_timer.h"

// For BLE
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "nvs_flash.h"

// Trajecto System
#include "trajecto_system.hpp"
#include "trajecto_protocol.h"
#include "swinging_door.hpp"

using namespace std::chrono_literals;

static const char* TAG = "Trajecto";

// Typedef for the IMU
using Imu = espp::Bmi270<espp::bmi270::Interface::I2C>;

static const char *device_name = "Trajecto";

// BLE Handles
static uint8_t ble_addr_type;
static uint16_t conn_handle;
static std::atomic<bool> is_connected(false);
static uint16_t trajecto_chr_val_handle; // For Notify (Data)
static uint16_t trajecto_cmd_val_handle; // For Write (Commands)

// Application State
enum class AppMode {
    IDLE,
    STREAMING_RAW,
    STREAMING_TRAJECTORY
};
static std::atomic<AppMode> current_mode(AppMode::IDLE);

// TFLite Initialization Status
static std::atomic<bool> tflite_ok(false);

// Current Configuration
static trajecto::protocol::ConfigPayload current_config = {
    .mode = 1,        // Default: Trajectory mode
    .odr_hz = 50,     // Fixed ODR
    .enable_sda = 1,  // Default: Swinging Door compression enabled
    .reserved = {0}
};

// Calibration Control
static std::atomic<bool> calibration_requested(false);

// Semaphore for IMU Data Ready
static SemaphoreHandle_t imu_sem = nullptr;

// Swinging Door compressor for trajectory stream
static trajecto::SwingingDoor compressor(
    0.001f,   // 1mm position tolerance
    0.01f,    // 1cm/s velocity tolerance
    0.01f,    // ~0.57° rotation tolerance
    50,       // Buffer up to 50 points (1 second @ 50Hz)
    500000    // Force send every 500ms
);

static void send_notification(void *data, size_t len);

// ============================================================================
// BMI270 Calibration (CRT & FOC)
// ============================================================================

static const char* NVS_NAMESPACE = "calibration";
static const char* KEY_GYRO_FOC_X = "gfoc_x";
static const char* KEY_GYRO_FOC_Y = "gfoc_y";
static const char* KEY_GYRO_FOC_Z = "gfoc_z";

struct GyroOffset {
    int16_t x;
    int16_t y;
    int16_t z;
};

static bool load_foc_from_nvs(GyroOffset& offset) {
    nvs_handle_t my_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &my_handle);
    if (err != ESP_OK) return false;

    int16_t x=0, y=0, z=0;
    if (nvs_get_i16(my_handle, KEY_GYRO_FOC_X, &x) == ESP_OK &&
        nvs_get_i16(my_handle, KEY_GYRO_FOC_Y, &y) == ESP_OK &&
        nvs_get_i16(my_handle, KEY_GYRO_FOC_Z, &z) == ESP_OK) {
        offset.x = x;
        offset.y = y;
        offset.z = z;
        nvs_close(my_handle);
        return true;
    }
    nvs_close(my_handle);
    return false;
}

static void save_foc_to_nvs(const GyroOffset& offset) {
    nvs_handle_t my_handle;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &my_handle) == ESP_OK) {
        nvs_set_i16(my_handle, KEY_GYRO_FOC_X, offset.x);
        nvs_set_i16(my_handle, KEY_GYRO_FOC_Y, offset.y);
        nvs_set_i16(my_handle, KEY_GYRO_FOC_Z, offset.z);
        nvs_commit(my_handle);
        nvs_close(my_handle);
    }
}

static void perform_calibration_with_status(Imu& imu) {
    using namespace trajecto::protocol;

    espp::Logger logger({.tag = "CALIB", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting runtime calibration...");
    std::error_code ec;

    logger.warn("========================================");
    logger.warn("STARTING CALIBRATION SEQUENCE");
    logger.warn("KEEP THE PEN STILL ON THE TABLE!");
    logger.warn("========================================");

    // Execute CRT (Sensitivity Fix)
    if (!imu.perform_crt(ec)) {
        logger.error("CRT FAILED code: {}. (Did you move the pen?)", ec.message());

        struct {
            Header h;
            uint8_t status;
        } rsp;
        rsp.h.type = PacketType::RSP_CALIB_STATUS;
        rsp.h.length = sizeof(uint8_t);
        rsp.status = 2; // Failed
        send_notification(&rsp, sizeof(rsp));
        return;
    }

    logger.info("CRT SUCCESS! Sensitivity Restored.");

    // Enable sensors for FOC (perform_gyro_foc handles this, but good to ensure)
    // espp driver handles enabling/disabling as needed.

    // Slight delay to ensure stability after CRT
    std::this_thread::sleep_for(100ms);

    if (!imu.perform_gyro_foc(ec)) {
        logger.error("Gyro FOC Failed: {}", ec.message());

        struct {
            Header h;
            uint8_t status;
        } rsp;
        rsp.h.type = PacketType::RSP_CALIB_STATUS;
        rsp.h.length = sizeof(uint8_t);
        rsp.status = 2; // Failed
        send_notification(&rsp, sizeof(rsp));
        return;
    }

    logger.info("Gyro FOC Done.");

    GyroOffset offset;
    if (imu.get_gyro_offset(offset.x, offset.y, offset.z, ec)) {
        save_foc_to_nvs(offset);
        logger.info("Calibration Saved to NVS (X={} Y={} Z={}).", offset.x, offset.y, offset.z);

        struct {
            Header h;
            uint8_t status;
        } rsp;
        rsp.h.type = PacketType::RSP_CALIB_STATUS;
        rsp.h.length = sizeof(uint8_t);
        rsp.status = 1; // Success
        send_notification(&rsp, sizeof(rsp));
    } else {
        logger.error("Failed to read back FOC offsets.");

        struct {
            Header h;
            uint8_t status;
        } rsp;
        rsp.h.type = PacketType::RSP_CALIB_STATUS;
        rsp.h.length = sizeof(uint8_t);
        rsp.status = 2; // Failed
        send_notification(&rsp, sizeof(rsp));
    }
}

// CRITICAL: CRT fixes 17m position error from BMI270 sensitivity drift
static bool ensure_calibration(Imu& imu) {
    espp::Logger logger({.tag = "CALIB", .level = espp::Logger::Verbosity::INFO});
    std::error_code ec;

    GyroOffset offset;
    if (load_foc_from_nvs(offset)) {
        logger.info("Found saved FOC offsets: X={} Y={} Z={}", offset.x, offset.y, offset.z);
        // Apply offsets
        if (imu.set_gyro_offset(offset.x, offset.y, offset.z, ec)) {
             // Enable offset compensation
             if (imu.set_gyro_offset_enable(true, ec)) {
                 logger.info("Calibration restored successfully.");
                 return true;
             }
        }
        logger.error("Failed to restore calibration: {}", ec.message());
        // If restore fails, fall through to full calibration?
        // Or just return false? Let's try to calibrate.
    }

    logger.warn("No calibration found (or restore failed) - running CRT+FOC sequence...");

    logger.warn("========================================");
    logger.warn("STARTING CALIBRATION SEQUENCE");
    logger.warn("KEEP THE PEN STILL ON THE TABLE!");
    logger.warn("========================================");

    // Execute CRT (Sensitivity Fix)
    if (imu.perform_crt(ec)) {
        logger.info("CRT SUCCESS! Sensitivity Restored.");

        std::this_thread::sleep_for(100ms);

        if (imu.perform_gyro_foc(ec)) {
            logger.info("Gyro FOC Done.");

            if (imu.get_gyro_offset(offset.x, offset.y, offset.z, ec)) {
                save_foc_to_nvs(offset);
                logger.info("Calibration Saved to NVS.");
                return true;
            } else {
                logger.error("Failed to read back FOC offsets: {}", ec.message());
                return false;
            }
        } else {
            logger.error("Gyro FOC Failed: {}", ec.message());
            return false;
        }
    } else {
        logger.error("CRT FAILED code: {}. (Did you move the pen?)", ec.message());
        return false;
    }
}

// ============================================================================
// BLE Protocol
// ============================================================================

static int ble_gap_event(struct ble_gap_event *event, void *arg);
static int gatt_svr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                              struct ble_gatt_access_ctxt *ctxt, void *arg);
static const ble_uuid128_t trajecto_service_uuid_val =
    BLE_UUID128_INIT(0x57, 0x45, 0x54, 0x53, 0x31, 0x54, 0x74, 0xB4, 0x94, 0x45, 0x49, 0xC5, 0x4E, 0x43, 0x43, 0xAD);
static const ble_uuid128_t trajecto_chr_uuid_val =
    BLE_UUID128_INIT(0x57, 0x45, 0x54, 0x53, 0x31, 0x54, 0x74, 0xB4, 0x94, 0x45, 0x49, 0xC5, 0x4F, 0x43, 0x43, 0xAD);
static const ble_uuid128_t trajecto_cmd_uuid_val =
    BLE_UUID128_INIT(0x57, 0x45, 0x54, 0x53, 0x31, 0x54, 0x74, 0xB4, 0x94, 0x45, 0x49, 0xC5, 0x4D, 0x43, 0x43, 0xAD);


static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &trajecto_service_uuid_val.u,
        .characteristics = (struct ble_gatt_chr_def[]){
            {
                .uuid = &trajecto_chr_uuid_val.u,
                .access_cb = gatt_svr_access_cb,
                .flags = BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &trajecto_chr_val_handle,

            },
            {
                .uuid = &trajecto_cmd_uuid_val.u,
                .access_cb = gatt_svr_access_cb,
                .flags = BLE_GATT_CHR_F_WRITE,
                .val_handle = &trajecto_cmd_val_handle,
            },
            {0}, /* No more characteristics in this service */
        },
    },
    {0},
};

static void send_notification(void *data, size_t len) {
    if (is_connected && trajecto_chr_val_handle != 0) {
        struct os_mbuf *txom = ble_hs_mbuf_from_flat(data, len);
        ble_gatts_notify_custom(conn_handle, trajecto_chr_val_handle, txom);
    }
}

static int gatt_svr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                   struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        if (attr_handle == trajecto_cmd_val_handle) {
            if (os_mbuf_len(ctxt->om) >= sizeof(trajecto::protocol::Header)) {
                uint8_t buf[32];
                uint16_t len = os_mbuf_len(ctxt->om);
                if (len > sizeof(buf)) len = sizeof(buf);
                ble_hs_mbuf_to_flat(ctxt->om, buf, len, NULL);

                auto* header = reinterpret_cast<trajecto::protocol::Header*>(buf);
                using namespace trajecto::protocol;

                switch (header->type) {
                    case PacketType::CMD_PING: {
                        ESP_LOGI(TAG, "CMD: Ping");
                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_PONG;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_GET_CONFIG: {
                        ESP_LOGI(TAG, "CMD: Get Config");
                        struct {
                            Header h;
                            ConfigPayload cfg;
                        } rsp;
                        rsp.h.type = PacketType::RSP_CONFIG;
                        rsp.h.length = sizeof(ConfigPayload);
                        rsp.cfg = current_config;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_SET_CONFIG: {
                        if (len >= sizeof(Header) + sizeof(ConfigPayload)) {
                            auto* cfg = reinterpret_cast<ConfigPayload*>(buf + sizeof(Header));
                            current_config = *cfg;

                            if (cfg->mode == 0) {
                                current_mode = AppMode::STREAMING_RAW;
                                ESP_LOGI(TAG, "CMD: Set Mode RAW");
                            } else {
                                if (tflite_ok) {
                                    current_mode = AppMode::STREAMING_TRAJECTORY;
                                    ESP_LOGI(TAG, "CMD: Set Mode TRAJECTORY");
                                } else {
                                    ESP_LOGW(TAG, "CMD: TRAJECTORY mode rejected - TFLite not initialized");
                                    current_mode = AppMode::IDLE;
                                }
                            }

                            struct {
                                Header h;
                            } rsp;
                            rsp.h.type = PacketType::RSP_CONFIG_OK;
                            rsp.h.length = 0;
                            send_notification(&rsp, sizeof(rsp));
                        }
                        break;
                    }

                    case PacketType::CMD_START_STREAM: {
                        if (current_mode == AppMode::IDLE) {
                            if (current_config.mode == 0) {
                                current_mode = AppMode::STREAMING_RAW;
                            } else {
                                if (tflite_ok) {
                                    current_mode = AppMode::STREAMING_TRAJECTORY;
                                    // Reset swinging door compressor for new stream
                                    compressor.reset();
                                } else {
                                    ESP_LOGW(TAG, "CMD: TRAJECTORY mode rejected - TFLite not initialized, using RAW mode");
                                    current_mode = AppMode::STREAMING_RAW;
                                }
                            }
                        }
                        ESP_LOGI(TAG, "CMD: Start Stream (Mode: %d)", static_cast<uint8_t>(current_mode.load()));

                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_STREAM_STARTED;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_STOP_STREAM: {
                        // Flush any buffered trajectory points before stopping
                        if (current_mode == AppMode::STREAMING_TRAJECTORY && current_config.enable_sda) {
                            auto flush_callback = [](const trajecto::SwingingDoor::Point& pt,
                                                    trajecto::SwingingDoor::SendReason reason) {
                                struct __attribute__((packed)) {
                                    Header h;
                                    TrajectoryPacket p;
                                } pkt;

                                pkt.h.type = PacketType::DATA_TRAJECTORY;
                                pkt.h.length = sizeof(TrajectoryPacket);
                                pkt.p.timestamp_us = pt.timestamp_us;
                                pkt.p.pos[0] = pt.position.x();
                                pkt.p.pos[1] = pt.position.y();
                                pkt.p.pos[2] = pt.position.z();
                                pkt.p.vel[0] = pt.velocity.x();
                                pkt.p.vel[1] = pt.velocity.y();
                                pkt.p.vel[2] = pt.velocity.z();
                                pkt.p.quat[0] = pt.quat.w();
                                pkt.p.quat[1] = pt.quat.x();
                                pkt.p.quat[2] = pt.quat.y();
                                pkt.p.quat[3] = pt.quat.z();
                                pkt.p.zupt_prob = 0.0f;  // Not available in flush context

                                // Set flags for flushed packet
                                pkt.p.flags = 0;
                                if (pt.pen_state) {
                                    pkt.p.flags |= TrajectoryFlags::TRAJ_FLAG_PEN_DOWN;
                                }
                                pkt.p.reserved[0] = 0;
                                pkt.p.reserved[1] = 0;
                                pkt.p.reserved[2] = 0;

                                send_notification(&pkt, sizeof(pkt));
                            };
                            compressor.flush(flush_callback);

                            // Log compression statistics
                            auto stats = compressor.get_stats();
                            ESP_LOGI(TAG, "Swinging Door Stats: Received=%lu, Sent=%lu, Ratio=%.2fx",
                                    stats.points_received, stats.points_sent, stats.compression_ratio);
                        }

                        current_mode = AppMode::IDLE;
                        ESP_LOGI(TAG, "CMD: Stop Stream");

                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_STREAM_STOPPED;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_CALIBRATE: {
                        ESP_LOGI(TAG, "CMD: Calibrate (requesting...)");
                        calibration_requested = true;

                        struct {
                            Header h;
                            uint8_t status; // 0: In Progress, 1: Success, 2: Failed
                        } rsp;
                        rsp.h.type = PacketType::RSP_CALIB_STATUS;
                        rsp.h.length = sizeof(uint8_t);
                        rsp.status = 0; // In Progress
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    default:
                        ESP_LOGW(TAG, "Unknown Command: 0x%02X", static_cast<uint8_t>(header->type));
                        break;
                }
            }
        }
        return 0;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

static void ble_advertise(void) {
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;
    int rc;

    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.tx_pwr_lvl_is_present = 1;
    fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;

    fields.name = reinterpret_cast<uint8_t *>(const_cast<char *>(device_name));
    fields.name_len = strlen(device_name);
    fields.name_is_complete = 1;

    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) return;

    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(ble_addr_type, NULL, BLE_HS_FOREVER, &adv_params, ble_gap_event, NULL);
}

static int ble_gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            is_connected = true;
            ESP_LOGI(TAG, "Connected");

            using namespace trajecto::protocol;
            struct {
                Header h;
            } packet;
            packet.h.type = PacketType::RSP_STREAM_STOPPED;
            packet.h.length = 0;
            send_notification(&packet, sizeof(packet));

        } else {
            ble_advertise();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        is_connected = false;
        current_mode = AppMode::IDLE;
        ESP_LOGI(TAG, "Disconnected");
        ble_advertise();
        break;
    case BLE_GAP_EVENT_ADV_COMPLETE:
        ble_advertise();
        break;
    }
    return 0;
}

void ble_on_sync(void) {
    int rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    rc = ble_hs_id_infer_auto(0, &ble_addr_type);
    assert(rc == 0);
    ble_advertise();
}

void ble_on_reset([[maybe_unused]] int reason) {}

void nimble_host_task(void *param) {
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void IRAM_ATTR isr_handler(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(imu_sem, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// ============================================================================
// Main
// ============================================================================

extern "C" void app_main(void) {
    espp::Logger logger({.tag = "Trajecto System", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting Trajecto System!");

    esp_err_t ret = nvs_flash_init();
    int rc = nimble_port_init();
    if (rc != 0) return;
    ble_hs_cfg.sync_cb = ble_on_sync;
    ble_hs_cfg.reset_cb = ble_on_reset;
    ble_svc_gap_device_name_set(device_name);
    ble_svc_gap_init();
    ble_svc_gatt_init();
    rc = ble_gatts_count_cfg(gatt_svr_svcs);
    assert(rc == 0);
    rc = ble_gatts_add_svcs(gatt_svr_svcs);
    assert(rc == 0);

    nimble_port_freertos_init(nimble_host_task);

    static constexpr gpio_num_t DATA_LED_PIN = GPIO_NUM_7;
    gpio_config_t led_io_conf = {};
    led_io_conf.mode = GPIO_MODE_OUTPUT;
    led_io_conf.pin_bit_mask = (1ULL << DATA_LED_PIN);
    gpio_config(&led_io_conf);
    gpio_set_level(DATA_LED_PIN, 0);

    static constexpr gpio_num_t FSR_ENABLE_PIN = GPIO_NUM_6;
    gpio_config_t fsr_io_conf = {};
    fsr_io_conf.mode = GPIO_MODE_OUTPUT;
    fsr_io_conf.pin_bit_mask = (1ULL << FSR_ENABLE_PIN);
    gpio_config(&fsr_io_conf);
    gpio_set_level(FSR_ENABLE_PIN, 1);
    static constexpr auto FSR_ADC_UNIT = ADC_UNIT_1;
    static constexpr auto FSR_ADC_CHANNEL = ADC_CHANNEL_3;
    adc_oneshot_unit_handle_t adc_handle;
    adc_oneshot_unit_init_cfg_t adc_init_config = { .unit_id = FSR_ADC_UNIT };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&adc_init_config, &adc_handle));

    adc_oneshot_chan_cfg_t adc_chan_config = {
        .atten = ADC_ATTEN_DB_11,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, FSR_ADC_CHANNEL, &adc_chan_config));

    imu_sem = xSemaphoreCreateBinary();

    using Imu = espp::Bmi270<espp::bmi270::Interface::I2C>;

    static constexpr auto i2c_port = I2C_NUM_0;
    static constexpr auto i2c_clock_speed = 400 * 1000;
    static constexpr gpio_num_t i2c_sda = static_cast<gpio_num_t>(10);
    static constexpr gpio_num_t i2c_scl = static_cast<gpio_num_t>(4);
    espp::I2c i2c({.port = i2c_port,
                    .sda_io_num = i2c_sda,
                    .scl_io_num = i2c_scl,
                    .sda_pullup_en = GPIO_PULLUP_ENABLE,
                    .scl_pullup_en = GPIO_PULLUP_ENABLE,
                    .timeout_ms = 200,
                    .clk_speed = i2c_clock_speed});

    uint8_t bmi270_address = Imu::DEFAULT_ADDRESS;
    const uint8_t addresses[] = {Imu::DEFAULT_ADDRESS, Imu::DEFAULT_ADDRESS_SDO_HIGH};
    for (auto address : addresses) {
        if (i2c.probe_device(address)) {
            logger.info("Found BMI270 at address: 0x{:02X}", address);
            bmi270_address = address;
            break;
        }
    }

    Imu::Config config{
        .device_address = bmi270_address,
        .write = std::bind(&espp::I2c::write, &i2c, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        .read = std::bind(&espp::I2c::read, &i2c, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
        .imu_config = {
            .accelerometer_range = Imu::AccelerometerRange::RANGE_4G,
            .accelerometer_odr = Imu::AccelerometerODR::ODR_50_HZ,
            .accelerometer_bandwidth = Imu::AccelerometerBandwidth::NORMAL_AVG4,
            .gyroscope_range = Imu::GyroscopeRange::RANGE_500DPS,
            .gyroscope_odr = Imu::GyroscopeODR::ODR_50_HZ,
            .gyroscope_bandwidth = Imu::GyroscopeBandwidth::NORMAL_MODE,
            .gyroscope_performance_mode = Imu::GyroscopePerformanceMode::PERFORMANCE_OPTIMIZED,
            .enable_advanced_features = true,
        },
        .burst_write_size = 128,
        .auto_init = true,
        .log_level = espp::Logger::Verbosity::INFO,
    };

    gpio_num_t interrupt_pin = GPIO_NUM_1;

    espp::Bmi270<>::InterruptConfig int_config{
        .pin = espp::Bmi270<>::InterruptPin::INT1,
        .output_type = espp::Bmi270<>::InterruptOutput::OPEN_DRAIN,
        .active_level = espp::Bmi270<>::InterruptLevel::ACTIVE_LOW,
        .enable_data_ready = true,
    };

    Imu imu(config);

    // --- CRT & FOC Calibration (check/perform if needed) ---
    vTaskDelay(pdMS_TO_TICKS(500));
    bool calib_ok = ensure_calibration(imu);
    if (!calib_ok) {
        logger.error("Calibration failed! System may not work correctly.");
    }

    std::error_code ec;
    imu.configure_interrupts(int_config, ec);

    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_NEGEDGE;
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << interrupt_pin);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);

    gpio_install_isr_service(0);
    gpio_isr_handler_add(interrupt_pin, isr_handler, nullptr);

    static trajecto::TrajectoSystem sys;
    if (!sys.setup()) {
        logger.error("Failed to setup Trajecto System (TFLite)!");
        logger.error("System will enter IDLE mode. Only RAW IMU streaming available.");
        tflite_ok = false;
    } else {
        logger.info("Trajecto System (TFLite) setup complete.");
        tflite_ok = true;
    }


    printf("accel_x_g,accel_y_g,accel_z_g,gyro_x_rads,gyro_y_rads,gyro_z_rads,temp_c\n");

    auto task_fn = [&]() -> bool {
        if (xSemaphoreTake(imu_sem, portMAX_DELAY) == pdTRUE) {
            gpio_set_level(DATA_LED_PIN, 0);

            auto now = esp_timer_get_time();
            static auto t0 = now;
            auto t1 = now;
            float dt = (t1 - t0) / 1'000'000.0f;
            t0 = t1;

            std::error_code ec;
            if (!imu.update(dt, ec)) {
                logger.error("Failed to update IMU: {}", ec.message());
                return false;
            }

            auto accel = imu.get_accelerometer();
            auto gyro = imu.get_gyroscope();
            float temp = imu.get_temperature();

            int fsr_raw;
            ESP_ERROR_CHECK(adc_oneshot_read(adc_handle, FSR_ADC_CHANNEL, &fsr_raw));

            constexpr float DEG_TO_RAD = M_PI / 180.0f;
            printf("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.2f\n",
                   accel.x, accel.y, accel.z,
                   gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD,
                   temp);

            Eigen::Vector3f accel_vec(accel.x * 9.81f, accel.y * 9.81f, accel.z * 9.81f);
            Eigen::Vector3f gyro_vec(gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD);
            float force_val = static_cast<float>(fsr_raw);

            if (is_connected) {
                using namespace trajecto::protocol;

                if (current_mode == AppMode::STREAMING_RAW) {
                    gpio_set_level(DATA_LED_PIN, 1);
                    struct __attribute__((packed)) {
                        Header h;
                        RawImuPacket p;
                    } pkt;
                    pkt.h.type = PacketType::DATA_RAW_IMU;
                    pkt.h.length = sizeof(RawImuPacket);
                    pkt.p.timestamp_us = static_cast<uint32_t>(now);
                    pkt.p.accel[0] = accel_vec.x();
                    pkt.p.accel[1] = accel_vec.y();
                    pkt.p.accel[2] = accel_vec.z();
                    pkt.p.gyro[0] = gyro_vec.x();
                    pkt.p.gyro[1] = gyro_vec.y();
                    pkt.p.gyro[2] = gyro_vec.z();
                    pkt.p.force = static_cast<int16_t>(force_val);
                    pkt.p.temperature = temp;

                    send_notification(&pkt, sizeof(pkt));
                }
                else if (current_mode == AppMode::STREAMING_TRAJECTORY) {
                    if (!tflite_ok) {
                        current_mode = AppMode::STREAMING_RAW;
                        logger.warn_rate_limited("Trajectory mode unavailable - TFLite not initialized", 1s);
                        return false;
                    }

                    gpio_set_level(DATA_LED_PIN, 1);

                    sys.step(accel_vec, gyro_vec, force_val);
                    const auto& state = sys.get_state();

                    // Prepare point for Swinging Door algorithm
                    trajecto::SwingingDoor::Point point;
                    point.timestamp_us = static_cast<uint32_t>(now);
                    point.position = state.pos;
                    point.velocity = state.vel;
                    point.quat = state.quat;
                    point.pen_state = (force_val > 100);  // Pen down if force > threshold

                    // Swinging Door callback: sends packet via BLE with appropriate flags
                    auto send_callback = [&](const trajecto::SwingingDoor::Point& pt,
                                            trajecto::SwingingDoor::SendReason reason) {
                        struct __attribute__((packed)) {
                            Header h;
                            TrajectoryPacket p;
                        } pkt;

                        pkt.h.type = PacketType::DATA_TRAJECTORY;
                        pkt.h.length = sizeof(TrajectoryPacket);
                        pkt.p.timestamp_us = pt.timestamp_us;
                        pkt.p.pos[0] = pt.position.x();
                        pkt.p.pos[1] = pt.position.y();
                        pkt.p.pos[2] = pt.position.z();
                        pkt.p.vel[0] = pt.velocity.x();
                        pkt.p.vel[1] = pt.velocity.y();
                        pkt.p.vel[2] = pt.velocity.z();
                        pkt.p.quat[0] = pt.quat.w();
                        pkt.p.quat[1] = pt.quat.x();
                        pkt.p.quat[2] = pt.quat.y();
                        pkt.p.quat[3] = pt.quat.z();
                        pkt.p.zupt_prob = sys.get_zupt_prob();

                        // Set flags based on send reason
                        pkt.p.flags = 0;
                        if (reason == trajecto::SwingingDoor::SendReason::TIME_GAP) {
                            pkt.p.flags |= TrajectoryFlags::TRAJ_FLAG_ABSOLUTE_REF;
                        }
                        if (pt.pen_state) {
                            pkt.p.flags |= TrajectoryFlags::TRAJ_FLAG_PEN_DOWN;
                        }
                        if (reason == trajecto::SwingingDoor::SendReason::PEN_STATE_CHANGE) {
                            pkt.p.flags |= TrajectoryFlags::TRAJ_FLAG_KEYFRAME;
                        }
                        pkt.p.reserved[0] = 0;
                        pkt.p.reserved[1] = 0;
                        pkt.p.reserved[2] = 0;

                        send_notification(&pkt, sizeof(pkt));
                    };

                    // Process through Swinging Door (only sends when necessary)
                    if (current_config.enable_sda) {
                        compressor.process(point, send_callback);
                    } else {
                        // Bypass compression: send every point
                        send_callback(point, trajecto::SwingingDoor::SendReason::DOOR_EXCEEDED);
                    }
                }
            }
        }
        return false;
    };

  espp::Task imu_task({
      .callback = task_fn,
      .task_config = {
          .name = "BMI270",
          .stack_size_bytes = 10 * 1024,
          .priority = 10,
          .core_id = 0,
      }});

  logger.info("Starting tasks...");
  imu_task.start();

  while (true) {
    if (calibration_requested.load()) {
      logger.info("Calibration requested, pausing IMU task...");

      auto prev_mode = current_mode.load();
      current_mode = AppMode::IDLE;

      std::this_thread::sleep_for(100ms);

      perform_calibration_with_status(imu);

      calibration_requested = false;

      current_mode = prev_mode;

      logger.info("Calibration complete, resuming operations.");
    }

    std::this_thread::sleep_for(100ms);
  }
}