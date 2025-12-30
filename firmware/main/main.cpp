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
#include "esp_timer.h"

// Bosch Driver Headers
#include "bmi270.h"

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

using namespace std::chrono_literals;

// ----------------------------------------------------------------------------
// Globals & Constants
// ----------------------------------------------------------------------------

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
    .mode = 1,      // Default: Trajectory mode
    .odr_hz = 50,   // Fixed ODR
    .reserved = {0, 0}
};

// Calibration Control
static std::atomic<bool> calibration_requested(false);
static espp::I2c* g_i2c = nullptr;
static uint8_t g_bmi270_address = 0;

// Semaphore for IMU Data Ready
static SemaphoreHandle_t imu_sem = nullptr;

// Forward declarations
static void send_notification(void *data, size_t len);

// ----------------------------------------------------------------------------
// Calibration / Bosch Driver Bridges (from data_acquire.cpp)
// ----------------------------------------------------------------------------

// Context for I2C communication with device address
struct I2cContext {
    espp::I2c *i2c;
    uint8_t address;
};

static int8_t bmi2_i2c_read_bridge(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
    auto *ctx = static_cast<I2cContext *>(intf_ptr);
    if (ctx->i2c->read_at_register(ctx->address, reg_addr, data, len)) {
        return BMI2_OK;
    }
    return BMI2_E_COM_FAIL;
}

static int8_t bmi2_i2c_write_bridge(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
    auto *ctx = static_cast<I2cContext *>(intf_ptr);
    
    constexpr size_t MAX_WRITE_BUFFER_SIZE = 256;
    if (len + 1 > MAX_WRITE_BUFFER_SIZE) {
        return BMI2_E_COM_FAIL;
    }

    uint8_t write_buffer[MAX_WRITE_BUFFER_SIZE];
    write_buffer[0] = reg_addr;
    memcpy(&write_buffer[1], data, len);

    // Use the generic write function
    if (ctx->i2c->write(ctx->address, write_buffer, len + 1)) {
        return BMI2_OK;
    }
    return BMI2_E_COM_FAIL;
}

static void bmi2_delay_us_bridge(uint32_t period, void *intf_ptr) {
    if (period == 0) return;
    uint32_t delay_ms = (period / 1000) + 1;
    TickType_t ticks = pdMS_TO_TICKS(delay_ms);
    if (ticks == 0) ticks = 1; // Ensure at least 1 tick wait
    vTaskDelay(ticks);
}

// NVS Configuration
static const char* NVS_NAMESPACE = "calibration";
static const char* KEY_GYRO_FOC_X = "gfoc_x";
static const char* KEY_GYRO_FOC_Y = "gfoc_y";
static const char* KEY_GYRO_FOC_Z = "gfoc_z";

static bool load_foc_from_nvs(struct bmi2_sens_axes_data* foc) {
    nvs_handle_t my_handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &my_handle);
    if (err != ESP_OK) return false;

    int16_t x=0, y=0, z=0;
    if (nvs_get_i16(my_handle, KEY_GYRO_FOC_X, &x) == ESP_OK &&
        nvs_get_i16(my_handle, KEY_GYRO_FOC_Y, &y) == ESP_OK &&
        nvs_get_i16(my_handle, KEY_GYRO_FOC_Z, &z) == ESP_OK) {
        foc->x = x;
        foc->y = y;
        foc->z = z;
        nvs_close(my_handle);
        return true;
    }
    nvs_close(my_handle);
    return false;
}

static void save_foc_to_nvs(const struct bmi2_sens_axes_data* foc) {
    nvs_handle_t my_handle;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &my_handle) == ESP_OK) {
        nvs_set_i16(my_handle, KEY_GYRO_FOC_X, foc->x);
        nvs_set_i16(my_handle, KEY_GYRO_FOC_Y, foc->y);
        nvs_set_i16(my_handle, KEY_GYRO_FOC_Z, foc->z);
        nvs_commit(my_handle);
        nvs_close(my_handle);
    }
}

// Perform calibration and send BLE status update
static void perform_calibration_with_status() {
    using namespace trajecto::protocol;

    espp::Logger logger({.tag = "CALIB", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting runtime calibration...");

    if (!g_i2c || g_bmi270_address == 0) {
        logger.error("Calibration failed: I2C not initialized");

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

    // Perform calibration (reuse ensure_calibration logic inline)
    struct bmi2_dev dev;
    I2cContext ctx = {g_i2c, g_bmi270_address};

    dev.read = bmi2_i2c_read_bridge;
    dev.write = bmi2_i2c_write_bridge;
    dev.delay_us = bmi2_delay_us_bridge;
    dev.intf = BMI2_I2C_INTF;
    dev.read_write_len = 32;
    dev.intf_ptr = &ctx;
    dev.config_file_ptr = NULL;

    int8_t rslt = bmi270_init(&dev);
    if (rslt != BMI2_OK) {
        logger.error("Failed to init raw driver: {}", rslt);

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

    logger.warn("========================================");
    logger.warn("STARTING CALIBRATION SEQUENCE");
    logger.warn("KEEP THE PEN STILL ON THE TABLE!");
    logger.warn("========================================");

    // Execute CRT (Sensitivity Fix)
    rslt = bmi2_do_crt(&dev);
    if (rslt != BMI2_OK) {
        logger.error("CRT FAILED code: {}. (Did you move the pen?)", rslt);

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

    // FOC (Offset Compensation)
    uint8_t sens_list[1] = {BMI2_GYRO};
    rslt = bmi2_sensor_enable(sens_list, 1, &dev);
    if (rslt == BMI2_OK) {
        vTaskDelay(pdMS_TO_TICKS(100));
        rslt = bmi2_perform_gyro_foc(&dev);
    }

    if (rslt != BMI2_OK) {
        logger.error("Gyro FOC Failed: {}", rslt);

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

    // Read back and save offsets
    struct bmi2_sens_axes_data foc_offset;
    if (bmi2_read_gyro_offset_comp_axes(&foc_offset, &dev) == BMI2_OK) {
        save_foc_to_nvs(&foc_offset);
        logger.info("Calibration Saved to NVS.");

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

// Executes CRT to fix 17m error and performs Gyro FOC, using NVS to persist results
// Returns true if calibration exists or was successfully completed
static bool ensure_calibration(espp::I2c *i2c, uint8_t dev_addr) {
    espp::Logger logger({.tag = "CALIB", .level = espp::Logger::Verbosity::INFO});

    // 1. Check NVS for existing FOC data
    struct bmi2_sens_axes_data foc_offset;
    if (load_foc_from_nvs(&foc_offset)) {
        logger.info("Found saved FOC offsets: X={} Y={} Z={}", foc_offset.x, foc_offset.y, foc_offset.z);
        logger.info("Calibration will be restored after IMU initialization.");
        return true; // Calibration exists, will be applied later
    }

    logger.warn("No calibration found in NVS - running CRT+FOC sequence...");

    // 2. No calibration found - run CRT + FOC
    struct bmi2_dev dev;
    I2cContext ctx = {i2c, dev_addr};

    // Configure Bosch BMI2 driver struct
    dev.read = bmi2_i2c_read_bridge;
    dev.write = bmi2_i2c_write_bridge;
    dev.delay_us = bmi2_delay_us_bridge;
    dev.intf = BMI2_I2C_INTF;
    dev.read_write_len = 32;
    dev.intf_ptr = &ctx;
    dev.config_file_ptr = NULL;

    // Initialize sensor with raw driver (needed to access calibration APIs)
    int8_t rslt = bmi270_init(&dev);
    if (rslt != BMI2_OK) {
        logger.error("Failed to init raw driver: {}", rslt);
        return false;
    }

    logger.warn("========================================");
    logger.warn("STARTING CALIBRATION SEQUENCE");
    logger.warn("KEEP THE PEN STILL ON THE TABLE!");
    logger.warn("========================================");

    // Execute CRT (Sensitivity Fix)
    rslt = bmi2_do_crt(&dev);
    if (rslt == BMI2_OK) {
        logger.info("CRT SUCCESS! Sensitivity Restored.");

        // FOC (Offset Compensation)
        uint8_t sens_list[1] = {BMI2_GYRO};
        rslt = bmi2_sensor_enable(sens_list, 1, &dev);
        if (rslt == BMI2_OK) {
             vTaskDelay(pdMS_TO_TICKS(100));
             rslt = bmi2_perform_gyro_foc(&dev);
        }

        if (rslt == BMI2_OK) {
            logger.info("Gyro FOC Done.");

            // Read back the calculated offsets
            if (bmi2_read_gyro_offset_comp_axes(&foc_offset, &dev) == BMI2_OK) {
                save_foc_to_nvs(&foc_offset);
                logger.info("Calibration Saved to NVS.");
                return true;
            } else {
                logger.error("Failed to read back FOC offsets.");
                return false;
            }
        } else {
            logger.error("Gyro FOC Failed: {}", rslt);
            return false;
        }
    } else {
        logger.error("CRT FAILED code: {}. (Did you move the pen?)", rslt);
        return false;
    }
}

// ----------------------------------------------------------------------------
// BLE Logic
// ----------------------------------------------------------------------------

static int ble_gap_event(struct ble_gap_event *event, void *arg);

static int gatt_svr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                              struct ble_gatt_access_ctxt *ctxt, void *arg);

// UUIDs
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
    {0}, /* No more services */
};

// Send a packet notification
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
            // Minimal size check (Header)
            if (os_mbuf_len(ctxt->om) >= sizeof(trajecto::protocol::Header)) {
                // Copy data to buffer
                uint8_t buf[32]; 
                uint16_t len = os_mbuf_len(ctxt->om);
                if (len > sizeof(buf)) len = sizeof(buf);
                ble_hs_mbuf_to_flat(ctxt->om, buf, len, NULL);

                auto* header = reinterpret_cast<trajecto::protocol::Header*>(buf);
                using namespace trajecto::protocol;

                switch (header->type) {
                    case PacketType::CMD_PING: {
                        printf("CMD: Ping\n");
                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_PONG;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_GET_CONFIG: {
                        printf("CMD: Get Config\n");
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
                                printf("CMD: Set Mode RAW\n");
                            } else {
                                if (tflite_ok) {
                                    current_mode = AppMode::STREAMING_TRAJECTORY;
                                    printf("CMD: Set Mode TRAJECTORY\n");
                                } else {
                                    printf("CMD: TRAJECTORY mode rejected - TFLite not initialized\n");
                                    // Keep in current mode or force to RAW
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
                                } else {
                                    printf("CMD: TRAJECTORY mode rejected - TFLite not initialized, using RAW mode\n");
                                    current_mode = AppMode::STREAMING_RAW;
                                }
                            }
                        }
                        printf("CMD: Start Stream (Mode: %d)\n", (uint8_t)current_mode.load());

                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_STREAM_STARTED;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_STOP_STREAM: {
                        current_mode = AppMode::IDLE;
                        printf("CMD: Stop Stream\n");

                        struct {
                            Header h;
                        } rsp;
                        rsp.h.type = PacketType::RSP_STREAM_STOPPED;
                        rsp.h.length = 0;
                        send_notification(&rsp, sizeof(rsp));
                        break;
                    }

                    case PacketType::CMD_CALIBRATE: {
                        printf("CMD: Calibrate (requesting...)\n");
                        calibration_requested = true;

                        // Send initial status: calibration started
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
                        printf("Unknown Command: 0x%02X\n", (uint8_t)header->type);
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

    fields.name = (uint8_t *)device_name;
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
            printf("Connected\n");
            
            // HANDSHAKE: Send status packet immediately
            using namespace trajecto::protocol;
            struct {
                Header h;
                // No payload for generic ready, or use RSP_STREAM_STOPPED
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
        current_mode = AppMode::IDLE; // Stop streaming on disconnect
        printf("Disconnected\n");
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

void ble_on_reset(int reason) { (void)reason; }

void nimble_host_task(void *param) {
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void IRAM_ATTR isr_handler(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(imu_sem, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

extern "C" void app_main(void) {
    espp::Logger logger({.tag = "Trajecto System", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting Trajecto System!");

    // Initialize NVS.
    esp_err_t ret = nvs_flash_init();

    // BLE Init
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

    // Configure data LED pin
    static constexpr gpio_num_t DATA_LED_PIN = GPIO_NUM_7;
    gpio_config_t led_io_conf = {};
    led_io_conf.mode = GPIO_MODE_OUTPUT;
    led_io_conf.pin_bit_mask = (1ULL << DATA_LED_PIN);
    gpio_config(&led_io_conf);
    gpio_set_level(DATA_LED_PIN, 0); 


    // Configure GPIO 6 to control the FSR resistor
    static constexpr gpio_num_t FSR_ENABLE_PIN = GPIO_NUM_6;
    gpio_config_t fsr_io_conf = {};
    fsr_io_conf.mode = GPIO_MODE_OUTPUT;
    fsr_io_conf.pin_bit_mask = (1ULL << FSR_ENABLE_PIN);
    gpio_config(&fsr_io_conf);
    gpio_set_level(FSR_ENABLE_PIN, 1); // When High 10M, Low 10k

    // FSR ADC configuration
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
    static constexpr gpio_num_t i2c_sda = (gpio_num_t)10;
    static constexpr gpio_num_t i2c_scl = (gpio_num_t)4;
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

    // Store for runtime calibration
    g_i2c = &i2c;
    g_bmi270_address = bmi270_address;

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

    // --- CRT & FOC Calibration (check/perform if needed) ---
    vTaskDelay(pdMS_TO_TICKS(500));
    bool calib_ok = ensure_calibration(&i2c, bmi270_address);
    if (!calib_ok) {
        logger.error("Calibration failed! System may not work correctly.");
    }
    // -------------------------------------------------------

    // Create the IMU
    Imu imu(config);

    std::error_code ec;
    imu.configure_interrupts(int_config, ec);

    // Apply calibration offsets after IMU initialization
    struct bmi2_sens_axes_data foc_offset;
    if (load_foc_from_nvs(&foc_offset)) {
        logger.info("Restoring calibration offsets to sensor...");
        // Use raw driver to restore offsets to the now-initialized sensor
        struct bmi2_dev dev;
        I2cContext ctx = {&i2c, bmi270_address};
        dev.read = bmi2_i2c_read_bridge;
        dev.write = bmi2_i2c_write_bridge;
        dev.delay_us = bmi2_delay_us_bridge;
        dev.intf = BMI2_I2C_INTF;
        dev.read_write_len = 32;
        dev.intf_ptr = &ctx;
        dev.config_file_ptr = NULL;

        // Note: We don't call bmi270_init() here since sensor is already initialized
        // Just write the calibration offsets directly
        int8_t rslt = bmi2_write_gyro_offset_comp_axes(&foc_offset, &dev);
        if (rslt == BMI2_OK) {
            rslt = bmi2_set_gyro_offset_comp(BMI2_ENABLE, &dev);
            if (rslt == BMI2_OK) {
                logger.info("Calibration restored successfully!");
            } else {
                logger.error("Failed to enable gyro compensation: {}", rslt);
            }
        } else {
            logger.error("Failed to write gyro offsets: {}", rslt);
        }
    }

    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_NEGEDGE; 
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << interrupt_pin);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);

    gpio_install_isr_service(0);
    gpio_isr_handler_add(interrupt_pin, isr_handler, nullptr);

    // --- Trajecto System Initialization ---
    static trajecto::TrajectoSystem sys;
    if (!sys.setup()) {
        logger.error("Failed to setup Trajecto System (TFLite)!");
        logger.error("System will enter IDLE mode. Only RAW IMU streaming available.");
        tflite_ok = false;
    } else {
        logger.info("Trajecto System (TFLite) setup complete.");
        tflite_ok = true;
    }

    // Print CSV Header for Raw Logger
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

            auto accel = imu.get_accelerometer(); // g
            auto gyro = imu.get_gyroscope();     // °/s (from BMI270 driver)
            float temp = imu.get_temperature();  // °C

            int fsr_raw;
            ESP_ERROR_CHECK(adc_oneshot_read(adc_handle, FSR_ADC_CHANNEL, &fsr_raw));

            // --- Raw Data Logging (CSV to UART) ---
            // Matches format from raw_data_logger.cpp
            // Convert gyro from °/s to rad/s to match header
            constexpr float DEG_TO_RAD = M_PI / 180.0f;
            printf("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.2f\n",
                   accel.x, accel.y, accel.z,
                   gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD,
                   temp);

            // Unit conversion for BLE/ESKF
            Eigen::Vector3f accel_vec(accel.x * 9.81f, accel.y * 9.81f, accel.z * 9.81f);  // g → m/s²
            Eigen::Vector3f gyro_vec(gyro.x * DEG_TO_RAD, gyro.y * DEG_TO_RAD, gyro.z * DEG_TO_RAD);  // °/s → rad/s
            float force_val = (float)fsr_raw;

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
                    pkt.p.timestamp_us = (uint32_t)now;
                    pkt.p.accel[0] = accel_vec.x(); // m/s^2
                    pkt.p.accel[1] = accel_vec.y();
                    pkt.p.accel[2] = accel_vec.z();
                    pkt.p.gyro[0] = gyro_vec.x();   // rad/s
                    pkt.p.gyro[1] = gyro_vec.y();
                    pkt.p.gyro[2] = gyro_vec.z();
                    pkt.p.force = (int16_t)force_val;
                    pkt.p.temperature = temp;       // °C

                    send_notification(&pkt, sizeof(pkt));
                }
                else if (current_mode == AppMode::STREAMING_TRAJECTORY) {
                    // Only run trajectory mode if TFLite initialized successfully
                    if (!tflite_ok) {
                        // Force back to IDLE if TFLite not available
                        current_mode = AppMode::IDLE;
                        logger.warn_rate_limited("Trajectory mode unavailable - TFLite not initialized", 1s);
                        return false;
                    }

                    gpio_set_level(DATA_LED_PIN, 1);

                    // Run TCN + Filter
                    sys.step(accel_vec, gyro_vec, force_val);
                    const auto& state = sys.get_state();

                    struct __attribute__((packed)) {
                        Header h;
                        TrajectoryPacket p;
                    } pkt;
                    pkt.h.type = PacketType::DATA_TRAJECTORY;
                    pkt.h.length = sizeof(TrajectoryPacket);
                    pkt.p.timestamp_us = (uint32_t)now;
                    pkt.p.pos[0] = state.pos.x();
                    pkt.p.pos[1] = state.pos.y();
                    pkt.p.pos[2] = state.pos.z();
                    pkt.p.vel[0] = state.vel.x();
                    pkt.p.vel[1] = state.vel.y();
                    pkt.p.vel[2] = state.vel.z();
                    pkt.p.quat[0] = state.quat.w();
                    pkt.p.quat[1] = state.quat.x();
                    pkt.p.quat[2] = state.quat.y();
                    pkt.p.quat[3] = state.quat.z();
                    pkt.p.prob_zupt = sys.get_zupt_prob(); 

                    send_notification(&pkt, sizeof(pkt));
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
    // Check for calibration request
    if (calibration_requested.load()) {
      logger.info("Calibration requested, pausing IMU task...");

      // Stop streaming during calibration
      auto prev_mode = current_mode.load();
      current_mode = AppMode::IDLE;

      // Wait a bit for current operations to complete
      std::this_thread::sleep_for(100ms);

      // Perform calibration with status updates
      perform_calibration_with_status();

      // Clear the request
      calibration_requested = false;

      // Resume previous mode if it was streaming
      current_mode = prev_mode;

      logger.info("Calibration complete, resuming operations.");
    }

    std::this_thread::sleep_for(100ms);
  }
}

/*
Current Monitor Logs.

# Check it save Calibration (CRT data)


[CALIB/I][6.633]: Found saved FOC offsets: X=3 Y=-1 Z=4
[CALIB/I][6.703]: Restored Calibration from NVS.
Didn't find op for builtin opcode 'CONCATENATION'
Failed to get registration from op code CONCATENATION
 
AllocateTensors failed!
[Trajecto System/E][6.703]: Failed to setup Trajecto System (TFLite)!
accel_x_g,accel_y_g,accel_z_g,gyro_x_rads,gyro_y_rads,gyro_z_rads
[Trajecto System/I][6.713]: Starting tasks...
*/