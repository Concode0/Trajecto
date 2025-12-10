#include <atomic>
#include <chrono>
#include <vector>

#include "bmi270.hpp"
#include "i2c.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"

// For BLE
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "nvs_flash.h"

using namespace std::chrono_literals;

// BLE
static uint8_t ble_addr_type;
static uint16_t conn_handle;
static std::atomic<bool> is_connected(false);
static std::atomic<bool> should_send_data(false);
static const char *device_name = "Trajecto";
static uint16_t trajecto_chr_val_handle;
static uint16_t trajecto_cmd_val_handle;

struct SensorData {
    float time;
    float accel[3];
    float gyro[3];
    int fsr;
};

static int ble_gap_event(struct ble_gap_event *event, void *arg);

static int gatt_svr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                              struct ble_gatt_access_ctxt *ctxt, void *arg);

// Define the UUIDs as static const variables to avoid rvalue issues
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

static int
gatt_svr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                   struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        if (attr_handle == trajecto_cmd_val_handle) {
            if (os_mbuf_len(ctxt->om) >= 4) {
                char cmd[5];
                memcpy(cmd, ctxt->om->om_data, 4);
                cmd[4] = '\0';
                if (strcmp(cmd, "strt") == 0) {
                    printf("Start command received\n");
                    should_send_data = true;
                } else if (strcmp(cmd, "stop") == 0) {
                    printf("Stop command received\n");
                    should_send_data = false;
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
    if (rc != 0) {
        return;
    }

    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(ble_addr_type, NULL, BLE_HS_FOREVER, &adv_params, ble_gap_event, NULL);
    if (rc != 0) {
        return;
    }
}

static int ble_gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            is_connected = true;
            printf("Connected\n");
        } else {
            ble_advertise();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        is_connected = false;
        should_send_data = false;
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

void ble_on_reset(int reason) {
    (void)reason;
    // Handle reset
}

void nimble_host_task(void *param) {
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static SemaphoreHandle_t imu_sem = nullptr;

static void IRAM_ATTR isr_handler(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(imu_sem, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

extern "C" void app_main(void) {
    espp::Logger logger({.tag = "BMI270 Example", .level = espp::Logger::Verbosity::INFO});
    logger.info("Starting example!");

    // Initialize NVS.
    esp_err_t ret = nvs_flash_init();

    // BLE Init
    int rc = nimble_port_init();
    if (rc != 0) {
        return;
    }
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
    gpio_set_level(DATA_LED_PIN, 0); // Start with LED off


    // Configure GPIO 6 to control the FSR resistor
    static constexpr gpio_num_t FSR_ENABLE_PIN = GPIO_NUM_6;
    gpio_config_t fsr_io_conf = {};
    fsr_io_conf.mode = GPIO_MODE_OUTPUT;
    fsr_io_conf.pin_bit_mask = (1ULL << FSR_ENABLE_PIN);
    gpio_config(&fsr_io_conf);
    // and set it high
    gpio_set_level(FSR_ENABLE_PIN, 1); // When High 10M, Low 10k

    // FSR ADC configuration
    static constexpr auto FSR_ADC_UNIT = ADC_UNIT_1;
    static constexpr auto FSR_ADC_CHANNEL = ADC_CHANNEL_3;
    adc_oneshot_unit_handle_t adc_handle;
    adc_oneshot_unit_init_cfg_t adc_init_config = {
        .unit_id = FSR_ADC_UNIT,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&adc_init_config, &adc_handle));

    adc_oneshot_chan_cfg_t adc_chan_config = {
        .atten = ADC_ATTEN_DB_11,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, FSR_ADC_CHANNEL, &adc_chan_config));

    imu_sem = xSemaphoreCreateBinary();

    //! [bmi270 example]
    using Imu = espp::Bmi270<espp::bmi270::Interface::I2C>;

    // make the i2c we'll use to communicate
    static constexpr auto i2c_port = I2C_NUM_0;
    static constexpr auto i2c_clock_speed = 400 * 1000;
    static constexpr gpio_num_t i2c_sda = (gpio_num_t)10;
    static constexpr gpio_num_t i2c_scl = (gpio_num_t)4;
    espp::I2c i2c({.port = i2c_port,
                    .sda_io_num = i2c_sda,
                    .scl_io_num = i2c_scl,
                    .sda_pullup_en = GPIO_PULLUP_ENABLE,
                    .scl_pullup_en = GPIO_PULLUP_ENABLE,
                    .timeout_ms = 200, // need to be long enough for writing config file (8kb)
                    .clk_speed = i2c_clock_speed});

    // use the i2c to ping both of the possible BMI270 addresses and use the one that responds
    // This is necessary because the BMI270 can be configured to use either address
    // SDO pulled high or low.
    uint8_t bmi270_address = Imu::DEFAULT_ADDRESS;
    std::vector<uint8_t> addresses = {Imu::DEFAULT_ADDRESS, Imu::DEFAULT_ADDRESS_SDO_HIGH};
    for (auto address : addresses) {
        if (i2c.probe_device(address)) {
        logger.info("Found BMI270 at address: 0x{:02X}", address);
        bmi270_address = address;
        break;
        } else {
        logger.warn("No BMI270 found at address: 0x{:02X}", address);
        }
    }

    // make the IMU config
    Imu::Config config{
        .device_address = bmi270_address,
        .write = std::bind(&espp::I2c::write, &i2c, std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
        .read = std::bind(&espp::I2c::read, &i2c, std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
        .imu_config =
            {
                .accelerometer_range = Imu::AccelerometerRange::RANGE_4G,
                .accelerometer_odr = Imu::AccelerometerODR::ODR_400_HZ,
                .accelerometer_bandwidth = Imu::AccelerometerBandwidth::NORMAL_AVG4,
                .gyroscope_range = Imu::GyroscopeRange::RANGE_1000DPS,
                .gyroscope_odr = Imu::GyroscopeODR::ODR_400_HZ,
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

    // create the IMU
    Imu imu(config);

    std::error_code ec;
    imu.configure_interrupts(int_config, ec);

    // configure the interrupt pin
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_NEGEDGE; // Trigger on FALLING edge
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << interrupt_pin);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);

    // Install the ISR service
    gpio_install_isr_service(0);
    // Hook the ISR handler for our specific GPIO pin
    gpio_isr_handler_add(interrupt_pin, isr_handler, nullptr);

    static SensorData batch_buffer[3]; 
    static int batch_idx = 0;

    // make a task to read out the IMU data and print it to console
    auto task_fn = [&]() -> bool {
        if (xSemaphoreTake(imu_sem, portMAX_DELAY) == pdTRUE) {
            // Turn off data LED
            gpio_set_level(DATA_LED_PIN, 0);

            auto now = esp_timer_get_time(); // time in microseconds
            static auto t0 = now;
            auto t1 = now;
            float dt = (t1 - t0) / 1'000'000.0f; // convert us to s
            t0 = t1;

            std::error_code ec;
            // update the imu data
            if (!imu.update(dt, ec)) {
                logger.error("Failed to update IMU: {}", ec.message());
                return false;
            }

            // get accel
            auto accel = imu.get_accelerometer();
            auto gyro = imu.get_gyroscope();
            auto temp = imu.get_temperature();

            // get FSR
            int fsr_raw;
            ESP_ERROR_CHECK(adc_oneshot_read(adc_handle, FSR_ADC_CHANNEL, &fsr_raw));

            batch_buffer[batch_idx].time = now;
            batch_buffer[batch_idx].accel[0] = accel.x;
            batch_buffer[batch_idx].accel[1] = accel.y;
            batch_buffer[batch_idx].accel[2] = accel.z;
            batch_buffer[batch_idx].gyro[0] = gyro.x;
            batch_buffer[batch_idx].gyro[1] = gyro.y;
            batch_buffer[batch_idx].gyro[2] = gyro.z;
            batch_buffer[batch_idx].fsr = fsr_raw;

            batch_idx++;

            if (batch_idx >= 3) {
                if (is_connected && should_send_data) {
                    gpio_set_level(DATA_LED_PIN, 1);
                    struct os_mbuf *txom = ble_hs_mbuf_from_flat(batch_buffer, sizeof(batch_buffer));
                
                    if (trajecto_chr_val_handle != 0) {
                        int rc = ble_gatts_notify_custom(conn_handle, trajecto_chr_val_handle, txom);
                        if (rc != 0) {
                            logger.warn("Failed to send notification; rc={}", rc);
                        }
                    }
                }
                batch_idx = 0;
            }
        }
        return false;
    };

  espp::Task imu_task({
      .callback = task_fn,
      .task_config = {
          .name = "BMI270",
          .stack_size_bytes = 6 * 1024,
          .priority = 10,
          .core_id = 0,
      }});

  // print the header for the IMU data (for plotting)
  fmt::print("% Time (s), "
             // raw IMU data (accel, gyro, temp)
             "Accel X (g), Accel Y (g), Accel Z (g), "
             "Gyro X (°/s), Gyro Y (°/s), Gyro Z (°/s), "
             "Temp (°C), "
             "FSR (raw)");

  logger.info("Starting tasks...");
  imu_task.start();

  // loop forever
  while (true) {
    std::this_thread::sleep_for(1s);
  }
  //! [bmi270 example]
}