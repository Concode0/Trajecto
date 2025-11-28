#include "fsr_task.hpp"
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"

static const char *TAG = "FSR_TASK";

static adc_oneshot_unit_handle_t adc1_handle;
static uint8_t current_gain_state = 0; // 0: Low Gain, 1: High Gain (Initial state)

esp_err_t fsr_init() {
    esp_err_t ret;

    // 1. Initialize ADC
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = FSR_ADC_UNIT,
    };
    ret = adc_oneshot_new_unit(&init_config, &adc1_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create ADC unit: %s", esp_err_to_name(ret));
        return ret;
    }

    adc_oneshot_chan_cfg_t config = {
        .atten = ADC_ATTEN_DB_11, // Max attenuation, 0-3.9V input
        .bitwidth = ADC_BITWIDTH_12, // 12-bit resolution
    };
    ret = adc_oneshot_config_channel(adc1_handle, FSR_ADC_CHANNEL, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure ADC channel: %s", esp_err_to_name(ret));
        return ret;
    }

    // 2. Initialize GPIO for gain control
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << FSR_GAIN_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure FSR gain GPIO: %s", esp_err_to_name(ret));
        return ret;
    }

    // Set initial gain state (e.g., low gain)
    gpio_set_level(FSR_GAIN_PIN, current_gain_state); // Assuming 0 for low gain, 1 for high gain

    ESP_LOGI(TAG, "FSR initialized (ADC Unit %d, Channel %d, Gain Pin %d)",
             FSR_ADC_UNIT, FSR_ADC_CHANNEL, FSR_GAIN_PIN);
    return ESP_OK;
}

esp_err_t fsr_read(uint16_t *out_fsr_val, uint8_t *out_gain_state) {
    int adc_raw = 0;
    esp_err_t ret = adc_oneshot_read(adc1_handle, FSR_ADC_CHANNEL, &adc_raw);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read ADC: %s", esp_err_to_name(ret));
        *out_fsr_val = 0;
        *out_gain_state = current_gain_state; // Return current state on error
        return ret;
    }

    // Hysteresis Logic for Dual-Gain Control
    if (current_gain_state == 0) { // Currently in Low Gain
        // If ADC value is near saturation, switch to High Gain
        if (adc_raw > FSR_LOW_GAIN_SWITCH_THRESHOLD) {
            gpio_set_level(FSR_GAIN_PIN, 1);
            current_gain_state = 1;
            ESP_LOGD(TAG, "Switched to High Gain. ADC: %d", adc_raw);
        }
    } else { // Currently in High Gain
        // If ADC value is near noise floor, switch to Low Gain
        if (adc_raw < FSR_HIGH_GAIN_SWITCH_THRESHOLD) {
            gpio_set_level(FSR_GAIN_PIN, 0);
            current_gain_state = 0;
            ESP_LOGD(TAG, "Switched to Low Gain. ADC: %d", adc_raw);
        }
    }

    *out_fsr_val = (uint16_t)adc_raw;
    *out_gain_state = current_gain_state;
    return ESP_OK;
}
