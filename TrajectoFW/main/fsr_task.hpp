#ifndef FSR_TASK_HPP
#define FSR_TASK_HPP

#include <driver/adc.h>
#include <hal/adc_types.h>
#include <driver/gpio.h>
#include <esp_err.h>

// Define FSR related GPIOs and ADC channel
#define FSR_ADC_UNIT        ADC_UNIT_1
#define FSR_ADC_CHANNEL     ADC_CHANNEL_2    // GPIO3 (Analog Pin 3)
#define FSR_GAIN_PIN        GPIO_NUM_6       // Digital Pin 6

// Define hysteresis thresholds (to be tuned)
// These values are placeholders and depend on the FSR circuit and ADC resolution.
#define FSR_HIGH_GAIN_SATURATION_THRESHOLD      3800 // ADC value when high gain is saturated (near 4095 for 12-bit ADC)
#define FSR_LOW_GAIN_NOISE_FLOOR_THRESHOLD      500  // ADC value when low gain is at noise floor (above 0)
#define FSR_LOW_GAIN_SWITCH_THRESHOLD           3000 // ADC value to switch from low to high gain
#define FSR_HIGH_GAIN_SWITCH_THRESHOLD          1000 // ADC value to switch from high to low gain


/**
 * @brief Initialize FSR sensor components (ADC and GPIO for gain control).
 *
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t fsr_init();

/**
 * @brief Read FSR value and manage dual-gain logic with hysteresis.
 *
 * This function reads the ADC value from the FSR, applies hysteresis to
 * dynamically switch between high and low gain states, and returns the
 * raw ADC value along with the current gain state.
 *
 * @param out_fsr_val Pointer to store the raw FSR ADC value.
 * @param out_gain_state Pointer to store the current gain state (0 for low, 1 for high).
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t fsr_read(uint16_t *out_fsr_val, uint8_t *out_gain_state);

#endif // FSR_TASK_HPP
