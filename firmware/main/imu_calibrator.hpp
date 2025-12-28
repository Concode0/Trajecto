// main/imu_calibrator.hpp
#pragma once

#include "bmi270.h"       // components/bmi270_driver에 있는 헤더
#include "espp/i2c.hpp"   // 아키텍트님이 이미 쓰고 계신 라이브러리
#include "esp_log.h"

static const char* TAG_CALIB = "IMU_CALIB";

class ImuCalibrator {
public:
    // ---------------------------------------------------------
    // 1. Bosch API <-> ESPP 연결 다리 (Bridge Functions)
    // ---------------------------------------------------------
    static int8_t i2c_read(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
        auto *i2c = static_cast<espp::I2c *>(intf_ptr);
        // espp i2c read wrapper (주소 0x68 또는 0x69 확인 필요)
        // 여기서는 0x68을 가정, 만약 SDO 핀 설정이 다르면 0x69로 변경
        if (i2c->read_at_register(BMI2_I2C_PRIM_ADDR, reg_addr, data, len)) {
            return BMI2_OK;
        }
        return BMI2_E_COM_FAIL;
    }

    static int8_t i2c_write(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
        auto *i2c = static_cast<espp::I2c *>(intf_ptr);
        if (i2c->write_at_register(BMI2_I2C_PRIM_ADDR, reg_addr, data, len)) {
            return BMI2_OK;
        }
        return BMI2_E_COM_FAIL;
    }

    static void delay_us(uint32_t period, void *intf_ptr) {
        // Microsecond delay using FreeRTOS ticks (min 1ms)
        uint32_t delay_ms = (period / 1000) + 1;
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }

    // ---------------------------------------------------------
    // 2. 핵심 기능: CRT(감도 보정) 실행
    // ---------------------------------------------------------
    static void run_crt(espp::I2c *i2c_handle) {
        if (!i2c_handle) {
            ESP_LOGE(TAG_CALIB, "I2C Handle is NULL");
            return;
        }

        struct bmi2_dev dev;
        int8_t rslt;

        // 구조체 세팅
        dev.read = i2c_read;
        dev.write = i2c_write;
        dev.delay_us = delay_us;
        dev.intf = BMI2_I2C_INTF;
        dev.read_write_len = 32; 
        dev.intf_ptr = (void*)i2c_handle; // 내 C++ 객체를 Bosch에게 넘겨줌
        dev.config_file_ptr = NULL; // 이미 espp::Bmi270에서 초기화 했다면 NULL

        ESP_LOGW(TAG_CALIB, "------------------------------------------------");
        ESP_LOGW(TAG_CALIB, "WARNING: STARTING CRT CALIBRATION");
        ESP_LOGW(TAG_CALIB, "DO NOT TOUCH THE PEN! (KEEP IT STILL ON TABLE)");
        ESP_LOGW(TAG_CALIB, "------------------------------------------------");

        // [중요] 초기화 확인 (이미 되어 있겠지만, 안전장치)
        // Bosch 드라이버가 칩을 인식하는지 확인
        rslt = bmi270_init(&dev);
        if (rslt != BMI2_OK) {
            ESP_LOGE(TAG_CALIB, "Init failed (Code: %d). Sensor connected?", rslt);
            return;
        }

        // [핵심] CRT 실행
        // 납땜으로 망가진 감도를 공장 초기화 상태로 복구
        rslt = bmi2_do_crt(&dev);

        if (rslt == BMI2_OK) {
            ESP_LOGI(TAG_CALIB, "✅ CRT SUCCESS! Sensitivity restored.");
            
            // [옵션] 영점(Offset) 보정도 같이 수행 (추천)
            struct bmi2_sens_config config;
            config.type = BMI2_ACCEL | BMI2_GYRO;
            bmi2_perform_accel_offset_compensation(&config, &dev);
            bmi2_perform_gyro_offset_compensation(&config, &dev);
            ESP_LOGI(TAG_CALIB, "✅ Offset Compensation SUCCESS!");
            
        } else {
            ESP_LOGE(TAG_CALIB, "❌ CRT FAILED (Code: %d).", rslt);
            ESP_LOGE(TAG_CALIB, "Did you move the pen? Is it flat?");
        }
    }
};