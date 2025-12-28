idf_component_register(
    SRCS
        "data_acquire.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        espp__bmi270
        BMI270_SensorAPI
        espp__i2c
        espp__task
        espp__logger
        esp_adc
        #esp_nn
        bt
        nvs_flash
        eigen
        esp-tflite-micro
)
