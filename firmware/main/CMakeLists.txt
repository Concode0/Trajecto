idf_component_register(
    SRCS
        "main.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        espp__bmi270
        BMI270_SensorAPI
        espp__i2c
        espp__task
        espp__logger
        esp_adc
        bt
        nvs_flash
        trajecto_core
        trajecto_protocol
)