idf_component_register(
    SRCS
        "allan_variance_main.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        bmi270
        i2c
        task
        logger
        esp_adc
        bt
        nvs_flash
)
