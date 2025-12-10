idf_component_register(
    SRCS
        "main.cpp"
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
