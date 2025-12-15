idf_component_register(
    SRCS
        "main.cpp"
        "eskf.cpp"
        "tcn_wrapper.cpp"
        "tcn_model.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        espp__bmi270
        espp__i2c
        espp__task
        espp__logger
        esp_adc
        bt
        nvs_flash
        eigen
        espressif__esp-tflite-micro
)
