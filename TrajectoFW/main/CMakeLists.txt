idf_component_register(SRCS "ble_task.cpp" "imu_task.cpp" "fsr_task.cpp" "ring_buffer.cpp" "main.cpp"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES "driver" "nimble")
