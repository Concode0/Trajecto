# Trajecto Firmware

This directory contains the ESP-IDF firmware for the Trajecto project. The firmware is written in C++ and utilizes the `espp` component library.

## Project Overview

The firmware is designed to run on an ESP32 microcontroller connected to a BMI270 Inertial Measurement Unit (IMU) and a Force Sensitive Resistor (FSR). Its primary function is to:
1.  Acquire sensor data (accelerometer and gyroscope) from the BMI270 at a high frequency (400Hz).
2.  Read analog values from an FSR.
3.  Transmit the batched sensor data over Bluetooth Low Energy (BLE) to a connected client.
4.  Receive commands ("start" and "stop") from the client via BLE to control the data transmission.

The firmware is structured around FreeRTOS tasks and uses hardware interrupts for efficient, low-latency data capture from the IMU.

## Hardware

*   **Microcontroller:** ESP32 series MCU.
*   **IMU:** Bosch BMI270 connected via I2C.
    *   **SCL:** GPIO 4
    *   **SDA:** GPIO 10
    *   **Interrupt Pin:** GPIO 1
*   **FSR:** Force Sensitive Resistor connected to an ADC pin.
    *   **ADC Input:** GPIO 3
    *   **Resisotr Selector:** GPIO 6
*   **Status LED:**
    *   **Data LED:** GPIO 7 (lights up when sending data via BLE)

## Features

*   **High-Frequency Data Acquisition:** Reads IMU data at 400Hz, driven by the BMI270's data-ready interrupt.
*   **Sensor Fusion:** Provides raw data streams for 3-axis accelerometer and 3-axis gyroscope.
*   **Analog Sensing:** Includes support for reading an FSR to detect force/pressure.
*   **BLE Communication:** Implements a custom GATT service for wireless data streaming and remote control.
*   **Data Batching:** Batches sensor readings into groups of three to optimize BLE throughput and reduce overhead.
*   **Remote Control:** A client can start and stop the data stream by writing to a specific BLE characteristic.
*   **C++ Abstraction:** Built using the `espp` C++ library, which provides high-level, modern C++ abstractions for ESP-IDF features and peripherals.

## BLE Service

The firmware exposes a custom BLE service to facilitate communication.

*   **Device Name:** `Trajecto`
*   **Service UUID:** `AD43434E-C549-4594-B474-543153544557`
*   **Characteristics:**
    *   **Data Characteristic (Notify):** `AD43434F-C549-4594-B474-543153544557`
        *   Streams batched sensor data to the client when notifications are enabled.
        *   Each notification contains a packet of 3 consecutive sensor readings.
    *   **Command Characteristic (Write):** `AD43434D-C549-4594-B474-543153544557`
        *   Allows a connected client to control the firmware.
        *   Write `"strt"` (4 bytes) to start the data stream.
        *   Write `"stop"` (4 bytes) to stop the data stream.

## Building and Flashing

This project is based on the ESP-IDF framework.

### Prerequisites

1.  **ESP-IDF:** Ensure you have the ESP-IDF development environment installed and configured. Follow the official [ESP-IDF Get Started guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html).
2.  **Dependencies:** The project dependencies are managed by the ESP-IDF component manager and are defined in the `dependencies.lock` file. They will be downloaded automatically during the first build.

### Build Commands

1.  **Open a terminal** and navigate to the `TrajectoFW` directory.
2.  **Set the target chip** (e.g., `esp32`, `esp32s3`):
    ```sh
    idf.py set-target esp32c3
    ```
3.  **Build the project:**
    ```sh
    idf.py build
    ```
4.  **Flash the firmware** to the device:
    ```sh
    idf.py flash
    ```
5.  **Monitor the serial output:**
    ```sh
    idf.py monitor
    ```

You can also combine these commands:
```sh
idf.py flash monitor
```