#pragma once
#include <stdint.h>

namespace trajecto {
namespace protocol {

// ----------------------------------------------------------------------------
// Packet Types
// ----------------------------------------------------------------------------

enum class PacketType : uint8_t {
    CMD_PING = 0x01,
    RSP_PONG = 0x02,
    
    CMD_SET_CONFIG = 0x10,
    RSP_CONFIG_OK  = 0x11,
    CMD_GET_CONFIG = 0x12,
    RSP_CONFIG     = 0x13,

    CMD_START_STREAM = 0x20,
    RSP_STREAM_STARTED = 0x21,
    CMD_STOP_STREAM  = 0x22,
    RSP_STREAM_STOPPED = 0x23,

    CMD_CALIBRATE    = 0x30, // Trigger CRT or Zero offset
    RSP_CALIB_STATUS = 0x31,

    DATA_RAW_IMU     = 0x80, // High throughput raw stream
    DATA_TRAJECTORY  = 0x81  // TCN output stream
};

// ----------------------------------------------------------------------------
// Payload Structures (Packed to ensure byte alignment over BLE)
// ----------------------------------------------------------------------------

#pragma pack(push, 1)

// -- Headers --
struct Header {
    PacketType type;
    uint8_t length; // Length of payload following this header
};

// -- Config --
struct ConfigPayload {
    uint8_t mode;      // 0: Raw, 1: Inference
    uint8_t odr_hz;    // e.g., 50
    uint8_t reserved[2];
};

// -- Data: Raw IMU --
struct RawImuPacket {
    uint32_t timestamp_us;
    float accel[3]; // x, y, z (m/s^2)
    float gyro[3];  // x, y, z (rad/s)
    int16_t force;  // FSR
    float temperature; // °C
};

// -- Data: Trajectory --
struct TrajectoryPacket {
    uint32_t timestamp_us;
    float pos[3];   // x, y, z (m)
    float vel[3];   // x, y, z (m/s)
    float quat[4];  // w, x, y, z
    float prob_zupt;
};

#pragma pack(pop)

} // namespace protocol
} // namespace trajecto
