/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System
 * Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * [PATENT NOTICE]
 * This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
 * Commercial use without a separate license is strictly prohibited.
 *
 * Contact: nemonanconcode@gmail.com
 */

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
    uint8_t mode;           // 0: Raw, 1: Inference
    uint8_t odr_hz;         // e.g., 50
    uint8_t enable_sda;     // 0: Disabled, 1: Enabled (Swinging Door compression)
    uint8_t reserved[1];
};

// -- Data: Raw IMU --
struct RawImuPacket {
    uint32_t timestamp_us;
    float accel[3]; // x, y, z (m/s^2)
    float gyro[3];  // x, y, z (rad/s)
    int16_t force;  // FSR
    float temperature; // °C
};

// -- Trajectory Packet Flags --
enum TrajectoryFlags : uint8_t {
    TRAJ_FLAG_NONE = 0x00,
    TRAJ_FLAG_ABSOLUTE_REF = 0x01,  // Sent due to max_time_gap (absolute reference)
    TRAJ_FLAG_PEN_DOWN = 0x02,      // Pen is in contact (writing)
    TRAJ_FLAG_KEYFRAME = 0x04       // Pen state changed (stroke boundary)
};

// -- Data: Trajectory --
struct TrajectoryPacket {
    uint32_t timestamp_us;
    float pos[3];       // x, y, z (m)
    float vel[3];       // x, y, z (m/s)
    float quat[4];      // w, x, y, z (orientation)
    float zupt_prob;    // Probability of Zero-Velocity
    uint8_t flags;      // Bitfield of TrajectoryFlags
    uint8_t reserved[3]; // Padding for 4-byte alignment
};

#pragma pack(pop)

} // namespace protocol
} // namespace trajecto
