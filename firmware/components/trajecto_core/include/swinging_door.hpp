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

#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace trajecto {

/**
 * @brief Swinging Door Algorithm (SDA) for trajectory compression
 *
 * SDA reduces data transmission by only sending points when the trajectory
 * deviates beyond a threshold from linear interpolation. This provides:
 * - 5-10x bandwidth reduction for typical trajectories
 * - Lossless reconstruction within error tolerance
 * - Adaptive compression based on motion complexity
 *
 * Algorithm:
 * 1. Maintain a "door" (tolerance band) from last sent point
 * 2. Buffer incoming points while they stay within door
 * 3. When a point exceeds door, send previous buffered point
 * 4. Update door from new anchor point
 *
 * Reference: "Swinging Door Trending: Adaptive Trend Recording"
 *           Bristol Babcock Inc., 1990
 */
class SwingingDoor {
public:
    struct Point {
        uint32_t timestamp_us;
        Eigen::Vector3f position;    // meters
        Eigen::Vector3f velocity;    // m/s
        Eigen::Quaternionf quat;     // orientation
        bool pen_state;              // true = pen down, false = pen up

        Point() : timestamp_us(0), position(0,0,0), velocity(0,0,0),
                  quat(1,0,0,0), pen_state(false) {}
    };

    enum class SendReason : uint8_t {
        FIRST_POINT,        // First point in stream
        TIME_GAP,          // max_time_gap_us exceeded (absolute reference)
        DOOR_EXCEEDED,     // Trajectory deviated beyond tolerance
        BUFFER_FULL,       // Buffer capacity reached
        PEN_STATE_CHANGE,  // Pen up/down transition (keyframe)
        FLUSH              // Stream ended, flushing buffer
    };

    using SendCallback = std::function<void(const Point&, SendReason)>;

    /**
     * @brief Construct Swinging Door compressor
     *
     * @param position_tolerance Maximum position deviation in meters (e.g., 0.001 = 1mm)
     * @param velocity_tolerance Maximum velocity deviation in m/s (e.g., 0.01)
     * @param rotation_tolerance Maximum rotation deviation in radians (e.g., 0.01)
     * @param max_buffer_size Maximum points to buffer before forced send
     * @param max_time_gap_us Maximum time between sends in microseconds (forced send)
     */
    SwingingDoor(
        float position_tolerance = 0.001f,     // 1mm position error
        float velocity_tolerance = 0.01f,      // 1cm/s velocity error
        float rotation_tolerance = 0.01f,      // ~0.57 degrees rotation error
        size_t max_buffer_size = 50,           // Buffer up to 50 points (1 second @ 50Hz)
        uint32_t max_time_gap_us = 500000      // Force send every 500ms
    );

    /**
     * @brief Process incoming trajectory point
     *
     * @param point New trajectory point from ESKF
     * @param send_callback Callback to send compressed point via BLE (receives Point and SendReason)
     *
     * The callback is invoked only when compression determines a point should be sent.
     * Typical compression ratio: 5-10x (send every 5-10 points instead of all)
     *
     * Send behavior:
     * - When max_time_gap_us exceeded: Sends current point as absolute reference
     * - When door tolerance exceeded or buffer full: Sends second-to-last buffered point
     *
     * This ensures fresh absolute data after long gaps to prevent error accumulation.
     *
     * Example usage in main.cpp:
     * @code
     * SwingingDoor compressor;
     * auto callback = [](const SwingingDoor::Point& pt, SwingingDoor::SendReason reason) {
     *     TrajectoryPacket pkt;
     *     pkt.timestamp_us = pt.timestamp_us;
     *     // ... copy pos, vel, quat ...
     *     pkt.flags = (reason == SendReason::TIME_GAP) ? TRAJ_FLAG_ABSOLUTE_REF : 0;
     *     if (pt.pen_state) pkt.flags |= TRAJ_FLAG_PEN_DOWN;
     *     if (reason == SendReason::PEN_STATE_CHANGE) pkt.flags |= TRAJ_FLAG_KEYFRAME;
     *     send_ble_packet(&pkt);
     * };
     * compressor.process(point, callback);
     * @endcode
     */
    void process(const Point& point, const SendCallback& send_callback);

    /**
     * @brief Force send all buffered points (call on stream stop)
     */
    void flush(const SendCallback& send_callback);

    /**
     * @brief Reset compression state (call on stream start)
     */
    void reset();

    /**
     * @brief Get compression statistics
     */
    struct Stats {
        uint32_t points_received;
        uint32_t points_sent;
        float compression_ratio;
    };
    Stats get_stats() const;

private:
    // Check if point exceeds door tolerance
    bool exceeds_door(const Point& p) const;

    // Update door bounds from new anchor point
    void update_door(const Point& anchor, const Point& latest);

    // Send point and update state
    void send_and_update(const Point& p, const SendCallback& callback, SendReason reason);

    // Configuration
    float pos_tol_;
    float vel_tol_;
    float rot_tol_;
    size_t max_buffer_;
    uint32_t max_time_gap_;

    // State
    bool has_anchor_;
    Point anchor_;           // Last sent point
    Point door_upper_;       // Upper bound of tolerance door
    Point door_lower_;       // Lower bound of tolerance door
    std::vector<Point> buffer_;

    // Statistics
    uint32_t rx_count_;
    uint32_t tx_count_;
};

} // namespace trajecto
