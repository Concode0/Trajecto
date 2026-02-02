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

#include "swinging_door.hpp"
#include <cmath>
#include <algorithm>

namespace trajecto {

SwingingDoor::SwingingDoor(
    float position_tolerance,
    float velocity_tolerance,
    float rotation_tolerance,
    size_t max_buffer_size,
    uint32_t max_time_gap_us
) : pos_tol_(position_tolerance),
    vel_tol_(velocity_tolerance),
    rot_tol_(rotation_tolerance),
    max_buffer_(max_buffer_size),
    max_time_gap_(max_time_gap_us),
    has_anchor_(false),
    rx_count_(0),
    tx_count_(0)
{
    buffer_.reserve(max_buffer_size);
}

void SwingingDoor::reset() {
    has_anchor_ = false;
    buffer_.clear();
    rx_count_ = 0;
    tx_count_ = 0;
}

void SwingingDoor::process(const Point& point, const SendCallback& send_callback) {
    rx_count_++;

    // First point: always send and set as anchor
    if (!has_anchor_) {
        send_and_update(point, send_callback, SendReason::FIRST_POINT);
        return;
    }

    // Check forced send conditions
    SendReason reason = SendReason::DOOR_EXCEEDED;  // Default
    bool force_send = false;

    // Condition 1: Time gap too large
    uint32_t time_gap = point.timestamp_us - anchor_.timestamp_us;
    if (time_gap >= max_time_gap_) {
        force_send = true;
        reason = SendReason::TIME_GAP;
    }

    // Condition 2: Buffer full
    if (!force_send && buffer_.size() >= max_buffer_) {
        force_send = true;
        reason = SendReason::BUFFER_FULL;
    }

    // Add to buffer
    buffer_.push_back(point);

    // Condition 3: Point exceeds door tolerance
    if (!force_send && exceeds_door(point)) {
        force_send = true;
        // Check if it's due to pen state change
        if (point.pen_state != anchor_.pen_state) {
            reason = SendReason::PEN_STATE_CHANGE;
        } else {
            reason = SendReason::DOOR_EXCEEDED;
        }
    }

    if (force_send) {
        // When max_time_gap triggers, send current point as absolute reference
        // This prevents error accumulation after long gaps
        if (reason == SendReason::TIME_GAP) {
            send_and_update(point, send_callback, reason);
        }
        // For other conditions, send second-to-last point for smoother interpolation
        else if (buffer_.size() >= 2) {
            Point to_send = buffer_[buffer_.size() - 2];
            send_and_update(to_send, send_callback, reason);

            // Keep current point in buffer for next iteration
            buffer_.clear();
            buffer_.push_back(point);
        } else {
            send_and_update(point, send_callback, reason);
        }
    } else {
        // Update door to include this point
        update_door(anchor_, point);
    }
}

void SwingingDoor::flush(const SendCallback& send_callback) {
    // Send last buffered point if any
    if (!buffer_.empty()) {
        send_and_update(buffer_.back(), send_callback, SendReason::FLUSH);
    }
}

bool SwingingDoor::exceeds_door(const Point& p) const {
    if (!has_anchor_) return true;

    // KEYFRAME CHECK: Pen state change (pen up/down transition)
    // This is critical for stroke segmentation - always send immediately
    if (p.pen_state != anchor_.pen_state) {
        return true;  // Pen state changed - force send!
    }

    // Check position deviation (each axis independently)
    for (int i = 0; i < 3; i++) {
        if (p.position[i] < door_lower_.position[i] - pos_tol_ ||
            p.position[i] > door_upper_.position[i] + pos_tol_) {
            return true;
        }
    }

    // Check velocity deviation
    for (int i = 0; i < 3; i++) {
        if (p.velocity[i] < door_lower_.velocity[i] - vel_tol_ ||
            p.velocity[i] > door_upper_.velocity[i] + vel_tol_) {
            return true;
        }
    }

    // Check rotation deviation (angle between quaternions)
    // Using inner product: angle = 2 * acos(|q1 · q2|)
    float dot = std::abs(
        anchor_.quat.w() * p.quat.w() +
        anchor_.quat.x() * p.quat.x() +
        anchor_.quat.y() * p.quat.y() +
        anchor_.quat.z() * p.quat.z()
    );

    // Clamp to [0, 1] to avoid numerical errors in acos
    dot = std::min(1.0f, std::max(0.0f, dot));
    float angle = 2.0f * std::acos(dot);

    if (angle > rot_tol_) {
        return true;
    }

    return false;
}

void SwingingDoor::update_door(const Point& anchor, const Point& latest) {
    if (buffer_.size() <= 1) {
        // Initialize door bounds
        door_lower_ = anchor;
        door_upper_ = anchor;
        return;
    }

    // Calculate linear interpolation bounds
    // For each dimension, compute slope and tolerance band

    uint32_t dt = latest.timestamp_us - anchor.timestamp_us;
    if (dt == 0) return; // Avoid division by zero

    float t_ratio = static_cast<float>(latest.timestamp_us - anchor.timestamp_us) / dt;

    // Position door
    for (int i = 0; i < 3; i++) {
        float slope = (latest.position[i] - anchor.position[i]) / dt;
        float interp = anchor.position[i] + slope * (latest.timestamp_us - anchor.timestamp_us);

        door_lower_.position[i] = interp - pos_tol_;
        door_upper_.position[i] = interp + pos_tol_;
    }

    // Velocity door
    for (int i = 0; i < 3; i++) {
        float slope = (latest.velocity[i] - anchor.velocity[i]) / dt;
        float interp = anchor.velocity[i] + slope * (latest.timestamp_us - anchor.timestamp_us);

        door_lower_.velocity[i] = interp - vel_tol_;
        door_upper_.velocity[i] = interp + vel_tol_;
    }

    // Rotation door (simplified - use anchor bounds)
    door_lower_.quat = anchor.quat;
    door_upper_.quat = anchor.quat;
}

void SwingingDoor::send_and_update(const Point& p, const SendCallback& callback, SendReason reason) {
    callback(p, reason);

    anchor_ = p;
    has_anchor_ = true;
    buffer_.clear();

    tx_count_++;
}

SwingingDoor::Stats SwingingDoor::get_stats() const {
    Stats s;
    s.points_received = rx_count_;
    s.points_sent = tx_count_;
    s.compression_ratio = (rx_count_ > 0) ?
        static_cast<float>(rx_count_) / std::max(uint32_t{1}, tx_count_) : 1.0f;
    return s;
}

} // namespace trajecto
