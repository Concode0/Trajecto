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

#include <cstdint>

namespace trajecto {

class PenStateDetector {
public:
    explicit PenStateDetector(
        int16_t touch_threshold = 100,
        int16_t hysteresis = 20
    );

    bool detect(int16_t force_value);
    bool get_state() const { return is_touching_; }
    void set_threshold(int16_t threshold, int16_t hysteresis = 20);
    void reset();

private:
    int16_t threshold_;
    int16_t hysteresis_;
    bool is_touching_;
};

} // namespace trajecto
