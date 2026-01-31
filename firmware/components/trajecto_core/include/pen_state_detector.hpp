/*
 * Trajecto: Real-time 3D Trajectory Reconstruction System
 * Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * NOTICE: This software is protected under the following ROK Patent Applications:
 * 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
 * 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
 *
 * Commercial use or redistribution of the core logic requires a separate license.
 * For inquiries, contact: nemonanconcode@gmail.com
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
