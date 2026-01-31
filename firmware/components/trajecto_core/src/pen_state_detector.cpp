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

#include "pen_state_detector.hpp"

namespace trajecto {

PenStateDetector::PenStateDetector(int16_t touch_threshold, int16_t hysteresis)
    : threshold_(touch_threshold),
      hysteresis_(hysteresis),
      is_touching_(false)
{
}

bool PenStateDetector::detect(int16_t force_value) {
    int16_t upper_threshold = threshold_ + (hysteresis_ / 2);
    int16_t lower_threshold = threshold_ - (hysteresis_ / 2);

    if (is_touching_) {
        if (force_value < lower_threshold) {
            is_touching_ = false;
        }
    } else {
        if (force_value > upper_threshold) {
            is_touching_ = true;
        }
    }

    return is_touching_;
}

void PenStateDetector::set_threshold(int16_t threshold, int16_t hysteresis) {
    threshold_ = threshold;
    hysteresis_ = hysteresis;
}

void PenStateDetector::reset() {
    is_touching_ = false;
}

} // namespace trajecto
