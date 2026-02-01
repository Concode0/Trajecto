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
