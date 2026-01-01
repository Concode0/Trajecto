#include "pen_state_detector.hpp"

namespace trajecto {

PenStateDetector::PenStateDetector(int16_t touch_threshold, int16_t hysteresis)
    : threshold_(touch_threshold),
      hysteresis_(hysteresis),
      is_touching_(false)
{
}

bool PenStateDetector::detect(int16_t force_value) {
    // Hysteresis to prevent jitter:
    // - When pen is UP: need force > (threshold + hyst/2) to go DOWN
    // - When pen is DOWN: need force < (threshold - hyst/2) to go UP

    int16_t upper_threshold = threshold_ + (hysteresis_ / 2);
    int16_t lower_threshold = threshold_ - (hysteresis_ / 2);

    if (is_touching_) {
        // Currently touching - check for lift-off
        if (force_value < lower_threshold) {
            is_touching_ = false;
        }
    } else {
        // Currently not touching - check for touch-down
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
