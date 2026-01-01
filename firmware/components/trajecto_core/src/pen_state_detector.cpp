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
