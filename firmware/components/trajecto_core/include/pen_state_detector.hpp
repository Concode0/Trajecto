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
