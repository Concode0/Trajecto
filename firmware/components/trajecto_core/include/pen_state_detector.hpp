#pragma once

#include <cstdint>

namespace trajecto {

/**
 * @brief Pen contact state detector
 *
 * Determines pen up/down state from force sensor reading.
 * Current implementation uses simple threshold, but designed to be
 * easily replaceable with ML-based detection in the future.
 *
 * Usage:
 *   PenStateDetector detector(threshold=100, hysteresis=20);
 *   bool is_touching = detector.detect(force_value);
 */
class PenStateDetector {
public:
    /**
     * @brief Construct pen state detector with threshold
     *
     * @param touch_threshold Force value above which pen is considered touching
     * @param hysteresis Hysteresis to prevent jitter (default: 20)
     *
     * Example:
     *   - FSR range: 0-4095 (12-bit ADC)
     *   - No contact: ~0-50
     *   - Light touch: ~100-500
     *   - Normal writing: ~500-2000
     *   - Heavy pressure: >2000
     *
     * Recommended defaults:
     *   - touch_threshold = 100 (detects light touch)
     *   - hysteresis = 20 (prevents flutter)
     */
    explicit PenStateDetector(
        int16_t touch_threshold = 100,
        int16_t hysteresis = 20
    );

    /**
     * @brief Detect pen state from force sensor reading
     *
     * Uses hysteresis to prevent rapid switching:
     * - Pen goes DOWN when force > (threshold + hysteresis/2)
     * - Pen goes UP when force < (threshold - hysteresis/2)
     *
     * @param force_value Raw force sensor reading (ADC units)
     * @return true if pen is touching (down), false if lifted (up)
     */
    bool detect(int16_t force_value);

    /**
     * @brief Get current pen state without updating
     */
    bool get_state() const { return is_touching_; }

    /**
     * @brief Update detection threshold (can be changed at runtime)
     */
    void set_threshold(int16_t threshold, int16_t hysteresis = 20);

    /**
     * @brief Reset state (call on stream start)
     */
    void reset();

private:
    int16_t threshold_;
    int16_t hysteresis_;
    bool is_touching_;
};

} // namespace trajecto
