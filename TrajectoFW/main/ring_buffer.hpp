#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP

#include <vector>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include "sensor_data.hpp"

/**
 * @brief Thread-safe ring buffer for sensor_sample_t.
 */
class RingBuffer {
public:
    RingBuffer(size_t capacity);
    ~RingBuffer();

    bool write(const sensor_sample_t& sample);
    bool read(sensor_sample_t& sample);
    size_t available_items();
    size_t free_space();
    size_t get_capacity();

private:
    std::vector<sensor_sample_t> buffer;
    size_t head;
    size_t tail;
    size_t capacity;
    SemaphoreHandle_t mutex;       ///< Protects head, tail, and buffer access
    SemaphoreHandle_t full_sem;    ///< Counts available items (for consumers)
    SemaphoreHandle_t empty_sem;   ///< Counts free slots (for producers)
};

#endif // RING_BUFFER_HPP
