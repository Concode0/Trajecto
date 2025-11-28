#include "ring_buffer.hpp"
#include "esp_log.h"

static const char *TAG = "RING_BUFFER";

RingBuffer::RingBuffer(size_t capacity) :
    head(0),
    tail(0),
    capacity(capacity)
{
    buffer.resize(capacity);
    mutex = xSemaphoreCreateMutex();
    full_sem = xSemaphoreCreateCounting(capacity, 0); // Initially no items
    empty_sem = xSemaphoreCreateCounting(capacity, capacity); // Initially all slots empty

    if (mutex == NULL || full_sem == NULL || empty_sem == NULL) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS semaphores!");
        // Handle error, maybe abort
    }
    ESP_LOGI(TAG, "RingBuffer initialized with capacity: %d", capacity);
}

RingBuffer::~RingBuffer() {
    if (mutex != NULL) vSemaphoreDelete(mutex);
    if (full_sem != NULL) vSemaphoreDelete(full_sem);
    if (empty_sem != NULL) vSemaphoreDelete(empty_sem);
}

bool RingBuffer::write(const sensor_sample_t& sample) {
    if (xSemaphoreTake(empty_sem, portMAX_DELAY) == pdTRUE) { // Wait for an empty slot
        if (xSemaphoreTake(mutex, portMAX_DELAY) == pdTRUE) { // Protect critical section
            buffer[head] = sample;
            head = (head + 1) % capacity;
            xSemaphoreGive(mutex);
            xSemaphoreGive(full_sem); // Signal that an item is available
            return true;
        }
        // If mutex acquisition failed after empty_sem, return empty_sem
        xSemaphoreGive(empty_sem);
    }
    return false;
}

bool RingBuffer::read(sensor_sample_t& sample) {
    if (xSemaphoreTake(full_sem, portMAX_DELAY) == pdTRUE) { // Wait for an item to be available
        if (xSemaphoreTake(mutex, portMAX_DELAY) == pdTRUE) { // Protect critical section
            sample = buffer[tail];
            tail = (tail + 1) % capacity;
            xSemaphoreGive(mutex);
            xSemaphoreGive(empty_sem); // Signal that a slot is empty
            return true;
        }
        // If mutex acquisition failed after full_sem, return full_sem
        xSemaphoreGive(full_sem);
    }
    return false;
}

size_t RingBuffer::available_items() {
    size_t count;
    if (xSemaphoreTake(mutex, portMAX_DELAY) == pdTRUE) {
        if (head >= tail) {
            count = head - tail;
        } else {
            count = capacity - (tail - head);
        }
        xSemaphoreGive(mutex);
        return count;
    }
    return 0; // Error or mutex not available
}

size_t RingBuffer::free_space() {
    return capacity - available_items();
}

size_t RingBuffer::get_capacity() {
    return capacity;
}
