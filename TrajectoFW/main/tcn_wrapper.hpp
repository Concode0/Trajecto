#pragma once

#include <vector>
#include <vector> // Redundant but harmless
#include <Eigen/Dense>
#include "eskf.hpp"
#include "model_params.hpp"

// Forward declarations for TFLite
namespace tflite {
    class MicroInterpreter;
    class Model;
}

namespace trajecto {

struct TCNOutput {
    Eigen::Vector3f vel_corr;
    Eigen::Matrix<float, 6, 1> R_params;
    float zupt_prob;
    bool valid; 
};

class TCNWrapper {
public:
    TCNWrapper();
    ~TCNWrapper();

    bool setup();

    /**
     * @brief Process a new data point (Stateful Step).
     */
    TCNOutput process_step(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        float force_raw,
        const ESKF& eskf,
        const Eigen::Matrix<float, 6, 1>& last_innovation,
        bool is_zupt
    );

private:
    void extract_features(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        float force_raw,
        const ESKF& eskf,
        const Eigen::Matrix<float, 6, 1>& last_innovation,
        bool is_zupt,
        std::vector<float>& out_features
    );

    // TFLite Micro components
    const tflite::Model* model_ = nullptr;
    tflite::MicroInterpreter* interpreter_ = nullptr;
    uint8_t* tensor_arena_ = nullptr;
    
    // Persistent State Buffers
    // Each element corresponds to a layer's state buffer.
    // We store them as flat vectors.
    std::vector<std::vector<float>> state_buffers_;
    
    static constexpr int kTensorArenaSize = 60 * 1024; 
    static constexpr int kInputSize = TCN_INPUT_SIZE;
};

} // namespace trajecto