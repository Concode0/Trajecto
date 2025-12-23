#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include "model_params.hpp"

namespace trajecto {

// Error State Layout
// 0-2: Position Error
// 3-5: Velocity Error
// 6-8: Orientation Error (Axis-Angle)
// 9-11: Gyro Bias Error
// 12-14: Accel Bias Error
constexpr int STATE_DIM = 15;

struct NominalState {
    Eigen::Vector3f pos;       // Position in World Frame
    Eigen::Vector3f vel;       // Velocity in World Frame
    Eigen::Quaternionf quat;   // Orientation (Body to World)
    Eigen::Vector3f gyro_bias; // Gyro Bias in Body Frame
    Eigen::Vector3f accel_bias;// Accel Bias in Body Frame

    NominalState() {
        pos.setZero();
        vel.setZero();
        quat.setIdentity();
        gyro_bias.setZero();
        accel_bias.setZero();
    }
};

class ESKF {
public:
    ESKF(float dt);

    /**
     * @brief Initialize the filter state.
     * 
     * @param accel_init Initial accelerometer reading (for gravity alignment).
     */
    void initialize(const Eigen::Vector3f& accel_init);

    /**
     * @brief Prediction Step (Propagate Nominal State + Predict Error Covariance).
     * 
     * @param gyro_raw Raw gyroscope measurement (rad/s).
     * @param accel_raw Raw accelerometer measurement (m/s^2).
     */
    void predict(const Eigen::Vector3f& gyro_raw, const Eigen::Vector3f& accel_raw);

    /**
     * @brief Update step using Zero Velocity Update (ZUPT).
     */
    void update_zupt();

    /**
     * @brief Update step using TCN predicted velocity correction.
     * 
     * @param vel_corr_body Velocity correction in Body Frame (m/s).
     * @param R_params Measurement noise parameters (not fully used yet, placeholder).
     */
    void update_tcn_vel(const Eigen::Vector3f& vel_corr_body, const Eigen::Matrix<float, 6, 1>& R_params);

    /**
     * @brief Standard Measurement Update using IMU data (to estimate biases/errors).
     * 
     * @param accel_raw Raw Accel
     * @param gyro_raw Raw Gyro
     * @param R_diag Measurement noise variance (6D)
     * @return Eigen::Matrix<float, 6, 1> Innovation vector (Accel 3 + Gyro 3)
     */
    Eigen::Matrix<float, 6, 1> update_imu(
        const Eigen::Vector3f& accel_raw,
        const Eigen::Vector3f& gyro_raw,
        const Eigen::Matrix<float, 6, 1>& R_diag
    );

    /**
     * @brief Check if ZUPT condition is met (simple thresholding).
     * 
     * @param accel_raw Raw accelerometer reading.
     * @return true if stationary.
     */
    bool check_zupt(const Eigen::Vector3f& accel_raw);

    // Getters
    const NominalState& get_state() const { return state_; }
    const Eigen::Matrix<float, STATE_DIM, STATE_DIM>& get_covariance() const { return P_; }

private:
    void inject_error(const Eigen::Matrix<float, STATE_DIM, 1>& delta_x);
    void enforce_symmetry();

    float dt_;
    NominalState state_;
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> P_; // Error Covariance

    // Process Noise Covariance (Diagonal)
    Eigen::Matrix<float, STATE_DIM, 1> Q_diag_;
    
    // Measurement Noise
    float zupt_noise_std_;
    float tcn_vel_noise_std_;

    // Gravity Vector (World Frame)
    Eigen::Vector3f gravity_w_;
};

} // namespace trajecto
