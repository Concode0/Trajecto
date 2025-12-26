module CRLB

using LinearAlgebra
using StaticArrays
using Statistics

export CRLBConfig, compute_crlb

# --- Configuration ---
Base.@kwdef struct CRLBConfig
    # Continuous Time Noise Spectral Densities
    # Units: unit / sqrt(Hz)
    acc_noise_density::Float64 = 0.993e-3  # m/s^2 / sqrt(Hz) (Avg from config.py)
    gyro_noise_density::Float64 = 4.82e-3 # rad/s / sqrt(Hz) (Avg from config.py)
    
    # Bias Instability (Random Walk)
    # Units: unit / sqrt(s)
    acc_bias_instability::Float64 = 3.91e-4 
    gyro_bias_instability::Float64 = 1.19e-3 
    
    # Initial State Uncertainties (Standard Deviation)
    # Based on P_init = 1e-4 * I => std = 0.01
    init_pos_std::Float64 = 0.01 # m
    init_vel_std::Float64 = 0.01 # m/s
    init_att_std::Float64 = 0.01 # rad
    init_acc_bias_std::Float64 = 0.01 # m/s^2
    init_gyro_bias_std::Float64 = 0.01 # rad/s
    
    # Measurement Noise (Standard Deviation)
    zupt_vel_std::Float64 = 0.01 # m/s
end

# --- Helpers ---
function skew(v::AbstractVector)
    SMatrix{3,3}(
        0, v[3], -v[2],
        -v[3], 0, v[1],
        v[2], -v[1], 0
    )
end

# --- Main Computation ---
"""
    compute_crlb(acc, gyro, rot, zupt, dt, config)

Computes the Cramer-Rao Lower Bound (Covariance propagation) for the trajectory.

# Arguments
- `acc`: (N, 3) Accelerometer readings (Body frame, m/s^2)
- `gyro`: (N, 3) Gyroscope readings (Body frame, rad/s)
- `rot`: (N, 3, 3) Rotation matrices (Body to World)
- `zupt`: (N,) Boolean mask or 0/1 integers where true indicates zero velocity.
- `dt`: Sampling time (s)
- `config`: CRLBConfig

# Returns
NamedTuple with standard deviations of the error states over time.
"""
function compute_crlb(acc::AbstractMatrix, gyro::AbstractMatrix, rot::AbstractArray, zupt::AbstractVector, dt::Float64, config::CRLBConfig)
    N = size(acc, 1)
    
    # State dimension: 15 (Pos(3), Vel(3), Ang(3), Ba(3), Bg(3))
    
    # Initial Covariance P
    P = zeros(15, 15)
    P[1:3, 1:3] .= Diagonal(fill(config.init_pos_std^2, 3))
    P[4:6, 4:6] .= Diagonal(fill(config.init_vel_std^2, 3))
    P[7:9, 7:9] .= Diagonal(fill(config.init_att_std^2, 3))
    P[10:12, 10:12] .= Diagonal(fill(config.init_acc_bias_std^2, 3))
    P[13:15, 13:15] .= Diagonal(fill(config.init_gyro_bias_std^2, 3))
    
    # Process Noise Covariance scaling
    # Q_discrete approx G * (sigma^2) * G' * dt
    # where sigma is the spectral density (unit/sqrt(Hz)) -> sigma^2 is power density (unit^2/Hz)
    
    q_acc = config.acc_noise_density^2 * dt
    q_gyro = config.gyro_noise_density^2 * dt
    q_ba = config.acc_bias_instability^2 * dt
    q_bg = config.gyro_bias_instability^2 * dt
    
    # Measurement Noise Covariance
    R_zupt = Diagonal(fill(config.zupt_vel_std^2, 3))
    
    # Storage
    std_pos = zeros(N, 3)
    std_vel = zeros(N, 3)
    std_att = zeros(N, 3)
    
    # Store initial
    std_pos[1, :] = sqrt.(diag(P[1:3, 1:3]))
    std_vel[1, :] = sqrt.(diag(P[4:6, 4:6]))
    std_att[1, :] = sqrt.(diag(P[7:9, 7:9]))
    
    # Pre-allocate matrices to reduce GC (optional optimization, keeping it simple for now)
    
    for k in 1:N-1
        # 1. Linearization Point
        a_b = acc[k, :]
        # w_b = gyro[k, :] # Not used in simplified Global-Error Jacobian
        R_wb = rot[k, :, :] # Body to World
        
        # 2. State Transition Matrix F
        # x_dot = F x + G n
        # Discrete: F_k = I + F * dt
        
        F = Matrix{Float64}(I, 15, 15)
        
        # Pos -> Vel
        F[1:3, 4:6] .= I(3) * dt
        
        # Vel -> Att: -R_wb * skew(a_b)
        R_mat = SMatrix{3,3}(R_wb)
        S_a = skew(a_b)
        F[4:6, 7:9] = -R_mat * S_a * dt
        
        # Vel -> AccBias: -R_wb
        F[4:6, 10:12] = -R_mat * dt
        
        # Att -> GyroBias: -R_wb (assuming global error def)
        F[7:9, 13:15] = -R_mat * dt
        
        # 3. Process Noise Q
        Q = zeros(15, 15)
        Q[4:6, 4:6] = Diagonal(fill(q_acc, 3)) # Vel noise from Acc
        Q[7:9, 7:9] = Diagonal(fill(q_gyro, 3)) # Att noise from Gyro
        Q[10:12, 10:12] = Diagonal(fill(q_ba, 3))
        Q[13:15, 13:15] = Diagonal(fill(q_bg, 3))
        
        # 4. Predict
        P = F * P * F' + Q
        
        # 5. Update (ZUPT)
        if zupt[k+1] > 0 # Check next step or current? Usually applied at step k+1
            H = zeros(3, 15)
            H[1:3, 4:6] = I(3) # Observe Velocity
            
            # K = P H' (H P H' + R)^-1
            S_mat = H * P * H' + R_zupt
            K = P * H' / S_mat
            
            # Update P = (I - KH) P (I - KH)' + KRK'
            ImKH = I(15) - K * H
            P = ImKH * P * ImKH' + K * R_zupt * K'
        end
        
        # Ensure Symmetry
        P = (P + P') / 2
        
        # Store
        std_pos[k+1, :] = sqrt.(max.(0.0, diag(P[1:3, 1:3])))
        std_vel[k+1, :] = sqrt.(max.(0.0, diag(P[4:6, 4:6])))
        std_att[k+1, :] = sqrt.(max.(0.0, diag(P[7:9, 7:9])))
    end
    
    return (
        std_pos = std_pos,
        std_vel = std_vel,
        std_att = std_att
    )
end

end
