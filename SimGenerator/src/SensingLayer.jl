# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [PATENT NOTICE]
# This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
# Commercial use without a separate license is strictly prohibited.
#
# Contact: nemonanconcode@gmail.com

"""
Sensing Layer: IMU and force sensor simulation with enhanced noise models.

Generates realistic accelerometer, gyroscope, and FSR data from trajectory and orientation.
Uses Allan variance noise parameters from real BMI270 sensor characterization.

Enhanced noise features:
- Physiological tremor (8-12 Hz band)
- 1/f (pink) noise
- FSR baseline drift with Ornstein-Uhlenbeck process
"""
module SensingLayer

using LinearAlgebra
using Random
using Statistics
using FFTW
using ..Config
using ..KinematicLayer: TrajectoryResult
using ..PostureLayer: OrientationResult, quaternion_to_rotation_matrix, quaternion_inverse

export SensorResult, generate_sensor_data, ADSRParams, DEFAULT_ADSR
export generate_tremor_noise, generate_pink_noise, generate_fsr_baseline

"""
    SensorResult

Result of sensor data generation.

# Fields
- `sensor_data::Matrix{Float64}`: Combined IMU data [T, 7] (accel, gyro, fsr)
- `gravity_b::Matrix{Float64}`: Gravity vector in body frame [T, 3]
"""
struct SensorResult
    sensor_data::Matrix{Float64}
    gravity_b::Matrix{Float64}
end

"""
    rotate_to_body(vec_w::Vector{Float64}, quat::Vector{Float64})

Rotate a world-frame vector to body frame using quaternion.
R_w2b = R_b2w^T = quaternion_to_rotation_matrix(q)^T
"""
function rotate_to_body(vec_w::Vector{Float64}, quat::Vector{Float64})
    R_b2w = quaternion_to_rotation_matrix(quat)
    R_w2b = R_b2w'  # Transpose for world-to-body
    return R_w2b * vec_w
end

# =============================================================================
# PHYSIOLOGICAL TREMOR (8-12 Hz)
# =============================================================================

"""
    generate_tremor_noise(n::Int, dt::Float64;
                          params::TremorParams=DEFAULT_TREMOR_PARAMS,
                          rng=Random.default_rng())

Generate physiological tremor noise for accelerometer and gyroscope.

Model:
- Band-limited sinusoid at 8-12 Hz (sample-specific frequency)
- Amplitude modulation for fatigue
- Cross-axis correlation via shared driver signal

Returns (accel_tremor, gyro_tremor) tuple of [n, 3] matrices.
"""
function generate_tremor_noise(n::Int, dt::Float64;
                                params::TremorParams=DEFAULT_TREMOR_PARAMS,
                                rng=Random.default_rng())
    # Sample-specific tremor frequency
    f_tremor = params.freq_mean + randn(rng) * params.freq_std
    f_tremor = clamp(f_tremor, 6.0, 14.0)  # Keep in reasonable range

    # Time vector
    t = collect(0:n-1) .* dt

    # Shared driver signal (correlated component)
    # Main sinusoid with slight frequency modulation for realism
    freq_mod = 1.0 .+ 0.05 .* sin.(2π * 0.3 * t)  # Slow frequency wobble
    driver = sin.(2π * f_tremor .* t .* freq_mod)

    # Amplitude modulation for fatigue (increases over time)
    amp_mod = 1.0 .+ params.fatigue_mod_amp .* sin.(2π * params.fatigue_mod_freq .* t)

    # Add bandwidth via slight noise on the driver
    bandwidth_noise = randn(rng, n) .* 0.1
    driver = driver .* amp_mod .+ bandwidth_noise .* 0.3

    # Pre-allocate output
    accel_tremor = zeros(n, 3)
    gyro_tremor = zeros(n, 3)

    corr = params.correlation
    corr_weight = sqrt(corr)
    indep_weight = sqrt(1 - corr)

    for axis in 1:3
        # Accelerometer tremor
        indep_accel = randn(rng, n) .* 0.3  # Independent component
        accel_tremor[:, axis] = params.amp_accel[axis] .* (
            corr_weight .* driver .+ indep_weight .* indep_accel
        )

        # Gyroscope tremor (slightly different phase per axis)
        phase_offset = (axis - 1) * π / 4
        driver_gyro = sin.(2π * f_tremor .* t .* freq_mod .+ phase_offset) .* amp_mod
        indep_gyro = randn(rng, n) .* 0.3
        gyro_tremor[:, axis] = params.amp_gyro[axis] .* (
            corr_weight .* driver_gyro .+ indep_weight .* indep_gyro
        )
    end

    return accel_tremor, gyro_tremor
end

# =============================================================================
# 1/f (PINK) NOISE
# =============================================================================

"""
    generate_pink_noise(n::Int, dt::Float64;
                        params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                        rng=Random.default_rng())

Generate 1/f (pink) noise using spectral filtering.

Model: PSD ∝ 1/f^α where α ≈ 1

Uses FFT-based filtering:
1. Generate white noise
2. Apply 1/f^(α/2) filter in frequency domain
3. IFFT to time domain

Returns pink noise vector of length n.
"""
function generate_pink_noise(n::Int, dt::Float64;
                              params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                              rng=Random.default_rng())
    # Generate white noise
    white = randn(rng, n)

    # FFT
    white_fft = fft(white)

    # Frequency vector
    fs = 1.0 / dt
    freqs = fftfreq(n, fs)

    # Build 1/f filter (avoid division by zero)
    H = zeros(ComplexF64, n)
    for i in 1:n
        f = abs(freqs[i])
        if f < params.low_freq_cutoff
            # Low frequency cutoff to avoid DC divergence
            H[i] = 1.0 / (params.low_freq_cutoff ^ (params.alpha / 2))
        else
            H[i] = 1.0 / (f ^ (params.alpha / 2))
        end
    end

    # Apply filter
    pink_fft = white_fft .* H

    # IFFT and return real part
    pink = real(ifft(pink_fft))

    # Normalize to unit variance and scale
    pink_std = std(pink)
    if pink_std > 0
        pink ./= pink_std
    end
    pink .*= params.amplitude_scale

    return pink
end

"""
    generate_pink_noise_3d(n::Int, dt::Float64;
                           params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                           rng=Random.default_rng())

Generate 3D pink noise (independent per axis).

Returns [n, 3] matrix.
"""
function generate_pink_noise_3d(n::Int, dt::Float64;
                                 params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                                 rng=Random.default_rng())
    pink_3d = zeros(n, 3)
    for axis in 1:3
        pink_3d[:, axis] = generate_pink_noise(n, dt; params=params, rng=rng)
    end
    return pink_3d
end

# =============================================================================
# FSR BASELINE DRIFT
# =============================================================================

"""
    generate_fsr_baseline(n::Int, dt::Float64;
                          params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS,
                          rng=Random.default_rng())

Generate FSR baseline drift using Ornstein-Uhlenbeck process.

Model:
- Mean-reverting random walk (OU process)
- Slow oscillation for grip pressure variation
- Initial value from configured range

The OU process is:
    dX = -θ(X - μ)dt + σdW

Returns baseline vector of length n.
"""
function generate_fsr_baseline(n::Int, dt::Float64;
                                params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS,
                                rng=Random.default_rng())
    baseline = zeros(n)

    # Random initial value
    baseline[1] = rand(rng) * (params.initial_range[2] - params.initial_range[1]) + params.initial_range[1]

    # OU process parameters
    theta = 1.0 / params.drift_tau  # Mean-reversion rate
    mu = params.mean_baseline
    sigma = params.drift_sigma

    # Random oscillation frequency for grip variation
    f_osc = rand(rng) * (params.oscillation_freq_range[2] - params.oscillation_freq_range[1]) +
            params.oscillation_freq_range[1]

    # Simulate OU process
    sqrt_dt = sqrt(dt)
    for i in 2:n
        # OU drift + diffusion
        drift = -theta * (baseline[i-1] - mu) * dt
        diffusion = sigma * sqrt_dt * randn(rng)
        baseline[i] = baseline[i-1] + drift + diffusion
    end

    # Add slow oscillation (grip pressure variation)
    t = collect(0:n-1) .* dt
    oscillation = params.oscillation_amp .* sin.(2π * f_osc .* t)
    baseline .+= oscillation

    # Clamp to valid range [0, 1]
    baseline .= clamp.(baseline, 0.0, 1.0)

    return baseline
end

# =============================================================================
# LEGACY IMU NOISE (white + bias instability)
# =============================================================================

"""
    add_imu_noise(accel_clean::Matrix{Float64}, gyro_clean::Matrix{Float64}, dt::Float64;
                  rng=Random.default_rng())

Add realistic IMU noise based on Allan variance parameters.

Noise model:
- White noise (VRW/ARW scaled to discrete time)
- Bias instability (random walk)
- Constant bias offset
"""
function add_imu_noise(accel_clean::Matrix{Float64}, gyro_clean::Matrix{Float64}, dt::Float64;
                       rng=Random.default_rng())
    n = size(accel_clean, 1)

    # White noise (VRW/ARW scaled to discrete time: sigma_discrete = sigma_continuous / sqrt(dt))
    accel_noise = randn(rng, n, 3) .* VRW' ./ sqrt(dt)
    gyro_noise = randn(rng, n, 3) .* ARW' ./ sqrt(dt)

    # Bias instability (random walk)
    bias_a = zeros(n, 3)
    bias_g = zeros(n, 3)
    for i in 2:n
        bias_a[i, :] = bias_a[i-1, :] + randn(rng, 3) .* ACCEL_BI * sqrt(dt)
        bias_g[i, :] = bias_g[i-1, :] + randn(rng, 3) .* GYRO_BI * sqrt(dt)
    end

    # Constant bias offset (per-run calibration error)
    const_bias_a = randn(rng, 3) .* 0.1
    const_bias_g = randn(rng, 3) .* 0.01

    # Apply noise
    accel_noisy = accel_clean .+ accel_noise .+ bias_a .+ const_bias_a'
    gyro_noisy = gyro_clean .+ gyro_noise .+ bias_g .+ const_bias_g'

    return accel_noisy, gyro_noisy
end

"""
    add_enhanced_imu_noise(accel_clean::Matrix{Float64}, gyro_clean::Matrix{Float64}, dt::Float64;
                           tremor_params::TremorParams=DEFAULT_TREMOR_PARAMS,
                           pink_params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                           rng=Random.default_rng())

Add enhanced IMU noise including:
1. White noise + bias instability (Allan variance)
2. Physiological tremor (8-12 Hz)
3. 1/f (pink) noise
"""
function add_enhanced_imu_noise(accel_clean::Matrix{Float64}, gyro_clean::Matrix{Float64}, dt::Float64;
                                 tremor_params::TremorParams=DEFAULT_TREMOR_PARAMS,
                                 pink_params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                                 rng=Random.default_rng())
    n = size(accel_clean, 1)

    # Start with legacy noise model
    accel_noisy, gyro_noisy = add_imu_noise(accel_clean, gyro_clean, dt; rng=rng)

    # Add physiological tremor
    accel_tremor, gyro_tremor = generate_tremor_noise(n, dt; params=tremor_params, rng=rng)
    accel_noisy .+= accel_tremor
    gyro_noisy .+= gyro_tremor

    # Add 1/f noise
    accel_pink = generate_pink_noise_3d(n, dt; params=pink_params, rng=rng)
    gyro_pink = generate_pink_noise_3d(n, dt; params=pink_params, rng=rng)

    # Scale pink noise by Allan variance parameters
    accel_noisy .+= accel_pink .* VRW' .* 0.5
    gyro_noisy .+= gyro_pink .* ARW' .* 0.5

    return accel_noisy, gyro_noisy
end

# =============================================================================
# FSR GENERATION
# =============================================================================

"""
    ADSRParams

ADSR envelope parameters for FSR modeling.

# Fields
- `attack_time::Float64`: Time to reach peak (seconds)
- `decay_time::Float64`: Time to decay to sustain level (seconds)
- `sustain_level::Float64`: Base sustain level [0, 1]
- `release_time::Float64`: Time to release to zero (seconds)
- `peak_level::Float64`: Peak level during attack [0, 1]
"""
struct ADSRParams
    attack_time::Float64
    decay_time::Float64
    sustain_level::Float64
    release_time::Float64
    peak_level::Float64
end

# Default ADSR parameters for pen pressure
const DEFAULT_ADSR = ADSRParams(0.05, 0.1, 0.5, 0.15, 0.9)

"""
    compute_curvature(vel_w::Matrix{Float64}, dt::Float64)

Compute trajectory curvature from velocity.
κ = |v × a| / |v|³  (2D approximation using XY plane)
"""
function compute_curvature(vel_w::Matrix{Float64}, dt::Float64)
    n = size(vel_w, 1)
    curvature = zeros(n)

    for i in 2:n-1
        # Velocity and acceleration
        v = vel_w[i, :]
        a = (vel_w[i+1, :] - vel_w[i-1, :]) / (2 * dt)

        v_mag = norm(v)
        if v_mag > 1e-6
            # Cross product magnitude (2D: v × a = vx*ay - vy*ax)
            cross_mag = abs(v[1] * a[2] - v[2] * a[1])
            curvature[i] = cross_mag / (v_mag^3)
        end
    end

    # Boundary handling
    curvature[1] = curvature[2]
    curvature[end] = curvature[end-1]

    return curvature
end

"""
    detect_stroke_events(vel_w::Matrix{Float64}, static_samples::Int; vel_threshold::Float64=0.005)

Detect stroke start/end events based on velocity threshold.
Returns vector of (start_idx, end_idx) tuples for each stroke.
"""
function detect_stroke_events(vel_w::Matrix{Float64}, static_samples::Int; vel_threshold::Float64=0.005)
    n = size(vel_w, 1)
    vel_mag = sqrt.(sum(vel_w.^2, dims=2))[:]

    strokes = Tuple{Int,Int}[]
    in_stroke = false
    stroke_start = 0

    for i in (static_samples+1):n
        if !in_stroke && vel_mag[i] > vel_threshold
            in_stroke = true
            stroke_start = i
        elseif in_stroke && vel_mag[i] <= vel_threshold
            in_stroke = false
            push!(strokes, (stroke_start, i))
        end
    end

    # Handle case where stroke extends to end
    if in_stroke
        push!(strokes, (stroke_start, n))
    end

    # If no strokes detected, treat entire writing portion as one stroke
    if isempty(strokes)
        push!(strokes, (static_samples + 1, n))
    end

    return strokes
end

"""
    generate_adsr_envelope(n::Int, start_idx::Int, end_idx::Int, dt::Float64;
                           params::ADSRParams=DEFAULT_ADSR)

Generate ADSR envelope for a single stroke.
"""
function generate_adsr_envelope(n::Int, start_idx::Int, end_idx::Int, dt::Float64;
                                 params::ADSRParams=DEFAULT_ADSR)
    envelope = zeros(n)

    attack_samples = round(Int, params.attack_time / dt)
    decay_samples = round(Int, params.decay_time / dt)
    release_samples = round(Int, params.release_time / dt)

    stroke_len = end_idx - start_idx + 1

    for i in start_idx:end_idx
        t_in_stroke = i - start_idx

        if t_in_stroke < attack_samples
            # Attack phase: linear rise to peak
            envelope[i] = params.peak_level * (t_in_stroke / attack_samples)

        elseif t_in_stroke < attack_samples + decay_samples
            # Decay phase: exponential decay to sustain
            decay_progress = (t_in_stroke - attack_samples) / decay_samples
            envelope[i] = params.sustain_level + (params.peak_level - params.sustain_level) * (1 - decay_progress)

        elseif t_in_stroke > stroke_len - release_samples
            # Release phase: linear decay to zero
            release_progress = (t_in_stroke - (stroke_len - release_samples)) / release_samples
            envelope[i] = params.sustain_level * (1 - release_progress)

        else
            # Sustain phase
            envelope[i] = params.sustain_level
        end
    end

    return envelope
end

"""
    generate_fsr(vel_w::Matrix{Float64}, pen_down::Vector{Bool}, static_samples::Int, dt::Float64;
                 rng=Random.default_rng(),
                 adsr_params::ADSRParams=DEFAULT_ADSR,
                 baseline_params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS,
                 vel_weight::Float64=0.2,
                 curvature_weight::Float64=0.3)

Generate force sensor (FSR) readings with ADSR envelope and baseline drift.

Model:
- ADSR envelope for each detected stroke (pen down/up events)
- Velocity modulation: higher velocity → slightly lighter touch
- Curvature modulation: sharper turns → more pressure for control
- Baseline drift: slow OU process + grip oscillation
- FSR = 0 during pen lift periods

FSR(t) = (pen_down ? ADSR(t) × modulation : 0) + baseline + noise
"""
function generate_fsr(vel_w::Matrix{Float64}, pen_down::Vector{Bool}, static_samples::Int, dt::Float64;
                      rng=Random.default_rng(),
                      adsr_params::ADSRParams=DEFAULT_ADSR,
                      baseline_params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS,
                      vel_weight::Float64=0.2,
                      curvature_weight::Float64=0.3)
    n = size(vel_w, 1)
    fsr = zeros(n)

    # Generate baseline drift
    baseline = generate_fsr_baseline(n, dt; params=baseline_params, rng=rng)

    # Static portion: baseline only with small noise
    fsr[1:static_samples] .= baseline[1:static_samples] .+ abs.(randn(rng, static_samples)) .* 0.02

    # Compute velocity magnitude and curvature
    vel_mag = sqrt.(sum(vel_w.^2, dims=2))[:]
    curvature = compute_curvature(vel_w, dt)

    # Normalize for modulation (avoid division by zero)
    max_vel = maximum(vel_mag)
    max_curv = maximum(curvature)
    vel_norm = max_vel > 1e-6 ? vel_mag ./ max_vel : zeros(n)
    curv_norm = max_curv > 1e-6 ? curvature ./ max_curv : zeros(n)

    # Detect stroke events
    strokes = detect_stroke_events(vel_w, static_samples)

    # Generate ADSR envelope for each stroke
    for (stroke_start, stroke_end) in strokes
        envelope = generate_adsr_envelope(n, stroke_start, stroke_end, dt; params=adsr_params)

        # Combine with existing (allows overlapping strokes)
        fsr .= max.(fsr, envelope)
    end

    # Apply pen_down mask (zero force during lift)
    for i in (static_samples+1):n
        if !pen_down[i]
            fsr[i] = 0.0
        end
    end

    # Apply velocity and curvature modulation to writing portion
    for i in (static_samples+1):n
        if fsr[i] > 0.01  # Only modulate when pen is "down"
            # Velocity: inverse correlation (faster → lighter touch)
            vel_mod = 1.0 - vel_weight * vel_norm[i]

            # Curvature: positive correlation (sharper → more pressure)
            curv_mod = 1.0 + curvature_weight * curv_norm[i]

            # Apply modulation
            fsr[i] *= vel_mod * curv_mod

            # Add noise
            fsr[i] += randn(rng) * 0.03

            # Add baseline
            fsr[i] += baseline[i]
        else
            # During lift, show only baseline
            fsr[i] = baseline[i] * 0.1  # Reduced baseline when lifted
        end
    end

    # Smooth the FSR signal (low-pass filter simulation)
    fsr_smooth = copy(fsr)
    alpha = 0.3  # Smoothing factor
    for i in 2:n
        fsr_smooth[i] = alpha * fsr[i] + (1 - alpha) * fsr_smooth[i-1]
    end

    # Clamp to valid range
    fsr_smooth = clamp.(fsr_smooth, 0.0, 1.5)

    return fsr_smooth
end

# =============================================================================
# MAIN SENSOR DATA GENERATION
# =============================================================================

"""
    compute_angular_acceleration(omega_b::Matrix{Float64}, dt::Float64)

Compute angular acceleration from angular velocity using central differences.
"""
function compute_angular_acceleration(omega_b::Matrix{Float64}, dt::Float64)
    n = size(omega_b, 1)
    alpha_b = zeros(n, 3)

    for i in 2:n-1
        alpha_b[i, :] = (omega_b[i+1, :] - omega_b[i-1, :]) / (2 * dt)
    end

    # Boundary handling
    alpha_b[1, :] = alpha_b[2, :]
    alpha_b[end, :] = alpha_b[end-1, :]

    return alpha_b
end

"""
    compute_lever_arm_acceleration(omega_b::Vector{Float64}, alpha_b::Vector{Float64},
                                   r::Vector{Float64})

Compute the additional acceleration at the IMU due to lever arm effect.

The IMU is offset from the pen tip by vector r (in body frame).
When the pen rotates, the IMU experiences:
- Centripetal acceleration: ω × (ω × r)
- Tangential acceleration: α × r

Returns the lever arm acceleration in body frame.
"""
function compute_lever_arm_acceleration(omega_b::Vector{Float64}, alpha_b::Vector{Float64},
                                        r::Vector{Float64})
    # Centripetal: ω × (ω × r)
    omega_cross_r = cross(omega_b, r)
    a_centripetal = cross(omega_b, omega_cross_r)

    # Tangential: α × r
    a_tangential = cross(alpha_b, r)

    return a_centripetal + a_tangential
end

"""
    cross(a::Vector{Float64}, b::Vector{Float64})

Compute cross product of two 3D vectors.
"""
function cross(a::Vector{Float64}, b::Vector{Float64})
    return [
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1]
    ]
end

"""
    generate_sensor_data(trajectory::TrajectoryResult, orientation::OrientationResult;
                         static_samples::Int, dt::Float64=DT, rng=Random.default_rng(),
                         use_enhanced_noise::Bool=true,
                         tremor_params::TremorParams=DEFAULT_TREMOR_PARAMS,
                         pink_params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                         baseline_params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS)

Generate complete sensor data from trajectory and orientation.

IMU physics with lever arm effect:
- The IMU is offset from pen tip by PEN_TIP_OFFSET (typically 6cm in -Y direction)
- When pen rotates, IMU experiences additional centripetal and tangential acceleration
- Accelerometer: a_b = R_w2b * (a_tip - g) + a_lever_arm + bias + noise
- Gyroscope: omega_b = omega_true + bias + noise

Lever arm acceleration:
- Centripetal: ω × (ω × r) - acceleration toward rotation axis
- Tangential: α × r - acceleration from angular acceleration

Enhanced noise features (when use_enhanced_noise=true):
- Physiological tremor (8-12 Hz)
- 1/f (pink) noise
- FSR baseline drift
"""

"""
    sample_pen_tip_offset(rng=Random.default_rng())

Sample a randomized pen tip offset to simulate grip position variance.
Returns the offset vector from IMU to pen tip in body frame.
"""
function sample_pen_tip_offset(rng=Random.default_rng())
    # Sample offset with Gaussian variation around mean
    offset = collect(PEN_TIP_OFFSET_MEAN) + randn(rng, 3) .* collect(PEN_TIP_OFFSET_STD)

    # Clamp to reasonable physical range (pen grip between 4-8cm from tip)
    offset[2] = clamp(offset[2], -0.08, -0.04)  # Y: along pen shaft
    offset[1] = clamp(offset[1], -0.01, 0.01)   # X: lateral
    offset[3] = clamp(offset[3], -0.005, 0.005) # Z: minimal

    return offset
end

export sample_pen_tip_offset

function generate_sensor_data(trajectory::TrajectoryResult, orientation::OrientationResult;
                              static_samples::Int,
                              dt::Float64=DT,
                              rng=Random.default_rng(),
                              use_enhanced_noise::Bool=true,
                              tremor_params::TremorParams=DEFAULT_TREMOR_PARAMS,
                              pink_params::PinkNoiseParams=DEFAULT_PINK_NOISE_PARAMS,
                              baseline_params::FSRBaselineParams=DEFAULT_FSR_BASELINE_PARAMS,
                              gravity_w::Union{Nothing, Vector{Float64}}=nothing)
    n = size(trajectory.pos_w, 1)

    # Pre-allocate
    accel_b_clean = zeros(n, 3)
    gyro_b_clean = zeros(n, 3)
    gravity_b = zeros(n, 3)

    # Sample randomized pen tip offset for this sample (grip position variance)
    pen_tip_offset = sample_pen_tip_offset(rng)

    # Lever arm: vector from pen tip to IMU in body frame
    # pen_tip_offset is IMU-to-tip, so r_lever = -pen_tip_offset (tip-to-IMU)
    r_lever = -pen_tip_offset

    # Compute angular acceleration for lever arm effect
    alpha_b = compute_angular_acceleration(orientation.omega_b, dt)

    # Gravity in world frame (can be rotated for world rotation augmentation)
    g_w_const = gravity_w !== nothing ? gravity_w : [0.0, 0.0, -GRAVITY_MAGNITUDE]

    for i in 1:n
        quat = orientation.quat[i, :]
        omega_b = orientation.omega_b[i, :]
        alpha_b_i = alpha_b[i, :]

        # Kinematic acceleration of PEN TIP in world frame
        a_tip_w = trajectory.accel_w[i, :]

        # Specific force in world frame: a_sf = a_kinematic - g
        # (accelerometer measures specific force, not coordinate acceleration)
        a_sf_w = a_tip_w - g_w_const

        # Rotate pen tip specific force to body frame
        a_tip_b = rotate_to_body(a_sf_w, quat)

        # Lever arm acceleration (IMU offset from pen tip)
        # This is the additional acceleration the IMU experiences due to rotation
        a_lever = compute_lever_arm_acceleration(omega_b, alpha_b_i, r_lever)

        # Total IMU acceleration in body frame
        accel_b_clean[i, :] = a_tip_b + a_lever

        # Angular velocity (already in body frame from PostureLayer)
        gyro_b_clean[i, :] = omega_b

        # Gravity in body frame
        gravity_b[i, :] = rotate_to_body(g_w_const, quat)
    end

    # Add noise (enhanced or legacy)
    if use_enhanced_noise
        accel_b_noisy, gyro_b_noisy = add_enhanced_imu_noise(
            accel_b_clean, gyro_b_clean, dt;
            tremor_params=tremor_params,
            pink_params=pink_params,
            rng=rng
        )
    else
        accel_b_noisy, gyro_b_noisy = add_imu_noise(accel_b_clean, gyro_b_clean, dt; rng=rng)
    end

    # Generate FSR with ADSR envelope, velocity/curvature modulation, and baseline drift
    fsr = generate_fsr(trajectory.vel_w, trajectory.pen_down, static_samples, dt;
                       rng=rng,
                       baseline_params=baseline_params)

    # Combine into sensor_data [T, 7]: accel(3), gyro(3), fsr(1)
    sensor_data = hcat(accel_b_noisy, gyro_b_noisy, fsr)

    return SensorResult(sensor_data, gravity_b)
end

end # module
