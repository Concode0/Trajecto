# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
Kinematic Layer: Sigma-Lognormal trajectory generation with superposition.

Implements the Sigma-Lognormal model for smooth, realistic handwriting trajectories.
Strokes are superimposed (velocities summed) for natural transitions between strokes.

References:
- Plamondon (1995): A kinematic theory of rapid human movements
- Flash & Hogan (1985): Minimum-jerk trajectories for pen lift modeling
"""
module KinematicLayer

using LinearAlgebra
using Random
using ..Config

export SigmaLogNormalStroke, generate_random_strokes, generate_trajectory, TrajectoryResult
export PenLiftEvent, detect_stroke_boundaries, generate_pen_lift_events
export StrokePattern, CURSIVE_WORD, PRINT_LETTERS, SIGNATURE, SCRIBBLE, LINE_BY_LINE
export PoseConstraints, DEFAULT_POSE_CONSTRAINTS, apply_pose_constraints!

"""
    StrokePattern

Enumeration of handwriting stroke patterns for geometric connectivity.
"""
@enum StrokePattern begin
    CURSIVE_WORD = 1      # Connected letters flowing left-to-right
    PRINT_LETTERS = 2     # Discrete letters with small gaps
    SIGNATURE = 3         # Flowing signature-like motion
    SCRIBBLE = 4          # Random scribbling pattern
    LINE_BY_LINE = 5      # Multiple lines of text
end

"""
    SigmaLogNormalStroke

A single stroke in the Sigma-LogNormal model.

The Sigma-LogNormal model describes rapid human movements where velocity
follows a lognormal distribution over time.

# Fields
- `D::Float64`: Amplitude (stroke length in meters)
- `t0::Float64`: Onset time (seconds)
- `mu::Float64`: Log-mean of velocity profile
- `sigma::Float64`: Log-std of velocity profile
- `theta::Float64`: Direction angle (radians)
"""
struct SigmaLogNormalStroke
    D::Float64      # Amplitude (m)
    t0::Float64     # Onset time (s)
    mu::Float64     # Log-mean
    sigma::Float64  # Log-std
    theta::Float64  # Direction (rad)
end

"""
    PenLiftEvent

A pen lift event between strokes.

# Fields
- `t_start::Float64`: Time pen starts lifting (end of previous stroke)
- `t_lift_end::Float64`: Time pen reaches maximum height
- `t_land_start::Float64`: Time pen starts descending
- `t_end::Float64`: Time pen touches surface (start of next stroke)
- `z_height::Float64`: Maximum lift height (m)
"""
struct PenLiftEvent
    t_start::Float64
    t_lift_end::Float64
    t_land_start::Float64
    t_end::Float64
    z_height::Float64
end

"""
    TrajectoryResult

Result of trajectory generation.

# Fields
- `pos_w::Matrix{Float64}`: Position in world frame [T, 3]
- `vel_w::Matrix{Float64}`: Velocity in world frame [T, 3]
- `accel_w::Matrix{Float64}`: Acceleration in world frame [T, 3]
- `time::Vector{Float64}`: Time vector
- `pen_down::Vector{Bool}`: Pen contact state (true = touching surface)
- `lift_events::Vector{PenLiftEvent}`: Pen lift events
"""
struct TrajectoryResult
    pos_w::Matrix{Float64}
    vel_w::Matrix{Float64}
    accel_w::Matrix{Float64}
    time::Vector{Float64}
    pen_down::Vector{Bool}
    lift_events::Vector{PenLiftEvent}
end

# =============================================================================
# SIGMA-LOGNORMAL MODEL WITH SUPERPOSITION
# =============================================================================

"""
    velocity_profile(stroke::SigmaLogNormalStroke, t::AbstractVector)

Compute the velocity magnitude profile for a single stroke.

The Sigma-LogNormal formula:
    v(t) = D / (sigma * sqrt(2*pi) * (t - t0)) * exp(-(log(t-t0) - mu)^2 / (2*sigma^2))

This produces a bell-shaped velocity profile when plotted against log-time.
"""
function velocity_profile(stroke::SigmaLogNormalStroke, t::AbstractVector{Float64})
    n = length(t)
    v_mag = zeros(n)

    for i in 1:n
        tau = t[i] - stroke.t0
        if tau > 1e-9
            coeff = stroke.D / (stroke.sigma * sqrt(2 * pi))
            exponent = -(log(tau) - stroke.mu)^2 / (2 * stroke.sigma^2)
            v_mag[i] = coeff / tau * exp(exponent)
        end
    end

    return v_mag
end

"""
    velocity_derivative(stroke::SigmaLogNormalStroke, t::AbstractVector)

Compute the acceleration (time derivative of velocity) for a single stroke.

Analytical derivative of the Sigma-LogNormal velocity profile.
"""
function velocity_derivative(stroke::SigmaLogNormalStroke, t::AbstractVector{Float64})
    n = length(t)
    a_mag = zeros(n)

    D, t0, mu, sigma = stroke.D, stroke.t0, stroke.mu, stroke.sigma
    sigma2 = sigma^2

    for i in 1:n
        tau = t[i] - t0
        if tau > 1e-9
            log_tau = log(tau)
            coeff = D / (sigma * sqrt(2 * pi))
            exponent = -(log_tau - mu)^2 / (2 * sigma2)

            # Derivative: d/dt[v(t)] = v(t) * (-1/tau - (log(tau) - mu) / (sigma^2 * tau))
            v = coeff / tau * exp(exponent)
            a_mag[i] = v * (-1/tau - (log_tau - mu) / (sigma2 * tau))
        end
    end

    return a_mag
end

"""
    generate_superposition_strokes(num_strokes::Int, duration::Float64;
                                   pattern::StrokePattern=CURSIVE_WORD,
                                   overlap_factor::Float64=0.5,
                                   rng=Random.default_rng())

Generate Sigma-LogNormal strokes with smooth superposition for natural curved trajectories.

Enhanced human-like dynamics:
- Variable rhythm/tempo (some strokes fast, some slow)
- Micro-hesitations at stroke beginnings
- Natural timing irregularity (jitter)
- Variable stroke sharpness (sigma variation)
- Amplitude variation mimicking natural letter size changes
- "Rush" vs "careful" stroke modes

# Arguments
- `num_strokes`: Number of logical strokes to generate
- `duration`: Total writing duration
- `pattern`: Stroke pattern for geometric connectivity
- `overlap_factor`: Fraction of stroke duration that overlaps (0.3-0.6 recommended)
- `rng`: Random number generator

# Returns
Vector of SigmaLogNormalStroke with overlapping timing for smooth curved transitions.
"""
function generate_superposition_strokes(num_strokes::Int, duration::Float64;
                                        pattern::StrokePattern=CURSIVE_WORD,
                                        overlap_factor::Float64=0.5,
                                        rng=Random.default_rng())
    strokes = SigmaLogNormalStroke[]

    # Pattern-specific parameters
    direction_params = get_direction_params(pattern)

    # Number of sub-strokes per logical stroke for smooth curves
    sub_strokes_per_stroke = pattern == SIGNATURE ? 5 : 4  # More sub-strokes for smoother curves

    # Total number of sub-strokes
    total_sub_strokes = num_strokes * sub_strokes_per_stroke

    # Base time spacing (will be modulated by rhythm)
    effective_duration = duration - 0.1
    base_stroke_spacing = effective_duration / (total_sub_strokes * (1 - overlap_factor) + overlap_factor)

    # Initialize direction state
    base_direction = 0.0
    current_direction = base_direction
    strokes_per_line = max(3, num_strokes ÷ 3)

    # Track cumulative curve for natural flow
    curve_phase = rand(rng) * 2 * pi

    # ===== HUMAN-LIKE RHYTHM GENERATION =====
    # Generate a rhythm pattern for the whole sequence
    # Rhythm varies between "rushed" (fast) and "careful" (slow) phases
    rhythm_frequency = 0.3 + rand(rng) * 0.4  # Slow rhythm variation
    rhythm_phase = rand(rng) * 2 * pi
    base_tempo = 1.0  # Nominal tempo

    # Cumulative time tracker for natural timing
    cumulative_time = 0.03

    for i in 1:num_strokes
        # ===== STROKE-LEVEL DYNAMICS =====

        # Determine if this stroke is "rushed" or "careful"
        # Based on rhythm + random variation
        rhythm_mod = 0.3 * sin(rhythm_frequency * i * 2 * pi + rhythm_phase)
        stroke_tempo = base_tempo * (1.0 + rhythm_mod + randn(rng) * 0.15)
        stroke_tempo = clamp(stroke_tempo, 0.6, 1.5)  # Don't go too extreme

        # Rushed strokes: faster, slightly less precise, larger amplitude variation
        # Careful strokes: slower, more precise, consistent amplitude
        is_rushed = stroke_tempo > 1.1
        is_careful = stroke_tempo < 0.85

        # Micro-hesitation at stroke start (brief pause before some strokes)
        # More likely at careful strokes or after direction changes
        hesitation_prob = is_careful ? 0.4 : 0.15
        if rand(rng) < hesitation_prob && i > 1
            cumulative_time += rand(rng) * 0.04 + 0.01  # 10-50ms hesitation
        end

        # Stroke curve parameters with tempo-dependent variation
        curve_amplitude_base = pattern == SIGNATURE ? 0.6 : 0.4
        stroke_curve_amplitude = curve_amplitude_base * (0.6 + rand(rng) * 0.8)
        if is_rushed
            stroke_curve_amplitude *= 0.8  # Rushed = slightly flatter curves
        end
        stroke_curve_direction = rand(rng) < 0.5 ? 1.0 : -1.0

        # Base direction for this logical stroke
        stroke_base_direction = compute_stroke_direction(i, pattern, current_direction,
                                                         base_direction, direction_params,
                                                         strokes_per_line, rng)

        # ===== SUB-STROKE GENERATION =====
        for j in 1:sub_strokes_per_stroke
            sub_idx = (i - 1) * sub_strokes_per_stroke + j

            # Timing with natural jitter
            timing_jitter = randn(rng) * 0.008  # ±8ms jitter
            sub_spacing = base_stroke_spacing / stroke_tempo  # Tempo affects spacing

            t0 = cumulative_time + timing_jitter
            t0 = clamp(t0, 0.03, duration - 0.1)

            # Advance cumulative time (with overlap)
            cumulative_time += sub_spacing * (1 - overlap_factor * 0.7)

            # ===== AMPLITUDE (D) - Human-like variation =====
            # Base amplitude depends on stroke type and position
            if is_rushed
                # Rushed: larger, more variable amplitude
                base_D = 0.014 * (0.8 + rand(rng) * 0.6)
            elseif is_careful
                # Careful: smaller, more consistent amplitude
                base_D = 0.010 * (0.9 + rand(rng) * 0.3)
            else
                # Normal: moderate variation
                base_D = 0.012 * (0.7 + rand(rng) * 0.5)
            end

            # Sub-stroke position affects amplitude (peak in middle)
            sub_progress = (j - 1) / max(1, sub_strokes_per_stroke - 1)
            position_mod = 0.8 + 0.4 * sin(pi * sub_progress)  # Peak at middle

            D = base_D * position_mod * direction_params.stroke_size_variation
            D = clamp(D, 0.004, 0.028)

            # ===== TIMING PARAMETERS (mu, sigma) =====
            # mu controls peak time, sigma controls spread/sharpness

            if is_rushed
                # Rushed: earlier peak, sharper profile
                mu = rand(rng) * 0.25 - 2.8  # Earlier peak
                sigma = rand(rng) * 0.10 + 0.30  # Sharper (smaller sigma)
            elseif is_careful
                # Careful: later peak, smoother profile
                mu = rand(rng) * 0.25 - 2.4  # Later peak
                sigma = rand(rng) * 0.12 + 0.45  # Smoother (larger sigma)
            else
                # Normal: moderate variation
                mu = rand(rng) * 0.35 - 2.6
                sigma = rand(rng) * 0.15 + 0.38
            end

            # Add occasional "accent" strokes (sharper, more prominent)
            if rand(rng) < 0.12  # 12% chance of accent
                D *= 1.3
                sigma *= 0.85  # Sharper peak
            end

            # ===== DIRECTION =====
            # Smooth curve with natural irregularity
            curve_offset = stroke_curve_amplitude * sin(pi * sub_progress) * stroke_curve_direction

            # Global wave for flow
            global_wave = 0.12 * sin(curve_phase + sub_idx * 0.35)

            # Direction jitter (more for rushed strokes)
            dir_jitter = randn(rng) * (is_rushed ? 0.12 : 0.06)

            theta = stroke_base_direction + curve_offset + global_wave + dir_jitter

            push!(strokes, SigmaLogNormalStroke(D, t0, mu, sigma, theta))
        end

        # Update direction with momentum
        current_direction = direction_params.direction_momentum * current_direction +
                           (1 - direction_params.direction_momentum) * stroke_base_direction

        # Advance curve phase
        curve_phase += 0.6 + rand(rng) * 0.5
    end

    return strokes
end

"""
    DirectionParams

Parameters controlling stroke direction evolution for different patterns.
"""
struct DirectionParams
    direction_momentum::Float64
    direction_variation::Float64
    return_to_baseline::Float64
    stroke_size_variation::Float64
end

"""
    get_direction_params(pattern::StrokePattern)

Get direction parameters for a specific stroke pattern.
Returns DirectionParams(momentum, variation, baseline_pull, size_variation)

Enhanced for more dynamic, human-like motion with increased variation.
"""
function get_direction_params(pattern::StrokePattern)
    if pattern == CURSIVE_WORD
        # Cursive: smooth flowing motion with natural loops and curves
        # Higher variation for more dynamic letter-like shapes
        DirectionParams(0.55, 0.75, 0.20, 1.15)
    elseif pattern == PRINT_LETTERS
        # Print: more vertical strokes, discrete letter shapes
        # Sharp direction changes between letters
        DirectionParams(0.35, 1.0, 0.15, 1.25)
    elseif pattern == SIGNATURE
        # Signature: flowing with large sweeping curved motions
        # High momentum, large sweeps
        DirectionParams(0.70, 0.9, 0.08, 1.4)
    elseif pattern == SCRIBBLE
        # Scribble: random curved wandering
        # Low momentum, high variation
        DirectionParams(0.45, 1.2, 0.0, 1.35)
    else  # LINE_BY_LINE
        # Lines: horizontal progression with oscillation
        DirectionParams(0.50, 0.6, 0.35, 1.0)
    end
end

"""
    compute_stroke_direction(i::Int, pattern::StrokePattern,
                             current_direction::Float64, base_direction::Float64,
                             params::DirectionParams, strokes_per_line::Int, rng)

Compute direction for stroke i based on pattern and previous direction.
Creates natural oscillating patterns that mimic real handwriting motion.

Enhanced with:
- Multi-frequency oscillation for letter-like shapes
- Natural direction "bursts" mimicking letter strokes
- Pattern-specific dynamics (loops, angles, sweeps)
"""
function compute_stroke_direction(i::Int, pattern::StrokePattern,
                                  current_direction::Float64, base_direction::Float64,
                                  params::DirectionParams, strokes_per_line::Int, rng)
    if pattern == LINE_BY_LINE && i > 1 && (i - 1) % strokes_per_line == 0
        # New line: carriage return stroke
        return pi + randn(rng) * 0.25
    end

    # Smooth direction evolution with momentum
    random_delta = randn(rng) * params.direction_variation
    baseline_pull = (base_direction - current_direction) * params.return_to_baseline

    new_direction = current_direction + random_delta + baseline_pull

    # Add pattern-specific oscillation for natural curved motion
    if pattern == CURSIVE_WORD
        # Cursive: natural letter-like oscillation (m, n, u, w, e, l patterns)
        # Three frequency components for realistic variation
        # Low freq: overall flow direction
        # Mid freq: letter-level oscillation (up-down-up)
        # High freq: within-letter detail

        low_freq = 0.2 * sin(i * 0.4 + randn(rng) * 0.1)  # Slow drift
        mid_freq = 0.55 * sin(i * 1.4 + randn(rng) * 0.15)  # Letter oscillation
        high_freq = 0.25 * sin(i * 3.2)  # Detail

        # Occasional "loop" - sharper direction change (like 'l', 'e', 'o')
        if rand(rng) < 0.15
            loop_burst = randn(rng) * 0.4
            new_direction += loop_burst
        end

        new_direction += low_freq + mid_freq + high_freq

    elseif pattern == SIGNATURE
        # Signature: large sweeping curves with flourishes
        # Signatures have dramatic direction changes and loops

        sweep = 0.5 * sin(i * 0.45)  # Large slow sweep
        flourish = 0.35 * sin(i * 1.6 + randn(rng) * 0.2)  # Quick flourishes

        # Occasional dramatic flourish
        if rand(rng) < 0.12
            dramatic = randn(rng) * 0.6
            new_direction += dramatic
        end

        new_direction += sweep + flourish

    elseif pattern == SCRIBBLE
        # Scribble: chaotic but locally smooth wandering
        # Random walk with occasional direction reversals

        wander = randn(rng) * 0.5 + 0.35 * sin(i * 1.3)

        # Occasional sharp turn
        if rand(rng) < 0.18
            sharp_turn = randn(rng) * 0.7
            new_direction += sharp_turn
        end

        new_direction += wander

    elseif pattern == PRINT_LETTERS
        # Print: angular discrete letter shapes
        # More structured oscillation mimicking print letter strokes
        # (vertical down, horizontal across, diagonal)

        # Letter structure: alternate between down, across, up strokes
        stroke_type = i % 4
        if stroke_type == 0
            angle_target = -pi/2 + randn(rng) * 0.2  # Down
        elseif stroke_type == 1
            angle_target = 0.0 + randn(rng) * 0.2    # Right
        elseif stroke_type == 2
            angle_target = pi/2 + randn(rng) * 0.2   # Up
        else
            angle_target = -pi/4 + randn(rng) * 0.3  # Diagonal
        end

        # Blend toward target with some randomness
        letter_pull = 0.4 * (angle_target - new_direction)
        new_direction += letter_pull + randn(rng) * 0.2

    else  # LINE_BY_LINE
        # Lines: subtle oscillation within line, staying mostly horizontal
        line_osc = 0.3 * sin(i * 0.85)
        new_direction += line_osc
    end

    return new_direction
end

"""
    generate_random_strokes(num_strokes::Int, duration::Float64; rng=Random.default_rng(),
                           separated::Bool=false, gap_duration::Float64=0.3,
                           pattern::StrokePattern=CURSIVE_WORD)

Generate Sigma-LogNormal strokes with geometric connectivity for realistic handwriting.

When separated=false (default), uses superposition for smooth stroke transitions.
When separated=true, creates distinct strokes with gaps (legacy mode).

# Arguments
- `num_strokes`: Number of strokes to generate
- `duration`: Total writing duration
- `rng`: Random number generator
- `separated`: If true, creates distinct strokes with gaps (legacy mode)
- `gap_duration`: Minimum gap between strokes when separated=true (seconds)
- `pattern`: Stroke pattern for geometric connectivity
"""
function generate_random_strokes(num_strokes::Int, duration::Float64;
                                 rng=Random.default_rng(),
                                 separated::Bool=false,
                                 gap_duration::Float64=0.3,
                                 pattern::StrokePattern=CURSIVE_WORD)
    if !separated
        # Use superposition mode for smooth curved connections (default)
        return generate_superposition_strokes(num_strokes, duration;
                                              pattern=pattern,
                                              overlap_factor=0.5,
                                              rng=rng)
    end

    # Legacy separated mode
    strokes = SigmaLogNormalStroke[]

    # Calculate timing
    time_per_stroke = duration / num_strokes
    stroke_duration = time_per_stroke - gap_duration

    if stroke_duration < 0.15
        stroke_duration = 0.15
        gap_duration = max(0.05, time_per_stroke - stroke_duration)
    end

    # Get pattern parameters
    params = get_direction_params(pattern)

    # Initialize direction state
    base_direction = 0.0
    current_direction = base_direction
    strokes_per_line = max(3, num_strokes ÷ 3)

    for i in 1:num_strokes
        # Start time of this stroke
        t_start = (i - 1) * time_per_stroke + gap_duration/2
        t0 = t_start + 0.01
        t0 = clamp(t0, 0.01, duration - 0.1)

        # Stroke amplitude
        base_D = 0.015
        D = base_D * (1.0 + (rand(rng) - 0.5) * 2 * params.stroke_size_variation)
        D = clamp(D, 0.008, 0.03)

        # Sigma-LogNormal timing parameters
        mu = rand(rng) * 0.5 - 3.0
        sigma = rand(rng) * 0.15 + 0.25

        # Direction
        theta = compute_stroke_direction(i, pattern, current_direction,
                                        base_direction, params, strokes_per_line, rng)
        current_direction = params.direction_momentum * current_direction +
                           (1 - params.direction_momentum) * theta

        push!(strokes, SigmaLogNormalStroke(D, t0, mu, sigma, theta))
    end

    return strokes
end

export generate_random_strokes, generate_curved_strokes, generate_spiral_stroke, generate_zigzag_strokes

"""
    minimum_jerk_trajectory(t::Float64, T::Float64, x0::Float64, xf::Float64)

Compute position at time t for minimum-jerk trajectory from x0 to xf over duration T.

Based on Flash & Hogan (1985): The coordination of arm movements.
The trajectory minimizes the integral of squared jerk (third derivative).

Returns (position, velocity, acceleration) tuple.
"""
function minimum_jerk_trajectory(t::Float64, T::Float64, x0::Float64, xf::Float64)
    if T <= 0
        return (xf, 0.0, 0.0)
    end

    # Normalized time
    tau = clamp(t / T, 0.0, 1.0)

    # Polynomial coefficients for minimum jerk: x(tau) = x0 + (xf - x0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    # Position
    s = 10 * tau3 - 15 * tau4 + 6 * tau5
    x = x0 + (xf - x0) * s

    # Velocity: ds/dt = (1/T) * ds/dtau = (1/T) * (30*tau^2 - 60*tau^3 + 30*tau^4)
    ds_dtau = 30 * tau2 - 60 * tau3 + 30 * tau4
    v = (xf - x0) * ds_dtau / T

    # Acceleration: dv/dt = (1/T^2) * d^2s/dtau^2 = (1/T^2) * (60*tau - 180*tau^2 + 120*tau^3)
    d2s_dtau2 = 60 * tau - 180 * tau2 + 120 * tau3
    a = (xf - x0) * d2s_dtau2 / (T * T)

    return (x, v, a)
end

"""
    detect_stroke_boundaries(vel_w::Matrix{Float64}, dt::Float64, static_samples::Int;
                             vel_threshold::Float64=0.005,
                             min_gap_duration::Float64=0.05)

Detect start/end times of strokes based on velocity threshold.

Returns vector of (start_time, end_time) tuples.

# Arguments
- `vel_w`: Velocity array [n, 3]
- `dt`: Time step
- `static_samples`: Number of static buffer samples
- `vel_threshold`: Velocity threshold for stroke detection
- `min_gap_duration`: Minimum gap duration to count as separate strokes
"""
function detect_stroke_boundaries(vel_w::Matrix{Float64}, dt::Float64, static_samples::Int;
                                  vel_threshold::Float64=0.005,
                                  min_gap_duration::Float64=0.05)
    n = size(vel_w, 1)
    vel_mag = sqrt.(sum(vel_w.^2, dims=2))[:]

    strokes = Tuple{Float64,Float64}[]
    in_stroke = false
    stroke_start_time = 0.0
    min_gap_samples = round(Int, min_gap_duration / dt)

    for i in (static_samples+1):n
        t = (i - static_samples - 1) * dt  # Time relative to writing start

        if !in_stroke && vel_mag[i] > vel_threshold
            # Starting a new stroke
            in_stroke = true
            stroke_start_time = t
        elseif in_stroke && vel_mag[i] <= vel_threshold
            # Potentially ending a stroke
            # Look ahead to see if this is a real gap or just a brief pause
            gap_samples = 0
            for j in i:min(i + min_gap_samples*2, n)
                if vel_mag[j] <= vel_threshold
                    gap_samples += 1
                else
                    break
                end
            end

            # If gap is long enough, mark stroke end
            if gap_samples >= min_gap_samples
                in_stroke = false
                push!(strokes, (stroke_start_time, t))
            end
        end
    end

    # Handle stroke extending to end
    if in_stroke
        t_end = (n - static_samples - 1) * dt
        push!(strokes, (stroke_start_time, t_end))
    end

    return strokes
end

"""
    generate_pen_lift_events(stroke_boundaries::Vector{Tuple{Float64,Float64}};
                             params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS, rng=Random.default_rng())

Generate pen lift events between detected strokes.
"""
function generate_pen_lift_events(stroke_boundaries::Vector{Tuple{Float64,Float64}};
                                   params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS, rng=Random.default_rng())
    lift_events = PenLiftEvent[]

    if length(stroke_boundaries) < 2
        return lift_events
    end

    for i in 1:(length(stroke_boundaries)-1)
        # Gap between end of stroke i and start of stroke i+1
        gap_start = stroke_boundaries[i][2]
        gap_end = stroke_boundaries[i+1][1]
        gap_duration = gap_end - gap_start

        if gap_duration < 0.05  # Skip if gap is too short
            continue
        end

        # Random lift parameters
        lift_dur = rand(rng) * (params.lift_duration_range[2] - params.lift_duration_range[1]) + params.lift_duration_range[1]
        land_dur = rand(rng) * (params.land_duration_range[2] - params.land_duration_range[1]) + params.land_duration_range[1]
        hover_dur = rand(rng) * (params.hover_duration_range[2] - params.hover_duration_range[1]) + params.hover_duration_range[1]
        z_height = rand(rng) * (params.lift_height_range[2] - params.lift_height_range[1]) + params.lift_height_range[1]

        # Total required duration
        total_required = lift_dur + hover_dur + land_dur

        # Scale durations if gap is too short
        if total_required > gap_duration
            scale = gap_duration / total_required * 0.9  # Leave 10% margin
            lift_dur *= scale
            hover_dur *= scale
            land_dur *= scale
        end

        t_start = gap_start
        t_lift_end = t_start + lift_dur
        t_land_start = t_lift_end + hover_dur
        t_end = t_land_start + land_dur

        push!(lift_events, PenLiftEvent(t_start, t_lift_end, t_land_start, t_end, z_height))
    end

    return lift_events
end

"""
    compute_z_trajectory(time::Vector{Float64}, lift_events::Vector{PenLiftEvent},
                        static_samples::Int)

Compute Z position, velocity, and acceleration from pen lift events using minimum-jerk.
Returns (z_pos, z_vel, z_accel, pen_down) tuple.
"""
function compute_z_trajectory(time::Vector{Float64}, lift_events::Vector{PenLiftEvent},
                              static_samples::Int)
    n = length(time)
    z_pos = zeros(n)
    z_vel = zeros(n)
    z_accel = zeros(n)
    pen_down = fill(true, n)

    for i in 1:n
        # Time relative to writing start
        if i <= static_samples
            # Static period: pen is down at z=0
            continue
        end

        t_write = time[i]  # Already relative time from time vector

        # Check if we're in a lift event
        in_lift = false
        for event in lift_events
            if t_write >= event.t_start && t_write <= event.t_end
                in_lift = true
                pen_down[i] = false

                if t_write <= event.t_lift_end
                    # Lifting phase: 0 -> z_height
                    t_local = t_write - event.t_start
                    T_phase = event.t_lift_end - event.t_start
                    z_pos[i], z_vel[i], z_accel[i] = minimum_jerk_trajectory(t_local, T_phase, 0.0, event.z_height)

                elseif t_write <= event.t_land_start
                    # Hover phase: constant z_height
                    z_pos[i] = event.z_height
                    z_vel[i] = 0.0
                    z_accel[i] = 0.0

                else
                    # Landing phase: z_height -> 0
                    t_local = t_write - event.t_land_start
                    T_phase = event.t_end - event.t_land_start
                    z_pos[i], z_vel[i], z_accel[i] = minimum_jerk_trajectory(t_local, T_phase, event.z_height, 0.0)
                end

                break
            end
        end
    end

    return z_pos, z_vel, z_accel, pen_down
end

"""
    generate_trajectory(strokes::Vector{SigmaLogNormalStroke}, dt::Float64, duration::Float64;
                        static_buffer::Float64=STATIC_BUFFER_S,
                        pen_lift_params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS,
                        rng=Random.default_rng())

Generate a complete trajectory from Sigma-LogNormal strokes with superposition.

The velocity at each time point is the sum of all stroke velocity profiles (superposition).
This creates smooth, continuous trajectories without velocity discontinuities.

Includes a static buffer at the start for ESKF initialization.
"""
function generate_trajectory(strokes::Vector{SigmaLogNormalStroke}, dt::Float64, duration::Float64;
                             static_buffer::Float64=STATIC_BUFFER_S,
                             pen_lift_params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS,
                             rng=Random.default_rng())
    # Time vector for writing portion
    t_write = collect(0.0:dt:duration)
    n_write = length(t_write)

    # Static buffer
    n_static = round(Int, static_buffer / dt)
    n_total = n_static + n_write

    # Initialize arrays
    vel_w = zeros(n_total, 3)
    accel_w = zeros(n_total, 3)

    # Compute velocity from strokes using SUPERPOSITION (sum of all stroke profiles)
    for stroke in strokes
        v_mag = velocity_profile(stroke, t_write)
        a_mag = velocity_derivative(stroke, t_write)

        # Direction in XY plane
        cos_theta = cos(stroke.theta)
        sin_theta = sin(stroke.theta)

        for i in 1:n_write
            vel_w[n_static + i, 1] += v_mag[i] * cos_theta
            vel_w[n_static + i, 2] += v_mag[i] * sin_theta
            accel_w[n_static + i, 1] += a_mag[i] * cos_theta
            accel_w[n_static + i, 2] += a_mag[i] * sin_theta
        end
    end

    # Full time vector (relative to writing start, negative for static portion)
    t_full = [(i - n_static - 1) * dt for i in 1:n_total]

    # Detect stroke boundaries for pen lift
    stroke_boundaries = detect_stroke_boundaries(vel_w, dt, n_static;
                                                  vel_threshold=pen_lift_params.vel_threshold)

    # Generate pen lift events
    lift_events = generate_pen_lift_events(stroke_boundaries;
                                            params=pen_lift_params, rng=rng)

    # Compute Z trajectory from lift events
    z_pos, z_vel, z_accel, pen_down = compute_z_trajectory(t_full, lift_events, n_static)

    # Integrate XY velocity to position (trapezoidal rule)
    pos_w = zeros(n_total, 3)
    for i in 2:n_total
        pos_w[i, 1:2] = pos_w[i-1, 1:2] + (vel_w[i-1, 1:2] + vel_w[i, 1:2]) * dt / 2
    end
    pos_w[:, 3] = z_pos
    vel_w[:, 3] = z_vel
    accel_w[:, 3] = z_accel

    return TrajectoryResult(pos_w, vel_w, accel_w, t_full, pen_down, lift_events)
end

"""
    generate_trajectory(num_strokes::Int, duration::Float64; kwargs...)

Convenience method: generate strokes and trajectory in one call.

Uses superposition by default for smooth stroke connections.

# Arguments
- `separated`: If true, creates distinct strokes with gaps. Default: false (uses superposition)
- `gap_duration`: Minimum gap between strokes when separated=true. Default: 0.3s
- `pattern`: Stroke pattern for geometric connectivity. Default: CURSIVE_WORD
"""
function generate_trajectory(num_strokes::Int, duration::Float64;
                             dt::Float64=DT, static_buffer::Float64=STATIC_BUFFER_S,
                             pen_lift_params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS,
                             separated::Bool=false,
                             gap_duration::Float64=0.3,
                             pattern::StrokePattern=CURSIVE_WORD,
                             rng=Random.default_rng())
    strokes = generate_random_strokes(num_strokes, duration;
                                      rng=rng,
                                      separated=separated,
                                      gap_duration=gap_duration,
                                      pattern=pattern)
    return generate_trajectory(strokes, dt, duration;
                               static_buffer=static_buffer,
                               pen_lift_params=pen_lift_params,
                               rng=rng)
end

"""
    generate_curved_strokes(num_strokes::Int, duration::Float64; rng=Random.default_rng())

Generate curved strokes using multiple overlapping sub-strokes for smooth curves.

Each stroke follows a smooth curve with varying curvature, mimicking natural writing motion.
Uses superposition of sub-strokes for natural acceleration profiles.
"""
function generate_curved_strokes(num_strokes::Int, duration::Float64;
                                  rng=Random.default_rng())
    strokes = SigmaLogNormalStroke[]
    time_per_stroke = duration / num_strokes

    for i in 1:num_strokes
        # Create multiple overlapping sub-strokes for a curved path
        num_sub = 3 + rand(rng, 1:2)  # 3-4 sub-strokes per curve
        sub_duration = time_per_stroke / num_sub

        base_theta = randn(rng) * 0.5

        for j in 1:num_sub
            # Overlapping timing for smooth superposition
            t_base = (i - 1) * time_per_stroke + (j - 1) * sub_duration * 0.7
            t0 = t_base + 0.02

            # Gradually varying theta creates smooth curve
            curve_offset = (j - 1) * (rand(rng) * 0.6 - 0.3)
            theta = base_theta + curve_offset

            # Smaller D for sub-strokes, superposition creates smooth curve
            D = rand(rng) * 0.008 + 0.006
            mu = rand(rng) * 0.3 - 2.7  # Adjusted for better overlap
            sigma = rand(rng) * 0.15 + 0.35

            push!(strokes, SigmaLogNormalStroke(D, t0, mu, sigma, theta))
        end
    end

    return strokes
end

"""
    generate_spiral_stroke(num_loops::Int, radius_start::Float64, radius_end::Float64,
                          duration::Float64; rng=Random.default_rng())

Generate a spiral pattern using multiple small overlapping strokes arranged in a spiral.

Note: Spiral is centered at origin (0,0). Position offset is handled at trajectory level.

# Arguments
- `num_loops`: Number of complete loops
- `radius_start`: Starting radius
- `radius_end`: Ending radius
- `duration`: Total duration
"""
function generate_spiral_stroke(num_loops::Int, radius_start::Float64, radius_end::Float64,
                                duration::Float64; rng=Random.default_rng())
    num_segments = num_loops * 12  # 12 segments per loop
    strokes = SigmaLogNormalStroke[]

    for i in 1:num_segments
        progress = (i - 1) / num_segments
        angle = progress * num_loops * 2 * pi
        radius = radius_start + (radius_end - radius_start) * progress

        # Tangent direction for spiral
        dr_dtheta = (radius_end - radius_start) / (num_loops * 2 * pi)
        theta = atan(dr_dtheta, radius) + angle

        # Overlapping timing for smooth spiral
        t0 = progress * duration * 0.9 + 0.02
        D = 0.008 + rand(rng) * 0.004
        mu = rand(rng) * 0.2 - 2.6
        sigma = rand(rng) * 0.15 + 0.35

        push!(strokes, SigmaLogNormalStroke(D, t0, mu, sigma, theta))
    end

    return strokes
end

"""
    generate_zigzag_strokes(num_strokes::Int, duration::Float64;
                           amplitude::Float64=0.02, rng=Random.default_rng())

Generate zigzag pattern strokes with sharp directional changes.

Uses superposition with controlled overlap for smoother transitions
while maintaining the zigzag character.
"""
function generate_zigzag_strokes(num_strokes::Int, duration::Float64;
                                  amplitude::Float64=0.02, rng=Random.default_rng())
    strokes = SigmaLogNormalStroke[]
    time_per_stroke = duration / num_strokes

    for i in 1:num_strokes
        # Overlapping timing
        t0 = (i - 1) * time_per_stroke * 0.85 + 0.05

        # Alternate between positive and negative angles for zigzag
        base_direction = (i % 2 == 0) ? pi/3 : -pi/3  # +/- 60 degrees
        theta = base_direction + randn(rng) * 0.2

        # Sharper strokes for zigzag character
        D = amplitude * (0.8 + rand(rng) * 0.4)
        mu = rand(rng) * 0.25 - 2.7
        sigma = rand(rng) * 0.12 + 0.28  # Slightly sharper peaks

        push!(strokes, SigmaLogNormalStroke(D, t0, mu, sigma, theta))
    end

    return strokes
end

"""
    generate_trajectory(method::String, num_strokes::Int, duration::Float64; kwargs...)

Enhanced trajectory generation supporting multiple methods via string identifier.

All methods use superposition for smooth stroke connections.

# Methods
- "sigma_lognormal" or "superposition": Superposition mode (default, smooth connections)
- "separated": Legacy separated strokes with gaps
- "curved": Smooth curved strokes using overlapping sub-strokes
- "spiral": Spiral pattern (requires center position in kwargs)
- "zigzag": Sharp zigzag pattern with controlled transitions
"""
function generate_trajectory(method::String, num_strokes::Int, duration::Float64;
                            dt::Float64=DT, static_buffer::Float64=STATIC_BUFFER_S,
                            pen_lift_params::PenLiftParams=DEFAULT_PEN_LIFT_PARAMS,
                            rng=Random.default_rng(),
                            pattern::StrokePattern=CURSIVE_WORD,
                            # Spiral-specific parameters
                            spiral_radius_range::Tuple{Float64,Float64}=(0.01, 0.04),
                            # Zigzag-specific parameters
                            zigzag_amplitude::Float64=0.02)

    strokes = if method == "curved"
        generate_curved_strokes(num_strokes, duration; rng=rng)
    elseif method == "spiral"
        num_loops = max(1, num_strokes ÷ 3)
        generate_spiral_stroke(num_loops, spiral_radius_range[1], spiral_radius_range[2],
                              duration; rng=rng)
    elseif method == "zigzag"
        generate_zigzag_strokes(num_strokes, duration;
                               amplitude=zigzag_amplitude, rng=rng)
    elseif method == "separated"
        generate_random_strokes(num_strokes, duration; rng=rng, separated=true, pattern=pattern)
    else  # Default: "sigma_lognormal" or "superposition"
        generate_random_strokes(num_strokes, duration; rng=rng, separated=false, pattern=pattern)
    end

    return generate_trajectory(strokes, dt, duration;
                              static_buffer=static_buffer,
                              pen_lift_params=pen_lift_params,
                              rng=rng)
end

# =============================================================================
# POSE-AWARE CONSTRAINTS
# =============================================================================

"""
    PoseConstraints

Biomechanical constraints that affect trajectory generation based on pen pose.

These constraints ensure the generated trajectory respects physical limitations
of human arm/wrist motion and creates more realistic handwriting.

# Fields
- `max_velocity::Float64`: Maximum writing velocity (m/s)
- `min_curvature_radius::Float64`: Minimum turn radius at max velocity (m)
- `velocity_curvature_coupling::Float64`: How much velocity affects curvature (higher = stricter)
- `workspace_center::Vector{Float64}`: Center of comfortable writing workspace (m)
- `workspace_radius::Float64`: Radius of comfortable workspace (m)
- `preferred_direction::Float64`: Preferred stroke direction based on handedness (rad)
- `direction_cost_weight::Float64`: How much non-preferred directions are penalized
- `acceleration_limit::Float64`: Maximum comfortable acceleration (m/s²)
"""
struct PoseConstraints
    max_velocity::Float64
    min_curvature_radius::Float64
    velocity_curvature_coupling::Float64
    workspace_center::Vector{Float64}
    workspace_radius::Float64
    preferred_direction::Float64
    direction_cost_weight::Float64
    acceleration_limit::Float64
end

const DEFAULT_POSE_CONSTRAINTS = PoseConstraints(
    0.25,                    # max_velocity (m/s) - typical fast writing
    0.005,                   # min_curvature_radius (m) - 5mm minimum turn radius
    2.0,                     # velocity_curvature_coupling
    [0.05, 0.05, 0.0],       # workspace_center (m) - slightly right and up from origin
    0.15,                    # workspace_radius (m) - 15cm comfortable reach
    0.0,                     # preferred_direction (rad) - 0 = right (for right-handed)
    0.3,                     # direction_cost_weight
    5.0                      # acceleration_limit (m/s²)
)

"""
    apply_pose_constraints!(trajectory::TrajectoryResult, constraints::PoseConstraints;
                            dt::Float64=DT)

Apply biomechanical pose constraints to a generated trajectory in-place.

Constraints applied:
1. Velocity limiting with smooth clamping
2. Velocity-curvature relationship (faster motion = smoother curves)
3. Workspace bounds (soft constraint, smoothly pulls back to workspace)
4. Acceleration limiting

Note: This modifies the trajectory in-place and recalculates acceleration.
"""
function apply_pose_constraints!(trajectory::TrajectoryResult, constraints::PoseConstraints;
                                  dt::Float64=DT)
    n = size(trajectory.pos_w, 1)
    pos = trajectory.pos_w
    vel = trajectory.vel_w
    accel = trajectory.accel_w

    # 1. Apply velocity-curvature constraint
    # At high velocity, limit curvature (enforce smoother curves)
    for i in 3:n-2
        v_mag = norm(vel[i, 1:2])  # XY velocity magnitude

        if v_mag > 0.001  # Only when moving
            # Compute local curvature from velocity change
            v_prev = vel[i-1, 1:2]
            v_curr = vel[i, 1:2]

            # Direction change rate
            if norm(v_curr) > 0.001 && norm(v_prev) > 0.001
                cos_angle = dot(v_curr, v_prev) / (norm(v_curr) * norm(v_prev))
                cos_angle = clamp(cos_angle, -1.0, 1.0)
                direction_change = acos(cos_angle)

                # Approximate curvature
                curvature = direction_change / (v_mag * dt)

                # Maximum allowed curvature decreases with velocity
                # κ_max = 1 / (r_min * (1 + c * v²))
                max_curvature = 1.0 / (constraints.min_curvature_radius *
                               (1.0 + constraints.velocity_curvature_coupling * v_mag^2))

                # If curvature exceeds limit, reduce velocity
                if curvature > max_curvature && curvature > 0
                    reduction_factor = sqrt(max_curvature / curvature)
                    reduction_factor = clamp(reduction_factor, 0.5, 1.0)  # Don't reduce too much
                    vel[i, :] *= reduction_factor
                end
            end
        end
    end

    # 2. Apply velocity magnitude limit with smooth clamping
    for i in 1:n
        v_mag = norm(vel[i, 1:2])
        if v_mag > constraints.max_velocity
            # Smooth velocity limiting using tanh-like function
            scale = constraints.max_velocity / v_mag
            # Soft clamp: don't cut abruptly
            soft_scale = scale + (1 - scale) * 0.1
            vel[i, 1:2] *= soft_scale
        end
    end

    # 3. Apply workspace constraint (soft boundary)
    for i in 1:n
        pos_xy = pos[i, 1:2]
        center_xy = constraints.workspace_center[1:2]
        dist_from_center = norm(pos_xy - center_xy)

        if dist_from_center > constraints.workspace_radius * 0.8
            # Soft pull back toward center
            overshoot = dist_from_center - constraints.workspace_radius * 0.8
            pull_strength = min(0.1, overshoot / constraints.workspace_radius)

            direction_to_center = (center_xy - pos_xy) / max(dist_from_center, 0.001)
            vel[i, 1:2] += direction_to_center * pull_strength * norm(vel[i, 1:2])
        end
    end

    # 4. Apply acceleration limit
    for i in 1:n
        a_mag = norm(accel[i, 1:2])
        if a_mag > constraints.acceleration_limit
            accel[i, 1:2] *= constraints.acceleration_limit / a_mag
        end
    end

    # 5. Recalculate position from constrained velocity (forward integration)
    for i in 2:n
        pos[i, 1:2] = pos[i-1, 1:2] + (vel[i-1, 1:2] + vel[i, 1:2]) * dt / 2
    end

    # 6. Recalculate acceleration from constrained velocity
    for i in 2:n-1
        accel[i, :] = (vel[i+1, :] - vel[i-1, :]) / (2 * dt)
    end
    accel[1, :] = accel[2, :]
    accel[end, :] = accel[end-1, :]

    return trajectory
end

"""
    dot(a::AbstractVector, b::AbstractVector)

Compute dot product of two vectors.
"""
function dot(a::AbstractVector, b::AbstractVector)
    return sum(a .* b)
end

end # module
