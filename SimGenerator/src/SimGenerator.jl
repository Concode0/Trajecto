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
SimGenerator: Synthetic Trajecto Data Generator for Sim2Real Training.

Three-layer architecture:
1. KinematicLayer: Sigma-LogNormal trajectory generation with superposition
2. PostureLayer: Wrist/arm orientation simulation with inverse kinematics
3. SensingLayer: IMU + FSR sensor modeling with enhanced noise

Usage:
    using SimGenerator
    samples = generate_samples(100; num_strokes=10, duration=5.0)
    export_hdf5("data/sim.h5", samples)
"""
module SimGenerator

# Include submodules in dependency order
include("Config.jl")
using .Config

include("KinematicLayer.jl")
using .KinematicLayer

include("PostureLayer.jl")
using .PostureLayer

include("SensingLayer.jl")
using .SensingLayer

include("HDF5Writer.jl")
using .HDF5Writer

include("Statistics.jl")
using .DataStatistics

include("Visualization.jl")
using .Visualization

# Re-export public API
export Config, KinematicLayer, PostureLayer, SensingLayer, HDF5Writer, DataStatistics, Visualization

# Re-export commonly used types and functions
export TrajectoryResult, OrientationResult, SensorResult, TrajectoryData
export generate_trajectory, generate_orientation, generate_sensor_data, export_hdf5

# Re-export pen lift and IK types
export PenLiftEvent, PenLiftParams, DEFAULT_PEN_LIFT_PARAMS
export WristJointState, ArmKinematicParams, JointLimits, WristIKParams
export DEFAULT_ARM_PARAMS, DEFAULT_JOINT_LIMITS, DEFAULT_WRIST_IK_PARAMS

# Re-export noise parameter types
export TremorParams, PinkNoiseParams, FSRBaselineParams
export DEFAULT_TREMOR_PARAMS, DEFAULT_PINK_NOISE_PARAMS, DEFAULT_FSR_BASELINE_PARAMS

# Re-export grip style and handedness types
export GripStyle, TRIPOD, LATERAL_TRIPOD, QUADRUPOD, DYNAMIC_TRIPOD, OVERHAND
export GripStyleParams, GRIP_TRIPOD, GRIP_LATERAL_TRIPOD, GRIP_QUADRUPOD
export GRIP_DYNAMIC_TRIPOD, GRIP_OVERHAND, ALL_GRIP_STYLES, DEFAULT_GRIP_STYLE
export Handedness, RIGHT_HAND, LEFT_HAND
export HandednessParams, RIGHT_HAND_PARAMS, LEFT_HAND_PARAMS, DEFAULT_HANDEDNESS
export random_grip_style, apply_handedness, apply_grip_style, generate_orientation_with_style
export generate_world_rotation, apply_world_rotation_to_trajectory, get_rotated_gravity

# World rotation params
export WorldRotationParams, DEFAULT_WORLD_ROTATION_PARAMS, DISABLED_WORLD_ROTATION_PARAMS

# Re-export stroke pattern types
export StrokePattern, CURSIVE_WORD, PRINT_LETTERS, SIGNATURE, SCRIBBLE, LINE_BY_LINE

# Configuration constants
export DT, STATIC_BUFFER_S, DEFAULT_OUTPUT_PATH, DEFAULT_NUM_STROKES, DEFAULT_WRITING_DURATION

# Re-export Statistics module functions
export DatasetStatistics, compute_dataset_statistics, print_config_update, print_statistics_summary

# Re-export Visualization module functions
export visualize_sample, visualize_6dof, visualize_imu, visualize_trajectory_3d
export visualize_batch_summary, save_visualization, quat_to_euler

using Random
using Statistics: mean  # Julia stdlib Statistics

"""
    SimulationParams

Combined parameters for the full simulation pipeline.

All parameters have sensible defaults, allowing incremental customization.
Uses Sigma-LogNormal with superposition for smooth trajectory generation.
"""
struct SimulationParams
    # Kinematic layer
    pen_lift_params::PenLiftParams
    use_superposition::Bool  # true = smooth superposition, false = separated strokes

    # Posture layer
    arm_params::ArmKinematicParams
    joint_limits::JointLimits
    ik_params::WristIKParams
    use_ik::Bool

    # Grip style and handedness
    grip_style::Union{GripStyleParams, Nothing}  # nothing = random selection
    handedness::Union{HandednessParams, Nothing} # nothing = random selection
    randomize_grip::Bool  # If true, randomly select grip style for each sample
    randomize_handedness::Bool # If true, randomly select handedness for each sample

    # World rotation augmentation
    world_rotation_params::WorldRotationParams

    # Sensing layer
    tremor_params::TremorParams
    pink_params::PinkNoiseParams
    baseline_params::FSRBaselineParams
    use_enhanced_noise::Bool
end

# Default simulation parameters (using Sigma-LogNormal with superposition)
const DEFAULT_SIM_PARAMS = SimulationParams(
    DEFAULT_PEN_LIFT_PARAMS,
    true,    # use_superposition = true for smooth stroke connections
    DEFAULT_ARM_PARAMS,
    DEFAULT_JOINT_LIMITS,
    DEFAULT_WRIST_IK_PARAMS,
    true,  # use_ik
    DEFAULT_GRIP_STYLE,  # grip_style
    DEFAULT_HANDEDNESS,  # handedness
    true,  # randomize_grip
    true,  # randomize_handedness
    DEFAULT_WORLD_ROTATION_PARAMS,  # world rotation augmentation
    DEFAULT_TREMOR_PARAMS,
    DEFAULT_PINK_NOISE_PARAMS,
    DEFAULT_FSR_BASELINE_PARAMS,
    true   # use_enhanced_noise
)

export SimulationParams, DEFAULT_SIM_PARAMS

"""
    generate_sample(; num_strokes::Int=DEFAULT_NUM_STROKES,
                     duration::Float64=DEFAULT_WRITING_DURATION,
                     label::String="sim_sample",
                     sim_params::SimulationParams=DEFAULT_SIM_PARAMS,
                     stroke_pattern::StrokePattern=CURSIVE_WORD,
                     rng=Random.default_rng())

Generate a single trajectory sample through all three layers.

Uses Sigma-LogNormal model with superposition by default for smooth, natural trajectories.
Set sim_params.use_superposition = false for legacy separated stroke mode.

# Arguments
- `num_strokes`: Number of strokes (affects trajectory complexity)
- `duration`: Writing duration in seconds
- `label`: Sample label for HDF5 export
- `sim_params`: Combined simulation parameters
- `stroke_pattern`: Geometric pattern for stroke connectivity (CURSIVE_WORD, PRINT_LETTERS, SIGNATURE, SCRIBBLE, LINE_BY_LINE)
- `rng`: Random number generator
"""
function generate_sample(; num_strokes::Int=DEFAULT_NUM_STROKES,
                          duration::Float64=DEFAULT_WRITING_DURATION,
                          label::String="sim_sample",
                          sim_params::SimulationParams=DEFAULT_SIM_PARAMS,
                          stroke_pattern::StrokePattern=CURSIVE_WORD,
                          rng=Random.default_rng())
    # Layer 1: Kinematic - Generate trajectory with Sigma-LogNormal superposition
    trajectory = generate_trajectory(num_strokes, duration;
                                     rng=rng,
                                     separated=!sim_params.use_superposition,
                                     pattern=stroke_pattern,
                                     pen_lift_params=sim_params.pen_lift_params)

    # World rotation augmentation - apply BEFORE gravity projection
    # This simulates writing on tilted surfaces by rotating the entire world frame
    R_world = generate_world_rotation(sim_params.world_rotation_params; rng=rng)
    trajectory_rotated = apply_world_rotation_to_trajectory(trajectory, R_world)
    gravity_w_rotated = get_rotated_gravity(R_world)

    # Determine grip style
    grip_style = if sim_params.randomize_grip || sim_params.grip_style === nothing
        random_grip_style(rng)
    else
        sim_params.grip_style
    end

    # Determine handedness
    handedness = if sim_params.randomize_handedness || sim_params.handedness === nothing
        rand(rng) < 0.5 ? RIGHT_HAND_PARAMS : LEFT_HAND_PARAMS
    else
        sim_params.handedness
    end

    # Layer 2: Posture - Generate orientation with grip style and handedness
    # Uses rotated trajectory so orientation is relative to rotated world frame
    orientation = if sim_params.use_ik
        generate_orientation_with_style(trajectory_rotated;
                                        grip_style=grip_style,
                                        handedness=handedness,
                                        arm_params=sim_params.arm_params,
                                        joint_limits=sim_params.joint_limits,
                                        ik_params=sim_params.ik_params,
                                        rng=rng)
    else
        generate_orientation(trajectory_rotated;
                            rng=rng,
                            use_ik=false,
                            arm_params=sim_params.arm_params,
                            joint_limits=sim_params.joint_limits,
                            ik_params=sim_params.ik_params)
    end

    # Layer 3: Sensing - Generate IMU data with enhanced noise
    # Pass rotated gravity so body-frame gravity projection is correct
    static_samples = round(Int, STATIC_BUFFER_S / DT)
    sensor_result = generate_sensor_data(trajectory_rotated, orientation;
                                         static_samples=static_samples,
                                         rng=rng,
                                         use_enhanced_noise=sim_params.use_enhanced_noise,
                                         tremor_params=sim_params.tremor_params,
                                         pink_params=sim_params.pink_params,
                                         baseline_params=sim_params.baseline_params,
                                         gravity_w=gravity_w_rotated)

    # Compute metadata (use rotated trajectory for consistency)
    vel_mag = sqrt.(sum(trajectory_rotated.vel_w.^2, dims=2))[:]
    max_vel = maximum(vel_mag)

    # Compute path length (same in any frame due to rotation invariance)
    path_length = 0.0
    for i in 2:size(trajectory_rotated.pos_w, 1)
        dx = trajectory_rotated.pos_w[i, :] - trajectory_rotated.pos_w[i-1, :]
        path_length += sqrt(sum(dx.^2))
    end

    # Generation method string for metadata
    gen_method_str = sim_params.use_superposition ? "sigma_lognormal_superposition" : "sigma_lognormal_separated"

    metadata = GenerationMetadata(
        num_strokes,
        duration,
        sim_params.use_ik,
        sim_params.use_enhanced_noise,
        gen_method_str,
        !sim_params.use_superposition,  # separated strokes
        DT,
        STATIC_BUFFER_S,
        length(trajectory_rotated.lift_events),
        max_vel,
        path_length,
        string(grip_style.style),
        string(handedness.hand)
    )

    # Package as TrajectoryData (use rotated world-frame data)
    return TrajectoryData(
        sensor_result.sensor_data,
        trajectory_rotated.pos_w,
        trajectory_rotated.vel_w,
        sensor_result.gravity_b,
        trajectory_rotated.pen_down,
        size(trajectory_rotated.pos_w, 1),
        label,
        metadata
    )
end

export generate_sample

"""
    generate_samples(n::Int; kwargs...)

Generate multiple trajectory samples.

# Arguments
- `n`: Number of samples to generate
- `num_strokes`: Number of strokes per sample
- `duration`: Writing duration per sample (seconds)
- `seed`: Random seed for reproducibility
- `sim_params`: Combined simulation parameters
- `stroke_pattern`: Geometric pattern for stroke connectivity
"""
function generate_samples(n::Int; num_strokes::Int=DEFAULT_NUM_STROKES,
                          duration::Float64=DEFAULT_WRITING_DURATION,
                          seed::Int=42,
                          sim_params::SimulationParams=DEFAULT_SIM_PARAMS,
                          stroke_pattern::StrokePattern=CURSIVE_WORD)
    rng = Random.MersenneTwister(seed)
    samples = Vector{TrajectoryData}(undef, n)

    for i in 1:n
        samples[i] = generate_sample(;
            num_strokes=num_strokes,
            duration=duration,
            label="sim_sample_$(lpad(i-1, 3, '0'))",
            sim_params=sim_params,
            stroke_pattern=stroke_pattern,
            rng=rng
        )
    end

    return samples
end

export generate_samples

"""
    generate_sample_simple(; kwargs...)

Simplified sample generation using default parameters.

This is a convenience wrapper that uses non-IK, non-enhanced settings
for backward compatibility or faster generation.
"""
function generate_sample_simple(; num_strokes::Int=DEFAULT_NUM_STROKES,
                                 duration::Float64=DEFAULT_WRITING_DURATION,
                                 label::String="sim_sample",
                                 rng=Random.default_rng())
    simple_params = SimulationParams(
        DEFAULT_PEN_LIFT_PARAMS,
        true,    # use_superposition = true for smooth connections
        DEFAULT_ARM_PARAMS,
        DEFAULT_JOINT_LIMITS,
        DEFAULT_WRIST_IK_PARAMS,
        false,  # use_ik=false (simplified orientation)
        DEFAULT_GRIP_STYLE,
        DEFAULT_HANDEDNESS,
        false,  # randomize_grip
        DISABLED_WORLD_ROTATION_PARAMS,  # no world rotation for simple mode
        DEFAULT_TREMOR_PARAMS,
        DEFAULT_PINK_NOISE_PARAMS,
        DEFAULT_FSR_BASELINE_PARAMS,
        false   # use_enhanced_noise=false (basic Allan variance only)
    )

    return generate_sample(;
        num_strokes=num_strokes,
        duration=duration,
        label=label,
        sim_params=simple_params,
        rng=rng
    )
end

export generate_sample_simple

end # module
