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
Posture Layer: Wrist and pen orientation simulation with inverse kinematics.

Simulates pen/IMU orientation based on:
1. Geometric IK for elbow angle from pen position
2. Biomechanical wrist joint model (pronation, flexion, deviation)
3. Velocity-induced dynamic coupling
"""
module PostureLayer

using LinearAlgebra
using Random
using Statistics
using StaticArrays
using ..Config
using ..KinematicLayer: TrajectoryResult

export OrientationResult, generate_orientation, generate_orientation_ik
export WristJointState, solve_elbow_ik, compute_wrist_angles
export apply_handedness, apply_grip_style, random_grip_style
export generate_world_rotation, apply_world_rotation_to_trajectory

"""
    OrientationResult

Result of orientation generation.

# Fields
- `quat::Matrix{Float64}`: Quaternion [T, 4] (w, x, y, z convention)
- `omega_b::Matrix{Float64}`: Angular velocity in body frame [T, 3]
- `joint_angles::Union{Nothing, Matrix{Float64}}`: Joint angles if IK was used [T, 4]
"""
struct OrientationResult
    quat::Matrix{Float64}
    omega_b::Matrix{Float64}
    joint_angles::Union{Nothing, Matrix{Float64}}  # [elbow, pronation, flexion, deviation]
end

"""
    WristJointState

Wrist joint angles in degrees.
"""
struct WristJointState
    pronation::Float64    # Pronation (+) / Supination (-)
    flexion::Float64      # Flexion (+) / Extension (-)
    deviation::Float64    # Radial (+) / Ulnar (-)
end

"""
    euler_to_quaternion(pitch::Float64, roll::Float64, yaw::Float64)

Convert Euler angles (ZYX convention) to quaternion (w, x, y, z).
"""
function euler_to_quaternion(pitch::Float64, roll::Float64, yaw::Float64)
    cy = cos(yaw / 2)
    sy = sin(yaw / 2)
    cp = cos(pitch / 2)
    sp = sin(pitch / 2)
    cr = cos(roll / 2)
    sr = sin(roll / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]
end

"""
    quaternion_multiply(q1, q2)

Multiply two quaternions (w, x, y, z convention).
Result = q1 * q2 (composition: q2 applied first, then q1)
"""
function quaternion_multiply(q1::Vector{Float64}, q2::Vector{Float64})
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]
end

"""
    quaternion_inverse(q)

Compute the inverse of a unit quaternion.
"""
function quaternion_inverse(q::Vector{Float64})
    return [q[1], -q[2], -q[3], -q[4]]
end

"""
    quaternion_to_rotation_matrix(q)

Convert quaternion to 3x3 rotation matrix.
"""
function quaternion_to_rotation_matrix(q::Vector{Float64})
    w, x, y, z = q

    return [
        1 - 2*(y^2 + z^2)    2*(x*y - w*z)       2*(x*z + w*y);
        2*(x*y + w*z)        1 - 2*(x^2 + z^2)   2*(y*z - w*x);
        2*(x*z - w*y)        2*(y*z + w*x)       1 - 2*(x^2 + y^2)
    ]
end

"""
    axis_angle_to_quaternion(axis::Vector{Float64}, angle::Float64)

Convert axis-angle representation to quaternion.
"""
function axis_angle_to_quaternion(axis::Vector{Float64}, angle::Float64)
    axis_normalized = axis / norm(axis)
    s = sin(angle / 2)
    return [cos(angle / 2), axis_normalized[1]*s, axis_normalized[2]*s, axis_normalized[3]*s]
end

"""
    compute_angular_velocity(quat_seq::Matrix{Float64}, dt::Float64)

Compute angular velocity in body frame from quaternion sequence.
Uses finite differences of quaternion.
"""
function compute_angular_velocity(quat_seq::Matrix{Float64}, dt::Float64)
    n = size(quat_seq, 1)
    omega_b = zeros(n, 3)

    for i in 2:n-1
        q_next = quat_seq[i+1, :]
        q_curr = quat_seq[i, :]

        # q_delta = q_curr^-1 * q_next (relative rotation)
        q_curr_inv = quaternion_inverse(q_curr)
        q_delta = quaternion_multiply(q_curr_inv, q_next)

        # For small rotations: omega = 2 * [x, y, z] / dt
        # Scale factor for finite difference over 2*dt
        omega_b[i, :] = [q_delta[2], q_delta[3], q_delta[4]] / dt
    end

    omega_b[1, :] = omega_b[2, :]
    omega_b[end, :] = omega_b[end-1, :]

    return omega_b
end

"""
    solve_elbow_ik(pen_pos::Vector{Float64}, arm_params::ArmKinematicParams)

Solve for elbow angle given pen tip position using geometric IK.

Returns elbow angle in radians, or nothing if unreachable.
"""
function solve_elbow_ik(pen_pos::Vector{Float64}, arm_params::ArmKinematicParams)
    # Vector from shoulder to pen tip
    d = pen_pos - collect(arm_params.shoulder_pos)
    dist = norm(d)

    # Combined reach (upper arm + forearm + hand)
    L1 = arm_params.L_upper
    L2 = arm_params.L_forearm + arm_params.L_hand

    # Check reachability
    if dist > L1 + L2 || dist < abs(L1 - L2)
        # Clamp to reachable range
        dist = clamp(dist, abs(L1 - L2) + 0.01, L1 + L2 - 0.01)
    end

    # Law of cosines for elbow angle
    cos_elbow = (L1^2 + L2^2 - dist^2) / (2 * L1 * L2)
    cos_elbow = clamp(cos_elbow, -1.0, 1.0)

    # Elbow angle (0 = fully extended, π = fully flexed)
    elbow_angle = π - acos(cos_elbow)

    return elbow_angle
end

"""
    compute_wrist_angles(pen_pos::Vector{Float64}, pen_vel::Vector{Float64}, pen_down::Bool,
                        ik_params::WristIKParams, workspace_center::Vector{Float64})

Compute wrist joint angles based on pen position and velocity.

Biomechanical model:
- Pronation: Coupled to X velocity (writing direction)
- Flexion: Coupled to Y position (workspace coverage)
- Deviation: Coupled to Y velocity (lateral movements)
"""
function compute_wrist_angles(pen_pos::Vector{Float64}, pen_vel::Vector{Float64}, pen_down::Bool,
                              ik_params::WristIKParams, workspace_center::Vector{Float64})
    # Normalize velocity
    v_x = pen_vel[1] / ik_params.v_max
    v_y = pen_vel[2] / ik_params.v_max

    # Normalize position relative to workspace center
    y_rel = (pen_pos[2] - workspace_center[2]) / ik_params.workspace_width

    # Compute angles (in degrees)
    # Pronation: decreases with positive X velocity (writing to the right)
    pronation = ik_params.base_pronation - ik_params.k_pronate * clamp(v_x, -1.0, 1.0)

    # Flexion: increases with Y position (moving up on the page)
    flexion = ik_params.base_flexion + ik_params.k_flex * clamp(y_rel, -1.0, 1.0)

    # Deviation: coupled to Y velocity (lateral movements)
    deviation = ik_params.base_deviation + ik_params.k_dev * clamp(v_y, -1.0, 1.0)

    # Significant wrist/arm pose change when pen is lifted
    # Pen lifts involve arm retraction and wrist relaxation
    if !pen_down
        # Strong relaxation toward neutral position during lift
        relax_factor = 0.25  # Much stronger relaxation (was 0.5)

        # Add lift-specific pose changes:
        # 1. Wrist extends slightly (pen tip moves away from surface)
        lift_flexion_offset = 15.0  # degrees - wrist extends during lift

        # 2. Slight supination (natural arm retraction motion)
        lift_pronation_offset = -10.0  # degrees - supinate during lift

        # 3. Ulnar deviation as arm pulls back
        lift_deviation_offset = -8.0  # degrees

        pronation = pronation * relax_factor + (ik_params.base_pronation + lift_pronation_offset) * (1 - relax_factor)
        flexion = flexion * relax_factor + (ik_params.base_flexion + lift_flexion_offset) * (1 - relax_factor)
        deviation = deviation * relax_factor + (ik_params.base_deviation + lift_deviation_offset) * (1 - relax_factor)
    end

    return WristJointState(pronation, flexion, deviation)
end

"""
    clamp_joint_angles(state::WristJointState, elbow::Float64, limits::JointLimits)

Clamp joint angles to biomechanical limits.
"""
function clamp_joint_angles(state::WristJointState, elbow::Float64, limits::JointLimits)
    elbow_clamped = clamp(rad2deg(elbow), limits.elbow_range[1], limits.elbow_range[2])
    pronation_clamped = clamp(state.pronation, limits.pronation_range[1], limits.pronation_range[2])
    flexion_clamped = clamp(state.flexion, limits.flexion_range[1], limits.flexion_range[2])
    deviation_clamped = clamp(state.deviation, limits.deviation_range[1], limits.deviation_range[2])

    return (deg2rad(elbow_clamped), pronation_clamped, flexion_clamped, deviation_clamped)
end

"""
    joints_to_quaternion(pronation::Float64, flexion::Float64, deviation::Float64,
                        base_pitch::Float64, base_roll::Float64, base_yaw::Float64)

Convert wrist joint angles to pen orientation quaternion.

The orientation is composed of:
1. Base grip orientation (pitch, roll, yaw)
2. Wrist joint rotations (pronation, flexion, deviation)

Note: Elbow angle affects arm pose but not pen orientation directly,
so it's stored in joint_angles but not used for quaternion computation.
"""
function joints_to_quaternion(pronation::Float64, flexion::Float64, deviation::Float64,
                               base_pitch::Float64, base_roll::Float64, base_yaw::Float64)
    # Base grip orientation
    q_base = euler_to_quaternion(base_pitch, base_roll, base_yaw)

    # Convert wrist angles from degrees to radians
    pronation_rad = deg2rad(pronation)
    flexion_rad = deg2rad(flexion)
    deviation_rad = deg2rad(deviation)

    # Wrist rotation quaternions (applied in order: pronation -> flexion -> deviation)
    # Pronation/supination: rotation around forearm axis (X in body frame)
    q_pronation = axis_angle_to_quaternion([1.0, 0.0, 0.0], pronation_rad * 0.5)  # Scale for pen

    # Flexion/extension: rotation around lateral axis (Y in body frame)
    q_flexion = axis_angle_to_quaternion([0.0, 1.0, 0.0], flexion_rad * 0.5)

    # Radial/ulnar deviation: rotation around vertical axis (Z in body frame)
    q_deviation = axis_angle_to_quaternion([0.0, 0.0, 1.0], deviation_rad * 0.3)

    # Compose: q_final = q_base * q_pronation * q_flexion * q_deviation
    q_wrist = quaternion_multiply(q_pronation, quaternion_multiply(q_flexion, q_deviation))
    q_final = quaternion_multiply(q_base, q_wrist)

    # Normalize
    q_final ./= norm(q_final)

    return q_final
end

"""
    smooth_joint_trajectory!(joint_seq::Matrix{Float64}, dt::Float64, tau::Float64)

Apply exponential smoothing (low-pass filter) to joint angle sequence.
"""
function smooth_joint_trajectory!(joint_seq::Matrix{Float64}, dt::Float64, tau::Float64)
    alpha = dt / (tau + dt)

    for j in 1:size(joint_seq, 2)
        for i in 2:size(joint_seq, 1)
            joint_seq[i, j] = alpha * joint_seq[i, j] + (1 - alpha) * joint_seq[i-1, j]
        end
    end
end

"""
    generate_orientation_ik(trajectory::TrajectoryResult;
                            arm_params::ArmKinematicParams=DEFAULT_ARM_PARAMS,
                            joint_limits::JointLimits=DEFAULT_JOINT_LIMITS,
                            ik_params::WristIKParams=DEFAULT_WRIST_IK_PARAMS,
                            rng=Random.default_rng(), dt::Float64=DT)

Generate pen orientation using inverse kinematics with wrist modeling.

This is the enhanced version that computes orientation based on:
1. Geometric IK for elbow angle
2. Biomechanical wrist joint model
3. Smooth joint trajectories
"""
function generate_orientation_ik(trajectory::TrajectoryResult;
                                  arm_params::ArmKinematicParams=DEFAULT_ARM_PARAMS,
                                  joint_limits::JointLimits=DEFAULT_JOINT_LIMITS,
                                  ik_params::WristIKParams=DEFAULT_WRIST_IK_PARAMS,
                                  rng=Random.default_rng(), dt::Float64=DT)
    n = size(trajectory.pos_w, 1)
    pos_w = trajectory.pos_w
    vel_w = trajectory.vel_w
    pen_down = trajectory.pen_down

    # Random base grip orientation
    base_pitch = rand(rng) * (BASE_PITCH_RANGE[2] - BASE_PITCH_RANGE[1]) + BASE_PITCH_RANGE[1]
    base_roll = rand(rng) * (BASE_ROLL_RANGE[2] - BASE_ROLL_RANGE[1]) + BASE_ROLL_RANGE[1]
    base_yaw = rand(rng) * 2 * pi - pi

    # Compute workspace center (mean position during writing)
    writing_start = findfirst(x -> x > 0, trajectory.time)
    if writing_start === nothing
        writing_start = 1
    end
    workspace_center = vec(mean(pos_w[writing_start:end, :], dims=1))

    # Pre-allocate joint angle array [elbow, pronation, flexion, deviation]
    joint_angles = zeros(n, 4)
    quat = zeros(n, 4)

    # Slow yaw drift (random walk)
    yaw_drift = zeros(n)
    drift_noise = randn(rng, n) * 0.001
    for i in 2:n
        yaw_drift[i] = yaw_drift[i-1] + drift_noise[i]
    end

    # Compute joint angles for each timestep
    for i in 1:n
        pen_pos = pos_w[i, :]
        pen_vel = vel_w[i, :]

        # Solve elbow IK
        elbow = solve_elbow_ik(pen_pos, arm_params)

        # Compute wrist angles
        wrist_state = compute_wrist_angles(pen_pos, pen_vel, pen_down[i], ik_params, workspace_center)

        # Clamp to limits
        elbow_c, pron_c, flex_c, dev_c = clamp_joint_angles(wrist_state, elbow, joint_limits)

        joint_angles[i, :] = [elbow_c, pron_c, flex_c, dev_c]
    end

    # Smooth joint trajectories
    smooth_joint_trajectory!(joint_angles, dt, ik_params.smoothing_tau)

    # Convert to quaternions
    for i in 1:n
        _, pron, flex, dev = joint_angles[i, :]  # Elbow stored but not used for orientation
        current_yaw = base_yaw + yaw_drift[i]

        quat[i, :] = joints_to_quaternion(pron, flex, dev, base_pitch, base_roll, current_yaw)
    end

    # Compute angular velocity
    omega_b = compute_angular_velocity(quat, dt)

    return OrientationResult(quat, omega_b, joint_angles)
end

"""
    generate_orientation(trajectory::TrajectoryResult; kwargs...)

Generate pen/IMU orientation from trajectory.

Models:
1. Base grip orientation (pitch, roll)
2. Velocity-induced wobble
3. Slow arm yaw drift

This is the original simplified version. Use generate_orientation_ik for the
full biomechanical model.
"""
function generate_orientation(trajectory::TrajectoryResult;
                              rng=Random.default_rng(),
                              dt::Float64=DT,
                              use_ik::Bool=true,
                              arm_params::ArmKinematicParams=DEFAULT_ARM_PARAMS,
                              joint_limits::JointLimits=DEFAULT_JOINT_LIMITS,
                              ik_params::WristIKParams=DEFAULT_WRIST_IK_PARAMS)

    # Use IK-based orientation if requested
    if use_ik
        return generate_orientation_ik(trajectory;
                                        arm_params=arm_params,
                                        joint_limits=joint_limits,
                                        ik_params=ik_params,
                                        rng=rng, dt=dt)
    end

    # Fallback to simplified model
    n = size(trajectory.pos_w, 1)
    vel_w = trajectory.vel_w

    # Random base grip orientation
    base_pitch = rand(rng) * (BASE_PITCH_RANGE[2] - BASE_PITCH_RANGE[1]) + BASE_PITCH_RANGE[1]
    base_roll = rand(rng) * (BASE_ROLL_RANGE[2] - BASE_ROLL_RANGE[1]) + BASE_ROLL_RANGE[1]
    base_yaw = rand(rng) * 2 * pi - pi  # Random initial yaw

    base_quat = euler_to_quaternion(base_pitch, base_roll, base_yaw)

    # Initialize quaternion array
    quat = zeros(n, 4)

    # Slow yaw drift (random walk)
    yaw_drift = zeros(n)
    drift_noise = randn(rng, n) * 0.001  # ~0.06 deg/step
    for i in 2:n
        yaw_drift[i] = yaw_drift[i-1] + drift_noise[i]
    end

    # Generate orientations
    for i in 1:n
        # Velocity-induced wobble
        pitch_wobble = WRIST_PIVOT_GAIN * vel_w[i, 1]  # X velocity -> pitch
        roll_wobble = WRIST_PIVOT_GAIN * vel_w[i, 2]   # Y velocity -> roll

        # Small sinusoidal wobble for realism
        t = i * dt
        pitch_wobble += 0.02 * sin(2 * pi * 0.5 * t)
        roll_wobble += 0.02 * cos(2 * pi * 0.3 * t)

        # Delta quaternion
        delta_quat = euler_to_quaternion(pitch_wobble, roll_wobble, yaw_drift[i])

        # Compose: q_final = q_base * q_delta
        quat[i, :] = quaternion_multiply(base_quat, delta_quat)

        # Normalize quaternion
        quat[i, :] ./= norm(quat[i, :])
    end

    # Compute angular velocity
    omega_b = compute_angular_velocity(quat, dt)

    return OrientationResult(quat, omega_b, nothing)
end

# =============================================================================
# GRIP STYLE AND HANDEDNESS SUPPORT
# =============================================================================

"""
    random_grip_style(rng=Random.default_rng(); weights=nothing)

Select a random grip style, optionally with custom weights.

# Arguments
- `rng`: Random number generator
- `weights`: Optional weight vector for each style (default: equal probability)

# Returns
- `GripStyleParams`: Selected grip style parameters
"""
function random_grip_style(rng=Random.default_rng(); weights::Union{Nothing,Vector{Float64}}=nothing)
    if weights === nothing
        # Equal probability for all styles
        return ALL_GRIP_STYLES[rand(rng, 1:length(ALL_GRIP_STYLES))]
    else
        # Weighted selection
        @assert length(weights) == length(ALL_GRIP_STYLES) "Weight vector must match number of grip styles"
        cumulative = cumsum(weights) / sum(weights)
        r = rand(rng)
        for (i, threshold) in enumerate(cumulative)
            if r <= threshold
                return ALL_GRIP_STYLES[i]
            end
        end
        return ALL_GRIP_STYLES[end]
    end
end

"""
    apply_grip_style(grip::GripStyleParams, ik_params::WristIKParams, rng=Random.default_rng())

Apply grip style parameters to modify IK parameters.

# Returns
- `Tuple{Float64, Float64, Float64, WristIKParams}`: (base_pitch, base_roll, base_yaw, modified_ik_params)
"""
function apply_grip_style(grip::GripStyleParams, ik_params::WristIKParams, rng=Random.default_rng())
    # Sample base orientation from grip-specific ranges
    base_pitch = rand(rng) * (grip.base_pitch_range[2] - grip.base_pitch_range[1]) + grip.base_pitch_range[1]
    base_roll = rand(rng) * (grip.base_roll_range[2] - grip.base_roll_range[1]) + grip.base_roll_range[1]
    base_yaw = rand(rng) * 2 * pi - pi  # Random initial yaw

    # Modify IK params based on grip style
    modified_ik = WristIKParams(
        ik_params.base_pronation,
        ik_params.base_flexion,
        ik_params.base_deviation,
        ik_params.k_pronate * grip.wrist_mobility,    # Scale wrist sensitivity
        ik_params.k_flex * grip.wrist_mobility,
        ik_params.k_dev * grip.wrist_mobility,
        ik_params.v_max,
        ik_params.workspace_width,
        ik_params.smoothing_tau / grip.wrist_mobility  # Faster response for mobile grips
    )

    return (base_pitch, base_roll, base_yaw, modified_ik)
end

"""
    apply_handedness(trajectory::TrajectoryResult, handedness::HandednessParams)

Apply handedness transformation to a trajectory.

For left-handed writing:
- Mirrors X coordinates
- Adjusts shoulder position
- Maintains consistent writing direction

# Returns
- `TrajectoryResult`: Transformed trajectory
"""
function apply_handedness(trajectory::TrajectoryResult, handedness::HandednessParams)
    if handedness.hand == RIGHT_HAND
        return trajectory  # No transformation needed
    end

    # Mirror X coordinates for left hand
    n = size(trajectory.pos_w, 1)
    pos_w_new = copy(trajectory.pos_w)
    vel_w_new = copy(trajectory.vel_w)
    accel_w_new = copy(trajectory.accel_w)

    if handedness.mirror_x
        # Mirror around center of trajectory
        x_center = (maximum(trajectory.pos_w[:, 1]) + minimum(trajectory.pos_w[:, 1])) / 2

        for i in 1:n
            # Mirror X position around center
            pos_w_new[i, 1] = 2 * x_center - trajectory.pos_w[i, 1]
            # Reverse X velocity and acceleration
            vel_w_new[i, 1] = -trajectory.vel_w[i, 1]
            accel_w_new[i, 1] = -trajectory.accel_w[i, 1]
        end
    end

    return TrajectoryResult(
        pos_w_new,
        vel_w_new,
        accel_w_new,
        trajectory.time,
        trajectory.pen_down,
        trajectory.lift_events
    )
end

"""
    generate_orientation_with_style(trajectory::TrajectoryResult;
                                    grip_style::GripStyleParams=DEFAULT_GRIP_STYLE,
                                    handedness::HandednessParams=DEFAULT_HANDEDNESS,
                                    arm_params::ArmKinematicParams=DEFAULT_ARM_PARAMS,
                                    joint_limits::JointLimits=DEFAULT_JOINT_LIMITS,
                                    ik_params::WristIKParams=DEFAULT_WRIST_IK_PARAMS,
                                    rng=Random.default_rng(),
                                    dt::Float64=DT)

Generate pen orientation with grip style and handedness support.

This is the full-featured version that supports:
1. Different grip styles (tripod, lateral, quadrupod, etc.)
2. Left/right handedness
3. Biomechanical wrist model
"""
function generate_orientation_with_style(trajectory::TrajectoryResult;
                                          grip_style::GripStyleParams=DEFAULT_GRIP_STYLE,
                                          handedness::HandednessParams=DEFAULT_HANDEDNESS,
                                          arm_params::ArmKinematicParams=DEFAULT_ARM_PARAMS,
                                          joint_limits::JointLimits=DEFAULT_JOINT_LIMITS,
                                          ik_params::WristIKParams=DEFAULT_WRIST_IK_PARAMS,
                                          rng=Random.default_rng(),
                                          dt::Float64=DT)

    # Apply handedness transformation to trajectory
    traj = apply_handedness(trajectory, handedness)

    # Modify arm params for handedness
    shoulder_pos = if handedness.hand == LEFT_HAND
        SVector{3,Float64}(
            arm_params.shoulder_pos[1],
            arm_params.shoulder_pos[2] + handedness.shoulder_offset,
            arm_params.shoulder_pos[3]
        )
    else
        arm_params.shoulder_pos
    end

    modified_arm_params = ArmKinematicParams(
        arm_params.L_upper,
        arm_params.L_forearm,
        arm_params.L_hand,
        shoulder_pos
    )

    # Apply grip style to get base orientation and modified IK params
    base_pitch, base_roll, base_yaw, modified_ik = apply_grip_style(grip_style, ik_params, rng)

    # Adjust pronation for handedness
    pronation_offset = handedness.pronation_offset

    # Generate orientation using IK
    n = size(traj.pos_w, 1)
    pos_w = traj.pos_w
    vel_w = traj.vel_w
    pen_down = traj.pen_down

    # Compute workspace center (mean position during writing)
    writing_start = findfirst(x -> x > 0, traj.time)
    if writing_start === nothing
        writing_start = 1
    end
    workspace_center = vec(mean(pos_w[writing_start:end, :], dims=1))

    # Pre-allocate joint angle array [elbow, pronation, flexion, deviation]
    joint_angles = zeros(n, 4)
    quat = zeros(n, 4)

    # Slow yaw drift (random walk)
    yaw_drift = zeros(n)
    drift_noise = randn(rng, n) * 0.001
    for i in 2:n
        yaw_drift[i] = yaw_drift[i-1] + drift_noise[i]
    end

    # Compute joint angles for each timestep
    for i in 1:n
        pen_pos = pos_w[i, :]
        pen_vel = vel_w[i, :]

        # Apply writing direction for handedness
        pen_vel_adjusted = copy(pen_vel)
        pen_vel_adjusted[1] *= handedness.writing_direction

        # Solve elbow IK
        elbow = solve_elbow_ik(pen_pos, modified_arm_params)

        # Compute wrist angles
        wrist_state = compute_wrist_angles(pen_pos, pen_vel_adjusted, pen_down[i], modified_ik, workspace_center)

        # Apply handedness pronation offset
        adjusted_pronation = wrist_state.pronation + pronation_offset

        # Create adjusted wrist state
        adjusted_wrist = WristJointState(adjusted_pronation, wrist_state.flexion, wrist_state.deviation)

        # Clamp to limits
        elbow_c, pron_c, flex_c, dev_c = clamp_joint_angles(adjusted_wrist, elbow, joint_limits)

        joint_angles[i, :] = [elbow_c, pron_c, flex_c, dev_c]
    end

    # Smooth joint trajectories
    smooth_joint_trajectory!(joint_angles, dt, modified_ik.smoothing_tau)

    # Convert to quaternions
    for i in 1:n
        _, pron, flex, dev = joint_angles[i, :]
        current_yaw = base_yaw + yaw_drift[i]

        quat[i, :] = joints_to_quaternion(pron, flex, dev, base_pitch, base_roll, current_yaw)
    end

    # Compute angular velocity
    omega_b = compute_angular_velocity(quat, dt)

    return OrientationResult(quat, omega_b, joint_angles)
end

export generate_orientation_with_style

# =============================================================================
# WORLD ROTATION AUGMENTATION
# =============================================================================

"""
    generate_world_rotation(params::WorldRotationParams; rng=Random.default_rng())

Generate a random SO(3) rotation matrix for world-frame augmentation.

This simulates writing on tilted surfaces by rotating the entire world frame.
The rotation is composed of:
1. Random tilt from vertical (0 to max_tilt)
2. Random tilt azimuth direction (0 to 2π)
3. Random yaw rotation (0 to 2π if full_yaw is true)

Returns a 3x3 rotation matrix R_world such that:
- pos_rotated = R_world * pos
- gravity_rotated = R_world * gravity
"""
function generate_world_rotation(params::WorldRotationParams; rng=Random.default_rng())
    if !params.enabled
        return Matrix{Float64}(I, 3, 3)  # Identity matrix
    end

    # Random tilt magnitude (0 to max_tilt)
    tilt = rand(rng) * params.max_tilt

    # Random tilt direction (azimuth around vertical)
    tilt_azimuth = rand(rng) * 2 * π

    # Random yaw rotation
    yaw = params.full_yaw ? rand(rng) * 2 * π : 0.0

    # Build tilt rotation using axis-angle (Rodrigues formula)
    # Tilt axis is in XY plane: [cos(azimuth), sin(azimuth), 0]
    tilt_axis = [cos(tilt_azimuth), sin(tilt_azimuth), 0.0]

    # Rodrigues formula for tilt rotation
    if tilt > 1e-8
        K_tilt = [
            0.0 -tilt_axis[3] tilt_axis[2];
            tilt_axis[3] 0.0 -tilt_axis[1];
            -tilt_axis[2] tilt_axis[1] 0.0
        ]
        R_tilt = I + sin(tilt) * K_tilt + (1 - cos(tilt)) * (K_tilt * K_tilt)
    else
        R_tilt = Matrix{Float64}(I, 3, 3)
    end

    # Yaw rotation around Z-axis
    c, s = cos(yaw), sin(yaw)
    R_yaw = [
        c -s 0.0;
        s c 0.0;
        0.0 0.0 1.0
    ]

    # Combined rotation: first tilt, then yaw
    return R_yaw * R_tilt
end

"""
    apply_world_rotation_to_trajectory(trajectory::TrajectoryResult, R_world::Matrix{Float64})

Apply world rotation to a trajectory, transforming all world-frame vectors.

This rotates:
- pos_w: World position
- vel_w: World velocity
- accel_w: World acceleration

The gravity vector will be correctly handled when projected to body frame in SensingLayer,
since gravity is defined as [0, 0, -g] in the rotated world frame.

Returns a new TrajectoryResult with rotated world-frame data.
"""
function apply_world_rotation_to_trajectory(trajectory::TrajectoryResult, R_world::Matrix{Float64})
    n = size(trajectory.pos_w, 1)

    # Allocate rotated arrays
    pos_w_rot = zeros(n, 3)
    vel_w_rot = zeros(n, 3)
    accel_w_rot = zeros(n, 3)

    # Apply rotation to each timestep
    for i in 1:n
        pos_w_rot[i, :] = R_world * trajectory.pos_w[i, :]
        vel_w_rot[i, :] = R_world * trajectory.vel_w[i, :]
        accel_w_rot[i, :] = R_world * trajectory.accel_w[i, :]
    end

    return TrajectoryResult(
        pos_w_rot,
        vel_w_rot,
        accel_w_rot,
        trajectory.time,
        trajectory.pen_down,
        trajectory.lift_events
    )
end

"""
    get_rotated_gravity(R_world::Matrix{Float64})

Get the gravity vector in the rotated world frame.

Since gravity is always [0, 0, -g] in the original world frame,
in the rotated frame it becomes R_world * [0, 0, -g].

This is needed by the SensingLayer to correctly project gravity to body frame.
"""
function get_rotated_gravity(R_world::Matrix{Float64})
    g_original = [0.0, 0.0, -GRAVITY_MAGNITUDE]
    return R_world * g_original
end

export get_rotated_gravity

end # module
