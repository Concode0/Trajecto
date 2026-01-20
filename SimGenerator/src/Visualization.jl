"""
6DOF Motion Visualization for SimGenerator sanity checks.

Provides comprehensive visualization of generated trajectory data:
- 3D trajectory plot
- Orientation (quaternion to Euler angles)
- IMU time series (accelerometer, gyroscope)
- Force sensor (FSR) plot
- Pen lift events

Usage:
    using .Visualization
    visualize_sample(sample)
    visualize_6dof(sample, save_path="output.png")
"""
module Visualization

using Plots
using LinearAlgebra: norm
using Printf

export visualize_sample, visualize_6dof, visualize_imu, visualize_trajectory_3d
export quat_to_euler, save_visualization

"""
    quat_to_euler(quat::Vector{Float64}) -> (roll, pitch, yaw)

Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.
Uses ZYX convention (yaw-pitch-roll).
"""
function quat_to_euler(quat::AbstractVector{<:Real})
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x^2 + y^2)
    roll = atan(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = if abs(sinp) >= 1
        copysign(pi/2, sinp)  # Use 90 degrees if out of range
    else
        asin(sinp)
    end

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y^2 + z^2)
    yaw = atan(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
end

"""
    visualize_trajectory_3d(sample; title="3D Trajectory")

Create a 3D plot of the trajectory.
"""
function visualize_trajectory_3d(sample; title="3D Trajectory", show_pen_lift=true)
    pos = sample.gt_pos_data
    T = sample.sequence_length

    # Convert to cm for better readability
    x = pos[1:T, 1] * 100
    y = pos[1:T, 2] * 100
    z = pos[1:T, 3] * 100

    p = plot3d(x, y, z,
        xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)",
        title=title,
        linewidth=2,
        color=:viridis,
        line_z=1:T,
        colorbar_title="Time (samples)",
        legend=false
    )

    # Mark start and end
    scatter3d!([x[1]], [y[1]], [z[1]], markersize=8, color=:green, label="Start")
    scatter3d!([x[end]], [y[end]], [z[end]], markersize=8, color=:red, label="End")

    # Show pen lift events if available
    if show_pen_lift && hasfield(typeof(sample), :pen_down) && sample.pen_down !== nothing
        pen_up_indices = findall(.!sample.pen_down[1:T])
        if !isempty(pen_up_indices)
            scatter3d!(x[pen_up_indices], y[pen_up_indices], z[pen_up_indices],
                markersize=2, alpha=0.3, color=:gray, label="Pen Up")
        end
    end

    return p
end

"""
    visualize_imu(sample; title_prefix="")

Create IMU time series plots (accelerometer and gyroscope).
"""
function visualize_imu(sample; title_prefix="")
    sensor = sample.sensor_data
    T = sample.sequence_length
    dt = hasfield(typeof(sample), :metadata) && sample.metadata !== nothing ?
         sample.metadata.dt : 0.02

    t = (0:T-1) * dt

    # Accelerometer
    p_accel = plot(t, sensor[1:T, 1:3],
        xlabel="Time (s)", ylabel="Acceleration (m/s^2)",
        title="$(title_prefix)Accelerometer",
        label=["X" "Y" "Z"],
        linewidth=1.5,
        legend=:topright
    )

    # Gyroscope
    p_gyro = plot(t, rad2deg.(sensor[1:T, 4:6]),
        xlabel="Time (s)", ylabel="Angular Velocity (deg/s)",
        title="$(title_prefix)Gyroscope",
        label=["X" "Y" "Z"],
        linewidth=1.5,
        legend=:topright
    )

    # FSR
    p_fsr = plot(t, sensor[1:T, 7],
        xlabel="Time (s)", ylabel="Force (normalized)",
        title="$(title_prefix)Force Sensor (FSR)",
        linewidth=1.5,
        color=:purple,
        legend=false,
        ylims=(0, max(1.5, maximum(sensor[1:T, 7]) * 1.1))
    )

    return p_accel, p_gyro, p_fsr
end

"""
    visualize_orientation(sample; title_prefix="")

Create orientation visualization (Euler angles from quaternion).
"""
function visualize_orientation(sample; title_prefix="")
    T = sample.sequence_length
    dt = hasfield(typeof(sample), :metadata) && sample.metadata !== nothing ?
         sample.metadata.dt : 0.02

    t = (0:T-1) * dt

    # Check if we have orientation data (from OrientationResult embedded in sensor generation)
    # For now, compute from gravity_b using gravity alignment assumption
    gravity_b = sample.gt_gravity_b_data[1:T, :]

    # Estimate orientation from gravity vector
    # When pen is level: gravity_b ≈ [0, 0, -1]
    # Pitch: rotation around Y axis (nose up/down)
    # Roll: rotation around X axis (bank left/right)

    pitch = asin.(clamp.(gravity_b[:, 1], -1, 1))  # Forward tilt
    roll = atan.(-gravity_b[:, 2], -gravity_b[:, 3])  # Side tilt

    # Estimate yaw from angular velocity integration (rough approximation)
    # Note: This is simplified; actual yaw requires full quaternion tracking
    gyro_z = sample.sensor_data[1:T, 6]
    yaw = cumsum(gyro_z) * dt

    p_orient = plot(t, [rad2deg.(roll) rad2deg.(pitch) rad2deg.(yaw)],
        xlabel="Time (s)", ylabel="Angle (deg)",
        title="$(title_prefix)Estimated Orientation",
        label=["Roll" "Pitch" "Yaw"],
        linewidth=1.5,
        legend=:topright
    )

    return p_orient
end

"""
    visualize_velocity(sample; title_prefix="")

Create velocity visualization (ground truth and magnitude).
"""
function visualize_velocity(sample; title_prefix="")
    vel = sample.gt_vel_data
    T = sample.sequence_length
    dt = hasfield(typeof(sample), :metadata) && sample.metadata !== nothing ?
         sample.metadata.dt : 0.02

    t = (0:T-1) * dt

    # Velocity components
    p_vel = plot(t, vel[1:T, :] * 100,  # Convert to cm/s
        xlabel="Time (s)", ylabel="Velocity (cm/s)",
        title="$(title_prefix)Velocity",
        label=["Vx" "Vy" "Vz"],
        linewidth=1.5,
        legend=:topright
    )

    # Velocity magnitude
    vel_mag = [norm(vel[i, :]) for i in 1:T] * 100
    p_mag = plot(t, vel_mag,
        xlabel="Time (s)", ylabel="Speed (cm/s)",
        title="$(title_prefix)Speed Magnitude",
        linewidth=1.5,
        color=:orange,
        legend=false,
        fill=0, alpha=0.3
    )

    return p_vel, p_mag
end

"""
    visualize_pen_state(sample; title_prefix="")

Visualize pen up/down state and Z-axis motion.
"""
function visualize_pen_state(sample; title_prefix="")
    pos = sample.gt_pos_data
    T = sample.sequence_length
    dt = hasfield(typeof(sample), :metadata) && sample.metadata !== nothing ?
         sample.metadata.dt : 0.02

    t = (0:T-1) * dt

    # Z-axis position (pen height)
    z_mm = pos[1:T, 3] * 1000  # Convert to mm

    p_z = plot(t, z_mm,
        xlabel="Time (s)", ylabel="Height (mm)",
        title="$(title_prefix)Pen Height (Z-axis)",
        linewidth=1.5,
        color=:teal,
        legend=false,
        fill=0, alpha=0.3
    )

    # Pen down state if available
    if hasfield(typeof(sample), :pen_down) && sample.pen_down !== nothing
        pen_down = Float64.(sample.pen_down[1:T])

        # Overlay pen state
        p_state = twinx()
        plot!(p_state, t, pen_down,
            ylabel="Pen Down",
            linewidth=1,
            color=:red,
            alpha=0.5,
            linestyle=:dash,
            legend=false
        )
    end

    return p_z
end

"""
    visualize_6dof(sample; save_path=nothing, title="6DOF Motion Visualization")

Create comprehensive 6DOF visualization with all sensor and trajectory data.

# Arguments
- `sample`: TrajectoryData from SimGenerator
- `save_path`: Optional path to save the figure
- `title`: Main title for the figure

# Returns
- Combined plot object
"""
function visualize_6dof(sample; save_path::Union{String,Nothing}=nothing, title="6DOF Motion Visualization")
    # Get metadata for title
    meta_str = ""
    if hasfield(typeof(sample), :metadata) && sample.metadata !== nothing
        meta = sample.metadata
        meta_str = " | $(meta.num_strokes) strokes, $(meta.duration)s, $(meta.grip_style)"
    end

    # Create individual plots
    p_3d = visualize_trajectory_3d(sample, title="3D Trajectory")
    p_accel, p_gyro, p_fsr = visualize_imu(sample)
    p_orient = visualize_orientation(sample)
    p_vel, p_mag = visualize_velocity(sample)
    p_z = visualize_pen_state(sample)

    # XY trajectory (2D top view)
    pos = sample.gt_pos_data
    T = sample.sequence_length
    x_cm, y_cm = pos[1:T, 1] * 100, pos[1:T, 2] * 100

    p_xy = plot(x_cm, y_cm,
        xlabel="X (cm)", ylabel="Y (cm)",
        title="XY Trajectory (Top View)",
        linewidth=1.5,
        color=:blue,
        aspect_ratio=:equal,
        legend=false
    )
    scatter!([x_cm[1]], [y_cm[1]], markersize=6, color=:green, label="Start")
    scatter!([x_cm[end]], [y_cm[end]], markersize=6, color=:red, label="End")

    # Combine into layout
    layout = @layout [
        a{0.4w} [b; c]
        d e f
        g h i
    ]

    combined = plot(
        p_3d, p_xy, p_z,
        p_accel, p_gyro, p_fsr,
        p_vel, p_mag, p_orient,
        layout=(3, 3),
        size=(1800, 1200),
        plot_title="$title$meta_str",
        margin=5Plots.mm
    )

    if save_path !== nothing
        savefig(combined, save_path)
        println("Saved visualization to: $save_path")
    end

    return combined
end

"""
    visualize_sample(sample; save_path=nothing)

Convenience function for quick visualization of a single sample.
Alias for visualize_6dof with default settings.
"""
visualize_sample(sample; kwargs...) = visualize_6dof(sample; kwargs...)

"""
    save_visualization(plot_obj, path::String)

Save a plot to file.
"""
function save_visualization(plot_obj, path::String)
    savefig(plot_obj, path)
    println("Saved: $path")
end

"""
    visualize_batch_summary(samples::Vector; save_path=nothing)

Create a summary visualization for a batch of samples.
Shows distribution of key metrics.
"""
function visualize_batch_summary(samples::Vector; save_path::Union{String,Nothing}=nothing)
    n = length(samples)

    # Collect metrics
    path_lengths = Float64[]
    max_vels = Float64[]
    durations = Float64[]
    stroke_counts = Int[]

    for sample in samples
        T = sample.sequence_length
        pos = sample.gt_pos_data
        vel = sample.gt_vel_data

        # Path length
        path_len = sum(norm(pos[i, :] - pos[i-1, :]) for i in 2:T)
        push!(path_lengths, path_len * 100)  # cm

        # Max velocity
        max_vel = maximum(norm(vel[i, :]) for i in 1:T)
        push!(max_vels, max_vel * 100)  # cm/s

        # From metadata
        if sample.metadata !== nothing
            push!(durations, sample.metadata.duration)
            push!(stroke_counts, sample.metadata.num_strokes)
        end
    end

    # Create histograms
    p1 = histogram(path_lengths, xlabel="Path Length (cm)", ylabel="Count",
                   title="Path Length Distribution", legend=false, bins=20)

    p2 = histogram(max_vels, xlabel="Max Velocity (cm/s)", ylabel="Count",
                   title="Max Velocity Distribution", legend=false, bins=20)

    p3 = if !isempty(durations)
        histogram(durations, xlabel="Duration (s)", ylabel="Count",
                  title="Duration Distribution", legend=false, bins=20)
    else
        plot(title="Duration (N/A)")
    end

    p4 = if !isempty(stroke_counts)
        histogram(stroke_counts, xlabel="Stroke Count", ylabel="Count",
                  title="Stroke Count Distribution", legend=false, bins=maximum(stroke_counts))
    else
        plot(title="Stroke Count (N/A)")
    end

    combined = plot(p1, p2, p3, p4,
                    layout=(2, 2),
                    size=(1000, 800),
                    plot_title="Batch Summary ($n samples)")

    if save_path !== nothing
        savefig(combined, save_path)
        println("Saved batch summary to: $save_path")
    end

    return combined
end

export visualize_batch_summary

end # module
