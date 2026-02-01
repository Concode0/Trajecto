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
DataStatistics: Statistics computation for SimGenerator output.

Computes statistics needed for model/config.py:
- Velocity statistics (mean, std, L2 norm)
- Sensor statistics (accelerometer, gyroscope, FSR)
- Trajectory metrics (path length, duration)

Usage:
    using .DataStatistics
    stats = compute_dataset_statistics(samples)
    print_config_update(stats)
"""
module DataStatistics

using Statistics: mean, std, quantile
using LinearAlgebra: norm
using Printf
using HDF5

export DatasetStatistics, compute_dataset_statistics, compute_statistics_from_hdf5
export print_config_update, print_statistics_summary

"""
    DatasetStatistics

Container for computed statistics from generated data.
"""
struct DatasetStatistics
    # Velocity statistics
    vel_mean::Vector{Float64}
    vel_std::Vector{Float64}
    vel_std_l2::Float64
    vel_mag_mean::Float64
    vel_mag_std::Float64
    vel_mag_max::Float64
    vel_mag_95th::Float64

    # Accelerometer statistics
    accel_mean::Vector{Float64}
    accel_std::Vector{Float64}

    # Gyroscope statistics
    gyro_mean::Vector{Float64}
    gyro_std::Vector{Float64}

    # FSR statistics
    fsr_mean::Float64
    fsr_std::Float64

    # Gravity in body frame
    gravity_b_mean::Vector{Float64}
    gravity_b_std::Vector{Float64}

    # Dataset info
    num_samples::Int
    total_timesteps::Int
    total_path_length::Float64
    avg_path_length::Float64
    avg_duration::Float64
end

"""
    compute_dataset_statistics(samples::Vector{TrajectoryData}) -> DatasetStatistics

Compute comprehensive statistics from generated trajectory samples.

# Arguments
- `samples`: Vector of TrajectoryData from SimGenerator

# Returns
- DatasetStatistics struct with all computed statistics
"""
function compute_dataset_statistics(samples::Vector)
    # Collect all data
    all_vels = Vector{Float64}[]
    all_accels = Vector{Float64}[]
    all_gyros = Vector{Float64}[]
    all_fsr = Float64[]
    all_gravity_b = Vector{Float64}[]

    total_path_length = 0.0
    total_timesteps = 0
    total_duration = 0.0

    for sample in samples
        T = sample.sequence_length

        # Velocity (ground truth)
        for t in 1:T
            push!(all_vels, sample.gt_vel_data[t, :])
        end

        # Sensor data: [accel(3), gyro(3), fsr(1)]
        for t in 1:T
            push!(all_accels, sample.sensor_data[t, 1:3])
            push!(all_gyros, sample.sensor_data[t, 4:6])
            push!(all_fsr, sample.sensor_data[t, 7])
        end

        # Gravity in body frame
        for t in 1:T
            push!(all_gravity_b, sample.gt_gravity_b_data[t, :])
        end

        # Path length
        for t in 2:T
            dx = sample.gt_pos_data[t, :] - sample.gt_pos_data[t-1, :]
            total_path_length += norm(dx)
        end

        total_timesteps += T

        # Duration from metadata if available
        if sample.metadata !== nothing
            total_duration += sample.metadata.duration
        end
    end

    # Convert to matrices for easier computation
    vel_mat = hcat(all_vels...)'  # [N, 3]
    accel_mat = hcat(all_accels...)'
    gyro_mat = hcat(all_gyros...)'
    gravity_mat = hcat(all_gravity_b...)'

    # Velocity statistics
    vel_mean = vec(mean(vel_mat, dims=1))
    vel_std = vec(std(vel_mat, dims=1))
    vel_std_l2 = sqrt(sum(vel_std.^2))

    vel_mag = [norm(vel_mat[i, :]) for i in 1:size(vel_mat, 1)]
    vel_mag_mean = mean(vel_mag)
    vel_mag_std = std(vel_mag)
    vel_mag_max = maximum(vel_mag)
    vel_mag_95th = quantile(vel_mag, 0.95)

    # Accelerometer statistics
    accel_mean = vec(mean(accel_mat, dims=1))
    accel_std = vec(std(accel_mat, dims=1))

    # Gyroscope statistics
    gyro_mean = vec(mean(gyro_mat, dims=1))
    gyro_std = vec(std(gyro_mat, dims=1))

    # FSR statistics
    fsr_mean = mean(all_fsr)
    fsr_std = std(all_fsr)

    # Gravity statistics
    gravity_b_mean = vec(mean(gravity_mat, dims=1))
    gravity_b_std = vec(std(gravity_mat, dims=1))

    return DatasetStatistics(
        vel_mean, vel_std, vel_std_l2,
        vel_mag_mean, vel_mag_std, vel_mag_max, vel_mag_95th,
        accel_mean, accel_std,
        gyro_mean, gyro_std,
        fsr_mean, fsr_std,
        gravity_b_mean, gravity_b_std,
        length(samples), total_timesteps, total_path_length,
        total_path_length / length(samples),
        total_duration / length(samples)
    )
end

"""
    compute_statistics_from_hdf5(path::String) -> DatasetStatistics

Compute statistics directly from an HDF5 file.
"""
function compute_statistics_from_hdf5(path::String)
    h5open(path, "r") do f
        sample_keys = filter(k -> startswith(k, "sim_sample"), keys(f))

        all_vels = Vector{Float64}[]
        all_accels = Vector{Float64}[]
        all_gyros = Vector{Float64}[]
        all_fsr = Float64[]
        all_gravity_b = Vector{Float64}[]

        total_path_length = 0.0
        total_timesteps = 0

        for key in sample_keys
            grp = f[key]

            # Read data (stored as [F, T], need to transpose)
            sensor = read(grp["sensor_data"])'  # [T, 7]
            vel = read(grp["gt_vel_data"])'     # [T, 3]
            pos = read(grp["gt_pos_data"])'     # [T, 3]

            seq_len = read_attribute(grp, "sequence_length")

            for t in 1:seq_len
                push!(all_vels, vel[t, :])
                push!(all_accels, sensor[t, 1:3])
                push!(all_gyros, sensor[t, 4:6])
                push!(all_fsr, sensor[t, 7])
            end

            if haskey(grp, "gt_gravity_b_data")
                gravity_b = read(grp["gt_gravity_b_data"])'
                for t in 1:seq_len
                    push!(all_gravity_b, gravity_b[t, :])
                end
            end

            # Path length
            for t in 2:seq_len
                dx = pos[t, :] - pos[t-1, :]
                total_path_length += norm(dx)
            end

            total_timesteps += seq_len
        end

        # Compute statistics (same as above)
        vel_mat = hcat(all_vels...)'
        accel_mat = hcat(all_accels...)'
        gyro_mat = hcat(all_gyros...)'

        vel_mean = vec(mean(vel_mat, dims=1))
        vel_std = vec(std(vel_mat, dims=1))
        vel_std_l2 = sqrt(sum(vel_std.^2))

        vel_mag = [norm(vel_mat[i, :]) for i in 1:size(vel_mat, 1)]
        vel_mag_mean = mean(vel_mag)
        vel_mag_std = std(vel_mag)
        vel_mag_max = maximum(vel_mag)
        vel_mag_95th = quantile(vel_mag, 0.95)

        accel_mean = vec(mean(accel_mat, dims=1))
        accel_std = vec(std(accel_mat, dims=1))

        gyro_mean = vec(mean(gyro_mat, dims=1))
        gyro_std = vec(std(gyro_mat, dims=1))

        fsr_mean = mean(all_fsr)
        fsr_std = std(all_fsr)

        gravity_b_mean = if !isempty(all_gravity_b)
            gravity_mat = hcat(all_gravity_b...)'
            vec(mean(gravity_mat, dims=1))
        else
            [0.0, 0.0, 0.0]
        end

        gravity_b_std = if !isempty(all_gravity_b)
            gravity_mat = hcat(all_gravity_b...)'
            vec(std(gravity_mat, dims=1))
        else
            [0.0, 0.0, 0.0]
        end

        return DatasetStatistics(
            vel_mean, vel_std, vel_std_l2,
            vel_mag_mean, vel_mag_std, vel_mag_max, vel_mag_95th,
            accel_mean, accel_std,
            gyro_mean, gyro_std,
            fsr_mean, fsr_std,
            gravity_b_mean, gravity_b_std,
            length(sample_keys), total_timesteps, total_path_length,
            total_path_length / length(sample_keys),
            0.0  # avg_duration not available from HDF5
        )
    end
end

"""
    print_statistics_summary(stats::DatasetStatistics)

Print a human-readable summary of the statistics.
"""
function print_statistics_summary(stats::DatasetStatistics)
    println("=" ^ 70)
    println("DATASET STATISTICS SUMMARY")
    println("=" ^ 70)

    println("\nVelocity Statistics:")
    @printf("  Mean: [%.4f, %.4f, %.4f] m/s\n", stats.vel_mean...)
    @printf("  Std:  [%.4f, %.4f, %.4f] m/s\n", stats.vel_std...)
    @printf("  Std L2 norm: %.6f m/s\n", stats.vel_std_l2)
    @printf("  Magnitude: mean=%.4f, std=%.4f, max=%.4f, 95th=%.4f m/s\n",
            stats.vel_mag_mean, stats.vel_mag_std, stats.vel_mag_max, stats.vel_mag_95th)

    println("\nAccelerometer Statistics:")
    @printf("  Mean: [%.4f, %.4f, %.4f] m/s^2\n", stats.accel_mean...)
    @printf("  Std:  [%.4f, %.4f, %.4f] m/s^2\n", stats.accel_std...)

    println("\nGyroscope Statistics:")
    @printf("  Mean: [%.6f, %.6f, %.6f] rad/s\n", stats.gyro_mean...)
    @printf("  Std:  [%.6f, %.6f, %.6f] rad/s\n", stats.gyro_std...)

    println("\nFSR Statistics:")
    @printf("  Mean: %.4f, Std: %.4f\n", stats.fsr_mean, stats.fsr_std)

    println("\nGravity (Body Frame) Statistics:")
    @printf("  Mean: [%.4f, %.4f, %.4f]\n", stats.gravity_b_mean...)
    @printf("  Std:  [%.4f, %.4f, %.4f]\n", stats.gravity_b_std...)

    println("\nDataset Summary:")
    @printf("  Samples: %d\n", stats.num_samples)
    @printf("  Total timesteps: %d\n", stats.total_timesteps)
    @printf("  Total path length: %.2f m\n", stats.total_path_length)
    @printf("  Avg path length: %.4f m\n", stats.avg_path_length)
    if stats.avg_duration > 0
        @printf("  Avg duration: %.2f s\n", stats.avg_duration)
    end
end

"""
    print_config_update(stats::DatasetStatistics)

Print Python code to update model/config.py with computed statistics.
"""
function print_config_update(stats::DatasetStatistics)
    println("\n" * "=" ^ 70)
    println("PYTHON CONFIG UPDATE CODE (paste into model/config.py)")
    println("=" ^ 70)

    println("""
# =============================================================================
# COMPUTED STATISTICS FROM SIMULATED DATA (SimGenerator)
# =============================================================================
# Generated by: julia --project=. main.jl --samples N --stats
# Samples: $(stats.num_samples), Timesteps: $(stats.total_timesteps)
# =============================================================================

# --- Velocity Statistics (from ground truth) ---
VEL_STD_L2 = $(stats.vel_std_l2)  # m/s (L2 norm of per-axis std)
VEL_MAG_95TH = $(stats.vel_mag_95th)  # m/s (95th percentile magnitude)

# --- Sensor Statistics (for normalization reference) ---
# Accelerometer
SIM_ACCEL_MEAN = [$(stats.accel_mean[1]), $(stats.accel_mean[2]), $(stats.accel_mean[3])]  # m/s^2
SIM_ACCEL_STD = [$(stats.accel_std[1]), $(stats.accel_std[2]), $(stats.accel_std[3])]  # m/s^2

# Gyroscope
SIM_GYRO_MEAN = [$(stats.gyro_mean[1]), $(stats.gyro_mean[2]), $(stats.gyro_mean[3])]  # rad/s
SIM_GYRO_STD = [$(stats.gyro_std[1]), $(stats.gyro_std[2]), $(stats.gyro_std[3])]  # rad/s

# FSR
SIM_FSR_MEAN = $(stats.fsr_mean)
SIM_FSR_STD = $(stats.fsr_std)
""")
end

end # module
