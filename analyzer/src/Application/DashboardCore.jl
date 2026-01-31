# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: This software is protected under the following ROK Patent Applications:
# 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
# 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
#
# Commercial use or redistribution of the core logic requires a separate license.
# For inquiries, contact: nemonanconcode@gmail.com

"""
Core types and utilities for the Trajecto Dashboard.

This module provides the base MakieDashboard type and shared utility functions
used across all dashboard visualizations.
"""
module DashboardCore

using ..Config
using GeometryBasics
using LinearAlgebra

"""
    MakieDashboard

Dashboard application for visualizing trajectory analysis results.

# Fields
- `window_size::Tuple{Int, Int}`: Window resolution (width, height) in pixels

# Example
```julia
app = MakieDashboard()  # Uses default size (1600, 1000)
app = MakieDashboard((1920, 1080))  # Custom size
```
"""
struct MakieDashboard
    window_size::Tuple{Int, Int}
end

"""
    MakieDashboard()

Create a MakieDashboard with default window size from Config.DEFAULT_WINDOW_SIZE.
"""
MakieDashboard() = MakieDashboard(Config.DEFAULT_WINDOW_SIZE)


"""
    calculate_ellipsoid_model(cov_3x3::Matrix, pos_3d::Vector) -> Mat4f

Calculate a 4x4 transformation matrix for rendering a 3-sigma uncertainty ellipsoid.

Uses eigen decomposition of the covariance matrix to determine orientation and scale.
The resulting transformation can be directly applied to a unit sphere mesh.

# Arguments
- `cov_3x3::Matrix`: 3×3 position covariance matrix
- `pos_3d::Vector`: 3D position center point

# Returns
- `Mat4f`: 4×4 transformation matrix combining translation, rotation, and scaling

# Mathematical Details
The transformation is computed as T × R × S where:
- S: Diagonal scaling by 3σ (3 × sqrt(eigenvalues))
- R: Rotation matrix from eigenvectors
- T: Translation to position center

Eigenvalues are clamped to MIN_EIGENVALUE to avoid numerical instability.

# Example
```julia
cov = [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]
pos = [1.0, 2.0, 3.0]
transform = calculate_ellipsoid_model(cov, pos)
mesh!(ax, Sphere(Point3f(0), 1.0), model=transform)
```
"""
function calculate_ellipsoid_model(cov_3x3::Matrix, pos_3d::Vector)
    # Ensure symmetric for eigen decomposition
    cov_sym = Symmetric(cov_3x3)
    F = eigen(cov_sym)

    # 3-sigma scaling with numerical stability
    # Clamp to avoid negative values from numerical noise
    radii = sqrt.(max.(F.values, Config.MIN_EIGENVALUE)) .* Config.SIGMA_SCALE

    # Rotation matrix from eigenvectors
    # F.vectors are columns corresponding to eigenvalues
    rot_mat = F.vectors

    # Construct 4x4 transformation matrix: M = T * R * S

    # 1. Scale matrix
    S = Diagonal([radii; 1.0])

    # 2. Rotation matrix (expand to 4x4)
    R = Matrix{Float64}(I, 4, 4)
    R[1:3, 1:3] = rot_mat

    # 3. Translation matrix
    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 4] = pos_3d

    return Mat4f(T * R * S)
end


"""
    calculate_cumulative_distance(positions::Matrix) -> Vector{Float64}

Calculate cumulative distance along a trajectory path.

# Arguments
- `positions::Matrix`: N×3 matrix of 3D positions

# Returns
- `Vector{Float64}`: N-length vector of cumulative distances (first element is 0.0)

# Example
```julia
trajectory = rand(100, 3)  # 100 positions
cumulative_dist = calculate_cumulative_distance(trajectory)
# cumulative_dist[1] == 0.0
# cumulative_dist[end] == total path length
```
"""
function calculate_cumulative_distance(positions::Matrix)
    # Calculate deltas between consecutive positions
    deltas = vcat([0.0 0.0 0.0], positions[2:end, :] .- positions[1:end-1, :])

    # Calculate segment distances
    segment_distances = sqrt.(sum(deltas.^2, dims=2))[:]

    # Return cumulative sum
    return cumsum(segment_distances)
end


"""
    calculate_local_error_ratio(errors::Vector, cumulative_distance::Vector,
                                segment_distances::Vector; window_size::Int) -> Vector{Float64}

Calculate local error-to-distance ratio with smoothing window.

# Arguments
- `errors::Vector`: Point-to-point errors
- `cumulative_distance::Vector`: Cumulative distance along path
- `segment_distances::Vector`: Distance of each segment
- `window_size::Int`: Number of frames for smoothing window

# Returns
- `Vector{Float64}`: Local error/distance ratio for each frame

# Example
```julia
errors = [0.01, 0.02, 0.015, ...]
cum_dist = calculate_cumulative_distance(trajectory)
seg_dist = diff([0.0; cum_dist])
ratio = calculate_local_error_ratio(errors, cum_dist, seg_dist, window_size=10)
```
"""
function calculate_local_error_ratio(errors::Vector, cumulative_distance::Vector,
                                     segment_distances::Vector;
                                     window_size::Int=Config.ERROR_WINDOW_SIZE)
    seq_len = length(errors)
    local_error_dist = zeros(seq_len)

    for i in 1:seq_len
        window_start = max(1, i - window_size + 1)
        window_error = sum(errors[window_start:i])
        window_dist = max(
            cumulative_distance[i] - cumulative_distance[window_start] + segment_distances[window_start],
            Config.MIN_DISTANCE_THRESHOLD
        )
        local_error_dist[i] = window_error / window_dist
    end

    return local_error_dist
end


export MakieDashboard
export calculate_ellipsoid_model, calculate_cumulative_distance, calculate_local_error_ratio

end
