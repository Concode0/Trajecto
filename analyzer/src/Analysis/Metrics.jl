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
Trajectory Alignment and Metrics Calculation.

This module provides functions for aligning predicted trajectories to ground truth
and calculating standardized evaluation metrics similar to the evo trajectory evaluation toolkit.

# Key Functions
- `align_trajectory`: Sim(3) or SE(3) alignment using Umeyama algorithm
- `calculate_metrics`: Comprehensive trajectory evaluation metrics

# Mathematical Background
Alignment uses the Umeyama algorithm which finds the optimal similarity transformation
(rotation, translation, and optionally scale) that minimizes the least-squares error
between two point sets. This is equivalent to Sim(3) alignment when scale is enabled,
or SE(3) alignment when scale is fixed to 1.

# References
- Umeyama, S. (1991). "Least-Squares Estimation of Transformation Parameters
  Between Two Point Patterns". IEEE PAMI 13(4): 376-380.
- evo trajectory evaluation toolkit: https://github.com/MichaelGrupp/evo
"""
module Metrics

using LinearAlgebra
using Statistics

export calculate_metrics, align_trajectory

"""
    align_trajectory(gt::AbstractMatrix, pred::AbstractMatrix; with_scale::Bool=true) -> Matrix

Align predicted trajectory to ground truth using the Umeyama algorithm.

Computes the optimal similarity transformation (Sim(3)) or rigid transformation (SE(3))
that minimizes the least-squares error between corresponding points.

# Arguments
- `gt::AbstractMatrix`: Ground truth trajectory (N×3) with N 3D positions
- `pred::AbstractMatrix`: Predicted trajectory (N×3), must have same length as gt
- `with_scale::Bool`: Whether to estimate scale factor (default: true)

# Returns
- `Matrix{Float64}`: Aligned prediction trajectory (N×3)

# Algorithm
Uses SVD-based Umeyama algorithm to find optimal rotation, translation, and scale.
Handles reflection case by checking determinant and adjusting smallest singular value.

# See Also
- `calculate_metrics`: Computes metrics after alignment
"""
function align_trajectory(gt::AbstractMatrix, pred::AbstractMatrix; with_scale=true)
    n = size(gt, 1)
    if n != size(pred, 1)
        error("Trajectory lengths mismatch: $(size(gt, 1)) vs $(size(pred, 1))")
    end

    # 1. Compute centroids
    mean_gt = mean(gt, dims=1)
    mean_pred = mean(pred, dims=1)

    # 2. Center points
    gt_centered = gt .- mean_gt
    pred_centered = pred .- mean_pred

    # 3. Covariance
    H = pred_centered' * gt_centered

    # 4. SVD
    F = svd(H)
    U, S, Vt = F.U, F.S, F.Vt
    # Note: Julia svd returns U, S, Vt where H = U * Diagonal(S) * Vt
    # Umeyama defines R = V * D * U' where H = U * S * V' (in some notations)
    # Let's check dimensions.
    # H is 3x3.
    # R should be 3x3.
    
    # Standard Kabsch: R = V * U'
    # Julia's Vt is V' so V = Vt'
    V = Vt'
    
    # Determinant check for reflection
    d = det(V * U')
    
    D = Matrix{Float64}(I, 3, 3)
    if d < 0
        D[3, 3] = -1
    end
    
    R = V * D * U'

    # 5. Scale
    if with_scale
        # var_pred = sum(norm.(eachrow(pred_centered)).^2) / n # This is bias
        # But we need sum of squared norms
        ss_pred = sum(abs2, pred_centered)
        c = sum(S .* diag(D)) / (ss_pred / n) # Wait, Umeyama formula
        # c = 1/sigma_x^2 * trace(D * S)
        # Using the formulation from "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns"
        c = sum(S) / (ss_pred / n) # Simplified if D is identity?
        # Actually: c = Trace(DS) / (Sum(pred_centered^2)/N)
        # We need to handle the reflection case S is singular values (always positive), D handles sign.
        # So trace(D * Diagonal(S)) is sum(S) if det>0, or S[1]+S[2]-S[3] if det<0
        trace_DS = sum(S[1:2]) + D[3,3]*S[3]
        
        # Variance of source
        var_pred = sum(abs2, pred_centered) / n
        
        s = trace_DS / var_pred # Scaling factor
    else
        s = 1.0
    end

    # 6. Translation
    # t = mu_q - s * R * mu_p
    # aligned = s * (pred - mu_p) * R' + mu_q
    
    # Apply transformation
    # Note on rotation:
    # If points are rows P (Nx3), then P_aligned = P * R^T (if R is column-vector basis)
    # Usually: y = s R x + t
    # Here x and y are column vectors (3x1).
    # In our matrices, rows are points.
    # row_aligned = (s * R * row')' + t'
    #             = s * row * R' + t'
    
    # Calculate aligned prediction
    # aligned_pred = (pred .- mean_pred) * (s .* R') .+ mean_gt
    
    # Let's verify dimensions:
    # (Nx3) * (3x3) -> (Nx3)
    aligned_pred = (pred_centered * (s * R')) .+ mean_gt
    
    return aligned_pred
end

"""
    calculate_metrics(gt::AbstractMatrix, pred::AbstractMatrix, dt::Float64) -> NamedTuple

Calculate comprehensive trajectory evaluation metrics after Sim(3) alignment.

# Arguments
- `gt::AbstractMatrix`: Ground truth trajectory (N×3) in meters
- `pred::AbstractMatrix`: Predicted trajectory (N×3) in meters
- `dt::Float64`: Time step between consecutive positions in seconds

# Returns
A NamedTuple with fields:
- `ape_rmse`: Absolute Pose Error (RMSE) in meters after alignment
- `error_over_dist`: APE normalized by total path length
- `error_over_time`: APE normalized by total duration (m/s)
- `axis_rmse`: Per-axis RMSE [x, y, z] in meters
- `aligned_traj`: Sim(3)-aligned prediction trajectory (N×3)

# Metrics
Computes APE (Absolute Pose Error) similar to evo toolkit, plus normalized
metrics for error/distance ratio and error/time. Compatible with evo evaluation.

# See Also
- `align_trajectory`: Underlying alignment function
"""
function calculate_metrics(gt::AbstractMatrix, pred::AbstractMatrix, dt::Float64)
    # Align
    pred_aligned = align_trajectory(gt, pred, with_scale=true)
    
    # Errors
    diff = gt .- pred_aligned
    squared_diff = diff.^2
    
    # 1. APE RMSE
    mse = mean(sum(squared_diff, dims=2))
    rmse = sqrt(mse)
    
    # 2. Path Length & Duration
    # Path length of GT
    gt_deltas = gt[2:end, :] .- gt[1:end-1, :]
    path_len = sum(sqrt.(sum(gt_deltas.^2, dims=2)))
    
    duration = size(gt, 1) * dt
    
    safe_path_len = max(path_len, 1e-3)
    safe_duration = max(duration, 1e-3)
    
    error_over_dist = rmse / safe_path_len
    error_over_time = rmse / safe_duration
    
    # 3. Axis RMSE
    rmse_axis = sqrt.(mean(squared_diff, dims=1))[:]
    
    return (
        ape_rmse = rmse,
        error_over_dist = error_over_dist,
        error_over_time = error_over_time,
        axis_rmse = rmse_axis, # [x, y, z]
        aligned_traj = pred_aligned
    )
end

end
