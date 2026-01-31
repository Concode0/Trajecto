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
Common plotting functions for the Trajecto Dashboard.

This module provides reusable plotting utilities to reduce code duplication
across single and comparison visualizations.
"""
module DashboardPlots

using ..Config
using ..DashboardCore
using ..Metrics
using GLMakie
using GeometryBasics
using Statistics

"""
    plot_3d_trajectory!(ax::Axis3, gt_pos::Matrix, pred_pos::Matrix, aligned_pred::Matrix,
                       frame_idx::Observable; show_covariance::Bool=true, pred_cov=nothing)

Plot 3D trajectory with ground truth, prediction, and aligned trajectories.

# Arguments
- `ax::Axis3`: 3D axis to plot on
- `gt_pos::Matrix`: Ground truth positions (N×3)
- `pred_pos::Matrix`: Raw prediction positions (N×3)
- `aligned_pred::Matrix`: Sim(3)-aligned prediction (N×3)
- `frame_idx::Observable{Int}`: Current frame index for dynamic markers
- `show_covariance::Bool`: Whether to show uncertainty ellipsoid (default: true)
- `pred_cov`: Prediction covariance (N×15×15), required if show_covariance=true

# Example
```julia
ax = Axis3(fig[1, 1])
frame_idx = Observable(1)
plot_3d_trajectory!(ax, gt, pred, aligned, frame_idx)
```
"""
function plot_3d_trajectory!(ax::Axis3, gt_pos::Matrix, pred_pos::Matrix, aligned_pred::Matrix,
                            frame_idx::Observable; show_covariance::Bool=true, pred_cov=nothing)
    # Static full trajectories
    lines!(ax, gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3],
           color = Config.COLOR_GROUND_TRUTH,
           linewidth = Config.LINEWIDTH_GROUND_TRUTH,
           label = "Ground Truth",
           linestyle = Config.LINESTYLE_GROUND_TRUTH,
           alpha = Config.ALPHA_TRAJECTORY)

    lines!(ax, pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3],
           color = Config.COLOR_RAW_PREDICTION,
           linewidth = Config.LINEWIDTH_PREDICTION,
           label = "Raw Prediction",
           alpha = Config.ALPHA_TRAJECTORY)

    lines!(ax, aligned_pred[:, 1], aligned_pred[:, 2], aligned_pred[:, 3],
           color = Config.COLOR_ALIGNED_PREDICTION,
           linewidth = Config.LINEWIDTH_ALIGNED,
           label = "Aligned Pred",
           linestyle = Config.LINESTYLE_ALIGNED,
           alpha = Config.ALPHA_ALIGNED)

    # Dynamic markers
    gt_pt = @lift(Point3f(gt_pos[$frame_idx, 1], gt_pos[$frame_idx, 2], gt_pos[$frame_idx, 3]))
    pred_pt = @lift(Point3f(pred_pos[$frame_idx, 1], pred_pos[$frame_idx, 2], pred_pos[$frame_idx, 3]))

    scatter!(ax, gt_pt, color = Config.COLOR_GROUND_TRUTH,
            markersize = Config.MARKERSIZE_HEAD, label = "GT Head")
    scatter!(ax, pred_pt, color = Config.COLOR_RAW_PREDICTION,
            markersize = Config.MARKERSIZE_HEAD, label = "Pred Head")

    # Optional uncertainty ellipsoid
    if show_covariance && pred_cov !== nothing
        ellipsoid_model = @lift(calculate_ellipsoid_model(
            pred_cov[$frame_idx, 1:3, 1:3],
            pred_pos[$frame_idx, :]
        ))
        mesh!(ax, Sphere(Point3f(0), 1.0f0),
             color = (Config.COLOR_UNCERTAINTY, Config.ALPHA_UNCERTAINTY),
             transparency = true,
             model = ellipsoid_model,
             label = "3σ Covariance")
    end

    # Disable interactive legend to avoid GLMakie hover bug
    axislegend(ax, framevisible=true, backgroundcolor=(:white, 0.8))
end


"""
    plot_accelerometer!(ax::Axis, sensor::Matrix, frame_idx::Observable)

Plot 3-axis accelerometer time series with frame indicator.

# Arguments
- `ax::Axis`: 2D axis to plot on
- `sensor::Matrix`: Sensor data (N×7), first 3 columns are accelerometer [X,Y,Z]
- `frame_idx::Observable{Int}`: Current frame index for vertical line indicator
"""
function plot_accelerometer!(ax::Axis, sensor::Matrix, frame_idx::Observable)
    for i in 1:3
        lines!(ax, sensor[:, i],
              color = (Config.COLORS_ACCEL_AXES[i], Config.ALPHA_ACCEL),
              label = "Acc $(['X','Y','Z'][i])")
    end

    # Frame indicator
    vlines!(ax, @lift([$frame_idx]), color = :black,
           linestyle = Config.LINESTYLE_INDICATOR)
end


"""
    plot_error_distribution!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix, frame_idx::Observable)

Plot point-to-point error distribution over time.

# Arguments
- `ax::Axis`: 2D axis to plot on
- `gt_pos::Matrix`: Ground truth positions (N×3)
- `aligned_pred::Matrix`: Aligned prediction positions (N×3)
- `frame_idx::Observable{Int}`: Current frame index for vertical line indicator
"""
function plot_error_distribution!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix, frame_idx::Observable)
    # Calculate point-to-point errors
    dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]

    # Band visualization
    band!(ax, 1:length(dist), zeros(length(dist)), dist,
         color = (Config.COLOR_ALIGNED_PREDICTION, Config.ALPHA_ERROR_BAND))
    lines!(ax, dist, color = Config.COLOR_ALIGNED_PREDICTION,
          linewidth = Config.LINEWIDTH_ERROR)

    # Frame indicator
    vlines!(ax, @lift([$frame_idx]), color = :black,
           linestyle = Config.LINESTYLE_INDICATOR)
end


"""
    plot_error_distance_ratio!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix,
                               frame_idx::Observable, global_error_dist::Float64)

Plot local error/distance ratio vs cumulative distance.

# Arguments
- `ax::Axis`: 2D axis to plot on
- `gt_pos::Matrix`: Ground truth positions (N×3)
- `aligned_pred::Matrix`: Aligned prediction positions (N×3)
- `frame_idx::Observable{Int}`: Current frame index for position indicator
- `global_error_dist::Float64`: Global average error/distance ratio for reference line

# Returns
- `Vector{Float64}`: Cumulative distance vector (for external use)
"""
function plot_error_distance_ratio!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix,
                                   frame_idx::Observable, global_error_dist::Float64)
    # Calculate errors
    dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]

    # Calculate cumulative distance
    cumulative_distance = calculate_cumulative_distance(gt_pos)
    gt_deltas = vcat([0.0 0.0 0.0], gt_pos[2:end, :] .- gt_pos[1:end-1, :])
    segment_distances = sqrt.(sum(gt_deltas.^2, dims=2))[:]

    # Calculate local error/distance ratio
    local_error_dist = calculate_local_error_ratio(dist, cumulative_distance, segment_distances)

    # Plot local ratio
    lines!(ax, cumulative_distance, local_error_dist .* 100,
          color = :purple,
          linewidth = Config.LINEWIDTH_DEFAULT,
          label = "Local Ratio ($(Config.ERROR_WINDOW_SIZE)-frame)")

    # Plot global average reference
    hlines!(ax, [global_error_dist * 100],
           color = Config.COLOR_UNCERTAINTY,
           linestyle = :dash,
           linewidth = Config.LINEWIDTH_DEFAULT,
           label = "Global Avg")

    # Disable interactive legend to avoid GLMakie hover bug
    Legend(ax.parent, ax, framevisible=true, backgroundcolor=(:white, 0.8),
           halign=:right, valign=:top, margin=(10, 10, 10, 10))

    # Dynamic position indicator
    current_dist = @lift(cumulative_distance[$frame_idx])
    vlines!(ax, current_dist, color = :black, linestyle = :dot)

    return cumulative_distance
end


"""
    plot_comparison_3d!(ax::Axis3, gt_pos::Matrix, model_metrics::Vector, frame_idx::Observable)

Plot 3D trajectory comparison with multiple models.

# Arguments
- `ax::Axis3`: 3D axis to plot on
- `gt_pos::Matrix`: Ground truth positions (N×3)
- `model_metrics::Vector`: Vector of NamedTuples with (name, metrics, trajectory)
- `frame_idx::Observable{Int}`: Current frame index for dynamic markers

# Example
```julia
ax = Axis3(fig[1, 1])
model_metrics = [(name="ESKF", metrics=..., trajectory=...),  ...]
plot_comparison_3d!(ax, gt, model_metrics, frame_idx)
```
"""
function plot_comparison_3d!(ax::Axis3, gt_pos::Matrix, model_metrics::Vector, frame_idx::Observable)
    # Ground truth (shared)
    lines!(ax, gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3],
          color = Config.COLOR_GROUND_TRUTH,
          linewidth = Config.LINEWIDTH_GROUND_TRUTH,
          label = "Ground Truth",
          linestyle = Config.LINESTYLE_GROUND_TRUTH)

    # Plot each model's trajectory
    for (idx, mm) in enumerate(model_metrics)
        pred_pos = mm.trajectory.pos
        color = Config.COLORS_MODEL_COMPARISON[mod1(idx, length(Config.COLORS_MODEL_COMPARISON))]

        lines!(ax, pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3],
              color = color,
              linewidth = Config.LINEWIDTH_DEFAULT,
              label = mm.name,
              alpha = Config.ALPHA_MODEL_COMPARISON)
    end

    # Add dynamic markers
    for (idx, mm) in enumerate(model_metrics)
        pred_pos = mm.trajectory.pos
        color = Config.COLORS_MODEL_COMPARISON[mod1(idx, length(Config.COLORS_MODEL_COMPARISON))]

        pred_pt = @lift(Point3f0(pred_pos[$frame_idx, 1],
                                pred_pos[$frame_idx, 2],
                                pred_pos[$frame_idx, 3]))
        scatter!(ax, pred_pt, color = color, markersize = Config.MARKERSIZE_HEAD_COMPARISON)
    end

    # GT marker (star)
    gt_pt = @lift(Point3f0(gt_pos[$frame_idx, 1],
                          gt_pos[$frame_idx, 2],
                          gt_pos[$frame_idx, 3]))
    scatter!(ax, gt_pt, color = Config.COLOR_GROUND_TRUTH,
            markersize = Config.MARKERSIZE_GT_STAR, marker = :star5)

    # Disable interactive legend to avoid GLMakie hover bug
    Legend(ax.parent, ax, framevisible=true, backgroundcolor=(:white, 0.8),
           halign=:left, valign=:top, margin=(10, 10, 10, 10))
end


"""
    format_metrics_title(metrics::NamedTuple; prefix::String="") -> String

Format metrics into a title string for display.

# Arguments
- `metrics::NamedTuple`: Metrics from calculate_metrics (must have ape_rmse, error_over_dist, error_over_time)
- `prefix::String`: Optional prefix for the title (default: "")

# Returns
- `String`: Formatted title with metrics in cm, %, and cm/s

# Example
```julia
metrics = calculate_metrics(gt, pred, 0.02)
title = format_metrics_title(metrics, prefix="ESKF-TCN Model")
# Returns: "ESKF-TCN Model\nAPE(RMSE): 1.23 cm | Err/Dist: 2.45 % | Err/Time: 0.61 cm/s"
```
"""
function format_metrics_title(metrics::NamedTuple; prefix::String="")
    ape_str = "APE(RMSE): $(round(metrics.ape_rmse*100, digits=Config.DISPLAY_PRECISION_APE_CM)) cm"
    err_dist_str = "Err/Dist: $(round(metrics.error_over_dist*100, digits=Config.DISPLAY_PRECISION_ERROR_DIST)) %"
    err_time_str = "Err/Time: $(round(metrics.error_over_time*100, digits=Config.DISPLAY_PRECISION_ERROR_TIME)) cm/s"

    if isempty(prefix)
        return "$ape_str | $err_dist_str | $err_time_str"
    else
        return "$prefix\n$ape_str | $err_dist_str | $err_time_str"
    end
end


"""
    plot_axis_errors!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix, frame_idx::Observable)

Plot per-axis (X, Y, Z) errors over time to identify directional biases.

# Arguments
- `ax::Axis`: 2D axis to plot on
- `gt_pos::Matrix`: Ground truth positions (N×3)
- `aligned_pred::Matrix`: Aligned prediction positions (N×3)
- `frame_idx::Observable{Int}`: Current frame index for vertical line indicator

# Visualization
Shows three separate lines for X, Y, Z axis errors (signed errors, not absolute):
- Red: X-axis error
- Green: Y-axis error
- Blue: Z-axis error

A horizontal zero line helps identify systematic biases. Useful for:
- Detecting gravity alignment issues (consistent Z bias)
- Identifying sensor calibration problems (consistent X/Y bias)
- Understanding directional performance characteristics
"""
function plot_axis_errors!(ax::Axis, gt_pos::Matrix, aligned_pred::Matrix, frame_idx::Observable)
    # Calculate per-axis signed errors (not absolute)
    errors = gt_pos .- aligned_pred  # (N, 3)

    # Plot each axis
    lines!(ax, errors[:, 1], color = :red, linewidth = Config.LINEWIDTH_DEFAULT,
          label = "X Error", alpha = 0.8)
    lines!(ax, errors[:, 2], color = :green, linewidth = Config.LINEWIDTH_DEFAULT,
          label = "Y Error", alpha = 0.8)
    lines!(ax, errors[:, 3], color = :blue, linewidth = Config.LINEWIDTH_DEFAULT,
          label = "Z Error", alpha = 0.8)

    # Zero reference line
    hlines!(ax, [0.0], color = :black, linestyle = :dash, linewidth = 1, alpha = 0.5)

    # Frame indicator
    vlines!(ax, @lift([$frame_idx]), color = :black,
           linestyle = Config.LINESTYLE_INDICATOR)

    # Disable interactive legend to avoid GLMakie hover bug
    Legend(ax.parent, ax, framevisible=true, backgroundcolor=(:white, 0.8),
           halign=:right, valign=:top, margin=(10, 10, 10, 10))
end


export plot_3d_trajectory!, plot_accelerometer!, plot_error_distribution!
export plot_error_distance_ratio!, plot_comparison_3d!, format_metrics_title
export plot_axis_errors!

end

