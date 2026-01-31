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
Multi-model comparison visualization for the Trajecto Dashboard.

This module implements side-by-side comparison of multiple trajectory models
with shared ground truth reference.
"""
module DashboardComparison

using ..Config
using ..Metrics
using ..DashboardCore
using ..DashboardControls
using ..DashboardPlots
using GLMakie
using Statistics


"""
    run_comparison_app(app::MakieDashboard, results::Vector, input_stream)

Display multi-model comparison dashboard with synchronized views.

# Arguments
- `app::MakieDashboard`: Dashboard application instance
- `results::Vector`: Vector of NamedTuples with (name::String, trajectory::NamedTuple or nothing)
- `input_stream::NamedTuple`: Input data with (sensor=(Seq,7), gt_pos=(Seq,3))

# Dashboard Layout
1. **Top Panel**: 3D trajectory comparison with all models and ground truth
2. **Middle Panels**: Individual XY projection views for each model
3. **Bottom Panel 1**: Point-to-point error comparison over time
4. **Bottom Panel 2**: Error/distance ratio comparison vs cumulative distance
5. **Bottom Panel 3**: Interactive controls (slider, play/pause)

# Features
- Filters out failed models (trajectory === nothing)
- Color-coded trajectories (up to 6 models supported)
- Synchronized frame indicators across all panels
- Dynamic markers that update with slider position
- Real-time metrics display in title

# Example
```julia
app = MakieDashboard()
results = [
    (name="Pure ESKF", trajectory=traj1),
    (name="ESKF-TCN", trajectory=traj2),
    (name="TCN-only", trajectory=traj3)
]
run_comparison_app(app, results, input_stream)
```
"""
function run_comparison_app(app::MakieDashboard, results::Vector, input_stream)
    gt_pos = input_stream.gt_pos
    seq_len = size(gt_pos, 1)

    # Filter out failed models
    valid_results = filter(r -> r.trajectory !== nothing, results)
    n_models = length(valid_results)

    if n_models == 0
        println("No valid models to display!")
        return
    end

    # Calculate metrics for all models
    model_metrics = []
    for result in valid_results
        metrics = calculate_metrics(gt_pos, result.trajectory.pos, Config.DEFAULT_DT)
        push!(model_metrics, (name=result.name, metrics=metrics, trajectory=result.trajectory))
    end

    # Create figure with larger resolution for multiple plots
    fig = Figure(size = app.window_size, font = Config.DEFAULT_FONT)

    # ========================================================================
    # TOP: 3D Trajectory Comparison (Shared View)
    # ========================================================================

    # Build title with all models
    title_lines = ["Multi-Model Trajectory Comparison"]
    for mm in model_metrics
        ape = round(mm.metrics.ape_rmse * 100, digits=Config.DISPLAY_PRECISION_APE_CM)
        err_dist = round(mm.metrics.error_over_dist * 100, digits=Config.DISPLAY_PRECISION_ERROR_DIST)
        push!(title_lines, "$(mm.name): APE=$(ape)cm, Err/Dist=$(err_dist)%")
    end
    title_str = join(title_lines, " | ")

    ax3d = Axis3(fig[1, 1:n_models], title = title_str,
                aspect = :data, perspectiveness = 0.5)

    # Create controls
    controls = create_playback_controls(fig, fig[5, 1:n_models], seq_len)

    # Plot 3D comparison
    plot_comparison_3d!(ax3d, gt_pos, model_metrics, controls.frame_idx)

    # ========================================================================
    # MIDDLE: Individual Model Panels (Side-by-Side XY Projections)
    # ========================================================================

    for col_idx in 1:n_models
        mm = model_metrics[col_idx]
        pred_pos = mm.trajectory.pos
        color = Config.COLORS_MODEL_COMPARISON[mod1(col_idx, length(Config.COLORS_MODEL_COMPARISON))]

        # XY projection
        ax_xy = Axis(fig[2, col_idx],
                    title = "$(mm.name)\nXY View",
                    xlabel = "X (m)", ylabel = "Y (m)",
                    aspect = DataAspect())

        lines!(ax_xy, gt_pos[:, 1], gt_pos[:, 2],
              color = Config.COLOR_GROUND_TRUTH,
              linewidth = Config.LINEWIDTH_ALIGNED,
              linestyle = Config.LINESTYLE_GROUND_TRUTH,
              alpha = Config.ALPHA_TRAJECTORY)
        lines!(ax_xy, pred_pos[:, 1], pred_pos[:, 2],
              color = color,
              linewidth = Config.LINEWIDTH_DEFAULT)
    end

    # ========================================================================
    # BOTTOM: Error Comparison
    # ========================================================================

    ax_err_cmp = Axis(fig[3, 1:n_models],
                     title = "Point-to-Point Error Comparison",
                     xlabel = "Frame", ylabel = "Error (m)")

    for (idx, mm) in enumerate(model_metrics)
        aligned_pred = mm.metrics.aligned_traj
        dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]
        color = Config.COLORS_MODEL_COMPARISON[mod1(idx, length(Config.COLORS_MODEL_COMPARISON))]

        lines!(ax_err_cmp, dist, color = color,
              linewidth = Config.LINEWIDTH_DEFAULT, label = mm.name)
    end

    # Disable interactive legend to avoid GLMakie hover bug
    Legend(ax_err_cmp.parent, ax_err_cmp, framevisible=true, backgroundcolor=(:white, 0.8),
           halign=:right, valign=:top, margin=(10, 10, 10, 10))
    vlines!(ax_err_cmp, @lift([$(controls.frame_idx)]), color = :black,
           linestyle = Config.LINESTYLE_INDICATOR)

    # ========================================================================
    # Error/Distance Comparison
    # ========================================================================

    ax_err_dist_cmp = Axis(fig[4, 1:n_models],
                          title = "Error/Distance Ratio Comparison",
                          xlabel = "Cumulative Distance (m)",
                          ylabel = "Error/Distance (%)")

    # Calculate cumulative distance (same for all models - using GT)
    cumulative_distance = calculate_cumulative_distance(gt_pos)
    gt_deltas = vcat([0.0 0.0 0.0], gt_pos[2:end, :] .- gt_pos[1:end-1, :])
    segment_distances = sqrt.(sum(gt_deltas.^2, dims=2))[:]

    for (idx, mm) in enumerate(model_metrics)
        aligned_pred = mm.metrics.aligned_traj
        dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]
        color = Config.COLORS_MODEL_COMPARISON[mod1(idx, length(Config.COLORS_MODEL_COMPARISON))]

        # Calculate local error/distance ratio with smoothing
        local_error_dist = calculate_local_error_ratio(dist, cumulative_distance, segment_distances)

        err_pct = round(mm.metrics.error_over_dist*100, digits=Config.DISPLAY_PRECISION_ERROR_DIST)
        lines!(ax_err_dist_cmp, cumulative_distance, local_error_dist .* 100,
              color = color,
              linewidth = Config.LINEWIDTH_DEFAULT,
              label = "$(mm.name) ($(err_pct)%)")
    end

    # Disable interactive legend to avoid GLMakie hover bug
    Legend(ax_err_dist_cmp.parent, ax_err_dist_cmp, framevisible=true, backgroundcolor=(:white, 0.8),
           halign=:right, valign=:top, margin=(10, 10, 10, 10))

    # Dynamic position indicator
    current_dist_cmp = @lift(cumulative_distance[$(controls.frame_idx)])
    vlines!(ax_err_dist_cmp, current_dist_cmp, color = :black, linestyle = :dash)

    # ========================================================================
    # Start playback loop
    # ========================================================================
    start_playback_loop(controls, seq_len)

    # Display figure
    display(fig)
end


export run_comparison_app

end
