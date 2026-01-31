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
Single model visualization for the Trajecto Dashboard.

This module implements the detailed 4-panel dashboard for analyzing
a single model's trajectory prediction.
"""
module DashboardSingle

using ..AbstractLayers
using ..Config
using ..Metrics
using ..DashboardCore
using ..DashboardControls
using ..DashboardPlots
using GLMakie


"""
    run_single_model_app(app::MakieDashboard, trajectory_data, input_stream)

Display detailed single-model analysis dashboard with 5 panels.

# Arguments
- `app::MakieDashboard`: Dashboard application instance
- `trajectory_data::NamedTuple`: Prediction results with (pos=(Seq,3), cov=(Seq,15,15))
- `input_stream::NamedTuple`: Input data with (sensor=(Seq,7), gt_pos=(Seq,3))

# Dashboard Panels
1. **3D Trajectory (Row 1)**: Ground truth, raw prediction, aligned prediction,
   3-sigma uncertainty ellipsoid, and dynamic markers
2. **Accelerometer (Row 2, Left)**: X, Y, Z acceleration time series
3. **Point-to-Point Error (Row 2, Middle)**: Error magnitude over time after alignment
4. **Error/Distance Ratio (Row 2, Right)**: Local error ratio vs cumulative distance
5. **Axis Errors (Row 3)**: Separate X, Y, Z signed errors to identify directional biases

# Interactive Controls
- **Time Slider**: Navigate frame-by-frame through the sequence
- **Play/Pause Button**: Automatic playback at 50 FPS
- Dynamic markers update in real-time with slider position

# Example
```julia
app = MakieDashboard()
trajectory = predict_trajectory(estimator, sensor_data)
run_single_model_app(app, trajectory, input_stream)
```
"""
function run_single_model_app(app::MakieDashboard, trajectory_data, input_stream)
    # Extract data
    pred_pos = trajectory_data.pos
    pred_cov = trajectory_data.cov
    gt_pos = input_stream.gt_pos
    sensor = input_stream.sensor

    seq_len = size(gt_pos, 1)

    # Calculate detailed metrics (assuming Config.DEFAULT_DT = 0.02s for 50Hz)
    metrics = calculate_metrics(gt_pos, pred_pos, Config.DEFAULT_DT)
    aligned_pred = metrics.aligned_traj

    # Create figure
    fig = Figure(size = app.window_size, font = Config.DEFAULT_FONT)

    # ========================================================================
    # Title with metrics
    # ========================================================================
    title_str = format_metrics_title(metrics, prefix="Trajecto Analysis")

    # ========================================================================
    # Panel 1: 3D Trajectory (Row 1, spanning all columns)
    # ========================================================================
    ax3d = Axis3(fig[1, 1:3], title = title_str,
                aspect = :data, perspectiveness = 0.5)

    # Create controls first to get frame_idx observable
    controls = create_playback_controls(fig, fig[4, 1:3], seq_len)

    # Plot 3D trajectory with covariance
    plot_3d_trajectory!(ax3d, gt_pos, pred_pos, aligned_pred, controls.frame_idx,
                       show_covariance=true, pred_cov=pred_cov)

    # ========================================================================
    # Panel 2: Accelerometer (Row 2, Column 1)
    # ========================================================================
    ax_acc = Axis(fig[2, 1], title = "Accelerometer",
                 xlabel = "Frame", ylabel = "m/s²")

    plot_accelerometer!(ax_acc, sensor, controls.frame_idx)

    # ========================================================================
    # Panel 3: Error Distribution (Row 2, Column 2)
    # ========================================================================
    ax_err = Axis(fig[2, 2], title = "Aligned Point-to-Point Error",
                 xlabel = "Frame", ylabel = "Error (m)")

    plot_error_distribution!(ax_err, gt_pos, aligned_pred, controls.frame_idx)

    # ========================================================================
    # Panel 4: Error/Distance Ratio (Row 2, Column 3)
    # ========================================================================
    ax_err_dist = Axis(fig[2, 3],
                      title = "Error/Distance Ratio",
                      xlabel = "Cumulative Distance (m)",
                      ylabel = "Error/Distance (%)")

    plot_error_distance_ratio!(ax_err_dist, gt_pos, aligned_pred,
                              controls.frame_idx, metrics.error_over_dist)

    # ========================================================================
    # Panel 5: Axis Errors (Row 3, spanning all columns)
    # ========================================================================
    ax_axis_err = Axis(fig[3, 1:3],
                      title = "Per-Axis Errors (X, Y, Z)",
                      xlabel = "Frame",
                      ylabel = "Error (m)")

    plot_axis_errors!(ax_axis_err, gt_pos, aligned_pred, controls.frame_idx)

    # ========================================================================
    # Start playback loop
    # ========================================================================
    start_playback_loop(controls, seq_len)

    # Display figure
    display(fig)
end


export run_single_model_app

end
