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
Trajecto Dashboard - Main Module

This module provides an interactive visualization dashboard for trajectory analysis.
It has been refactored into modular components for better maintainability:

- **DashboardCore**: Base types and utilities
- **DashboardControls**: Interactive controls (slider, playback)
- **DashboardPlots**: Reusable plotting functions
- **DashboardSingle**: Single model detailed analysis view
- **DashboardComparison**: Multi-model comparison view

# Usage

```julia
using TrajectoLab

# Single model analysis
app = MakieDashboard()
run_app(app, trajectory_data, input_stream)

# Multi-model comparison
results = [(name="Model A", trajectory=traj_a), ...]
run_app_comparison(app, results, input_stream)
```
"""
module Dashboard

using ..AbstractLayers
using ..Config
using ..Metrics

# Import all submodules
include("DashboardCore.jl")
using .DashboardCore

include("DashboardControls.jl")
using .DashboardControls

include("DashboardPlots.jl")
using .DashboardPlots

include("DashboardSingle.jl")
using .DashboardSingle

include("DashboardComparison.jl")
using .DashboardComparison

# Re-export core types
export MakieDashboard

# Re-export utilities (for advanced users)
export calculate_ellipsoid_model, calculate_cumulative_distance, calculate_local_error_ratio
export create_playback_controls, start_playback_loop
export plot_3d_trajectory!, plot_accelerometer!, plot_error_distribution!
export plot_error_distance_ratio!, plot_comparison_3d!, format_metrics_title


"""
    run_app(app::MakieDashboard, trajectory_data, input_stream)

Launch single-model analysis dashboard.

This is the main entry point for detailed analysis of a single trajectory model.
Displays a 4-panel dashboard with 3D visualization, sensor data, error analysis,
and interactive controls.

# Arguments
- `app::MakieDashboard`: Dashboard application instance
- `trajectory_data::NamedTuple`: Model prediction with (pos=(Seq,3), cov=(Seq,15,15))
- `input_stream::NamedTuple`: Input data with (sensor=(Seq,7), gt_pos=(Seq,3), seq_len=Int)

# Example
```julia
app = MakieDashboard()
estimator = TrajectoEstimator("eskf", "model.pth", model_path, scaler_path)
trajectory = predict_trajectory(estimator, sensor_data)
run_app(app, trajectory, input_stream)
```

# See Also
- `run_app_comparison`: For multi-model comparison
"""
function AbstractLayers.run_app(app::MakieDashboard, trajectory_data, input_stream)
    run_single_model_app(app, trajectory_data, input_stream)
end


"""
    run_app_comparison(app::MakieDashboard, results::Vector, input_stream)

Launch multi-model comparison dashboard.

Displays synchronized visualizations of multiple models side-by-side,
enabling direct performance comparison with shared ground truth.

# Arguments
- `app::MakieDashboard`: Dashboard application instance
- `results::Vector`: Vector of NamedTuples with (name::String, trajectory::NamedTuple or nothing)
  - Failed models (trajectory=nothing) are automatically filtered out
- `input_stream::NamedTuple`: Input data with (sensor=(Seq,7), gt_pos=(Seq,3), seq_len=Int)

# Example
```julia
app = MakieDashboard()
results = [
    (name="Pure Integration", trajectory=traj_baseline),
    (name="Pure ESKF", trajectory=traj_eskf),
    (name="ESKF-TCN", trajectory=traj_hybrid)
]
run_app_comparison(app, results, input_stream)
```

# Dashboard Features
- Color-coded trajectories (supports up to 6 models)
- Synchronized playback across all views
- Real-time metrics comparison
- Individual XY projections for each model

# See Also
- `run_app`: For single-model detailed analysis
"""
function run_app_comparison(app::MakieDashboard, results::Vector, input_stream)
    run_comparison_app(app, results, input_stream)
end


export run_app_comparison

end
