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
Configuration constants for Trajecto Analyzer.

This module centralizes all configuration parameters and magic numbers
used throughout the analyzer pipeline.
"""
module Config

# ============================================================================
# Data Processing Constants
# ============================================================================

"""Default IMU sampling rate in Hz (Trajecto standard protocol)"""
const DEFAULT_SAMPLING_RATE_HZ = 50

"""Default time step (dt) in seconds"""
const DEFAULT_DT = 1.0 / DEFAULT_SAMPLING_RATE_HZ  # 0.02s

"""Maximum sequence length for trajectory analysis"""
const MAX_SEQUENCE_LENGTH = 1750  # 35 seconds @ 50Hz


# ============================================================================
# Visualization Constants
# ============================================================================

"""Default dashboard window size (width, height) in pixels"""
const DEFAULT_WINDOW_SIZE = (1600, 1200)

"""Default figure font family"""
const DEFAULT_FONT = "sans"

"""Playback frame rate for dashboard animation (FPS)"""
const PLAYBACK_FPS = 50

"""Playback sleep time in seconds"""
const PLAYBACK_SLEEP_TIME = 1.0 / PLAYBACK_FPS  # 0.02s

"""3-sigma scaling factor for uncertainty ellipsoids"""
const SIGMA_SCALE = 3.0

"""Minimum eigenvalue for numerical stability in ellipsoid calculation"""
const MIN_EIGENVALUE = 1e-9


# ============================================================================
# Error Analysis Constants
# ============================================================================

"""Window size for local error/distance ratio smoothing (number of frames)"""
const ERROR_WINDOW_SIZE = 10

"""Minimum distance threshold for error/distance calculations (meters)"""
const MIN_DISTANCE_THRESHOLD = 1e-6


# ============================================================================
# Color Palettes
# ============================================================================

"""Color palette for ground truth visualization"""
const COLOR_GROUND_TRUTH = :blue

"""Color palette for raw prediction visualization"""
const COLOR_RAW_PREDICTION = :red

"""Color palette for aligned prediction visualization"""
const COLOR_ALIGNED_PREDICTION = :green

"""Color palette for uncertainty ellipsoid"""
const COLOR_UNCERTAINTY = :orange

"""Color palette for accelerometer axes [X, Y, Z]"""
const COLORS_ACCEL_AXES = [:red, :green, :blue]

"""Color palette for multi-model comparison (up to 6 models)"""
const COLORS_MODEL_COMPARISON = [:red, :orange, :purple, :cyan, :magenta, :yellow]


# ============================================================================
# Model Configuration
# ============================================================================

"""Supported model types and their Python module paths"""
const MODEL_CONFIGS = Dict(
    "eskf" => (
        module_path = "model.ESKF_TCN",
        class_name = "ESKFTCN_model",
        requires_weights = true,
        description = "ESKF-TCN Hybrid (ML-augmented physics filter)"
    ),
    "aekf" => (
        module_path = "model.AEKF_TCN",
        class_name = "AEKFTCN_model",
        requires_weights = true,
        description = "AEKF-TCN Adaptive Variant"
    ),
    "tcn" => (
        module_path = "model.onlyTCN",
        class_name = "OnlyTCN",
        requires_weights = true,
        description = "TCN-only Direct Regression"
    ),
    "pure_eskf" => (
        module_path = "model.pure_eskf",
        class_name = "PureESKFModel",
        requires_weights = false,
        description = "Physics-only Baseline (no ML)"
    ),
    "pure_integration" => (
        module_path = "model.pure_integration",
        class_name = "PureIntegrationModel",
        requires_weights = false,
        description = "Simple Dead Reckoning Baseline"
    )
)

"""Default PyTorch device for inference"""
const DEFAULT_DEVICE = "cpu"


# ============================================================================
# Alignment Configuration
# ============================================================================

"""Enable scale estimation in Sim(3) alignment (Umeyama algorithm)"""
const ALIGNMENT_WITH_SCALE = true

"""Minimum SVD singular value threshold for reflection detection"""
const ALIGNMENT_SVD_THRESHOLD = 1e-10


# ============================================================================
# Display Formatting
# ============================================================================

"""Number of decimal places for APE RMSE display (cm)"""
const DISPLAY_PRECISION_APE_CM = 2

"""Number of decimal places for error/distance ratio display (%)"""
const DISPLAY_PRECISION_ERROR_DIST = 2

"""Number of decimal places for error/time display (cm/s)"""
const DISPLAY_PRECISION_ERROR_TIME = 2

"""Number of decimal places for drift display (mm)"""
const DISPLAY_PRECISION_DRIFT_MM = 2


# ============================================================================
# Transparency and Styling
# ============================================================================

"""Alpha transparency for trajectory lines in 3D plot"""
const ALPHA_TRAJECTORY = 0.5

"""Alpha transparency for aligned prediction"""
const ALPHA_ALIGNED = 0.6

"""Alpha transparency for uncertainty ellipsoid"""
const ALPHA_UNCERTAINTY = 0.3

"""Alpha transparency for error band visualization"""
const ALPHA_ERROR_BAND = 0.2

"""Alpha transparency for accelerometer traces"""
const ALPHA_ACCEL = 0.7

"""Alpha transparency for model comparison trajectories"""
const ALPHA_MODEL_COMPARISON = 0.7


# ============================================================================
# Line and Marker Styles
# ============================================================================

"""Ground truth line width (pixels)"""
const LINEWIDTH_GROUND_TRUTH = 2

"""Prediction line width (pixels)"""
const LINEWIDTH_PREDICTION = 2

"""Aligned prediction line width (pixels)"""
const LINEWIDTH_ALIGNED = 1

"""Default line width for plots (pixels)"""
const LINEWIDTH_DEFAULT = 2

"""Error plot line width (pixels)"""
const LINEWIDTH_ERROR = 1

"""Ground truth line style"""
const LINESTYLE_GROUND_TRUTH = :dash

"""Aligned prediction line style"""
const LINESTYLE_ALIGNED = :dot

"""Indicator vline style"""
const LINESTYLE_INDICATOR = :dash

"""Marker size for trajectory heads (single view)"""
const MARKERSIZE_HEAD = 15

"""Marker size for trajectory heads (comparison view)"""
const MARKERSIZE_HEAD_COMPARISON = 15

"""Marker size for ground truth star marker"""
const MARKERSIZE_GT_STAR = 20


# ============================================================================
# Export all constants
# ============================================================================

export DEFAULT_SAMPLING_RATE_HZ, DEFAULT_DT, MAX_SEQUENCE_LENGTH
export DEFAULT_WINDOW_SIZE, DEFAULT_FONT, PLAYBACK_FPS, PLAYBACK_SLEEP_TIME
export SIGMA_SCALE, MIN_EIGENVALUE
export ERROR_WINDOW_SIZE, MIN_DISTANCE_THRESHOLD
export COLOR_GROUND_TRUTH, COLOR_RAW_PREDICTION, COLOR_ALIGNED_PREDICTION, COLOR_UNCERTAINTY
export COLORS_ACCEL_AXES, COLORS_MODEL_COMPARISON
export MODEL_CONFIGS, DEFAULT_DEVICE
export ALIGNMENT_WITH_SCALE, ALIGNMENT_SVD_THRESHOLD
export DISPLAY_PRECISION_APE_CM, DISPLAY_PRECISION_ERROR_DIST, DISPLAY_PRECISION_ERROR_TIME, DISPLAY_PRECISION_DRIFT_MM
export ALPHA_TRAJECTORY, ALPHA_ALIGNED, ALPHA_UNCERTAINTY, ALPHA_ERROR_BAND, ALPHA_ACCEL, ALPHA_MODEL_COMPARISON
export LINEWIDTH_GROUND_TRUTH, LINEWIDTH_PREDICTION, LINEWIDTH_ALIGNED, LINEWIDTH_DEFAULT, LINEWIDTH_ERROR
export LINESTYLE_GROUND_TRUTH, LINESTYLE_ALIGNED, LINESTYLE_INDICATOR
export MARKERSIZE_HEAD, MARKERSIZE_HEAD_COMPARISON, MARKERSIZE_GT_STAR

end
