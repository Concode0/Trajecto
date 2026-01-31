# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

module AbstractLayers

# --- 1. Data Perception Layer ---
abstract type AbstractPerception end

"""
    process_input(perception::AbstractPerception, raw_data)

Converts raw input data into a standardized format for the estimator.
"""
function process_input end

# --- 2. Estimation Core Layer ---
abstract type AbstractEstimator end

"""
    predict_trajectory(estimator::AbstractEstimator, input_stream)

Takes standardized input and returns the estimated state/trajectory.
"""
function predict_trajectory end

"""
    load_model(estimator::AbstractEstimator, model_path)

Loads the model weights/parameters.
"""
function load_model end

# --- 3. Application Layer ---
abstract type AbstractApplication end

"""
    run_app(app::AbstractApplication, trajectory_data, sensor_data)

Executes the application logic (e.g., visualization, recognition) using the estimated trajectory.
"""
function run_app end

export AbstractPerception, process_input
export AbstractEstimator, predict_trajectory, load_model
export AbstractApplication, run_app

end
