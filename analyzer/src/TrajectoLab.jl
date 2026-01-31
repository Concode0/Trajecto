# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

module TrajectoLab

# Configuration Constants
include("Config.jl")
using .Config

# Abstract Interfaces
include("AbstractLayers.jl")
using .AbstractLayers
export AbstractPerception, process_input
export AbstractEstimator, predict_trajectory, load_model
export AbstractApplication, run_app

# Perception Plugins
include("Perception/HDF5Loader.jl")
using .HDF5Loader
export HDF5Perception

include("Perception/APSPlugin.jl")
using .APSPlugin
export APSPerception

include("Perception/PressurePlugin.jl")
using .PressurePlugin
export PressurePerception

# Estimation Plugins
include("Estimation/PyTorchEstimator.jl")
using .PyTorchEstimator
export TrajectoEstimator

# Analysis Tools
include("Analysis/Metrics.jl")
using .Metrics
export calculate_metrics, align_trajectory

include("Analysis/CRLB.jl")
using .CRLB
export CRLBConfig, compute_crlb

# Application Plugins
include("Application/Dashboard.jl")
using .Dashboard
export MakieDashboard, run_app_comparison

end