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