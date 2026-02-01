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