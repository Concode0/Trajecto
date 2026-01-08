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