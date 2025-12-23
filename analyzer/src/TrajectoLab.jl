module TrajectoLab

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

# Application Plugins
include("Application/Dashboard.jl")
using .Dashboard
export MakieDashboard

end