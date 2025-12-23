using Pkg
Pkg.activate(@__DIR__)

using TrajectoLab

# 1. Environment Setup (Python)
const PROJECT_ROOT = dirname(@__DIR__)
const VENV_PATH = joinpath(PROJECT_ROOT, ".venv", "bin", "python")
ENV["JULIA_PYTHONCALL_EXE"] = VENV_PATH

# 2. Configuration
# Adjust these paths according to your actual data location
h5_path = joinpath(PROJECT_ROOT, "data/dataset.h5")
model_path = joinpath(PROJECT_ROOT, "eskf_tcn_model.pth")
script_path = joinpath(PROJECT_ROOT, "model")
scaler_path = joinpath(PROJECT_ROOT, "data/scaler_stats.h5")

# Select Model Type: "eskf", "aekf", "tcn"
const MODEL_TYPE = "eskf"

# 3. Instantiate Components
# A. Perception Layer
loader = HDF5Perception(h5_path)

# B. Estimation Core Layer
estimator = TrajectoEstimator(MODEL_TYPE, model_path, script_path, scaler_path)

# C. Application Layer
app = MakieDashboard()

# 4. Execution Pipeline
function run_analysis(sample_id::String)
    println("=== Trajecto Framework Execution ===")
    println("Model Type: $MODEL_TYPE")
    println("Sample: $sample_id")

    println("[1/3] Perception Layer: Loading and Standardizing Input...")
    try
        input_stream = process_input(loader, sample_id)

        println("[2/3] Estimation Layer: Running Inference...")
        trajectory = predict_trajectory(estimator, input_stream.sensor)

        println("[3/3] Application Layer: Launching Dashboard...")
        run_app(app, trajectory, input_stream)

    catch e
        println("Error during execution: ", e)
        rethrow(e)
    end
end

# Example usage:
run_analysis("sample_001_seg0")

println("Analysis complete. Press Enter to exit...")
readline()
