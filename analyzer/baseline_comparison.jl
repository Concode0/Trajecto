"""
Baseline Model Comparison Script

This script demonstrates how to run and compare different trajectory estimation models:
1. pure_integration - Simple dead reckoning (double integration, no filtering)
2. pure_eskf - Physics-based ESKF without TCN corrections
3. eskf_tcn - Full hybrid model (ESKF + TCN)

Usage:
    julia --project=. analyzer/baseline_comparison.jl

The script will:
- Load the same sample data for all models
- Run inference with each model
- Display trajectories side-by-side for comparison
- Calculate error metrics (if ground truth is available)
"""

using Pkg
Pkg.activate(@__DIR__)

using TrajectoLab

# Configuration
const PROJECT_ROOT = dirname(@__DIR__)
const VENV_PATH = joinpath(PROJECT_ROOT, ".venv", "bin", "python")
ENV["JULIA_PYTHONCALL_EXE"] = VENV_PATH

# Data paths
h5_path = joinpath(PROJECT_ROOT, "data/validation_dataset.h5")
model_path = joinpath(PROJECT_ROOT, "eskf_tcn_model.pth")
script_path = joinpath(PROJECT_ROOT, "model")
scaler_path = joinpath(PROJECT_ROOT, "data/scaler_stats.h5")

# Sample to analyze
sample_id = "sample_002_seg0"

println("=" ^ 80)
println("TRAJECTO BASELINE MODEL COMPARISON")
println("=" ^ 80)
println("Sample: $sample_id")
println()

# Load data once (shared across all models)
println("[1/4] Loading data from HDF5...")
loader = HDF5Perception(h5_path)
input_stream = process_input(loader, sample_id)
println("   ✓ Loaded sensor data: $(size(input_stream.sensor))")
println("   ✓ Loaded ground truth: $(size(input_stream.gt_pos))")
println()

# Define models to compare
models_config = [
    (name="Pure Integration", type="pure_integration", path=""),
    (name="Pure ESKF", type="pure_eskf", path=""),
    (name="ESKF-TCN Hybrid", type="eskf", path=model_path),
]

# Storage for results
results_for_dashboard = []

# Run each model
for (idx, config) in enumerate(models_config)
    println("[$idx/$(length(models_config))] Running $(config.name)...")

    try
        # Create estimator for this model type
        estimator = TrajectoEstimator(config.type, config.path, script_path, scaler_path)

        # Run inference
        trajectory = predict_trajectory(estimator, input_stream.sensor)

        # Store results
        push!(results_for_dashboard, (name=config.name, trajectory=trajectory))

        println("   ✓ Generated trajectory: $(size(trajectory.pos))")

        # Calculate drift metrics
        start_pos = trajectory.pos[1, :]
        end_pos = trajectory.pos[end, :]
        total_drift = sqrt(sum((end_pos - start_pos).^2))
        println("   ✓ Total drift: $(round(total_drift * 1000, digits=1)) mm")

    catch e
        println("   ✗ Error: $e")
        push!(results_for_dashboard, (name=config.name, trajectory=nothing))
    end
    println()
end

# Display comparison summary
println("=" ^ 80)
println("COMPARISON SUMMARY")
println("=" ^ 80)

for res in results_for_dashboard
    if res.trajectory !== nothing
        traj = res.trajectory
        drift = sqrt(sum((traj.pos[end, :] - traj.pos[1, :]).^2)) * 1000
        println("$(res.name):")
        println("  Drift: $(round(drift, digits=2)) mm")

        # Calculate path length
        path_length = 0.0
        for i in 2:size(traj.pos, 1)
            path_length += sqrt(sum((traj.pos[i, :] - traj.pos[i-1, :]).^2))
        end
        println("  Path length: $(round(path_length * 1000, digits=1)) mm")
        println()
    end
end

println("=" ^ 80)
println("VISUALIZATION")
println("=" ^ 80)
println("Launching interactive dashboard for visual comparison...")
println()

app = MakieDashboard()
run_app_comparison(app, results_for_dashboard, input_stream)

println()
println("Analysis complete.")
