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
script_path = joinpath(PROJECT_ROOT, "model")
scaler_path = joinpath(PROJECT_ROOT, "data/scaler_stats.h5")

# ============================================================================
# COMPARISON MODE CONFIGURATION
# ============================================================================
# Set to true to compare multiple models side-by-side
const COMPARE_MODE = false

# Models to compare (used when COMPARE_MODE = true)
# Format: (name, type, model_path)
const MODELS_TO_COMPARE = [
    ("Pure Integration", "pure_integration", ""),  # Empty path for baselines
    ("Pure ESKF", "pure_eskf", ""),
    ("ESKF-TCN", "eskf", joinpath(PROJECT_ROOT, "eskf_tcn_model.pth"))
]

# Single model configuration (used when COMPARE_MODE = false)
const MODEL_TYPE = "pure_eskf"
const MODEL_PATH = ""  # Empty for baselines, path to .pth for trained models

# Sample to analyze
const SAMPLE_ID = "sample_003_seg0"

#
# EXECUTION PIPELINE
# ============================================================================

# 3. Instantiate Perception Layer (shared across all models)
loader = HDF5Perception(h5_path)

# 4A. Single Model Analysis
function run_single_analysis(sample_id::String)
    println("=" ^ 80)
    println("TRAJECTO SINGLE MODEL ANALYSIS")
    println("=" ^ 80)
    println("Model Type: $MODEL_TYPE")
    println("Sample: $sample_id")
    println()

    println("[1/3] Perception Layer: Loading and Standardizing Input...")
    input_stream = process_input(loader, sample_id)
    println("   ✓ Loaded sensor data: $(size(input_stream.sensor))")
    println("   ✓ Loaded ground truth: $(size(input_stream.gt_pos))")
    println("   ✓ Actual sequence length: $(input_stream.seq_len) (padding removed)")
    println()

    println("[2/3] Estimation Layer: Running Inference...")
    estimator = TrajectoEstimator(MODEL_TYPE, MODEL_PATH, script_path, scaler_path)
    trajectory = predict_trajectory(estimator, input_stream.sensor)
    println("   ✓ Generated trajectory: $(size(trajectory.pos))")
    println()

    # --- CRLB Analysis ---
    if MODEL_TYPE in ["eskf", "aekf", "pure_eskf"]
        println("   > Running CRLB Analysis...")
        try
            # Extract sensor data
            acc = input_stream.sensor[:, 1:3]
            gyro = input_stream.sensor[:, 4:6]
            
            # Use estimated rotation and ZUPT for linearization
            rot = trajectory.rot
            zupt = trajectory.zupt .> 0.5 # Threshold probability
            
            config = CRLBConfig() # Use defaults
            crlb = compute_crlb(acc, gyro, rot, zupt, Config.DEFAULT_DT, config)
            
            # Compare uncertainties at the end of trajectory
            final_crlb_pos = norm(crlb.std_pos[end, :])
            final_est_std = sqrt(tr(trajectory.cov[end, 1:3, 1:3]))
            
            # Calculate actual error (drift)
            drift = norm(trajectory.pos[end, :] - trajectory.pos[1, :]) # Simplified drift
            # Or better: error w.r.t Ground Truth (if aligned) - but here we check drift/uncertainty consistency
            
            println("     CRLB Position Std (End): $(round(final_crlb_pos * 1000, digits=2)) mm")
            println("     Est. Position Std (End): $(round(final_est_std * 1000, digits=2)) mm")
            println("     Actual Drift (End):      $(round(drift * 1000, digits=2)) mm")
            
            if final_est_std < final_crlb_pos
                println("     ⚠ WARNING: Estimator is over-confident (Est < CRLB)")
            else
                println("     ✓ Estimator is consistent (Est >= CRLB)")
            end
        catch e
            println("     ⚠ CRLB Analysis failed: $e")
        end
        println()
    end
    # ---------------------

    println("[3/3] Application Layer: Launching Dashboard...")
    app = MakieDashboard()
    run_app(app, trajectory, input_stream)

    println()
    println("Analysis complete.")
end

# 4B. Comparison Analysis
function run_comparison_analysis(sample_id::String)
    println("=" ^ 80)
    println("TRAJECTO MULTI-MODEL COMPARISON")
    println("=" ^ 80)
    println("Comparing $(length(MODELS_TO_COMPARE)) models")
    println("Sample: $sample_id")
    println()

    # Load data once (shared across all models)
    println("[1/$(length(MODELS_TO_COMPARE) + 2)] Loading data...")
    input_stream = process_input(loader, sample_id)
    println("   ✓ Loaded sensor data: $(size(input_stream.sensor))")
    println("   ✓ Loaded ground truth: $(size(input_stream.gt_pos))")
    println("   ✓ Actual sequence length: $(input_stream.seq_len) (padding removed)")
    println()

    # Run each model
    results = []
    for (idx, (name, type, path)) in enumerate(MODELS_TO_COMPARE)
        println("[$(idx + 1)/$(length(MODELS_TO_COMPARE) + 2)] Running $name ($type)...")

        try
            estimator = TrajectoEstimator(type, path, script_path, scaler_path)
            trajectory = predict_trajectory(estimator, input_stream.sensor)

            # Calculate metrics
            drift = sqrt(sum((trajectory.pos[end, :] - trajectory.pos[1, :]).^2)) * 1000

            println("   ✓ Generated trajectory: $(size(trajectory.pos))")
            println("   ✓ Drift: $(round(drift, digits=2)) mm")

            push!(results, (name=name, trajectory=trajectory))
        catch e
            println("   ✗ Error: $e")
            push!(results, (name=name, trajectory=nothing))
        end
        println()
    end

    # Display comparison
    println("[$(length(MODELS_TO_COMPARE) + 2)/$(length(MODELS_TO_COMPARE) + 2)] Launching Comparison Dashboard...")
    println()

    app = MakieDashboard()
    run_app_comparison(app, results, input_stream)

    println()
    println("Comparison complete.")
end


# ============================================================================
# MAIN EXECUTION
# ============================================================================

try
    if COMPARE_MODE
        run_comparison_analysis(SAMPLE_ID)
    else
        run_single_analysis(SAMPLE_ID)
    end
catch e
    println("Error during execution: ", e)
    rethrow(e)
end

println()
println("Press Enter to exit...")
readline()
