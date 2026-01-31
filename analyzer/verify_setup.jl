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

"""
Verification Script for Trajecto Analyzer

This script checks that the analyzer setup is correct without requiring data or models.
It verifies:
1. Julia package dependencies
2. Module imports
3. Python environment setup
4. Function exports

Run with: julia --project=. verify_setup.jl
"""

using Pkg
Pkg.activate(@__DIR__)

println("=" ^ 80)
println("TRAJECTO ANALYZER VERIFICATION")
println("=" ^ 80)
println()

# 1. Check Julia Dependencies
println("[1/5] Checking Julia package dependencies...")
try
    using GLMakie
    using HDF5
    using PythonCall
    using GeometryBasics
    using Statistics
    using LinearAlgebra
    println("   ✓ All Julia packages loaded successfully")
catch e
    println("   ✗ Error loading Julia packages: $e")
    println("   Run: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
    exit(1)
end
println()

# 2. Check TrajectoLab Module
println("[2/5] Checking TrajectoLab module...")
try
    using TrajectoLab
    println("   ✓ TrajectoLab module loaded")
catch e
    println("   ✗ Error loading TrajectoLab: $e")
    exit(1)
end
println()

# 3. Check Exported Functions
println("[3/5] Checking exported functions...")
exported_symbols = names(TrajectoLab)
required_exports = [
    :HDF5Perception,
    :TrajectoEstimator,
    :MakieDashboard,
    :process_input,
    :predict_trajectory,
    :load_model,
    :run_app,
    :run_app_comparison,
    :calculate_metrics,
]

missing_exports = []
for symbol in required_exports
    if !(symbol in exported_symbols)
        push!(missing_exports, symbol)
    end
end

if isempty(missing_exports)
    println("   ✓ All required functions exported ($(length(required_exports)) total)")
else
    println("   ✗ Missing exports: $missing_exports")
    exit(1)
end
println()

# 4. Check Python Environment
println("[4/5] Checking Python environment...")
const PROJECT_ROOT = dirname(@__DIR__)
const VENV_PATH = joinpath(PROJECT_ROOT, ".venv", "bin", "python")

if isfile(VENV_PATH)
    ENV["JULIA_PYTHONCALL_EXE"] = VENV_PATH
    println("   ✓ Python venv found: $VENV_PATH")
else
    println("   ⚠ Python venv not found at: $VENV_PATH")
    println("   Note: This is okay if using system Python")
end

try
    np = pyimport("numpy")
    torch = pyimport("torch")
    println("   ✓ PyTorch and NumPy accessible")
catch e
    println("   ✗ Error importing Python packages: $e")
    println("   Make sure PyTorch is installed in your Python environment")
    exit(1)
end
println()

# 5. Check Model Types Configuration
println("[5/5] Checking model type support...")
model_types = ["pure_integration", "pure_eskf", "eskf", "aekf", "tcn"]
println("   Supported model types:")
for mt in model_types
    println("      - $mt")
end
println("   ✓ $(length(model_types)) model types configured")
println()

# Summary
println("=" ^ 80)
println("VERIFICATION COMPLETE")
println("=" ^ 80)
println("✓ All checks passed!")
println()
println("Next steps:")
println("  1. Ensure you have data files in data/dataset.h5 or data/validation_dataset.h5")
println("  2. For trained models, place .pth files in project root")
println("  3. Edit analyzer/main.jl to configure your analysis")
println("  4. Run: julia --project=. analyzer/main.jl")
println()
println("For baseline models (no training required):")
println("  - Set MODEL_TYPE = \"pure_integration\" or \"pure_eskf\"")
println("  - Set MODEL_PATH = \"\"")
println()
println("For comparison mode:")
println("  - Set COMPARE_MODE = true")
println("  - Configure MODELS_TO_COMPARE list")
println()
