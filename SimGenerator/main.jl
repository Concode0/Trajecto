# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

#!/usr/bin/env julia
"""
SimGenerator: Synthetic Trajecto Data Generator for Sim2Real Training.

Three-layer architecture:
1. Kinematic Layer: Sigma-LogNormal trajectory generation
2. Posture Layer: Wrist/arm biomechanics simulation
3. Sensing Layer: IMU + force sensor modeling

Usage:
    julia --project=. main.jl --samples 100 --output data/simulated_dataset.h5
    julia --project=. main.jl -n 50 --strokes 15 --duration 8.0
"""

using Pkg
Pkg.activate(@__DIR__)

using ProgressMeter
using Random

# Include the SimGenerator module
include("src/SimGenerator.jl")
using .SimGenerator

function parse_commandline()
    # Simple argument parsing without ArgParse dependency
    args = Dict{String,Any}(
        "samples" => 100,
        "output" => joinpath(dirname(@__DIR__), "data", "simulated_dataset.h5"),
        "duration" => DEFAULT_WRITING_DURATION,
        "strokes" => DEFAULT_NUM_STROKES,
        "seed" => 42,
        "verbose" => false,
        "stats" => false,
        "visualize" => false,
        "visualize_path" => nothing,
        "random_grip" => true,
        "random_handedness" => true
    )

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ["--samples", "-n"]
            args["samples"] = parse(Int, ARGS[i+1])
            i += 2
        elseif arg in ["--output", "-o"]
            args["output"] = ARGS[i+1]
            i += 2
        elseif arg == "--duration"
            args["duration"] = parse(Float64, ARGS[i+1])
            i += 2
        elseif arg == "--strokes"
            args["strokes"] = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--seed"
            args["seed"] = parse(Int, ARGS[i+1])
            i += 2
        elseif arg in ["--verbose", "-v"]
            args["verbose"] = true
            i += 1
        elseif arg == "--stats"
            args["stats"] = true
            i += 1
        elseif arg == "--visualize"
            args["visualize"] = true
            i += 1
        elseif arg == "--visualize-path"
            args["visualize_path"] = ARGS[i+1]
            i += 2
        elseif arg == "--no-random-grip"
            args["random_grip"] = false
            i += 1
        elseif arg == "--no-random-handedness"
            args["random_handedness"] = false
            i += 1
        elseif arg == "--random-grip"
            args["random_grip"] = true
            i += 1
        elseif arg == "--random-handedness"
            args["random_handedness"] = true
            i += 1
        elseif arg in ["--help", "-h"]
            println("""
Usage: julia --project=. main.jl [options]

Options:
  -n, --samples N       Number of samples to generate (default: 100)
  -o, --output PATH     Output HDF5 file path (default: data/simulated_dataset.h5)
  --duration SECS       Writing duration per sample in seconds (default: 5.0)
  --strokes N           Number of strokes per trajectory (default: 10)
  --seed N              Random seed for reproducibility (default: 42)
  -v, --verbose         Verbose output
  --stats               Compute and print statistics for config.py
  --visualize           Generate visualization for first sample
  --visualize-path PATH Save visualization to specified path
  --random-grip         Randomize grip style for each sample (default: true)
  --no-random-grip      Disable randomization of grip style
  --random-handedness   Randomize handedness for each sample (default: true)
  --no-random-handedness Disable randomization of handedness
  -h, --help            Show this help message

Examples:
  # Generate 100 samples with statistics
  julia --project=. main.jl -n 100 --stats

  # Generate with visualization
  julia --project=. main.jl -n 10 --visualize --visualize-path output.png

  # Generate for training with specific parameters
  julia --project=. main.jl -n 500 --strokes 15 --duration 8.0 -o data/train_sim.h5

  # Generate with random grip and handedness (default)
  julia --project=. main.jl -n 100
""")
            exit(0)
        else
            println("Unknown argument: $arg")
            exit(1)
        end
    end

    return args
end

function main()
    args = parse_commandline()

    println("=" ^ 60)
    println("SimGenerator: Synthetic Trajecto Data Generator")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Samples:     $(args["samples"])")
    println("  Output:      $(args["output"])")
    println("  Duration:    $(args["duration"]) s")
    println("  Strokes:     $(args["strokes"])")
    println("  Seed:        $(args["seed"])")
    println("  Random Grip: $(args["random_grip"])")
    println("  Random Hand: $(args["random_handedness"])")
    println()

    # Set random seed
    rng = Random.MersenneTwister(args["seed"])

    # Construct simulation parameters
    # Copy defaults and apply overrides
    sim_params = SimulationParams(
        DEFAULT_SIM_PARAMS.pen_lift_params,
        DEFAULT_SIM_PARAMS.use_superposition,
        DEFAULT_SIM_PARAMS.arm_params,
        DEFAULT_SIM_PARAMS.joint_limits,
        DEFAULT_SIM_PARAMS.ik_params,
        DEFAULT_SIM_PARAMS.use_ik,
        DEFAULT_SIM_PARAMS.grip_style,
        DEFAULT_SIM_PARAMS.handedness,
        args["random_grip"],          # Override
        args["random_handedness"],    # Override
        DEFAULT_SIM_PARAMS.world_rotation_params,
        DEFAULT_SIM_PARAMS.tremor_params,
        DEFAULT_SIM_PARAMS.pink_params,
        DEFAULT_SIM_PARAMS.baseline_params,
        DEFAULT_SIM_PARAMS.use_enhanced_noise
    )

    # Generate samples
    println("Generating $(args["samples"]) synthetic samples...")
    samples = Vector{TrajectoryData}(undef, args["samples"])

    @showprogress for i in 1:args["samples"]
        samples[i] = generate_sample(;
            num_strokes=args["strokes"],
            duration=args["duration"],
            label="sim_sample_$(lpad(i-1, 3, '0'))",
            sim_params=sim_params,
            rng=rng
        )

        if args["verbose"]
            seq_len = samples[i].sequence_length
            println("  Sample $i: seq_len=$seq_len")
        end
    end

    # Export to HDF5
    println()
    println("Exporting to HDF5: $(args["output"])")
    export_hdf5(args["output"], samples)

    println()
    println("Done! Generated $(length(samples)) samples.")
    println("  Total timesteps: $(sum(s.sequence_length for s in samples))")
    println("  Output file: $(args["output"])")

    # Compute and print statistics if requested
    if args["stats"]
        println()
        stats = compute_dataset_statistics(samples)
        print_statistics_summary(stats)
        print_config_update(stats)
    end

    # Generate visualization if requested
    if args["visualize"]
        println()
        println("Generating 6DOF visualization...")

        # Visualize first sample
        if !isempty(samples)
            save_path = args["visualize_path"]
            if save_path === nothing
                save_path = replace(args["output"], ".h5" => "_viz.png")
            end

            visualize_6dof(samples[1]; save_path=save_path, title="Sample 1 - 6DOF Motion")
            println("Visualization saved to: $save_path")

            # Also generate batch summary if multiple samples
            if length(samples) > 1
                batch_path = replace(save_path, ".png" => "_batch_summary.png")
                visualize_batch_summary(samples; save_path=batch_path)
                println("Batch summary saved to: $batch_path")
            end
        end
    end
end

main()
