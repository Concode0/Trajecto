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
Hyperparameter Tuning Wrapper for HPO Integration

This script acts as a bridge between the HPO agent and the training script (`train_eskf.py`).
It parses command-line arguments provided by the HPO agent, configures the `TrainConfig`
object with these hyperparameters, and runs the training loop.

Crucially, it adapts the HPO's search space (e.g., `tcn_channel_size` as a single int)
to the model's expected format (e.g., `tcn_channels` as a list of 4 ints).

Output Format:
    The final validation loss is printed to stdout as: "FINAL_LOSS: <value>"
    This allows the HPO agent to parse the result.
"""

import argparse
import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_eskf import train, TrainConfig
from model.config import Config

def main():
    parser = argparse.ArgumentParser(description="Tune ESKF-TCN Hyperparameters")

    # Training loop parameters (often fixed or low for tuning speed)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for tuning run")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="data/hpo_dataset.h5")
    parser.add_argument("--val_dataset", type=str, default="data/hpo_dataset.h5")

    # Hyperparameters from HPO/kernel.py PARAM_SPACE
    # Architecture
    parser.add_argument("--dropout", type=float, default=Config.ESKFTCN.DROPOUT)
    parser.add_argument("--kernel_size", type=int, default=Config.ESKFTCN.KERNEL_SIZE)
    parser.add_argument("--tcn_channel_size", type=int, default=96, help="Single channel size for all layers")

    # Regularization
    parser.add_argument("--reg_weight", type=float, default=Config.LOSS.REG_WEIGHT_ESKF_TCN)

    # ESKF
    parser.add_argument("--mahalanobis_threshold", type=float, default=Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD)

    # Loss weights (initial values for DWA)
    parser.add_argument("--w_mag", type=float, default=1.0)
    parser.add_argument("--w_cos", type=float, default=1.0)
    parser.add_argument("--w_zupt", type=float, default=0.5)
    parser.add_argument("--w_cov", type=float, default=0.01)
    parser.add_argument("--w_fft", type=float, default=0.5)
    parser.add_argument("--w_delta", type=float, default=0.5)

    # ZUPT
    parser.add_argument("--zupt_vel_threshold", type=float, default=0.005)

    args = parser.parse_args()

    # Adapt HPO parameters to Model Configuration
    # The model expects a list of 4 channels for its Y-shaped architecture
    tcn_channels = [args.tcn_channel_size] * 4

    # Create Configuration
    config = TrainConfig(
        # Data
        dataset_path=args.dataset,
        val_dataset_path=args.val_dataset,

        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,

        # Loss weights (initial values for DWA)
        w_mag=args.w_mag,
        w_cos=args.w_cos,
        w_zupt=args.w_zupt,
        w_cov=args.w_cov,
        w_fft=args.w_fft,
        w_delta=args.w_delta,
        w_reg=args.reg_weight,

        # Model Hyperparameters
        tcn_channels=tcn_channels,
        tcn_kernel_size=args.kernel_size,
        tcn_dropout=args.dropout,
        mahalanobis_threshold=args.mahalanobis_threshold,
        zupt_vel_threshold=args.zupt_vel_threshold,

        # Output (Use a temp dir or specific run dir to avoid clutter)
        checkpoint_dir="hpo_checkpoints",
        model_name=f"hpo_lr{args.lr:.1e}_k{args.kernel_size}_c{args.tcn_channel_size}"
    )

    try:
        # Run Training
        final_loss = train(config)

        # Report Result to HPO Agent
        print(f"FINAL_LOSS: {final_loss}")

    except Exception as e:
        print(f"Training failed with error: {e}", file=sys.stderr)
        # Return a high loss to indicate failure, or let the agent handle the non-zero exit code
        print("FINAL_LOSS: 999.99")
        sys.exit(1)

if __name__ == "__main__":
    main()
