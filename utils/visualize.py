"""This module provides functions for visualizing model predictions against ground
truth data.

It includes functions to load a trained model, generate trajectory predictions for a
given sample, and plot the results in 2D and 3D space, as well as an axis-wise
comparison over time.
"""

import argparse
import os
from typing import Any, Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.AEKF_TCN import AEKFTCN_model
from model.ESKF_TCN import ESKFTCN_model
from model.onlyTCN import TCN as OnlyTCN
from model.dataset import TrajectoryDataset


def load_model(
    model_name: str, model_path: str, device: str, dt: float
) -> torch.nn.Module:
    """Initializes a model and loads its weights from a file.

    Args:
        model_name: The name of the model to load.
            Choices are 'eskf_tcn', 'aekf_tcn', 'only_tcn'.
        model_path: The path to the saved model weights.
        device: The device to load the model on ('cpu' or 'cuda').
        dt: The time step used for the model's internal calculations.

    Returns:
        The loaded and initialized model.

    Raises:
        ValueError: If the model_name is not recognized.
        FileNotFoundError: If the model_path does not exist.
    """
    model_configs: Dict[str, Dict[str, Any]] = {
        "eskf_tcn": {"tcn_input_size": 20, "use_zupt": False, "use_tcn_zupt": False},
        "aekf_tcn": {"tcn_input_size": 20, "use_zupt": False},
        "only_tcn": {"input_size": 7, "output_size": 3},
    }

    config = model_configs.get(model_name)
    if not config:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Loading {model_name} from {model_path}...")

    if model_name == "eskf_tcn":
        model: torch.nn.Module = ESKFTCN_model(device=device, dt=dt, **config)
    elif model_name == "aekf_tcn":
        model = AEKFTCN_model(device=device, dt=dt, **config)
    elif model_name == "only_tcn":
        model = OnlyTCN(
            device=device, dt=dt, **config
        )  # dt is required by this model
    else:
        # This case should not be reached due to the initial check
        raise ValueError(f"Unknown model name: {model_name}")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")


def plot_trajectory(
    sample_idx: int,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    dt: float,
    save_dir: str = "results",
) -> None:
    """Visualizes the ground truth and predicted trajectories.

    This function generates and saves three plots:
    1. A 2D top-down view of the XY plane.
    2. A 3D plot of the trajectory.
    3. An axis-wise comparison of X, Y, and Z positions over time.

    Args:
        sample_idx: The index of the sample being plotted, used for titles
            and filenames.
        gt_pos: The ground truth position data, shape (T, 3).
        pred_pos: The predicted position data, shape (T, 3).
        dt: The time step between data points, used for the time axis.
        save_dir: The directory where the plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 2D Plot (XY Plane - Top Down View)
    plt.figure(figsize=(10, 8))
    plt.plot(
        gt_pos[:, 0], gt_pos[:, 1], "k--", label="Ground Truth", linewidth=2
    )
    plt.plot(
        pred_pos[:, 0],
        pred_pos[:, 1],
        "r-",
        label="Predicted",
        linewidth=1.5,
    )
    plt.title(f"Sample {sample_idx}: XY Plane Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_2d.png"))
    plt.show()

    # 2. 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], "k--", label="Ground Truth")
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], "r-", label="Predicted")
    ax.set_title(f"Sample {sample_idx}: 3D Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_3d.png"))
    plt.show()

    # 3. Axis-wise Comparison (Time-series)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    time = np.arange(len(gt_pos)) * dt

    axes[0].plot(time, gt_pos[:, 0], "k--", label="GT X")
    axes[0].plot(time, pred_pos[:, 0], "r-", label="Pred X")
    axes[0].set_ylabel("X (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(time, gt_pos[:, 1], "k--", label="GT Y")
    axes[1].plot(time, pred_pos[:, 1], "r-", label="Pred Y")
    axes[1].set_ylabel("Y (m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    axes[2].plot(time, gt_pos[:, 2], "k--", label="GT Z")
    axes[2].plot(time, pred_pos[:, 2], "r-", label="Pred Z")
    axes[2].set_ylabel("Z (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)

    plt.suptitle(f"Sample {sample_idx}: Axis-wise Drift Check")
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_axis.png"))
    plt.show()


def main() -> None:
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize model trajectory predictions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eskf_tcn",
        choices=["eskf_tcn", "aekf_tcn", "only_tcn"],
        help="The name of the model to evaluate.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.h5",
        help="Path to the preprocessed HDF5 dataset.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="data/scaler_stats.h5",
        help="Path to the HDF5 file with scaler statistics (mean/std).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="eskf_tcn_model.pth",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample to visualize.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g., 'cpu', 'cuda').",
    )
    args = parser.parse_args()

    # Define dt, as it's a fixed property of the dataset after preprocessing.
    # Note: If this changes, it should be loaded from metadata.
    dt = 0.0025 * 4  # 0.0025s original, subsampled by 4

    # 1. Load Dataset
    dataset = TrajectoryDataset(args.data)
    if args.sample_idx >= len(dataset):
        print(
            f"Error: Sample index {args.sample_idx} is out of range. "
            f"The dataset has {len(dataset)} samples."
        )
        return

    # 2. Get Data Sample
    data = dataset[args.sample_idx]
    imu_raw = data["imu_seq_raw"].unsqueeze(0).to(args.device)
    gt_pos = data["gt_pos_w"].numpy()

    # 3. Load Scaler and Normalize Data
    with h5py.File(args.scaler, "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float().to(args.device)
        std = torch.from_numpy(f["std"][:]).float().to(args.device)
    imu_norm = (imu_raw - mean) / (std + 1e-9)

    # 4. Load Model & Run Inference
    model = load_model(args.model, args.model_path, args.device, dt=dt)

    print(f"Running inference on sample {args.sample_idx}...")
    with torch.no_grad():
        if args.model == "only_tcn":
            # The 'OnlyTCN' model directly outputs the position tensor.
            pred_pos = model(imu_raw, imu_norm)
        else:
            # Hybrid models return a dictionary; we need the 'pred_pos_w' key.
            outputs = model(imu_raw, imu_norm)
            pred_pos = outputs["pred_pos_w"]

    pred_pos = pred_pos.squeeze(0).cpu().numpy()

    # 5. Calculate Error Metrics
    # Note: For a more accurate error, one might exclude padded sections.
    # Here, we calculate over the entire sequence for simplicity.
    mse = np.mean((gt_pos - pred_pos) ** 2)
    rmse = np.sqrt(mse)
    end_error = np.linalg.norm(gt_pos[-1, :2] - pred_pos[-1, :2])

    print(f"\n--- Metrics for Sample {args.sample_idx} ---")
    print(f"RMSE (All axes): {rmse:.4f} m")
    print(f"2D End-point Error: {end_error*100:.2f} cm")

    # 6. Plotting
    plot_trajectory(args.sample_idx, gt_pos, pred_pos, dt)


if __name__ == "__main__":
    main()