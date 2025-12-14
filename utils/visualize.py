"""This module provides functions for visualizing model predictions against ground
truth data.

It includes functions to load a trained model, generate trajectory predictions for a
given sample, and plot the results in 2D and 3D space, as well as an axis-wise
comparison over time.
"""

import argparse
import os, sys
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from torch.utils.data import DataLoader

# Add parent directory to sys.path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.AEKF_TCN import AEKFTCN_model
from model.ESKF_TCN import ESKFTCN_model
from model.onlyTCN import OnlyTCN
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

def align_trajectories(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align Pred and GT Trajectory using SVD with Scaling (Umeyama Algorithm).

    Args:
        gt (np.ndarray): Ground truth trajectory.
            - Shape: (N, 3)
            - Unit: Meter
            - Frame: World
        pred (np.ndarray): Predicted trajectory.
            - Shape: (N, 3)
            - Unit: Meter
            - Frame: Arbitrary (aligned to World in output)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - gt_centered: Centered ground truth trajectory.
                - Shape: (N, 3) | Unit: Meter | Frame: World (Centered)
            - pred_aligned: Aligned predicted trajectory.
                - Shape: (N, 3) | Unit: Meter | Frame: World (Centered)
    """

    gt_mean = np.mean(gt, axis=0)
    pred_mean = np.mean(pred, axis=0)
    gt_centered = gt - gt_mean
    pred_centered = pred - pred_mean

    gt_std = np.linalg.norm(gt_centered)
    pred_std = np.linalg.norm(pred_centered)
    scale = gt_std / pred_std if pred_std > 1e-6 else 1.0

    print(f"[Info] Auto-Scaling Factor: {scale:.4f} (Pred를 GT 크기에 맞춤)")

    pred_scaled = pred_centered * scale
    H = np.dot(pred_scaled.T, gt_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    pred_aligned = np.dot(pred_scaled, R.T)

    return gt_centered, pred_aligned

def plot_uncertainty_ellipse(
    ax: plt.Axes,
    last_pos: np.ndarray,
    last_cov: np.ndarray,
    n_std: float = 2.0,
    facecolor: str = "blue",
    edgecolor: str = "blue",
    alpha: float = 0.2,
    **kwargs,
) -> None:
    """Plots a 2D uncertainty ellipse for a given position and covariance."""
    cov_xy = last_cov[0:2, 0:2]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)

    # Get the angle of the major axis
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Eigenvalues are variance, so sqrt gives std dev
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create the ellipse patch
    ellipse = Ellipse(
        xy=last_pos[0:2],
        width=width,
        height=height,
        angle=np.degrees(angle),
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        **kwargs,
    )
    ax.add_patch(ellipse)

def plot_trajectory(
    sample_idx: int,
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    gt_vel: np.ndarray,
    pred_vel: np.ndarray,
    pred_zupt: Optional[np.ndarray],
    dt: float,
    save_dir: str = "results",
    pred_cov: Optional[np.ndarray] = None,
) -> None:
    """Visualizes the ground truth and predicted trajectories.

    This function generates and saves three plots:
    1. A 2D top-down view of the XY plane.
    2. A 3D plot of the trajectory.
    3. An axis-wise comparison of X, Y, and Z positions over time.

    Args:
        sample_idx (int): The index of the sample being plotted.
        gt_pos (np.ndarray): The ground truth position data.
            - Shape: (T, 3)
            - Unit: Meter
            - Frame: World
        pred_pos (np.ndarray): The predicted position data.
            - Shape: (T, 3)
            - Unit: Meter
            - Frame: World
        gt_vel (np.ndarray): The ground truth velocity data.
            - Shape: (T, 3)
            - Unit: m/s
            - Frame: World
        pred_vel (np.ndarray): The predicted velocity data.
            - Shape: (T, 3)
            - Unit: m/s
            - Frame: World
        pred_zupt (Optional[np.ndarray]): Predicted ZUPT probabilities.
            - Shape: (T,) or (T, 1)
            - Range: [0, 1]
        dt (float): The time step between data points (s).
        save_dir (str): The directory where the plots will be saved.
        pred_cov (Optional[np.ndarray]): Predicted Covariance diagonal elements.
            - Shape: (T, 6)
    """
    os.makedirs(save_dir, exist_ok=True)
    time = np.arange(len(gt_pos)) * dt
    # 1. 2D Trajectory ( Aligned )
    gt_align, pred_align = align_trajectories(gt_pos, pred_pos)
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(gt_align[:, 0], gt_align[:, 1], "k--", label="GT", alpha=0.7)
    ax1.plot(pred_align[:, 0], pred_align[:, 1], "r-", label="Pred (Aligned)")
    if pred_cov is not None:
        # Plot every 50th covariance ellipse to avoid clutter
        for i in range(0, len(pred_pos), 50):
            # We need to rotate the covariance as well
            # This is a bit tricky, so we'll skip it for now
            pass
    ax1.set_title("Aligned Trajectory")
    ax1.legend()
    ax1.axis("equal")
    ax1.grid(True)
    plt.suptitle(f"Sample {sample_idx}: 2D Trajectory")
    plt.savefig(f"{save_dir}/sample_{sample_idx}_2d_trajectory.png")
    plt.close()
    # 2. Velocity & ZUPT Analysis
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    gt_speed = np.linalg.norm(gt_vel, axis=1)
    pred_speed = np.linalg.norm(pred_vel, axis=1)
    ax[0].plot(time, gt_speed, "k--", label="GT Speed")
    ax[0].plot(time, pred_speed, "g-", label="Pred Speed")
    ax[0].set_ylabel("Speed (m/s)")
    ax[0].set_title("Velocity Tracking")
    ax[0].legend()
    ax[0].grid(True)
    if pred_zupt is not None:
        ax[1].plot(time, pred_zupt, "m-", label="ZUPT Prob", linewidth=1.5)
        ax[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax[1].set_ylabel("Probability")
        ax[1].set_title("Zero-Velocity Detection (Model Output)")
        ax[1].legend()
        ax[1].grid(True)
    # 3. Position Error over Time (Drift Analysis)
    error = np.linalg.norm(gt_pos - pred_pos, axis=1)
    ax[2].plot(time, error, "r-", label="Pos Error")
    ax[2].set_ylabel("Error (m)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_title("Drift Accumulation over Time")
    ax[2].legend()
    ax[2].grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{sample_idx}_physics.png")
    plt.close()
    # 4. 3D Trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_align[:, 0], gt_align[:, 1], gt_align[:, 2], "k--", label="GT")
    ax.plot(pred_align[:, 0], pred_align[:, 1], pred_align[:, 2], "r-", label="Pred")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title(f"Sample {sample_idx}: 3D Trajectory (Aligned)")
    plt.savefig(f"{save_dir}/sample_{sample_idx}_3d_trajectory.png")
    plt.close()
    # 5. Axis-wise plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i in range(3):
        axes[i].plot(time, gt_align[:, i], "k--", label="GT")
        axes[i].plot(time, pred_align[:, i], "r-", label="Pred")
        axes[i].set_ylabel(f"{labels[i]} (m)")
        axes[i].legend()
        axes[i].grid(True)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Sample {sample_idx}: Axis-wise Position (Aligned)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{sample_idx}_axis_wise.png")
    plt.close()
    print(f"Saved plots to {save_dir}/")
    plt.show()


def main() -> None:
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="eskf_tcn")
    parser.add_argument("--data", type=str, default="data/dataset.h5")
    parser.add_argument("--scaler", type=str, default="data/scaler_stats.h5")
    parser.add_argument("--model_path", type=str, default="eskf_tcn_model.pth")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Define dt, as it's a fixed property of the dataset after preprocessing.
    # Note: If this changes, it should be loaded from metadata.
    dt = 0.0025 * 8  # 0.0025s original, subsampled by 4

    # 1. Load Dataset
    dataset = TrajectoryDataset(args.data)
    data = dataset[args.sample_idx]

    # 2. Get Data Sample
    imu_raw = data["imu_seq_raw"].unsqueeze(0).to(args.device)
    gt_pos = data["gt_pos_w"].numpy()
    gt_vel = data["gt_vel_w"].numpy()

    # 3. Load Scaler and Normalize Data
    with h5py.File(args.scaler, "r") as f:
        mean = torch.from_numpy(f["mean"][:]).float().to(args.device)
        std = torch.from_numpy(f["std"][:]).float().to(args.device)
    imu_norm = (imu_raw - mean) / (std + 1e-9)

    # 4. Load Model & Run Inference
    model = load_model(args.model, args.model_path, args.device, dt)

    print(f"Visualizing Sample {args.sample_idx}...")
    with torch.no_grad():
        if args.model == "only_tcn":
            pred_pos = model(imu_raw, imu_norm)
            pred_vel = torch.zeros_like(pred_pos)
            pred_zupt = None
            pred_cov = None
        else:
            outputs = model(imu_raw, imu_norm)
            pred_pos = outputs["pred_pos_w"]
            pred_vel = outputs["filter_vel_w"] + outputs.get("pred_vel_resid_b", 0)
            pred_zupt = torch.sigmoid(outputs["pred_zupt_prob"])
            pred_cov = outputs.get("pred_covariance_p")

    # Convert to Numpy
    pred_pos = pred_pos.squeeze(0).cpu().numpy()
    pred_vel = pred_vel.squeeze(0).cpu().numpy()
    if pred_zupt is not None:
        pred_zupt = pred_zupt.squeeze(0).cpu().numpy()
    if pred_cov is not None:
        pred_cov = pred_cov.squeeze(0).cpu().numpy()

    # Plot
    plot_trajectory(
        args.sample_idx, gt_pos, pred_pos, gt_vel, pred_vel, pred_zupt, dt, "results", pred_cov
    )

if __name__ == "__main__":
    main()