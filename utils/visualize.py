"""This module provides a comprehensive Debugging and Tuning Station for the
Trajecto project.

It integrates data validation tools (checking synchronization and sensor health)
with model performance visualization (trajectory alignment, error metrics) into
a unified `TrajectoDebugger` class. This allows for rapid iteration and
diagnosis of both data quality and model accuracy issues.
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist

# Add parent directory to sys.path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.AEKF_TCN import AEKFTCN_model
from model.config import Config
from model.dataset import TrajectoryDataset
from model.ESKF_TCN import ESKFTCN_model
from model.onlyTCN import OnlyTCN
from model.rotation_utils import *


def load_model(
    model_name: str, model_path: str, device: str, dt: float
) -> torch.nn.Module:
    """Initializes a model and loads its weights from a file.

    Args:
        model_name (str): The name of the model to load.
            Choices are 'eskf_tcn', 'aekf_tcn', 'only_tcn'.
        model_path (str): The path to the saved model weights.
        device (str): The device to load the model on ('cpu' or 'cuda').
        dt (float): The time step used for the model's internal calculations.

    Returns:
        torch.nn.Module: The loaded and initialized model.

    Raises:
        ValueError: If the model_name is not recognized.
        FileNotFoundError: If the model_path does not exist.
    """
    config = None
    if model_name == "eskf_tcn":
        config = {
            "tcn_input_size": Config.ESKFTCN.TCN_INPUT_SIZE,
            "use_zupt": Config.ESKFTCN.USE_ZUPT,
            "use_tcn_zupt": Config.ESKFTCN.USE_TCN_ZUPT,
        }
    elif model_name == "aekf_tcn":
        config = {
            "tcn_input_size": Config.AEKFTCN.TCN_INPUT_SIZE,
            "use_zupt": False, # AEKF uses TCN outputs for velocity residuals, not direct ZUPT
        }
    elif model_name == "only_tcn":
        config = {
            "input_size": Config.OnlyTCN.INPUT_SIZE,
            "output_size": Config.OnlyTCN.OUTPUT_SIZE,
        }

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
        raise ValueError(f"Unknown model name: {model_name}")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        print(f"[Warning] Model file not found at {model_path}. Initializing with random weights.")
        model.eval()
        return model


def align_trajectories(
    gt: np.ndarray, pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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

    print(f"[Info] Auto-Scaling Factor: {scale:.4f} (Aligning Pred to GT scale)")

    pred_scaled = pred_centered * scale
    H = np.dot(pred_scaled.T, gt_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    pred_aligned = np.dot(pred_scaled, R.T)

    return gt_centered, pred_aligned


def calculate_dtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Calculates Dynamic Time Warping (DTW) distance using Euclidean metric.

    Args:
        seq1 (np.ndarray): First sequence.
            - Shape: (N, D)
        seq2 (np.ndarray): Second sequence.
            - Shape: (M, D)

    Returns:
        float: The accumulated DTW distance.
    """
    n, m = len(seq1), len(seq2)
    # Compute pairwise distance matrix
    dist_matrix = cdist(seq1, seq2, metric='euclidean')

    # Initialize DP matrix with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j],    # Insertion
                                   dtw[i, j - 1],    # Deletion
                                   dtw[i - 1, j - 1]) # Match

    return dtw[n, m]


class TrajectoDebugger:
    """A unified station for debugging data preprocessing and tuning model performance.

    This class encapsulates the visualization logic for:
    1.  **Data Validation:** Checking sensor signal health and synchronization with GT.
    2.  **Model Evaluation:** Running inference and plotting trajectory/error metrics.
    """

    def __init__(
        self,
        dataset_path: str,
        scaler_path: str,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        dt: float = 0.02,
    ):
        """Initializes the TrajectoDebugger.

        Args:
            dataset_path (str): Path to the HDF5 dataset file.
            scaler_path (str): Path to the scaler statistics file.
            model (Optional[torch.nn.Module]): The trained PyTorch model.
            device (str): Computation device ('cpu' or 'cuda').
            dt (float): Time step in seconds.
        """
        self.dataset = TrajectoryDataset(dataset_path)
        self.model = model
        self.device = device
        self.dt = dt

        # Load scaler stats
        if os.path.exists(scaler_path):
            with h5py.File(scaler_path, "r") as f:
                self.mean = torch.from_numpy(f["mean"][:]).float().to(self.device)
                self.std = torch.from_numpy(f["std"][:]).float().to(self.device)
        else:
            print(f"[Warning] Scaler stats not found at {scaler_path}. Using identity scaling.")
            self.mean = torch.zeros(1).to(self.device)
            self.std = torch.ones(1).to(self.device)

    def _get_sample_data(self, sample_idx: int) -> Dict[str, Any]:
        """Retrieves and prepares data for a specific sample."""
        data = self.dataset[sample_idx]
        imu_raw = data["imu_seq_raw"].unsqueeze(0).to(self.device)
        imu_norm = (imu_raw - self.mean) / (self.std + 1e-9)
        seq_len = int(data["len"].item())

        return {
            "imu_raw": imu_raw,
            "imu_norm": imu_norm,
            "gt_pos": data["gt_pos_w"].numpy()[:seq_len],
            "gt_vel": data["gt_vel_w"].numpy()[:seq_len],
            "sensor_np": data["imu_seq_raw"].numpy()[:seq_len],
            "len": seq_len,
        }

    def validate_data(
        self, ax_sync: plt.Axes, ax_sensor: plt.Axes, sample_data: Dict[str, Any]
    ) -> None:
        """Visualizes data synchronization and sensor health.

        Args:
            ax_sync (plt.Axes): Axes for synchronization plot.
            ax_sensor (plt.Axes): Axes for raw sensor plot.
            sample_data (Dict[str, Any]): Dictionary containing sample data.
        """
        sensor = sample_data["sensor_np"]
        gt_vel = sample_data["gt_vel"]
        time = np.arange(sample_data["len"]) * self.dt

        # --- 1. Synchronization Check ---
        # Plot normalized Accel Magnitude vs GT Velocity Magnitude
        # This allows checking if the "motion" in IMU aligns with GT movement.
        accel_norm = np.linalg.norm(sensor[:, :3], axis=1)
        gt_speed = np.linalg.norm(gt_vel, axis=1)

        # Normalize for visualization overlay
        accel_vis = (accel_norm - np.mean(accel_norm)) / (np.std(accel_norm) + 1e-6)
        speed_vis = (gt_speed - np.mean(gt_speed)) / (np.std(gt_speed) + 1e-6)

        ax_sync.plot(time, accel_vis, "b-", alpha=0.6, label="Accel Norm (Norm)")
        ax_sync.plot(time, speed_vis, "r--", alpha=0.8, label="GT Speed (Norm)")

        # If FSR exists (7th channel), plot it on a secondary y-axis
        if sensor.shape[1] > 6:
            fsr = sensor[:, 6]
            if np.std(fsr) > 0:
                ax_sync_twin = ax_sync.twinx() # Create a twin Axes for shared x-axis
                fsr_vis = (fsr - np.mean(fsr)) / (np.std(fsr) + 1e-6)
                ax_sync_twin.plot(time, fsr_vis, "g:", alpha=0.5, label="FSR (Norm)")
                ax_sync_twin.set_ylabel("FSR (Norm)", color="g")
                ax_sync_twin.tick_params(axis="y", labelcolor="g")
                # Update legend to include twin axis's label
                lines, labels = ax_sync.get_legend_handles_labels()
                lines2, labels2 = ax_sync_twin.get_legend_handles_labels()
                ax_sync.legend(lines + lines2, labels + labels2, loc="upper left")


        ax_sync.set_title("Synchronization Check (Normalized Overlay)")
        ax_sync.set_ylabel("Normalized Amplitude")
        ax_sync.legend(loc="upper left") # Adjusted legend location due to twinx
        ax_sync.grid(True)

        # --- 2. Sensor Health ---
        # Plot Gyroscope data to check for noise/bias
        ax_sensor.plot(time, sensor[:, 3], label="Gyro X")
        ax_sensor.plot(time, sensor[:, 4], label="Gyro Y")
        ax_sensor.plot(time, sensor[:, 5], label="Gyro Z")
        ax_sensor.set_title("Raw Gyroscope Data")
        ax_sensor.set_ylabel("rad/s")
        ax_sensor.set_xlabel("Time (s)")
        ax_sensor.legend(loc="upper right")
        ax_sensor.grid(True)

    def evaluate_model(
        self,
        ax_traj: plt.Axes,
        ax_err: plt.Axes,
        ax_3d: plt.Axes,
        sample_data: Dict[str, Any],
    ) -> None:
        """Runs model inference and visualizes trajectory/errors.

        Args:
            ax_traj (plt.Axes): Axes for 2D trajectory plot.
            ax_err (plt.Axes): Axes for error/drift plot.
            ax_3d (plt.Axes): Axes for 3D trajectory plot.
            sample_data (Dict[str, Any]): Dictionary containing sample data.
        """
        if self.model is None:
            print("[Info] No model provided. Skipping model evaluation.")
            return

        with torch.no_grad():
            outputs = self.model(sample_data["imu_raw"], sample_data["imu_norm"])

            # Handle different model output structures
            if isinstance(outputs, dict) and "pred_pos_w" in outputs:
                pred_pos = outputs["pred_pos_w"]
            else:
                # Fallback for simple models
                pred_pos = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        # Slice to valid sequence length
        seq_len = sample_data["len"]
        pred_pos_np = pred_pos.squeeze(0).cpu().numpy()[:seq_len]
        gt_pos_np = sample_data["gt_pos"]
        gt_vel_np = sample_data["gt_vel"]
        time = np.arange(seq_len) * self.dt

        # Align trajectories
        gt_aligned, pred_aligned = align_trajectories(gt_pos_np, pred_pos_np)

        # --- 1. 2D Trajectory with Color Gradient ---
        # Use a colormap to show progress over time
        cmap = plt.cm.viridis
        ax_traj.scatter(gt_aligned[:, 0], gt_aligned[:, 1], c=time, cmap=cmap, s=5, label="GT", alpha=0.7)
        ax_traj.scatter(pred_aligned[:, 0], pred_aligned[:, 1], c=time, cmap=cmap, s=5, marker='x', label="Pred", alpha=0.7)
        ax_traj.set_title("2D Alignment (XY Plane) with Time Gradient")
        ax_traj.axis("equal")
        ax_traj.grid(True)
        ax_traj.legend()
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax_traj, label="Time (s)")


        # --- 2. Velocity Components Graph ---
        pred_vel_tensor: Optional[torch.Tensor] = None
        if isinstance(outputs, dict):
            # Hybrid models (ESKF_TCN, AEKF_TCN)
            if "filter_vel_w" in outputs and "pred_vel_resid_b" in outputs and "filter_quat" in outputs:
                # Reconstruct predicted velocity for hybrid models
                filter_vel_w = outputs["filter_vel_w"]
                pred_vel_resid_b = outputs["pred_vel_resid_b"]
                filter_quat = outputs["filter_quat"]

                # Ensure filter_quat is (Batch, Seq, 4) or (Seq, 4) for this sample
                # Squeeze if batch_size=1, then transpose to (Seq, 4)
                filter_quat_sq = filter_quat.squeeze(0) # (Seq, 4)

                # Convert quaternions to rotation matrices (Batch, Seq, 3, 3)
                rot_mat_b_to_w = quaternion_to_rotation_matrix(filter_quat_sq).cpu() # (Seq, 3, 3)

                # Rotate residual velocity from body to world frame
                pred_vel_resid_w = (
                    rot_mat_b_to_w @ pred_vel_resid_b.squeeze(0).cpu().unsqueeze(-1)
                ).squeeze(-1) # (Seq, 3)

                pred_vel_tensor = filter_vel_w.squeeze(0).cpu() + pred_vel_resid_w
            else:
                print("[Warning] Could not reconstruct predicted velocity from hybrid model outputs. Differentiating position.")
                # Fallback: numerically differentiate position
                # Needs original pred_pos (before numpy conversion)
                if isinstance(outputs, dict): # Should be True if we are in this block
                    raw_pred_pos = outputs["pred_pos_w"].squeeze(0).cpu() # Get tensor form
                else: # Fallback to outputs as tensor, unlikely for hybrid
                    raw_pred_pos = outputs.squeeze(0).cpu()
                pred_vel_tensor = torch.from_numpy(np.gradient(raw_pred_pos.numpy(), axis=0) / self.dt)


        else: # OnlyTCN case, outputs is a tensor directly
            # Numerically differentiate position to get velocity for OnlyTCN
            raw_pred_pos = outputs.squeeze(0).cpu() # Get tensor form
            pred_vel_tensor = torch.from_numpy(np.gradient(raw_pred_pos.numpy(), axis=0) / self.dt)

        ax_err.plot(time, gt_vel_np[:, 0], "k-", label="GT Vel X")
        ax_err.plot(time, gt_vel_np[:, 1], "k--", label="GT Vel Y")
        ax_err.plot(time, gt_vel_np[:, 2], "k:", label="GT Vel Z")

        if pred_vel_tensor is not None:
            pred_vel_np = pred_vel_tensor.numpy()[:seq_len]
            ax_err.plot(time, pred_vel_np[:, 0], "r-", label="Pred Vel X")
            ax_err.plot(time, pred_vel_np[:, 1], "r--", label="Pred Vel Y")
            ax_err.plot(time, pred_vel_np[:, 2], "r:", label="Pred Vel Z")
            ax_err.set_title("Velocity Components (World Frame)")
        else:
            ax_err.set_title("Ground Truth Velocity Components (Pred Vel N/A)")
            print("[Warning] Predicted velocity could not be obtained from model output or by differentiation.")

        ax_err.set_xlabel("Time (s)")
        ax_err.set_ylabel("Velocity (m/s)")
        ax_err.legend(loc="upper right")
        ax_err.grid(True)

        # --- 3. 3D Trajectory ---
        ax_3d.plot(gt_aligned[:, 0], gt_aligned[:, 1], gt_aligned[:, 2], "k--", label="GT")
        ax_3d.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], "r-", label="Pred")
        ax_3d.set_title("3D Trajectory")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z") # type: ignore

    def run_dashboard(self, sample_idx: int) -> None:
        """Creates and displays the unified debugging dashboard."""
        print(f"--- Launching Trajecto Dashboard for Sample {sample_idx} ---")
        sample_data = self._get_sample_data(sample_idx)

        fig = plt.figure(figsize=(18, 10))

        # Grid layout:
        # Row 1: Sync Check (Left), Sensor Health (Right)
        # Row 2: 2D Traj (Left), Error (Middle), 3D Traj (Right)
        gs = fig.add_gridspec(2, 3)

        ax_sync = fig.add_subplot(gs[0, :2]) # Span 2 columns
        ax_sensor = fig.add_subplot(gs[0, 2])

        ax_traj = fig.add_subplot(gs[1, 0])
        ax_err = fig.add_subplot(gs[1, 1])
        ax_3d = fig.add_subplot(gs[1, 2], projection="3d")

        # 1. Data Validation
        self.validate_data(ax_sync, ax_sensor, sample_data)

        # 2. Model Evaluation
        self.evaluate_model(ax_traj, ax_err, ax_3d, sample_data)

        plt.tight_layout()
        save_path = f"results/dashboard_sample_{sample_idx}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(save_path)
        print(f"Dashboard saved to {save_path}")
        plt.show()


def main() -> None:
    """Main CLI entry point for the Trajecto Debugging Station."""
    parser = argparse.ArgumentParser(description="Trajecto Debugging & Tuning Station")
    parser.add_argument("--model", type=str, default="eskf_tcn", help="Model type")
    parser.add_argument("--model_path", type=str, default="eskf_tcn_model.pth", help="Path to .pth file")
    parser.add_argument("--data", type=str, default="data/dataset.h5", help="Dataset path")
    parser.add_argument("--scaler", type=str, default="data/scaler_stats.h5", help="Scaler stats path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # New argument for visualization mode
    parser.add_argument(
        "--mode",
        type=str,
        default="dashboard",
        choices=["dashboard", "data_only", "model_only"],
        help="Visualization mode: 'dashboard' (all), 'data_only' (preprocess checks), 'model_only' (predictions)"
    )

    args = parser.parse_args()

    # Model Setup (only needed for dashboard or model_only)
    model = None
    dt = Config.DT

    if args.mode != "data_only":
        try:
            model = load_model(args.model, args.model_path, args.device, dt)
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            if args.mode == "model_only":
                return # Can't proceed
            print("[Info] Proceeding with data visualization only.")

    debugger = TrajectoDebugger(args.data, args.scaler, model, args.device, dt)

    if args.mode == "dashboard":
        debugger.run_dashboard(args.sample_idx)
    elif args.mode == "data_only":
        # Create a simplified figure for data checks
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        sample_data = debugger._get_sample_data(args.sample_idx)
        debugger.validate_data(ax1, ax2, sample_data)
        plt.show()
    elif args.mode == "model_only":
        # Create a figure for model results
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')
        sample_data = debugger._get_sample_data(args.sample_idx)
        debugger.evaluate_model(ax1, ax2, ax3, sample_data)
        plt.show()

if __name__ == "__main__":
    main()