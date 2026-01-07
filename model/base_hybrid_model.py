"""
Base class for hybrid Kalman Filter-TCN models.

Defines common architecture for ESKF/AEKF-TCN hybrid systems with
configurable closed-loop or open-loop TCN correction integration.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TCN import TCN
from config import Config

# Import rotation utilities
from rotation_utils import quaternion_to_rotation_matrix


class BaseFilterTCNModel(nn.Module):
    """Base class for a hybrid Kalman Filter-TCN model architecture.

    This class sets up the common structure for models that integrate a
    state-estimation filter (like an ESKF or AEKF) with a Temporal
    Convolutional Network (TCN). The filter operates step-by-step on raw
    sensor data, while the TCN processes a sliding window of features
    derived from the filter's output. The TCN's predictions (e.g., velocity
    corrections, adaptive noise parameters, ZUPT probabilities) can then be
    used to refine the filter's estimates in a closed-loop fashion, or applied
    as post-processing in an open-loop setup.

    Attributes:
        device (str): The computation device ('cpu', 'cuda', 'mps').
        dt (float): The time step (delta time) used for filter integration.
        tcn_input_size (int): The dimensionality of the feature vector fed into the TCN.
        loop_type (str): Specifies how TCN corrections are applied ('closed' or 'open').
        filter (nn.Module): Placeholder for the specific Kalman filter
            implementation (e.g., ESKF, AEKF), to be instantiated by subclasses.
        tcn (TCN): The Temporal Convolutional Network used for feature processing.
        input_norm_layer (nn.LayerNorm): Layer normalization applied to TCN inputs.
        pen_tip_offset_b (nn.Parameter): Learnable offset from IMU to pen tip in body frame.
        initial_pen_tip_offset_b (torch.Tensor): Initial constant offset for regularization.
        gravity_w (torch.Tensor): Constant gravity vector in world frame.
    """

    def __init__(
        self,
        tcn_input_size: int = 20,
        tcn_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
        tcn_dilation_factors: Optional[List[int]] = None,
        dt: float = 0.01,
        loop_type: str = "closed",
        separable: bool = False,
    ):
        """Initializes the BaseFilterTCNModel.

        Args:
            tcn_input_size: The number of features in the TCN input vector.
            tcn_channels: A list specifying the number of channels (filters)
                for each layer in the TCN.
            kernel_size: The kernel size for TCN convolutions.
            dropout: The dropout rate applied within the TCN for regularization.
            device: The computation device ('cpu', 'cuda', 'mps').
            tcn_dilation_factors: Optional list of dilation factors for each
                TCN layer. If None, default dilation factors are used.
            dt: The time step (delta time) in seconds, crucial for the filter's
                integration steps.
            loop_type: Defines how TCN corrections are applied.
                'closed': TCN outputs directly influence the filter's state/covariance updates.
                'open': TCN outputs are used for post-correction of the filter's trajectory.
            separable: Whether to use Depthwise Separable Convolutions in TCN.
        """
        super().__init__()
        self.device = device
        self.dt = dt
        self.tcn_input_size = tcn_input_size
        self.loop_type = loop_type

        self.filter: Optional[nn.Module] = None

        self.tcn = TCN(
            input_size=tcn_input_size,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            tcn_dilation_factors=tcn_dilation_factors,
            separable=separable,
        )

        # r_pt: IMU→pen tip offset (learnable for lever arm correction)
        self.pen_tip_offset_b = nn.Parameter(
            torch.tensor(Config.INITIAL_PEN_TIP_OFFSET, device=device, dtype=torch.float32)
        )

        self.register_buffer(
            "initial_pen_tip_offset_b",
            torch.tensor(Config.INITIAL_PEN_TIP_OFFSET, device=device, dtype=torch.float32)
        )

        self.register_buffer(
            "gravity_w",
            torch.tensor([0.0, 0.0, -Config.GRAVITY_MAGNITUDE], device=device)
        )

        # Register normalization constants as buffers (optimization)
        # These are created once and automatically moved to device, avoiding
        # repeated tensor creation in the forward loop
        # NOTE: vel_mean is NOT registered because body-frame velocity normalization
        #       only uses std (mean subtraction would be incorrect due to frame rotation)
        self.register_buffer(
            "vel_std",
            torch.tensor(Config.VEL_STD, device=device, dtype=torch.float32)
        )

        # Register measurement noise std from Allan variance (for innovation normalization)
        # Use ISOTROPIC normalization (max noise) to preserve directional information
        # Rationale:
        #   - Innovation can represent directional errors (e.g., gravity misalignment)
        #   - Variation is small: accel 1.39×, gyro 1.11× (not 10× like velocity)
        #   - Consistent with isotropic velocity scaling philosophy
        #   - TCN learns physical relationships, not statistical artifacts
        # Channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        max_vrw = max(Config.VRW_X, Config.VRW_Y, Config.VRW_Z)  # Max accel noise
        max_arw = max(Config.ARW_X, Config.ARW_Y, Config.ARW_Z)  # Max gyro noise
        self.register_buffer(
            "measurement_noise_std",
            torch.tensor([max_vrw, max_vrw, max_vrw, max_arw, max_arw, max_arw],
                        device=device, dtype=torch.float32)
        )

    def get_pen_tip_regularization_loss(self) -> torch.Tensor:
        """Regularizes pen tip offset to maintain physical plausibility.

        Returns:
            L2 norm of deviation from initial physical measurement.
        """
        return torch.norm(self.pen_tip_offset_b - self.initial_pen_tip_offset_b)

    def _initialize_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        imu_data_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Abstract method to initialize the filter's state and covariance.

        Subclasses must implement this method to provide initial conditions
        for their specific Kalman filter (e.g., ESKF or AEKF).

        Args:
            batch_size: The number of sequences in the current batch.
            dtype: The desired data type for the state and covariance tensors.
            imu_data_seq: Optional. Initial IMU data that might be used for
                state initialization (e.g., for 'leveling' the initial orientation).

        Returns:
            A tuple containing the initialized state vector and covariance matrix.
        """
        raise NotImplementedError

    def _filter_step(
        self,
        state_tuple: Tuple[torch.Tensor, ...],
        imu_data: Tuple[torch.Tensor, ...],
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Abstract method to perform a single step of the Kalman filter.

        Subclasses must implement this method to execute one predict-update
        cycle of their specific Kalman filter.

        Args:
            state_tuple: A tuple containing the current state vector and
                its covariance matrix.
            imu_data: A tuple containing raw IMU measurements (gyro, accel, force)
                for the current time step.
            tcn_output: Optional. A dictionary of outputs from the TCN that
                can be used to influence the filter's current step (e.g., noise
                adaptation, state corrections).

        Returns:
            A tuple containing the updated state vector, covariance matrix,
            and any other filter-specific outputs (e.g., innovation, features
            for the TCN).
        """
        raise NotImplementedError

    def _get_position_and_quaternion(
        self, filter_output: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to extract position and quaternion from the filter's state.

        Subclasses must implement this to correctly parse their filter's
        state vector format.

        Args:
            filter_output: The full output tuple from the filter's step,
                where the first element is typically the state vector.

        Returns:
            A tuple containing:
                - pos_w (torch.Tensor): The 3D position in the world frame.
                - quat_b_to_w (torch.Tensor): The 4-element body-to-world quaternion.
        """
        raise NotImplementedError

    def _get_gyro_bias(
        self, filter_output: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Abstract method to extract gyroscope bias from the filter's state.

        Subclasses must implement this to correctly parse their filter's
        state vector format.

        Args:
            filter_output: The full output tuple from the filter's step,
                where the first element is typically the state vector.

        Returns:
            gyro_bias_b (torch.Tensor): The 3D gyroscope bias in the body frame.
        """
        raise NotImplementedError

    def forward(
        self,
        imu_data_raw: torch.Tensor,
        imu_data_norm: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the hybrid filter-TCN model.

        This method processes an entire sequence of IMU data, iteratively
        applying the Kalman filter step by step, feeding filter-derived
        features to the TCN, and applying TCN corrections.

        Args:
            imu_data_raw (torch.Tensor): Raw IMU data sequence.
                - Shape: (Batch, Seq_Len, Features)
                - Channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, force]
                - Unit: m/s^2 (Accel), rad/s (Gyro), N (Force)
                - Frame: Body Frame
            imu_data_norm (torch.Tensor): Normalized IMU data sequence.
                - Shape: (Batch, Seq_Len, Features)
                - Unit: Normalized
                - Frame: Body Frame
            initial_state (Optional[Tuple[torch.Tensor, ...]]): Optional. A tuple containing
                the initial state vector and its covariance matrix.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing predicted outputs:
                - "pred_pos_w": Final corrected trajectory.
                    - Shape: (Batch, Seq_Len, 3) | Unit: Meter | Frame: World
                - "pred_vel_resid_b": TCN-predicted residual velocity.
                    - Shape: (Batch, Seq_Len, 3) | Unit: m/s | Frame: Body
                - "pred_zupt_prob": TCN-predicted ZUPT probability.
                    - Shape: (Batch, Seq_Len, 1) | Range: [0, 1]
                - "pred_covariance_R": TCN-predicted measurement noise covariance parameters.
                    - Shape: (Batch, Seq_Len, 6) | Unit: varies
                - "filter_vel_w": Velocity from the filter.
                    - Shape: (Batch, Seq_Len, 3) | Unit: m/s | Frame: World
                - "filter_quat": Orientation quaternions from the filter.
                    - Shape: (Batch, Seq_Len, 4) | Frame: Body-to-World
                - "filter_innovation": Raw innovation from the filter.
                    - Shape: (Batch, Seq_Len, 6) | Unit: m/s^2, rad/s
                - "tcn_output_mask": Mask indicating valid TCN outputs.
                    - Shape: (Batch, Seq_Len) | Type: Bool
        """
        batch_size, seq_len, _ = imu_data_raw.shape

        # Initialize the filter's state and covariance.
        state_tuple = (
            self._initialize_state(batch_size, imu_data_raw.dtype, imu_data_raw)
            if initial_state is None
            else initial_state
        )

        # Retrieve the TCN's receptive field size. This determines how much
        # historical data the TCN needs to make a prediction.
        receptive_field = self.tcn.receptive_field

        # Buffers for storing sequences of filter outputs and TCN predictions.
        positions_w_seq: List[torch.Tensor] = []
        quaternions_b_to_w_seq: List[torch.Tensor] = []
        P_error_seq: List[torch.Tensor] = []
        pred_vel_resid_b_seq: List[torch.Tensor] = []
        pred_zupt_prob_seq: List[torch.Tensor] = []
        pred_covariance_R_seq: List[torch.Tensor] = []
        filter_vel_w_seq: List[torch.Tensor] = []
        filter_innovation_seq: List[torch.Tensor] = []

        # Buffer to store TCN input features for the entire sequence.
        # This allows for efficient windowing for the TCN.
        tcn_feature_seq = torch.zeros(
            batch_size,
            seq_len,
            self.tcn_input_size,
            device=self.device,
            dtype=imu_data_norm.dtype,
        )
        # Mask to indicate which time steps had valid TCN outputs (i.e., after RF).
        tcn_output_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=self.device
        )

        # Iterate through the sequence, applying the filter step by step.
        for t in range(seq_len):
            # Extract raw IMU data for the current time step.
            accel_b_raw = imu_data_raw[:, t, :3]
            gyro_b_raw = imu_data_raw[:, t, 3:6]
            force_raw = imu_data_raw[:, t, 6:]

            tcn_output: Optional[Dict[str, torch.Tensor]] = None
            # TCN prediction: only if enough history (at least receptive_field steps) is available.
            if t >= receptive_field:
                # Create a sliding window of features for the TCN.
                start_idx = t - receptive_field
                tcn_input_window = tcn_feature_seq[:, start_idx:t, :].clone()

                # Get predictions from the TCN. The TCN typically outputs a sequence
                # of corrections/parameters; we take the latest prediction.
                tcn_output_seq = self.tcn(tcn_input_window)
                tcn_output = {k: v[:, -1, :] for k, v in tcn_output_seq.items()}

                # Store TCN predictions for later use or loss calculation.
                pred_vel_resid_b_seq.append(tcn_output["vel_corr"])
                pred_zupt_prob_seq.append(tcn_output["zupt_prob"])
                pred_covariance_R_seq.append(tcn_output["covariance_R"])
                tcn_output_mask[:, t] = True

            # Perform one step of the Kalman filter (prediction and update).
            # The tcn_output is passed to allow for closed-loop adaptation.
            filter_output = self._filter_step(
                state_tuple, (gyro_b_raw, accel_b_raw, force_raw), tcn_output
            )

            # Unpack results from the filter step for the next iteration and logging.
            pos_w, quat_b_to_w = self._get_position_and_quaternion(filter_output)
            gyro_bias_b = self._get_gyro_bias(filter_output)
            # The last element of filter_output is usually a dict of TCN features.
            tcn_features_from_filter: Dict[str, torch.Tensor] = filter_output[-1]
            # The state and covariance are updated for the next iteration.
            state_tuple = filter_output[:-1]

            # Store the raw innovation from the filter for potential loss functions.
            raw_innovation = tcn_features_from_filter["innovation"]
            filter_innovation_seq.append(raw_innovation)

            # --- Feature Engineering for the *next* TCN input ---
            # These features are crucial for the TCN to learn effective corrections.
            # They are derived from normalized IMU data and current filter estimates.
            accel_b_norm_t = imu_data_norm[:, t, :3]
            gyro_b_norm_t = imu_data_norm[:, t, 3:6]
            force_norm_t = imu_data_norm[:, t, 6:]

            # Rotate current quaternion to get world-to-body rotation matrix.
            rot_mat_b_to_w_t = quaternion_to_rotation_matrix(quat_b_to_w)
            rot_mat_w_to_b_t = rot_mat_b_to_w_t.transpose(-1, -2)

            # Calculate estimated gravity vector in the body frame and normalize it.
            # This provides a sense of the current orientation relative to gravity.
            gravity_b_raw = (
                rot_mat_w_to_b_t @ self.gravity_w.expand(batch_size, -1).unsqueeze(-1)
            ).squeeze(-1)
            gravity_b_norm = gravity_b_raw / Config.GRAVITY_MAGNITUDE  # Normalize by magnitude of gravity

            # Calculate angular velocity and tangential velocity of pen tip.
            angular_velocity_b = gyro_b_raw - gyro_bias_b
            # Tangential velocity of a point (pen tip) due to rotation relative to IMU origin.
            tangential_vel_b = torch.cross(
                angular_velocity_b, self.pen_tip_offset_b.unsqueeze(0), dim=-1
            )
            # Combined pen tip velocity in body frame (filter's linear velocity + tangential).
            pen_tip_vel_b = tcn_features_from_filter["body_velocity"] + tangential_vel_b

            # Normalize velocity by std only (not mean) because:
            # 1. pen_tip_vel_b is in BODY frame (rotates with pen)
            # 2. VEL_MEAN/STD are from WORLD frame ground truth
            # 3. Body frame statistics are non-stationary due to rotation
            # 4. Handwriting is approximately zero-mean in body frame (symmetric strokes)
            # Use cached buffers (registered in __init__) - avoids repeated tensor creation
            # Safety epsilon 1e-3: prevents numerical explosion if std corrupts (Z-axis std ~0.01)
            pen_tip_vel_b_norm = pen_tip_vel_b / (self.vel_std + 1e-3)

            # Normalize innovation by MEASUREMENT NOISE (not empirical drift-contaminated stats)
            # Theoretical foundation: In well-calibrated KF, innovation ~ N(0, R)
            # Using empirical stats from pure ESKF is wrong because:
            #   1. Pure ESKF drifts → innovation std is contaminated by drift (inflated)
            #   2. During training, ESKF-TCN has minimal drift (TCN corrects it)
            #   3. Training/statistics mismatch causes over-normalization
            # Solution: Normalize by measurement noise R (from Allan variance)
            #   - Accel noise: VRW (Velocity Random Walk)
            #   - Gyro noise: ARW (Angular Random Walk)
            # This is theoretically principled and independent of drift
            # Use cached buffers for measurement noise std (registered in __init__)
            # Safety epsilon 1e-3: prevents numerical explosion (ARW ~7e-4, VRW ~9e-3)
            innovation_norm = raw_innovation / (self.measurement_noise_std + 1e-3)

            # Construct the comprehensive TCN input vector for this time step.
            # Note: zupt_flag removed to avoid circular dependency when using TCN-based ZUPT
            # All features now use z-score normalization (preserves magnitude info)
            tcn_input_vec = torch.cat(
                [
                    gyro_b_norm_t,           # [3] - Already z-score normalized
                    accel_b_norm_t,          # [3] - Already z-score normalized
                    force_norm_t,            # [1] - Already z-score normalized
                    pen_tip_vel_b_norm,      # [3] - Z-score normalized (not squashed!)
                    gravity_b_norm,          # [3] - Unit normalized (÷ g)
                    innovation_norm,         # [6] - Z-score normalized (not squashed!)
                ],
                dim=-1,
            )
            # Store the constructed feature vector.
            tcn_feature_seq[:, t, :] = tcn_input_vec

            # Log current position and quaternion from the filter for final trajectory calculation.
            positions_w_seq.append(pos_w)
            quaternions_b_to_w_seq.append(quat_b_to_w)

            # Also log filter's world-frame velocity.
            filter_vel_w = (
                rot_mat_b_to_w_t @ pen_tip_vel_b.unsqueeze(-1)
            ).squeeze(-1)
            filter_vel_w_seq.append(filter_vel_w)
            
            # Log covariance
            P_error_seq.append(filter_output[-2])

        # --- Helper for padding sequences ---
        def pad_sequence(
            seq: List[torch.Tensor], final_dim: Union[int, Tuple[int, ...]], dtype: torch.dtype
        ) -> torch.Tensor:
            """Pads a list of tensors to a consistent sequence length for concatenation."""
            if isinstance(final_dim, int):
                shape_suffix = (final_dim,)
            else:
                shape_suffix = final_dim

            if not seq:
                # If sequence is empty, return a zero tensor of the expected final shape.
                return torch.zeros(
                    batch_size, seq_len, *shape_suffix, device=self.device, dtype=dtype
                )
            # Calculate how many initial time steps were skipped (due to TCN receptive field).
            num_missing = seq_len - len(seq)
            # Create padding tensor for the skipped steps.
            padding_tensor = torch.zeros(
                batch_size, num_missing, *shape_suffix, device=self.device, dtype=dtype
            )
            stacked_seq = torch.stack(seq, dim=1)
            return torch.cat([padding_tensor, stacked_seq], dim=1)

        # --- Stack and process collected sequences ---
        raw_trajectory_w = torch.stack(positions_w_seq, dim=1)
        quaternions_b_to_w = torch.stack(quaternions_b_to_w_seq, dim=1)
        # Convert all quaternions in the sequence to rotation matrices (body to world).
        rot_mat_b_to_w = quaternion_to_rotation_matrix(
            quaternions_b_to_w.reshape(-1, 4)
        ).view(batch_size, seq_len, 3, 3)

        # Pad TCN prediction sequences.
        pred_vel_resid_b = pad_sequence(pred_vel_resid_b_seq, 3, imu_data_raw.dtype)
        pred_zupt_prob = pad_sequence(pred_zupt_prob_seq, 1, imu_data_raw.dtype)
        # Note: If TCN predicts R as a full matrix, final_dim would be different.
        # Assuming diagonal elements or specific covariance parameterization.
        pred_covariance_R = pad_sequence(pred_covariance_R_seq, 6, imu_data_raw.dtype)
        filter_innovation = pad_sequence(filter_innovation_seq, 6, imu_data_raw.dtype)
        stacked_filter_vel_w = torch.stack(filter_vel_w_seq, dim=1)
        
        # Pad and stack P_error sequence
        # Determine covariance shape dynamically from filter type
        # ESKF uses error_state_dim (15), AEKF uses state_dim (16)
        if hasattr(self.filter, 'error_state_dim'):
            dim = self.filter.error_state_dim  # ESKF
        elif hasattr(self.filter, 'state_dim'):
            dim = self.filter.state_dim  # AEKF
        else:
            dim = 15  # Fallback to ESKF default

        cov_shape = (dim, dim)
        if P_error_seq:
            cov_shape = P_error_seq[0].shape[1:]  # Override if sequence is non-empty

        stacked_P_error = pad_sequence(P_error_seq, cov_shape, imu_data_raw.dtype)

        # --- Final Trajectory Calculation (Open-loop vs. Closed-loop) ---
        final_pen_tip_trajectory_w: torch.Tensor
        if self.loop_type == "open":
            # In open-loop, TCN corrections are applied *after* the filter has run.
            # This means the filter's state itself is not directly modified by TCN velocity corrections.
            # 1. Rotate residual velocity from body to world frame using filter's orientation.
            pred_vel_resid_w = (
                rot_mat_b_to_w @ pred_vel_resid_b.unsqueeze(-1)
            ).squeeze(-1)

            # 2. Add residual velocity to the filter's world-frame velocity.
            corrected_vel_w = stacked_filter_vel_w + pred_vel_resid_w

            # 3. Integrate the corrected velocity to obtain the final trajectory.
            # Use the first position from the filter as the initial condition for integration.
            initial_pos_w = raw_trajectory_w[:, 0, :].unsqueeze(1)

            # Cumulative sum of velocity deltas (delta_p = v * dt) gives displacement.
            # We start integration from the second time step as we have initial_pos_w.
            corrected_pos_deltas = torch.cumsum(
                corrected_vel_w[:, 1:, :] * self.dt, dim=1
            )

            # Reconstruct the trajectory by adding displacements to the initial position.
            corrected_trajectory_w = torch.cat(
                [initial_pos_w, initial_pos_w + corrected_pos_deltas], dim=1
            )

            # Add pen tip offset in the world frame to get the final pen tip trajectory.
            pen_tip_offset_w = (
                rot_mat_b_to_w @ self.pen_tip_offset_b.view(1, 1, 3, 1)
            ).squeeze(-1)
            final_pen_tip_trajectory_w = corrected_trajectory_w + pen_tip_offset_w
        else:  # loop_type == 'closed'
            # In closed-loop, TCN corrections (e.g., velocity residuals) are
            # directly integrated into the filter's state propagation or update.
            # Therefore, `raw_trajectory_w` already reflects the TCN's influence.
            # We just need to add the pen tip offset.
            pen_tip_offset_w = (
                rot_mat_b_to_w @ self.pen_tip_offset_b.view(1, 1, 3, 1)
            ).squeeze(-1)
            final_pen_tip_trajectory_w = raw_trajectory_w + pen_tip_offset_w

        return {
            "pred_pos_w": final_pen_tip_trajectory_w,
            "pred_vel_resid_b": pred_vel_resid_b,
            "pred_zupt_prob": pred_zupt_prob,
            "pred_covariance_R": pred_covariance_R,
            "filter_vel_w": stacked_filter_vel_w,
            "filter_quat": quaternions_b_to_w,
            "filter_innovation": filter_innovation,
            "filter_covariance": stacked_P_error,
            "tcn_output_mask": tcn_output_mask,
        }