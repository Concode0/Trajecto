import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from TCN import TCN
from rotation_utils import quaternion_to_rotation_matrix

class BaseFilterTCNModel(nn.Module):
    """Base class for a hybrid Kalman Filter-TCN model for trajectory reconstruction.

    This class defines a hybrid architecture that combines a physics-based Kalman
    filter (like AEKF or ESKF) with a data-driven Temporal Convolutional Network (TCN).
    The architecture operates as follows:
    1.  A Kalman filter processes raw IMU data sequentially to produce a baseline
        state estimate (position, velocity, orientation).
    2.  At each time step, a rich feature vector is constructed, including raw
        IMU data, filter state estimates, and internal filter metrics (e.g., innovation).
    3.  A TCN processes the sequence of these feature vectors to learn and predict
        the residual error of the Kalman filter's position estimate.
    4.  This predicted error is added back to the filter's output to produce a
        final, corrected trajectory.
    
    Subclasses must implement the filter-specific methods for state initialization
    and single-step filtering.
    """
    def __init__(self, tcn_input_size: int = 17, tcn_channels: List[int] = [64, 64, 64, 64], kernel_size: int = 3, dropout: float = 0.1, device: str = 'cpu'):
        super(BaseFilterTCNModel, self).__init__()
        self.device = device
        
        # The specific Kalman Filter (AEKF or ESKF) is defined by the subclass.
        self.filter = None 

        # --- TCN for Residual Correction ---
        self.tcn_input_norm = nn.BatchNorm1d(tcn_input_size)
        self.tcn = TCN(input_size=tcn_input_size, output_size=3, tcn_channels=tcn_channels, kernel_size=kernel_size, dropout=dropout)
        
        # --- Lever Arm Compensation ---
        # This constant represents the physical offset from the IMU sensor to the pen tip.
        # It is crucial for accurately tracking the tip's trajectory.
        self.register_buffer('pen_tip_offset_b', torch.tensor([0.145, 0.002, -0.02], device=device))

    def _initialize_state(self, batch_size: int, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Initializes the state and covariance for the Kalman filter."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _filter_step(self, state_tuple: Tuple[torch.Tensor, ...], imu_data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Performs a single predict-update step of the Kalman filter."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_position_and_quaternion(self, filter_output: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts position and orientation from the filter's output tuple."""
        raise NotImplementedError("This method must be implemented by the subclass.")
    
    def _get_gyro_bias(self, filter_output: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extracts gyroscope bias from the filter's output tuple."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def forward(self, imu_data_seq: torch.Tensor, initial_state: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.Tensor:
        """Performs a full forward pass of the hybrid model.

        Args:
            imu_data_seq (torch.Tensor): A sequence of IMU data with shape `[B, T, 7]`,
                containing (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, force).
            initial_state (Optional[Tuple[torch.Tensor, ...]]): An optional tuple to
                initialize the Kalman filter's state and covariance.

        Returns:
            torch.Tensor: The final corrected 3D trajectory of the pen tip, with shape `[B, T, 3]`.
        """
        batch_size, seq_len, _ = imu_data_seq.shape
        
        state_tuple = self._initialize_state(batch_size, imu_data_seq.dtype) if initial_state is None else initial_state
        
        positions_w_seq = []
        quaternions_b_to_w_seq = []
        tcn_feature_seq = []

        for t in range(seq_len):
            accel_b_raw = imu_data_seq[:, t, :3]
            gyro_b_raw = imu_data_seq[:, t, 3:6]
            force_raw = imu_data_seq[:, t, 6:]
            
            filter_output = self._filter_step(state_tuple, (gyro_b_raw, accel_b_raw, force_raw))
            
            pos_w, quat_b_to_w = self._get_position_and_quaternion(filter_output)
            gyro_bias_b = self._get_gyro_bias(filter_output)

            tcn_features_from_filter = filter_output[-1]
            state_tuple = filter_output[:-1]

            # --- TCN Feature Engineering ---
            # A rich feature vector is created to help the TCN learn the filter's error dynamics.
            
            # Lever Arm Velocity Compensation: The filter tracks the IMU's velocity, but we need the
            # velocity of the pen tip. This is calculated by adding the tangential velocity
            # caused by rotation around the IMU center. v_tip = v_imu + ω × r
            angular_velocity_b = gyro_b_raw - gyro_bias_b
            tangential_vel_b = torch.cross(angular_velocity_b, self.pen_tip_offset_b.unsqueeze(0), dim=-1)
            pen_tip_vel_b = tcn_features_from_filter['body_velocity'] + tangential_vel_b

            tcn_input_vec = torch.cat([
                gyro_b_raw, accel_b_raw, force_raw,
                pen_tip_vel_b,
                tcn_features_from_filter['zupt_flag'],
                tcn_features_from_filter['innovation']
            ], dim=-1)

            positions_w_seq.append(pos_w)
            quaternions_b_to_w_seq.append(quat_b_to_w)
            tcn_feature_seq.append(tcn_input_vec)

        positions_w = torch.stack(positions_w_seq, dim=1)
        quaternions_b_to_w = torch.stack(quaternions_b_to_w_seq, dim=1)
        tcn_features = torch.stack(tcn_feature_seq, dim=1)

        # --- Final Trajectory Correction ---
        # 1. Apply lever arm correction to the filter's position output.
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quaternions_b_to_w.reshape(-1, 4))
        rot_mat_b_to_w = rot_mat_b_to_w.view(batch_size, seq_len, 3, 3)
        offset_w = (rot_mat_b_to_w @ self.pen_tip_offset_b.view(1, 1, 3, 1)).squeeze(-1)
        pen_tip_pos_w_base = positions_w + offset_w
        
        # 2. Use the TCN to predict the residual error.
        tcn_features_norm = self.tcn_input_norm(tcn_features.permute(0, 2, 1)).permute(0, 2, 1)
        position_correction_w = self.tcn(tcn_features_norm)

        # 3. Add the learned correction to the baseline trajectory.
        final_trajectory_w = pen_tip_pos_w_base + position_correction_w

        return final_trajectory_w
