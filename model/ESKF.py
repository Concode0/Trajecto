"""
Error-State Kalman Filter (ESKF) for 3D pen trajectory reconstruction.

Maintains a nominal state propagated through non-linear dynamics and a
linear error state for corrections. This approach combines accurate
non-linear propagation with computationally efficient linear filtering.

Supports Block-Parallel Scan for accelerated covariance computation:
- Cache F, Q, W, K, R matrices during sequential forward pass
- Use cached matrices with parallel scan for O(log T) covariance computation
- Unified operator: F̃ = W @ F, Q̃ = W @ Q @ W^T + K @ R @ K^T
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

# Import rotation utilities
from rotation_utils import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    small_angle_to_quaternion,
    quaternion_from_two_vectors,
)

from zupt_detector import ZuptDetector


@dataclass
class ESKFStepCache:
    """Cache for a single ESKF timestep."""
    F: torch.Tensor  # Transition Jacobian [batch, 15, 15]
    Q: torch.Tensor  # Process noise [batch, 15, 15]
    W: torch.Tensor  # Combined compensation matrix I - K @ H [batch, 15, 15]
    noise_inj: torch.Tensor  # Accumulated noise injection from updates [batch, 15, 15]


@dataclass
class ESKFSequenceCache:
    """Cache for entire ESKF sequence, used for parallel covariance computation.

    The unified operator formulation:
        F̃ = W @ F
        Q̃ = W @ Q @ W^T + noise_inj
        P[t+1] = F̃ @ P[t] @ F̃^T + Q̃
    """
    F_seq: torch.Tensor  # [batch, seq_len, 15, 15] - Transition Jacobians
    Q_seq: torch.Tensor  # [batch, seq_len, 15, 15] - Process noise
    W_seq: torch.Tensor  # [batch, seq_len, 15, 15] - Combined compensation matrices
    noise_inj_seq: torch.Tensor  # [batch, seq_len, 15, 15] - Accumulated noise injection
    P_init: torch.Tensor  # [batch, 15, 15] - Initial covariance

    def to(self, device: torch.device) -> 'ESKFSequenceCache':
        """Move cache to device."""
        return ESKFSequenceCache(
            F_seq=self.F_seq.to(device),
            Q_seq=self.Q_seq.to(device),
            W_seq=self.W_seq.to(device),
            noise_inj_seq=self.noise_inj_seq.to(device),
            P_init=self.P_init.to(device),
        )


class ErrorStateKalmanFilter(nn.Module):
    """Batch-aware Error-State Kalman Filter for IMU state estimation.

    State Representation:
    - Nominal State (16D): [pos_w(3), vel_w(3), quat_b_to_w(4), gyro_bias_b(3), accel_bias_b(3)]
    - Error State (15D): [δpos_w(3), δvel_w(3), δtheta_b(3), δgyro_bias_b(3), δaccel_bias_b(3)]

    Filter Cycle:
    1. Predict: Propagate nominal state and error covariance
    2. Update: Correct error state using measurements
    3. Inject: Apply error correction to nominal state and reset error
    """

    def __init__(
        self,
        error_state_dim: int = 15,
        obs_dim: int = 6,  # 3 accel + 3 gyro
        dt: float = Config.DT,
        device: str = "cpu",
        use_zupt: bool = True,
        use_tcn_zupt: bool = False,
        use_virtual_measurements: bool = False,
        mahalanobis_threshold: float = Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD,
        eskf_learnable_params: bool = Config.ESKFTCN.ESKF_LEARNABLE_PARAMS,
    ):
        """Initializes the ESKF module.

        Args:
            error_state_dim: Dimension of the error state vector. Fixed at 15
                for this implementation (3 for position, 3 for velocity, 3 for
                orientation (small angle), 3 for gyro bias, 3 for accel bias).
            obs_dim: Dimension of the observation vector. Fixed at 6 for
                3-axis accelerometer and 3-axis gyroscope measurements.
            dt: Time step (delta time) in seconds.
            device: The compute device ('cpu', 'cuda', 'mps').
            use_zupt: Boolean flag to enable or disable traditional ZUPT detection.
            use_tcn_zupt: Boolean flag to enable ZUPT decisions based on TCN output.
            use_virtual_measurements: Boolean flag to enable virtual measurement
                updates during motion (for pure ESKF mode without TCN).
            mahalanobis_threshold: Threshold for Mahalanobis gating.
            eskf_learnable_params: If True, ESKF parameters (R_diag, zupt_noise_std,
                virtual_meas_weights) are learnable via backpropagation. If False,
                they are fixed buffers (faster training, no BPTT).
        """
        super().__init__()

        self.error_state_dim = error_state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.device = device
        self.mahalanobis_threshold = mahalanobis_threshold

        # Allan Variance parameters (sensor characterization)
        arw_x, arw_y, arw_z = Config.ARW_X, Config.ARW_Y, Config.ARW_Z
        gyro_bi_x, gyro_bi_y, gyro_bi_z = Config.GYRO_BI_X, Config.GYRO_BI_Y, Config.GYRO_BI_Z
        vrw_x, vrw_y, vrw_z = Config.VRW_X, Config.VRW_Y, Config.VRW_Z
        accel_bi_x, accel_bi_y, accel_bi_z = Config.ACCEL_BI_X, Config.ACCEL_BI_Y, Config.ACCEL_BI_Z

        # Q diagonal: [Pos(3), Vel(3), Ori(3), GyBias(3), AcBias(3)]
        Q_diag_tensor = torch.zeros(self.error_state_dim, device=device)
        Q_diag_tensor[0:3] = 0.0  # Pos: driven by vel
        Q_diag_tensor[3:6] = torch.tensor([vrw_x**2, vrw_y**2, vrw_z**2], device=device)  # Vel: accel VRW
        Q_diag_tensor[6:9] = torch.tensor([arw_x**2, arw_y**2, arw_z**2], device=device)  # Ori: gyro ARW
        Q_diag_tensor[9:12] = torch.tensor([gyro_bi_x**2, gyro_bi_y**2, gyro_bi_z**2], device=device)  # Gyro bias: BI
        Q_diag_tensor[12:15] = torch.tensor([accel_bi_x**2, accel_bi_y**2, accel_bi_z**2], device=device)  # Accel bias: BI
        self.register_buffer("Q_diag", Q_diag_tensor)

        # ZUPT noise: learnable or fixed based on flag
        if use_zupt or use_tcn_zupt:
            zupt_noise_tensor = torch.tensor(Config.ESKFTCN.ZUPT_NOISE_STD_ESKF, device=device)
            if eskf_learnable_params:
                self.zupt_noise_std = nn.Parameter(zupt_noise_tensor)
            else:
                self.register_buffer("zupt_noise_std", zupt_noise_tensor)

        # R_diag: learnable or fixed based on flag
        R_diag_tensor = torch.ones(self.obs_dim, device=device) * 1e-4
        if eskf_learnable_params:
            self.R_diag = nn.Parameter(R_diag_tensor)
        else:
            self.register_buffer("R_diag", R_diag_tensor)

        self.register_buffer("gravity_w", torch.tensor([0.0, 0.0, -Config.GRAVITY_MAGNITUDE], device=device))

        self.zupt_detector = ZuptDetector(
            window_size=Config.ZUPT_WINDOW_SIZE,
            accel_var_threshold=Config.ZUPT_ACCEL_THRESHOLD,
            force_var_threshold=Config.ZUPT_FORCE_VAR_THRESHOLD,
            force_delta_threshold=Config.ZUPT_FORCE_DELTA_THRESHOLD,
            device=device,
        )
        self.use_zupt = use_zupt
        self.use_tcn_zupt = use_tcn_zupt
        self.use_virtual_measurements = use_virtual_measurements
        self.adaptive_gain = Config.ESKFTCN.ADAPTIVE_GAIN_ESKF

        # Virtual measurement correction weights (for blind period / pure ESKF mode)
        # Initialized from Allan variance ratios - provides principled starting point
        # Can be learnable or fixed based on eskf_learnable_params flag
        if use_virtual_measurements:
            virtual_meas_tensor = torch.tensor([
                0.0, 0.0, 0.0,  # Position (blocked - no absolute reference)
                0.0, 0.0, 0.0,  # Velocity (blocked - would cause scale drift)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,  # Orientation (from 1/ARW)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ORIENTATION,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,    # Gyro bias (from 1/GYRO_BI)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_GYRO_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,   # Accel bias (from 1/ACCEL_BI)
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,
                Config.ESKFTCN.VIRTUAL_MEAS_WEIGHT_ACCEL_BIAS,
            ], device=device, dtype=torch.float32)
            if eskf_learnable_params:
                self.virtual_meas_weights = nn.Parameter(virtual_meas_tensor)
            else:
                self.register_buffer("virtual_meas_weights", virtual_meas_tensor)


    def get_Q(self):
        """Returns the scaled process noise covariance matrix Q."""
        return torch.diag(self.Q_diag) * self.dt

    def get_R_zupt(self):
        """Returns the measurement noise covariance R for ZUPT."""
        return torch.diag(self.zupt_noise_std ** 2)

    # =========================================================================
    # CACHING SUPPORT FOR BLOCK-PARALLEL SCAN
    # =========================================================================

    def init_cache(self, batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
        """Initialize cache storage for parallel covariance computation.

        Call this before the forward pass sequence to enable caching.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type for cache tensors
            device: Device for cache tensors. If None, uses self.device.
        """
        if device is None:
            device = self.device

        self._cache_enabled = True
        self._cache_step = 0
        self._cache_batch_size = batch_size
        self._cache_seq_len = seq_len
        self._cache_device = device

        # Pre-allocate cache tensors (torch.zeros creates contiguous tensors by default)
        D = self.error_state_dim
        self._cache_F = torch.zeros(
            batch_size, seq_len, D, D, device=device, dtype=dtype
        )
        self._cache_Q = torch.zeros(
            batch_size, seq_len, D, D, device=device, dtype=dtype
        )
        self._cache_W = torch.zeros(
            batch_size, seq_len, D, D, device=device, dtype=dtype
        )
        self._cache_noise_inj = torch.zeros(
            batch_size, seq_len, D, D, device=device, dtype=dtype
        )
        self._cache_P_init = None

    def finalize_cache(self) -> Optional[ESKFSequenceCache]:
        """Finalize and return the collected cache.

        Call this after the forward pass sequence completes.

        Returns:
            ESKFSequenceCache with all cached matrices, or None if caching disabled.
        """
        if not getattr(self, '_cache_enabled', False):
            return None

        cache = ESKFSequenceCache(
            F_seq=self._cache_F[:, :self._cache_step].clone(),
            Q_seq=self._cache_Q[:, :self._cache_step].clone(),
            W_seq=self._cache_W[:, :self._cache_step].clone(),
            noise_inj_seq=self._cache_noise_inj[:, :self._cache_step].clone(),
            P_init=self._cache_P_init.clone() if self._cache_P_init is not None else None,
        )

        # Disable caching
        self._cache_enabled = False
        return cache

    def _cache_step_matrices(
        self,
        F: torch.Tensor,
        Q: torch.Tensor,
        W: torch.Tensor,
        noise_inj: torch.Tensor,
    ):
        """Cache matrices for current timestep.

        Called internally during forward pass when caching is enabled.

        Args:
            F: Transition Jacobian [batch, 15, 15]
            Q: Process noise [batch, 15, 15]
            W: Combined compensation matrix from all updates [batch, 15, 15]
            noise_inj: Accumulated noise injection from all updates [batch, 15, 15]
        """
        if not getattr(self, '_cache_enabled', False):
            return

        t = self._cache_step
        if t >= self._cache_seq_len:
            return

        # Move to cache device if necessary (CPU state + GPU cache)
        cache_device = self._cache_F.device
        if F.device != cache_device:
            F = F.to(cache_device)
            Q = Q.to(cache_device)
        # W and noise_inj are already on cache device from cache_accum

        self._cache_F[:, t] = F
        self._cache_Q[:, t] = Q
        self._cache_W[:, t] = W
        self._cache_noise_inj[:, t] = noise_inj
        self._cache_step += 1

    def _init_cache_accumulators(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> dict:
        """Initialize accumulators for tracking combined update matrices.

        Returns a dict with:
        - W_combined: Product of all compensation matrices (I - K @ H)
        - noise_accumulated: Cumulative noise injection from all updates

        For multiple updates: P_final = W_combined @ P_pred @ W_combined^T + noise_accumulated
        where noise_accumulated = sum of W_remaining @ K_i @ R_i @ K_i^T @ W_remaining^T
        """
        return {
            'W_combined': torch.eye(self.error_state_dim, device=device, dtype=dtype)
                         .unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'noise_accumulated': torch.zeros(batch_size, self.error_state_dim, self.error_state_dim,
                                             device=device, dtype=dtype),
        }

    def _accumulate_update_noise(
        self,
        cache_accum: dict,
        W_new: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Accumulate noise injection from an update step.

        After this update:
        - noise_new = W_new @ noise_old @ W_new^T + K @ R @ K^T
        - W_combined = W_new @ W_combined

        Args:
            cache_accum: Cache accumulator dict
            W_new: Compensation matrix (I - K @ H) from current update [batch, 15, 15]
            K: Kalman gain [batch, 15, m] where m is measurement dim
            R: Measurement noise [batch, m, m]
            mask: Optional boolean mask for selective update [batch]

        BPTT Note: Uses torch.where for gradient-safe conditional assignment.
        """
        batch_size = cache_accum['noise_accumulated'].shape[0]
        cache_device = cache_accum['noise_accumulated'].device

        # Move update matrices to cache device if necessary (CPU state + GPU cache)
        if W_new.device != cache_device:
            W_new = W_new.to(cache_device)
            K = K.to(cache_device)
            R = R.to(cache_device)
            if mask is not None:
                mask = mask.to(cache_device)

        if mask is not None:
            # For masked updates, use torch.where for gradient-safe conditional assignment
            # Expand mask for broadcasting: [batch] -> [batch, 1, 1]
            mask_expanded = mask.view(batch_size, 1, 1)

            # Compute updates for all samples (W_new already has correct batch dim for masked samples)
            # We need to handle the case where W_new/K/R have different batch sizes
            if W_new.shape[0] == mask.sum():
                # W_new, K, R are only for masked samples - need to scatter back
                noise_old_masked = cache_accum['noise_accumulated'][mask]
                noise_transformed = torch.einsum("bij,bjk,blk->bil", W_new, noise_old_masked, W_new)
                noise_injection = torch.einsum("bij,bjk,blk->bil", K, R, K)
                new_noise_masked = noise_transformed + noise_injection

                W_combined_masked = torch.einsum(
                    "bij,bjk->bik", W_new, cache_accum['W_combined'][mask]
                )

                # Create full batch tensors with updates at masked positions
                # Use index_copy for gradient-safe scatter
                indices = torch.nonzero(mask, as_tuple=True)[0]
                new_noise_full = cache_accum['noise_accumulated'].clone()
                new_noise_full[indices] = new_noise_masked

                new_W_full = cache_accum['W_combined'].clone()
                new_W_full[indices] = W_combined_masked

                cache_accum['noise_accumulated'] = new_noise_full
                cache_accum['W_combined'] = new_W_full
            else:
                # W_new has full batch size - use torch.where
                noise_transformed = torch.einsum(
                    "bij,bjk,blk->bil", W_new, cache_accum['noise_accumulated'], W_new
                )
                noise_injection = torch.einsum("bij,bjk,blk->bil", K, R, K)
                new_noise = noise_transformed + noise_injection

                new_W = torch.einsum(
                    "bij,bjk->bik", W_new, cache_accum['W_combined']
                )

                cache_accum['noise_accumulated'] = torch.where(
                    mask_expanded, new_noise, cache_accum['noise_accumulated']
                )
                cache_accum['W_combined'] = torch.where(
                    mask_expanded, new_W, cache_accum['W_combined']
                )
        else:
            # Full batch update
            # noise_new = W_new @ noise_old @ W_new^T + K @ R @ K^T
            noise_transformed = torch.einsum("bij,bjk,blk->bil", W_new, cache_accum['noise_accumulated'], W_new)
            noise_injection = torch.einsum("bij,bjk,blk->bil", K, R, K)
            cache_accum['noise_accumulated'] = noise_transformed + noise_injection

            # W_combined = W_new @ W_combined
            cache_accum['W_combined'] = torch.einsum(
                "bij,bjk->bik", W_new, cache_accum['W_combined']
            )

    def compute_F_Q_matrices(
        self,
        quat_b_to_w: torch.Tensor,
        accel_b_raw: torch.Tensor,
        accel_bias_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute transition Jacobian F and process noise Q matrices.

        This is the core computation from predict() extracted for caching.

        Args:
            quat_b_to_w: Body-to-world quaternion [batch, 4]
            accel_b_raw: Raw accelerometer [batch, 3]
            accel_bias_b: Accelerometer bias [batch, 3]

        Returns:
            F: Transition Jacobian [batch, 15, 15]
            Q: Process noise [batch, 15, 15]
        """
        batch_size = quat_b_to_w.shape[0]
        dtype = quat_b_to_w.dtype
        device = quat_b_to_w.device

        # Build F matrix (state transition Jacobian)
        F_error_matrix = (
            torch.eye(self.error_state_dim, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Skew-symmetric matrix for acceleration
        accel_ssm = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # Fill F blocks
        F_error_matrix[:, 0:3, 3:6] = torch.eye(3, device=device, dtype=dtype) * self.dt
        F_error_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        F_error_matrix[:, 6:9, 9:12] = -torch.eye(3, device=device, dtype=dtype) * self.dt

        # Build Q matrix (process noise with trapezoidal integration)
        Q_continuous = torch.diag(self.Q_diag).unsqueeze(0).expand(batch_size, -1, -1)
        Q_error_matrix = 0.5 * (
            F_error_matrix @ Q_continuous @ F_error_matrix.transpose(-2, -1) + Q_continuous
        ) * self.dt

        return F_error_matrix, Q_error_matrix

    def parallel_covariance_from_cache(
        self,
        cache: ESKFSequenceCache,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Compute covariances using parallel scan with cached matrices.

        Uses the unified operator formulation:
            F̃_t = W_t @ F_t
            Q̃_t = W_t @ Q_t @ W_t^T + noise_inj_t
            P_{t|t} = F̃_t @ P_{t-1|t-1} @ F̃_t^T + Q̃_t

        The noise_inj_t is the properly accumulated noise injection from all updates,
        accounting for the transformation by subsequent W matrices.

        Args:
            cache: ESKFSequenceCache with F, Q, W, noise_inj sequences
            block_size: Block size for parallel scan

        Returns:
            P_seq: Covariance sequence [batch, seq_len, 15, 15]
        """
        from model.parallel_scan_ops import parallel_covariance_scan_blocked

        # Build unified operators: F̃ = W @ F, Q̃ = W @ Q @ W^T + noise_inj
        F_unified = torch.einsum("btij,btjk->btik", cache.W_seq, cache.F_seq)

        WQW = torch.einsum("btij,btjk,btlk->btil", cache.W_seq, cache.Q_seq, cache.W_seq)
        Q_unified = WQW + cache.noise_inj_seq

        # Parallel covariance scan
        P_seq = parallel_covariance_scan_blocked(F_unified, Q_unified, cache.P_init, block_size)

        return P_seq

    # =========================================================================
    # HYBRID CPU/GPU MODE: CPU nominal state + GPU parallel P computation
    # =========================================================================

    def init_cache_hybrid_cpu(
        self,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize cache on CPU for hybrid CPU/GPU mode.

        In hybrid mode:
        - Sequential nominal state propagation runs on CPU (avoids GPU kernel overhead)
        - F, Q matrices are cached on CPU during the loop
        - After loop, F, Q are transferred to GPU for parallel P computation

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type for cache tensors
        """
        self._hybrid_mode = True
        self._hybrid_cache_step = 0
        self._hybrid_batch_size = batch_size
        self._hybrid_seq_len = seq_len

        # Pre-allocate cache tensors ON CPU
        D = self.error_state_dim
        self._hybrid_cache_F = torch.zeros(batch_size, seq_len, D, D, device='cpu', dtype=dtype)
        self._hybrid_cache_Q = torch.zeros(batch_size, seq_len, D, D, device='cpu', dtype=dtype)
        self._hybrid_cache_P_init = None

        # CPU copies of filter parameters for fast access
        self._cpu_Q_diag = self.Q_diag.cpu()
        self._cpu_dt = self.dt

    def forward_nominal_cpu(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_b_raw: torch.Tensor,
        accel_b_raw: torch.Tensor,
        gravity_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU-based nominal state propagation with F, Q caching.

        This method runs the sequential nominal state update on CPU to avoid
        GPU kernel launch overhead for small O(1) operations. F and Q matrices
        are computed and cached for later GPU-based parallel P computation.

        All input tensors should be on CPU. Returns updated state on CPU.

        Args:
            pos_w: Position in world frame [batch, 3] (CPU)
            vel_w: Velocity in world frame [batch, 3] (CPU)
            quat_b_to_w: Body-to-world quaternion [batch, 4] (CPU)
            gyro_bias_b: Gyroscope bias [batch, 3] (CPU)
            accel_bias_b: Accelerometer bias [batch, 3] (CPU)
            gyro_b_raw: Raw gyroscope measurement [batch, 3] (CPU)
            accel_b_raw: Raw accelerometer measurement [batch, 3] (CPU)
            gravity_w: Gravity vector in world frame [batch, 3] (CPU)

        Returns:
            Tuple of (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new,
                     accel_bias_b_new, F_matrix, Q_matrix) all on CPU
        """
        batch_size = pos_w.shape[0]
        dt = self._cpu_dt

        # --- Nominal State Propagation (CPU) ---
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Quaternion propagation
        angle_change = gyro_b_corrected * dt
        delta_quat = small_angle_to_quaternion(angle_change)
        quat_b_to_w_new = quaternion_multiply(quat_b_to_w, delta_quat)
        quat_b_to_w_new = torch.nn.functional.normalize(quat_b_to_w_new, p=2, dim=-1)

        # Rotation matrices
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        rot_mat_b_to_w_new = quaternion_to_rotation_matrix(quat_b_to_w_new)

        # Acceleration in world frame
        accel_w = (rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - gravity_w
        accel_w_new = (rot_mat_b_to_w_new @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - gravity_w

        # Trapezoidal integration
        vel_w_new = vel_w + 0.5 * (accel_w + accel_w_new) * dt
        pos_w_new = pos_w + 0.5 * (vel_w + vel_w_new) * dt

        # Biases (random walk - unchanged)
        gyro_bias_b_new = gyro_bias_b
        accel_bias_b_new = accel_bias_b

        # --- Compute F, Q matrices (CPU) ---
        D = self.error_state_dim
        F_matrix = torch.eye(D, device='cpu', dtype=pos_w.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        # Skew-symmetric matrix for acceleration
        accel_ssm = torch.zeros(batch_size, 3, 3, device='cpu', dtype=pos_w.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # F matrix entries
        F_matrix[:, 0:3, 3:6] = torch.eye(3, device='cpu') * dt
        F_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * dt
        F_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * dt
        F_matrix[:, 6:9, 9:12] = -torch.eye(3, device='cpu') * dt

        # Q matrix (trapezoidal)
        Q_continuous = torch.diag(self._cpu_Q_diag).unsqueeze(0).expand(batch_size, -1, -1)
        Q_matrix = 0.5 * (F_matrix @ Q_continuous @ F_matrix.transpose(-2, -1) + Q_continuous) * dt

        # Cache F, Q
        if getattr(self, '_hybrid_mode', False):
            t = self._hybrid_cache_step
            if t == 0:
                self._hybrid_cache_P_init = None  # Will be set from GPU P_error
            self._hybrid_cache_F[:, t] = F_matrix
            self._hybrid_cache_Q[:, t] = Q_matrix
            self._hybrid_cache_step += 1

        return pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new, F_matrix, Q_matrix

    def compute_P_parallel_gpu(
        self,
        P_init: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Compute P sequence using GPU parallel scan from CPU-cached F, Q.

        Call this after the CPU nominal state loop completes. Transfers
        cached F, Q to GPU and computes all P matrices using parallel scan.

        Args:
            P_init: Initial covariance matrix [batch, 15, 15] (can be GPU or CPU)
            block_size: Block size for parallel scan

        Returns:
            P_seq: Covariance sequence [batch, seq_len, 15, 15] on GPU
        """
        from model.parallel_scan_ops import parallel_covariance_scan_blocked

        if not getattr(self, '_hybrid_mode', False):
            raise RuntimeError("Hybrid mode not initialized. Call init_cache_hybrid_cpu first.")

        seq_len = self._hybrid_cache_step

        # Transfer F, Q from CPU to GPU
        F_gpu = self._hybrid_cache_F[:, :seq_len].to(self.device)
        Q_gpu = self._hybrid_cache_Q[:, :seq_len].to(self.device)
        P_init_gpu = P_init.to(self.device) if P_init.device.type == 'cpu' else P_init

        # Parallel scan on GPU
        P_seq = parallel_covariance_scan_blocked(F_gpu, Q_gpu, P_init_gpu, block_size)

        return P_seq

    def finalize_hybrid_cache(self):
        """Clean up hybrid mode cache."""
        self._hybrid_mode = False
        if hasattr(self, '_hybrid_cache_F'):
            del self._hybrid_cache_F
        if hasattr(self, '_hybrid_cache_Q'):
            del self._hybrid_cache_Q
        if hasattr(self, '_hybrid_cache_P_init'):
            del self._hybrid_cache_P_init
        if hasattr(self, '_cpu_Q_diag'):
            del self._cpu_Q_diag

    def _make_symmetric(self, P_covariance: torch.Tensor) -> torch.Tensor:
        """Enforces symmetry on covariance matrix to prevent numerical drift.

        Args:
            P_covariance: Covariance matrix to symmetrize.

        Returns:
            Symmetrized covariance matrix via (P + P^T) / 2.
        """
        return 0.5 * (P_covariance + P_covariance.transpose(-2, -1))

    def _solve_symmetric_system(
        self,
        S: torch.Tensor,
        B: torch.Tensor,
        compute_quadratic: bool = False,
        quadratic_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Solve symmetric positive definite linear system using Cholesky decomposition.

        Computes X = S^{-1} @ B using Cholesky decomposition for numerical stability.
        Optionally computes quadratic form v^T @ S^{-1} @ v for Mahalanobis distance.

        This method consolidates the repeated try-except Cholesky pattern throughout
        the codebase, providing a single point for numerical stability handling.

        Args:
            S: Symmetric positive definite matrix [batch, n, n].
            B: Right-hand side matrix [batch, n, m] or [batch, n].
            compute_quadratic: If True, compute v^T @ S^{-1} @ v.
            quadratic_vec: Vector v for quadratic form [batch, n]. Required if compute_quadratic=True.

        Returns:
            Tuple of:
                - X: Solution to S @ X = B, shape [batch, n, m] or [batch, n].
                - quadratic_result: v^T @ S^{-1} @ v if compute_quadratic, else None.
        """
        quadratic_result = None
        B_is_1d = B.dim() == 2

        if B_is_1d:
            B = B.unsqueeze(-1)

        # Direct Cholesky solve (no try-except) for torch.compile compatibility.
        # Regularization ensures S is positive definite.
        L = torch.linalg.cholesky(S)
        X = torch.cholesky_solve(B, L)

        if compute_quadratic and quadratic_vec is not None:
            sol_v = torch.cholesky_solve(quadratic_vec.unsqueeze(-1), L)
            quadratic_result = (quadratic_vec.unsqueeze(1) @ sol_v).squeeze(-1).squeeze(-1)

        if B_is_1d:
            X = X.squeeze(-1)

        return X, quadratic_result

    def _propagate_nominal_state(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_b_raw: torch.Tensor,
        accel_b_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagates nominal state using non-linear dynamics f(x_nom, u).

        Args:
            pos_w: Current position in world frame.
            vel_w: Current velocity in world frame.
            quat_b_to_w: Current body-to-world quaternion.
            gyro_bias_b: Current gyroscope bias in body frame.
            accel_bias_b: Current accelerometer bias in body frame.
            gyro_b_raw: Raw gyroscope measurements.
            accel_b_raw: Raw accelerometer measurements.

        Returns:
            Propagated state (pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b).
        """
        gyro_b_corrected = gyro_b_raw - gyro_bias_b
        accel_b_corrected = accel_b_raw - accel_bias_b

        # Quaternion propagation via exponential map
        angle_change = gyro_b_corrected * self.dt
        delta_quat = small_angle_to_quaternion(angle_change)
        quat_b_to_w_new = quaternion_multiply(quat_b_to_w, delta_quat)

        # Trapezoidal integration for position and velocity
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_w = (rot_mat_b_to_w @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w

        rot_mat_b_to_w_new = quaternion_to_rotation_matrix(quat_b_to_w_new)
        accel_w_new = (rot_mat_b_to_w_new @ accel_b_corrected.unsqueeze(-1)).squeeze(-1) - self.gravity_w

        vel_w_new = vel_w + 0.5 * (accel_w + accel_w_new) * self.dt
        pos_w_new = pos_w + 0.5 * (vel_w + vel_w_new) * self.dt

        # Biases modeled as random walks
        gyro_bias_b_new = gyro_bias_b
        accel_bias_b_new = accel_bias_b

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new)

    def predict(
        self,
        P_error_covariance: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        accel_b_raw: torch.Tensor,
        accel_bias_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts error covariance using linearized dynamics.

        Computes P_k|k-1 = F * P_k-1|k-1 * F^T + Q

        Args:
            P_error_covariance: Current 15x15 error covariance (P_k-1|k-1).
            quat_b_to_w: Current body-to-world quaternion.
            accel_b_raw: Raw accelerometer measurements.
            accel_bias_b: Current accelerometer bias.

        Returns:
            Predicted 15x15 error covariance (P_k|k-1).
        """
        batch_size = P_error_covariance.shape[0]
        # Infer device from input tensors to support CPU state propagation
        device = P_error_covariance.device

        F_error_matrix = (
            torch.eye(self.error_state_dim, device=device, dtype=P_error_covariance.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        accel_b_corrected = accel_b_raw - accel_bias_b

        # [a]_x for Jacobian
        accel_ssm = torch.zeros(batch_size, 3, 3, device=device, dtype=P_error_covariance.dtype)
        accel_ssm[:, 0, 1] = -accel_b_corrected[:, 2]
        accel_ssm[:, 0, 2] = accel_b_corrected[:, 1]
        accel_ssm[:, 1, 0] = accel_b_corrected[:, 2]
        accel_ssm[:, 1, 2] = -accel_b_corrected[:, 0]
        accel_ssm[:, 2, 0] = -accel_b_corrected[:, 1]
        accel_ssm[:, 2, 1] = accel_b_corrected[:, 0]

        # F: δp' = δp + δv*dt, δv' = δv - R[a]_x*δθ*dt - R*δa*dt, δθ' = δθ - δω*dt
        F_error_matrix[:, 0:3, 3:6] = torch.eye(3, device=device) * self.dt
        F_error_matrix[:, 3:6, 6:9] = -rot_mat_b_to_w @ accel_ssm * self.dt
        F_error_matrix[:, 3:6, 12:15] = -rot_mat_b_to_w * self.dt
        F_error_matrix[:, 6:9, 9:12] = -torch.eye(3, device=device) * self.dt

        # Q: trapezoidal integration
        Q_continuous = torch.diag(self.Q_diag).unsqueeze(0).expand(batch_size, -1, -1)
        Q_error_matrix = 0.5 * (
            F_error_matrix @ Q_continuous @ F_error_matrix.transpose(-2, -1) + Q_continuous
        ) * self.dt

        # P_k|k-1 = FPF^T + Q
        P_predicted = (
            F_error_matrix @ P_error_covariance @ F_error_matrix.transpose(-2, -1)
            + Q_error_matrix
        )

        return self._make_symmetric(P_predicted)

    def update(
        self,
        P_error_pred: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        accel_bias_b: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        measurement: torch.Tensor,
        R_override: Optional[torch.Tensor] = None,
        gating_threshold: Optional[float] = None,
        return_update_matrices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Performs bias-only measurement update (attitude handled by gravity alignment).

        This update observes accelerometer and gyroscope biases only.
        Attitude correction is handled separately by _apply_gravity_alignment_update
        to avoid double gravity observation.

        Args:
            P_error_pred: Predicted 15x15 error covariance (P_k|k-1).
            quat_b_to_w: Current body-to-world quaternion.
            accel_bias_b: Current accelerometer bias.
            gyro_bias_b: Current gyroscope bias.
            measurement: 6D sensor measurement (z_k).
            R_override: Optional measurement noise covariance override.
            gating_threshold: Optional Mahalanobis distance threshold.
            return_update_matrices: If True, also return (W, K, R) for caching.

        Returns:
            Tuple of (delta_x, P_error_new, innovation, mahalanobis_sq) or
            (delta_x, P_error_new, innovation, mahalanobis_sq, (W, K, R)) if return_update_matrices=True.
        """
        batch_size = P_error_pred.shape[0]
        device = P_error_pred.device

        H_error_matrix = torch.zeros(
            batch_size, self.obs_dim, self.error_state_dim, device=device, dtype=P_error_pred.dtype
        )

        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w).transpose(-2, -1)
        gravity_body = (rot_mat_world_to_body @ self.gravity_w.to(device).unsqueeze(0).T).squeeze(-1)

        # H: bias-only observation (no attitude coupling)
        # Attitude correction handled by gravity alignment update
        # - Accel measures: gravity + accel_bias → observe accel_bias only
        # - Gyro measures: gyro_bias → observe gyro_bias
        H_error_matrix[:, 0:3, 12:15] = torch.eye(3, device=device)  # accel → accel_bias
        H_error_matrix[:, 3:6, 9:12] = torch.eye(3, device=device)   # gyro → gyro_bias

        # y = z - h(x)
        accel_pred = gravity_body + accel_bias_b
        gyro_pred = gyro_bias_b
        h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)
        innovation = measurement - h_predicted

        # Adaptive R: scale by |‖a‖ - g|
        if R_override is not None:
            R_noise_matrix = R_override + torch.eye(self.obs_dim, device=device) * 1e-6
        else:
            accel_meas = measurement[..., 0:3]
            gravity_w = self.gravity_w.to(device)
            accel_norm_diff = torch.abs(torch.norm(accel_meas, dim=-1, keepdim=True) - torch.norm(gravity_w))
            # CRITICAL: Clamp accel_norm_diff to prevent exponential explosion
            # Max diff of 20 m/s² (realistic for handwriting + safety margin)
            accel_norm_diff = torch.clamp(accel_norm_diff, max=20.0)
            scaling_factor = torch.exp(self.adaptive_gain * accel_norm_diff)
            # Additional safety: clamp scaling factor to [1.0, 1000.0]
            scaling_factor = torch.clamp(scaling_factor, min=1.0, max=1000.0)
            base_R = torch.diag_embed(F.softplus(self.R_diag.to(device)) + 1e-6)
            R_noise_matrix = base_R.unsqueeze(0).expand(batch_size, -1, -1).clone()
            R_noise_matrix[..., 0:3, 0:3] *= scaling_factor.unsqueeze(-1)

        # S = HPH^T + R, K = PH^T S^-1
        S_matrix = H_error_matrix @ P_error_pred @ H_error_matrix.transpose(-2, -1) + R_noise_matrix
        # Add regularization to ensure S_matrix is well-conditioned for solve operations
        S_matrix = S_matrix + torch.eye(self.obs_dim, device=device) * 1e-5

        # Solve for Kalman gain and Mahalanobis distance using unified helper
        # K = P @ H^T @ S^{-1} = (S^{-1} @ H @ P^T)^T
        K_gain_T, mahalanobis_sq = self._solve_symmetric_system(
            S_matrix,
            H_error_matrix @ P_error_pred.transpose(-2, -1),
            compute_quadratic=True,
            quadratic_vec=innovation,
        )
        K_gain = K_gain_T.transpose(-2, -1)

        # δx = Ky
        delta_x = (K_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        # Clamp to reasonable range: [0, 1e6] prevents numerical issues in gating logic
        # Chi-square distribution for 6 DOF has p=0.99999 at ~27, so 1e6 is extremely conservative
        mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0, max=1e6)

        # P: Joseph form for numerical stability
        I_matrix = torch.eye(self.error_state_dim, device=device)
        ImKH = I_matrix - K_gain @ H_error_matrix
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gain @ R_noise_matrix @ K_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        # Mahalanobis gating
        if gating_threshold is not None:
             reject_mask = mahalanobis_sq > gating_threshold
             delta_x = torch.where(reject_mask.unsqueeze(-1), torch.zeros_like(delta_x), delta_x)
             P_error_new = torch.where(reject_mask.unsqueeze(-1).unsqueeze(-1), P_error_pred, P_error_new)
             # For gated samples, use identity W and zero K (so noise_inj = K @ R @ K^T = 0)
             if return_update_matrices:
                 ImKH = torch.where(
                     reject_mask.unsqueeze(-1).unsqueeze(-1),
                     I_matrix.unsqueeze(0).expand(batch_size, -1, -1),
                     ImKH
                 )
                 # Zero out K for gated samples to ensure noise_inj = 0
                 K_gain = torch.where(
                     reject_mask.unsqueeze(-1).unsqueeze(-1),
                     torch.zeros_like(K_gain),
                     K_gain
                 )

        if return_update_matrices:
            return delta_x, P_error_new, innovation, mahalanobis_sq, (ImKH, K_gain, R_noise_matrix)

        return delta_x, P_error_new, innovation, mahalanobis_sq, None

    def _calculate_stationary_update(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        gyro_pred: torch.Tensor,
        tcn_zupt_prob: Optional[torch.Tensor] = None,
        use_zaru: bool = False,
        return_update_matrices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Applies stationary update: ZUPT (Zero-Velocity) and optionally ZARU (Zero Angular Rate).

        For ESKF-TCN: Applies both ZUPT and ZARU constraints when stationary.
        For classical ZUPT: Only applies velocity constraint.

        Args:
            vel_w_pred: Predicted velocity in world frame.
            P_error_pred: Predicted 15x15 error covariance.
            gyro_pred: Predicted gyroscope (bias estimate) in body frame.
            tcn_zupt_prob: Optional TCN-predicted zero-velocity probability [0,1].
            use_zaru: If True, also applies zero angular rate constraint (ESKF-TCN only).
            return_update_matrices: If True, also return (W, K, R) for caching.

        Returns:
            Tuple of (delta_x_stationary, P_error_new) or
            (delta_x_stationary, P_error_new, (W, K, R)) if return_update_matrices=True.
        """
        batch_size = P_error_pred.shape[0]
        device = P_error_pred.device

        onset = Config.ESKFTCN.ZUPT_DECAY_ONSET
        exponent = Config.ESKFTCN.ZUPT_DECAY_EXPONENT

        if use_zaru:
            # ZUPT + ZARU: Constrain both velocity and angular rate
            meas_dim = 6
            H_stationary = torch.zeros(batch_size, meas_dim, self.error_state_dim, device=device, dtype=P_error_pred.dtype)
            H_stationary[:, 0:3, 3:6] = torch.eye(3, device=device)  # Velocity error δv
            H_stationary[:, 3:6, 9:12] = torch.eye(3, device=device)  # Gyro bias error δω_bias

            # Innovation: [velocity_error; gyro_error]
            innovation_stationary = torch.cat([-vel_w_pred, -gyro_pred], dim=-1)

            # R: adaptive via TCN probability for both velocity and gyro
            if tcn_zupt_prob is not None:
                if tcn_zupt_prob.ndim == 1:
                    tcn_zupt_prob = tcn_zupt_prob.unsqueeze(-1)

                clamped_prob = torch.clamp(tcn_zupt_prob, 1e-4, 1.0)

                prob_above_onset = torch.clamp((clamped_prob - onset) / (1.0 - onset + 1e-6), 0.0, 1.0)
                alpha = prob_above_onset ** exponent

                min_R_val = self.zupt_noise_std**2
                max_R_val = min_R_val * 1e6

                log_R_min = torch.log(torch.tensor(min_R_val, device=device))
                log_R_max = torch.log(torch.tensor(max_R_val, device=device))

                log_R_val = (1.0 - alpha) * log_R_max + alpha * log_R_min
                R_val = torch.exp(log_R_val)

                R_zupt_scaled_diag = R_val.repeat(1, 3) if R_val.shape[-1] == 1 else R_val
                # ZARU noise: slightly higher than ZUPT (gyro bias has more uncertainty)
                R_zaru_scaled_diag = R_zupt_scaled_diag * 2.0

                R_combined_diag = torch.cat([R_zupt_scaled_diag, R_zaru_scaled_diag], dim=-1)
                R_stationary_matrix = torch.diag_embed(R_combined_diag)
            else:
                R_zupt_base = self.get_R_zupt()
                R_zaru_base = R_zupt_base * 2.0
                R_combined = torch.cat([torch.diag(R_zupt_base), torch.diag(R_zaru_base)], dim=0)
                R_stationary_matrix = torch.diag(R_combined).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # ZUPT only: Constrain velocity
            meas_dim = 3
            H_stationary = torch.zeros(batch_size, meas_dim, self.error_state_dim, device=device, dtype=P_error_pred.dtype)
            H_stationary[:, :, 3:6] = torch.eye(3, device=device)

            innovation_stationary = -vel_w_pred

            # R_ZUPT: adaptive via TCN probability (high prob → low R)
            if tcn_zupt_prob is not None:
                if tcn_zupt_prob.ndim == 1:
                    tcn_zupt_prob = tcn_zupt_prob.unsqueeze(-1)

                clamped_prob = torch.clamp(tcn_zupt_prob, 1e-4, 1.0)

                prob_above_onset = torch.clamp((clamped_prob - onset) / (1.0 - onset + 1e-6), 0.0, 1.0)
                alpha = prob_above_onset ** exponent

                min_R_val = self.zupt_noise_std**2
                max_R_val = min_R_val * 1e6

                log_R_min = torch.log(torch.tensor(min_R_val, device=device))
                log_R_max = torch.log(torch.tensor(max_R_val, device=device))

                log_R_val = (1.0 - alpha) * log_R_max + alpha * log_R_min
                R_val = torch.exp(log_R_val)

                R_zupt_scaled_diag = R_val.repeat(1, 3) if R_val.shape[-1] == 1 else R_val

                R_stationary_matrix = torch.diag_embed(R_zupt_scaled_diag)
            else:
                R_stationary_matrix = self.get_R_zupt().unsqueeze(0).expand(batch_size, -1, -1)

        S_stationary_matrix = H_stationary @ P_error_pred @ H_stationary.transpose(-2, -1) + R_stationary_matrix
        # Regularization for numerical stability
        S_stationary_matrix = S_stationary_matrix + torch.eye(meas_dim, device=device) * Config.MOTION.S_MATRIX_REGULARIZATION

        # Solve for Kalman gain using unified helper
        K_stationary_gain_T, _ = self._solve_symmetric_system(
            S_stationary_matrix,
            H_stationary @ P_error_pred.transpose(-2, -1),
        )
        K_stationary_gain = K_stationary_gain_T.transpose(-2, -1)

        delta_x_stationary = (K_stationary_gain @ innovation_stationary.unsqueeze(-1)).squeeze(-1)

        # P: Joseph form
        I_matrix = torch.eye(self.error_state_dim, device=device)
        ImKH_stationary = I_matrix - K_stationary_gain @ H_stationary
        P_error_new = (
            ImKH_stationary @ P_error_pred @ ImKH_stationary.transpose(-2, -1)
            + K_stationary_gain @ R_stationary_matrix @ K_stationary_gain.transpose(-2, -1)
        )
        P_error_new = self._make_symmetric(P_error_new)

        if return_update_matrices:
            # Pad K and R to obs_dim=6 for consistent caching
            batch_size = P_error_pred.shape[0]
            K_padded = torch.zeros(batch_size, self.error_state_dim, self.obs_dim,
                                   device=self.device, dtype=P_error_pred.dtype)
            R_padded = torch.zeros(batch_size, self.obs_dim, self.obs_dim,
                                   device=self.device, dtype=P_error_pred.dtype)
            K_padded[:, :, :meas_dim] = K_stationary_gain
            R_padded[:, :meas_dim, :meas_dim] = R_stationary_matrix
            return delta_x_stationary, P_error_new, (ImKH_stationary, K_padded, R_padded)

        return delta_x_stationary, P_error_new, None

    def _apply_tcn_velocity_correction(
        self,
        vel_w_pred: torch.Tensor,
        P_error_pred: torch.Tensor,
        vel_corr_b: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        R_vel_diag: Optional[torch.Tensor] = None,
        return_update_matrices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply TCN velocity correction as pseudo-measurement.

        Args:
            vel_w_pred: Predicted velocity in world frame [batch, 3].
            P_error_pred: Error covariance [batch, 15, 15].
            vel_corr_b: Velocity correction in body frame [batch, 3].
            quat_b_to_w: Body-to-world quaternion [batch, 4].
            R_vel_diag: Measurement noise diagonal [batch, 3], already processed.
            return_update_matrices: If True, also return (W, K, R) for caching.

        Returns:
            delta_x_tcn: State correction [batch, 15].
            P_error_new: Updated covariance [batch, 15, 15].
            update_matrices: (W, K, R) if return_update_matrices=True, else None.
        """
        batch_size = P_error_pred.shape[0]
        dtype = P_error_pred.dtype
        device = P_error_pred.device

        # H matrix: velocity observation
        H_tcn = torch.zeros(batch_size, 3, self.error_state_dim, device=device, dtype=dtype)
        H_tcn[:, :, 3:6] = torch.eye(3, device=device, dtype=dtype)

        # Transform velocity correction to world frame
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        vel_corr_w = (rot_mat_b_to_w @ vel_corr_b.unsqueeze(-1)).squeeze(-1)

        # R matrix from pre-processed TCN covariance
        if R_vel_diag is not None:
            R_tcn_matrix = torch.diag_embed(R_vel_diag)
        else:
            R_tcn_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1) * 1e-2

        # Kalman update using unified helper
        S_tcn = H_tcn @ P_error_pred @ H_tcn.transpose(-2, -1) + R_tcn_matrix
        # Add regularization for numerical stability (critical for Cholesky)
        S_tcn = S_tcn + torch.eye(3, device=device, dtype=dtype) * 1e-6
        K_T, _ = self._solve_symmetric_system(
            S_tcn,
            H_tcn @ P_error_pred.transpose(-2, -1),
        )
        K = K_T.transpose(-2, -1)

        delta_x_tcn = (K @ vel_corr_w.unsqueeze(-1)).squeeze(-1)

        # Joseph form covariance update
        I = torch.eye(self.error_state_dim, device=device, dtype=dtype)
        ImKH = I - K @ H_tcn
        P_error_new = ImKH @ P_error_pred @ ImKH.transpose(-2, -1) + K @ R_tcn_matrix @ K.transpose(-2, -1)
        P_error_new = self._make_symmetric(P_error_new)

        if return_update_matrices:
            # Pad K and R to obs_dim=6 for consistent caching
            K_padded = torch.zeros(batch_size, self.error_state_dim, self.obs_dim,
                                   device=device, dtype=dtype)
            R_padded = torch.zeros(batch_size, self.obs_dim, self.obs_dim,
                                   device=device, dtype=dtype)
            K_padded[:, :, :3] = K
            R_padded[:, :3, :3] = R_tcn_matrix
            return delta_x_tcn, P_error_new, (ImKH, K_padded, R_padded)

        return delta_x_tcn, P_error_new, None

    def _apply_gravity_alignment_update(
        self,
        P_error_pred: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gravity_measured_b: torch.Tensor,
        R_gravity_diag: torch.Tensor,
        accel_b_raw: Optional[torch.Tensor] = None,
        return_update_matrices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply gravity alignment update to correct orientation error.

        Uses gravity observation to correct roll/pitch:
        - ZUPT: accelerometer as gravity measurement
        - Motion: TCN-predicted gravity direction

        Measurement model: h(q) = R(q)^T @ g_world
        Jacobian: H = -[g_body]_× (skew-symmetric)

        Args:
            P_error_pred: Error covariance [batch, 15, 15].
            quat_b_to_w: Body-to-world quaternion [batch, 4].
            gravity_measured_b: Gravity direction in body frame [batch, 3].
            R_gravity_diag: Measurement noise [batch, 3].
            accel_b_raw: Raw accelerometer for gating [batch, 3].
            return_update_matrices: If True, also return (W, K, R) for caching.

        Returns:
            delta_x_gravity: State correction [batch, 15].
            P_error_new: Updated covariance [batch, 15, 15].
            update_matrices: (W, K, R) if return_update_matrices=True, else None.
        """
        batch_size = P_error_pred.shape[0]
        dtype = P_error_pred.dtype
        device = P_error_pred.device

        # Expected gravity in body frame from current orientation
        rot_mat_b_to_w = quaternion_to_rotation_matrix(quat_b_to_w)
        rot_mat_w_to_b = rot_mat_b_to_w.transpose(-1, -2)
        gravity_w = self.gravity_w.to(device)
        gravity_expected_b = (rot_mat_w_to_b @ gravity_w.unsqueeze(-1)).squeeze(-1)
        gravity_expected_b_unit = F.normalize(gravity_expected_b, p=2, dim=-1, eps=1e-6)

        # Innovation: measured - expected (both unit vectors)
        innovation_gravity = gravity_measured_b - gravity_expected_b_unit

        # Gating: reject when accel magnitude far from gravity (high dynamics)
        if accel_b_raw is not None:
            accel_norm = torch.norm(accel_b_raw, dim=-1, keepdim=True)
            accel_min = Config.ESKFTCN.GRAVITY_ACCEL_MIN
            accel_max = Config.ESKFTCN.GRAVITY_ACCEL_MAX
            valid_mask = ((accel_norm > accel_min) & (accel_norm < accel_max)).squeeze(-1)
        else:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # H matrix: Jacobian of gravity w.r.t. orientation error
        # H[:, :, 6:9] = -[g_body]_× (skew-symmetric)
        H_gravity = torch.zeros(batch_size, 3, self.error_state_dim, device=device, dtype=dtype)
        g = gravity_expected_b_unit
        skew_g = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        skew_g[:, 0, 1] = -g[:, 2]
        skew_g[:, 0, 2] = g[:, 1]
        skew_g[:, 1, 0] = g[:, 2]
        skew_g[:, 1, 2] = -g[:, 0]
        skew_g[:, 2, 0] = -g[:, 1]
        skew_g[:, 2, 1] = g[:, 0]
        H_gravity[:, :, 6:9] = -skew_g

        # R matrix from TCN's adaptive covariance (per-sample diagonal)
        R_gravity_matrix = torch.diag_embed(R_gravity_diag)

        # Kalman update using unified helper
        S_gravity = H_gravity @ P_error_pred @ H_gravity.transpose(-2, -1) + R_gravity_matrix
        # Regularization for numerical stability
        S_gravity = S_gravity + torch.eye(3, device=device, dtype=dtype) * Config.MOTION.S_MATRIX_REGULARIZATION

        K_gravity_T, _ = self._solve_symmetric_system(
            S_gravity,
            H_gravity @ P_error_pred.transpose(-2, -1),
        )
        K_gravity = K_gravity_T.transpose(-2, -1)

        delta_x_gravity = (K_gravity @ innovation_gravity.unsqueeze(-1)).squeeze(-1)

        # Apply gating mask
        delta_x_gravity = torch.where(
            valid_mask.unsqueeze(-1), delta_x_gravity, torch.zeros_like(delta_x_gravity)
        )

        # Joseph form covariance update
        I_matrix = torch.eye(self.error_state_dim, device=device, dtype=dtype)
        ImKH = I_matrix - K_gravity @ H_gravity
        P_error_new = (
            ImKH @ P_error_pred @ ImKH.transpose(-2, -1)
            + K_gravity @ R_gravity_matrix @ K_gravity.transpose(-2, -1)
        )
        P_error_new = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1), P_error_new, P_error_pred
        )
        P_error_new = self._make_symmetric(P_error_new)

        if return_update_matrices:
            # Pad K and R to obs_dim=6 for consistent caching
            K_padded = torch.zeros(batch_size, self.error_state_dim, self.obs_dim,
                                   device=device, dtype=dtype)
            R_padded = torch.zeros(batch_size, self.obs_dim, self.obs_dim,
                                   device=device, dtype=dtype)
            K_padded[:, :, :3] = K_gravity
            R_padded[:, :3, :3] = R_gravity_matrix
            # Apply valid_mask to W (identity for invalid samples)
            W_masked = torch.where(
                valid_mask.unsqueeze(-1).unsqueeze(-1), ImKH, I_matrix
            )
            # Zero out K for invalid samples to ensure noise_inj = 0
            K_padded = torch.where(
                valid_mask.unsqueeze(-1).unsqueeze(-1), K_padded, torch.zeros_like(K_padded)
            )
            return delta_x_gravity, P_error_new, (W_masked, K_padded, R_padded)

        return delta_x_gravity, P_error_new, None

    def inject_correction(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        delta_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Injects error state correction into nominal state.

        Args:
            pos_w: Current position in world frame.
            vel_w: Current velocity in world frame.
            quat_b_to_w: Current body-to-world quaternion.
            gyro_bias_b: Current gyroscope bias.
            accel_bias_b: Current accelerometer bias.
            delta_x: 15D error state.

        Returns:
            Corrected nominal state components.
        """
        d_pos_w, d_vel_w, d_theta_b, d_gyro_bias_b, d_accel_bias_b = delta_x.split([3, 3, 3, 3, 3], dim=-1)

        pos_w_new = pos_w + d_pos_w
        vel_w_new = vel_w + d_vel_w

        quat_b_to_w_new = quaternion_multiply(
            quat_b_to_w, small_angle_to_quaternion(d_theta_b)
        )
        quat_b_to_w_new = F.normalize(quat_b_to_w_new, p=2, dim=-1)

        gyro_bias_b_new = gyro_bias_b + d_gyro_bias_b
        accel_bias_b_new = accel_bias_b + d_accel_bias_b

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new)

    def forward(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        quat_b_to_w: torch.Tensor,
        gyro_bias_b: torch.Tensor,
        accel_bias_b: torch.Tensor,
        P_error: torch.Tensor,
        gyro_b_raw: torch.Tensor,
        accel_b_raw: torch.Tensor,
        force_raw: torch.Tensor,
        measurement: torch.Tensor,
        tcn_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """Executes one full predict-update-inject cycle of the ESKF.

        Args:
            pos_w (torch.Tensor): Current nominal position.
                - Shape: (Batch, 3) | Unit: m | Frame: World
            vel_w (torch.Tensor): Current nominal velocity.
                - Shape: (Batch, 3) | Unit: m/s | Frame: World
            quat_b_to_w (torch.Tensor): Current nominal quaternion.
                - Shape: (Batch, 4) | Frame: Body-to-World
            gyro_bias_b (torch.Tensor): Current nominal gyroscope bias.
                - Shape: (Batch, 3) | Unit: rad/s | Frame: Body
            accel_bias_b (torch.Tensor): Current nominal accelerometer bias.
                - Shape: (Batch, 3) | Unit: m/s^2 | Frame: Body
            P_error (torch.Tensor): Current error covariance matrix.
                - Shape: (Batch, 15, 15)
            gyro_b_raw (torch.Tensor): Raw gyroscope measurements.
                - Shape: (Batch, 3) | Unit: rad/s | Frame: Body
            accel_b_raw (torch.Tensor): Raw accelerometer measurements.
                - Shape: (Batch, 3) | Unit: m/s^2 | Frame: Body
            force_raw (torch.Tensor): Raw force sensor measurement.
                - Shape: (Batch, 1) | Unit: N
            measurement (torch.Tensor): The 6D sensor measurement vector.
                - Shape: (Batch, 6) | [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            tcn_output (Optional[Dict[str, torch.Tensor]]): Optional TCN predictions.

        Returns:
            Tuple[...]: Updated state components and features:
                - pos_w_new: (Batch, 3)
                - vel_w_new: (Batch, 3)
                - quat_b_to_w_new: (Batch, 4)
                - gyro_bias_b_new: (Batch, 3)
                - accel_bias_b_new: (Batch, 3)
                - P_error_final: (Batch, 15, 15)
                - tcn_features: Dict with keys "body_velocity" (Batch, 3), "zupt_flag" (Batch, 1), "innovation" (Batch, 6), "mahalanobis" (Batch, 1)
        """
        batch_size = pos_w.shape[0]
        caching_enabled = getattr(self, '_cache_enabled', False)

        # --- 0. Initialize Caching ---
        if caching_enabled:
            # Use cache device (GPU) even when state is on CPU
            cache_device = getattr(self, '_cache_device', self.device)
            # Cache P_init on first step
            if self._cache_step == 0:
                # Move to cache device if necessary
                if P_error.device != cache_device:
                    self._cache_P_init = P_error.to(cache_device).clone()
                else:
                    self._cache_P_init = P_error.clone()
            # Initialize cache accumulators for tracking combined update matrices
            cache_accum = self._init_cache_accumulators(batch_size, pos_w.dtype, cache_device)
        else:
            cache_accum = None

        # --- 1. Prediction Step ---
        pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred = (
            self._propagate_nominal_state(
                pos_w, vel_w, quat_b_to_w, gyro_bias_b, accel_bias_b, gyro_b_raw, accel_b_raw
            )
        )
        P_error_pred = self.predict(P_error, quat_b_to_w, accel_b_raw, accel_bias_b)

        # Compute F, Q for caching (same computation as in predict())
        if caching_enabled:
            F_cache, Q_cache = self.compute_F_Q_matrices(quat_b_to_w, accel_b_raw, accel_bias_b)

        # --- 2. Update Step ---
        device = pos_w.device
        P_error_final = P_error_pred
        total_delta_x = torch.zeros(batch_size, self.error_state_dim, device=device, dtype=pos_w.dtype)
        innovation_output = torch.zeros(batch_size, self.obs_dim, device=device, dtype=pos_w.dtype)
        mahalanobis_output = torch.zeros(batch_size, device=device, dtype=pos_w.dtype)

        # Determine if ZUPT should be applied.
        # TCN outputs logits for BCEWithLogitsLoss compatibility; apply sigmoid for probability
        zupt_prob: Optional[torch.Tensor] = None
        if self.use_tcn_zupt and tcn_output is not None:
            zupt_prob = torch.sigmoid(tcn_output["zupt_prob"]).squeeze(-1)
            is_zupt = zupt_prob > Config.ESKFTCN.ZUPT_PROB_THRESHOLD
        elif self.use_zupt:
            is_zupt = self.zupt_detector(accel_b_raw, force_raw)
        else:
            is_zupt = torch.zeros(accel_b_raw.shape[0], dtype=torch.bool, device=device)

        # Apply stationary update (ZUPT + ZARU) where applicable.
        if torch.any(is_zupt):
            zupt_mask = is_zupt
            zupt_prob_to_pass = None
            if self.use_tcn_zupt and zupt_prob is not None:
                zupt_prob_to_pass = zupt_prob[zupt_mask].unsqueeze(-1)

            # Compute gyro prediction (bias estimate)
            gyro_pred = gyro_bias_b_pred

            # Enable ZARU only for ESKF-TCN (when using TCN-based ZUPT)
            use_zaru = self.use_tcn_zupt

            delta_x_stationary, P_after_stationary, zupt_update_mats = self._calculate_stationary_update(
                vel_w_pred[zupt_mask],
                P_error_pred[zupt_mask],
                gyro_pred[zupt_mask],
                tcn_zupt_prob=zupt_prob_to_pass,
                use_zaru=use_zaru,
                return_update_matrices=caching_enabled,
            )
            total_delta_x[zupt_mask] += delta_x_stationary
            P_error_final[zupt_mask] = P_after_stationary

            # Accumulate update noise for masked samples
            if caching_enabled and zupt_update_mats is not None:
                W_zupt, K_zupt, R_zupt = zupt_update_mats
                self._accumulate_update_noise(cache_accum, W_zupt, K_zupt, R_zupt, mask=zupt_mask)

        # Apply TCN-based corrections or standard measurement update.
        if tcn_output is not None:
            # Process TCN covariance once (shared by all updates)
            # covariance_R: [batch, 6] -> first 3 for accel/vel, last 3 for gyro
            tcn_cov_raw = tcn_output.get("covariance_R", None)
            if tcn_cov_raw is not None:
                tcn_cov_diag = F.softplus(tcn_cov_raw) + Config.ESKFTCN.R_MIN
                tcn_cov_diag = torch.clamp(tcn_cov_diag, min=Config.ESKFTCN.R_MIN, max=Config.ESKFTCN.R_MAX)
                R_accel = tcn_cov_diag[:, :3]  # For velocity and gravity updates
            else:
                tcn_cov_diag = None
                R_accel = None

            # 1. TCN Velocity Correction
            vel_corr_body = tcn_output["vel_corr"]
            if torch.any(is_zupt):
                vel_corr_body = torch.where(is_zupt.unsqueeze(-1), torch.zeros_like(vel_corr_body), vel_corr_body)

            delta_x_tcn, P_after_tcn, tcn_vel_update_mats = self._apply_tcn_velocity_correction(
                vel_w_pred, P_error_final, vel_corr_body, quat_b_to_w_pred,
                R_vel_diag=R_accel,
                return_update_matrices=caching_enabled,
            )
            total_delta_x += delta_x_tcn
            P_error_final = P_after_tcn

            # Accumulate update noise
            if caching_enabled and tcn_vel_update_mats is not None:
                W_tcn, K_tcn, R_tcn = tcn_vel_update_mats
                self._accumulate_update_noise(cache_accum, W_tcn, K_tcn, R_tcn)

            # 2. Gravity Alignment Update
            # Skip during warmup period to let TCN stabilize before trusting its gravity predictions
            gravity_warmup_done = getattr(self, '_current_epoch', 0) >= Config.ESKFTCN.GRAVITY_WARMUP_EPOCHS
            if Config.ESKFTCN.USE_GRAVITY_ALIGNMENT and gravity_warmup_done and "gravity_b" in tcn_output and R_accel is not None:
                # Select gravity source: accel (ZUPT) or TCN prediction (motion)
                accel_gravity_b = F.normalize(accel_b_raw, p=2, dim=-1, eps=1e-6)
                gravity_measured_b = torch.where(
                    is_zupt.unsqueeze(-1).expand(-1, 3),
                    accel_gravity_b,
                    tcn_output["gravity_b"]
                )

                # Scale R by ZUPT state (lower R = higher trust during ZUPT)
                zupt_scale = Config.ESKFTCN.GRAVITY_R_STATIC / Config.ESKFTCN.GRAVITY_R_DYNAMIC
                R_gravity_diag = torch.where(
                    is_zupt.unsqueeze(-1).expand(-1, 3),
                    R_accel * zupt_scale,
                    R_accel
                )

                delta_x_gravity, P_after_gravity, gravity_update_mats = self._apply_gravity_alignment_update(
                    P_error_final, quat_b_to_w_pred, gravity_measured_b,
                    R_gravity_diag=R_gravity_diag, accel_b_raw=accel_b_raw,
                    return_update_matrices=caching_enabled,
                )
                total_delta_x += delta_x_gravity
                P_error_final = P_after_gravity

                # Accumulate update noise
                if caching_enabled and gravity_update_mats is not None:
                    W_grav, K_grav, R_grav = gravity_update_mats
                    self._accumulate_update_noise(cache_accum, W_grav, K_grav, R_grav)

            # 3. Standard Measurement Update
            R_tcn_override = torch.diag_embed(tcn_cov_diag) if tcn_cov_diag is not None else None

            # Pass gating threshold from Config
            gating_thresh = Config.ESKFTCN.MAHALANOBIS_GATE_THRESHOLD

            delta_x_up, P_after_up, innovation, mahalanobis_sq, meas_update_mats = self.update(
                P_error_final,
                quat_b_to_w_pred,
                accel_bias_b_pred,
                gyro_bias_b_pred,
                measurement,
                R_override=R_tcn_override,
                gating_threshold=gating_thresh,
                return_update_matrices=caching_enabled,
            )
            total_delta_x += delta_x_up
            P_error_final = P_after_up
            innovation_output = innovation
            mahalanobis_output = mahalanobis_sq

            # Accumulate update noise
            if caching_enabled and meas_update_mats is not None:
                W_meas, K_meas, R_meas = meas_update_mats
                self._accumulate_update_noise(cache_accum, W_meas, K_meas, R_meas)
        else:
            # Pure ESKF mode (without TCN)
            # Recompute necessary variables for innovation
            rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w_pred).transpose(-2, -1)
            gravity_body = (rot_mat_world_to_body @ self.gravity_w.unsqueeze(0).T).squeeze(-1)

            accel_pred = gravity_body + accel_bias_b_pred
            gyro_pred = gyro_bias_b_pred
            h_predicted = torch.cat([accel_pred, gyro_pred], dim=-1)

            innovation_output = measurement - h_predicted

            # Virtual measurement update for Pure ESKF mode
            # Apply weak measurement updates during motion to reduce drift
            if self.use_virtual_measurements and not torch.all(is_zupt):
                # Compute motion level from accelerometer innovation
                accel_innovation = innovation_output[:, 0:3]
                motion_level = torch.norm(accel_innovation, dim=-1, keepdim=True)

                # Adaptive R: High-motion samples need stronger corrections (inverse relationship)
                # Analysis showed high-motion samples suffer more from gyro bias drift.
                # Range: [MIN_SCALE, MAX_SCALE]
                motion_normalized = torch.clamp(motion_level / Config.GRAVITY_MAGNITUDE, 0.0, 1.0)

                max_scale = Config.MOTION.VIRTUAL_MEAS_MAX_SCALE
                min_scale = Config.MOTION.VIRTUAL_MEAS_MIN_SCALE
                motion_scale = max_scale - (max_scale - min_scale) * motion_normalized

                # Create adaptive R matrix (only for non-ZUPT samples)
                # R_diag: [6], motion_scale: [batch, 1] -> expand to [batch, 6]
                R_virtual_diag = F.softplus(self.R_diag).unsqueeze(0) * motion_scale + 1e-2  # [batch, 6]
                R_virtual = torch.diag_embed(R_virtual_diag)  # [batch, 6, 6]

                # Apply update only to non-ZUPT samples
                non_zupt_mask = ~is_zupt
                if torch.any(non_zupt_mask):
                    # Create a temporary delta_x tensor for the full batch
                    delta_x_virtual_full = torch.zeros_like(total_delta_x)
                    P_virtual_full = P_error_final.clone()

                    # Apply update to masked samples
                    delta_x_virtual, P_after_virtual, _, _, virtual_update_mats = self.update(
                        P_error_final[non_zupt_mask],
                        quat_b_to_w_pred[non_zupt_mask],
                        accel_bias_b_pred[non_zupt_mask],
                        gyro_bias_b_pred[non_zupt_mask],
                        measurement[non_zupt_mask],
                        R_override=R_virtual[non_zupt_mask],
                        gating_threshold=None,  # No gating for virtual measurements
                        return_update_matrices=caching_enabled,
                    )

                    # Accumulate update noise for non-ZUPT samples in pure ESKF mode
                    if caching_enabled and virtual_update_mats is not None:
                        W_virt, K_virt, R_virt = virtual_update_mats
                        self._accumulate_update_noise(cache_accum, W_virt, K_virt, R_virt, mask=non_zupt_mask)

                    # Assign back to full batch tensor with learned correction weights
                    # Weights are learnable, initialized from Allan variance ratios
                    # Position/velocity weights fixed at 0 (no absolute reference)
                    # Orientation/bias weights learned to compensate for blind period drift
                    # Broadcasting: [num_masked, 15] * [15] -> [num_masked, 15]
                    delta_x_virtual_full[non_zupt_mask] = delta_x_virtual * self.virtual_meas_weights
                    P_virtual_full[non_zupt_mask] = P_after_virtual

                    # Add to total correction
                    total_delta_x += delta_x_virtual_full
                    P_error_final = P_virtual_full

        # --- 3. Error Injection ---
        if torch.any(total_delta_x != 0):
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = self.inject_correction(
                pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred, total_delta_x
            )
        else:
            pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new = (pos_w_pred, vel_w_pred, quat_b_to_w_pred, gyro_bias_b_pred, accel_bias_b_pred)


        # # --- 3.5. Gradual Velocity Decay ---
        # # Instead of hard reset at threshold, apply smooth decay proportional to zupt_prob.
        # # This prevents discontinuities while still enforcing zero velocity at high confidence.
        # #
        # # decay_weight = clamp((zupt_prob - onset) / (1 - onset), 0, 1) ^ exponent
        # # vel_new = vel * (1 - decay_weight)
        # #
        # # Examples (with onset=0.5, exponent=2):
        # #   zupt_prob = 0.5  → decay = 0.0  → vel unchanged
        # #   zupt_prob = 0.75 → decay = 0.25 → vel reduced by 25%
        # #   zupt_prob = 1.0  → decay = 1.0  → vel = 0
        # if self.use_tcn_zupt and zupt_prob is not None:
        #     onset = Config.ESKFTCN.ZUPT_DECAY_ONSET
        #     exponent = Config.ESKFTCN.ZUPT_DECAY_EXPONENT
        #
        #     # Compute normalized probability above onset threshold
        #     prob_above_onset = (zupt_prob - onset) / (1.0 - onset + 1e-6)
        #     prob_above_onset = torch.clamp(prob_above_onset, min=0.0, max=1.0)
        #
        #     # Apply exponent for smooth onset (quadratic by default)
        #     decay_weight = prob_above_onset ** exponent
        #
        #     # Apply gradual decay: vel_new = vel * (1 - decay_weight)
        #     vel_w_new = vel_w_new * (1.0 - decay_weight.unsqueeze(-1))

        # --- 4. Assemble Features for next TCN step ---
        rot_mat_world_to_body = quaternion_to_rotation_matrix(quat_b_to_w_new).transpose(-2, -1)
        vel_body = (rot_mat_world_to_body @ vel_w_new.unsqueeze(-1)).squeeze(-1)
        tcn_features: Dict[str, torch.Tensor] = {
            "body_velocity": vel_body,
            "zupt_flag": is_zupt.float().unsqueeze(-1),
            "innovation": innovation_output,
            "mahalanobis": mahalanobis_output.unsqueeze(-1),
        }

        # --- 5. Cache matrices for parallel covariance computation ---
        if caching_enabled:
            self._cache_step_matrices(
                F_cache,
                Q_cache,
                cache_accum['W_combined'],
                cache_accum['noise_accumulated'],
            )

        return (pos_w_new, vel_w_new, quat_b_to_w_new, gyro_bias_b_new, accel_bias_b_new, P_error_final, tcn_features)
