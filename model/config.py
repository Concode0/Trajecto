# model/config.py

class Config:
    """
    Centralized configuration class for trajectory estimation models.
    This class holds various parameters for Kalman filters, TCNs, dataset
    handling, and training, allowing for easy management and modification
    of global settings.
    """

    # --- Global Training Parameters ---
    TARGET_SAMPLING_RATE_HZ = 50.107  # Target sampling rate (Hz) for data acquisition and model
    DT = 1.0 / TARGET_SAMPLING_RATE_HZ  # Time delta (s) for model integration (50.107 Hz = 0.019957291396 s)
    INITIAL_PEN_TIP_OFFSET = [0.0, -0.06, 0.0] # [x, y, z] offset from IMU to pen tip (m)

    # --- Dataset Parameters ---
    DATASET_H5_PATH = "./data/dataset.h5"
    VALIDATION_DATASET_H5_PATH = "./data/validation_dataset.h5"
    AUGMENT_MULTIPLIER = 2
    SUBSAMPLE_STEP = 1
    DO_AUGMENT = False
    YAW_ANGLE = (-0.78, 0.78)   # Set small angle in first and increase when fine tunning.
    SIGMA_TILT = 0.00           # Same ( Don't increase too large (0.02 ~ 0.52))
    SCALER_STATS_H5_PATH = "./data/scaler_stats.h5"

    # --- ZUPT Parameters (used by ESKF and AEKF) ---
    ZUPT_WINDOW_SIZE = 20
    ZUPT_ACCEL_THRESHOLD = 0.1430  # Optimized from GT analysis: 92% ZUPT coverage, 51% moving rejection (was: 0.5)
    ZUPT_FORCE_VAR_THRESHOLD = 36660  # Optimized from GT analysis: 90% ZUPT coverage, 18% moving rejection (was: 100000)
    ZUPT_FORCE_DELTA_THRESHOLD = 154  # Optimized from GT analysis: 87% ZUPT coverage, 20% moving rejection (was: 2000)

    # --- Allan Variance Noise Parameters (used by ESKF and AEKF) ---

    # Base values from Allan variance analysis (empirical measurements)
    ARW_X, ARW_Y, ARW_Z = 7.1664e-05, 7.9283e-05, 7.5335e-05
    GYRO_BI_X, GYRO_BI_Y, GYRO_BI_Z = 1.6441e-05, 2.8196e-05, 1.2203e-05
    VRW_X, VRW_Y, VRW_Z = 8.3297e-04, 6.7196e-04, 9.3271e-04
    ACCEL_BI_X, ACCEL_BI_Y, ACCEL_BI_Z = 4.3723e-04, 1.7697e-04, 2.8099e-04

    # --- Physical Constants ---
    GRAVITY_MAGNITUDE = 9.80665  # Standard gravity (m/s²) - CODATA 2018

    # --- Model Specific Parameters ---
    class ESKFTCN:
        TCN_INPUT_SIZE = 19  # Reduced from 20: removed zupt_flag to avoid circular dependency with TCN ZUPT
        TCN_CHANNELS = [64, 64, 64, 64]
        KERNEL_SIZE = 5
        DROPOUT = 0.1
        TCN_DILATION_FACTORS = [1, 4, 8, 16] # Added TCN Dilation Factors
        USE_ZUPT = False  # Classic threshold-based ZUPT disabled (TCN handles ZUPT detection)
        USE_TCN_ZUPT = True  # TCN predicts zupt_prob directly from physics features
        ADAPTIVE_GAIN_ESKF = 0.5 # Specific to ESKF's R adaptivity
        # Initial standard deviation for ZUPT measurement noise in ESKF.
        ZUPT_NOISE_STD_ESKF = [0.01, 0.01, 0.01]
        # Whether to use Depthwise Separable Convolutions in TCN for ESKFTCN.
        USE_SEPARABLE_CONV = False
        # Mahalanobis distance threshold for measurement gating (Chi-square dist, dof=6, p=0.99 => ~16.8)
        MAHALANOBIS_GATE_THRESHOLD = 16.8

    class AEKFTCN:
        TCN_INPUT_SIZE = 19  # Reduced from 20: removed zupt_flag to avoid circular dependency
        TCN_OUTPUT_SIZE = 3 # Only predict 3D velocity residual
        TCN_NUM_CHANNELS = [64, 64, 64, 64]
        TCN_KERNEL_SIZE = 3
        TCN_DROPOUT = 0.2
        TCN_DILATION_FACTORS = [1, 2, 4, 8] # Added TCN Dilation Factors
        ADAPTIVE_R_FACTOR_AEKF = 0.1 # Specific to AEKF's R adaptivity
        ZUPT_R_FACTOR_AEKF = 1e-6 # Specific to AEKF's ZUPT R
        # Initial standard deviation for ZUPT measurement noise in AEKF.
        ZUPT_NOISE_STD_AEKF = [0.01, 0.01, 0.01]
        # Whether to use Depthwise Separable Convolutions in TCN for AEKFTCN.
        USE_SEPARABLE_CONV = False


    class OnlyTCN:
        INPUT_SIZE = 7
        OUTPUT_SIZE = 3
        TCN_CHANNELS = [64, 64, 64, 64] # Default for OnlyTCN
        KERNEL_SIZE = 3
        DROPOUT = 0.1

    # --- Loss Parameters ---
    class LOSS:
        REG_WEIGHT_ESKF_TCN = 1e-4
        REG_WEIGHT_AEKF_TCN = 1e-5
        # No specific reg weight for OnlyTCN in current loss setup