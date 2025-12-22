# model/config.py

class Config:
    """
    Centralized configuration class for trajectory estimation models.
    This class holds various parameters for Kalman filters, TCNs, dataset
    handling, and training, allowing for easy management and modification
    of global settings.
    """

    # --- Global Training Parameters ---
    DT = 0.02 # Time delta (s) for model integration
    INITIAL_PEN_TIP_OFFSET = [0.0, 0.125, 0.0] # [x, y, z] offset from IMU to pen tip (m)

    # --- Dataset Parameters ---
    DATASET_H5_PATH = "./data/dataset.h5"
    VALIDATION_DATASET_H5_PATH = "./data/validation_dataset.h5"
    AUGMENT_MULTIPLIER = 1
    SUBSAMPLE_STEP = 1
    DO_AUGMENT = False
    SCALER_STATS_H5_PATH = "./data/scaler_stats.h5"

    # --- ZUPT Parameters (used by ESKF and AEKF) ---
    ZUPT_WINDOW_SIZE = 20
    ZUPT_ACCEL_THRESHOLD = 0.1
    ZUPT_FORCE_VAR_THRESHOLD = 0.01
    ZUPT_FORCE_DELTA_THRESHOLD = 0.1

    # --- Allan Variance Noise Parameters (used by ESKF and AEKF) ---
    # These values are derived from sensor characterization (e.g., Allan Variance plots).
    # Gyroscope
    ARW_X, ARW_Y, ARW_Z = 4.6726e-03, 5.0027e-03, 4.7839e-03 # Angle Random Walk (ARW)
    GYRO_BI_X, GYRO_BI_Y, GYRO_BI_Z = 1.0330e-03, 1.5368e-03, 9.9458e-04 # Bias Instability (BI)
    # Accelerometer
    VRW_X, VRW_Y, VRW_Z = 1.1339e-03, 8.3872e-04, 1.0075e-03 # Velocity Random Walk (VRW)
    ACCEL_BI_X, ACCEL_BI_Y, ACCEL_BI_Z = 5.4210e-04, 2.8454e-04, 3.4573e-04 # Bias Instability (BI)

    # --- Physical Constants ---
    GRAVITY_MAGNITUDE = 9.81 # Magnitude of gravity (m/s^2)

    # --- Model Specific Parameters ---
    class ESKFTCN:
        TCN_INPUT_SIZE = 20
        TCN_CHANNELS = [64, 64, 64, 64]
        KERNEL_SIZE = 5
        DROPOUT = 0.1
        TCN_DILATION_FACTORS = [1, 2, 4, 8] # Added TCN Dilation Factors
        USE_ZUPT = False
        USE_TCN_ZUPT = True
        ADAPTIVE_GAIN_ESKF = 0.5 # Specific to ESKF's R adaptivity
        # Initial standard deviation for ZUPT measurement noise in ESKF.
        ZUPT_NOISE_STD_ESKF = [0.01, 0.01, 0.01]
        # Whether to use Depthwise Separable Convolutions in TCN for ESKFTCN.
        USE_SEPARABLE_CONV = False

    class AEKFTCN:
        TCN_INPUT_SIZE = 20
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