"""
Configuration constants for SimGenerator.
Mirrors model/config.py for compatibility.
"""
module Config

using StaticArrays

# --- Timing ---
const TARGET_SAMPLING_RATE_HZ = 50.107
const DT = 1.0 / TARGET_SAMPLING_RATE_HZ
const MAX_SEQUENCE_LENGTH = 1750  # 35 seconds @ 50Hz

# --- Physical Constants ---
const GRAVITY_MAGNITUDE = 9.80665
const GRAVITY_W = SVector{3,Float64}(0.0, 0.0, -GRAVITY_MAGNITUDE)

# Pen tip offset: vector from IMU to pen tip in body frame
# Mean offset and variance for grip position variation
const PEN_TIP_OFFSET_MEAN = SVector{3,Float64}(0.0, -0.06, 0.0)  # m, nominal IMU to pen tip
const PEN_TIP_OFFSET_STD = SVector{3,Float64}(0.005, 0.015, 0.003)  # m, grip position variance
# - Y varies most (5-7.5cm grip range along pen shaft)
# - X varies slightly (lateral grip position)
# - Z varies minimally (finger wrap around pen)

# Legacy constant for backwards compatibility
const PEN_TIP_OFFSET = PEN_TIP_OFFSET_MEAN

# --- Allan Variance (from BMI270 characterization) ---
const VRW = SVector{3,Float64}(8.33e-4, 6.72e-4, 9.33e-4)  # m/s^2/sqrt(Hz)
const ARW = SVector{3,Float64}(7.17e-5, 7.93e-5, 7.53e-5)  # rad/s/sqrt(Hz)
const ACCEL_BI = SVector{3,Float64}(4.37e-4, 1.77e-4, 2.81e-4)  # m/s^2
const GYRO_BI = SVector{3,Float64}(1.64e-5, 2.82e-5, 1.22e-5)   # rad/s

# --- Sigma-LogNormal Parameters ---
const SIGMALOG_MU_RANGE = (0.0, 0.5)       # Log-mean range
const SIGMALOG_SIGMA_RANGE = (0.3, 0.6)    # Log-std range
const SIGMALOG_D_RANGE = (0.005, 0.05)     # Stroke amplitude (m)
const STROKE_DURATION_RANGE = (0.2, 0.8)   # Seconds per stroke

# --- Posture Parameters ---
const BASE_PITCH_RANGE = (0.4, 1.0)   # rad (~23-57 deg)
const BASE_ROLL_RANGE = (-0.3, 0.3)   # rad (~-17 to +17 deg)
const WRIST_PIVOT_GAIN = 0.05         # rad/(m/s) - velocity-to-wobble coupling

# --- Static Buffer ---
const STATIC_BUFFER_S = 2.5  # seconds of static data before writing

# --- Output ---
const DEFAULT_OUTPUT_PATH = "data/simulated_dataset.h5"
const DEFAULT_NUM_STROKES = 10
const DEFAULT_WRITING_DURATION = 5.0  # seconds

# =============================================================================
# PEN LIFT PARAMETERS (KinematicLayer)
# =============================================================================

"""
    PenLiftParams

Parameters for pen lift/land motion between strokes.

# Fields
- `lift_duration_range`: Time range for pen to lift from surface to max height (s)
- `land_duration_range`: Time range for pen to descend from max height to surface (s)
- `hover_duration_range`: Time range for pen to hover at max height (s)
- `lift_height_range`: Range of max lift height above writing surface (m)
- `vel_threshold`: Velocity threshold for detecting stroke boundaries (m/s)
"""
struct PenLiftParams
    lift_duration_range::Tuple{Float64,Float64}
    land_duration_range::Tuple{Float64,Float64}
    hover_duration_range::Tuple{Float64,Float64}
    lift_height_range::Tuple{Float64,Float64}
    vel_threshold::Float64
end

const DEFAULT_PEN_LIFT_PARAMS = PenLiftParams(
    (0.25, 0.45),   # lift_duration_range (s) - slower, more natural lift
    (0.20, 0.35),   # land_duration_range (s) - slower, controlled landing
    (0.15, 0.40),   # hover_duration_range (s) - brief pause at top
    (0.008, 0.012), # lift_height_range (m) - slightly lower lifts
    0.008           # vel_threshold (m/s) - higher threshold for detecting stroke gaps
)

# =============================================================================
# ARM KINEMATICS PARAMETERS (PostureLayer)
# =============================================================================

"""
    ArmKinematicParams

Arm segment lengths and shoulder position for IK.

# Fields
- `L_upper`: Upper arm length (shoulder to elbow) in meters
- `L_forearm`: Forearm length (elbow to wrist) in meters
- `L_hand`: Hand length (wrist to pen tip) in meters
- `shoulder_pos`: Shoulder position in world frame (m)
"""
struct ArmKinematicParams
    L_upper::Float64
    L_forearm::Float64
    L_hand::Float64
    shoulder_pos::SVector{3,Float64}
end

const DEFAULT_ARM_PARAMS = ArmKinematicParams(
    0.30,   # L_upper (m)
    0.26,   # L_forearm (m)
    0.10,   # L_hand (m)
    SVector{3,Float64}(0.0, -0.20, 0.40)  # shoulder_pos (m)
)

"""
    JointLimits

Biomechanical joint angle limits.

# Fields
- `elbow_range`: Elbow flexion/extension range (deg)
- `pronation_range`: Wrist pronation/supination range (deg)
- `flexion_range`: Wrist flexion/extension range (deg)
- `deviation_range`: Wrist radial/ulnar deviation range (deg)
"""
struct JointLimits
    elbow_range::Tuple{Float64,Float64}
    pronation_range::Tuple{Float64,Float64}
    flexion_range::Tuple{Float64,Float64}
    deviation_range::Tuple{Float64,Float64}
end

const DEFAULT_JOINT_LIMITS = JointLimits(
    (30.0, 145.0),   # elbow_range (deg)
    (-85.0, 90.0),   # pronation_range (deg)
    (-80.0, 70.0),   # flexion_range (deg)
    (-40.0, 20.0)    # deviation_range (deg)
)

"""
    WristIKParams

Parameters for inverse kinematics wrist angle computation.

# Fields
- `base_pronation`: Base pronation angle (deg)
- `base_flexion`: Base flexion angle (deg)
- `base_deviation`: Base deviation angle (deg)
- `k_pronate`: Pronation sensitivity to X velocity
- `k_flex`: Flexion sensitivity to Y position
- `k_dev`: Deviation sensitivity to Y velocity
- `v_max`: Max velocity for normalization (m/s)
- `workspace_width`: Typical workspace width (m)
- `smoothing_tau`: Low-pass filter time constant (s)
"""
struct WristIKParams
    base_pronation::Float64
    base_flexion::Float64
    base_deviation::Float64
    k_pronate::Float64
    k_flex::Float64
    k_dev::Float64
    v_max::Float64
    workspace_width::Float64
    smoothing_tau::Float64
end

const DEFAULT_WRIST_IK_PARAMS = WristIKParams(
    -30.0,   # base_pronation (deg)
    -15.0,   # base_flexion (deg)
    -10.0,   # base_deviation (deg)
    25.0,    # k_pronate (deg per normalized velocity) - increased for dynamic motion
    30.0,    # k_flex (deg per normalized position) - increased for workspace coupling
    18.0,    # k_dev (deg per normalized velocity) - increased for lateral response
    0.20,    # v_max (m/s) - lowered for higher sensitivity at writing speeds
    0.12,    # workspace_width (m)
    0.04     # smoothing_tau (s) - faster response
)

# =============================================================================
# NOISE PARAMETERS (SensingLayer)
# =============================================================================

"""
    TremorParams

Physiological tremor parameters (8-12 Hz band).

# Fields
- `freq_mean`: Mean tremor frequency (Hz)
- `freq_std`: Standard deviation of tremor frequency (Hz)
- `amp_accel`: Tremor amplitude for accelerometer (m/s²) per axis
- `amp_gyro`: Tremor amplitude for gyroscope (rad/s) per axis
- `bandwidth`: Tremor bandwidth (Hz)
- `correlation`: Cross-axis correlation coefficient
- `fatigue_mod_freq`: Fatigue modulation frequency (Hz)
- `fatigue_mod_amp`: Fatigue modulation amplitude (relative)
"""
struct TremorParams
    freq_mean::Float64
    freq_std::Float64
    amp_accel::SVector{3,Float64}
    amp_gyro::SVector{3,Float64}
    bandwidth::Float64
    correlation::Float64
    fatigue_mod_freq::Float64
    fatigue_mod_amp::Float64
end

const DEFAULT_TREMOR_PARAMS = TremorParams(
    10.0,   # freq_mean (Hz)
    1.0,    # freq_std (Hz)
    SVector{3,Float64}(0.08, 0.08, 0.05),  # amp_accel (m/s²)
    SVector{3,Float64}(0.02, 0.02, 0.015), # amp_gyro (rad/s)
    2.0,    # bandwidth (Hz)
    0.7,    # correlation
    0.1,    # fatigue_mod_freq (Hz)
    0.3     # fatigue_mod_amp
)

"""
    PinkNoiseParams

1/f (pink) noise parameters.

# Fields
- `alpha`: Spectral exponent (1.0 for pure 1/f)
- `low_freq_cutoff`: Low frequency cutoff to avoid DC divergence (Hz)
- `amplitude_scale`: Scale relative to white noise amplitude
"""
struct PinkNoiseParams
    alpha::Float64
    low_freq_cutoff::Float64
    amplitude_scale::Float64
end

const DEFAULT_PINK_NOISE_PARAMS = PinkNoiseParams(
    1.0,    # alpha
    0.01,   # low_freq_cutoff (Hz)
    0.5     # amplitude_scale
)

"""
    FSRBaselineParams

FSR baseline drift parameters using Ornstein-Uhlenbeck process.

# Fields
- `initial_range`: Initial baseline value range [0, 1]
- `mean_baseline`: Long-term mean baseline value
- `drift_tau`: Mean-reversion time constant (s)
- `drift_sigma`: Diffusion coefficient for random walk
- `oscillation_amp`: Amplitude of slow grip pressure oscillation
- `oscillation_freq_range`: Frequency range of grip oscillation (Hz)
"""
struct FSRBaselineParams
    initial_range::Tuple{Float64,Float64}
    mean_baseline::Float64
    drift_tau::Float64
    drift_sigma::Float64
    oscillation_amp::Float64
    oscillation_freq_range::Tuple{Float64,Float64}
end

const DEFAULT_FSR_BASELINE_PARAMS = FSRBaselineParams(
    (0.1, 0.3),   # initial_range
    0.2,          # mean_baseline
    20.0,         # drift_tau (s)
    0.015,        # drift_sigma
    0.03,         # oscillation_amp
    (0.05, 0.2)   # oscillation_freq_range (Hz)
)

# =============================================================================
# GRIP STYLE PARAMETERS (PostureLayer)
# =============================================================================

"""
    GripStyle

Enumeration of common pen grip styles.

- `TRIPOD`: Standard tripod grip (thumb, index, middle finger)
- `LATERAL_TRIPOD`: Thumb crosses over index finger
- `QUADRUPOD`: Four fingers grip the pen
- `DYNAMIC_TRIPOD`: Fingers move during writing (more flexible)
- `OVERHAND`: Pen held from above (like a brush)
"""
@enum GripStyle begin
    TRIPOD = 1
    LATERAL_TRIPOD = 2
    QUADRUPOD = 3
    DYNAMIC_TRIPOD = 4
    OVERHAND = 5
end

"""
    GripStyleParams

Parameters for a specific grip style.

# Fields
- `style`: GripStyle enum value
- `base_pitch_range`: Pen tilt angle range (rad)
- `base_roll_range`: Pen roll angle range (rad)
- `finger_spread`: How spread fingers are on pen (affects stability)
- `wrist_mobility`: Wrist flexibility multiplier (1.0 = normal)
- `pressure_variation`: Force sensor variation (higher = more variable pressure)
"""
struct GripStyleParams
    style::GripStyle
    base_pitch_range::Tuple{Float64,Float64}
    base_roll_range::Tuple{Float64,Float64}
    finger_spread::Float64
    wrist_mobility::Float64
    pressure_variation::Float64
end

# Grip style presets
const GRIP_TRIPOD = GripStyleParams(
    TRIPOD,
    (0.5, 0.9),     # pitch: 29-52 deg (moderate tilt)
    (-0.2, 0.2),    # roll: ±11 deg
    0.5,            # medium finger spread
    1.0,            # normal wrist mobility
    0.15            # low pressure variation
)

const GRIP_LATERAL_TRIPOD = GripStyleParams(
    LATERAL_TRIPOD,
    (0.4, 0.8),     # pitch: 23-46 deg (slightly flatter)
    (-0.4, 0.1),    # roll: more negative (thumb over)
    0.4,            # tighter finger spread
    0.8,            # reduced wrist mobility
    0.20            # medium pressure variation
)

const GRIP_QUADRUPOD = GripStyleParams(
    QUADRUPOD,
    (0.5, 1.0),     # pitch: 29-57 deg
    (-0.25, 0.25),  # roll: ±14 deg
    0.6,            # wider finger spread
    0.9,            # slightly reduced mobility
    0.12            # stable pressure
)

const GRIP_DYNAMIC_TRIPOD = GripStyleParams(
    DYNAMIC_TRIPOD,
    (0.35, 0.85),   # pitch: 20-49 deg (more variable)
    (-0.35, 0.35),  # roll: ±20 deg (more variable)
    0.55,           # medium finger spread
    1.2,            # higher wrist mobility
    0.25            # high pressure variation
)

const GRIP_OVERHAND = GripStyleParams(
    OVERHAND,
    (0.8, 1.3),     # pitch: 46-74 deg (steeper angle)
    (-0.15, 0.15),  # roll: ±9 deg (more stable)
    0.7,            # wider grip
    0.7,            # reduced wrist mobility
    0.10            # stable pressure
)

const ALL_GRIP_STYLES = [GRIP_TRIPOD, GRIP_LATERAL_TRIPOD, GRIP_QUADRUPOD,
                         GRIP_DYNAMIC_TRIPOD, GRIP_OVERHAND]

const DEFAULT_GRIP_STYLE = GRIP_TRIPOD

# =============================================================================
# HANDEDNESS PARAMETERS
# =============================================================================

"""
    Handedness

Enumeration for hand dominance.
"""
@enum Handedness begin
    RIGHT_HAND = 1
    LEFT_HAND = 2
end

"""
    HandednessParams

Parameters for left/right hand writing simulation.

# Fields
- `hand`: Handedness enum value
- `mirror_x`: Whether to mirror X coordinates
- `pronation_offset`: Additional pronation for this hand (deg)
- `writing_direction`: Primary writing direction multiplier (+1 right-to-left, -1 left-to-right)
- `shoulder_offset`: Lateral shoulder position offset (m)
"""
struct HandednessParams
    hand::Handedness
    mirror_x::Bool
    pronation_offset::Float64
    writing_direction::Float64
    shoulder_offset::Float64
end

const RIGHT_HAND_PARAMS = HandednessParams(
    RIGHT_HAND,
    false,          # no mirror
    0.0,            # no pronation offset
    1.0,            # normal writing direction
    0.0             # no shoulder offset
)

const LEFT_HAND_PARAMS = HandednessParams(
    LEFT_HAND,
    true,           # mirror X coordinates
    30.0,           # additional pronation (supinated position)
    -1.0,           # reversed writing direction
    0.40            # shoulder offset to left side (m)
)

const DEFAULT_HANDEDNESS = RIGHT_HAND_PARAMS

# =============================================================================
# WORLD ROTATION AUGMENTATION
# =============================================================================

"""
    WorldRotationParams

Parameters for world rotation augmentation (simulates tilted writing surfaces).

# Fields
- `enabled::Bool`: Whether world rotation is applied
- `max_tilt::Float64`: Maximum tilt from vertical in radians (e.g., 0.52 ≈ 30°)
- `full_yaw::Bool`: If true, allow full 360° yaw rotation
"""
struct WorldRotationParams
    enabled::Bool
    max_tilt::Float64
    full_yaw::Bool
end

const DEFAULT_WORLD_ROTATION_PARAMS = WorldRotationParams(
    true,   # enabled
    0.52,   # max_tilt (30 degrees)
    true    # full_yaw
)

# Disabled version for testing
const DISABLED_WORLD_ROTATION_PARAMS = WorldRotationParams(
    false,
    0.0,
    false
)

# =============================================================================
# EXPORTS
# =============================================================================

export WorldRotationParams, DEFAULT_WORLD_ROTATION_PARAMS, DISABLED_WORLD_ROTATION_PARAMS
export TARGET_SAMPLING_RATE_HZ, DT, MAX_SEQUENCE_LENGTH,
       GRAVITY_MAGNITUDE, GRAVITY_W, PEN_TIP_OFFSET,
       PEN_TIP_OFFSET_MEAN, PEN_TIP_OFFSET_STD,
       VRW, ARW, ACCEL_BI, GYRO_BI,
       SIGMALOG_MU_RANGE, SIGMALOG_SIGMA_RANGE, SIGMALOG_D_RANGE,
       STROKE_DURATION_RANGE, BASE_PITCH_RANGE, BASE_ROLL_RANGE,
       WRIST_PIVOT_GAIN, STATIC_BUFFER_S,
       DEFAULT_OUTPUT_PATH, DEFAULT_NUM_STROKES, DEFAULT_WRITING_DURATION

# Pen Lift exports
export PenLiftParams, DEFAULT_PEN_LIFT_PARAMS

# Arm Kinematics exports
export ArmKinematicParams, JointLimits, WristIKParams
export DEFAULT_ARM_PARAMS, DEFAULT_JOINT_LIMITS, DEFAULT_WRIST_IK_PARAMS

# Noise exports
export TremorParams, PinkNoiseParams, FSRBaselineParams
export DEFAULT_TREMOR_PARAMS, DEFAULT_PINK_NOISE_PARAMS, DEFAULT_FSR_BASELINE_PARAMS

# Grip style exports
export GripStyle, TRIPOD, LATERAL_TRIPOD, QUADRUPOD, DYNAMIC_TRIPOD, OVERHAND
export GripStyleParams, GRIP_TRIPOD, GRIP_LATERAL_TRIPOD, GRIP_QUADRUPOD
export GRIP_DYNAMIC_TRIPOD, GRIP_OVERHAND, ALL_GRIP_STYLES, DEFAULT_GRIP_STYLE

# Handedness exports
export Handedness, RIGHT_HAND, LEFT_HAND
export HandednessParams, RIGHT_HAND_PARAMS, LEFT_HAND_PARAMS, DEFAULT_HANDEDNESS

end # module
