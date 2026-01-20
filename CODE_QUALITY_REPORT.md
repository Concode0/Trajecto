# Code Quality Analysis Report - Trajecto

**Date:** 2026-01-19
**Analysis Scope:** Complete codebase review
**Total Issues Found:** 50+

---

## 🔴 Critical Issues (Fix Immediately)

### 1\. Missing Input Validation in Training Loop

**File:** `train_eskf.py:108-208`, `train_two_stage.py:113-206`
**Impact:** Cryptic PyTorch errors, wasted GPU time, hard debugging

**Problem:**

```python
def train_step(model, batch, config, mean, std, task_weights=None):
    sensor_raw = batch["imu_seq_raw"].to(device)  # No shape validation!
    # If sensor_raw is [B, T, 9] instead of [B, T, 7] → cryptic error 50 lines later
```

**Fix:**

```python
def train_step(model, batch, config, mean, std, task_weights=None):
    sensor_raw = batch["imu_seq_raw"].to(device)

    # Add validation
    assert sensor_raw.ndim == 3, f"Expected 3D tensor, got shape {sensor_raw.shape}"
    assert sensor_raw.shape[2] == 7, f"Expected 7 channels, got {sensor_raw.shape[2]}"
    assert gt_vel.shape == sensor_raw.shape[:-1] + (3,), "Velocity shape mismatch"

    # ... rest of function
```

---

### 2\. Silent NaN Loss Handling

**File:** `train_eskf.py:318`, `train_two_stage.py:303, 440`
**Impact:** Training issues invisible, models converge poorly without warning

**Problem:**

```python
if torch.isnan(losses["total"]):
    continue  # Silent skip - WHY did this happen?
```

**Fix:**

```python
import logging
logger = logging.getLogger(__name__)

if torch.isnan(losses["total"]):
    logger.warning(f"NaN loss at epoch {epoch}, batch {i_batch}")
    logger.debug(f"Loss breakdown: {losses}")
    logger.debug(f"Sensor stats: min={sensor_raw.min()}, max={sensor_raw.max()}")
    continue
```

---

### 3\. Missing File Existence Checks

**File:** Multiple (`validate.py:568`, `train_eskf.py:246`, etc.)
**Impact:** Confusing h5py/torch errors instead of clear "file not found"

**Problem:**

```python
with h5py.File("./data/scaler_stats.h5", "r") as f:  # What if file doesn't exist?
    mean = torch.tensor(f["mean"][:])
```

**Fix:**

```python
scaler_path = Path("./data/scaler_stats.h5")
if not scaler_path.exists():
    raise FileNotFoundError(
        f"Scaler stats not found at {scaler_path}.\n"
        f"Run data preprocessing first: python utils/acquire.py --reprocess"
    )

with h5py.File(scaler_path, "r") as f:
    mean = torch.tensor(f["mean"][:])
```

---

### 4\. Unsafe Cholesky Decomposition Fallback

**File:** `model/ESKF.py:202-224`
**Impact:** Numerical instability masked, training continues silently

**Problem:**

```python
try:
    L = torch.linalg.cholesky(S)
except RuntimeError:
    # Fallback to slow pinv - masks real numerical issues
    X = torch.linalg.pinv(S) @ B
    print("Cholesky failed, using pinv")  # Only prints to console
```

**Fix:**

```python
try:
    L = torch.linalg.cholesky(S)
    X = torch.cholesky_solve(B, L)
except RuntimeError as e:
    # Log for post-analysis
    logging.warning(f"Cholesky failed at step {self.step_count}: {e}")

    # Add small regularization instead of pinv
    S_reg = S + torch.eye(S.shape[-1], device=S.device) * 1e-6
    try:
        L = torch.linalg.cholesky(S_reg)
        X = torch.cholesky_solve(B, L)
    except RuntimeError:
        # Only use pinv as last resort
        logging.error("Cholesky failed even with regularization!")
        X = torch.linalg.pinv(S_reg) @ B
```

---

## 🟠 High Priority Issues (Fix Soon)

### 5\. Massive Code Duplication - Training Loops

**Files:** `train_eskf.py:304-379`, `train_two_stage.py:294-340, 431-463`
**Impact:** 200+ lines duplicated, bug fixes need 3x changes

**Problem:** Nearly identical training loops in multiple files

**Fix:** Create unified training infrastructure

```python
# Create new file: model/training.py
class TrainingLoop:
    def __init__(self, model, optimizer, scheduler, loss_fn, dwa_updater=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.dwa_updater = dwa_updater

    def train_epoch(self, dataloader, config):
        """Single epoch of training"""
        # Unified logic here
        ...

    def validate(self, dataloader, config):
        """Validation pass"""
        ...

# Usage in train_eskf.py
trainer = TrainingLoop(model, optimizer, scheduler, train_step, dwa_updater)
for epoch in range(config.epochs):
    train_losses = trainer.train_epoch(train_loader, config)
    val_losses = trainer.validate(val_loader, config)
```

---

### 6\. Duplicated Dataset Augmentation

**Files:** `model/dataset.py:153-214`, `model/sim_dataset.py:115-159`
**Impact:** 60+ lines duplicated, bug fixes need 2x changes

**Fix:**

```python
# Create new file: model/data_augmentation.py
class AugmentationPipeline:
    def __init__(self, yaw_range, sigma_tilt):
        self.yaw_range = yaw_range
        self.sigma_tilt = sigma_tilt

    def augment_sample(self, sensor, gt_pos, gt_vel):
        """Apply random augmentation to a single sample"""
        # Extract common logic here
        ...
        return sensor_aug, gt_pos_aug, gt_vel_aug

# Usage in dataset.py
class TrajectoryDataset(Dataset):
    def __init__(self, ...):
        self.augmentation = AugmentationPipeline(yaw_range, sigma_tilt)

    def __getitem__(self, idx):
        if self.do_augment:
            return self.augmentation.augment_sample(sensor, gt_pos, gt_vel)
```

---

### 7\. Magic Numbers Throughout Codebase

**Files:** Multiple
**Impact:** Hard to tune, maintenance burden

**Examples:**

*   `model/ESKF_TCN.py:133` → `0.002` (gyro variance threshold)
*   `model/dataset.py:202-212` → `1.5`, `8.33e-3` (noise multipliers)
*   `utils/acquire.py:86` → `0.01` (force threshold)
*   `model/ESKF.py:392` → `20.0` (accel norm clamp)

**Fix:** Add to `model/config.py`:

```python
class MOTION_DETECTION:
    GYRO_VAR_THRESHOLD = 0.002
    ACCEL_NORM_DIFF_MAX_CLAMP = 20.0
    FORCE_THRESHOLD_MOVING = 0.01

class AUGMENTATION:
    NOISE_MULTIPLIER_GYRO = 1.5
    NOISE_MULTIPLIER_ACCEL = 8.33e-3
```

---

### 8\. Overly Complex \_initialize\_state()

**File:** `model/ESKF_TCN.py:94-300`
**Impact:** 200+ lines, untestable, hard to debug

**Problem:** Single function does 6 different things

**Fix:** Break into smaller methods

```python
def _initialize_state(self, sensor_raw, sensor_norm):
    """Initialize filter state from first timesteps."""

    # 1. Detect static period
    static_samples = self._detect_static_period(sensor_raw)

    if static_samples is None:
        return self._fallback_initialization(sensor_raw)

    # 2. Estimate gravity and biases
    gravity, gyro_bias, accel_bias = self._estimate_gravity_and_biases(
        sensor_raw, static_samples
    )

    # 3. Initialize covariance
    P_error = self._initialize_covariance(sensor_raw, static_samples)

    # 4. Set initial state
    self.filter.set_initial_state(gravity, gyro_bias, accel_bias, P_error)

def _detect_static_period(self, sensor_raw):
    """Detect static period for initialization (20-50 samples)."""
    # Focused logic here
    ...

def _estimate_gravity_and_biases(self, sensor_raw, static_samples):
    """Estimate gravity vector and sensor biases from static data."""
    # Focused logic here
    ...
```

---

## 🟡 Medium Priority Issues (Refactoring)

### 9\. Inefficient Dataset Loading

**File:** `model/dataset.py:69-109`
**Impact:** 2x memory usage, slow startup

**Problem:**

```python
for key in tqdm(keys, desc="Caching dataset"):
    sensor = f[key]["sensor_data"][:]  # Full copy to RAM
    self.cached_data.append(data_dict)  # Another copy!
```

**Fix:** Implement lazy loading

```python
class LazyTrajectoryDataset(Dataset):
    def __init__(self, h5_path, use_cache=False):
        self.h5_path = h5_path
        self.use_cache = use_cache
        self._cache = {} if use_cache else None

        # Only load metadata
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())

    def __getitem__(self, idx):
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        # Load on-demand
        with h5py.File(self.h5_path, "r") as f:
            data = self._load_sample(f, self.keys[idx])

        if self.use_cache:
            self._cache[idx] = data

        return data
```

---

### 10\. Repeated Normalization in Training Loop

**File:** `train_eskf.py:130`, `train_two_stage.py:135`
**Impact:** Unnecessary compute every batch

**Problem:**

```python
def train_step(...):
    sensor_norm = (sensor_raw - mean) / (std + 1e-6)  # Every batch!
```

**Fix:** Normalize in dataset

```python
class TrajectoryDataset(Dataset):
    def __init__(self, ..., mean=None, std=None):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        sensor_raw = ...
        if self.mean is not None:
            sensor_norm = (sensor_raw - self.mean) / (self.std + 1e-6)
            return {"imu_seq_raw": sensor_raw, "imu_seq_norm": sensor_norm, ...}
```

---

### 11\. Missing Docstrings on Key Functions

**Files:** Multiple
**Impact:** Onboarding difficulty, unclear contracts

**Examples:**

*   `train_step()` - Complex DWA logic undocumented
*   `_filter_step()` - Abstract method without clear contract
*   `preprocess_gt_data()` - 70+ line transformation pipeline

**Fix:** Add comprehensive docstrings

```python
def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    config: TrainConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
    task_weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Execute single training step with optional DWA weighting.

    Args:
        model: ESKFTCN_model instance in train mode
        batch: Batch dictionary with keys:
            - imu_seq_raw: [B, T, 7] sensor readings (accel, gyro, force)
            - gt_vel_w: [B, T, 3] ground truth velocity (world frame)
            - gt_pos_w: [B, T, 3] ground truth position (world frame)
            - len: [B] valid sequence lengths
        config: Training configuration with loss weights
        mean: [7] sensor normalization mean
        std: [7] sensor normalization std
        task_weights: Optional [5] DWA weights [mag, cos, zupt, cov, fft]

    Returns:
        Dictionary with loss components:
            - mag: Magnitude loss (scalar)
            - cos: Cosine loss (scalar)
            - zupt: ZUPT loss (scalar)
            - cov: Covariance NLL loss (scalar)
            - fft: FFT loss (scalar)
            - reg: Regularization loss (scalar)
            - delta: Delta loss (scalar)
            - total: Weighted sum of all losses (scalar)

    Note:
        If task_weights is None, uses fixed config weights.
        NaN losses are detected by caller and skipped.

    Example:
        >>> losses = train_step(model, batch, config, mean, std)
        >>> losses["total"].backward()
        >>> optimizer.step()
    """
    ...
```

---

### 12\. Inconsistent Variable Naming

**Files:** Throughout
**Impact:** Code harder to read, errors from confusion

**Examples:**

*   `rot_mat` vs `R` vs `rot` (rotation matrices)
*   `vel_resid_b` (should be `vel_correction_b`)
*   `tcn_output_mask` vs `mask` (unclear difference)
*   `seq_lens` vs `lengths` vs `len` (sequence lengths)

**Fix:** Establish naming conventions

```python
# NAMING CONVENTIONS
# Coordinate frames: _w (world), _b (body), _s (sensor)
# Positions: pos_w, pos_b
# Velocities: vel_w, vel_b, vel_correction_b (not vel_resid_b)
# Rotations: R_w_to_b, R_b_to_w (clear direction)
# Quaternions: quat (not q)
# Masks: mask, mask_valid, mask_tcn, mask_moving (clear purpose)
# Lengths: seq_len (not len, lengths, seq_lens)
```

---

## 🟢 Low Priority Issues (Polish)

### 13\. Missing Type Hints

**Files:** Throughout
**Impact:** Limited IDE support, unclear function contracts

**Fix:** Use TypedDict for complex returns

```python
from typing import TypedDict, Dict
import torch

class TrainStepOutput(TypedDict):
    mag: torch.Tensor
    cos: torch.Tensor
    zupt: torch.Tensor
    cov: torch.Tensor
    fft: torch.Tensor
    reg: torch.Tensor
    delta: torch.Tensor
    total: torch.Tensor

def train_step(...) -> TrainStepOutput:
    ...
```

---

### 14\. Dead Code / Deprecated Files

**Files:** `model_old_deprecated/`, commented blocks
**Impact:** Confusion, maintenance burden

**Fix:**

*   Move `model_old_deprecated/` to Git archive branch
*   Add `@deprecated` decorator for old functions
*   Update docs with "Use X instead of Y"

---

### 15\. Tight Coupling Filter-TCN

**File:** `model/ESKF_TCN.py`
**Impact:** Hard to swap filters, test independently

**Fix:** Use dependency injection

```python
class HybridFilterTCN(nn.Module):
    def __init__(
        self,
        filter: KalmanFilterBase,  # Inject filter
        tcn: TCN,                   # Inject TCN
        device: str = "cpu"
    ):
        super().__init__()
        self.filter = filter
        self.tcn = tcn
        self.device = device

# Usage
eskf = ESKF(device=device, ...)
tcn = TCN(input_size=19, ...)
model = HybridFilterTCN(filter=eskf, tcn=tcn, device=device)
```

---

## Summary Statistics

| Category | Count | Lines of Code Affected |
| --- | --- | --- |
| 🔴 Critical Issues | 4 | ~50 LOC to fix |
| 🟠 High Priority | 4 | ~500 LOC to refactor |
| 🟡 Medium Priority | 5 | ~200 LOC to improve |
| 🟢 Low Priority | 3 | ~100 LOC to polish |
| **Total** | **16** | **~850 LOC** |

---

## Recommended Action Plan

### Week 1: Critical Fixes (Before Stage 2 Training)

*   Add input validation to training loops
*   Implement proper NaN loss logging
*   Add file existence checks
*   Improve Cholesky fallback logic

### Week 2: High Priority Refactoring

*   Extract common training loop logic
*   Unify dataset augmentation
*   Move magic numbers to config
*   Split `_initialize_state()` into smaller methods

### Week 3: Medium Priority Improvements

*   Implement lazy dataset loading
*   Move normalization to dataset
*   Add comprehensive docstrings
*   Standardize variable naming

### Week 4: Low Priority Polish

*   Add type hints with TypedDict
*   Archive deprecated code
*   Implement dependency injection for filters

---

## Testing Strategy

After each fix, run:

```
# 1. Unit tests (if available)
pytest tests/

# 2. Quick training smoke test (1 epoch)
python train_eskf.py --epochs 1 --batch-size 4

# 3. Validation on existing model
python validate.py --model_type eskf_tcn --model_path checkpoints/eskf_tcn_two_stage_stage1_best.pth

# 4. Full training (after major refactoring)
python train_two_stage.py --skip-stage1
```

---

## Additional Resources

*   **Detailed Analysis:** See exploration agent output above
*   **Loss System:** `LOSS_DWA_ANALYSIS.md`
*   **Recent Improvements:** `model/losses.py` (FFT windowing, mutual exclusivity)
*   **Configuration:** `model/config.py`

---

## Questions?

For each issue, you can:

1.  Refer to specific file:line mentioned
2.  Check existing implementation
3.  Test suggested fix in isolation
4.  Submit PR with fix + tests