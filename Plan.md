# Plan: Use Apple Pencil Pose for gravity_b Ground Truth

## Problem Statement

Currently, `gravity_b` GT is derived from accelerometer during low-dynamics periods:
```python
# train.py:261
gt_gravity_b = accel_raw / (accel_norm + 1e-6)  # Unit vector
```

**Limitations:**
1. Only valid when `accel_norm ≈ 9.81 m/s²` (static or slow motion)
2. During dynamic motion, accelerometer = gravity + linear_accel (unusable)
3. Results in sparse supervision signal (only ~30-50% of timesteps)

**Solution:** Use Apple Pencil's pose data (`azimuth`, `altitude`, `rollAngle`) to compute GT gravity vector in body frame at **every timestep**.

---

## Available Data

From `acquired_data/Sample_N.csv` (stored in `raw_acquired_data.h5`):

| Column | Type | Description |
|--------|------|-------------|
| `azimuth` | radians | Angle in x-y plane from positive x-axis (counterclockwise) |
| `altitude` | radians | Angle from iPad surface plane (0 = flat, π/2 = vertical) |
| `rollAngle` | radians | Rotation around pencil's longitudinal axis |

**Note:** Data at ~240Hz, resampled to 50Hz during preprocessing.

---

## Coordinate Frames

### iPad Frame (World Reference)
- **Origin:** iPad screen center
- **X:** Right (landscape)
- **Y:** Up (portrait direction)
- **Z:** Out of screen (towards user)
- **Gravity:** `g_ipad = [0, 0, -9.81]` (into table when flat)

### Pencil Body Frame (IMU)
- **Origin:** IMU location inside pencil
- **X:** Along pencil tip direction
- **Y:** Perpendicular (in-plane with screen during normal writing)
- **Z:** Out of pencil (completing right-hand system)

---

## Conversion Algorithm

### Step 1: Apple Pencil Angles → Rotation Matrix

Apple Pencil orientation is defined by:
1. **Azimuth (ψ):** Rotation around iPad's Z-axis
2. **Altitude (θ):** Elevation angle from the x-y plane
3. **Roll (φ):** Rotation around pencil's own axis

**Rotation sequence:** ZYX Euler angles (extrinsic) or XYZ (intrinsic body-fixed)

```python
def pencil_angles_to_rotation(azimuth, altitude, roll):
    """
    Convert Apple Pencil angles to rotation matrix R_pencil_to_ipad.

    Args:
        azimuth: Angle in x-y plane from +x axis (radians)
        altitude: Angle above x-y plane (radians)
        roll: Rotation around pencil axis (radians)

    Returns:
        R: 3x3 rotation matrix (body-to-world)
    """
    # Azimuth rotation around Z
    Rz = np.array([
        [cos(azimuth), -sin(azimuth), 0],
        [sin(azimuth),  cos(azimuth), 0],
        [0,             0,            1]
    ])

    # Altitude rotation around Y (after azimuth)
    Ry = np.array([
        [cos(altitude),  0, sin(altitude)],
        [0,              1, 0            ],
        [-sin(altitude), 0, cos(altitude)]
    ])

    # Roll rotation around X (pencil axis)
    Rx = np.array([
        [1, 0,          0         ],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll),  cos(roll)]
    ])

    # Combined rotation: R = Rz @ Ry @ Rx
    R_pencil_to_ipad = Rz @ Ry @ Rx
    return R_pencil_to_ipad
```

### Step 2: Compute Gravity in Body Frame

```python
g_ipad = np.array([0, 0, -9.81])  # Gravity in iPad frame

# Transform to body frame
R_pencil_to_ipad = pencil_angles_to_rotation(azimuth, altitude, roll)
R_ipad_to_pencil = R_pencil_to_ipad.T

g_body = R_ipad_to_pencil @ g_ipad  # Gravity in pencil's body frame
g_body_unit = g_body / np.linalg.norm(g_body)  # Unit vector
```

---

## Implementation Steps

### Phase 1: Data Pipeline Modification (`utils/acquire.py`)

1. **Modify `preprocess_gt_data()`:**
   - Extract `azimuth`, `altitude`, `rollAngle` columns
   - Interpolate pose angles to 50Hz (same as position)
   - Return extended DataFrame with pose columns

2. **Modify HDF5 saving:**
   - Add new dataset: `gt_gravity_b_data` (shape: [seq_len, 3])
   - Or add pose angles: `gt_pose_data` (shape: [seq_len, 3]) for on-the-fly computation

### Phase 2: Dataset Modification (`model/dataset.py`)

1. **Update `TrajectoryDataset.__init__()`:**
   - Load `gt_gravity_b_data` or compute from pose angles
   - Add to cached data dict

2. **Update `__getitem__()`:**
   - Return `gravity_b_gt` tensor in batch dict

### Phase 3: Training Modification (`train.py`)

1. **Update `UncertaintyLoss.forward()`:**
   ```python
   # Replace current gravity GT (from accelerometer):
   # gt_gravity_b = accel_raw / (accel_norm + 1e-6)

   # With pencil-derived GT:
   gt_gravity_b = batch["gravity_b_gt"]  # [B, T, 3]

   # No need for accel_valid mask - valid at all timesteps!
   gravity_mask = tcn_output_mask  # Only mask TCN warmup period
   ```

2. **Remove low-dynamics filtering:**
   - Delete `accel_valid = (accel_norm > 8.0) & (accel_norm < 12.0)`
   - Full supervision during both static and dynamic motion

---

## Critical Considerations

### 1. Coordinate Frame Alignment

**IMPORTANT:** The IMU body frame and Apple Pencil angle reference frame may not be perfectly aligned. Need to determine:

- Is the IMU X-axis aligned with pencil tip direction?
- Verify with static calibration: hold pencil at known angles and compare computed `g_body` with accelerometer reading

**Calibration procedure:**
```python
# During static hold (accel = gravity):
g_from_accel = accel_raw / np.linalg.norm(accel_raw)
g_from_pencil = compute_gravity_from_pencil_angles(azimuth, altitude, roll)

# Find rotation offset R_offset such that:
# g_from_accel = R_offset @ g_from_pencil
```

### 2. Handling Hover State

When `isHovering = 1`:
- Pose data may be less accurate (not in contact)
- Position data is approximate
- **Recommendation:** Keep pose data but add confidence weighting or mask

### 3. rollAngle Availability

Apple Pencil Pro provides `rollAngle`; older pencils may not:
- Check if `rollAngle ≠ 0` in data
- If unavailable, assume `roll = 0` (minor impact on gravity direction)

### 4. Timestamp Synchronization

Pose angles and IMU data must be time-aligned:
- Pose is already resampled to 50Hz during preprocessing
- Two-tap sync aligns iPad and ESP32 clocks
- Same interpolation should apply to pose angles

---

## Validation Plan

1. **Unit Test:** Compare computed `g_body` with accelerometer during static periods
   - Expected: Cosine similarity > 0.99

2. **Visual Validation:** Plot both signals over time
   - Pencil-derived should be smooth
   - Accelerometer-derived should match during static, diverge during motion

3. **Training Comparison:**
   - Train with accelerometer-based GT (baseline)
   - Train with pencil-based GT (proposed)
   - Compare APE metrics on validation set

---

## Files to Modify

| File | Changes |
|------|---------|
| `utils/acquire.py` | Add pose preprocessing, compute gravity_b, save to HDF5 |
| `model/dataset.py` | Load gravity_b GT from HDF5, add to batch dict |
| `train.py` | Use pencil-derived GT, remove dynamics filtering |
| `model/config.py` | Add `GT_GRAVITY_FROM_PENCIL = True` flag |

---

## Alternative: Reprocess from Raw Data

Since `raw_acquired_data.h5` contains all pose angles, we can:

1. Add `--recompute-gravity` flag to `acquire.py`
2. Read from raw, compute `gt_gravity_b`, write to processed HDF5
3. No need to recollect data

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Frame misalignment | Medium | Calibration procedure, validation check |
| rollAngle unavailable | Low | Check data, fallback to roll=0 |
| Pose noise during motion | Low | Apple's sensor fusion is robust |
| Training regression | Low | A/B test before full adoption |

---

## Implementation Status: COMPLETE

### Files Modified:
1. **`utils/acquire.py`:**
   - Added `pencil_angles_to_rotation_matrix()` - converts azimuth/altitude/roll to rotation matrix
   - Added `compute_gravity_body_from_pencil_pose()` - transforms gravity to body frame
   - Added `calibrate_imu_pencil_alignment()` - uses Kabsch algorithm to find frame alignment
   - Modified `preprocess_gt_data()` - preserves pose angles with azimuth unwrapping
   - Modified `preprocess_single()` - computes `gt_gravity_b` using first static frame for calibration
   - Modified `save_data()` and `_finalize_dataset()` - saves `gt_gravity_b_data` to HDF5

2. **`model/dataset.py`:**
   - Modified `__init__()` - loads `gt_gravity_b_data` if available
   - Modified `__getitem__()` - returns `gt_gravity_b` and applies local grip augmentation

3. **`train.py`:**
   - Modified `UncertaintyLoss.forward()` - uses pencil-derived GT when available, falls back to accelerometer

### Validation Results:
- All rotation matrices are proper (det=1, R@R.T=I)
- Gravity vectors are unit normalized
- Calibration achieves <1° alignment error with synthetic data
- Real data shows frame misalignment (expected) - calibration step is essential

### To Reprocess Existing Data:
```bash
python utils/acquire.py --reprocess
```
This will regenerate the dataset with `gt_gravity_b_data` included.

---
---

# Plan: Gradient Conflict Resolution

## Current Issues

Based on gradient conflict analysis:
```
[Grad Conflict] vel↔cos: +0.091 | vel↔zupt: -0.119 | vel↔cov: +0.304 | cos↔cov: +0.211
[Grad Norms] vel: 6.76e+00 | cos: 7.41e+00 | zupt: 3.25e+00 | cov: 1.05e+01
```

### Problems Identified:
1. **vel↔zupt conflict (-0.119)**: Velocity and ZUPT gradients oppose each other
2. **Gradient imbalance**: Cov (10.5) dominates, zupt (3.25) is weakest (~3x difference)
3. **vel↔cos near-orthogonal (+0.091)**: Both velocity losses barely cooperate

---

## Solution 1: PCGrad (Projecting Conflicting Gradients)

**Paper**: "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)

**Idea**: When two task gradients conflict (negative cosine), project one onto the normal plane of the other to remove the conflicting component.

```python
def pcgrad(grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Apply PCGrad to a list of task gradients."""
    num_tasks = len(grads)
    projected_grads = [g.clone() for g in grads]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                cos_sim = (projected_grads[i] @ grads[j]) / (grads[j].norm() ** 2 + 1e-8)
                if cos_sim < 0:  # Only project if conflicting
                    projected_grads[i] = projected_grads[i] - cos_sim * grads[j]

    return projected_grads
```

**Implementation location**: After computing individual loss gradients, before `optimizer.step()`

### PCGrad Integration in train.py:
```python
# After computing loss, instead of loss.backward():
losses_list = [loss_vel, loss_cos, loss_zupt, loss_cov, loss_reg]
grads_list = []

for l in losses_list:
    model.zero_grad()
    l.backward(retain_graph=True)
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    grads_list.append(grads)

# Apply PCGrad
projected_grads = pcgrad(grads_list)
final_grad = sum(projected_grads)

# Set gradients manually
idx = 0
for p in model.parameters():
    if p.grad is not None:
        numel = p.grad.numel()
        p.grad = final_grad[idx:idx+numel].view_as(p.grad)
        idx += numel

optimizer.step()
```

---

## Solution 2: GradNorm (Gradient Balancing)

**Idea**: Normalize gradient magnitudes so no single loss dominates.

### Simple Version (Gradient Scaling):
```python
# Compute target norm (average of all)
target_norm = np.mean([norm_vel, norm_cos, norm_zupt, norm_cov])

# Scale each loss to achieve balanced gradients
scale_vel = target_norm / (norm_vel + 1e-8)
scale_cos = target_norm / (norm_cos + 1e-8)
scale_zupt = target_norm / (norm_zupt + 1e-8)
scale_cov = target_norm / (norm_cov + 1e-8)

# Apply scales
loss = scale_vel * loss_vel + scale_cos * loss_cos + scale_zupt * loss_zupt + scale_cov * loss_cov
```

### Advanced Version (Learnable Weights):
```python
# In UncertaintyLoss.__init__
self.grad_weights = nn.ParameterDict({
    'vel': nn.Parameter(torch.tensor(1.0)),
    'cos': nn.Parameter(torch.tensor(1.0)),
    'zupt': nn.Parameter(torch.tensor(1.0)),
    'cov': nn.Parameter(torch.tensor(1.0)),
})

# Normalize weights to sum to num_tasks
weights = F.softmax(torch.stack([self.grad_weights[k] for k in ['vel', 'cos', 'zupt', 'cov']]), dim=0) * 4
```

---

## Solution 3: Conflict Penalty Term

**Idea**: Add regularization that penalizes gradient conflicts during training.

```python
def gradient_conflict_penalty(grad_dict: Dict[str, torch.Tensor], pairs: List[Tuple[str, str]]) -> torch.Tensor:
    """Penalize negative cosine similarities between task gradients."""
    penalty = torch.tensor(0.0, device=next(iter(grad_dict.values())).device)

    for k1, k2 in pairs:
        if k1 in grad_dict and k2 in grad_dict:
            g1, g2 = grad_dict[k1], grad_dict[k2]
            cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0))
            if cos_sim < 0:
                penalty = penalty - cos_sim  # Add positive penalty for conflicts

    return penalty

# In training loop:
conflict_penalty = gradient_conflict_penalty(grad_dict, [('vel', 'zupt'), ('cos', 'reg')])
total_loss = total_loss + 0.1 * conflict_penalty
```

---

## Solution 4: Loss Redesign (Address Root Cause)

### vel↔zupt Conflict Root Cause:
- **vel loss**: Minimize |pred_vel - gt_vel| for ALL timesteps
- **zupt loss**: Force pred_zupt → 1 when gt_vel ≈ 0

**Conflict scenario**: During transition (slow motion):
- vel loss pushes toward small nonzero velocity
- zupt might trigger, creating opposing gradients

### Fix: Mutual Exclusivity
```python
# Apply vel loss only when clearly moving
moving_mask = (gt_vel_norm > 0.01).squeeze(-1)  # Clear motion threshold
vel_loss = (per_element_vel_loss * mask * moving_mask.unsqueeze(-1)).sum() / ...

# Apply zupt loss only when clearly static
static_mask = (gt_vel_norm < 0.005).squeeze(-1)  # Clear static threshold
zupt_loss = (per_element_zupt_loss * mask * static_mask.unsqueeze(-1)).sum() / ...
```

This creates a **dead zone** (0.005 < vel < 0.01) where neither loss applies strongly, reducing conflicts.

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Immediate)
- [ ] **Gradient normalization**: Scale each loss's gradient to similar magnitude
- [ ] **Mutual exclusivity masks**: Separate vel/zupt application regions

### Phase 2: PCGrad (Medium effort)
- [ ] Implement PCGrad helper function
- [ ] Integrate into training loop
- [ ] Verify vel↔zupt conflict resolves

### Phase 3: Advanced (If needed)
- [ ] Learnable gradient weights (GradNorm style)
- [ ] Conflict penalty term

---

## Implementation Checklist

- [ ] Add `pcgrad()` helper function in `train.py`
- [ ] Add gradient normalization option
- [ ] Implement mutual exclusivity masks for vel/zupt
- [ ] Re-run gradient conflict analysis after changes
- [ ] Compare validation APE before/after

---

## References

- PCGrad: https://arxiv.org/abs/2001.06782
- GradNorm: https://arxiv.org/abs/1711.02257
- CAGrad: https://arxiv.org/abs/2110.14048
