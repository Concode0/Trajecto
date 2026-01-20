# Loss and DWA System Analysis

**Date:** 2026-01-19
**System Version:** Trajecto ESKF-TCN Hybrid Model

---

## Executive Summary

Your training system uses **7 loss components** with **Dynamic Weight Averaging (DWA)** to automatically balance multi-task learning. The system separates losses into:
- **5 Learning Tasks** (managed by DWA): magnitude, cosine, ZUPT, covariance, FFT
- **2 Physics Constraints** (fixed weights): regularization, delta loss

---

## Loss Components Breakdown

### 1. **Magnitude Loss** (`mag`) - L1 Loss on Velocity Magnitude
**Purpose:** Match the speed (magnitude) of predicted velocity to ground truth
**Implementation:** `losses.py:6-11`
```python
pred_mag = ||pred_vel||
gt_mag = ||gt_vel||
loss = L1(pred_mag, gt_mag)
```

**Characteristics:**
- ✅ Simple and stable
- ✅ Rotation-invariant (only cares about speed, not direction)
- ⚠️ Doesn't penalize direction errors (needs cosine loss companion)

**DWA Status:** ✅ Managed by DWA (weight[0])

---

### 2. **Context-Aware Direction Loss** (`cos`) - Gyro-Weighted Cosine Similarity
**Purpose:** Match velocity direction, with **extra penalty during fast rotations**
**Implementation:** `losses.py:14-46`
```python
base_loss = 1 - cosine_similarity(pred_vel, gt_vel)
gyro_mag = ||gyro_raw||  # rad/s
context_weight = 1.0 + 2.0 * gyro_mag
weighted_loss = base_loss * context_weight
```

**Key Innovation:** During sharp turns (high gyro), direction errors get amplified (up to 7x weight).

**Example:**
- Straight line (gyro=0): weight = 1.0
- Sharp turn (gyro=3 rad/s): weight = 7.0

**Characteristics:**
- ✅ Physics-aware: recognizes that direction matters more during rotation
- ✅ Masked during ZUPT (direction is meaningless when stationary)
- ⚠️ Assumes gyro indices are [3:6] in sensor array

**DWA Status:** ✅ Managed by DWA (weight[1])

---

### 3. **ZUPT Loss** (`zupt`) - Binary Cross-Entropy for Zero-Velocity Detection
**Purpose:** Train TCN to predict when the pen is stationary
**Implementation:** `losses.py:48-51`
```python
gt_zupt = (||gt_vel|| < 0.005 m/s) ? 1.0 : 0.0
loss = BCE_with_logits(pred_zupt, gt_zupt)
```

**Characteristics:**
- ✅ Enables TCN to learn temporal patterns indicating stillness
- ✅ Provides supervision for ESKF's ZUPT update mechanism
- ⚠️ Threshold-sensitive (0.005 m/s = 5 mm/s)

**DWA Status:** ✅ Managed by DWA (weight[2])

---

### 4. **Covariance NLL Loss** (`cov`) - Negative Log-Likelihood for Uncertainty Calibration
**Purpose:** Train TCN to predict adaptive measurement noise (R matrix)
**Implementation:** `losses.py:54-70`
```python
variance = softplus(pred_R) + 1e-4  # Ensure positive
variance = clamp(variance, 1e-4, 3.0)
nll = 0.5 * (innovation^2 / variance + log(variance))
```

**Key Concept:** TCN learns to output low variance when confident, high variance when uncertain.

**Characteristics:**
- ✅ Enables Bayesian filtering: ESKF trusts TCN more when variance is low
- ✅ Prevents overconfidence via log(variance) penalty
- ⚠️ Only active after TCN warmup period (`tcn_mask`)
- ⚠️ Small weight (0.01 default) to avoid overwhelming other losses

**DWA Status:** ✅ Managed by DWA (weight[3])

---

### 5. **FFT Loss** (`fft`) - Frequency Domain Matching
**Purpose:** Match spectral characteristics of velocity over time
**Implementation:** `losses.py:79-104`
```python
pred_fft = rfft(pred_vel, dim=time)
gt_fft = rfft(gt_vel, dim=time)
loss = L1(log(|pred_fft| + eps), log(|gt_fft| + eps))
```

**Rationale:** Time-domain losses can miss high-frequency jitter or low-frequency drift. FFT loss enforces spectral matching.

**Characteristics:**
- ✅ Penalizes unrealistic frequency content
- ✅ Log-scale makes it robust to magnitude variations
- ⚠️ Masking strategy affects FFT quality (zeros introduce artifacts)

**DWA Status:** ✅ Managed by DWA (weight[4])

---

### 6. **Regularization Loss** (`reg`) - L2 Penalty on TCN Corrections
**Purpose:** Prevent TCN from making overly large velocity corrections
**Implementation:** `losses.py:73-76`
```python
loss = ||vel_correction||^2
```

**Philosophy:** ESKF physics should do most of the work. TCN should only make small adjustments.

**Characteristics:**
- ✅ Prevents TCN from "hijacking" the solution
- ✅ Encourages sparse corrections
- ⚠️ **Fixed weight (1e-7)** - NOT managed by DWA
- ⚠️ Extremely low weight (config: `REG_WEIGHT_ESKF_TCN = 1e-7`)

**DWA Status:** ❌ Fixed weight (physics constraint)

---

### 7. **Delta Loss** (`delta`) - Semi-Loop Closure via Sliding Windows
**Purpose:** Enforce displacement consistency over multiple time scales
**Implementation:** `losses.py:107-164`
```python
for window_size in [25, 100, 250]:  # 0.5s, 2s, 5s @ 50Hz
    for t in range(0, T - window_size, stride=10):
        delta_pred = pred_pos[t+W] - pred_pos[t]
        delta_gt = gt_pos[t+W] - gt_pos[t]
        loss += L1(delta_pred, delta_gt)
```

**Key Insight:** Even without actual loop closures, we can check if predicted displacement over any window matches GT. This prevents cumulative drift.

**Characteristics:**
- ✅ Multi-scale: short (0.5s), medium (2s), long (5s)
- ✅ Stride=10 for efficiency (don't check every timestep)
- ✅ Requires position integration from velocity
- ⚠️ **Fixed weight (0.5)** - NOT managed by DWA
- ⚠️ Computationally expensive (3 nested loops)

**DWA Status:** ❌ Fixed weight (physics constraint)

---

## Dynamic Weight Averaging (DWA)

### Algorithm Overview
**Paper:** "End-to-End Multi-Task Learning with Attention" (Liu et al., CVPR 2019)

**Core Principle:** Tasks that improve slowly need higher weight.

### Implementation (`model/dwa.py`)

```python
class DWALossUpdater:
    def get_weights(self, current_epoch_losses):
        # 1. Calculate learning rate ratio
        r_k = L(t-1) / L(t-2)  # How much did loss drop?

        # 2. Small r_k = fast learning = reduce weight
        #    Large r_k = slow learning = increase weight

        # 3. Softmax normalization with temperature
        w = exp(r_k / temp) / sum(exp(r_k / temp))

        return w * num_tasks  # Scale to sum = num_tasks
```

### Temperature Parameter (`temp=2.0`)
- **Low temp (1.0):** Aggressive - large weight differences
- **High temp (5.0):** Conservative - smoother weights
- **Current (2.0):** Balanced

### Managed Tasks (5)
| Index | Task | Initial Weight | DWA Controlled |
|-------|------|----------------|----------------|
| 0 | Magnitude | 1.0 | ✅ Yes |
| 1 | Cosine | 1.0 | ✅ Yes |
| 2 | ZUPT | 0.5 | ✅ Yes |
| 3 | Covariance | 0.01 | ✅ Yes |
| 4 | FFT | 0.5 | ✅ Yes |

### Excluded from DWA (2)
- **Regularization (1e-7):** Physics constraint, should always be tiny
- **Delta Loss (0.5):** Drift correction, fixed importance

---

## Current Configuration Summary

### Default Weights (Initial/Fallback)
```python
# From train_eskf.py:63-69
w_mag = 1.0
w_cos = 1.0
w_zupt = 0.5
w_cov = 0.01
w_fft = 0.5
w_reg = 1e-7  # FIXED (not in DWA)
w_delta = 0.5  # FIXED (not in DWA)
```

### Two-Stage Training Behavior
**Stage 1 (Sim Pretrain):**
- DWA manages: [mag, cos, zupt, cov, fft]
- ESKF parameters frozen
- Focus: Learn TCN feature extraction

**Stage 2 (Mixed Fine-tune):**
- DWA continues managing same 5 tasks
- ESKF parameters unfrozen (separate LR: 1e-6)
- Focus: Adapt to real-world data distribution

---

## Potential Issues & Observations

### ⚠️ Issue 1: Covariance Weight Disparity
**Problem:** Initial weight for covariance is 0.01 (100x smaller than mag/cos)

**Impact:**
- DWA ratios are relative, but initial weights matter for first 2 epochs
- Covariance may be under-trained early on

**Recommendation:**
- Consider starting covariance at 0.1 or 0.5
- Or use uncertainty-based automatic weighting (Kendall et al.) instead of DWA for this task

---

### ⚠️ Issue 2: Regularization is Too Weak
**Problem:** `w_reg = 1e-7` is extremely small

**Impact:**
- TCN can make arbitrarily large corrections without penalty
- May overfit to training noise

**Recommendation:**
- Increase to `1e-4` or `1e-3` and monitor validation loss
- Check magnitude of `vel_correction` outputs during training

---

### ⚠️ Issue 3: Delta Loss Computational Cost
**Problem:** Triple nested loop with FFT integration

**Impact:**
- Training slowdown (especially with large batch sizes)
- May not be worth 0.5 weight

**Recommendation:**
- Profile training time with/without delta loss
- Consider reducing window sizes or increasing stride

---

### ⚠️ Issue 4: FFT Masking Artifacts
**Problem:** Zeroing invalid timesteps creates discontinuities

**Impact:**
- FFT may pick up artificial high-frequency content from mask edges

**Recommendation:**
- Use window functions (Hann, Hamming) before FFT
- Or apply FFT only to valid continuous segments

---

### ⚠️ Issue 5: DWA Warmup Period
**Problem:** First 2 epochs use equal weights (requires history)

**Impact:**
- Training dynamics differ between early and late epochs
- May cause initial instability

**Recommendation:**
- Increase warmup to 5 epochs to let loss landscape stabilize
- Monitor weight transitions after warmup

---

## Diagnostic Tools

### Check Current DWA Behavior
During training, the pbar shows:
```python
pbar.set_postfix({
    "Loss": f"{losses['total'].item():.2f}",
    "MagW": f"{current_task_weights[0]:.2f}",
    "CosW": f"{current_task_weights[1]:.2f}",
    "FFTW": f"{current_task_weights[4]:.2f}"
})
```

### Epoch Summary
```
[DWA Weights] Mag: 1.23, Cos: 0.87, Zupt: 1.45, Cov: 0.92, FFT: 0.53
```

**Healthy Pattern:**
- Weights oscillate gently around 1.0
- No single weight dominates (>5.0) or collapses (<0.1)

**Unhealthy Pattern:**
- One weight explodes (>10.0) → That task isn't learning
- One weight vanishes (<0.01) → Task converged too fast

---

## Recommendations

### Short-Term (Immediate)
1. ✅ **Monitor DWA weights** during next training run
2. ✅ **Increase `w_reg`** from 1e-7 to 1e-4
3. ✅ **Profile delta loss** impact on training time

### Medium-Term (Next Iteration)
4. 🔄 **Adjust covariance initial weight** to 0.1
5. 🔄 **Add gradient conflict diagnostics** (from Plan.md)
6. 🔄 **Implement FFT windowing** to reduce artifacts

### Long-Term (Future Work)
7. 🔮 **Compare DWA vs Uncertainty Weighting** (Kendall et al.)
8. 🔮 **Experiment with PCGrad** for gradient surgery
9. 🔮 **Add task-specific learning rate scheduling**

---

## Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| Loss Functions | `model/losses.py` | 1-188 |
| DWA Implementation | `model/dwa.py` | 1-49 |
| Training Loop (Single-Stage) | `train_eskf.py` | 241-380 |
| Training Loop (Two-Stage) | `train_two_stage.py` | 113-206, 230-350 |
| Loss Config | `model/config.py` | 139-169 |

---

## Experiment Suggestions

### Experiment 1: Disable Delta Loss
```bash
python train_eskf.py --no-delta-loss
```
**Hypothesis:** Training will be faster with minimal accuracy loss.

### Experiment 2: Increase Regularization
```bash
python train_eskf.py --w-reg 1e-4
```
**Hypothesis:** TCN corrections will be smaller and more stable.

### Experiment 3: DWA Temperature Sweep
Modify `dwa.py:10` to test `temp ∈ [1.0, 2.0, 3.0, 5.0]`
**Hypothesis:** Higher temp = more stable training, lower temp = faster convergence.

---

## Conclusion

Your loss system is **well-designed and modular**, with clear separation between learning tasks (DWA-managed) and physics constraints (fixed). The DWA algorithm is correctly implemented and actively balancing task weights.

**Primary concerns:**
1. Regularization weight may be too weak (1e-7)
2. Covariance weight disparity (0.01 vs 1.0)
3. Delta loss computational cost vs benefit

**Next steps:**
1. Run Stage 2 training and monitor DWA weight evolution
2. Validate on hold-out set
3. Profile training time breakdown
4. Consider gradient conflict analysis from Plan.md
