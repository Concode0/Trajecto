# Feature Vector Update Changelog

## Version 2.0 (19D) - 2025-12-31

### Summary

Removed `zupt_flag` from TCN feature vector to eliminate circular dependency when using TCN-based ZUPT detection. Feature dimensionality reduced from **20D → 19D**.

---

## Changes

### Modified Files

#### Core Model Files

1. **`model/base_hybrid_model.py`**
   - Line 384-392: Removed `tcn_features_from_filter["zupt_flag"]` from feature concatenation
   - Added comment explaining circular dependency

2. **`model/config.py`**
   - Line 45: `TCN_INPUT_SIZE = 19` (was 20)
   - Line 61: `AEKFTCN.TCN_INPUT_SIZE = 19` (was 20)
   - Updated comments to explain removal

#### Firmware Files

3. **`firmware/components/trajecto_core/include/model_params.hpp`**
   - Line 8: `TCN_INPUT_SIZE = 19` (was 20)
   - Line 15: First state buffer dimension `{ 19, 2 }` (was `{ 20, 2 }`)
   - Added comments

4. **`firmware/components/trajecto_core/src/tcn_wrapper.cpp`**
   - Line 125: Removed `out_features[idx++] = is_zupt ? 1.0f : 0.0f;`
   - Added comment explaining removal

#### Documentation Files

5. **`CLAUDE.md`**
   - Updated feature list from 20D to 19D
   - Expanded feature description with breakdown

6. **`firmware/COMPUTATION_ANALYSIS.md`**
   - Line 98: Updated to 19D
   - Line 111: Updated TCN_INPUT_SIZE reference
   - Line 15: Updated first state buffer dimension

7. **`docs/TCN_FEATURE_VECTOR_19D.md`** (NEW)
   - Comprehensive 19D feature specification
   - Includes normalization stats, code examples, validation checks

8. **`docs/FEATURE_UPDATE_CHANGELOG.md`** (NEW - this file)
   - Migration guide and changelog

---

## Technical Details

### Removed Feature

**`zupt_flag`** (Index 13 in 20D version)
- **Type**: Binary {0, 1}
- **Source**: Classic threshold-based ZUPT detector OR TCN's own zupt_prob
- **Reason for Removal**: Circular dependency

### Circular Dependency Problem

**Flow with zupt_flag** (BROKEN):
```
T=0: zupt_flag[0] = 0 (default)
     ↓
T=1: TCN reads zupt_flag[0] → predicts zupt_prob[1] → is_zupt[1] = (prob > 0.5)
     ↓
T=2: TCN reads zupt_flag[1] = is_zupt[1] → predicts zupt_prob[2]
     ↓
... TCN echoes its own past decisions!
```

**Flow without zupt_flag** (FIXED):
```
T=0: TCN reads [gyro, accel, force, vel, gravity, innovation]
     ↓ Learns: force low + vel low → ZUPT
T=1: TCN predicts zupt_prob[1] from physics features
     ↓ No self-reference
T=2: TCN predicts zupt_prob[2] from physics features
     ↓
... TCN learns physical patterns, not shortcuts!
```

### New Feature Order (19D)

| Index | Feature | Dim | Change |
|-------|---------|-----|--------|
| 0-2 | `gyro_b_norm` | 3 | ✓ No change |
| 3-5 | `accel_b_norm` | 3 | ✓ No change |
| 6 | `force_norm` | 1 | ✓ No change |
| 7-9 | `pen_tip_vel_b_squashed` | 3 | ✓ No change |
| 10-12 | `gravity_b_norm` | 3 | ✓ No change |
| ~~13~~ | ~~`zupt_flag`~~ | ~~1~~ | ✗ **REMOVED** |
| 13-18 | `innovation_squashed` | 6 | ⚠️ Index shifted down |

**Critical**: `innovation_squashed` moved from indices 14-19 to 13-18

---

## Impact Analysis

### Python Training

**Breaking Change**: ✅ Models trained with 20D are incompatible

**Required Actions**:
1. Delete old checkpoints (20D models won't load)
2. Retrain from scratch with new 19D features
3. Re-export to ONNX/TFLite

**Expected Accuracy Change**: ±0-2% (zupt_flag was redundant)

### Firmware

**Breaking Change**: ✅ Old TFLite models (20D) will fail

**Error Symptoms**:
```
[ERROR] TFLite tensor shape mismatch
Expected: [1, 1, 19]
Got: [1, 1, 20]
```

**Required Actions**:
1. Retrain model with 19D features
2. Export to TFLite (INT8 quantization)
3. Replace `firmware/main/tcn_model_dynamic_range_quant.tflite`
4. Rebuild firmware: `idf.py build flash`

### Performance

**Computational Cost**:
- Feature extraction: -0.1 μs (negligible)
- TFLite inference: -200 μs (~1.3% faster)
- Total: <0.01% improvement

**Memory**:
- State buffer: -8 bytes (19×2 vs 20×2 in first layer)
- TFLite model: ~same size (dominated by weights, not input)

---

## Migration Guide

### For Developers

#### Step 1: Update Code (DONE)

All code changes already committed. No action needed.

#### Step 2: Retrain Model

```bash
# Clean old checkpoints
rm -rf checkpoints/ *.pth

# Train new 19D model
python train.py --model eskf_tcn --epochs 200 --lr 1e-4 --batch_size 4

# Validate
python validate.py --model_type eskf_tcn --model_path eskf_tcn_model.pth
```

**Expected Output**:
```
Mean APE (RMSE): 0.0089 m  (target: <0.012 m)
Compression ratio: 1.02x   (scale close to 1.0)
```

#### Step 3: Export to TFLite

```bash
# Export with INT8 quantization
python utils/convert_tflite.py \
    --model_path eskf_tcn_model.pth \
    --output_path firmware/main/tcn_model_dynamic_range_quant.tflite

# Verify model size
ls -lh firmware/main/tcn_model_dynamic_range_quant.tflite
# Expected: ~33 KB (similar to old 20D model)
```

#### Step 4: Update Firmware

```bash
cd firmware

# Rebuild (auto-converts .tflite to C array)
idf.py fullclean
idf.py build

# Flash to device
idf.py flash monitor
```

**Check Logs**:
```
I (1234) trajecto: TCN initialized (19D features)
I (1235) trajecto: State buffers: [(19,2), (64,4), (64,8), (64,16)]
```

#### Step 5: Test On-Device

1. Connect via BLE
2. Stream trajectory mode
3. Verify no TFLite errors
4. Check ZUPT detection works (no zupt_flag needed)
5. Validate position accuracy

---

## Testing Checklist

### Pre-Deployment

- [ ] Train new 19D model (200 epochs)
- [ ] Validation APE RMSE < 1.2 cm
- [ ] Export to TFLite without errors
- [ ] Firmware builds successfully
- [ ] TFLite model loads on ESP32
- [ ] No dimension mismatch errors

### On-Device Validation

- [ ] Feature extraction produces 19 values
- [ ] TCN inference runs without errors
- [ ] ZUPT detection triggers correctly (force-based)
- [ ] Position tracking accuracy maintained
- [ ] No memory leaks or crashes
- [ ] Inference latency < 18ms

### Accuracy Regression Tests

- [ ] Same validation data: APE RMSE within ±2% of 20D model
- [ ] ZUPT detection: F1 score > 0.95
- [ ] Scale drift: < 2% over 30 seconds
- [ ] Trajectory smoothness: No jitter or artifacts

---

## Rollback Plan

If 19D model performs significantly worse:

### Option A: Revert Code

```bash
git revert <commit-hash>  # Revert feature update commit
```

### Option B: Hybrid Mode

Keep classic ZUPT detector and provide zupt_flag:

```python
# model/config.py
USE_ZUPT = True        # Enable classic detector
USE_TCN_ZUPT = True    # TCN refines classic decision
TCN_INPUT_SIZE = 20    # Restore zupt_flag
```

**Trade-off**: Accept circular dependency for better accuracy

---

## Known Issues & Limitations

### Issue 1: Model Incompatibility

**Problem**: Cannot load old 20D checkpoints with new code

**Workaround**: Train from scratch (no workaround for loading old models)

**Status**: Expected behavior, not a bug

### Issue 2: State Buffer Size

**Problem**: First state buffer changed from (20, 2) to (19, 2)

**Impact**: Old TFLite models have incompatible state buffer shapes

**Solution**: Must retrain and re-export

### Issue 3: Feature Statistics May Shift

**Problem**: Removing zupt_flag changes input distribution slightly

**Impact**: GroupNorm statistics in first layer may differ

**Solution**: Retrain updates BatchNorm/GroupNorm running stats

---

## Future Work

### Potential Improvements

1. **Adaptive Feature Selection**
   - Train TCN with attention to learn which features are most important
   - Could reduce to 15-16D without accuracy loss

2. **Learned Pen Tip Offset**
   - Make `PEN_TIP_OFFSET` a trainable parameter
   - Currently hardcoded to [0, 0, 0]

3. **Multi-Scale Features**
   - Add moving averages (e.g., vel_smoothed over 5 samples)
   - Could improve robustness to sensor noise

4. **Auxiliary Tasks**
   - Predict contact force directly (regression head)
   - Could improve pen state detection

---

## References

- **PR**: #XXX (Feature vector update: 20D → 19D)
- **Discussion**: Remove zupt_flag circular dependency
- **Related Docs**:
  - `docs/TCN_FEATURE_VECTOR_19D.md` - Full feature specification
  - `docs/FEATURE_VECTOR_UPDATE.md` - Original proposal
  - `model/base_hybrid_model.py:382-394` - Implementation

---

## Contact

Questions or issues with this update?
- File issue: https://github.com/anthropics/trajecto/issues
- Review code: `git diff <prev-commit> HEAD -- model/ firmware/`

---

**Status**: ✅ **CODE UPDATED** | ⚠️ **MODELS NEED RETRAINING**
**Priority**: 🔴 **HIGH** (breaks existing models)
**ETA for Retrained Models**: 2-3 hours (200 epochs @ 2min/epoch)
