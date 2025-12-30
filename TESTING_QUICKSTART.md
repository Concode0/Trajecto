# Testing Quick Start Guide

## 🚀 Get Testing in 5 Minutes

### Step 1: Install Test Dependencies

```bash
# Make sure you're in the Trajecto directory
cd /Users/haro/works/Trajecto

# Install pytest and dependencies
uv add --dev pytest pytest-cov
```

### Step 2: Run Your First Test

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run just ESKF tests
pytest tests/test_eskf.py

# Run a specific test
pytest tests/test_eskf.py::TestESKFInitialization::test_initial_state_is_zero
```

### Step 3: See the Results

You should see output like:
```
============================================ test session starts =============================================
platform darwin -- Python 3.11.x, pytest-x.x.x
collected 15 items

tests/test_eskf.py::TestESKFInitialization::test_initial_state_is_zero PASSED                        [  6%]
tests/test_eskf.py::TestESKFInitialization::test_initial_quaternion_is_identity PASSED              [ 13%]
tests/test_eskf.py::TestESKFPredict::test_stationary_no_drift PASSED                                [ 20%]
...

========================================== 15 passed in 2.45s =============================================
```

---

## 📊 Common Test Commands

### Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=model --cov=utils --cov-report=html
# Then open: htmlcov/index.html

# Only fast tests (skip slow ones)
pytest -m "not slow"

# Run tests in parallel (faster!)
pip install pytest-xdist
pytest -n auto

# Stop at first failure
pytest -x

# Run last failed tests
pytest --lf

# Show print statements
pytest -s
```

### Watching for Changes

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests when files change
ptw
```

### Writing Your First Test

**Example: Test a new feature**

1. **Create test file**: `tests/test_myfeature.py`

```python
import pytest
from model.my_module import my_function

def test_my_function_basic():
    """Test basic functionality"""
    result = my_function(input_data=5)
    assert result == 10  # Expected output
```

2. **Run it**:
```bash
pytest tests/test_myfeature.py -v
```

3. **See it fail** (if function doesn't exist yet - that's OK!)

4. **Implement the function** in `model/my_module.py`

5. **Run again** - it should pass! ✅

---

## 🔧 Firmware Tests

### Running Firmware Tests

```bash
cd firmware

# Build firmware with tests
idf.py build

# Flash to ESP32 and monitor output
idf.py flash monitor

# You'll see test results in the serial output
```

### Expected Output

```
========================================
  ESKF Unit Tests (Firmware)
========================================

test_eskf_initialization:PASS
test_quaternion_normalization:PASS
test_stationary_no_drift:PASS
...

-----------------------
15 Tests 0 Failures 0 Ignored
OK
========================================
  Tests Complete!
========================================
```

---

## 🎯 Test-Driven Development (TDD) Workflow

### Example: Adding a new ESKF feature

1. **Write the test first** (it will fail):
```python
# tests/test_eskf.py
def test_eskf_resets_state():
    eskf = ESKF()
    # Add some state
    eskf.pos = torch.tensor([[1.0, 2.0, 3.0]])

    # Reset should zero everything
    eskf.reset()

    assert torch.allclose(eskf.pos, torch.zeros(1, 3))
```

2. **Run the test** (it fails):
```bash
$ pytest tests/test_eskf.py::test_eskf_resets_state
FAILED - AttributeError: 'ESKF' object has no attribute 'reset'
```

3. **Implement the feature**:
```python
# model/ESKF.py
class ESKF:
    def reset(self):
        """Reset filter to initial state"""
        self.pos = torch.zeros(1, 3)
        self.vel = torch.zeros(1, 3)
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # ... reset other state
```

4. **Run the test again**:
```bash
$ pytest tests/test_eskf.py::test_eskf_resets_state
PASSED ✓
```

5. **Success!** 🎉 You've just done TDD!

---

## 📈 Checking Code Coverage

```bash
# Run tests with coverage
pytest --cov=model --cov=utils --cov-report=html --cov-report=term

# Output shows coverage percentages:
# model/ESKF.py         85%
# model/TCN.py          72%
# utils/acquire.py      45%   <- Needs more tests!

# Open detailed HTML report
open htmlcov/index.html
```

The HTML report shows:
- Which lines are covered (green)
- Which lines are NOT covered (red)
- Helps you find untested code

---

## 🐛 Debugging Failed Tests

### Option 1: Add print statements

```python
def test_my_feature():
    result = my_function(5)
    print(f"Result: {result}")  # This prints during test
    assert result == 10
```

Run with `-s` to see prints:
```bash
pytest tests/test_myfeature.py -s
```

### Option 2: Use pytest's built-in debugger

```python
def test_my_feature():
    result = my_function(5)

    import pdb; pdb.set_trace()  # Stops execution here

    assert result == 10
```

### Option 3: Use VS Code debugger

1. Set breakpoint in test file (click left of line number)
2. Click "Debug Test" in VS Code
3. Step through code!

---

## 🎓 Next Steps

### Priority 1: Critical Tests (Do These First!)

1. **ESKF Tests** ✅ (Already created!)
   ```bash
   pytest tests/test_eskf.py
   ```

2. **Dataset Tests** ✅ (Already created!)
   ```bash
   pytest tests/test_dataset.py
   ```

3. **Firmware ESKF Tests** ✅ (Already created!)
   ```bash
   cd firmware && idf.py build flash monitor
   ```

### Priority 2: Add More Tests

4. **TCN Tests** (Create `tests/test_tcn.py`):
   - Test forward pass
   - Test output shapes
   - Test with different sequence lengths

5. **Preprocessing Tests** (Create `tests/test_preprocessing.py`):
   - Test two-tap synchronization
   - Test data segmentation
   - Test coordinate transforms

6. **Calibration Tests** (Create `tests/test_calibration.py`):
   - Test NVS save/load
   - Mock BMI270 responses
   - Test CRT/FOC logic

### Priority 3: Integration Tests

7. **Python-C++ Parity Test** (Create `tests/test_parity.py`):
   - Compare Python ESKF vs C++ ESKF outputs
   - Run same input through both
   - Assert outputs match within tolerance

---

## 📚 Resources

- **Pytest Docs**: https://docs.pytest.org
- **Testing Guide**: See `TESTING_GUIDE.md` for detailed explanations
- **Ask for Help**: "How do I test X?" - I can provide specific examples!

---

## ✅ Checklist: Am I Testing Correctly?

- [ ] Tests are fast (< 1 second each)
- [ ] Tests are independent (can run in any order)
- [ ] Test names describe what they test
- [ ] Each test checks ONE thing
- [ ] Tests use assertions (`assert ...`)
- [ ] I run tests before committing code
- [ ] Failed tests are investigated immediately
- [ ] Coverage is gradually increasing

---

## 🎉 You're Ready to Test!

**Try it now:**

```bash
# Run the ESKF tests
pytest tests/test_eskf.py -v

# See which tests pass/fail
# Fix any failures
# Add more tests!
```

**Remember**: Tests are your safety net. They catch bugs before users do! 🐛🛡️
