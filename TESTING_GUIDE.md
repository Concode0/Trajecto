# Trajecto Testing Guide

## Table of Contents
1. [Testing Basics](#testing-basics)
2. [Test Types](#test-types)
3. [Python Testing](#python-testing)
4. [Firmware Testing](#firmware-testing)
5. [Integration Testing](#integration-testing)
6. [Running Tests](#running-tests)

---

## Testing Basics

### What is Testing?

Testing is writing code that automatically checks if your main code works correctly. Instead of manually testing every time you make a change, tests run automatically and tell you immediately if something breaks.

### Why Test?

**For Trajecto specifically:**
- **Math-heavy code**: ESKF quaternion operations, matrix multiplications can have subtle bugs
- **Python-C++ sync**: Ensure firmware ESKF matches Python ESKF
- **Calibration logic**: Verify NVS save/load, CRT/FOC sequences
- **Model accuracy**: Check TCN output quality over time
- **Regression prevention**: Catch bugs before they reach production

### Test-Driven Development (TDD) - Optional but Recommended

1. **Write the test first** (it will fail - that's expected!)
2. **Write minimal code** to make the test pass
3. **Refactor** your code while tests ensure it still works

---

## Test Types

### 1. **Unit Tests**
Test individual functions/classes in isolation.

**Example**: Test quaternion multiplication without initializing the whole ESKF system.

```cpp
// Test that quaternion multiplication is correct
TEST_CASE("Quaternion multiplication", "[quaternion]") {
    Quaternionf q1(1, 0, 0, 0);  // Identity
    Quaternionf q2(0.707, 0.707, 0, 0);  // 90° around X
    Quaternionf result = q1 * q2;

    REQUIRE(result.w() == Approx(0.707));
    REQUIRE(result.x() == Approx(0.707));
}
```

### 2. **Integration Tests**
Test how multiple components work together.

**Example**: Test that ESKF + TCN pipeline produces reasonable trajectory.

### 3. **End-to-End Tests**
Test the complete system from sensor input to BLE output.

### 4. **Parity Tests**
Verify Python implementation matches C++ implementation.

**Critical for Trajecto**: Ensure firmware ESKF produces same results as training ESKF.

---

## Python Testing

### Setup

```bash
# Install pytest
uv add --dev pytest pytest-cov numpy

# Run all tests
pytest

# Run with coverage report
pytest --cov=model --cov-report=html
```

### Directory Structure

```
Trajecto/
├── tests/                    # Python tests
│   ├── __init__.py
│   ├── test_eskf.py         # ESKF unit tests
│   ├── test_tcn.py          # TCN unit tests
│   ├── test_dataset.py      # Dataset tests
│   ├── test_preprocessing.py
│   └── test_parity.py       # Python-C++ parity tests
├── model/
└── utils/
```

### Example: Testing ESKF Predict Step

**tests/test_eskf.py**
```python
import pytest
import torch
from model.ESKF import ESKF
from model.config import Config

class TestESKF:
    @pytest.fixture
    def eskf(self):
        """Create ESKF instance for testing"""
        return ESKF()

    def test_predict_stationary(self, eskf):
        """Test that ESKF doesn't drift when stationary"""
        batch_size = 1

        # Zero motion input (stationary)
        accel = torch.tensor([[0.0, 0.0, 9.81]])  # Just gravity
        gyro = torch.zeros(batch_size, 3)

        # Run 100 prediction steps
        for _ in range(100):
            eskf.predict(accel, gyro)

        # Position should not drift significantly
        assert torch.abs(eskf.pos).max() < 0.01, "Position drifted on stationary input"

    def test_gravity_alignment(self, eskf):
        """Test that gravity is correctly aligned to Z-axis"""
        accel = torch.tensor([[0.0, 0.0, 9.81]])
        gyro = torch.zeros(1, 3)

        # Initial gravity should point down in body frame
        eskf.predict(accel, gyro)

        # After alignment, world Z acceleration should be ~0
        accel_world = eskf.quat_rotate(accel)
        assert torch.abs(accel_world[0, 2] - 9.81) < 0.1

    def test_quaternion_normalization(self, eskf):
        """Test that quaternion stays normalized after many updates"""
        accel = torch.randn(1, 3) * 0.1 + torch.tensor([[0, 0, 9.81]])
        gyro = torch.randn(1, 3) * 0.1

        for _ in range(1000):
            eskf.predict(accel, gyro)

            # Quaternion should remain unit quaternion
            quat_norm = torch.norm(eskf.quat, dim=1)
            assert torch.abs(quat_norm - 1.0) < 1e-5, f"Quaternion not normalized: {quat_norm}"

    def test_velocity_integration(self, eskf):
        """Test that constant acceleration integrates to linear velocity"""
        dt = Config.DT
        accel_body = torch.tensor([[1.0, 0.0, 9.81]])  # 1 m/s² forward + gravity
        gyro = torch.zeros(1, 3)

        # Initialize with known orientation (identity = aligned with world)
        eskf.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Integrate for 1 second (50 steps @ 50Hz)
        for _ in range(50):
            eskf.predict(accel_body, gyro)

        # Velocity in X direction should be ~1 m/s (after 1 second)
        # (Gravity in Z should cancel out in world frame)
        expected_vel_x = 1.0 * 1.0  # accel * time
        assert torch.abs(eskf.vel[0, 0] - expected_vel_x) < 0.1


class TestESKFUpdate:
    @pytest.fixture
    def eskf(self):
        return ESKF()

    def test_zupt_update(self, eskf):
        """Test Zero-velocity update reduces velocity uncertainty"""
        # Add some velocity
        eskf.vel = torch.tensor([[1.0, 1.0, 0.0]])

        # Apply ZUPT (should force velocity to zero)
        eskf.update_zupt()

        # Velocity should be close to zero
        assert torch.norm(eskf.vel) < 0.01, "ZUPT didn't zero velocity"
```

### Example: Testing Data Preprocessing

**tests/test_preprocessing.py**
```python
import pytest
import numpy as np
from utils.acquire import estimate_time_alignment_two_taps

class TestSynchronization:
    def test_two_tap_correlation(self):
        """Test that two-tap sync finds correct alignment"""
        # Create synthetic data
        dt = 0.02  # 50 Hz
        time_pen = np.arange(0, 10, dt)
        time_imu = np.arange(0, 10, dt) + 0.05  # 50ms offset

        # Create tap signals (impulses at t=1s and t=9s)
        force_signal = np.zeros_like(time_pen)
        force_signal[np.abs(time_pen - 1.0) < dt] = 1.0
        force_signal[np.abs(time_pen - 9.0) < dt] = 1.0

        accel_signal = np.zeros_like(time_imu)
        accel_signal[np.abs(time_imu - 1.0) < dt] = 10.0
        accel_signal[np.abs(time_imu - 9.0) < dt] = 10.0

        # Estimate alignment
        slope, intercept = estimate_time_alignment_two_taps(
            time_pen, force_signal, time_imu, accel_signal
        )

        # Should find ~50ms offset
        assert abs(intercept - 0.05) < 0.01, f"Expected 50ms offset, got {intercept*1000}ms"
        assert abs(slope - 1.0) < 0.01, "Clock drift should be minimal"
```

---

## Firmware Testing

ESP-IDF supports two types of tests:

### 1. **Host Tests** (Run on PC - Faster!)

Tests run on your computer using gcc/clang, not on ESP32. Much faster for development.

**Setup Directory Structure:**
```
firmware/
├── components/
│   └── trajecto_core/
│       ├── src/
│       ├── include/
│       └── test/              # Host tests
│           ├── test_eskf.cpp
│           ├── test_quaternion.cpp
│           └── CMakeLists.txt
├── main/
└── test/                       # Target tests (run on ESP32)
```

### 2. **Target Tests** (Run on ESP32)

Tests that actually run on the device. Needed for hardware-specific code (I2C, BLE, etc.).

### Creating Your First Firmware Test

**Step 1: Create test directory**

```bash
cd firmware/components/trajecto_core
mkdir -p test
```

**Step 2: Create test file**

**firmware/components/trajecto_core/test/test_eskf.cpp**
```cpp
#include <cmath>
#include "unity.h"
#include "eskf.hpp"

// Unity test framework macros:
// TEST_ASSERT_EQUAL(expected, actual)
// TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual)
// TEST_ASSERT_TRUE(condition)

void setUp(void) {
    // Runs before each test
}

void tearDown(void) {
    // Runs after each test
}

// Test that ESKF initializes correctly
void test_eskf_initialization(void) {
    trajecto::ESKF eskf;

    // Position should start at origin
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, eskf.get_state().pos.x());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, eskf.get_state().pos.y());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, eskf.get_state().pos.z());

    // Quaternion should be identity (w=1, x=y=z=0)
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, eskf.get_state().quat.w());
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, eskf.get_state().quat.x());
}

// Test quaternion normalization
void test_quaternion_normalization(void) {
    trajecto::ESKF eskf;

    // Simulate 1000 prediction steps
    Eigen::Vector3f accel(0.1f, 0.1f, 9.81f);
    Eigen::Vector3f gyro(0.01f, 0.01f, 0.01f);

    for (int i = 0; i < 1000; i++) {
        eskf.predict(accel, gyro);
    }

    // Check quaternion is still normalized
    auto quat = eskf.get_state().quat;
    float norm = std::sqrt(quat.w()*quat.w() + quat.x()*quat.x() +
                          quat.y()*quat.y() + quat.z()*quat.z());

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, norm);
}

// Test stationary case (should not drift)
void test_stationary_no_drift(void) {
    trajecto::ESKF eskf;

    // Zero motion (only gravity)
    Eigen::Vector3f accel(0.0f, 0.0f, 9.81f);
    Eigen::Vector3f gyro(0.0f, 0.0f, 0.0f);

    // Run for 5 seconds (250 steps @ 50Hz)
    for (int i = 0; i < 250; i++) {
        eskf.predict(accel, gyro);
    }

    // Position drift should be minimal
    auto pos = eskf.get_state().pos;
    float drift = std::sqrt(pos.x()*pos.x() + pos.y()*pos.y() + pos.z()*pos.z());

    TEST_ASSERT_LESS_THAN(0.01f, drift); // Less than 1cm drift
}

// Test ZUPT update
void test_zupt_reduces_velocity(void) {
    trajecto::ESKF eskf;

    // Add some velocity
    Eigen::Vector3f vel(1.0f, 1.0f, 0.5f);
    // Note: You'll need to add a setter or make this testable
    // eskf.set_velocity(vel);  // Add this method to ESKF for testing

    // Apply ZUPT
    eskf.update_zupt();

    // Velocity should be reduced significantly
    auto state = eskf.get_state();
    float vel_magnitude = state.vel.norm();

    TEST_ASSERT_LESS_THAN(0.1f, vel_magnitude);
}

// Main test runner
extern "C" void app_main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_eskf_initialization);
    RUN_TEST(test_quaternion_normalization);
    RUN_TEST(test_stationary_no_drift);
    RUN_TEST(test_zupt_reduces_velocity);

    UNITY_END();
}
```

**Step 3: Create CMakeLists.txt for test**

**firmware/components/trajecto_core/test/CMakeLists.txt**
```cmake
idf_component_register(
    SRC_DIRS "."
    INCLUDE_DIRS "." "../include"
    REQUIRES unity trajecto_core
)
```

**Step 4: Run the test**

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor

# In monitor, tests will run automatically
```

---

## Integration Testing

### Python-C++ Parity Test

This is **CRITICAL** for Trajecto - ensure firmware matches training code.

**tests/test_parity.py**
```python
import pytest
import torch
import numpy as np
import subprocess
import json
from model.ESKF import ESKF as PythonESKF

class TestParity:
    """Test that C++ firmware ESKF matches Python training ESKF"""

    def test_eskf_parity(self):
        """Run same input through Python and C++ ESKF, compare outputs"""

        # Generate test sequence
        np.random.seed(42)
        n_steps = 100
        accel_seq = np.random.randn(n_steps, 3) * 0.5
        accel_seq[:, 2] += 9.81  # Add gravity
        gyro_seq = np.random.randn(n_steps, 3) * 0.1

        # Save test data to file
        test_data = {
            'accel': accel_seq.tolist(),
            'gyro': gyro_seq.tolist()
        }
        with open('/tmp/test_input.json', 'w') as f:
            json.dump(test_data, f)

        # Run Python ESKF
        python_eskf = PythonESKF()
        python_results = []

        for i in range(n_steps):
            accel_t = torch.tensor(accel_seq[i:i+1], dtype=torch.float32)
            gyro_t = torch.tensor(gyro_seq[i:i+1], dtype=torch.float32)
            python_eskf.predict(accel_t, gyro_t)

            python_results.append({
                'pos': python_eskf.pos[0].numpy().tolist(),
                'vel': python_eskf.vel[0].numpy().tolist(),
                'quat': python_eskf.quat[0].numpy().tolist()
            })

        # Run C++ ESKF (you'll need to create a test binary)
        # firmware/test/eskf_parity_test
        result = subprocess.run(
            ['firmware/build/test/eskf_parity_test', '/tmp/test_input.json'],
            capture_output=True,
            text=True
        )

        cpp_results = json.loads(result.stdout)

        # Compare results
        for i, (py_res, cpp_res) in enumerate(zip(python_results, cpp_results)):
            # Position should match within 1mm
            pos_error = np.linalg.norm(
                np.array(py_res['pos']) - np.array(cpp_res['pos'])
            )
            assert pos_error < 0.001, f"Position mismatch at step {i}: {pos_error}m"

            # Velocity within 1cm/s
            vel_error = np.linalg.norm(
                np.array(py_res['vel']) - np.array(cpp_res['vel'])
            )
            assert vel_error < 0.01, f"Velocity mismatch at step {i}: {vel_error}m/s"

            # Quaternion within 0.01 rad (~0.6 degrees)
            quat_error = 1.0 - abs(np.dot(py_res['quat'], cpp_res['quat']))
            assert quat_error < 0.01, f"Quaternion mismatch at step {i}: {quat_error}"
```

---

## Running Tests

### Python Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_eskf.py

# Run specific test
pytest tests/test_eskf.py::TestESKF::test_predict_stationary

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=model --cov=utils --cov-report=html
# Open htmlcov/index.html to see coverage

# Run in watch mode (re-run on file changes)
pip install pytest-watch
ptw
```

### Firmware Tests

```bash
# Build and flash tests
cd firmware
idf.py build flash monitor

# For host tests (if configured)
idf.py build
./build/test_eskf.elf
```

### Continuous Integration (GitHub Actions)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest --cov=model --cov=utils --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  firmware-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup ESP-IDF
        uses: espressif/esp-idf-ci-action@v1
      - name: Build tests
        run: |
          cd firmware
          idf.py build
```

---

## Best Practices

### 1. **Test File Naming**
- Python: `test_*.py` or `*_test.py`
- C++: `test_*.cpp`

### 2. **Test Function Naming**
- Be descriptive: `test_quaternion_multiplication_identity()`
- Not just: `test1()`

### 3. **Arrange-Act-Assert Pattern**
```python
def test_example():
    # Arrange: Set up test data
    eskf = ESKF()
    accel = torch.tensor([[0, 0, 9.81]])

    # Act: Execute the code under test
    eskf.predict(accel, gyro)

    # Assert: Check the results
    assert eskf.pos[0, 2] < 0.01
```

### 4. **Use Fixtures for Common Setup**
```python
@pytest.fixture
def initialized_eskf():
    eskf = ESKF()
    # ... do initialization ...
    return eskf

def test_something(initialized_eskf):
    # Use the fixture
    initialized_eskf.predict(...)
```

### 5. **Test One Thing Per Test**
Bad:
```python
def test_everything():
    test_init()
    test_predict()
    test_update()
```

Good:
```python
def test_init():
    # ...

def test_predict():
    # ...
```

### 6. **Don't Test External Libraries**
Don't test that PyTorch works. Test that YOUR code works.

---

## What to Test First (Priority Order)

### High Priority (Do These First!)
1. ✅ **ESKF predict step** - Core algorithm
2. ✅ **Quaternion operations** - Easy to get wrong
3. ✅ **Python-C++ parity** - Critical for Sim2Real
4. ✅ **Calibration NVS save/load** - Data loss prevention
5. ✅ **Data synchronization** - Two-tap correlation

### Medium Priority
6. TCN forward pass (basic smoke test)
7. Dataset loading and augmentation
8. BLE protocol packet serialization
9. ZUPT detection logic

### Low Priority (Nice to Have)
10. UI interactions (iOS app)
11. Visualization code
12. CLI argument parsing

---

## Example Test Session

```bash
# 1. Write a test (it fails - expected!)
$ cat > tests/test_eskf.py
def test_gravity():
    eskf = ESKF()
    assert eskf.gravity == 9.81  # Fails: attribute doesn't exist

# 2. Run the test
$ pytest tests/test_eskf.py -v
FAILED - AttributeError: 'ESKF' object has no attribute 'gravity'

# 3. Fix the code
$ vim model/ESKF.py
# Add: self.gravity = 9.81

# 4. Run test again
$ pytest tests/test_eskf.py -v
PASSED ✓

# 5. Add more tests...
```

---

## Getting Help

- **pytest docs**: https://docs.pytest.org
- **Unity (ESP-IDF)**: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/unit-tests.html
- **Ask**: "How do I test X?" - I can help with specific examples!

---

## Next Steps

1. **Start small**: Pick ONE function to test (e.g., quaternion multiplication)
2. **Write the test**: Follow examples above
3. **Run it**: See it pass ✓
4. **Gradually add more**: Build up your test suite
5. **Run before commits**: Make it a habit

**Remember**: Tests are insurance. They take time upfront but save MUCH more time debugging later!
