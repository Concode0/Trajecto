# Parallel Scan Experiment for Trajecto

This experiment explores using parallel scan methods to accelerate training of the Trajecto ESKF-TCN model.

## Background

### Why Traditional ESKF Cannot Use Parallel Scan

The Error-State Kalman Filter (ESKF) used in Trajecto is inherently sequential because:

1. **State-dependent dynamics**: `x_{t+1} = f(x_t, u_t)` where `f` is nonlinear
2. **Quaternion multiplication**: Non-commutative operations for orientation
3. **Jacobian computation**: Depends on current state
4. **Cholesky decomposition**: Sequential matrix operations

### Where Parallel Scan CAN Help

Parallel scan (prefix sum) can accelerate computations with **associative binary operators**:

1. **Linear State Space Models (SSM)**: `x_{t+1} = Ax_t + Bu_t` can be reformulated as:
   ```
   (A_2, b_2) * (A_1, b_1) = (A_2 @ A_1, A_2 @ b_1 + b_2)
   ```
   This operator is associative, enabling parallel computation!

2. **S4/S5/Mamba-style models**: Modern sequence models use structured state spaces that are fully parallelizable during training.

## Module Overview

### `scan_ops.py` - Core Parallel Scan Primitives

- `sequential_scan()`: Baseline sequential implementation
- `parallel_scan()`: Work-efficient Blelloch algorithm
- `_parallel_scan_gpu()`: GPU-optimized iterative doubling
- `associative_scan()`: Generic scan for any associative operator
- `discretize_continuous_ssm()`: ZOH/Euler discretization

### `linear_ssm.py` - Linear State Space Models

- `S4DKernel`: Diagonal state space (S4D) with HiPPO initialization
  - Convolution view for training efficiency
  - FFT-based convolution for long sequences
  - Recurrent mode for inference

- `LinearSSM`: Multi-layer SSM with input/output projections
  - Drop-in replacement for TCN layers
  - Optional bidirectional mode

- `S5Layer`: Simplified State Space (S5) layer
  - Real-valued diagonal state matrix
  - Direct parallel scan (no FFT)

### `ssm_filter.py` - SSM-Based Filter

- `SSMFilter`: Complete filter replacement for ESKF
  - Multi-layer S5 backbone
  - Output heads for position, velocity, orientation
  - Uncertainty estimation (log variance)
  - Both parallel (training) and recurrent (inference) modes

- `HybridSSMESKF`: Hybrid approach
  - SSM for feature extraction (parallel)
  - Physics-based integration for position/velocity (sequential but fast)

- `ParallelIntegrator`: Parallel kinematic integration
  - Uses cumsum tricks for velocity/position integration

### `benchmark.py` - Performance Benchmarks

Comprehensive benchmarks for:
- Sequential vs parallel scan speedup
- SSM layer performance (S4D, S5)
- SSMFilter training throughput
- Integrator comparison

## Quick Start

```bash
# Run all benchmarks
python -m experiments.parallel_scan.benchmark

# Verify correctness only
python -m experiments.parallel_scan.benchmark --verify

# Benchmark specific sequence lengths
python -m experiments.parallel_scan.benchmark --sequence-lengths 500 1000 2000

# Run only scan operation benchmarks
python -m experiments.parallel_scan.benchmark --scan-only
```

## Usage Examples

### Replace TCN with SSM Filter

```python
from experiments.parallel_scan import SSMFilter

# Create SSM filter with same interface as ESKF-TCN output
filter = SSMFilter(
    input_dim=19,      # Same as TCN_INPUT_SIZE
    hidden_dim=64,     # Similar to TCN_CHANNELS
    state_dim=64,      # SSM state dimension
    num_layers=4,      # Same as TCN depth
)

# Training (parallel)
outputs = filter(features)  # [batch, seq_len, input_dim]
position = outputs["position"]  # [batch, seq_len, 3]
velocity = outputs["velocity"]  # [batch, seq_len, 3]

# Inference (recurrent)
filter.init_state(batch_size=1, device="cuda")
for t in range(seq_len):
    output = filter.step(features[:, t])
```

### Use Parallel Scan Directly

```python
from experiments.parallel_scan import parallel_scan

# Linear dynamics: x_{t+1} = A @ x_t + b_t
A_seq = A.unsqueeze(0).expand(batch, seq_len, state_dim, state_dim)
b_seq = B @ inputs  # [batch, seq_len, state_dim]

# Compute all states in parallel
states = parallel_scan(A_seq, b_seq, initial_state)
```

### Parallel Integration

```python
from experiments.parallel_scan.ssm_filter import ParallelIntegrator

integrator = ParallelIntegrator(dt=0.02)
positions, velocities = integrator.forward(
    accelerations,  # [batch, seq_len, 3]
    initial_pos,    # [batch, 3]
    initial_vel,    # [batch, 3]
)
```

## Algorithm Details

### Parallel Scan for Linear Recurrence

For the recurrence `x_{k+1} = A x_k + b_k`, we define tuples `(A_k, b_k)` with the associative operator:

```
(A_2, b_2) * (A_1, b_1) = (A_2 @ A_1, A_2 @ b_1 + b_2)
```

The parallel scan computes:
```
result[t] = (A_t, b_t) * (A_{t-1}, b_{t-1}) * ... * (A_0, b_0)
```

This gives all states with:
- **Work**: O(T * state_dim^2) - same as sequential
- **Depth**: O(log T * state_dim^3) - exponentially better!

### Blelloch Algorithm

The work-efficient parallel scan uses iterative doubling:

```
Iteration 0: stride = 1
  element[1] = op(element[0], element[1])
  element[3] = op(element[2], element[3])
  ...

Iteration 1: stride = 2
  element[3] = op(element[1], element[3])
  element[7] = op(element[5], element[7])
  ...

Iteration 2: stride = 4
  ...
```

After `log(T)` iterations, each element contains the prefix reduction up to that point.

## Expected Performance

| Sequence Length | Sequential (ms) | Parallel (ms) | Speedup |
|----------------|-----------------|---------------|---------|
| 100            | ~1.5            | ~1.2          | 1.2x    |
| 500            | ~7.5            | ~2.0          | 3.7x    |
| 1000           | ~15             | ~2.5          | 6x      |
| 1750           | ~26             | ~3.0          | 8.7x    |

*Note: Actual speedup depends on hardware and batch size. GPU parallelization is most effective for longer sequences.*

## Limitations & Trade-offs

1. **Linear Approximation**: SSM approximates nonlinear ESKF dynamics with a learned linear model. May need larger state dimension to match accuracy.

2. **Quaternion Handling**: Orientation is represented in vector space (rotation vectors) instead of unit quaternions. Singularities may occur for large rotations.

3. **Memory Usage**: Parallel scan requires storing all intermediate states, increasing memory by ~2x.

4. **GPU Requirement**: Full speedup only realized on GPU. CPU falls back to sequential.

5. **Initialization**: S4D uses HiPPO initialization for memory, but S5 uses random. May affect long-range dependencies.

## Integration with Trajecto

To integrate with the main training pipeline:

1. Create a new model class in `model/` that uses `SSMFilter` instead of ESKF
2. Modify `train.py` to support the new model type
3. Compare validation metrics (APE, Error/Distance) between approaches

Example model structure:
```python
class SSM_TCN_Model(nn.Module):
    def __init__(self, config):
        self.ssm_filter = SSMFilter(
            input_dim=config.TCN_INPUT_SIZE,
            hidden_dim=config.TCN_CHANNELS[0],
            state_dim=64,
            num_layers=len(config.TCN_CHANNELS),
        )

    def forward(self, sensor_data, ...):
        # Build features (same as ESKF-TCN)
        features = self._build_features(sensor_data)

        # SSM forward (parallel!)
        outputs = self.ssm_filter(features)

        return outputs["position"], outputs["velocity"]
```

## References

1. Blelloch, G.E. (1990). "Prefix Sums and Their Applications"
2. Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces" (S4)
3. Smith et al. (2023). "Simplified State Space Layers for Sequence Modeling" (S5)
4. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

## File Structure

```
experiments/parallel_scan/
├── __init__.py          # Package exports
├── README.md            # This documentation
├── scan_ops.py          # Core parallel scan primitives
├── linear_ssm.py        # S4D, S5, LinearSSM implementations
├── ssm_filter.py        # Drop-in filter replacement
└── benchmark.py         # Performance benchmarks
```
