# Julia Analyzer Usage Guide

## Quick Start

### Single Model Analysis

1. Edit `analyzer/main.jl` configuration:

```julia
# Line 21: Disable comparison mode
const COMPARE_MODE = false

# Line 32: Select model type
const MODEL_TYPE = "pure_eskf"  # Options: "pure_integration", "pure_eskf", "eskf", "aekf", "tcn"

# Line 33: Set model path (empty for baselines)
const MODEL_PATH = ""  # For trained models: "eskf_tcn_model.pth"

# Line 36: Choose sample
const SAMPLE_ID = "sample_002_seg0"
```

2. Run:

```bash
cd analyzer
julia --project=. main.jl
```

### Multi-Model Comparison

1. Edit `analyzer/main.jl` configuration:

```julia
# Line 21: Enable comparison mode
const COMPARE_MODE = true

# Lines 25-29: Configure models to compare
const MODELS_TO_COMPARE = [
    ("Pure Integration", "pure_integration", ""),
    ("Pure ESKF", "pure_eskf", ""),
    ("ESKF-TCN", "eskf", joinpath(PROJECT_ROOT, "eskf_tcn_model.pth")),
]
```

2. Run:

```bash
cd analyzer
julia --project=. main.jl
```

This will:
- Load data once
- Run all models
- Display side-by-side comparison in a single window

## Available Models

### Baseline Models (No Training Required)

| Model Type | Description | Model Path |
|------------|-------------|------------|
| `pure_integration` | Simple dead reckoning | `""` |
| `pure_eskf` | ESKF without TCN | `""` |

### Trained Models (Require .pth File)

| Model Type | Description | Model Path Example |
|------------|-------------|-------------------|
| `eskf` | ESKF-TCN hybrid | `"eskf_tcn_model.pth"` |
| `aekf` | AEKF-TCN variant | `"aekf_tcn_model.pth"` |
| `tcn` | TCN-only | `"only_tcn_model.pth"` |

## Comparison Dashboard Features

When running in comparison mode, you'll see:

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  3D Trajectory View (All Models + Ground Truth)        │
│  - Color-coded trajectories                             │
│  - Metrics in title (APE, Err/Dist for each model)     │
├──────────────┬──────────────┬──────────────────────────┤
│  Model 1     │  Model 2     │  Model 3                 │
│  XY View     │  XY View     │  XY View                 │
├──────────────┴──────────────┴──────────────────────────┤
│  Error Comparison (All Models)                         │
│  - Line plot showing error over time                   │
├─────────────────────────────────────────────────────────┤
│  Controls: [Slider] Frame: N  [Play/Pause]            │
└─────────────────────────────────────────────────────────┘
```

### Interactive Controls

- **Slider**: Scrub through time
- **Play/Pause**: Auto-playback at 50 FPS
- **3D View**: Rotate, zoom, pan
- **Dynamic Markers**: Follow current frame position

### Color Scheme

- **Blue**: Ground truth
- **Red**: First model (Pure Integration)
- **Orange**: Second model (Pure ESKF)
- **Purple**: Third model (ESKF-TCN)
- **Cyan/Magenta/Yellow**: Additional models

## Example Configurations

### Compare All Baselines

```julia
const COMPARE_MODE = true
const MODELS_TO_COMPARE = [
    ("Pure Integration", "pure_integration", ""),
    ("Pure ESKF", "pure_eskf", ""),
]
```

### Compare ESKF Variants

```julia
const COMPARE_MODE = true
const MODELS_TO_COMPARE = [
    ("Pure ESKF", "pure_eskf", ""),
    ("ESKF-TCN", "eskf", joinpath(PROJECT_ROOT, "eskf_tcn_model.pth")),
    ("AEKF-TCN", "aekf", joinpath(PROJECT_ROOT, "aekf_tcn_model.pth")),
]
```

### Single Model Deep Dive

```julia
const COMPARE_MODE = false
const MODEL_TYPE = "eskf"
const MODEL_PATH = joinpath(PROJECT_ROOT, "eskf_tcn_model.pth")
```

## Dataset Selection

### Using Validation Set

```julia
h5_path = joinpath(PROJECT_ROOT, "data/validation_dataset.h5")
```

### Using Training Set

```julia
h5_path = joinpath(PROJECT_ROOT, "data/dataset.h5")
```

### Available Samples

List available samples:

```bash
h5dump -n data/validation_dataset.h5
```

Common samples:
- `sample_001_seg0`
- `sample_002_seg0`
- `sample_003_seg0`
- etc.

## Metrics Explained

### APE (Absolute Pose Error)

- RMSE after Sim(3) alignment
- Units: cm
- **Lower is better**
- Typical values:
  - Pure Integration: >1000 cm
  - Pure ESKF: 3-5 cm
  - ESKF-TCN: 0.8-1.2 cm

### Err/Dist (Error over Distance)

- Normalized error by path length
- Units: %
- **Lower is better**
- Typical values:
  - Pure ESKF: 2-4%
  - ESKF-TCN: <1%

### Drift

- Euclidean distance between start and end positions
- Units: mm
- **Lower is better**
- Typical values:
  - Pure Integration: >10,000 mm
  - Pure ESKF: 1,000-5,000 mm
  - ESKF-TCN: <50 mm

## Troubleshooting

### Problem: Julia can't find Python modules

**Solution**: Verify VENV_PATH points to correct Python environment:

```julia
const VENV_PATH = joinpath(PROJECT_ROOT, ".venv", "bin", "python")
```

### Problem: Model fails to load

**Check**:
1. Model type spelling (case-sensitive)
2. .pth file exists for trained models
3. Model was trained with same architecture

### Problem: Comparison window doesn't show all models

**Solution**: One or more models failed. Check terminal output for errors.

### Problem: Performance is slow

**Optimize**:
1. Reduce sample length (use shorter sequences)
2. Close other Julia processes
3. Use fewer models in comparison

## Advanced Usage

### Custom Model Configuration

Add your own model to comparison:

1. Train model: `python train.py --model your_model`
2. Add to Julia loader (`PyTorchEstimator.jl`)
3. Update `MODELS_TO_COMPARE` in `main.jl`

### Batch Analysis

Run multiple samples:

```julia
samples = ["sample_001_seg0", "sample_002_seg0", "sample_003_seg0"]

for sample in samples
    println("Analyzing $sample...")
    # Run analysis...
end
```

### Export Results

Save metrics to file:

```julia
open("results.txt", "w") do io
    for result in model_metrics
        println(io, "$(result.name): APE=$(result.metrics.ape_rmse)")
    end
end
```

## Performance Tips

- **First run**: Slower due to compilation (Julia JIT)
- **Subsequent runs**: Much faster (~10x)
- **Large datasets**: Consider subsampling or shorter sequences
- **Multiple models**: Run in parallel (not currently supported, opens sequentially)

## Keyboard Shortcuts (GLMakie)

- **Left Mouse**: Rotate 3D view
- **Right Mouse**: Pan
- **Scroll**: Zoom
- **ESC**: Close window

## Next Steps

1. Try comparing baselines to understand drift
2. Validate your trained ESKF-TCN model
3. Analyze failure cases (high APE samples)
4. Experiment with different ZUPT thresholds (config.py)
5. Visualize covariance ellipsoids (uncertainty quantification)

## License

GNU General Public License v3.0
