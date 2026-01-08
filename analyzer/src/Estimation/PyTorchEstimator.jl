"""
PyTorch Model Estimator for Trajecto Analyzer.

This module provides a unified interface for loading and running PyTorch-based
trajectory estimation models through Julia's PythonCall.

Supported model types:
- ESKF-TCN: Hybrid physics + ML model
- AEKF-TCN: Adaptive EKF variant
- TCN-only: Pure data-driven regression
- Pure ESKF: Physics-only baseline
- Pure Integration: Simple dead reckoning baseline
"""
module PyTorchEstimator

using ..AbstractLayers
using ..Config
using PythonCall
using HDF5

"""
    TrajectoEstimator <: AbstractEstimator

Interface for PyTorch-based trajectory estimation models.

# Fields
- `model_type::String`: Model type identifier (eskf, aekf, tcn, pure_eskf, pure_integration)
- `model_path::String`: Path to .pth model weights file (empty for baseline models)
- `script_path::String`: Path to Python model directory (typically PROJECT_ROOT/model)
- `scaler_path::String`: Path to HDF5 scaler statistics file (mean/std for normalization)
- `engine::Ref{Any}`: Lazy-loaded Python model instance and torch module
- `scaler_stats::Ref{Any}`: Lazy-loaded normalization statistics (mean, std)

# Constructor
```julia
estimator = TrajectoEstimator(model_type, model_path, script_path, scaler_path)
```

Model types are automatically lowercased for consistency.

# Example
```julia
# Load hybrid model
estimator = TrajectoEstimator(
    "eskf",
    "/path/to/eskf_tcn_model.pth",
    "/path/to/model",
    "/path/to/scaler_stats.h5"
)

# Load baseline (no weights required)
estimator = TrajectoEstimator(
    "pure_eskf",
    "",  # Empty for baselines
    "/path/to/model",
    "/path/to/scaler_stats.h5"
)
```

# See Also
- `load_model`: Explicitly load model weights (called automatically on first inference)
- `predict_trajectory`: Run trajectory estimation
"""
struct TrajectoEstimator <: AbstractEstimator
    model_type::String
    model_path::String
    script_path::String
    scaler_path::String
    engine::Ref{Any}
    scaler_stats::Ref{Any}

    function TrajectoEstimator(model_type::String, model_path::String, script_path::String, scaler_path::String)
        new(lowercase(model_type), model_path, script_path, scaler_path, Ref{Any}(nothing), Ref{Any}(nothing))
    end
end

function AbstractLayers.load_model(estimator::TrajectoEstimator)
    if estimator.engine[] !== nothing
        return
    end

    # Load Scaler Stats
    if estimator.scaler_stats[] === nothing
        println(">>> Loading Scaler Stats: ", estimator.scaler_path)
        h5open(estimator.scaler_path, "r") do f
            mean_val = read(f["mean"])
            std_val = read(f["std"])
            estimator.scaler_stats[] = (mean=mean_val, std=std_val)
        end
    end

    sys = pyimport("sys")
    abs_script_path = abspath(estimator.script_path)
    # Add both the model directory and its parent to sys.path
    parent_dir = dirname(abs_script_path)
    if !(abs_script_path in pyconvert(Vector, sys.path))
        sys.path.append(abs_script_path)
    end
    if !(parent_dir in pyconvert(Vector, sys.path))
        sys.path.insert(0, parent_dir)
    end

    torch = pyimport("torch")

    # Validate model type
    if !haskey(Config.MODEL_CONFIGS, estimator.model_type)
        error("Unknown model type: $(estimator.model_type). Supported types: $(keys(Config.MODEL_CONFIGS))")
    end

    # Get model configuration
    model_config = Config.MODEL_CONFIGS[estimator.model_type]

    println(">>> Loading PyTorch Model [$(estimator.model_type)]: $(model_config.description)")
    if model_config.requires_weights
        println("    Model file: ", abspath(estimator.model_path))
    end

    # Import Python module and instantiate model
    model_module = pyimport(model_config.module_path)
    raw_model = getproperty(model_module, model_config.class_name)(device=Config.DEFAULT_DEVICE)

    # Load pre-trained weights if required
    if model_config.requires_weights
        state_dict = torch.load(abspath(estimator.model_path), map_location=Config.DEFAULT_DEVICE)
        raw_model.load_state_dict(state_dict)
    end
    raw_model.eval()

    estimator.engine[] = (model=raw_model, torch=torch)
end

function AbstractLayers.predict_trajectory(estimator::TrajectoEstimator, input_data)
    # input_data is expected to be (Seq, 7) Float32 Matrix
    if estimator.engine[] === nothing
        load_model(estimator)
    end
    
    engine = estimator.engine[]
    stats = estimator.scaler_stats[]
    np = pyimport("numpy")
    
    # 1. Prepare Raw Tensor
    py_imu_np = np.asarray(input_data)
    raw_tensor = engine.torch.from_numpy(py_imu_np).float().unsqueeze(0)
    
    # 2. Prepare Normalized Tensor
    # Standardize: (X - mean) / std
    norm_data = (input_data .- stats.mean') ./ stats.std'
    
    py_norm_np = np.asarray(norm_data)
    norm_tensor = engine.torch.from_numpy(py_norm_np).float().unsqueeze(0)

    pywith(engine.torch.no_grad()) do _
        # Pass both raw and normalized tensors
        output = engine.model(raw_tensor, norm_tensor)
        
        pred_pos_py = nothing
        pred_cov_py = nothing
        
        if estimator.model_type == "tcn"
            # OnlyTCN returns tensor directly
            pred_pos_py = output.squeeze(0) # (Seq, 3)
            # Dummy covariance
            seq_len = pred_pos_py.shape[0]
            # Create a zero covariance matrix (Seq, 15, 15)
            pred_cov_py = engine.torch.zeros(seq_len, 15, 15)
        elseif estimator.model_type in ["pure_eskf", "pure_integration"]
            # Baseline models return only position
            pred_pos_py = output["pred_pos_w"].squeeze(0) # (Seq, 3)
            # Dummy covariance for baselines
            seq_len = pred_pos_py.shape[0]
            pred_cov_py = engine.torch.zeros(seq_len, 15, 15)
        else
            # Hybrid models (ESKF-TCN, AEKF-TCN) return Dict with covariance
            pred_pos_py = output["pred_pos_w"].squeeze(0) # (Seq, 3)
            pred_cov_py = output["filter_covariance"].squeeze(0) # (Seq, 15, 15) or (Seq, 16, 16)
        end
        
        return (
            pos = pyconvert(Array, pred_pos_py),
            cov = pyconvert(Array, pred_cov_py)
        )
    end
end

export TrajectoEstimator

end
