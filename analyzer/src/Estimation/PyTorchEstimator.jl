module PyTorchEstimator

using ..AbstractLayers
using PythonCall
using HDF5

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
    
    println(">>> Loading PyTorch Model [$(estimator.model_type)]: ", abspath(estimator.model_path))
    
    raw_model = nothing
    
    if estimator.model_type == "eskf"
        model_module = pyimport("model.ESKF_TCN")
        raw_model = model_module.ESKFTCN_model(device="cpu")
    elseif estimator.model_type == "aekf"
        model_module = pyimport("model.AEKF_TCN")
        raw_model = model_module.AEKFTCN_model(device="cpu")
    elseif estimator.model_type == "tcn"
        model_module = pyimport("model.onlyTCN")
        raw_model = model_module.OnlyTCN(device="cpu")
    elseif estimator.model_type == "pure_eskf"
        model_module = pyimport("model.pure_eskf")
        raw_model = model_module.PureESKFModel(device="cpu")
    elseif estimator.model_type == "pure_integration"
        model_module = pyimport("model.pure_integration")
        raw_model = model_module.PureIntegrationModel(device="cpu")
    else
        error("Unknown model type: $(estimator.model_type)")
    end

    # Baseline models don't require pre-trained weights
    if !(estimator.model_type in ["pure_eskf", "pure_integration"])
        state_dict = torch.load(abspath(estimator.model_path), map_location="cpu")
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
