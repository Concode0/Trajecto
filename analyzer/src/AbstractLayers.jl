module AbstractLayers

# --- 1. Data Perception Layer ---
abstract type AbstractPerception end

"""
    process_input(perception::AbstractPerception, raw_data)

Converts raw input data into a standardized format for the estimator.
"""
function process_input end

# --- 2. Estimation Core Layer ---
abstract type AbstractEstimator end

"""
    predict_trajectory(estimator::AbstractEstimator, input_stream)

Takes standardized input and returns the estimated state/trajectory.
"""
function predict_trajectory end

"""
    load_model(estimator::AbstractEstimator, model_path)

Loads the model weights/parameters.
"""
function load_model end

# --- 3. Application Layer ---
abstract type AbstractApplication end

"""
    run_app(app::AbstractApplication, trajectory_data, sensor_data)

Executes the application logic (e.g., visualization, recognition) using the estimated trajectory.
"""
function run_app end

export AbstractPerception, process_input
export AbstractEstimator, predict_trajectory, load_model
export AbstractApplication, run_app

end
