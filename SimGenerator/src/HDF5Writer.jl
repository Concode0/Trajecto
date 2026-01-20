"""
HDF5 Writer: Export generated data to HDF5 format.

Output format matches model/dataset.py expectations:
- sensor_data: [T, 7] (accel, gyro, fsr)
- gt_pos_data: [T, 3]
- gt_vel_data: [T, 3]
- gt_gravity_b_data: [T, 3]
- pen_down: [T] (optional, pen contact state)
- attrs: sequence_length, original_label
"""
module HDF5Writer

using HDF5
using ..Config

export TrajectoryData, export_hdf5

"""
    GenerationMetadata

Metadata about generation parameters for reproducibility and analysis.

# Fields
- `num_strokes::Int`: Number of strokes in this sample
- `duration::Float64`: Writing duration (seconds)
- `use_ik::Bool`: Whether inverse kinematics was used
- `use_enhanced_noise::Bool`: Whether enhanced noise models were used
- `generation_method::String`: Method used (e.g., "sigma_lognormal", "curved", "spiral")
- `separated_strokes::Bool`: Whether strokes are separated with gaps
- `dt::Float64`: Time step used
- `static_buffer::Float64`: Static buffer duration
- `pen_lift_count::Int`: Number of pen lift events
- `max_velocity::Float64`: Maximum velocity in trajectory
- `path_length::Float64`: Total path length
- `grip_style::String`: Grip style name (e.g., "TRIPOD", "LATERAL_TRIPOD")
- `handedness::String`: Hand used ("RIGHT_HAND" or "LEFT_HAND")
"""
struct GenerationMetadata
    num_strokes::Int
    duration::Float64
    use_ik::Bool
    use_enhanced_noise::Bool
    generation_method::String
    separated_strokes::Bool
    dt::Float64
    static_buffer::Float64
    pen_lift_count::Int
    max_velocity::Float64
    path_length::Float64
    grip_style::String
    handedness::String
end

"""
    TrajectoryData

Complete trajectory sample ready for export.

# Fields
- `sensor_data::Matrix{Float64}`: IMU data [T, 7] (accel, gyro, fsr)
- `gt_pos_data::Matrix{Float64}`: Ground truth position [T, 3]
- `gt_vel_data::Matrix{Float64}`: Ground truth velocity [T, 3]
- `gt_gravity_b_data::Matrix{Float64}`: Gravity in body frame [T, 3]
- `pen_down::Vector{Bool}`: Pen contact state [T]
- `sequence_length::Int`: Actual sequence length before padding
- `label::String`: Sample label/identifier
- `metadata::Union{GenerationMetadata,Nothing}`: Optional generation metadata
"""
struct TrajectoryData
    sensor_data::Matrix{Float64}      # [T, 7]: accel, gyro, fsr
    gt_pos_data::Matrix{Float64}      # [T, 3]: world position
    gt_vel_data::Matrix{Float64}      # [T, 3]: world velocity
    gt_gravity_b_data::Matrix{Float64} # [T, 3]: gravity in body frame
    pen_down::Vector{Bool}            # [T]: pen contact state
    sequence_length::Int
    label::String
    metadata::Union{GenerationMetadata,Nothing}
end

# Constructor without metadata for backward compatibility
function TrajectoryData(sensor_data, gt_pos_data, gt_vel_data, gt_gravity_b_data,
                        pen_down, sequence_length, label)
    TrajectoryData(sensor_data, gt_pos_data, gt_vel_data, gt_gravity_b_data,
                   pen_down, sequence_length, label, nothing)
end

export GenerationMetadata

"""
    pad_sequence(data::Matrix{Float64}, max_len::Int)

Pad or truncate sequence to fixed length.
Uses edge padding (repeat last value).
"""
function pad_sequence(data::Matrix{Float64}, max_len::Int)
    T, F = size(data)
    if T >= max_len
        return data[1:max_len, :]
    else
        padded = zeros(max_len, F)
        padded[1:T, :] = data
        # Edge-pad with last value
        for i in (T+1):max_len
            padded[i, :] = data[end, :]
        end
        return padded
    end
end

"""
    pad_sequence_bool(data::Vector{Bool}, max_len::Int)

Pad or truncate boolean sequence to fixed length.
Uses edge padding (repeat last value).
"""
function pad_sequence_bool(data::Vector{Bool}, max_len::Int)
    T = length(data)
    if T >= max_len
        return data[1:max_len]
    else
        padded = fill(data[end], max_len)
        padded[1:T] = data
        return padded
    end
end

"""
    export_hdf5(path::String, samples::Vector{TrajectoryData};
                max_len::Int=MAX_SEQUENCE_LENGTH,
                include_pen_down::Bool=true)

Export trajectory samples to HDF5 file.

Format matches model/dataset.py:
- Each sample as a group: "sim_sample_XXX"
- Datasets: sensor_data, gt_pos_data, gt_vel_data, gt_gravity_b_data, pen_down
- Attributes: sequence_length, original_label

# Arguments
- `path`: Output file path
- `samples`: Vector of TrajectoryData to export
- `max_len`: Maximum sequence length (sequences are padded/truncated)
- `include_pen_down`: Whether to include pen_down flag in export
"""
function export_hdf5(path::String, samples::Vector{TrajectoryData};
                      max_len::Int=MAX_SEQUENCE_LENGTH,
                      include_pen_down::Bool=true)
    # Ensure output directory exists
    dir = dirname(path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    h5open(path, "w") do f
        for (i, sample) in enumerate(samples)
            grp_name = "sim_sample_$(lpad(i-1, 3, '0'))"
            grp = create_group(f, grp_name)

            # Pad sequences to fixed length and transpose for row-major (Python) storage
            # Julia [T, F] -> HDF5 stored as [F, T] -> Python reads as [T, F]
            grp["sensor_data"] = permutedims(pad_sequence(sample.sensor_data, max_len))
            grp["gt_pos_data"] = permutedims(pad_sequence(sample.gt_pos_data, max_len))
            grp["gt_vel_data"] = permutedims(pad_sequence(sample.gt_vel_data, max_len))
            grp["gt_gravity_b_data"] = permutedims(pad_sequence(sample.gt_gravity_b_data, max_len))

            # Optional pen_down flag
            if include_pen_down
                # Convert Bool to Int8 for HDF5 compatibility
                pen_down_padded = pad_sequence_bool(sample.pen_down, max_len)
                grp["pen_down"] = Int8.(pen_down_padded)
            end

            # Attributes - core
            attributes(grp)["sequence_length"] = sample.sequence_length
            attributes(grp)["original_label"] = sample.label

            # Attributes - generation metadata (if available)
            if !isnothing(sample.metadata)
                meta = sample.metadata
                attributes(grp)["num_strokes"] = meta.num_strokes
                attributes(grp)["duration"] = meta.duration
                attributes(grp)["use_ik"] = meta.use_ik
                attributes(grp)["use_enhanced_noise"] = meta.use_enhanced_noise
                attributes(grp)["generation_method"] = meta.generation_method
                attributes(grp)["separated_strokes"] = meta.separated_strokes
                attributes(grp)["dt"] = meta.dt
                attributes(grp)["static_buffer"] = meta.static_buffer
                attributes(grp)["pen_lift_count"] = meta.pen_lift_count
                attributes(grp)["max_velocity"] = meta.max_velocity
                attributes(grp)["path_length"] = meta.path_length
                attributes(grp)["grip_style"] = meta.grip_style
                attributes(grp)["handedness"] = meta.handedness
            end
        end

        # Global attributes - configuration constants
        attributes(f)["config/TARGET_SAMPLING_RATE_HZ"] = TARGET_SAMPLING_RATE_HZ
        attributes(f)["config/DT"] = DT
        attributes(f)["config/GRAVITY_MAGNITUDE"] = GRAVITY_MAGNITUDE
        attributes(f)["config/MAX_SEQUENCE_LENGTH"] = MAX_SEQUENCE_LENGTH
        attributes(f)["config/STATIC_BUFFER_S"] = STATIC_BUFFER_S

        # Allan variance parameters
        attributes(f)["config/VRW"] = collect(VRW)
        attributes(f)["config/ARW"] = collect(ARW)
        attributes(f)["config/ACCEL_BI"] = collect(ACCEL_BI)
        attributes(f)["config/GYRO_BI"] = collect(GYRO_BI)

        # Generation info
        attributes(f)["generator/version"] = "1.0.0"
        attributes(f)["generator/architecture"] = "three_layer_kinematic_posture_sensing"
        attributes(f)["generator/num_samples"] = length(samples)
    end

    return nothing
end

end # module
