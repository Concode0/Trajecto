module HDF5Loader

using ..AbstractLayers
using HDF5

struct HDF5Perception <: AbstractPerception
    file_path::String
end

function AbstractLayers.process_input(perception::HDF5Perception, sample_key::String)
    h5open(perception.file_path, "r") do f
        if !haskey(f, sample_key)
            error("Sample $sample_key not found in $(perception.file_path)")
        end
        g = f[sample_key]

        # Load data (HDF5 stores as (Seq, Features) but Julia reads in column-major)
        sensor = read(g["sensor_data"])' #  Transpose from (7, Seq) to (Seq, 7)
        gt_pos = read(g["gt_pos_data"])' # Transpose from (3, Seq) to (Seq, 3)

        # Read sequence length from attributes, but cap at actual data size
        attr_seq_len = haskey(attributes(g), "sequence_length") ? read(attributes(g)["sequence_length"]) : size(sensor, 1)
        actual_size = size(sensor, 1)
        seq_len = min(attr_seq_len, actual_size)

        # Truncate to actual sequence length (remove padding)
        sensor_truncated = Matrix(sensor[1:seq_len, :])
        gt_pos_truncated = Matrix(gt_pos[1:seq_len, :])

        # Return as (Seq, Feature) matrices
        return (sensor=sensor_truncated, gt_pos=gt_pos_truncated, seq_len=seq_len)
    end
end

export HDF5Perception

end
