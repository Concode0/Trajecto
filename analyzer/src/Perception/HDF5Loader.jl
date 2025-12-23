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
        
        # Load data
        sensor = read(g["sensor_data"]) # (7, Seq)
        gt_pos = read(g["gt_pos_data"]) # (3, Seq)
        
        # Standardize to (Seq, Feature) for processing if needed, 
        # but let's keep raw as loaded and transform in the estimator if needed,
        # OR standardize here. 
        # Let's standardize to (Seq, Feature) as that's what ML models usually consume individually.
        
        return (sensor=Matrix(sensor'), gt_pos=Matrix(gt_pos'))
    end
end

export HDF5Perception

end
