module APSPlugin

using ..AbstractLayers

struct APSPerception <: AbstractPerception
    # Configuration for Attitude based Plane Segmentation
end

function AbstractLayers.process_input(perception::APSPerception, raw_data)
    println("APS Processing...")
    # Implementation pending
    return raw_data
end

export APSPerception

end
