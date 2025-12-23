module Dashboard

using ..AbstractLayers
using GLMakie
using GeometryBasics
using Statistics
using LinearAlgebra

struct MakieDashboard <: AbstractApplication
    resolution::Tuple{Int, Int}
end

MakieDashboard() = MakieDashboard((1600, 1000))

function calculate_ate(gt::AbstractMatrix, pred::AbstractMatrix)
    # Expecting (Seq, 3)
    errors = sqrt.(sum((gt .- pred).^2, dims=2))
    return mean(errors)
end

function calculate_ellipsoid_model(cov_3x3, pos_3d)
    # Eigen decomposition for orientation and scale
    # Ensure symmetric for eigen
    cov_sym = Symmetric(cov_3x3)
    F = eigen(cov_sym)
    
    # 3-sigma scaling
    # Clamp to avoid negative values (numerical noise)
    radii = sqrt.(max.(F.values, 1e-9)) .* 3.0 
    
    # Rotation matrix from eigenvectors
    # F.vectors are columns corresponding to values
    rot_mat = F.vectors
    
    # Construct 4x4 transformation matrix manually
    # M = T * R * S
    
    # 1. Scale
    S = Diagonal([radii; 1.0])
    
    # 2. Rotation (expand to 4x4)
    R = Matrix{Float64}(I, 4, 4)
    R[1:3, 1:3] = rot_mat
    
    # 3. Translation
    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 4] = pos_3d
    
    return Mat4f(T * R * S)
end

function AbstractLayers.run_app(app::MakieDashboard, trajectory_data, input_stream)
    # input_stream: NamedTuple (sensor=(Seq, 7), gt_pos=(Seq, 3))
    # trajectory_data: NamedTuple (pos=(Seq, 3), cov=(Seq, 15, 15))
    
    pred_pos = trajectory_data.pos
    pred_cov = trajectory_data.cov
    gt_pos = input_stream.gt_pos
    sensor = input_stream.sensor
    
    seq_len = size(gt_pos, 1)
    ate = calculate_ate(gt_pos, pred_pos)
    
    # FIX: resolution -> size (Deprecated in Makie)
    fig = Figure(size = app.resolution, font = "sans")

    # --- Controls ---
    # Layout for controls
    ctrl_layout = GridLayout()
    fig[3, 1:2] = ctrl_layout

    # Slider
    # Explicitly using a GridLayout element to host the Slider
    time_slider = Slider(ctrl_layout[1, 1], range = 1:seq_len, startvalue = 1)
    
    # Label
    # Lift the slider value to display text
    frame_idx_obs = time_slider.value
    Label(ctrl_layout[1, 2], @lift("Frame: $($frame_idx_obs)"), width = 100)
    
    # Playback control
    is_playing = Observable(false)
    play_button = Button(ctrl_layout[1, 3], label = @lift($is_playing ? "Pause" : "Play"))
    
    on(play_button.clicks) do _
        is_playing[] = !is_playing[]
    end
    
    # Timer for playback
    @async begin
        while true
            sleep(0.02) # ~50 FPS
            if is_playing[]
                current_frame = time_slider.value[]
                if current_frame < seq_len
                    set_close_to!(time_slider, current_frame + 1)
                else
                    is_playing[] = false
                    set_close_to!(time_slider, 1)
                end
            end
        end
    end

    # 1. 3D Trajectory
    ax3d = Axis3(fig[1, 1:2], title = "Trajecto 3D Analysis (ATE: $(round(ate*100, digits=2))cm)",
                 aspect = :data, perspectiveness = 0.5)

    # Static full trajectories
    lines!(ax3d, gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3], 
           color = :blue, linewidth = 1, label = "Ground Truth", linestyle = :dash, alpha=0.5)
    lines!(ax3d, pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3], 
           color = :red, linewidth = 2, label = "Prediction", alpha=0.5)

    # Dynamic markers
    # Lift position based on slider value
    gt_pt = @lift(Point3f(gt_pos[$frame_idx_obs, 1], gt_pos[$frame_idx_obs, 2], gt_pos[$frame_idx_obs, 3]))
    pred_pt = @lift(Point3f(pred_pos[$frame_idx_obs, 1], pred_pos[$frame_idx_obs, 2], pred_pos[$frame_idx_obs, 3]))
    
    scatter!(ax3d, gt_pt, color = :blue, markersize = 15, label = "GT Head")
    scatter!(ax3d, pred_pt, color = :red, markersize = 15, label = "Pred Head")

    # Dynamic Error Ellipsoid (3-sigma)
    ellipsoid_model = @lift(calculate_ellipsoid_model(
        pred_cov[$frame_idx_obs, 1:3, 1:3], 
        pred_pos[$frame_idx_obs, :]
    ))
    mesh!(ax3d, Sphere(Point3f(0), 1.0f0), color=(:orange, 0.3), transparency=true, model=ellipsoid_model, label="3σ Covariance")
    
    axislegend(ax3d)

    # 2. Sensor Data (Accel)
    ax_acc = Axis(fig[2, 1], title = "Accelerometer", xlabel = "Frame", ylabel = "m/s²")
    colors = [:red, :green, :blue]
    for i in 1:3
        lines!(ax_acc, sensor[:, i], color = (colors[i], 0.7), label = "Acc $(['X','Y','Z'][i])")
    end
    # vlines! with lifted observable vector
    vlines!(ax_acc, @lift([$frame_idx_obs]), color = :black, linestyle = :dash)

    # 3. Error Distribution
    dist = sqrt.(sum((gt_pos .- pred_pos).^2, dims=2))[:]
    ax_err = Axis(fig[2, 2], title = "Point-to-Point Error", xlabel = "Frame", ylabel = "Error (m)")
    
    # band! with explicit vector for lower bound
    band!(ax_err, 1:length(dist), zeros(length(dist)), dist, color = (:red, 0.2))
    lines!(ax_err, dist, color = :red, linewidth = 1)
    vlines!(ax_err, @lift([$frame_idx_obs]), color = :black, linestyle = :dash)

    display(fig)
end

export MakieDashboard

end
