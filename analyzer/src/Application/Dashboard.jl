module Dashboard

using ..AbstractLayers
using ..Metrics
using GLMakie
using GeometryBasics
using Statistics
using LinearAlgebra

struct MakieDashboard <: AbstractApplication
    resolution::Tuple{Int, Int}
end

MakieDashboard() = MakieDashboard((1600, 1000))

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
    
    # Calculate detailed metrics
    # Assuming 50Hz (0.02s) as per standard Trajecto protocol
    metrics = calculate_metrics(gt_pos, pred_pos, 0.02)
    aligned_pred = metrics.aligned_traj
    
    # FIX: resolution -> size (Deprecated in Makie)
    fig = Figure(size = app.resolution, font = "sans")

    # --- Controls ---
    # Layout for controls
    ctrl_layout = GridLayout()
    fig[3, 1:3] = ctrl_layout

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

    # Title Info
    title_str = "Trajecto Analysis\n" * 
                "APE(RMSE): $(round(metrics.ape_rmse*100, digits=2)) cm | " *
                "Err/Dist: $(round(metrics.error_over_dist*100, digits=2)) % | " * 
                "Err/Time: $(round(metrics.error_over_time*100, digits=2)) cm/s"

    # 1. 3D Trajectory
    ax3d = Axis3(fig[1, 1:3], title = title_str,
                 aspect = :data, perspectiveness = 0.5)

    # Static full trajectories
    lines!(ax3d, gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3], 
           color = :blue, linewidth = 1, label = "Ground Truth", linestyle = :dash, alpha=0.5)
    lines!(ax3d, pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3], 
           color = :red, linewidth = 2, label = "Raw Prediction", alpha=0.5)
    lines!(ax3d, aligned_pred[:, 1], aligned_pred[:, 2], aligned_pred[:, 3], 
           color = :green, linewidth = 1, label = "Aligned Pred", linestyle = :dot, alpha=0.6)

    # Dynamic markers
    # Lift position based on slider value
    gt_pt = @lift(Point3f(gt_pos[$frame_idx_obs, 1], gt_pos[$frame_idx_obs, 2], gt_pos[$frame_idx_obs, 3]))
    pred_pt = @lift(Point3f(pred_pos[$frame_idx_obs, 1], pred_pos[$frame_idx_obs, 2], pred_pos[$frame_idx_obs, 3]))
    
    scatter!(ax3d, gt_pt, color = :blue, markersize = 15, label = "GT Head")
    scatter!(ax3d, pred_pt, color = :red, markersize = 15, label = "Pred Head")

    # Dynamic Error Ellipsoid (3-sigma) on Raw Prediction
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

    # 3. Error Distribution (Aligned)
    # Using aligned prediction for error to match APE metrics
    dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]
    ax_err = Axis(fig[2, 2], title = "Aligned Point-to-Point Error", xlabel = "Frame", ylabel = "Error (m)")

    # band! with explicit vector for lower bound
    band!(ax_err, 1:length(dist), zeros(length(dist)), dist, color = (:green, 0.2))
    lines!(ax_err, dist, color = :green, linewidth = 1)
    vlines!(ax_err, @lift([$frame_idx_obs]), color = :black, linestyle = :dash)

    # 4. Error vs Distance Graph (Cumulative)
    # Calculate cumulative distance along ground truth path
    gt_deltas = vcat([0.0 0.0 0.0], gt_pos[2:end, :] .- gt_pos[1:end-1, :])
    segment_distances = sqrt.(sum(gt_deltas.^2, dims=2))[:]
    cumulative_distance = cumsum(segment_distances)

    # Calculate cumulative error
    cumulative_error = cumsum(dist)

    # Calculate local error/distance ratio (with smoothing window)
    window_size = 10
    local_error_dist = zeros(seq_len)
    for i in 1:seq_len
        window_start = max(1, i - window_size + 1)
        window_error = sum(dist[window_start:i])
        window_dist = max(cumulative_distance[i] - cumulative_distance[window_start] + segment_distances[window_start], 1e-6)
        local_error_dist[i] = window_error / window_dist
    end

    ax_err_dist = Axis(fig[2, 3],
                       title = "Error/Distance Ratio",
                       xlabel = "Cumulative Distance (m)",
                       ylabel = "Error/Distance (%)")

    lines!(ax_err_dist, cumulative_distance, local_error_dist .* 100,
           color = :purple, linewidth = 2, label = "Local Ratio ($(window_size)-frame)")
    hlines!(ax_err_dist, [metrics.error_over_dist * 100],
            color = :orange, linestyle = :dash, linewidth = 2, label = "Global Avg")
    axislegend(ax_err_dist, position = :rt)

    # Add vline for current position
    current_dist = @lift(cumulative_distance[$frame_idx_obs])
    vlines!(ax_err_dist, current_dist, color = :black, linestyle = :dot)

    display(fig)
end

function run_app_comparison(app::MakieDashboard, results, input_stream)
    """
    Display multiple models side-by-side for comparison.

    Args:
        app: MakieDashboard instance
        results: Vector of NamedTuples with (name, trajectory)
        input_stream: Input data with ground truth
    """

    gt_pos = input_stream.gt_pos
    seq_len = size(gt_pos, 1)

    # Filter out failed models
    valid_results = filter(r -> r.trajectory !== nothing, results)
    n_models = length(valid_results)

    if n_models == 0
        println("No valid models to display!")
        return
    end

    # Calculate metrics for all models
    model_metrics = []
    for result in valid_results
        metrics = calculate_metrics(gt_pos, result.trajectory.pos, 0.02)
        push!(model_metrics, (name=result.name, metrics=metrics, trajectory=result.trajectory))
    end

    # Create figure with larger resolution for multiple plots
    fig = Figure(size = app.resolution, font = "sans")

    # Color palette for models
    colors = [:red, :orange, :purple, :cyan, :magenta, :yellow]

    # ========================================================================
    # TOP: 3D Trajectory Comparison (Shared View)
    # ========================================================================

    # Build title with all models
    title_lines = ["Multi-Model Trajectory Comparison"]
    for mm in model_metrics
        ape = round(mm.metrics.ape_rmse * 100, digits=2)
        err_dist = round(mm.metrics.error_over_dist * 100, digits=2)
        push!(title_lines, "$(mm.name): APE=$(ape)cm, Err/Dist=$(err_dist)%")
    end
    title_str = join(title_lines, " | ")

    ax3d = Axis3(fig[1, 1:n_models], title = title_str,
                 aspect = :data, perspectiveness = 0.5)

    # Ground truth (shared)
    lines!(ax3d, gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3],
           color = :blue, linewidth = 2, label = "Ground Truth", linestyle = :dash)

    # Plot each model's trajectory
    for (idx, mm) in enumerate(model_metrics)
        pred_pos = mm.trajectory.pos
        color = colors[mod1(idx, length(colors))]

        lines!(ax3d, pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3],
               color = color, linewidth = 2, label = mm.name, alpha = 0.7)
    end

    axislegend(ax3d, position = :lt)

    # ========================================================================
    # MIDDLE: Individual Model Panels (Side-by-Side)
    # ========================================================================

    for col_idx in 1:n_models
        mm = model_metrics[col_idx]
        pred_pos = mm.trajectory.pos
        color = colors[mod1(col_idx, length(colors))]

        # XY projection
        ax_xy = Axis(fig[2, col_idx],
                     title = "$(mm.name)\nXY View",
                     xlabel = "X (m)", ylabel = "Y (m)",
                     aspect = DataAspect())

        lines!(ax_xy, gt_pos[:, 1], gt_pos[:, 2],
               color = :blue, linewidth = 1, linestyle = :dash, alpha = 0.5)
        lines!(ax_xy, pred_pos[:, 1], pred_pos[:, 2],
               color = color, linewidth = 2)
    end

    # ========================================================================
    # BOTTOM: Error Comparison
    # ========================================================================

    ax_err_cmp = Axis(fig[3, 1:n_models],
                      title = "Point-to-Point Error Comparison",
                      xlabel = "Frame", ylabel = "Error (m)")

    for (idx, mm) in enumerate(model_metrics)
        aligned_pred = mm.metrics.aligned_traj
        dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]
        color = colors[mod1(idx, length(colors))]

        lines!(ax_err_cmp, dist, color = color, linewidth = 2, label = mm.name)
    end

    axislegend(ax_err_cmp, position = :rt)

    # ========================================================================
    # Error/Distance Comparison
    # ========================================================================

    ax_err_dist_cmp = Axis(fig[4, 1:n_models],
                           title = "Error/Distance Ratio Comparison",
                           xlabel = "Cumulative Distance (m)",
                           ylabel = "Error/Distance (%)")

    # Calculate cumulative distance (same for all models - using GT)
    gt_deltas = vcat([0.0 0.0 0.0], gt_pos[2:end, :] .- gt_pos[1:end-1, :])
    segment_distances = sqrt.(sum(gt_deltas.^2, dims=2))[:]
    cumulative_distance = cumsum(segment_distances)

    for (idx, mm) in enumerate(model_metrics)
        aligned_pred = mm.metrics.aligned_traj
        dist = sqrt.(sum((gt_pos .- aligned_pred).^2, dims=2))[:]
        color = colors[mod1(idx, length(colors))]

        # Calculate local error/distance ratio with smoothing
        window_size = 10
        local_error_dist = zeros(seq_len)
        for i in 1:seq_len
            window_start = max(1, i - window_size + 1)
            window_error = sum(dist[window_start:i])
            window_dist = max(cumulative_distance[i] - cumulative_distance[window_start] + segment_distances[window_start], 1e-6)
            local_error_dist[i] = window_error / window_dist
        end

        lines!(ax_err_dist_cmp, cumulative_distance, local_error_dist .* 100,
               color = color, linewidth = 2, label = "$(mm.name) ($(round(mm.metrics.error_over_dist*100, digits=2))%)")
    end

    axislegend(ax_err_dist_cmp, position = :rt)

    # ========================================================================
    # CONTROLS
    # ========================================================================

    ctrl_layout = GridLayout()
    fig[5, 1:n_models] = ctrl_layout

    time_slider = Slider(ctrl_layout[1, 1], range = 1:seq_len, startvalue = 1)
    frame_idx_obs = time_slider.value
    Label(ctrl_layout[1, 2], @lift("Frame: $($frame_idx_obs)"), width = 100)

    is_playing = Observable(false)
    play_button = Button(ctrl_layout[1, 3], label = @lift($is_playing ? "Pause" : "Play"))

    on(play_button.clicks) do _
        is_playing[] = !is_playing[]
    end

    # Playback timer
    @async begin
        while true
            sleep(0.02)
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

    # Add dynamic markers to 3D plot
    for (idx, mm) in enumerate(model_metrics)
        pred_pos = mm.trajectory.pos
        color = colors[mod1(idx, length(colors))]

        pred_pt = @lift(Point3f0(pred_pos[$frame_idx_obs, 1],
                                 pred_pos[$frame_idx_obs, 2],
                                 pred_pos[$frame_idx_obs, 3]))
        scatter!(ax3d, pred_pt, color = color, markersize = 15)
    end

    # GT marker
    gt_pt = @lift(Point3f0(gt_pos[$frame_idx_obs, 1],
                           gt_pos[$frame_idx_obs, 2],
                           gt_pos[$frame_idx_obs, 3]))
    scatter!(ax3d, gt_pt, color = :blue, markersize = 20, marker = :star5)

    # Add vlines to error plot
    vlines!(ax_err_cmp, @lift([$frame_idx_obs]), color = :black, linestyle = :dash)

    # Add vline to error/distance plot (by cumulative distance)
    current_dist_cmp = @lift(cumulative_distance[$frame_idx_obs])
    vlines!(ax_err_dist_cmp, current_dist_cmp, color = :black, linestyle = :dash)

    display(fig)
end

export MakieDashboard, run_app_comparison

end
