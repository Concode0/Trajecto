# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
UI controls for the Trajecto Dashboard.

This module provides interactive controls including time slider and playback functionality.
"""
module DashboardControls

using ..Config
using GLMakie

"""
    PlaybackState

Container for playback control state.

# Fields
- `slider::Slider`: Time slider widget
- `is_playing::Observable{Bool}`: Playback state (true = playing, false = paused)
- `play_button::Button`: Play/pause button widget
- `frame_idx::Observable{Int}`: Current frame index (lifted from slider.value)
"""
struct PlaybackState
    slider::Slider
    is_playing::Observable{Bool}
    play_button::Button
    frame_idx::Observable{Int}
end


"""
    create_playback_controls(fig::Figure, grid_position, seq_len::Int) -> PlaybackState

Create time slider and playback controls for dashboard visualization.

# Arguments
- `fig::Figure`: Makie figure to add controls to
- `grid_position`: Grid position specification (e.g., `fig[3, 1:3]`)
- `seq_len::Int`: Total sequence length (maximum slider value)

# Returns
- `PlaybackState`: Container with all control widgets and observables

# Controls Created
- **Time Slider**: Range from 1 to seq_len, starts at frame 1
- **Frame Label**: Displays current frame number
- **Play/Pause Button**: Toggles playback state

# Example
```julia
fig = Figure()
controls = create_playback_controls(fig, fig[3, 1:3], 1000)

# Access current frame
current_frame = controls.frame_idx[]

# Programmatically control playback
controls.is_playing[] = true  # Start playing
```
"""
function create_playback_controls(fig::Figure, grid_position, seq_len::Int)
    # Create control layout
    ctrl_layout = GridLayout()
    grid_position[] = ctrl_layout

    # Time slider
    time_slider = Slider(ctrl_layout[1, 1], range = 1:seq_len, startvalue = 1)

    # Frame index observable (lifted from slider)
    frame_idx_obs = time_slider.value

    # Frame label
    Label(ctrl_layout[1, 2], @lift("Frame: $($frame_idx_obs)"), width = 100)

    # Playback state
    is_playing = Observable(false)

    # Play/pause button with dynamic label
    play_button = Button(ctrl_layout[1, 3],
                        label = @lift($is_playing ? "Pause" : "Play"))

    # Button click handler
    on(play_button.clicks) do _
        is_playing[] = !is_playing[]
    end

    return PlaybackState(time_slider, is_playing, play_button, frame_idx_obs)
end


"""
    start_playback_loop(playback_state::PlaybackState, seq_len::Int;
                       fps::Int=Config.PLAYBACK_FPS)

Start asynchronous playback loop for frame-by-frame animation.

# Arguments
- `playback_state::PlaybackState`: Playback control state from `create_playback_controls`
- `seq_len::Int`: Total sequence length
- `fps::Int`: Target frames per second (default: Config.PLAYBACK_FPS)

# Behavior
- Runs indefinitely in background task
- When playing: advances slider by 1 frame per tick
- When reaching end: loops back to frame 1 and pauses
- Sleep time calculated as 1/fps

# Example
```julia
controls = create_playback_controls(fig, fig[3, 1:3], 1000)
start_playback_loop(controls, 1000)  # Default 50 FPS
start_playback_loop(controls, 1000, fps=30)  # Custom 30 FPS
```

# Note
This function returns immediately after starting the background task.
The loop continues running until the Julia session ends.
"""
function start_playback_loop(playback_state::PlaybackState, seq_len::Int;
                            fps::Int=Config.PLAYBACK_FPS)
    sleep_time = 1.0 / fps

    @async begin
        while true
            sleep(sleep_time)
            if playback_state.is_playing[]
                current_frame = playback_state.slider.value[]
                if current_frame < seq_len
                    set_close_to!(playback_state.slider, current_frame + 1)
                else
                    # Loop back to start and pause
                    playback_state.is_playing[] = false
                    set_close_to!(playback_state.slider, 1)
                end
            end
        end
    end
end


export PlaybackState, create_playback_controls, start_playback_loop

end
