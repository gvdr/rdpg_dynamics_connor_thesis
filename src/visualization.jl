"""
    Visualization functions for RDPG temporal network dynamics.

    Provides three types of plots:
    1. Phase portraits - 2D embedding trajectories
    2. Network snapshots - Graph visualization at key timesteps
    3. Probability heatmaps - P = L*R' matrices over time
"""

using CairoMakie: Figure, Axis, lines!, scatter!, arrows!, Legend, LineElement, MarkerElement,
                  save, heatmap!, Colorbar, Label, hidespines!, hidedecorations!, DataAspect
using Colors: distinguishable_colors, RGB
using LinearAlgebra

export plot_phase_portrait, plot_network_snapshots, plot_probability_heatmaps
export plot_all_diagnostics
export plot_single_target_trajectory

"""
    plot_phase_portrait(L_data, predictions, name; n, d, output_dir="results/")

Create 2D phase portrait showing node trajectories in embedding space.

Ground truth shown as solid lines, predictions as dashed.
Nodes colored by their index to track individual trajectories.
"""
function plot_phase_portrait(L_data::Vector, predictions::Matrix, name::String;
                             n::Int, d::Int, output_dir::String="results/",
                             show_arrows::Bool=true)
    @assert d == 2 "Phase portraits only work for d=2"

    mkpath(output_dir)
    timesteps = min(size(predictions, 2), length(L_data))

    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1],
        xlabel="Embedding dimension 1",
        ylabel="Embedding dimension 2",
        title=name * " - Phase Portrait",
        aspect=DataAspect()
    )

    # Color palette for nodes
    colors = distinguishable_colors(n, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    # Plot ground truth trajectories
    for node in 1:n
        traj_x = [L_data[t][node, 1] for t in 1:timesteps]
        traj_y = [L_data[t][node, 2] for t in 1:timesteps]

        lines!(ax, traj_x, traj_y, color=(colors[node], 0.7), linewidth=2)

        # Start point
        scatter!(ax, [traj_x[1]], [traj_y[1]], color=colors[node],
                markersize=12, marker=:circle)
        # End point
        scatter!(ax, [traj_x[end]], [traj_y[end]], color=colors[node],
                markersize=12, marker=:star5)

        # Arrow showing direction
        if show_arrows && timesteps > 5
            mid = div(timesteps, 2)
            dx = traj_x[mid+1] - traj_x[mid]
            dy = traj_y[mid+1] - traj_y[mid]
            arrows!(ax, [traj_x[mid]], [traj_y[mid]], [dx*2], [dy*2],
                   color=(colors[node], 0.8), linewidth=2, arrowsize=10)
        end
    end

    # Plot predictions as dashed lines
    for node in 1:n
        pred_x = [predictions[(node-1)*d + 1, t] for t in 1:timesteps]
        pred_y = [predictions[(node-1)*d + 2, t] for t in 1:timesteps]

        lines!(ax, pred_x, pred_y, color=(colors[node], 0.5),
              linewidth=1.5, linestyle=:dash)
    end

    # Legend
    Legend(fig[1, 2],
        [LineElement(color=:gray, linewidth=2),
         LineElement(color=:gray, linewidth=1.5, linestyle=:dash),
         MarkerElement(color=:gray, marker=:circle, markersize=12),
         MarkerElement(color=:gray, marker=:star5, markersize=12)],
        ["Ground Truth", "Prediction", "Start", "End"],
        framevisible=false)

    save(output_dir * "/" * name * "_phase_portrait.pdf", fig)
    println("Saved: " * output_dir * "/" * name * "_phase_portrait.pdf")

    return fig
end

"""
    plot_network_snapshots(L_data, name; n, d, timesteps_to_show=[1, 15, 30], output_dir="results/")

Create network visualizations at key timesteps.

Node positions based on embedding, edges from P = L*R' with threshold.
"""
function plot_network_snapshots(L_data::Vector, name::String;
                                n::Int, d::Int,
                                timesteps_to_show::Vector{Int}=[1, 15, 30],
                                threshold::Float64=0.3,
                                output_dir::String="results/")
    mkpath(output_dir)

    # Filter valid timesteps
    max_t = length(L_data)
    timesteps_to_show = filter(t -> t <= max_t, timesteps_to_show)

    n_plots = length(timesteps_to_show)
    fig = Figure(size=(400 * n_plots, 400))

    for (idx, t) in enumerate(timesteps_to_show)
        ax = Axis(fig[1, idx],
            title="t = " * string(t),
            aspect=DataAspect())
        hidespines!(ax)
        hidedecorations!(ax)

        L = L_data[t]

        # Node positions (use first 2 dimensions)
        x = L[:, 1]
        y = d >= 2 ? L[:, 2] : zeros(n)

        # Compute probability matrix P = L * L' (assuming R ≈ L for undirected)
        P = L * L'
        P = clamp.(P, 0.0, 1.0)

        # Draw edges where P > threshold
        for i in 1:n
            for j in (i+1):n
                if P[i, j] > threshold
                    alpha = min(1.0, P[i, j])
                    lines!(ax, [x[i], x[j]], [y[i], y[j]],
                          color=(:gray, alpha * 0.5), linewidth=1)
                end
            end
        end

        # Draw nodes
        # Color by connectivity (row sum of P)
        connectivity = vec(sum(P, dims=2)) .- 1  # Subtract self-connection
        scatter!(ax, x, y, color=connectivity, colormap=:viridis,
                markersize=8, strokewidth=0.5, strokecolor=:black)
    end

    Label(fig[0, :], name * " - Network Snapshots", fontsize=16)

    save(output_dir * "/" * name * "_network_snapshots.pdf", fig)
    println("Saved: " * output_dir * "/" * name * "_network_snapshots.pdf")

    return fig
end

"""
    plot_probability_heatmaps(L_data, predictions, name; n, d, timesteps_to_show=[1, 15, 30], output_dir="results/")

Create heatmaps of probability matrices P = L*R' at key timesteps.

Shows both ground truth and predicted probability matrices.
"""
function plot_probability_heatmaps(L_data::Vector, predictions::Matrix, name::String;
                                   n::Int, d::Int,
                                   timesteps_to_show::Vector{Int}=[1, 15, 30],
                                   output_dir::String="results/")
    mkpath(output_dir)

    max_t = min(length(L_data), size(predictions, 2))
    timesteps_to_show = filter(t -> t <= max_t, timesteps_to_show)

    n_times = length(timesteps_to_show)
    fig = Figure(size=(350 * n_times, 700))

    for (idx, t) in enumerate(timesteps_to_show)
        # Ground truth
        L_true = L_data[t]
        P_true = L_true * L_true'
        P_true = clamp.(P_true, 0.0, 1.0)

        # Prediction
        L_pred = reshape(predictions[:, t], d, n)'  # n × d
        P_pred = L_pred * L_pred'
        P_pred = clamp.(P_pred, 0.0, 1.0)

        # True P heatmap
        ax1 = Axis(fig[1, idx], title="t=" * string(t) * " (True)",
                   aspect=DataAspect(), yreversed=true)
        heatmap!(ax1, P_true, colormap=:blues, colorrange=(0, 1))
        hidedecorations!(ax1)

        # Predicted P heatmap
        ax2 = Axis(fig[2, idx], title="t=" * string(t) * " (Predicted)",
                   aspect=DataAspect(), yreversed=true)
        heatmap!(ax2, P_pred, colormap=:blues, colorrange=(0, 1))
        hidedecorations!(ax2)
    end

    Label(fig[0, :], name * " - Probability Matrices P = LL'", fontsize=16)
    Colorbar(fig[1:2, n_times+1], colormap=:blues, limits=(0, 1), label="P(edge)")

    save(output_dir * "/" * name * "_probability_heatmaps.pdf", fig)
    println("Saved: " * output_dir * "/" * name * "_probability_heatmaps.pdf")

    return fig
end

"""
    plot_all_diagnostics(L_data, predictions, name; n, d, output_dir="results/")

Generate all three diagnostic plots for a dataset.
"""
function plot_all_diagnostics(L_data::Vector, predictions::Matrix, name::String;
                              n::Int, d::Int, output_dir::String="results/")
    println("\nGenerating diagnostic plots for " * name)

    # 1. Phase portrait
    if d == 2
        plot_phase_portrait(L_data, predictions, name; n=n, d=d, output_dir=output_dir)
    else
        println("  Skipping phase portrait (d != 2)")
    end

    # 2. Network snapshots
    max_t = length(L_data)
    timesteps = [1, div(max_t, 2), max_t]
    plot_network_snapshots(L_data, name; n=n, d=d, timesteps_to_show=timesteps, output_dir=output_dir)

    # 3. Probability heatmaps
    plot_probability_heatmaps(L_data, predictions, name; n=n, d=d,
                             timesteps_to_show=timesteps, output_dir=output_dir)

    println("  All diagnostic plots saved to " * output_dir)
end

"""
    plot_single_target_trajectory(L_data, target_idx, predictions, name;
                                   d=2, output_dir="results/")

Create visualization comparing ground truth and predicted trajectory for a single target node.

Shows:
- 2D phase portrait (if d=2) for the single node
- Time series for each embedding dimension
- Clear comparison of prediction quality

# Arguments
- `L_data`: Vector of embedding matrices (n × d), one per timestep
- `target_idx`: Index of the target node being predicted
- `predictions`: Predicted trajectory (d × timesteps matrix)
- `name`: Dataset/experiment name for file naming
- `d`: Embedding dimension (default: 2)
- `output_dir`: Output directory for saved plots
"""
function plot_single_target_trajectory(L_data::Vector, target_idx::Int,
                                        predictions::Matrix, name::String;
                                        d::Int=2, output_dir::String="results/")
    mkpath(output_dir)
    timesteps = min(size(predictions, 2), length(L_data))

    # Extract ground truth for target node
    ground_truth = hcat([L_data[t][target_idx, :] for t in 1:timesteps]...)  # d × T

    # Compute MSE
    mse = sum(abs2, ground_truth .- predictions[:, 1:timesteps]) / length(ground_truth)

    # Create figure with subplots
    if d == 2
        fig = Figure(size=(1200, 500))

        # Left: 2D Phase portrait
        ax1 = Axis(fig[1, 1],
            xlabel="Embedding dimension 1",
            ylabel="Embedding dimension 2",
            title="Node " * string(target_idx) * " - Phase Portrait",
            aspect=DataAspect()
        )

        # Ground truth trajectory
        lines!(ax1, ground_truth[1, :], ground_truth[2, :],
               color=:blue, linewidth=2, label="Ground Truth")
        scatter!(ax1, [ground_truth[1, 1]], [ground_truth[2, 1]],
                color=:blue, markersize=15, marker=:circle)  # Start
        scatter!(ax1, [ground_truth[1, end]], [ground_truth[2, end]],
                color=:blue, markersize=15, marker=:star5)  # End

        # Predicted trajectory
        lines!(ax1, predictions[1, 1:timesteps], predictions[2, 1:timesteps],
               color=:red, linewidth=2, linestyle=:dash, label="Prediction")
        scatter!(ax1, [predictions[1, 1]], [predictions[2, 1]],
                color=:red, markersize=12, marker=:circle)  # Start
        scatter!(ax1, [predictions[1, timesteps]], [predictions[2, timesteps]],
                color=:red, markersize=12, marker=:star5)  # End

        Legend(fig[1, 2], ax1, framevisible=false)

        # Right: Time series for each dimension
        ax2 = Axis(fig[1, 3],
            xlabel="Time",
            ylabel="Embedding value",
            title="Trajectory over time (MSE=" * string(round(mse; digits=4)) * ")"
        )

        ts = 1:timesteps
        lines!(ax2, ts, ground_truth[1, :], color=:blue, linewidth=2, label="True dim 1")
        lines!(ax2, ts, ground_truth[2, :], color=:green, linewidth=2, label="True dim 2")
        lines!(ax2, ts, predictions[1, 1:timesteps], color=:blue, linewidth=2,
               linestyle=:dash, label="Pred dim 1")
        lines!(ax2, ts, predictions[2, 1:timesteps], color=:green, linewidth=2,
               linestyle=:dash, label="Pred dim 2")

        Legend(fig[1, 4], ax2, framevisible=false)

    else
        # For d != 2, just show time series
        fig = Figure(size=(800, 300 * d))

        for dim in 1:d
            ax = Axis(fig[dim, 1],
                xlabel= dim == d ? "Time" : "",
                ylabel="Dim " * string(dim),
                title= dim == 1 ? "Node " * string(target_idx) * " (MSE=" * string(round(mse; digits=4)) * ")" : ""
            )

            ts = 1:timesteps
            lines!(ax, ts, ground_truth[dim, :], color=:blue, linewidth=2, label="Ground Truth")
            lines!(ax, ts, predictions[dim, 1:timesteps], color=:red, linewidth=2,
                   linestyle=:dash, label="Prediction")
        end

        Legend(fig[1, 2],
            [LineElement(color=:blue, linewidth=2),
             LineElement(color=:red, linewidth=2, linestyle=:dash)],
            ["Ground Truth", "Prediction"],
            framevisible=false)
    end

    filename = output_dir * "/" * name * "_node" * string(target_idx) * "_trajectory.pdf"
    save(filename, fig)
    println("Saved: " * filename)

    return fig, mse
end
