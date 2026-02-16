#!/usr/bin/env -S julia --project
"""
Alg4 Dynamics Visualization

Generates diagnostic plots for all three dynamics families:
- Linear symmetric
- Polynomial in P
- Message-passing

Plots include:
1. Phase portraits (2D trajectories of X(t))
2. Spectral gap over time
3. Skew gauge error over time
4. P(t) validity (min/max values)
5. P matrix heatmaps at key timesteps

Usage:
    julia --project scripts/alg4/plot_dynamics.jl [family]

    family: "linear", "polynomial", "message_passing", or "all" (default)
"""

using Serialization
using Statistics
using LinearAlgebra
using DotProductGraphs
using CairoMakie: Figure, Axis, lines!, scatter!, arrows2d!, heatmap!, band!,
    Colorbar, Label, Legend, MarkerElement, hidedecorations!, axislegend,
    text!, hlines!, save, DataAspect
using Colors: distinguishable_colors, RGB

# =============================================================================
# Utility Functions
# =============================================================================

"""
    load_data(family::String)

Load serialized data for a given dynamics family.
"""
function load_data(family::String)
    path = joinpath("data", "alg4", family, family * "_data.jls")
    if !isfile(path)
        error("Data file not found: " * path * "\nRun the generator script first.")
    end
    return deserialize(path)
end

"""
    get_distinguishable_colors(n::Int)

Generate n distinguishable colors for plotting.
"""
function get_distinguishable_colors(n::Int)
    return distinguishable_colors(n, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
end

# =============================================================================
# Plotting Functions
# =============================================================================

"""
    plot_phase_portrait(data, family; output_dir="results/alg4")

Create 2D phase portrait showing true X(t) trajectories.
"""
function plot_phase_portrait(data::Dict, family::String; output_dir::String="results/alg4")
    mkpath(output_dir)

    X_series = data["X_true_series"]
    n = size(X_series[1], 1)
    T = length(X_series)

    fig = Figure(size=(900, 700))
    ax = Axis(fig[1, 1],
        xlabel="Embedding dimension 1",
        ylabel="Embedding dimension 2",
        title=uppercase(family[1]) * family[2:end] * " Dynamics - True Trajectories",
        aspect=DataAspect()
    )

    colors = get_distinguishable_colors(n)

    for node in 1:n
        traj_x = [X_series[t][node, 1] for t in 1:T]
        traj_y = [X_series[t][node, 2] for t in 1:T]

        # Trajectory line
        lines!(ax, traj_x, traj_y, color=(colors[node], 0.6), linewidth=1.5)

        # Start point (circle)
        scatter!(ax, [traj_x[1]], [traj_y[1]], color=colors[node],
                markersize=10, marker=:circle)
        # End point (star)
        scatter!(ax, [traj_x[end]], [traj_y[end]], color=colors[node],
                markersize=10, marker=:star5)

        # Arrow at midpoint showing direction
        if T > 4
            mid = div(T, 2)
            dx = traj_x[mid+1] - traj_x[mid]
            dy = traj_y[mid+1] - traj_y[mid]
            scale = 2.0
            arrows2d!(ax, [traj_x[mid]], [traj_y[mid]], [dx*scale], [dy*scale],
                   color=(colors[node], 0.8), tipwidth=8, tiplength=8)
        end
    end

    # Legend for markers
    Legend(fig[1, 2],
        [MarkerElement(color=:gray, marker=:circle, markersize=10),
         MarkerElement(color=:gray, marker=:star5, markersize=10)],
        ["Start (t=0)", "End (t=T)"],
        framevisible=false)

    filename = joinpath(output_dir, family * "_phase_portrait.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_phase_portrait_with_embeddings(data, family; output_dir="results/alg4")

Phase portrait comparing true X(t) vs ASE embeddings X̂(t).
"""
function plot_phase_portrait_with_embeddings(data::Dict, family::String; output_dir::String="results/alg4")
    mkpath(output_dir)

    X_series = data["X_true_series"]
    X_hat_series = data["X_hat_series"]
    n = size(X_series[1], 1)
    T = length(X_series)

    fig = Figure(size=(1400, 600))

    # True trajectories
    ax1 = Axis(fig[1, 1],
        xlabel="Dimension 1",
        ylabel="Dimension 2",
        title="True X(t)",
        aspect=DataAspect()
    )

    # ASE embeddings
    ax2 = Axis(fig[1, 2],
        xlabel="Dimension 1",
        ylabel="Dimension 2",
        title="ASE Embeddings X̂(t)",
        aspect=DataAspect()
    )

    colors = get_distinguishable_colors(n)

    for node in 1:n
        # True trajectory
        traj_x = [X_series[t][node, 1] for t in 1:T]
        traj_y = [X_series[t][node, 2] for t in 1:T]
        lines!(ax1, traj_x, traj_y, color=(colors[node], 0.6), linewidth=1.5)
        scatter!(ax1, [traj_x[1]], [traj_y[1]], color=colors[node], markersize=8, marker=:circle)
        scatter!(ax1, [traj_x[end]], [traj_y[end]], color=colors[node], markersize=8, marker=:star5)

        # ASE embedding
        hat_x = [X_hat_series[t][node, 1] for t in 1:T]
        hat_y = [X_hat_series[t][node, 2] for t in 1:T]
        lines!(ax2, hat_x, hat_y, color=(colors[node], 0.6), linewidth=1.5)
        scatter!(ax2, [hat_x[1]], [hat_y[1]], color=colors[node], markersize=8, marker=:circle)
        scatter!(ax2, [hat_x[end]], [hat_y[end]], color=colors[node], markersize=8, marker=:star5)
    end

    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " Dynamics - True vs ASE Embeddings",
          fontsize=18)

    filename = joinpath(output_dir, family * "_phase_comparison.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_phase_portrait_methods(data, family; output_dir="results/alg4")

Plot 2D trajectories for multiple embedding methods on averaged adjacencies.
"""
function plot_phase_portrait_methods(data::Dict, family::String; output_dir::String="results/alg4")
    mkpath(output_dir)

    X_true_series = data["X_true_series"]
    X_hat_series = data["X_hat_series"]
    A_series = data["A_avg_series"]

    n = size(X_true_series[1], 1)
    T = length(X_true_series)
    d = size(X_hat_series[1], 2)

    if d < 2
        error("plot_phase_portrait_methods requires d >= 2")
    end

    methods = Dict{String, Vector{Matrix{Float64}}}()
    methods["true"] = X_true_series
    methods["ase"] = X_hat_series

    results_dir = joinpath("results", "alg4")
    init_methods = ["procrustes_chain", "none", "dpg_procrustes"]
    for init_method in init_methods
        result_path = joinpath(results_dir, family * "_alignment_results_" * init_method * ".jls")
        if isfile(result_path)
            res = deserialize(result_path)
            Q_series = res["Q_series"]
            methods["structure_constrained_" * init_method] = [X_hat_series[t] * Q_series[t] for t in 1:length(Q_series)]
        end
    end

    TNE = TemporalNetworkEmbedding(A_series, d, :procrustes)
    methods["dpg_procrustes"] = [TNE[t, :AL] for t in 1:length(TNE)]

    TNE = TemporalNetworkEmbedding(A_series, d, :omni)
    methods["dpg_omni"] = [TNE[t, :AL] for t in 1:length(TNE)]

    TNE = TemporalNetworkEmbedding(A_series, d, :uase)
    methods["dpg_uase"] = [TNE[t, :AL] for t in 1:length(TNE)]

    TNE = TemporalNetworkEmbedding(A_series, d, :mase)
    methods["dpg_mase"] = [TNE[t, :AL] for t in 1:length(TNE)]

    duase_res = duase_embedding(A_series, d)
    methods["dpg_duase"] = [duase_res.Y[:, :, t] for t in 1:size(duase_res.Y, 3)]

    gbdase_res = gbdase_MAP(A_series, d; max_iter=10)
    methods["dpg_gbdase_map"] = [Matrix(gbdase_res.X[t, :, :]) for t in 1:size(gbdase_res.X, 1)]

    method_names = collect(keys(methods))
    n_methods = length(method_names)
    ncols = 3
    nrows = cld(n_methods, ncols)

    fig = Figure(size=(420 * ncols, 360 * nrows))
    colors = get_distinguishable_colors(n)

    for (idx, name) in enumerate(method_names)
        row = div(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1

        ax = Axis(fig[row, col],
            xlabel="Dimension 1",
            ylabel="Dimension 2",
            title=name,
            aspect=DataAspect()
        )

        series = methods[name]
        for node in 1:n
            traj_x = [series[t][node, 1] for t in 1:T]
            traj_y = [series[t][node, 2] for t in 1:T]
            lines!(ax, traj_x, traj_y, color=(colors[node], 0.6), linewidth=1.2)
        end
    end

    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " - Embedding Method Comparison",
          fontsize=18)

    filename = joinpath(output_dir, family * "_phase_methods.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_diagnostics_timeseries(data, family; output_dir="results/alg4")

Plot spectral gap, skew error, and P validity over time.
"""
function plot_diagnostics_timeseries(data::Dict, family::String; output_dir::String="results/alg4")
    mkpath(output_dir)

    spectral_gaps = data["spectral_gaps"]
    skew_errors = data["skew_errors"]
    prob_stats = data["probability_stats"]
    config = data["config"]

    T = length(spectral_gaps)
    dt = config["dt"]
    times = range(0, step=dt, length=T)

    fig = Figure(size=(1200, 800))

    # Spectral gap
    ax1 = Axis(fig[1, 1],
        xlabel="Time",
        ylabel="λ_d (spectral gap)",
        title="Spectral Gap Over Time"
    )
    lines!(ax1, collect(times), spectral_gaps, color=:blue, linewidth=2)
    hlines!(ax1, [0.0], color=:red, linestyle=:dash, linewidth=1, label="Zero")

    # Skew error
    ax2 = Axis(fig[1, 2],
        xlabel="Time",
        ylabel="||Skew(X'dX)||_F",
        title="Gauge Skew Error Over Time"
    )
    if length(skew_errors) > 0
        times_skew = range(0, step=dt, length=length(skew_errors))
        lines!(ax2, collect(times_skew), skew_errors, color=:orange, linewidth=2)
    else
        text!(ax2, 0.5, 0.5, text="No data", align=(:center, :center))
    end

    # P validity: min and max
    min_P = [s.min_P for s in prob_stats]
    max_P = [s.max_P for s in prob_stats]

    ax3 = Axis(fig[2, 1],
        xlabel="Time",
        ylabel="P value",
        title="P(t) Range Over Time"
    )
    lines!(ax3, collect(times), min_P, color=:purple, linewidth=2, label="min(P)")
    lines!(ax3, collect(times), max_P, color=:green, linewidth=2, label="max(P)")
    hlines!(ax3, [0.0, 1.0], color=:red, linestyle=:dash, linewidth=1)
    axislegend(ax3, position=:rt)

    # P violation counts
    neg_counts = [s.neg_count for s in prob_stats]
    above_counts = [s.above_one_count for s in prob_stats]

    ax4 = Axis(fig[2, 2],
        xlabel="Time",
        ylabel="Count",
        title="P(t) Constraint Violations"
    )
    lines!(ax4, collect(times), neg_counts, color=:red, linewidth=2, label="P < 0")
    lines!(ax4, collect(times), above_counts, color=:orange, linewidth=2, label="P > 1")
    axislegend(ax4, position=:rt)

    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " Dynamics - Diagnostics",
          fontsize=18)

    filename = joinpath(output_dir, family * "_diagnostics.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_P_heatmaps(data, family; output_dir="results/alg4", n_snapshots=4)

Plot heatmaps of P = X X' at several timesteps.
"""
function plot_P_heatmaps(data::Dict, family::String; output_dir::String="results/alg4", n_snapshots::Int=4)
    mkpath(output_dir)

    X_series = data["X_true_series"]
    T = length(X_series)
    n = size(X_series[1], 1)

    # Select timesteps to show
    timesteps = unique(round.(Int, range(1, T, length=n_snapshots)))

    fig = Figure(size=(300 * length(timesteps), 350))

    for (idx, t) in enumerate(timesteps)
        X = X_series[t]
        P = X * X'

        ax = Axis(fig[1, idx],
            title="t = " * string(t),
            aspect=DataAspect(),
            yreversed=true
        )

        hm = heatmap!(ax, P, colormap=:viridis, colorrange=(0, 1))
        hidedecorations!(ax)

        if idx == length(timesteps)
            Colorbar(fig[1, idx+1], hm, label="P(i,j)")
        end
    end

    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " - Probability Matrices P(t) = X(t)X(t)'",
          fontsize=16)

    filename = joinpath(output_dir, family * "_P_heatmaps.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_P_heatmaps_clamped(data, family; output_dir="results/alg4", n_snapshots=4)

Plot heatmaps of P with values clamped to [0,1] and violations highlighted.
"""
function plot_P_heatmaps_clamped(data::Dict, family::String; output_dir::String="results/alg4", n_snapshots::Int=4)
    mkpath(output_dir)

    X_series = data["X_true_series"]
    T = length(X_series)
    n = size(X_series[1], 1)

    timesteps = unique(round.(Int, range(1, T, length=n_snapshots)))

    fig = Figure(size=(300 * length(timesteps), 700))

    for (idx, t) in enumerate(timesteps)
        X = X_series[t]
        P = X * X'

        # Raw P
        ax1 = Axis(fig[1, idx],
            title="t = " * string(t) * " (raw)",
            aspect=DataAspect(),
            yreversed=true
        )
        hm1 = heatmap!(ax1, P, colormap=:RdBu, colorrange=(-0.2, 1.2))
        hidedecorations!(ax1)

        # Clamped P
        P_clamped = clamp.(P, 0.0, 1.0)
        ax2 = Axis(fig[2, idx],
            title="t = " * string(t) * " (clamped)",
            aspect=DataAspect(),
            yreversed=true
        )
        hm2 = heatmap!(ax2, P_clamped, colormap=:viridis, colorrange=(0, 1))
        hidedecorations!(ax2)

        if idx == length(timesteps)
            Colorbar(fig[1, idx+1], hm1, label="P (raw)")
            Colorbar(fig[2, idx+1], hm2, label="P (clamped)")
        end
    end

    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " - Raw vs Clamped P(t)",
          fontsize=16)

    filename = joinpath(output_dir, family * "_P_heatmaps_comparison.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_summary(data, family; output_dir="results/alg4")

Create a single-page summary plot with key information.
"""
function plot_summary(data::Dict, family::String; output_dir::String="results/alg4")
    mkpath(output_dir)

    X_series = data["X_true_series"]
    spectral_gaps = data["spectral_gaps"]
    prob_stats = data["probability_stats"]
    config = data["config"]

    n = size(X_series[1], 1)
    T = length(X_series)
    dt = config["dt"]
    times = range(0, step=dt, length=T)

    fig = Figure(size=(1400, 900))

    # Title with config info
    config_str = "n=" * string(config["n"]) * ", d=" * string(config["d"]) *
                 ", T=" * string(config["T"]) * ", dt=" * string(config["dt"])
    Label(fig[0, :], uppercase(family[1]) * family[2:end] * " Dynamics Summary (" * config_str * ")",
          fontsize=20)

    # Phase portrait
    ax1 = Axis(fig[1, 1:2],
        xlabel="Dim 1", ylabel="Dim 2",
        title="Phase Portrait",
        aspect=DataAspect()
    )
    colors = get_distinguishable_colors(n)
    for node in 1:n
        traj_x = [X_series[t][node, 1] for t in 1:T]
        traj_y = [X_series[t][node, 2] for t in 1:T]
        lines!(ax1, traj_x, traj_y, color=(colors[node], 0.5), linewidth=1.2)
        scatter!(ax1, [traj_x[1]], [traj_y[1]], color=colors[node], markersize=6)
        scatter!(ax1, [traj_x[end]], [traj_y[end]], color=colors[node], markersize=6, marker=:star5)
    end

    # P validity over time
    min_P = [s.min_P for s in prob_stats]
    max_P = [s.max_P for s in prob_stats]

    ax2 = Axis(fig[1, 3],
        xlabel="Time", ylabel="P value",
        title="P(t) Range"
    )
    band!(ax2, collect(times), min_P, max_P, color=(:blue, 0.3))
    lines!(ax2, collect(times), min_P, color=:blue, linewidth=1.5, label="min")
    lines!(ax2, collect(times), max_P, color=:blue, linewidth=1.5, label="max")
    hlines!(ax2, [0.0, 1.0], color=:red, linestyle=:dash, linewidth=1)

    # Spectral gap
    ax3 = Axis(fig[2, 1],
        xlabel="Time", ylabel="λ_d",
        title="Spectral Gap"
    )
    lines!(ax3, collect(times), spectral_gaps, color=:green, linewidth=2)

    # P heatmaps at t=1 and t=T
    X_start = X_series[1]
    X_end = X_series[end]
    P_start = X_start * X_start'
    P_end = X_end * X_end'

    ax4 = Axis(fig[2, 2],
        title="P(t=0)",
        aspect=DataAspect(),
        yreversed=true
    )
    heatmap!(ax4, P_start, colormap=:viridis, colorrange=(0, 1))
    hidedecorations!(ax4)

    ax5 = Axis(fig[2, 3],
        title="P(t=T)",
        aspect=DataAspect(),
        yreversed=true
    )
    hm = heatmap!(ax5, P_end, colormap=:viridis, colorrange=(0, 1))
    hidedecorations!(ax5)
    Colorbar(fig[2, 4], hm, label="P")

    # Summary statistics text
    min_gap = round(minimum(spectral_gaps), digits=4)
    mean_gap = round(mean(spectral_gaps), digits=4)
    overall_min_P = round(minimum(min_P), digits=4)
    overall_max_P = round(maximum(max_P), digits=4)
    total_neg = sum([s.neg_count for s in prob_stats])
    total_above = sum([s.above_one_count for s in prob_stats])

    stats_text = "Spectral gap: min=" * string(min_gap) * ", mean=" * string(mean_gap) * "\n" *
                 "P range: [" * string(overall_min_P) * ", " * string(overall_max_P) * "]\n" *
                 "Violations: " * string(total_neg) * " neg, " * string(total_above) * " >1"

    Label(fig[3, :], stats_text, fontsize=14, halign=:left)

    filename = joinpath(output_dir, family * "_summary.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

"""
    plot_all_families_comparison(; output_dir="results/alg4")

Create a comparison plot showing all three dynamics families side by side.
"""
function plot_all_families_comparison(; output_dir::String="results/alg4")
    mkpath(output_dir)

    families = ["linear", "polynomial", "message_passing"]
    family_names = ["Linear Symmetric", "Polynomial in P", "Message-Passing"]

    # Load all data
    all_data = Dict{String, Dict}()
    for family in families
        try
            all_data[family] = load_data(family)
        catch e
            println("Warning: Could not load " * family * " data: " * string(e))
        end
    end

    if isempty(all_data)
        error("No data files found. Run the generator scripts first.")
    end

    fig = Figure(size=(1500, 500 * length(all_data)))

    row = 1
    for (family, fname) in zip(families, family_names)
        if !haskey(all_data, family)
            continue
        end

        data = all_data[family]
        X_series = data["X_true_series"]
        prob_stats = data["probability_stats"]
        config = data["config"]
        n = size(X_series[1], 1)
        T = length(X_series)
        dt = config["dt"]
        times = range(0, step=dt, length=T)

        # Phase portrait
        ax1 = Axis(fig[row, 1],
            xlabel="Dim 1", ylabel="Dim 2",
            title=fname * " - Trajectories",
            aspect=DataAspect()
        )
        colors = get_distinguishable_colors(n)
        for node in 1:n
            traj_x = [X_series[t][node, 1] for t in 1:T]
            traj_y = [X_series[t][node, 2] for t in 1:T]
            lines!(ax1, traj_x, traj_y, color=(colors[node], 0.5), linewidth=1.0)
        end

        # P validity
        min_P = [s.min_P for s in prob_stats]
        max_P = [s.max_P for s in prob_stats]

        ax2 = Axis(fig[row, 2],
            xlabel="Time", ylabel="P value",
            title=fname * " - P(t) Range"
        )
        band!(ax2, collect(times), min_P, max_P, color=(:blue, 0.3))
        lines!(ax2, collect(times), min_P, color=:blue, linewidth=1.5)
        lines!(ax2, collect(times), max_P, color=:blue, linewidth=1.5)
        hlines!(ax2, [0.0, 1.0], color=:red, linestyle=:dash, linewidth=1)

        # P heatmap at final time
        X_end = X_series[end]
        P_end = clamp.(X_end * X_end', 0.0, 1.0)

        ax3 = Axis(fig[row, 3],
            title=fname * " - P(t=T)",
            aspect=DataAspect(),
            yreversed=true
        )
        hm = heatmap!(ax3, P_end, colormap=:viridis, colorrange=(0, 1))
        hidedecorations!(ax3)

        if row == 1
            Colorbar(fig[1:length(all_data), 4], hm, label="P")
        end

        row += 1
    end

    filename = joinpath(output_dir, "all_families_comparison.pdf")
    save(filename, fig)
    println("Saved: " * filename)

    return fig
end

# =============================================================================
# Main Entry Point
# =============================================================================

"""
    generate_all_plots(family::String; output_dir="results/alg4")

Generate all plots for a given dynamics family.
"""
function generate_all_plots(family::String; output_dir::String="results/alg4")
    println("\n" * "="^60)
    println("Generating plots for: " * family)
    println("="^60)

    data = load_data(family)

    # Print summary statistics
    config = data["config"]
    prob_stats = data["probability_stats"]
    spectral_gaps = data["spectral_gaps"]

    println("\nConfiguration:")
    println("  n = " * string(config["n"]))
    println("  d = " * string(config["d"]))
    println("  T = " * string(config["T"]))
    println("  dt = " * string(config["dt"]))

    min_P = minimum([s.min_P for s in prob_stats])
    max_P = maximum([s.max_P for s in prob_stats])
    total_neg = sum([s.neg_count for s in prob_stats])
    total_above = sum([s.above_one_count for s in prob_stats])

    println("\nDiagnostics:")
    println("  Spectral gap: min=" * string(round(minimum(spectral_gaps), digits=4)) *
            ", mean=" * string(round(mean(spectral_gaps), digits=4)))
    println("  P range: [" * string(round(min_P, digits=4)) * ", " * string(round(max_P, digits=4)) * "]")
    println("  P violations: " * string(total_neg) * " negative, " * string(total_above) * " above 1")

    # Generate plots
    println("\nGenerating plots...")
    plot_phase_portrait(data, family; output_dir=output_dir)
    plot_phase_portrait_with_embeddings(data, family; output_dir=output_dir)
    plot_phase_portrait_methods(data, family; output_dir=output_dir)
    plot_diagnostics_timeseries(data, family; output_dir=output_dir)
    plot_P_heatmaps(data, family; output_dir=output_dir)
    plot_P_heatmaps_clamped(data, family; output_dir=output_dir)
    plot_summary(data, family; output_dir=output_dir)

    println("\nDone with " * family)
end

function main()
    output_dir = joinpath("results", "alg4")
    mkpath(output_dir)

    # Parse command line argument
    if length(ARGS) > 0
        family = ARGS[1]
    else
        family = "all"
    end

    families = ["linear", "polynomial", "message_passing"]

    if family == "all"
        for f in families
            try
                generate_all_plots(f; output_dir=output_dir)
            catch e
                println("Error processing " * f * ": " * string(e))
            end
        end

        # Generate comparison plot
        println("\n" * "="^60)
        println("Generating comparison plot")
        println("="^60)
        try
            plot_all_families_comparison(; output_dir=output_dir)
        catch e
            println("Error generating comparison: " * string(e))
        end
    elseif family in families
        generate_all_plots(family; output_dir=output_dir)
    else
        println("Unknown family: " * family)
        println("Valid options: " * join(families, ", ") * ", all")
        exit(1)
    end

    println("\n" * "="^60)
    println("All plots saved to: " * output_dir)
    println("="^60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
