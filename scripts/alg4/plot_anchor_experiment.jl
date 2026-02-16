#!/usr/bin/env -S julia --project
"""
Alg4 Anchor-Based Alignment Experiment — Phase 3: Figures for Paper

Generates two figures:
1. Main results (2×2 panels): anchor count, error vs T, drifting anchors, trajectory error timeseries
2. Spectral gap + phase portrait (1×2 panels)

Outputs PNGs to paper/plots/anchor-*.png
"""

using Serialization
using Statistics
using LinearAlgebra
using CairoMakie: Figure, Axis, lines!, scatter!, band!, vlines!, hlines!,
    Label, Legend, LineElement, MarkerElement, save, current_figure,
    Colorbar, PolyElement, text!, axislegend

# =============================================================================
# Load results
# =============================================================================

const RESULTS_PATH = joinpath("results", "alg4", "anchor_experiment", "anchor_experiment_results.jls")
const PLOT_DIR = joinpath("paper", "plots")

function load_results()
    if !isfile(RESULTS_PATH)
        error("Results file not found: " * RESULTS_PATH *
              "\nRun scripts/alg4/run_anchor_experiment.jl first.")
    end
    return deserialize(RESULTS_PATH)
end

# =============================================================================
# Color palette
# =============================================================================

const C_ANCHOR = :steelblue
const C_SEQ = :coral
const C_DRIFT = [:steelblue, :royalblue, :goldenrod, :darkorange, :firebrick]

# =============================================================================
# Figure 1: Main results (2×2)
# =============================================================================

function plot_main_results(results::Dict)
    fig = Figure(size=(900, 750), fontsize=12)

    # --- Panel (a): Error vs anchor count ---
    ax_a = Axis(fig[1, 1],
        xlabel="Number of anchors (n_a)",
        ylabel="Mean alignment error",
        title="(a) Alignment error vs. anchor count"
    )

    if haskey(results, "anchor_count")
        agg = results["anchor_count"]["aggregated"]
        ks = sort(collect(keys(agg)))

        na_vals = Float64.(ks)
        anchor_means = [agg[k]["anchor_mean"] for k in ks]
        anchor_stds = [agg[k]["anchor_std"] for k in ks]
        seq_means = [agg[k]["seq_mean"] for k in ks]
        seq_stds = [agg[k]["seq_std"] for k in ks]

        band!(ax_a, na_vals,
              anchor_means .- anchor_stds,
              anchor_means .+ anchor_stds,
              color=(C_ANCHOR, 0.2))
        lines!(ax_a, na_vals, anchor_means, color=C_ANCHOR, linewidth=2, label="Anchor")
        scatter!(ax_a, na_vals, anchor_means, color=C_ANCHOR, markersize=8)

        band!(ax_a, na_vals,
              seq_means .- seq_stds,
              seq_means .+ seq_stds,
              color=(C_SEQ, 0.2))
        lines!(ax_a, na_vals, seq_means, color=C_SEQ, linewidth=2, linestyle=:dash, label="Sequential")
        scatter!(ax_a, na_vals, seq_means, color=C_SEQ, markersize=8, marker=:utriangle)

        # Vertical dashed line at n_a = d = 2
        vlines!(ax_a, [2.0], color=:gray, linestyle=:dash, linewidth=1)
        text!(ax_a, 2.5, maximum(anchor_means) * 0.9, text="n_a = d", fontsize=10, color=:gray)
    end

    # --- Panel (b): Error vs T (log scale) ---
    ax_b = Axis(fig[1, 2],
        xlabel="Trajectory length T",
        ylabel="Mean alignment error",
        title="(b) Error accumulation over time",
        xscale=log10,
        yscale=log10
    )

    if haskey(results, "error_vs_T")
        agg = results["error_vs_T"]["aggregated"]
        ks = sort(collect(keys(agg)))

        T_vals = Float64.(ks)
        anchor_means = [agg[k]["anchor_mean"] for k in ks]
        anchor_stds = [agg[k]["anchor_std"] for k in ks]
        seq_means = [agg[k]["seq_mean"] for k in ks]
        seq_stds = [agg[k]["seq_std"] for k in ks]

        lines!(ax_b, T_vals, anchor_means, color=C_ANCHOR, linewidth=2, label="Anchor")
        scatter!(ax_b, T_vals, anchor_means, color=C_ANCHOR, markersize=8)

        lines!(ax_b, T_vals, seq_means, color=C_SEQ, linewidth=2, linestyle=:dash, label="Sequential")
        scatter!(ax_b, T_vals, seq_means, color=C_SEQ, markersize=8, marker=:utriangle)

        # Reference slope lines
        T_ref = T_vals
        # O(sqrt(T)) reference
        scale_sqrt = seq_means[1] / sqrt(T_vals[1])
        lines!(ax_b, T_ref, scale_sqrt .* sqrt.(T_ref),
               color=:gray, linestyle=:dot, linewidth=1)
        text!(ax_b, T_ref[end], scale_sqrt * sqrt(T_ref[end]) * 1.2,
              text="O(sqrt(T))", fontsize=9, color=:gray)
    end

    # --- Panel (c): Drifting anchors, error over time ---
    ax_c = Axis(fig[2, 1],
        xlabel="Time step t",
        ylabel="Alignment error",
        title="(c) Effect of anchor drift"
    )

    if haskey(results, "drifting")
        agg = results["drifting"]["aggregated"]
        ks = sort(collect(keys(agg)))

        for (idx, k) in enumerate(ks)
            a = agg[k]
            ts = 1:length(a["anchor_timeseries_mean"])
            c = idx <= length(C_DRIFT) ? C_DRIFT[idx] : :black

            band!(ax_c, collect(ts),
                  a["anchor_timeseries_mean"] .- a["anchor_timeseries_std"],
                  a["anchor_timeseries_mean"] .+ a["anchor_timeseries_std"],
                  color=(c, 0.15))
            lines!(ax_c, collect(ts), a["anchor_timeseries_mean"],
                   color=c, linewidth=1.5,
                   label="eps=" * string(k))
        end
    end

    # --- Panel (d): X-space trajectory error over time (T=200) ---
    ax_d = Axis(fig[2, 2],
        xlabel="Time step t",
        ylabel="Alignment error",
        title="(d) Trajectory error (T = 200)"
    )

    if haskey(results, "error_vs_T")
        agg = results["error_vs_T"]["aggregated"]
        # Use the longest trajectory for clearest demonstration
        T_key = maximum(collect(keys(agg)))
        a = agg[T_key]

        ts_anchor = collect(1:length(a["anchor_timeseries_mean"]))
        ts_seq = collect(1:length(a["seq_timeseries_mean"]))

        band!(ax_d, ts_anchor,
              a["anchor_timeseries_mean"] .- a["anchor_timeseries_std"],
              a["anchor_timeseries_mean"] .+ a["anchor_timeseries_std"],
              color=(C_ANCHOR, 0.2))
        lines!(ax_d, ts_anchor, a["anchor_timeseries_mean"],
               color=C_ANCHOR, linewidth=2, label="Anchor")

        band!(ax_d, ts_seq,
              a["seq_timeseries_mean"] .- a["seq_timeseries_std"],
              a["seq_timeseries_mean"] .+ a["seq_timeseries_std"],
              color=(C_SEQ, 0.2))
        lines!(ax_d, ts_seq, a["seq_timeseries_mean"],
               color=C_SEQ, linewidth=2, linestyle=:dash, label="Sequential")
    end

    # Legends
    Legend(fig[1, 3],
        [LineElement(color=C_ANCHOR, linewidth=2),
         LineElement(color=C_SEQ, linewidth=2, linestyle=:dash)],
        ["Anchor-based", "Sequential Procrustes"],
        framevisible=false, labelsize=10)

    mkpath(PLOT_DIR)
    out_path = joinpath(PLOT_DIR, "anchor-main-results.png")
    save(out_path, fig, px_per_unit=3)
    println("Saved: " * out_path)
    return fig
end

# =============================================================================
# Figure 2: Spectral gap + phase portrait
# =============================================================================

function plot_spectral_and_portrait(results::Dict)
    fig = Figure(size=(900, 380), fontsize=12)

    # --- Panel (a): Error vs spectral gap ---
    ax_a = Axis(fig[1, 1],
        xlabel="X0 norm scale",
        ylabel="Mean alignment error",
        title="(a) Alignment error vs. spectral gap"
    )

    if haskey(results, "spectral_gap")
        agg = results["spectral_gap"]["aggregated"]
        ks = sort(collect(keys(agg)))

        scales = Float64.(ks)
        anchor_means = [agg[k]["anchor_mean"] for k in ks]
        anchor_stds = [agg[k]["anchor_std"] for k in ks]
        seq_means = [agg[k]["seq_mean"] for k in ks]
        seq_stds = [agg[k]["seq_std"] for k in ks]

        band!(ax_a, scales,
              anchor_means .- anchor_stds,
              anchor_means .+ anchor_stds,
              color=(C_ANCHOR, 0.2))
        lines!(ax_a, scales, anchor_means, color=C_ANCHOR, linewidth=2, label="Anchor")
        scatter!(ax_a, scales, anchor_means, color=C_ANCHOR, markersize=8)

        band!(ax_a, scales,
              seq_means .- seq_stds,
              seq_means .+ seq_stds,
              color=(C_SEQ, 0.2))
        lines!(ax_a, scales, seq_means, color=C_SEQ, linewidth=2, linestyle=:dash, label="Sequential")
        scatter!(ax_a, scales, seq_means, color=C_SEQ, markersize=8, marker=:utriangle)

        axislegend(ax_a, position=:rt, framevisible=false, labelsize=10)
    end

    # --- Panel (b): Example phase portrait ---
    ax_b = Axis(fig[1, 2],
        xlabel="Dimension 1",
        ylabel="Dimension 2",
        title="(b) Example trajectory (n_a = 15)"
    )

    # Draw B+^2 boundary: quarter-circle arc + axes
    arc_theta = range(0, pi/2, length=100)
    arc_x = cos.(arc_theta)
    arc_y = sin.(arc_theta)
    lines!(ax_b, arc_x, arc_y, color=:black, linestyle=:dash, linewidth=1.2)
    lines!(ax_b, [0.0, 0.0], [0.0, 1.0], color=:black, linestyle=:dash, linewidth=1.2)
    lines!(ax_b, [0.0, 1.0], [0.0, 0.0], color=:black, linestyle=:dash, linewidth=1.2)

    # Try to load a single rep's raw data for a phase portrait
    example_path = joinpath("data", "alg4", "anchor_experiment", "anchor_count", "na15_rep1.jls")
    if isfile(example_path)
        data = deserialize(example_path)
        X_true = data["X_true_series"]
        anchor_mask = data["anchor_mask"]
        n = size(X_true[1], 1)
        T = length(X_true)

        # Plot non-anchor trajectories in gray
        for node in 1:n
            if !anchor_mask[node]
                traj_x = [X_true[t][node, 1] for t in 1:T]
                traj_y = [X_true[t][node, 2] for t in 1:T]
                lines!(ax_b, traj_x, traj_y, color=(:gray, 0.3), linewidth=0.5)
            end
        end

        # Plot anchor nodes as points (they are stationary, no lines needed)
        for node in 1:n
            if anchor_mask[node]
                scatter!(ax_b, [X_true[1][node, 1]], [X_true[1][node, 2]],
                         color=:firebrick, markersize=6, marker=:diamond)
            end
        end

        Legend(fig[1, 3],
            [LineElement(color=:black, linewidth=1.2, linestyle=:dash),
             LineElement(color=:gray, linewidth=1),
             MarkerElement(color=:firebrick, marker=:diamond, markersize=6)],
            ["B+^2 boundary", "Non-anchor trajectories", "Anchor nodes (fixed)"],
            framevisible=false, labelsize=10)
    else
        text!(ax_b, 0.5, 0.5, text="No example data", align=(:center, :center))
    end

    mkpath(PLOT_DIR)
    out_path = joinpath(PLOT_DIR, "anchor-spectral-portrait.png")
    save(out_path, fig, px_per_unit=3)
    println("Saved: " * out_path)
    return fig
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("Anchor-Based Alignment Experiment — Figure Generation")
    results = load_results()

    plot_main_results(results)
    plot_spectral_and_portrait(results)

    println("\nAll figures saved to: " * PLOT_DIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
