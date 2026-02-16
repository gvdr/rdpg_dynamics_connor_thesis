#!/usr/bin/env -S julia --project
"""
UDE Pipeline Experiment — Phase 3: Figures

Generates a 2×2 figure for the paper:
  (a) 3D trajectories: true (gray) + anchor-aligned ASE (blue) in B+³
  (b) NN residual accuracy: learned f_u vs true f_u scatter per condition
  (c) SymReg: Pareto frontier curves per condition
  (d) Recovered γ̂ per condition (bar chart with error bars across reps)

Output: paper/plots/ude-pipeline-results.png
"""

using Serialization
using Statistics
using LinearAlgebra
using CairoMakie: Figure, Axis, Axis3, lines!, scatter!, band!, barplot!,
    errorbars!, Label, Legend, LineElement, MarkerElement, PolyElement,
    save, current_figure, text!, axislegend, hlines!, ablines!,
    Colorbar, hexbin!, GridLayout

# =============================================================================
# Paths
# =============================================================================

const DATA_DIR = joinpath("data", "ude_experiment")
const RESULTS_PATH = joinpath("results", "ude_experiment", "ude_results.jls")
const PLOT_DIR = joinpath("paper", "plots")

# =============================================================================
# Colors
# =============================================================================

const C_ANCHOR = :steelblue
const C_SEQ = :coral
const C_UNALIGNED = :gray60
const C_TRUE = :gray40

# =============================================================================
# Load data
# =============================================================================

function load_results()
    if !isfile(RESULTS_PATH)
        error("Results file not found: " * RESULTS_PATH *
              "\nRun scripts/ude_experiment/run_spiral_ude.jl first.")
    end
    return deserialize(RESULTS_PATH)
end

function load_example_data(rep::Int=1)
    path = joinpath(DATA_DIR, "rep" * string(rep) * ".jls")
    if !isfile(path)
        return nothing
    end
    return deserialize(path)
end

# =============================================================================
# Figure: UDE Pipeline Results (2×2)
# =============================================================================

function plot_ude_results(results::Dict, example_data)
    fig = Figure(size=(1000, 850), fontsize=12)

    # --- Panel (a): 3D trajectories ---
    ax_a = Axis3(fig[1, 1],
        xlabel="d₁", ylabel="d₂", zlabel="d₃",
        title="(a) Trajectories in B₊³",
        azimuth=0.4π, elevation=0.15π
    )

    if !isnothing(example_data)
        X_true = example_data["X_true_series"]
        X_aligned = example_data["X_anchor_aligned"]
        anchor_mask = example_data["anchor_mask"]
        n = size(X_true[1], 1)
        T = length(X_true)

        # Plot a subset of non-anchor true trajectories (gray)
        mobile_nodes = findall(.!anchor_mask)
        plot_nodes = mobile_nodes[1:min(20, length(mobile_nodes))]
        for node in plot_nodes
            traj_x = [X_true[t][node, 1] for t in 1:T]
            traj_y = [X_true[t][node, 2] for t in 1:T]
            traj_z = [X_true[t][node, 3] for t in 1:T]
            lines!(ax_a, traj_x, traj_y, traj_z,
                   color=(C_TRUE, 0.4), linewidth=0.8)
        end

        # Plot same nodes from anchor-aligned ASE (blue)
        for node in plot_nodes
            traj_x = [X_aligned[t][node, 1] for t in 1:T]
            traj_y = [X_aligned[t][node, 2] for t in 1:T]
            traj_z = [X_aligned[t][node, 3] for t in 1:T]
            lines!(ax_a, traj_x, traj_y, traj_z,
                   color=(C_ANCHOR, 0.3), linewidth=0.8)
        end

        # Plot anchor nodes as diamonds (at t=1)
        anchor_nodes = findall(anchor_mask)
        anchor_subset = anchor_nodes[1:min(15, length(anchor_nodes))]
        scatter!(ax_a,
                 [X_true[1][i, 1] for i in anchor_subset],
                 [X_true[1][i, 2] for i in anchor_subset],
                 [X_true[1][i, 3] for i in anchor_subset],
                 color=:firebrick, markersize=5, marker=:diamond)
    end

    # --- Panel (b): NN residual accuracy hexbin (3 subpanels) ---
    rep_keys = sort(collect(keys(results)))

    gb = GridLayout(fig[1, 2])
    Label(gb[0, 1:3], "(b) NN residual accuracy", fontsize=12, halign=:center)

    cond_names_b = ["anchor", "sequential", "unaligned"]
    cond_labels_b = ["Anchor", "Sequential", "Unaligned"]
    cond_colors_b = [C_ANCHOR, C_SEQ, C_UNALIGNED]

    if !isempty(rep_keys)
        rep1 = results[rep_keys[1]]

        for (col, cond) in enumerate(cond_names_b)
            show_ylabel = col == 1
            ax = Axis(gb[1, col],
                xlabel="True f_u",
                ylabel=show_ylabel ? "Learned f_u" : "",
                title=cond_labels_b[col],
                titlesize=10,
                aspect=1,
                yticklabelsvisible=show_ylabel
            )

            if haskey(rep1, cond)
                r = rep1[cond]
                f_true = Float64.(vec(r["f_true_samples"]))
                f_nn = Float64.(vec(r["f_nn_samples"]))

                hexbin!(ax, f_true, f_nn,
                        cellsize=0.02, colormap=:dense,
                        threshold=1)

                # Identity line
                ablines!(ax, 0.0, 1.0, color=:black, linestyle=:dash, linewidth=1)
            end
        end
    end

    # --- Panel (c): SymReg Pareto frontiers ---
    ax_c = Axis(fig[2, 1],
        xlabel="Expression complexity",
        ylabel="Loss (MSE)",
        title="(c) Symbolic regression Pareto front",
        yscale=log10
    )

    if !isempty(rep_keys)
        rep1 = results[rep_keys[1]]
        cond_colors = Dict("anchor" => C_ANCHOR, "sequential" => C_SEQ, "unaligned" => C_UNALIGNED)

        for cond in ["anchor", "sequential", "unaligned"]
            if haskey(rep1, cond) && haskey(rep1[cond], "sr_results")
                sr_list = rep1[cond]["sr_results"]
                # Use first dimension as representative
                if !isempty(sr_list)
                    pareto = sr_list[1]
                    complexities = pareto["complexities"]
                    losses = pareto["losses"]
                    if !isempty(complexities)
                        # Sort by complexity
                        perm = sortperm(complexities)
                        lines!(ax_c, complexities[perm], losses[perm],
                               color=cond_colors[cond], linewidth=2)
                        scatter!(ax_c, complexities[perm], losses[perm],
                                 color=cond_colors[cond], markersize=6,
                                 label=cond)
                    end
                end
            end
        end
        axislegend(ax_c, position=:rt, framevisible=false, labelsize=10)
    end

    # --- Panel (d): Total dynamics MSE bar chart ---
    ax_d = Axis(fig[2, 2],
        xlabel="Alignment condition",
        ylabel="Total dynamics MSE",
        title="(d) Dynamics recovery accuracy",
        xticks=(1:3, ["Anchor", "Sequential", "Unaligned"]),
        yscale=log10
    )

    if !isempty(rep_keys)
        cond_names = ["anchor", "sequential", "unaligned"]
        cond_colors_vec = [C_ANCHOR, C_SEQ, C_UNALIGNED]

        dyn_means = Float64[]
        dyn_stds = Float64[]

        for cond in cond_names
            vals = Float64[]
            for rep_key in rep_keys
                if haskey(results[rep_key], cond) && haskey(results[rep_key][cond], "total_dynamics_mse")
                    push!(vals, results[rep_key][cond]["total_dynamics_mse"])
                end
            end
            if !isempty(vals)
                push!(dyn_means, mean(vals))
                push!(dyn_stds, std(vals))
            else
                push!(dyn_means, NaN)
                push!(dyn_stds, 0.0)
            end
        end

        barplot!(ax_d, 1:3, dyn_means, color=cond_colors_vec, width=0.6)
        errorbars!(ax_d, Float64.(1:3), dyn_means, dyn_stds,
                   color=:black, whiskerwidth=10)
    end

    # Global legend
    Legend(fig[0, :],
        [LineElement(color=C_TRUE, linewidth=2),
         LineElement(color=C_ANCHOR, linewidth=2),
         LineElement(color=C_SEQ, linewidth=2, linestyle=:dash),
         LineElement(color=C_UNALIGNED, linewidth=2, linestyle=:dot),
         MarkerElement(color=:firebrick, marker=:diamond, markersize=8)],
        ["True trajectory", "Anchor-aligned", "Sequential Procrustes",
         "Unaligned", "Anchor nodes"],
        orientation=:horizontal, framevisible=false, labelsize=10,
        tellwidth=false, tellheight=true)

    mkpath(PLOT_DIR)
    out_path = joinpath(PLOT_DIR, "ude-pipeline-results.png")
    save(out_path, fig, px_per_unit=3)
    println("Saved: " * out_path)
    return fig
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("UDE Pipeline Experiment — Figure Generation")

    results = load_results()
    example_data = load_example_data(1)

    plot_ude_results(results, example_data)

    println("\nFigure saved to: " * PLOT_DIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
