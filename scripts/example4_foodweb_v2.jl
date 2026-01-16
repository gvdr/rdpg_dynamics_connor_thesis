#!/usr/bin/env -S julia --project
"""
Example 4: Food Web with Heterogeneous Node Types (d=4)

Setup:
- 3 node types: Predator, Prey, Resource
- d=4 dimensional embedding
- Equivariant dynamics using type centroids (not fixed targets)

Dynamics:
- Predators: cohesion + attracted to prey centroid (hunting)
- Prey: cohesion + repelled by predator centroid (fleeing) + attracted to resource centroid
- Resources: cohesion + slow regrowth

This is rotation-equivariant: all interactions are relative to type centroids.

Usage:
    julia --project scripts/example4_foodweb_v2.jl generate
    julia --project scripts/example4_foodweb_v2.jl estimate
    julia --project scripts/example4_foodweb_v2.jl train <model>
    julia --project scripts/example4_foodweb_v2.jl evaluate
    julia --project scripts/example4_foodweb_v2.jl visualize
    julia --project scripts/example4_foodweb_v2.jl all
    julia --project scripts/example4_foodweb_v2.jl status
"""

using RDPGDynamics

# Additional imports needed for dynamics definitions
using LinearAlgebra: norm
using Statistics: mean
using Random
using Lux
using ComponentArrays: ComponentArray

const EXAMPLE_NAME = "example4_foodweb"

# =============================================================================
# Configuration
# =============================================================================

const N_PREDATORS = 15
const N_PREY = 25
const N_RESOURCES = 20
const N_TOTAL = N_PREDATORS + N_PREY + N_RESOURCES

const D = 4  # Higher dimensional embedding

# Node type indices
const TYPE_PREDATOR = 1
const TYPE_PREY = 2
const TYPE_RESOURCE = 3

# Node index ranges
const PREDATORS = 1:N_PREDATORS
const PREY = (N_PREDATORS + 1):(N_PREDATORS + N_PREY)
const RESOURCES = (N_PREDATORS + N_PREY + 1):N_TOTAL

# Dynamics parameters
const K_COHESION = 0.08        # Within-type cohesion
const K_HUNT = 0.04            # Predator → prey attraction
const K_FLEE = 0.06            # Prey → predator repulsion
const K_FEED = 0.03            # Prey → resource attraction
const K_REGROW = 0.02          # Resource regrowth toward center

# Time parameters
const T_END = 50.0
const DT = 1.0

# Training
const K_SAMPLES = 100
const TRAIN_FRACTION = 0.75
const EPOCHS_ADAM = 500
const EPOCHS_BFGS = 100

# =============================================================================
# True Dynamics (Equivariant: uses type centroids)
# =============================================================================

function get_node_type(i::Int)
    if i in PREDATORS
        return TYPE_PREDATOR
    elseif i in PREY
        return TYPE_PREY
    else
        return TYPE_RESOURCE
    end
end

"""
True dynamics: equivariant food web.

All interactions are relative to type centroids (no fixed targets).
This is rotation-equivariant.
"""
function true_dynamics!(du, u, p, t)
    X = reshape(u, N_TOTAL, D)
    dX = zeros(N_TOTAL, D)

    # Compute type centroids
    X_pred = mean(X[PREDATORS, :], dims=1)[1, :]
    X_prey = mean(X[PREY, :], dims=1)[1, :]
    X_res = mean(X[RESOURCES, :], dims=1)[1, :]

    # Global centroid for boundary reference
    X_global = mean(X, dims=1)[1, :]

    for i in 1:N_TOTAL
        node_type = get_node_type(i)

        if node_type == TYPE_PREDATOR
            # Cohesion with other predators
            dX[i, :] = -K_COHESION .* (X[i, :] .- X_pred)
            # Hunt: attracted to prey centroid
            dX[i, :] .+= K_HUNT .* (X_prey .- X[i, :])

        elseif node_type == TYPE_PREY
            # Cohesion with other prey
            dX[i, :] = -K_COHESION .* (X[i, :] .- X_prey)
            # Flee: repelled by predator centroid
            dir_flee = X[i, :] .- X_pred
            d_flee = norm(dir_flee)
            if d_flee > 0.01
                dX[i, :] .+= K_FLEE .* dir_flee ./ d_flee
            end
            # Feed: attracted to resource centroid
            dX[i, :] .+= K_FEED .* (X_res .- X[i, :])

        else  # TYPE_RESOURCE
            # Cohesion with other resources
            dX[i, :] = -K_COHESION .* (X[i, :] .- X_res)
            # Regrowth: drift toward global center (stability)
            dX[i, :] .+= K_REGROW .* (X_global .- X[i, :])
        end
    end

    # Soft boundary for B^d_+
    for i in 1:N_TOTAL
        for j in 1:D
            if X[i, j] < 0.1
                dX[i, j] += 0.25 * exp(-(X[i, j] - 0.05) / 0.02)
            end
            if X[i, j] > 0.9
                dX[i, j] -= 0.25 * exp((X[i, j] - 0.95) / 0.02)
            end
        end

        # Norm constraint (stay in unit ball)
        r = norm(X[i, :])
        if r > 0.85
            dX[i, :] .-= 0.2 * (r - 0.8) .* X[i, :] ./ r
        end
    end

    du .= vec(dX)
end

"""Generate initial positions for food web."""
function generate_initial_positions(seed::Int)
    rng = Random.MersenneTwister(seed)
    X0 = zeros(N_TOTAL, D)

    # Initialize in different regions of B^d_+ (with overlap)
    # Predators: higher in dims 1,2
    for i in PREDATORS
        X0[i, :] = [0.6, 0.5, 0.4, 0.4] .+ 0.1 .* randn(rng, D)
    end

    # Prey: middle region
    for i in PREY
        X0[i, :] = [0.45, 0.55, 0.5, 0.45] .+ 0.1 .* randn(rng, D)
    end

    # Resources: higher in dims 3,4
    for i in RESOURCES
        X0[i, :] = [0.4, 0.4, 0.6, 0.55] .+ 0.1 .* randn(rng, D)
    end

    # Ensure B^d_+
    X0 = max.(X0, 0.1)
    for i in 1:N_TOTAL
        r = norm(X0[i, :])
        if r > 0.85
            X0[i, :] .*= 0.85 / r
        end
    end

    return X0
end

# =============================================================================
# Neural ODE Dynamics
# =============================================================================

# One-hot encoding for node types (Zygote-compatible)
const TYPE_ONEHOTS = (
    F[1, 0, 0],  # Predator
    F[0, 1, 0],  # Prey
    F[0, 0, 1]   # Resource
)

"""
UDE: Known structure (centroid interactions) + NN corrections.
"""
function make_ude_dynamics(nn, st, n::Int, d::Int)
    nd = n * d

    function dynamics(u::AbstractVector{T}, p, t) where T
        X = reshape(u, n, d)

        # Compute type centroids
        X_pred = mean(X[PREDATORS, :], dims=1)
        X_prey = mean(X[PREY, :], dims=1)
        X_res = mean(X[RESOURCES, :], dims=1)

        dX_known = zeros(T, n, d)

        for i in 1:n
            node_type = get_node_type(i)

            if node_type == TYPE_PREDATOR
                # Known: cohesion + general attraction to prey
                dX_known[i, :] = -T(K_COHESION) .* (X[i, :] .- vec(X_pred))
                dX_known[i, :] .+= T(K_HUNT * 0.5) .* (vec(X_prey) .- X[i, :])

            elseif node_type == TYPE_PREY
                # Known: cohesion
                dX_known[i, :] = -T(K_COHESION) .* (X[i, :] .- vec(X_prey))

            else  # Resource
                # Known: cohesion
                dX_known[i, :] = -T(K_COHESION) .* (X[i, :] .- vec(X_res))
            end
        end

        # NN learns flee, feed, and corrections
        u_reshape = reshape(u, nd, 1)
        correction, _ = nn(u_reshape, p, st)

        return vec(dX_known) .+ vec(correction)
    end
    return dynamics
end

"""
Pure NN: learns everything.
"""
function make_pure_nn_dynamics(nn, st, n::Int, d::Int)
    nd = n * d

    function dynamics(u::AbstractVector{T}, p, t) where T
        u_reshape = reshape(u, nd, 1)
        out, _ = nn(u_reshape, p, st)
        return vec(out)
    end
    return dynamics
end

# =============================================================================
# Phase Implementations
# =============================================================================

function run_generate()
    X0 = generate_initial_positions(42)

    metadata = Dict(
        "communities" => [collect(PREDATORS), collect(PREY), collect(RESOURCES)],
        "node_types" => Dict("predators" => PREDATORS, "prey" => PREY, "resources" => RESOURCES),
        "n_predators" => N_PREDATORS,
        "n_prey" => N_PREY,
        "n_resources" => N_RESOURCES
    )

    phase_generate(EXAMPLE_NAME, true_dynamics!, X0;
                   T_end=T_END, dt=DT, metadata=metadata, seed=42)
end

function run_estimate()
    phase_estimate(EXAMPLE_NAME; K=K_SAMPLES)
end

function run_train(model_name::String)
    true_data = load_data(EXAMPLE_NAME, FILES.true_dynamics)
    n, d = true_data["n"], true_data["d"]
    nd = n * d

    rng = Random.Xoshiro(42)

    if model_name == "pure_nn"
        # Larger network for d=4
        nn = build_nn(nd, [128, 96, 64], nd; rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_pure_nn_dynamics(nn, st, n, d)

    elseif model_name == "ude"
        nn = build_correction_nn(nd, nd; hidden_size=64, rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_ude_dynamics(nn, st, n, d)

    else
        error("Unknown model: " * model_name * ". Use: pure_nn, ude")
    end

    phase_train(EXAMPLE_NAME, model_name, dynamics_fn, ps_init;
                train_fraction=TRAIN_FRACTION,
                epochs_adam=EPOCHS_ADAM, epochs_bfgs=EPOCHS_BFGS,
                lr=0.01f0, dt=F(DT))
end

function run_evaluate()
    phase_evaluate(EXAMPLE_NAME, ["pure_nn", "ude"])
end

function run_visualize()
    # Custom visualization for d=4 (show 2D projections)
    println("\n" * "=" ^ 60)
    println("PHASE 5: Visualizations (d=4 projections)")
    println("=" ^ 60)

    true_data = load_data(EXAMPLE_NAME, FILES.true_dynamics)
    est_data = load_data(EXAMPLE_NAME, FILES.estimated)
    eval_data = load_data(EXAMPLE_NAME, "evaluation.jls")

    X_true_series = true_data["X_true_series"]
    X_est_series = est_data["X_est_series"]
    T_train = eval_data["T_train"]

    n_times = length(X_true_series)

    # Create figure with 2D projections
    fig = CM.Figure(size=(1600, 800))

    # Dim 1-2 projection
    ax1 = CM.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                  title="True: Dims 1-2", aspect=1)
    ax2 = CM.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                  title="Estimated: Dims 1-2", aspect=1)

    # Dim 3-4 projection
    ax3 = CM.Axis(fig[2, 1], xlabel="Dim 3", ylabel="Dim 4",
                  title="True: Dims 3-4", aspect=1)
    ax4 = CM.Axis(fig[2, 2], xlabel="Dim 3", ylabel="Dim 4",
                  title="Estimated: Dims 3-4", aspect=1)

    type_colors = [:red, :blue, :green]
    type_names = ["Predator", "Prey", "Resource"]
    type_ranges = [PREDATORS, PREY, RESOURCES]

    for (ax_true, ax_est, dims) in [(ax1, ax2, (1, 2)), (ax3, ax4, (3, 4))]
        d1, d2 = dims

        for (tidx, type_range) in enumerate(type_ranges)
            color = type_colors[tidx]

            # True trajectories
            for i in type_range
                xs = [X_true_series[t][i, d1] for t in 1:n_times]
                ys = [X_true_series[t][i, d2] for t in 1:n_times]
                CM.lines!(ax_true, xs, ys, color=(color, 0.3), linewidth=1)
            end

            # Estimated trajectories
            for i in type_range
                xs = [X_est_series[t][i, d1] for t in 1:n_times]
                ys = [X_est_series[t][i, d2] for t in 1:n_times]
                CM.lines!(ax_est, xs, ys, color=(color, 0.3), linewidth=1)
            end
        end
    end

    # Legend
    elements = [CM.LineElement(color=c, linewidth=2) for c in type_colors]
    CM.Legend(fig[1:2, 3], elements, type_names, "Node Type")

    path = results_path(EXAMPLE_NAME, "projections_true_vs_est.pdf")
    CM.save(path, fig)
    println("  Saved: " * path)

    # Also generate standard plots for models
    for model_name in ["pure_nn", "ude"]
        model_file = "model_" * model_name * ".jls"
        if !data_exists(EXAMPLE_NAME, model_file)
            continue
        end

        model_data = load_data(EXAMPLE_NAME, model_file)
        X_rec_series = model_data["X_rec_series"]

        fig = CM.Figure(size=(1200, 600))

        ax1 = CM.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                      title=model_name * " Recovered: Dims 1-2", aspect=1)
        ax2 = CM.Axis(fig[1, 2], xlabel="Dim 3", ylabel="Dim 4",
                      title=model_name * " Recovered: Dims 3-4", aspect=1)

        for (ax, dims) in [(ax1, (1, 2)), (ax2, (3, 4))]
            d1, d2 = dims
            for (tidx, type_range) in enumerate(type_ranges)
                color = type_colors[tidx]
                for i in type_range
                    xs = [X_rec_series[t][i, d1] for t in 1:n_times]
                    ys = [X_rec_series[t][i, d2] for t in 1:n_times]
                    CM.lines!(ax, xs, ys, color=(color, 0.3), linewidth=1)
                end
            end
        end

        CM.Legend(fig[1, 3], elements, type_names, "Node Type")

        path = results_path(EXAMPLE_NAME, "projections_" * model_name * ".pdf")
        CM.save(path, fig)
        println("  Saved: " * path)
    end

    # Metrics and learning curves
    metrics_dict = eval_data["metrics_dict"]
    losses_dict = Dict{String, Vector{F}}()

    for model_name in ["pure_nn", "ude"]
        model_file = "model_" * model_name * ".jls"
        if data_exists(EXAMPLE_NAME, model_file)
            model_data = load_data(EXAMPLE_NAME, model_file)
            losses_dict[model_name] = model_data["losses"]
        end
    end

    fig = plot_metrics(metrics_dict, T_train; title="Example 4 Food Web - Metrics")
    CM.save(results_path(EXAMPLE_NAME, "metrics.pdf"), fig)
    println("  Saved: " * results_path(EXAMPLE_NAME, "metrics.pdf"))

    if !isempty(losses_dict)
        fig = plot_learning_curves(losses_dict; title="Example 4 - Learning Curves")
        CM.save(results_path(EXAMPLE_NAME, "learning_curves.pdf"), fig)
        println("  Saved: " * results_path(EXAMPLE_NAME, "learning_curves.pdf"))
    end

    println("\n  Visualizations complete!")
end

# =============================================================================
# Main Entry Point (Julia 1.11+)
# =============================================================================

function print_usage()
    println("Usage: julia --project scripts/example4_foodweb_v2.jl <command>")
    println("\nCommands:")
    println("  generate   - Generate true dynamics")
    println("  estimate   - RDPG estimation")
    println("  train <model>  - Train model (pure_nn, ude, or all)")
    println("  evaluate   - Evaluate models")
    println("  visualize  - Generate plots")
    println("  all        - Run full pipeline")
    println("  status     - Show pipeline status")
end

function (@main)(args)
    if isempty(args)
        print_usage()
        return 0
    end

    command = args[1]

    if command == "generate"
        run_generate()

    elseif command == "estimate"
        run_estimate()

    elseif command == "train"
        if length(args) < 2
            println("Specify model: pure_nn, ude, or all")
            return 1
        end
        model = args[2]
        if model == "all"
            run_train("pure_nn")
            run_train("ude")
        else
            run_train(model)
        end

    elseif command == "evaluate"
        run_evaluate()

    elseif command == "visualize"
        run_visualize()

    elseif command == "all"
        run_generate()
        run_estimate()
        run_train("pure_nn")
        run_train("ude")
        run_evaluate()
        run_visualize()

    elseif command == "status"
        print_status(EXAMPLE_NAME)

    else
        println("Unknown command: " * command)
        return 1
    end

    return 0
end
