#!/usr/bin/env -S julia --project
"""
Example 1: Bridge Node Between Two Communities

Setup:
- Two communities (30+ nodes each) clustered around centers A₁ and A₂
- One "bridge" node that oscillates between communities

Comparison:
- ABSOLUTE: Bridge attracted to fixed points A₁, A₂
- EQUIVARIANT: Bridge attracted to community centroids X̄₁, X̄₂

Usage:
    julia --project scripts/example1_bridge_node_v2.jl generate
    julia --project scripts/example1_bridge_node_v2.jl estimate
    julia --project scripts/example1_bridge_node_v2.jl train [model_name]
    julia --project scripts/example1_bridge_node_v2.jl evaluate
    julia --project scripts/example1_bridge_node_v2.jl visualize
    julia --project scripts/example1_bridge_node_v2.jl all
    julia --project scripts/example1_bridge_node_v2.jl status
"""

using RDPGDynamics

# Additional imports needed for dynamics definitions
using LinearAlgebra: norm
using Statistics: mean
using Random
using Lux
using ComponentArrays: ComponentArray

const EXAMPLE_NAME = "example1_bridge"

# =============================================================================
# Configuration
# =============================================================================

# Network structure
const N_COMMUNITY_1 = 35
const N_COMMUNITY_2 = 35
const N_BRIDGE = 1
const N_TOTAL = N_COMMUNITY_1 + N_COMMUNITY_2 + N_BRIDGE

const D = 2  # Embedding dimension

# Community centers (fixed points for absolute dynamics)
const CENTER_1 = [0.3, 0.7]
const CENTER_2 = [0.7, 0.3]

# Community indices
const COMMUNITY_1 = 1:N_COMMUNITY_1
const COMMUNITY_2 = (N_COMMUNITY_1 + 1):(N_COMMUNITY_1 + N_COMMUNITY_2)
const BRIDGE_IDX = N_TOTAL

# Dynamics parameters
const K_COHESION = 0.15      # Community cohesion strength
const K_BRIDGE = 0.08        # Bridge attraction strength
const K_SWITCH = 5.0         # Switching sharpness
const OSCILLATION_PERIOD = 12.0  # Time for one oscillation cycle

# Time parameters
const T_END = 50.0
const DT = 1.0

# Training
const K_SAMPLES = 100
const TRAIN_FRACTION = 0.75
const EPOCHS_ADAM = 600
const EPOCHS_BFGS = 100

# =============================================================================
# True Dynamics
# =============================================================================

"""
True dynamics for the bridge node system.

Community nodes: Cohesion toward their center
Bridge node: Oscillates between A₁ and A₂ with smooth switching
"""
function true_dynamics!(du, u, p, t)
    X = reshape(u, N_TOTAL, D)
    dX = zeros(N_TOTAL, D)

    # Compute community centroids (for reference, not used in absolute dynamics)
    X_bar1 = mean(X[COMMUNITY_1, :], dims=1)[1, :]
    X_bar2 = mean(X[COMMUNITY_2, :], dims=1)[1, :]

    # Community 1: cohesion toward CENTER_1
    for i in COMMUNITY_1
        delta = X[i, :] .- CENTER_1
        dX[i, :] = -K_COHESION .* delta
    end

    # Community 2: cohesion toward CENTER_2
    for i in COMMUNITY_2
        delta = X[i, :] .- CENTER_2
        dX[i, :] = -K_COHESION .* delta
    end

    # Bridge node: oscillates between centers
    # Use sinusoidal switching for smooth oscillation
    phase = sin(2π * t / OSCILLATION_PERIOD)
    weight_1 = 0.5 * (1 + phase)  # 0 to 1
    weight_2 = 1 - weight_1       # 1 to 0

    bridge_pos = X[BRIDGE_IDX, :]
    target = weight_1 .* CENTER_1 .+ weight_2 .* CENTER_2
    dX[BRIDGE_IDX, :] = K_BRIDGE .* (target .- bridge_pos)

    # Soft boundary to keep in B^d_+
    for i in 1:N_TOTAL
        for j in 1:D
            if X[i, j] < 0.1
                dX[i, j] += 0.3 * exp(-(X[i, j] - 0.05) / 0.02)
            end
            if X[i, j] > 0.9
                dX[i, j] -= 0.3 * exp((X[i, j] - 0.95) / 0.02)
            end
        end
    end

    du .= vec(dX)
end

"""Generate initial positions."""
function generate_initial_positions(seed::Int=42)
    rng = Random.MersenneTwister(seed)
    X0 = zeros(N_TOTAL, D)

    # Community 1: around CENTER_1 with small spread
    for i in COMMUNITY_1
        X0[i, :] = CENTER_1 .+ 0.08 .* randn(rng, D)
    end

    # Community 2: around CENTER_2 with small spread
    for i in COMMUNITY_2
        X0[i, :] = CENTER_2 .+ 0.08 .* randn(rng, D)
    end

    # Bridge: start near CENTER_1
    X0[BRIDGE_IDX, :] = CENTER_1 .+ 0.05 .* randn(rng, D)

    # Clamp to B^d_+
    X0 = clamp.(X0, 0.1, 0.9)

    return X0
end

# =============================================================================
# Neural ODE Dynamics (for training)
# =============================================================================

"""
Make ABSOLUTE dynamics: bridge attracted to fixed A₁, A₂.
Known: oscillation structure, community cohesion
Unknown: exact strengths, boundary terms
"""
function make_ude_absolute_dynamics(nn, st, n::Int, d::Int)
    nd = n * d
    center_1_f32 = F.(CENTER_1)
    center_2_f32 = F.(CENTER_2)

    function dynamics(u::AbstractVector{T}, p, t) where T
        X = reshape(u, n, d)

        # Known: Community cohesion (simplified - all toward their respective centers)
        dX_known = zeros(T, n, d)

        for i in COMMUNITY_1
            delta = X[i, :] .- T.(center_1_f32)
            dX_known[i, :] = -T(K_COHESION) .* delta
        end

        for i in COMMUNITY_2
            delta = X[i, :] .- T.(center_2_f32)
            dX_known[i, :] = -T(K_COHESION) .* delta
        end

        # Bridge: base attraction (NN learns the switching)
        # Known structure: attracted somewhere between centers
        mid = (T.(center_1_f32) .+ T.(center_2_f32)) ./ T(2)
        dX_known[BRIDGE_IDX, :] = T(K_BRIDGE) .* (mid .- X[BRIDGE_IDX, :])

        # NN correction for everything
        u_reshape = reshape(u, nd, 1)
        correction, _ = nn(u_reshape, p, st)

        return vec(dX_known) .+ vec(correction)
    end
    return dynamics
end

"""
Make EQUIVARIANT dynamics: bridge attracted to community centroids.
This is rotation-equivariant: only depends on relative positions.
"""
function make_ude_equivariant_dynamics(nn, st, n::Int, d::Int)
    nd = n * d

    function dynamics(u::AbstractVector{T}, p, t) where T
        X = reshape(u, n, d)

        # Compute community centroids (data-dependent, equivariant)
        X_bar1 = mean(X[COMMUNITY_1, :], dims=1)
        X_bar2 = mean(X[COMMUNITY_2, :], dims=1)

        dX_known = zeros(T, n, d)

        # Community 1: cohesion toward own centroid
        for i in COMMUNITY_1
            delta = X[i, :] .- vec(X_bar1)
            dX_known[i, :] = -T(K_COHESION) .* delta
        end

        # Community 2: cohesion toward own centroid
        for i in COMMUNITY_2
            delta = X[i, :] .- vec(X_bar2)
            dX_known[i, :] = -T(K_COHESION) .* delta
        end

        # Bridge: known to be attracted to both centroids (NN learns weights)
        mid = (vec(X_bar1) .+ vec(X_bar2)) ./ T(2)
        dX_known[BRIDGE_IDX, :] = T(K_BRIDGE) .* (mid .- X[BRIDGE_IDX, :])

        # NN correction
        u_reshape = reshape(u, nd, 1)
        correction, _ = nn(u_reshape, p, st)

        return vec(dX_known) .+ vec(correction)
    end
    return dynamics
end

"""
Pure Neural ODE: NN learns everything.
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
        "communities" => [collect(COMMUNITY_1), collect(COMMUNITY_2), [BRIDGE_IDX]],
        "center_1" => CENTER_1,
        "center_2" => CENTER_2,
        "bridge_idx" => BRIDGE_IDX
    )

    phase_generate(EXAMPLE_NAME, true_dynamics!, X0;
                   T_end=T_END, dt=DT, metadata=metadata, seed=42)
end

function run_estimate()
    phase_estimate(EXAMPLE_NAME; K=K_SAMPLES)
end

function run_train(model_name::String)
    # Load data to get dimensions
    true_data = load_data(EXAMPLE_NAME, FILES.true_dynamics)
    n, d = true_data["n"], true_data["d"]
    nd = n * d

    rng = Random.Xoshiro(42)

    if model_name == "pure_nn"
        nn = build_nn(nd, [64, 64, 32], nd; rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_pure_nn_dynamics(nn, st, n, d)

    elseif model_name == "ude_absolute"
        nn = build_correction_nn(nd, nd; hidden_size=48, rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_ude_absolute_dynamics(nn, st, n, d)

    elseif model_name == "ude_equivariant"
        nn = build_correction_nn(nd, nd; hidden_size=48, rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_ude_equivariant_dynamics(nn, st, n, d)

    else
        error("Unknown model: " * model_name * ". Use: pure_nn, ude_absolute, ude_equivariant")
    end

    phase_train(EXAMPLE_NAME, model_name, dynamics_fn, ps_init;
                train_fraction=TRAIN_FRACTION,
                epochs_adam=EPOCHS_ADAM, epochs_bfgs=EPOCHS_BFGS,
                lr=0.01f0, dt=F(DT))
end

function run_evaluate()
    model_names = ["pure_nn", "ude_absolute", "ude_equivariant"]
    phase_evaluate(EXAMPLE_NAME, model_names)
end

function run_visualize()
    model_names = ["pure_nn", "ude_absolute", "ude_equivariant"]
    phase_visualize(EXAMPLE_NAME, model_names)
end

# =============================================================================
# Main Entry Point (Julia 1.11+)
# =============================================================================

function print_usage()
    println("Usage: julia --project scripts/example1_bridge_node_v2.jl <command>")
    println("\nCommands:")
    println("  generate   - Generate true dynamics")
    println("  estimate   - RDPG estimation")
    println("  train <model>  - Train model (pure_nn, ude_absolute, ude_equivariant)")
    println("  evaluate   - Evaluate all trained models")
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
            println("Specify model: pure_nn, ude_absolute, ude_equivariant")
            println("  Or use 'train all' to train all models")
            return 1
        end
        model = args[2]
        if model == "all"
            for m in ["pure_nn", "ude_absolute", "ude_equivariant"]
                run_train(m)
            end
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
        for m in ["pure_nn", "ude_absolute", "ude_equivariant"]
            run_train(m)
        end
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
