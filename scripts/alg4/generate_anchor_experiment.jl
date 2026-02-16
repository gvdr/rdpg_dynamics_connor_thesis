#!/usr/bin/env -S julia --project
"""
Alg4 Anchor-Based Alignment Experiment — Phase 1: Data Generation

Generates synthetic RDPG data under 5 experimental conditions:
1. Anchor count sweep (n_a ∈ [0,1,2,5,10,15,20,30])
2. Drifting anchors (ε ∈ [0.0, 0.001, 0.005, 0.01, 0.02])
3. Error vs trajectory length T
4. Spectral gap dependence (scale on X0 norms)

Each condition × parameter × repetition generates:
- X_true_series: ground-truth trajectory
- X_hat_series: ASE embeddings from noisy adjacencies
- config: experiment parameters

Outputs:
- data/alg4/anchor_experiment/<condition>/<param>_rep<i>.jls
"""

using Random
using LinearAlgebra
using Statistics
using Serialization
using OrdinaryDiffEq

include(joinpath(@__DIR__, "alg4_utils.jl"))
using .Alg4Utils

# =============================================================================
# Local helper functions (not modifying shared utils)
# =============================================================================

"""
    generate_uniform_Bd_plus(n, d; rng)

Generate n points uniformly in the positive unit ball B_+^d via rejection sampling.
"""
function generate_uniform_Bd_plus(n::Int, d::Int; rng::AbstractRNG=Random.default_rng())
    X = zeros(Float64, n, d)
    count = 0
    while count < n
        x = rand(rng, d)  # uniform [0,1]^d
        if norm(x) <= 1.0
            count += 1
            X[count, :] = x
        end
    end
    return X
end

"""
    anchor_polynomial_dynamics!(du, u, p, t; n, d, alpha, anchor_mask)

Polynomial dynamics Ẋ = (α₀I + α₁P)X with zeroed derivatives for anchor nodes.
anchor_mask is a BitVector: true = anchor (frozen).
"""
function anchor_polynomial_dynamics!(du::AbstractVector, u::AbstractVector, p, t;
                                     n::Int, d::Int, alpha::AbstractVector,
                                     anchor_mask::BitVector)
    X = reshape(u, n, d)
    P = X * X'
    N_mat = alpha[1] .* Matrix{eltype(X)}(I, n, n)
    Pk = P
    for k in 2:length(alpha)
        N_mat .+= alpha[k] .* Pk
        Pk = Pk * P
    end
    dX = N_mat * X
    # Zero out anchor derivatives
    for i in 1:n
        if anchor_mask[i]
            for j in 1:d
                dX[i, j] = 0.0
            end
        end
    end
    du .= vec(dX)
    return nothing
end

"""
    drifting_anchor_dynamics!(du, u, p, t; n, d, alpha, anchor_mask, drift_directions, epsilon)

Same as anchor polynomial dynamics but anchors drift at rate ε in given directions.
"""
function drifting_anchor_dynamics!(du::AbstractVector, u::AbstractVector, p, t;
                                   n::Int, d::Int, alpha::AbstractVector,
                                   anchor_mask::BitVector,
                                   drift_directions::Matrix{Float64},
                                   epsilon::Float64)
    X = reshape(u, n, d)
    P = X * X'
    N_mat = alpha[1] .* Matrix{eltype(X)}(I, n, n)
    Pk = P
    for k in 2:length(alpha)
        N_mat .+= alpha[k] .* Pk
        Pk = Pk * P
    end
    dX = N_mat * X
    # Override anchor derivatives with drift
    for i in 1:n
        if anchor_mask[i]
            for j in 1:d
                dX[i, j] = epsilon * drift_directions[i, j]
            end
        end
    end
    du .= vec(dX)
    return nothing
end

"""
    simulate_anchor_series(dynamics!, X0; T, dt)

Simulate ODE and return vector of X(t) matrices (same interface as Alg4Utils.simulate_series
but accepts any in-place dynamics closure).
"""
function simulate_anchor_series(dynamics!, X0::AbstractMatrix; T::Int=50, dt::Float64=0.05)
    n, d = size(X0)
    tspan = (0.0, dt * (T - 1))
    tsteps = range(tspan[1], tspan[2]; length=T)
    prob = ODEProblem((du, u, p, t) -> dynamics!(du, u, p, t), vec(X0), tspan)
    sol = solve(prob, Tsit5(); saveat=tsteps, abstol=1e-8, reltol=1e-8)
    return [reshape(sol.u[t], n, d) for t in 1:length(sol.u)]
end

# =============================================================================
# Shared configuration
# =============================================================================

const N = 200
const D = 2
const K_SAMPLES = 3           # few samples → noisier ASE → visible gauge drift
const ALPHA = [-0.3, 0.003]   # degree K=1: Ẋ = (α₀I + α₁P)X
                               # rate ≈ -0.3 + 0.003*66 ≈ -0.1 (stable, visible dynamics)
const N_REPS = 20
const BASE_SEED = 2024

const BASE_DIR = joinpath("data", "alg4", "anchor_experiment")

# =============================================================================
# Condition 1: Anchor count sweep
# =============================================================================

function generate_condition_anchor_count()
    println("\n" * "="^60)
    println("Condition 1: Anchor count sweep")
    println("="^60)

    T = 50
    dt = 0.05
    n_anchors_list = [0, 1, 2, 5, 10, 15, 20, 30]

    out_dir = joinpath(BASE_DIR, "anchor_count")
    mkpath(out_dir)

    for n_a in n_anchors_list
        for rep in 1:N_REPS
            rng = Random.MersenneTwister(BASE_SEED + rep)

            X0 = generate_uniform_Bd_plus(N, D; rng=rng)
            anchor_mask = falses(N)
            anchor_mask[1:n_a] .= true

            function dyn!(du, u, p, t)
                return anchor_polynomial_dynamics!(du, u, p, t;
                    n=N, d=D, alpha=ALPHA, anchor_mask=anchor_mask)
            end

            X_true_series = simulate_anchor_series(dyn!, X0; T=T, dt=dt)
            A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
            X_hat_series = ase_series(A_avg_series, D)

            config = Dict(
                "n" => N, "d" => D, "T" => T, "dt" => dt,
                "K_samples" => K_SAMPLES, "alpha" => ALPHA,
                "n_anchors" => n_a, "rep" => rep,
                "seed" => BASE_SEED + rep,
                "condition" => "anchor_count"
            )

            data = Dict(
                "config" => config,
                "X_true_series" => X_true_series,
                "X_hat_series" => X_hat_series,
                "anchor_mask" => anchor_mask
            )

            fname = "na" * string(n_a) * "_rep" * string(rep) * ".jls"
            serialize(joinpath(out_dir, fname), data)
        end
        println("  n_a=" * string(n_a) * ": " * string(N_REPS) * " reps done")
    end
end

# =============================================================================
# Condition 2: Drifting anchors
# =============================================================================

function generate_condition_drifting()
    println("\n" * "="^60)
    println("Condition 2: Drifting anchors")
    println("="^60)

    T = 50
    dt = 0.05
    n_a = 15
    epsilons = [0.0, 0.005, 0.01, 0.05, 0.1]

    out_dir = joinpath(BASE_DIR, "drifting")
    mkpath(out_dir)

    for eps_val in epsilons
        for rep in 1:N_REPS
            rng = Random.MersenneTwister(BASE_SEED + rep)

            X0 = generate_uniform_Bd_plus(N, D; rng=rng)
            anchor_mask = falses(N)
            anchor_mask[1:n_a] .= true

            # Random unit drift directions for anchors
            drift_directions = zeros(Float64, N, D)
            for i in 1:N
                if anchor_mask[i]
                    v = randn(rng, D)
                    drift_directions[i, :] = v ./ norm(v)
                end
            end

            function dyn!(du, u, p, t)
                return drifting_anchor_dynamics!(du, u, p, t;
                    n=N, d=D, alpha=ALPHA, anchor_mask=anchor_mask,
                    drift_directions=drift_directions, epsilon=eps_val)
            end

            X_true_series = simulate_anchor_series(dyn!, X0; T=T, dt=dt)
            A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
            X_hat_series = ase_series(A_avg_series, D)

            config = Dict(
                "n" => N, "d" => D, "T" => T, "dt" => dt,
                "K_samples" => K_SAMPLES, "alpha" => ALPHA,
                "n_anchors" => n_a, "epsilon" => eps_val,
                "rep" => rep, "seed" => BASE_SEED + rep,
                "condition" => "drifting"
            )

            data = Dict(
                "config" => config,
                "X_true_series" => X_true_series,
                "X_hat_series" => X_hat_series,
                "anchor_mask" => anchor_mask,
                "drift_directions" => drift_directions
            )

            fname = "eps" * string(eps_val) * "_rep" * string(rep) * ".jls"
            serialize(joinpath(out_dir, fname), data)
        end
        println("  epsilon=" * string(eps_val) * ": " * string(N_REPS) * " reps done")
    end
end

# =============================================================================
# Condition 3: Error vs trajectory length T
# =============================================================================

function generate_condition_error_vs_T()
    println("\n" * "="^60)
    println("Condition 3: Error vs T")
    println("="^60)

    dt = 0.05
    n_a = 15
    T_values = [10, 20, 50, 100, 200]

    out_dir = joinpath(BASE_DIR, "error_vs_T")
    mkpath(out_dir)

    for T_val in T_values
        for rep in 1:N_REPS
            rng = Random.MersenneTwister(BASE_SEED + rep)

            X0 = generate_uniform_Bd_plus(N, D; rng=rng)
            anchor_mask = falses(N)
            anchor_mask[1:n_a] .= true

            function dyn!(du, u, p, t)
                return anchor_polynomial_dynamics!(du, u, p, t;
                    n=N, d=D, alpha=ALPHA, anchor_mask=anchor_mask)
            end

            X_true_series = simulate_anchor_series(dyn!, X0; T=T_val, dt=dt)
            A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
            X_hat_series = ase_series(A_avg_series, D)

            config = Dict(
                "n" => N, "d" => D, "T" => T_val, "dt" => dt,
                "K_samples" => K_SAMPLES, "alpha" => ALPHA,
                "n_anchors" => n_a, "rep" => rep,
                "seed" => BASE_SEED + rep,
                "condition" => "error_vs_T"
            )

            data = Dict(
                "config" => config,
                "X_true_series" => X_true_series,
                "X_hat_series" => X_hat_series,
                "anchor_mask" => anchor_mask
            )

            fname = "T" * string(T_val) * "_rep" * string(rep) * ".jls"
            serialize(joinpath(out_dir, fname), data)
        end
        println("  T=" * string(T_val) * ": " * string(N_REPS) * " reps done")
    end
end

# =============================================================================
# Condition 4: Spectral gap dependence
# =============================================================================

function generate_condition_spectral_gap()
    println("\n" * "="^60)
    println("Condition 4: Spectral gap dependence")
    println("="^60)

    T = 50
    dt = 0.05
    n_a = 15
    scales = [0.3, 0.5, 0.7, 0.9, 1.0]

    out_dir = joinpath(BASE_DIR, "spectral_gap")
    mkpath(out_dir)

    for scale in scales
        for rep in 1:N_REPS
            rng = Random.MersenneTwister(BASE_SEED + rep)

            X0 = generate_uniform_Bd_plus(N, D; rng=rng)
            # Scale norms to control spectral gap
            X0 .*= scale

            anchor_mask = falses(N)
            anchor_mask[1:n_a] .= true

            function dyn!(du, u, p, t)
                return anchor_polynomial_dynamics!(du, u, p, t;
                    n=N, d=D, alpha=ALPHA, anchor_mask=anchor_mask)
            end

            X_true_series = simulate_anchor_series(dyn!, X0; T=T, dt=dt)
            A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
            X_hat_series = ase_series(A_avg_series, D)

            # Compute spectral gap of initial X0
            gap = spectral_gap(X0)

            config = Dict(
                "n" => N, "d" => D, "T" => T, "dt" => dt,
                "K_samples" => K_SAMPLES, "alpha" => ALPHA,
                "n_anchors" => n_a, "scale" => scale,
                "spectral_gap_X0" => gap,
                "rep" => rep, "seed" => BASE_SEED + rep,
                "condition" => "spectral_gap"
            )

            data = Dict(
                "config" => config,
                "X_true_series" => X_true_series,
                "X_hat_series" => X_hat_series,
                "anchor_mask" => anchor_mask
            )

            fname = "scale" * string(scale) * "_rep" * string(rep) * ".jls"
            serialize(joinpath(out_dir, fname), data)
        end
        println("  scale=" * string(scale) * ": " * string(N_REPS) * " reps done")
    end
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("Anchor-Based Alignment Experiment — Data Generation")
    println("n=" * string(N) * ", d=" * string(D) * ", K=" * string(K_SAMPLES) *
            ", alpha=" * string(ALPHA) * ", n_reps=" * string(N_REPS))

    mkpath(BASE_DIR)

    generate_condition_anchor_count()
    generate_condition_drifting()
    generate_condition_error_vs_T()
    generate_condition_spectral_gap()

    println("\n" * "="^60)
    println("All data saved to: " * BASE_DIR)
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
