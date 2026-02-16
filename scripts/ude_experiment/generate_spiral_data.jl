#!/usr/bin/env -S julia --project
"""
UDE Pipeline Experiment — Phase 1: Data Generation

Generates damped spiral dynamics around community centroids in B+^3.
Non-gauge-equivariant dynamics requiring X-space estimates for parameter recovery.

True dynamics for mobile node i in community k:
  Ẋᵢ = (-γ + β||xᵢ - μ_k||²)(xᵢ - μ_k) + ω J(xᵢ - μ_k)

where J = (1/√3)[0 -1 1; 1 0 -1; -1 1 0] (rotation around (1,1,1)/√3).

Outputs per rep:
  data/ude_experiment/rep<i>.jls
"""

using Random
using LinearAlgebra
using Statistics
using Serialization
using OrdinaryDiffEq

include(joinpath(@__DIR__, "..", "alg4", "alg4_utils.jl"))
include(joinpath(@__DIR__, "..", "alg4", "alg4_alignment_utils.jl"))

using .Alg4Utils
using .Alg4AlignmentUtils

# =============================================================================
# Constants
# =============================================================================

const N = 200
const D = 3
const N_ANCHOR = 100
const K_COMMUNITIES = 3
const K_SAMPLES = 10
const T_STEPS = 50
const DT = 0.1
const N_REPS = 5
const BASE_SEED = 2025

# Dynamics parameters
const GAMMA = 0.3
const BETA = -0.5
const OMEGA = 1.0

# Community centroids near vertices of B+^3
const CENTROIDS = [
    [0.7, 0.2, 0.2],
    [0.2, 0.7, 0.2],
    [0.2, 0.2, 0.7]
]

const DATA_DIR = joinpath("data", "ude_experiment")

# =============================================================================
# Rotation generator J around (1,1,1)/√3
# =============================================================================

const J_MATRIX = (1.0 / sqrt(3.0)) .* [
    0.0  -1.0   1.0;
    1.0   0.0  -1.0;
   -1.0   1.0   0.0
]

# =============================================================================
# Spiral dynamics
# =============================================================================

"""
    spiral_dynamics!(du, u, p, t; n, d, community, centroids, anchor_mask, gamma, beta, omega)

In-place ODE for damped spiral around community centroids.
Mobile nodes: Ẋᵢ = (-γ + β r²)δᵢ + ω J δᵢ  where δᵢ = xᵢ - μ_k, r = ||δᵢ||
Anchor nodes: Ẋᵢ = 0
"""
function spiral_dynamics!(du::AbstractVector, u::AbstractVector, p, t;
                           n::Int, d::Int,
                           community::Vector{Int},
                           centroids::Vector{Vector{Float64}},
                           anchor_mask::BitVector,
                           gamma::Float64, beta::Float64, omega::Float64)
    X = reshape(u, n, d)
    dX = zeros(eltype(u), n, d)

    for i in 1:n
        if anchor_mask[i]
            # Anchor nodes are frozen
            continue
        end

        k = community[i]
        mu = centroids[k]

        # Offset from centroid
        delta = X[i, :] .- mu
        r2 = dot(delta, delta)

        # Damped radial + nonlinear stiffening
        radial = (-gamma + beta * r2) .* delta

        # Rotation: ω J δ
        rotation = omega .* (J_MATRIX * delta)

        dX[i, :] = radial .+ rotation
    end

    du .= vec(dX)
    return nothing
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_spiral_series(dynamics!, X0; T, dt)

Simulate ODE and return vector of X(t) matrices.
"""
function simulate_spiral_series(dynamics!, X0::AbstractMatrix; T::Int=50, dt::Float64=0.1)
    n, d = size(X0)
    tspan = (0.0, dt * (T - 1))
    tsteps = range(tspan[1], tspan[2]; length=T)
    prob = ODEProblem((du, u, p, t) -> dynamics!(du, u, p, t), vec(X0), tspan)
    sol = solve(prob, Tsit5(); saveat=tsteps, abstol=1e-8, reltol=1e-8)
    return [reshape(sol.u[t], n, d) for t in 1:length(sol.u)]
end

# =============================================================================
# Alignment functions (from anchor experiment)
# =============================================================================

"""
    anchor_procrustes(X_t, X_ref, anchor_idx)

Compute Procrustes rotation using only anchor rows.
"""
function anchor_procrustes(X_t::AbstractMatrix, X_ref::AbstractMatrix,
                           anchor_idx::Vector{Int})
    A = X_t[anchor_idx, :]
    B = X_ref[anchor_idx, :]
    return procrustes_rotation(A, B)
end

"""
    anchor_align_series(X_hat_series, anchor_idx; ref_time=1)

Align each frame independently to ref_time via anchor-node Procrustes.
"""
function anchor_align_series(X_hat_series::Vector{<:AbstractMatrix},
                             anchor_idx::Vector{Int};
                             ref_time::Int=1)
    T = length(X_hat_series)
    X_ref = X_hat_series[ref_time]
    aligned = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        Q = anchor_procrustes(X_hat_series[t], X_ref, anchor_idx)
        aligned[t] = X_hat_series[t] * Q
    end
    return aligned
end

"""
    sequential_procrustes_align(X_hat_series)

Chain Procrustes: align each frame to the previous one.
"""
function sequential_procrustes_align(X_hat_series::Vector{<:AbstractMatrix})
    T = length(X_hat_series)
    aligned = Vector{Matrix{Float64}}(undef, T)
    aligned[1] = copy(X_hat_series[1])
    for t in 2:T
        Q = procrustes_rotation(X_hat_series[t], aligned[t-1])
        aligned[t] = X_hat_series[t] * Q
    end
    return aligned
end

# =============================================================================
# Community assignment
# =============================================================================

"""
    assign_communities(n, k)

Assign n nodes to k communities in balanced fashion.
Returns vector of community indices (1:k).
"""
function assign_communities(n::Int, k::Int)
    community = zeros(Int, n)
    per = n ÷ k
    extra = n % k
    idx = 1
    for c in 1:k
        count = per + (c <= extra ? 1 : 0)
        for _ in 1:count
            community[idx] = c
            idx += 1
        end
    end
    return community
end

# =============================================================================
# Per-rep data generation
# =============================================================================

function generate_rep(rep::Int)
    rng = Random.MersenneTwister(BASE_SEED + rep)
    println("  Generating rep " * string(rep) * "...")

    # Community assignments: first N_ANCHOR are anchors, next N-N_ANCHOR are mobile
    # Both sets are balanced across communities
    anchor_community = assign_communities(N_ANCHOR, K_COMMUNITIES)
    mobile_community = assign_communities(N - N_ANCHOR, K_COMMUNITIES)
    community = vcat(anchor_community, mobile_community)

    anchor_mask = falses(N)
    anchor_mask[1:N_ANCHOR] .= true

    # Generate initial positions via generate_clustered_X0
    X0 = generate_clustered_X0(N, D;
        centers=[Float64.(c) for c in CENTROIDS],
        noise_std=0.15,
        rng=rng)

    # Build dynamics closure
    function dyn!(du, u, p, t)
        return spiral_dynamics!(du, u, p, t;
            n=N, d=D, community=community,
            centroids=[Float64.(c) for c in CENTROIDS],
            anchor_mask=anchor_mask,
            gamma=GAMMA, beta=BETA, omega=OMEGA)
    end

    # Simulate true trajectory
    X_true_series = simulate_spiral_series(dyn!, X0; T=T_STEPS, dt=DT)

    # Check for probability violations
    P_final = X_true_series[end] * X_true_series[end]'
    max_P = maximum(P_final)
    min_P = minimum(P_final)
    println("    P range at final time: [" * string(round(min_P, digits=4)) *
            ", " * string(round(max_P, digits=4)) * "]")

    # Sample noisy adjacency matrices and embed
    A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
    X_hat_series = ase_series(A_avg_series, D)

    anchor_idx = findall(anchor_mask)

    # Condition 1: Anchor-based alignment
    X_anchor_aligned = anchor_align_series(X_hat_series, anchor_idx; ref_time=1)

    # Condition 2: Sequential Procrustes alignment
    X_seq_aligned = sequential_procrustes_align(X_hat_series)

    # Condition 3: Unaligned — random orthogonal rotation per frame.
    # Raw ASE is misleadingly consistent because SVD of slowly-changing
    # matrices produces nearly identical eigenvectors. To simulate genuine
    # rotation ambiguity, apply an independent random orthogonal Q_t to each
    # frame. This preserves per-frame embedding quality but destroys
    # temporal coherence.
    X_unaligned = Vector{Matrix{Float64}}(undef, length(X_hat_series))
    for t in eachindex(X_hat_series)
        Q_rand = Matrix(qr(randn(rng, D, D)).Q)
        X_unaligned[t] = X_hat_series[t] * Q_rand
    end

    # Global alignment to true coordinate frame.
    # Each alignment method puts frames in a self-consistent frame, but that
    # frame is an arbitrary rotation of the true X frame (ASE rotation ambiguity).
    # Apply one Procrustes rotation (computed at t=1) to bring each series into
    # the true coordinate system. This is necessary because the UDE uses true
    # centroids and J lives in the true X-space.
    Q_anchor = procrustes_rotation(X_anchor_aligned[1], X_true_series[1])
    X_anchor_aligned = [X * Q_anchor for X in X_anchor_aligned]

    Q_seq = procrustes_rotation(X_seq_aligned[1], X_true_series[1])
    X_seq_aligned = [X * Q_seq for X in X_seq_aligned]

    # For unaligned: only frame 1 gets properly aligned; the rest have
    # independent random rotations so the single Q cannot fix them.
    Q_unaligned = procrustes_rotation(X_unaligned[1], X_true_series[1])
    X_unaligned = [X * Q_unaligned for X in X_unaligned]

    # Config dict
    config = Dict(
        "n" => N, "d" => D, "T" => T_STEPS, "dt" => DT,
        "n_anchor" => N_ANCHOR, "K_samples" => K_SAMPLES,
        "K_communities" => K_COMMUNITIES,
        "gamma" => GAMMA, "beta" => BETA, "omega" => OMEGA,
        "centroids" => CENTROIDS,
        "noise_std" => 0.15,
        "rep" => rep, "seed" => BASE_SEED + rep
    )

    data = Dict(
        "config" => config,
        "X_true_series" => X_true_series,
        "X_anchor_aligned" => X_anchor_aligned,
        "X_seq_aligned" => X_seq_aligned,
        "X_unaligned" => X_unaligned,
        "anchor_mask" => anchor_mask,
        "community" => community
    )

    fname = "rep" * string(rep) * ".jls"
    serialize(joinpath(DATA_DIR, fname), data)
    println("    Saved: " * fname)
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("UDE Pipeline Experiment — Data Generation")
    println("n=" * string(N) * ", d=" * string(D) *
            ", n_anchor=" * string(N_ANCHOR) *
            ", K=" * string(K_SAMPLES) *
            ", T=" * string(T_STEPS) *
            ", dt=" * string(DT))
    println("Dynamics: gamma=" * string(GAMMA) *
            ", beta=" * string(BETA) *
            ", omega=" * string(OMEGA))

    mkpath(DATA_DIR)

    for rep in 1:N_REPS
        generate_rep(rep)
    end

    println("\n" * "="^60)
    println("All data saved to: " * DATA_DIR)
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
