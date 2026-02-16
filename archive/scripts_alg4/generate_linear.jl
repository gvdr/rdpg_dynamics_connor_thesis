#!/usr/bin/env -S julia --project
"""
Alg4 Synthetic Dataset: Linear Symmetric Dynamics

Generates:
- True latent trajectory X(t) from Ẋ = N X with symmetric N
- Averaged adjacency series Ā(t) from repeated Bernoulli samples
- ASE embeddings X̂(t) from Ā(t)
- Diagnostics: spectral gap λ_d(t) and skew gauge error

Outputs:
- data/alg4/linear/linear_data.jls
"""

using Random
using LinearAlgebra
using Statistics
using Serialization

include(joinpath(@__DIR__, "alg4_utils.jl"))
using .Alg4Utils

# =============================================================================
# Configuration
# =============================================================================

const SEED = 42
const N = 30
const D = 2
const T = 10
const DT = 0.01
const K_SAMPLES = 20

# Linear symmetric N parameters
const A_INTRA = 0.05
const B_INTER = 0.01
const GAMMA = 0.04

# Initial clusters (two communities)
const CENTERS = [
    [0.7, 0.3],
    [0.3, 0.7],
]
const NOISE_STD = 0.06

# =============================================================================
# Setup
# =============================================================================

rng = Random.MersenneTwister(SEED)

# Initial positions
X0 = generate_clustered_X0(N, D; centers=CENTERS, noise_std=NOISE_STD, rng=rng)
normalize_rows!(X0; max_norm=1.0, min_pos=0.0)

# Dynamics matrix N
communities = [collect(1:(N ÷ 2)), collect((N ÷ 2 + 1):N)]
N_mat = build_linear_symmetric_N(
    N;
    a=A_INTRA,
    b=B_INTER,
    gamma=GAMMA,
    communities=communities
)

# Define dynamics closure
function dynamics!(du, u, p, t)
    return linear_dynamics!(du, u, p, t; n=N, d=D, N=N_mat)
end

# =============================================================================
# Simulate true trajectory
# =============================================================================

X_true_series = simulate_series(dynamics!, X0; T=T, dt=DT)

# Diagnostics: spectral gap on true X(t)
spectral_gaps = series_spectral_gaps(X_true_series)
prob_stats = series_probability_violation_stats(X_true_series)

# =============================================================================
# Observation model: averaged adjacency + ASE
# =============================================================================

A_avg_series = sample_adjacency_average(X_true_series; K=K_SAMPLES, rng=rng)
X_hat_series = ase_series(A_avg_series, D)

# Diagnostics: gauge skew on embeddings
skew_errors = series_skew_errors(X_hat_series; dt=DT)

# =============================================================================
# Save
# =============================================================================

out_dir = joinpath("data", "alg4", "linear")
mkpath(out_dir)
out_path = joinpath(out_dir, "linear_data.jls")

data = Dict(
    "config" => Dict(
        "seed" => SEED,
        "n" => N,
        "d" => D,
        "T" => T,
        "dt" => DT,
        "K_samples" => K_SAMPLES,
        "a_intra" => A_INTRA,
        "b_inter" => B_INTER,
        "gamma" => GAMMA,
        "centers" => CENTERS,
        "noise_std" => NOISE_STD
    ),
    "X0" => X0,
    "N" => N_mat,
    "X_true_series" => X_true_series,
    "A_avg_series" => A_avg_series,
    "X_hat_series" => X_hat_series,
    "spectral_gaps" => spectral_gaps,
    "skew_errors" => skew_errors,
    "probability_stats" => prob_stats
)

serialize(out_path, data)

println("Saved: " * out_path)
println("Spectral gap: min=" * string(round(minimum(spectral_gaps), digits=4)) *
        ", mean=" * string(round(mean(spectral_gaps), digits=4)))
println("Skew error: mean=" * string(round(mean(skew_errors), digits=4)))

min_P = minimum([s.min_P for s in prob_stats])
max_P = maximum([s.max_P for s in prob_stats])
neg_count = sum([s.neg_count for s in prob_stats])
above_one_count = sum([s.above_one_count for s in prob_stats])

println("P validity: min=" * string(round(min_P, digits=4)) *
        ", max=" * string(round(max_P, digits=4)) *
        ", neg_count=" * string(neg_count) *
        ", above_one_count=" * string(above_one_count))
