#!/usr/bin/env -S julia --project
"""
Alg4 Polynomial Alignment

Runs the structure-constrained alignment algorithm for the polynomial family:
X(t+1) ≈ M(P(t)) X(t), with M(P) = I + dt Σ_k α_k P^k.

Algorithm:
- α-step: gauge-free regression on P dynamics (trapezoid rule)
- Compute P-space residuals
- No R-step or gauge alignment

Outputs:
- results/alg4/polynomial_alignment_results.jls
"""

using Serialization
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "alg4_utils.jl"))
include(joinpath(@__DIR__, "alg4_alignment_utils.jl"))

using .Alg4Utils
using .Alg4AlignmentUtils

# =============================================================================
# Configuration
# =============================================================================

const FAMILY = "polynomial"
const MAX_ITERS = 50
const TOL = 1e-6
const OUTPUT_DIR = joinpath("results", "alg4")

const L2_REG = parse(Float64, get(ENV, "ALG4_ALPHA_L2", "0.0"))

# =============================================================================
# Load data
# =============================================================================

let
    data_path = joinpath("data", "alg4", FAMILY, FAMILY * "_data.jls")
if !isfile(data_path)
    error("Data file not found: " * data_path * "\nRun scripts/alg4/generate_polynomial.jl first.")
end

data = deserialize(data_path)
X_hat_series = data["X_hat_series"]
config = data["config"]

T = length(X_hat_series)
n, d = size(X_hat_series[1])
dt = config["dt"]

# Degree K inferred from true alpha if present
if haskey(data, "alpha")
    K = length(data["alpha"]) - 1
else
    K = 1
end

# =============================================================================
# Gauge-free setup (no R-series)
# =============================================================================

# =============================================================================
# Alternating optimization
# =============================================================================

# =============================================================================
# Gauge-free P-dynamics fit
# =============================================================================

P_series = compute_P_series(X_hat_series)
P_powers = compute_P_powers(P_series, K + 1)

alpha = polynomial_alpha_step(X_hat_series, Matrix{Float64}[], K; dt=dt, l2=L2_REG)
alpha_history = [alpha]

residuals = zeros(Float64, T - 1)
for t in 1:(T - 1)
    dP = P_series[t+1] .- P_series[t]
    dP_pred = zeros(Float64, n, n)
    for k in 1:(K+1)
        dP_pred .+= dt .* alpha[k] .* (P_powers[t][k+1] .+ P_powers[t+1][k+1])
    end
    residuals[t] = norm(dP .- dP_pred)
end

residual_history = [mean(residuals)]



# =============================================================================
# Save results
# =============================================================================

mkpath(OUTPUT_DIR)
out_path = joinpath(OUTPUT_DIR, "polynomial_alignment_results_gauge_free.jls")

result = Dict(
    "config" => config,
    "method" => "gauge_free",
    "K" => K,
    "alpha" => alpha_history[end],
    "alpha_history" => alpha_history,
    "p_residuals" => residuals,
    "residual_history" => residual_history
)

serialize(out_path, result)

println("Saved: " * out_path)

end
