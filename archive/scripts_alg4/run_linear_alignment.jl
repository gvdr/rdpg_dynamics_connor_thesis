#!/usr/bin/env -S julia --project
"""
Alg4 Linear Symmetric Alignment

Runs the structure-constrained alignment algorithm for the linear symmetric family:
X(t+1) ≈ M X(t), with M = Mᵀ.

Algorithm:
- Initialize R_t via Procrustes chain
- Alternate:
  1) M-step: least squares + symmetrize
  2) R-step: Procrustes per t
- Track residuals and gauge-skew diagnostic

Outputs:
- results/alg4/linear_alignment_results.jls
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

const FAMILY = "linear"
const MAX_ITERS = 50
const TOL = 1e-6
const OUTPUT_DIR = joinpath("results", "alg4")
const INIT_METHOD = get(ENV, "ALG4_INIT_METHOD", "procrustes_chain") # options: "procrustes_chain", "none", "dpg_procrustes", "true_gauges"
const L2_REG = parse(Float64, get(ENV, "ALG4_M_L2", "0.0"))
const USE_TRUE_SERIES = get(ENV, "ALG4_USE_TRUE_SERIES", "false") == "true"

# =============================================================================
# Load data
# =============================================================================

let
    data_path = joinpath("data", "alg4", FAMILY, FAMILY * "_data.jls")
if !isfile(data_path)
    error("Data file not found: " * data_path * "\nRun scripts/alg4/generate_linear.jl first.")
end

data = deserialize(data_path)
X_hat_series = data["X_hat_series"]
X_true_series = data["X_true_series"]
config = data["config"]
X_fit_series = USE_TRUE_SERIES ? X_true_series : X_hat_series

T = length(X_hat_series)
n, d = size(X_hat_series[1])

# Hard-coded communities (two equal blocks)
communities = [collect(1:(n ÷ 2)), collect((n ÷ 2 + 1):n)]
W_intra, W_inter = block_indicators(n, communities)

# =============================================================================
# Initialization: choose R_t strategy
# =============================================================================

if INIT_METHOD == "procrustes_chain"
    R_series = Vector{Matrix{Float64}}(undef, T - 1)
    for t in 1:(T - 1)
        R_series[t] = procrustes_rotation(X_fit_series[t+1], X_fit_series[t])
    end
elseif INIT_METHOD == "none"
    Q_series = [Matrix{Float64}(I, d, d) for _ in 1:T]
    R_series = build_relative_gauges(Q_series)
elseif INIT_METHOD == "dpg_procrustes"
    if haskey(data, "Q_series_dpg_procrustes")
        Q_series = data["Q_series_dpg_procrustes"]
        R_series = build_relative_gauges(Q_series)
    else
        error("INIT_METHOD=dpg_procrustes requires data[\"Q_series_dpg_procrustes\"]")
    end
elseif INIT_METHOD == "true_gauges"
    Q_series = [procrustes_rotation(X_hat_series[t], X_true_series[t]) for t in 1:T]
    R_series = build_relative_gauges(Q_series)
else
    error("Unknown INIT_METHOD: " * string(INIT_METHOD))
end

# =============================================================================
# Log initialization
# =============================================================================

params_init = linear_block_params_step(X_fit_series, R_series, W_intra, W_inter; dt=config["dt"], l2=L2_REG)
M_init = linear_block_M(n, communities, params_init[1], params_init[2], params_init[3]; dt=config["dt"])
println("Init params (a, b, gamma) from initial gauges:")
println(params_init)

# =============================================================================
# Alternating optimization
# =============================================================================

residual_history = Float64[]
M_history = Matrix{Float64}[]
params_history = Vector{Vector{Float64}}()

for iter in 1:MAX_ITERS
    # Params-step (block-structured M)
    params = linear_block_params_step(X_fit_series, R_series, W_intra, W_inter; dt=config["dt"], l2=L2_REG)
    push!(params_history, params)
    M = linear_block_M(n, communities, params[1], params[2], params[3]; dt=config["dt"])
    push!(M_history, M)

    # Residuals
    f_pred = X -> M * X
    res = alignment_residuals(X_fit_series, R_series, f_pred)
    mean_res = mean_alignment_residual(res)
    push!(residual_history, mean_res)

    # R-step
    R_prev = R_series
    R_series = linear_R_step(X_fit_series, M)
    r_changes = [norm(R_series[t] .- R_prev[t]) for t in 1:length(R_series)]
    r_change_mean = isempty(r_changes) ? 0.0 : mean(r_changes)
    println("Iter " * string(iter) * " | mean R change = " * string(round(r_change_mean, digits=6)))

    # Convergence check
    if iter > 1
        if abs(residual_history[end] - residual_history[end - 1]) < TOL
            println("Converged at iter " * string(iter) * ", mean residual=" * string(mean_res))
            break
        end
    end

    if iter % 10 == 0 || iter == 1
        println("Iter " * string(iter) * " | mean residual = " * string(round(mean_res, digits=6)))
    end
end

println("Final params (a, b, gamma) after fitting:")
println(params_history[end])

# =============================================================================
# Diagnostics: gauge skew on aligned series
# =============================================================================

# Recover Q_t from R_t (start at identity)
Q0 = Matrix{Float64}(I, d, d)
Q_series = reconstruct_gauges_from_rel(R_series; Q0=Q0)
X_aligned = apply_gauge_series(X_hat_series, Q_series)

skew_errors = series_skew_errors(X_aligned; dt=config["dt"])

# =============================================================================
# Save results
# =============================================================================

mkpath(OUTPUT_DIR)
out_path = joinpath(OUTPUT_DIR, "linear_alignment_results_" * INIT_METHOD * ".jls")

result = Dict(
    "config" => config,
    "M" => M_history[end],
    "params" => params_history[end],
    "params_history" => params_history,
    "R_series" => R_series,
    "Q_series" => Q_series,
    "residual_history" => residual_history,
    "skew_errors" => skew_errors
)

serialize(out_path, result)

println("Saved: " * out_path)

end
