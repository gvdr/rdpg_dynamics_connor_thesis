#!/usr/bin/env -S julia --project
"""
Alg4 Message-Passing Alignment

Runs the structure-constrained alignment algorithm for the message-passing family:
ẋ_i = ∑_j P_ij g(x_i, x_j), with g(x_i, x_j) = beta1 (x_j - x_i) + beta2 x_j.

Algorithm:
- Initialize R_t via Procrustes chain
- Alternate:
  1) beta-step: linear regression using forward-Euler velocities
  2) R-step: Procrustes per t using Euler-predicted X_pred
- Track residuals and gauge-skew diagnostic

Outputs:
- results/alg4/message_passing_alignment_results.jls
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

const FAMILY = "message_passing"
const MAX_ITERS = 50
const TOL = 1e-6
const OUTPUT_DIR = joinpath("results", "alg4")
const INIT_METHOD = get(ENV, "ALG4_INIT_METHOD", "procrustes_chain") # options: "procrustes_chain", "none", "dpg_procrustes"
const L2_REG = parse(Float64, get(ENV, "ALG4_BETA_L2", "0.0"))

# =============================================================================
# Load data
# =============================================================================

let
    data_path = joinpath("data", "alg4", FAMILY, FAMILY * "_data.jls")
    if !isfile(data_path)
        error("Data file not found: " * data_path * "\nRun scripts/alg4/generate_message_passing.jl first.")
    end

    data = deserialize(data_path)
    X_hat_series = data["X_hat_series"]
    config = data["config"]

    T = length(X_hat_series)
    n, d = size(X_hat_series[1])
    dt = config["dt"]

    # =============================================================================
    # Initialization: choose R_t strategy
    # =============================================================================

    if INIT_METHOD == "procrustes_chain"
        R_series = Vector{Matrix{Float64}}(undef, T - 1)
        for t in 1:(T - 1)
            R_series[t] = procrustes_rotation(X_hat_series[t+1], X_hat_series[t])
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
    else
        error("Unknown INIT_METHOD: " * string(INIT_METHOD))
    end

    # =============================================================================
    # Log initialization
    # =============================================================================

    beta_init = message_passing_beta_step(X_hat_series, R_series; dt=dt, l2=L2_REG)
    println("Init beta (from initial gauges):")
    println(beta_init)

    # =============================================================================
    # Alternating optimization
    # =============================================================================

    residual_history = Float64[]
    beta_history = Vector{Vector{Float64}}()

    for iter in 1:MAX_ITERS
        # beta-step
        beta = message_passing_beta_step(X_hat_series, R_series; dt=dt, l2=L2_REG)
        push!(beta_history, beta)

        # Residuals (Euler prediction)
        f_pred = X -> begin
            F1, F2, _ = message_passing_features(X)
            X .+ dt .* (beta[1] .* F1 .+ beta[2] .* F2)
        end

        res = alignment_residuals(X_hat_series, R_series, f_pred)
        mean_res = mean_alignment_residual(res)
        push!(residual_history, mean_res)

        # R-step
        R_series = message_passing_R_step(X_hat_series, beta; dt=dt)

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

    println("Final beta (after fitting):")
    println(beta_history[end])

    # =============================================================================
    # Diagnostics: gauge skew on aligned series
    # =============================================================================

    Q0 = Matrix{Float64}(I, d, d)
    Q_series = reconstruct_gauges_from_rel(R_series; Q0=Q0)
    X_aligned = apply_gauge_series(X_hat_series, Q_series)

    skew_errors = series_skew_errors(X_aligned; dt=dt)

    # =============================================================================
    # Save results
    # =============================================================================

    mkpath(OUTPUT_DIR)
    out_path = joinpath(OUTPUT_DIR, "message_passing_alignment_results_" * INIT_METHOD * ".jls")

    result = Dict(
        "config" => config,
        "init_method" => INIT_METHOD,
        "beta" => beta_history[end],
        "beta_history" => beta_history,
        "R_series" => R_series,
        "Q_series" => Q_series,
        "residual_history" => residual_history,
        "skew_errors" => skew_errors
    )

    serialize(out_path, result)

    println("Saved: " * out_path)
end
