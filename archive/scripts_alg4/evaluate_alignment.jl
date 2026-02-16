#!/usr/bin/env -S julia --project
"""
Alg4 Alignment Evaluation

Compares baselines and parameter recovery across dynamics families.

Baselines:
- None (no alignment)
- Procrustes chain
- Structure-constrained alignment (from saved results)

Metrics:
- P-error: ||P_hat - P_true||_F / ||P_true||_F (mean over time)
- Skew gauge error (mean over time)
- Parameter recovery (relative error vs. ground truth)

Outputs:
- results/alg4/evaluation_summary.jls
"""

using Serialization
using LinearAlgebra
using Statistics

const DPG_PATH = joinpath(@__DIR__, "..", "..", "..", "DotProductGraphs.jl", "src")
if !(DPG_PATH in LOAD_PATH)
    push!(LOAD_PATH, DPG_PATH)
end
using DotProductGraphs

include(joinpath(@__DIR__, "alg4_utils.jl"))
include(joinpath(@__DIR__, "alg4_alignment_utils.jl"))

using .Alg4Utils
using .Alg4AlignmentUtils

const DATA_DIR = joinpath("data", "alg4")
const RESULTS_DIR = joinpath("results", "alg4")
const SKIP_GBDASE = get(ENV, "ALG4_SKIP_GBDASE", "false") == "true"
const VERBOSE = get(ENV, "ALG4_VERBOSE", "true") == "true"

# =============================================================================
# Helpers
# =============================================================================

function procrustes_chain_gauges(X_series::Vector{<:AbstractMatrix})
    T = length(X_series)
    R_series = Vector{Matrix{Float64}}(undef, T - 1)
    for t in 1:(T - 1)
        R_series[t] = procrustes_rotation(X_series[t+1], X_series[t])
    end
    Q0 = Matrix{Float64}(I, size(X_series[1], 2), size(X_series[1], 2))
    Q_series = reconstruct_gauges_from_rel(R_series; Q0=Q0)
    return Q_series
end

function identity_gauges(X_series::Vector{<:AbstractMatrix})
    d = size(X_series[1], 2)
    T = length(X_series)
    return [Matrix{Float64}(I, d, d) for _ in 1:T]
end

function P_error_series(X_aligned::Vector{<:AbstractMatrix},
                        X_true::Vector{<:AbstractMatrix})
    T = min(length(X_aligned), length(X_true))
    errs = zeros(Float64, T)
    for t in 1:T
        P_hat = X_aligned[t] * X_aligned[t]'
        P_true = X_true[t] * X_true[t]'
        denom = norm(P_true)
        errs[t] = denom > 0 ? norm(P_hat .- P_true) / denom : 0.0
    end
    return errs
end

function mean_skew_error(X_aligned::Vector{<:AbstractMatrix}; dt::Float64=1.0)
    skew_errors = series_skew_errors(X_aligned; dt=dt)
    return isempty(skew_errors) ? 0.0 : mean(skew_errors)
end

function rel_error(A, B)
    denom = norm(B)
    return denom > 0 ? norm(A .- B) / denom : 0.0
end

function tne_to_series(TNE::TemporalNetworkEmbedding; side::Symbol=:AL)
    T = length(TNE)
    return [TNE[t, side] for t in 1:T]
end

function duase_to_series(result)
    T = size(result.Y, 3)
    return [result.Y[:, :, t] for t in 1:T]
end

function gbdase_to_series(result)
    X = result.X
    T = size(X, 1)
    return [Matrix(X[t, :, :]) for t in 1:T]
end

function embedding_baselines(A_series::Vector{<:AbstractMatrix}, d::Int)
    baselines = Dict{String, Vector{Matrix{Float64}}}()

    TNE = TemporalNetworkEmbedding(A_series, d, :procrustes)
    baselines["dpg_procrustes"] = tne_to_series(TNE)

    TNE = TemporalNetworkEmbedding(A_series, d, :omni)
    baselines["dpg_omni"] = tne_to_series(TNE)

    TNE = TemporalNetworkEmbedding(A_series, d, :uase)
    baselines["dpg_uase"] = tne_to_series(TNE)

    TNE = TemporalNetworkEmbedding(A_series, d, :mase)
    baselines["dpg_mase"] = tne_to_series(TNE)

    duase_res = duase_embedding(A_series, d)
    baselines["dpg_duase"] = duase_to_series(duase_res)

    if !SKIP_GBDASE
        if VERBOSE
            println("  Computing GBDASE baseline")
        end
        gbdase_res = gbdase(A_series, d)
        baselines["dpg_gbdase"] = gbdase_to_series(gbdase_res)
    elseif VERBOSE
        println("  Skipping GBDASE baseline")
    end

    return baselines
end

# =============================================================================
# Load data for each family
# =============================================================================

function load_data(family::String)
    data_path = joinpath(DATA_DIR, family, family * "_data.jls")
    if !isfile(data_path)
        error("Missing data file: " * data_path)
    end
    return deserialize(data_path)
end

function load_alignment_result(family::String, init_method::String)
    result_path = joinpath(RESULTS_DIR, family * "_alignment_results_" * init_method * ".jls")
    if isfile(result_path)
        return deserialize(result_path)
    end

    if init_method == "procrustes_chain"
        legacy_path = joinpath(RESULTS_DIR, family * "_alignment_results.jls")
        if isfile(legacy_path)
            return deserialize(legacy_path)
        end
    end

    return nothing
end

# =============================================================================
# Evaluation per family
# =============================================================================

function evaluate_family(family::String)
    data_path = joinpath(DATA_DIR, family, family * "_data.jls")
    data = load_data(family)
    X_hat_series = data["X_hat_series"]
    X_true_series = data["X_true_series"]
    A_series = data["A_avg_series"]
    config = data["config"]
    dt = config["dt"]
    d = config["d"]

    if VERBOSE
        println("")
        println("Evaluating family: " * family)
        println("  n=" * string(config["n"]) * " d=" * string(d) * " T=" * string(config["T"]) * " dt=" * string(dt))
    end

    results = Dict{String, Any}()

    results["true_reference"] = Dict(
        "P_error_mean" => 0.0,
        "skew_mean" => mean_skew_error(X_true_series; dt=dt)
    )

    # Baseline: none
    Q_none = identity_gauges(X_hat_series)
    X_none = apply_gauge_series(X_hat_series, Q_none)
    results["none"] = Dict(
        "P_error_mean" => mean(P_error_series(X_none, X_true_series)),
        "skew_mean" => mean_skew_error(X_none; dt=dt)
    )

    # Baseline: Procrustes chain
    Q_chain = procrustes_chain_gauges(X_hat_series)
    X_chain = apply_gauge_series(X_hat_series, Q_chain)
    results["procrustes_chain"] = Dict(
        "P_error_mean" => mean(P_error_series(X_chain, X_true_series)),
        "skew_mean" => mean_skew_error(X_chain; dt=dt)
    )

    # DotProductGraphs baselines
    if VERBOSE
        println("  Computing DotProductGraphs baselines")
    end
    embed_baselines = embedding_baselines(A_series, d)
    for (name, X_base) in embed_baselines
        results[name] = Dict(
            "P_error_mean" => mean(P_error_series(X_base, X_true_series)),
            "skew_mean" => mean_skew_error(X_base; dt=dt)
        )
    end

    if haskey(embed_baselines, "dpg_procrustes")
        X_dpg = embed_baselines["dpg_procrustes"]
        Q_series_dpg_procrustes = [procrustes_rotation(X_hat_series[t], X_dpg[t]) for t in 1:length(X_hat_series)]
        data["Q_series_dpg_procrustes"] = Q_series_dpg_procrustes
        serialize(data_path, data)
    end

    # Structure-constrained (if available)
    init_methods = ["procrustes_chain", "none", "dpg_procrustes"]
    for init_method in init_methods
        align_res = load_alignment_result(family, init_method)
        if !isnothing(align_res)
            Q_series = align_res["Q_series"]
            X_struct = apply_gauge_series(X_hat_series, Q_series)
            results["structure_constrained_" * init_method] = Dict(
                "P_error_mean" => mean(P_error_series(X_struct, X_true_series)),
                "skew_mean" => mean_skew_error(X_struct; dt=dt),
                "residual_mean" => mean(align_res["residual_history"])
            )
        end
    end

    # Parameter recovery (use procrustes_chain alignment result when available)
    align_res_pc = load_alignment_result(family, "procrustes_chain")
    if family == "linear"
        if haskey(data, "N") && !isnothing(align_res_pc)
            N_true = data["N"]
            M_hat = align_res_pc["M"]
            results["param_error"] = Dict(
                "M_rel_error" => rel_error(M_hat, N_true)
            )
        end
    elseif family == "polynomial"
        if haskey(data, "alpha") && !isnothing(align_res_pc)
            alpha_true = data["alpha"]
            alpha_hat = align_res_pc["alpha"]
            results["param_error"] = Dict(
                "alpha_rel_error" => rel_error(alpha_hat, alpha_true)
            )
        end
    elseif family == "message_passing"
        if haskey(data, "beta") && !isnothing(align_res_pc)
            beta_true = data["beta"]
            beta_hat = align_res_pc["beta"]
            results["param_error"] = Dict(
                "beta_rel_error" => rel_error(beta_hat, beta_true)
            )
        end
    end

    return results
end

# =============================================================================
# Run evaluations
# =============================================================================

families = ["linear", "polynomial", "message_passing"]
summary = Dict{String, Any}()

for family in families
    summary[family] = evaluate_family(family)
end

# =============================================================================
# Save + print summary
# =============================================================================

mkpath(RESULTS_DIR)
out_path = joinpath(RESULTS_DIR, "evaluation_summary.jls")
serialize(out_path, summary)

println("Saved: " * out_path)

for family in families
    println("Family: " * family)
    fam = summary[family]

    for (k, v) in fam
        if k == "param_error"
            for (pk, pv) in v
                println("  " * pk * " = " * string(round(pv, digits=6)))
            end
        else
            println("  " * k * ":")
            println("    P_error_mean = " * string(round(v["P_error_mean"], digits=6)))
            println("    skew_mean    = " * string(round(v["skew_mean"], digits=6)))
            if haskey(v, "residual_mean")
                println("    residual_mean = " * string(round(v["residual_mean"], digits=6)))
            end
        end
    end
end
