#!/usr/bin/env -S julia --project
"""
Alg4 Anchor-Based Alignment Experiment — Phase 2: Alignment & Evaluation

For each condition × parameter × rep:
- Apply anchor-based alignment → compute alignment error over time
- Apply sequential Procrustes (baseline) → compute alignment error over time
- Fit α̂ from aligned trajectory → dynamics recovery error

Aggregates across reps: mean ± std. Saves to results/alg4/anchor_experiment/.
"""

using Random
using LinearAlgebra
using Statistics
using Serialization

include(joinpath(@__DIR__, "alg4_utils.jl"))
include(joinpath(@__DIR__, "alg4_alignment_utils.jl"))

using .Alg4Utils
using .Alg4AlignmentUtils

# =============================================================================
# Local alignment functions (defined inline, not modifying shared utils)
# =============================================================================

"""
    anchor_procrustes(X_t, X_ref, anchor_idx)

Compute Procrustes rotation using only anchor rows.
Returns Q ∈ O(d) such that X_t[anchor_idx, :] * Q ≈ X_ref[anchor_idx, :].
"""
function anchor_procrustes(X_t::AbstractMatrix, X_ref::AbstractMatrix,
                           anchor_idx::Vector{Int})
    A = X_t[anchor_idx, :]
    B = X_ref[anchor_idx, :]
    return procrustes_rotation(A, B)
end

"""
    anchor_align_series(X_hat_series, anchor_idx; ref_time=1)

Align each frame independently to time ref_time via anchor-node Procrustes.
Returns vector of aligned X̂(t) matrices.
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
Baseline method that accumulates drift.
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

"""
    compute_alignment_error(X_aligned_series, X_true_series)

Compute per-timestep alignment error:
err(t) = (1/n) ||X̃(t) - X(t) Q*||_F
where Q* is the best global Procrustes on the concatenated series.
"""
function compute_alignment_error(X_aligned_series::Vector{<:AbstractMatrix},
                                 X_true_series::Vector{<:AbstractMatrix})
    T = length(X_aligned_series)
    n = size(X_aligned_series[1], 1)

    # Find best global Q* by concatenating all timesteps
    X_cat = vcat(X_aligned_series...)
    Y_cat = vcat(X_true_series...)
    Q_star = procrustes_rotation(Y_cat, X_cat)

    errors = zeros(Float64, T)
    for t in 1:T
        diff = X_aligned_series[t] .- X_true_series[t] * Q_star
        errors[t] = norm(diff) / n
    end
    return errors
end

"""
    fit_alpha_from_aligned(X_aligned_series, dt, K)

Fit polynomial coefficients from an aligned trajectory using the gauge-free
P-dynamics approach from alg4_alignment_utils.
"""
function fit_alpha_from_aligned(X_aligned_series::Vector{<:AbstractMatrix},
                                dt::Float64, K::Int)
    # polynomial_alpha_step ignores R_series for gauge-free fit
    return polynomial_alpha_step(X_aligned_series, Matrix{Float64}[], K; dt=dt, l2=0.0)
end

# =============================================================================
# Evaluation driver
# =============================================================================

"""
    evaluate_single(data_path)

Load a single .jls, run both alignment methods, compute errors, return results dict.
"""
function evaluate_single(data_path::String)
    data = deserialize(data_path)
    config = data["config"]
    X_true_series = data["X_true_series"]
    X_hat_series = data["X_hat_series"]
    anchor_mask = data["anchor_mask"]

    n_a = config["n_anchors"]
    dt = config["dt"]
    K = length(config["alpha"]) - 1  # polynomial degree

    anchor_idx = findall(anchor_mask)

    # --- Anchor-based alignment ---
    if n_a >= 2  # need at least d=2 anchors for Procrustes
        X_anchor_aligned = anchor_align_series(X_hat_series, anchor_idx; ref_time=1)
        anchor_errors = compute_alignment_error(X_anchor_aligned, X_true_series)
        alpha_anchor = fit_alpha_from_aligned(X_anchor_aligned, dt, K)
    else
        # Underdetermined: still try if n_a >= 1, but expect failure
        if n_a >= 1
            X_anchor_aligned = anchor_align_series(X_hat_series, anchor_idx; ref_time=1)
            anchor_errors = compute_alignment_error(X_anchor_aligned, X_true_series)
            alpha_anchor = fit_alpha_from_aligned(X_anchor_aligned, dt, K)
        else
            # No anchors: use unaligned embeddings
            anchor_errors = compute_alignment_error(X_hat_series, X_true_series)
            alpha_anchor = fit_alpha_from_aligned(X_hat_series, dt, K)
        end
    end

    # --- Sequential Procrustes (baseline) ---
    X_seq_aligned = sequential_procrustes_align(X_hat_series)
    seq_errors = compute_alignment_error(X_seq_aligned, X_true_series)
    alpha_seq = fit_alpha_from_aligned(X_seq_aligned, dt, K)

    return Dict(
        "config" => config,
        "anchor_errors" => anchor_errors,
        "seq_errors" => seq_errors,
        "alpha_anchor" => alpha_anchor,
        "alpha_seq" => alpha_seq,
        "mean_anchor_error" => mean(anchor_errors),
        "mean_seq_error" => mean(seq_errors)
    )
end

# =============================================================================
# Aggregation helpers
# =============================================================================

"""
    aggregate_results(results_list, group_key)

Group results by a config key and compute mean ± std of errors.
"""
function aggregate_results(results_list::Vector{Dict}, group_key::String)
    groups = Dict{Any, Vector{Dict}}()
    for r in results_list
        k = r["config"][group_key]
        if !haskey(groups, k)
            groups[k] = Dict[]
        end
        push!(groups[k], r)
    end

    agg = Dict{Any, Dict}()
    for (k, reps) in groups
        anchor_means = [r["mean_anchor_error"] for r in reps]
        seq_means = [r["mean_seq_error"] for r in reps]

        # Per-timestep aggregation
        T_max = maximum(length(r["anchor_errors"]) for r in reps)
        # Only aggregate timesteps where all reps have data
        T_min = minimum(length(r["anchor_errors"]) for r in reps)

        anchor_ts = zeros(Float64, T_min)
        anchor_ts_std = zeros(Float64, T_min)
        seq_ts = zeros(Float64, T_min)
        seq_ts_std = zeros(Float64, T_min)

        for t in 1:T_min
            ae = [r["anchor_errors"][t] for r in reps]
            se = [r["seq_errors"][t] for r in reps]
            anchor_ts[t] = mean(ae)
            anchor_ts_std[t] = std(ae)
            seq_ts[t] = mean(se)
            seq_ts_std[t] = std(se)
        end

        # Alpha recovery
        alpha_true = reps[1]["config"]["alpha"]
        alpha_anchor_all = hcat([r["alpha_anchor"] for r in reps]...)'
        alpha_seq_all = hcat([r["alpha_seq"] for r in reps]...)'

        agg[k] = Dict(
            "anchor_mean" => mean(anchor_means),
            "anchor_std" => std(anchor_means),
            "seq_mean" => mean(seq_means),
            "seq_std" => std(seq_means),
            "anchor_timeseries_mean" => anchor_ts,
            "anchor_timeseries_std" => anchor_ts_std,
            "seq_timeseries_mean" => seq_ts,
            "seq_timeseries_std" => seq_ts_std,
            "alpha_true" => alpha_true,
            "alpha_anchor_mean" => vec(mean(alpha_anchor_all, dims=1)),
            "alpha_anchor_std" => vec(std(alpha_anchor_all, dims=1)),
            "alpha_seq_mean" => vec(mean(alpha_seq_all, dims=1)),
            "alpha_seq_std" => vec(std(alpha_seq_all, dims=1)),
            "n_reps" => length(reps)
        )
    end
    return agg
end

# =============================================================================
# Process each condition
# =============================================================================

const DATA_DIR = joinpath("data", "alg4", "anchor_experiment")
const RESULTS_DIR = joinpath("results", "alg4", "anchor_experiment")

function process_condition(condition_name::String, group_key::String)
    println("\nProcessing condition: " * condition_name)
    cond_dir = joinpath(DATA_DIR, condition_name)

    if !isdir(cond_dir)
        println("  WARNING: directory not found: " * cond_dir)
        return nothing
    end

    files = filter(f -> endswith(f, ".jls"), readdir(cond_dir))
    println("  Found " * string(length(files)) * " data files")

    results = Dict[]
    for f in files
        try
            r = evaluate_single(joinpath(cond_dir, f))
            push!(results, r)
        catch e
            println("  Error processing " * f * ": " * string(e))
        end
    end

    if isempty(results)
        println("  No results computed")
        return nothing
    end

    agg = aggregate_results(results, group_key)

    # Print summary
    for k in sort(collect(keys(agg)))
        a = agg[k]
        println("  " * group_key * "=" * string(k) *
                ": anchor=" * string(round(a["anchor_mean"], digits=6)) *
                "±" * string(round(a["anchor_std"], digits=6)) *
                ", seq=" * string(round(a["seq_mean"], digits=6)) *
                "±" * string(round(a["seq_std"], digits=6)))
    end

    return Dict(
        "condition" => condition_name,
        "group_key" => group_key,
        "aggregated" => agg,
        "raw_results" => results
    )
end

function main()
    println("Anchor-Based Alignment Experiment — Alignment & Evaluation")
    mkpath(RESULTS_DIR)

    all_results = Dict{String, Any}()

    # Condition 1: anchor count
    r = process_condition("anchor_count", "n_anchors")
    if !isnothing(r)
        all_results["anchor_count"] = r
    end

    # Condition 2: drifting anchors
    r = process_condition("drifting", "epsilon")
    if !isnothing(r)
        all_results["drifting"] = r
    end

    # Condition 3: error vs T
    r = process_condition("error_vs_T", "T")
    if !isnothing(r)
        all_results["error_vs_T"] = r
    end

    # Condition 4: spectral gap
    r = process_condition("spectral_gap", "scale")
    if !isnothing(r)
        all_results["spectral_gap"] = r
    end

    # Save all results
    out_path = joinpath(RESULTS_DIR, "anchor_experiment_results.jls")
    serialize(out_path, all_results)
    println("\n" * "="^60)
    println("All results saved to: " * out_path)
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
