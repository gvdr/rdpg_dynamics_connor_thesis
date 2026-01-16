"""
Shared RDPG Pipeline Module for Examples

MODULAR DESIGN - Each phase saves/loads intermediate data:

Phase 1: generate_data     → saves data/example_X/true_dynamics.jls
Phase 2: rdpg_estimate     → saves data/example_X/estimated.jls
Phase 3: train_models      → saves data/example_X/model_Y.jls
Phase 4: evaluate          → saves data/example_X/metrics.jls
Phase 5: visualize         → saves results/example_X/*.pdf

Usage (from a script with --project activated):
    include("shared_pipeline.jl")

    # Use phase functions directly
    phase_generate(example_name, dynamics!, X0; T_end=25.0, dt=1.0)
    phase_estimate(example_name; K=100)
    phase_train(example_name, model_name, dynamics_fn, ps_init; ...)
    phase_evaluate(example_name, model_names)
    phase_visualize(example_name, model_names)

Note: This module expects the project environment to be already activated
(via `julia --project` or shebang `#!/usr/bin/env -S julia --project`).
"""

using LinearAlgebra
using Statistics
using Random
using Serialization

# SciML stack
using Lux
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using ComponentArrays
using Zygote

# Plotting
import CairoMakie
const CM = CairoMakie

# RDPG library
using RDPGDynamics: sample_adjacency_repeated, embed_temporal_network_Bd_plus

# =============================================================================
# Type alias and paths
# =============================================================================
const F = Float32

const DATA_DIR = "data"
const RESULTS_DIR = "results"

# =============================================================================
# Data Management
# =============================================================================

"""Get path for intermediate data file."""
function data_path(example_name::String, filename::String)
    dir = joinpath(DATA_DIR, example_name)
    mkpath(dir)
    return joinpath(dir, filename)
end

"""Get path for results file."""
function results_path(example_name::String, filename::String)
    dir = joinpath(RESULTS_DIR, example_name)
    mkpath(dir)
    return joinpath(dir, filename)
end

"""Save intermediate data."""
function save_data(example_name::String, filename::String, data)
    path = data_path(example_name, filename)
    serialize(path, data)
    println("  Saved: " * path)
    return path
end

"""Load intermediate data."""
function load_data(example_name::String, filename::String)
    path = data_path(example_name, filename)
    if !isfile(path)
        error("Data file not found: " * path * "\nRun earlier pipeline phases first.")
    end
    data = deserialize(path)
    println("  Loaded: " * path)
    return data
end

"""Check if data file exists."""
function data_exists(example_name::String, filename::String)
    return isfile(data_path(example_name, filename))
end

# =============================================================================
# RDPG Estimation Pipeline
# =============================================================================

"""
Embed temporal series with B^d_+ alignment.

Pipeline:
1. Generate K adjacency samples per timestep from true positions
2. Embed with temporal Procrustes + B^d_+ alignment
3. Convert to Float32 for training

Args:
    X_true_series: Vector of n×d matrices (true positions at each time)
    d: Embedding dimension
    K: Number of adjacency samples per timestep
    T: Output type (default Float32)

Returns:
    Vector of n×d matrices in B^d_+ (estimated positions)
"""
function rdpg_estimate(X_true_series::Vector, d::Int, K::Int, ::Type{T}=F) where T
    n_times = length(X_true_series)

    # Step 1: Generate averaged adjacency matrices
    A_series = Vector{Matrix{Float64}}(undef, n_times)
    for t in 1:n_times
        A_series[t] = sample_adjacency_repeated(X_true_series[t], K)
    end

    # Step 2: Embed with B^d_+ alignment
    L_series = embed_temporal_network_Bd_plus(A_series, d)

    # Step 3: Convert to target type
    X_est_series = Vector{Matrix{T}}(undef, n_times)
    for t in 1:n_times
        X_est_series[t] = T.(L_series[t])
    end

    return X_est_series
end

"""Print diagnostic info about estimated positions."""
function print_rdpg_diagnostics(X_est_series::Vector; name::String="")
    X_all = vcat(X_est_series...)
    n_times = length(X_est_series)
    n, d = size(X_est_series[1])

    neg_count = sum(X_all .< 0)
    max_norm = maximum([norm(X_all[i, :]) for i in 1:size(X_all, 1)])

    prefix = isempty(name) ? "" : "[" * name * "] "
    println("   " * prefix * "Embedded " * string(n_times) * " timesteps (n=" * string(n) * ", d=" * string(d) * ")")
    println("   " * prefix * "B^d_+ check: neg=" * string(neg_count) * ", max_norm=" * string(round(max_norm, digits=3)))

    for j in 1:d
        col = X_all[:, j]
        println("   " * prefix * "x" * string(j) * " range: [" *
                string(round(minimum(col), digits=3)) * ", " *
                string(round(maximum(col), digits=3)) * "]")
    end
end

# =============================================================================
# Evaluation Metrics
# =============================================================================

"""Compute pairwise distance matrix."""
function pairwise_distances(X::AbstractMatrix)
    sq_norms = sum(X .^ 2, dims=2)
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    D_sq = max.(D_sq, 0.0)
    return sqrt.(D_sq .+ 1e-10)
end

"""Extract upper triangle of matrix as vector."""
function upper_tri(M::AbstractMatrix)
    n = size(M, 1)
    return [M[i, j] for i in 1:n for j in i+1:n]
end

"""
Evaluate similarity between two position matrices using rotation-invariant metrics.

Returns:
    D_corr: Correlation of pairwise distances
    P_err: Relative error in probability matrix reconstruction
"""
function evaluate_positions(X1::AbstractMatrix, X2::AbstractMatrix)
    X1_f64 = Float64.(X1)
    X2_f64 = Float64.(X2)

    # Distance correlation
    D1 = pairwise_distances(X1_f64)
    D2 = pairwise_distances(X2_f64)
    d1_vec, d2_vec = upper_tri(D1), upper_tri(D2)
    D_corr = cor(d1_vec, d2_vec)

    # P reconstruction error
    P1 = X1_f64 * X1_f64'
    P2 = X2_f64 * X2_f64'
    P_err = norm(P1 - P2) / max(norm(P2), 1e-10)

    return (D_corr=D_corr, P_err=P_err)
end

"""Evaluate two temporal series, returning metrics at each timestep."""
function evaluate_series(X_series_1::Vector, X_series_2::Vector)
    n_times = length(X_series_1)
    D_corr = [evaluate_positions(X_series_1[t], X_series_2[t]).D_corr for t in 1:n_times]
    P_err = [evaluate_positions(X_series_1[t], X_series_2[t]).P_err for t in 1:n_times]
    return (D_corr=D_corr, P_err=P_err)
end

"""Print evaluation summary."""
function print_evaluation(metrics; name::String="", baseline_metrics=nothing)
    avg_D = mean(metrics.D_corr)
    avg_P = mean(metrics.P_err)

    if isnothing(baseline_metrics)
        println("  " * name * ": D_corr=" * string(round(avg_D, digits=4)) *
                ", P_err=" * string(round(avg_P, digits=4)))
    else
        ratio_D = avg_D / mean(baseline_metrics.D_corr)
        ratio_P = avg_P / mean(baseline_metrics.P_err)
        println("  " * name * ": D_corr=" * string(round(avg_D, digits=4)) *
                " (" * string(round(100*ratio_D, digits=1)) * "% of baseline)" *
                ", P_err=" * string(round(avg_P, digits=4)) *
                " (" * string(round(100*ratio_P, digits=1)) * "%)")
    end
end

# =============================================================================
# Training Utilities
# =============================================================================

"""
Position-based loss function for trajectory prediction.
"""
function position_loss(pred_series::Vector, target_series::Vector)
    loss = zero(F)
    for (pred, target) in zip(pred_series, target_series)
        loss += sum(abs2, pred .- target)
    end
    return loss / F(length(pred_series))
end

"""
Train a dynamics model using ADAM + BFGS with BacksolveAdjoint.

Args:
    X_train_series: Training data (vector of n×d matrices)
    dynamics_fn: Dynamics function (u, p, t) -> du
    ps_init: Initial parameters (ComponentArray)
    dt: Time step
    epochs_adam: ADAM iterations
    epochs_bfgs: BFGS iterations
    lr: Learning rate
    name: Model name for logging

Returns:
    (trained_params, loss_history)
"""
function train_dynamics(X_train_series::Vector{Matrix{F}},
                        dynamics_fn,
                        ps_init::ComponentArray{F};
                        dt::F=1.0f0,
                        epochs_adam::Int=500,
                        epochs_bfgs::Int=100,
                        lr::F=0.01f0,
                        name::String="model",
                        verbose::Bool=true)

    n, d = size(X_train_series[1])
    T_train = length(X_train_series)

    tsteps = range(zero(F), step=dt, length=T_train)
    tspan = (zero(F), tsteps[end])
    u0 = vec(X_train_series[1])

    # Optimized sensitivity algorithm
    sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP())

    function total_loss(p)
        prob = ODEProblem(dynamics_fn, u0, tspan, p)
        sol = solve(prob, Tsit5(),
                    saveat=tsteps,
                    sensealg=sensealg,
                    abstol=F(1e-5),
                    reltol=F(1e-5))

        if sol.retcode != :Success
            return F(1e8)
        end

        pred_series = [reshape(sol.u[t], n, d) for t in 1:T_train]

        for pred in pred_series
            if any(isnan, pred) || any(isinf, pred)
                return F(1e8)
            end
        end

        return position_loss(pred_series, X_train_series)
    end

    # Training callback
    iter = Ref(0)
    losses = F[]

    function callback(state, l)
        iter[] += 1
        push!(losses, F(l))
        if verbose && iter[] % 50 == 0
            println("    [" * name * "] Iter " * string(iter[]) * ": loss = " * string(round(l, digits=6)))
        end
        return false
    end

    if verbose
        println("  Training " * name * "...")
        println("    Initial loss: " * string(round(total_loss(ps_init), digits=4)))
        println("    ADAM phase (" * string(epochs_adam) * " iterations)...")
    end

    # ADAM phase
    optf = Optimization.OptimizationFunction((p, _) -> total_loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(lr),
        maxiters=epochs_adam,
        callback=callback
    )

    # BFGS phase
    if verbose
        println("    BFGS phase (" * string(epochs_bfgs) * " iterations)...")
    end

    optprob_bfgs = Optimization.OptimizationProblem(optf, result_adam.u)

    result_bfgs = try
        Optimization.solve(
            optprob_bfgs,
            OptimizationOptimJL.BFGS(initial_stepnorm=F(0.01)),
            maxiters=epochs_bfgs,
            callback=callback,
            allow_f_increases=false
        )
    catch e
        if verbose
            println("    BFGS failed, using ADAM result")
        end
        result_adam
    end

    if verbose
        println("    Final loss: " * string(round(result_bfgs.objective, digits=6)))
    end

    return result_bfgs.u, losses
end

"""
Recover full trajectory using trained dynamics.
"""
function recover_trajectory(X_est_series::Vector{Matrix{F}},
                           dynamics_fn,
                           ps_trained::ComponentArray{F};
                           dt::F=1.0f0)
    n, d = size(X_est_series[1])
    T_total = length(X_est_series)

    tsteps = range(zero(F), step=dt, length=T_total)
    tspan = (zero(F), tsteps[end])
    u0 = vec(X_est_series[1])

    prob = ODEProblem(dynamics_fn, u0, tspan, ps_trained)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=F(1e-7), reltol=F(1e-7))

    return [reshape(sol.u[t], n, d) for t in 1:T_total]
end

# =============================================================================
# Neural Network Builders
# =============================================================================

"""Build a neural network for dynamics learning."""
function build_nn(input_dim::Int, hidden_sizes::Vector{Int}, output_dim::Int;
                  activation=tanh, rng=Random.default_rng())
    layers = []
    in_dim = input_dim

    for h in hidden_sizes
        push!(layers, Lux.Dense(in_dim, h, activation))
        in_dim = h
    end
    push!(layers, Lux.Dense(in_dim, output_dim))

    return Lux.Chain(layers...)
end

"""Build a smaller correction network for UDE."""
function build_correction_nn(input_dim::Int, output_dim::Int;
                             hidden_size::Int=32, rng=Random.default_rng())
    return Lux.Chain(
        Lux.Dense(input_dim, hidden_size, tanh),
        Lux.Dense(hidden_size, hidden_size ÷ 2, tanh),
        Lux.Dense(hidden_size ÷ 2, output_dim)
    )
end

# =============================================================================
# Visualization Helpers
# =============================================================================

"""Plot trajectory evolution in 2D."""
function plot_trajectory_evolution(X_series, communities::Vector{<:AbstractVector};
                                   title::String="Trajectory Evolution",
                                   colors=[:blue, :red, :green, :orange])
    fig = CM.Figure(size=(600, 600))
    ax = CM.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title=title, aspect=CM.DataAspect())

    n_times = length(X_series)

    for (c_idx, community) in enumerate(communities)
        color = colors[mod1(c_idx, length(colors))]
        for i in community
            xs = [X_series[t][i, 1] for t in 1:n_times]
            ys = [X_series[t][i, 2] for t in 1:n_times]
            CM.lines!(ax, xs, ys, color=(color, 0.5), linewidth=1)
            CM.scatter!(ax, [xs[1]], [ys[1]], color=color, markersize=6)
            CM.scatter!(ax, [xs[end]], [ys[end]], color=color, marker=:star5, markersize=8)
        end
    end

    return fig
end

"""Plot comparison of true, estimated, and recovered trajectories."""
function plot_trajectory_comparison(X_true_series, X_est_series, X_rec_series,
                                   communities::Vector{<:AbstractVector}, T_train::Int;
                                   title::String="Trajectory Comparison")
    fig = CM.Figure(size=(1400, 500))

    n_times = length(X_true_series)
    timesteps = [1, T_train, n_times]
    titles = ["t=1 (start)", "t=" * string(T_train) * " (train end)", "t=" * string(n_times) * " (final)"]

    # Compute global bounds
    all_x1, all_x2 = Float64[], Float64[]
    for series in [X_true_series, X_est_series, X_rec_series]
        for t in timesteps
            append!(all_x1, vec(series[t][:, 1]))
            append!(all_x2, vec(series[t][:, 2]))
        end
    end
    x_min, x_max = extrema(all_x1)
    y_min, y_max = extrema(all_x2)
    x_pad = 0.1 * max(x_max - x_min, 0.1)
    y_pad = 0.1 * max(y_max - y_min, 0.1)

    colors = [:blue, :red, :green, :orange]

    for (idx, t) in enumerate(timesteps)
        ax = CM.Axis(fig[1, idx], xlabel="x₁", ylabel="x₂", title=titles[idx], aspect=CM.DataAspect())

        for (c_idx, community) in enumerate(communities)
            color = colors[mod1(c_idx, length(colors))]

            # True
            CM.scatter!(ax, X_true_series[t][community, 1], X_true_series[t][community, 2],
                       color=color, marker=:circle, markersize=10, label=idx==1 ? "True C" * string(c_idx) : nothing)
            # Estimated
            CM.scatter!(ax, X_est_series[t][community, 1], X_est_series[t][community, 2],
                       color=(color, 0.4), marker=:diamond, markersize=8, label=idx==1 ? "Est C" * string(c_idx) : nothing)
            # Recovered
            CM.scatter!(ax, X_rec_series[t][community, 1], X_rec_series[t][community, 2],
                       color=color, marker=:star5, markersize=12, label=idx==1 ? "Rec C" * string(c_idx) : nothing)
        end

        CM.xlims!(ax, x_min - x_pad, x_max + x_pad)
        CM.ylims!(ax, y_min - y_pad, y_max + y_pad)

        if idx == 1
            CM.Legend(fig[1, 4], ax, framevisible=false)
        end
    end

    CM.Label(fig[0, :], title, fontsize=16)
    return fig
end

"""Plot metrics comparison."""
function plot_metrics(metrics_dict::Dict, T_train::Int; title::String="Metrics Over Time")
    fig = CM.Figure(size=(1200, 400))
    colors = [:blue, :red, :green, :orange, :purple, :brown]

    ax1 = CM.Axis(fig[1, 1], xlabel="Time", ylabel="D Correlation", title="Distance Correlation")
    for (i, (name, metrics)) in enumerate(metrics_dict)
        n_t = length(metrics.D_corr)
        CM.lines!(ax1, 1:n_t, metrics.D_corr, color=colors[mod1(i, length(colors))],
                 linewidth=2, label=name)
    end
    CM.vlines!(ax1, [T_train], color=:gray, linestyle=:dash)
    CM.Legend(fig[1, 2], ax1, framevisible=false)

    ax2 = CM.Axis(fig[1, 3], xlabel="Time", ylabel="P Error", title="P Reconstruction Error")
    for (i, (name, metrics)) in enumerate(metrics_dict)
        n_t = length(metrics.P_err)
        CM.lines!(ax2, 1:n_t, metrics.P_err, color=colors[mod1(i, length(colors))],
                 linewidth=2)
    end
    CM.vlines!(ax2, [T_train], color=:gray, linestyle=:dash)

    CM.Label(fig[0, :], title, fontsize=16)
    return fig
end

"""Plot learning curves."""
function plot_learning_curves(losses_dict::Dict; title::String="Learning Curves")
    fig = CM.Figure(size=(800, 400))
    ax = CM.Axis(fig[1, 1], xlabel="Iteration", ylabel="Loss (log scale)",
                 title=title, yscale=log10)

    colors = [:blue, :red, :green, :orange, :purple, :brown]

    for (i, (name, losses)) in enumerate(losses_dict)
        valid_losses = [l for l in losses if l < 1e6 && l > 0]
        if !isempty(valid_losses)
            CM.lines!(ax, 1:length(valid_losses), valid_losses,
                     color=colors[mod1(i, length(colors))], linewidth=2, label=name)
        end
    end

    CM.Legend(fig[1, 2], ax, framevisible=false)
    return fig
end

# =============================================================================
# Summary Table
# =============================================================================

"""Print a summary table of results."""
function print_summary_table(metrics_dict::Dict, T_train::Int; baseline_key::String="Baseline")
    println("\n" * "=" ^ 80)
    println("SUMMARY TABLE")
    println("=" ^ 80)
    println()
    println("Model              | Avg D corr | Avg P err | Train D | Val D  | Train P | Val P")
    println("-" ^ 80)

    baseline = haskey(metrics_dict, baseline_key) ? metrics_dict[baseline_key] : nothing

    for (name, m) in metrics_dict
        if name == baseline_key
            println(rpad(name, 18) * " | " *
                    rpad(string(round(mean(m.D_corr), digits=3)), 10) * " | " *
                    rpad(string(round(mean(m.P_err), digits=3)), 9) * " | " *
                    "   -    |   -    |    -    |   -")
        else
            n_times = length(m.D_corr)
            D_train = round(mean(m.D_corr[1:T_train]), digits=3)
            D_val = round(mean(m.D_corr[T_train+1:end]), digits=3)
            P_train = round(mean(m.P_err[1:T_train]), digits=3)
            P_val = round(mean(m.P_err[T_train+1:end]), digits=3)

            println(rpad(name, 18) * " | " *
                    rpad(string(round(mean(m.D_corr), digits=3)), 10) * " | " *
                    rpad(string(round(mean(m.P_err), digits=3)), 9) * " | " *
                    rpad(string(D_train), 7) * " | " *
                    rpad(string(D_val), 6) * " | " *
                    rpad(string(P_train), 7) * " | " *
                    string(P_val))
        end
    end
end

# =============================================================================
# Phase Runners
# =============================================================================

"""
Standard file names for intermediate data.
"""
const FILES = (
    true_dynamics = "true_dynamics.jls",
    estimated = "estimated.jls",
    config = "config.jls"
)

"""
Phase 1: Generate true dynamics and save.

Args:
    example_name: Name for this example (used for file paths)
    dynamics_fn!: In-place dynamics function (du, u, p, t)
    X0: Initial positions (n × d matrix)
    T_end: End time
    dt: Time step
    metadata: Additional metadata to save (communities, centers, etc.)
"""
function phase_generate(example_name::String, dynamics_fn!, X0::Matrix{Float64};
                        T_end::Float64=25.0, dt::Float64=1.0,
                        metadata::Dict=Dict(), seed::Int=42)
    println("\n" * "=" ^ 60)
    println("PHASE 1: Generate True Dynamics")
    println("=" ^ 60)

    n, d = size(X0)
    T_steps = Int(T_end / dt) + 1
    tspan = (0.0, T_end)
    tsteps = range(0.0, T_end, length=T_steps)

    prob = ODEProblem(dynamics_fn!, vec(X0), tspan)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-7, reltol=1e-7)

    X_true_series = [reshape(sol.u[t], n, d) for t in 1:T_steps]

    println("  Generated " * string(T_steps) * " timesteps")
    println("  n=" * string(n) * " nodes, d=" * string(d) * " dimensions")

    # Save
    data = Dict(
        "X_true_series" => X_true_series,
        "X0" => X0,
        "n" => n,
        "d" => d,
        "T_end" => T_end,
        "dt" => dt,
        "T_steps" => T_steps,
        "seed" => seed,
        "metadata" => metadata
    )
    save_data(example_name, FILES.true_dynamics, data)

    return data
end

"""
Phase 2: RDPG estimation from true dynamics.

Args:
    example_name: Name for this example
    K: Number of adjacency samples per timestep
"""
function phase_estimate(example_name::String; K::Int=100)
    println("\n" * "=" ^ 60)
    println("PHASE 2: RDPG Estimation")
    println("=" ^ 60)

    # Load true dynamics
    true_data = load_data(example_name, FILES.true_dynamics)
    X_true_series = true_data["X_true_series"]
    n, d = true_data["n"], true_data["d"]

    println("  K=" * string(K) * " samples per timestep")

    # RDPG estimation
    X_est_series = rdpg_estimate(X_true_series, d, K, F)
    print_rdpg_diagnostics(X_est_series)

    # Baseline metrics (True ↔ Estimated)
    baseline_metrics = evaluate_series(X_true_series, X_est_series)
    println("\n  Baseline (True ↔ Estimated):")
    println("    Avg D corr: " * string(round(mean(baseline_metrics.D_corr), digits=4)))
    println("    Avg P err:  " * string(round(mean(baseline_metrics.P_err), digits=4)))

    # Save
    data = Dict(
        "X_est_series" => X_est_series,
        "K" => K,
        "baseline_metrics" => baseline_metrics
    )
    save_data(example_name, FILES.estimated, data)

    return data
end

"""
Phase 3: Train a single model.

Args:
    example_name: Name for this example
    model_name: Name for this model (e.g., "pure_nn", "ude_absolute")
    make_dynamics_fn: Function (nn, st, params...) -> dynamics_fn
    nn: Neural network
    train_fraction: Fraction of data for training
"""
function phase_train(example_name::String, model_name::String,
                     dynamics_fn, ps_init::ComponentArray{F};
                     train_fraction::Float64=0.75,
                     epochs_adam::Int=500, epochs_bfgs::Int=100,
                     lr::F=0.01f0, dt::F=1.0f0)
    println("\n" * "=" ^ 60)
    println("PHASE 3: Train Model - " * model_name)
    println("=" ^ 60)

    # Load estimated data
    est_data = load_data(example_name, FILES.estimated)
    X_est_series = est_data["X_est_series"]

    T_total = length(X_est_series)
    T_train = Int(floor(train_fraction * T_total))
    X_train = X_est_series[1:T_train]

    println("  Train: " * string(T_train) * ", Val: " * string(T_total - T_train))
    println("  Parameters: " * string(length(ps_init)))

    # Train
    ps_trained, losses = train_dynamics(
        X_train, dynamics_fn, ps_init;
        dt=dt, epochs_adam=epochs_adam, epochs_bfgs=epochs_bfgs,
        lr=lr, name=model_name
    )

    # Recover trajectory
    X_rec_series = recover_trajectory(X_est_series, dynamics_fn, ps_trained; dt=dt)

    # Save
    model_file = "model_" * model_name * ".jls"
    data = Dict(
        "ps_trained" => ps_trained,
        "losses" => losses,
        "X_rec_series" => X_rec_series,
        "T_train" => T_train,
        "train_fraction" => train_fraction
    )
    save_data(example_name, model_file, data)

    return data
end

"""
Phase 4: Evaluate all trained models.
"""
function phase_evaluate(example_name::String, model_names::Vector{String})
    println("\n" * "=" ^ 60)
    println("PHASE 4: Evaluate Models")
    println("=" ^ 60)

    # Load data
    true_data = load_data(example_name, FILES.true_dynamics)
    est_data = load_data(example_name, FILES.estimated)

    X_true_series = true_data["X_true_series"]
    X_est_series = est_data["X_est_series"]
    baseline_metrics = est_data["baseline_metrics"]

    T_train = nothing
    metrics_dict = Dict{String, Any}()
    metrics_dict["Baseline"] = baseline_metrics

    for model_name in model_names
        model_file = "model_" * model_name * ".jls"
        if !data_exists(example_name, model_file)
            println("  Skipping " * model_name * " (not trained)")
            continue
        end

        model_data = load_data(example_name, model_file)
        X_rec_series = model_data["X_rec_series"]
        T_train = model_data["T_train"]

        # True ↔ Recovered
        metrics = evaluate_series(X_true_series, X_rec_series)
        metrics_dict[model_name] = metrics

        print_evaluation(metrics; name=model_name, baseline_metrics=baseline_metrics)
    end

    # Save
    eval_data = Dict(
        "metrics_dict" => metrics_dict,
        "T_train" => T_train
    )
    save_data(example_name, "evaluation.jls", eval_data)

    # Print summary table
    if !isnothing(T_train)
        print_summary_table(metrics_dict, T_train)
    end

    return eval_data
end

"""
Phase 5: Generate visualizations.
"""
function phase_visualize(example_name::String, model_names::Vector{String};
                         communities::Vector{<:AbstractVector}=Vector{Int}[])
    println("\n" * "=" ^ 60)
    println("PHASE 5: Visualizations")
    println("=" ^ 60)

    # Load data
    true_data = load_data(example_name, FILES.true_dynamics)
    est_data = load_data(example_name, FILES.estimated)
    eval_data = load_data(example_name, "evaluation.jls")

    X_true_series = true_data["X_true_series"]
    X_est_series = est_data["X_est_series"]
    metrics_dict = eval_data["metrics_dict"]
    T_train = eval_data["T_train"]

    n = true_data["n"]
    metadata = true_data["metadata"]

    # Use communities from metadata if not provided
    if isempty(communities) && haskey(metadata, "communities")
        communities = metadata["communities"]
    end
    if isempty(communities)
        communities = [collect(1:n)]  # All nodes as one community
    end

    # True trajectory evolution
    fig = plot_trajectory_evolution(X_true_series, communities; title="True Dynamics")
    path = results_path(example_name, "evolution_true.pdf")
    CM.save(path, fig)
    println("  Saved: " * path)

    # Estimated trajectory evolution
    fig = plot_trajectory_evolution(X_est_series, communities; title="Estimated (RDPG)")
    path = results_path(example_name, "evolution_estimated.pdf")
    CM.save(path, fig)
    println("  Saved: " * path)

    # Each model
    losses_dict = Dict{String, Vector{F}}()

    for model_name in model_names
        model_file = "model_" * model_name * ".jls"
        if !data_exists(example_name, model_file)
            continue
        end

        model_data = load_data(example_name, model_file)
        X_rec_series = model_data["X_rec_series"]
        losses_dict[model_name] = model_data["losses"]

        # Trajectory comparison
        fig = plot_trajectory_comparison(X_true_series, X_est_series, X_rec_series,
                                        communities, T_train; title=model_name)
        path = results_path(example_name, "comparison_" * model_name * ".pdf")
        CM.save(path, fig)
        println("  Saved: " * path)

        # Recovered evolution
        fig = plot_trajectory_evolution(X_rec_series, communities;
                                       title="Recovered: " * model_name)
        path = results_path(example_name, "evolution_" * model_name * ".pdf")
        CM.save(path, fig)
        println("  Saved: " * path)
    end

    # Metrics over time
    fig = plot_metrics(metrics_dict, T_train; title=example_name * " - Metrics")
    path = results_path(example_name, "metrics.pdf")
    CM.save(path, fig)
    println("  Saved: " * path)

    # Learning curves
    if !isempty(losses_dict)
        fig = plot_learning_curves(losses_dict; title=example_name * " - Learning Curves")
        path = results_path(example_name, "learning_curves.pdf")
        CM.save(path, fig)
        println("  Saved: " * path)
    end

    println("\n  Visualizations complete!")
end

# =============================================================================
# Convenience: Run Multiple Phases
# =============================================================================

"""
Check which phases need to run based on existing data.
"""
function check_phases(example_name::String)
    status = Dict{Symbol, Bool}()
    status[:generate] = data_exists(example_name, FILES.true_dynamics)
    status[:estimate] = data_exists(example_name, FILES.estimated)
    status[:evaluate] = data_exists(example_name, "evaluation.jls")
    return status
end

"""Print pipeline status."""
function print_status(example_name::String)
    println("\nPipeline status for: " * example_name)
    status = check_phases(example_name)
    for (phase, exists) in status
        mark = exists ? "✓" : "✗"
        println("  " * mark * " " * string(phase))
    end

    # Check for models
    dir = joinpath(DATA_DIR, example_name)
    if isdir(dir)
        models = filter(f -> startswith(f, "model_") && endswith(f, ".jls"), readdir(dir))
        if !isempty(models)
            println("  Models found:")
            for m in models
                println("    - " * replace(m, r"^model_|\.jls$" => ""))
            end
        end
    end
end

println("Shared pipeline module loaded.")
