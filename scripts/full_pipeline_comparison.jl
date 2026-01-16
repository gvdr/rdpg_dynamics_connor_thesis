#!/usr/bin/env julia
"""
Full Pipeline Comparison: Pure NN vs UDE

Optimized version with Float32 throughout for performance.
Based on SciML best practices.

Compares:
1. Pure Neural ODE (NN learns everything)
2. UDE (known oscillation structure + NN correction)

With comprehensive diagnostics:
- Trajectory visualization
- Distance correlation over time
- P reconstruction error over time
- Learning curves
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Random

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

# =============================================================================
# Type alias for Float32 throughout
# =============================================================================
const F = Float32

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct PipelineConfig
    # Problem setup
    n::Int = 12                          # Number of nodes
    d::Int = 2                           # Embedding dimension
    T_end::F = 25.0f0                    # End time (Float32)
    dt::F = 1.0f0                        # Time step (Float32)

    # RDPG sampling
    K_samples::Int = 100                 # Samples per timestep

    # Training
    hidden_sizes::Vector{Int} = [64, 64, 32]  # Network architecture
    epochs_adam::Int = 800               # ADAM iterations
    epochs_bfgs::Int = 100               # BFGS iterations
    lr::F = 0.01f0                       # Learning rate (Float32)

    # UDE parameters (known structure)
    omega_known::F = 0.3f0               # Known oscillation frequency
    k_attract_known::F = 0.1f0           # Known attraction strength

    # Train/val split
    train_fraction::F = 0.75f0

    # Random seed
    seed::Int = 42
end

# =============================================================================
# True Dynamics (for data generation) - uses Float64 for accuracy
# =============================================================================

function create_true_dynamics(n::Int, d::Int, community_1, community_2,
                              center_1::Vector{Float64}, center_2::Vector{Float64})
    omega = 0.3
    k_attract = 0.1
    k_cohesion = 0.05

    function true_dynamics!(du, u, p, t)
        X = reshape(u, n, d)
        dX = zeros(n, d)

        X_bar1 = mean(X[community_1, :], dims=1)[1, :]
        X_bar2 = mean(X[community_2, :], dims=1)[1, :]

        for i in 1:n
            if i in community_1
                center = center_1
                X_bar = X_bar1
            else
                center = center_2
                X_bar = X_bar2
            end

            # Oscillation around center
            delta = X[i, :] .- center
            dX[i, 1] = -omega * delta[2] - k_attract * delta[1]
            dX[i, 2] = omega * delta[1] - k_attract * delta[2]

            # Cohesion
            dX[i, :] .+= k_cohesion .* (X_bar .- X[i, :])

            # Soft boundary
            for j in 1:d
                if X[i, j] < 0.15
                    dX[i, j] += 0.2 * exp(-(X[i, j] - 0.1) / 0.03)
                end
                if X[i, j] > 0.85
                    dX[i, j] -= 0.2 * exp((X[i, j] - 0.9) / 0.03)
                end
            end
        end

        du .= vec(dX)
    end

    return true_dynamics!
end

# =============================================================================
# RDPG Estimation - using RDPGDynamics library
# =============================================================================

# Import RDPG functions from RDPGDynamics
using RDPGDynamics: sample_adjacency_repeated, embed_temporal_network_Bd_plus

"""
Embed temporal series with B^d_+ alignment using RDPGDynamics functions.

Simplified pipeline using library functions:
1. Sample K adjacency matrices per timestep and average
2. Embed with temporal Procrustes + B^d_+ alignment (via embed_temporal_network_Bd_plus)
3. Convert to Float32 for training

Result: All estimated positions are in B^d_+ (x ≥ 0, ||x|| ≤ 1).
"""
function embed_series(X_true_series::Vector, d::Int, K::Int, ::Type{T}) where T
    n_times = length(X_true_series)

    # Step 1: Generate averaged adjacency matrices from true positions
    A_series = Vector{Matrix{Float64}}(undef, n_times)
    for t in 1:n_times
        A_series[t] = sample_adjacency_repeated(X_true_series[t], K)
    end

    # Step 2: Embed with B^d_+ alignment (handles Procrustes chain + global alignment + projection)
    L_series = embed_temporal_network_Bd_plus(A_series, d)

    # Step 3: Convert to target type (Float32 for training)
    X_est_series = Vector{Matrix{T}}(undef, n_times)
    for t in 1:n_times
        X_est_series[t] = T.(L_series[t])
    end

    # Print diagnostic info
    X_all = vcat(X_est_series...)
    neg_count = sum(X_all .< 0)
    max_norm = maximum([norm(X_all[i, :]) for i in 1:size(X_all, 1)])

    println("   Embedded " * string(n_times) * " timesteps")
    println("   B^d_+ check: neg=" * string(neg_count) * ", max_norm=" * string(round(max_norm, digits=3)))
    println("   Position range: x1=[" * string(round(minimum(X_all[:, 1]), digits=3)) * ", " *
            string(round(maximum(X_all[:, 1]), digits=3)) * "], x2=[" *
            string(round(minimum(X_all[:, 2]), digits=3)) * ", " *
            string(round(maximum(X_all[:, 2]), digits=3)) * "]")

    return X_est_series
end

# =============================================================================
# Evaluation Metrics - uses Float64 for accuracy
# =============================================================================

function pairwise_distances(X::AbstractMatrix)
    sq_norms = sum(X .^ 2, dims=2)
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    D_sq = max.(D_sq, 0.0)
    return sqrt.(D_sq .+ 1e-10)
end

function upper_tri(M::AbstractMatrix)
    n = size(M, 1)
    return [M[i, j] for i in 1:n for j in i+1:n]
end

function evaluate_at_time(X1::AbstractMatrix, X2::AbstractMatrix)
    # Convert to Float64 for accurate evaluation
    X1_f64 = Float64.(X1)
    X2_f64 = Float64.(X2)

    D1 = pairwise_distances(X1_f64)
    D2 = pairwise_distances(X2_f64)
    d1_vec, d2_vec = upper_tri(D1), upper_tri(D2)

    D_corr = cor(d1_vec, d2_vec)
    D_rmse = sqrt(mean((d1_vec .- d2_vec).^2))

    P1 = X1_f64 * X1_f64'
    P2 = X2_f64 * X2_f64'
    P_err = norm(P1 - P2) / max(norm(P2), 1e-10)

    return (D_corr=D_corr, D_rmse=D_rmse, P_err=P_err)
end

function evaluate_series(X_series_1::Vector, X_series_2::Vector)
    n_times = length(X_series_1)
    D_corr = [evaluate_at_time(X_series_1[t], X_series_2[t]).D_corr for t in 1:n_times]
    D_rmse = [evaluate_at_time(X_series_1[t], X_series_2[t]).D_rmse for t in 1:n_times]
    P_err = [evaluate_at_time(X_series_1[t], X_series_2[t]).P_err for t in 1:n_times]
    return (D_corr=D_corr, D_rmse=D_rmse, P_err=P_err)
end

# =============================================================================
# Neural Network Architectures
# =============================================================================

"""Build pure NN for dynamics with tanh activation."""
function build_pure_nn(input_dim::Int, hidden_sizes::Vector{Int}, output_dim::Int;
                       rng=Random.default_rng())
    layers = []
    in_dim = input_dim

    for h in hidden_sizes
        push!(layers, Lux.Dense(in_dim, h, tanh))
        in_dim = h
    end
    push!(layers, Lux.Dense(in_dim, output_dim))

    return Lux.Chain(layers...)
end

"""Build smaller NN for UDE correction term."""
function build_correction_nn(input_dim::Int, output_dim::Int; rng=Random.default_rng())
    return Lux.Chain(
        Lux.Dense(input_dim, 32, tanh),
        Lux.Dense(32, 16, tanh),
        Lux.Dense(16, output_dim)
    )
end

# =============================================================================
# Dynamics Functions - Type stable, Float32
# =============================================================================

"""Pure Neural ODE dynamics: du/dt = NN(u)"""
function make_pure_nn_dynamics(nn, st, n::Int, d::Int)
    nd = n * d
    function dynamics(u::AbstractVector{T}, p, t) where T
        u_reshape = reshape(u, nd, 1)
        out, _ = nn(u_reshape, p, st)
        return vec(out)
    end
    return dynamics
end

"""
UDE dynamics: du/dt = f_known(u) + NN_correction(u)

Known structure: oscillation around community centers
NN learns: corrections, boundary effects, inter-community coupling
Fully vectorized for Zygote compatibility.
"""
function make_ude_dynamics(nn, st, n::Int, d::Int, community_1, community_2,
                           center_1::Vector{T}, center_2::Vector{T},
                           omega::T, k_attract::T) where T
    # Precompute center matrix (n x d) - each row is the center for that node
    centers = zeros(T, n, d)
    for i in community_1
        centers[i, :] .= center_1
    end
    for i in community_2
        centers[i, :] .= center_2
    end

    nd = n * d

    function dynamics(u::AbstractVector{T2}, p, t) where T2
        X = reshape(u, n, d)

        # Compute delta from centers (vectorized, type-stable)
        delta = X .- T2.(centers)

        # Known oscillation (d=2 specific but could be generalized)
        dX_known_1 = @. -T2(omega) * delta[:, 2] - T2(k_attract) * delta[:, 1]
        dX_known_2 = @. T2(omega) * delta[:, 1] - T2(k_attract) * delta[:, 2]
        dX_known = hcat(dX_known_1, dX_known_2)

        # NN correction
        u_reshape = reshape(u, nd, 1)
        correction, _ = nn(u_reshape, p, st)

        return vec(dX_known) .+ vec(correction)
    end
    return dynamics
end

# =============================================================================
# Training - Optimized with BacksolveAdjoint
# =============================================================================

function position_loss(pred_series::Vector, target_series::Vector)
    loss = zero(F)
    for (pred, target) in zip(pred_series, target_series)
        loss += sum(abs2, pred .- target)
    end
    return loss / F(length(pred_series))
end

function train_dynamics(config::PipelineConfig,
                        X_train_series::Vector{Matrix{F}},
                        dynamics_fn,
                        ps_init::ComponentArray{F};
                        name::String="model")

    n, d = config.n, config.d
    T_train = length(X_train_series)

    # Float32 time parameters
    tsteps = range(zero(F), step=config.dt, length=T_train)
    tspan = (zero(F), tsteps[end])
    u0 = vec(X_train_series[1])

    # Optimized sensitivity algorithm for non-stiff Neural ODE
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
        if iter[] % 50 == 0
            println("    [" * name * "] Iter " * string(iter[]) * ": loss = " * string(round(l, digits=6)))
        end
        return false
    end

    println("  Training " * name * "...")
    println("    Initial loss: " * string(round(total_loss(ps_init), digits=4)))

    # ADAM phase
    println("    ADAM phase (" * string(config.epochs_adam) * " iterations)...")
    optf = Optimization.OptimizationFunction((p, _) -> total_loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(config.lr),
        maxiters=config.epochs_adam,
        callback=callback
    )

    # BFGS phase
    println("    BFGS phase (" * string(config.epochs_bfgs) * " iterations)...")
    optprob_bfgs = Optimization.OptimizationProblem(optf, result_adam.u)

    result_bfgs = Optimization.solve(
        optprob_bfgs,
        OptimizationOptimJL.BFGS(initial_stepnorm=F(0.01)),
        maxiters=config.epochs_bfgs,
        callback=callback,
        allow_f_increases=false
    )

    println("    Final loss: " * string(round(result_bfgs.objective, digits=6)))

    return result_bfgs.u, losses
end

# =============================================================================
# Trajectory Recovery
# =============================================================================

function recover_trajectory(config::PipelineConfig, X_est_series::Vector{Matrix{F}},
                           dynamics_fn, ps_trained::ComponentArray{F})
    n, d = config.n, config.d
    T_total = length(X_est_series)

    tsteps = range(zero(F), step=config.dt, length=T_total)
    tspan = (zero(F), tsteps[end])
    u0 = vec(X_est_series[1])

    prob = ODEProblem(dynamics_fn, u0, tspan, ps_trained)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=F(1e-7), reltol=F(1e-7))

    return [reshape(sol.u[t], n, d) for t in 1:T_total]
end

# =============================================================================
# Visualization
# =============================================================================

function plot_trajectories(X_true_series, X_est_series, X_rec_series,
                          community_1, community_2, T_train::Int;
                          title="Trajectory Comparison")
    fig = CM.Figure(size=(1400, 500))

    n_times = length(X_true_series)
    timesteps = [1, T_train, n_times]
    titles = ["t=1 (start)", "t=" * string(T_train) * " (train end)", "t=" * string(n_times) * " (final)"]

    # Compute global bounds across all series for consistent scaling
    all_x1 = Float64[]
    all_x2 = Float64[]
    for series in [X_true_series, X_est_series, X_rec_series]
        for t in timesteps
            append!(all_x1, vec(series[t][:, 1]))
            append!(all_x2, vec(series[t][:, 2]))
        end
    end
    x_min, x_max = extrema(all_x1)
    y_min, y_max = extrema(all_x2)
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)

    for (idx, t) in enumerate(timesteps)
        ax = CM.Axis(fig[1, idx],
                     xlabel="x₁", ylabel="x₂",
                     title=titles[idx],
                     aspect=CM.DataAspect())

        # True positions
        CM.scatter!(ax, X_true_series[t][community_1, 1], X_true_series[t][community_1, 2],
                   color=:blue, marker=:circle, markersize=12, label="True C1")
        CM.scatter!(ax, X_true_series[t][community_2, 1], X_true_series[t][community_2, 2],
                   color=:red, marker=:circle, markersize=12, label="True C2")

        # Estimated positions
        CM.scatter!(ax, X_est_series[t][community_1, 1], X_est_series[t][community_1, 2],
                   color=:lightblue, marker=:diamond, markersize=10, label="Est C1")
        CM.scatter!(ax, X_est_series[t][community_2, 1], X_est_series[t][community_2, 2],
                   color=:lightsalmon, marker=:diamond, markersize=10, label="Est C2")

        # Recovered positions
        CM.scatter!(ax, X_rec_series[t][community_1, 1], X_rec_series[t][community_1, 2],
                   color=:darkblue, marker=:star5, markersize=14, label="Rec C1")
        CM.scatter!(ax, X_rec_series[t][community_2, 1], X_rec_series[t][community_2, 2],
                   color=:darkred, marker=:star5, markersize=14, label="Rec C2")

        CM.xlims!(ax, x_min - x_pad, x_max + x_pad)
        CM.ylims!(ax, y_min - y_pad, y_max + y_pad)

        if idx == 3
            CM.Legend(fig[1, 4], ax, framevisible=false)
        end
    end

    CM.Label(fig[0, :], title, fontsize=16)
    return fig
end

function plot_trajectory_evolution(X_series, community_1, community_2;
                                   title="Trajectory Evolution",
                                   xlims_override=nothing, ylims_override=nothing)
    fig = CM.Figure(size=(600, 600))
    ax = CM.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title=title, aspect=CM.DataAspect())

    n_times = length(X_series)

    # Compute data bounds
    all_x1 = Float64[]
    all_x2 = Float64[]

    for i in community_1
        xs = [X_series[t][i, 1] for t in 1:n_times]
        ys = [X_series[t][i, 2] for t in 1:n_times]
        append!(all_x1, xs)
        append!(all_x2, ys)
        CM.lines!(ax, xs, ys, color=(:blue, 0.5), linewidth=1)
        CM.scatter!(ax, [xs[1]], [ys[1]], color=:blue, markersize=8)
        CM.scatter!(ax, [xs[end]], [ys[end]], color=:blue, marker=:star5, markersize=10)
    end

    for i in community_2
        xs = [X_series[t][i, 1] for t in 1:n_times]
        ys = [X_series[t][i, 2] for t in 1:n_times]
        append!(all_x1, xs)
        append!(all_x2, ys)
        CM.lines!(ax, xs, ys, color=(:red, 0.5), linewidth=1)
        CM.scatter!(ax, [xs[1]], [ys[1]], color=:red, markersize=8)
        CM.scatter!(ax, [xs[end]], [ys[end]], color=:red, marker=:star5, markersize=10)
    end

    # Auto-scale with padding, or use override
    if isnothing(xlims_override)
        x_min, x_max = extrema(all_x1)
        x_pad = 0.1 * (x_max - x_min)
        CM.xlims!(ax, x_min - x_pad, x_max + x_pad)
    else
        CM.xlims!(ax, xlims_override...)
    end

    if isnothing(ylims_override)
        y_min, y_max = extrema(all_x2)
        y_pad = 0.1 * (y_max - y_min)
        CM.ylims!(ax, y_min - y_pad, y_max + y_pad)
    else
        CM.ylims!(ax, ylims_override...)
    end

    return fig
end

function plot_metrics_over_time(metrics_dict::Dict, T_train::Int; title="Metrics Over Time")
    fig = CM.Figure(size=(1200, 400))
    colors = [:blue, :red, :green, :orange, :purple]

    ax1 = CM.Axis(fig[1, 1], xlabel="Time", ylabel="D Correlation", title="Distance Correlation")
    for (i, (name, metrics)) in enumerate(metrics_dict)
        n_t = length(metrics.D_corr)
        CM.lines!(ax1, 1:n_t, metrics.D_corr, color=colors[mod1(i, length(colors))],
                 linewidth=2, label=name)
    end
    CM.vlines!(ax1, [T_train], color=:gray, linestyle=:dash, label="Train/Val split")
    CM.Legend(fig[1, 2], ax1, framevisible=false)

    ax2 = CM.Axis(fig[1, 3], xlabel="Time", ylabel="P Error", title="P Reconstruction Error")
    for (i, (name, metrics)) in enumerate(metrics_dict)
        n_t = length(metrics.P_err)
        CM.lines!(ax2, 1:n_t, metrics.P_err, color=colors[mod1(i, length(colors))],
                 linewidth=2, label=name)
    end
    CM.vlines!(ax2, [T_train], color=:gray, linestyle=:dash)

    CM.Label(fig[0, :], title, fontsize=16)
    return fig
end

function plot_learning_curves(losses_dict::Dict; title="Learning Curves")
    fig = CM.Figure(size=(800, 400))
    ax = CM.Axis(fig[1, 1], xlabel="Iteration", ylabel="Loss (log scale)",
                 title=title, yscale=log10)

    colors = [:blue, :red, :green, :orange]

    for (i, (name, losses)) in enumerate(losses_dict)
        valid_losses = [l for l in losses if l < 1e6]
        CM.lines!(ax, 1:length(valid_losses), valid_losses,
                 color=colors[mod1(i, length(colors))], linewidth=2, label=name)
    end

    CM.Legend(fig[1, 2], ax, framevisible=false)
    return fig
end

# =============================================================================
# Main Pipeline
# =============================================================================

function run_full_pipeline(config::PipelineConfig)
    println("=" ^ 70)
    println("FULL PIPELINE COMPARISON (Float32 Optimized)")
    println("  n=" * string(config.n) * ", d=" * string(config.d) *
            ", T=" * string(config.T_end) * ", K=" * string(config.K_samples))
    println("=" ^ 70)

    Random.seed!(config.seed)

    n, d = config.n, config.d
    community_1 = 1:(n ÷ 2)
    community_2 = (n ÷ 2 + 1):n

    # Centers as Float64 for true dynamics generation
    center_1_f64 = [0.35, 0.65]
    center_2_f64 = [0.65, 0.35]

    # Centers as Float32 for UDE
    center_1_f32 = F[0.35, 0.65]
    center_2_f32 = F[0.65, 0.35]

    # 1. Generate true data (Float64 for accuracy)
    println("\n1. Generating true dynamics...")
    true_dynamics! = create_true_dynamics(n, d, community_1, community_2, center_1_f64, center_2_f64)

    rng = Random.MersenneTwister(config.seed)
    X0 = zeros(n, d)
    for i in community_1
        X0[i, :] = center_1_f64 .+ 0.12 .* randn(rng, d)
    end
    for i in community_2
        X0[i, :] = center_2_f64 .+ 0.12 .* randn(rng, d)
    end
    X0 = clamp.(X0, 0.15, 0.85)

    T_steps = Int(config.T_end / config.dt) + 1
    tspan_f64 = (0.0, Float64(config.T_end))
    tsteps_f64 = range(0.0, Float64(config.T_end), length=T_steps)
    prob = ODEProblem(true_dynamics!, vec(X0), tspan_f64)
    sol = solve(prob, Tsit5(), saveat=tsteps_f64, abstol=1e-7, reltol=1e-7)

    # Store true series as Float64 for evaluation
    X_true_series = [reshape(sol.u[t], n, d) for t in 1:T_steps]
    println("   Generated " * string(T_steps) * " timesteps")

    # 2. RDPG estimation (output Float32)
    println("\n2. RDPG estimation (K=" * string(config.K_samples) * ")...")
    X_est_series = embed_series(X_true_series, d, config.K_samples, F)
    println("   Embedded " * string(T_steps) * " timesteps (Float32)")

    # 3. Train/val split
    T_train = Int(floor(config.train_fraction * T_steps))
    X_train = X_est_series[1:T_train]
    println("\n3. Train/val split: " * string(T_train) * " train, " *
            string(T_steps - T_train) * " val")

    # Store results
    results = Dict{String, Any}()
    losses_dict = Dict{String, Vector{F}}()

    # =========================================================================
    # Model A: Pure Neural ODE
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("MODEL A: Pure Neural ODE")
    println("=" ^ 70)

    rng_nn = Random.Xoshiro(config.seed)
    nn_pure = build_pure_nn(n * d, config.hidden_sizes, n * d; rng=rng_nn)
    ps_pure, st_pure = Lux.setup(rng_nn, nn_pure)
    ps_pure_f32 = ComponentArray{F}(ps_pure)
    println("   Parameters: " * string(length(ps_pure_f32)))

    dynamics_pure = make_pure_nn_dynamics(nn_pure, st_pure, n, d)
    ps_pure_trained, losses_pure = train_dynamics(config, X_train, dynamics_pure, ps_pure_f32;
                                                   name="Pure NN")

    X_rec_pure = recover_trajectory(config, X_est_series, dynamics_pure, ps_pure_trained)
    results["Pure NN"] = (X_rec=X_rec_pure, ps=ps_pure_trained, dynamics=dynamics_pure)
    losses_dict["Pure NN"] = losses_pure

    # =========================================================================
    # Model B: UDE (known oscillation + NN correction)
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("MODEL B: UDE (Known Oscillation + NN Correction)")
    println("=" ^ 70)

    rng_ude = Random.Xoshiro(config.seed + 1)
    nn_correction = build_correction_nn(n * d, n * d; rng=rng_ude)
    ps_corr, st_corr = Lux.setup(rng_ude, nn_correction)
    ps_corr_f32 = ComponentArray{F}(ps_corr)
    println("   Correction NN parameters: " * string(length(ps_corr_f32)))

    dynamics_ude = make_ude_dynamics(nn_correction, st_corr, n, d,
                                     community_1, community_2, center_1_f32, center_2_f32,
                                     config.omega_known, config.k_attract_known)
    ps_ude_trained, losses_ude = train_dynamics(config, X_train, dynamics_ude, ps_corr_f32;
                                                 name="UDE")

    X_rec_ude = recover_trajectory(config, X_est_series, dynamics_ude, ps_ude_trained)
    results["UDE"] = (X_rec=X_rec_ude, ps=ps_ude_trained, dynamics=dynamics_ude)
    losses_dict["UDE"] = losses_ude

    # =========================================================================
    # Evaluation
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("EVALUATION")
    println("=" ^ 70)

    # Baseline: True vs Estimated
    metrics_baseline = evaluate_series(X_true_series, X_est_series)

    println("\n--- Baseline (True ↔ Estimated) ---")
    println("  Avg D corr: " * string(round(mean(metrics_baseline.D_corr), digits=4)))
    println("  Avg P err:  " * string(round(mean(metrics_baseline.P_err), digits=4)))

    metrics_dict = Dict{String, Any}()
    metrics_dict["Baseline"] = metrics_baseline

    for (name, result) in results
        metrics = evaluate_series(X_true_series, result.X_rec)
        metrics_dict[name] = metrics

        println("\n--- " * name * " (True ↔ Recovered) ---")
        println("  Avg D corr: " * string(round(mean(metrics.D_corr), digits=4)) *
                " (ratio: " * string(round(mean(metrics.D_corr) / mean(metrics_baseline.D_corr), digits=2)) * ")")
        println("  Avg P err:  " * string(round(mean(metrics.P_err), digits=4)) *
                " (ratio: " * string(round(mean(metrics.P_err) / mean(metrics_baseline.P_err), digits=2)) * ")")

        # Train vs Val performance
        D_corr_train = mean(metrics.D_corr[1:T_train])
        D_corr_val = mean(metrics.D_corr[T_train+1:end])
        P_err_train = mean(metrics.P_err[1:T_train])
        P_err_val = mean(metrics.P_err[T_train+1:end])

        println("  Train D corr: " * string(round(D_corr_train, digits=4)) *
                ", Val D corr: " * string(round(D_corr_val, digits=4)))
        println("  Train P err: " * string(round(P_err_train, digits=4)) *
                ", Val P err: " * string(round(P_err_val, digits=4)))
    end

    # =========================================================================
    # Visualization
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("GENERATING VISUALIZATIONS")
    println("=" ^ 70)

    mkpath("results")

    # Trajectory comparisons
    for (name, result) in results
        fig = plot_trajectories(X_true_series, X_est_series, result.X_rec,
                               community_1, community_2, T_train;
                               title=name * ": Trajectory Comparison")
        filename = "results/trajectory_" * replace(lowercase(name), " " => "_") * ".pdf"
        CM.save(filename, fig)
        println("  Saved: " * filename)
    end

    # Trajectory evolution
    fig_true = plot_trajectory_evolution(X_true_series, community_1, community_2;
                                         title="True Trajectory Evolution")
    CM.save("results/evolution_true.pdf", fig_true)
    println("  Saved: results/evolution_true.pdf")

    for (name, result) in results
        fig = plot_trajectory_evolution(result.X_rec, community_1, community_2;
                                       title=name * ": Recovered Trajectory Evolution")
        filename = "results/evolution_" * replace(lowercase(name), " " => "_") * ".pdf"
        CM.save(filename, fig)
    end

    # Metrics over time
    fig_metrics = plot_metrics_over_time(metrics_dict, T_train;
                                         title="Metrics Over Time (All Models)")
    CM.save("results/metrics_comparison.pdf", fig_metrics)
    println("  Saved: results/metrics_comparison.pdf")

    # Learning curves
    fig_learning = plot_learning_curves(losses_dict; title="Learning Curves")
    CM.save("results/learning_curves.pdf", fig_learning)
    println("  Saved: results/learning_curves.pdf")

    # =========================================================================
    # Summary Table
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("SUMMARY TABLE")
    println("=" ^ 70)
    println()
    println("Model          | Avg D corr | Avg P err | Train D | Val D  | Train P | Val P")
    println("-" ^ 85)

    for name in ["Baseline", "Pure NN", "UDE"]
        if haskey(metrics_dict, name)
            m = metrics_dict[name]
            if name == "Baseline"
                println(rpad(name, 14) * " | " *
                        rpad(string(round(mean(m.D_corr), digits=3)), 10) * " | " *
                        rpad(string(round(mean(m.P_err), digits=3)), 9) * " | " *
                        "  -     |   -    |   -     |   -")
            else
                D_train = round(mean(m.D_corr[1:T_train]), digits=3)
                D_val = round(mean(m.D_corr[T_train+1:end]), digits=3)
                P_train = round(mean(m.P_err[1:T_train]), digits=3)
                P_val = round(mean(m.P_err[T_train+1:end]), digits=3)

                println(rpad(name, 14) * " | " *
                        rpad(string(round(mean(m.D_corr), digits=3)), 10) * " | " *
                        rpad(string(round(mean(m.P_err), digits=3)), 9) * " | " *
                        rpad(string(D_train), 7) * " | " *
                        rpad(string(D_val), 6) * " | " *
                        rpad(string(P_train), 7) * " | " *
                        string(P_val))
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("Pipeline complete!")
    println("=" ^ 70)

    return results, metrics_dict, losses_dict, X_true_series, X_est_series
end

# =============================================================================
# Run
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    config = PipelineConfig()
    results, metrics, losses, X_true, X_est = run_full_pipeline(config)
end
