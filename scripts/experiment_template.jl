#!/usr/bin/env julia
"""
Neural ODE Experiment Template for RDPG Dynamics

This template implements the full evaluation framework:
- Proper Neural ODE training with SciML autodiff
- Both position-based and distance-based loss options
- Three evaluation comparisons: True↔Est, True↔Rec, Est↔Rec
- Rotation-invariant metrics (D correlation, P error)

Usage:
  julia --project scripts/experiment_template.jl
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

# Plotting (import qualified to avoid Axis conflict with ComponentArrays)
import CairoMakie
const CM = CairoMakie

# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct ExperimentConfig
    # Problem setup
    n::Int = 16                          # Number of nodes
    d::Int = 2                           # Embedding dimension
    T_end::Float64 = 30.0                # End time
    dt::Float64 = 1.0                    # Time step

    # RDPG sampling
    K_samples::Int = 50                  # Samples per timestep

    # Training
    loss_type::Symbol = :position        # :position or :distance
    hidden_sizes::Vector{Int} = [64, 64] # NN architecture
    epochs_adam::Int = 300               # ADAM iterations
    epochs_bfgs::Int = 100               # BFGS iterations
    lr::Float64 = 0.01                   # Learning rate

    # Train/val split
    train_fraction::Float64 = 0.7        # Fraction of timesteps for training

    # Random seed
    seed::Int = 42
end

# =============================================================================
# Evaluation Metrics
# =============================================================================

"""Compute pairwise distance matrix (non-mutating for Zygote compatibility)."""
function pairwise_distances(X::AbstractMatrix)
    n = size(X, 1)
    # Compute squared norms
    sq_norms = sum(X .^ 2, dims=2)
    # D_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    # Clamp to avoid numerical issues with sqrt of small negatives
    D_sq = max.(D_sq, 0.0)
    return sqrt.(D_sq .+ 1e-10)  # Small epsilon for numerical stability
end

"""Compute squared pairwise distances (more stable for training)."""
function pairwise_distances_squared(X::AbstractMatrix)
    sq_norms = sum(X .^ 2, dims=2)
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    return max.(D_sq, 0.0)
end

"""Extract upper triangular elements."""
function upper_tri(M::AbstractMatrix)
    n = size(M, 1)
    return [M[i, j] for i in 1:n for j in i+1:n]
end

"""
Compute all evaluation metrics between two position series.

Returns NamedTuple with:
- D_corr: distance correlation (per timestep)
- D_rmse: distance RMSE (per timestep)
- P_err: P reconstruction error (per timestep)
- pos_rmse: position RMSE (per timestep, only meaningful for same-rotation data)
"""
function evaluate_series(X_series_1::Vector, X_series_2::Vector)
    T = length(X_series_1)
    @assert length(X_series_2) == T

    D_corr = zeros(T)
    D_rmse = zeros(T)
    P_err = zeros(T)
    pos_rmse = zeros(T)

    for t in 1:T
        X1, X2 = X_series_1[t], X_series_2[t]

        # Distance metrics
        D1 = pairwise_distances(X1)
        D2 = pairwise_distances(X2)
        d1_vec, d2_vec = upper_tri(D1), upper_tri(D2)

        D_corr[t] = cor(d1_vec, d2_vec)
        D_rmse[t] = sqrt(mean((d1_vec .- d2_vec).^2))

        # P reconstruction
        P1 = X1 * X1'
        P2 = X2 * X2'
        P_err[t] = norm(P1 - P2) / max(norm(P2), 1e-10)

        # Position RMSE
        pos_rmse[t] = sqrt(mean((X1 .- X2).^2))
    end

    return (D_corr=D_corr, D_rmse=D_rmse, P_err=P_err, pos_rmse=pos_rmse)
end

"""Print summary of evaluation metrics."""
function print_evaluation(name::String, metrics; timesteps=[1, 10, 20, 30])
    println("\n--- " * name * " ---")
    println("  t  | D corr | D RMSE | P err  | Pos RMSE")
    println("  " * "-" ^ 45)

    for t in timesteps
        if t <= length(metrics.D_corr)
            println("  " * lpad(string(t), 2) * " | " *
                    rpad(string(round(metrics.D_corr[t], digits=3)), 6) * " | " *
                    rpad(string(round(metrics.D_rmse[t], digits=4)), 6) * " | " *
                    rpad(string(round(metrics.P_err[t], digits=4)), 6) * " | " *
                    string(round(metrics.pos_rmse[t], digits=4)))
        end
    end

    println("\n  Averages:")
    println("    D corr:   " * string(round(mean(metrics.D_corr), digits=4)))
    println("    D RMSE:   " * string(round(mean(metrics.D_rmse), digits=4)))
    println("    P err:    " * string(round(mean(metrics.P_err), digits=4)))
    println("    Pos RMSE: " * string(round(mean(metrics.pos_rmse), digits=4)))
end

# =============================================================================
# RDPG Sampling and Embedding
# =============================================================================

"""Sample adjacency matrix from RDPG."""
function sample_adjacency(X::AbstractMatrix{T}) where T
    n = size(X, 1)
    P = X * X'
    P = clamp.(P, zero(T), one(T))
    A = T.(rand(n, n) .< P)
    # Make symmetric
    for i in 1:n, j in 1:i-1
        A[i, j] = A[j, i]
    end
    return A
end

"""SVD embedding of adjacency matrix."""
function svd_embed(A::AbstractMatrix, d::Int)
    F = svd(A)
    # Handle case where we have fewer than d significant singular values
    d_actual = min(d, length(F.S))
    L = F.U[:, 1:d_actual] .* sqrt.(F.S[1:d_actual]')
    # Pad with zeros if needed
    if d_actual < d
        L = hcat(L, zeros(size(L, 1), d - d_actual))
    end
    return L
end

"""Embed a series of true positions via RDPG sampling."""
function embed_series(X_true_series::Vector, d::Int, K::Int)
    T = length(X_true_series)
    X_est_series = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        A_avg = zeros(size(X_true_series[t], 1), size(X_true_series[t], 1))
        for k in 1:K
            A_avg .+= sample_adjacency(X_true_series[t])
        end
        A_avg ./= K
        X_est_series[t] = svd_embed(A_avg, d)
    end

    return X_est_series
end

# =============================================================================
# Neural ODE Setup
# =============================================================================

"""Build neural network for dynamics."""
function build_nn(input_dim::Int, hidden_sizes::Vector{Int}, output_dim::Int; rng=Random.default_rng())
    layers = []
    in_dim = input_dim

    for h in hidden_sizes
        push!(layers, Lux.Dense(in_dim, h, tanh))
        in_dim = h
    end
    push!(layers, Lux.Dense(in_dim, output_dim))

    return Lux.Chain(layers...)
end

"""
Create dynamics function for Neural ODE.

The NN takes flattened positions and outputs flattened velocities.
"""
function make_dynamics(nn, ps, st, n, d)
    function dynamics!(du, u, p, t)
        # u is flattened (n*d,)
        u_reshape = reshape(u, 1, n * d)  # (1, n*d) for Lux
        out, _ = nn(u_reshape, p, st)
        du .= vec(out)
        return nothing
    end
    return dynamics!
end

"""
Create dynamics function that outputs for ODE solve (non-mutating version).
"""
function make_dynamics_oop(nn, st, n, d)
    function dynamics(u, p, t)
        # Lux expects (features, batch) = (n*d, 1)
        u_reshape = reshape(u, n * d, 1)
        out, _ = nn(u_reshape, p, st)
        return vec(out)
    end
    return dynamics
end

# =============================================================================
# Loss Functions
# =============================================================================

"""
Position-based loss: ||X_pred - X_target||²
"""
function position_loss(pred_series::Vector, target_series::Vector)
    loss = 0.0
    for (pred, target) in zip(pred_series, target_series)
        loss += sum(abs2, pred .- target)
    end
    return loss / length(pred_series)
end

"""
Distance-based loss: ||D(X_pred) - D(X_target)||²

This is rotation-invariant. Uses Frobenius norm normalized by number of pairs.
"""
function distance_loss(pred_series::Vector, target_series::Vector)
    loss = 0.0
    n = size(pred_series[1], 1)
    n_pairs = n * (n - 1) / 2  # Number of unique pairs

    for (pred, target) in zip(pred_series, target_series)
        # Use squared distances for numerical stability (avoids sqrt gradient issues)
        D_sq_pred = pairwise_distances_squared(pred)
        D_sq_target = pairwise_distances_squared(target)

        # Compare sqrt of squared distances with epsilon for stability
        # This is equivalent to comparing distances but with stable gradients
        D_pred = sqrt.(D_sq_pred .+ 1e-6)
        D_target = sqrt.(D_sq_target .+ 1e-6)

        # Normalize by number of pairs for comparable scale to position loss
        loss += sum(abs2, D_pred .- D_target) / (2 * n_pairs)
    end
    return loss / length(pred_series)
end

# =============================================================================
# Training
# =============================================================================

"""
Simulate Neural ODE from initial condition.
"""
function simulate_node(u0::Vector, tspan, tsteps, p, dynamics_fn; solver=Tsit5())
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, solver, saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1e-5, reltol=1e-5)
    return sol
end

"""
Main training function.

Returns trained parameters and training history.
"""
function train_node(config::ExperimentConfig,
                    X_train_series::Vector{Matrix{Float64}},
                    nn, ps_init, st)

    n, d = config.n, config.d
    T_train = length(X_train_series)
    tsteps = range(0.0, step=config.dt, length=T_train)
    tspan = (0.0, tsteps[end])

    # Initial condition
    u0 = vec(X_train_series[1])

    # Dynamics function
    dynamics_fn = make_dynamics_oop(nn, st, n, d)

    # Choose loss function
    loss_fn = config.loss_type == :distance ? distance_loss : position_loss

    # Training loss
    function total_loss(p)
        sol = simulate_node(u0, tspan, tsteps, p, dynamics_fn)

        # Check if solve succeeded - return large finite loss instead of Inf
        # (Inf causes NaN gradients)
        if sol.retcode != :Success
            return 1e8
        end

        # Extract predictions
        pred_series = [reshape(sol.u[t], n, d) for t in 1:T_train]

        # Check for NaN/Inf in predictions
        for pred in pred_series
            if any(isnan, pred) || any(isinf, pred)
                return 1e8
            end
        end

        return loss_fn(pred_series, X_train_series)
    end

    # Training callback
    iter = Ref(0)
    losses = Float64[]

    function callback(state, l)
        iter[] += 1
        push!(losses, l)
        if iter[] % 50 == 0
            println("    Iter " * string(iter[]) * ": loss = " * string(round(l, digits=6)))
        end
        return false
    end

    println("  Training with " * string(config.loss_type) * " loss...")
    println("  Initial loss: " * string(round(total_loss(ps_init), digits=4)))

    # Optimization
    optf = Optimization.OptimizationFunction((p, _) -> total_loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    # ADAM phase
    println("\n  ADAM phase (" * string(config.epochs_adam) * " iterations)...")
    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(config.lr),
        maxiters=config.epochs_adam,
        callback=callback
    )

    # BFGS phase
    println("\n  BFGS phase (" * string(config.epochs_bfgs) * " iterations)...")
    iter[] = config.epochs_adam
    optprob2 = Optimization.OptimizationProblem(optf, result_adam.u)

    result_bfgs = try
        Optimization.solve(
            optprob2,
            OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
            maxiters=config.epochs_bfgs,
            callback=callback,
            allow_f_increases=false
        )
    catch e
        println("    BFGS failed: " * string(e)[1:min(50, end)])
        result_adam
    end

    final_loss = total_loss(result_bfgs.u)
    println("\n  Final loss: " * string(round(final_loss, digits=6)))

    return result_bfgs.u, losses
end

"""
Recover full trajectory using trained model.
"""
function recover_trajectory(config::ExperimentConfig,
                           X0::Matrix{Float64},
                           T_total::Int,
                           ps_trained, nn, st)
    n, d = config.n, config.d
    tsteps = range(0.0, step=config.dt, length=T_total)
    tspan = (0.0, tsteps[end])
    u0 = vec(X0)

    dynamics_fn = make_dynamics_oop(nn, st, n, d)
    sol = simulate_node(u0, tspan, tsteps, ps_trained, dynamics_fn)

    return [reshape(sol.u[t], n, d) for t in 1:T_total]
end

# =============================================================================
# Example: Two Communities Joining
# =============================================================================

"""
Generate true dynamics: two communities oscillating around fixed points.
This maintains community structure over time (unlike joining dynamics that collapse).
"""
function generate_communities_data(config::ExperimentConfig)
    n, d = config.n, config.d
    T_steps = Int(config.T_end / config.dt) + 1

    community_1 = 1:(n ÷ 2)
    community_2 = (n ÷ 2 + 1):n

    # Fixed points for each community (in positive orthant for valid probabilities)
    center_1 = [0.35, 0.65]
    center_2 = [0.65, 0.35]

    # Parameters for oscillation
    omega = 0.3     # Angular frequency
    k_attract = 0.1  # Attraction to center
    k_cohesion = 0.05  # Cohesion within community

    # True dynamics: oscillation around community centers
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

            # Oscillation: rotate around community center
            delta = X[i, :] .- center
            dX[i, 1] = -omega * delta[2] - k_attract * delta[1]
            dX[i, 2] = omega * delta[1] - k_attract * delta[2]

            # Cohesion within community
            dX[i, :] .+= k_cohesion .* (X_bar .- X[i, :])

            # Soft boundary to keep in valid region [0.1, 0.9]
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

    # Initial positions: scattered around community centers
    rng = MersenneTwister(config.seed)
    X0 = zeros(n, d)
    for i in community_1
        X0[i, :] = center_1 .+ 0.12 .* randn(rng, d)
    end
    for i in community_2
        X0[i, :] = center_2 .+ 0.12 .* randn(rng, d)
    end

    # Clamp to valid region
    X0 = clamp.(X0, 0.15, 0.85)

    # Simulate
    tspan = (0.0, config.T_end)
    tsteps = range(0.0, config.T_end, length=T_steps)
    prob = ODEProblem(true_dynamics!, vec(X0), tspan)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-7, reltol=1e-7)

    X_true_series = [reshape(sol.u[t], n, d) for t in 1:T_steps]

    return X_true_series, community_1, community_2
end

# =============================================================================
# Main Experiment
# =============================================================================

function run_experiment(config::ExperimentConfig)
    println("=" ^ 70)
    println("RDPG Dynamics Experiment")
    println("  Loss type: " * string(config.loss_type))
    println("  n=" * string(config.n) * ", d=" * string(config.d) *
            ", T=" * string(config.T_end) * ", K=" * string(config.K_samples))
    println("=" ^ 70)

    # 1. Generate true data
    println("\n1. Generating true dynamics...")
    X_true_series, c1, c2 = generate_communities_data(config)
    T_total = length(X_true_series)
    println("   Generated " * string(T_total) * " timesteps")

    # 2. RDPG estimation
    println("\n2. RDPG estimation (K=" * string(config.K_samples) * ")...")
    X_est_series = embed_series(X_true_series, config.d, config.K_samples)
    println("   Embedded " * string(T_total) * " timesteps")

    # 3. Train/val split
    T_train = Int(floor(config.train_fraction * T_total))
    X_train = X_est_series[1:T_train]
    println("\n3. Train/val split: " * string(T_train) * " train, " *
            string(T_total - T_train) * " val")

    # 4. Setup Neural ODE
    println("\n4. Setting up Neural ODE...")
    rng = Random.Xoshiro(config.seed)
    nn = build_nn(config.n * config.d, config.hidden_sizes, config.n * config.d; rng=rng)
    ps, st = Lux.setup(rng, nn)
    # Convert to Float64 to match ODE state type
    ps_ca = ComponentArray{Float64}(ps)
    println("   Parameters: " * string(length(ps_ca)))

    # 5. Train
    println("\n5. Training...")
    ps_trained, losses = train_node(config, X_train, nn, ps_ca, st)

    # 6. Recover full trajectory
    println("\n6. Recovering trajectory...")
    X_rec_series = recover_trajectory(config, X_est_series[1], T_total, ps_trained, nn, st)
    println("   Recovered " * string(length(X_rec_series)) * " timesteps")

    # 7. Evaluation
    println("\n" * "=" ^ 70)
    println("EVALUATION")
    println("=" ^ 70)

    eval_timesteps = [1, T_train ÷ 2, T_train, T_total]

    metrics_true_est = evaluate_series(X_true_series, X_est_series)
    print_evaluation("True ↔ Estimated (RDPG baseline)", metrics_true_est;
                     timesteps=eval_timesteps)

    metrics_true_rec = evaluate_series(X_true_series, X_rec_series)
    print_evaluation("True ↔ Recovered (dynamics quality)", metrics_true_rec;
                     timesteps=eval_timesteps)

    metrics_est_rec = evaluate_series(X_est_series, X_rec_series)
    print_evaluation("Estimated ↔ Recovered (training fit)", metrics_est_rec;
                     timesteps=eval_timesteps)

    # 8. Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    avg_d_true_est = mean(metrics_true_est.D_corr)
    avg_d_true_rec = mean(metrics_true_rec.D_corr)
    avg_p_true_est = mean(metrics_true_est.P_err)
    avg_p_true_rec = mean(metrics_true_rec.P_err)

    println("\nDistance Correlation (avg):")
    println("  True ↔ Est: " * string(round(avg_d_true_est, digits=4)))
    println("  True ↔ Rec: " * string(round(avg_d_true_rec, digits=4)))
    println("  Ratio:      " * string(round(avg_d_true_rec / max(avg_d_true_est, 0.01), digits=2)))

    println("\nP Error (avg):")
    println("  True ↔ Est: " * string(round(avg_p_true_est, digits=4)))
    println("  True ↔ Rec: " * string(round(avg_p_true_rec, digits=4)))
    println("  Ratio:      " * string(round(avg_p_true_rec / max(avg_p_true_est, 0.01), digits=2)))

    # Success criteria
    d_ratio = avg_d_true_rec / max(avg_d_true_est, 0.01)
    p_ratio = avg_p_true_rec / max(avg_p_true_est, 0.01)

    println("\n" * "-" ^ 50)
    if d_ratio > 0.9 && p_ratio < 1.5
        println("✓ SUCCESS: Dynamics learning is nearly as good as RDPG estimation")
    elseif d_ratio > 0.7 && p_ratio < 3.0
        println("~ PARTIAL: Dynamics learning is reasonable but degrades")
    else
        println("✗ FAIL: Dynamics learning significantly degrades from estimation")
    end

    # 9. Visualization
    println("\n7. Creating visualization...")

    fig = CM.Figure(size=(1400, 800))

    ts = 1:T_total

    # Row 1: Metrics over time
    ax1 = CM.Axis(fig[1, 1], xlabel="Time", ylabel="D Correlation",
                  title="Distance Correlation with True")
    CM.lines!(ax1, ts, metrics_true_est.D_corr, linewidth=2, label="Estimated", color=:blue)
    CM.lines!(ax1, ts, metrics_true_rec.D_corr, linewidth=2, label="Recovered", color=:red)
    CM.vlines!(ax1, [T_train], color=:gray, linestyle=:dash, label="Train/Val split")
    CM.axislegend(ax1, position=:lb)

    ax2 = CM.Axis(fig[1, 2], xlabel="Time", ylabel="P Error",
                  title="P Reconstruction Error")
    CM.lines!(ax2, ts, metrics_true_est.P_err, linewidth=2, label="Estimated", color=:blue)
    CM.lines!(ax2, ts, metrics_true_rec.P_err, linewidth=2, label="Recovered", color=:red)
    CM.vlines!(ax2, [T_train], color=:gray, linestyle=:dash)
    CM.axislegend(ax2, position=:lt)

    ax3 = CM.Axis(fig[1, 3], xlabel="Iteration", ylabel="Loss",
                  title="Training Loss", yscale=log10)
    CM.lines!(ax3, 1:length(losses), losses .+ 1e-10, linewidth=2, color=:black)

    # Row 2: Inter-community distance
    D_true = [pairwise_distances(X) for X in X_true_series]
    D_est = [pairwise_distances(X) for X in X_est_series]
    D_rec = [pairwise_distances(X) for X in X_rec_series]

    inter_true = [mean([D_true[t][i,j] for i in c1 for j in c2]) for t in ts]
    inter_est = [mean([D_est[t][i,j] for i in c1 for j in c2]) for t in ts]
    inter_rec = [mean([D_rec[t][i,j] for i in c1 for j in c2]) for t in ts]

    ax4 = CM.Axis(fig[2, 1], xlabel="Time", ylabel="Inter-community Distance",
                  title="Community Dynamics")
    CM.lines!(ax4, ts, inter_true, linewidth=3, label="True", color=:black)
    CM.lines!(ax4, ts, inter_est, linewidth=2, label="Estimated", color=:blue, linestyle=:dash)
    CM.lines!(ax4, ts, inter_rec, linewidth=2, label="Recovered", color=:red, linestyle=:dot)
    CM.vlines!(ax4, [T_train], color=:gray, linestyle=:dash)
    CM.axislegend(ax4, position=:rt)

    # Scatter: True vs Recovered distances at mid and end
    t_mid = T_train
    d_true_mid = upper_tri(D_true[t_mid])
    d_rec_mid = upper_tri(D_rec[t_mid])

    ax5 = CM.Axis(fig[2, 2], xlabel="True Distance", ylabel="Recovered Distance",
                  title="Distance Recovery (t=" * string(t_mid) * ", train end)")
    CM.scatter!(ax5, d_true_mid, d_rec_mid, alpha=0.5, markersize=6)
    lims = (0, max(maximum(d_true_mid), maximum(d_rec_mid)) * 1.1)
    CM.lines!(ax5, [lims[1], lims[2]], [lims[1], lims[2]], color=:red, linestyle=:dash)
    CM.text!(ax5, lims[1] + 0.05*(lims[2]-lims[1]), lims[2] - 0.1*(lims[2]-lims[1]),
             text="r=" * string(round(cor(d_true_mid, d_rec_mid), digits=3)))

    t_end = T_total
    d_true_end = upper_tri(D_true[t_end])
    d_rec_end = upper_tri(D_rec[t_end])

    ax6 = CM.Axis(fig[2, 3], xlabel="True Distance", ylabel="Recovered Distance",
                  title="Distance Recovery (t=" * string(t_end) * ", validation)")
    CM.scatter!(ax6, d_true_end, d_rec_end, alpha=0.5, markersize=6)
    lims = (0, max(maximum(d_true_end), maximum(d_rec_end)) * 1.1)
    CM.lines!(ax6, [lims[1], lims[2]], [lims[1], lims[2]], color=:red, linestyle=:dash)
    CM.text!(ax6, lims[1] + 0.05*(lims[2]-lims[1]), lims[2] - 0.1*(lims[2]-lims[1]),
             text="r=" * string(round(cor(d_true_end, d_rec_end), digits=3)))

    mkpath("results")
    loss_str = string(config.loss_type)
    CM.save("results/experiment_" * loss_str * "_loss.pdf", fig)
    println("Saved: results/experiment_" * loss_str * "_loss.pdf")

    return (
        config = config,
        X_true = X_true_series,
        X_est = X_est_series,
        X_rec = X_rec_series,
        metrics_true_est = metrics_true_est,
        metrics_true_rec = metrics_true_rec,
        metrics_est_rec = metrics_est_rec,
        ps_trained = ps_trained,
        losses = losses
    )
end

# =============================================================================
# Run Experiments
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "=" ^ 70)
    println("Running experiments with POSITION loss and DISTANCE loss")
    println("=" ^ 70)

    # Experiment A: Position loss
    config_pos = ExperimentConfig(
        loss_type = :position,
        epochs_adam = 300,
        epochs_bfgs = 100
    )
    results_pos = run_experiment(config_pos)

    println("\n\n")

    # Experiment B: Distance loss
    config_dist = ExperimentConfig(
        loss_type = :distance,
        epochs_adam = 500,  # More iterations since gradients are weaker
        epochs_bfgs = 100,
        lr = 0.001  # Much lower LR for distance loss stability
    )
    results_dist = run_experiment(config_dist)

    # Final comparison
    println("\n" * "=" ^ 70)
    println("FINAL COMPARISON: Position Loss vs Distance Loss")
    println("=" ^ 70)

    println("\nPosition Loss:")
    println("  D corr (True↔Rec): " * string(round(mean(results_pos.metrics_true_rec.D_corr), digits=4)))
    println("  P err (True↔Rec):  " * string(round(mean(results_pos.metrics_true_rec.P_err), digits=4)))

    println("\nDistance Loss:")
    println("  D corr (True↔Rec): " * string(round(mean(results_dist.metrics_true_rec.D_corr), digits=4)))
    println("  P err (True↔Rec):  " * string(round(mean(results_dist.metrics_true_rec.P_err), digits=4)))

    println("\nBaseline (RDPG estimation):")
    println("  D corr (True↔Est): " * string(round(mean(results_pos.metrics_true_est.D_corr), digits=4)))
    println("  P err (True↔Est):  " * string(round(mean(results_pos.metrics_true_est.P_err), digits=4)))
end
