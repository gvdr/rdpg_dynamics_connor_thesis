#!/usr/bin/env -S julia --project
"""
Smooth Embedding: Find X(t) that minimizes both P reconstruction and tangent smoothness.

Objective: min Σ_t ||X(t)X(t)' - P(t)||² + λ Σ_t ||X(t+1) - 2X(t) + X(t-1)||²

The second term penalizes acceleration (second derivative), encouraging smooth tangent evolution.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Random
using OrdinaryDiffEq
using Statistics
using Optim
using CairoMakie

# ============================================================================
# Setup (same as before)
# ============================================================================

const N_PRED, N_PREY, N_RES = 12, 15, 10
const N_TOTAL = N_PRED + N_PREY + N_RES
const D_EMBED = 2
const T_TOTAL = 25
const SEED = 42

const TYPE_P, TYPE_Y, TYPE_R = 1, 2, 3
const NODE_TYPES = vcat(fill(TYPE_P, N_PRED), fill(TYPE_Y, N_PREY), fill(TYPE_R, N_RES))
const KNOWN_SELF_RATES = Dict(TYPE_P => -0.002, TYPE_Y => -0.001, TYPE_R => 0.000)
const HOLLING_ALPHA, HOLLING_BETA = 0.025, 2.0
const TYPE_COLORS = Dict(TYPE_P => :red, TYPE_Y => :blue, TYPE_R => :green)

function κ_true(ti, tj, p)
    ti == TYPE_P && tj == TYPE_Y && return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)
    ti == TYPE_Y && tj == TYPE_P && return -0.02 * p
    ti == TYPE_Y && tj == TYPE_R && return 0.012 * p
    ti == TYPE_R && tj == TYPE_Y && return -0.006 * p
    ti == TYPE_P && tj == TYPE_P && return -0.004
    ti == TYPE_Y && tj == TYPE_Y && return 0.003
    ti == TYPE_R && tj == TYPE_R && return 0.005
    return 0.0
end

function compute_N_matrix(X)
    n = size(X, 1)
    P = clamp.(X * X', 0.0, 1.0)
    [i == j ? KNOWN_SELF_RATES[NODE_TYPES[i]] : κ_true(NODE_TYPES[i], NODE_TYPES[j], P[i,j]) for i in 1:n, j in 1:n]
end

true_dynamics!(dX, X, p, t) = (dX .= compute_N_matrix(X) * X)

function generate_initial_X(rng)
    X = zeros(N_TOTAL, D_EMBED)
    centers = [[0.35, 0.70], [0.70, 0.50], [0.55, 0.25]]
    spread = 0.08
    idx = 1
    for (type_idx, count) in [(TYPE_P, N_PRED), (TYPE_Y, N_PREY), (TYPE_R, N_RES)]
        for _ in 1:count
            X[idx, :] = centers[type_idx] .+ spread .* randn(rng, 2)
            idx += 1
        end
    end
    clamp.(X, 0.05, 0.95)
end

function compute_curvature(X_traj)
    T = length(X_traj)
    n = size(X_traj[1], 1)
    velocities = [X_traj[t+1] - X_traj[t] for t in 1:(T-1)]
    angles = Float64[]
    for t in 1:(T-2), i in 1:n
        v1, v2 = velocities[t][i, :], velocities[t+1][i, :]
        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            push!(angles, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
        end
    end
    mean(angles)
end

# ============================================================================
# Smooth Embedding Optimization
# ============================================================================

function pack_X(X_list)
    # Pack list of matrices into single vector for optimization
    vcat([vec(X) for X in X_list]...)
end

function unpack_X(θ, n, d, T)
    # Unpack vector into list of matrices
    X_list = Vector{Matrix{Float64}}(undef, T)
    chunk_size = n * d
    for t in 1:T
        start_idx = (t-1) * chunk_size + 1
        end_idx = t * chunk_size
        X_list[t] = reshape(θ[start_idx:end_idx], n, d)
    end
    X_list
end

function smooth_embedding_loss(θ, P_list, n, d, λ_smooth)
    T = length(P_list)
    X_list = unpack_X(θ, n, d, T)

    # P reconstruction loss
    loss_P = 0.0
    for t in 1:T
        P_hat = X_list[t] * X_list[t]'
        loss_P += sum((P_hat - P_list[t]).^2)
    end

    # Smoothness loss (second derivative / acceleration)
    loss_smooth = 0.0
    for t in 2:(T-1)
        accel = X_list[t+1] - 2*X_list[t] + X_list[t-1]
        loss_smooth += sum(accel.^2)
    end

    return loss_P + λ_smooth * loss_smooth
end

function smooth_embedding_gradient!(G, θ, P_list, n, d, λ_smooth)
    T = length(P_list)
    X_list = unpack_X(θ, n, d, T)
    chunk_size = n * d

    G .= 0.0

    for t in 1:T
        P_hat = X_list[t] * X_list[t]'
        dP = 2 * (P_hat - P_list[t])
        dX_P = 2 * dP * X_list[t]  # Gradient of ||XX' - P||² w.r.t. X

        start_idx = (t-1) * chunk_size + 1
        end_idx = t * chunk_size
        G[start_idx:end_idx] .= vec(dX_P)
    end

    # Smoothness gradient
    for t in 2:(T-1)
        accel = X_list[t+1] - 2*X_list[t] + X_list[t-1]

        # d/dX[t-1] of ||X[t+1] - 2X[t] + X[t-1]||² = 2*accel
        # d/dX[t] = -4*accel
        # d/dX[t+1] = 2*accel

        idx_prev = (t-2) * chunk_size + 1
        idx_curr = (t-1) * chunk_size + 1
        idx_next = t * chunk_size + 1

        G[idx_prev:idx_prev+chunk_size-1] .+= λ_smooth * 2 * vec(accel)
        G[idx_curr:idx_curr+chunk_size-1] .+= λ_smooth * (-4) * vec(accel)
        G[idx_next:idx_next+chunk_size-1] .+= λ_smooth * 2 * vec(accel)
    end

    G
end

function fit_smooth_embedding(P_list, d, λ_smooth; X_init=nothing, max_iter=1000, verbose=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    # Initialize with ASE + sequential Procrustes if not provided
    if X_init === nothing
        X_init = Vector{Matrix{Float64}}(undef, T)
        for t in 1:T
            U, S, _ = svd(P_list[t])
            X_init[t] = U[:, 1:d] * Diagonal(sqrt.(S[1:d]))
        end
        for t in 2:T
            F = svd(X_init[t]' * X_init[t-1])
            X_init[t] = X_init[t] * (F.U * F.Vt)
        end
    end

    θ0 = pack_X(X_init)

    f(θ) = smooth_embedding_loss(θ, P_list, n, d, λ_smooth)
    g!(G, θ) = smooth_embedding_gradient!(G, θ, P_list, n, d, λ_smooth)

    if verbose
        println("  Initial loss: ", round(f(θ0), digits=2))
    end

    result = optimize(f, g!, θ0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose, show_every=100))

    θ_opt = Optim.minimizer(result)
    X_opt = unpack_X(θ_opt, n, d, T)

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_opt
end

# ============================================================================
# Main
# ============================================================================

function main()
    # Generate data
    rng = MersenneTwister(SEED)
    X0 = generate_initial_X(rng)
    sol = solve(ODEProblem(true_dynamics!, X0, (0.0, Float64(T_TOTAL-1))), Tsit5(),
                saveat=0.0:1.0:Float64(T_TOTAL-1))
    X_true = [sol.u[t] for t in 1:T_TOTAL]
    P_list = [clamp.(X_true[t] * X_true[t]', 0.0, 1.0) for t in 1:T_TOTAL]

    println("=" ^ 70)
    println("SMOOTH EMBEDDING OPTIMIZATION")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue trajectory curvature: ", round(curv_true, digits=2), "°")

    # Baseline: ASE + Sequential Procrustes
    println("\n--- Baseline: ASE + Sequential Procrustes ---")
    X_baseline = Vector{Matrix{Float64}}(undef, T_TOTAL)
    for t in 1:T_TOTAL
        U, S, _ = svd(P_list[t])
        X_baseline[t] = U[:, 1:D_EMBED] * Diagonal(sqrt.(S[1:D_EMBED]))
    end
    for t in 2:T_TOTAL
        F = svd(X_baseline[t]' * X_baseline[t-1])
        X_baseline[t] = X_baseline[t] * (F.U * F.Vt)
    end

    # Test different smoothness weights
    λ_values = [0.0, 1.0, 10.0, 100.0, 1000.0]
    results = Dict{Float64, Vector{Matrix{Float64}}}()

    for λ in λ_values
        println("\n--- λ_smooth = ", λ, " ---")
        X_smooth = fit_smooth_embedding(P_list, D_EMBED, λ, X_init=deepcopy(X_baseline),
                                        max_iter=500, verbose=false)
        results[λ] = X_smooth
    end

    # Align all to true at t=1 for comparison
    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T_TOTAL]
    end

    X_baseline_al = align_to_true(X_baseline)
    for λ in λ_values
        results[λ] = align_to_true(results[λ])
    end

    # Compute metrics
    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
    end

    function pos_error(X_est)
        mean([norm(X_est[t] - X_true[t]) / norm(X_true[t]) for t in 1:T_TOTAL])
    end

    println("\n" * "=" ^ 70)
    println("RESULTS")
    println("=" ^ 70)
    println("\n", rpad("Method", 20), rpad("Curvature", 12), rpad("P-error", 12), "Pos-error")
    println("-" ^ 56)
    println(rpad("True", 20), rpad(string(round(curv_true, digits=2)) * "°", 12), "-", " " ^ 10, "-")
    println(rpad("Baseline (λ=0)", 20),
            rpad(string(round(compute_curvature(X_baseline_al), digits=2)) * "°", 12),
            rpad(string(round(100*p_error(X_baseline_al), digits=1)) * "%", 12),
            string(round(100*pos_error(X_baseline_al), digits=1)) * "%")

    for λ in λ_values[2:end]
        X = results[λ]
        println(rpad("λ=" * string(λ), 20),
                rpad(string(round(compute_curvature(X), digits=2)) * "°", 12),
                rpad(string(round(100*p_error(X), digits=1)) * "%", 12),
                string(round(100*pos_error(X), digits=1)) * "%")
    end

    # Visualization
    fig = Figure(size=(1600, 400))

    datasets = [
        (X_true, "True"),
        (X_baseline_al, "Baseline (λ=0)"),
        (results[10.0], "λ=10"),
        (results[100.0], "λ=100"),
        (results[1000.0], "λ=1000"),
    ]

    for (idx, (X_data, title)) in enumerate(datasets)
        ax = Axis(fig[1, idx], title=title, xlabel="X₁", ylabel="X₂", aspect=DataAspect())
        for type_idx in [TYPE_P, TYPE_Y, TYPE_R]
            nodes = findall(==(type_idx), NODE_TYPES)
            for i in nodes
                lines!(ax, [X_data[t][i, 1] for t in 1:T_TOTAL], [X_data[t][i, 2] for t in 1:T_TOTAL],
                       color=(TYPE_COLORS[type_idx], 0.3), linewidth=1)
            end
            cx = [mean(X_data[t][nodes, 1]) for t in 1:T_TOTAL]
            cy = [mean(X_data[t][nodes, 2]) for t in 1:T_TOTAL]
            lines!(ax, cx, cy, color=TYPE_COLORS[type_idx], linewidth=3)
            scatter!(ax, [cx[1]], [cy[1]], color=TYPE_COLORS[type_idx], markersize=12, marker=:circle)
            scatter!(ax, [cx[end]], [cy[end]], color=TYPE_COLORS[type_idx], markersize=12, marker=:star5)
        end
        xlims!(ax, -0.5, 1.5)
        ylims!(ax, -0.6, 1.2)
    end

    save(joinpath(dirname(@__DIR__), "results", "smooth_embedding.png"), fig, px_per_unit=2)
    println("\nSaved: results/smooth_embedding.png")
end

main()
