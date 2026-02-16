#!/usr/bin/env -S julia --project
"""
Sylvester Embedding: Extract velocity from ΔP to recover curvature.

Key idea: Given P(t) and ΔP(t) = P(t+1) - P(t), solve the Sylvester equation
    V̂X̂' + X̂V̂' = ΔP
for velocity V̂, then propagate X̂(t+1) = X̂(t) + V̂(t).

This should recover curvature that DUASE loses.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Random
using OrdinaryDiffEq
using Statistics
using CairoMakie

# ============================================================================
# Setup (same ecological model)
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
const TYPE_NAMES = Dict(TYPE_P => "Predator", TYPE_Y => "Prey", TYPE_R => "Resource")

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
# DUASE Embedding (for comparison)
# ============================================================================

function duase_embed(P_list, d)
    T = length(P_list)
    n = size(P_list[1], 1)

    Unfolded = hcat(P_list...)
    U, S, V = svd(Unfolded)
    G = U[:, 1:d]

    X_duase = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        Qt = G' * P_list[t] * G
        Qt_sym = (Qt + Qt') / 2
        eig = eigen(Symmetric(Qt_sym))
        sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
        X_duase[t] = G * sqrt_Q
    end
    return X_duase
end

# ============================================================================
# Sylvester Embedding
# ============================================================================

"""
Solve the Sylvester equation A·Λ^{1/2} + Λ^{1/2}·A' = S for symmetric A.

For d=2 with Λ^{1/2} = diag(σ₁, σ₂):
- a₁₁ = s₁₁/(2σ₁)
- a₂₂ = s₂₂/(2σ₂)
- a₁₂ = a₂₁ = s₁₂/(σ₁ + σ₂)
"""
function solve_sylvester_symmetric(S::Matrix{Float64}, σ::Vector{Float64})
    d = length(σ)
    A = zeros(d, d)

    for i in 1:d
        A[i, i] = S[i, i] / (2 * σ[i])
    end

    for i in 1:d, j in (i+1):d
        A[i, j] = S[i, j] / (σ[i] + σ[j])
        A[j, i] = A[i, j]  # symmetric
    end

    return A
end

"""
ASE embedding of a single P matrix.
Returns X̂ = U·Λ^{1/2} and the components U, λ.
"""
function ase_embed(P::Matrix{Float64}, d::Int)
    # Symmetrize and ensure PSD
    P_sym = (P + P') / 2
    eig = eigen(Symmetric(P_sym), sortby = x -> -x)  # descending order

    # Take top d positive eigenvalues
    λ = max.(eig.values[1:d], 0.0)
    U = eig.vectors[:, 1:d]

    X = U * Diagonal(sqrt.(λ))
    return X, U, λ
end

"""
Sylvester propagation embedding.

Algorithm:
1. Embed P(1) to get X̂(1)
2. For each t:
   a. Compute ΔP(t) = P(t+1) - P(t)
   b. Solve Sylvester for A: A·Λ^{1/2} + Λ^{1/2}·A' = U'·ΔP·U
   c. V̂(t) = U·A
   d. Propagate: X̂(t+1) = X̂(t) + V̂(t)
"""
function sylvester_embed(P_list, d)
    T = length(P_list)
    n = size(P_list[1], 1)

    X_syl = Vector{Matrix{Float64}}(undef, T)
    V_syl = Vector{Matrix{Float64}}(undef, T-1)

    # Initialize with ASE of P(1)
    X_syl[1], U, λ = ase_embed(P_list[1], d)
    σ = sqrt.(λ)

    # Propagate
    for t in 1:(T-1)
        # Current embedding and its SVD structure
        # We need U and σ from current X̂(t)
        # For propagated X, re-extract U and λ
        if t > 1
            # Re-compute U and λ from current X̂
            # X̂ = U·Λ^{1/2}, so X̂·X̂' = U·Λ·U'
            # But X̂ may have drifted, so re-embed from X̂·X̂'
            P_current = X_syl[t] * X_syl[t]'
            _, U, λ = ase_embed(P_current, d)
            σ = sqrt.(max.(λ, 1e-10))
        end

        # Compute ΔP
        ΔP = P_list[t+1] - P_list[t]

        # Project onto eigenbasis: S = U'·ΔP·U
        S = U' * ΔP * U
        S = (S + S') / 2  # ensure symmetric

        # Solve Sylvester equation
        A = solve_sylvester_symmetric(S, σ)

        # Velocity in original space: V̂ = U·A
        V_syl[t] = U * A

        # Propagate: X̂(t+1) = X̂(t) + V̂(t)
        X_syl[t+1] = X_syl[t] + V_syl[t]
    end

    return X_syl, V_syl
end

"""
Sylvester embedding with periodic re-projection to P-consistent manifold.
"""
function sylvester_embed_projected(P_list, d; project_every=5)
    T = length(P_list)
    n = size(P_list[1], 1)

    X_syl = Vector{Matrix{Float64}}(undef, T)

    # Initialize with ASE of P(1)
    X_syl[1], U, λ = ase_embed(P_list[1], d)
    σ = sqrt.(λ)

    for t in 1:(T-1)
        # Re-embed if at projection step, aligning to current X̂
        if t > 1 && (t-1) % project_every == 0
            X_new, _, _ = ase_embed(P_list[t], d)
            # Procrustes align X_new to X_syl[t]
            F = svd(X_new' * X_syl[t])
            Q = F.U * F.Vt
            X_syl[t] = X_new * Q
        end

        # Get current structure
        P_current = X_syl[t] * X_syl[t]'
        _, U, λ = ase_embed(P_current, d)
        σ = sqrt.(max.(λ, 1e-10))

        # Compute ΔP and solve Sylvester
        ΔP = P_list[t+1] - P_list[t]
        S = U' * ΔP * U
        S = (S + S') / 2
        A = solve_sylvester_symmetric(S, σ)
        V = U * A

        # Propagate
        X_syl[t+1] = X_syl[t] + V
    end

    return X_syl
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
    println("SYLVESTER EMBEDDING TEST")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue trajectory curvature: ", round(curv_true, digits=2), "°")

    # Align function
    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T_TOTAL]
    end

    # DUASE embedding
    println("\n--- DUASE Embedding ---")
    X_duase = duase_embed(P_list, D_EMBED)
    X_duase_al = align_to_true(X_duase)
    curv_duase = compute_curvature(X_duase_al)
    println("DUASE curvature: ", round(curv_duase, digits=2), "° (", round(curv_duase/curv_true, digits=2), "× true)")

    # Sylvester embedding (pure propagation)
    println("\n--- Sylvester Embedding (pure propagation) ---")
    X_syl, V_syl = sylvester_embed(P_list, D_EMBED)
    X_syl_al = align_to_true(X_syl)
    curv_syl = compute_curvature(X_syl_al)
    println("Sylvester curvature: ", round(curv_syl, digits=2), "° (", round(curv_syl/curv_true, digits=2), "× true)")

    # Sylvester with periodic projection
    println("\n--- Sylvester Embedding (with projection every 5 steps) ---")
    X_syl_proj = sylvester_embed_projected(P_list, D_EMBED, project_every=5)
    X_syl_proj_al = align_to_true(X_syl_proj)
    curv_syl_proj = compute_curvature(X_syl_proj_al)
    println("Sylvester+proj curvature: ", round(curv_syl_proj, digits=2), "° (", round(curv_syl_proj/curv_true, digits=2), "× true)")

    # Metrics
    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
    end

    function pos_error(X_est)
        mean([norm(X_est[t] - X_true[t]) / norm(X_true[t]) for t in 1:T_TOTAL])
    end

    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("\n", rpad("Method", 25), rpad("Curvature", 15), rpad("P-error", 12), "Pos-error")
    println("-" ^ 65)
    println(rpad("True", 25), rpad(string(round(curv_true, digits=2)) * "°", 15), "-", " " ^ 10, "-")
    println(rpad("DUASE", 25),
            rpad(string(round(curv_duase, digits=2)) * "°", 15),
            rpad(string(round(100*p_error(X_duase_al), digits=1)) * "%", 12),
            string(round(100*pos_error(X_duase_al), digits=1)) * "%")
    println(rpad("Sylvester (propagate)", 25),
            rpad(string(round(curv_syl, digits=2)) * "°", 15),
            rpad(string(round(100*p_error(X_syl_al), digits=1)) * "%", 12),
            string(round(100*pos_error(X_syl_al), digits=1)) * "%")
    println(rpad("Sylvester (proj/5)", 25),
            rpad(string(round(curv_syl_proj, digits=2)) * "°", 15),
            rpad(string(round(100*p_error(X_syl_proj_al), digits=1)) * "%", 12),
            string(round(100*pos_error(X_syl_proj_al), digits=1)) * "%")

    # Check P error over time
    println("\n--- P reconstruction error over time ---")
    println("t\tDUASE\t\tSylvester\tSyl+proj")
    for t in [1, 5, 10, 15, 20, 25]
        err_d = norm(X_duase_al[t] * X_duase_al[t]' - P_list[t]) / norm(P_list[t])
        err_s = norm(X_syl_al[t] * X_syl_al[t]' - P_list[t]) / norm(P_list[t])
        err_sp = norm(X_syl_proj_al[t] * X_syl_proj_al[t]' - P_list[t]) / norm(P_list[t])
        println(t, "\t", round(100*err_d, digits=1), "%\t\t",
                round(100*err_s, digits=1), "%\t\t", round(100*err_sp, digits=1), "%")
    end

    # Visualization
    fig = Figure(size=(1600, 800))

    datasets = [
        (X_true, "True"),
        (X_duase_al, "DUASE"),
        (X_syl_al, "Sylvester (propagate)"),
        (X_syl_proj_al, "Sylvester (proj/5)"),
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
    end

    # Row 2: Velocity vectors at selected times
    for (idx, (X_data, title)) in enumerate(datasets)
        ax = Axis(fig[2, idx], title=title * " (velocity)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())

        t_show = 10  # Show velocities at t=10
        if idx == 1
            V_data = [X_true[t+1] - X_true[t] for t in 1:(T_TOTAL-1)]
        elseif idx == 2
            V_data = [X_duase_al[t+1] - X_duase_al[t] for t in 1:(T_TOTAL-1)]
        elseif idx == 3
            V_data = [X_syl_al[t+1] - X_syl_al[t] for t in 1:(T_TOTAL-1)]
        else
            V_data = [X_syl_proj_al[t+1] - X_syl_proj_al[t] for t in 1:(T_TOTAL-1)]
        end

        scale = 5.0  # Scale up velocities for visibility
        for type_idx in [TYPE_P, TYPE_Y, TYPE_R]
            nodes = findall(==(type_idx), NODE_TYPES)
            for i in nodes
                x, y = X_data[t_show][i, :]
                vx, vy = V_data[t_show][i, :] * scale
                arrows!(ax, [x], [y], [vx], [vy], color=TYPE_COLORS[type_idx], linewidth=1.5)
            end
        end
        scatter!(ax, X_data[t_show][:, 1], X_data[t_show][:, 2],
                 color=[TYPE_COLORS[NODE_TYPES[i]] for i in 1:N_TOTAL], markersize=6)
    end

    save(joinpath(dirname(@__DIR__), "results", "sylvester_embedding.png"), fig, px_per_unit=2)
    println("\nSaved: results/sylvester_embedding.png")
end

main()
