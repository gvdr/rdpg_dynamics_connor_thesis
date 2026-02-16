#!/usr/bin/env -S julia --project
"""
Three-Point Sylvester: Use t-1, t, t+1 to couple velocities via acceleration.

Key equations:
- dP/dt = VX' + XV'  (velocity)
- d²P/dt² = AX' + XA' + 2VV'  (acceleration)

Algorithm:
1. Initialize V(1) from first-order Sylvester
2. For each t: compute A from d²P - 2VV', then V(t) = V(t-1) + A
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Random
using OrdinaryDiffEq
using Statistics
using CairoMakie

# ============================================================================
# Setup
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
# Embedding methods
# ============================================================================

function ase_embed(P::Matrix{Float64}, d::Int)
    P_sym = (P + P') / 2
    eig = eigen(Symmetric(P_sym), sortby = x -> -x)
    λ = max.(eig.values[1:d], 0.0)
    U = eig.vectors[:, 1:d]
    X = U * Diagonal(sqrt.(λ))
    return X, U, λ
end

function solve_sylvester_symmetric(S::Matrix{Float64}, σ::Vector{Float64})
    d = length(σ)
    A = zeros(d, d)
    for i in 1:d
        A[i, i] = S[i, i] / (2 * σ[i] + 1e-10)
    end
    for i in 1:d, j in (i+1):d
        A[i, j] = S[i, j] / (σ[i] + σ[j] + 1e-10)
        A[j, i] = A[i, j]
    end
    return A
end

function duase_embed(P_list, d)
    T = length(P_list)
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
# Three-Point Sylvester
# ============================================================================

"""
Three-point Sylvester embedding with acceleration coupling.

Algorithm:
1. Initialize X(1) from ASE, V(1) from first-order Sylvester
2. For t = 2 to T-1:
   - Compute d²P = P(t+1) - 2P(t) + P(t-1)
   - Solve for A: AX' + XA' = d²P - 2VV'
   - Update V(t) = V(t-1) + A
   - Propagate X(t+1) = X(t) + V(t)
"""
function three_point_sylvester(P_list, d)
    T = length(P_list)
    n = size(P_list[1], 1)

    X = Vector{Matrix{Float64}}(undef, T)
    V = Vector{Matrix{Float64}}(undef, T-1)

    # Initialize: X(1) from ASE
    X[1], U, λ = ase_embed(P_list[1], d)
    σ = sqrt.(max.(λ, 1e-10))

    # V(1) from first-order Sylvester: VX' + XV' = ΔP(1)
    ΔP_1 = P_list[2] - P_list[1]
    S = U' * ΔP_1 * U
    S = (S + S') / 2
    A_coef = solve_sylvester_symmetric(S, σ)
    V[1] = U * A_coef

    # X(2) from first propagation
    X[2] = X[1] + V[1]

    # Three-point propagation
    for t in 2:(T-1)
        # Get eigenbasis from current X
        P_current = X[t] * X[t]'
        _, U, λ = ase_embed(P_current, d)
        σ = sqrt.(max.(λ, 1e-10))

        # Second derivative: d²P = P(t+1) - 2P(t) + P(t-1)
        d2P = P_list[t+1] - 2*P_list[t] + P_list[t-1]

        # Right-hand side for acceleration: d²P - 2VV'
        VVt = V[t-1] * V[t-1]'
        M = d2P - 2*VVt

        # Project and solve for acceleration
        S = U' * M * U
        S = (S + S') / 2
        B = solve_sylvester_symmetric(S, σ)
        A = U * B

        # Update velocity: V(t) = V(t-1) + A
        V[t] = V[t-1] + A

        # Propagate position: X(t+1) = X(t) + V(t)
        X[t+1] = X[t] + V[t]
    end

    return X, V
end

"""
Variant: Re-anchor to P periodically to prevent drift.
"""
function three_point_sylvester_anchored(P_list, d; anchor_every=5)
    T = length(P_list)
    n = size(P_list[1], 1)

    X = Vector{Matrix{Float64}}(undef, T)
    V = Vector{Matrix{Float64}}(undef, T-1)

    # Initialize
    X[1], U, λ = ase_embed(P_list[1], d)
    σ = sqrt.(max.(λ, 1e-10))

    ΔP_1 = P_list[2] - P_list[1]
    S = U' * ΔP_1 * U
    S = (S + S') / 2
    A_coef = solve_sylvester_symmetric(S, σ)
    V[1] = U * A_coef
    X[2] = X[1] + V[1]

    for t in 2:(T-1)
        # Re-anchor periodically
        if (t - 1) % anchor_every == 0 && t > 2
            X_ase, _, _ = ase_embed(P_list[t], d)
            # Procrustes align to current X
            F = svd(X_ase' * X[t])
            X[t] = X_ase * (F.U * F.Vt)
            # Re-compute V from finite difference
            V[t-1] = X[t] - X[t-1]
        end

        P_current = X[t] * X[t]'
        _, U, λ = ase_embed(P_current, d)
        σ = sqrt.(max.(λ, 1e-10))

        d2P = P_list[t+1] - 2*P_list[t] + P_list[t-1]
        VVt = V[t-1] * V[t-1]'
        M = d2P - 2*VVt

        S = U' * M * U
        S = (S + S') / 2
        B = solve_sylvester_symmetric(S, σ)
        A = U * B

        V[t] = V[t-1] + A
        X[t+1] = X[t] + V[t]
    end

    return X, V
end

# ============================================================================
# Main
# ============================================================================

function main()
    rng = MersenneTwister(SEED)
    X0 = generate_initial_X(rng)
    sol = solve(ODEProblem(true_dynamics!, X0, (0.0, Float64(T_TOTAL-1))), Tsit5(),
                saveat=0.0:1.0:Float64(T_TOTAL-1))
    X_true = [sol.u[t] for t in 1:T_TOTAL]
    P_list = [clamp.(X_true[t] * X_true[t]', 0.0, 1.0) for t in 1:T_TOTAL]

    println("=" ^ 70)
    println("THREE-POINT SYLVESTER EMBEDDING")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue curvature: ", round(curv_true, digits=2), "°")

    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T_TOTAL]
    end

    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
    end

    # DUASE
    println("\n--- DUASE ---")
    X_duase = duase_embed(P_list, D_EMBED)
    X_duase_al = align_to_true(X_duase)
    println("Curvature: ", round(compute_curvature(X_duase_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_al), digits=1), "%")

    # Three-point Sylvester
    println("\n--- Three-Point Sylvester ---")
    X_3pt, V_3pt = three_point_sylvester(P_list, D_EMBED)
    X_3pt_al = align_to_true(X_3pt)
    println("Curvature: ", round(compute_curvature(X_3pt_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_3pt_al), digits=1), "%")

    # Three-point with anchoring
    println("\n--- Three-Point Sylvester (anchored/5) ---")
    X_3pt_anch, _ = three_point_sylvester_anchored(P_list, D_EMBED, anchor_every=5)
    X_3pt_anch_al = align_to_true(X_3pt_anch)
    println("Curvature: ", round(compute_curvature(X_3pt_anch_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_3pt_anch_al), digits=1), "%")

    # P error over time
    println("\n--- P error over time ---")
    println("t\tDUASE\t\t3-pt\t\t3-pt-anch")
    for t in [1, 5, 10, 15, 20, 25]
        e_d = norm(X_duase_al[t] * X_duase_al[t]' - P_list[t]) / norm(P_list[t])
        e_3 = norm(X_3pt_al[t] * X_3pt_al[t]' - P_list[t]) / norm(P_list[t])
        e_3a = norm(X_3pt_anch_al[t] * X_3pt_anch_al[t]' - P_list[t]) / norm(P_list[t])
        println(t, "\t", round(100*e_d, digits=1), "%\t\t",
                round(100*e_3, digits=1), "%\t\t", round(100*e_3a, digits=1), "%")
    end

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println(rpad("Method", 30), rpad("Curvature", 15), "P-error")
    println("-" ^ 55)
    println(rpad("True", 30), rpad(string(round(curv_true, digits=2)) * "°", 15), "-")
    println(rpad("DUASE", 30),
            rpad(string(round(compute_curvature(X_duase_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_al), digits=1)) * "%")
    println(rpad("Three-Point Sylvester", 30),
            rpad(string(round(compute_curvature(X_3pt_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_3pt_al), digits=1)) * "%")
    println(rpad("Three-Point (anchored/5)", 30),
            rpad(string(round(compute_curvature(X_3pt_anch_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_3pt_anch_al), digits=1)) * "%")

    # Visualization
    fig = Figure(size=(1600, 400))

    datasets = [
        (X_true, "True"),
        (X_duase_al, "DUASE"),
        (X_3pt_al, "3-Point Sylvester"),
        (X_3pt_anch_al, "3-Point (anchored)"),
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

    save(joinpath(dirname(@__DIR__), "results", "three_point_sylvester.png"), fig, px_per_unit=2)
    println("\nSaved: results/three_point_sylvester.png")
end

main()
