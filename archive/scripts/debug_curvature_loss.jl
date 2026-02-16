#!/usr/bin/env -S julia --project
"""
Diagnose: Where does DUASE lose trajectory curvature?

The core issue: True trajectories have smooth curvature that SciML can learn.
DUASE trajectories are either jagged (noisy) or linearized (lose higher-order effects).

Hypothesis testing:
1. Is it the shared basis G constraining dynamics?
2. Is it sqrt(Q) losing nonlinear structure?
3. Is it sampling noise in A_obs?
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using Statistics

# ============================================================================
# Setup (same as test_3x_nodes.jl)
# ============================================================================

const N_PRED = 12
const N_PREY = 15
const N_RES = 10
const N_TOTAL = N_PRED + N_PREY + N_RES
const D_EMBED = 2
const T_TOTAL = 25
const K_SAMPLES = 50
const SEED = 42

const TYPE_P, TYPE_Y, TYPE_R = 1, 2, 3
const NODE_TYPES = vcat(fill(TYPE_P, N_PRED), fill(TYPE_Y, N_PREY), fill(TYPE_R, N_RES))
const KNOWN_SELF_RATES = Dict(TYPE_P => -0.002, TYPE_Y => -0.001, TYPE_R => 0.000)
const HOLLING_ALPHA = 0.025
const HOLLING_BETA = 2.0

function κ_true(type_i::Int, type_j::Int, p::Real)
    if type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)  # Nonlinear!
    elseif type_i == TYPE_Y && type_j == TYPE_P
        return -0.02 * p
    elseif type_i == TYPE_Y && type_j == TYPE_R
        return 0.012 * p
    elseif type_i == TYPE_R && type_j == TYPE_Y
        return -0.006 * p
    elseif type_i == TYPE_P && type_j == TYPE_P
        return -0.004
    elseif type_i == TYPE_Y && type_j == TYPE_Y
        return 0.003
    elseif type_i == TYPE_R && type_j == TYPE_R
        return 0.005
    else
        return 0.0
    end
end

function compute_N_matrix(X::Matrix{Float64})
    n = size(X, 1)
    P = clamp.(X * X', 0.0, 1.0)
    N = zeros(n, n)
    for i in 1:n, j in 1:n
        N[i,j] = i == j ? KNOWN_SELF_RATES[NODE_TYPES[i]] : κ_true(NODE_TYPES[i], NODE_TYPES[j], P[i,j])
    end
    return N
end

function true_dynamics!(dX, X, p, t)
    N = compute_N_matrix(X)
    dX .= N * X
end

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
    return clamp.(X, 0.05, 0.95)
end

function simulate_trajectory(X0, T)
    prob = ODEProblem(true_dynamics!, X0, (0.0, Float64(T-1)))
    sol = solve(prob, Tsit5(), saveat=0.0:1.0:Float64(T-1), abstol=1e-8, reltol=1e-8)
    return [sol.u[t] for t in 1:T]
end

function generate_adjacencies(X_true, K, rng)
    T = length(X_true)
    n = size(X_true[1], 1)
    A_obs = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        P = clamp.(X_true[t] * X_true[t]', 0.0, 1.0)
        A_sum = zeros(n, n)
        for _ in 1:K
            A = Float64.(rand(rng, n, n) .< P)
            A = (A + A') / 2
            A_sum .+= A
        end
        A_obs[t] = A_sum / K
    end
    return A_obs
end

# Curvature = mean angle between consecutive velocity vectors
function compute_curvature(X_traj)
    T = length(X_traj)
    n = size(X_traj[1], 1)
    velocities = [X_traj[t+1] - X_traj[t] for t in 1:(T-1)]

    angles = Float64[]
    for t in 1:(T-2), i in 1:n
        v1, v2 = velocities[t][i, :], velocities[t+1][i, :]
        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            cos_angle = clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            push!(angles, acos(cos_angle) * 180 / π)
        end
    end
    return mean(angles), std(angles)
end

# ============================================================================
# Main analysis
# ============================================================================

function main()
    rng = MersenneTwister(SEED)

    println("=" ^ 70)
    println("CURVATURE LOSS DIAGNOSIS")
    println("=" ^ 70)

    # Generate true trajectory
    X0 = generate_initial_X(rng)
    X_true = simulate_trajectory(X0, T_TOTAL)

    curv_true, _ = compute_curvature(X_true)
    println("\nTrue trajectory curvature: ", round(curv_true, digits=2), "°")

    # ========================================================================
    # Test 1: Perfect P matrices (no sampling noise)
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 1: DUASE on PERFECT P matrices (no sampling)")
    println("=" ^ 70)

    P_perfect = [clamp.(X_true[t] * X_true[t]', 0.0, 1.0) for t in 1:T_TOTAL]
    G_perf, X_duase_perf = duase_embedding(P_perfect, D_EMBED)

    # Procrustes align to true
    F = svd(X_duase_perf[1]' * X_true[1])
    Q = F.U * F.Vt
    X_aligned_perf = [X_duase_perf[t] * Q for t in 1:T_TOTAL]

    curv_perf, _ = compute_curvature(X_aligned_perf)
    println("DUASE curvature (perfect P): ", round(curv_perf, digits=2), "°")
    println("Curvature ratio: ", round(curv_perf / curv_true, digits=2), "×")

    # P reconstruction error
    P_err = mean([norm(X_aligned_perf[t] * X_aligned_perf[t]' - P_perfect[t]) / norm(P_perfect[t]) for t in 1:T_TOTAL])
    println("Mean P reconstruction error: ", round(100 * P_err, digits=2), "%")

    # ========================================================================
    # Test 2: Independent ASE + Procrustes chain
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 2: Independent ASE + Procrustes chain (on perfect P)")
    println("=" ^ 70)

    X_ase = Vector{Matrix{Float64}}(undef, T_TOTAL)
    for t in 1:T_TOTAL
        U, S, V = svd(P_perfect[t])
        X_ase[t] = U[:, 1:D_EMBED] * Diagonal(sqrt.(S[1:D_EMBED]))
    end

    # Procrustes chain alignment
    X_ase_aligned = copy(X_ase)
    for t in 2:T_TOTAL
        F = svd(X_ase_aligned[t]' * X_ase_aligned[t-1])
        Q = F.U * F.Vt
        X_ase_aligned[t] = X_ase_aligned[t] * Q
    end

    # Align first to true
    F = svd(X_ase_aligned[1]' * X_true[1])
    Q = F.U * F.Vt
    X_ase_aligned = [X_ase_aligned[t] * Q for t in 1:T_TOTAL]

    curv_ase, _ = compute_curvature(X_ase_aligned)
    println("ASE+Procrustes curvature: ", round(curv_ase, digits=2), "°")
    println("Curvature ratio: ", round(curv_ase / curv_true, digits=2), "×")

    # ========================================================================
    # Test 3: Compare velocities directly
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 3: Velocity magnitude comparison")
    println("=" ^ 70)

    v_true = [X_true[t+1] - X_true[t] for t in 1:(T_TOTAL-1)]
    v_duase = [X_aligned_perf[t+1] - X_aligned_perf[t] for t in 1:(T_TOTAL-1)]
    v_ase = [X_ase_aligned[t+1] - X_ase_aligned[t] for t in 1:(T_TOTAL-1)]

    for t in [1, 5, 10, 15, 20]
        mag_true = mean([norm(v_true[t][i, :]) for i in 1:N_TOTAL])
        mag_duase = mean([norm(v_duase[t][i, :]) for i in 1:N_TOTAL])
        mag_ase = mean([norm(v_ase[t][i, :]) for i in 1:N_TOTAL])
        println("t=", t, ": true=", round(mag_true, digits=4),
                " DUASE=", round(mag_duase, digits=4), " (", round(mag_duase/mag_true, digits=2), "×)",
                " ASE=", round(mag_ase, digits=4), " (", round(mag_ase/mag_true, digits=2), "×)")
    end

    # ========================================================================
    # Test 4: Second derivative (acceleration) comparison
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 4: Acceleration comparison (2nd derivative)")
    println("=" ^ 70)

    a_true = [v_true[t+1] - v_true[t] for t in 1:(T_TOTAL-2)]
    a_duase = [v_duase[t+1] - v_duase[t] for t in 1:(T_TOTAL-2)]
    a_ase = [v_ase[t+1] - v_ase[t] for t in 1:(T_TOTAL-2)]

    mag_a_true = mean([mean([norm(a_true[t][i, :]) for i in 1:N_TOTAL]) for t in 1:length(a_true)])
    mag_a_duase = mean([mean([norm(a_duase[t][i, :]) for i in 1:N_TOTAL]) for t in 1:length(a_duase)])
    mag_a_ase = mean([mean([norm(a_ase[t][i, :]) for i in 1:N_TOTAL]) for t in 1:length(a_ase)])

    println("Mean acceleration magnitude:")
    println("  True:  ", round(mag_a_true, digits=5))
    println("  DUASE: ", round(mag_a_duase, digits=5), " (", round(mag_a_duase/mag_a_true, digits=2), "× true)")
    println("  ASE:   ", round(mag_a_ase, digits=5), " (", round(mag_a_ase/mag_a_true, digits=2), "× true)")

    # ========================================================================
    # Test 5: Check if DUASE preserves P evolution correctly
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 5: P matrix evolution preservation")
    println("=" ^ 70)

    # dP/dt from true trajectory
    dP_true = [P_perfect[t+1] - P_perfect[t] for t in 1:(T_TOTAL-1)]

    # dP/dt from DUASE trajectory
    P_duase = [X_aligned_perf[t] * X_aligned_perf[t]' for t in 1:T_TOTAL]
    dP_duase = [P_duase[t+1] - P_duase[t] for t in 1:(T_TOTAL-1)]

    println("dP/dt alignment (should be high if P dynamics are preserved):")
    for t in [1, 5, 10, 15, 20]
        dPt, dPd = dP_true[t], dP_duase[t]
        # Flatten and compute correlation
        corr = dot(vec(dPt), vec(dPd)) / (norm(dPt) * norm(dPd))
        println("  t=", t, ": correlation = ", round(corr, digits=3))
    end

    # ========================================================================
    # Key insight: Check the STRUCTURE of Q(t) eigenvalues
    # ========================================================================
    println("\n" * "=" ^ 70)
    println("TEST 6: DUASE Q(t) eigenvalue structure")
    println("=" ^ 70)

    # Get Q(t) = G' * P(t) * G
    Q_series = [G_perf' * P_perfect[t] * G_perf for t in 1:T_TOTAL]

    println("Q(t) eigenvalues over time:")
    for t in [1, 5, 10, 15, 20, 25]
        eigs = eigvals(Q_series[t])
        println("  t=", t, ": λ = [", round(eigs[1], digits=3), ", ", round(eigs[2], digits=3), "]")
    end

    # The key question: does sqrt(Q) linearize the dynamics?
    println("\nQ(t) vs sqrt(Q(t))*sqrt(Q(t))' reconstruction:")
    for t in [1, 13, 25]
        sqrt_Q = sqrt(Symmetric(Q_series[t]))
        P_recon = G_perf * sqrt_Q * sqrt_Q * G_perf'
        err = norm(P_recon - P_perfect[t]) / norm(P_perfect[t])
        println("  t=", t, ": reconstruction error = ", round(100*err, digits=2), "%")
    end

    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("\nCurvature comparison:")
    println("  True trajectory:        ", round(curv_true, digits=2), "° (baseline)")
    println("  DUASE (perfect P):      ", round(curv_perf, digits=2), "° (", round(curv_perf/curv_true, digits=2), "× inflation)")
    println("  ASE + Procrustes chain: ", round(curv_ase, digits=2), "° (", round(curv_ase/curv_true, digits=2), "× inflation)")
end

main()
