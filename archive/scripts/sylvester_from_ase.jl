#!/usr/bin/env -S julia --project
"""
Sylvester from ASE: Use ASE embedding (not DUASE) for Sylvester velocity.

ASE gives X̂ = U·Λ^{1/2} where X̂·X̂' = P exactly (top d components).
This should make Sylvester work properly.
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
# Embedding
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

"""
Sylvester propagation from ASE.

Key: At each step, use ASE of P(t) directly (which gives X̂X̂' = P exactly),
then compute Sylvester velocity from ΔP.
"""
function sylvester_propagate_from_ase(P_list, d)
    T = length(P_list)

    X_syl = Vector{Matrix{Float64}}(undef, T)

    # Start with ASE of P(1)
    X_syl[1], U, λ = ase_embed(P_list[1], d)
    σ = sqrt.(max.(λ, 1e-10))

    println("\n--- Sylvester propagation diagnostics ---")

    for t in 1:(T-1)
        # Get U, λ from current P (not from propagated X!)
        # This ensures X̂X̂' = P exactly
        _, U, λ = ase_embed(P_list[t], d)
        σ = sqrt.(max.(λ, 1e-10))

        # Current X in this eigenbasis
        # We need to express X_syl[t] in terms of this U
        # X_syl[t] is in a different gauge, so we Procrustes align
        X_ase_t, _, _ = ase_embed(P_list[t], d)
        F = svd(X_ase_t' * X_syl[t])
        Q = F.U * F.Vt
        X_aligned = X_syl[t] * Q'  # Align X_syl to X_ase gauge

        # Now X_aligned ≈ X_ase_t, both satisfy XX' ≈ P(t)

        # Compute ΔP
        ΔP = P_list[t+1] - P_list[t]

        # Project onto eigenbasis: S = U'·ΔP·U
        S = U' * ΔP * U
        S = (S + S') / 2

        # Solve Sylvester
        A = solve_sylvester_symmetric(S, σ)

        # Velocity in eigenbasis
        V_eigen = U * A

        # Propagate (in eigenbasis gauge, then transform back to X_syl gauge)
        X_next_eigen = X_ase_t + V_eigen
        X_syl[t+1] = X_next_eigen * Q  # Transform back

        # Diagnostic: check Sylvester equation satisfaction
        if t in [1, 5, 10, 15, 20]
            residual = V_eigen * X_ase_t' + X_ase_t * V_eigen' - ΔP
            rel_err = norm(residual) / (norm(ΔP) + 1e-10)
            p_err = norm(X_syl[t+1] * X_syl[t+1]' - P_list[t+1]) / norm(P_list[t+1])
            println("t=", t, ": Sylvester residual ", round(100*rel_err, digits=1),
                    "%, P error ", round(100*p_err, digits=1), "%")
        end
    end

    return X_syl
end

"""
Alternative: pure ASE + Procrustes chain (for comparison)
"""
function ase_procrustes_embed(P_list, d)
    T = length(P_list)
    X_ase = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        X_ase[t], _, _ = ase_embed(P_list[t], d)
    end

    # Procrustes chain
    for t in 2:T
        F = svd(X_ase[t]' * X_ase[t-1])
        X_ase[t] = X_ase[t] * (F.U * F.Vt)
    end

    return X_ase
end

"""
DUASE for comparison
"""
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
    println("SYLVESTER FROM ASE")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue curvature: ", round(curv_true, digits=2), "°")

    # Align function
    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T_TOTAL]
    end

    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
    end

    # Methods
    println("\n--- DUASE ---")
    X_duase = duase_embed(P_list, D_EMBED)
    X_duase_al = align_to_true(X_duase)
    println("Curvature: ", round(compute_curvature(X_duase_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_al), digits=1), "%")

    println("\n--- ASE + Procrustes ---")
    X_ase = ase_procrustes_embed(P_list, D_EMBED)
    X_ase_al = align_to_true(X_ase)
    println("Curvature: ", round(compute_curvature(X_ase_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_ase_al), digits=1), "%")

    println("\n--- Sylvester from ASE ---")
    X_syl = sylvester_propagate_from_ase(P_list, D_EMBED)
    X_syl_al = align_to_true(X_syl)
    println("Curvature: ", round(compute_curvature(X_syl_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_syl_al), digits=1), "%")

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println(rpad("Method", 25), rpad("Curvature", 15), "P-error")
    println("-" ^ 50)
    println(rpad("True", 25), rpad(string(round(curv_true, digits=2)) * "°", 15), "-")
    println(rpad("DUASE", 25),
            rpad(string(round(compute_curvature(X_duase_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_al), digits=1)) * "%")
    println(rpad("ASE + Procrustes", 25),
            rpad(string(round(compute_curvature(X_ase_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_ase_al), digits=1)) * "%")
    println(rpad("Sylvester from ASE", 25),
            rpad(string(round(compute_curvature(X_syl_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_syl_al), digits=1)) * "%")

    # Visualization
    fig = Figure(size=(1600, 400))

    datasets = [
        (X_true, "True"),
        (X_duase_al, "DUASE"),
        (X_ase_al, "ASE+Procrustes"),
        (X_syl_al, "Sylvester from ASE"),
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

    save(joinpath(dirname(@__DIR__), "results", "sylvester_from_ase.png"), fig, px_per_unit=2)
    println("\nSaved: results/sylvester_from_ase.png")
end

main()
