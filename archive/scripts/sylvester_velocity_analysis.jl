#!/usr/bin/env -S julia --project
"""
Sylvester Velocity Analysis

Instead of propagating with Sylvester, compute Sylvester-derived velocities
from DUASE embeddings and analyze their curvature.

Key question: Does V_Sylvester have better curvature than V_DUASE = X(t+1) - X(t)?
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

# ============================================================================
# Embedding methods
# ============================================================================

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
    return X_duase, G
end

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
Compute Sylvester velocity from X̂ and ΔP.
V̂ satisfies V̂X̂' + X̂V̂' = ΔP (projected to eigenbasis).
"""
function sylvester_velocity(X::Matrix{Float64}, ΔP::Matrix{Float64}, d::Int)
    # Get eigenbasis of XX'
    P_current = X * X'
    _, U, λ = ase_embed(P_current, d)
    σ = sqrt.(max.(λ, 1e-10))

    # Project ΔP
    S = U' * ΔP * U
    S = (S + S') / 2

    # Solve Sylvester
    A = solve_sylvester_symmetric(S, σ)

    # Velocity
    return U * A
end

# ============================================================================
# Curvature analysis
# ============================================================================

function compute_curvature_from_velocities(V_list)
    T = length(V_list)
    n = size(V_list[1], 1)
    angles = Float64[]
    for t in 1:(T-1), i in 1:n
        v1, v2 = V_list[t][i, :], V_list[t+1][i, :]
        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            push!(angles, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
        end
    end
    mean(angles)
end

function compute_velocity_alignment(V_est, V_true)
    # Measure how well estimated velocities align with true velocities
    T = length(V_est)
    n = size(V_est[1], 1)
    alignments = Float64[]
    for t in 1:T, i in 1:n
        v1, v2 = V_est[t][i, :], V_true[t][i, :]
        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            push!(alignments, dot(v1, v2) / (n1 * n2))
        end
    end
    mean(alignments)
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
    println("SYLVESTER VELOCITY ANALYSIS")
    println("=" ^ 70)

    # True velocities
    V_true = [X_true[t+1] - X_true[t] for t in 1:(T_TOTAL-1)]
    curv_true = compute_curvature_from_velocities(V_true)
    println("\nTrue velocity curvature: ", round(curv_true, digits=2), "°")

    # DUASE embedding
    X_duase, G = duase_embed(P_list, D_EMBED)

    # Align DUASE to true
    F = svd(X_duase[1]' * X_true[1])
    Q_align = F.U * F.Vt
    X_duase_al = [X_duase[t] * Q_align for t in 1:T_TOTAL]

    # DUASE velocities (finite difference)
    V_duase = [X_duase_al[t+1] - X_duase_al[t] for t in 1:(T_TOTAL-1)]
    curv_duase = compute_curvature_from_velocities(V_duase)
    println("DUASE velocity curvature: ", round(curv_duase, digits=2), "°")

    # Sylvester velocities from DUASE positions
    V_sylvester = Vector{Matrix{Float64}}(undef, T_TOTAL-1)
    for t in 1:(T_TOTAL-1)
        ΔP = P_list[t+1] - P_list[t]
        V_sylvester[t] = sylvester_velocity(X_duase_al[t], ΔP, D_EMBED)
    end
    curv_sylvester = compute_curvature_from_velocities(V_sylvester)
    println("Sylvester velocity curvature: ", round(curv_sylvester, digits=2), "°")

    # Velocity alignment with true
    align_duase = compute_velocity_alignment(V_duase, V_true)
    align_sylvester = compute_velocity_alignment(V_sylvester, V_true)
    println("\nVelocity alignment with true:")
    println("  DUASE: ", round(align_duase, digits=3))
    println("  Sylvester: ", round(align_sylvester, digits=3))

    # Per-type curvature
    println("\n--- Per-type curvature ---")
    for type_idx in [TYPE_P, TYPE_Y, TYPE_R]
        nodes = findall(==(type_idx), NODE_TYPES)

        # True
        angles_true = Float64[]
        for t in 1:(T_TOTAL-2), i in nodes
            v1, v2 = V_true[t][i, :], V_true[t+1][i, :]
            n1, n2 = norm(v1), norm(v2)
            if n1 > 1e-10 && n2 > 1e-10
                push!(angles_true, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
            end
        end

        # DUASE
        angles_duase = Float64[]
        for t in 1:(T_TOTAL-2), i in nodes
            v1, v2 = V_duase[t][i, :], V_duase[t+1][i, :]
            n1, n2 = norm(v1), norm(v2)
            if n1 > 1e-10 && n2 > 1e-10
                push!(angles_duase, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
            end
        end

        # Sylvester
        angles_syl = Float64[]
        for t in 1:(T_TOTAL-2), i in nodes
            v1, v2 = V_sylvester[t][i, :], V_sylvester[t+1][i, :]
            n1, n2 = norm(v1), norm(v2)
            if n1 > 1e-10 && n2 > 1e-10
                push!(angles_syl, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
            end
        end

        println(TYPE_NAMES[type_idx], ":")
        println("  True: ", round(mean(angles_true), digits=2), "°")
        println("  DUASE: ", round(mean(angles_duase), digits=2), "°")
        println("  Sylvester: ", round(mean(angles_syl), digits=2), "°")
    end

    # Check if Sylvester velocity is consistent with ΔP
    println("\n--- Sylvester velocity consistency check ---")
    println("||V̂X̂' + X̂V̂' - ΔP|| / ||ΔP||:")
    for t in [1, 5, 10, 15, 20]
        X = X_duase_al[t]
        V = V_sylvester[t]
        ΔP = P_list[t+1] - P_list[t]
        residual = V * X' + X * V' - ΔP
        rel_err = norm(residual) / norm(ΔP)
        println("  t=", t, ": ", round(100*rel_err, digits=1), "%")
    end

    # Visualization
    fig = Figure(size=(1400, 600))

    # Plot 1: Velocity directions at t=10
    ax1 = Axis(fig[1, 1], title="Velocities at t=10", xlabel="X₁", ylabel="X₂", aspect=DataAspect())

    t_show = 10
    scale = 8.0

    for type_idx in [TYPE_P, TYPE_Y, TYPE_R]
        nodes = findall(==(type_idx), NODE_TYPES)
        for i in nodes[1:min(3, length(nodes))]  # Show first 3 of each type
            x, y = X_duase_al[t_show][i, :]

            # True velocity (black, dashed)
            vx_t, vy_t = V_true[t_show][i, :] * scale
            arrows!(ax1, [x], [y], [vx_t], [vy_t], color=:black, linewidth=2, linestyle=:dash)

            # DUASE velocity
            vx_d, vy_d = V_duase[t_show][i, :] * scale
            arrows!(ax1, [x], [y], [vx_d], [vy_d], color=TYPE_COLORS[type_idx], linewidth=2)

            # Sylvester velocity (dotted)
            vx_s, vy_s = V_sylvester[t_show][i, :] * scale
            arrows!(ax1, [x+0.01], [y+0.01], [vx_s], [vy_s], color=TYPE_COLORS[type_idx], linewidth=2, linestyle=:dot)
        end
    end
    # Legend
    scatter!(ax1, [-10], [-10], color=:black, label="True")
    scatter!(ax1, [-10], [-10], color=:red, label="DUASE")
    # axislegend(ax1, position=:lt)

    # Plot 2: Curvature over time
    ax2 = Axis(fig[1, 2], title="Curvature over time", xlabel="Time", ylabel="Curvature (°)")

    # Compute curvature at each timestep
    curv_true_t = Float64[]
    curv_duase_t = Float64[]
    curv_syl_t = Float64[]

    for t in 1:(T_TOTAL-2)
        # True
        angles = [let v1=V_true[t][i,:], v2=V_true[t+1][i,:], n1=norm(v1), n2=norm(v2)
                    n1 > 1e-10 && n2 > 1e-10 ? acos(clamp(dot(v1,v2)/(n1*n2), -1, 1))*180/π : 0.0
                  end for i in 1:N_TOTAL]
        push!(curv_true_t, mean(filter(x -> x > 0, angles)))

        # DUASE
        angles = [let v1=V_duase[t][i,:], v2=V_duase[t+1][i,:], n1=norm(v1), n2=norm(v2)
                    n1 > 1e-10 && n2 > 1e-10 ? acos(clamp(dot(v1,v2)/(n1*n2), -1, 1))*180/π : 0.0
                  end for i in 1:N_TOTAL]
        push!(curv_duase_t, mean(filter(x -> x > 0, angles)))

        # Sylvester
        angles = [let v1=V_sylvester[t][i,:], v2=V_sylvester[t+1][i,:], n1=norm(v1), n2=norm(v2)
                    n1 > 1e-10 && n2 > 1e-10 ? acos(clamp(dot(v1,v2)/(n1*n2), -1, 1))*180/π : 0.0
                  end for i in 1:N_TOTAL]
        push!(curv_syl_t, mean(filter(x -> x > 0, angles)))
    end

    lines!(ax2, 1:(T_TOTAL-2), curv_true_t, color=:black, linewidth=2, label="True")
    lines!(ax2, 1:(T_TOTAL-2), curv_duase_t, color=:blue, linewidth=2, label="DUASE")
    lines!(ax2, 1:(T_TOTAL-2), curv_syl_t, color=:red, linewidth=2, label="Sylvester")
    axislegend(ax2, position=:rt)

    # Plot 3: Velocity magnitude comparison
    ax3 = Axis(fig[1, 3], title="Velocity magnitude over time", xlabel="Time", ylabel="Mean |V|")

    mag_true = [mean([norm(V_true[t][i, :]) for i in 1:N_TOTAL]) for t in 1:(T_TOTAL-1)]
    mag_duase = [mean([norm(V_duase[t][i, :]) for i in 1:N_TOTAL]) for t in 1:(T_TOTAL-1)]
    mag_syl = [mean([norm(V_sylvester[t][i, :]) for i in 1:N_TOTAL]) for t in 1:(T_TOTAL-1)]

    lines!(ax3, 1:(T_TOTAL-1), mag_true, color=:black, linewidth=2, label="True")
    lines!(ax3, 1:(T_TOTAL-1), mag_duase, color=:blue, linewidth=2, label="DUASE")
    lines!(ax3, 1:(T_TOTAL-1), mag_syl, color=:red, linewidth=2, label="Sylvester")
    axislegend(ax3, position=:rt)

    save(joinpath(dirname(@__DIR__), "results", "sylvester_velocity_analysis.png"), fig, px_per_unit=2)
    println("\nSaved: results/sylvester_velocity_analysis.png")
end

main()
