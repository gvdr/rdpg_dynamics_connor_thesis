#!/usr/bin/env -S julia --project
"""
Compare embedding methods for curvature preservation:
1. DUASE (Q(t) method) - current approach
2. V-block method - use V blocks from unfolded SVD directly
3. ASE + Procrustes chain - independent embedding with alignment
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
const TYPE_COLORS = Dict(TYPE_P => :red, TYPE_Y => :blue, TYPE_R => :green)

function κ_true(type_i::Int, type_j::Int, p::Real)
    if type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)
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

function compute_N_matrix(X)
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
    return mean(angles)
end

# ============================================================================
# EMBEDDING METHODS
# ============================================================================

# Method 1: DUASE (current approach) - Q(t) projection
function duase_embed(A_list, d)
    T = length(A_list)
    n = size(A_list[1], 1)

    # Unfold horizontally: [A(1) | A(2) | ... | A(T)]
    Unfolded = hcat(A_list...)  # n × (n*T)

    # SVD
    U, S, V = svd(Unfolded)
    G = U[:, 1:d]  # Shared basis

    # Q(t) approach: project each A(t) onto shared basis
    X_duase = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        Qt = G' * A_list[t] * G
        Qt_sym = (Qt + Qt') / 2
        eig = eigen(Symmetric(Qt_sym))
        sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
        X_duase[t] = G * sqrt_Q
    end
    return X_duase
end

# Method 2: V-block approach - use V blocks from unfolded SVD directly
function vblock_embed(A_list, d)
    T = length(A_list)
    n = size(A_list[1], 1)

    # Unfold horizontally: [A(1) | A(2) | ... | A(T)]
    Unfolded = hcat(A_list...)  # n × (n*T)

    # SVD: Unfolded = U * S * V'
    # V is (n*T) × rank, contains time-specific information
    U, S, V = svd(Unfolded)

    # Partition V into T blocks of n rows each
    X_vblock = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        start_idx = (t-1)*n + 1
        end_idx = t*n
        Vt = V[start_idx:end_idx, 1:d]
        # Scale by sqrt(singular values)
        X_vblock[t] = Vt * Diagonal(sqrt.(S[1:d]))
    end
    return X_vblock
end

# Method 3: Independent ASE + Procrustes chain
function ase_procrustes_embed(A_list, d)
    T = length(A_list)

    # Independent ASE at each timestep
    X_ase = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        U, S, _ = svd(A_list[t])
        X_ase[t] = U[:, 1:d] * Diagonal(sqrt.(S[1:d]))
    end

    # Procrustes chain alignment
    for t in 2:T
        F = svd(X_ase[t]' * X_ase[t-1])
        Q = F.U * F.Vt
        X_ase[t] = X_ase[t] * Q
    end
    return X_ase
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    rng = MersenneTwister(SEED)
    X0 = generate_initial_X(rng)
    X_true = simulate_trajectory(X0, T_TOTAL)
    P_perfect = [clamp.(X_true[t] * X_true[t]', 0.0, 1.0) for t in 1:T_TOTAL]
    A_noisy = generate_adjacencies(X_true, K_SAMPLES, rng)

    println("=" ^ 70)
    println("COMPARING EMBEDDING METHODS")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue trajectory curvature: ", round(curv_true, digits=2), "°")

    # Align each to true at t=1
    function align_to_true(X_est, X_true)
        F = svd(X_est[1]' * X_true[1])
        Q = F.U * F.Vt
        return [X_est[t] * Q for t in 1:length(X_est)]
    end

    # P reconstruction error
    function p_error(X_est, P_true)
        mean([norm(X_est[t] * X_est[t]' - P_true[t]) / norm(P_true[t]) for t in 1:length(X_est)])
    end

    # ========================================================================
    # Test on PERFECT P (no noise)
    # ========================================================================
    println("\n--- Perfect P (no noise) ---")

    X_duase = duase_embed(P_perfect, D_EMBED)
    X_vblock = vblock_embed(P_perfect, D_EMBED)
    X_ase = ase_procrustes_embed(P_perfect, D_EMBED)

    X_duase_al = align_to_true(X_duase, X_true)
    X_vblock_al = align_to_true(X_vblock, X_true)
    X_ase_al = align_to_true(X_ase, X_true)

    curv_duase = compute_curvature(X_duase_al)
    curv_vblock = compute_curvature(X_vblock_al)
    curv_ase = compute_curvature(X_ase_al)

    println("DUASE (Q(t) method):    ", round(curv_duase, digits=2), "° (", round(curv_duase/curv_true, digits=2), "× true)")
    println("V-block method:         ", round(curv_vblock, digits=2), "° (", round(curv_vblock/curv_true, digits=2), "× true)")
    println("ASE + Procrustes chain: ", round(curv_ase, digits=2), "° (", round(curv_ase/curv_true, digits=2), "× true)")

    println("\nP reconstruction error:")
    println("DUASE:   ", round(100*p_error(X_duase_al, P_perfect), digits=2), "%")
    println("V-block: ", round(100*p_error(X_vblock_al, P_perfect), digits=2), "%")
    println("ASE:     ", round(100*p_error(X_ase_al, P_perfect), digits=2), "%")

    # ========================================================================
    # Test on NOISY A_obs
    # ========================================================================
    println("\n--- Noisy A_obs (K=", K_SAMPLES, " samples) ---")

    X_duase_n = duase_embed(A_noisy, D_EMBED)
    X_vblock_n = vblock_embed(A_noisy, D_EMBED)
    X_ase_n = ase_procrustes_embed(A_noisy, D_EMBED)

    X_duase_n_al = align_to_true(X_duase_n, X_true)
    X_vblock_n_al = align_to_true(X_vblock_n, X_true)
    X_ase_n_al = align_to_true(X_ase_n, X_true)

    curv_duase_n = compute_curvature(X_duase_n_al)
    curv_vblock_n = compute_curvature(X_vblock_n_al)
    curv_ase_n = compute_curvature(X_ase_n_al)

    println("DUASE (Q(t) method):    ", round(curv_duase_n, digits=2), "° (", round(curv_duase_n/curv_true, digits=2), "× true)")
    println("V-block method:         ", round(curv_vblock_n, digits=2), "° (", round(curv_vblock_n/curv_true, digits=2), "× true)")
    println("ASE + Procrustes chain: ", round(curv_ase_n, digits=2), "° (", round(curv_ase_n/curv_true, digits=2), "× true)")

    println("\nP reconstruction error (vs perfect P):")
    println("DUASE:   ", round(100*p_error(X_duase_n_al, P_perfect), digits=2), "%")
    println("V-block: ", round(100*p_error(X_vblock_n_al, P_perfect), digits=2), "%")
    println("ASE:     ", round(100*p_error(X_ase_n_al, P_perfect), digits=2), "%")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    fig = Figure(size=(1600, 900))

    # Row 1: Perfect P
    ax1 = Axis(fig[1,1], title="True", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax2 = Axis(fig[1,2], title="DUASE (perfect P)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax3 = Axis(fig[1,3], title="V-block (perfect P)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax4 = Axis(fig[1,4], title="ASE+Procrustes (perfect P)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())

    # Row 2: Noisy
    ax5 = Axis(fig[2,1], title="True (reference)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax6 = Axis(fig[2,2], title="DUASE (K=$K_SAMPLES)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax7 = Axis(fig[2,3], title="V-block (K=$K_SAMPLES)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())
    ax8 = Axis(fig[2,4], title="ASE+Procrustes (K=$K_SAMPLES)", xlabel="X₁", ylabel="X₂", aspect=DataAspect())

    datasets_row1 = [(ax1, X_true), (ax2, X_duase_al), (ax3, X_vblock_al), (ax4, X_ase_al)]
    datasets_row2 = [(ax5, X_true), (ax6, X_duase_n_al), (ax7, X_vblock_n_al), (ax8, X_ase_n_al)]

    for (ax, X_data) in vcat(datasets_row1, datasets_row2)
        for type_idx in [TYPE_P, TYPE_Y, TYPE_R]
            nodes = findall(==(type_idx), NODE_TYPES)
            for i in nodes
                traj_x = [X_data[t][i, 1] for t in 1:T_TOTAL]
                traj_y = [X_data[t][i, 2] for t in 1:T_TOTAL]
                lines!(ax, traj_x, traj_y, color=(TYPE_COLORS[type_idx], 0.3), linewidth=1)
            end
            cx = [mean(X_data[t][nodes, 1]) for t in 1:T_TOTAL]
            cy = [mean(X_data[t][nodes, 2]) for t in 1:T_TOTAL]
            lines!(ax, cx, cy, color=TYPE_COLORS[type_idx], linewidth=3)
            scatter!(ax, [cx[1]], [cy[1]], color=TYPE_COLORS[type_idx], markersize=12, marker=:circle)
            scatter!(ax, [cx[end]], [cy[end]], color=TYPE_COLORS[type_idx], markersize=12, marker=:star5)
        end
        xlims!(ax, 0, 1)
        ylims!(ax, 0, 1)
    end

    Label(fig[0, :], "Embedding Methods Comparison (Row 1: Perfect P, Row 2: Noisy)", fontsize=20)

    save(joinpath(dirname(@__DIR__), "results", "embedding_methods_comparison.png"), fig, px_per_unit=2)
    println("\nSaved: results/embedding_methods_comparison.png")
end

main()
