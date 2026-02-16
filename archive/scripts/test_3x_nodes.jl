#!/usr/bin/env -S julia --project
"""
Test: Does 3× nodes per type improve trajectory curvature estimation?

Compare trajectory curvature between:
- Original: 12/15/10 = 37 nodes
- 3× scale: 36/45/30 = 111 nodes
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using CairoMakie
using Statistics

const CM = CairoMakie

# ============================================================================
# Configuration - 3× nodes
# ============================================================================

const N_PRED = 12    # Original
const N_PREY = 15    # Original
const N_RES = 10     # Original
const N_TOTAL = N_PRED + N_PREY + N_RES  # 37 total (original)

const D_EMBED = 2
const T_TOTAL = 25
const K_SAMPLES = 50  # Edge samples per timestep
const SEED = 42

# Type labels
const TYPE_P = 1
const TYPE_Y = 2
const TYPE_R = 3

const NODE_TYPES = vcat(
    fill(TYPE_P, N_PRED),
    fill(TYPE_Y, N_PREY),
    fill(TYPE_R, N_RES)
)

# Type colors
const TYPE_COLORS = Dict(
    TYPE_P => :red,
    TYPE_Y => :blue,
    TYPE_R => :green
)

# ============================================================================
# Dynamics (same as Example 4)
# ============================================================================

const KNOWN_SELF_RATES = Dict(
    TYPE_P => -0.002,
    TYPE_Y => -0.001,
    TYPE_R =>  0.000
)

const HOLLING_ALPHA = 0.025
const HOLLING_BETA = 2.0

function κ_true(type_i::Int, type_j::Int, p::Real)
    if type_i == TYPE_P && type_j == TYPE_P
        return -0.004
    elseif type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)
    elseif type_i == TYPE_P && type_j == TYPE_R
        return 0.0
    elseif type_i == TYPE_Y && type_j == TYPE_P
        return -0.02 * p
    elseif type_i == TYPE_Y && type_j == TYPE_Y
        return 0.003
    elseif type_i == TYPE_Y && type_j == TYPE_R
        return 0.012 * p
    elseif type_i == TYPE_R && type_j == TYPE_P
        return 0.0
    elseif type_i == TYPE_R && type_j == TYPE_Y
        return -0.006 * p
    elseif type_i == TYPE_R && type_j == TYPE_R
        return 0.005
    else
        return 0.0
    end
end

function compute_N_matrix(X::Matrix{Float64})
    n = size(X, 1)
    P = X * X'
    P = clamp.(P, 0.0, 1.0)

    N = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i == j
                N[i,i] = KNOWN_SELF_RATES[NODE_TYPES[i]]
            else
                N[i,j] = κ_true(NODE_TYPES[i], NODE_TYPES[j], P[i,j])
            end
        end
    end
    return N
end

function true_dynamics!(dX, X, p, t)
    N = compute_N_matrix(X)
    mul!(dX, N, X)
end

# ============================================================================
# Initial conditions
# ============================================================================

function generate_initial_X(rng::AbstractRNG)
    X = zeros(N_TOTAL, D_EMBED)

    # Type centers (same as Example 4)
    center_P = [0.35, 0.70]
    center_Y = [0.70, 0.50]
    center_R = [0.55, 0.25]

    spread = 0.08

    for i in 1:N_PRED
        X[i, :] = center_P .+ spread .* randn(rng, 2)
    end
    for i in (N_PRED+1):(N_PRED+N_PREY)
        X[i, :] = center_Y .+ spread .* randn(rng, 2)
    end
    for i in (N_PRED+N_PREY+1):N_TOTAL
        X[i, :] = center_R .+ spread .* randn(rng, 2)
    end

    X = clamp.(X, 0.05, 0.95)
    return X
end

# ============================================================================
# Simulation and embedding
# ============================================================================

function simulate_true_trajectory(X0::Matrix{Float64}, T::Int)
    prob = ODEProblem(true_dynamics!, X0, (0.0, Float64(T-1)))
    sol = solve(prob, Tsit5(), saveat=0.0:1.0:Float64(T-1), abstol=1e-8, reltol=1e-8)
    return [sol.u[t] for t in 1:T]
end

function generate_adjacencies(X_true::Vector{Matrix{Float64}}, K::Int, rng::AbstractRNG)
    T = length(X_true)
    n = size(X_true[1], 1)
    A_obs = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        P = X_true[t] * X_true[t]'
        P = clamp.(P, 0.0, 1.0)

        A_sum = zeros(n, n)
        for _ in 1:K
            A = Float64.(rand(rng, n, n) .< P)
            A = (A + A') / 2
            # KEEP self-loops! DUASE needs diagonal for proper eigenvalue estimation
            A_sum .+= A
        end
        A_obs[t] = A_sum / K
    end
    return A_obs
end

# ============================================================================
# Curvature metric
# ============================================================================

function compute_trajectory_curvature(X_traj::Vector{Matrix{Float64}})
    T = length(X_traj)
    n = size(X_traj[1], 1)

    # Velocity vectors
    velocities = [X_traj[t+1] - X_traj[t] for t in 1:(T-1)]

    # Angle between consecutive velocities
    angles = Float64[]
    for t in 1:(T-2)
        for i in 1:n
            v1 = velocities[t][i, :]
            v2 = velocities[t+1][i, :]

            norm1 = norm(v1)
            norm2 = norm(v2)

            if norm1 > 1e-10 && norm2 > 1e-10
                cos_angle = clamp(dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle = acos(cos_angle) * 180 / π
                push!(angles, angle)
            end
        end
    end

    return mean(angles), std(angles)
end

# ============================================================================
# Main comparison
# ============================================================================

function main()
    rng = MersenneTwister(SEED)

    println("=" ^ 60)
    println("Testing 3× Nodes: Effect on Trajectory Curvature")
    println("=" ^ 60)
    println("\nConfiguration:")
    println("  Nodes: ", N_PRED, " pred + ", N_PREY, " prey + ", N_RES, " res = ", N_TOTAL, " total")
    println("  Dimension: ", D_EMBED)
    println("  Timesteps: ", T_TOTAL)
    println("  Edge samples: ", K_SAMPLES)

    # Generate true trajectory
    println("\n1. Generating true trajectory...")
    X0 = generate_initial_X(rng)
    X_true = simulate_true_trajectory(X0, T_TOTAL)

    # Sample adjacencies
    println("2. Sampling adjacencies (K=", K_SAMPLES, ")...")
    A_obs = generate_adjacencies(X_true, K_SAMPLES, rng)

    # Embed with DUASE
    println("3. Embedding with DUASE...")
    G, X_raw = duase_embedding(A_obs, D_EMBED)
    # NOTE: duase_embedding returns (G, X_series) where X_series already has sqrt(Q) applied!

    # Sign correction
    sign_flips = ones(D_EMBED)
    for j in 1:D_EMBED
        if sum(X_raw[1][:, j] .< 0) > N_TOTAL / 2
            sign_flips[j] = -1.0
        end
    end
    X_est = [X_raw[t] .* sign_flips' for t in 1:T_TOTAL]

    # Compute curvature
    println("\n4. Computing trajectory curvature...")
    curv_true_mean, curv_true_std = compute_trajectory_curvature(X_true)
    curv_est_mean, curv_est_std = compute_trajectory_curvature(X_est)

    println("\n   True trajectories:")
    println("     Mean angle: ", round(curv_true_mean, digits=2), "° ± ", round(curv_true_std, digits=2), "°")
    println("   DUASE estimated:")
    println("     Mean angle: ", round(curv_est_mean, digits=2), "° ± ", round(curv_est_std, digits=2), "°")

    # Compute P-error
    P_errors = Float64[]
    for t in 1:T_TOTAL
        P_true = X_true[t] * X_true[t]'
        P_est = X_est[t] * X_est[t]'
        push!(P_errors, norm(P_true - P_est) / norm(P_true))
    end
    println("\n   Mean P-error: ", round(100 * mean(P_errors), digits=2), "%")

    # ========================================================================
    # Visualization
    # ========================================================================
    println("\n5. Creating visualization...")

    fig = Figure(size=(1400, 600))

    # Left: True trajectories
    ax1 = Axis(fig[1,1],
        title="True Trajectories (n=" * string(N_TOTAL) * ")",
        xlabel="X₁", ylabel="X₂",
        aspect=DataAspect())

    # Plot one representative node per type
    rep_nodes = [1, N_PRED+1, N_PRED+N_PREY+1]

    for (idx, node) in enumerate(rep_nodes)
        traj_x = [X_true[t][node, 1] for t in 1:T_TOTAL]
        traj_y = [X_true[t][node, 2] for t in 1:T_TOTAL]
        type_idx = NODE_TYPES[node]

        lines!(ax1, traj_x, traj_y, color=TYPE_COLORS[type_idx], linewidth=2)
        scatter!(ax1, [traj_x[1]], [traj_y[1]], color=TYPE_COLORS[type_idx],
            markersize=15, marker=:circle)
        scatter!(ax1, [traj_x[end]], [traj_y[end]], color=TYPE_COLORS[type_idx],
            markersize=15, marker=:star5)
    end

    # Also plot type centroids
    for t_idx in [1, T_TOTAL÷2, T_TOTAL]
        for type_idx in 1:3
            nodes = findall(==(type_idx), NODE_TYPES)
            cx = mean(X_true[t_idx][nodes, 1])
            cy = mean(X_true[t_idx][nodes, 2])
            scatter!(ax1, [cx], [cy], color=TYPE_COLORS[type_idx],
                markersize=25, marker=:diamond, strokewidth=2, strokecolor=:black)
        end
    end

    xlims!(ax1, 0, 1)
    ylims!(ax1, 0, 1)

    # Middle: Estimated trajectories
    ax2 = Axis(fig[1,2],
        title="DUASE Estimated (n=" * string(N_TOTAL) * ")",
        xlabel="X₁", ylabel="X₂",
        aspect=DataAspect())

    for (idx, node) in enumerate(rep_nodes)
        traj_x = [X_est[t][node, 1] for t in 1:T_TOTAL]
        traj_y = [X_est[t][node, 2] for t in 1:T_TOTAL]
        type_idx = NODE_TYPES[node]

        lines!(ax2, traj_x, traj_y, color=TYPE_COLORS[type_idx], linewidth=2)
        scatter!(ax2, [traj_x[1]], [traj_y[1]], color=TYPE_COLORS[type_idx],
            markersize=15, marker=:circle)
        scatter!(ax2, [traj_x[end]], [traj_y[end]], color=TYPE_COLORS[type_idx],
            markersize=15, marker=:star5)
    end

    # Type centroids for estimated
    for t_idx in [1, T_TOTAL÷2, T_TOTAL]
        for type_idx in 1:3
            nodes = findall(==(type_idx), NODE_TYPES)
            cx = mean(X_est[t_idx][nodes, 1])
            cy = mean(X_est[t_idx][nodes, 2])
            scatter!(ax2, [cx], [cy], color=TYPE_COLORS[type_idx],
                markersize=25, marker=:diamond, strokewidth=2, strokecolor=:black)
        end
    end

    xlims!(ax2, 0, 1)
    ylims!(ax2, 0, 1)

    # Right: Type centroid trajectories (averaged)
    ax3 = Axis(fig[1,3],
        title="Type Centroid Trajectories",
        xlabel="X₁", ylabel="X₂",
        aspect=DataAspect())

    type_names = ["Predator", "Prey", "Resource"]
    for type_idx in 1:3
        nodes = findall(==(type_idx), NODE_TYPES)

        # True centroid trajectory
        traj_true_x = [mean(X_true[t][nodes, 1]) for t in 1:T_TOTAL]
        traj_true_y = [mean(X_true[t][nodes, 2]) for t in 1:T_TOTAL]
        lines!(ax3, traj_true_x, traj_true_y, color=TYPE_COLORS[type_idx],
            linewidth=3, linestyle=:solid, label=type_names[type_idx] * " (true)")

        # Estimated centroid trajectory
        traj_est_x = [mean(X_est[t][nodes, 1]) for t in 1:T_TOTAL]
        traj_est_y = [mean(X_est[t][nodes, 2]) for t in 1:T_TOTAL]
        lines!(ax3, traj_est_x, traj_est_y, color=TYPE_COLORS[type_idx],
            linewidth=2, linestyle=:dash, label=type_names[type_idx] * " (est)")
    end

    xlims!(ax3, 0, 1)
    ylims!(ax3, 0, 1)
    axislegend(ax3, position=:lb)

    save(joinpath(dirname(@__DIR__), "results", "test_3x_nodes.png"), fig, px_per_unit=2)
    println("   Saved: results/test_3x_nodes.png")

    # ========================================================================
    # Also compute centroid-level curvature
    # ========================================================================
    println("\n6. Centroid-level curvature analysis...")

    # Get centroid trajectories
    centroid_true = Vector{Matrix{Float64}}(undef, T_TOTAL)
    centroid_est = Vector{Matrix{Float64}}(undef, T_TOTAL)

    for t in 1:T_TOTAL
        centroid_true[t] = zeros(3, 2)
        centroid_est[t] = zeros(3, 2)
        for type_idx in 1:3
            nodes = findall(==(type_idx), NODE_TYPES)
            centroid_true[t][type_idx, :] = mean(X_true[t][nodes, :], dims=1)
            centroid_est[t][type_idx, :] = mean(X_est[t][nodes, :], dims=1)
        end
    end

    curv_cent_true_mean, curv_cent_true_std = compute_trajectory_curvature(centroid_true)
    curv_cent_est_mean, curv_cent_est_std = compute_trajectory_curvature(centroid_est)

    println("\n   Type centroid trajectories:")
    println("     True: ", round(curv_cent_true_mean, digits=2), "° ± ", round(curv_cent_true_std, digits=2), "°")
    println("     Estimated: ", round(curv_cent_est_mean, digits=2), "° ± ", round(curv_cent_est_std, digits=2), "°")

    println("\n" * "=" ^ 60)
    println("Done!")
    println("=" ^ 60)
end

main()
