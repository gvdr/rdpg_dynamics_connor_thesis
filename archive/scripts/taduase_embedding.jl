#!/usr/bin/env -S julia --project
"""
Type-Augmented DUASE (TA-DUASE): Allow type-specific rotations to recover curvature.

Key insight: DUASE constrains X(t) = G · √Q(t), meaning all nodes undergo the SAME
transformation. TA-DUASE relaxes this:

    X̂(t)[i,:] = G[i,:] · √Q(t) · Ω_type(i)(t)

where Ω_τ(t) ∈ O(d) is a type-specific rotation.

Properties:
- Within-type P preserved exactly (Ω_τ Ω_τ' = I)
- Cross-type P depends on relative rotation Ω_τ Ω_σ'
- Different types can now move in different directions
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
# Setup (same ecological model)
# ============================================================================

const N_PRED, N_PREY, N_RES = 12, 15, 10
const N_TOTAL = N_PRED + N_PREY + N_RES
const D_EMBED = 2
const T_TOTAL = 25
const SEED = 42

const TYPE_P, TYPE_Y, TYPE_R = 1, 2, 3
const NODE_TYPES = vcat(fill(TYPE_P, N_PRED), fill(TYPE_Y, N_PREY), fill(TYPE_R, N_RES))
const NUM_TYPES = 3
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

function compute_type_curvature(X_traj, type_idx)
    T = length(X_traj)
    nodes = findall(==(type_idx), NODE_TYPES)
    velocities = [X_traj[t+1] - X_traj[t] for t in 1:(T-1)]
    angles = Float64[]
    for t in 1:(T-2), i in nodes
        v1, v2 = velocities[t][i, :], velocities[t+1][i, :]
        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            push!(angles, acos(clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)) * 180 / π)
        end
    end
    isempty(angles) ? 0.0 : mean(angles)
end

# ============================================================================
# Standard DUASE
# ============================================================================

function duase_embed(A_list, d)
    T = length(A_list)
    n = size(A_list[1], 1)

    Unfolded = hcat(A_list...)
    U, S, V = svd(Unfolded)
    G = U[:, 1:d]

    X_duase = Vector{Matrix{Float64}}(undef, T)
    sqrt_Q_list = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        Qt = G' * A_list[t] * G
        Qt_sym = (Qt + Qt') / 2
        eig = eigen(Symmetric(Qt_sym))
        sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
        sqrt_Q_list[t] = sqrt_Q
        X_duase[t] = G * sqrt_Q
    end

    return G, sqrt_Q_list, X_duase
end

# ============================================================================
# Type-Augmented DUASE (TA-DUASE)
# ============================================================================

"""
Build rotation matrix from angle (2D case).
"""
function rotation_matrix(θ::Real)
    c, s = cos(θ), sin(θ)
    [c -s; s c]
end

"""
Pack type-specific angles into vector for optimization.
θ has shape (NUM_TYPES, T)
"""
function pack_angles(θ_matrix)
    vec(θ_matrix)
end

function unpack_angles(θ_vec, num_types, T)
    reshape(θ_vec, num_types, T)
end

"""
Apply type-specific rotations to DUASE embedding.
"""
function apply_type_rotations(G, sqrt_Q_list, θ_matrix)
    T = length(sqrt_Q_list)
    n = size(G, 1)
    d = size(G, 2)

    X_taduase = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        X_t = zeros(n, d)
        for i in 1:n
            τ = NODE_TYPES[i]
            Ω_τ = rotation_matrix(θ_matrix[τ, t])
            X_t[i, :] = G[i, :] * sqrt_Q_list[t] * Ω_τ
        end
        X_taduase[t] = X_t
    end

    return X_taduase
end

"""
Loss function for TA-DUASE optimization.

Balances:
1. Cross-type P reconstruction
2. Smoothness of rotation angles over time
3. Smoothness of resulting trajectories
"""
function taduase_loss(θ_vec, G, sqrt_Q_list, P_list, λ_smooth, λ_traj)
    θ_matrix = unpack_angles(θ_vec, NUM_TYPES, length(P_list))
    X_taduase = apply_type_rotations(G, sqrt_Q_list, θ_matrix)

    T = length(P_list)

    # P reconstruction loss (all pairs)
    loss_P = 0.0
    for t in 1:T
        P_hat = X_taduase[t] * X_taduase[t]'
        loss_P += sum((P_hat - P_list[t]).^2)
    end

    # Angle smoothness (penalize rapid rotation changes)
    loss_angle_smooth = 0.0
    for τ in 1:NUM_TYPES
        for t in 1:(T-1)
            dθ = θ_matrix[τ, t+1] - θ_matrix[τ, t]
            loss_angle_smooth += dθ^2
        end
    end

    # Trajectory smoothness (second derivative / acceleration)
    loss_traj_smooth = 0.0
    for t in 2:(T-1)
        accel = X_taduase[t+1] - 2*X_taduase[t] + X_taduase[t-1]
        loss_traj_smooth += sum(accel.^2)
    end

    return loss_P + λ_smooth * loss_angle_smooth + λ_traj * loss_traj_smooth
end

"""
Fit TA-DUASE by optimizing type-specific rotations.
"""
function fit_taduase(P_list, d; λ_smooth=10.0, λ_traj=1.0, max_iter=1000, verbose=true)
    T = length(P_list)

    # Start with standard DUASE
    G, sqrt_Q_list, X_duase = duase_embed(P_list, d)

    # Initialize angles to zero (start from DUASE)
    θ0 = zeros(NUM_TYPES * T)

    f(θ) = taduase_loss(θ, G, sqrt_Q_list, P_list, λ_smooth, λ_traj)

    if verbose
        println("  Initial loss: ", round(f(θ0), digits=2))
    end

    result = optimize(f, θ0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose, show_every=100))

    θ_opt = Optim.minimizer(result)
    θ_matrix = unpack_angles(θ_opt, NUM_TYPES, T)
    X_taduase = apply_type_rotations(G, sqrt_Q_list, θ_matrix)

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_taduase, θ_matrix, G, sqrt_Q_list
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
    println("TYPE-AUGMENTED DUASE (TA-DUASE)")
    println("=" ^ 70)

    curv_true = compute_curvature(X_true)
    println("\nTrue trajectory curvature: ", round(curv_true, digits=2), "°")
    println("\nPer-type curvature (true):")
    for τ in [TYPE_P, TYPE_Y, TYPE_R]
        println("  ", TYPE_NAMES[τ], ": ", round(compute_type_curvature(X_true, τ), digits=2), "°")
    end

    # Standard DUASE
    println("\n--- Standard DUASE ---")
    G, sqrt_Q_list, X_duase = duase_embed(P_list, D_EMBED)

    # Align to true
    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T_TOTAL]
    end

    X_duase_al = align_to_true(X_duase)
    curv_duase = compute_curvature(X_duase_al)
    println("DUASE curvature: ", round(curv_duase, digits=2), "° (", round(curv_duase/curv_true, digits=2), "× true)")

    # TA-DUASE with different parameters
    println("\n--- TA-DUASE Optimization ---")

    configs = [
        (λ_smooth=1.0, λ_traj=0.0, name="λ_s=1, λ_t=0"),
        (λ_smooth=10.0, λ_traj=0.0, name="λ_s=10, λ_t=0"),
        (λ_smooth=10.0, λ_traj=1.0, name="λ_s=10, λ_t=1"),
        (λ_smooth=1.0, λ_traj=10.0, name="λ_s=1, λ_t=10"),
    ]

    results = Dict()

    for cfg in configs
        println("\n--- ", cfg.name, " ---")
        X_ta, θ_matrix, _, _ = fit_taduase(P_list, D_EMBED;
                                           λ_smooth=cfg.λ_smooth, λ_traj=cfg.λ_traj,
                                           max_iter=500, verbose=false)
        X_ta_al = align_to_true(X_ta)
        results[cfg.name] = (X=X_ta_al, θ=θ_matrix)

        curv = compute_curvature(X_ta_al)
        println("  Curvature: ", round(curv, digits=2), "° (", round(curv/curv_true, digits=2), "× true)")

        # P reconstruction error
        P_err = mean([norm(X_ta_al[t] * X_ta_al[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
        println("  P error: ", round(100*P_err, digits=1), "%")

        # Max angle deviation
        max_angle = maximum(abs.(θ_matrix)) * 180 / π
        println("  Max rotation: ", round(max_angle, digits=1), "°")
    end

    # Metrics summary
    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T_TOTAL])
    end

    function pos_error(X_est)
        mean([norm(X_est[t] - X_true[t]) / norm(X_true[t]) for t in 1:T_TOTAL])
    end

    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("\n", rpad("Method", 20), rpad("Curvature", 12), rpad("P-error", 12), "Pos-error")
    println("-" ^ 56)
    println(rpad("True", 20), rpad(string(round(curv_true, digits=2)) * "°", 12), "-", " " ^ 10, "-")
    println(rpad("DUASE", 20),
            rpad(string(round(curv_duase, digits=2)) * "°", 12),
            rpad(string(round(100*p_error(X_duase_al), digits=1)) * "%", 12),
            string(round(100*pos_error(X_duase_al), digits=1)) * "%")

    for cfg in configs
        X = results[cfg.name].X
        println(rpad(cfg.name, 20),
                rpad(string(round(compute_curvature(X), digits=2)) * "°", 12),
                rpad(string(round(100*p_error(X), digits=1)) * "%", 12),
                string(round(100*pos_error(X), digits=1)) * "%")
    end

    # Visualization
    fig = Figure(size=(1800, 800))

    # Row 1: Trajectories
    best_cfg = "λ_s=10, λ_t=1"
    datasets = [
        (X_true, "True"),
        (X_duase_al, "DUASE"),
        (results["λ_s=1, λ_t=0"].X, "TA-DUASE (λs=1)"),
        (results[best_cfg].X, "TA-DUASE (λs=10,λt=1)"),
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
        xlims!(ax, -0.2, 1.2)
        ylims!(ax, -0.2, 1.2)
    end

    # Row 2: Type-specific rotation angles over time
    ax_angles = Axis(fig[2, 1:2], title="Type-Specific Rotation Angles (TA-DUASE)",
                     xlabel="Time", ylabel="Rotation (degrees)")
    θ_best = results[best_cfg].θ
    for τ in [TYPE_P, TYPE_Y, TYPE_R]
        lines!(ax_angles, 1:T_TOTAL, θ_best[τ, :] .* (180/π),
               color=TYPE_COLORS[τ], linewidth=2, label=TYPE_NAMES[τ])
    end
    axislegend(ax_angles, position=:lt)

    # Row 2: Curvature comparison
    ax_curv = Axis(fig[2, 3:4], title="Per-Type Curvature Comparison",
                   xlabel="Type", ylabel="Curvature (degrees)")

    types = ["Predator", "Prey", "Resource"]
    curv_true_by_type = [compute_type_curvature(X_true, τ) for τ in [TYPE_P, TYPE_Y, TYPE_R]]
    curv_duase_by_type = [compute_type_curvature(X_duase_al, τ) for τ in [TYPE_P, TYPE_Y, TYPE_R]]
    curv_ta_by_type = [compute_type_curvature(results[best_cfg].X, τ) for τ in [TYPE_P, TYPE_Y, TYPE_R]]

    barplot!(ax_curv, [1, 2, 3] .- 0.25, curv_true_by_type, width=0.2, color=:black, label="True")
    barplot!(ax_curv, [1, 2, 3], curv_duase_by_type, width=0.2, color=:gray, label="DUASE")
    barplot!(ax_curv, [1, 2, 3] .+ 0.25, curv_ta_by_type, width=0.2, color=:steelblue, label="TA-DUASE")
    ax_curv.xticks = ([1, 2, 3], types)
    axislegend(ax_curv, position=:rt)

    save(joinpath(dirname(@__DIR__), "results", "taduase_embedding.png"), fig, px_per_unit=2)
    println("\nSaved: results/taduase_embedding.png")

    # Detailed trajectory shape analysis
    println("\n" * "=" ^ 70)
    println("TRAJECTORY SHAPE ANALYSIS")
    println("=" ^ 70)

    # Compare centroid trajectories
    println("\nCentroid trajectory comparison (predator type):")
    pred_nodes = findall(==(TYPE_P), NODE_TYPES)

    cx_true = [mean(X_true[t][pred_nodes, 1]) for t in 1:T_TOTAL]
    cy_true = [mean(X_true[t][pred_nodes, 2]) for t in 1:T_TOTAL]
    cx_duase = [mean(X_duase_al[t][pred_nodes, 1]) for t in 1:T_TOTAL]
    cy_duase = [mean(X_duase_al[t][pred_nodes, 2]) for t in 1:T_TOTAL]
    cx_ta = [mean(results[best_cfg].X[t][pred_nodes, 1]) for t in 1:T_TOTAL]
    cy_ta = [mean(results[best_cfg].X[t][pred_nodes, 2]) for t in 1:T_TOTAL]

    for t in [1, 5, 10, 15, 20, 25]
        println("  t=", t, ":")
        println("    True:    (", round(cx_true[t], digits=3), ", ", round(cy_true[t], digits=3), ")")
        println("    DUASE:   (", round(cx_duase[t], digits=3), ", ", round(cy_duase[t], digits=3), ")")
        println("    TA-DUASE: (", round(cx_ta[t], digits=3), ", ", round(cy_ta[t], digits=3), ")")
    end
end

main()
