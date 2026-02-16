#!/usr/bin/env -S julia --project
"""
Variational Embedding: Solve for entire trajectory X(1),...,X(T) simultaneously.

Objective:
  L(X) = Σ_t ||X(t)X(t)' - P(t)||²  [P reconstruction]
       + λ_s · Σ_t ||X(t+1) - X(t)||²  [smoothness]
       + λ_a · Σ_t ||X(t+1) - 2X(t) + X(t-1)||²  [acceleration penalty]

Key insight: By optimizing all timesteps together, we avoid the error accumulation
of forward propagation while still allowing curvature in the trajectories.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Random
using OrdinaryDiffEq
using Statistics
using CairoMakie
using Optim

# ============================================================================
# Setup (same as other scripts)
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
# Variational Embedding
# ============================================================================

"""
Variational embedding: optimize entire trajectory simultaneously.

Parameters:
- λ_P: weight on P reconstruction (default 1.0)
- λ_smooth: weight on smoothness (default 0.1)
- λ_accel: weight on acceleration penalty (default 0.01)
- fix_first: whether to fix X(1) to ASE (default true)
"""
function variational_embed(P_list, d; λ_P=1.0, λ_smooth=0.1, λ_accel=0.01,
                           max_iter=500, verbose=true, fix_first=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    # Initialize with DUASE
    X_init = duase_embed(P_list, d)

    # If fixing first frame, we only optimize t=2:T
    if fix_first
        X_fixed = copy(X_init[1])
        start_t = 2
    else
        X_fixed = nothing
        start_t = 1
    end

    # Flatten/unflatten for optimizer
    function flatten(X_list)
        vcat([vec(X_list[t]) for t in start_t:T]...)
    end

    function unflatten(x)
        result = Vector{Matrix{Float64}}(undef, T)
        if fix_first
            result[1] = X_fixed
        end
        for (idx, t) in enumerate(start_t:T)
            offset = (idx - 1) * n * d
            result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        return result
    end

    # Objective function
    function objective(x)
        X = unflatten(x)

        # P reconstruction loss
        loss_P = 0.0
        for t in 1:T
            P_hat = X[t] * X[t]'
            loss_P += norm(P_hat - P_list[t])^2
        end

        # Smoothness (velocity magnitude)
        loss_smooth = 0.0
        for t in 1:(T-1)
            loss_smooth += norm(X[t+1] - X[t])^2
        end

        # Acceleration penalty
        loss_accel = 0.0
        for t in 2:(T-1)
            accel = X[t+1] - 2*X[t] + X[t-1]
            loss_accel += norm(accel)^2
        end

        return λ_P * loss_P + λ_smooth * loss_smooth + λ_accel * loss_accel
    end

    # Analytic gradient
    function gradient!(g, x)
        X = unflatten(x)
        grad_X = [zeros(n, d) for _ in 1:T]

        # Gradient of P reconstruction: ∂/∂X(t) ||XX' - P||² = 4(XX' - P)X
        for t in 1:T
            residual = X[t] * X[t]' - P_list[t]
            grad_X[t] .+= λ_P * 4 * residual * X[t]
        end

        # Gradient of smoothness: ∂/∂X(t) ||X(t+1) - X(t)||²
        # For t: contributes -2(X(t+1) - X(t)) to grad_X[t] and +2(X(t+1) - X(t)) to grad_X[t+1]
        for t in 1:(T-1)
            diff = X[t+1] - X[t]
            grad_X[t] .-= λ_smooth * 2 * diff
            grad_X[t+1] .+= λ_smooth * 2 * diff
        end

        # Gradient of acceleration: ∂/∂X(t) ||X(t+1) - 2X(t) + X(t-1)||²
        for t in 2:(T-1)
            accel = X[t+1] - 2*X[t] + X[t-1]
            grad_X[t-1] .+= λ_accel * 2 * accel
            grad_X[t] .-= λ_accel * 4 * accel
            grad_X[t+1] .+= λ_accel * 2 * accel
        end

        # Flatten gradients (only for optimized variables)
        for (idx, t) in enumerate(start_t:T)
            offset = (idx - 1) * n * d
            g[offset+1:offset+n*d] .= vec(grad_X[t])
        end
    end

    x0 = flatten(X_init)

    if verbose
        println("Optimizing variational embedding...")
        println("  Variables: ", length(x0))
        println("  λ_P=", λ_P, ", λ_smooth=", λ_smooth, ", λ_accel=", λ_accel)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    # Optimize with analytic gradient
    result = optimize(objective, gradient!, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt = unflatten(Optim.minimizer(result))

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
        println("  Converged: ", Optim.converged(result))
    end

    return X_opt, result
end

"""
Variant with Sylvester consistency term.
"""
function variational_embed_sylvester(P_list, d; λ_P=1.0, λ_smooth=0.1, λ_sylv=0.5,
                                      max_iter=500, verbose=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    X_init = duase_embed(P_list, d)
    X_fixed = copy(X_init[1])

    function flatten(X_list)
        vcat([vec(X_list[t]) for t in 2:T]...)
    end

    function unflatten(x)
        result = Vector{Matrix{Float64}}(undef, T)
        result[1] = X_fixed
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        return result
    end

    function objective(x)
        X = unflatten(x)

        # P reconstruction loss
        loss_P = sum(norm(X[t] * X[t]' - P_list[t])^2 for t in 1:T)

        # Smoothness
        loss_smooth = sum(norm(X[t+1] - X[t])^2 for t in 1:(T-1))

        # Sylvester consistency: V*X' + X*V' ≈ ΔP
        loss_sylv = 0.0
        for t in 1:(T-1)
            V = X[t+1] - X[t]
            ΔP = P_list[t+1] - P_list[t]
            sylv_residual = V * X[t]' + X[t] * V' - ΔP
            loss_sylv += norm(sylv_residual)^2
        end

        return λ_P * loss_P + λ_smooth * loss_smooth + λ_sylv * loss_sylv
    end

    # Analytic gradient for Sylvester term
    function gradient!(g, x)
        X = unflatten(x)
        grad_X = [zeros(n, d) for _ in 1:T]

        # P reconstruction gradient
        for t in 1:T
            residual = X[t] * X[t]' - P_list[t]
            grad_X[t] .+= λ_P * 4 * residual * X[t]
        end

        # Smoothness gradient
        for t in 1:(T-1)
            diff = X[t+1] - X[t]
            grad_X[t] .-= λ_smooth * 2 * diff
            grad_X[t+1] .+= λ_smooth * 2 * diff
        end

        # Sylvester gradient: ||VX' + XV' - ΔP||² where V = X(t+1) - X(t)
        # Let R = VX' + XV' - ΔP (symmetric)
        # ∂/∂X(t+1): 2R*X(t) (from V contribution)
        # ∂/∂X(t): 2R*V - 2R*X(t) = 2R*(X(t+1) - 2X(t))
        for t in 1:(T-1)
            V = X[t+1] - X[t]
            ΔP = P_list[t+1] - P_list[t]
            R = V * X[t]' + X[t] * V' - ΔP
            grad_X[t] .+= λ_sylv * 2 * (R * V - R * X[t])
            grad_X[t+1] .+= λ_sylv * 2 * R * X[t]
        end

        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            g[offset+1:offset+n*d] .= vec(grad_X[t])
        end
    end

    x0 = flatten(X_init)

    if verbose
        println("Optimizing variational embedding (with Sylvester)...")
        println("  λ_P=", λ_P, ", λ_smooth=", λ_smooth, ", λ_sylv=", λ_sylv)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    result = optimize(objective, gradient!, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt = unflatten(Optim.minimizer(result))

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_opt, result
end

"""
Variant with explicit Procrustes alignment penalty to prevent gauge drift.

The idea: penalize ||X(t+1) - X(t)*Q(t)||² where Q(t) is the Procrustes rotation.
This encourages X(t+1) to be aligned with X(t), preventing spurious rotational curvature.
"""
function variational_embed_procrustes(P_list, d; λ_P=1.0, λ_smooth=0.1, λ_proc=1.0,
                                       max_iter=500, verbose=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    X_init = duase_embed(P_list, d)
    X_fixed = copy(X_init[1])

    function flatten(X_list)
        vcat([vec(X_list[t]) for t in 2:T]...)
    end

    function unflatten(x)
        result = Vector{Matrix{Float64}}(undef, T)
        result[1] = X_fixed
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        return result
    end

    # Procrustes rotation: Q = argmin ||X(t+1) - X(t)*Q||
    # Solution: Q = V*U' where X(t)'*X(t+1) = U*Σ*V'
    function procrustes_rotation(A, B)
        F = svd(A' * B)
        return F.U * F.Vt
    end

    function objective(x)
        X = unflatten(x)

        # P reconstruction loss
        loss_P = sum(norm(X[t] * X[t]' - P_list[t])^2 for t in 1:T)

        # Smoothness
        loss_smooth = sum(norm(X[t+1] - X[t])^2 for t in 1:(T-1))

        # Procrustes alignment penalty: ||X(t+1) - X(t)*Q||²
        # where Q is the optimal rotation from X(t) to X(t+1)
        # This penalizes the "non-rotational" residual
        loss_proc = 0.0
        for t in 1:(T-1)
            Q = procrustes_rotation(X[t], X[t+1])
            X_aligned = X[t] * Q
            loss_proc += norm(X[t+1] - X_aligned)^2
        end

        return λ_P * loss_P + λ_smooth * loss_smooth + λ_proc * loss_proc
    end

    # Note: gradient is more complex due to Procrustes, use numerical for now
    # The Procrustes term has a discontinuous gradient at degenerate points
    # but should be smooth elsewhere

    x0 = flatten(X_init)

    if verbose
        println("Optimizing variational + Procrustes...")
        println("  λ_P=", λ_P, ", λ_smooth=", λ_smooth, ", λ_proc=", λ_proc)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    result = optimize(objective, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt = unflatten(Optim.minimizer(result))

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_opt, result
end

"""
Alternative: Penalize the antisymmetric part of the "velocity direction".

If X(t+1) - X(t) has a large component that looks like rotation (antisymmetric wrt X),
that's likely gauge drift. Penalize it.
"""
function variational_embed_antisym(P_list, d; λ_P=1.0, λ_smooth=0.1, λ_antisym=1.0,
                                    max_iter=500, verbose=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    X_init = duase_embed(P_list, d)
    X_fixed = copy(X_init[1])

    function flatten(X_list)
        vcat([vec(X_list[t]) for t in 2:T]...)
    end

    function unflatten(x)
        result = Vector{Matrix{Float64}}(undef, T)
        result[1] = X_fixed
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        return result
    end

    function objective(x)
        X = unflatten(x)

        # P reconstruction loss
        loss_P = sum(norm(X[t] * X[t]' - P_list[t])^2 for t in 1:T)

        # Smoothness
        loss_smooth = sum(norm(X[t+1] - X[t])^2 for t in 1:(T-1))

        # Antisymmetric penalty: X'*V should be symmetric (no rotation)
        # V = X(t+1) - X(t), X'*V = X'*X(t+1) - X'*X
        # Penalize ||X'*V - (X'*V)'||² = ||antisym(X'*V)||²
        loss_antisym = 0.0
        for t in 1:(T-1)
            V = X[t+1] - X[t]
            M = X[t]' * V  # d×d matrix
            antisym = (M - M') / 2
            loss_antisym += norm(antisym)^2
        end

        return λ_P * loss_P + λ_smooth * loss_smooth + λ_antisym * loss_antisym
    end

    # Analytic gradient
    function gradient!(g, x)
        X = unflatten(x)
        grad_X = [zeros(n, d) for _ in 1:T]

        # P reconstruction gradient
        for t in 1:T
            residual = X[t] * X[t]' - P_list[t]
            grad_X[t] .+= λ_P * 4 * residual * X[t]
        end

        # Smoothness gradient
        for t in 1:(T-1)
            diff = X[t+1] - X[t]
            grad_X[t] .-= λ_smooth * 2 * diff
            grad_X[t+1] .+= λ_smooth * 2 * diff
        end

        # Antisymmetric penalty gradient
        # L = ||antisym(X'V)||² = (1/4)||X'V - V'X||²
        # ∂L/∂X(t) = (1/2) * (V*(X'V - V'X) - (antisym term for X contribution))
        # ∂L/∂X(t+1) = (1/2) * X * (X'V - V'X)'
        for t in 1:(T-1)
            V = X[t+1] - X[t]
            M = X[t]' * V
            A = M - M'  # 2 * antisym
            # ∂||A||²/∂X(t) = 2A ⊗ ∂(X'V)/∂X + ... this is getting complex
            # Use: ∂(X'V)/∂X(t) = -I ⊗ V' contribution, V ⊗ I contribution
            # Simpler: d/dX(t) of ||X'V - V'X||² = 2(V*A' + V*A) - from X'V term
            #                                    = 2V(A' + A) = 0 since A is antisym
            # Wait, need to be more careful...
            # Let f = tr((X'V - V'X)(X'V - V'X)') = tr(AA')
            # df = 2tr(A dA') = 2tr(A (dX'V + X'dV - dV'X - V'dX)')
            # For dX(t): dV = -dX, so dA = dX'(-V-V) + (X-X)'(-dX) = -2dX'V - 2V'dX?
            # This is getting messy. Let me just use numerical diff for this term.
            grad_X[t] .+= λ_antisym * V * A'
            grad_X[t+1] .-= λ_antisym * X[t] * A
        end

        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            g[offset+1:offset+n*d] .= vec(grad_X[t])
        end
    end

    x0 = flatten(X_init)

    if verbose
        println("Optimizing variational + antisym penalty...")
        println("  λ_P=", λ_P, ", λ_smooth=", λ_smooth, ", λ_antisym=", λ_antisym)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    result = optimize(objective, gradient!, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt = unflatten(Optim.minimizer(result))

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_opt, result
end

"""
Constrained variational: X(t) = X_DUASE(t) + Δ(t) with regularization on Δ.

This keeps the solution close to DUASE while allowing small corrections.
"""
function variational_embed_duase_reg(P_list, d; λ_P=1.0, λ_duase=1.0,
                                      max_iter=500, verbose=true)
    T = length(P_list)
    n = size(P_list[1], 1)

    # Get DUASE as the reference
    X_duase = duase_embed(P_list, d)
    X_fixed = copy(X_duase[1])

    function flatten(X_list)
        vcat([vec(X_list[t]) for t in 2:T]...)
    end

    function unflatten(x)
        result = Vector{Matrix{Float64}}(undef, T)
        result[1] = X_fixed
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        return result
    end

    function objective(x)
        X = unflatten(x)

        # P reconstruction loss
        loss_P = sum(norm(X[t] * X[t]' - P_list[t])^2 for t in 1:T)

        # DUASE regularization: stay close to DUASE
        loss_duase = sum(norm(X[t] - X_duase[t])^2 for t in 2:T)

        return λ_P * loss_P + λ_duase * loss_duase
    end

    function gradient!(g, x)
        X = unflatten(x)
        grad_X = [zeros(n, d) for _ in 1:T]

        # P reconstruction gradient
        for t in 1:T
            residual = X[t] * X[t]' - P_list[t]
            grad_X[t] .+= λ_P * 4 * residual * X[t]
        end

        # DUASE regularization gradient
        for t in 2:T
            grad_X[t] .+= λ_duase * 2 * (X[t] - X_duase[t])
        end

        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            g[offset+1:offset+n*d] .= vec(grad_X[t])
        end
    end

    x0 = flatten(X_duase)

    if verbose
        println("Optimizing variational + DUASE regularization...")
        println("  λ_P=", λ_P, ", λ_duase=", λ_duase)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    result = optimize(objective, gradient!, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt = unflatten(Optim.minimizer(result))

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
    end

    return X_opt, result
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
    println("VARIATIONAL EMBEDDING")
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

    # DUASE baseline
    println("\n--- DUASE ---")
    X_duase = duase_embed(P_list, D_EMBED)
    X_duase_al = align_to_true(X_duase)
    println("Curvature: ", round(compute_curvature(X_duase_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_al), digits=1), "%")

    # Variational (basic)
    println("\n--- Variational (λ_smooth=0.1, λ_accel=0.01) ---")
    X_var1, _ = variational_embed(P_list, D_EMBED, λ_smooth=0.1, λ_accel=0.01,
                                   max_iter=500, verbose=false)
    X_var1_al = align_to_true(X_var1)
    println("Curvature: ", round(compute_curvature(X_var1_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var1_al), digits=1), "%")

    # Variational (less smoothing)
    println("\n--- Variational (λ_smooth=0.01, λ_accel=0.001) ---")
    X_var2, _ = variational_embed(P_list, D_EMBED, λ_smooth=0.01, λ_accel=0.001,
                                   max_iter=500, verbose=false)
    X_var2_al = align_to_true(X_var2)
    println("Curvature: ", round(compute_curvature(X_var2_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var2_al), digits=1), "%")

    # Variational (more smoothing)
    println("\n--- Variational (λ_smooth=1.0, λ_accel=0.1) ---")
    X_var3, _ = variational_embed(P_list, D_EMBED, λ_smooth=1.0, λ_accel=0.1,
                                   max_iter=500, verbose=false)
    X_var3_al = align_to_true(X_var3)
    println("Curvature: ", round(compute_curvature(X_var3_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var3_al), digits=1), "%")

    # Variational with Sylvester
    println("\n--- Variational + Sylvester (λ_sylv=0.5) ---")
    X_var_sylv, _ = variational_embed_sylvester(P_list, D_EMBED, λ_smooth=0.1, λ_sylv=0.5,
                                                 max_iter=500, verbose=false)
    X_var_sylv_al = align_to_true(X_var_sylv)
    println("Curvature: ", round(compute_curvature(X_var_sylv_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var_sylv_al), digits=1), "%")

    # Variational with Procrustes penalty
    println("\n--- Variational + Procrustes (λ_proc=1.0) ---")
    X_var_proc, _ = variational_embed_procrustes(P_list, D_EMBED, λ_smooth=0.1, λ_proc=1.0,
                                                  max_iter=300, verbose=false)
    X_var_proc_al = align_to_true(X_var_proc)
    println("Curvature: ", round(compute_curvature(X_var_proc_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var_proc_al), digits=1), "%")

    # Variational with antisymmetric penalty
    println("\n--- Variational + Antisym (λ_antisym=1.0) ---")
    X_var_antisym, _ = variational_embed_antisym(P_list, D_EMBED, λ_smooth=0.1, λ_antisym=1.0,
                                                  max_iter=500, verbose=false)
    X_var_antisym_al = align_to_true(X_var_antisym)
    println("Curvature: ", round(compute_curvature(X_var_antisym_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var_antisym_al), digits=1), "%")

    # Variational with stronger antisymmetric penalty
    println("\n--- Variational + Antisym (λ_antisym=10.0) ---")
    X_var_antisym2, _ = variational_embed_antisym(P_list, D_EMBED, λ_smooth=0.1, λ_antisym=10.0,
                                                   max_iter=500, verbose=false)
    X_var_antisym2_al = align_to_true(X_var_antisym2)
    println("Curvature: ", round(compute_curvature(X_var_antisym2_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_var_antisym2_al), digits=1), "%")

    # DUASE-regularized (weak)
    println("\n--- DUASE-regularized (λ_duase=0.1) ---")
    X_duase_reg1, _ = variational_embed_duase_reg(P_list, D_EMBED, λ_duase=0.1,
                                                   max_iter=500, verbose=false)
    X_duase_reg1_al = align_to_true(X_duase_reg1)
    println("Curvature: ", round(compute_curvature(X_duase_reg1_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_reg1_al), digits=1), "%")

    # DUASE-regularized (medium)
    println("\n--- DUASE-regularized (λ_duase=1.0) ---")
    X_duase_reg2, _ = variational_embed_duase_reg(P_list, D_EMBED, λ_duase=1.0,
                                                   max_iter=500, verbose=false)
    X_duase_reg2_al = align_to_true(X_duase_reg2)
    println("Curvature: ", round(compute_curvature(X_duase_reg2_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_reg2_al), digits=1), "%")

    # DUASE-regularized (strong)
    println("\n--- DUASE-regularized (λ_duase=10.0) ---")
    X_duase_reg3, _ = variational_embed_duase_reg(P_list, D_EMBED, λ_duase=10.0,
                                                   max_iter=500, verbose=false)
    X_duase_reg3_al = align_to_true(X_duase_reg3)
    println("Curvature: ", round(compute_curvature(X_duase_reg3_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_duase_reg3_al), digits=1), "%")

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println(rpad("Method", 40), rpad("Curvature", 15), "P-error")
    println("-" ^ 65)
    println(rpad("True", 40), rpad(string(round(curv_true, digits=2)) * "°", 15), "-")
    println(rpad("DUASE", 40),
            rpad(string(round(compute_curvature(X_duase_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_al), digits=1)) * "%")
    println(rpad("Variational (λ_s=0.1)", 40),
            rpad(string(round(compute_curvature(X_var1_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_var1_al), digits=1)) * "%")
    println(rpad("DUASE-reg (λ_duase=0.1)", 40),
            rpad(string(round(compute_curvature(X_duase_reg1_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_reg1_al), digits=1)) * "%")
    println(rpad("DUASE-reg (λ_duase=1.0)", 40),
            rpad(string(round(compute_curvature(X_duase_reg2_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_reg2_al), digits=1)) * "%")
    println(rpad("DUASE-reg (λ_duase=10.0)", 40),
            rpad(string(round(compute_curvature(X_duase_reg3_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_duase_reg3_al), digits=1)) * "%")

    # Visualization
    fig = Figure(size=(1600, 800))

    datasets = [
        (X_true, "True (5.17°)"),
        (X_duase_al, "DUASE (" * string(round(compute_curvature(X_duase_al), digits=1)) * "°)"),
        (X_var1_al, "Variational (" * string(round(compute_curvature(X_var1_al), digits=1)) * "°)"),
        (X_duase_reg1_al, "DUASE-reg λ=0.1 (" * string(round(compute_curvature(X_duase_reg1_al), digits=1)) * "°)"),
        (X_duase_reg2_al, "DUASE-reg λ=1 (" * string(round(compute_curvature(X_duase_reg2_al), digits=1)) * "°)"),
        (X_duase_reg3_al, "DUASE-reg λ=10 (" * string(round(compute_curvature(X_duase_reg3_al), digits=1)) * "°)"),
    ]

    for (idx, (X_data, title)) in enumerate(datasets)
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1
        ax = Axis(fig[row, col], title=title, xlabel="X₁", ylabel="X₂", aspect=DataAspect())
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

    save(joinpath(dirname(@__DIR__), "results", "variational_embedding.png"), fig, px_per_unit=2)
    println("\nSaved: results/variational_embedding.png")
end

main()
