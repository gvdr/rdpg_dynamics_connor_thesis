#!/usr/bin/env -S julia --project
"""
Joint Embedding + Dynamics Learning

Key idea: Learn the embedding X(t) and dynamics parameters θ simultaneously.
The dynamics model acts as a "physics prior" that constrains the gauge.

Objective:
  L = λ_P · Σ_t ||X(t)X(t)' - P(t)||²           [P reconstruction]
    + λ_dyn · Σ_t ||X(t+1) - X(t) - f(X(t);θ)||²  [dynamics consistency]
    + λ_reg · ||θ||²                             [parameter regularization]

Dynamics model (message-passing with types):
  Ẋᵢ = β₀[type(i)] · Xᵢ + Σⱼ κ(type(i), type(j), Pᵢⱼ) · (Xⱼ - Xᵢ)
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
# Setup
# ============================================================================

const N_PRED, N_PREY, N_RES = 12, 15, 10
const N_TOTAL = N_PRED + N_PREY + N_RES
const D_EMBED = 2
const T_TOTAL = 25
const SEED = 42
const N_TYPES = 3

const TYPE_P, TYPE_Y, TYPE_R = 1, 2, 3
const NODE_TYPES = vcat(fill(TYPE_P, N_PRED), fill(TYPE_Y, N_PREY), fill(TYPE_R, N_RES))
const KNOWN_SELF_RATES = Dict(TYPE_P => -0.002, TYPE_Y => -0.001, TYPE_R => 0.000)
const HOLLING_ALPHA, HOLLING_BETA = 0.025, 2.0
const TYPE_COLORS = Dict(TYPE_P => :red, TYPE_Y => :blue, TYPE_R => :green)

# True dynamics
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
    return X
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
# Dynamics Models
# ============================================================================

"""
Linear message-passing dynamics:
  Ẋᵢ = β₀[type(i)] · Xᵢ + Σⱼ α[type(i),type(j)] · Pᵢⱼ · (Xⱼ - Xᵢ)

Parameters: β₀ (3 values), α (3×3 = 9 values) = 12 total
"""
struct LinearMsgPass
    β₀::Vector{Float64}  # Self-rates by type (3)
    α::Matrix{Float64}   # Interaction coefficients (3×3)
end

function n_params(::Type{LinearMsgPass})
    return N_TYPES + N_TYPES^2  # 3 + 9 = 12
end

function pack_params(model::LinearMsgPass)
    vcat(model.β₀, vec(model.α))
end

function unpack_params(::Type{LinearMsgPass}, θ::Vector{Float64})
    β₀ = θ[1:N_TYPES]
    α = reshape(θ[N_TYPES+1:end], N_TYPES, N_TYPES)
    return LinearMsgPass(β₀, α)
end

function dynamics_rhs(model::LinearMsgPass, X::Matrix{Float64})
    n, d = size(X)
    P = X * X'
    dX = zeros(n, d)

    for i in 1:n
        ti = NODE_TYPES[i]
        # Self-interaction
        dX[i, :] .= model.β₀[ti] .* X[i, :]

        # Message-passing
        for j in 1:n
            if i != j
                tj = NODE_TYPES[j]
                weight = model.α[ti, tj] * P[i, j]
                dX[i, :] .+= weight .* (X[j, :] .- X[i, :])
            end
        end
    end
    return dX
end

"""
Fixed-self message-passing: self-rates are KNOWN (from domain knowledge).
Only learn the interaction coefficients α.

Parameters: α (3×3 = 9 values) only
"""
struct FixedSelfMsgPass
    α::Matrix{Float64}   # Interaction coefficients (3×3)
end

const KNOWN_SELF_RATES_VEC = [KNOWN_SELF_RATES[TYPE_P], KNOWN_SELF_RATES[TYPE_Y], KNOWN_SELF_RATES[TYPE_R]]

function n_params(::Type{FixedSelfMsgPass})
    return N_TYPES^2  # 9
end

function pack_params(model::FixedSelfMsgPass)
    vec(model.α)
end

function unpack_params(::Type{FixedSelfMsgPass}, θ::Vector{Float64})
    α = reshape(θ, N_TYPES, N_TYPES)
    return FixedSelfMsgPass(α)
end

function dynamics_rhs(model::FixedSelfMsgPass, X::Matrix{Float64})
    n, d = size(X)
    P = X * X'
    dX = zeros(n, d)

    for i in 1:n
        ti = NODE_TYPES[i]
        # Self-interaction (FIXED)
        dX[i, :] .= KNOWN_SELF_RATES_VEC[ti] .* X[i, :]

        # Message-passing (learned)
        for j in 1:n
            if i != j
                tj = NODE_TYPES[j]
                weight = model.α[ti, tj] * P[i, j]
                dX[i, :] .+= weight .* (X[j, :] .- X[i, :])
            end
        end
    end
    return dX
end

"""
Sparse fixed-self: like FixedSelfMsgPass but only allow specific interactions:
- Predator ← Prey (predator gains from prey)
- Prey ← Predator (prey loses to predator)
- Prey ← Resource (prey gains from resource)
- Resource ← Prey (resource loses to prey)

Parameters: 4 specific interaction coefficients
"""
struct SparseMsgPass
    α_PY::Float64   # Predator from Prey
    α_YP::Float64   # Prey from Predator
    α_YR::Float64   # Prey from Resource
    α_RY::Float64   # Resource from Prey
end

function n_params(::Type{SparseMsgPass})
    return 4
end

function pack_params(model::SparseMsgPass)
    [model.α_PY, model.α_YP, model.α_YR, model.α_RY]
end

function unpack_params(::Type{SparseMsgPass}, θ::Vector{Float64})
    return SparseMsgPass(θ[1], θ[2], θ[3], θ[4])
end

function dynamics_rhs(model::SparseMsgPass, X::Matrix{Float64})
    n, d = size(X)
    P = X * X'
    dX = zeros(n, d)

    for i in 1:n
        ti = NODE_TYPES[i]
        # Self-interaction (FIXED)
        dX[i, :] .= KNOWN_SELF_RATES_VEC[ti] .* X[i, :]

        # Sparse message-passing
        for j in 1:n
            if i != j
                tj = NODE_TYPES[j]
                p_ij = P[i, j]

                weight = 0.0
                if ti == TYPE_P && tj == TYPE_Y
                    weight = model.α_PY * p_ij
                elseif ti == TYPE_Y && tj == TYPE_P
                    weight = model.α_YP * p_ij
                elseif ti == TYPE_Y && tj == TYPE_R
                    weight = model.α_YR * p_ij
                elseif ti == TYPE_R && tj == TYPE_Y
                    weight = model.α_RY * p_ij
                end

                if weight != 0.0
                    dX[i, :] .+= weight .* (X[j, :] .- X[i, :])
                end
            end
        end
    end
    return dX
end

"""
Holling message-passing dynamics:
  Ẋᵢ = β₀[type(i)] · Xᵢ + Σⱼ α[ti,tj] · Pᵢⱼ / (1 + β[ti,tj] · Pᵢⱼ) · (Xⱼ - Xᵢ)

Parameters: β₀ (3), α (9), β (9) = 21 total
"""
struct HollingMsgPass
    β₀::Vector{Float64}  # Self-rates by type (3)
    α::Matrix{Float64}   # Interaction numerators (3×3)
    β::Matrix{Float64}   # Saturation denominators (3×3)
end

function n_params(::Type{HollingMsgPass})
    return N_TYPES + 2 * N_TYPES^2  # 3 + 18 = 21
end

function pack_params(model::HollingMsgPass)
    vcat(model.β₀, vec(model.α), vec(model.β))
end

function unpack_params(::Type{HollingMsgPass}, θ::Vector{Float64})
    β₀ = θ[1:N_TYPES]
    α = reshape(θ[N_TYPES+1:N_TYPES+N_TYPES^2], N_TYPES, N_TYPES)
    β = reshape(θ[N_TYPES+N_TYPES^2+1:end], N_TYPES, N_TYPES)
    return HollingMsgPass(β₀, α, abs.(β))  # β must be positive
end

function dynamics_rhs(model::HollingMsgPass, X::Matrix{Float64})
    n, d = size(X)
    P = X * X'
    dX = zeros(n, d)

    for i in 1:n
        ti = NODE_TYPES[i]
        dX[i, :] .= model.β₀[ti] .* X[i, :]

        for j in 1:n
            if i != j
                tj = NODE_TYPES[j]
                p = P[i, j]
                weight = model.α[ti, tj] * p / (1 + abs(model.β[ti, tj]) * p + 1e-8)
                dX[i, :] .+= weight .* (X[j, :] .- X[i, :])
            end
        end
    end
    return dX
end

# ============================================================================
# Joint Optimization
# ============================================================================

"""
Joint optimization of embedding X(t) and dynamics parameters θ.

Variables:
- X(2), ..., X(T): embeddings (X(1) fixed)
- θ: dynamics parameters

Objective:
  L = λ_P · Σ_t ||X(t)X(t)' - P(t)||²
    + λ_dyn · Σ_t ||X(t+1) - X(t) - f(X(t);θ)||²
    + λ_reg · ||θ||²
"""
function joint_optimize(P_list, d, ::Type{ModelType};
                        λ_P=1.0, λ_dyn=1.0, λ_reg=0.01,
                        max_iter=1000, verbose=true) where ModelType

    T = length(P_list)
    n = size(P_list[1], 1)
    n_θ = n_params(ModelType)

    # Initialize X with DUASE
    X_init = duase_embed(P_list, d)
    X_fixed = copy(X_init[1])

    # Total variables: (T-1) * n * d for X, plus n_θ for dynamics
    n_X = (T - 1) * n * d
    n_total = n_X + n_θ

    function flatten_all(X_list, θ)
        x_part = vcat([vec(X_list[t]) for t in 2:T]...)
        vcat(x_part, θ)
    end

    function unflatten_all(x)
        # Extract X
        X_result = Vector{Matrix{Float64}}(undef, T)
        X_result[1] = X_fixed
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            X_result[t] = reshape(x[offset+1:offset+n*d], n, d)
        end
        # Extract θ
        θ = x[n_X+1:end]
        return X_result, θ
    end

    function objective(x)
        X, θ = unflatten_all(x)
        model = unpack_params(ModelType, θ)

        # P reconstruction loss
        loss_P = sum(norm(X[t] * X[t]' - P_list[t])^2 for t in 1:T)

        # Dynamics consistency loss
        loss_dyn = 0.0
        for t in 1:(T-1)
            dX_pred = dynamics_rhs(model, X[t])
            X_next_pred = X[t] + dX_pred
            loss_dyn += norm(X[t+1] - X_next_pred)^2
        end

        # Regularization
        loss_reg = norm(θ)^2

        return λ_P * loss_P + λ_dyn * loss_dyn + λ_reg * loss_reg
    end

    # Analytic gradient (for X part only; θ uses numerical diff)
    function gradient!(g, x)
        X, θ = unflatten_all(x)
        model = unpack_params(ModelType, θ)

        grad_X = [zeros(n, d) for _ in 1:T]

        # P reconstruction gradient: 4(XX' - P)X
        for t in 1:T
            residual = X[t] * X[t]' - P_list[t]
            grad_X[t] .+= λ_P * 4 * residual * X[t]
        end

        # Dynamics consistency gradient
        # L_dyn = Σ_t ||X(t+1) - X(t) - f(X(t);θ)||²
        # ∂L/∂X(t+1) = 2(X(t+1) - X(t) - f(X(t)))
        # ∂L/∂X(t) = -2(X(t+1) - X(t) - f(X(t))) + terms from ∂f/∂X(t)
        # For simplicity, ignore ∂f/∂X term (it's O(1) correction)
        for t in 1:(T-1)
            dX_pred = dynamics_rhs(model, X[t])
            residual = X[t+1] - X[t] - dX_pred
            grad_X[t] .-= λ_dyn * 2 * residual
            grad_X[t+1] .+= λ_dyn * 2 * residual
        end

        # Fill gradient vector for X
        for (idx, t) in enumerate(2:T)
            offset = (idx - 1) * n * d
            g[offset+1:offset+n*d] .= vec(grad_X[t])
        end

        # For θ, use numerical differentiation (small dimension)
        ε = 1e-6
        for i in 1:n_θ
            x_plus = copy(x)
            x_plus[n_X + i] += ε
            x_minus = copy(x)
            x_minus[n_X + i] -= ε
            g[n_X + i] = (objective(x_plus) - objective(x_minus)) / (2ε)
        end
    end

    # Initialize
    θ_init = 0.01 * randn(n_θ)  # Small random initialization
    x0 = flatten_all(X_init, θ_init)

    if verbose
        println("Joint optimization: embedding + dynamics")
        println("  X variables: ", n_X, ", θ variables: ", n_θ)
        println("  λ_P=", λ_P, ", λ_dyn=", λ_dyn, ", λ_reg=", λ_reg)
        println("  Initial loss: ", round(objective(x0), digits=2))
    end

    result = optimize(objective, gradient!, x0, LBFGS(),
                      Optim.Options(iterations=max_iter, show_trace=verbose,
                                    show_every=100))

    X_opt, θ_opt = unflatten_all(Optim.minimizer(result))
    model_opt = unpack_params(ModelType, θ_opt)

    if verbose
        println("  Final loss: ", round(Optim.minimum(result), digits=2))
        println("  Converged: ", Optim.converged(result))
    end

    return X_opt, model_opt, result
end

# ============================================================================
# Evaluation
# ============================================================================

"""
Extrapolate dynamics from initial condition using learned model.
"""
function extrapolate_dynamics(model, X0, T; dt=1.0)
    X_traj = Vector{Matrix{Float64}}(undef, T)
    X_traj[1] = copy(X0)

    for t in 1:(T-1)
        dX = dynamics_rhs(model, X_traj[t])
        X_traj[t+1] = X_traj[t] + dt * dX
    end

    return X_traj
end

"""
Compare learned dynamics to true dynamics.
"""
function compare_dynamics(model_learned, X_true, P_list)
    T = length(X_true)

    # Extrapolate from true X(1) using learned model
    X_extrap = extrapolate_dynamics(model_learned, X_true[1], T)

    # Compute P errors
    p_errors = [norm(X_extrap[t] * X_extrap[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T]

    return X_extrap, p_errors
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
    println("JOINT EMBEDDING + DYNAMICS OPTIMIZATION")
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

    # Joint optimization with Linear model
    println("\n--- Joint: Linear MsgPass (λ_dyn=0.1) ---")
    X_joint_lin1, model_lin1, _ = joint_optimize(P_list, D_EMBED, LinearMsgPass,
                                                  λ_P=1.0, λ_dyn=0.1, λ_reg=0.01,
                                                  max_iter=500, verbose=false)
    X_joint_lin1_al = align_to_true(X_joint_lin1)
    println("Curvature: ", round(compute_curvature(X_joint_lin1_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_lin1_al), digits=1), "%")

    # Joint optimization with stronger dynamics weight
    println("\n--- Joint: Linear MsgPass (λ_dyn=1.0) ---")
    X_joint_lin2, model_lin2, _ = joint_optimize(P_list, D_EMBED, LinearMsgPass,
                                                  λ_P=1.0, λ_dyn=1.0, λ_reg=0.01,
                                                  max_iter=500, verbose=false)
    X_joint_lin2_al = align_to_true(X_joint_lin2)
    println("Curvature: ", round(compute_curvature(X_joint_lin2_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_lin2_al), digits=1), "%")

    # Joint optimization with even stronger dynamics
    println("\n--- Joint: Linear MsgPass (λ_dyn=10.0) ---")
    X_joint_lin3, model_lin3, _ = joint_optimize(P_list, D_EMBED, LinearMsgPass,
                                                  λ_P=1.0, λ_dyn=10.0, λ_reg=0.01,
                                                  max_iter=500, verbose=false)
    X_joint_lin3_al = align_to_true(X_joint_lin3)
    println("Curvature: ", round(compute_curvature(X_joint_lin3_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_lin3_al), digits=1), "%")

    # Joint with Fixed-self (only learn interactions)
    println("\n--- Joint: FixedSelf MsgPass (λ_dyn=1.0) ---")
    X_joint_fixed, model_fixed, _ = joint_optimize(P_list, D_EMBED, FixedSelfMsgPass,
                                                    λ_P=1.0, λ_dyn=1.0, λ_reg=0.01,
                                                    max_iter=500, verbose=false)
    X_joint_fixed_al = align_to_true(X_joint_fixed)
    println("Curvature: ", round(compute_curvature(X_joint_fixed_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_fixed_al), digits=1), "%")

    # Joint with Sparse model (only 4 parameters!)
    println("\n--- Joint: Sparse MsgPass (λ_dyn=1.0) ---")
    X_joint_sparse, model_sparse, _ = joint_optimize(P_list, D_EMBED, SparseMsgPass,
                                                      λ_P=1.0, λ_dyn=1.0, λ_reg=0.001,
                                                      max_iter=500, verbose=false)
    X_joint_sparse_al = align_to_true(X_joint_sparse)
    println("Curvature: ", round(compute_curvature(X_joint_sparse_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_sparse_al), digits=1), "%")

    # Joint with Sparse and stronger dynamics weight
    println("\n--- Joint: Sparse MsgPass (λ_dyn=10.0) ---")
    X_joint_sparse2, model_sparse2, _ = joint_optimize(P_list, D_EMBED, SparseMsgPass,
                                                        λ_P=1.0, λ_dyn=10.0, λ_reg=0.001,
                                                        max_iter=500, verbose=false)
    X_joint_sparse2_al = align_to_true(X_joint_sparse2)
    println("Curvature: ", round(compute_curvature(X_joint_sparse2_al), digits=2), "°")
    println("P error: ", round(100*p_error(X_joint_sparse2_al), digits=1), "%")

    # Test extrapolation: use learned dynamics to predict from true X(1)
    println("\n--- Extrapolation Test ---")
    X_extrap_lin, p_err_lin = compare_dynamics(model_lin2, X_true, P_list)
    X_extrap_sparse, p_err_sparse = compare_dynamics(model_sparse, X_true, P_list)
    X_extrap_sparse2, p_err_sparse2 = compare_dynamics(model_sparse2, X_true, P_list)

    println("Linear model extrapolation P-error at T=25: ", round(100*p_err_lin[end], digits=1), "%")
    println("Sparse (λ=1) extrapolation P-error at T=25: ", round(100*p_err_sparse[end], digits=1), "%")
    println("Sparse (λ=10) extrapolation P-error at T=25: ", round(100*p_err_sparse2[end], digits=1), "%")

    # Print learned parameters
    println("\n--- Learned Parameters (Sparse, λ_dyn=1.0) ---")
    println("α_PY (Predator←Prey): ", round(model_sparse.α_PY, digits=4))
    println("α_YP (Prey←Predator): ", round(model_sparse.α_YP, digits=4))
    println("α_YR (Prey←Resource): ", round(model_sparse.α_YR, digits=4))
    println("α_RY (Resource←Prey): ", round(model_sparse.α_RY, digits=4))

    println("\n--- True Parameters ---")
    println("α_PY (Predator←Prey): 0.025 * p / (1 + 2p) ≈ 0.0125 at p=0.5")
    println("α_YP (Prey←Predator): -0.02")
    println("α_YR (Prey←Resource): 0.012")
    println("α_RY (Resource←Prey): -0.006")

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
    println(rpad("Joint Linear (λ_dyn=1.0)", 40),
            rpad(string(round(compute_curvature(X_joint_lin2_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_joint_lin2_al), digits=1)) * "%")
    println(rpad("Joint FixedSelf (λ_dyn=1.0)", 40),
            rpad(string(round(compute_curvature(X_joint_fixed_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_joint_fixed_al), digits=1)) * "%")
    println(rpad("Joint Sparse (λ_dyn=1.0)", 40),
            rpad(string(round(compute_curvature(X_joint_sparse_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_joint_sparse_al), digits=1)) * "%")
    println(rpad("Joint Sparse (λ_dyn=10.0)", 40),
            rpad(string(round(compute_curvature(X_joint_sparse2_al), digits=2)) * "°", 15),
            string(round(100*p_error(X_joint_sparse2_al), digits=1)) * "%")

    # Visualization
    fig = Figure(size=(1600, 800))

    datasets = [
        (X_true, "True (" * string(round(curv_true, digits=1)) * "°)"),
        (X_duase_al, "DUASE (" * string(round(compute_curvature(X_duase_al), digits=1)) * "°)"),
        (X_joint_lin2_al, "Joint Lin (" * string(round(compute_curvature(X_joint_lin2_al), digits=1)) * "°)"),
        (X_joint_fixed_al, "Joint Fixed (" * string(round(compute_curvature(X_joint_fixed_al), digits=1)) * "°)"),
        (X_joint_sparse_al, "Sparse λ=1 (" * string(round(compute_curvature(X_joint_sparse_al), digits=1)) * "°)"),
        (X_joint_sparse2_al, "Sparse λ=10 (" * string(round(compute_curvature(X_joint_sparse2_al), digits=1)) * "°)"),
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

    save(joinpath(dirname(@__DIR__), "results", "joint_embedding_dynamics.png"), fig, px_per_unit=2)
    println("\nSaved: results/joint_embedding_dynamics.png")
end

main()
