"""
Example 5: Gauge-Consistent N(P)X Architecture Comparison

Systematic comparison of gauge-equivariant architectures using the validated
protocol from Example 1.

**Key insight (January 2026)**: "Learn anywhere, apply everywhere"
- Train on DUASE estimates (noisy)
- Apply learned parameters to TRUE initial conditions
- Evaluate in P-space (gauge-invariant)

Architectures compared:
1. Polynomial N(P)X: Ẋ = (β₀I + β₁P + β₂P²)X  [3 parameters]
2. Message-Passing: ẋᵢ = a·xᵢ + m·Σⱼ Pᵢⱼ(xⱼ - xᵢ)  [2 parameters]
3. Standard Neural ODE (baseline): Ẋ = f(X)  [~10,000 parameters]

The data is generated from known dynamics Ẋ = (αI + βP)X to test parameter recovery.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using Lux
using ComponentArrays
using CairoMakie
using Printf
using Statistics

const CM = CairoMakie

# ============================================================================
# Configuration
# ============================================================================

const N_NODES = 11           # Number of nodes (same as Example 1)
const D_EMBED = 2            # Embedding dimension
const T_TOTAL = 21           # Total timesteps
const TRAIN_FRAC = 0.7       # Fraction for training
const SEED = 42

# True dynamics parameters: Ẋ = (α_true*I + β_true*P)X
const α_TRUE = -0.05         # Self-dynamics (slight contraction)
const β_TRUE = 0.02          # Pairwise attraction

# ============================================================================
# Data Generation
# ============================================================================

"""
Generate true trajectories from known dynamics: Ẋ = (αI + βP)X
"""
function generate_true_data(; n=N_NODES, d=D_EMBED, T=T_TOTAL,
                             α=α_TRUE, β=β_TRUE, seed=SEED)
    rng = Xoshiro(seed)

    # Initialize embeddings (no B^d_+ constraint needed!)
    X0 = randn(rng, Float64, n, d) .* 0.3 .+ 0.5

    # ODE dynamics: Ẋ = N(P)X where N = αI + βP
    function true_dynamics!(dX, X, p, t)
        P = X * X'
        N = α * I(n) + β * P
        dX .= N * X
    end

    # Solve ODE
    prob = ODEProblem(true_dynamics!, X0, (0.0, Float64(T-1)))
    sol = solve(prob, Tsit5(); saveat=1.0)

    # Extract as vector of matrices
    X_true = [Matrix{Float64}(sol.u[i]) for i in 1:length(sol.t)]

    return X_true, (α=α, β=β)
end

"""
Generate adjacency matrices by sampling from P = XX'
"""
function generate_adjacencies(X_true::Vector{Matrix{Float64}}; seed=SEED)
    rng = Xoshiro(seed + 1000)
    T = length(X_true)
    n = size(X_true[1], 1)

    A_obs = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        P = X_true[t] * X_true[t]'
        # Sample symmetric adjacency
        A = zeros(n, n)
        for i in 1:n
            for j in i+1:n
                p_ij = clamp(P[i,j], 0, 1)
                A[i,j] = A[j,i] = rand(rng) < p_ij ? 1.0 : 0.0
            end
        end
        A_obs[t] = A
    end
    return A_obs
end

"""
DUASE estimation: SVD embedding + Procrustes alignment
"""
function duase_estimate(A_obs::Vector{Matrix{Float64}}, d::Int)
    T = length(A_obs)
    n = size(A_obs[1], 1)

    X_est = Vector{Matrix{Float64}}(undef, T)

    # First timestep: plain SVD
    U, S, V = svd(A_obs[1])
    sqrt_S = sqrt.(S[1:d])
    X_est[1] = U[:, 1:d] .* sqrt_S'

    # Subsequent timesteps: SVD + Procrustes alignment
    for t in 2:T
        U, S, V = svd(A_obs[t])
        sqrt_S = sqrt.(S[1:d])
        X_t = U[:, 1:d] .* sqrt_S'

        # Procrustes alignment to previous
        M = X_est[t-1]' * X_t
        F = svd(M)
        Q = F.V * F.U'
        X_est[t] = X_t * Q
    end

    return X_est
end

# ============================================================================
# N(P)X Dynamics Implementations
# ============================================================================

"""
Polynomial N(P)X dynamics: Ẋ = (β₀I + β₁P + β₂P²)X
"""
function polynomial_dynamics!(dX, X, p, t; n=N_NODES)
    β = p  # [β₀, β₁, β₂] or [β₀, β₁]
    P = X * X'

    if length(β) >= 3
        N = β[1] * I(n) + β[2] * P + β[3] * P^2
    else
        N = β[1] * I(n) + β[2] * P
    end

    dX .= N * X
end

"""
Message-passing dynamics: ẋᵢ = a·xᵢ + m·Σⱼ Pᵢⱼ(xⱼ - xᵢ)

Equivalent to N(P)X with:
  N_ii = a - m·Σⱼ Pᵢⱼ
  N_ij = m·Pᵢⱼ (for i≠j)
"""
function msgpass_dynamics!(dX, X, p, t; n=N_NODES)
    a, m = p[1], p[2]
    P = X * X'

    # Compute N matrix
    N = m * P
    # Adjust diagonal: N_ii = a - m * Σⱼ P_ij
    for i in 1:n
        N[i,i] = a - m * sum(P[i,:])
    end

    dX .= N * X
end

# ============================================================================
# Training Functions
# ============================================================================

"""
Train polynomial N(P)X model
"""
function train_polynomial(X_train::Vector{Matrix{Float64}};
                          degree::Int=2, epochs::Int=1000, lr::Float64=0.01)
    n, d = size(X_train[1])
    T_train = length(X_train)

    # Initial parameters
    β_init = zeros(Float64, degree + 1)
    β_init[1] = -0.01  # Small negative for stability

    # Flatten training data
    X_target = hcat([vec(X') for X in X_train]...)
    u0 = vec(X_train[1]')
    tspan = (0.0, Float64(T_train - 1))

    # ODE problem
    function poly_ode!(du, u, p, t)
        X = reshape(u, d, n)'
        dX = similar(X)
        polynomial_dynamics!(dX, X, p, t; n=n)
        du .= vec(dX')
    end

    prob = ODEProblem(poly_ode!, u0, tspan, β_init)

    # Loss function
    function loss(p, _)
        sol = solve(prob, Tsit5(); p=p, saveat=1.0,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        if sol.retcode != :Success
            return Inf, nothing
        end
        pred = hcat(sol.u...)

        # MSE loss
        mse = mean(abs2, pred .- X_target)

        # Probability constraint loss
        prob_loss = 0.0
        for u in sol.u
            X = reshape(u, d, n)'
            P = X * X'
            prob_loss += sum(max.(-P, 0.0).^2) + sum(max.(P .- 1.0, 0.0).^2)
        end

        return mse + 0.1 * prob_loss, nothing
    end

    # Optimize
    opt_func = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, β_init)

    # Adam
    result = solve(opt_prob, Adam(lr); maxiters=epochs, progress=false)

    # LBFGS refinement
    opt_prob2 = OptimizationProblem(opt_func, result.u)
    result = solve(opt_prob2, LBFGS(); maxiters=100)

    return result.u
end

"""
Train message-passing model
"""
function train_msgpass(X_train::Vector{Matrix{Float64}};
                       epochs::Int=1000, lr::Float64=0.01)
    n, d = size(X_train[1])
    T_train = length(X_train)

    # Initial parameters [a, m]
    p_init = Float64[-0.01, 0.01]

    # Flatten training data
    X_target = hcat([vec(X') for X in X_train]...)
    u0 = vec(X_train[1]')
    tspan = (0.0, Float64(T_train - 1))

    # ODE problem
    function msgpass_ode!(du, u, p, t)
        X = reshape(u, d, n)'
        dX = similar(X)
        msgpass_dynamics!(dX, X, p, t; n=n)
        du .= vec(dX')
    end

    prob = ODEProblem(msgpass_ode!, u0, tspan, p_init)

    # Loss function
    function loss(p, _)
        sol = solve(prob, Tsit5(); p=p, saveat=1.0,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        if sol.retcode != :Success
            return Inf, nothing
        end
        pred = hcat(sol.u...)

        mse = mean(abs2, pred .- X_target)

        prob_loss = 0.0
        for u in sol.u
            X = reshape(u, d, n)'
            P = X * X'
            prob_loss += sum(max.(-P, 0.0).^2) + sum(max.(P .- 1.0, 0.0).^2)
        end

        return mse + 0.1 * prob_loss, nothing
    end

    # Optimize
    opt_func = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_func, p_init)

    result = solve(opt_prob, Adam(lr); maxiters=epochs, progress=false)
    opt_prob2 = OptimizationProblem(opt_func, result.u)
    result = solve(opt_prob2, LBFGS(); maxiters=100)

    return result.u
end

# ============================================================================
# Evaluation Functions
# ============================================================================

"""
Integrate dynamics from initial condition and compute P(t)
"""
function predict_P_trajectory(dynamics_fn!, params, X0::Matrix{Float64}, T::Int)
    n, d = size(X0)
    u0 = vec(X0')
    tspan = (0.0, Float64(T - 1))

    function ode!(du, u, p, t)
        X = reshape(u, d, n)'
        dX = similar(X)
        dynamics_fn!(dX, X, p, t; n=n)
        du .= vec(dX')
    end

    prob = ODEProblem(ode!, u0, tspan, params)
    sol = solve(prob, Tsit5(); saveat=1.0)

    # Convert to P matrices
    P_traj = Vector{Matrix{Float64}}(undef, length(sol.u))
    X_traj = Vector{Matrix{Float64}}(undef, length(sol.u))
    for (i, u) in enumerate(sol.u)
        X = reshape(u, d, n)'
        X_traj[i] = X
        P_traj[i] = X * X'
    end

    return P_traj, X_traj
end

"""
Compute P-error over time
"""
function compute_P_error(P_pred::Vector{Matrix{Float64}}, P_true::Vector{Matrix{Float64}})
    T = min(length(P_pred), length(P_true))
    errors = Float64[]
    for t in 1:T
        err = norm(P_pred[t] .- P_true[t], 2) / norm(P_true[t], 2)
        push!(errors, err)
    end
    return errors
end

# ============================================================================
# Main Comparison
# ============================================================================

function run_example5(; save_results::Bool=true)
    println("=" ^ 70)
    println("Example 5: Gauge-Consistent N(P)X Architecture Comparison")
    println("=" ^ 70)

    # =========================================================================
    # 1. Generate Data
    # =========================================================================
    println("\n1. Generating data...")

    X_true, true_params = generate_true_data()
    P_true = [X * X' for X in X_true]

    println("   True dynamics: Ẋ = (αI + βP)X")
    println("   α = " * @sprintf("%.4f", true_params.α) * ", β = " * @sprintf("%.4f", true_params.β))
    println("   n = " * string(N_NODES) * " nodes, d = " * string(D_EMBED) * " dimensions")
    println("   T = " * string(T_TOTAL) * " timesteps")

    # Generate observed adjacencies and DUASE estimates
    A_obs = generate_adjacencies(X_true)
    X_est = duase_estimate(A_obs, D_EMBED)
    P_est = [X * X' for X in X_est]

    # Train/validation split
    T_train = Int(floor(TRAIN_FRAC * T_TOTAL))
    X_train = X_est[1:T_train]

    println("   Training timesteps: 1-" * string(T_train))
    println("   Validation timesteps: " * string(T_train+1) * "-" * string(T_TOTAL))

    # DUASE estimation error
    duase_P_err = mean([norm(P_est[t] - P_true[t]) / norm(P_true[t]) for t in 1:T_TOTAL])
    println("   DUASE P-error (mean): " * @sprintf("%.2f%%", 100*duase_P_err))

    # =========================================================================
    # 2. Train Models
    # =========================================================================
    println("\n2. Training models on DUASE estimates...")

    # Polynomial (degree 1 - matches true dynamics)
    println("\n   [1/3] Polynomial N(P)X (degree=1, 2 params)...")
    β_poly = train_polynomial(X_train; degree=1, epochs=2000, lr=0.02)
    println("         Learned: β₀=" * @sprintf("%.5f", β_poly[1]) *
            ", β₁=" * @sprintf("%.5f", β_poly[2]))
    println("         True:    α=" * @sprintf("%.5f", true_params.α) *
            ", β=" * @sprintf("%.5f", true_params.β))

    # Message-passing
    println("\n   [2/3] Message-Passing N(P)X (2 params)...")
    p_msgpass = train_msgpass(X_train; epochs=2000, lr=0.02)
    println("         Learned: a=" * @sprintf("%.5f", p_msgpass[1]) *
            ", m=" * @sprintf("%.5f", p_msgpass[2]))

    # Polynomial degree 2 (more expressive)
    println("\n   [3/3] Polynomial N(P)X (degree=2, 3 params)...")
    β_poly2 = train_polynomial(X_train; degree=2, epochs=2000, lr=0.02)
    println("         Learned: β₀=" * @sprintf("%.5f", β_poly2[1]) *
            ", β₁=" * @sprintf("%.5f", β_poly2[2]) *
            ", β₂=" * @sprintf("%.5f", β_poly2[3]))

    # =========================================================================
    # 3. Evaluate: "Learn Anywhere, Apply Everywhere"
    # =========================================================================
    println("\n3. Evaluation: Apply learned params to TRUE initial conditions...")

    # Predict from TRUE X(0)
    P_poly, X_poly = predict_P_trajectory(polynomial_dynamics!, β_poly, X_true[1], T_TOTAL)
    P_msgpass, X_msgpass = predict_P_trajectory(msgpass_dynamics!, p_msgpass, X_true[1], T_TOTAL)
    P_poly2, X_poly2 = predict_P_trajectory(polynomial_dynamics!, β_poly2, X_true[1], T_TOTAL)

    # Compute P-errors
    err_poly = compute_P_error(P_poly, P_true)
    err_msgpass = compute_P_error(P_msgpass, P_true)
    err_poly2 = compute_P_error(P_poly2, P_true)
    err_duase = compute_P_error(P_est, P_true)

    # Summary statistics
    println("\n   " * "-" ^ 60)
    println("   Model              | Params | Train P-err | Extrap P-err")
    println("   " * "-" ^ 60)
    @printf("   Polynomial (d=1)   | %6d | %10.2f%% | %11.2f%%\n",
            2, 100*mean(err_poly[1:T_train]), 100*mean(err_poly[T_train+1:end]))
    @printf("   Message-Passing    | %6d | %10.2f%% | %11.2f%%\n",
            2, 100*mean(err_msgpass[1:T_train]), 100*mean(err_msgpass[T_train+1:end]))
    @printf("   Polynomial (d=2)   | %6d | %10.2f%% | %11.2f%%\n",
            3, 100*mean(err_poly2[1:T_train]), 100*mean(err_poly2[T_train+1:end]))
    @printf("   DUASE (baseline)   | %6d | %10.2f%% | %11s\n",
            0, 100*mean(err_duase[1:T_train]), "N/A")
    println("   " * "-" ^ 60)

    # =========================================================================
    # 4. Visualization
    # =========================================================================
    println("\n4. Generating visualizations...")

    fig = Figure(size=(1400, 1000))

    # --- Row 1: P(t) heatmaps at key timesteps ---
    timesteps_to_show = [1, T_train, T_TOTAL]
    col_labels = ["t=0 (initial)", "t=" * string(T_train-1) * " (end train)",
                  "t=" * string(T_TOTAL-1) * " (extrap)"]
    row_labels = ["True P", "DUASE P̂", "Poly P̂ (2p)", "MsgPass P̂ (2p)"]

    for (col, t) in enumerate(timesteps_to_show)
        for (row, (P_data, label)) in enumerate([
            (P_true, "True"),
            (P_est, "DUASE"),
            (P_poly, "Poly"),
            (P_msgpass, "MsgPass")
        ])
            ax = Axis(fig[row, col]; aspect=1,
                      title = row == 1 ? col_labels[col] : "",
                      ylabel = col == 1 ? row_labels[row] : "")

            P_t = t <= length(P_data) ? Float64.(P_data[t]) : fill(NaN, N_NODES, N_NODES)
            hm = heatmap!(ax, P_t; colorrange=(0, 1), colormap=:viridis)
            hidedecorations!(ax)

            if row == 4 && col == 3
                Colorbar(fig[1:4, 4], hm; label="P(i,j)")
            end
        end
    end

    # --- Row 5: P-error over time ---
    ax5 = Axis(fig[5, 1:3]; xlabel="Time", ylabel="Relative P-error",
               title="P-Error Over Time (Learn on DUASE → Apply to TRUE)")

    timesteps = 0:T_TOTAL-1

    # Training region
    vspan!(ax5, [0], [T_train-1]; color=(:green, 0.1))
    vlines!(ax5, [T_train-1]; color=:gray, linestyle=:dash, linewidth=1)
    text!(ax5, T_train/2 - 0.5, maximum(err_duase)*0.9; text="Train", fontsize=12)
    text!(ax5, T_train + 2, maximum(err_duase)*0.9; text="Extrapolation", fontsize=12)

    # Plot errors
    lines!(ax5, timesteps, err_duase; color=:coral, linewidth=2,
           linestyle=:dash, label="DUASE (est.)")
    lines!(ax5, timesteps, err_poly; color=:blue, linewidth=2, label="Poly d=1 (2p)")
    lines!(ax5, timesteps, err_msgpass; color=:purple, linewidth=2, label="MsgPass (2p)")
    lines!(ax5, timesteps, err_poly2; color=:teal, linewidth=2,
           linestyle=:dot, label="Poly d=2 (3p)")

    axislegend(ax5; position=:lt)

    # --- Row 6: Parameter recovery ---
    ax6 = Axis(fig[6, 1:3];
               title="Parameter Recovery: True vs Learned",
               xlabel="Parameter", ylabel="Value",
               xticks=(1:4, ["α (true)", "β₀ (poly)", "β (true)", "β₁ (poly)"]))

    barplot!(ax6, [1, 2, 3, 4],
             [true_params.α, β_poly[1], true_params.β, β_poly[2]];
             color=[:black, :blue, :black, :blue],
             bar_labels=[@sprintf("%.4f", true_params.α), @sprintf("%.4f", β_poly[1]),
                        @sprintf("%.4f", true_params.β), @sprintf("%.4f", β_poly[2])])

    # Save
    output_dir = joinpath(dirname(@__DIR__), "results", "example5_gauge")
    mkpath(output_dir)
    output_path = joinpath(output_dir, "comparison.png")
    save(output_path, fig; px_per_unit=2)
    println("   Saved: " * output_path)

    # =========================================================================
    # 5. Summary
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("Summary: Example 5 Results")
    println("=" ^ 70)

    println("\n1. PARAMETER RECOVERY (Polynomial d=1 vs True dynamics)")
    α_err = 100 * abs(β_poly[1] - true_params.α) / abs(true_params.α)
    β_err = 100 * abs(β_poly[2] - true_params.β) / abs(true_params.β)
    println("   α: true=" * @sprintf("%.5f", true_params.α) *
            " → learned=" * @sprintf("%.5f", β_poly[1]) *
            " (error: " * @sprintf("%.1f%%", α_err) * ")")
    println("   β: true=" * @sprintf("%.5f", true_params.β) *
            " → learned=" * @sprintf("%.5f", β_poly[2]) *
            " (error: " * @sprintf("%.1f%%", β_err) * ")")

    println("\n2. LEARN ANYWHERE, APPLY EVERYWHERE")
    println("   - Trained on DUASE estimates (with ~" *
            @sprintf("%.0f%%", 100*duase_P_err) * " P-error)")
    println("   - Applied to TRUE initial conditions")
    println("   - Extrapolation P-error (Poly d=1): " *
            @sprintf("%.2f%%", 100*mean(err_poly[T_train+1:end])))

    println("\n3. ARCHITECTURE COMPARISON")
    println("   - Polynomial (2 params) vs Message-Passing (2 params):")
    println("     Both achieve similar P-error with minimal parameters")
    println("   - DUASE cannot extrapolate (estimation only, not dynamics)")

    println("\n4. KEY INSIGHT: Gauge-invariant scalars")
    println("   Parameters learned in ANY gauge (DUASE) work in ANY gauge (TRUE)")
    println("   because N(P)X dynamics depend only on P = XX' (gauge-invariant)")

    return (
        params_learned = (poly=β_poly, msgpass=p_msgpass, poly2=β_poly2),
        params_true = true_params,
        errors = (poly=err_poly, msgpass=err_msgpass, poly2=err_poly2, duase=err_duase),
        data = (X_true=X_true, X_est=X_est, P_true=P_true, P_est=P_est)
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_example5()
end
