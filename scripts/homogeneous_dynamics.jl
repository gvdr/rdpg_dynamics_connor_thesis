#!/usr/bin/env julia
"""
Homogeneous Dynamics Experiment

All nodes follow the SAME differential equation:
    dL_i/dt = f(L_i)

This gives us n × T data points to learn ONE function f.
Much more data-efficient than learning separate dynamics per node.

Scenario: All nodes rotate around a common center with the same angular velocity.
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Lux
using OrdinaryDiffEq
using Optimization
using OptimizationOptimisers
using ComponentArrays
using SciMLSensitivity

println("=" ^ 60)
println("Homogeneous Dynamics: All nodes follow same ODE")
println("=" ^ 60)

# =============================================================================
# Generate data: All nodes rotating with same angular velocity
# =============================================================================

function generate_rotating_community(; n::Int=50, d::Int=2, T::Int=40,
                                      omega::Float64=0.15, r_mean::Float64=0.7,
                                      r_std::Float64=0.1, seed::Int=1234,
                                      wrap_angle::Bool=false)
    """
    All nodes rotate around origin with same angular velocity omega.
    Each node has a fixed radius (distance from origin).

    True dynamics: dθ/dt = ω  (constant angular velocity)
    In Cartesian:  dL/dt = ω * [-L[2], L[1]]  (rotation matrix derivative)

    Note: If wrap_angle=true, angles wrap at π/2 (first quadrant only).
    If wrap_angle=false, continuous rotation (may leave B^d_+).
    """
    rng = Random.MersenneTwister(seed)

    # Initial positions: random angles, random radii
    # Start in first quadrant but allow smooth rotation
    θ0 = rand(rng, n) .* (π/4) .+ π/8  # Centered in first quadrant
    radii = r_mean .+ r_std .* randn(rng, n)
    radii = clamp.(radii, 0.3, 0.95)  # Keep in valid range

    L_series = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        L_t = zeros(n, d)
        for i in 1:n
            θ_i = θ0[i] + omega * (t - 1)
            if wrap_angle
                θ_i = mod(θ_i, π/2)
            end
            L_t[i, :] = radii[i] * [cos(θ_i), sin(θ_i)]
        end
        L_series[t] = L_t
    end

    return (
        L_series = L_series,
        n = n,
        d = d,
        T = T,
        omega = omega,
        radii = radii,
        θ0 = θ0,
        # True dynamics function
        true_dynamics = (L) -> omega * [-L[2], L[1]]
    )
end

println("\n1. Generating rotating community data...")
data = generate_rotating_community(n=50, T=40, omega=0.1)
n, d, T = data.n, data.d, data.T

println("  n=", n, ", d=", d, ", T=", T)
println("  True omega: ", data.omega)

# =============================================================================
# Prepare training data: Pool ALL node trajectories
# =============================================================================
println("\n2. Preparing pooled training data...")

# Collect (L, dL/dt) pairs from all nodes and all timesteps
# We estimate dL/dt via finite differences
L_inputs = Vector{Vector{Float64}}()
dL_targets = Vector{Vector{Float64}}()

for t in 1:(T-1)
    for i in 1:n
        L_now = data.L_series[t][i, :]
        L_next = data.L_series[t+1][i, :]

        # Finite difference approximation of derivative
        dL_approx = L_next - L_now  # dt = 1

        push!(L_inputs, L_now)
        push!(dL_targets, dL_approx)
    end
end

X = Float32.(hcat(L_inputs...))      # d × (n*T)
Y = Float32.(hcat(dL_targets...))    # d × (n*T)

println("  Training samples: ", size(X, 2), " (", n, " nodes × ", T-1, " transitions)")
println("  Input dim: ", size(X, 1), ", Output dim: ", size(Y, 1))

# =============================================================================
# Train a simple NN to learn f(L) -> dL/dt
# =============================================================================
println("\n3. Training neural network to learn f(L)...")

rng = Random.Xoshiro(1234)

# Simple MLP: L (2D) -> dL/dt (2D)
model = Lux.Chain(
    Lux.Dense(d, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, d)
)

params, state = Lux.setup(rng, model)
params_ca = ComponentArray(params)

# Simple supervised loss
function loss_fn(p, _)
    pred, _ = model(X, p, state)
    return sum(abs2, Y .- pred) / size(X, 2)
end

optf = Optimization.OptimizationFunction(loss_fn, Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, params_ca, nothing)

println("  Training...")
result = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); maxiters=2000,
    callback=(st, l) -> (st.iter % 500 == 0 && println("    iter ", st.iter, ": loss=", round(l, digits=6)); false))

final_params = result.u
println("  Final loss: ", round(loss_fn(final_params, nothing), digits=6))

# =============================================================================
# Evaluate: Use learned f to predict trajectories via ODE
# =============================================================================
println("\n4. Evaluating learned dynamics...")

function predict_trajectory_homogeneous(L0::Vector{Float64}, T::Int, params)
    function dudt(u, p, t)
        pred, _ = model(reshape(Float32.(u), :, 1), p, state)
        return vec(pred)
    end

    prob = ODEProblem(dudt, Float32.(L0), (0.0f0, Float32(T-1)), params)
    sol = solve(prob, Tsit5(); saveat=0:T-1)
    return Array(sol)
end

# Predict for a few test nodes
test_nodes = [1, 10, 25, 50]
errors = Float64[]

for i in test_nodes
    L0 = data.L_series[1][i, :]
    pred = predict_trajectory_homogeneous(L0, T, final_params)
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)

    err = mean([norm(pred[:, t] - true_traj[:, t]) for t in 1:T])
    push!(errors, err)
end

println("  Mean prediction error: ", round(mean(errors), digits=4))

# =============================================================================
# Visualize
# =============================================================================
println("\n5. Creating visualization...")

fig = Figure(size=(1200, 600))

# Phase portrait for a few nodes
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                       title="Phase Portrait: True vs Predicted", aspect=1)

θ_arc = range(0, π/2, length=100)
lines!(ax1, cos.(θ_arc), sin.(θ_arc), color=:gray, linestyle=:dash)

colors = [:blue, :red, :green, :purple]
for (idx, i) in enumerate(test_nodes[1:min(4, length(test_nodes))])
    L0 = data.L_series[1][i, :]
    pred = predict_trajectory_homogeneous(L0, T, final_params)
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)

    lines!(ax1, true_traj[1, :], true_traj[2, :], color=colors[idx], linewidth=2,
           label=(idx == 1 ? "True" : nothing))
    lines!(ax1, pred[1, :], pred[2, :], color=colors[idx], linewidth=2, linestyle=:dash,
           label=(idx == 1 ? "Predicted" : nothing))
end

axislegend(ax1, position=:lb)
xlims!(ax1, 0, 1)
ylims!(ax1, 0, 1)

# Learned vector field
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                       title="Learned Vector Field f(L)", aspect=1)

# Grid of points
xs = range(0.1, 0.9, length=10)
ys = range(0.1, 0.9, length=10)

for x in xs
    for y in ys
        L = Float32.([x, y])
        dL, _ = model(reshape(L, :, 1), final_params, state)
        dL = vec(dL)

        # Scale arrows for visibility
        scale = 2.0
        arrows!(ax2, [x], [y], [scale * dL[1]], [scale * dL[2]],
                color=:blue, linewidth=1.5)
    end
end

# Add true vector field for comparison (offset slightly for visibility)
for x in xs
    for y in ys
        dL_true = data.true_dynamics([x, y])
        scale = 2.0
        arrows!(ax2, [x + 0.02], [y + 0.02], [scale * dL_true[1]], [scale * dL_true[2]],
                color=:red, linewidth=1)
    end
end

xlims!(ax2, 0, 1)
ylims!(ax2, 0, 1)

save("results/homogeneous_dynamics.pdf", fig)
println("\nSaved: results/homogeneous_dynamics.pdf")

# =============================================================================
# Step 6: Symbolic Regression to find closed-form equations
# =============================================================================
println("\n6. Running symbolic regression...")

using SymbolicRegression
import SymbolicRegression: calculate_pareto_frontier, compute_complexity

# Use the same input-output pairs as training
# X is d × N, Y is d × N
# For symbolic regression, we need Float64
X_sr = Float64.(X)
Y_sr = Float64.(Y)

# Configure symbolic regression
# True dynamics: dL/dt = omega * [-L[2], L[1]]
# So we expect: dL1/dt = -a*L2, dL2/dt = a*L1 where a ≈ omega
options = Options(
    populations=20,
    binary_operators=[+, -, *],
    unary_operators=[],  # No need for trig - it's just linear
    should_optimize_constants=true,
    should_simplify=true,
    maxsize=10,  # Keep expressions simple
    timeout_in_seconds=60.0  # Don't spend too long
)

println("  Searching for symbolic equations...")
println("  (True dynamics: dL/dt = omega * [-L2, L1], omega=" * string(data.omega) * ")")

hall_of_fame = equation_search(
    X_sr, Y_sr;
    niterations=30,
    options=options,
    parallelism=:multithreading
)

# Get best equations
dominating = calculate_pareto_frontier.(hall_of_fame)

println("\n  Found equations:")
for (dim, front) in enumerate(dominating)
    if !isempty(front)
        # Pick best by loss among reasonable complexity
        best_idx = argmin([f.loss for f in front])
        best = front[best_idx]
        complexity = compute_complexity(best.tree, options)
        println("    dL" * string(dim) * "/dt = " * string(best.tree) *
                " (complexity=" * string(complexity) * ", loss=" * string(round(best.loss; digits=6)) * ")")
    end
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Homogeneous Dynamics")
println("=" ^ 60)
println("Training samples: " * string(size(X, 2)) * " (pooled from all nodes)")
println("Mean prediction error: " * string(round(mean(errors), digits=4)))
println("\nKey insight: Learning ONE shared dynamics function is much")
println("more data-efficient than learning per-node dynamics.")
println("\nSymbolic regression can recover interpretable equations")
println("when all nodes follow the same dynamics.")
