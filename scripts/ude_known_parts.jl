#!/usr/bin/env julia
"""
Universal Differential Equations (UDE) with Known Parts

Instead of learning the entire dynamics dL/dt = NN(L), we specify
known parts of the dynamics and only learn the unknown residual:

    dL/dt = f_known(L) + NN(L)

This can help when:
1. We have prior knowledge about the system (e.g., rotation, attraction)
2. The unknown part is simpler than the full dynamics
3. We want interpretable results by isolating the learned component

Scenario: Nodes rotate around origin (known) + attraction/repulsion (unknown)
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
println("UDE: Known Rotation + Learned Radial Dynamics")
println("=" ^ 60)

# =============================================================================
# Generate data: Rotation (known) + Radial oscillation (unknown)
# =============================================================================

function generate_rotating_oscillating_community(; n::Int=30, d::Int=2, T::Int=50,
                                                   omega::Float64=0.1,
                                                   r_target::Float64=0.7,
                                                   k_radial::Float64=0.05,
                                                   seed::Int=1234)
    """
    Nodes rotate AND oscillate radially toward a target radius.

    True dynamics:
        dL/dt = omega * [-L[2], L[1]]  +  k * (r_target - ||L||) * L_hat
              = (rotation, KNOWN)       +  (radial spring, UNKNOWN)

    where L_hat = L / ||L|| is the unit vector direction.

    IMPORTANT: No angle wrapping - continuous rotation for smooth derivatives.
    """
    rng = Random.MersenneTwister(seed)

    # Initial positions: random angles in first quadrant, perturbed radii
    # Center angles away from boundaries for the short simulation
    theta0 = rand(rng, n) .* (pi/6) .+ pi/6  # Between π/6 and π/3
    radii0 = r_target .+ 0.2 .* randn(rng, n)  # Larger perturbation for more radial dynamics
    radii0 = clamp.(radii0, 0.4, 0.95)

    # Simulate true dynamics using actual ODE integration (more accurate)
    L_series = Vector{Matrix{Float64}}(undef, T)

    # State: [theta_1, r_1, theta_2, r_2, ...]
    state = vcat([[theta0[i], radii0[i]] for i in 1:n]...)

    dt = 1.0
    for t in 1:T
        L_t = zeros(n, d)
        for i in 1:n
            theta_i = state[2*(i-1) + 1]
            r_i = state[2*(i-1) + 2]

            # NO wrapping - continuous rotation
            L_t[i, :] = r_i * [cos(theta_i), sin(theta_i)]
        end
        L_series[t] = L_t

        # Update state (Euler step)
        for i in 1:n
            theta_i = state[2*(i-1) + 1]
            r_i = state[2*(i-1) + 2]

            # dtheta/dt = omega (rotation)
            state[2*(i-1) + 1] = theta_i + omega * dt

            # dr/dt = k * (r_target - r) (radial spring toward target)
            state[2*(i-1) + 2] = r_i + k_radial * (r_target - r_i) * dt
        end
    end

    return (
        L_series = L_series,
        n = n,
        d = d,
        T = T,
        omega = omega,
        r_target = r_target,
        k_radial = k_radial,
        # The known part (rotation)
        known_dynamics = (L) -> omega * [-L[2], L[1]],
        # The unknown part (radial spring) - what we want to learn
        unknown_dynamics = (L) -> begin
            r = norm(L)
            if r < 1e-6
                return zeros(2)
            end
            L_hat = L ./ r
            return k_radial * (r_target - r) .* L_hat
        end
    )
end

println("\n1. Generating data with rotation + radial dynamics...")
# k_radial=0.15 makes unknown part ~30% of known part (was 0.5% with k=0.05)
data = generate_rotating_oscillating_community(n=30, T=50, omega=0.1, k_radial=0.15)
n, d, T = data.n, data.d, data.T

println("  n=" * string(n) * ", d=" * string(d) * ", T=" * string(T))
println("  Known: rotation with omega=" * string(data.omega))
println("  Unknown: radial spring with k=" * string(data.k_radial) * ", r_target=" * string(data.r_target))

# =============================================================================
# Prepare training data
# =============================================================================
println("\n2. Preparing training data...")

L_inputs = Vector{Vector{Float64}}()
dL_targets = Vector{Vector{Float64}}()
dL_unknown_targets = Vector{Vector{Float64}}()  # Residual after removing known part

# CRITICAL: For UDE, we need to use finite-difference version of known dynamics
# The instantaneous rotation derivative omega*[-L2, L1] doesn't match finite difference!
# Finite diff for rotation: R(omega*dt)*L - L where R is rotation matrix
omega = data.omega
dt = 1.0
cos_omega = cos(omega * dt)
sin_omega = sin(omega * dt)

function rotation_finite_diff(L)
    # R(theta) * L - L where theta = omega * dt
    L_rotated = [cos_omega * L[1] - sin_omega * L[2],
                 sin_omega * L[1] + cos_omega * L[2]]
    return L_rotated - L
end

for t in 1:(T-1)
    for i in 1:n
        L_now = data.L_series[t][i, :]
        L_next = data.L_series[t+1][i, :]

        # Total derivative (finite difference)
        dL_total = L_next - L_now

        # Known part - USE FINITE DIFFERENCE VERSION for training!
        dL_known = rotation_finite_diff(L_now)

        # Unknown part = total - known (now both are finite differences)
        dL_unknown = dL_total - dL_known

        push!(L_inputs, L_now)
        push!(dL_targets, dL_total)
        push!(dL_unknown_targets, dL_unknown)
    end
end

X = Float32.(hcat(L_inputs...))           # d × N
Y_total = Float32.(hcat(dL_targets...))   # d × N
Y_unknown = Float32.(hcat(dL_unknown_targets...))  # d × N

mean_total = mean(norm.(eachcol(Y_total)))
mean_known = mean(norm.(eachcol(Float32.(hcat([data.known_dynamics(L) for L in L_inputs]...)))))
mean_unknown = mean(norm.(eachcol(Y_unknown)))

println("  Training samples: " * string(size(X, 2)))
println("  Mean ||dL_total||: " * string(round(mean_total; digits=4)))
println("  Mean ||dL_known||: " * string(round(mean_known; digits=4)))
println("  Mean ||dL_unknown|| (residual): " * string(round(mean_unknown; digits=4)))
println("  Ratio unknown/known: " * string(round(100 * mean_unknown / mean_known; digits=1)) * "%")

# =============================================================================
# Approach 1: Learn FULL dynamics (baseline)
# =============================================================================
println("\n3. Training NN to learn FULL dynamics (baseline)...")

rng = Random.Xoshiro(1234)

model_full = Lux.Chain(
    Lux.Dense(d, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, d)
)

params_full, state_full = Lux.setup(rng, model_full)
params_full_ca = ComponentArray(params_full)

function loss_full(p, _)
    pred, _ = model_full(X, p, state_full)
    return sum(abs2, Y_total .- pred) / size(X, 2)
end

optf_full = Optimization.OptimizationFunction(loss_full, Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, params_full_ca, nothing)

result_full = Optimization.solve(optprob_full, OptimizationOptimisers.Adam(0.01); maxiters=2000,
    callback=(st, l) -> (st.iter % 500 == 0 && println("    iter " * string(st.iter) * ": loss=" * string(round(l, digits=6))); false))

params_full_trained = result_full.u
loss_full_final = loss_full(params_full_trained, nothing)
println("  Final loss (full dynamics): " * string(round(loss_full_final, digits=6)))

# =============================================================================
# Approach 2: UDE - Learn only UNKNOWN part (residual)
# =============================================================================
println("\n4. Training UDE to learn only UNKNOWN residual...")

model_ude = Lux.Chain(
    Lux.Dense(d, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, d)
)

params_ude, state_ude = Lux.setup(rng, model_ude)
params_ude_ca = ComponentArray(params_ude)

function loss_ude(p, _)
    pred, _ = model_ude(X, p, state_ude)
    return sum(abs2, Y_unknown .- pred) / size(X, 2)
end

optf_ude = Optimization.OptimizationFunction(loss_ude, Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, params_ude_ca, nothing)

result_ude = Optimization.solve(optprob_ude, OptimizationOptimisers.Adam(0.01); maxiters=2000,
    callback=(st, l) -> (st.iter % 500 == 0 && println("    iter " * string(st.iter) * ": loss=" * string(round(l, digits=6))); false))

params_ude_trained = result_ude.u
loss_ude_final = loss_ude(params_ude_trained, nothing)
println("  Final loss (UDE residual): " * string(round(loss_ude_final, digits=6)))

# =============================================================================
# Compare predictions via ODE integration
# =============================================================================
println("\n5. Comparing trajectory predictions...")

omega = data.omega

function predict_trajectory_full(L0::Vector{Float64}, T::Int, params)
    function dudt(u, p, t)
        pred, _ = model_full(reshape(Float32.(u), :, 1), p, state_full)
        return vec(pred)
    end

    prob = ODEProblem(dudt, Float32.(L0), (0.0f0, Float32(T-1)), params)
    sol = solve(prob, Tsit5(); saveat=0:T-1)
    return Array(sol)
end

function predict_trajectory_ude(L0::Vector{Float64}, T::Int, params)
    # Use DISCRETE steps matching training (not ODE integration)
    # This ensures consistency: training used finite differences with dt=1
    trajectory = zeros(Float32, 2, T)
    trajectory[:, 1] = Float32.(L0)

    for t in 1:(T-1)
        u = trajectory[:, t]

        # Known part: rotation finite difference (matches training)
        dL_known = [cos_omega * u[1] - sin_omega * u[2] - u[1],
                    sin_omega * u[1] + cos_omega * u[2] - u[2]]

        # Learned part: unknown residual
        dL_unknown, _ = model_ude(reshape(u, :, 1), params, state_ude)

        # Discrete update: L[t+1] = L[t] + dL_known + dL_unknown
        trajectory[:, t+1] = u .+ Float32.(dL_known) .+ vec(dL_unknown)
    end

    return trajectory
end

# Test on a few nodes
test_nodes = [1, 10, 20, 30]
errors_full = Float64[]
errors_ude = Float64[]

for i in test_nodes
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)

    pred_full = predict_trajectory_full(L0, T, params_full_trained)
    pred_ude = predict_trajectory_ude(L0, T, params_ude_trained)

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T])

    push!(errors_full, err_full)
    push!(errors_ude, err_ude)
end

println("  Mean prediction error (FULL NN):  " * string(round(mean(errors_full), digits=4)))
println("  Mean prediction error (UDE):      " * string(round(mean(errors_ude), digits=4)))
println("  Improvement: " * string(round(100 * (1 - mean(errors_ude) / mean(errors_full)), digits=1)) * "%")

# =============================================================================
# Visualize
# =============================================================================
println("\n6. Creating visualization...")

fig = Figure(size=(1400, 600))

# Panel 1: Phase portraits for both methods
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                       title="Full NN: Learned All Dynamics", aspect=1)

theta_arc = range(0, pi/2, length=100)
lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash)

colors = [:blue, :red, :green, :purple]
for (idx, i) in enumerate(test_nodes)
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)
    pred_full = predict_trajectory_full(L0, T, params_full_trained)

    lines!(ax1, true_traj[1, :], true_traj[2, :], color=colors[idx], linewidth=2)
    lines!(ax1, pred_full[1, :], pred_full[2, :], color=colors[idx], linewidth=2, linestyle=:dash)
end
xlims!(ax1, 0, 1)
ylims!(ax1, 0, 1)

ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                       title="UDE: Known Rotation + Learned Residual", aspect=1)

lines!(ax2, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash)

for (idx, i) in enumerate(test_nodes)
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)
    pred_ude = predict_trajectory_ude(L0, T, params_ude_trained)

    lines!(ax2, true_traj[1, :], true_traj[2, :], color=colors[idx], linewidth=2,
           label=(idx == 1 ? "True" : nothing))
    lines!(ax2, pred_ude[1, :], pred_ude[2, :], color=colors[idx], linewidth=2, linestyle=:dash,
           label=(idx == 1 ? "UDE" : nothing))
end
axislegend(ax2, position=:lb)
xlims!(ax2, 0, 1)
ylims!(ax2, 0, 1)

# Panel 3: Learned residual vector field
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="Dim 1", ylabel="Dim 2",
                       title="Learned Residual (Radial Dynamics)", aspect=1)

lines!(ax3, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash)

# Circle at target radius
theta_full = range(0, 2*pi, length=100)
lines!(ax3, data.r_target .* cos.(theta_full), data.r_target .* sin.(theta_full),
       color=:green, linestyle=:dot, linewidth=2, label="Target r=" * string(data.r_target))

xs = range(0.2, 0.9, length=8)
ys = range(0.2, 0.9, length=8)

for x in xs
    for y in ys
        L = Float32.([x, y])
        dL_learned, _ = model_ude(reshape(L, :, 1), params_ude_trained, state_ude)
        dL_learned = vec(dL_learned)

        dL_true = data.unknown_dynamics([x, y])

        scale = 3.0
        arrows!(ax3, [x], [y], [scale * dL_learned[1]], [scale * dL_learned[2]],
                color=:blue, linewidth=1.5)
        arrows!(ax3, [x + 0.02], [y + 0.02], [scale * dL_true[1]], [scale * dL_true[2]],
                color=:red, linewidth=1)
    end
end

xlims!(ax3, 0, 1)
ylims!(ax3, 0, 1)

save("results/ude_known_rotation.pdf", fig)
println("\nSaved: results/ude_known_rotation.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: UDE with Known Rotation")
println("=" ^ 60)
println("Full NN loss:           " * string(round(loss_full_final, digits=6)))
println("UDE residual loss:      " * string(round(loss_ude_final, digits=6)))
println("\nPrediction errors (mean over test nodes):")
println("  Full NN:  " * string(round(mean(errors_full), digits=4)))
println("  UDE:      " * string(round(mean(errors_ude), digits=4)))
println("\nKey insight: When we KNOW part of the dynamics (rotation),")
println("the UDE only needs to learn the simpler residual (radial spring),")
println("leading to better generalization and interpretability.")
