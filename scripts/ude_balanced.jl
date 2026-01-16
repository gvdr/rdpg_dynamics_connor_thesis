#!/usr/bin/env julia
"""
UDE with Flipped Known/Unknown Parts

Key insight: Make the DOMINANT dynamics the known part!

Scenario:
- Known: Rotation (dθ/dt = omega) - the dominant, easy-to-specify part
- Unknown: Radial attraction (dr/dt = -k * (r - r_target)) - the smaller residual

This is realistic: you'd encode the obvious dynamics as prior knowledge,
and let the NN learn the subtle corrections.
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
println("UDE: Balanced Known/Unknown (50/50 split)")
println("=" ^ 60)

# =============================================================================
# Generate data: Rotation (known) + Radial attraction (unknown)
# =============================================================================

function generate_balanced_dynamics(; n::Int=30, d::Int=2, T::Int=50,
                                     k_radial::Float64=0.1,  # Unknown: radial attraction
                                     omega::Float64=0.1,      # Known: rotation
                                     r_target::Float64=0.7,
                                     seed::Int=1234)
    """
    Dynamics with FLIPPED known/unknown roles.

    The dominant dynamics (rotation) is now KNOWN - this is realistic since
    you would encode the obvious, dominant dynamics as prior knowledge.

    True dynamics in polar coordinates:
        dθ/dt = omega                    (KNOWN: rotation - dominant)
        dr/dt = -k * (r - r_target)     (UNKNOWN: radial attraction - residual)

    In Cartesian:
        dL/dt = known(L) + unknown(L)

    where:
        known(L) = omega * [-L[2], L[1]]           (rotation)
        unknown(L) = -k * (1 - r_target/||L||) * L (radial attraction)
    """
    rng = Random.MersenneTwister(seed)

    # Initial positions: start FAR from r_target for significant radial dynamics
    theta0 = rand(rng, n) .* (pi/3) .+ pi/6  # Between π/6 and π/2
    # Half start inside (r=0.4-0.5), half start outside (r=0.85-0.95)
    radii0 = zeros(n)
    for i in 1:n
        if rand(rng) < 0.5
            radii0[i] = 0.4 + 0.1 * rand(rng)  # Inside: 0.4-0.5
        else
            radii0[i] = 0.85 + 0.1 * rand(rng)  # Outside: 0.85-0.95
        end
    end

    L_series = Vector{Matrix{Float64}}(undef, T)

    # State: [theta_1, r_1, theta_2, r_2, ...]
    state = vcat([[theta0[i], radii0[i]] for i in 1:n]...)

    dt = 1.0
    for t in 1:T
        L_t = zeros(n, d)
        for i in 1:n
            theta_i = state[2*(i-1) + 1]
            r_i = state[2*(i-1) + 2]
            L_t[i, :] = r_i * [cos(theta_i), sin(theta_i)]
        end
        L_series[t] = L_t

        # Update state (Euler step)
        for i in 1:n
            theta_i = state[2*(i-1) + 1]
            r_i = state[2*(i-1) + 2]

            # dθ/dt = omega (rotation - KNOWN)
            state[2*(i-1) + 1] = theta_i + omega * dt

            # dr/dt = -k * (r - r_target) (radial attraction - UNKNOWN)
            state[2*(i-1) + 2] = r_i - k_radial * (r_i - r_target) * dt
        end
    end

    return (
        L_series = L_series,
        n = n,
        d = d,
        T = T,
        k_radial = k_radial,
        omega = omega,
        r_target = r_target,
        # FLIPPED: The known part is now ROTATION (dominant)
        known_dynamics = (L) -> omega * [-L[2], L[1]],
        # FLIPPED: The unknown part is now RADIAL ATTRACTION (smaller residual)
        unknown_dynamics = (L) -> begin
            r = norm(L)
            if r < 1e-6
                return zeros(2)
            end
            return -k_radial * (1.0 - r_target / r) .* L
        end
    )
end

println("\n1. Generating balanced dynamics data...")
# AGGRESSIVE BALANCING:
# - Low omega (0.03) so rotation is smaller
# - High k_radial (0.5) so radial dynamics is significant
# - Nodes start far from r_target (0.7)
data = generate_balanced_dynamics(n=30, T=50, k_radial=0.5, omega=0.03)
n, d, T = data.n, data.d, data.T

println("  n=" * string(n) * ", d=" * string(d) * ", T=" * string(T))
println("  Known: rotation with omega=" * string(data.omega))
println("  Unknown: radial attraction with k=" * string(data.k_radial) * ", r_target=" * string(data.r_target))

# =============================================================================
# Prepare training data
# =============================================================================
println("\n2. Preparing training data...")

L_inputs = Vector{Vector{Float64}}()
dL_targets = Vector{Vector{Float64}}()
dL_unknown_targets = Vector{Vector{Float64}}()

for t in 1:(T-1)
    for i in 1:n
        L_now = data.L_series[t][i, :]
        L_next = data.L_series[t+1][i, :]

        # Total derivative (finite difference)
        dL_total = L_next - L_now

        # Known part (radial attraction) - using instantaneous for simplicity
        # since radial attraction is well-approximated by Euler
        dL_known = data.known_dynamics(L_now)

        # Unknown part = total - known
        dL_unknown = dL_total - dL_known

        push!(L_inputs, L_now)
        push!(dL_targets, dL_total)
        push!(dL_unknown_targets, dL_unknown)
    end
end

X = Float32.(hcat(L_inputs...))
Y_total = Float32.(hcat(dL_targets...))
Y_unknown = Float32.(hcat(dL_unknown_targets...))

mean_total = mean(norm.(eachcol(Y_total)))
mean_known = mean(norm.(eachcol(Float32.(hcat([data.known_dynamics(L) for L in L_inputs]...)))))
mean_unknown = mean(norm.(eachcol(Y_unknown)))

println("  Training samples: " * string(size(X, 2)))
println("  Mean ||dL_total||: " * string(round(mean_total; digits=4)))
println("  Mean ||dL_known|| (rotation): " * string(round(mean_known; digits=4)))
println("  Mean ||dL_unknown|| (radial): " * string(round(mean_unknown; digits=4)))
println("  Ratio unknown/known: " * string(round(100 * mean_unknown / mean_known; digits=1)) * "%")
println("  Split: " * string(round(100 * mean_known / mean_total; digits=0)) * "% known, " *
        string(round(100 * mean_unknown / mean_total; digits=0)) * "% unknown")

# =============================================================================
# Approach 1: Learn FULL dynamics (baseline)
# =============================================================================
println("\n3. Training NN to learn FULL dynamics (baseline)...")

rng = Random.Xoshiro(1234)

model_full = Lux.Chain(
    Lux.Dense(d, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, d)
)

params_full, state_full = Lux.setup(rng, model_full)
params_full_ca = ComponentArray(params_full)

function loss_full(p, _)
    pred, _ = model_full(X, p, state_full)
    return sum(abs2, Y_total .- pred) / size(X, 2)
end

optf_full = Optimization.OptimizationFunction(loss_full, Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, params_full_ca, nothing)

result_full = Optimization.solve(optprob_full, OptimizationOptimisers.Adam(0.01); maxiters=3000,
    callback=(st, l) -> (st.iter % 500 == 0 && println("    iter " * string(st.iter) * ": loss=" * string(round(l, digits=7))); false))

params_full_trained = result_full.u
loss_full_final = loss_full(params_full_trained, nothing)
println("  Final loss (full dynamics): " * string(round(loss_full_final, sigdigits=3)))

# =============================================================================
# Approach 2: UDE - Learn only UNKNOWN part (radial attraction)
# =============================================================================
println("\n4. Training UDE to learn only UNKNOWN radial dynamics...")

model_ude = Lux.Chain(
    Lux.Dense(d, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, d)
)

params_ude, state_ude = Lux.setup(rng, model_ude)
params_ude_ca = ComponentArray(params_ude)

function loss_ude(p, _)
    pred, _ = model_ude(X, p, state_ude)
    return sum(abs2, Y_unknown .- pred) / size(X, 2)
end

optf_ude = Optimization.OptimizationFunction(loss_ude, Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, params_ude_ca, nothing)

result_ude = Optimization.solve(optprob_ude, OptimizationOptimisers.Adam(0.01); maxiters=3000,
    callback=(st, l) -> (st.iter % 500 == 0 && println("    iter " * string(st.iter) * ": loss=" * string(round(l, digits=7))); false))

params_ude_trained = result_ude.u
loss_ude_final = loss_ude(params_ude_trained, nothing)
println("  Final loss (UDE radial): " * string(round(loss_ude_final, sigdigits=3)))

# =============================================================================
# Compare predictions
# =============================================================================
println("\n5. Comparing trajectory predictions...")

function predict_trajectory_full(L0::Vector{Float64}, T::Int, params)
    function dudt(u, p, t)
        pred, _ = model_full(reshape(Float32.(u), :, 1), p, state_full)
        return vec(pred)
    end

    prob = ODEProblem(dudt, Float32.(L0), (0.0f0, Float32(T-1)), params)
    sol = solve(prob, Tsit5(); saveat=0:T-1)
    return Array(sol)
end

omega = data.omega

function predict_trajectory_ude(L0::Vector{Float64}, T::Int, params)
    function dudt(u, p, t)
        # Known part: rotation (dL/dt = omega * [-u[2], u[1]])
        dL_known = Float32.(omega * [-u[2], u[1]])

        # Learned part: unknown radial attraction
        dL_unknown, _ = model_ude(reshape(Float32.(u), :, 1), p, state_ude)

        return dL_known .+ vec(dL_unknown)
    end

    prob = ODEProblem(dudt, Float32.(L0), (0.0f0, Float32(T-1)), params)
    sol = solve(prob, Tsit5(); saveat=0:T-1)
    return Array(sol)
end

# Test on all nodes
errors_full = Float64[]
errors_ude = Float64[]

for i in 1:n
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)

    pred_full = predict_trajectory_full(L0, T, params_full_trained)
    pred_ude = predict_trajectory_ude(L0, T, params_ude_trained)

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T])

    push!(errors_full, err_full)
    push!(errors_ude, err_ude)
end

mean_err_full = mean(errors_full)
mean_err_ude = mean(errors_ude)
improvement = 100 * (1 - mean_err_ude / mean_err_full)

println("  Mean prediction error (FULL NN):  " * string(round(mean_err_full, digits=4)))
println("  Mean prediction error (UDE):      " * string(round(mean_err_ude, digits=4)))
println("  Improvement: " * string(round(improvement, digits=1)) * "%")

# =============================================================================
# Visualize
# =============================================================================
println("\n6. Creating visualization...")

r_target = data.r_target  # For plotting

fig = Figure(size=(1400, 600))

# Panel 1: Full NN trajectories
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                       title="Full NN (error=" * string(round(mean_err_full, digits=4)) * ")", aspect=1)

theta_arc = range(0, pi/2, length=100)
lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash)
lines!(ax1, r_target .* cos.(theta_arc), r_target .* sin.(theta_arc), color=:green, linestyle=:dot, linewidth=2)

test_nodes = [1, 10, 20, 30]
colors = [:blue, :red, :orange, :purple]
for (idx, i) in enumerate(test_nodes)
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)
    pred_full = predict_trajectory_full(L0, T, params_full_trained)

    lines!(ax1, true_traj[1, :], true_traj[2, :], color=colors[idx], linewidth=2)
    lines!(ax1, pred_full[1, :], pred_full[2, :], color=colors[idx], linewidth=2, linestyle=:dash)
    scatter!(ax1, [true_traj[1, 1]], [true_traj[2, 1]], color=colors[idx], markersize=10)
end
xlims!(ax1, 0, 1)
ylims!(ax1, 0, 1)

# Panel 2: UDE trajectories
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                       title="UDE (error=" * string(round(mean_err_ude, digits=4)) * ")", aspect=1)

lines!(ax2, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash)
lines!(ax2, r_target .* cos.(theta_arc), r_target .* sin.(theta_arc), color=:green, linestyle=:dot, linewidth=2)

for (idx, i) in enumerate(test_nodes)
    L0 = data.L_series[1][i, :]
    true_traj = hcat([data.L_series[t][i, :] for t in 1:T]...)
    pred_ude = predict_trajectory_ude(L0, T, params_ude_trained)

    lines!(ax2, true_traj[1, :], true_traj[2, :], color=colors[idx], linewidth=2,
           label=(idx == 1 ? "True" : nothing))
    lines!(ax2, pred_ude[1, :], pred_ude[2, :], color=colors[idx], linewidth=2, linestyle=:dash,
           label=(idx == 1 ? "UDE" : nothing))
    scatter!(ax2, [true_traj[1, 1]], [true_traj[2, 1]], color=colors[idx], markersize=10)
end
axislegend(ax2, position=:lb)
xlims!(ax2, 0, 1)
ylims!(ax2, 0, 1)

# Panel 3: Error comparison
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="Node index", ylabel="Mean trajectory error",
                       title="Per-node Error Comparison")

barplot!(ax3, 1:n, errors_full, color=:blue, label="Full NN", dodge=1, width=0.4)
barplot!(ax3, (1:n) .+ 0.4, errors_ude, color=:red, label="UDE", dodge=2, width=0.4)
axislegend(ax3, position=:rt)

save("results/ude_balanced.pdf", fig)
println("\nSaved: results/ude_balanced.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: UDE with Balanced Dynamics")
println("=" ^ 60)
println("\nDynamics split:")
println("  Known (rotation): " * string(round(100 * mean_known / mean_total; digits=0)) * "%")
println("  Unknown (radial): " * string(round(100 * mean_unknown / mean_total; digits=0)) * "%")
println("\nTraining losses:")
println("  Full NN: " * string(round(loss_full_final, sigdigits=3)))
println("  UDE:     " * string(round(loss_ude_final, sigdigits=3)))
println("\nPrediction errors:")
println("  Full NN: " * string(round(mean_err_full, digits=4)))
println("  UDE:     " * string(round(mean_err_ude, digits=4)))
println("  Improvement: " * string(round(improvement, digits=1)) * "%")

if improvement > 0
    println("\n✓ UDE outperforms Full NN when known/unknown are balanced!")
else
    println("\n✗ UDE still underperforms - may need different scenario or more training")
end
