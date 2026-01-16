#!/usr/bin/env julia
"""
UDE with Proper SciML Training

Key insight from SciML tutorials:
- Train through ODE integration, NOT on derivative residuals
- Loss = |predicted_trajectory - true_trajectory|²
- Backpropagate through ODE solver using adjoint sensitivity

This is fundamentally different from our previous approach which trained
the NN on pointwise residuals and then hoped it worked when integrated.

Usage:
  julia --project scripts/ude_proper.jl           # Full run (train + visualize)
  julia --project scripts/ude_proper.jl --viz     # Visualization only (load saved)
"""

using Pkg
Pkg.activate(".")

# Check for visualization-only mode
VIZ_ONLY = "--viz" in ARGS || "-v" in ARGS

using LinearAlgebra
using Random
using Statistics
using CairoMakie
using Lux
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using ComponentArrays
using Zygote
using Serialization

# File paths for saved results
const RESULTS_DIR = "results"
const SAVE_FILE = joinpath(RESULTS_DIR, "ude_proper_results.jls")

println("=" ^ 60)
println("UDE with Proper SciML Training")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# Generate data: Rotation (known) + Radial attraction (unknown)
# =============================================================================

function generate_dynamics(; n::Int=20, d::Int=2, T::Int=30,
                            omega::Float64=0.08,      # Known: rotation
                            k_radial::Float64=0.12,   # Unknown: radial attraction
                            r_target::Float64=0.7,
                            seed::Int=1234)
    """
    True dynamics: dL/dt = rotation(L) + radial(L)

    Known part (rotation):   omega * [-L[2], L[1]]
    Unknown part (radial):   -k * (1 - r_target/||L||) * L
    """
    rng = Random.MersenneTwister(seed)

    # Initial positions: mix of inside and outside r_target
    theta0 = rand(rng, n) .* (pi/3) .+ pi/6
    radii0 = zeros(n)
    for i in 1:n
        if rand(rng) < 0.5
            radii0[i] = 0.4 + 0.15 * rand(rng)  # Inside: 0.4-0.55
        else
            radii0[i] = 0.85 + 0.1 * rand(rng)  # Outside: 0.85-0.95
        end
    end

    # Generate trajectories by solving the true ODE
    L_series = Vector{Matrix{Float64}}(undef, T)

    function true_dynamics!(du, u, p, t)
        for i in 1:n
            L = @view u[(2*(i-1)+1):(2*i)]
            dL = @view du[(2*(i-1)+1):(2*i)]

            r = sqrt(L[1]^2 + L[2]^2)

            # Rotation (known)
            dL[1] = omega * (-L[2])
            dL[2] = omega * L[1]

            # Radial attraction (unknown)
            if r > 1e-6
                radial_factor = -k_radial * (1.0 - r_target / r)
                dL[1] += radial_factor * L[1]
                dL[2] += radial_factor * L[2]
            end
        end
    end

    # Initial state vector
    u0 = zeros(2 * n)
    for i in 1:n
        u0[2*(i-1)+1] = radii0[i] * cos(theta0[i])
        u0[2*(i-1)+2] = radii0[i] * sin(theta0[i])
    end

    # Solve true ODE
    prob = ODEProblem(true_dynamics!, u0, (0.0, Float64(T-1)))
    sol = solve(prob, Tsit5(), saveat=0:(T-1))

    # Convert to L_series format
    for t in 1:T
        L_t = zeros(n, d)
        for i in 1:n
            L_t[i, 1] = sol.u[t][2*(i-1)+1]
            L_t[i, 2] = sol.u[t][2*(i-1)+2]
        end
        L_series[t] = L_t
    end

    return (
        L_series = L_series,
        n = n,
        d = d,
        T = T,
        omega = omega,
        k_radial = k_radial,
        r_target = r_target,
        u0 = u0
    )
end

println("\n1. Generating dynamics data...")
data = generate_dynamics(n=20, T=30, omega=0.08, k_radial=0.12)
n, d, T = data.n, data.d, data.T

println("  n=" * string(n) * ", d=" * string(d) * ", T=" * string(T))
println("  Known: rotation with omega=" * string(data.omega))
println("  Unknown: radial attraction with k=" * string(data.k_radial))

# Compute magnitude of each component for reference
sample_magnitudes_rot = Float64[]
sample_magnitudes_rad = Float64[]
for t in 1:(T-1)
    for i in 1:n
        L = data.L_series[t][i, :]
        r = norm(L)
        rot_mag = data.omega * r
        rad_mag = abs(data.k_radial * (1.0 - data.r_target / r)) * r
        push!(sample_magnitudes_rot, rot_mag)
        push!(sample_magnitudes_rad, rad_mag)
    end
end
println("  Mean ||rotation||: " * string(round(mean(sample_magnitudes_rot), digits=4)))
println("  Mean ||radial||: " * string(round(mean(sample_magnitudes_rad), digits=4)))
println("  Ratio radial/rotation: " * string(round(100 * mean(sample_magnitudes_rad) / mean(sample_magnitudes_rot), digits=1)) * "%")

# =============================================================================
# Setup neural network and models (always needed for dynamics functions)
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# Neural network for learning unknown dynamics (radial attraction)
# Input: 2D position, Output: 2D velocity contribution
nn_unknown = Lux.Chain(
    Lux.Dense(2, 24, tanh),
    Lux.Dense(24, 24, tanh),
    Lux.Dense(24, 2)
)

# Neural network for learning full dynamics (baseline)
nn_full = Lux.Chain(
    Lux.Dense(2, 24, tanh),
    Lux.Dense(24, 24, tanh),
    Lux.Dense(24, 2)
)

ps_unknown, st_unknown = Lux.setup(rng, nn_unknown)
ps_full, st_full = Lux.setup(rng, nn_full)

# Convert to ComponentArrays
ps_unknown_ca = ComponentArray(ps_unknown)
ps_full_ca = ComponentArray(ps_full)

println("  NN parameters (unknown): " * string(length(ps_unknown_ca)))
println("  NN parameters (full): " * string(length(ps_full_ca)))

# =============================================================================
# Prepare training data
# =============================================================================
println("\n3. Preparing training data...")

# Use a subset of nodes for training (leave some for validation)
train_nodes = 1:15
val_nodes = 16:20

# Extract training trajectories as a single matrix for each node
# Shape: (2, T) for each node
train_data = Dict{Int, Matrix{Float32}}()
for i in train_nodes
    traj = zeros(Float32, 2, T)
    for t in 1:T
        traj[:, t] = Float32.(data.L_series[t][i, :])
    end
    train_data[i] = traj
end

println("  Training nodes: " * string(length(train_nodes)))
println("  Validation nodes: " * string(length(val_nodes)))

# =============================================================================
# Define UDE dynamics and prediction functions
# =============================================================================

omega = data.omega  # Known parameter

# UDE dynamics: known rotation + learned unknown (OUT-OF-PLACE for Zygote compatibility)
# IMPORTANT: Must be type-stable - return same type as u
function ude_dynamics(u, p, t)
    # Known part: rotation (use eltype(u) for type stability)
    T = eltype(u)
    omega_T = T(omega)
    du_known = [omega_T * (-u[2]), omega_T * u[1]]

    # Unknown part: learned by NN
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_unknown(u_input, p, st_unknown)

    return du_known .+ vec(nn_out)
end

# Full NN dynamics (baseline) - OUT-OF-PLACE
function full_dynamics(u, p, t)
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# Prediction function for a single node
function predict_node(u0, tspan, tsteps, p, dynamics_fn; sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps, sensealg=sensealg, abstol=1f-6, reltol=1f-6)
    return Array(sol)
end

# =============================================================================
# Loss functions - KEY CHANGE: Loss on trajectory, not derivatives!
# =============================================================================
println("\n4. Defining loss functions (trajectory-based)...")

tspan = (0.0f0, Float32(T-1))
tsteps = Float32.(0:(T-1))

# Loss for UDE
function loss_ude(p)
    total_loss = 0.0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]

        # Predict trajectory by solving ODE with NN inside
        pred = predict_node(u0, tspan, tsteps, p, ude_dynamics)

        # Loss on trajectory
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T)
end

# Loss for full NN
function loss_full(p)
    total_loss = 0.0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]

        pred = predict_node(u0, tspan, tsteps, p, full_dynamics)
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T)
end

# Test initial losses (only if not loading)
if !VIZ_ONLY || !isfile(SAVE_FILE)
    println("  Initial loss (UDE): " * string(round(loss_ude(ps_unknown_ca), digits=4)))
    println("  Initial loss (Full): " * string(round(loss_full(ps_full_ca), digits=4)))
end

# =============================================================================
# Load saved results OR run training
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n5-7. Loading saved results from " * SAVE_FILE * "...")
    saved = deserialize(SAVE_FILE)

    # Load metrics
    errors_full_train = saved["errors_full_train"]
    errors_ude_train = saved["errors_ude_train"]
    errors_full_val = saved["errors_full_val"]
    errors_ude_val = saved["errors_ude_val"]
    loss_full_final = saved["loss_full_final"]
    loss_ude_final = saved["loss_ude_final"]

    # Load data parameters
    r_target_saved = saved["r_target"]

    # Load precomputed trajectories
    viz_train_nodes = saved["viz_train_nodes"]
    viz_val_node = saved["viz_val_node"]
    viz_train_true = saved["viz_train_true"]
    viz_train_full = saved["viz_train_full"]
    viz_train_ude = saved["viz_train_ude"]
    viz_val_true = saved["viz_val_true"]
    viz_val_full = saved["viz_val_full"]
    viz_val_ude = saved["viz_val_ude"]

    improvement_train = 100 * (1 - mean(errors_ude_train) / mean(errors_full_train))
    improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))

    println("  Loaded!")
    println("\n  Training losses (from saved):")
    println("    Full NN: " * string(round(loss_full_final, digits=6)))
    println("    UDE:     " * string(round(loss_ude_final, digits=6)))
    println("\n  Validation errors (from saved):")
    println("    Full NN: " * string(round(mean(errors_full_val), digits=4)))
    println("    UDE:     " * string(round(mean(errors_ude_val), digits=4)))
    println("    Improvement: " * string(round(improvement_val, digits=1)) * "%")

else  # Run training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file found. Running full training...")
end

# =============================================================================
# Training with two-stage optimization (ADAM then BFGS)
# =============================================================================
println("\n5. Training Full NN (baseline)...")

# Callback for monitoring
iter_full = Ref(0)
function callback_full(state, l)
    iter_full[] += 1
    if iter_full[] % 50 == 0
        println("    iter " * string(iter_full[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

# Stage 1: ADAM
println("  Stage 1: ADAM optimization...")
optf_full = Optimization.OptimizationFunction((p, _) -> loss_full(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full_adam = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_full
)

# Stage 2: BFGS
println("  Stage 2: BFGS refinement...")
iter_full[] = 300
optprob_full2 = Optimization.OptimizationProblem(optf_full, result_full_adam.u)

result_full = try
    Optimization.solve(
        optprob_full2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=100,
        callback=callback_full,
        allow_f_increases=false
    )
catch e
    println("    BFGS failed, using ADAM result: " * string(e))
    result_full_adam
end

ps_full_trained = result_full.u
loss_full_final = loss_full(ps_full_trained)
println("  Final loss (Full NN): " * string(round(loss_full_final, digits=6)))

# =============================================================================
# Training UDE
# =============================================================================
println("\n6. Training UDE...")

iter_ude = Ref(0)
function callback_ude(state, l)
    iter_ude[] += 1
    if iter_ude[] % 50 == 0
        println("    iter " * string(iter_ude[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

# Stage 1: ADAM
println("  Stage 1: ADAM optimization...")
optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude(p), Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_unknown_ca)

result_ude_adam = Optimization.solve(
    optprob_ude,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_ude
)

# Stage 2: BFGS
println("  Stage 2: BFGS refinement...")
iter_ude[] = 300
optprob_ude2 = Optimization.OptimizationProblem(optf_ude, result_ude_adam.u)

result_ude = try
    Optimization.solve(
        optprob_ude2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=100,
        callback=callback_ude,
        allow_f_increases=false
    )
catch e
    println("    BFGS failed, using ADAM result: " * string(e))
    result_ude_adam
end

ps_ude_trained = result_ude.u
loss_ude_final = loss_ude(ps_ude_trained)
println("  Final loss (UDE): " * string(round(loss_ude_final, digits=6)))

# =============================================================================
# Evaluate on validation set
# =============================================================================
println("\n7. Evaluating on validation nodes...")

errors_full_train = Float64[]
errors_ude_train = Float64[]
errors_full_val = Float64[]
errors_ude_val = Float64[]

for i in 1:n
    u0 = Float32.(data.L_series[1][i, :])
    true_traj = hcat([Float32.(data.L_series[t][i, :]) for t in 1:T]...)

    pred_full = predict_node(u0, tspan, tsteps, ps_full_trained, full_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    pred_ude = predict_node(u0, tspan, tsteps, ps_ude_trained, ude_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T])

    if i in train_nodes
        push!(errors_full_train, err_full)
        push!(errors_ude_train, err_ude)
    else
        push!(errors_full_val, err_full)
        push!(errors_ude_val, err_ude)
    end
end

println("\n  Training set errors:")
println("    Full NN: " * string(round(mean(errors_full_train), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_train), digits=4)))

println("\n  Validation set errors:")
println("    Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_val), digits=4)))

improvement_train = 100 * (1 - mean(errors_ude_train) / mean(errors_full_train))
improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))

println("\n  Improvement (UDE vs Full):")
println("    Training:   " * string(round(improvement_train, digits=1)) * "%")
println("    Validation: " * string(round(improvement_val, digits=1)) * "%")

# =============================================================================
# Save results for later visualization (precompute all trajectories)
# =============================================================================
if !VIZ_ONLY
    println("\n  Precomputing trajectories for visualization...")

    # Nodes to visualize
    viz_train_nodes = [1, 5, 10, 15]
    viz_val_node = first(val_nodes)

    # Precompute trajectories for training nodes
    viz_train_true = Dict{Int, Matrix{Float32}}()
    viz_train_full = Dict{Int, Matrix{Float32}}()
    viz_train_ude = Dict{Int, Matrix{Float32}}()

    for i in viz_train_nodes
        u0 = Float32.(data.L_series[1][i, :])
        viz_train_true[i] = hcat([Float32.(data.L_series[t][i, :]) for t in 1:T]...)
        viz_train_full[i] = predict_node(u0, tspan, tsteps, ps_full_trained, full_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        viz_train_ude[i] = predict_node(u0, tspan, tsteps, ps_ude_trained, ude_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    end

    # Precompute trajectories for validation node
    u0_val = Float32.(data.L_series[1][viz_val_node, :])
    viz_val_true = hcat([Float32.(data.L_series[t][viz_val_node, :]) for t in 1:T]...)
    viz_val_full = predict_node(u0_val, tspan, tsteps, ps_full_trained, full_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    viz_val_ude = predict_node(u0_val, tspan, tsteps, ps_ude_trained, ude_dynamics, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

    println("  Saving results to " * SAVE_FILE * "...")
    mkpath(RESULTS_DIR)
    serialize(SAVE_FILE, Dict(
        # Metrics
        "errors_full_train" => errors_full_train,
        "errors_ude_train" => errors_ude_train,
        "errors_full_val" => errors_full_val,
        "errors_ude_val" => errors_ude_val,
        "loss_full_final" => loss_full_final,
        "loss_ude_final" => loss_ude_final,
        # Data parameters for reference circles
        "r_target" => data.r_target,
        "omega" => data.omega,
        "k_radial" => data.k_radial,
        # Precomputed trajectories for visualization
        "viz_train_nodes" => viz_train_nodes,
        "viz_val_node" => viz_val_node,
        "viz_train_true" => viz_train_true,
        "viz_train_full" => viz_train_full,
        "viz_train_ude" => viz_train_ude,
        "viz_val_true" => viz_val_true,
        "viz_val_full" => viz_val_full,
        "viz_val_ude" => viz_val_ude
    ))
    println("  Saved!")
end

end  # end of else block (training vs loading)

# =============================================================================
# Visualize learned dynamics (using precomputed trajectories)
# =============================================================================
println("\n8. Creating visualization...")

# Get r_target for reference circles (from saved or from data)
r_target_viz = VIZ_ONLY && isfile(SAVE_FILE) ? r_target_saved : data.r_target

fig = Figure(size=(1600, 500))

# Draw reference circles (full circle since trajectories rotate beyond first quadrant)
theta_full = range(0, 2*pi, length=200)

# Panel 1: Training trajectories - Full NN
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                       title="Full NN (training nodes)", aspect=1)

lines!(ax1, cos.(theta_full), sin.(theta_full), color=:gray, linestyle=:dash, alpha=0.3)
lines!(ax1, r_target_viz .* cos.(theta_full), r_target_viz .* sin.(theta_full),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

# Plot main node (node 5) for legend
main_node = 5
true_traj_main = viz_train_true[main_node]
pred_full_main = viz_train_full[main_node]

lines!(ax1, true_traj_main[1, :], true_traj_main[2, :], color=:black, linewidth=2, label="True")
lines!(ax1, pred_full_main[1, :], pred_full_main[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Predicted")
scatter!(ax1, [true_traj_main[1, 1]], [true_traj_main[2, 1]], color=:black, markersize=12, marker=:circle)
scatter!(ax1, [true_traj_main[1, end]], [true_traj_main[2, end]], color=:black, markersize=12, marker=:star5)

# Add other nodes in lighter colors (no legend)
for i in viz_train_nodes
    i == main_node && continue
    lines!(ax1, viz_train_true[i][1, :], viz_train_true[i][2, :], color=(:black, 0.3), linewidth=1)
    lines!(ax1, viz_train_full[i][1, :], viz_train_full[i][2, :], color=(:blue, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax1, position=:lt)
xlims!(ax1, -1, 1)
ylims!(ax1, -0.2, 1)

# Panel 2: UDE trajectories
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                       title="UDE (training nodes)", aspect=1)

lines!(ax2, cos.(theta_full), sin.(theta_full), color=:gray, linestyle=:dash, alpha=0.3)
lines!(ax2, r_target_viz .* cos.(theta_full), r_target_viz .* sin.(theta_full),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

# Plot main node for legend
pred_ude_main = viz_train_ude[main_node]

lines!(ax2, true_traj_main[1, :], true_traj_main[2, :], color=:black, linewidth=2, label="True")
lines!(ax2, pred_ude_main[1, :], pred_ude_main[2, :], color=:red, linewidth=2, linestyle=:dash, label="Predicted")
scatter!(ax2, [true_traj_main[1, 1]], [true_traj_main[2, 1]], color=:black, markersize=12, marker=:circle)
scatter!(ax2, [true_traj_main[1, end]], [true_traj_main[2, end]], color=:black, markersize=12, marker=:star5)

# Add other nodes in lighter colors
for i in viz_train_nodes
    i == main_node && continue
    lines!(ax2, viz_train_true[i][1, :], viz_train_true[i][2, :], color=(:black, 0.3), linewidth=1)
    lines!(ax2, viz_train_ude[i][1, :], viz_train_ude[i][2, :], color=(:red, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax2, position=:lt)
xlims!(ax2, -1, 1)
ylims!(ax2, -0.2, 1)

# Panel 3: Comparison on validation node
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="Dim 1", ylabel="Dim 2",
                       title="Validation Node Comparison", aspect=1)

lines!(ax3, cos.(theta_full), sin.(theta_full), color=:gray, linestyle=:dash, alpha=0.3)
lines!(ax3, r_target_viz .* cos.(theta_full), r_target_viz .* sin.(theta_full),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

lines!(ax3, viz_val_true[1, :], viz_val_true[2, :], color=:black, linewidth=2.5, label="True")
lines!(ax3, viz_val_full[1, :], viz_val_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, viz_val_ude[1, :], viz_val_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
scatter!(ax3, [viz_val_true[1, 1]], [viz_val_true[2, 1]], color=:black, markersize=12, marker=:circle, label="Start")
scatter!(ax3, [viz_val_true[1, end]], [viz_val_true[2, end]], color=:black, markersize=12, marker=:star5, label="End")

axislegend(ax3, position=:lt)
xlims!(ax3, -1, 1)
ylims!(ax3, -0.2, 1)

save("results/ude_proper.pdf", fig)
println("\nSaved: results/ude_proper.pdf")

# =============================================================================
# Analyze what the NN learned (only in training mode - needs NN)
# =============================================================================
if !VIZ_ONLY
    println("\n9. Analyzing learned unknown dynamics...")

    # Sample points and compare true vs learned unknown dynamics
    test_points = [
        [0.5, 0.5],   # r ≈ 0.71, near target
        [0.3, 0.3],   # r ≈ 0.42, inside target
        [0.6, 0.6],   # r ≈ 0.85, outside target
        [0.4, 0.6],   # r ≈ 0.72, near target
    ]

    println("\n  True vs Learned unknown dynamics:")
    println("  " * "-"^60)
    for pt in test_points
        r = norm(pt)

        # True unknown (radial attraction)
        true_unknown = -data.k_radial * (1.0 - data.r_target / r) .* pt

        # Learned unknown
        pt_input = reshape(Float32.(pt), 2, 1)
        learned_unknown, _ = nn_unknown(pt_input, ps_ude_trained, st_unknown)
        learned_unknown = vec(learned_unknown)

        println("  Point: [" * string(round(pt[1], digits=2)) * ", " * string(round(pt[2], digits=2)) *
                "] (r=" * string(round(r, digits=2)) * ")")
        println("    True:    [" * string(round(true_unknown[1], digits=4)) * ", " *
                string(round(true_unknown[2], digits=4)) * "]")
        println("    Learned: [" * string(round(learned_unknown[1], digits=4)) * ", " *
                string(round(learned_unknown[2], digits=4)) * "]")
        println("    Error:   " * string(round(norm(true_unknown - learned_unknown), digits=4)))
    end
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: UDE with Proper SciML Training")
println("=" ^ 60)
println("\nKey change: Loss computed on ODE trajectories, not derivative residuals")
println("\nTraining losses:")
println("  Full NN: " * string(round(loss_full_final, digits=6)))
println("  UDE:     " * string(round(loss_ude_final, digits=6)))
println("\nPrediction errors (training):")
println("  Full NN: " * string(round(mean(errors_full_train), digits=4)))
println("  UDE:     " * string(round(mean(errors_ude_train), digits=4)))
println("\nPrediction errors (validation):")
println("  Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("  UDE:     " * string(round(mean(errors_ude_val), digits=4)))
println("\nImprovement (positive = UDE better):")
println("  Training:   " * string(round(improvement_train, digits=1)) * "%")
println("  Validation: " * string(round(improvement_val, digits=1)) * "%")

if improvement_val > 0
    println("\n✓ UDE outperforms Full NN on validation set!")
    println("  This demonstrates the benefit of encoding known physics.")
else
    println("\n✗ UDE underperforms - consider:")
    println("  - Multiple shooting for long trajectories")
    println("  - More training iterations")
    println("  - Different network architecture")
end
