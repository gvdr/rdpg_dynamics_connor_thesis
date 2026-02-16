#!/usr/bin/env julia
"""
Example 2: Homogeneous Circulation in B^d_+

Dynamics: All nodes follow the same physics - circulation with boundary confinement.
This demonstrates data pooling efficiency when nodes share dynamics.

Key features:
- AUTONOMOUS dynamics: dx/dt = f(x), no explicit time dependence
- Stays in B^d_+ (non-negative unit ball): x₁ ≥ 0, x₂ ≥ 0, ||x|| ≤ 1
- Known part: circulation ω * [-x₂, x₁]
- Unknown part: boundary confinement + radial equilibrium

Usage:
  julia --project scripts/example2_circulation.jl           # Full run
  julia --project scripts/example2_circulation.jl --viz     # Visualization only
"""

using Pkg
Pkg.activate(".")

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

const RESULTS_DIR = "results"
const SAVE_FILE = joinpath(RESULTS_DIR, "example2_circulation_results.jls")

println("=" ^ 60)
println("Example 2: Homogeneous Circulation in B^d_+")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# True dynamics: Circulation + Boundary Confinement (stays in B^d_+)
# =============================================================================

# Parameters for true dynamics
const TRUE_OMEGA = 0.15f0        # Circulation speed (KNOWN)
const TRUE_ALPHA = 0.5f0         # Boundary repulsion strength (UNKNOWN)
const TRUE_EPSILON = 0.08f0      # Boundary repulsion sharpness (UNKNOWN)
const TRUE_K_RADIAL = 0.1f0      # Radial attraction strength (UNKNOWN)
const TRUE_R_TARGET = 0.6f0      # Target radius (UNKNOWN)

"""
True dynamics function (for data generation).
Autonomous: depends only on position, not time.
"""
function true_dynamics(u::Vector{T}) where T
    x1, x2 = u[1], u[2]
    r = sqrt(x1^2 + x2^2)

    # Known: Circulation (tangent to circles)
    dx_circ = TRUE_OMEGA * T[-x2, x1]

    # Unknown: Soft repulsion from boundaries
    # Repel from x₁=0 edge
    repel_x1 = TRUE_ALPHA * exp(-x1 / TRUE_EPSILON)
    # Repel from x₂=0 edge
    repel_x2 = TRUE_ALPHA * exp(-x2 / TRUE_EPSILON)
    # Repel from unit circle arc
    repel_arc = TRUE_ALPHA * exp((r - 1) / TRUE_EPSILON)

    # Unknown: Radial attraction to target radius
    radial = r > 1f-6 ? -TRUE_K_RADIAL * (r - TRUE_R_TARGET) / r : zero(T)

    # Combine
    dx = dx_circ .+ T[repel_x1, repel_x2] .- (repel_arc / r) .* T[x1, x2]
    dx .+= radial .* T[x1, x2]

    return dx
end

# In-place version for ODE solver (data generation)
function true_dynamics!(du, u, p, t)
    result = true_dynamics(Vector{eltype(u)}(u))
    du[1] = result[1]
    du[2] = result[2]
end

"""
Generate synthetic data: n nodes following circulation dynamics in B^d_+.
"""
function generate_data(; n::Int=20, T_end::Float32=40f0, dt::Float32=1f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    # Initial positions in B^d_+ (first quadrant of unit disk)
    # Spread across different radii and angles
    u0_all = Vector{Vector{Float32}}(undef, n)
    for i in 1:n
        # Random angle in [π/8, 3π/8] (away from axes)
        theta = Float32(pi/8 + rand(rng) * pi/4)
        # Random radius in [0.3, 0.9]
        r = Float32(0.3 + 0.6 * rand(rng))
        u0_all[i] = Float32[r * cos(theta), r * sin(theta)]
    end

    # Solve ODE for each node
    tspan = (0f0, T_end)
    tsteps = 0f0:dt:T_end
    n_steps = length(tsteps)

    trajectories = Dict{Int, Matrix{Float32}}()

    for i in 1:n
        prob = ODEProblem(true_dynamics!, u0_all[i], tspan)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-7, reltol=1f-7)

        traj = zeros(Float32, 2, n_steps)
        for (t_idx, t) in enumerate(tsteps)
            traj[:, t_idx] = sol(t)
        end
        trajectories[i] = traj
    end

    return (
        trajectories = trajectories,
        n = n,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        u0_all = u0_all
    )
end

println("\n1. Generating circulation dynamics data...")
data = generate_data(n=20, T_end=40f0, dt=1f0)
n = data.n
tsteps = data.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " nodes, T=" * string(data.T_end) * ", steps=" * string(T_steps))
println("  True parameters:")
println("    omega (known):     " * string(TRUE_OMEGA))
println("    alpha (unknown):   " * string(TRUE_ALPHA))
println("    epsilon (unknown): " * string(TRUE_EPSILON))
println("    k_radial (unknown):" * string(TRUE_K_RADIAL))
println("    r_target (unknown):" * string(TRUE_R_TARGET))

# Check that trajectories stay in B^d_+
println("\n  Verifying B^d_+ constraint...")

function verify_bdplus(trajectories, n_nodes, n_steps)
    min_x1, min_x2, max_r = Inf, Inf, 0.0
    for i in 1:n_nodes
        traj = trajectories[i]
        min_x1 = min(min_x1, minimum(traj[1, :]))
        min_x2 = min(min_x2, minimum(traj[2, :]))
        for t in 1:n_steps
            max_r = max(max_r, norm(traj[:, t]))
        end
    end
    return min_x1, min_x2, max_r
end

bdplus_min_x1, bdplus_min_x2, bdplus_max_r = verify_bdplus(data.trajectories, n, T_steps)
println("    min(x₁) = " * string(round(bdplus_min_x1, digits=4)) * " (should be ≥ 0)")
println("    min(x₂) = " * string(round(bdplus_min_x2, digits=4)) * " (should be ≥ 0)")
println("    max(||x||) = " * string(round(bdplus_max_r, digits=4)) * " (should be ≤ 1)")

if bdplus_min_x1 < 0 || bdplus_min_x2 < 0 || bdplus_max_r > 1
    println("  WARNING: B^d_+ constraint violated!")
else
    println("  ✓ All trajectories stay in B^d_+")
end

# =============================================================================
# Setup neural networks
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# NN for learning unknown dynamics (boundary confinement + radial)
nn_unknown = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

# NN for learning full dynamics (baseline - no known structure)
nn_full = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

ps_unknown, st_unknown = Lux.setup(rng, nn_unknown)
ps_full, st_full = Lux.setup(rng, nn_full)

ps_unknown_ca = ComponentArray(ps_unknown)
ps_full_ca = ComponentArray(ps_full)

println("  NN parameters (unknown): " * string(length(ps_unknown_ca)))
println("  NN parameters (full): " * string(length(ps_full_ca)))

# =============================================================================
# Define UDE and Full NN dynamics
# =============================================================================

# UDE: Known circulation + learned unknown
function ude_dynamics(u, p, t)
    T = eltype(u)
    omega_T = T(TRUE_OMEGA)

    # Known: circulation
    dx_known = omega_T .* T[-u[2], u[1]]

    # Unknown: learned by NN
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_unknown(u_input, p, st_unknown)

    return dx_known .+ vec(nn_out)
end

# Full NN: learns everything
function full_dynamics(u, p, t)
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# =============================================================================
# Prepare training data
# =============================================================================
println("\n3. Preparing training data...")

train_nodes = 1:15
val_nodes = 16:20

train_data = Dict{Int, Matrix{Float32}}()
for i in train_nodes
    train_data[i] = data.trajectories[i]
end

println("  Training nodes: " * string(length(train_nodes)))
println("  Validation nodes: " * string(length(val_nodes)))

# =============================================================================
# Loss functions (trajectory-based)
# =============================================================================
println("\n4. Defining loss functions...")

tspan = (0f0, data.T_end)

function predict_node(u0, p, dynamics_fn)
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-6, reltol=1f-6)
    return Array(sol)
end

function loss_ude(p)
    total_loss = 0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]
        pred = predict_node(u0, p, ude_dynamics)
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T_steps)
end

function loss_full(p)
    total_loss = 0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]
        pred = predict_node(u0, p, full_dynamics)
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T_steps)
end

if !VIZ_ONLY || !isfile(SAVE_FILE)
    println("  Initial loss (UDE): " * string(round(loss_ude(ps_unknown_ca), digits=4)))
    println("  Initial loss (Full): " * string(round(loss_full(ps_full_ca), digits=4)))
end

# =============================================================================
# Training or load saved results
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n5-7. Loading saved results...")
    saved = deserialize(SAVE_FILE)

    errors_full_train = saved["errors_full_train"]
    errors_ude_train = saved["errors_ude_train"]
    errors_full_val = saved["errors_full_val"]
    errors_ude_val = saved["errors_ude_val"]
    loss_full_final = saved["loss_full_final"]
    loss_ude_final = saved["loss_ude_final"]

    viz_train_nodes = saved["viz_train_nodes"]
    viz_val_node = saved["viz_val_node"]
    viz_train_true = saved["viz_train_true"]
    viz_train_full = saved["viz_train_full"]
    viz_train_ude = saved["viz_train_ude"]
    viz_val_true = saved["viz_val_true"]
    viz_val_full = saved["viz_val_full"]
    viz_val_ude = saved["viz_val_ude"]

    # Load trained parameters for vector field visualization
    ps_full_trained = saved["ps_full_trained"]
    ps_ude_trained = saved["ps_ude_trained"]

    improvement_train = 100 * (1 - mean(errors_ude_train) / mean(errors_full_train))
    improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))

    println("  Loaded!")

else  # Run training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

println("\n5. Training Full NN (baseline)...")

iter_full = Ref(0)
function callback_full(state, l)
    iter_full[] += 1
    if iter_full[] % 50 == 0
        println("    iter " * string(iter_full[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

# Stage 1: ADAM
println("  Stage 1: ADAM...")
optf_full = Optimization.OptimizationFunction((p, _) -> loss_full(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full_adam = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_full
)

# Stage 2: BFGS
println("  Stage 2: BFGS...")
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
    println("    BFGS failed: " * string(e))
    result_full_adam
end

ps_full_trained = result_full.u
loss_full_final = loss_full(ps_full_trained)
println("  Final loss (Full NN): " * string(round(loss_full_final, digits=6)))

# Train UDE
println("\n6. Training UDE...")

iter_ude = Ref(0)
function callback_ude(state, l)
    iter_ude[] += 1
    if iter_ude[] % 50 == 0
        println("    iter " * string(iter_ude[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

println("  Stage 1: ADAM...")
optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude(p), Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_unknown_ca)

result_ude_adam = Optimization.solve(
    optprob_ude,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_ude
)

println("  Stage 2: BFGS...")
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
    println("    BFGS failed: " * string(e))
    result_ude_adam
end

ps_ude_trained = result_ude.u
loss_ude_final = loss_ude(ps_ude_trained)
println("  Final loss (UDE): " * string(round(loss_ude_final, digits=6)))

# =============================================================================
# Evaluate on all nodes
# =============================================================================
println("\n7. Evaluating...")

errors_full_train = Float64[]
errors_ude_train = Float64[]
errors_full_val = Float64[]
errors_ude_val = Float64[]

for i in 1:n
    u0 = data.trajectories[i][:, 1]
    true_traj = data.trajectories[i]

    pred_full = predict_node(u0, ps_full_trained, full_dynamics)
    pred_ude = predict_node(u0, ps_ude_trained, ude_dynamics)

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T_steps])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T_steps])

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

# Save results
println("\n  Saving results...")

viz_train_nodes = [1, 5, 10, 15]
viz_val_node = first(val_nodes)

viz_train_true = Dict{Int, Matrix{Float32}}()
viz_train_full = Dict{Int, Matrix{Float32}}()
viz_train_ude = Dict{Int, Matrix{Float32}}()

for i in viz_train_nodes
    u0 = data.trajectories[i][:, 1]
    viz_train_true[i] = data.trajectories[i]
    viz_train_full[i] = predict_node(u0, ps_full_trained, full_dynamics)
    viz_train_ude[i] = predict_node(u0, ps_ude_trained, ude_dynamics)
end

u0_val = data.trajectories[viz_val_node][:, 1]
viz_val_true = data.trajectories[viz_val_node]
viz_val_full = predict_node(u0_val, ps_full_trained, full_dynamics)
viz_val_ude = predict_node(u0_val, ps_ude_trained, ude_dynamics)

mkpath(RESULTS_DIR)
serialize(SAVE_FILE, Dict(
    "errors_full_train" => errors_full_train,
    "errors_ude_train" => errors_ude_train,
    "errors_full_val" => errors_full_val,
    "errors_ude_val" => errors_ude_val,
    "loss_full_final" => loss_full_final,
    "loss_ude_final" => loss_ude_final,
    "viz_train_nodes" => viz_train_nodes,
    "viz_val_node" => viz_val_node,
    "viz_train_true" => viz_train_true,
    "viz_train_full" => viz_train_full,
    "viz_train_ude" => viz_train_ude,
    "viz_val_true" => viz_val_true,
    "viz_val_full" => viz_val_full,
    "viz_val_ude" => viz_val_ude,
    "ps_full_trained" => ps_full_trained,
    "ps_ude_trained" => ps_ude_trained
))
println("  Saved to " * SAVE_FILE)

end  # end training block

# =============================================================================
# Visualization
# =============================================================================
println("\n8. Creating visualization...")

fig = Figure(size=(1800, 1000))

# Draw B^d_+ region boundary
theta_arc = range(0, pi/2, length=100)

# --- Row 1: Trajectory plots ---

# Panel 1: Full NN trajectories
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂",
                       title="Full NN (training nodes)", aspect=1)

# Draw B^d_+ boundary
lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)
# Target radius
lines!(ax1, TRUE_R_TARGET .* cos.(theta_arc), TRUE_R_TARGET .* sin.(theta_arc),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

main_node = 5
lines!(ax1, viz_train_true[main_node][1, :], viz_train_true[main_node][2, :],
       color=:black, linewidth=2, label="True")
lines!(ax1, viz_train_full[main_node][1, :], viz_train_full[main_node][2, :],
       color=:blue, linewidth=2, linestyle=:dash, label="Predicted")
scatter!(ax1, [viz_train_true[main_node][1, 1]], [viz_train_true[main_node][2, 1]],
         color=:black, markersize=12, marker=:circle)
scatter!(ax1, [viz_train_true[main_node][1, end]], [viz_train_true[main_node][2, end]],
         color=:black, markersize=12, marker=:star5)

for i in viz_train_nodes
    i == main_node && continue
    lines!(ax1, viz_train_true[i][1, :], viz_train_true[i][2, :], color=(:black, 0.3), linewidth=1)
    lines!(ax1, viz_train_full[i][1, :], viz_train_full[i][2, :], color=(:blue, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax1, position=:lt)
xlims!(ax1, -0.05, 1.05)
ylims!(ax1, -0.05, 1.05)

# Panel 2: UDE trajectories
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="x₁", ylabel="x₂",
                       title="UDE (training nodes)", aspect=1)

lines!(ax2, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax2, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax2, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax2, TRUE_R_TARGET .* cos.(theta_arc), TRUE_R_TARGET .* sin.(theta_arc),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

lines!(ax2, viz_train_true[main_node][1, :], viz_train_true[main_node][2, :],
       color=:black, linewidth=2, label="True")
lines!(ax2, viz_train_ude[main_node][1, :], viz_train_ude[main_node][2, :],
       color=:red, linewidth=2, linestyle=:dash, label="Predicted")
scatter!(ax2, [viz_train_true[main_node][1, 1]], [viz_train_true[main_node][2, 1]],
         color=:black, markersize=12, marker=:circle)
scatter!(ax2, [viz_train_true[main_node][1, end]], [viz_train_true[main_node][2, end]],
         color=:black, markersize=12, marker=:star5)

for i in viz_train_nodes
    i == main_node && continue
    lines!(ax2, viz_train_true[i][1, :], viz_train_true[i][2, :], color=(:black, 0.3), linewidth=1)
    lines!(ax2, viz_train_ude[i][1, :], viz_train_ude[i][2, :], color=(:red, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax2, position=:lt)
xlims!(ax2, -0.05, 1.05)
ylims!(ax2, -0.05, 1.05)

# Panel 3: Validation comparison
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="x₁", ylabel="x₂",
                       title="Validation Node Comparison", aspect=1)

lines!(ax3, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax3, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax3, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax3, TRUE_R_TARGET .* cos.(theta_arc), TRUE_R_TARGET .* sin.(theta_arc),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

lines!(ax3, viz_val_true[1, :], viz_val_true[2, :], color=:black, linewidth=2.5, label="True")
lines!(ax3, viz_val_full[1, :], viz_val_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, viz_val_ude[1, :], viz_val_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
scatter!(ax3, [viz_val_true[1, 1]], [viz_val_true[2, 1]], color=:black, markersize=12, marker=:circle, label="Start")
scatter!(ax3, [viz_val_true[1, end]], [viz_val_true[2, end]], color=:black, markersize=12, marker=:star5, label="End")

axislegend(ax3, position=:lt)
xlims!(ax3, -0.05, 1.05)
ylims!(ax3, -0.05, 1.05)

# --- Row 2: Vector field visualization ---

# Grid for vector field
grid_pts = range(0.05, 0.95, length=15)
xs = [x for x in grid_pts, y in grid_pts]
ys = [y for x in grid_pts, y in grid_pts]

# Panel 4: True vector field
ax4 = CairoMakie.Axis(fig[2, 1], xlabel="x₁", ylabel="x₂",
                       title="True Vector Field", aspect=1)

lines!(ax4, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax4, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax4, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

# Compute true vector field
us_true = zeros(Float32, length(grid_pts), length(grid_pts))
vs_true = zeros(Float32, length(grid_pts), length(grid_pts))

for (i, x) in enumerate(grid_pts)
    for (j, y) in enumerate(grid_pts)
        if x^2 + y^2 <= 1  # Inside B^d_+
            du = true_dynamics(Float32[x, y])
            us_true[i, j] = du[1]
            vs_true[i, j] = du[2]
        end
    end
end

arrows!(ax4, vec(xs), vec(ys), vec(us_true), vec(vs_true),
        lengthscale=0.4, arrowsize=8, color=:black)

xlims!(ax4, -0.05, 1.05)
ylims!(ax4, -0.05, 1.05)

# Panel 5: Full NN vector field
ax5 = CairoMakie.Axis(fig[2, 2], xlabel="x₁", ylabel="x₂",
                       title="Full NN Vector Field", aspect=1)

lines!(ax5, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax5, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax5, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

us_full = zeros(Float32, length(grid_pts), length(grid_pts))
vs_full = zeros(Float32, length(grid_pts), length(grid_pts))

for (i, x) in enumerate(grid_pts)
    for (j, y) in enumerate(grid_pts)
        if x^2 + y^2 <= 1
            du = full_dynamics(Float32[x, y], ps_full_trained, 0f0)
            us_full[i, j] = du[1]
            vs_full[i, j] = du[2]
        end
    end
end

arrows!(ax5, vec(xs), vec(ys), vec(us_full), vec(vs_full),
        lengthscale=0.4, arrowsize=8, color=:blue)

xlims!(ax5, -0.05, 1.05)
ylims!(ax5, -0.05, 1.05)

# Panel 6: UDE vector field
ax6 = CairoMakie.Axis(fig[2, 3], xlabel="x₁", ylabel="x₂",
                       title="UDE Vector Field", aspect=1)

lines!(ax6, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax6, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax6, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

us_ude = zeros(Float32, length(grid_pts), length(grid_pts))
vs_ude = zeros(Float32, length(grid_pts), length(grid_pts))

for (i, x) in enumerate(grid_pts)
    for (j, y) in enumerate(grid_pts)
        if x^2 + y^2 <= 1
            du = ude_dynamics(Float32[x, y], ps_ude_trained, 0f0)
            us_ude[i, j] = du[1]
            vs_ude[i, j] = du[2]
        end
    end
end

arrows!(ax6, vec(xs), vec(ys), vec(us_ude), vec(vs_ude),
        lengthscale=0.4, arrowsize=8, color=:red)

xlims!(ax6, -0.05, 1.05)
ylims!(ax6, -0.05, 1.05)

save("results/example2_circulation.pdf", fig)
println("\nSaved: results/example2_circulation.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 2 - Homogeneous Circulation in B^d_+")
println("=" ^ 60)
println("\nDynamics: Circulation (known) + Boundary confinement (unknown)")
println("All " * string(n) * " nodes follow identical physics.")
println("\nTraining losses:")
println("  Full NN: " * string(round(loss_full_final, digits=6)))
println("  UDE:     " * string(round(loss_ude_final, digits=6)))
println("\nPrediction errors (validation):")
println("  Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("  UDE:     " * string(round(mean(errors_ude_val), digits=4)))
println("\nImprovement (positive = UDE better):")
println("  Validation: " * string(round(improvement_val, digits=1)) * "%")

if improvement_val > 0
    println("\n✓ UDE outperforms Full NN!")
    println("  Known circulation structure helps learning.")
else
    println("\n✗ Full NN outperforms UDE.")
    println("  The unknown part may be too complex for the NN to learn as a residual.")
end
