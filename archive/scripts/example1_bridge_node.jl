#!/usr/bin/env julia
"""
Example 1: Bridge Node Oscillation (Lead Example)

Story: A "bridge" node oscillates between two community attractors within B^d_+.
This is the primary example demonstrating the UDE approach.

Dynamics (autonomous, stays in B^d_+):
- Two attractors A₁ and A₂ in B^d_+
- Smooth switching: when closer to A₁, attract to A₂ and vice versa
- Creates limit cycle oscillation between communities

Known part: The two-attractor switching structure
Unknown part: Attractor locations A₁, A₂

Usage:
  julia --project scripts/example1_bridge_node.jl           # Full run
  julia --project scripts/example1_bridge_node.jl --viz     # Visualization only
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
const SAVE_FILE = joinpath(RESULTS_DIR, "example1_bridge_node_results.jls")

println("=" ^ 60)
println("Example 1: Bridge Node Oscillation")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# True dynamics parameters
# =============================================================================

# Two attractors in B^d_+ (first quadrant)
const A1 = Float32[0.25, 0.65]   # Upper-left community
const A2 = Float32[0.65, 0.25]   # Lower-right community

# Dynamics parameters - tuned for visible oscillation
const K_ATTRACT = 0.5f0          # Attraction strength (UNKNOWN magnitude)
const OVERSHOOT_FACTOR = 1.5f0   # How much to overshoot target (creates oscillation)
const SWITCH_DIST = 0.15f0       # Distance at which to switch attractors

"""
Sigmoid function for smooth switching.
"""
sigmoid(x) = 1f0 / (1f0 + exp(-x))

"""
True dynamics: Two-attractor oscillation (limit cycle).

Uses a "push when close" mechanism: when you reach one attractor,
you get strongly pushed toward the other. This creates persistent
oscillation rather than settling at an equilibrium.
"""
function true_dynamics(u::Vector{T}) where T
    # Distances to each attractor
    d1 = norm(u - T.(A1))
    d2 = norm(u - T.(A2))

    # Proximity functions: high when close to attractor
    prox1 = exp(-d1 / T(SWITCH_DIST))  # ~1 when at A1, ~0 when far
    prox2 = exp(-d2 / T(SWITCH_DIST))  # ~1 when at A2, ~0 when far

    # Direction vectors (normalized)
    dir_to_A1 = (T.(A1) - u) / max(d1, T(0.01))
    dir_to_A2 = (T.(A2) - u) / max(d2, T(0.01))

    # Key insight: When close to A1 (prox1 high), push STRONGLY toward A2
    # When close to A2 (prox2 high), push STRONGLY toward A1
    # When in between, gentle attraction to nearest

    push_to_A2 = T(K_ATTRACT) * T(OVERSHOOT_FACTOR) * prox1 .* dir_to_A2
    push_to_A1 = T(K_ATTRACT) * T(OVERSHOOT_FACTOR) * prox2 .* dir_to_A1

    # Gentle general attraction to midpoint creates bounded motion
    midpoint = (T.(A1) .+ T.(A2)) ./ T(2)
    attract_center = T(0.1) * (midpoint .- u)

    dx = push_to_A1 .+ push_to_A2 .+ attract_center

    # Add soft boundary repulsion to stay in B^d_+
    epsilon = T(0.05)
    alpha = T(0.2)
    repel_x1 = alpha * exp(-u[1] / epsilon)
    repel_x2 = alpha * exp(-u[2] / epsilon)
    r = norm(u)
    repel_arc = r > T(0.1) ? alpha * exp((r - one(T)) / epsilon) : zero(T)

    dx = dx .+ T[repel_x1, repel_x2]
    if r > T(0.1)
        dx = dx .- (repel_arc / r) .* u
    end

    return dx
end

function true_dynamics!(du, u, p, t)
    result = true_dynamics(Vector{eltype(u)}(u))
    du[1] = result[1]
    du[2] = result[2]
end

"""
Generate bridge node trajectory data.
"""
function generate_data(; n_trajectories::Int=10, T_end::Float32=60f0, dt::Float32=0.5f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    # Initial positions scattered around the center of B^d_+
    u0_all = Vector{Vector{Float32}}(undef, n_trajectories)
    for i in 1:n_trajectories
        # Start between the two attractors with some variation
        center = (A1 .+ A2) ./ 2
        perturbation = Float32.(0.1 .* randn(rng, 2))
        u0 = center .+ perturbation
        # Ensure in B^d_+
        u0 = max.(u0, 0.1f0)
        u0 ./= max(norm(u0), 1f0) * 1.05f0  # Keep inside unit ball
        u0_all[i] = u0
    end

    tspan = (0f0, T_end)
    tsteps = 0f0:dt:T_end
    n_steps = length(tsteps)

    trajectories = Dict{Int, Matrix{Float32}}()

    for i in 1:n_trajectories
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
        n = n_trajectories,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        u0_all = u0_all
    )
end

println("\n1. Generating bridge node data...")
data = generate_data(n_trajectories=10, T_end=60f0, dt=0.5f0)
n = data.n
tsteps = data.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " trajectories, T=" * string(data.T_end) * ", steps=" * string(T_steps))
println("  Attractor A₁: [" * string(A1[1]) * ", " * string(A1[2]) * "]")
println("  Attractor A₂: [" * string(A2[1]) * ", " * string(A2[2]) * "]")

# Verify B^d_+ constraint
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

min_x1, min_x2, max_r = verify_bdplus(data.trajectories, n, T_steps)
println("\n  B^d_+ constraint check:")
println("    min(x₁) = " * string(round(min_x1, digits=4)))
println("    min(x₂) = " * string(round(min_x2, digits=4)))
println("    max(||x||) = " * string(round(max_r, digits=4)))

if min_x1 >= 0 && min_x2 >= 0 && max_r <= 1
    println("  ✓ All trajectories stay in B^d_+")
else
    println("  WARNING: B^d_+ constraint violated!")
end

# =============================================================================
# Setup neural networks
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# Full NN learns everything
nn_full = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

# UDE: NN learns the attractor locations/strengths
# Known structure: two-attractor switching
nn_ude = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 4)  # Output: [A1_x, A1_y, A2_x, A2_y] corrections or full locations
)

ps_full, st_full = Lux.setup(rng, nn_full)
ps_ude, st_ude = Lux.setup(Random.Xoshiro(42), nn_ude)

ps_full_ca = ComponentArray(ps_full)
ps_ude_ca = ComponentArray(ps_ude)

println("  NN parameters (full): " * string(length(ps_full_ca)))
println("  NN parameters (UDE): " * string(length(ps_ude_ca)))

# =============================================================================
# Define dynamics
# =============================================================================

# Full NN dynamics
function full_dynamics(u, p, t)
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# UDE dynamics: Known switching structure + learned attractor locations
function ude_dynamics(u, p, t)
    T = eltype(u)

    # NN outputs attractor location corrections (scaled)
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_ude(u_input, p, st_ude)
    nn_out = vec(nn_out)

    # Interpret NN output as attractor corrections
    # Use fixed base attractors + NN corrections
    A1_learned = T.(A1) .+ 0.1f0 .* nn_out[1:2]  # Small corrections to A1
    A2_learned = T.(A2) .+ 0.1f0 .* nn_out[3:4]  # Small corrections to A2

    # Known structure: two-attractor switching
    d1 = norm(u - A1_learned)
    d2 = norm(u - A2_learned)

    f = sigmoid(T(SWITCH_SHARPNESS) * (d1 - d2))

    attract_to_A1 = -T(K_ATTRACT) * (u - A1_learned)
    attract_to_A2 = -T(K_ATTRACT) * (u - A2_learned)

    dx = f .* attract_to_A1 .+ (one(T) - f) .* attract_to_A2

    # Boundary repulsion (known/fixed) - non-mutating for Zygote
    epsilon = T(0.05)
    alpha = T(0.2)
    repel_x1 = alpha * exp(-u[1] / epsilon)
    repel_x2 = alpha * exp(-u[2] / epsilon)
    r = norm(u)

    dx = dx .+ T[repel_x1, repel_x2]

    if r > T(0.1)
        repel_arc = alpha * exp((r - one(T)) / epsilon)
        dx = dx .- (repel_arc / r) .* u
    end

    return dx
end

# =============================================================================
# Training setup
# =============================================================================
println("\n3. Preparing training data...")

train_nodes = 1:7
val_nodes = 8:10

train_data = Dict{Int, Matrix{Float32}}()
for i in train_nodes
    train_data[i] = data.trajectories[i]
end

println("  Training trajectories: " * string(length(train_nodes)))
println("  Validation trajectories: " * string(length(val_nodes)))

tspan = (0f0, data.T_end)

function predict_node(u0, p, dynamics_fn)
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-6, reltol=1f-6)
    return Array(sol)
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

if !VIZ_ONLY || !isfile(SAVE_FILE)
    println("  Initial loss (Full): " * string(round(loss_full(ps_full_ca), digits=4)))
    println("  Initial loss (UDE): " * string(round(loss_ude(ps_ude_ca), digits=4)))
end

# =============================================================================
# Training or load results
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n4-6. Loading saved results...")
    saved = deserialize(SAVE_FILE)

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

    ps_full_trained = saved["ps_full_trained"]
    ps_ude_trained = saved["ps_ude_trained"]

    improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))
    println("  Loaded!")

else  # Training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

println("\n4. Training Full NN...")

iter_full = Ref(0)
function callback_full(state, l)
    iter_full[] += 1
    if iter_full[] % 50 == 0
        println("    iter " * string(iter_full[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

optf_full = Optimization.OptimizationFunction((p, _) -> loss_full(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full_adam = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_full
)

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

println("\n5. Training UDE...")

iter_ude = Ref(0)
function callback_ude(state, l)
    iter_ude[] += 1
    if iter_ude[] % 50 == 0
        println("    iter " * string(iter_ude[]) * ": loss=" * string(round(l, digits=6)))
    end
    return false
end

optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude(p), Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_ude_ca)

result_ude_adam = Optimization.solve(
    optprob_ude,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_ude
)

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
# Evaluate
# =============================================================================
println("\n6. Evaluating...")

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

improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))
println("\n  UDE improvement: " * string(round(improvement_val, digits=1)) * "%")

# Save precomputed trajectories
println("\n  Saving results...")

viz_train_nodes = [1, 3, 5]
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
println("\n7. Creating visualization...")

fig = Figure(size=(1800, 1000))

theta_arc = range(0, pi/2, length=100)

# Row 1: Trajectory comparisons
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂",
                       title="Full NN (training)", aspect=1)

lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

# Mark attractors
scatter!(ax1, [A1[1]], [A1[2]], color=:purple, markersize=20, marker=:star5)
scatter!(ax1, [A2[1]], [A2[2]], color=:orange, markersize=20, marker=:star5)

main_node = first(viz_train_nodes)
lines!(ax1, viz_train_true[main_node][1, :], viz_train_true[main_node][2, :],
       color=:black, linewidth=2, label="True")
lines!(ax1, viz_train_full[main_node][1, :], viz_train_full[main_node][2, :],
       color=:blue, linewidth=2, linestyle=:dash, label="Predicted")

for i in viz_train_nodes
    i == main_node && continue
    lines!(ax1, viz_train_true[i][1, :], viz_train_true[i][2, :],
           color=(:black, 0.3), linewidth=1)
    lines!(ax1, viz_train_full[i][1, :], viz_train_full[i][2, :],
           color=(:blue, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax1, position=:rt)
xlims!(ax1, -0.05, 1.05)
ylims!(ax1, -0.05, 1.05)

# Panel 2: UDE
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="x₁", ylabel="x₂",
                       title="UDE (training)", aspect=1)

lines!(ax2, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax2, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax2, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

scatter!(ax2, [A1[1]], [A1[2]], color=:purple, markersize=20, marker=:star5, label="A₁")
scatter!(ax2, [A2[1]], [A2[2]], color=:orange, markersize=20, marker=:star5, label="A₂")

lines!(ax2, viz_train_true[main_node][1, :], viz_train_true[main_node][2, :],
       color=:black, linewidth=2, label="True")
lines!(ax2, viz_train_ude[main_node][1, :], viz_train_ude[main_node][2, :],
       color=:red, linewidth=2, linestyle=:dash, label="Predicted")

for i in viz_train_nodes
    i == main_node && continue
    lines!(ax2, viz_train_true[i][1, :], viz_train_true[i][2, :],
           color=(:black, 0.3), linewidth=1)
    lines!(ax2, viz_train_ude[i][1, :], viz_train_ude[i][2, :],
           color=(:red, 0.3), linewidth=1, linestyle=:dash)
end

axislegend(ax2, position=:rt)
xlims!(ax2, -0.05, 1.05)
ylims!(ax2, -0.05, 1.05)

# Panel 3: Validation comparison
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="x₁", ylabel="x₂",
                       title="Validation Comparison", aspect=1)

lines!(ax3, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax3, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax3, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

scatter!(ax3, [A1[1]], [A1[2]], color=:purple, markersize=20, marker=:star5)
scatter!(ax3, [A2[1]], [A2[2]], color=:orange, markersize=20, marker=:star5)

lines!(ax3, viz_val_true[1, :], viz_val_true[2, :], color=:black, linewidth=2.5, label="True")
lines!(ax3, viz_val_full[1, :], viz_val_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, viz_val_ude[1, :], viz_val_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")

axislegend(ax3, position=:rt)
xlims!(ax3, -0.05, 1.05)
ylims!(ax3, -0.05, 1.05)

# Row 2: Vector fields
grid_pts = range(0.1, 0.9, length=12)
xs = [x for x in grid_pts, y in grid_pts]
ys = [y for x in grid_pts, y in grid_pts]

# True vector field
ax4 = CairoMakie.Axis(fig[2, 1], xlabel="x₁", ylabel="x₂",
                       title="True Vector Field", aspect=1)

lines!(ax4, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
scatter!(ax4, [A1[1]], [A1[2]], color=:purple, markersize=15, marker=:star5)
scatter!(ax4, [A2[1]], [A2[2]], color=:orange, markersize=15, marker=:star5)

us_true = zeros(Float32, length(grid_pts), length(grid_pts))
vs_true = zeros(Float32, length(grid_pts), length(grid_pts))

for (i, x) in enumerate(grid_pts)
    for (j, y) in enumerate(grid_pts)
        if x^2 + y^2 <= 1
            du = true_dynamics(Float32[x, y])
            us_true[i, j] = du[1]
            vs_true[i, j] = du[2]
        end
    end
end

arrows!(ax4, vec(xs), vec(ys), vec(us_true), vec(vs_true),
        lengthscale=0.3, arrowsize=8, color=:black)

xlims!(ax4, -0.05, 1.05)
ylims!(ax4, -0.05, 1.05)

# Full NN vector field
ax5 = CairoMakie.Axis(fig[2, 2], xlabel="x₁", ylabel="x₂",
                       title="Full NN Vector Field", aspect=1)

lines!(ax5, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
scatter!(ax5, [A1[1]], [A1[2]], color=:purple, markersize=15, marker=:star5)
scatter!(ax5, [A2[1]], [A2[2]], color=:orange, markersize=15, marker=:star5)

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
        lengthscale=0.3, arrowsize=8, color=:blue)

xlims!(ax5, -0.05, 1.05)
ylims!(ax5, -0.05, 1.05)

# UDE vector field
ax6 = CairoMakie.Axis(fig[2, 3], xlabel="x₁", ylabel="x₂",
                       title="UDE Vector Field", aspect=1)

lines!(ax6, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
scatter!(ax6, [A1[1]], [A1[2]], color=:purple, markersize=15, marker=:star5)
scatter!(ax6, [A2[1]], [A2[2]], color=:orange, markersize=15, marker=:star5)

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
        lengthscale=0.3, arrowsize=8, color=:red)

xlims!(ax6, -0.05, 1.05)
ylims!(ax6, -0.05, 1.05)

save("results/example1_bridge_node.pdf", fig)
println("\nSaved: results/example1_bridge_node.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 1 - Bridge Node Oscillation")
println("=" ^ 60)

println("\nStory: A bridge node oscillates between two community attractors")
println("  A₁ = [" * string(A1[1]) * ", " * string(A1[2]) * "] (upper-left community)")
println("  A₂ = [" * string(A2[1]) * ", " * string(A2[2]) * "] (lower-right community)")

println("\nTraining losses:")
println("  Full NN: " * string(round(loss_full_final, digits=6)))
println("  UDE:     " * string(round(loss_ude_final, digits=6)))

println("\nValidation errors:")
println("  Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("  UDE:     " * string(round(mean(errors_ude_val), digits=4)))

println("\nUDE improvement: " * string(round(improvement_val, digits=1)) * "%")

if improvement_val > 0
    println("\n✓ UDE outperforms Full NN!")
    println("  Known two-attractor structure helps learning.")
else
    println("\n✗ Full NN outperforms UDE.")
end
