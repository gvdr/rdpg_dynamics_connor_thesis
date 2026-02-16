#!/usr/bin/env julia
"""
Example 4: Food Web Heterogeneous Dynamics (d=4)

Story: Different node types (predator, prey, resource) follow different dynamics.
This demonstrates handling of heterogeneous dynamics and higher dimensions.

Setup:
- d=4 dimensional embedding space (B^4_+)
- 3 node types: Predator (P), Prey (Y), Resource (R)
- Each type has different ODE parameters
- All trajectories stay in B^d_+

Known part: General Lotka-Volterra-like structure
Unknown part: Type-specific interaction parameters

Usage:
  julia --project scripts/example4_food_web.jl           # Full run
  julia --project scripts/example4_food_web.jl --viz     # Visualization only
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
const SAVE_FILE = joinpath(RESULTS_DIR, "example4_food_web_results.jls")

println("=" ^ 60)
println("Example 4: Food Web Heterogeneous Dynamics (d=4)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# Node types and parameters
# =============================================================================

# Node types
const TYPE_PREDATOR = 1
const TYPE_PREY = 2
const TYPE_RESOURCE = 3

# Target positions in B^4_+ for each type (attractors)
# Each type tends toward a different region of the embedding space
const TARGET_PREDATOR = Float32[0.6, 0.3, 0.4, 0.5]
const TARGET_PREY = Float32[0.3, 0.6, 0.5, 0.4]
const TARGET_RESOURCE = Float32[0.4, 0.4, 0.6, 0.3]

# Normalize to be in B^4_+
normalize_bd(x) = x ./ max(norm(x), 1f0)

# Dynamics parameters (different for each type)
const K_PREDATOR = 0.08f0    # Predator attraction rate (UNKNOWN)
const K_PREY = 0.12f0        # Prey attraction rate (UNKNOWN)
const K_RESOURCE = 0.15f0    # Resource attraction rate (UNKNOWN)

# Cross-type interaction (known structure, unknown magnitude)
const INTERACTION_STRENGTH = 0.05f0  # How much types influence each other

"""
Dynamics for a single node given its type and the mean positions of other types.
"""
function node_dynamics(u::Vector{T}, node_type::Int, mean_predator, mean_prey, mean_resource) where T
    d = length(u)

    # Get target and attraction rate for this node type
    target, k = if node_type == TYPE_PREDATOR
        T.(TARGET_PREDATOR), T(K_PREDATOR)
    elseif node_type == TYPE_PREY
        T.(TARGET_PREY), T(K_PREY)
    else  # Resource
        T.(TARGET_RESOURCE), T(K_RESOURCE)
    end

    # Self-attraction to type target (KNOWN structure, UNKNOWN parameters)
    dx = -k .* (u .- target)

    # Cross-type interactions (KNOWN structure, UNKNOWN strength)
    int_str = T(INTERACTION_STRENGTH)

    if node_type == TYPE_PREDATOR
        # Predators are attracted to prey (hunting)
        dx .+= int_str .* (T.(mean_prey) .- u)
    elseif node_type == TYPE_PREY
        # Prey are repelled by predators (fleeing)
        dx .-= int_str .* (T.(mean_predator) .- u)
        # Prey attracted to resources (feeding)
        dx .+= 0.5f0 .* int_str .* (T.(mean_resource) .- u)
    else  # Resource
        # Resources grow toward their target (regrowth)
        dx .+= 0.5f0 .* k .* (target .- u)
    end

    # Soft boundary repulsion to stay in B^d_+
    epsilon = T(0.05)
    alpha = T(0.3)

    for i in 1:d
        # Repel from zero boundaries
        dx[i] += alpha * exp(-u[i] / epsilon)
    end

    # Repel from unit sphere boundary
    r = norm(u)
    if r > T(0.5)
        repel_sphere = alpha * exp((r - one(T)) / epsilon)
        dx .-= (repel_sphere / r) .* u
    end

    return dx
end

"""
Generate food web data with heterogeneous node types.
"""
function generate_data(; n_predators::Int=5, n_prey::Int=8, n_resource::Int=7,
                        T_end::Float32=50f0, dt::Float32=1f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    n_total = n_predators + n_prey + n_resource
    d = 4

    # Assign types
    node_types = vcat(
        fill(TYPE_PREDATOR, n_predators),
        fill(TYPE_PREY, n_prey),
        fill(TYPE_RESOURCE, n_resource)
    )

    # Initialize positions near type targets with perturbation
    u0_all = Vector{Vector{Float32}}(undef, n_total)
    for i in 1:n_total
        target = if node_types[i] == TYPE_PREDATOR
            TARGET_PREDATOR
        elseif node_types[i] == TYPE_PREY
            TARGET_PREY
        else
            TARGET_RESOURCE
        end
        perturbation = Float32.(0.15 .* randn(rng, d))
        u0 = target .+ perturbation
        u0 = max.(u0, 0.1f0)  # Keep positive
        u0 = normalize_bd(u0) .* 0.8f0  # Keep in B^d_+
        u0_all[i] = u0
    end

    # Define ODE system for all nodes
    function system_dynamics!(du, u_flat, p, t)
        # Reshape to (d, n)
        U = reshape(u_flat, d, n_total)

        # Compute mean positions by type
        mean_predator = zeros(Float32, d)
        mean_prey = zeros(Float32, d)
        mean_resource = zeros(Float32, d)
        count_p, count_y, count_r = 0, 0, 0

        for i in 1:n_total
            if node_types[i] == TYPE_PREDATOR
                mean_predator .+= U[:, i]
                count_p += 1
            elseif node_types[i] == TYPE_PREY
                mean_prey .+= U[:, i]
                count_y += 1
            else
                mean_resource .+= U[:, i]
                count_r += 1
            end
        end

        mean_predator ./= count_p
        mean_prey ./= count_y
        mean_resource ./= count_r

        # Compute dynamics for each node
        DU = reshape(du, d, n_total)
        for i in 1:n_total
            DU[:, i] .= node_dynamics(U[:, i], node_types[i], mean_predator, mean_prey, mean_resource)
        end
    end

    # Solve ODE
    u0_flat = vcat(u0_all...)
    tspan = (0f0, T_end)
    tsteps = 0f0:dt:T_end
    n_steps = length(tsteps)

    prob = ODEProblem(system_dynamics!, u0_flat, tspan)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-6, reltol=1f-6)

    # Extract individual trajectories
    trajectories = Dict{Int, Matrix{Float32}}()
    for i in 1:n_total
        traj = zeros(Float32, d, n_steps)
        for (t_idx, t) in enumerate(tsteps)
            idx_start = (i-1)*d + 1
            idx_end = i*d
            traj[:, t_idx] = sol.u[t_idx][idx_start:idx_end]
        end
        trajectories[i] = traj
    end

    return (
        trajectories = trajectories,
        node_types = node_types,
        n = n_total,
        d = d,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        u0_all = u0_all,
        n_predators = n_predators,
        n_prey = n_prey,
        n_resource = n_resource
    )
end

println("\n1. Generating food web data...")
data = generate_data(n_predators=5, n_prey=8, n_resource=7, T_end=50f0, dt=1f0)
n = data.n
d = data.d
tsteps = data.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " nodes (P=" * string(data.n_predators) *
        ", Y=" * string(data.n_prey) * ", R=" * string(data.n_resource) * ")")
println("  d=" * string(d) * " dimensions")
println("  T=" * string(data.T_end) * ", steps=" * string(T_steps))

# Verify B^d_+ constraint
println("\n  Verifying B^d_+ constraint...")

function verify_bdplus_4d(trajectories, n_nodes, n_steps, dims)
    min_coords = fill(Inf, dims)
    max_r = 0.0
    for i in 1:n_nodes
        traj = trajectories[i]
        for j in 1:dims
            min_coords[j] = min(min_coords[j], minimum(traj[j, :]))
        end
        for t in 1:n_steps
            max_r = max(max_r, norm(traj[:, t]))
        end
    end
    return min_coords, max_r
end

min_coords, max_r = verify_bdplus_4d(data.trajectories, n, T_steps, d)

for j in 1:d
    println("    min(x_" * string(j) * ") = " * string(round(min_coords[j], digits=4)))
end
println("    max(||x||) = " * string(round(max_r, digits=4)))

if all(min_coords .>= 0) && max_r <= 1
    println("  ✓ All trajectories stay in B^d_+")
else
    println("  WARNING: B^d_+ constraint violated!")
end

# =============================================================================
# Setup neural networks
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# Full NN: learns everything from scratch
# Input: position (d) + one-hot type encoding (3) = d+3
nn_full = Lux.Chain(
    Lux.Dense(d + 3, 48, tanh),
    Lux.Dense(48, 48, tanh),
    Lux.Dense(48, d)
)

# UDE NN: learns type-specific parameters
# Same architecture but interprets output differently
nn_ude = Lux.Chain(
    Lux.Dense(d + 3, 48, tanh),
    Lux.Dense(48, 48, tanh),
    Lux.Dense(48, d)
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

# One-hot encode node type (Zygote-compatible - no mutations)
const TYPE_ONEHOTS = (
    Float32[1, 0, 0],  # TYPE_PREDATOR = 1
    Float32[0, 1, 0],  # TYPE_PREY = 2
    Float32[0, 0, 1]   # TYPE_RESOURCE = 3
)

function type_onehot(node_type::Int)
    return TYPE_ONEHOTS[node_type]
end

# Full NN dynamics: NN predicts everything
function full_node_dynamics(u, node_type, p)
    type_enc = type_onehot(node_type)
    input = vcat(u, type_enc)
    input_mat = reshape(input, length(input), 1)
    nn_out, _ = nn_full(input_mat, p, st_full)
    return vec(nn_out)
end

# UDE dynamics: Known structure + learned corrections
function ude_node_dynamics(u, node_type, p, mean_predator, mean_prey, mean_resource)
    T = eltype(u)

    # Known structure: attraction to type-specific targets
    target = if node_type == TYPE_PREDATOR
        T.(TARGET_PREDATOR)
    elseif node_type == TYPE_PREY
        T.(TARGET_PREY)
    else
        T.(TARGET_RESOURCE)
    end

    # Base known dynamics: simple attraction
    k_base = T(0.1)
    dx_known = -k_base .* (u .- target)

    # NN learns corrections/adjustments
    type_enc = type_onehot(node_type)
    input = vcat(u, type_enc)
    input_mat = reshape(input, length(input), 1)
    nn_out, _ = nn_ude(input_mat, p, st_ude)
    dx_correction = vec(nn_out)

    # Boundary repulsion (known) - non-mutating for Zygote
    epsilon = T(0.05)
    alpha = T(0.3)

    # Repulsion from coordinate planes (vectorized, no mutation)
    dx_boundary = alpha .* exp.(-u ./ epsilon)

    # Repulsion from unit sphere
    r = norm(u)
    sphere_repel = r > T(0.5) ? alpha * exp((r - one(T)) / epsilon) / r : zero(T)
    dx_boundary = dx_boundary .- sphere_repel .* u

    return dx_known .+ dx_correction .+ dx_boundary
end

# =============================================================================
# Training setup
# =============================================================================
println("\n3. Preparing training data...")

# Split by type to ensure balanced train/val
pred_idx = findall(t -> t == TYPE_PREDATOR, data.node_types)
prey_idx = findall(t -> t == TYPE_PREY, data.node_types)
res_idx = findall(t -> t == TYPE_RESOURCE, data.node_types)

train_nodes = vcat(pred_idx[1:4], prey_idx[1:6], res_idx[1:5])
val_nodes = vcat(pred_idx[5:5], prey_idx[7:8], res_idx[6:7])

train_data = Dict{Int, Matrix{Float32}}()
for i in train_nodes
    train_data[i] = data.trajectories[i]
end

println("  Training nodes: " * string(length(train_nodes)) *
        " (P=" * string(sum(data.node_types[train_nodes] .== TYPE_PREDATOR)) *
        ", Y=" * string(sum(data.node_types[train_nodes] .== TYPE_PREY)) *
        ", R=" * string(sum(data.node_types[train_nodes] .== TYPE_RESOURCE)) * ")")
println("  Validation nodes: " * string(length(val_nodes)))

tspan = (0f0, data.T_end)

# Prediction function for a single node
function predict_full_node(u0, node_type, p)
    function dynamics_wrapper(u, p_inner, t)
        return full_node_dynamics(u, node_type, p_inner)
    end

    prob = ODEProblem(dynamics_wrapper, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-5, reltol=1f-5)
    return Array(sol)
end

function predict_ude_node(u0, node_type, p, mean_p, mean_y, mean_r)
    function dynamics_wrapper(u, p_inner, t)
        return ude_node_dynamics(u, node_type, p_inner, mean_p, mean_y, mean_r)
    end

    prob = ODEProblem(dynamics_wrapper, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-5, reltol=1f-5)
    return Array(sol)
end

# Compute mean positions for UDE (use training data means)
mean_p = mean([mean(train_data[i], dims=2)[:] for i in train_nodes if data.node_types[i] == TYPE_PREDATOR])
mean_y = mean([mean(train_data[i], dims=2)[:] for i in train_nodes if data.node_types[i] == TYPE_PREY])
mean_r = mean([mean(train_data[i], dims=2)[:] for i in train_nodes if data.node_types[i] == TYPE_RESOURCE])

function loss_full(p)
    total_loss = 0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]
        pred = predict_full_node(u0, data.node_types[i], p)
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T_steps * d)
end

function loss_ude(p)
    total_loss = 0f0
    for i in train_nodes
        u0 = train_data[i][:, 1]
        true_traj = train_data[i]
        pred = predict_ude_node(u0, data.node_types[i], p, mean_p, mean_y, mean_r)
        total_loss += sum(abs2, true_traj .- pred)
    end
    return total_loss / (length(train_nodes) * T_steps * d)
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

    viz_data = saved["viz_data"]

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
callback_full(state, l) = (iter_full[] += 1; iter_full[] % 50 == 0 && println("    iter " * string(iter_full[]) * ": " * string(round(l, digits=6))); false)

optf_full = Optimization.OptimizationFunction((p, _) -> loss_full(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full_adam = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=200,
    callback=callback_full
)

iter_full[] = 200
optprob_full2 = Optimization.OptimizationProblem(optf_full, result_full_adam.u)

result_full = try
    Optimization.solve(
        optprob_full2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=50,
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
callback_ude(state, l) = (iter_ude[] += 1; iter_ude[] % 50 == 0 && println("    iter " * string(iter_ude[]) * ": " * string(round(l, digits=6))); false)

optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude(p), Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_ude_ca)

result_ude_adam = Optimization.solve(
    optprob_ude,
    OptimizationOptimisers.Adam(0.01),
    maxiters=200,
    callback=callback_ude
)

iter_ude[] = 200
optprob_ude2 = Optimization.OptimizationProblem(optf_ude, result_ude_adam.u)

result_ude = try
    Optimization.solve(
        optprob_ude2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=50,
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

errors_full_val = Float64[]
errors_ude_val = Float64[]

for i in val_nodes
    u0 = data.trajectories[i][:, 1]
    true_traj = data.trajectories[i]

    pred_full = predict_full_node(u0, data.node_types[i], ps_full_trained)
    pred_ude = predict_ude_node(u0, data.node_types[i], ps_ude_trained, mean_p, mean_y, mean_r)

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T_steps])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T_steps])

    push!(errors_full_val, err_full)
    push!(errors_ude_val, err_ude)
end

println("\n  Validation errors:")
println("    Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_val), digits=4)))

improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))
println("    Improvement: " * string(round(improvement_val, digits=1)) * "%")

# Save visualization data
viz_data = Dict{String, Any}()

# Sample one node of each type for visualization
viz_pred = first(pred_idx)
viz_prey = first(prey_idx)
viz_res = first(res_idx)

for (label, idx) in [("predator", viz_pred), ("prey", viz_prey), ("resource", viz_res)]
    u0 = data.trajectories[idx][:, 1]
    viz_data[label * "_true"] = data.trajectories[idx]
    viz_data[label * "_full"] = predict_full_node(u0, data.node_types[idx], ps_full_trained)
    viz_data[label * "_ude"] = predict_ude_node(u0, data.node_types[idx], ps_ude_trained, mean_p, mean_y, mean_r)
    viz_data[label * "_type"] = data.node_types[idx]
end

# Save
println("\n  Saving results...")
mkpath(RESULTS_DIR)
serialize(SAVE_FILE, Dict(
    "errors_full_val" => errors_full_val,
    "errors_ude_val" => errors_ude_val,
    "loss_full_final" => loss_full_final,
    "loss_ude_final" => loss_ude_final,
    "viz_data" => viz_data,
    "ps_full_trained" => ps_full_trained,
    "ps_ude_trained" => ps_ude_trained
))
println("  Saved to " * SAVE_FILE)

end  # end training block

# =============================================================================
# Visualization
# =============================================================================
println("\n7. Creating visualization...")

fig = Figure(size=(1600, 1000))

# Row 1: 2D projections (dims 1-2) for each node type
type_colors = Dict("predator" => :red, "prey" => :blue, "resource" => :green)
type_markers = Dict("predator" => :circle, "prey" => :diamond, "resource" => :star5)

# Panel 1: Predator trajectory (dim 1 vs dim 2)
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Dim 1", ylabel="Dim 2",
                       title="Predator (dims 1-2)", aspect=1)

pred_true = viz_data["predator_true"]
pred_full = viz_data["predator_full"]
pred_ude = viz_data["predator_ude"]

lines!(ax1, pred_true[1, :], pred_true[2, :], color=:black, linewidth=2, label="True")
lines!(ax1, pred_full[1, :], pred_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax1, pred_ude[1, :], pred_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
scatter!(ax1, [TARGET_PREDATOR[1]], [TARGET_PREDATOR[2]], color=:red, markersize=15, marker=:star5)

axislegend(ax1, position=:lt)

# Panel 2: Prey trajectory
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Dim 1", ylabel="Dim 2",
                       title="Prey (dims 1-2)", aspect=1)

prey_true = viz_data["prey_true"]
prey_full = viz_data["prey_full"]
prey_ude = viz_data["prey_ude"]

lines!(ax2, prey_true[1, :], prey_true[2, :], color=:black, linewidth=2, label="True")
lines!(ax2, prey_full[1, :], prey_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax2, prey_ude[1, :], prey_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
scatter!(ax2, [TARGET_PREY[1]], [TARGET_PREY[2]], color=:blue, markersize=15, marker=:diamond)

axislegend(ax2, position=:lt)

# Panel 3: Resource trajectory
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="Dim 1", ylabel="Dim 2",
                       title="Resource (dims 1-2)", aspect=1)

res_true = viz_data["resource_true"]
res_full = viz_data["resource_full"]
res_ude = viz_data["resource_ude"]

lines!(ax3, res_true[1, :], res_true[2, :], color=:black, linewidth=2, label="True")
lines!(ax3, res_full[1, :], res_full[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, res_ude[1, :], res_ude[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
scatter!(ax3, [TARGET_RESOURCE[1]], [TARGET_RESOURCE[2]], color=:green, markersize=15, marker=:star5)

axislegend(ax3, position=:lt)

# Row 2: Time series for dimensions 3-4
ax4 = CairoMakie.Axis(fig[2, 1], xlabel="Time", ylabel="Dim 3",
                       title="Predator Dim 3 over time")

lines!(ax4, collect(tsteps), pred_true[3, :], color=:black, linewidth=2, label="True")
lines!(ax4, collect(tsteps), pred_full[3, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax4, collect(tsteps), pred_ude[3, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")

axislegend(ax4, position=:rt)

ax5 = CairoMakie.Axis(fig[2, 2], xlabel="Time", ylabel="Dim 3",
                       title="Prey Dim 3 over time")

lines!(ax5, collect(tsteps), prey_true[3, :], color=:black, linewidth=2)
lines!(ax5, collect(tsteps), prey_full[3, :], color=:blue, linewidth=2, linestyle=:dash)
lines!(ax5, collect(tsteps), prey_ude[3, :], color=:red, linewidth=2, linestyle=:dot)

ax6 = CairoMakie.Axis(fig[2, 3], xlabel="Time", ylabel="Dim 3",
                       title="Resource Dim 3 over time")

lines!(ax6, collect(tsteps), res_true[3, :], color=:black, linewidth=2)
lines!(ax6, collect(tsteps), res_full[3, :], color=:blue, linewidth=2, linestyle=:dash)
lines!(ax6, collect(tsteps), res_ude[3, :], color=:red, linewidth=2, linestyle=:dot)

save("results/example4_food_web.pdf", fig)
println("\nSaved: results/example4_food_web.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 4 - Food Web Heterogeneous Dynamics")
println("=" ^ 60)

println("\nSetup:")
println("  d=4 dimensional B^d_+")
println("  3 node types: Predator (" * string(data.n_predators) *
        "), Prey (" * string(data.n_prey) *
        "), Resource (" * string(data.n_resource) * ")")

println("\nTraining losses:")
println("  Full NN: " * string(round(loss_full_final, digits=6)))
println("  UDE:     " * string(round(loss_ude_final, digits=6)))

println("\nValidation errors:")
println("  Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("  UDE:     " * string(round(mean(errors_ude_val), digits=4)))

println("\nUDE improvement: " * string(round(improvement_val, digits=1)) * "%")

if improvement_val > 0
    println("\n✓ UDE outperforms Full NN on heterogeneous d=4 system!")
else
    println("\n✗ Full NN outperforms UDE.")
end
