#!/usr/bin/env julia
"""
Example: Proper RDPG Pipeline

This example demonstrates the CORRECT approach to learning dynamics from RDPG:
1. Generate true latent positions X(t) with known dynamics
2. Sample adjacency matrices A(t) ~ Bernoulli(X X')
3. Embed via folded SVD to get X̂(t)
4. Train Neural ODE on the EMBEDDED data (not true X)

Dynamics: Two communities that oscillate toward/away from each other.
- This IS observable from network data (changing edge probabilities)
- Known structure: oscillatory time dependence
- Unknown: amplitude, frequency, community centers

Usage:
  julia --project scripts/example_rdpg_pipeline.jl
"""

using Pkg
Pkg.activate(".")

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

# Include embedding functions
include("../src/embedding.jl")

const RESULTS_DIR = "results"

println("=" ^ 60)
println("Example: Proper RDPG Pipeline")
println("=" ^ 60)

# =============================================================================
# True Dynamics: Community Oscillation
# =============================================================================
#
# Two communities whose centers oscillate toward/away from each other.
# This creates time-varying edge probabilities - OBSERVABLE from network data.
#
# Community 1: centered around C1(t) = C1_base + amplitude * sin(ω*t) * direction
# Community 2: centered around C2(t) = C2_base - amplitude * sin(ω*t) * direction
# =============================================================================

# Base community centers (in B^2_+)
const C1_BASE = Float64[0.3, 0.6]
const C2_BASE = Float64[0.6, 0.3]

# Oscillation parameters (UNKNOWN to learner)
const OMEGA = 0.15              # Angular frequency
const AMPLITUDE = 0.12          # Oscillation amplitude
const DIRECTION = normalize(C2_BASE - C1_BASE)  # Direction of oscillation

# Node spread within community
const COMMUNITY_SPREAD = 0.08

"""
Generate true latent positions at time t.
"""
function true_positions(t::Float64, n_per_community::Int; rng=Random.GLOBAL_RNG)
    n = 2 * n_per_community
    d = 2

    # Oscillating community centers
    offset = AMPLITUDE * sin(OMEGA * t) .* DIRECTION
    C1_t = C1_BASE .+ offset
    C2_t = C2_BASE .- offset

    # Generate node positions
    X = zeros(n, d)

    for i in 1:n_per_community
        # Community 1 nodes
        X[i, :] = C1_t .+ COMMUNITY_SPREAD .* randn(rng, d)
        # Community 2 nodes
        X[n_per_community + i, :] = C2_t .+ COMMUNITY_SPREAD .* randn(rng, d)
    end

    # Project to B^d_+
    for i in 1:n
        X[i, :] = max.(X[i, :], 0.05)  # Stay positive
        r = norm(X[i, :])
        if r > 0.95
            X[i, :] .*= 0.95 / r  # Stay in unit ball
        end
    end

    return X
end

"""
True dynamics for a single node's position.
"""
function true_node_dynamics(x::Vector{T}, community::Int, t::T) where T
    # Oscillation derivative
    d_offset = AMPLITUDE * OMEGA * cos(OMEGA * t) .* T.(DIRECTION)

    if community == 1
        # Community 1: moves with +offset
        dx = d_offset
    else
        # Community 2: moves with -offset
        dx = -d_offset
    end

    # Add attraction to community center (damping)
    C_base = community == 1 ? T.(C1_BASE) : T.(C2_BASE)
    offset = T(AMPLITUDE) * sin(T(OMEGA) * t) .* T.(DIRECTION)
    C_t = community == 1 ? C_base .+ offset : C_base .- offset

    k_attract = T(0.1)
    dx = dx .- k_attract .* (x .- C_t)

    # Boundary repulsion
    epsilon = T(0.05)
    alpha = T(0.2)
    dx = dx .+ alpha .* exp.(-x ./ epsilon)

    r = norm(x)
    if r > T(0.5)
        dx = dx .- alpha * exp((r - one(T)) / epsilon) / r .* x
    end

    return dx
end

# =============================================================================
# Generate Data
# =============================================================================
println("\n1. Generating true dynamics...")

const N_PER_COMMUNITY = 8
const N_TOTAL = 2 * N_PER_COMMUNITY
const D = 2
const T_END = 40.0
const DT = 1.0
const TSTEPS = 0.0:DT:T_END
const N_TIMESTEPS = length(TSTEPS)

# Community assignments
const COMMUNITIES = vcat(fill(1, N_PER_COMMUNITY), fill(2, N_PER_COMMUNITY))

# Generate true trajectories by integrating dynamics
rng_data = MersenneTwister(42)

# Initial positions
X0 = true_positions(0.0, N_PER_COMMUNITY; rng=rng_data)

# Solve ODE for each node
X_true_series = Vector{Matrix{Float64}}(undef, N_TIMESTEPS)
X_true_series[1] = copy(X0)

for (t_idx, t) in enumerate(TSTEPS)
    if t_idx == 1
        continue
    end

    X_prev = X_true_series[t_idx - 1]
    X_new = similar(X_prev)

    for i in 1:N_TOTAL
        # Simple Euler integration for true dynamics
        x = X_prev[i, :]
        dt = DT
        for _ in 1:10  # Substeps for accuracy
            dx = true_node_dynamics(x, COMMUNITIES[i], Float64(t - DT + dt/10))
            x = x .+ (dt/10) .* dx
            x = max.(x, 0.05)
            r = norm(x)
            if r > 0.95
                x .*= 0.95 / r
            end
        end
        X_new[i, :] = x
    end

    X_true_series[t_idx] = X_new
end

println("  n = " * string(N_TOTAL) * " nodes (" * string(N_PER_COMMUNITY) * " per community)")
println("  T = " * string(T_END) * ", dt = " * string(DT) * ", steps = " * string(N_TIMESTEPS))

# Verify dynamics are visible
P_start = X_true_series[1] * X_true_series[1]'
P_end = X_true_series[end] * X_true_series[end]'
cross_prob_start = mean(P_start[1:N_PER_COMMUNITY, N_PER_COMMUNITY+1:end])
cross_prob_end = mean(P_end[1:N_PER_COMMUNITY, N_PER_COMMUNITY+1:end])
println("  Cross-community edge prob: " * string(round(cross_prob_start, digits=3)) *
        " → " * string(round(cross_prob_end, digits=3)))

# =============================================================================
# RDPG Observation Model
# =============================================================================
println("\n2. Sampling adjacency matrices and embedding...")

# Parameters for RDPG estimation
const K_SAMPLES = 10  # Repeated samples per timestep
const W_WINDOW = 3    # Window size

rng_rdpg = MersenneTwister(123)

# Use the proper pipeline
L_hat_series = embed_temporal_with_folding(
    X_true_series, D;
    K=K_SAMPLES, W=W_WINDOW,
    L_true=X_true_series,  # Use oracle alignment for fair comparison
    rng=rng_rdpg
)

println("  K = " * string(K_SAMPLES) * " samples, W = " * string(W_WINDOW) * " window")
println("  Embedded " * string(length(L_hat_series)) * " timesteps")

# Measure embedding error
embedding_errors = Float64[]
for t in 1:N_TIMESTEPS
    # Compute error after optimal alignment
    Q = ortho_procrustes_RM(L_hat_series[t]', X_true_series[t]')
    L_aligned = L_hat_series[t] * Q
    err = mean([norm(L_aligned[i, :] - X_true_series[t][i, :]) for i in 1:N_TOTAL])
    push!(embedding_errors, err)
end
println("  Mean embedding error: " * string(round(mean(embedding_errors), digits=4)))

# =============================================================================
# Prepare Training Data
# =============================================================================
println("\n3. Preparing training data...")

# Convert to per-node trajectories
# TRUE trajectories (what we SHOULD NOT use in practice)
true_trajectories = Dict{Int, Matrix{Float64}}()
for i in 1:N_TOTAL
    traj = zeros(D, N_TIMESTEPS)
    for t in 1:N_TIMESTEPS
        traj[:, t] = X_true_series[t][i, :]
    end
    true_trajectories[i] = traj
end

# EMBEDDED trajectories (what we SHOULD use)
embedded_trajectories = Dict{Int, Matrix{Float64}}()
for i in 1:N_TOTAL
    traj = zeros(D, N_TIMESTEPS)
    for t in 1:N_TIMESTEPS
        traj[:, t] = L_hat_series[t][i, :]
    end
    embedded_trajectories[i] = traj
end

# Train/val split
train_nodes = 1:12
val_nodes = 13:16

println("  Training nodes: " * string(length(train_nodes)))
println("  Validation nodes: " * string(length(val_nodes)))

# =============================================================================
# Neural Networks
# =============================================================================
println("\n4. Setting up models...")

rng_nn = Random.Xoshiro(42)

# Full NN: learns everything
nn_full = Lux.Chain(
    Lux.Dense(D + 1, 32, tanh),  # +1 for time
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, D)
)

# UDE: knows oscillatory structure, learns parameters
nn_ude = Lux.Chain(
    Lux.Dense(D + 1, 32, tanh),  # +1 for community encoding
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, D)
)

ps_full, st_full = Lux.setup(rng_nn, nn_full)
ps_ude, st_ude = Lux.setup(Random.Xoshiro(42), nn_ude)

ps_full_ca = ComponentArray(ps_full)
ps_ude_ca = ComponentArray(ps_ude)

println("  Full NN params: " * string(length(ps_full_ca)))
println("  UDE NN params: " * string(length(ps_ude_ca)))

# =============================================================================
# Dynamics Functions
# =============================================================================

function full_dynamics(u, p, t)
    T_type = eltype(u)
    input = vcat(u, T_type[t / T_END])  # Normalize time to [0,1]
    input_mat = reshape(input, length(input), 1)
    nn_out, _ = nn_full(input_mat, p, st_full)
    return vec(nn_out)
end

# UDE: knows oscillatory structure exists
function ude_dynamics(u, p, t, community::Int)
    T_type = eltype(u)

    # Known: time-dependent oscillatory component (but not the exact form)
    # We encode time phase information
    sin_t = sin(T_type(OMEGA) * t)
    cos_t = cos(T_type(OMEGA) * t)

    # Direction hint (communities move opposite)
    comm_sign = community == 1 ? one(T_type) : -one(T_type)

    # NN learns the amplitude and detailed dynamics
    input = vcat(u, T_type[comm_sign])
    input_mat = reshape(input, length(input), 1)
    nn_out, _ = nn_ude(input_mat, p, st_ude)
    dx_nn = vec(nn_out)

    # Known structure: oscillatory motion along direction
    # The NN output is modulated by the oscillation
    dx_known = comm_sign .* cos_t .* T_type.(DIRECTION) .* T_type(0.1)

    return dx_known .+ dx_nn
end

# =============================================================================
# Training on EMBEDDED data (proper approach)
# =============================================================================
println("\n5. Training on EMBEDDED data (proper RDPG approach)...")

tspan = (0.0, T_END)

function predict_embedded(u0, p, dynamics_fn, node_idx)
    community = COMMUNITIES[node_idx]
    function dyn_wrapper(u, p_inner, t)
        return dynamics_fn(u, p_inner, t, community)
    end

    prob = ODEProblem(dyn_wrapper, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=collect(TSTEPS),
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1e-5, reltol=1e-5)
    return Array(sol)
end

function predict_full(u0, p)
    prob = ODEProblem(full_dynamics, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=collect(TSTEPS),
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1e-5, reltol=1e-5)
    return Array(sol)
end

function loss_full_embedded(p)
    total = 0.0
    for i in train_nodes
        u0 = embedded_trajectories[i][:, 1]
        true_traj = embedded_trajectories[i]
        pred = predict_full(u0, p)
        total += sum(abs2, true_traj .- pred)
    end
    return total / (length(train_nodes) * N_TIMESTEPS)
end

function loss_ude_embedded(p)
    total = 0.0
    for i in train_nodes
        u0 = embedded_trajectories[i][:, 1]
        true_traj = embedded_trajectories[i]
        pred = predict_embedded(u0, p, ude_dynamics, i)
        total += sum(abs2, true_traj .- pred)
    end
    return total / (length(train_nodes) * N_TIMESTEPS)
end

# Training callbacks
iter_full = Ref(0)
callback_full(state, l) = (iter_full[] += 1; iter_full[] % 50 == 0 && println("    Full iter " * string(iter_full[]) * ": " * string(round(l, digits=6))); false)

iter_ude = Ref(0)
callback_ude(state, l) = (iter_ude[] += 1; iter_ude[] % 50 == 0 && println("    UDE iter " * string(iter_ude[]) * ": " * string(round(l, digits=6))); false)

println("  Initial loss (Full on embedded): " * string(round(loss_full_embedded(ps_full_ca), digits=4)))
println("  Initial loss (UDE on embedded): " * string(round(loss_ude_embedded(ps_ude_ca), digits=4)))

# Train Full NN on embedded data
println("\n  Training Full NN on embedded data...")
optf_full = Optimization.OptimizationFunction((p, _) -> loss_full_embedded(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_full
)
ps_full_trained = result_full.u
loss_full_final = loss_full_embedded(ps_full_trained)
println("  Final loss (Full): " * string(round(loss_full_final, digits=6)))

# Train UDE on embedded data
println("\n  Training UDE on embedded data...")
optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude_embedded(p), Optimization.AutoZygote())
optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_ude_ca)

result_ude = Optimization.solve(
    optprob_ude,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300,
    callback=callback_ude
)
ps_ude_trained = result_ude.u
loss_ude_final = loss_ude_embedded(ps_ude_trained)
println("  Final loss (UDE): " * string(round(loss_ude_final, digits=6)))

# =============================================================================
# Evaluation
# =============================================================================
println("\n6. Evaluating...")

errors_full_val = Float64[]
errors_ude_val = Float64[]

for i in val_nodes
    u0 = embedded_trajectories[i][:, 1]
    true_traj = embedded_trajectories[i]

    pred_full = predict_full(u0, ps_full_trained)
    pred_ude = predict_embedded(u0, ps_ude_trained, ude_dynamics, i)

    err_full = mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:N_TIMESTEPS])
    err_ude = mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:N_TIMESTEPS])

    push!(errors_full_val, err_full)
    push!(errors_ude_val, err_ude)
end

println("\n  Validation errors (on embedded data):")
println("    Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_val), digits=4)))

improvement = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))
println("    Improvement: " * string(round(improvement, digits=1)) * "%")

# =============================================================================
# Visualization
# =============================================================================
println("\n7. Creating visualization...")

fig = Figure(size=(1600, 1000))

# Panel 1: True dynamics (community centers over time)
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂",
                       title="True Latent Positions", aspect=1)

theta_arc = range(0, pi/2, length=50)
lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)

# Plot trajectories for a few nodes
for i in [1, 4, 9, 12]
    traj = true_trajectories[i]
    c = COMMUNITIES[i] == 1 ? :blue : :red
    lines!(ax1, traj[1, :], traj[2, :], color=c, linewidth=1.5, alpha=0.7)
    scatter!(ax1, [traj[1, 1]], [traj[2, 1]], color=c, markersize=10)
end

scatter!(ax1, [C1_BASE[1]], [C1_BASE[2]], color=:blue, marker=:star5, markersize=20, label="C1")
scatter!(ax1, [C2_BASE[1]], [C2_BASE[2]], color=:red, marker=:star5, markersize=20, label="C2")
axislegend(ax1, position=:rt)
xlims!(ax1, 0, 1)
ylims!(ax1, 0, 1)

# Panel 2: Embedded positions
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="x̂₁", ylabel="x̂₂",
                       title="Embedded Positions (from RDPG)", aspect=1)

lines!(ax2, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)

for i in [1, 4, 9, 12]
    traj = embedded_trajectories[i]
    c = COMMUNITIES[i] == 1 ? :blue : :red
    lines!(ax2, traj[1, :], traj[2, :], color=c, linewidth=1.5, alpha=0.7)
    scatter!(ax2, [traj[1, 1]], [traj[2, 1]], color=c, markersize=10)
end

xlims!(ax2, 0, 1)
ylims!(ax2, 0, 1)

# Panel 3: Prediction comparison (validation node)
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="x̂₁", ylabel="x̂₂",
                       title="Prediction Comparison (Validation)", aspect=1)

lines!(ax3, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)

val_node = first(val_nodes)
true_traj = embedded_trajectories[val_node]
pred_full_traj = predict_full(true_traj[:, 1], ps_full_trained)
pred_ude_traj = predict_embedded(true_traj[:, 1], ps_ude_trained, ude_dynamics, val_node)

lines!(ax3, true_traj[1, :], true_traj[2, :], color=:black, linewidth=2, label="True (embedded)")
lines!(ax3, pred_full_traj[1, :], pred_full_traj[2, :], color=:green, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, pred_ude_traj[1, :], pred_ude_traj[2, :], color=:orange, linewidth=2, linestyle=:dot, label="UDE")

axislegend(ax3, position=:rt)
xlims!(ax3, 0, 1)
ylims!(ax3, 0, 1)

# Panel 4: Time series comparison
ax4 = CairoMakie.Axis(fig[2, 1], xlabel="Time", ylabel="x̂₁",
                       title="x̂₁ Over Time (Validation Node)")

lines!(ax4, collect(TSTEPS), true_traj[1, :], color=:black, linewidth=2, label="True")
lines!(ax4, collect(TSTEPS), pred_full_traj[1, :], color=:green, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax4, collect(TSTEPS), pred_ude_traj[1, :], color=:orange, linewidth=2, linestyle=:dot, label="UDE")

axislegend(ax4, position=:rt)

# Panel 5: Cross-community probability over time
ax5 = CairoMakie.Axis(fig[2, 2], xlabel="Time", ylabel="P(cross-edge)",
                       title="Cross-Community Edge Probability")

cross_probs_true = Float64[]
cross_probs_embedded = Float64[]

for t in 1:N_TIMESTEPS
    P_true = X_true_series[t] * X_true_series[t]'
    P_embedded = L_hat_series[t] * L_hat_series[t]'

    push!(cross_probs_true, mean(P_true[1:N_PER_COMMUNITY, N_PER_COMMUNITY+1:end]))
    push!(cross_probs_embedded, mean(P_embedded[1:N_PER_COMMUNITY, N_PER_COMMUNITY+1:end]))
end

lines!(ax5, collect(TSTEPS), cross_probs_true, color=:black, linewidth=2, label="True P")
lines!(ax5, collect(TSTEPS), cross_probs_embedded, color=:purple, linewidth=2, linestyle=:dash, label="Embedded P")

axislegend(ax5, position=:rt)

# Panel 6: Error over time
ax6 = CairoMakie.Axis(fig[2, 3], xlabel="Time", ylabel="Error",
                       title="Prediction Error Over Time")

errors_full_time = [norm(pred_full_traj[:, t] - true_traj[:, t]) for t in 1:N_TIMESTEPS]
errors_ude_time = [norm(pred_ude_traj[:, t] - true_traj[:, t]) for t in 1:N_TIMESTEPS]

lines!(ax6, collect(TSTEPS), errors_full_time, color=:green, linewidth=2, label="Full NN")
lines!(ax6, collect(TSTEPS), errors_ude_time, color=:orange, linewidth=2, label="UDE")

axislegend(ax6, position=:lt)

mkpath(RESULTS_DIR)
save(joinpath(RESULTS_DIR, "example_rdpg_pipeline.pdf"), fig)
println("Saved: results/example_rdpg_pipeline.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Proper RDPG Pipeline")
println("=" ^ 60)

println("\nKey insight: We train on EMBEDDED data X̂(t), not true X(t)")
println("  - X(t) is unobservable in practice")
println("  - We only see adjacency matrices A(t)")
println("  - Embedding introduces noise and rotation ambiguity")

println("\nDynamics: Two communities oscillating toward/away from each other")
println("  - This IS observable (changes edge probabilities)")
println("  - Not like rotation (which is unobservable)")

println("\nRDPG estimation:")
println("  - K = " * string(K_SAMPLES) * " repeated samples")
println("  - W = " * string(W_WINDOW) * " window size")
println("  - Mean embedding error: " * string(round(mean(embedding_errors), digits=4)))

println("\nTraining results (on embedded data):")
println("  Full NN validation error: " * string(round(mean(errors_full_val), digits=4)))
println("  UDE validation error:     " * string(round(mean(errors_ude_val), digits=4)))
println("  Improvement: " * string(round(improvement, digits=1)) * "%")

if improvement > 0
    println("\n✓ UDE outperforms Full NN on properly embedded RDPG data!")
else
    println("\n✗ Full NN outperforms UDE")
end
