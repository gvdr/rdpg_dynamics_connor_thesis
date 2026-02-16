#!/usr/bin/env julia
"""
Example 3a: Wrong Known Part (Clean Data)

Hypothesis: UDE might be WORSE than Full NN when the known structure is wrong.
The rigid incorrect structure constrains the NN to learn corrections from a
wrong baseline. Full NN has freedom to find the right solution.

Setup (using pairwise Lennard-Jones dynamics like Example 2):
- True dynamics: A_ATTRACT=0.02, B_REPEL=0.0008
- UDE correct: Same parameters
- UDE mild: A_ATTRACT=0.024, B_REPEL=0.00096 (20% stronger forces)
- UDE severe: A_ATTRACT=0.04, B_REPEL=0.0016 (100% stronger forces)

Usage:
  julia --project scripts/example3a_wrong_known.jl           # Full run
  julia --project scripts/example3a_wrong_known.jl --viz     # Visualization only
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
using RDPGDynamics: project_embedding_to_Bd_plus

const RESULTS_DIR = "results/example3a"
const DATA_DIR = "data/example3a"
const SAVE_FILE = joinpath(DATA_DIR, "results.jls")

mkpath(RESULTS_DIR)
mkpath(DATA_DIR)

println("=" ^ 60)
println("Example 3a: Wrong Known Part (Pairwise Dynamics)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# System parameters
# =============================================================================

const N_NODES = 30          # Number of nodes
const D = 2                 # Embedding dimension
const STATE_DIM = N_NODES * D

# True dynamics parameters (Lennard-Jones like: -a/r + b/r³)
const TRUE_A = 0.02f0       # Long-range attraction
const TRUE_B = 0.0008f0     # Short-range repulsion
const R_SOFT = 0.05f0       # Soft cutoff for numerical stability

# B^d_+ barrier parameters
const BOUND_K = 0.5f0
const BOUND_SCALE = 0.02f0

# Misspecification levels
const MILD_A = 1.2f0 * TRUE_A    # 20% stronger
const MILD_B = 1.2f0 * TRUE_B
const SEVERE_A = 2.0f0 * TRUE_A  # 100% stronger
const SEVERE_B = 2.0f0 * TRUE_B

# =============================================================================
# Dynamics functions
# =============================================================================

"""
B^d_+ barrier force (non-mutating for Zygote).
"""
function bdplus_barrier(X::AbstractMatrix{T}) where T
    # Positive orthant barrier: repulsion from x=0
    dX_pos = BOUND_K .* exp.(-X ./ BOUND_SCALE)

    # Unit ball barrier: repulsion from ||x||=1
    row_norms = sqrt.(sum(X .^ 2, dims=2))
    # Avoid division by zero
    safe_norms = max.(row_norms, T(0.01))
    unit_dir = X ./ safe_norms
    ball_force = BOUND_K .* exp.((row_norms .- one(T)) ./ BOUND_SCALE)
    dX_ball = -ball_force .* unit_dir

    return dX_pos .+ dX_ball
end

"""
Pairwise Lennard-Jones dynamics with given parameters (non-mutating for Zygote).
"""
function pairwise_dynamics(X::AbstractMatrix{T}, A::T, B::T) where T
    n, d = size(X)

    # Compute pairwise distances
    sq_norms = sum(X .^ 2, dims=2)
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    D_mat = sqrt.(max.(D_sq, T(R_SOFT)^2))

    # Force magnitudes: f(d) = -a/d + b/d³
    F_mag_raw = -A ./ D_mat .+ B ./ (D_mat .^ 3)

    # Zero diagonal using mask (non-mutating)
    diag_mask = one(T) .- T.(I(n))
    F_mag = F_mag_raw .* diag_mask

    # Compute forces for each dimension (vectorized)
    dX_list = map(1:d) do dim
        diff_dim = X[:, dim] .- X[:, dim]'
        sum(F_mag .* diff_dim ./ D_mat, dims=2)
    end
    dX = hcat(dX_list...)

    # Add boundary forces
    return dX .+ bdplus_barrier(X)
end

"""
True dynamics (used for data generation).
"""
function true_dynamics!(du, u, p, t)
    X = reshape(u, N_NODES, D)
    dX = pairwise_dynamics(X, TRUE_A, TRUE_B)
    du .= vec(dX)
end

# =============================================================================
# Generate data
# =============================================================================

function generate_data(; T_end::Float32=40f0, dt::Float32=1f0, seed::Int=42)
    println("\n1. Generating data...")
    rng = Random.MersenneTwister(seed)

    # Initial positions in B^d_+
    X0 = 0.3f0 .+ 0.4f0 .* rand(rng, Float32, N_NODES, D)
    X0 = Float32.(project_embedding_to_Bd_plus(Float64.(X0)))
    u0 = vec(X0)

    tspan = (0f0, T_end)
    tsteps = collect(0f0:dt:T_end)

    prob = ODEProblem(true_dynamics!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-7, reltol=1f-7)

    # Convert to trajectory matrix (STATE_DIM × T)
    T_steps = length(tsteps)
    trajectory = zeros(Float32, STATE_DIM, T_steps)
    for (t_idx, t) in enumerate(tsteps)
        trajectory[:, t_idx] = sol(t)
    end

    println("  n=" * string(N_NODES) * " nodes, d=" * string(D) * ", T=" * string(T_end))
    println("  TRUE A=" * string(TRUE_A) * ", B=" * string(TRUE_B))
    println("  MILD A=" * string(MILD_A) * " (+" * string(round(Int, 100*(MILD_A/TRUE_A - 1))) * "%)")
    println("  SEVERE A=" * string(SEVERE_A) * " (+" * string(round(Int, 100*(SEVERE_A/TRUE_A - 1))) * "%)")

    return (
        trajectory = trajectory,
        u0 = u0,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps
    )
end

# =============================================================================
# Neural network setup
# =============================================================================

println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# Full NN (no prior structure)
nn_full = Lux.Chain(
    Lux.Dense(STATE_DIM, 64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, STATE_DIM)
)

# UDE correction networks (smaller - only learns residuals)
nn_ude = Lux.Chain(
    Lux.Dense(STATE_DIM, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, STATE_DIM)
)

ps_full, st_full = Lux.setup(rng, nn_full)
ps_correct, st_correct = Lux.setup(Random.Xoshiro(42), nn_ude)
ps_mild, st_mild = Lux.setup(Random.Xoshiro(42), nn_ude)
ps_severe, st_severe = Lux.setup(Random.Xoshiro(42), nn_ude)

ps_full_ca = ComponentArray{Float32}(ps_full)
ps_correct_ca = ComponentArray{Float32}(ps_correct)
ps_mild_ca = ComponentArray{Float32}(ps_mild)
ps_severe_ca = ComponentArray{Float32}(ps_severe)

println("  Full NN parameters: " * string(length(ps_full_ca)))
println("  UDE NN parameters: " * string(length(ps_correct_ca)))

# =============================================================================
# Define dynamics functions for training
# =============================================================================

# Full NN dynamics
function full_dynamics(u, p, t)
    u_input = reshape(u, STATE_DIM, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# UDE with CORRECT parameters
function ude_correct_dynamics(u, p, t)
    X = reshape(u, N_NODES, D)
    dx_known = vec(pairwise_dynamics(X, TRUE_A, TRUE_B))
    u_input = reshape(u, STATE_DIM, 1)
    nn_out, _ = nn_ude(u_input, p, st_correct)
    return dx_known .+ vec(nn_out)
end

# UDE with MILD wrong parameters
function ude_mild_dynamics(u, p, t)
    X = reshape(u, N_NODES, D)
    dx_known = vec(pairwise_dynamics(X, MILD_A, MILD_B))
    u_input = reshape(u, STATE_DIM, 1)
    nn_out, _ = nn_ude(u_input, p, st_mild)
    return dx_known .+ vec(nn_out)
end

# UDE with SEVERE wrong parameters
function ude_severe_dynamics(u, p, t)
    X = reshape(u, N_NODES, D)
    dx_known = vec(pairwise_dynamics(X, SEVERE_A, SEVERE_B))
    u_input = reshape(u, STATE_DIM, 1)
    nn_out, _ = nn_ude(u_input, p, st_severe)
    return dx_known .+ vec(nn_out)
end

# =============================================================================
# Training
# =============================================================================

function train_model(dynamics_fn, ps_init, trajectory, tsteps;
                     name="Model", max_adam=400, max_bfgs=100)
    println("\n  Training " * name * "...")

    u0 = trajectory[:, 1]
    tspan = (tsteps[1], tsteps[end])
    T_steps = length(tsteps)

    function predict(p)
        prob = ODEProblem(dynamics_fn, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=tsteps,
                    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                    abstol=1f-5, reltol=1f-5)
        return Array(sol)
    end

    function loss(p)
        pred = predict(p)
        return sum(abs2, trajectory .- pred) / (STATE_DIM * T_steps)
    end

    iter = Ref(0)
    function callback(state, l)
        iter[] += 1
        if iter[] % 50 == 0
            println("    iter " * string(iter[]) * ": loss=" * string(round(l, digits=4)))
        end
        return false
    end

    # Initial loss
    init_loss = loss(ps_init)
    println("    Initial loss: " * string(round(init_loss, digits=4)))

    # Stage 1: ADAM
    optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(0.01f0),
        maxiters=max_adam,
        callback=callback
    )

    # Stage 2: BFGS
    iter[] = max_adam
    optprob2 = Optimization.OptimizationProblem(optf, result_adam.u)

    result = try
        Optimization.solve(
            optprob2,
            OptimizationOptimJL.BFGS(initial_stepnorm=0.01f0),
            maxiters=max_bfgs,
            callback=callback,
            allow_f_increases=false
        )
    catch e
        println("    BFGS failed, using ADAM result")
        result_adam
    end

    final_loss = loss(result.u)
    println("    Final loss: " * string(round(final_loss, digits=6)))

    # Return trained parameters and predictions
    final_pred = predict(result.u)
    return result.u, final_loss, final_pred
end

# =============================================================================
# Evaluation metrics
# =============================================================================

function compute_metrics(pred, true_traj)
    # Position MSE
    mse = mean(abs2, pred .- true_traj)

    # Distance correlation (reshape to n×d×T)
    T_steps = size(pred, 2)
    d_corrs = Float64[]

    for t in 1:T_steps
        X_pred = reshape(pred[:, t], N_NODES, D)
        X_true = reshape(true_traj[:, t], N_NODES, D)

        # Compute pairwise distances
        D_pred = [norm(X_pred[i, :] - X_pred[j, :]) for i in 1:N_NODES for j in i+1:N_NODES]
        D_true = [norm(X_true[i, :] - X_true[j, :]) for i in 1:N_NODES for j in i+1:N_NODES]

        if std(D_pred) > 1e-6 && std(D_true) > 1e-6
            push!(d_corrs, cor(D_pred, D_true))
        end
    end

    avg_d_corr = isempty(d_corrs) ? 0.0 : mean(d_corrs)

    return (mse=mse, d_corr=avg_d_corr)
end

# =============================================================================
# Main execution
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n3-4. Loading saved results...")
    saved = deserialize(SAVE_FILE)
    results = saved
    println("  Loaded!")

else  # Run training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

data = generate_data()

println("\n3. Training all models...")

ps_full_trained, loss_full, pred_full = train_model(
    full_dynamics, ps_full_ca, data.trajectory, data.tsteps, name="Full NN")

ps_correct_trained, loss_correct, pred_correct = train_model(
    ude_correct_dynamics, ps_correct_ca, data.trajectory, data.tsteps, name="UDE (correct params)")

ps_mild_trained, loss_mild, pred_mild = train_model(
    ude_mild_dynamics, ps_mild_ca, data.trajectory, data.tsteps, name="UDE (20% wrong)")

ps_severe_trained, loss_severe, pred_severe = train_model(
    ude_severe_dynamics, ps_severe_ca, data.trajectory, data.tsteps, name="UDE (100% wrong)")

# Compute metrics
println("\n4. Computing evaluation metrics...")

metrics_full = compute_metrics(pred_full, data.trajectory)
metrics_correct = compute_metrics(pred_correct, data.trajectory)
metrics_mild = compute_metrics(pred_mild, data.trajectory)
metrics_severe = compute_metrics(pred_severe, data.trajectory)

println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)

println("\nTraining Loss:")
println("  Full NN:           " * string(round(loss_full, digits=6)))
println("  UDE (correct):     " * string(round(loss_correct, digits=6)))
println("  UDE (20% wrong):   " * string(round(loss_mild, digits=6)))
println("  UDE (100% wrong):  " * string(round(loss_severe, digits=6)))

println("\nDistance Correlation:")
println("  Full NN:           " * string(round(metrics_full.d_corr, digits=4)))
println("  UDE (correct):     " * string(round(metrics_correct.d_corr, digits=4)))
println("  UDE (20% wrong):   " * string(round(metrics_mild.d_corr, digits=4)))
println("  UDE (100% wrong):  " * string(round(metrics_severe.d_corr, digits=4)))

println("\nRelative to Full NN (positive = better than Full NN):")
println("  UDE (correct):     " * string(round(100*(metrics_correct.d_corr/metrics_full.d_corr - 1), digits=1)) * "%")
println("  UDE (20% wrong):   " * string(round(100*(metrics_mild.d_corr/metrics_full.d_corr - 1), digits=1)) * "%")
println("  UDE (100% wrong):  " * string(round(100*(metrics_severe.d_corr/metrics_full.d_corr - 1), digits=1)) * "%")

# Save results
results = Dict(
    "data" => data,
    "loss_full" => loss_full,
    "loss_correct" => loss_correct,
    "loss_mild" => loss_mild,
    "loss_severe" => loss_severe,
    "metrics_full" => metrics_full,
    "metrics_correct" => metrics_correct,
    "metrics_mild" => metrics_mild,
    "metrics_severe" => metrics_severe,
    "pred_full" => pred_full,
    "pred_correct" => pred_correct,
    "pred_mild" => pred_mild,
    "pred_severe" => pred_severe,
    "ps_full" => ps_full_trained,
    "ps_correct" => ps_correct_trained,
    "ps_mild" => ps_mild_trained,
    "ps_severe" => ps_severe_trained
)

serialize(SAVE_FILE, results)
println("\nSaved results to " * SAVE_FILE)

end  # if VIZ_ONLY

# =============================================================================
# Visualization
# =============================================================================

println("\n5. Generating visualizations...")

data = results["data"]
pred_full = results["pred_full"]
pred_correct = results["pred_correct"]
pred_mild = results["pred_mild"]
pred_severe = results["pred_severe"]

# Plot 1: Trajectory comparison for a few nodes
fig1 = Figure(size=(1200, 800))

# Select representative nodes
plot_nodes = [1, 5, 10, 15]
colors = [:blue, :red, :orange, :purple]
labels = ["Full NN", "UDE correct", "UDE 20% wrong", "UDE 100% wrong"]
preds = [pred_full, pred_correct, pred_mild, pred_severe]

for (row, node_i) in enumerate(plot_nodes)
    ax = Axis(fig1[row, 1],
              xlabel=row==length(plot_nodes) ? "x₁" : "",
              ylabel="x₂",
              title="Node " * string(node_i))

    # True trajectory
    true_x1 = [data.trajectory[(node_i-1)*D + 1, t] for t in 1:length(data.tsteps)]
    true_x2 = [data.trajectory[(node_i-1)*D + 2, t] for t in 1:length(data.tsteps)]
    lines!(ax, true_x1, true_x2, color=:black, linewidth=2, label="True")

    # Model predictions
    for (pred, color, label) in zip(preds, colors, labels)
        pred_x1 = [pred[(node_i-1)*D + 1, t] for t in 1:length(data.tsteps)]
        pred_x2 = [pred[(node_i-1)*D + 2, t] for t in 1:length(data.tsteps)]
        lines!(ax, pred_x1, pred_x2, color=color, linewidth=1.5, linestyle=:dash, label=label)
    end

    if row == 1
        axislegend(ax, position=:rt)
    end
end

save(joinpath(RESULTS_DIR, "trajectory_comparison.pdf"), fig1)
println("  Saved: trajectory_comparison.pdf")

# Plot 2: Bar chart of metrics
fig2 = Figure(size=(800, 400))

ax2 = Axis(fig2[1, 1],
           xlabel="Model",
           ylabel="Distance Correlation",
           title="Example 3a: Effect of Wrong Known Structure",
           xticks=(1:4, ["Full NN", "UDE\n(correct)", "UDE\n(20% wrong)", "UDE\n(100% wrong)"]))

d_corrs = [results["metrics_full"].d_corr,
           results["metrics_correct"].d_corr,
           results["metrics_mild"].d_corr,
           results["metrics_severe"].d_corr]

barplot!(ax2, 1:4, d_corrs, color=[:gray, :green, :orange, :red])

# Add horizontal line for Full NN baseline
hlines!(ax2, [results["metrics_full"].d_corr], color=:black, linestyle=:dash,
        label="Full NN baseline")

save(joinpath(RESULTS_DIR, "metrics_comparison.pdf"), fig2)
println("  Saved: metrics_comparison.pdf")

# Plot 3: Loss comparison
fig3 = Figure(size=(800, 400))

ax3 = Axis(fig3[1, 1],
           xlabel="Model",
           ylabel="Training Loss (log scale)",
           title="Training Loss by Model",
           xticks=(1:4, ["Full NN", "UDE\n(correct)", "UDE\n(20% wrong)", "UDE\n(100% wrong)"]),
           yscale=log10)

losses = [results["loss_full"], results["loss_correct"],
          results["loss_mild"], results["loss_severe"]]

barplot!(ax3, 1:4, losses, color=[:gray, :green, :orange, :red])

save(joinpath(RESULTS_DIR, "loss_comparison.pdf"), fig3)
println("  Saved: loss_comparison.pdf")

println("\n" * "=" ^ 60)
println("Example 3a complete!")
println("=" ^ 60)
