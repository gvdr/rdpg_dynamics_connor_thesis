#!/usr/bin/env julia
"""
Example 3b: Correct Known Part (Noisy Data)

Hypothesis: UDE might be BETTER than Full NN when the known structure is correct
but data is noisy. The correct structure acts as regularization, preventing
overfitting to noise. Full NN might chase the noise.

Setup (using pairwise Lennard-Jones dynamics like Example 2):
- True dynamics: A_ATTRACT=0.02, B_REPEL=0.0008
- UDE uses: Correct parameters (same as true)
- Full NN: No prior structure

Noise levels tested:
- Low:    σ = 0.01 (1% of typical position magnitude)
- Medium: σ = 0.03 (3%)
- High:   σ = 0.05 (5%)

Usage:
  julia --project scripts/example3b_noisy_data.jl           # Full run
  julia --project scripts/example3b_noisy_data.jl --viz     # Visualization only
"""

using Pkg
Pkg.activate(".")

VIZ_ONLY = "--viz" in ARGS || "-v" in ARGS

using LinearAlgebra
using Random
using Statistics
import CairoMakie as CM
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

const RESULTS_DIR = "results/example3b"
const DATA_DIR = "data/example3b"
const SAVE_FILE = joinpath(DATA_DIR, "results.jls")

mkpath(RESULTS_DIR)
mkpath(DATA_DIR)

println("=" ^ 60)
println("Example 3b: Correct Known Part (Noisy Data)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# System parameters
# =============================================================================

const N_NODES = 30
const D = 2
const STATE_DIM = N_NODES * D

# True dynamics parameters
const TRUE_A = 0.02f0
const TRUE_B = 0.0008f0
const R_SOFT = 0.05f0

# B^d_+ barrier parameters
const BOUND_K = 0.5f0
const BOUND_SCALE = 0.02f0

# Noise levels
const NOISE_LOW = 0.01f0
const NOISE_MEDIUM = 0.03f0
const NOISE_HIGH = 0.05f0

# =============================================================================
# Dynamics functions (same as Example 3a)
# =============================================================================

function bdplus_barrier(X::AbstractMatrix{T}) where T
    # Positive orthant barrier: repulsion from x=0
    dX_pos = BOUND_K .* exp.(-X ./ BOUND_SCALE)

    # Unit ball barrier: repulsion from ||x||=1
    row_norms = sqrt.(sum(X .^ 2, dims=2))
    safe_norms = max.(row_norms, T(0.01))
    unit_dir = X ./ safe_norms
    ball_force = BOUND_K .* exp.((row_norms .- one(T)) ./ BOUND_SCALE)
    dX_ball = -ball_force .* unit_dir

    return dX_pos .+ dX_ball
end

function pairwise_dynamics(X::AbstractMatrix{T}, A::T, B::T) where T
    n, d = size(X)

    sq_norms = sum(X .^ 2, dims=2)
    D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
    D_mat = sqrt.(max.(D_sq, T(R_SOFT)^2))

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

    return dX .+ bdplus_barrier(X)
end

function true_dynamics!(du, u, p, t)
    X = reshape(u, N_NODES, D)
    dX = pairwise_dynamics(X, TRUE_A, TRUE_B)
    du .= vec(dX)
end

# =============================================================================
# Generate data with noise
# =============================================================================

function generate_data(; T_end::Float32=40f0, dt::Float32=1f0, noise_level::Float32=0f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    # Initial positions in B^d_+
    X0 = 0.3f0 .+ 0.4f0 .* rand(rng, Float32, N_NODES, D)
    X0 = Float32.(project_embedding_to_Bd_plus(Float64.(X0)))
    u0 = vec(X0)

    tspan = (0f0, T_end)
    tsteps = collect(0f0:dt:T_end)

    prob = ODEProblem(true_dynamics!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-7, reltol=1f-7)

    T_steps = length(tsteps)
    clean_trajectory = zeros(Float32, STATE_DIM, T_steps)
    for (t_idx, t) in enumerate(tsteps)
        clean_trajectory[:, t_idx] = sol(t)
    end

    # Add Gaussian noise
    noisy_trajectory = clean_trajectory .+ noise_level .* randn(rng, Float32, STATE_DIM, T_steps)

    return (
        clean_trajectory = clean_trajectory,
        noisy_trajectory = noisy_trajectory,
        u0 = u0,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        noise_level = noise_level
    )
end

# =============================================================================
# Neural network setup
# =============================================================================

function setup_networks()
    rng = Random.Xoshiro(42)

    # Full NN
    nn_full = Lux.Chain(
        Lux.Dense(STATE_DIM, 64, tanh),
        Lux.Dense(64, 64, tanh),
        Lux.Dense(64, STATE_DIM)
    )

    # UDE correction network
    nn_ude = Lux.Chain(
        Lux.Dense(STATE_DIM, 32, tanh),
        Lux.Dense(32, 32, tanh),
        Lux.Dense(32, STATE_DIM)
    )

    ps_full, st_full = Lux.setup(rng, nn_full)
    ps_ude, st_ude = Lux.setup(Random.Xoshiro(42), nn_ude)

    return (
        nn_full = nn_full, ps_full = ComponentArray{Float32}(ps_full), st_full = st_full,
        nn_ude = nn_ude, ps_ude = ComponentArray{Float32}(ps_ude), st_ude = st_ude
    )
end

# =============================================================================
# Training
# =============================================================================

function train_model(dynamics_fn, ps_init, trajectory, tsteps;
                     name="Model", max_adam=400, max_bfgs=100)
    println("    Training " * name * "...")

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
        if iter[] % 100 == 0
            println("      iter " * string(iter[]) * ": loss=" * string(round(l, digits=4)))
        end
        return false
    end

    init_loss = loss(ps_init)
    println("      Initial loss: " * string(round(init_loss, digits=4)))

    optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(0.01f0),
        maxiters=max_adam,
        callback=callback
    )

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
        println("      BFGS failed, using ADAM result")
        result_adam
    end

    final_loss = loss(result.u)
    println("      Final loss: " * string(round(final_loss, digits=6)))

    final_pred = predict(result.u)
    return result.u, final_loss, final_pred
end

# =============================================================================
# Evaluation metrics
# =============================================================================

function compute_metrics(pred, clean_traj)
    # MSE against CLEAN trajectory (not noisy training data)
    mse = mean(abs2, pred .- clean_traj)

    T_steps = size(pred, 2)
    d_corrs = Float64[]

    for t in 1:T_steps
        X_pred = reshape(pred[:, t], N_NODES, D)
        X_true = reshape(clean_traj[:, t], N_NODES, D)

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
    println("\nLoading saved results...")
    results = deserialize(SAVE_FILE)
    println("  Loaded!")

else

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

println("\n1. Generating data at multiple noise levels...")

data_clean = generate_data(noise_level=0f0)
data_low = generate_data(noise_level=NOISE_LOW)
data_medium = generate_data(noise_level=NOISE_MEDIUM)
data_high = generate_data(noise_level=NOISE_HIGH)

println("  Clean data generated")
println("  Low noise (σ=" * string(NOISE_LOW) * ") generated")
println("  Medium noise (σ=" * string(NOISE_MEDIUM) * ") generated")
println("  High noise (σ=" * string(NOISE_HIGH) * ") generated")

# Store all results
results = Dict{String, Any}()
results["clean_trajectory"] = data_clean.clean_trajectory
results["tsteps"] = data_clean.tsteps

noise_levels = [
    ("low", NOISE_LOW, data_low),
    ("medium", NOISE_MEDIUM, data_medium),
    ("high", NOISE_HIGH, data_high)
]

for (noise_name, noise_val, data) in noise_levels
    println("\n" * "=" ^ 60)
    println("Training with " * uppercase(noise_name) * " noise (σ=" * string(noise_val) * ")")
    println("=" ^ 60)

    # Fresh networks for each noise level
    nets = setup_networks()

    # Create dynamics functions with these networks
    function full_dynamics(u, p, t)
        u_input = reshape(u, STATE_DIM, 1)
        nn_out, _ = nets.nn_full(u_input, p, nets.st_full)
        return vec(nn_out)
    end

    function ude_dynamics(u, p, t)
        X = reshape(u, N_NODES, D)
        dx_known = vec(pairwise_dynamics(X, TRUE_A, TRUE_B))
        u_input = reshape(u, STATE_DIM, 1)
        nn_out, _ = nets.nn_ude(u_input, p, nets.st_ude)
        return dx_known .+ vec(nn_out)
    end

    # Train on NOISY data
    ps_full_trained, loss_full, pred_full = train_model(
        full_dynamics, nets.ps_full, data.noisy_trajectory, data.tsteps,
        name="Full NN")

    ps_ude_trained, loss_ude, pred_ude = train_model(
        ude_dynamics, nets.ps_ude, data.noisy_trajectory, data.tsteps,
        name="UDE (correct params)")

    # Evaluate on CLEAN trajectory
    metrics_full = compute_metrics(pred_full, data.clean_trajectory)
    metrics_ude = compute_metrics(pred_ude, data.clean_trajectory)

    println("\n  Results (evaluated on clean trajectory):")
    println("    Full NN - D_corr: " * string(round(metrics_full.d_corr, digits=4)) *
            ", MSE: " * string(round(metrics_full.mse, digits=6)))
    println("    UDE     - D_corr: " * string(round(metrics_ude.d_corr, digits=4)) *
            ", MSE: " * string(round(metrics_ude.mse, digits=6)))

    improvement = 100 * (metrics_ude.d_corr / metrics_full.d_corr - 1)
    println("    UDE improvement: " * string(round(improvement, digits=1)) * "%")

    # Store results
    results[noise_name] = Dict(
        "noise_level" => noise_val,
        "noisy_trajectory" => data.noisy_trajectory,
        "loss_full" => loss_full,
        "loss_ude" => loss_ude,
        "pred_full" => pred_full,
        "pred_ude" => pred_ude,
        "metrics_full" => metrics_full,
        "metrics_ude" => metrics_ude,
        "ps_full" => ps_full_trained,
        "ps_ude" => ps_ude_trained
    )
end

# Summary table
println("\n" * "=" ^ 60)
println("SUMMARY: UDE vs Full NN across noise levels")
println("=" ^ 60)
println("\nNoise Level | Full NN D_corr | UDE D_corr | UDE Improvement")
println("-" ^ 60)

for noise_name in ["low", "medium", "high"]
    r = results[noise_name]
    full_d = r["metrics_full"].d_corr
    ude_d = r["metrics_ude"].d_corr
    improvement = 100 * (ude_d / full_d - 1)

    println("  " * rpad(noise_name, 10) * " | " *
            rpad(string(round(full_d, digits=4)), 14) * " | " *
            rpad(string(round(ude_d, digits=4)), 10) * " | " *
            string(round(improvement, digits=1)) * "%")
end

# Save results
serialize(SAVE_FILE, results)
println("\nSaved results to " * SAVE_FILE)

end  # if VIZ_ONLY

# =============================================================================
# Visualization
# =============================================================================

println("\n5. Generating visualizations...")

clean_traj = results["clean_trajectory"]
tsteps = results["tsteps"]

# Plot 1: D_corr comparison bar chart
fig1 = CM.Figure(size=(800, 400))

ax1 = CM.Axis(fig1[1, 1],
              xlabel="Noise Level",
              ylabel="Distance Correlation (vs clean)",
              title="Example 3b: UDE Regularization Effect Under Noise",
              xticks=(1:3, ["Low\n(σ=0.01)", "Medium\n(σ=0.03)", "High\n(σ=0.05)"]))

x_full = [0.8, 1.8, 2.8]
x_ude = [1.2, 2.2, 3.2]

d_full = [results[n]["metrics_full"].d_corr for n in ["low", "medium", "high"]]
d_ude = [results[n]["metrics_ude"].d_corr for n in ["low", "medium", "high"]]

CM.barplot!(ax1, x_full, d_full, color=:gray, label="Full NN", width=0.35)
CM.barplot!(ax1, x_ude, d_ude, color=:green, label="UDE (correct)", width=0.35)

CM.axislegend(ax1, position=:rt)

CM.save(joinpath(RESULTS_DIR, "noise_comparison.pdf"), fig1)
println("  Saved: noise_comparison.pdf")

# Plot 2: Trajectory comparison at high noise
fig2 = CM.Figure(size=(1000, 600))

high_results = results["high"]
pred_full = high_results["pred_full"]
pred_ude = high_results["pred_ude"]
noisy_traj = high_results["noisy_trajectory"]

plot_nodes = [1, 5, 10]

for (col, node_i) in enumerate(plot_nodes)
    ax = CM.Axis(fig2[1, col],
                 xlabel="x₁", ylabel="x₂",
                 title="Node " * string(node_i) * " (High Noise)")

    # Clean trajectory
    clean_x1 = [clean_traj[(node_i-1)*D + 1, t] for t in 1:length(tsteps)]
    clean_x2 = [clean_traj[(node_i-1)*D + 2, t] for t in 1:length(tsteps)]
    CM.lines!(ax, clean_x1, clean_x2, color=:black, linewidth=2, label="True (clean)")

    # Noisy observations
    noisy_x1 = [noisy_traj[(node_i-1)*D + 1, t] for t in 1:length(tsteps)]
    noisy_x2 = [noisy_traj[(node_i-1)*D + 2, t] for t in 1:length(tsteps)]
    CM.scatter!(ax, noisy_x1, noisy_x2, color=:lightgray, markersize=4, label="Noisy obs")

    # Full NN prediction
    full_x1 = [pred_full[(node_i-1)*D + 1, t] for t in 1:length(tsteps)]
    full_x2 = [pred_full[(node_i-1)*D + 2, t] for t in 1:length(tsteps)]
    CM.lines!(ax, full_x1, full_x2, color=:blue, linewidth=1.5, linestyle=:dash, label="Full NN")

    # UDE prediction
    ude_x1 = [pred_ude[(node_i-1)*D + 1, t] for t in 1:length(tsteps)]
    ude_x2 = [pred_ude[(node_i-1)*D + 2, t] for t in 1:length(tsteps)]
    CM.lines!(ax, ude_x1, ude_x2, color=:green, linewidth=1.5, linestyle=:dash, label="UDE")

    if col == 1
        CM.axislegend(ax, position=:lt)
    end
end

CM.save(joinpath(RESULTS_DIR, "trajectory_high_noise.pdf"), fig2)
println("  Saved: trajectory_high_noise.pdf")

# Plot 3: MSE comparison
fig3 = CM.Figure(size=(800, 400))

ax3 = CM.Axis(fig3[1, 1],
              xlabel="Noise Level",
              ylabel="MSE (vs clean trajectory, log scale)",
              title="Prediction Error by Model",
              xticks=(1:3, ["Low", "Medium", "High"]),
              yscale=log10)

mse_full = [results[n]["metrics_full"].mse for n in ["low", "medium", "high"]]
mse_ude = [results[n]["metrics_ude"].mse for n in ["low", "medium", "high"]]

CM.barplot!(ax3, x_full, mse_full, color=:gray, label="Full NN", width=0.35)
CM.barplot!(ax3, x_ude, mse_ude, color=:green, label="UDE", width=0.35)

CM.axislegend(ax3, position=:lt)

CM.save(joinpath(RESULTS_DIR, "mse_comparison.pdf"), fig3)
println("  Saved: mse_comparison.pdf")

println("\n" * "=" ^ 60)
println("Example 3b complete!")
println("=" ^ 60)
