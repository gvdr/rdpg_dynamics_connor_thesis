#!/usr/bin/env julia
"""
Example 3b: Correct Known Part (Noisy Data)

Hypothesis: UDE might be BETTER than Full NN when the known structure is correct
but data is noisy. The correct structure acts as regularization, preventing
overfitting to noise. Full NN might chase the noise.

Setup:
- True dynamics: Same circulation as Example 2
- UDE uses: Correct circulation with ω_true
- Noise: Gaussian noise added to observed trajectories

This tests multiple noise levels:
- Low:    σ = 0.01 (1% of typical magnitude)
- Medium: σ = 0.05 (5%)
- High:   σ = 0.10 (10%)

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
const SAVE_FILE = joinpath(RESULTS_DIR, "example3b_noisy_data_results.jls")

println("=" ^ 60)
println("Example 3b: Correct Known Part (Noisy Data)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# True dynamics parameters (same as Example 2)
# =============================================================================

const TRUE_OMEGA = 0.15f0
const TRUE_ALPHA = 0.5f0
const TRUE_EPSILON = 0.08f0
const TRUE_K_RADIAL = 0.1f0
const TRUE_R_TARGET = 0.6f0

# Noise levels
const NOISE_LOW = 0.01f0
const NOISE_MEDIUM = 0.05f0
const NOISE_HIGH = 0.10f0

"""
True dynamics function.
"""
function true_dynamics(u::Vector{T}) where T
    x1, x2 = u[1], u[2]
    r = sqrt(x1^2 + x2^2)

    dx_circ = TRUE_OMEGA * T[-x2, x1]

    repel_x1 = TRUE_ALPHA * exp(-x1 / TRUE_EPSILON)
    repel_x2 = TRUE_ALPHA * exp(-x2 / TRUE_EPSILON)
    repel_arc = TRUE_ALPHA * exp((r - 1) / TRUE_EPSILON)

    radial = r > 1f-6 ? -TRUE_K_RADIAL * (r - TRUE_R_TARGET) / r : zero(T)

    dx = dx_circ .+ T[repel_x1, repel_x2] .- (repel_arc / r) .* T[x1, x2]
    dx .+= radial .* T[x1, x2]

    return dx
end

function true_dynamics!(du, u, p, t)
    result = true_dynamics(Vector{eltype(u)}(u))
    du[1] = result[1]
    du[2] = result[2]
end

"""
Generate clean data, then add noise at specified level.
"""
function generate_data(; n::Int=20, T_end::Float32=40f0, dt::Float32=1f0,
                        noise_level::Float32=0f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    u0_all = Vector{Vector{Float32}}(undef, n)
    for i in 1:n
        theta = Float32(pi/8 + rand(rng) * pi/4)
        r = Float32(0.3 + 0.6 * rand(rng))
        u0_all[i] = Float32[r * cos(theta), r * sin(theta)]
    end

    tspan = (0f0, T_end)
    tsteps = 0f0:dt:T_end
    n_steps = length(tsteps)

    # Generate clean trajectories
    clean_trajectories = Dict{Int, Matrix{Float32}}()
    for i in 1:n
        prob = ODEProblem(true_dynamics!, u0_all[i], tspan)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-7, reltol=1f-7)

        traj = zeros(Float32, 2, n_steps)
        for (t_idx, t) in enumerate(tsteps)
            traj[:, t_idx] = sol(t)
        end
        clean_trajectories[i] = traj
    end

    # Add noise (using different seed for noise)
    noise_rng = Random.MersenneTwister(seed + 1000)
    noisy_trajectories = Dict{Int, Matrix{Float32}}()
    for i in 1:n
        clean = clean_trajectories[i]
        noise = Float32.(noise_level .* randn(noise_rng, size(clean)))
        noisy = clean .+ noise

        # Clip to stay in B^d_+ (simple projection)
        noisy[1, :] = max.(noisy[1, :], 0.01f0)
        noisy[2, :] = max.(noisy[2, :], 0.01f0)
        for t in 1:n_steps
            r = norm(noisy[:, t])
            if r > 0.99f0
                noisy[:, t] .*= 0.99f0 / r
            end
        end

        noisy_trajectories[i] = noisy
    end

    return (
        clean_trajectories = clean_trajectories,
        noisy_trajectories = noisy_trajectories,
        n = n,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        u0_all = u0_all,
        noise_level = noise_level
    )
end

println("\n1. Generating data at multiple noise levels...")

# Generate data at each noise level
data_low = generate_data(n=20, noise_level=NOISE_LOW)
data_medium = generate_data(n=20, noise_level=NOISE_MEDIUM)
data_high = generate_data(n=20, noise_level=NOISE_HIGH)

n = data_low.n
tsteps = data_low.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " nodes")
println("  Noise levels: " * string(NOISE_LOW) * ", " * string(NOISE_MEDIUM) * ", " * string(NOISE_HIGH))

# =============================================================================
# Training function (reusable)
# =============================================================================

function setup_and_train(noisy_data, clean_data; noise_name="")
    println("\n" * "="^40)
    println("Training with " * noise_name * " noise (σ=" * string(noisy_data.noise_level) * ")")
    println("="^40)

    rng = Random.Xoshiro(42)

    # Create fresh NNs for this noise level
    nn_full = Lux.Chain(
        Lux.Dense(2, 32, tanh),
        Lux.Dense(32, 32, tanh),
        Lux.Dense(32, 2)
    )

    nn_ude = Lux.Chain(
        Lux.Dense(2, 32, tanh),
        Lux.Dense(32, 32, tanh),
        Lux.Dense(32, 2)
    )

    ps_full, st_full = Lux.setup(rng, nn_full)
    ps_ude, st_ude = Lux.setup(Random.Xoshiro(42), nn_ude)

    ps_full_ca = ComponentArray(ps_full)
    ps_ude_ca = ComponentArray(ps_ude)

    # Define dynamics with these specific NNs
    function full_dynamics_local(u, p, t)
        u_input = reshape(u, 2, 1)
        nn_out, _ = nn_full(u_input, p, st_full)
        return vec(nn_out)
    end

    function ude_dynamics_local(u, p, t)
        T = eltype(u)
        dx_known = T(TRUE_OMEGA) .* T[-u[2], u[1]]
        u_input = reshape(u, 2, 1)
        nn_out, _ = nn_ude(u_input, p, st_ude)
        return dx_known .+ vec(nn_out)
    end

    # Training setup
    train_nodes = 1:15
    val_nodes = 16:20
    tspan = (0f0, noisy_data.T_end)

    train_data = Dict{Int, Matrix{Float32}}()
    for i in train_nodes
        train_data[i] = noisy_data.noisy_trajectories[i]
    end

    function predict_node_local(u0, p, dynamics_fn)
        prob = ODEProblem(dynamics_fn, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=tsteps,
                    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                    abstol=1f-6, reltol=1f-6)
        return Array(sol)
    end

    function loss_full_local(p)
        total_loss = 0f0
        for i in train_nodes
            u0 = train_data[i][:, 1]
            true_traj = train_data[i]
            pred = predict_node_local(u0, p, full_dynamics_local)
            total_loss += sum(abs2, true_traj .- pred)
        end
        return total_loss / (length(train_nodes) * T_steps)
    end

    function loss_ude_local(p)
        total_loss = 0f0
        for i in train_nodes
            u0 = train_data[i][:, 1]
            true_traj = train_data[i]
            pred = predict_node_local(u0, p, ude_dynamics_local)
            total_loss += sum(abs2, true_traj .- pred)
        end
        return total_loss / (length(train_nodes) * T_steps)
    end

    # Train Full NN
    println("  Training Full NN...")
    iter_full = Ref(0)
    callback_full(state, l) = (iter_full[] += 1; iter_full[] % 100 == 0 && println("    iter " * string(iter_full[]) * ": " * string(round(l, digits=6))); false)

    optf_full = Optimization.OptimizationFunction((p, _) -> loss_full_local(p), Optimization.AutoZygote())
    optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

    result_full_adam = Optimization.solve(optprob_full, OptimizationOptimisers.Adam(0.01), maxiters=300, callback=callback_full)

    iter_full[] = 300
    optprob_full2 = Optimization.OptimizationProblem(optf_full, result_full_adam.u)
    result_full = try
        Optimization.solve(optprob_full2, OptimizationOptimJL.BFGS(initial_stepnorm=0.01), maxiters=100, callback=callback_full, allow_f_increases=false)
    catch
        result_full_adam
    end

    ps_full_trained = result_full.u
    println("    Final loss: " * string(round(loss_full_local(ps_full_trained), digits=6)))

    # Train UDE
    println("  Training UDE...")
    iter_ude = Ref(0)
    callback_ude(state, l) = (iter_ude[] += 1; iter_ude[] % 100 == 0 && println("    iter " * string(iter_ude[]) * ": " * string(round(l, digits=6))); false)

    optf_ude = Optimization.OptimizationFunction((p, _) -> loss_ude_local(p), Optimization.AutoZygote())
    optprob_ude = Optimization.OptimizationProblem(optf_ude, ps_ude_ca)

    result_ude_adam = Optimization.solve(optprob_ude, OptimizationOptimisers.Adam(0.01), maxiters=300, callback=callback_ude)

    iter_ude[] = 300
    optprob_ude2 = Optimization.OptimizationProblem(optf_ude, result_ude_adam.u)
    result_ude = try
        Optimization.solve(optprob_ude2, OptimizationOptimJL.BFGS(initial_stepnorm=0.01), maxiters=100, callback=callback_ude, allow_f_increases=false)
    catch
        result_ude_adam
    end

    ps_ude_trained = result_ude.u
    println("    Final loss: " * string(round(loss_ude_local(ps_ude_trained), digits=6)))

    # Evaluate on CLEAN validation data
    println("  Evaluating on clean validation data...")

    val_errs_full = Float64[]
    val_errs_ude = Float64[]

    for i in val_nodes
        u0 = clean_data.clean_trajectories[i][:, 1]
        true_traj = clean_data.clean_trajectories[i]

        pred_full = predict_node_local(u0, ps_full_trained, full_dynamics_local)
        pred_ude = predict_node_local(u0, ps_ude_trained, ude_dynamics_local)

        push!(val_errs_full, mean([norm(pred_full[:, t] - true_traj[:, t]) for t in 1:T_steps]))
        push!(val_errs_ude, mean([norm(pred_ude[:, t] - true_traj[:, t]) for t in 1:T_steps]))
    end

    val_full = mean(val_errs_full)
    val_ude = mean(val_errs_ude)

    println("    Full NN val error (on clean): " * string(round(val_full, digits=4)))
    println("    UDE val error (on clean):     " * string(round(val_ude, digits=4)))

    improvement = 100 * (1 - val_ude / val_full)
    println("    UDE improvement: " * string(round(improvement, digits=1)) * "%")

    # Compute trajectories for visualization
    viz_node = first(val_nodes)
    u0_viz = clean_data.clean_trajectories[viz_node][:, 1]
    viz_clean = clean_data.clean_trajectories[viz_node]
    viz_noisy = noisy_data.noisy_trajectories[viz_node]
    viz_full = predict_node_local(u0_viz, ps_full_trained, full_dynamics_local)
    viz_ude = predict_node_local(u0_viz, ps_ude_trained, ude_dynamics_local)

    return Dict(
        "val_full" => val_full,
        "val_ude" => val_ude,
        "improvement" => improvement,
        "viz_clean" => viz_clean,
        "viz_noisy" => viz_noisy,
        "viz_full" => viz_full,
        "viz_ude" => viz_ude,
        "viz_node" => viz_node
    )
end

# =============================================================================
# Run training or load results
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n2-4. Loading saved results...")
    saved = deserialize(SAVE_FILE)

    results_low = saved["results_low"]
    results_medium = saved["results_medium"]
    results_high = saved["results_high"]

    println("  Loaded!")

else  # Run training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

# Generate clean reference data (no noise)
data_clean = generate_data(n=20, noise_level=0f0)

println("\n2-4. Training at each noise level...")

results_low = setup_and_train(data_low, data_clean, noise_name="low")
results_medium = setup_and_train(data_medium, data_clean, noise_name="medium")
results_high = setup_and_train(data_high, data_clean, noise_name="high")

# Save results
println("\n  Saving results...")
mkpath(RESULTS_DIR)
serialize(SAVE_FILE, Dict(
    "results_low" => results_low,
    "results_medium" => results_medium,
    "results_high" => results_high
))
println("  Saved to " * SAVE_FILE)

end  # end training block

# =============================================================================
# Visualization
# =============================================================================
println("\n5. Creating visualization...")

fig = Figure(size=(1600, 800))

theta_arc = range(0, pi/2, length=100)

# Row 1: Trajectories at each noise level
for (col, (results, noise_val, noise_name)) in enumerate([
    (results_low, NOISE_LOW, "Low (σ=0.01)"),
    (results_medium, NOISE_MEDIUM, "Medium (σ=0.05)"),
    (results_high, NOISE_HIGH, "High (σ=0.10)")
])
    ax = CairoMakie.Axis(fig[1, col], xlabel="x₁", ylabel="x₂",
                          title="Noise: " * noise_name, aspect=1)

    lines!(ax, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
    lines!(ax, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
    lines!(ax, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)

    # Clean (ground truth)
    lines!(ax, results["viz_clean"][1, :], results["viz_clean"][2, :],
           color=:black, linewidth=2.5, label="True (clean)")

    # Noisy training data
    scatter!(ax, results["viz_noisy"][1, :], results["viz_noisy"][2, :],
             color=(:gray, 0.3), markersize=4, label="Noisy observations")

    # Predictions
    lines!(ax, results["viz_full"][1, :], results["viz_full"][2, :],
           color=:blue, linewidth=2, label="Full NN")
    lines!(ax, results["viz_ude"][1, :], results["viz_ude"][2, :],
           color=:red, linewidth=2, linestyle=:dash, label="UDE")

    axislegend(ax, position=:lt)
    xlims!(ax, -0.05, 1.05)
    ylims!(ax, -0.05, 1.05)
end

# Row 2: Summary comparison
ax_summary = CairoMakie.Axis(fig[2, 1:2],
                              xlabel="Noise Level",
                              ylabel="Validation Error (on clean data)",
                              title="Generalization Error Comparison",
                              xticks=(1:3, ["Low (σ=0.01)", "Medium (σ=0.05)", "High (σ=0.10)"]))

x_pos = [1, 2, 3]
width = 0.35

# Full NN bars
full_vals = [results_low["val_full"], results_medium["val_full"], results_high["val_full"]]
barplot!(ax_summary, x_pos .- width/2, full_vals, width=width, color=:blue, label="Full NN")

# UDE bars
ude_vals = [results_low["val_ude"], results_medium["val_ude"], results_high["val_ude"]]
barplot!(ax_summary, x_pos .+ width/2, ude_vals, width=width, color=:red, label="UDE")

axislegend(ax_summary, position=:lt)

# Improvement chart
ax_imp = CairoMakie.Axis(fig[2, 3],
                          xlabel="Noise Level",
                          ylabel="UDE Improvement vs Full NN (%)",
                          title="UDE Advantage\n(positive = UDE better)",
                          xticks=(1:3, ["Low", "Medium", "High"]))

improvements = [results_low["improvement"], results_medium["improvement"], results_high["improvement"]]
colors_imp = [imp > 0 ? :green : :red for imp in improvements]

barplot!(ax_imp, 1:3, improvements, color=colors_imp)
hlines!(ax_imp, [0], color=:black, linestyle=:solid, alpha=0.5)

save("results/example3b_noisy_data.pdf", fig)
println("\nSaved: results/example3b_noisy_data.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 3b - Noisy Data with Correct Structure")
println("=" ^ 60)

println("\nHypothesis: UDE with correct structure should outperform Full NN")
println("because structure provides regularization against overfitting to noise.")

println("\nValidation Errors (evaluated on CLEAN data):")
println("  Noise Level | Full NN  | UDE      | UDE Improvement")
println("  " * "-"^50)
for (noise_name, results) in [("Low   ", results_low), ("Medium", results_medium), ("High  ", results_high)]
    println("  " * noise_name * "     | " *
            string(round(results["val_full"], digits=4)) * " | " *
            string(round(results["val_ude"], digits=4)) * " | " *
            string(round(results["improvement"], digits=1)) * "%")
end

println("\n" * "-"^40)
# Interpret results
all_positive = all([results_low["improvement"] > 0, results_medium["improvement"] > 0, results_high["improvement"] > 0])
if all_positive
    println("✓ UDE outperforms Full NN at all noise levels!")
    println("  → Correct structure provides regularization benefit")
else
    println("? Mixed results - structure doesn't always help")
end

# Check if improvement increases with noise
if results_high["improvement"] > results_low["improvement"]
    println("✓ UDE advantage increases with noise level")
    println("  → More noise makes regularization more valuable")
end
