#!/usr/bin/env julia
"""
Example 3a: Wrong Known Part (Clean Data)

Hypothesis: UDE might be WORSE than Full NN when the known structure is wrong.
The rigid incorrect structure constrains the NN to learn corrections from a
wrong baseline. Full NN has freedom to find the right solution.

Setup:
- True dynamics: Circulation with ω_true
- UDE assumes: Circulation with ω_wrong ≠ ω_true
- Full NN: No assumptions

This tests multiple misspecification levels:
- Mild: ω_wrong = 1.2 * ω_true (20% off)
- Severe: ω_wrong = 2.0 * ω_true (100% off)

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

const RESULTS_DIR = "results"
const SAVE_FILE = joinpath(RESULTS_DIR, "example3a_wrong_known_results.jls")

println("=" ^ 60)
println("Example 3a: Wrong Known Part (Clean Data)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# True dynamics parameters (same as Example 2)
# =============================================================================

const TRUE_OMEGA = 0.15f0        # TRUE circulation speed
const TRUE_ALPHA = 0.5f0         # Boundary repulsion strength
const TRUE_EPSILON = 0.08f0      # Boundary repulsion sharpness
const TRUE_K_RADIAL = 0.1f0      # Radial attraction strength
const TRUE_R_TARGET = 0.6f0      # Target radius

# Misspecification levels for UDE
const OMEGA_MILD = 1.2f0 * TRUE_OMEGA    # 20% off
const OMEGA_SEVERE = 2.0f0 * TRUE_OMEGA  # 100% off

"""
True dynamics function (for data generation).
"""
function true_dynamics(u::Vector{T}) where T
    x1, x2 = u[1], u[2]
    r = sqrt(x1^2 + x2^2)

    # Circulation (with TRUE omega)
    dx_circ = TRUE_OMEGA * T[-x2, x1]

    # Boundary repulsion
    repel_x1 = TRUE_ALPHA * exp(-x1 / TRUE_EPSILON)
    repel_x2 = TRUE_ALPHA * exp(-x2 / TRUE_EPSILON)
    repel_arc = TRUE_ALPHA * exp((r - 1) / TRUE_EPSILON)

    # Radial attraction
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
Generate synthetic data.
"""
function generate_data(; n::Int=20, T_end::Float32=40f0, dt::Float32=1f0, seed::Int=42)
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

println("\n1. Generating data (same dynamics as Example 2)...")
data = generate_data(n=20, T_end=40f0, dt=1f0)
n = data.n
tsteps = data.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " nodes, T=" * string(data.T_end))
println("  TRUE omega: " * string(TRUE_OMEGA))
println("  MILD wrong omega: " * string(OMEGA_MILD) * " (" * string(round(100*(OMEGA_MILD/TRUE_OMEGA - 1), digits=0)) * "% off)")
println("  SEVERE wrong omega: " * string(OMEGA_SEVERE) * " (" * string(round(100*(OMEGA_SEVERE/TRUE_OMEGA - 1), digits=0)) * "% off)")

# =============================================================================
# Setup neural networks (one for each model)
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# Full NN (baseline)
nn_full = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

# UDE with correct omega
nn_ude_correct = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

# UDE with mildly wrong omega
nn_ude_mild = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

# UDE with severely wrong omega
nn_ude_severe = Lux.Chain(
    Lux.Dense(2, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 2)
)

ps_full, st_full = Lux.setup(rng, nn_full)
ps_ude_correct, st_ude_correct = Lux.setup(Random.Xoshiro(42), nn_ude_correct)
ps_ude_mild, st_ude_mild = Lux.setup(Random.Xoshiro(42), nn_ude_mild)
ps_ude_severe, st_ude_severe = Lux.setup(Random.Xoshiro(42), nn_ude_severe)

ps_full_ca = ComponentArray(ps_full)
ps_ude_correct_ca = ComponentArray(ps_ude_correct)
ps_ude_mild_ca = ComponentArray(ps_ude_mild)
ps_ude_severe_ca = ComponentArray(ps_ude_severe)

println("  NN parameters per model: " * string(length(ps_full_ca)))

# =============================================================================
# Define dynamics functions
# =============================================================================

# Full NN dynamics
function full_dynamics(u, p, t)
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# UDE with CORRECT omega
function ude_correct_dynamics(u, p, t)
    T = eltype(u)
    dx_known = T(TRUE_OMEGA) .* T[-u[2], u[1]]
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_ude_correct(u_input, p, st_ude_correct)
    return dx_known .+ vec(nn_out)
end

# UDE with MILD wrong omega
function ude_mild_dynamics(u, p, t)
    T = eltype(u)
    dx_known = T(OMEGA_MILD) .* T[-u[2], u[1]]
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_ude_mild(u_input, p, st_ude_mild)
    return dx_known .+ vec(nn_out)
end

# UDE with SEVERE wrong omega
function ude_severe_dynamics(u, p, t)
    T = eltype(u)
    dx_known = T(OMEGA_SEVERE) .* T[-u[2], u[1]]
    u_input = reshape(u, 2, 1)
    nn_out, _ = nn_ude_severe(u_input, p, st_ude_severe)
    return dx_known .+ vec(nn_out)
end

# =============================================================================
# Training setup
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

tspan = (0f0, data.T_end)

function predict_node(u0, p, dynamics_fn)
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-6, reltol=1f-6)
    return Array(sol)
end

function make_loss(dynamics_fn)
    function loss(p)
        total_loss = 0f0
        for i in train_nodes
            u0 = train_data[i][:, 1]
            true_traj = train_data[i]
            pred = predict_node(u0, p, dynamics_fn)
            total_loss += sum(abs2, true_traj .- pred)
        end
        return total_loss / (length(train_nodes) * T_steps)
    end
    return loss
end

loss_full = make_loss(full_dynamics)
loss_ude_correct = make_loss(ude_correct_dynamics)
loss_ude_mild = make_loss(ude_mild_dynamics)
loss_ude_severe = make_loss(ude_severe_dynamics)

# =============================================================================
# Training function
# =============================================================================

function train_model(loss_fn, ps_init; name="Model", max_adam=300, max_bfgs=100)
    println("\n  Training " * name * "...")

    iter = Ref(0)
    function callback(state, l)
        iter[] += 1
        if iter[] % 100 == 0
            println("    iter " * string(iter[]) * ": loss=" * string(round(l, digits=6)))
        end
        return false
    end

    # Stage 1: ADAM
    optf = Optimization.OptimizationFunction((p, _) -> loss_fn(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps_init)

    result_adam = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(0.01),
        maxiters=max_adam,
        callback=callback
    )

    # Stage 2: BFGS
    iter[] = max_adam
    optprob2 = Optimization.OptimizationProblem(optf, result_adam.u)

    result = try
        Optimization.solve(
            optprob2,
            OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
            maxiters=max_bfgs,
            callback=callback,
            allow_f_increases=false
        )
    catch e
        println("    BFGS failed: " * string(e))
        result_adam
    end

    final_loss = loss_fn(result.u)
    println("    Final loss: " * string(round(final_loss, digits=6)))

    return result.u, final_loss
end

# =============================================================================
# Run training or load results
# =============================================================================

if VIZ_ONLY && isfile(SAVE_FILE)
    println("\n4-5. Loading saved results...")
    saved = deserialize(SAVE_FILE)

    results = saved["results"]
    val_errors = saved["val_errors"]
    train_errors = saved["train_errors"]

    # Load trained parameters for viz
    ps_full_trained = saved["ps_full_trained"]
    ps_correct_trained = saved["ps_correct_trained"]
    ps_mild_trained = saved["ps_mild_trained"]
    ps_severe_trained = saved["ps_severe_trained"]

    # Load precomputed trajectories
    viz_node = saved["viz_node"]
    viz_true = saved["viz_true"]
    viz_full = saved["viz_full"]
    viz_correct = saved["viz_correct"]
    viz_mild = saved["viz_mild"]
    viz_severe = saved["viz_severe"]

    println("  Loaded!")

else  # Run training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz specified but no saved file. Running training...")
end

println("\n4. Training all models...")

ps_full_trained, loss_full_final = train_model(loss_full, ps_full_ca, name="Full NN")
ps_correct_trained, loss_correct_final = train_model(loss_ude_correct, ps_ude_correct_ca, name="UDE (correct ω)")
ps_mild_trained, loss_mild_final = train_model(loss_ude_mild, ps_ude_mild_ca, name="UDE (20% wrong ω)")
ps_severe_trained, loss_severe_final = train_model(loss_ude_severe, ps_ude_severe_ca, name="UDE (100% wrong ω)")

# =============================================================================
# Evaluate all models
# =============================================================================
println("\n5. Evaluating on validation set...")

function evaluate_model(ps_trained, dynamics_fn)
    train_errs = Float64[]
    val_errs = Float64[]

    for i in 1:n
        u0 = data.trajectories[i][:, 1]
        true_traj = data.trajectories[i]
        pred = predict_node(u0, ps_trained, dynamics_fn)

        err = mean([norm(pred[:, t] - true_traj[:, t]) for t in 1:T_steps])

        if i in train_nodes
            push!(train_errs, err)
        else
            push!(val_errs, err)
        end
    end

    return mean(train_errs), mean(val_errs)
end

train_full, val_full = evaluate_model(ps_full_trained, full_dynamics)
train_correct, val_correct = evaluate_model(ps_correct_trained, ude_correct_dynamics)
train_mild, val_mild = evaluate_model(ps_mild_trained, ude_mild_dynamics)
train_severe, val_severe = evaluate_model(ps_severe_trained, ude_severe_dynamics)

results = Dict(
    "full" => (loss=loss_full_final,),
    "correct" => (loss=loss_correct_final,),
    "mild" => (loss=loss_mild_final,),
    "severe" => (loss=loss_severe_final,)
)

val_errors = Dict(
    "full" => val_full,
    "correct" => val_correct,
    "mild" => val_mild,
    "severe" => val_severe
)

train_errors = Dict(
    "full" => train_full,
    "correct" => train_correct,
    "mild" => train_mild,
    "severe" => train_severe
)

println("\n  Validation errors:")
println("    Full NN:              " * string(round(val_full, digits=4)))
println("    UDE (correct ω):      " * string(round(val_correct, digits=4)))
println("    UDE (20% wrong ω):    " * string(round(val_mild, digits=4)))
println("    UDE (100% wrong ω):   " * string(round(val_severe, digits=4)))

# Compute relative performance
println("\n  Relative to Full NN (negative = worse):")
println("    UDE (correct ω):      " * string(round(100*(1 - val_correct/val_full), digits=1)) * "%")
println("    UDE (20% wrong ω):    " * string(round(100*(1 - val_mild/val_full), digits=1)) * "%")
println("    UDE (100% wrong ω):   " * string(round(100*(1 - val_severe/val_full), digits=1)) * "%")

# Precompute trajectories for visualization
println("\n  Precomputing trajectories for visualization...")

viz_node = first(val_nodes)
u0_viz = data.trajectories[viz_node][:, 1]

viz_true = data.trajectories[viz_node]
viz_full = predict_node(u0_viz, ps_full_trained, full_dynamics)
viz_correct = predict_node(u0_viz, ps_correct_trained, ude_correct_dynamics)
viz_mild = predict_node(u0_viz, ps_mild_trained, ude_mild_dynamics)
viz_severe = predict_node(u0_viz, ps_severe_trained, ude_severe_dynamics)

# Save results
println("  Saving results...")
mkpath(RESULTS_DIR)
serialize(SAVE_FILE, Dict(
    "results" => results,
    "val_errors" => val_errors,
    "train_errors" => train_errors,
    "ps_full_trained" => ps_full_trained,
    "ps_correct_trained" => ps_correct_trained,
    "ps_mild_trained" => ps_mild_trained,
    "ps_severe_trained" => ps_severe_trained,
    "viz_node" => viz_node,
    "viz_true" => viz_true,
    "viz_full" => viz_full,
    "viz_correct" => viz_correct,
    "viz_mild" => viz_mild,
    "viz_severe" => viz_severe
))
println("  Saved to " * SAVE_FILE)

end  # end training block

# =============================================================================
# Visualization
# =============================================================================
println("\n6. Creating visualization...")

fig = Figure(size=(1400, 500))

theta_arc = range(0, pi/2, length=100)

# Panel 1: Trajectory comparison on validation node
ax1 = CairoMakie.Axis(fig[1, 1], xlabel="x₁", ylabel="x₂",
                       title="Validation Node: Trajectory Comparison", aspect=1)

# B^d_+ boundary
lines!(ax1, cos.(theta_arc), sin.(theta_arc), color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 1], [0, 0], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, [0, 0], [0, 1], color=:gray, linestyle=:dash, alpha=0.5)
lines!(ax1, TRUE_R_TARGET .* cos.(theta_arc), TRUE_R_TARGET .* sin.(theta_arc),
       color=:green, linestyle=:dot, linewidth=1, alpha=0.5)

lines!(ax1, viz_true[1, :], viz_true[2, :], color=:black, linewidth=3, label="True")
lines!(ax1, viz_full[1, :], viz_full[2, :], color=:blue, linewidth=2, label="Full NN")
lines!(ax1, viz_correct[1, :], viz_correct[2, :], color=:green, linewidth=2, linestyle=:dash, label="UDE (correct)")
lines!(ax1, viz_mild[1, :], viz_mild[2, :], color=:orange, linewidth=2, linestyle=:dot, label="UDE (20% wrong)")
lines!(ax1, viz_severe[1, :], viz_severe[2, :], color=:red, linewidth=2, linestyle=:dashdot, label="UDE (100% wrong)")

scatter!(ax1, [viz_true[1, 1]], [viz_true[2, 1]], color=:black, markersize=15, marker=:circle)
scatter!(ax1, [viz_true[1, end]], [viz_true[2, end]], color=:black, markersize=15, marker=:star5)

axislegend(ax1, position=:lt)
xlims!(ax1, -0.05, 1.05)
ylims!(ax1, -0.05, 1.05)

# Panel 2: Bar chart of validation errors
ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Model", ylabel="Validation Error",
                       title="Validation Error Comparison",
                       xticks=(1:4, ["Full NN", "UDE\n(correct)", "UDE\n(20% wrong)", "UDE\n(100% wrong)"]))

errors_bar = [val_errors["full"], val_errors["correct"], val_errors["mild"], val_errors["severe"]]
colors_bar = [:blue, :green, :orange, :red]

barplot!(ax2, 1:4, errors_bar, color=colors_bar)

# Add horizontal line at Full NN level
hlines!(ax2, [val_errors["full"]], color=:blue, linestyle=:dash, alpha=0.5)

# Panel 3: Improvement relative to Full NN
ax3 = CairoMakie.Axis(fig[1, 3], xlabel="Model", ylabel="% Improvement vs Full NN",
                       title="Relative Performance\n(positive = better than Full NN)",
                       xticks=(1:3, ["UDE\n(correct)", "UDE\n(20% wrong)", "UDE\n(100% wrong)"]))

improvements = [
    100 * (1 - val_errors["correct"] / val_errors["full"]),
    100 * (1 - val_errors["mild"] / val_errors["full"]),
    100 * (1 - val_errors["severe"] / val_errors["full"])
]
colors_imp = [imp > 0 ? :green : :red for imp in improvements]

barplot!(ax3, 1:3, improvements, color=colors_imp)
hlines!(ax3, [0], color=:black, linestyle=:solid, alpha=0.5)

save("results/example3a_wrong_known.pdf", fig)
println("\nSaved: results/example3a_wrong_known.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 3a - Wrong Known Part")
println("=" ^ 60)

println("\nHypothesis: UDE with wrong known structure might be WORSE than Full NN")

println("\nValidation Errors:")
println("  Full NN:              " * string(round(val_errors["full"], digits=4)))
println("  UDE (correct ω):      " * string(round(val_errors["correct"], digits=4)))
println("  UDE (20% wrong ω):    " * string(round(val_errors["mild"], digits=4)))
println("  UDE (100% wrong ω):   " * string(round(val_errors["severe"], digits=4)))

println("\nRelative to Full NN:")
for (name, key) in [("correct", "correct"), ("20% wrong", "mild"), ("100% wrong", "severe")]
    imp = 100 * (1 - val_errors[key] / val_errors["full"])
    status = imp > 0 ? "better" : "worse"
    println("  UDE (" * name * " ω): " * string(round(imp, digits=1)) * "% " * status)
end

# Interpret results
println("\n" * "-"^40)
if val_errors["correct"] < val_errors["full"]
    println("✓ UDE with correct structure outperforms Full NN (as expected)")
else
    println("✗ Unexpected: Full NN outperforms UDE even with correct structure")
end

if val_errors["severe"] > val_errors["full"]
    println("✓ UDE with severely wrong structure underperforms Full NN")
    println("  → Confirms hypothesis: wrong structure hurts more than no structure")
else
    println("? UDE with wrong structure still outperforms Full NN")
    println("  → Hypothesis not confirmed for this misspecification level")
end
