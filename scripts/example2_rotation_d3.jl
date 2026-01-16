#!/usr/bin/env julia
"""
Example 2 (v2): Rotation in B³₊ - Learning in Hyperspherical Space

Convert all data to hyperspherical coordinates (r, φ₁, φ₂) and learn directly
in that space. This avoids coordinate transforms during differentiation.

Dynamics in hyperspherical space:
- dr/dt = f(r, φ) [UNKNOWN - radial dynamics + boundary]
- dφ₁/dt = ω₁ [KNOWN - constant rotation]
- dφ₂/dt = ω₂ [KNOWN - constant rotation]

The NN learns the unknown radial dynamics, while rotation is known exactly.
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
const SAVE_FILE = joinpath(RESULTS_DIR, "example2_rotation_d3_results.jls")

println("=" ^ 60)
println("Example 2 (v2): Rotation in B³₊ (Hyperspherical Space)")
if VIZ_ONLY
    println("  (Visualization-only mode)")
end
println("=" ^ 60)

# =============================================================================
# Coordinate conversions (for data preprocessing only)
# =============================================================================

"""
Convert Cartesian to hyperspherical: (x₁, x₂, x₃) → (r, φ₁, φ₂)
For B³₊: r ∈ [0,1], φ₁ ∈ [0, π/2], φ₂ ∈ [0, π/2]

x₁ = r cos(φ₁)
x₂ = r sin(φ₁) cos(φ₂)
x₃ = r sin(φ₁) sin(φ₂)
"""
function cartesian_to_hyperspherical(x::AbstractVector{T}) where T
    r = norm(x)
    if r < T(1e-10)
        return T[zero(T), T(π/4), T(π/4)]  # Default angles at origin
    end

    # φ₁ = acos(x₁/r)
    phi1 = acos(clamp(x[1] / r, -one(T), one(T)))

    # φ₂ from x₂, x₃
    sin_phi1 = sin(phi1)
    if sin_phi1 < T(1e-10)
        phi2 = T(π/4)  # Default when on x₁ axis
    else
        # x₂ = r sin(φ₁) cos(φ₂), x₃ = r sin(φ₁) sin(φ₂)
        cos_phi2 = x[2] / (r * sin_phi1)
        sin_phi2 = x[3] / (r * sin_phi1)
        phi2 = atan(sin_phi2, cos_phi2)
        phi2 = clamp(phi2, zero(T), T(π/2))
    end

    return T[r, phi1, phi2]
end

"""
Convert hyperspherical to Cartesian: (r, φ₁, φ₂) → (x₁, x₂, x₃)
Applies angle wrapping to ensure result is in B³₊.
"""
function hyperspherical_to_cartesian(u::AbstractVector{T}; wrap::Bool=true) where T
    r = u[1]
    phi1 = wrap ? wrap_angle(u[2]) : u[2]
    phi2 = wrap ? wrap_angle(u[3]) : u[3]
    x1 = r * cos(phi1)
    x2 = r * sin(phi1) * cos(phi2)
    x3 = r * sin(phi1) * sin(phi2)
    return T[x1, x2, x3]
end

"""
Apply wrapping to hyperspherical coordinates.
"""
function wrap_hyperspherical(u::AbstractVector{T}) where T
    return T[u[1], wrap_angle(u[2]), wrap_angle(u[3])]
end

# =============================================================================
# True dynamics parameters
# =============================================================================

# Known angular rotation (faster to see multiple rotations)
const TRUE_OMEGA1 = 0.15f0   # dφ₁/dt - completes ~1.2 rotations in T=50
const TRUE_OMEGA2 = 0.12f0   # dφ₂/dt - completes ~1 rotation in T=50

# Unknown radial dynamics
const TRUE_K_RADIAL = 0.12f0   # Radial relaxation
const TRUE_R_TARGET = 0.55f0   # Target radius

# Unknown boundary repulsion (only for radial, angles wrap)
const TRUE_ALPHA = 0.2f0       # Boundary strength
const TRUE_EPSILON = 0.1f0     # Boundary sharpness

# Angular period for modular wrapping (use π/2 for B³₊ oscillation)
const ANGLE_PERIOD = Float32(π/2)

"""
Wrap angle to [0, π/2] using triangular wave (bounce at boundaries).
This keeps trajectories in B³₊ while allowing continuous rotation.
"""
function wrap_angle(phi::T) where T
    # Normalize to [0, 2π)
    phi_mod = mod(phi, T(2π))
    # Triangular wave with period π (bounces at 0 and π/2)
    # Map [0, π/2] → [0, π/2], [π/2, π] → [π/2, 0], [π, 3π/2] → [0, π/2], etc.
    quarter = T(π/2)
    n_quarters = floor(Int, phi_mod / quarter)
    remainder = phi_mod - n_quarters * quarter
    if iseven(n_quarters)
        return remainder
    else
        return quarter - remainder
    end
end

"""
True dynamics in hyperspherical coordinates.
State: u = [r, φ₁, φ₂] where angles are unbounded (wrap applied later)
"""
function true_hyperspherical_dynamics(u::Vector{T}) where T
    r, phi1, phi2 = u[1], u[2], u[3]

    # Known: constant angular rotation (angles evolve freely, wrap later)
    dphi1 = T(TRUE_OMEGA1)
    dphi2 = T(TRUE_OMEGA2)

    # Unknown: radial dynamics - attraction to target radius
    dr = -TRUE_K_RADIAL * (r - TRUE_R_TARGET)

    # Unknown: boundary repulsion in radial direction only
    # Keep away from r=0 and r=1
    dr += TRUE_ALPHA * exp(-r / TRUE_EPSILON)  # Push away from r=0
    dr -= TRUE_ALPHA * exp((r - one(T)) / TRUE_EPSILON)  # Push away from r=1

    # No angular boundary repulsion - angles wrap via modular arithmetic

    return T[dr, dphi1, dphi2]
end

function true_hyperspherical_dynamics!(du, u, p, t)
    result = true_hyperspherical_dynamics(Vector{eltype(u)}(u))
    du[1] = result[1]
    du[2] = result[2]
    du[3] = result[3]
end

# =============================================================================
# Generate data
# =============================================================================

function generate_data(; n::Int=25, T_end::Float32=50f0, dt::Float32=1f0, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    # Initial positions in hyperspherical: (r, φ₁, φ₂)
    u0_all = Vector{Vector{Float32}}(undef, n)
    for i in 1:n
        r = Float32(0.4 + 0.35 * rand(rng))      # r in [0.4, 0.75]
        phi1 = Float32(π/6 + π/6 * rand(rng))    # φ₁ in [π/6, π/3]
        phi2 = Float32(π/6 + π/6 * rand(rng))    # φ₂ in [π/6, π/3]
        u0_all[i] = Float32[r, phi1, phi2]
    end

    # Solve ODE in hyperspherical coordinates
    tspan = (0f0, T_end)
    tsteps = 0f0:dt:T_end
    n_steps = length(tsteps)

    trajectories_raw = Dict{Int, Matrix{Float32}}()      # Unwrapped angles (for training)
    trajectories_wrapped = Dict{Int, Matrix{Float32}}()  # Wrapped angles (for visualization)
    trajectories_cartesian = Dict{Int, Matrix{Float32}}()

    for i in 1:n
        prob = ODEProblem(true_hyperspherical_dynamics!, u0_all[i], tspan)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1f-7, reltol=1f-7)

        traj_raw = zeros(Float32, 3, n_steps)
        traj_wrapped = zeros(Float32, 3, n_steps)
        traj_cart = zeros(Float32, 3, n_steps)
        for (t_idx, t) in enumerate(tsteps)
            u = sol(t)
            traj_raw[:, t_idx] = u
            traj_wrapped[:, t_idx] = wrap_hyperspherical(u)
            traj_cart[:, t_idx] = hyperspherical_to_cartesian(u)  # Uses wrapping
        end
        trajectories_raw[i] = traj_raw
        trajectories_wrapped[i] = traj_wrapped
        trajectories_cartesian[i] = traj_cart
    end

    return (
        trajectories = trajectories_raw,              # Raw unwrapped (for NN training)
        trajectories_wrapped = trajectories_wrapped,  # Wrapped (for some viz)
        trajectories_cartesian = trajectories_cartesian,  # Cartesian (for 3D viz)
        n = n,
        T_end = T_end,
        dt = dt,
        tsteps = tsteps,
        u0_all = u0_all
    )
end

println("\n1. Generating rotation dynamics data...")
data = generate_data(n=25, T_end=50f0, dt=1f0)
n = data.n
tsteps = data.tsteps
T_steps = length(tsteps)

println("  n=" * string(n) * " nodes, T=" * string(data.T_end) * ", steps=" * string(T_steps))
println("  Learning in hyperspherical coordinates (r, φ₁, φ₂)")
println("\n  True parameters:")
println("    ω₁ (known):        " * string(TRUE_OMEGA1))
println("    ω₂ (known):        " * string(TRUE_OMEGA2))
println("    k_radial (unknown):" * string(TRUE_K_RADIAL))
println("    r_target (unknown):" * string(TRUE_R_TARGET))
println("    alpha (unknown):   " * string(TRUE_ALPHA))

# Verify constraints
println("\n  Verifying constraints...")

function verify_constraints(trajectories, n_nodes, n_steps)
    min_r, max_r = Inf, 0.0
    min_phi1, max_phi1 = Inf, 0.0
    min_phi2, max_phi2 = Inf, 0.0

    for i in 1:n_nodes
        traj = trajectories[i]
        min_r = min(min_r, minimum(traj[1, :]))
        max_r = max(max_r, maximum(traj[1, :]))
        min_phi1 = min(min_phi1, minimum(traj[2, :]))
        max_phi1 = max(max_phi1, maximum(traj[2, :]))
        min_phi2 = min(min_phi2, minimum(traj[3, :]))
        max_phi2 = max(max_phi2, maximum(traj[3, :]))
    end
    return (min_r=min_r, max_r=max_r, min_phi1=min_phi1, max_phi1=max_phi1,
            min_phi2=min_phi2, max_phi2=max_phi2)
end

c = verify_constraints(data.trajectories, n, T_steps)
println("    r ∈ [" * string(round(c.min_r, digits=3)) * ", " * string(round(c.max_r, digits=3)) * "] (should be in [0, 1])")
println("    φ₁ ∈ [" * string(round(c.min_phi1, digits=3)) * ", " * string(round(c.max_phi1, digits=3)) * "] (should be in [0, π/2])")
println("    φ₂ ∈ [" * string(round(c.min_phi2, digits=3)) * ", " * string(round(c.max_phi2, digits=3)) * "] (should be in [0, π/2])")

if c.min_r >= 0 && c.max_r <= 1 && c.min_phi1 >= 0 && c.max_phi1 <= π/2 && c.min_phi2 >= 0 && c.max_phi2 <= π/2
    println("  ✓ All trajectories stay in B³₊")
else
    println("  WARNING: Constraint violated!")
end

# =============================================================================
# Setup neural networks
# =============================================================================
println("\n2. Setting up models...")

rng = Random.Xoshiro(42)

# NN for unknown part: learns dr/dt and angular boundary corrections
# Input: [r, φ₁, φ₂], Output: [dr, dφ₁_correction, dφ₂_correction]
nn_unknown = Lux.Chain(
    Lux.Dense(3, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 3)
)

# Full NN: learns everything
nn_full = Lux.Chain(
    Lux.Dense(3, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 3)
)

ps_unknown, st_unknown = Lux.setup(rng, nn_unknown)
ps_full, st_full = Lux.setup(rng, nn_full)

ps_unknown_ca = ComponentArray(ps_unknown)
ps_full_ca = ComponentArray(ps_full)

println("  NN parameters (unknown): " * string(length(ps_unknown_ca)))
println("  NN parameters (full): " * string(length(ps_full_ca)))

# =============================================================================
# UDE and Full NN dynamics (in hyperspherical space)
# =============================================================================

# UDE: Known angular rotation + learned corrections
function ude_dynamics(u, p, t)
    T = eltype(u)

    # Known: constant angular rotation
    dx_known = T[0, TRUE_OMEGA1, TRUE_OMEGA2]

    # Unknown: NN learns radial dynamics and boundary corrections
    u_input = reshape(u, 3, 1)
    nn_out, _ = nn_unknown(u_input, p, st_unknown)

    return dx_known .+ vec(nn_out)
end

# Full NN: learns everything
function full_dynamics(u, p, t)
    u_input = reshape(u, 3, 1)
    nn_out, _ = nn_full(u_input, p, st_full)
    return vec(nn_out)
end

# =============================================================================
# Training data
# =============================================================================
println("\n3. Preparing training data...")

train_nodes = 1:18
val_nodes = 19:25

train_data = Dict{Int, Matrix{Float32}}()
for i in train_nodes
    train_data[i] = data.trajectories[i]
end

println("  Training nodes: " * string(length(train_nodes)))
println("  Validation nodes: " * string(length(val_nodes)))

# =============================================================================
# Loss functions
# =============================================================================
println("\n4. Defining loss functions...")

tspan = (0f0, data.T_end)

function predict_node(u0, p, dynamics_fn)
    prob = ODEProblem(dynamics_fn, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-4, reltol=1f-4)
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
# Training
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
    ps_full_trained = saved["ps_full_trained"]
    ps_ude_trained = saved["ps_ude_trained"]

    improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))
    println("  Loaded!")

else  # Training

if VIZ_ONLY && !isfile(SAVE_FILE)
    println("\n  WARNING: --viz but no saved file. Running training...")
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

println("  Stage 1: ADAM...")
optf_full = Optimization.OptimizationFunction((p, _) -> loss_full(p), Optimization.AutoZygote())
optprob_full = Optimization.OptimizationProblem(optf_full, ps_full_ca)

result_full_adam = Optimization.solve(
    optprob_full,
    OptimizationOptimisers.Adam(0.01),
    maxiters=400,
    callback=callback_full
)

println("  Stage 2: BFGS...")
iter_full[] = 400
optprob_full2 = Optimization.OptimizationProblem(optf_full, result_full_adam.u)

result_full = try
    Optimization.solve(
        optprob_full2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=150,
        callback=callback_full,
        allow_f_increases=false
    )
catch e
    println("    BFGS stopped: " * string(typeof(e)))
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
    maxiters=400,
    callback=callback_ude
)

println("  Stage 2: BFGS...")
iter_ude[] = 400
optprob_ude2 = Optimization.OptimizationProblem(optf_ude, result_ude_adam.u)

result_ude = try
    Optimization.solve(
        optprob_ude2,
        OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
        maxiters=150,
        callback=callback_ude,
        allow_f_increases=false
    )
catch e
    println("    BFGS stopped: " * string(typeof(e)))
    result_ude_adam
end

ps_ude_trained = result_ude.u
loss_ude_final = loss_ude(ps_ude_trained)
println("  Final loss (UDE): " * string(round(loss_ude_final, digits=6)))

# =============================================================================
# Evaluation
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

println("\n  Training set errors (hyperspherical):")
println("    Full NN: " * string(round(mean(errors_full_train), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_train), digits=4)))

println("\n  Validation set errors (hyperspherical):")
println("    Full NN: " * string(round(mean(errors_full_val), digits=4)))
println("    UDE:     " * string(round(mean(errors_ude_val), digits=4)))

improvement_train = 100 * (1 - mean(errors_ude_train) / mean(errors_full_train))
improvement_val = 100 * (1 - mean(errors_ude_val) / mean(errors_full_val))

println("\n  Improvement (UDE vs Full):")
println("    Training:   " * string(round(improvement_train, digits=1)) * "%")
println("    Validation: " * string(round(improvement_val, digits=1)) * "%")

# Save results
println("\n  Saving results...")
mkpath(RESULTS_DIR)
serialize(SAVE_FILE, Dict(
    "errors_full_train" => errors_full_train,
    "errors_ude_train" => errors_ude_train,
    "errors_full_val" => errors_full_val,
    "errors_ude_val" => errors_ude_val,
    "loss_full_final" => loss_full_final,
    "loss_ude_final" => loss_ude_final,
    "ps_full_trained" => ps_full_trained,
    "ps_ude_trained" => ps_ude_trained
))
println("  Saved to " * SAVE_FILE)

end  # end training block

# =============================================================================
# Visualization
# =============================================================================
println("\n8. Creating visualization...")

fig = Figure(size=(1800, 1200))

viz_nodes = [1, 5, 10, 18, 20, 25]
colors = [:blue, :red, :green, :orange, :purple, :cyan]

# --- Panel 1: True trajectories in Cartesian (3D) ---
ax1 = CairoMakie.Axis3(fig[1, 1],
    xlabel="x₁", ylabel="x₂", zlabel="x₃",
    title="True Trajectories (Cartesian)",
    aspect=(1, 1, 1))

theta_range = range(0, π/2, length=30)
for phi in [0f0, π/4, π/2]
    xs = [cos(theta) for theta in theta_range]
    ys = [sin(theta) * cos(phi) for theta in theta_range]
    zs = [sin(theta) * sin(phi) for theta in theta_range]
    lines!(ax1, xs, ys, zs, color=:gray, alpha=0.3)
end

for (idx, i) in enumerate(viz_nodes)
    traj = data.trajectories_cartesian[i]
    c = colors[mod1(idx, length(colors))]
    lines!(ax1, traj[1, :], traj[2, :], traj[3, :], color=c, linewidth=2)
    scatter!(ax1, [traj[1, 1]], [traj[2, 1]], [traj[3, 1]], color=c, markersize=10)
end

# --- Panel 2: Trajectories in hyperspherical space ---
ax2 = CairoMakie.Axis3(fig[1, 2],
    xlabel="r", ylabel="φ₁", zlabel="φ₂",
    title="Trajectories (Hyperspherical)",
    aspect=(1, 1, 1))

for (idx, i) in enumerate(viz_nodes)
    traj = data.trajectories[i]
    c = colors[mod1(idx, length(colors))]
    lines!(ax2, traj[1, :], traj[2, :], traj[3, :], color=c, linewidth=2)
    scatter!(ax2, [traj[1, 1]], [traj[2, 1]], [traj[3, 1]], color=c, markersize=10)
end

# --- Panel 3: UDE vs Full NN in hyperspherical ---
ax3 = CairoMakie.Axis3(fig[1, 3],
    xlabel="r", ylabel="φ₁", zlabel="φ₂",
    title="Predictions (Hyperspherical)",
    aspect=(1, 1, 1))

val_node = first(val_nodes)
traj_val = data.trajectories[val_node]
u0_val = traj_val[:, 1]

pred_full_val = predict_node(u0_val, ps_full_trained, full_dynamics)
pred_ude_val = predict_node(u0_val, ps_ude_trained, ude_dynamics)

lines!(ax3, traj_val[1, :], traj_val[2, :], traj_val[3, :],
       color=:black, linewidth=2, label="True")
lines!(ax3, pred_full_val[1, :], pred_full_val[2, :], pred_full_val[3, :],
       color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax3, pred_ude_val[1, :], pred_ude_val[2, :], pred_ude_val[3, :],
       color=:red, linewidth=2, linestyle=:dot, label="UDE")

# --- Row 2: Component-wise plots ---

# Panel 4: Radius over time
ax4 = CairoMakie.Axis(fig[2, 1], xlabel="Time", ylabel="r",
                       title="Radius Evolution")

lines!(ax4, collect(tsteps), traj_val[1, :], color=:black, linewidth=2, label="True")
lines!(ax4, collect(tsteps), pred_full_val[1, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax4, collect(tsteps), pred_ude_val[1, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")
hlines!(ax4, [TRUE_R_TARGET], color=:green, linestyle=:dash, alpha=0.5, label="Target")

axislegend(ax4, position=:rt)

# Panel 5: φ₁ over time
ax5 = CairoMakie.Axis(fig[2, 2], xlabel="Time", ylabel="φ₁",
                       title="φ₁ Evolution (Angular Rotation)")

lines!(ax5, collect(tsteps), traj_val[2, :], color=:black, linewidth=2, label="True")
lines!(ax5, collect(tsteps), pred_full_val[2, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax5, collect(tsteps), pred_ude_val[2, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")

# Expected linear trend from known rotation
phi1_expected = traj_val[2, 1] .+ TRUE_OMEGA1 .* collect(tsteps)
lines!(ax5, collect(tsteps), phi1_expected, color=:green, linestyle=:dash, alpha=0.5, label="Expected (ω₁t)")

axislegend(ax5, position=:lt)

# Panel 6: φ₂ over time
ax6 = CairoMakie.Axis(fig[2, 3], xlabel="Time", ylabel="φ₂",
                       title="φ₂ Evolution (Angular Rotation)")

lines!(ax6, collect(tsteps), traj_val[3, :], color=:black, linewidth=2, label="True")
lines!(ax6, collect(tsteps), pred_full_val[3, :], color=:blue, linewidth=2, linestyle=:dash, label="Full NN")
lines!(ax6, collect(tsteps), pred_ude_val[3, :], color=:red, linewidth=2, linestyle=:dot, label="UDE")

phi2_expected = traj_val[3, 1] .+ TRUE_OMEGA2 .* collect(tsteps)
lines!(ax6, collect(tsteps), phi2_expected, color=:green, linestyle=:dash, alpha=0.5, label="Expected (ω₂t)")

axislegend(ax6, position=:lt)

save("results/example2_rotation_d3.pdf", fig)
println("\nSaved: results/example2_rotation_d3.pdf")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 60)
println("SUMMARY: Example 2 (v2) - Rotation in B³₊")
println("=" ^ 60)

println("\nDynamics in hyperspherical coordinates (r, φ₁, φ₂):")
println("  KNOWN:   dφ₁/dt = ω₁ = " * string(TRUE_OMEGA1) * ", dφ₂/dt = ω₂ = " * string(TRUE_OMEGA2))
println("  UNKNOWN: dr/dt (radial relaxation + boundary)")
println("\nAll " * string(n) * " nodes follow identical physics.")

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
    println("  Known angular rotation (dφ/dt = ω) helps learning.")
elseif improvement_val > -10
    println("\n~ UDE performs similarly to Full NN.")
else
    println("\n✗ Full NN outperforms UDE.")
end
