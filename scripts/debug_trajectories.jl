#!/usr/bin/env julia
"""
Debug script: Compare true, SVD-estimated, and predicted trajectories.

This helps diagnose whether the problem is:
1. SVD embedding not recovering true positions
2. Neural ODE not learning the dynamics
3. Both
"""

using Pkg
Pkg.activate(".")

using RDPGDynamics
using LinearAlgebra
using CairoMakie
using JSON
using Random

# Include the data generation function
include("generate_synthetic_data.jl")

println("=" ^ 60)
println("Trajectory Comparison: True vs SVD-Estimated vs Predicted")
println("=" ^ 60)

# =============================================================================
# Step 1: Generate data with KNOWN ground truth
# =============================================================================
println("\n1. Generating bridge node data with known ground truth...")

data = oscillating_bridge_node(n1=25, n2=24, d=2, timesteps=40, omega=0.2, seed=1234)
n, d, T = data.n, data.d, data.timesteps
bridge_idx = data.bridge_node_idx

# True trajectory (ground truth L matrices, stored as d × n)
# Convert to our n × d format
L_true = [data.L_series[t]' for t in 1:T]  # Now n × d

println("  n = " * string(n) * ", d = " * string(d) * ", T = " * string(T))
println("  Bridge node: " * string(bridge_idx))

# =============================================================================
# Step 2: Sample adjacency matrices from true probabilities
# =============================================================================
println("\n2. Sampling adjacency matrices from P = L * L'...")

A_series = Vector{Matrix{Float64}}(undef, T)
for t in 1:T
    L_t = L_true[t]
    P = L_t * L_t'  # Probability matrix
    P = clamp.(P, 0.0, 1.0)

    # Sample binary adjacency
    A = zeros(n, n)
    for i in 1:n
        for j in (i+1):n
            if rand() < P[i, j]
                A[i, j] = 1.0
                A[j, i] = 1.0
            end
        end
    end
    A_series[t] = A
end

# Check sparsity
avg_density = mean([sum(A) / (n * (n-1)) for A in A_series])
println("  Average edge density: " * string(round(avg_density; digits=3)))

# =============================================================================
# Step 3: Re-embed using SVD with B^d_+ alignment
# =============================================================================
println("\n3. Re-embedding via SVD with B^d_+ alignment...")

# Use our temporal embedding function
L_svd = embed_temporal_network_Bd_plus(A_series, d; L_true=L_true)

# Check alignment quality
println("  Checking alignment quality (Frobenius distance to true):")
for t in [1, 10, 20, 30, 40]
    dist = norm(L_svd[t] - L_true[t])
    println("    t=" * string(t) * ": ||L_svd - L_true|| = " * string(round(dist; digits=4)))
end

# =============================================================================
# Step 4: Train Neural ODE (with tanh, smaller network per SciML best practices)
# =============================================================================
println("\n4. Training Neural ODE...")
println("  Using tanh activation, smaller network [32, 32]")

# Override config to use tanh and smaller network
config = SingleTargetConfig(
    n = n,
    d = d,
    target_node = bridge_idx,
    datasize = 35,
    hidden_sizes = [32, 32],  # Smaller, per SciML examples
    activation = tanh,         # tanh instead of celu
    learning_rate = 0.01,
    epochs = 500,
    seed = 1234
)

# Train on SVD-estimated data (what we'd have in practice)
trained = train_single_target(L_svd, config; verbose=true)

# Predict
predictions = predict_single_target(trained, L_svd, T)

# =============================================================================
# Step 5: Create comparison visualization
# =============================================================================
println("\n5. Creating comparison visualization...")

fig = Figure(size=(1200, 800))

# Extract trajectories for bridge node
true_traj = hcat([L_true[t][bridge_idx, :] for t in 1:T]...)  # 2 × T
svd_traj = hcat([L_svd[t][bridge_idx, :] for t in 1:T]...)    # 2 × T
pred_traj = predictions  # d × T

# Panel A: Phase portrait (2D trajectory)
ax1 = Axis(fig[1, 1],
    xlabel="Dimension 1", ylabel="Dimension 2",
    title="Phase Portrait: Bridge Node Trajectory",
    aspect=1)

# Plot B^d_+ boundary (quarter circle in positive quadrant)
θ_arc = range(0, π/2; length=100)
lines!(ax1, cos.(θ_arc), sin.(θ_arc), color=:gray, linestyle=:dash, label="B^d_+ boundary")

# True trajectory
lines!(ax1, true_traj[1, :], true_traj[2, :], color=:blue, linewidth=2, label="True")
scatter!(ax1, [true_traj[1, 1]], [true_traj[2, 1]], color=:blue, markersize=15, marker=:circle)
scatter!(ax1, [true_traj[1, end]], [true_traj[2, end]], color=:blue, markersize=15, marker=:star5)

# SVD-estimated trajectory
lines!(ax1, svd_traj[1, :], svd_traj[2, :], color=:orange, linewidth=2, label="SVD-estimated")
scatter!(ax1, [svd_traj[1, 1]], [svd_traj[2, 1]], color=:orange, markersize=12, marker=:circle)

# Predicted trajectory
lines!(ax1, pred_traj[1, :], pred_traj[2, :], color=:red, linewidth=2, linestyle=:dash, label="Predicted")
scatter!(ax1, [pred_traj[1, 1]], [pred_traj[2, 1]], color=:red, markersize=12, marker=:circle)

axislegend(ax1, position=:lb)
xlims!(ax1, -0.1, 1.1)
ylims!(ax1, -0.1, 1.1)

# Panel B: Dimension 1 over time
ax2 = Axis(fig[2, 1],
    xlabel="Time", ylabel="Dimension 1",
    title="Dimension 1 over Time")

lines!(ax2, 1:T, true_traj[1, :], color=:blue, linewidth=2, label="True")
lines!(ax2, 1:T, svd_traj[1, :], color=:orange, linewidth=2, label="SVD-estimated")
lines!(ax2, 1:T, pred_traj[1, :], color=:red, linewidth=2, linestyle=:dash, label="Predicted")
axislegend(ax2, position=:rt)

# Panel C: Dimension 2 over time
ax3 = Axis(fig[2, 2],
    xlabel="Time", ylabel="Dimension 2",
    title="Dimension 2 over Time")

lines!(ax3, 1:T, true_traj[2, :], color=:blue, linewidth=2, label="True")
lines!(ax3, 1:T, svd_traj[2, :], color=:orange, linewidth=2, label="SVD-estimated")
lines!(ax3, 1:T, pred_traj[2, :], color=:red, linewidth=2, linestyle=:dash, label="Predicted")
axislegend(ax3, position=:rt)

# Panel D: Error metrics
ax4 = Axis(fig[1, 2],
    xlabel="Time", ylabel="Error (Euclidean distance)",
    title="Reconstruction Errors over Time")

# SVD error (how well SVD recovers true positions)
svd_error = [norm(svd_traj[:, t] - true_traj[:, t]) for t in 1:T]
# Prediction error (how well NN predicts SVD trajectory)
pred_error = [norm(pred_traj[:, t] - svd_traj[:, t]) for t in 1:T]
# Total error (prediction vs true)
total_error = [norm(pred_traj[:, t] - true_traj[:, t]) for t in 1:T]

lines!(ax4, 1:T, svd_error, color=:orange, linewidth=2, label="SVD vs True")
lines!(ax4, 1:T, pred_error, color=:red, linewidth=2, label="Pred vs SVD")
lines!(ax4, 1:T, total_error, color=:purple, linewidth=2, linestyle=:dash, label="Pred vs True")
axislegend(ax4, position=:rt)

# Add summary statistics as text
avg_svd_err = mean(svd_error)
avg_pred_err = mean(pred_error)
avg_total_err = mean(total_error)

Label(fig[3, 1:2],
    "Mean Errors: SVD reconstruction = " * string(round(avg_svd_err; digits=4)) *
    " | NN prediction = " * string(round(avg_pred_err; digits=4)) *
    " | Total = " * string(round(avg_total_err; digits=4)),
    fontsize=14)

save("results/debug_trajectory_comparison.pdf", fig)
println("\nSaved: results/debug_trajectory_comparison.pdf")

# =============================================================================
# Step 6: Print diagnostic summary
# =============================================================================
println("\n" * "=" ^ 60)
println("DIAGNOSTIC SUMMARY")
println("=" ^ 60)

println("\nSVD Reconstruction Quality:")
println("  Mean error: " * string(round(avg_svd_err; digits=4)))
println("  Max error:  " * string(round(maximum(svd_error); digits=4)))
if avg_svd_err > 0.1
    println("  ⚠️  SVD reconstruction is poor - embedding may not be recoverable from adjacency")
else
    println("  ✓  SVD reconstruction is good")
end

println("\nNeural ODE Prediction Quality:")
println("  Mean error: " * string(round(avg_pred_err; digits=4)))
println("  Max error:  " * string(round(maximum(pred_error); digits=4)))

# Check if prediction is constant (collapsing to mean)
pred_variance = var(pred_traj[1, :]) + var(pred_traj[2, :])
true_variance = var(true_traj[1, :]) + var(true_traj[2, :])
variance_ratio = pred_variance / true_variance

println("\nVariance Analysis:")
println("  True trajectory variance: " * string(round(true_variance; digits=4)))
println("  Predicted variance:       " * string(round(pred_variance; digits=4)))
println("  Ratio (pred/true):        " * string(round(variance_ratio; digits=4)))

if variance_ratio < 0.1
    println("  ⚠️  Prediction is nearly constant - NN collapsed to mean!")
    println("     Possible causes:")
    println("     - Learning rate too high/low")
    println("     - Network too large (overfitting to initial condition)")
    println("     - Gradient vanishing through ODE solver")
elseif variance_ratio < 0.5
    println("  ⚠️  Prediction has low variance - NN is underfitting dynamics")
else
    println("  ✓  Prediction captures trajectory variance")
end

println("\n" * "=" ^ 60)
