#!/usr/bin/env julia
"""
Test: Does training with DISTANCE-BASED LOSS solve the rotation problem?

Instead of:  L = ||X_pred - X_target||²  (position loss)
Use:         L = ||D(X_pred) - D(X_target)||²  (distance loss)

This should be rotation-invariant and might not require equivariant architectures.
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq
using CairoMakie

println("=" ^ 70)
println("Testing Distance-Based Loss vs Position Loss")
println("=" ^ 70)

# =============================================================================
# Utilities
# =============================================================================

function sample_adjacency(X::Matrix{T}) where T
    n = size(X, 1)
    P = X * X'
    P = clamp.(P, zero(T), one(T))
    A = T.(rand(n, n) .< P)
    for i in 1:n, j in 1:i-1
        A[i, j] = A[j, i]
    end
    return A
end

function svd_embed(A::Matrix, d::Int)
    F = svd(A)
    L = F.U[:, 1:d] .* sqrt.(F.S[1:d]')
    return L
end

function pairwise_distances(X::Matrix)
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        D[i, j] = norm(X[i, :] - X[j, :])
        D[j, i] = D[i, j]
    end
    return D
end

function upper_tri(M::Matrix)
    n = size(M, 1)
    return [M[i, j] for i in 1:n for j in i+1:n]
end

# =============================================================================
# True dynamics
# =============================================================================

function true_dynamics!(du, u, p, t)
    n, d = p.n, p.d
    X = reshape(u, n, d)
    dX = zeros(n, d)

    k_in, k_out = p.k_in, p.k_out
    c1, c2 = p.community_1, p.community_2

    X_bar1 = mean(X[c1, :], dims=1)[1, :]
    X_bar2 = mean(X[c2, :], dims=1)[1, :]

    for i in 1:n
        if i in c1
            dX[i, :] = -k_in .* (X[i, :] .- X_bar1)
            dX[i, :] .+= k_out .* (X_bar2 .- X[i, :])
        else
            dX[i, :] = -k_in .* (X[i, :] .- X_bar2)
            dX[i, :] .+= k_out .* (X_bar1 .- X[i, :])
        end

        for j in 1:d
            if X[i, j] < 0.1
                dX[i, j] += 0.5 * exp(-X[i, j] / 0.05)
            end
        end
        r = norm(X[i, :])
        if r > 0.8
            dX[i, :] .-= 0.5 * exp((r - 1) / 0.1) .* X[i, :] ./ r
        end
    end

    du .= vec(dX)
end

# =============================================================================
# Simple NN with two training modes
# =============================================================================

mutable struct SimpleNN
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function SimpleNN(input_dim::Int, hidden::Int, output_dim::Int; rng=Random.GLOBAL_RNG)
    scale = 0.1
    return SimpleNN(
        scale * randn(rng, hidden, input_dim),
        zeros(hidden),
        scale * randn(rng, output_dim, hidden),
        zeros(output_dim)
    )
end

function forward(nn::SimpleNN, x::Vector)
    h = tanh.(nn.W1 * x .+ nn.b1)
    return nn.W2 * h .+ nn.b2
end

function train_position_loss!(nn::SimpleNN, X_series::Vector{Matrix{Float64}}, dt::Float64;
                              epochs::Int=500, lr::Float64=0.01)
    n, d = size(X_series[1])
    T_len = length(X_series)

    for epoch in 1:epochs
        total_loss = 0.0
        grad_W1, grad_b1 = zeros(size(nn.W1)), zeros(size(nn.b1))
        grad_W2, grad_b2 = zeros(size(nn.W2)), zeros(size(nn.b2))

        for t in 1:T_len-1
            x_curr = vec(X_series[t])
            x_next = vec(X_series[t+1])

            dx_pred = forward(nn, x_curr)
            x_pred = x_curr .+ dt .* dx_pred

            loss = sum((x_pred .- x_next).^2)
            total_loss += loss

            dloss = 2 .* (x_pred .- x_next) .* dt
            h = tanh.(nn.W1 * x_curr .+ nn.b1)

            grad_W2 .+= dloss * h'
            grad_b2 .+= dloss

            dh = nn.W2' * dloss
            dtanh = dh .* (1 .- h.^2)
            grad_W1 .+= dtanh * x_curr'
            grad_b1 .+= dtanh
        end

        nn.W1 .-= lr * grad_W1 / (T_len - 1)
        nn.b1 .-= lr * grad_b1 / (T_len - 1)
        nn.W2 .-= lr * grad_W2 / (T_len - 1)
        nn.b2 .-= lr * grad_b2 / (T_len - 1)

        if epoch % 100 == 0
            println("    Epoch " * string(epoch) * ": loss = " * string(round(total_loss / (T_len-1), digits=6)))
        end
    end
end

function train_distance_loss!(nn::SimpleNN, X_series::Vector{Matrix{Float64}}, dt::Float64;
                              epochs::Int=500, lr::Float64=0.01)
    n, d = size(X_series[1])
    T_len = length(X_series)

    # Precompute target distances
    D_targets = [pairwise_distances(X) for X in X_series]

    for epoch in 1:epochs
        total_loss = 0.0
        grad_W1, grad_b1 = zeros(size(nn.W1)), zeros(size(nn.b1))
        grad_W2, grad_b2 = zeros(size(nn.W2)), zeros(size(nn.b2))

        for t in 1:T_len-1
            x_curr = vec(X_series[t])
            D_target = D_targets[t+1]

            # Forward: predict next positions
            dx_pred = forward(nn, x_curr)
            x_pred = x_curr .+ dt .* dx_pred
            X_pred = reshape(x_pred, n, d)

            # Compute predicted distances
            D_pred = pairwise_distances(X_pred)

            # Distance loss
            loss = sum((D_pred .- D_target).^2) / 2  # Symmetric matrix
            total_loss += loss

            # Gradient of distance loss w.r.t. positions (manual)
            # ∂||Xi-Xj||/∂Xi = (Xi-Xj)/||Xi-Xj||
            dL_dX = zeros(n, d)
            for i in 1:n, j in i+1:n
                diff = X_pred[i, :] .- X_pred[j, :]
                d_ij = norm(diff)
                if d_ij > 1e-8
                    dL_dD = 2 * (D_pred[i, j] - D_target[i, j])
                    dD_dX = diff ./ d_ij
                    dL_dX[i, :] .+= dL_dD .* dD_dX
                    dL_dX[j, :] .-= dL_dD .* dD_dX
                end
            end

            dloss = vec(dL_dX) .* dt

            # Backprop through NN
            h = tanh.(nn.W1 * x_curr .+ nn.b1)
            grad_W2 .+= dloss * h'
            grad_b2 .+= dloss

            dh = nn.W2' * dloss
            dtanh = dh .* (1 .- h.^2)
            grad_W1 .+= dtanh * x_curr'
            grad_b1 .+= dtanh
        end

        nn.W1 .-= lr * grad_W1 / (T_len - 1)
        nn.b1 .-= lr * grad_b1 / (T_len - 1)
        nn.W2 .-= lr * grad_W2 / (T_len - 1)
        nn.b2 .-= lr * grad_b2 / (T_len - 1)

        if epoch % 100 == 0
            println("    Epoch " * string(epoch) * ": loss = " * string(round(total_loss / (T_len-1), digits=6)))
        end
    end
end

function simulate_nn(nn::SimpleNN, X0::Matrix{Float64}, dt::Float64, T_len::Int)
    n, d = size(X0)
    X_series = [copy(X0)]

    x = vec(X0)
    for t in 1:T_len-1
        dx = forward(nn, x)
        x = x .+ dt .* dx
        push!(X_series, reshape(copy(x), n, d))
    end

    return X_series
end

# =============================================================================
# Generate data
# =============================================================================

println("\n1. Generating true trajectory...")

n = 16
d = 2
T_len = 30
dt = 1.0

community_1 = 1:8
community_2 = 9:16

rng = MersenneTwister(42)
X0_true = zeros(n, d)
for i in community_1
    X0_true[i, :] = [0.3, 0.7] .+ 0.1 .* randn(rng, 2)
end
for i in community_2
    X0_true[i, :] = [0.7, 0.3] .+ 0.1 .* randn(rng, 2)
end

X0_true = max.(X0_true, 0.1)
for i in 1:n
    r = norm(X0_true[i, :])
    if r > 0.9
        X0_true[i, :] .*= 0.9 / r
    end
end

params = (n=n, d=d, k_in=0.15, k_out=0.08, community_1=community_1, community_2=community_2)
prob = ODEProblem(true_dynamics!, vec(X0_true), (0.0, (T_len-1) * dt), params)
sol = solve(prob, Tsit5(), saveat=dt)
X_true_series = [reshape(sol.u[t], n, d) for t in 1:T_len]

println("  Generated " * string(T_len) * " timesteps")

# Embed
println("\n2. RDPG estimation...")
K = 50
X_est_series = Vector{Matrix{Float64}}(undef, T_len)
for t in 1:T_len
    A_avg = zeros(n, n)
    for k in 1:K
        A_avg .+= sample_adjacency(X_true_series[t])
    end
    A_avg ./= K
    X_est_series[t] = svd_embed(A_avg, d)
end

# =============================================================================
# Train both models
# =============================================================================

println("\n3. Training with POSITION LOSS...")
nn_pos = SimpleNN(n * d, 64, n * d; rng=MersenneTwister(123))
train_position_loss!(nn_pos, X_est_series, dt; epochs=500, lr=0.005)

println("\n4. Training with DISTANCE LOSS...")
nn_dist = SimpleNN(n * d, 64, n * d; rng=MersenneTwister(123))
train_distance_loss!(nn_dist, X_est_series, dt; epochs=500, lr=0.0005)  # Lower LR for stability

# Simulate
X_rec_pos = simulate_nn(nn_pos, X_est_series[1], dt, T_len)
X_rec_dist = simulate_nn(nn_dist, X_est_series[1], dt, T_len)

# =============================================================================
# Evaluate
# =============================================================================

println("\n" * "=" ^ 70)
println("COMPARISON: Position Loss vs Distance Loss")
println("=" ^ 70)

D_true = [pairwise_distances(X) for X in X_true_series]
D_est = [pairwise_distances(X) for X in X_est_series]
D_rec_pos = [pairwise_distances(X) for X in X_rec_pos]
D_rec_dist = [pairwise_distances(X) for X in X_rec_dist]

P_true = [X * X' for X in X_true_series]
P_rec_pos = [X * X' for X in X_rec_pos]
P_rec_dist = [X * X' for X in X_rec_dist]

println("\n--- True ↔ Recovered: Distance Correlation ---")
println("  t  | Position Loss | Distance Loss")
println("  " * "-" ^ 35)

for t in [1, 10, 20, 30]
    corr_pos = cor(upper_tri(D_true[t]), upper_tri(D_rec_pos[t]))
    corr_dist = cor(upper_tri(D_true[t]), upper_tri(D_rec_dist[t]))
    println("  " * lpad(string(t), 2) * " | " *
            rpad(string(round(corr_pos, digits=3)), 13) * " | " *
            string(round(corr_dist, digits=3)))
end

println("\n--- True ↔ Recovered: P Reconstruction Error ---")
println("  t  | Position Loss | Distance Loss")
println("  " * "-" ^ 35)

for t in [1, 10, 20, 30]
    err_pos = norm(P_true[t] - P_rec_pos[t]) / norm(P_true[t])
    err_dist = norm(P_true[t] - P_rec_dist[t]) / norm(P_true[t])
    println("  " * lpad(string(t), 2) * " | " *
            rpad(string(round(err_pos, digits=4)), 13) * " | " *
            string(round(err_dist, digits=4)))
end

# Summary
avg_d_corr_pos = mean([cor(upper_tri(D_true[t]), upper_tri(D_rec_pos[t])) for t in 1:T_len])
avg_d_corr_dist = mean([cor(upper_tri(D_true[t]), upper_tri(D_rec_dist[t])) for t in 1:T_len])

avg_p_err_pos = mean([norm(P_true[t] - P_rec_pos[t]) / norm(P_true[t]) for t in 1:T_len])
avg_p_err_dist = mean([norm(P_true[t] - P_rec_dist[t]) / norm(P_true[t]) for t in 1:T_len])

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nAverage Distance Correlation with True:")
println("  Position Loss: " * string(round(avg_d_corr_pos, digits=4)))
println("  Distance Loss: " * string(round(avg_d_corr_dist, digits=4)))

println("\nAverage P Error vs True:")
println("  Position Loss: " * string(round(avg_p_err_pos, digits=4)))
println("  Distance Loss: " * string(round(avg_p_err_dist, digits=4)))

if avg_d_corr_dist > avg_d_corr_pos + 0.1
    println("\n✓ Distance loss significantly outperforms position loss!")
elseif avg_d_corr_dist > avg_d_corr_pos
    println("\n✓ Distance loss outperforms position loss")
else
    println("\n~ Results are similar or position loss is better")
end

println("""

CONCLUSION:
If distance loss works better, then we can achieve rotation-invariant goals
by simply using the right loss function - no need for equivariant architectures!

The key insight: train on what you care about (distances, P), not positions.
""")

# Visualization
fig = Figure(size=(1200, 400))

ts = 1:T_len
d_corr_pos_series = [cor(upper_tri(D_true[t]), upper_tri(D_rec_pos[t])) for t in ts]
d_corr_dist_series = [cor(upper_tri(D_true[t]), upper_tri(D_rec_dist[t])) for t in ts]
d_corr_est_series = [cor(upper_tri(D_true[t]), upper_tri(D_est[t])) for t in ts]

ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Distance Corr with True",
           title="Distance Recovery: Position vs Distance Loss")

lines!(ax1, ts, d_corr_est_series, linewidth=2, label="Estimated (baseline)", color=:gray)
lines!(ax1, ts, d_corr_pos_series, linewidth=2, label="Position Loss", color=:blue)
lines!(ax1, ts, d_corr_dist_series, linewidth=2, label="Distance Loss", color=:red)
hlines!(ax1, [0], color=:black, linestyle=:dash, alpha=0.3)
axislegend(ax1, position=:lb)

# Inter-community distance
inter_true = [mean([D_true[t][i,j] for i in community_1 for j in community_2]) for t in ts]
inter_est = [mean([D_est[t][i,j] for i in community_1 for j in community_2]) for t in ts]
inter_pos = [mean([D_rec_pos[t][i,j] for i in community_1 for j in community_2]) for t in ts]
inter_dist = [mean([D_rec_dist[t][i,j] for i in community_1 for j in community_2]) for t in ts]

ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="Inter-community Distance",
           title="Community Dynamics Recovery")

lines!(ax2, ts, inter_true, linewidth=3, label="True", color=:black)
lines!(ax2, ts, inter_est, linewidth=2, label="Estimated", color=:gray, linestyle=:dash)
lines!(ax2, ts, inter_pos, linewidth=2, label="Pos Loss", color=:blue, linestyle=:dot)
lines!(ax2, ts, inter_dist, linewidth=2, label="Dist Loss", color=:red, linestyle=:dashdot)
axislegend(ax2, position=:rt)

# P error over time
p_err_pos_series = [norm(P_true[t] - P_rec_pos[t]) / norm(P_true[t]) for t in ts]
p_err_dist_series = [norm(P_true[t] - P_rec_dist[t]) / norm(P_true[t]) for t in ts]

ax3 = Axis(fig[1, 3], xlabel="Time", ylabel="P Error (relative)",
           title="Probability Matrix Recovery", yscale=log10)

lines!(ax3, ts, p_err_pos_series .+ 1e-10, linewidth=2, label="Position Loss", color=:blue)
lines!(ax3, ts, p_err_dist_series .+ 1e-10, linewidth=2, label="Distance Loss", color=:red)
axislegend(ax3, position=:lt)

mkpath("results")
save("results/distance_vs_position_loss.pdf", fig)
println("\nSaved: results/distance_vs_position_loss.pdf")
