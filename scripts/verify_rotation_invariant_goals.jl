#!/usr/bin/env julia
"""
Verify: Can we achieve rotation-invariant goals without explicit equivariance?

Key insight: If we only care about:
- Pairwise distances D
- Probability matrix P = XX'
- Functional form (via symbolic regression on distances)

Then rotation ambiguity might not matter for training!

Three comparisons:
1. True ↔ Estimated:    D, P  (RDPG estimation quality)
2. True ↔ Recovered:    D, P  (dynamics learning quality)
3. Estimated ↔ Recovered: positions, D  (training fit)
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq
using CairoMakie

println("=" ^ 70)
println("Testing Rotation-Invariant Goals")
println("=" ^ 70)

# =============================================================================
# Utility functions
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

function P_matrix(X::Matrix)
    return X * X'
end

# =============================================================================
# True dynamics: Two communities with attraction
# =============================================================================

function true_dynamics!(du, u, p, t)
    n, d = p.n, p.d
    X = reshape(u, n, d)
    dX = zeros(n, d)

    k_in, k_out = p.k_in, p.k_out
    c1, c2 = p.community_1, p.community_2

    # Compute centroids
    X_bar1 = mean(X[c1, :], dims=1)[1, :]
    X_bar2 = mean(X[c2, :], dims=1)[1, :]

    for i in 1:n
        if i in c1
            # Cohesion to own community
            dX[i, :] = -k_in .* (X[i, :] .- X_bar1)
            # Attraction to other community
            dX[i, :] .+= k_out .* (X_bar2 .- X[i, :])
        else
            dX[i, :] = -k_in .* (X[i, :] .- X_bar2)
            dX[i, :] .+= k_out .* (X_bar1 .- X[i, :])
        end

        # Soft boundary (stay in B^d_+)
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
# Simple Neural ODE (NOT explicitly equivariant)
# =============================================================================

struct SimpleNN
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function SimpleNN(input_dim::Int, hidden::Int, output_dim::Int; rng=Random.GLOBAL_RNG)
    scale = 0.1
    W1 = scale * randn(rng, hidden, input_dim)
    b1 = zeros(hidden)
    W2 = scale * randn(rng, output_dim, hidden)
    b2 = zeros(output_dim)
    return SimpleNN(W1, b1, W2, b2)
end

function (nn::SimpleNN)(x::Vector)
    h = tanh.(nn.W1 * x .+ nn.b1)
    return nn.W2 * h .+ nn.b2
end

function nn_dynamics!(du, u, p, t)
    du .= p.nn(u)
end

# =============================================================================
# Training utilities
# =============================================================================

function train_simple_nn!(nn::SimpleNN, X_series::Vector{Matrix{Float64}}, dt::Float64;
                          epochs::Int=500, lr::Float64=0.01)
    n, d = size(X_series[1])
    T_len = length(X_series)

    # Training: predict next state from current
    for epoch in 1:epochs
        total_loss = 0.0

        # Gradient accumulation (simple finite differences)
        grad_W1 = zeros(size(nn.W1))
        grad_b1 = zeros(size(nn.b1))
        grad_W2 = zeros(size(nn.W2))
        grad_b2 = zeros(size(nn.b2))

        for t in 1:T_len-1
            x_curr = vec(X_series[t])
            x_next = vec(X_series[t+1])

            # Forward
            dx_pred = nn(x_curr)
            x_pred = x_curr .+ dt .* dx_pred

            # Loss
            loss = sum((x_pred .- x_next).^2)
            total_loss += loss

            # Backward (manual gradient for simplicity)
            dloss = 2 .* (x_pred .- x_next) .* dt

            # Gradient through output layer
            h = tanh.(nn.W1 * x_curr .+ nn.b1)
            grad_W2 .+= dloss * h'
            grad_b2 .+= dloss

            # Gradient through hidden layer
            dh = nn.W2' * dloss
            dtanh = dh .* (1 .- h.^2)
            grad_W1 .+= dtanh * x_curr'
            grad_b1 .+= dtanh
        end

        # Update
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
        dx = nn(x)
        x = x .+ dt .* dx
        push!(X_series, reshape(copy(x), n, d))
    end

    return X_series
end

# =============================================================================
# Main experiment
# =============================================================================

println("\n1. Generating true trajectory...")

n = 16
d = 2
T_len = 30
dt = 1.0

community_1 = 1:8
community_2 = 9:16

# Initial positions
rng = MersenneTwister(42)
X0_true = zeros(n, d)
for i in community_1
    X0_true[i, :] = [0.3, 0.7] .+ 0.1 .* randn(rng, 2)
end
for i in community_2
    X0_true[i, :] = [0.7, 0.3] .+ 0.1 .* randn(rng, 2)
end

# Clamp to B^d_+
X0_true = max.(X0_true, 0.1)
for i in 1:n
    r = norm(X0_true[i, :])
    if r > 0.9
        X0_true[i, :] .*= 0.9 / r
    end
end

# Simulate true dynamics
params = (n=n, d=d, k_in=0.15, k_out=0.08, community_1=community_1, community_2=community_2)
prob = ODEProblem(true_dynamics!, vec(X0_true), (0.0, (T_len-1) * dt), params)
sol = solve(prob, Tsit5(), saveat=dt)

X_true_series = [reshape(sol.u[t], n, d) for t in 1:T_len]

println("  Generated " * string(T_len) * " timesteps")
println("  Community 1 centroid moves: " *
        string(round.(mean(X_true_series[1][community_1, :], dims=1)[1,:], digits=3)) * " → " *
        string(round.(mean(X_true_series[end][community_1, :], dims=1)[1,:], digits=3)))

# =============================================================================
# 2. RDPG Estimation
# =============================================================================

println("\n2. RDPG estimation (K=50 samples per timestep)...")

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

println("  Embedded " * string(T_len) * " timesteps")

# =============================================================================
# 3. Train on ESTIMATED positions (not equivariant)
# =============================================================================

println("\n3. Training simple NN on ESTIMATED positions...")

nn = SimpleNN(n * d, 64, n * d; rng=MersenneTwister(123))
train_simple_nn!(nn, X_est_series, dt; epochs=500, lr=0.005)

# Simulate learned dynamics
X_rec_series = simulate_nn(nn, X_est_series[1], dt, T_len)

println("  Recovered " * string(T_len) * " timesteps")

# =============================================================================
# 4. Evaluate on rotation-invariant metrics
# =============================================================================

println("\n" * "=" ^ 70)
println("EVALUATION: Rotation-Invariant Metrics")
println("=" ^ 70)

# Compute all distances and P matrices
D_true = [pairwise_distances(X) for X in X_true_series]
D_est = [pairwise_distances(X) for X in X_est_series]
D_rec = [pairwise_distances(X) for X in X_rec_series]

P_true = [P_matrix(X) for X in X_true_series]
P_est = [P_matrix(X) for X in X_est_series]
P_rec = [P_matrix(X) for X in X_rec_series]

println("\n--- Comparison 1: True ↔ Estimated (RDPG quality) ---")
println("  t  | D corr | D RMSE | P RMSE")
println("  " * "-" ^ 40)

for t in [1, 10, 20, 30]
    d_corr = cor(upper_tri(D_true[t]), upper_tri(D_est[t]))
    d_rmse = sqrt(mean((upper_tri(D_true[t]) .- upper_tri(D_est[t])).^2))
    p_rmse = norm(P_true[t] - P_est[t]) / norm(P_true[t])

    println("  " * lpad(string(t), 2) * " | " *
            rpad(string(round(d_corr, digits=3)), 5) * "  | " *
            rpad(string(round(d_rmse, digits=4)), 6) * " | " *
            string(round(p_rmse, digits=4)))
end

println("\n--- Comparison 2: True ↔ Recovered (dynamics quality) ---")
println("  t  | D corr | D RMSE | P RMSE")
println("  " * "-" ^ 40)

for t in [1, 10, 20, 30]
    d_corr = cor(upper_tri(D_true[t]), upper_tri(D_rec[t]))
    d_rmse = sqrt(mean((upper_tri(D_true[t]) .- upper_tri(D_rec[t])).^2))
    p_rmse = norm(P_true[t] - P_rec[t]) / norm(P_true[t])

    println("  " * lpad(string(t), 2) * " | " *
            rpad(string(round(d_corr, digits=3)), 5) * "  | " *
            rpad(string(round(d_rmse, digits=4)), 6) * " | " *
            string(round(p_rmse, digits=4)))
end

println("\n--- Comparison 3: Estimated ↔ Recovered (training fit) ---")
println("  t  | Position RMSE | D corr | D RMSE")
println("  " * "-" ^ 45)

for t in [1, 10, 20, 30]
    pos_rmse = sqrt(mean((X_est_series[t] .- X_rec_series[t]).^2))
    d_corr = cor(upper_tri(D_est[t]), upper_tri(D_rec[t]))
    d_rmse = sqrt(mean((upper_tri(D_est[t]) .- upper_tri(D_rec[t])).^2))

    println("  " * lpad(string(t), 2) * " | " *
            rpad(string(round(pos_rmse, digits=5)), 13) * " | " *
            rpad(string(round(d_corr, digits=3)), 5) * "  | " *
            string(round(d_rmse, digits=4)))
end

# =============================================================================
# 5. Summary statistics
# =============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

# Average over all timesteps
avg_d_corr_true_est = mean([cor(upper_tri(D_true[t]), upper_tri(D_est[t])) for t in 1:T_len])
avg_d_corr_true_rec = mean([cor(upper_tri(D_true[t]), upper_tri(D_rec[t])) for t in 1:T_len])
avg_d_corr_est_rec = mean([cor(upper_tri(D_est[t]), upper_tri(D_rec[t])) for t in 1:T_len])

avg_p_rmse_true_est = mean([norm(P_true[t] - P_est[t]) / norm(P_true[t]) for t in 1:T_len])
avg_p_rmse_true_rec = mean([norm(P_true[t] - P_rec[t]) / norm(P_true[t]) for t in 1:T_len])
avg_p_rmse_est_rec = mean([norm(P_est[t] - P_rec[t]) / norm(P_est[t]) for t in 1:T_len])

println("\nAverage Distance Correlation:")
println("  True ↔ Estimated:  " * string(round(avg_d_corr_true_est, digits=4)))
println("  True ↔ Recovered:  " * string(round(avg_d_corr_true_rec, digits=4)))
println("  Est ↔ Recovered:   " * string(round(avg_d_corr_est_rec, digits=4)))

println("\nAverage P Reconstruction Error:")
println("  True ↔ Estimated:  " * string(round(avg_p_rmse_true_est, digits=4)))
println("  True ↔ Recovered:  " * string(round(avg_p_rmse_true_rec, digits=4)))
println("  Est ↔ Recovered:   " * string(round(avg_p_rmse_est_rec, digits=4)))

# Key question: does recovered match true as well as estimated matches true?
println("\n" * "-" ^ 50)
println("KEY QUESTION: Does training on estimated data recover true dynamics?")
println("-" ^ 50)

if avg_d_corr_true_rec > 0.9 * avg_d_corr_true_est
    println("✓ YES: Recovered distances are " *
            string(round(100 * avg_d_corr_true_rec / avg_d_corr_true_est, digits=1)) *
            "% as good as estimated")
else
    println("✗ NO: Recovered distances are only " *
            string(round(100 * avg_d_corr_true_rec / avg_d_corr_true_est, digits=1)) *
            "% as good as estimated")
end

if avg_p_rmse_true_rec < 1.5 * avg_p_rmse_true_est
    println("✓ YES: Recovered P error is only " *
            string(round(100 * avg_p_rmse_true_rec / avg_p_rmse_true_est, digits=1)) *
            "% of estimation error")
else
    println("✗ PARTIAL: Recovered P error is " *
            string(round(100 * avg_p_rmse_true_rec / avg_p_rmse_true_est, digits=1)) *
            "% of estimation error")
end

println("""

CONCLUSION:
If the above checks pass, then explicit equivariance may NOT be necessary!
Training on estimated positions with standard loss functions can achieve
our rotation-invariant goals (D, P recovery) without special architectures.

The rotation ambiguity is "absorbed" by the neural network, and the
rotation-invariant quantities we care about are still recovered correctly.
""")

# =============================================================================
# Visualization
# =============================================================================

fig = Figure(size=(1400, 500))

# Panel 1: Distance correlation over time
ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Distance Correlation with True",
           title="Distance Recovery Quality")

ts = 1:T_len
d_corr_est = [cor(upper_tri(D_true[t]), upper_tri(D_est[t])) for t in ts]
d_corr_rec = [cor(upper_tri(D_true[t]), upper_tri(D_rec[t])) for t in ts]

lines!(ax1, ts, d_corr_est, linewidth=2, label="Estimated", color=:blue)
lines!(ax1, ts, d_corr_rec, linewidth=2, label="Recovered", color=:red)
axislegend(ax1, position=:lb)

# Panel 2: P reconstruction error over time
ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="P Reconstruction Error (relative)",
           title="P Recovery Quality")

p_err_est = [norm(P_true[t] - P_est[t]) / norm(P_true[t]) for t in ts]
p_err_rec = [norm(P_true[t] - P_rec[t]) / norm(P_true[t]) for t in ts]

lines!(ax2, ts, p_err_est, linewidth=2, label="Estimated", color=:blue)
lines!(ax2, ts, p_err_rec, linewidth=2, label="Recovered", color=:red)
axislegend(ax2, position=:lt)

# Panel 3: Trajectory comparison (inter-community distance)
ax3 = Axis(fig[1, 3], xlabel="Time", ylabel="Inter-community Distance",
           title="Community Dynamics")

inter_true = [mean([D_true[t][i,j] for i in community_1 for j in community_2]) for t in ts]
inter_est = [mean([D_est[t][i,j] for i in community_1 for j in community_2]) for t in ts]
inter_rec = [mean([D_rec[t][i,j] for i in community_1 for j in community_2]) for t in ts]

lines!(ax3, ts, inter_true, linewidth=2, label="True", color=:black)
lines!(ax3, ts, inter_est, linewidth=2, label="Estimated", color=:blue, linestyle=:dash)
lines!(ax3, ts, inter_rec, linewidth=2, label="Recovered", color=:red, linestyle=:dot)
axislegend(ax3, position=:rt)

mkpath("results")
save("results/rotation_invariant_goals.pdf", fig)
println("\nSaved: results/rotation_invariant_goals.pdf")
