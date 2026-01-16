#!/usr/bin/env julia
"""Diagnostic script to check data ranges and trajectory issues."""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq

const F = Float32
n, d = 12, 2
community_1 = 1:(n ÷ 2)
community_2 = (n ÷ 2 + 1):n

# === TRUE DYNAMICS ===
center_1 = [0.35, 0.65]
center_2 = [0.65, 0.35]

function true_dynamics!(du, u, p, t)
    X = reshape(u, n, d)
    dX = zeros(n, d)
    X_bar1 = mean(X[community_1, :], dims=1)[1, :]
    X_bar2 = mean(X[community_2, :], dims=1)[1, :]
    for i in 1:n
        center = i in community_1 ? center_1 : center_2
        X_bar = i in community_1 ? X_bar1 : X_bar2
        delta = X[i, :] .- center
        dX[i, 1] = -0.3 * delta[2] - 0.1 * delta[1]
        dX[i, 2] = 0.3 * delta[1] - 0.1 * delta[2]
        dX[i, :] .+= 0.05 .* (X_bar .- X[i, :])
    end
    du .= vec(dX)
end

# Initial positions
Random.seed!(42)
X0 = zeros(n, d)
for i in community_1
    X0[i, :] = center_1 .+ 0.12 .* randn(d)
end
for i in community_2
    X0[i, :] = center_2 .+ 0.12 .* randn(d)
end
X0 = clamp.(X0, 0.15, 0.85)

# Solve true
sol = solve(ODEProblem(true_dynamics!, vec(X0), (0.0, 25.0)), Tsit5(), saveat=0:1.0:25)
X_true = [reshape(sol.u[t], n, d) for t in 1:26]

println("=" ^ 60)
println("TRUE DYNAMICS")
println("=" ^ 60)
X_true_all = vcat(X_true...)
println("x1 range: ", round.(extrema(X_true_all[:, 1]), digits=3))
println("x2 range: ", round.(extrema(X_true_all[:, 2]), digits=3))

# === RDPG ESTIMATION ===
function sample_adj(X)
    P = clamp.(X * X', 0, 1)
    A = Float64.(rand(n, n) .< P)
    for i in 1:n
        for j in 1:i-1
            A[i, j] = A[j, i]
        end
    end
    return A
end

function svd_embed(A, d)
    Fd = svd(A)
    sqrt_S = sqrt.(Fd.S[1:d])
    L = Fd.U[:, 1:d] .* sqrt_S'
    return F.(L)
end

K = 100
X_est = Vector{Matrix{F}}(undef, 26)
for t in 1:26
    A_avg = zeros(n, n)
    for k in 1:K
        A_avg .+= sample_adj(X_true[t])
    end
    A_avg ./= K
    X_est[t] = svd_embed(A_avg, d)
end

println("\n" * "=" ^ 60)
println("ESTIMATED (RDPG)")
println("=" ^ 60)
X_est_all = vcat(X_est...)
println("x1 range: ", round.(extrema(X_est_all[:, 1]), digits=3))
println("x2 range: ", round.(extrema(X_est_all[:, 2]), digits=3))
println("Negative values: ", sum(X_est_all .< 0), " / ", length(X_est_all))
println("Values > 1: ", sum(X_est_all .> 1), " / ", length(X_est_all))

# Check motion in estimated series
println("\nMotion in estimated series:")
for t in [1, 13, 26]
    c1 = mean(X_est[t][community_1, :], dims=1)[1, :]
    c2 = mean(X_est[t][community_2, :], dims=1)[1, :]
    println("  t=$t: C1=", round.(c1, digits=3), " C2=", round.(c2, digits=3),
            " dist=", round(norm(c1 - c2), digits=3))
end

# === SIMULATE WHAT NEURAL ODE WOULD DO ===
println("\n" * "=" ^ 60)
println("SIMULATING RECOVERED TRAJECTORIES")
println("=" ^ 60)

# Train positions (first 19 timesteps)
X_train = X_est[1:19]

# Check training data range
X_train_all = vcat(X_train...)
println("Training data range:")
println("  x1: ", round.(extrema(X_train_all[:, 1]), digits=3))
println("  x2: ", round.(extrema(X_train_all[:, 2]), digits=3))

# Check if training data has any motion
train_motion = 0.0
for t in 2:19
    global train_motion += sum(abs2, X_train[t] - X_train[t-1])
end
println("Training data total motion: ", round(sqrt(train_motion), digits=4))

# If motion is very small, the NN might learn to output near-zero derivatives
avg_vel = sqrt(train_motion) / (19 * n)
println("Average velocity magnitude: ", round(avg_vel, digits=6))

# Check what range a constant model would predict
println("\nIf model stays at initial position:")
u0 = X_est[1]
println("  Would be at: x1=", round.(extrema(u0[:, 1]), digits=3),
        " x2=", round.(extrema(u0[:, 2]), digits=3))
