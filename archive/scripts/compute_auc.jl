#!/usr/bin/env julia
using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Statistics
using Random
using RDPGDynamics

# Simple AUC calculation via Mann-Whitney U
function compute_auc(y_true::Vector{Float64}, y_pred::Vector{Float64})
    n_pos = sum(y_true)
    n_neg = length(y_true) - n_pos

    if n_pos == 0 || n_neg == 0
        return 0.5
    end

    # Efficient AUC via sorting
    order = sortperm(y_pred, rev=true)
    y_sorted = y_true[order]

    # Count pairs where positive is ranked higher than negative
    auc = 0.0
    tp_so_far = 0.0
    for i in 1:length(y_sorted)
        if y_sorted[i] == 1.0
            tp_so_far += 1
        else
            auc += tp_so_far
        end
    end

    return auc / (n_pos * n_neg)
end

# Generate same test data
Random.seed!(42)
n, d, m = 30, 2, 20

X_true = zeros(m, n, d)
X_true[1, :, :] = 0.4 .+ 0.15 * randn(n, d)
X_true[1, :, :] = clamp.(X_true[1, :, :], 0.15, 0.85)
for t in 2:m
    X_true[t, :, :] = X_true[t-1, :, :] + 0.015 * randn(n, d)
    X_true[t, :, :] = clamp.(X_true[t, :, :], 0.1, 0.9)
end

P_true = [clamp.(X_true[t, :, :] * X_true[t, :, :]', 0.0, 1.0) for t in 1:m]

# Generate binary adjacencies
A_binary = Vector{Matrix{Float64}}(undef, m)
for t in 1:m
    A_binary[t] = Float64.(rand(n, n) .< P_true[t])
    A_binary[t] = (A_binary[t] + A_binary[t]') / 2
    A_binary[t][diagind(A_binary[t])] .= 0.0
    A_binary[t] = Float64.(A_binary[t] .> 0.5)
end

# Extract lower triangular
function get_lower_tri(matrices)
    m_len = length(matrices)
    n_nodes = size(matrices[1], 1)
    y = Float64[]
    for t in 1:m_len
        for i in 2:n_nodes
            for j in 1:(i-1)
                push!(y, matrices[t][i, j])
            end
        end
    end
    return y
end

y_obs = get_lower_tri(A_binary)
p_true_vec = get_lower_tri(P_true)

println("=== AUC Results (their metric) ===")
println("AUC measures ranking: can predicted probabilities distinguish edges from non-edges?")
println()

# Oracle: P_true vs observed A
auc_oracle = compute_auc(y_obs, p_true_vec)
println("P_true (oracle):        AUC = " * string(round(auc_oracle, digits=4)))

# DUASE
function duase_embed(A_input, d_emb)
    m_loc = length(A_input)
    Unfolded = hcat(A_input...)
    U, S, V = svd(Unfolded)
    G = U[:, 1:d_emb]

    X_duase = zeros(m_loc, size(A_input[1], 1), d_emb)
    for t in 1:m_loc
        Qt = G' * A_input[t] * G
        Qt_sym = (Qt + Qt') / 2
        eig = eigen(Symmetric(Qt_sym))
        sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
        X_duase[t, :, :] = G * sqrt_Q
    end
    return X_duase
end

X_duase = duase_embed(A_binary, d)
P_duase = [clamp.(X_duase[t, :, :] * X_duase[t, :, :]', 0.0, 1.0) for t in 1:m]
p_duase_vec = get_lower_tri(P_duase)
auc_duase = compute_auc(y_obs, p_duase_vec)
println("DUASE:                  AUC = " * string(round(auc_duase, digits=4)))

# SVD + Procrustes
X_svd = zeros(m, n, d)
for t in 1:m
    emb = svd_embedding(A_binary[t], d)
    X_svd[t, :, :] = emb.L_hat
end
for t in 2:m
    R = ortho_procrustes_RM(X_svd[t, :, :]', X_svd[t-1, :, :]')
    X_svd[t, :, :] = X_svd[t, :, :] * R
end

P_svd = [clamp.(X_svd[t, :, :] * X_svd[t, :, :]', 0.0, 1.0) for t in 1:m]
p_svd_vec = get_lower_tri(P_svd)
auc_svd = compute_auc(y_obs, p_svd_vec)
println("SVD+Procrustes:         AUC = " * string(round(auc_svd, digits=4)))

# Now run GB-DASE and compute its AUC
println("\n--- Running GB-DASE Gibbs (500+500 iterations) ---")

include("gbdase_gibbs.jl")

result_gbdase = gbdase_gibbs(A_binary, d;
                             rw_order=2,
                             n_burnin=500,
                             n_samples=500,
                             scale="auto",
                             verbose=false)

P_gbdase = [clamp.(result_gbdase.X_mean[t, :, :] * result_gbdase.X_mean[t, :, :]', 0.0, 1.0) for t in 1:m]
p_gbdase_vec = get_lower_tri(P_gbdase)
auc_gbdase = compute_auc(y_obs, p_gbdase_vec)
println("GB-DASE Gibbs:          AUC = " * string(round(auc_gbdase, digits=4)))

println()
println("=== Summary ===")
println("Method              AUC      (vs oracle " * string(round(auc_oracle, digits=4)) * ")")
println("-" ^ 50)
println("P_true (oracle):    " * string(round(auc_oracle, digits=4)))
println("DUASE:              " * string(round(auc_duase, digits=4)) * "    (" * string(round(100*(auc_oracle - auc_duase)/auc_oracle, digits=2)) * "% below oracle)")
println("GB-DASE Gibbs:      " * string(round(auc_gbdase, digits=4)) * "    (" * string(round(100*(auc_oracle - auc_gbdase)/auc_oracle, digits=2)) * "% below oracle)")
println("SVD+Procrustes:     " * string(round(auc_svd, digits=4)) * "    (" * string(round(100*(auc_oracle - auc_svd)/auc_oracle, digits=2)) * "% below oracle)")
println()
println("Random guessing: AUC = 0.5")
println("Perfect ranking: AUC = 1.0")
