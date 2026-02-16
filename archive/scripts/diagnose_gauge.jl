#!/usr/bin/env -S julia --project
"""
Diagnostic: Analyze gauge mismatch vs learning quality in Example 4
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using Serialization
using LinearAlgebra
using Statistics
using Printf
using OrdinaryDiffEq
using Lux
using ComponentArrays

# Load data and model
data = deserialize("results/example4_type_kernel/data.jls")
X_true = data["X_true"]
X_est = data["X_est"]
P_true = [X * transpose(X) for X in X_true]
P_est = [X * transpose(X) for X in X_est]

model = deserialize("results/example4_type_kernel/model.jls")
ps = model["ps_learned"]
nn = model["nn"]
st = model["st"]

# Node configuration
const N_PRED = 12
const N_PREY = 15
const N_RES = 10
const N_TOTAL = N_PRED + N_PREY + N_RES
const D_EMBED = 2

# Type constants
const TYPE_P = 1
const TYPE_Y = 2
const TYPE_R = 3
const K_TYPES = 3

const NODE_TYPES = vcat(fill(TYPE_P, N_PRED), fill(TYPE_Y, N_PREY), fill(TYPE_R, N_RES))
const TYPE_PAIR_MATRIX = [3*(NODE_TYPES[i]-1) + NODE_TYPES[j] for i in 1:N_TOTAL, j in 1:N_TOTAL]
const TYPE_SCALAR = Dict(TYPE_P => 1.0f0, TYPE_Y => 0.0f0, TYPE_R => -1.0f0)
const KNOWN_SELF_RATES = Dict(TYPE_P => -0.002, TYPE_Y => -0.001, TYPE_R => 0.0)
const KERNEL_SCALE = 0.05f0

# Pre-computed arrays
const _TYPE_I_BATCH = let
    arr = zeros(Float32, N_TOTAL * (N_TOTAL - 1))
    idx = 1
    for i in 1:N_TOTAL, j in 1:N_TOTAL
        i == j && continue
        arr[idx] = TYPE_SCALAR[NODE_TYPES[i]]
        idx += 1
    end
    reshape(arr, 1, :)
end

const _TYPE_J_BATCH = let
    arr = zeros(Float32, N_TOTAL * (N_TOTAL - 1))
    idx = 1
    for i in 1:N_TOTAL, j in 1:N_TOTAL
        i == j && continue
        arr[idx] = TYPE_SCALAR[NODE_TYPES[j]]
        idx += 1
    end
    reshape(arr, 1, :)
end

const _OFFDIAG_IDX = let
    indices = CartesianIndex{2}[]
    for i in 1:N_TOTAL, j in 1:N_TOTAL
        i == j && continue
        push!(indices, CartesianIndex(i, j))
    end
    indices
end

const _TYPE_PAIR_IDX_BATCH = let
    indices = Int[]
    for i in 1:N_TOTAL, j in 1:N_TOTAL
        i == j && continue
        push!(indices, TYPE_PAIR_MATRIX[i, j])
    end
    indices
end

const _SELF_RATES_DIAG = Float32[KNOWN_SELF_RATES[NODE_TYPES[i]] for i in 1:N_TOTAL]

const _SCATTER_MAT = let
    n_pairs = N_TOTAL * (N_TOTAL - 1)
    mat = zeros(Float32, N_TOTAL, N_TOTAL, n_pairs)
    for k in 1:n_pairs
        idx = _OFFDIAG_IDX[k]
        mat[idx[1], idx[2], k] = 1.0f0
    end
    mat
end

function compute_N_nn(P, nn, ps, st)
    n = size(P, 1)
    n_pairs = n * (n - 1)
    T = eltype(P)

    P_flat = P[_OFFDIAG_IDX]
    input_batch = vcat(reshape(P_flat, 1, n_pairs), T.(_TYPE_I_BATCH), T.(_TYPE_J_BATCH))
    output_batch, _ = nn(input_batch, ps, st)
    scaled_output = output_batch .* T(KERNEL_SCALE)

    linear_idx = _TYPE_PAIR_IDX_BATCH .+ (0:n_pairs-1) .* 9
    κ_values = scaled_output[linear_idx]

    scatter_mat_T = T.(_SCATTER_MAT)
    N_offdiag = reshape(reshape(scatter_mat_T, n*n, n_pairs) * κ_values, n, n)

    row_sums = sum(N_offdiag; dims=2)
    diag_vals = T.(_SELF_RATES_DIAG) .- vec(row_sums)

    return N_offdiag + Diagonal(diag_vals)
end

function nn_dynamics(X, ps, nn, st)
    P = X * transpose(X)
    N = compute_N_nn(P, nn, ps, st)
    return N * X
end

function predict_trajectory(X0, T_end)
    n, d = size(X0)
    u0 = vec(collect(transpose(X0)))
    tspan = (0.0, Float64(T_end - 1))

    function ode(u, p, t)
        X = permutedims(reshape(u, d, n))
        dX = nn_dynamics(Float32.(X), ps, nn, st)
        return vec(permutedims(Float64.(dX)))
    end

    prob = ODEProblem{false}(ode, u0, tspan, ps)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_traj = [collect(transpose(reshape(u, d, n))) for u in sol.u]
    P_traj = [X * transpose(X) for X in X_traj]
    return P_traj, X_traj
end

function compute_P_error(P_pred, P_true_ref)
    T = min(length(P_pred), length(P_true_ref))
    [norm(P_pred[t] .- P_true_ref[t]) / norm(P_true_ref[t]) for t in 1:T]
end

println("=" ^ 60)
println("Diagnostic: Gauge Analysis for Example 4")
println("=" ^ 60)

# 1. Initial condition comparison
println("\n1. Initial condition analysis:")
p_init_err = norm(P_true[1] - P_est[1]) / norm(P_true[1])
println("   P_true[1] vs P_est[1] error: " * @sprintf("%.4f%%", 100*p_init_err))

# Check if X_est is in a rotated gauge vs X_true
# Find the best orthogonal alignment
U, _, V = svd(transpose(X_true[1]) * X_est[1])
Q_align = V * transpose(U)
X_est_aligned_1 = X_est[1] * Q_align
gauge_rotation_angle = acos(clamp(tr(Q_align) / 2, -1, 1)) * 180 / π
println("   Gauge rotation angle at t=1: " * @sprintf("%.2f°", gauge_rotation_angle))

# 2. Predict from both starting points
T_total = length(X_true)
T_train = Int(floor(0.7 * T_total))

println("\n2. UDE predictions from different starting points:")

# From X_true[1]
P_pred_true, X_pred_true = predict_trajectory(X_true[1], T_total)
err_true = compute_P_error(P_pred_true, P_true)
println("   From X_true[1]:")
println("     Training P-error:      " * @sprintf("%.2f%%", 100*mean(err_true[1:T_train])))
println("     Extrapolation P-error: " * @sprintf("%.2f%%", 100*mean(err_true[T_train+1:end])))

# From X_est[1]
P_pred_est, X_pred_est = predict_trajectory(X_est[1], T_total)
err_est = compute_P_error(P_pred_est, P_true)
println("   From X_est[1]:")
println("     Training P-error:      " * @sprintf("%.2f%%", 100*mean(err_est[1:T_train])))
println("     Extrapolation P-error: " * @sprintf("%.2f%%", 100*mean(err_est[T_train+1:end])))

# DUASE baseline
err_duase = compute_P_error(P_est, P_true)
println("   DUASE baseline:          " * @sprintf("%.2f%%", 100*mean(err_duase)))

# 3. Compare X trajectories
println("\n3. X trajectory comparison (training period):")
mse_train_est = mean([sum((X_pred_est[t] .- X_est[t]).^2) / length(X_est[t]) for t in 1:T_train])
println("   MSE(X_pred from X_est[1], X_est): " * @sprintf("%.6f", mse_train_est))

# Compare UDE predictions to X_true (with best alignment)
function aligned_mse(X_pred_list, X_ref_list, T)
    total_mse = 0.0
    for t in 1:T
        U, _, V = svd(transpose(X_ref_list[t]) * X_pred_list[t])
        Q = V * transpose(U)
        X_aligned = X_pred_list[t] * Q
        total_mse += sum((X_aligned .- X_ref_list[t]).^2) / length(X_ref_list[t])
    end
    return total_mse / T
end

mse_true_aligned = aligned_mse(X_pred_true, X_true, T_train)
println("   MSE(X_pred aligned to X_true):   " * @sprintf("%.6f", mse_true_aligned))

# 4. Analyze learned N matrix vs true N matrix
println("\n4. Learned vs True N matrix at t=1:")

# True N computation
function compute_N_true(P)
    N = zeros(N_TOTAL, N_TOTAL)
    for i in 1:N_TOTAL
        for j in 1:N_TOTAL
            if i != j
                ti, tj = NODE_TYPES[i], NODE_TYPES[j]
                p = P[i, j]
                # True kernel values (from example4 script)
                if ti == 1 && tj == 1  # P-P
                    N[i, j] = -0.004
                elseif ti == 1 && tj == 2  # P-Y (Holling)
                    N[i, j] = 0.025 * p / (1 + 2 * p)
                elseif ti == 1 && tj == 3  # P-R
                    N[i, j] = 0.0
                elseif ti == 2 && tj == 1  # Y-P
                    N[i, j] = -0.02 * p
                elseif ti == 2 && tj == 2  # Y-Y
                    N[i, j] = 0.003
                elseif ti == 2 && tj == 3  # Y-R
                    N[i, j] = 0.012 * p
                elseif ti == 3 && tj == 1  # R-P
                    N[i, j] = 0.0
                elseif ti == 3 && tj == 2  # R-Y
                    N[i, j] = -0.006 * p
                elseif ti == 3 && tj == 3  # R-R
                    N[i, j] = 0.005
                end
            end
        end
        N[i, i] = KNOWN_SELF_RATES[NODE_TYPES[i]] - sum(N[i, :])
    end
    return N
end

N_true_1 = compute_N_true(P_true[1])
N_learned_1 = Float64.(compute_N_nn(Float32.(P_true[1]), nn, ps, st))

n_error = norm(N_learned_1 .- N_true_1) / norm(N_true_1)
println("   ||N_learned - N_true|| / ||N_true||: " * @sprintf("%.4f", n_error))

# Show sample kernel comparisons
println("\n   Sample kernel values at P_ij = 0.5:")
p_sample = 0.5f0
input_sample = vcat(reshape([p_sample], 1, 1), reshape([1.0f0], 1, 1), reshape([0.0f0], 1, 1))  # P→Y
out, _ = nn(input_sample, ps, st)
κ_pred_py = out[2] * KERNEL_SCALE  # Index 2 = P→Y
κ_true_py = 0.025 * 0.5 / (1 + 2 * 0.5)
println("   κ_P→Y (p=0.5): learned=" * @sprintf("%.6f", κ_pred_py) * ", true=" * @sprintf("%.6f", κ_true_py))

println("\n" * "=" ^ 60)
println("Conclusion:")
if abs(mean(err_true) - mean(err_est)) < 0.02
    println("   Similar errors from both gauges -> gauge is not the main issue")
    if mean(err_est[1:T_train]) > mean(err_duase)
        println("   UDE P-error > DUASE even when starting from X_est[1]")
        println("   -> The NN is not learning dynamics that improve on DUASE")
    end
else
    println("   Different errors -> potential gauge sensitivity issue")
end
println("=" ^ 60)
