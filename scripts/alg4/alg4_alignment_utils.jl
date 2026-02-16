module Alg4AlignmentUtils

using LinearAlgebra
using Statistics

export procrustes_rotation, apply_gauge, apply_gauge_series
export build_relative_gauges, reconstruct_gauges_from_rel
export linear_M_step, linear_R_step
export block_indicators, linear_block_M, linear_block_params_step
export polynomial_alpha_step, polynomial_R_step
export message_passing_features, message_passing_beta_step, message_passing_R_step
export compute_P_series, compute_P_powers
export alignment_residuals, mean_alignment_residual

# =============================================================================
# Gauge utilities
# =============================================================================

"""
    procrustes_rotation(A, B)

Compute orthogonal Procrustes rotation Ω that minimizes ||AΩ - B||_F.
Returns Ω.
"""
function procrustes_rotation(A::AbstractMatrix, B::AbstractMatrix)
    U, _, V = svd(A' * B)
    return U * V'
end

"""
    apply_gauge(X, Q)

Apply gauge Q to embedding X, returning X * Q.
"""
apply_gauge(X::AbstractMatrix, Q::AbstractMatrix) = X * Q

"""
    apply_gauge_series(X_series, Q_series)

Apply gauge sequence to embedding series. Returns vector of X_t * Q_t.
"""
function apply_gauge_series(X_series::Vector{<:AbstractMatrix},
                            Q_series::Vector{<:AbstractMatrix})
    T = length(X_series)
    return [X_series[t] * Q_series[t] for t in 1:T]
end

"""
    build_relative_gauges(Q_series)

Compute relative gauges R_t = Q_t' * Q_{t+1}.
"""
function build_relative_gauges(Q_series::Vector{<:AbstractMatrix})
    T = length(Q_series)
    return [Q_series[t]' * Q_series[t+1] for t in 1:(T-1)]
end

"""
    reconstruct_gauges_from_rel(R_series; Q0=I)

Reconstruct Q_t from R_t with Q_{t+1} = Q_t * R_t.
"""
function reconstruct_gauges_from_rel(R_series::Vector{<:AbstractMatrix};
                                     Q0::AbstractMatrix)
    Tm1 = length(R_series)
    Q_series = Vector{Matrix{Float64}}(undef, Tm1 + 1)
    Q_series[1] = Matrix(Q0)
    for t in 1:Tm1
        Q_series[t+1] = Q_series[t] * R_series[t]
    end
    return Q_series
end

# =============================================================================
# Linear symmetric family: M-step and R-step
# =============================================================================

"""
    linear_M_step(X_series, R_series; l2=0.0)

Compute symmetric M that minimizes Σ ||X_{t+1} R_t' - M X_t||_F^2 with L2 regularization.
Returns symmetric M.
"""
function linear_M_step(X_series::Vector{<:AbstractMatrix},
                       R_series::Vector{<:AbstractMatrix};
                       l2::Float64=0.0)
    Tm1 = length(R_series)
    Y = hcat([X_series[t+1] * R_series[t]' for t in 1:Tm1]...)
    Z = hcat([X_series[t] for t in 1:Tm1]...)
    M_star = Y * Z' * inv(Z * Z' + (1e-8 + l2) * I(size(Z, 1)))
    return (M_star + M_star') ./ 2
end

"""
    linear_R_step(X_series, M)

Compute R_t via Procrustes for each t:
R_t = argmin ||X_{t+1} R_t' - M X_t||_F.
"""
function linear_R_step(X_series::Vector{<:AbstractMatrix},
                       M::AbstractMatrix)
    T = length(X_series)
    R_series = Vector{Matrix{Float64}}(undef, T-1)
    for t in 1:(T-1)
        A = X_series[t+1]
        B = M * X_series[t]
        R_series[t] = procrustes_rotation(A, B)'
    end
    return R_series
end

# =============================================================================
# Linear structured block family: params-step and utilities
# =============================================================================

"""
    block_indicators(n, communities)

Return (W_intra, W_inter) indicator matrices for within- and between-community pairs.
"""
function block_indicators(n::Int, communities::Vector{Vector{Int}})
    comm_id = zeros(Int, n)
    for (k, comm) in enumerate(communities)
        for i in comm
            comm_id[i] = k
        end
    end
    if any(comm_id .== 0)
        error("All nodes must belong to a community")
    end

    W_intra = zeros(Float64, n, n)
    W_inter = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if comm_id[i] == comm_id[j]
            W_intra[i, j] = 1.0
        else
            W_inter[i, j] = 1.0
        end
    end
    return W_intra, W_inter
end

"""
    linear_block_M(n, communities, a, b, gamma; dt=1.0)

Build discrete-time M = I + dt * (a * W_intra + b * W_inter - gamma * I).
"""
function linear_block_M(n::Int, communities::Vector{Vector{Int}},
                        a::Float64, b::Float64, gamma::Float64;
                        dt::Float64=1.0)
    W_intra, W_inter = block_indicators(n, communities)
    M = Matrix{Float64}(I, n, n)
    M .+= dt .* (a .* W_intra .+ b .* W_inter)
    M .-= dt .* gamma .* Matrix{Float64}(I, n, n)
    return M
end

"""
    linear_block_params_step(X_series, R_series, communities; dt=1.0, l2=0.0)

Fit (a, b, gamma) in M = I + dt * (a * W_intra + b * W_inter - gamma * I) via ridge regression.
Returns a 3-vector [a, b, gamma].
"""
function linear_block_params_step(X_series::Vector{<:AbstractMatrix},
                                  R_series::Vector{<:AbstractMatrix},
                                  communities::Vector{Vector{Int}};
                                  dt::Float64=1.0,
                                  l2::Float64=0.0)
    n = size(X_series[1], 1)
    W_intra, W_inter = block_indicators(n, communities)
    return linear_block_params_step(X_series, R_series, W_intra, W_inter; dt=dt, l2=l2)
end

"""
    linear_block_params_step(X_series, R_series, W_intra, W_inter; dt=1.0, l2=0.0)

Fit (a, b, gamma) given precomputed indicators, using (Y_t - X_t)/dt.
"""
function linear_block_params_step(X_series::Vector{<:AbstractMatrix},
                                  R_series::Vector{<:AbstractMatrix},
                                  W_intra::AbstractMatrix,
                                  W_inter::AbstractMatrix;
                                  dt::Float64=1.0,
                                  l2::Float64=0.0)
    Tm1 = length(R_series)
    G = zeros(Float64, 3, 3)
    b = zeros(Float64, 3)

    for t in 1:Tm1
        X_t = X_series[t]
        Y_t = X_series[t+1] * R_series[t]'
        V_t = (Y_t .- X_t) ./ dt

        Phi1 = W_intra * X_t
        Phi2 = W_inter * X_t
        Phi3 = -X_t

        Phis = (Phi1, Phi2, Phi3)
        for i in 1:3
            b[i] += sum(Phis[i] .* V_t)
            for j in 1:3
                G[i, j] += sum(Phis[i] .* Phis[j])
            end
        end
    end

    params = (G + (1e-8 + l2) * I(3)) \ b
    return params
end

# =============================================================================
# Polynomial family: alpha-step and R-step
# =============================================================================

"""
    compute_P_series(X_series)

Return vector of P_t = X_t * X_t'.
"""
function compute_P_series(X_series::Vector{<:AbstractMatrix})
    return [X * X' for X in X_series]
end

"""
    compute_P_powers(P_series, K)

Return vector of vectors: P_powers[t][k] = (P_t)^(k-1), k=1..K+1.
k=1 corresponds to I.
"""
function compute_P_powers(P_series::Vector{<:AbstractMatrix}, K::Int)
    T = length(P_series)
    P_powers = Vector{Vector{Matrix{Float64}}}(undef, T)
    for t in 1:T
        P = P_series[t]
        n = size(P, 1)
        powers = Vector{Matrix{Float64}}(undef, K+1)
        powers[1] = Matrix{Float64}(I, n, n)
        if K >= 1
            powers[2] = P
        end
        for k in 3:(K+1)
            powers[k] = powers[k-1] * P
        end
        P_powers[t] = powers
    end
    return P_powers
end

"""
    polynomial_alpha_step(X_series, R_series, K; dt=1.0, l2=0.0)

Solve for α using gauge-free P-dynamics with L2 regularization:

dot(P) = 2 * Σ_k α_k P^(k+1)
P_{t+1} - P_t ≈ dt * Σ_k α_k (P_t^(k+1) + P_{t+1}^(k+1))

Uses only P_t = X_t * X_t', ignoring R_series.
"""
function polynomial_alpha_step(X_series::Vector{<:AbstractMatrix},
                               R_series::Vector{<:AbstractMatrix},
                               K::Int; dt::Float64=1.0, l2::Float64=0.0)
    Tm1 = length(X_series) - 1
    P_series = compute_P_series(X_series)
    P_powers = compute_P_powers(P_series, K + 1)

    # Build Gram matrix and RHS
    G = zeros(Float64, K+1, K+1)
    b = zeros(Float64, K+1)

    for t in 1:Tm1
        dP = P_series[t+1] .- P_series[t]
        for k in 1:(K+1)
            Psi_k = dt .* (P_powers[t][k+1] .+ P_powers[t+1][k+1])
            b[k] += sum(Psi_k .* dP)
            for j in 1:(K+1)
                Psi_j = dt .* (P_powers[t][j+1] .+ P_powers[t+1][j+1])
                G[k, j] += sum(Psi_k .* Psi_j)
            end
        end
    end

    alpha = (G + (1e-8 + l2) * I(size(G, 1))) \ b
    return alpha
end

"""
    polynomial_R_step(X_series, alpha, K; dt=1.0)

Compute R_t via Procrustes with M(P_t) = I + dt Σ α_k P^k.
"""
function polynomial_R_step(X_series::Vector{<:AbstractMatrix},
                           alpha::AbstractVector,
                           K::Int; dt::Float64=1.0)
    T = length(X_series)
    R_series = Vector{Matrix{Float64}}(undef, T-1)

    P_series = compute_P_series(X_series)
    P_powers = compute_P_powers(P_series, K)

    for t in 1:(T-1)
        X_t = X_series[t]
        X_tp1 = X_series[t+1]
        n = size(X_t, 1)

        M = Matrix{Float64}(I, n, n)
        for k in 1:(K+1)
            M .+= dt .* alpha[k] .* P_powers[t][k]
        end

        R_series[t] = procrustes_rotation(X_tp1, M * X_t)'
    end

    return R_series
end

# =============================================================================
# Message-passing family: beta-step and R-step
# =============================================================================

"""
    message_passing_features(X)

Return F1 = P*X - degrees .* X, F2 = P*X, and P, where P = X*X'.
"""
function message_passing_features(X::AbstractMatrix)
    P = X * X'
    PX = P * X
    degrees = sum(P, dims=2)
    F1 = PX .- degrees .* X
    F2 = PX
    return F1, F2, P
end

"""
    message_passing_beta_step(X_series, R_series; dt=1.0, l2=0.0)

Solve for beta via linear regression with L2 regularization:
(Y_t - X_t)/dt ≈ beta1 * F1 + beta2 * F2
"""
function message_passing_beta_step(X_series::Vector{<:AbstractMatrix},
                                   R_series::Vector{<:AbstractMatrix};
                                   dt::Float64=1.0, l2::Float64=0.0)
    Tm1 = length(R_series)
    G = zeros(Float64, 2, 2)
    b = zeros(Float64, 2)

    for t in 1:Tm1
        X_t = X_series[t]
        Y_t = X_series[t+1] * R_series[t]'
        V_t = (Y_t .- X_t) ./ dt

        F1, F2, _ = message_passing_features(X_t)

        G[1, 1] += sum(F1 .* F1)
        G[1, 2] += sum(F1 .* F2)
        G[2, 2] += sum(F2 .* F2)

        b[1] += sum(F1 .* V_t)
        b[2] += sum(F2 .* V_t)
    end

    G[2, 1] = G[1, 2]
    beta = (G + (1e-8 + l2) * I(2)) \ b
    return beta
end

"""
    message_passing_R_step(X_series, beta; dt=1.0)

Compute R_t via Procrustes with forward Euler prediction.
"""
function message_passing_R_step(X_series::Vector{<:AbstractMatrix},
                                beta::AbstractVector;
                                dt::Float64=1.0)
    T = length(X_series)
    R_series = Vector{Matrix{Float64}}(undef, T-1)

    for t in 1:(T-1)
        X_t = X_series[t]
        X_tp1 = X_series[t+1]

        F1, F2, _ = message_passing_features(X_t)
        X_pred = X_t .+ dt .* (beta[1] .* F1 .+ beta[2] .* F2)

        R_series[t] = procrustes_rotation(X_tp1, X_pred)'
    end

    return R_series
end

# =============================================================================
# Diagnostics
# =============================================================================

"""
    alignment_residuals(X_series, R_series, f_pred)

Compute residuals ||X_{t+1} R_t' - f_pred(X_t)||_F for t=1..T-1.
"""
function alignment_residuals(X_series::Vector{<:AbstractMatrix},
                             R_series::Vector{<:AbstractMatrix},
                             f_pred::Function)
    Tm1 = length(R_series)
    res = zeros(Float64, Tm1)
    for t in 1:Tm1
        Y_t = X_series[t+1] * R_series[t]'
        res[t] = norm(Y_t .- f_pred(X_series[t]))
    end
    return res
end

"""
    mean_alignment_residual(residuals)

Return mean residual.
"""
mean_alignment_residual(residuals::AbstractVector) = mean(residuals)

end
