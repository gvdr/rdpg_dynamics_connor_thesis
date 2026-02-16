module Alg4Utils

using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq

export normalize_rows!, generate_clustered_X0
export build_linear_symmetric_N, linear_dynamics!, polynomial_dynamics!, message_passing_dynamics!
export simulate_series, sample_adjacency_average, ase_embedding, ase_series
export spectral_gap, skew_gauge_error, series_spectral_gaps, series_skew_errors
export probability_violation_stats, series_probability_violation_stats

# =============================================================================
# Utilities: Initialization
# =============================================================================

"""
    normalize_rows!(X; max_norm=1.0, min_pos=0.0)

Clamp rows to be non-negative and rescale rows with norm > max_norm.
"""
function normalize_rows!(X::AbstractMatrix; max_norm::Float64=1.0, min_pos::Float64=0.0)
    n = size(X, 1)
    for i in 1:n
        X[i, :] = max.(X[i, :], min_pos)
        row_norm = norm(X[i, :])
        if row_norm > max_norm
            X[i, :] .= X[i, :] .* (max_norm / row_norm)
        end
    end
    return X
end

"""
    generate_clustered_X0(n, d; centers, noise_std=0.05, rng=Random.default_rng())

Generate clustered initial positions with optional normalization.
"""
function generate_clustered_X0(n::Int, d::Int;
                               centers::Vector{<:AbstractVector},
                               noise_std::Float64=0.05,
                               rng::AbstractRNG=Random.default_rng())
    k = length(centers)
    X0 = zeros(Float64, n, d)
    nodes_per = fill(n ÷ k, k)
    for i in 1:(n % k)
        nodes_per[i] += 1
    end

    idx = 1
    for (c_idx, center) in enumerate(centers)
        for _ in 1:nodes_per[c_idx]
            noise = noise_std .* randn(rng, d)
            X0[idx, :] = center .+ noise
            idx += 1
        end
    end

    normalize_rows!(X0; max_norm=1.0, min_pos=0.0)
    return X0
end

# =============================================================================
# Utilities: Dynamics Constructors
# =============================================================================

"""
    build_linear_symmetric_N(n; a=0.08, b=0.02, gamma=0.12, communities=nothing)

Construct a symmetric N with block-community structure.
If `communities` is nothing, uses two equal blocks.
"""
function build_linear_symmetric_N(n::Int;
                                  a::Float64=0.08,
                                  b::Float64=0.02,
                                  gamma::Float64=0.12,
                                  communities::Union{Nothing, Vector{Vector{Int}}}=nothing)
    if isnothing(communities)
        split = n ÷ 2
        communities = [collect(1:split), collect(split+1:n)]
    end

    N = zeros(Float64, n, n)
    for comm in communities
        for i in comm, j in comm
            N[i, j] += a
        end
    end
    for i in 1:length(communities)
        for j in 1:length(communities)
            if i != j
                for u in communities[i], v in communities[j]
                    N[u, v] += b
                end
            end
        end
    end
    N .-= gamma .* I(n)
    N = Symmetric(N)
    return Matrix(N)
end

"""
    linear_dynamics!(du, u, p, t; n, d, N)

In-place linear symmetric dynamics: Ẋ = N X
"""
function linear_dynamics!(du::AbstractVector, u::AbstractVector, p, t; n::Int, d::Int, N::AbstractMatrix)
    X = reshape(u, n, d)
    dX = N * X
    du .= vec(dX)
    return nothing
end

"""
    polynomial_dynamics!(du, u, p, t; n, d, alpha)

In-place polynomial dynamics: Ẋ = (sum_k alpha_k P^k) X, P = X X'
"""
function polynomial_dynamics!(du::AbstractVector, u::AbstractVector, p, t; n::Int, d::Int, alpha::AbstractVector)
    X = reshape(u, n, d)
    P = X * X'
    N = alpha[1] .* Matrix{eltype(X)}(I, n, n)
    Pk = P
    for k in 2:length(alpha)
        N .+= alpha[k] .* Pk
        Pk = Pk * P
    end
    dX = N * X
    du .= vec(dX)
    return nothing
end

"""
    message_passing_dynamics!(du, u, p, t; n, d, beta)

In-place message-passing dynamics: Ẋ_i = sum_j P_ij g(x_i, x_j)
with g(x_i, x_j) = beta1 (x_j - x_i) + beta2 x_j.
"""
function message_passing_dynamics!(du::AbstractVector, u::AbstractVector, p, t; n::Int, d::Int, beta::AbstractVector)
    X = reshape(u, n, d)
    P = X * X'
    PX = P * X
    degrees = sum(P, dims=2)
    dX = beta[1] .* (PX .- degrees .* X) .+ beta[2] .* PX
    du .= vec(dX)
    return nothing
end

# =============================================================================
# Utilities: Simulation and Observation
# =============================================================================

"""
    simulate_series(dynamics!, X0; T=40, dt=1.0)

Simulate an ODE and return a vector of X(t) matrices.
"""
function simulate_series(dynamics!, X0::AbstractMatrix; T::Int=40, dt::Float64=1.0)
    n, d = size(X0)
    tspan = (0.0, dt * (T - 1))
    tsteps = range(tspan[1], tspan[2]; length=T)
    prob = ODEProblem((du, u, p, t) -> dynamics!(du, u, p, t), vec(X0), tspan)
    sol = solve(prob, Tsit5(); saveat=tsteps, abstol=1e-6, reltol=1e-6)
    return [reshape(sol.u[t], n, d) for t in 1:length(sol.u)]
end

"""
    sample_adjacency_average(X_series; K=20, rng=Random.default_rng())

For each t, sample K adjacency matrices and return the average.
"""
function sample_adjacency_average(X_series::Vector{<:AbstractMatrix};
                                  K::Int=20,
                                  rng::AbstractRNG=Random.default_rng())
    T = length(X_series)
    n = size(X_series[1], 1)
    A_avg = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        P = X_series[t] * X_series[t]'
        P = clamp.(P, 0.0, 1.0)
        A_sum = zeros(Float64, n, n)
        for _ in 1:K
            A = rand(rng, n, n) .< P
            A = Float64.(A)
            A_sum .+= (A .+ A') ./ 2
        end
        A_avg[t] = A_sum ./ K
    end
    return A_avg
end

"""
    ase_embedding(A, d)

Compute ASE with SVD. Returns n × d matrix.
"""
function ase_embedding(A::AbstractMatrix, d::Int)
    F = svd(A)
    U = F.U[:, 1:d]
    S = F.S[1:d]
    return U .* sqrt.(S)'
end

"""
    ase_series(A_series, d)

Embed a series of averaged adjacency matrices.
"""
function ase_series(A_series::Vector{<:AbstractMatrix}, d::Int)
    return [ase_embedding(A_series[t], d) for t in 1:length(A_series)]
end

# =============================================================================
# Diagnostics
# =============================================================================

"""
    spectral_gap(X)

Return smallest non-zero singular value of X.
"""
function spectral_gap(X::AbstractMatrix)
    s = svdvals(X)
    s_sorted = sort(s; rev=true)
    if length(s_sorted) == 0
        return 0.0
    end
    return s_sorted[end]
end

"""
    probability_violation_stats(X)

Compute basic validity diagnostics for P = X X':
- neg_count: number of entries < 0
- above_one_count: number of entries > 1
- min_P, max_P: extrema of P
"""
function probability_violation_stats(X::AbstractMatrix)
    P = X * X'
    min_P = minimum(P)
    max_P = maximum(P)
    neg_count = count(<(0.0), P)
    above_one_count = count(>(1.0), P)
    return (neg_count=neg_count, above_one_count=above_one_count, min_P=min_P, max_P=max_P)
end

"""
    series_probability_violation_stats(X_series)

Return a vector of validity diagnostics over time.
"""
function series_probability_violation_stats(X_series::Vector{<:AbstractMatrix})
    return [probability_violation_stats(X) for X in X_series]
end

"""
    skew_gauge_error(X_t, X_tp1; dt=1.0)

Compute ||Skew(X_t' * dX)||_F where dX ≈ (X_{t+1}-X_t)/dt.
"""
function skew_gauge_error(X_t::AbstractMatrix, X_tp1::AbstractMatrix; dt::Float64=1.0)
    dX = (X_tp1 .- X_t) ./ dt
    S = X_t' * dX
    skew = S .- S'
    return norm(skew)
end

"""
    series_spectral_gaps(X_series)

Return vector of spectral gaps over time.
"""
function series_spectral_gaps(X_series::Vector{<:AbstractMatrix})
    return [spectral_gap(X) for X in X_series]
end

"""
    series_skew_errors(X_series; dt=1.0)

Return vector of skew gauge errors over time.
"""
function series_skew_errors(X_series::Vector{<:AbstractMatrix}; dt::Float64=1.0)
    T = length(X_series)
    if T < 2
        return Float64[]
    end
    return [skew_gauge_error(X_series[t], X_series[t+1]; dt=dt) for t in 1:(T-1)]
end

end
