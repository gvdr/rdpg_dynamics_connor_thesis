"""
    Core types for RDPG temporal network dynamics.
"""

export TemporalNetworkEmbedding, without_node, target_node

"""
    TemporalNetworkEmbedding{T<:AbstractFloat}

A container for left and right SVD embeddings of a temporal network.

Stores the L and R matrices from RDPG decomposition across multiple timesteps,
with support for indexing, slicing, and linear interpolation.

# Fields
- `AL::Array{T,3}`: Left embeddings, shape (n, d, timesteps)
- `AR::Array{T,3}`: Right embeddings, shape (n, d, timesteps)
- `n::Int`: Number of nodes in the network
- `d::Int`: Embedding dimension

# Indexing
- `tne[t]` returns a Dict with :AL and :AR matrices at integer time t
- `tne[t, :AL]` returns just the left embedding at time t
- `tne[t::Float]` returns linearly interpolated embedding
- `tne[1:5]` returns a new TemporalNetworkEmbedding with subset of timesteps
"""
struct TemporalNetworkEmbedding{T<:AbstractFloat}
    AL::Array{T,3}
    AR::Array{T,3}
    n::Int
    d::Int

    function TemporalNetworkEmbedding(AL::Array{T,3}, AR::Array{T,3}, n::Int, d::Int) where T<:AbstractFloat
        size(AL) == size(AR) || throw(DimensionMismatch("AL and AR must have same shape"))
        size(AL, 1) == n || throw(DimensionMismatch("First dimension must equal n=" * string(n)))
        size(AL, 2) == d || throw(DimensionMismatch("Second dimension must equal d=" * string(d)))
        new{T}(AL, AR, n, d)
    end
end

# Integer indexing - returns Dict with both embeddings
function Base.getindex(X::TemporalNetworkEmbedding, t::Int)
    return Dict(:AL => X.AL[:, :, t], :AR => X.AR[:, :, t])
end

# Integer indexing with side selection
function Base.getindex(X::TemporalNetworkEmbedding, t::Int, side::Symbol)
    if side == :AL
        return X.AL[:, :, t]
    elseif side == :AR
        return X.AR[:, :, t]
    else
        throw(ArgumentError("side must be :AL or :AR, got :" * string(side)))
    end
end

# Float indexing with linear interpolation
@inline function Base.getindex(X::TemporalNetworkEmbedding, t::T, side::Symbol=:AL) where T<:AbstractFloat
    t_floor = Int(floor(t))
    t_ceil = Int(ceil(t))
    alpha = t - t_floor

    if t_floor == t_ceil
        return X[t_floor, side]
    end

    # Linear interpolation: (1-α)*X[floor] + α*X[ceil]
    return X[t_floor, side] .* (1 - alpha) .+ X[t_ceil, side] .* alpha
end

# Range indexing - returns new TemporalNetworkEmbedding with subset
function Base.getindex(X::TemporalNetworkEmbedding, t::UnitRange{Int64})
    return TemporalNetworkEmbedding(X.AL[:, :, t], X.AR[:, :, t], X.n, X.d)
end

# Length and lastindex
Base.length(X::TemporalNetworkEmbedding) = size(X.AL, 3)
Base.lastindex(X::TemporalNetworkEmbedding) = size(X.AL, 3)
Base.firstindex(X::TemporalNetworkEmbedding) = 1

# Iteration protocol - iterate over timesteps
function Base.iterate(X::TemporalNetworkEmbedding)
    length(X) == 0 && return nothing
    return (X[1], 2)
end

function Base.iterate(X::TemporalNetworkEmbedding, state::Int)
    state > length(X) && return nothing
    return (X[state], state + 1)
end

# Collect all timesteps as vector of dicts
function Base.collect(X::TemporalNetworkEmbedding)
    return [X[t] for t in 1:length(X)]
end

"""
    without_node(X::TemporalNetworkEmbedding, node_idx::Int) -> TemporalNetworkEmbedding

Return a new embedding with the specified node removed.

Useful for leave-one-out prediction where we predict one node's trajectory
from the others.
"""
function without_node(X::TemporalNetworkEmbedding{T}, node_idx::Int) where T
    mask = [i != node_idx for i in 1:X.n]
    return TemporalNetworkEmbedding(
        X.AL[mask, :, :],
        X.AR[mask, :, :],
        X.n - 1,
        X.d
    )
end

"""
    target_node(X::TemporalNetworkEmbedding, node_idx::Int) -> TemporalNetworkEmbedding

Return a new embedding containing only the specified node.

Useful for extracting the target node's trajectory for prediction.
"""
function target_node(X::TemporalNetworkEmbedding{T}, node_idx::Int) where T
    return TemporalNetworkEmbedding(
        X.AL[node_idx:node_idx, :, :],
        X.AR[node_idx:node_idx, :, :],
        1,
        X.d
    )
end

# Pretty printing
function Base.show(io::IO, X::TemporalNetworkEmbedding{T}) where T
    print(io, "TemporalNetworkEmbedding{" * string(T) * "}(n=" * string(X.n) * ", d=" * string(X.d) * ", timesteps=" * string(length(X)) * ")")
end

function Base.println(X::TemporalNetworkEmbedding)
    println("TemporalNetworkEmbedding")
    println("  Time Steps: " * string(length(X)))
    println("  Nodes: " * string(X.n))
    println("  Dimension: " * string(X.d))
end

"""
    TemporalNetworkEmbedding(graphs::AbstractVector{<:AbstractMatrix}, d::Int) -> TemporalNetworkEmbedding

Construct a TemporalNetworkEmbedding from a sequence of adjacency matrices.

Performs SVD embedding on each graph and aligns embeddings across time using
orthogonal Procrustes alignment.

# Arguments
- `graphs`: Vector of adjacency matrices (all same size n×n)
- `d`: Embedding dimension

# Returns
- TemporalNetworkEmbedding with aligned L and R embeddings
"""
function TemporalNetworkEmbedding(graphs::AbstractVector{T}, d::Int) where T<:AbstractMatrix
    n = size(graphs[1], 1)
    num_timesteps = length(graphs)

    # Pre-allocate output arrays
    AL = zeros(Float32, n, d, num_timesteps)
    AR = zeros(Float32, n, d, num_timesteps)

    # Embed first graph (reference for alignment)
    emb = svd_embedding(graphs[1], d)
    AL[:, :, 1] = emb.L_hat
    AR[:, :, 1] = emb.R_hat

    prev_L = emb.L_hat
    prev_R = emb.R_hat

    # Embed and align subsequent graphs
    for i in 2:num_timesteps
        emb = svd_embedding(graphs[i], d)
        L, R = emb.L_hat, emb.R_hat

        # Align to previous timestep using Procrustes
        rotation = ortho_procrustes_RM(L', prev_L')
        L = L * rotation
        R = R * rotation

        AL[:, :, i] = L
        AR[:, :, i] = R

        prev_L = L
        prev_R = R
    end

    return TemporalNetworkEmbedding(AL, AR, n, d)
end
