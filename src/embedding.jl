"""
    Embedding functions for Random Dot Product Graphs (RDPG).

    Provides SVD-based embedding and Procrustes alignment for temporal networks.
    These functions were originally in DotProductGraphs.jl and are inlined here
    for reproducibility.

    IMPORTANT: SVD identifies L, R only up to orthogonal rotation. When the true
    latent space is B^d_+ (non-negative unit ball), we need to:
    1. Find the rotation Q that maps SVD output back into B^d_+
    2. Project to handle numerical issues
"""

using LinearAlgebra
using Arpack
using Random

export ortho_procrustes_RM, svd_embedding, truncated_svd
export project_to_Bd_plus, in_Bd_plus, align_to_Bd_plus, align_to_reference
export project_embedding_to_Bd_plus, svd_embedding_Bd_plus
export embed_temporal_network_Bd_plus
export embed_temporal_network_smoothed
export sample_adjacency, sample_adjacency_repeated
export folded_embedding, embed_temporal_with_folding
export duase_embedding, omni_embedding
export embed_temporal_duase, embed_temporal_duase_raw, embed_temporal_omni
export align_series_to_Bd_plus, find_global_orthogonal_to_Bd_plus, find_global_rotation_to_Bd_plus

"""
    ortho_procrustes_RM(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Compute the orthogonal Procrustes rotation matrix Ω that minimizes ‖ΩA - B‖_F.

Returns the rotation matrix Ω = VU' from the SVD of AB'.

# Arguments
- `A`: Source matrix to be rotated
- `B`: Target matrix

# Returns
- Orthogonal matrix Ω such that ΩA ≈ B in the least-squares sense
"""
function ortho_procrustes_RM(A::AbstractMatrix, B::AbstractMatrix)
    U, _, V = svd(A * B')
    return V * U'
end

"""
    truncated_svd(A::AbstractMatrix, d::Int) -> Tuple{Matrix, Vector, Matrix}

Compute truncated SVD of matrix A with d singular values using Arpack.

Uses a deterministic starting vector for reproducibility.

# Arguments
- `A`: Matrix to decompose
- `d`: Number of singular values/vectors to compute

# Returns
- Tuple (U, Σ, V) where A ≈ U * Diagonal(Σ) * V'
"""
function truncated_svd(A::AbstractMatrix, d::Int)
    n = minimum(size(A))
    # Deterministic starting vector for reproducibility
    v0 = [Float64(i % 7) for i in 1:n]
    result = Arpack.svds(A; nsv=d, v0=v0)[1]
    return result.U, result.S, result.V
end

"""
    svd_embedding(A::AbstractMatrix, d::Int) -> NamedTuple{(:L_hat, :R_hat)}

Compute the RDPG embedding of adjacency matrix A with embedding dimension d.

For a graph with adjacency matrix A, computes L̂ = UΣ^{1/2} and R̂ = VΣ^{1/2}
such that A ≈ L̂R̂'.

# Arguments
- `A`: Adjacency matrix (n × n)
- `d`: Embedding dimension

# Returns
- Named tuple with L_hat and R_hat matrices (both n × d)

# Example
```julia
A = [0 1 1; 1 0 1; 1 1 0]
emb = svd_embedding(A, 2)
# emb.L_hat * emb.R_hat' ≈ A
```
"""
function svd_embedding(A::AbstractMatrix, d::Int)
    U, Sigma, V = truncated_svd(A, d)
    sqrt_Sigma = sqrt.(Sigma)
    # Use broadcasting to avoid allocating full diagonal matrix
    L_hat = U .* sqrt_Sigma'
    R_hat = V .* sqrt_Sigma'
    return (L_hat=L_hat, R_hat=R_hat)
end

"""
    svd_embedding(A::AbstractMatrix, svd_engine::Function, d::Int) -> NamedTuple

Compute RDPG embedding using a custom SVD engine.

This variant allows using different SVD implementations (e.g., full SVD for small matrices).

# Arguments
- `A`: Adjacency matrix
- `svd_engine`: Function (A, d) -> (U, Σ, V)
- `d`: Embedding dimension
"""
function svd_embedding(A::AbstractMatrix, svd_engine::Function, d::Int)
    U, Sigma, V = svd_engine(A, d)
    sqrt_Sigma = sqrt.(Sigma)
    L_hat = U .* sqrt_Sigma'
    R_hat = V .* sqrt_Sigma'
    return (L_hat=L_hat, R_hat=R_hat)
end

# =============================================================================
# B^d_+ (Non-negative Unit Ball) Utilities
# =============================================================================
#
# For valid RDPG, we need L_i · L_j ∈ [0, 1] for all pairs.
# The canonical solution is to constrain L to B^d_+ = {x ∈ R^d : x ≥ 0, ||x|| ≤ 1}
#
# Challenge: SVD gives L up to arbitrary orthogonal rotation Q.
# Even if true positions are in B^d_+, SVD output might be in a rotated space.
# =============================================================================

"""
    in_Bd_plus(x::AbstractVector; tol=1e-10) -> Bool

Check if point lies in the non-negative unit ball B^d_+.
A point is in B^d_+ if all coordinates are ≥ 0 and ||x|| ≤ 1.
"""
function in_Bd_plus(x::AbstractVector; tol::Float64=1e-10)
    return all(>=(−tol), x) && norm(x) <= 1.0 + tol
end

"""
    project_to_Bd_plus(x::AbstractVector) -> Vector

Project a point onto B^d_+ (non-negative unit ball).
Two-step projection: (1) clamp to non-negative, (2) scale if norm > 1.
"""
function project_to_Bd_plus(x::AbstractVector)
    # Step 1: Clamp to non-negative orthant
    x_pos = max.(x, 0.0)
    # Step 2: Scale to unit ball if needed
    n = norm(x_pos)
    return n > 1.0 ? x_pos ./ n : x_pos
end

"""
    project_embedding_to_Bd_plus(L::AbstractMatrix) -> Matrix

Project each row of embedding matrix L onto B^d_+.
Returns a new matrix where each row is in B^d_+.
"""
function project_embedding_to_Bd_plus(L::AbstractMatrix)
    n = size(L, 1)
    L_proj = similar(L)
    for i in 1:n
        L_proj[i, :] = project_to_Bd_plus(L[i, :])
    end
    return L_proj
end

"""
    align_to_reference(L_hat::AbstractMatrix, L_ref::AbstractMatrix) -> Matrix

Align estimated embedding L_hat to reference L_ref using Procrustes.

This is used when we KNOW the ground truth positions (e.g., synthetic data)
and want to find the rotation Q that maps L_hat → L_ref.

Returns L_aligned = L_hat * Q where Q minimizes ||L_hat * Q - L_ref||_F.
"""
function align_to_reference(L_hat::AbstractMatrix, L_ref::AbstractMatrix)
    # Procrustes: find Q minimizing ||L_hat * Q - L_ref||
    Q = ortho_procrustes_RM(L_hat', L_ref')
    return L_hat * Q
end

"""
    align_to_Bd_plus(L::AbstractMatrix; max_iters::Int=100) -> Tuple{Matrix, Matrix}

Find orthogonal rotation Q that maps L into B^d_+ as much as possible.

This is for cases where we DON'T know the ground truth but believe the
true positions should be in B^d_+.

Strategy: Iterative sign flipping and rotation to minimize negative entries.
This is a heuristic - finding the optimal Q is NP-hard in general.

Returns (L_aligned, Q) where L_aligned = L * Q.
"""
function align_to_Bd_plus(L::AbstractMatrix; max_iters::Int=100)
    n, d = size(L)
    Q = Matrix{Float64}(I, d, d)
    L_current = copy(L)

    for iter in 1:max_iters
        improved = false

        # Try flipping signs of each column
        for j in 1:d
            neg_count = sum(L_current[:, j] .< 0)
            if neg_count > n / 2
                # Flip this column
                L_current[:, j] .*= -1
                Q[:, j] .*= -1
                improved = true
            end
        end

        # Try swapping columns if it helps (for d=2)
        if d == 2
            # Check if swapping would reduce negatives
            neg_orig = sum(L_current .< 0)
            L_swap = L_current[:, [2, 1]]
            neg_swap = sum(L_swap .< 0)
            if neg_swap < neg_orig
                L_current = L_swap
                Q = Q[:, [2, 1]]
                improved = true
            end
        end

        if !improved
            break
        end
    end

    # Final projection to ensure B^d_+ constraints
    L_proj = project_embedding_to_Bd_plus(L_current)

    return L_proj, Q
end

"""
    find_global_orthogonal_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                                       max_iters::Int=100,
                                       n_angles::Int=36) -> Tuple{Matrix, Matrix}

Find an orthogonal transformation Q that maps all matrices in L_series toward B^d_+.

The transformation is decomposed as Q = S * R where:
- S is a sign matrix (reflection/sign flips)
- R is a proper rotation (det = +1)

# Algorithm
1. Sign flips: Flip column signs to maximize positive entries
2. Rotation: For d=2, optimize angle θ to minimize B^d_+ violations
   For d>2, apply pairwise Givens rotations

# Arguments
- `L_series`: Vector of embedding matrices (all same size n × d)
- `max_iters`: Maximum iterations for sign flipping
- `n_angles`: Number of angles to try for rotation optimization (d=2)

# Returns
- Tuple (Q, S) where Q = S * R is the full transformation and S is the sign matrix
"""
function find_global_orthogonal_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                                            max_iters::Int=100,
                                            n_angles::Int=36)
    # Stack all matrices
    L_stacked = vcat(L_series...)
    n_total, d = size(L_stacked)

    # ===== Step 1: Sign flips (reflections) =====
    S = Matrix{Float64}(I, d, d)  # Sign matrix
    L_current = copy(L_stacked)

    for iter in 1:max_iters
        improved = false

        for j in 1:d
            neg_count = sum(L_current[:, j] .< 0)
            if neg_count > n_total / 2
                L_current[:, j] .*= -1
                S[j, j] *= -1
                improved = true
            end
        end

        if !improved
            break
        end
    end

    # ===== Step 2: Rotation optimization =====
    R = Matrix{Float64}(I, d, d)  # Rotation matrix

    if d == 2
        # For d=2, optimize single angle θ
        R = _optimize_rotation_2d(L_current, n_angles)
        L_current = L_current * R
    elseif d > 2
        # For d>2: joint optimization (d≤4) or iterative Givens with multiple passes (d>4)
        R = _optimize_rotation_nd(L_current, n_angles)
        L_current = L_current * R
    end

    # Full orthogonal transformation
    Q = S * R

    return Q, S
end

"""
Optimize rotation angle for d=2 to minimize B^d_+ violations.
"""
function _optimize_rotation_2d(L::AbstractMatrix, n_angles::Int)
    best_angle = 0.0
    best_violation = _compute_Bd_plus_violation(L)

    for i in 0:n_angles-1
        θ = 2π * i / n_angles
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        L_rot = L * R
        violation = _compute_Bd_plus_violation(L_rot)

        if violation < best_violation
            best_violation = violation
            best_angle = θ
        end
    end

    # Fine-tune around best angle
    for δ in [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]
        θ = best_angle + δ
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        L_rot = L * R
        violation = _compute_Bd_plus_violation(L_rot)

        if violation < best_violation
            best_violation = violation
            best_angle = θ
        end
    end

    return [cos(best_angle) -sin(best_angle); sin(best_angle) cos(best_angle)]
end

"""
Optimize rotation for d>2 using either joint optimization (small d) or iterative Givens (large d).
"""
function _optimize_rotation_nd(L::AbstractMatrix, n_angles::Int; max_passes::Int=10, tol::Float64=1e-6)
    n, d = size(L)

    if d <= 4
        # Joint optimization for small d - feasible to search over all angles
        return _optimize_rotation_joint(L, n_angles)
    else
        # Iterative Givens for large d
        return _optimize_rotation_givens_iterative(L, n_angles; max_passes=max_passes, tol=tol)
    end
end

"""
Joint optimization over all d(d-1)/2 rotation angles for small d (≤4).
Uses grid search + refinement.
"""
function _optimize_rotation_joint(L::AbstractMatrix, n_angles::Int)
    n, d = size(L)
    n_params = d * (d - 1) ÷ 2  # Number of rotation parameters

    # For d=3: 3 angles, for d=4: 6 angles
    # Use coarser grid for higher dimensions
    angles_per_dim = d == 3 ? n_angles : max(8, n_angles ÷ 2)

    best_angles = zeros(n_params)
    best_violation = _compute_Bd_plus_violation(L)
    best_R = Matrix{Float64}(I, d, d)

    if d == 3
        # 3 angles: (1,2), (1,3), (2,3)
        for i1 in 0:angles_per_dim-1
            θ1 = 2π * i1 / angles_per_dim
            for i2 in 0:angles_per_dim-1
                θ2 = 2π * i2 / angles_per_dim
                for i3 in 0:angles_per_dim-1
                    θ3 = 2π * i3 / angles_per_dim

                    R = _build_rotation_from_angles(d, [θ1, θ2, θ3])
                    violation = _compute_Bd_plus_violation(L * R)

                    if violation < best_violation
                        best_violation = violation
                        best_angles = [θ1, θ2, θ3]
                        best_R = R
                    end
                end
            end
        end
    elseif d == 4
        # 6 angles - use sparser grid + local refinement
        coarse_angles = max(6, angles_per_dim ÷ 2)

        # Coarse search
        for i1 in 0:coarse_angles-1, i2 in 0:coarse_angles-1, i3 in 0:coarse_angles-1
            θ1, θ2, θ3 = 2π .* [i1, i2, i3] ./ coarse_angles
            for i4 in 0:coarse_angles-1, i5 in 0:coarse_angles-1, i6 in 0:coarse_angles-1
                θ4, θ5, θ6 = 2π .* [i4, i5, i6] ./ coarse_angles

                angles = [θ1, θ2, θ3, θ4, θ5, θ6]
                R = _build_rotation_from_angles(d, angles)
                violation = _compute_Bd_plus_violation(L * R)

                if violation < best_violation
                    best_violation = violation
                    best_angles = angles
                    best_R = R
                end
            end
        end
    end

    # Local refinement around best angles
    best_R = _refine_rotation(L, d, best_angles, best_violation)

    return best_R
end

"""
Build rotation matrix from d(d-1)/2 angles using Givens rotations.
"""
function _build_rotation_from_angles(d::Int, angles::Vector{Float64})
    R = Matrix{Float64}(I, d, d)
    idx = 1

    for i in 1:d-1
        for j in i+1:d
            θ = angles[idx]
            G = Matrix{Float64}(I, d, d)
            G[i, i] = cos(θ)
            G[j, j] = cos(θ)
            G[i, j] = -sin(θ)
            G[j, i] = sin(θ)
            R = R * G
            idx += 1
        end
    end

    return R
end

"""
Local refinement of rotation angles.
"""
function _refine_rotation(L::AbstractMatrix, d::Int, angles::Vector{Float64}, current_violation::Float64)
    best_angles = copy(angles)
    best_violation = current_violation

    # Try small perturbations
    deltas = [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]

    for pass in 1:3
        improved = false
        for (idx, _) in enumerate(angles)
            for δ in deltas
                test_angles = copy(best_angles)
                test_angles[idx] += δ
                R = _build_rotation_from_angles(d, test_angles)
                violation = _compute_Bd_plus_violation(L * R)

                if violation < best_violation
                    best_violation = violation
                    best_angles = test_angles
                    improved = true
                end
            end
        end
        if !improved
            break
        end
    end

    return _build_rotation_from_angles(d, best_angles)
end

"""
Iterative Givens rotations with multiple passes until convergence (for d>4).
"""
function _optimize_rotation_givens_iterative(L::AbstractMatrix, n_angles::Int;
                                              max_passes::Int=10, tol::Float64=1e-6)
    n, d = size(L)
    R_total = Matrix{Float64}(I, d, d)
    L_current = copy(L)

    prev_violation = _compute_Bd_plus_violation(L_current)

    for pass in 1:max_passes
        # Single pass over all pairs
        for i in 1:d-1
            for j in i+1:d
                # Find best rotation for this (i,j) plane
                best_angle = 0.0
                best_local_violation = _compute_Bd_plus_violation(L_current)

                for k in 0:n_angles-1
                    θ = 2π * k / n_angles
                    G = _make_givens(d, i, j, θ)
                    L_rot = L_current * G
                    violation = _compute_Bd_plus_violation(L_rot)

                    if violation < best_local_violation
                        best_local_violation = violation
                        best_angle = θ
                    end
                end

                # Fine-tune
                for δ in [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]
                    θ = best_angle + δ
                    G = _make_givens(d, i, j, θ)
                    L_rot = L_current * G
                    violation = _compute_Bd_plus_violation(L_rot)

                    if violation < best_local_violation
                        best_local_violation = violation
                        best_angle = θ
                    end
                end

                # Apply best rotation for this pair
                G = _make_givens(d, i, j, best_angle)
                R_total = R_total * G
                L_current = L_current * G
            end
        end

        # Check convergence
        current_violation = _compute_Bd_plus_violation(L_current)
        if abs(prev_violation - current_violation) < tol
            break
        end
        prev_violation = current_violation
    end

    return R_total
end

"""
Create a Givens rotation matrix for plane (i,j) with angle θ.
"""
function _make_givens(d::Int, i::Int, j::Int, θ::Float64)
    G = Matrix{Float64}(I, d, d)
    G[i, i] = cos(θ)
    G[j, j] = cos(θ)
    G[i, j] = -sin(θ)
    G[j, i] = sin(θ)
    return G
end

"""
Compute violation score for B^d_+ constraints.
Lower is better. Measures: sum of |negative entries| + sum of (norm - 1) for points outside ball.
"""
function _compute_Bd_plus_violation(L::AbstractMatrix)
    n = size(L, 1)
    violation = 0.0

    # Penalty for negative entries
    violation += sum(abs.(min.(L, 0.0)))

    # Penalty for points outside unit ball
    for i in 1:n
        norm_i = norm(L[i, :])
        if norm_i > 1.0
            violation += (norm_i - 1.0)
        end
    end

    return violation
end

# Keep old name as alias for backwards compatibility
find_global_rotation_to_Bd_plus(L_series::Vector{<:AbstractMatrix}; kwargs...) =
    find_global_orthogonal_to_Bd_plus(L_series; kwargs...)[1]

"""
    align_series_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                            method::Symbol=:project,
                            Q::Union{Nothing, AbstractMatrix}=nothing) -> NamedTuple

Align a temporal series of embeddings to B^d_+ using a SINGLE global rotation.

# Arguments
- `L_series`: Vector of embedding matrices (all same size n × d)
- `method`: How to handle points outside B^d_+ after rotation:
  - `:project` (default): Per-point projection (clamp negatives, scale if ||x|| > 1)
  - `:rescale`: Global rescaling (find s such that all ||s*x|| ≤ 1)
  - `:none`: Only apply rotation, no projection or rescaling
- `Q`: Optional pre-computed rotation matrix. If not provided, computes via `find_global_rotation_to_Bd_plus`

# Returns
Named tuple with:
- `L_aligned`: Vector of aligned matrices
- `Q`: The rotation matrix used
- `scale`: Scale factor (1.0 if method != :rescale)
- `stats`: Dict with violation statistics before/after

# Example
```julia
result = align_series_to_Bd_plus(L_series; method=:rescale)
L_aligned = result.L_aligned
println("Scale factor: ", result.scale)
println("Violations before: ", result.stats[:neg_before])
```
"""
function align_series_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                                  method::Symbol=:project,
                                  Q::Union{Nothing, AbstractMatrix}=nothing)
    T = length(L_series)
    n, d = size(L_series[1])

    # Step 1: Find global rotation if not provided
    if isnothing(Q)
        Q = find_global_rotation_to_Bd_plus(L_series)
    end

    # Step 2: Apply rotation to all matrices
    L_rotated = [L_series[t] * Q for t in 1:T]

    # Compute statistics before further processing
    L_stacked = vcat(L_rotated...)
    neg_count_before = sum(L_stacked .< 0)
    max_norm_before = maximum([norm(L_stacked[i, :]) for i in 1:size(L_stacked, 1)])
    outside_ball_before = sum([norm(L_stacked[i, :]) > 1.0 for i in 1:size(L_stacked, 1)])

    # Step 3: Handle points outside B^d_+
    scale = 1.0
    L_aligned = Vector{Matrix{Float64}}(undef, T)

    if method == :rescale
        # Global rescaling: find s such that all points satisfy constraints
        # First clamp negatives, then find global scale
        L_clamped = [max.(L_rotated[t], 0.0) for t in 1:T]
        L_clamped_stacked = vcat(L_clamped...)
        max_norm = maximum([norm(L_clamped_stacked[i, :]) for i in 1:size(L_clamped_stacked, 1)])

        if max_norm > 1.0
            scale = 1.0 / max_norm
            for t in 1:T
                L_aligned[t] = L_clamped[t] .* scale
            end
        else
            L_aligned = L_clamped
        end

    elseif method == :project
        # Per-point projection
        for t in 1:T
            L_aligned[t] = project_embedding_to_Bd_plus(L_rotated[t])
        end

    elseif method == :none
        # Just rotation, no projection
        L_aligned = L_rotated

    else
        error("Unknown method: " * string(method) * ". Use :project, :rescale, or :none")
    end

    # Compute statistics after processing
    L_aligned_stacked = vcat(L_aligned...)
    neg_count_after = sum(L_aligned_stacked .< 0)
    max_norm_after = maximum([norm(L_aligned_stacked[i, :]) for i in 1:size(L_aligned_stacked, 1)])
    outside_ball_after = sum([norm(L_aligned_stacked[i, :]) > 1.0 + 1e-10 for i in 1:size(L_aligned_stacked, 1)])

    stats = Dict(
        :neg_before => neg_count_before,
        :neg_after => neg_count_after,
        :max_norm_before => max_norm_before,
        :max_norm_after => max_norm_after,
        :outside_ball_before => outside_ball_before,
        :outside_ball_after => outside_ball_after,
        :total_points => n * T
    )

    return (L_aligned=L_aligned, Q=Q, scale=scale, stats=stats)
end

"""
    svd_embedding_Bd_plus(A::AbstractMatrix, d::Int;
                          reference::Union{Nothing, AbstractMatrix}=nothing) -> NamedTuple

Compute RDPG embedding with B^d_+ constraint handling.

If `reference` is provided (ground truth), uses Procrustes alignment.
Otherwise, uses heuristic alignment to B^d_+.

Returns (L_hat, R_hat) where both are projected to ensure B^d_+ constraints.
"""
function svd_embedding_Bd_plus(A::AbstractMatrix, d::Int;
                                reference::Union{Nothing, AbstractMatrix}=nothing)
    # Standard SVD embedding
    emb = svd_embedding(A, d)
    L_hat, R_hat = emb.L_hat, emb.R_hat

    if !isnothing(reference)
        # Align to known ground truth
        L_hat = align_to_reference(L_hat, reference)
        R_hat = align_to_reference(R_hat, reference)
    else
        # Heuristic alignment
        L_hat, _ = align_to_Bd_plus(L_hat)
        R_hat, _ = align_to_Bd_plus(R_hat)
    end

    # Final projection for numerical safety
    L_hat = project_embedding_to_Bd_plus(L_hat)
    R_hat = project_embedding_to_Bd_plus(R_hat)

    return (L_hat=L_hat, R_hat=R_hat)
end

"""
    embed_temporal_network_Bd_plus(A_series::Vector{<:AbstractMatrix}, d::Int;
                                    L_true::Union{Nothing, Vector}=nothing) -> Vector{Matrix}

Embed a sequence of adjacency matrices with B^d_+ constraints.

Two-step process:
1. Embed each graph and align temporally (Procrustes to previous timestep)
2. Find a SINGLE rotation Q that maps the entire trajectory into B^d_+

If `L_true` is provided, aligns to ground truth instead.

# Arguments
- `A_series`: Vector of adjacency matrices
- `d`: Embedding dimension
- `L_true`: Optional ground truth L matrices for alignment

# Returns
- Vector of L matrices, all in B^d_+, temporally aligned
"""
function embed_temporal_network_Bd_plus(A_series::Vector{<:AbstractMatrix}, d::Int;
                                         L_true::Union{Nothing, Vector}=nothing)
    T = length(A_series)

    # Step 1: SVD embed each graph
    L_raw = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        emb = svd_embedding(A_series[t], d)
        L_raw[t] = emb.L_hat
    end

    # Step 2: Temporal alignment (Procrustes chain)
    L_aligned = Vector{Matrix{Float64}}(undef, T)
    L_aligned[1] = L_raw[1]

    for t in 2:T
        # Align to previous timestep
        Q = ortho_procrustes_RM(L_raw[t]', L_aligned[t-1]')
        L_aligned[t] = L_raw[t] * Q
    end

    # Step 3: Find global rotation into B^d_+
    if !isnothing(L_true)
        # Align entire trajectory to ground truth at t=1
        Q_global = ortho_procrustes_RM(L_aligned[1]', L_true[1]')
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    else
        # Find Q that minimizes B^d_+ violations across ALL timesteps
        # Stack all L matrices and find best global alignment
        L_stacked = vcat(L_aligned...)
        _, Q_global = align_to_Bd_plus(L_stacked)

        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    end

    # Step 4: Project for numerical safety
    for t in 1:T
        L_aligned[t] = project_embedding_to_Bd_plus(L_aligned[t])
    end

    return L_aligned
end

# =============================================================================
# Sliding Window Embedding (for noisy temporal networks)
# =============================================================================
#
# Key insight: Binary adjacency matrices have high noise. Averaging adjacent
# time slices dramatically improves the eigenvalue gap and embedding recovery.
# This is inspired by graspologic and the UASE literature.
# =============================================================================

"""
    embed_temporal_network_smoothed(A_series::Vector{<:AbstractMatrix}, d::Int;
                                     window::Int=5,
                                     L_true::Union{Nothing, Vector}=nothing) -> Vector{Matrix}

Embed a temporal network using sliding window averaging to reduce noise.

This method averages adjacent adjacency matrices before embedding, which:
1. Improves the eigenvalue gap (better signal-to-noise ratio)
2. Reduces variance in the embedding estimates
3. Provides smoother trajectories suitable for Neural ODE training

# Arguments
- `A_series`: Vector of adjacency matrices (one per timestep)
- `d`: Embedding dimension
- `window`: Size of sliding window for averaging (default: 5)
- `L_true`: Optional ground truth for Procrustes alignment (oracle mode)

# Returns
- Vector of L matrices (n × d), temporally aligned

# Example
```julia
# Embed with window averaging
L_series = embed_temporal_network_smoothed(A_series, 2; window=5)
```

# Notes
- Window=1 is equivalent to standard per-timestep embedding
- Larger windows give smoother trajectories but may blur fast dynamics
- Recommended: window=5 for typical temporal networks with n~50-100
"""
function embed_temporal_network_smoothed(A_series::Vector{<:AbstractMatrix}, d::Int;
                                          window::Int=5,
                                          L_true::Union{Nothing, Vector}=nothing)
    T = length(A_series)
    n = size(A_series[1], 1)

    # Step 1: Compute smoothed adjacency matrices via sliding window
    A_smoothed = Vector{Matrix{Float64}}(undef, T)
    half_window = window ÷ 2

    for t in 1:T
        t_start = max(1, t - half_window)
        t_end = min(T, t + half_window)

        A_avg = zeros(n, n)
        for s in t_start:t_end
            A_avg .+= A_series[s]
        end
        A_avg ./= (t_end - t_start + 1)
        A_smoothed[t] = A_avg
    end

    # Step 2: Embed each smoothed matrix
    L_raw = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        # Use full SVD for small matrices, truncated for large
        if n <= 100
            F = svd(A_smoothed[t])
            L_raw[t] = F.U[:, 1:d] .* sqrt.(F.S[1:d])'
        else
            emb = svd_embedding(A_smoothed[t], d)
            L_raw[t] = emb.L_hat
        end
    end

    # Step 3: Temporal alignment via Procrustes chain
    L_aligned = Vector{Matrix{Float64}}(undef, T)
    L_aligned[1] = L_raw[1]

    for t in 2:T
        Q = ortho_procrustes_RM(L_raw[t]', L_aligned[t-1]')
        L_aligned[t] = L_raw[t] * Q
    end

    # Step 4: Global alignment (to ground truth or B^d_+)
    if !isnothing(L_true)
        # Oracle mode: align to ground truth at t=1
        Q_global = ortho_procrustes_RM(L_aligned[1]', L_true[1]')
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    else
        # Heuristic: find global Q that maps into B^d_+
        L_stacked = vcat(L_aligned...)
        _, Q_global = align_to_Bd_plus(L_stacked)
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    end

    # Step 5: Project onto B^d_+ for numerical safety
    for t in 1:T
        L_aligned[t] = project_embedding_to_Bd_plus(L_aligned[t])
    end

    return L_aligned
end

# =============================================================================
# Adjacency Matrix Generation (RDPG Observation Model)
# =============================================================================
#
# In RDPG, we observe adjacency matrices A sampled from P = XX' (or XY' for directed).
# These functions generate the observation layer that was missing from the examples.
# =============================================================================

"""
    sample_adjacency(X::AbstractMatrix; symmetric::Bool=true, self_loops::Bool=true, rng=Random.GLOBAL_RNG) -> Matrix

Sample a single adjacency matrix from RDPG latent positions.

For undirected graphs, P(i,j) = X_i · X_j, and edges are Bernoulli(P(i,j)).

# Arguments
- `X`: Latent position matrix (n × d), rows are node positions
- `symmetric`: If true, generate undirected graph (default: true)
- `self_loops`: If true, include diagonal entries (default: true)
- `rng`: Random number generator for reproducibility

# Returns
- Binary adjacency matrix A (n × n)

# Notes
Self-loops (diagonal entries) are CRITICAL for RDPG estimation. The diagonal of P = XX'
contains P_ii = ||X_i||², which is essential for recovering X via SVD. Setting
self_loops=false will cause the averaged adjacency to NOT converge to P.

# Example
```julia
X = rand(10, 2)  # 10 nodes in 2D
A = sample_adjacency(X)
```
"""
function sample_adjacency(X::AbstractMatrix; symmetric::Bool=true, self_loops::Bool=true, rng=Random.GLOBAL_RNG)
    n = size(X, 1)

    # Compute probability matrix P = XX'
    P = X * X'

    # Clamp to valid probabilities [0, 1]
    P = clamp.(P, 0.0, 1.0)

    # Sample edges via Bernoulli for ALL entries (including diagonal)
    A = rand(rng, n, n) .< P

    # Convert to Float64
    A = Float64.(A)

    if symmetric
        # Make symmetric: keep upper triangle + diagonal, mirror to lower
        # A_ij = A_ji for i != j, diagonal stays as sampled
        for i in 1:n
            for j in 1:i-1
                A[i, j] = A[j, i]  # Mirror upper to lower
            end
        end
    end

    if !self_loops
        # Zero diagonal if self-loops not wanted (NOT recommended for RDPG estimation)
        for i in 1:n
            A[i, i] = 0.0
        end
    end

    return A
end

"""
    sample_adjacency_repeated(X::AbstractMatrix, K::Int;
                               symmetric::Bool=true, self_loops::Bool=true, rng=Random.GLOBAL_RNG) -> Matrix

Generate K repeated adjacency samples and average them.

This reduces noise in the observation by averaging multiple Bernoulli samples.
The result is a weighted adjacency matrix with values in [0, 1].

# Arguments
- `X`: Latent position matrix (n × d)
- `K`: Number of repeated samples
- `symmetric`: If true, generate undirected graphs
- `self_loops`: If true, include diagonal entries (default: true)
- `rng`: Random number generator

# Returns
- Averaged adjacency matrix (n × n) with entries in [0, 1]

# Notes
As K → ∞, the average converges to the true probability matrix P = XX'.
For K=10, we get good noise reduction while maintaining realistic network structure.
Self-loops are included by default (see sample_adjacency docstring for rationale).
"""
function sample_adjacency_repeated(X::AbstractMatrix, K::Int;
                                    symmetric::Bool=true, self_loops::Bool=true, rng=Random.GLOBAL_RNG)
    n = size(X, 1)
    A_sum = zeros(n, n)

    for k in 1:K
        A_sum .+= sample_adjacency(X; symmetric=symmetric, self_loops=self_loops, rng=rng)
    end

    return A_sum ./ K
end

# =============================================================================
# Folded Embedding (Horizontal Stacking)
# =============================================================================
#
# Key insight from graspologic: stack multiple adjacency matrices horizontally
# and perform a single SVD. This embeds all graphs in a common latent space.
# =============================================================================

"""
    folded_embedding(A_list::Vector{<:AbstractMatrix}, d::Int) -> Matrix

Embed multiple adjacency matrices using horizontal folding.

Stacks matrices horizontally: [A_1 | A_2 | ... | A_m] and performs SVD.
All graphs are embedded in a common latent space.

# Arguments
- `A_list`: Vector of adjacency matrices (all same size n × n)
- `d`: Embedding dimension

# Returns
- Embedding matrix L (n × d) in the common latent space

# Notes
This is faster than Omnibus embedding but may not preserve structure as well.
Best used with averaged/smoothed adjacency matrices.
"""
function folded_embedding(A_list::Vector{<:AbstractMatrix}, d::Int)
    n = size(A_list[1], 1)
    m = length(A_list)

    # Horizontally stack: A_folded is n × (n*m)
    A_folded = hcat(A_list...)

    # SVD on folded matrix
    if n <= 100
        # Full SVD for small matrices
        F = svd(A_folded)
        L = F.U[:, 1:d] .* sqrt.(F.S[1:d])'
    else
        # Truncated SVD for large matrices
        U, S, _ = truncated_svd(A_folded, d)
        L = U .* sqrt.(S)'
    end

    return L
end

"""
    embed_temporal_with_folding(X_series::Vector{<:AbstractMatrix}, d::Int;
                                 K::Int=10, W::Int=3,
                                 L_true::Union{Nothing, Vector}=nothing,
                                 rng=Random.GLOBAL_RNG) -> Vector{Matrix}

Full RDPG pipeline: latent positions → adjacency samples → folded embedding → alignment.

This is the PROPER way to learn dynamics from RDPG - going through the observation model.

# Pipeline
1. For each time t, generate K adjacency samples from X(t)
2. Apply sliding window of size W (average adjacent time points)
3. Stack windowed samples horizontally and embed via folded SVD
4. Align temporally using Procrustes chain
5. Align globally to B^d_+ (or to ground truth if provided)

# Arguments
- `X_series`: Vector of true latent position matrices (one per timestep)
- `d`: Embedding dimension
- `K`: Number of repeated samples per timestep (default: 10)
- `W`: Window size for temporal smoothing (default: 3)
- `L_true`: Optional ground truth for oracle alignment
- `rng`: Random number generator

# Returns
- Vector of estimated L matrices, temporally aligned

# Example
```julia
# Generate true dynamics
X_series = [evolve_dynamics(X0, t) for t in 1:T]

# Embed through RDPG observation model
L_hat_series = embed_temporal_with_folding(X_series, 2; K=10, W=3)

# Now train Neural ODE on L_hat_series (not X_series!)
```
"""
function embed_temporal_with_folding(X_series::Vector{<:AbstractMatrix}, d::Int;
                                      K::Int=10, W::Int=3,
                                      L_true::Union{Nothing, Vector}=nothing,
                                      rng=Random.GLOBAL_RNG)
    T = length(X_series)
    n = size(X_series[1], 1)
    half_W = W ÷ 2

    # Step 1: Generate K adjacency samples at each time point and AVERAGE them
    A_avg_series = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        A_sum = zeros(n, n)
        for _ in 1:K
            A_sum .+= sample_adjacency(X_series[t]; rng=rng)
        end
        A_avg_series[t] = A_sum ./ K
    end

    # Step 2: For each time t, use temporal window folding
    # - FOLD across time window to get stable common basis U
    # - PROJECT the center time point onto U
    # - EIGENDECOMPOSE to get time-specific embedding
    L_raw = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        # Define window bounds
        t_start = max(1, t - half_W)
        t_end = min(T, t + half_W)
        window_size = t_end - t_start + 1

        # FOLD: horizontally stack averaged adjacency matrices in window
        A_folded = hcat([A_avg_series[s] for s in t_start:t_end]...)

        # SVD on folded matrix to get common basis U
        F = svd(A_folded)
        U = F.U[:, 1:d]

        # PROJECT: compute Λ = U' A_avg(t) U for the CENTER time point
        Lambda = U' * A_avg_series[t] * U

        # EIGENDECOMPOSE Λ to get time-specific embedding
        # Λ should be approximately diagonal with eigenvalues = squared singular values of X(t)
        E = eigen(Symmetric(Lambda))
        # X̂(t) = U · (eigenvectors · √eigenvalues)
        sqrt_eigs = sqrt.(max.(E.values, 0.0))
        L_raw[t] = U * (E.vectors .* sqrt_eigs')
    end

    # Step 3: Temporal alignment via Procrustes chain
    L_aligned = Vector{Matrix{Float64}}(undef, T)
    L_aligned[1] = L_raw[1]

    for t in 2:T
        Q = ortho_procrustes_RM(L_raw[t]', L_aligned[t-1]')
        L_aligned[t] = L_raw[t] * Q
    end

    # Step 4: Global alignment
    if !isnothing(L_true)
        # Oracle mode: align to ground truth at t=1
        Q_global = ortho_procrustes_RM(L_aligned[1]', L_true[1]')
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    else
        # Heuristic: find global Q that maps into B^d_+
        L_stacked = vcat(L_aligned...)
        _, Q_global = align_to_Bd_plus(L_stacked)
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    end

    return L_aligned
end

# =============================================================================
# DUASE: Doubly Unfolded Adjacency Spectral Embedding
# =============================================================================
#
# Reference: Baum, Sanna Passino & Gandy (2024), arXiv:2410.09810
#
# Model: A(t) ≈ G Q(t) G' where:
#   - G is shared node embedding (n × d) across all times
#   - Q(t) is time-specific score matrix (d × d)
#
# For single-layer temporal networks (our case), DUASE gives:
#   X̂(t) = G · √Q(t) where √Q(t) is the matrix square root
# =============================================================================

"""
    duase_embedding(A_series::Vector{<:AbstractMatrix}, d::Int;
                    window::Union{Nothing, Int}=nothing) -> Tuple{Matrix, Vector{Matrix}}

DUASE embedding for temporal networks.

Implements the Doubly Unfolded Adjacency Spectral Embedding from Baum et al. (2024).

# Algorithm
1. Unfold: Stack horizontally `A = [A(1) | A(2) | ... | A(T)]` (n × nT)
2. SVD: `A = U Σ V'`
3. Extract shared basis: `G = U[:, 1:d]` (NOT scaled by √Σ)
4. For each time t:
   - Project: `Q(t) = G' A(t) G`
   - Matrix square root: `X̂(t) = G · √Q(t)`

# Arguments
- `A_series`: Vector of adjacency matrices (one per time point)
- `d`: Embedding dimension
- `window`: If provided, use sliding window of this size instead of all T (default: nothing = full)

# Returns
- Tuple (G, X_series) where:
  - G is the shared basis (n × d)
  - X_series is vector of per-time embeddings (each n × d)

# Notes
- DUASE separates shared structure (G) from temporal variation (Q(t))
- Best for dynamics where node positions evolve but community structure is stable
- For K repeated samples per time, average first then apply DUASE
"""
function duase_embedding(A_series::Vector{<:AbstractMatrix}, d::Int;
                          window::Union{Nothing, Int}=nothing)
    T = length(A_series)
    n = size(A_series[1], 1)

    if isnothing(window) || window >= T
        # Full unfolding: use all T time points
        return _duase_full(A_series, d)
    else
        # Windowed DUASE: sliding window of size W
        return _duase_windowed(A_series, d, window)
    end
end

"""
Internal: Full DUASE using all time points.
"""
function _duase_full(A_series::Vector{<:AbstractMatrix}, d::Int)
    T = length(A_series)
    n = size(A_series[1], 1)

    # Step 1: Horizontally unfold all adjacency matrices
    A_unfolded = hcat(A_series...)  # n × (n*T)

    # Step 2: SVD to get shared basis G
    if n <= 100
        F = svd(A_unfolded)
        G = F.U[:, 1:d]  # Shared basis (NOT scaled by √Σ)
    else
        U, _, _ = truncated_svd(A_unfolded, d)
        G = U
    end

    # Step 3: For each time point, compute time-specific embedding
    X_series = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        # Project: Q(t) = G' A(t) G
        Q_t = G' * A_series[t] * G

        # Make symmetric (numerical stability)
        Q_t = (Q_t + Q_t') / 2

        # Matrix square root via eigendecomposition
        E = eigen(Symmetric(Q_t))
        sqrt_eigs = sqrt.(max.(E.values, 0.0))  # Clamp negative eigenvalues
        sqrt_Q = E.vectors * Diagonal(sqrt_eigs) * E.vectors'

        # Embedding: X̂(t) = G · √Q(t)
        X_series[t] = G * sqrt_Q
    end

    return G, X_series
end

"""
Internal: Windowed DUASE with sliding window of size W.
"""
function _duase_windowed(A_series::Vector{<:AbstractMatrix}, d::Int, W::Int)
    T = length(A_series)
    n = size(A_series[1], 1)
    half_W = W ÷ 2

    X_series = Vector{Matrix{Float64}}(undef, T)
    G_series = Vector{Matrix{Float64}}(undef, T)  # G varies per window

    for t in 1:T
        # Define window bounds
        t_start = max(1, t - half_W)
        t_end = min(T, t + half_W)

        # Unfold within window
        A_window = hcat([A_series[s] for s in t_start:t_end]...)

        # SVD for window-specific G
        if n <= 100
            F = svd(A_window)
            G_t = F.U[:, 1:d]
        else
            U, _, _ = truncated_svd(A_window, d)
            G_t = U
        end

        G_series[t] = G_t

        # Project and compute embedding for center time point
        Q_t = G_t' * A_series[t] * G_t
        Q_t = (Q_t + Q_t') / 2

        E = eigen(Symmetric(Q_t))
        sqrt_eigs = sqrt.(max.(E.values, 0.0))
        sqrt_Q = E.vectors * Diagonal(sqrt_eigs) * E.vectors'

        X_series[t] = G_t * sqrt_Q
    end

    # Return the middle G as representative (or could return all)
    return G_series[T ÷ 2 + 1], X_series
end

"""
    embed_temporal_duase(X_series::Vector{<:AbstractMatrix}, d::Int;
                          K::Int=10, window::Union{Nothing, Int}=nothing,
                          L_true::Union{Nothing, Vector}=nothing,
                          rng=Random.GLOBAL_RNG) -> Vector{Matrix}

Full RDPG pipeline using DUASE: latent positions → adjacency samples → DUASE embedding → alignment.

# Pipeline
1. For each time t, generate K adjacency samples from X(t) and average
2. Apply DUASE (optionally with windowing)
3. Align temporally using Procrustes chain
4. Align globally to B^d_+ (or to ground truth if provided)

# Arguments
- `X_series`: Vector of true latent position matrices (one per timestep)
- `d`: Embedding dimension
- `K`: Number of repeated samples per timestep (default: 10)
- `window`: Sliding window size for DUASE (default: nothing = use all T)
- `L_true`: Optional ground truth for oracle alignment
- `rng`: Random number generator

# Returns
- Vector of estimated L matrices, temporally aligned

# Example
```julia
X_series = [evolve_dynamics(X0, t) for t in 1:T]
L_hat = embed_temporal_duase(X_series, 2; K=30)
```
"""
function embed_temporal_duase(X_series::Vector{<:AbstractMatrix}, d::Int;
                               K::Int=10, window::Union{Nothing, Int}=nothing,
                               L_true::Union{Nothing, Vector}=nothing,
                               rng=Random.GLOBAL_RNG)
    T = length(X_series)
    n = size(X_series[1], 1)

    # Step 1: Generate K adjacency samples at each time point and average
    A_avg_series = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        A_sum = zeros(n, n)
        for _ in 1:K
            A_sum .+= sample_adjacency(X_series[t]; rng=rng)
        end
        A_avg_series[t] = A_sum ./ K
    end

    # Step 2: Apply DUASE
    _, X_raw = duase_embedding(A_avg_series, d; window=window)

    # Step 3: Temporal alignment via Procrustes chain
    L_aligned = Vector{Matrix{Float64}}(undef, T)
    L_aligned[1] = X_raw[1]

    for t in 2:T
        Q = ortho_procrustes_RM(X_raw[t]', L_aligned[t-1]')
        L_aligned[t] = X_raw[t] * Q
    end

    # Step 4: Global alignment
    if !isnothing(L_true)
        Q_global = ortho_procrustes_RM(L_aligned[1]', L_true[1]')
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    else
        L_stacked = vcat(L_aligned...)
        _, Q_global = align_to_Bd_plus(L_stacked)
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    end

    return L_aligned
end

"""
    embed_temporal_duase_raw(X_series::Vector{<:AbstractMatrix}, d::Int;
                              K::Int=10, window::Union{Nothing, Int}=nothing,
                              rng=Random.GLOBAL_RNG) -> Vector{Matrix}

DUASE embedding WITHOUT B^d_+ alignment - returns embeddings in their natural space.

The key insight: DUASE's shared basis G already provides temporal consistency.
We don't need to align to B^d_+ for dynamics learning - we can learn in whatever
rotated space the embedding gives us.

# Pipeline
1. For each time t, generate K adjacency samples from X(t) and average
2. Apply DUASE (optionally with windowing)
3. Apply consistent sign flips (not full Procrustes or B^d_+ alignment)

# Arguments
- `X_series`: Vector of true latent position matrices (one per timestep)
- `d`: Embedding dimension
- `K`: Number of repeated samples per timestep (default: 10)
- `window`: Sliding window size for DUASE (default: nothing = use all T)
- `rng`: Random number generator

# Returns
- Vector of estimated L matrices, temporally aligned via shared basis

# Notes
- NO B^d_+ projection (which can distort geometry)
- Dynamics can be learned in the rotated space since P = XX' is rotation invariant
"""
function embed_temporal_duase_raw(X_series::Vector{<:AbstractMatrix}, d::Int;
                                   K::Int=10, window::Union{Nothing, Int}=nothing,
                                   rng=Random.GLOBAL_RNG)
    T = length(X_series)
    n = size(X_series[1], 1)

    # Step 1: Generate K adjacency samples at each time point and average
    A_avg_series = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        A_sum = zeros(n, n)
        for _ in 1:K
            A_sum .+= sample_adjacency(X_series[t]; rng=rng)
        end
        A_avg_series[t] = A_sum ./ K
    end

    # Step 2: Apply DUASE - gives embeddings aligned via shared basis G
    G, X_raw = duase_embedding(A_avg_series, d; window=window)

    # Step 3: Consistent sign flips only (minimal alignment)
    # Flip signs so that the majority of entries are positive (heuristic)
    L_aligned = Vector{Matrix{Float64}}(undef, T)

    # Determine sign flips from first timestep
    sign_flips = ones(d)
    for j in 1:d
        if sum(X_raw[1][:, j] .< 0) > n / 2
            sign_flips[j] = -1.0
        end
    end

    # Apply consistent sign flips to all timesteps
    for t in 1:T
        L_aligned[t] = X_raw[t] .* sign_flips'
    end

    return L_aligned
end


# =============================================================================
# OMNI: Omnibus Embedding
# =============================================================================
#
# Reference: Levin et al. (2017), IEEE ICDMW
#
# Construct omnibus matrix M where:
#   - Diagonal blocks: M[ii] = A(i)
#   - Off-diagonal blocks: M[ij] = (A(i) + A(j)) / 2
#
# Joint embedding ensures all graphs in common latent space.
# =============================================================================

"""
    omni_embedding(A_series::Vector{<:AbstractMatrix}, d::Int) -> Vector{Matrix}

Omnibus embedding for multiple graphs.

Constructs the omnibus matrix M (nT × nT) and embeds via SVD.

# Algorithm
1. Construct M with diagonal blocks A(i) and off-diagonal (A(i)+A(j))/2
2. SVD: M = U Σ V'
3. Extract per-time embeddings from U row blocks: X̂(t) = U[(t-1)n+1:tn, 1:d] · √Σ

# Arguments
- `A_series`: Vector of adjacency matrices
- `d`: Embedding dimension

# Returns
- Vector of embedding matrices (one per time point, each n × d)

# Notes
- Memory: O((nT)²) for the omnibus matrix
- Guaranteed joint embedding: all graphs in same latent space
- Empirically performs as well as DUASE for RDPG recovery
"""
function omni_embedding(A_series::Vector{<:AbstractMatrix}, d::Int)
    T = length(A_series)
    n = size(A_series[1], 1)

    # Construct omnibus matrix M (nT × nT)
    M = zeros(n * T, n * T)

    for i in 1:T
        for j in 1:T
            row_start = (i - 1) * n + 1
            row_end = i * n
            col_start = (j - 1) * n + 1
            col_end = j * n

            if i == j
                # Diagonal block: A(i)
                M[row_start:row_end, col_start:col_end] = A_series[i]
            else
                # Off-diagonal block: average of A(i) and A(j)
                M[row_start:row_end, col_start:col_end] = (A_series[i] + A_series[j]) / 2
            end
        end
    end

    # SVD of omnibus matrix
    F = svd(M)
    U = F.U[:, 1:d]
    sqrt_S = sqrt.(F.S[1:d])

    # Extract per-time embeddings from U row blocks
    X_series = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        row_start = (t - 1) * n + 1
        row_end = t * n
        X_series[t] = U[row_start:row_end, :] .* sqrt_S'
    end

    return X_series
end

"""
    embed_temporal_omni(X_series::Vector{<:AbstractMatrix}, d::Int;
                         K::Int=10,
                         L_true::Union{Nothing, Vector}=nothing,
                         rng=Random.GLOBAL_RNG) -> Vector{Matrix}

Full RDPG pipeline using Omnibus: latent positions → adjacency samples → OMNI embedding → alignment.

# Arguments
- `X_series`: Vector of true latent position matrices
- `d`: Embedding dimension
- `K`: Number of repeated samples per timestep (default: 10)
- `L_true`: Optional ground truth for oracle alignment
- `rng`: Random number generator

# Returns
- Vector of estimated L matrices, temporally aligned

# Notes
- OMNI guarantees joint embedding (no Procrustes chain needed)
- Higher memory cost than DUASE: O((nT)²) vs O(n × nT)
"""
function embed_temporal_omni(X_series::Vector{<:AbstractMatrix}, d::Int;
                              K::Int=10,
                              L_true::Union{Nothing, Vector}=nothing,
                              rng=Random.GLOBAL_RNG)
    T = length(X_series)
    n = size(X_series[1], 1)

    # Step 1: Generate K adjacency samples at each time point and average
    A_avg_series = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        A_sum = zeros(n, n)
        for _ in 1:K
            A_sum .+= sample_adjacency(X_series[t]; rng=rng)
        end
        A_avg_series[t] = A_sum ./ K
    end

    # Step 2: Apply Omnibus embedding
    X_raw = omni_embedding(A_avg_series, d)

    # Step 3: Global alignment only (OMNI already gives joint embedding)
    L_aligned = copy(X_raw)

    if !isnothing(L_true)
        Q_global = ortho_procrustes_RM(L_aligned[1]', L_true[1]')
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    else
        L_stacked = vcat(L_aligned...)
        _, Q_global = align_to_Bd_plus(L_stacked)
        for t in 1:T
            L_aligned[t] = L_aligned[t] * Q_global
        end
    end

    return L_aligned
end
