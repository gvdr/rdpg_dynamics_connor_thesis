"""
    Constraint loss functions for RDPG Neural ODE training.

    RDPG models predict probability matrices where entries should be in [0, 1].
    These functions provide differentiable penalty terms for constraint violations.
"""

using LinearAlgebra

export probability_constraint_loss, below_zero_penalty, above_one_penalty

"""
    below_zero_penalty(P::AbstractMatrix{T}; exclude_diagonal::Bool=true) where T

Compute penalty for off-diagonal entries below zero.

Returns sum of |min(P_ij, 0)| for off-diagonal entries, penalizing negative probabilities.

# Arguments
- `P`: Probability matrix (n × n)
- `exclude_diagonal`: If true (default), ignore diagonal entries

# Returns
- Non-negative penalty value
"""
function below_zero_penalty(P::AbstractMatrix{T}; exclude_diagonal::Bool=true) where T
    if exclude_diagonal
        n = size(P, 1)
        # Create off-diagonal mask efficiently
        penalty = zero(T)
        @inbounds for j in axes(P, 2)
            for i in axes(P, 1)
                if i != j && P[i, j] < zero(T)
                    penalty -= P[i, j]
                end
            end
        end
        return penalty
    else
        return sum(max.(-P, zero(T)))
    end
end

"""
    above_one_penalty(P::AbstractMatrix{T}; exclude_diagonal::Bool=true) where T

Compute penalty for off-diagonal entries above one.

Returns sum of max(P_ij - 1, 0) for off-diagonal entries, penalizing probabilities > 1.

# Arguments
- `P`: Probability matrix (n × n)
- `exclude_diagonal`: If true (default), ignore diagonal entries

# Returns
- Non-negative penalty value
"""
function above_one_penalty(P::AbstractMatrix{T}; exclude_diagonal::Bool=true) where T
    if exclude_diagonal
        n = size(P, 1)
        penalty = zero(T)
        @inbounds for j in axes(P, 2)
            for i in axes(P, 1)
                if i != j && P[i, j] > one(T)
                    penalty += P[i, j] - one(T)
                end
            end
        end
        return penalty
    else
        return sum(max.(P .- one(T), zero(T)))
    end
end

"""
    probability_constraint_loss(P::AbstractMatrix{T};
                                 lambda_lower::Real=1.0,
                                 lambda_upper::Real=1.0,
                                 exclude_diagonal::Bool=true) where T

Combined constraint loss for probability matrices.

Computes weighted sum of penalties for values outside [0, 1].

# Arguments
- `P`: Probability matrix (n × n)
- `lambda_lower`: Weight for below-zero penalty (default: 1.0)
- `lambda_upper`: Weight for above-one penalty (default: 1.0)
- `exclude_diagonal`: If true (default), ignore diagonal entries

# Returns
- Non-negative combined penalty value

# Example
```julia
P = [0.5 -0.1 1.2; 0.3 0.5 0.8; 0.9 0.1 0.5]
loss = probability_constraint_loss(P)  # Penalizes -0.1 and 1.2
```
"""
function probability_constraint_loss(P::AbstractMatrix{T};
                                      lambda_lower::Real=1.0,
                                      lambda_upper::Real=1.0,
                                      exclude_diagonal::Bool=true) where T
    lower_penalty = below_zero_penalty(P; exclude_diagonal=exclude_diagonal)
    upper_penalty = above_one_penalty(P; exclude_diagonal=exclude_diagonal)
    return T(lambda_lower) * lower_penalty + T(lambda_upper) * upper_penalty
end

"""
    probability_constraint_loss(predictions::AbstractVector{<:AbstractMatrix};
                                 lambda_lower::Real=1.0,
                                 lambda_upper::Real=1.0) -> Real

Compute total constraint loss over a batch of probability matrices.

# Arguments
- `predictions`: Vector of probability matrices
- `lambda_lower`: Weight for below-zero penalty
- `lambda_upper`: Weight for above-one penalty

# Returns
- Sum of constraint losses across all matrices
"""
function probability_constraint_loss(predictions::AbstractVector{<:AbstractMatrix};
                                      lambda_lower::Real=1.0,
                                      lambda_upper::Real=1.0)
    return sum(probability_constraint_loss(P; lambda_lower=lambda_lower, lambda_upper=lambda_upper)
               for P in predictions)
end
