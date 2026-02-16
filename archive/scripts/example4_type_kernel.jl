#!/usr/bin/env -S julia --project
"""
Example 4: Heterogeneous Dynamics with Type-Specific Kernels + Symbolic Regression

**The Full Pipeline:**
1. Domain knowledge (UDE): Node type membership + self-rates are known
2. Flexible learning: NN kernel learns κ(P_ij, type_i, type_j)
3. Symbolic discovery: SymReg extracts interpretable equations per type-pair

**Known physics (UDE):**
- Node types: predator, prey, resource
- Self-rates: a_P, a_Y, a_R (decay rates per type)

**Unknown (learned by NN):**
- Input: (P_ij, type_i, type_j) with one-hot encoding for types
- Output: 9 message kernels [κ_PP, κ_PY, κ_PR, κ_YP, κ_YY, κ_YR, κ_RP, κ_RY, κ_RR]
- For SymReg: analyze each output NN(·,·,·)[k] as a separate function

**True dynamics include Holling Type II:**
- κ_PY(p) = α·p / (1 + β·p)  -- Saturating predation (to be discovered!)
- Other interactions: linear or constant

Usage:
    julia --project scripts/example4_type_kernel.jl
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

using RDPGDynamics
using LinearAlgebra
using Random
using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using Zygote
using Lux
using ComponentArrays
using CairoMakie
using Printf
using Statistics
using Serialization  # For saving intermediate results
using SymbolicRegression  # For discovering kernel equations

const CM = CairoMakie

# Output directory for all results
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "results", "example4_type_kernel")
mkpath(OUTPUT_DIR)

# ============================================================================
# Configuration
# ============================================================================

# Node counts per type (full scale for proper results)
const N_PRED = 12    # Predators
const N_PREY = 15    # Prey
const N_RES = 10     # Resources
const N_TOTAL = N_PRED + N_PREY + N_RES  # 37 total

const D_EMBED = 2    # Embedding dimension
const T_TOTAL = 25   # Total timesteps
const TRAIN_FRAC = 0.7  # Use 70% for training
const SEED = 42

# Type indices (1-indexed ranges)
const IDX_PRED = 1:N_PRED
const IDX_PREY = (N_PRED+1):(N_PRED+N_PREY)
const IDX_RES = (N_PRED+N_PREY+1):N_TOTAL

# Type labels
const TYPE_P = 1  # Predator
const TYPE_Y = 2  # Prey
const TYPE_R = 3  # Resource
const K_TYPES = 3

# Type assignment for each node
const NODE_TYPES = vcat(
    fill(TYPE_P, N_PRED),
    fill(TYPE_Y, N_PREY),
    fill(TYPE_R, N_RES)
)

# ============================================================================
# True Dynamics: Type-Specific with Holling Type II
# ============================================================================
#
# N_ij = κ_{type(i), type(j)}(P_ij)
# ẋ = N·x (equivalent to message-passing form)
#
# Key feature: Predator-Prey interaction uses Holling Type II (saturating)
#   κ_PY(p) = α·p / (1 + β·p)
#
# This nonlinearity is what SymReg should discover!

# Self-rates (KNOWN PHYSICS - part of UDE structure)
# Small decay for stability
const KNOWN_SELF_RATES = Dict(
    TYPE_P => -0.002,   # Predator: small decay
    TYPE_Y => -0.001,   # Prey: minimal decay
    TYPE_R =>  0.000    # Resource: stable
)

# Message kernels κ_ab(p) - stored as functions
# Holling Type II for predator-prey - scaled for n~37 nodes
const HOLLING_ALPHA = 0.025
const HOLLING_BETA = 2.0

function κ_true(type_i::Int, type_j::Int, p::Real)
    # Predator-Predator: mild repulsion (constant)
    if type_i == TYPE_P && type_j == TYPE_P
        return -0.004

    # Predator-Prey: HOLLING TYPE II (saturating attraction) - CHASE!
    elseif type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)

    # Predator-Resource: ignore
    elseif type_i == TYPE_P && type_j == TYPE_R
        return 0.0

    # Prey-Predator: FLEE! (linear, negative)
    elseif type_i == TYPE_Y && type_j == TYPE_P
        return -0.02 * p

    # Prey-Prey: mild cohesion (constant) - herding
    elseif type_i == TYPE_Y && type_j == TYPE_Y
        return 0.003

    # Prey-Resource: attraction (linear)
    elseif type_i == TYPE_Y && type_j == TYPE_R
        return 0.012 * p

    # Resource-Predator: ignore
    elseif type_i == TYPE_R && type_j == TYPE_P
        return 0.0

    # Resource-Prey: depletion (linear, negative)
    elseif type_i == TYPE_R && type_j == TYPE_Y
        return -0.006 * p

    # Resource-Resource: cohesion (constant)
    elseif type_i == TYPE_R && type_j == TYPE_R
        return 0.005

    else
        return 0.0
    end
end

"""
Compute N matrix from true type-specific kernels.
"""
function compute_N_true(P::Matrix{Float64})
    n = size(P, 1)
    N = zeros(n, n)

    for i in 1:n
        ti = NODE_TYPES[i]
        # Diagonal: self-rate minus sum of outgoing messages
        N[i,i] = KNOWN_SELF_RATES[ti]
        for j in 1:n
            if j != i
                tj = NODE_TYPES[j]
                κ_ij = κ_true(ti, tj, P[i,j])
                N[i,j] = κ_ij
                N[i,i] -= κ_ij  # Message-passing form: subtract from diagonal
            end
        end
    end

    return N
end

"""
True dynamics: ẋ = N(P)·x with type-specific kernels including Holling Type II.
"""
function true_dynamics!(dX::Matrix{Float64}, X::Matrix{Float64}, p, t)
    P = X * X'
    N = compute_N_true(P)
    dX .= N * X
end

function true_dynamics_vec!(du::Vector{Float64}, u::Vector{Float64}, p, t)
    n, d = N_TOTAL, D_EMBED
    X = collect(transpose(reshape(u, d, n)))  # Materialize to concrete Matrix
    dX = similar(X)
    true_dynamics!(dX, X, p, t)
    du .= vec(transpose(dX))
end

# ============================================================================
# Data Generation
# ============================================================================

function generate_true_data(; seed=SEED)
    rng = Xoshiro(seed)

    # Initialize: spread types in DISTINCT regions for better 2D structure
    # More separation so SVD can see 2D structure from the start
    X0 = zeros(N_TOTAL, D_EMBED)

    for i in IDX_PRED
        X0[i, :] = [0.7, 0.6] .+ 0.06 .* randn(rng, D_EMBED)  # Upper right
    end
    for i in IDX_PREY
        X0[i, :] = [0.4, 0.6] .+ 0.06 .* randn(rng, D_EMBED)  # Upper left
    end
    for i in IDX_RES
        X0[i, :] = [0.5, 0.3] .+ 0.06 .* randn(rng, D_EMBED)  # Lower middle
    end

    X0 = max.(X0, 0.15)  # Keep away from origin

    u0 = vec(collect(transpose(X0)))
    tspan = (0.0, Float64(T_TOTAL - 1))

    prob = ODEProblem(true_dynamics_vec!, u0, tspan)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_true = [collect(transpose(reshape(sol.u[i], D_EMBED, N_TOTAL))) for i in 1:length(sol.t)]
    return X_true
end

# Use RDPGDynamics functions for adjacency sampling and embedding
# sample_adjacency, embed_temporal_duase_raw are from the package

"""
Generate adjacencies using RDPGDynamics.sample_adjacency with K repeated samples.
"""
function generate_adjacencies(X_true::Vector{Matrix{Float64}}; K::Int=10, seed=SEED)
    rng = Xoshiro(seed + 1000)
    T = length(X_true)

    A_obs = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        # Use package function with K repeated samples (averaged)
        A_obs[t] = sample_adjacency_repeated(X_true[t], K; rng=rng)
    end
    return A_obs
end

"""
DUASE estimation - uses shared basis G for natural temporal alignment.
NO Procrustes chain! That would fight against DUASE's alignment and introduce noise.
Only apply consistent sign flips (determined from t=1) to all timesteps.
"""
function duase_estimate(A_obs::Vector{Matrix{Float64}}, d::Int; window::Union{Nothing,Int}=nothing)
    T = length(A_obs)
    n = size(A_obs[1], 1)

    # Use DUASE from the package - gives embeddings X(t) = G · √Q(t) with shared G
    _, X_raw = duase_embedding(A_obs, d; window=window)

    # Only apply CONSISTENT sign flips (same for all t, determined from t=1)
    # This fixes eigenvector sign ambiguity without introducing rotation noise
    sign_flips = ones(d)
    for j in 1:d
        if sum(X_raw[1][:, j] .< 0) > n / 2
            sign_flips[j] = -1.0
        end
    end

    X_est = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        X_est[t] = X_raw[t] .* sign_flips'
    end

    return X_est
end

# ============================================================================
# Neural Network Kernel (UDE: knows types + self-rates, learns κ flexibly)
# ============================================================================
#
# KNOWN PHYSICS (not learned):
#   - Node types (predator, prey, resource)
#   - Self-rates a_P, a_Y, a_R
#
# LEARNED BY NN:
#   - Input: (P_ij, type_i_onehot[3], type_j_onehot[3]) = 7 features
#   - Output: 9 message kernels κ(P_ij) for each type pair
#
# Output indices (all 9 asymmetric type pairs):
#   1: κ_PP, 2: κ_PY, 3: κ_PR
#   4: κ_YP, 5: κ_YY, 6: κ_YR
#   7: κ_RP, 8: κ_RY, 9: κ_RR

# Map (type_i, type_j) to output index
const TYPE_PAIR_TO_IDX = Dict(
    (TYPE_P, TYPE_P) => 1,
    (TYPE_P, TYPE_Y) => 2,
    (TYPE_P, TYPE_R) => 3,
    (TYPE_Y, TYPE_P) => 4,
    (TYPE_Y, TYPE_Y) => 5,
    (TYPE_Y, TYPE_R) => 6,
    (TYPE_R, TYPE_P) => 7,
    (TYPE_R, TYPE_Y) => 8,
    (TYPE_R, TYPE_R) => 9
)

"""
Build the kernel NN with scalar type encoding.
Input: 3 features (P_ij, type_i_scalar, type_j_scalar)
  - type encoding: Predator=1, Prey=0, Resource=-1
Output: 9 message kernels (self-rates are known, not learned)
  - tanh output bounded to [-0.1, 0.1] for stable dynamics
  - Small weight initialization for ODE stability at start of training
"""
function build_kernel_nn(; hidden_sizes=[16, 16], rng=Random.default_rng())
    # Small weight initializer for stable ODE at start of training
    small_init(rng, dims...) = 0.1f0 * randn(rng, Float32, dims...)

    layers = []
    in_dim = 3  # (P_ij, type_i, type_j)

    for h in hidden_sizes
        push!(layers, Lux.Dense(in_dim, h, tanh; init_weight=small_init, init_bias=Lux.zeros32))
        in_dim = h
    end

    # Output: 9 kernels with tanh to bound in [-1, 1], then scale
    # Even smaller weights for output layer
    tiny_init(rng, dims...) = 0.01f0 * randn(rng, Float32, dims...)
    push!(layers, Lux.Dense(in_dim, 9, tanh; init_weight=tiny_init, init_bias=Lux.zeros32))

    return Lux.Chain(layers...)
end

# Scalar type encoding: Predator=1, Prey=0, Resource=-1
const TYPE_SCALAR = Float32[1.0, 0.0, -1.0]

# Pre-computed type pair index lookup table (n×n matrix for fast lookup)
function make_type_pair_matrix(types::Vector{Int})
    n = length(types)
    idx_mat = zeros(Int, n, n)
    for i in 1:n, j in 1:n
        idx_mat[i,j] = TYPE_PAIR_TO_IDX[(types[i], types[j])]
    end
    return idx_mat
end

const TYPE_PAIR_MATRIX = make_type_pair_matrix(NODE_TYPES)

# Output scale: tanh gives [-1,1], scale to reasonable kernel magnitude
# True kernels are roughly in [-0.05, 0.03], so scale of 0.15 gives good range
# while still preventing exploding dynamics
const KERNEL_SCALE = 0.15f0

# Pre-computed constant arrays for type scalars (avoid rebuilding each call)
# These are Float32 constants, not tracked
const _TYPE_I_BATCH = let
    n = N_TOTAL
    arr = zeros(Float32, n * (n - 1))
    idx = 1
    for i in 1:n, j in 1:n
        i == j && continue
        arr[idx] = TYPE_SCALAR[NODE_TYPES[i]]
        idx += 1
    end
    reshape(arr, 1, :)
end

const _TYPE_J_BATCH = let
    n = N_TOTAL
    arr = zeros(Float32, n * (n - 1))
    idx = 1
    for i in 1:n, j in 1:n
        i == j && continue
        arr[idx] = TYPE_SCALAR[NODE_TYPES[j]]
        idx += 1
    end
    reshape(arr, 1, :)
end

# Pre-computed CartesianIndex arrays for scattering kernel outputs to N matrix
# _OFFDIAG_IDX[k] = (i, j) for the k-th off-diagonal pair
const _OFFDIAG_IDX = let
    n = N_TOTAL
    indices = CartesianIndex{2}[]
    for i in 1:n, j in 1:n
        i == j && continue
        push!(indices, CartesianIndex(i, j))
    end
    indices
end

# _TYPE_PAIR_IDX_BATCH[k] = which of the 9 kernel outputs to use for pair k
const _TYPE_PAIR_IDX_BATCH = let
    n = N_TOTAL
    indices = Int[]
    for i in 1:n, j in 1:n
        i == j && continue
        push!(indices, TYPE_PAIR_MATRIX[i, j])
    end
    indices
end

# Diagonal self-rates as a vector
const _SELF_RATES_DIAG = Float32[KNOWN_SELF_RATES[NODE_TYPES[i]] for i in 1:N_TOTAL]

# Pre-computed scatter matrix: _SCATTER_MAT[i, j, k] = 1 if off-diagonal pair k is at position (i,j)
const _SCATTER_MAT = let
    n = N_TOTAL
    n_pairs = n * (n - 1)
    mat = zeros(Float32, n, n, n_pairs)
    for k in 1:n_pairs
        idx = _OFFDIAG_IDX[k]
        mat[idx[1], idx[2], k] = 1.0f0
    end
    mat
end

"""
Compute N matrix using batched NN with scalar type encoding.
Fully non-mutating for Zygote compatibility using einsum-style scatter.
"""
function compute_N_nn(P::Matrix{T}, nn, ps, st) where T
    n = size(P, 1)
    n_pairs = n * (n - 1)

    # Extract P values at off-diagonal positions
    P_flat = P[_OFFDIAG_IDX]  # Vector of length n_pairs

    # Build input batch: vcat tracked P values with constant type scalars
    input_batch = vcat(
        reshape(P_flat, 1, n_pairs),
        T.(_TYPE_I_BATCH),
        T.(_TYPE_J_BATCH)
    )

    # Batched NN forward pass - output already bounded by tanh in [-1, 1]
    output_batch, _ = nn(input_batch, ps, st)  # 9 × n_pairs

    # Scale to small kernel values
    scaled_output = output_batch .* T(KERNEL_SCALE)

    # Extract the correct kernel for each pair using linear indexing
    linear_idx = _TYPE_PAIR_IDX_BATCH .+ (0:n_pairs-1) .* 9
    κ_values = scaled_output[linear_idx]  # Vector of length n_pairs

    # Scatter κ_values to N_offdiag using pre-computed scatter matrix
    # N_offdiag[i,j] = sum_k scatter_mat[i,j,k] * κ_values[k]
    scatter_mat_T = T.(_SCATTER_MAT)
    N_offdiag = reshape(reshape(scatter_mat_T, n*n, n_pairs) * κ_values, n, n)

    # Diagonal: self_rates - row_sums
    row_sums = sum(N_offdiag; dims=2)
    diag_vals = T.(_SELF_RATES_DIAG) .- vec(row_sums)

    # Add diagonal (non-mutating)
    return N_offdiag + Diagonal(diag_vals)
end

"""
Learned dynamics using kernel NN (non-mutating for Zygote compatibility).
"""
function nn_dynamics(X::Matrix{T}, ps, nn, st) where T
    # Compute P = X * X' directly
    P = X * transpose(X)
    N = compute_N_nn(P, nn, ps, st)
    return N * X  # Non-mutating
end

# ============================================================================
# Training
# ============================================================================

function train_kernel_nn(X_train::Vector{Matrix{Float64}};
                         epochs::Int=500, lr::Float64=0.01, epochs_lbfgs::Int=100)
    n, d = size(X_train[1])
    T_train = length(X_train)

    # Convert training data to Float32 for efficient NN ops
    X_train_f32 = [Float32.(X) for X in X_train]
    X_target = hcat([vec(permutedims(X)) for X in X_train_f32]...)
    u0 = vec(permutedims(X_train_f32[1]))
    tspan = (0.0f0, Float32(T_train - 1))

    # Build NN (Lux defaults to Float32)
    rng = Xoshiro(123)
    nn = build_kernel_nn(; hidden_sizes=[32, 32], rng=rng)
    ps, st = Lux.setup(rng, nn)
    ps = ComponentArray(ps)

    # ODE with NN dynamics (out-of-place for Zygote compatibility)
    function nn_ode(u, p, t)
        X = permutedims(reshape(u, d, n))  # (n, d)
        dX = nn_dynamics(X, p, nn, st)
        return vec(permutedims(dX))
    end

    prob = ODEProblem{false}(nn_ode, u0, tspan, ps)

    # Loss = MSE only (simpler, faster)
    function loss(p, _)
        _prob = remake(prob; p=p)
        sol = solve(_prob, Tsit5();
                    saveat=1.0f0,
                    sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()),
                    abstol=1f-4, reltol=1f-4)
        sol.retcode != ReturnCode.Success && return 1.0f6

        pred = reduce(hcat, sol.u)
        mse = sum(abs2, pred .- X_target) / length(X_target)
        return mse
    end

    # Use Zygote for outer optimization (works with Lux)
    opt_func = OptimizationFunction(loss, Optimization.AutoZygote())
    opt_prob = OptimizationProblem(opt_func, ps)

    println("      Training kernel NN with Adam (" * string(epochs) * " epochs)...")
    result_adam = solve(opt_prob, OptimizationOptimisers.Adam(Float32(lr));
                        maxiters=epochs, progress=true)

    println("      Adam loss: " * string(round(result_adam.objective; digits=6)))

    # LBFGS refinement
    println("      Refining with LBFGS (" * string(epochs_lbfgs) * " iterations)...")
    opt_prob_lbfgs = OptimizationProblem(opt_func, result_adam.u)
    result_lbfgs = solve(opt_prob_lbfgs, OptimizationOptimJL.LBFGS();
                         maxiters=epochs_lbfgs, progress=true)

    println("      Final loss: " * string(round(result_lbfgs.objective; digits=6)))

    return result_lbfgs.u, nn, st
end

# ============================================================================
# Symbolic Regression
# ============================================================================

"""
Sample the learned kernel NN for symbolic regression.

For each type pair (ti, tj), generate (P_ij, κ) pairs.
Uses scalar type encoding: Predator=1, Prey=0, Resource=-1
Returns a Dict mapping (ti, tj) to (p, κ) vectors for SymReg.
"""
function sample_kernel_for_symreg(nn, ps, st; n_samples=500, seed=999)
    rng = Xoshiro(seed)

    # Sample P_ij values
    p_values = rand(rng, Float32, n_samples) .* 0.8f0 .+ 0.1f0  # P_ij in [0.1, 0.9]

    # For each type pair, collect samples
    samples = Dict{Tuple{Int,Int}, @NamedTuple{p::Vector{Float64}, κ::Vector{Float64}}}()

    for ti in 1:K_TYPES
        ti_scalar = TYPE_SCALAR[ti]

        for tj in 1:K_TYPES
            tj_scalar = TYPE_SCALAR[tj]

            # Build batched input: all p_values at once
            input_batch = zeros(Float32, 3, n_samples)
            input_batch[1, :] .= p_values
            input_batch[2, :] .= ti_scalar
            input_batch[3, :] .= tj_scalar

            output_batch, _ = nn(input_batch, ps, st)
            output_batch = output_batch .* KERNEL_SCALE

            # Get the kernel for this type pair
            idx = TYPE_PAIR_TO_IDX[(ti, tj)]
            κ_values = Float64.(output_batch[idx, :])

            samples[(ti, tj)] = (p=Float64.(p_values), κ=κ_values)
        end
    end

    return samples
end

"""
Run SymbolicRegression.jl on each type pair to discover kernel equations.

Returns a Dict mapping (ti, tj) to discovered equations with their complexity/loss.
"""
function run_symbolic_regression(samples;
                                  niterations::Int=30,
                                  maxsize::Int=15,
                                  populations::Int=15,
                                  timeout_seconds::Int=60)
    type_names = ["Pred", "Prey", "Res"]

    # SR options - allow division for Holling Type II discovery
    options = Options(
        binary_operators=[+, -, *, /],
        unary_operators=[],
        populations=populations,
        population_size=33,
        maxsize=maxsize,
        timeout_in_seconds=timeout_seconds,
        progress=false,  # Quiet mode
    )

    results = Dict{Tuple{Int,Int}, @NamedTuple{
        equation::String,
        complexity::Int,
        loss::Float64,
        hall_of_fame::Any
    }}()

    for ti in 1:K_TYPES
        for tj in 1:K_TYPES
            data = samples[(ti, tj)]
            p_vals = data.p
            κ_vals = data.κ

            # Skip if data is essentially constant (SR won't find anything useful)
            κ_range = maximum(κ_vals) - minimum(κ_vals)
            if κ_range < 1e-8
                results[(ti, tj)] = (
                    equation=@sprintf("%.6f", mean(κ_vals)),
                    complexity=1,
                    loss=0.0,
                    hall_of_fame=nothing
                )
                continue
            end

            # Prepare data for SR: X is (1, n_samples), y is (n_samples,)
            X = reshape(p_vals, 1, :)
            y = κ_vals

            pair_name = type_names[ti] * "→" * type_names[tj]
            println("      Running SR for κ_" * pair_name * "...")

            # Run symbolic regression
            hall_of_fame = equation_search(
                X, y;
                options=options,
                niterations=niterations,
                parallelism=:serial
            )

            # Get best equation (lowest loss among reasonable complexity)
            best_eq = nothing
            best_loss = Inf
            best_complexity = 0

            for member in hall_of_fame.members
                member === nothing && continue
                # Prefer equations with complexity <= maxsize and finite loss
                if isfinite(member.loss) && member.loss < best_loss
                    best_eq = member
                    best_loss = member.loss
                    best_complexity = compute_complexity(member.tree, options)
                end
            end

            if best_eq !== nothing
                eq_str = string(best_eq.tree)
                results[(ti, tj)] = (
                    equation=eq_str,
                    complexity=best_complexity,
                    loss=best_loss,
                    hall_of_fame=hall_of_fame
                )
            else
                results[(ti, tj)] = (
                    equation="FAILED",
                    complexity=0,
                    loss=Inf,
                    hall_of_fame=hall_of_fame
                )
            end
        end
    end

    return results
end

"""
Format SR results for display, showing Pareto frontier of complexity vs loss.
"""
function format_sr_results(sr_results)
    type_names = ["Pred", "Prey", "Res"]

    println("\n   Discovered equations (via SymbolicRegression.jl):")
    println("   " * "-"^70)

    for ti in 1:K_TYPES
        for tj in 1:K_TYPES
            result = sr_results[(ti, tj)]
            pair_name = type_names[ti] * "→" * type_names[tj]

            # Clean up equation string for display
            eq_str = result.equation
            # Replace x1 with p for readability
            eq_str = replace(eq_str, "x1" => "p")

            loss_str = result.loss < 1e-10 ? "<1e-10" : @sprintf("%.2e", result.loss)

            println("     κ_" * pair_name * ": " * eq_str)
            println("       (complexity=" * string(result.complexity) * ", loss=" * loss_str * ")")
        end
    end
    println("   " * "-"^70)
end

"""
Extract Pareto-optimal equations from hall of fame for a type pair.
Returns vector of (complexity, loss, equation_string) tuples.
"""
function get_pareto_frontier(sr_result)
    hof = sr_result.hall_of_fame
    hof === nothing && return []

    frontier = Tuple{Int, Float64, String}[]
    for member in hof.members
        member === nothing && continue
        !isfinite(member.loss) && continue
        push!(frontier, (
            compute_complexity(member.tree, hof.options),
            member.loss,
            string(member.tree)
        ))
    end

    # Sort by complexity
    sort!(frontier, by=x->x[1])
    return frontier
end

# ============================================================================
# Evaluation
# ============================================================================

function predict_P_trajectory(ps, nn, st, X0::Matrix{Float64}, T::Int)
    n, d = size(X0)
    u0 = vec(collect(transpose(X0)))
    tspan = (0.0, Float64(T - 1))

    # Use out-of-place ODE for consistency with training
    function ode(u, p, t)
        X = permutedims(reshape(u, d, n))
        dX = nn_dynamics(Float32.(X), p, nn, st)
        return vec(permutedims(Float64.(dX)))
    end

    prob = ODEProblem{false}(ode, u0, tspan, ps)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_traj = [collect(transpose(reshape(u, d, n))) for u in sol.u]
    P_traj = [X * X' for X in X_traj]

    return P_traj, X_traj
end

function compute_P_error(P_pred, P_true)
    T = min(length(P_pred), length(P_true))
    [norm(P_pred[t] .- P_true[t]) / norm(P_true[t]) for t in 1:T]
end

# ============================================================================
# Preliminary Trajectory Visualization
# ============================================================================

"""
Visualize trajectories to check they stay in reasonable gauge before training.
"""
function visualize_trajectories(X_true, X_est, P_true, P_est)
    T = length(X_true)
    type_colors = [:red, :blue, :green]  # Pred, Prey, Res

    fig = CM.Figure(size=(1400, 800))

    # Row 1: True X trajectories
    ax1 = CM.Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", title="True X trajectories")
    for i in 1:N_TOTAL
        ti = NODE_TYPES[i]
        xs = [X_true[t][i, 1] for t in 1:T]
        ys = [X_true[t][i, 2] for t in 1:T]
        CM.lines!(ax1, xs, ys; color=(type_colors[ti], 0.5), linewidth=1)
        CM.scatter!(ax1, [xs[1]], [ys[1]]; color=type_colors[ti], markersize=8)
        CM.scatter!(ax1, [xs[end]], [ys[end]]; color=type_colors[ti], marker=:star5, markersize=10)
    end

    # Row 1: Estimated X trajectories
    ax2 = CM.Axis(fig[1, 2]; xlabel="x₁", ylabel="x₂", title="DUASE X̂ trajectories")
    for i in 1:N_TOTAL
        ti = NODE_TYPES[i]
        xs = [X_est[t][i, 1] for t in 1:T]
        ys = [X_est[t][i, 2] for t in 1:T]
        CM.lines!(ax2, xs, ys; color=(type_colors[ti], 0.5), linewidth=1)
        CM.scatter!(ax2, [xs[1]], [ys[1]]; color=type_colors[ti], markersize=8)
        CM.scatter!(ax2, [xs[end]], [ys[end]]; color=type_colors[ti], marker=:star5, markersize=10)
    end

    # Row 1: P error over time
    ax3 = CM.Axis(fig[1, 3]; xlabel="Time", ylabel="Relative P-error", title="DUASE estimation error")
    err = compute_P_error(P_est, P_true)
    CM.lines!(ax3, 0:T-1, err; color=:coral, linewidth=2)

    # Row 2: P heatmaps at t=1, mid, end
    times = [1, T ÷ 2, T]
    for (col, t) in enumerate(times)
        ax_true = CM.Axis(fig[2, col]; aspect=1, title="P_true(t=" * string(t-1) * ")")
        CM.heatmap!(ax_true, P_true[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax_true)

        ax_est = CM.Axis(fig[3, col]; aspect=1, title="P̂_est(t=" * string(t-1) * ")")
        CM.heatmap!(ax_est, P_est[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax_est)
    end

    # Legend
    CM.Legend(fig[1, 4], [CM.MarkerElement(color=c, marker=:circle) for c in type_colors],
           ["Predator", "Prey", "Resource"]; framevisible=false)

    return fig
end

# ============================================================================
# Save/Load Functions
# ============================================================================

function save_data(X_true, A_obs, X_est)
    data = Dict(
        "X_true" => X_true,
        "A_obs" => A_obs,
        "X_est" => X_est,
        "config" => Dict(
            "N_PRED" => N_PRED, "N_PREY" => N_PREY, "N_RES" => N_RES,
            "D_EMBED" => D_EMBED, "T_TOTAL" => T_TOTAL, "SEED" => SEED,
            "HOLLING_ALPHA" => HOLLING_ALPHA, "HOLLING_BETA" => HOLLING_BETA
        )
    )
    serialize(joinpath(OUTPUT_DIR, "data.jls"), data)
    println("   Saved: " * joinpath(OUTPUT_DIR, "data.jls"))
end

function load_data()
    data = deserialize(joinpath(OUTPUT_DIR, "data.jls"))
    return data["X_true"], data["A_obs"], data["X_est"]
end

function save_model(ps_learned, nn, st, samples, sr_results)
    # Extract serializable SR results (hall_of_fame objects don't serialize well)
    sr_serializable = Dict{Tuple{Int,Int}, @NamedTuple{equation::String, complexity::Int, loss::Float64}}()
    for (key, val) in sr_results
        sr_serializable[key] = (equation=val.equation, complexity=val.complexity, loss=val.loss)
    end

    model = Dict(
        "ps_learned" => ps_learned,
        "nn" => nn,
        "st" => st,
        "samples" => samples,
        "sr_results" => sr_serializable
    )
    serialize(joinpath(OUTPUT_DIR, "model.jls"), model)
    println("   Saved: " * joinpath(OUTPUT_DIR, "model.jls"))
end

function save_evaluation(err_pred, err_duase, P_pred, P_true)
    eval_data = Dict(
        "err_pred" => err_pred,
        "err_duase" => err_duase,
        "P_pred" => P_pred,
        "P_true" => P_true
    )
    serialize(joinpath(OUTPUT_DIR, "evaluation.jls"), eval_data)
    println("   Saved: " * joinpath(OUTPUT_DIR, "evaluation.jls"))
end

# ============================================================================
# Main
# ============================================================================

function run_example4(; regenerate_data::Bool=false, skip_training::Bool=false)
    println("=" ^ 70)
    println("Example 4: Type-Specific Kernels with Symbolic Regression")
    println("=" ^ 70)

    # =========================================================================
    # 1. Generate or Load Data
    # =========================================================================
    data_file = joinpath(OUTPUT_DIR, "data.jls")

    if isfile(data_file) && !regenerate_data
        println("\n1. Loading existing data...")
        X_true, A_obs, X_est = load_data()
        P_true = [X * X' for X in X_true]
        P_est = [X * X' for X in X_est]
    else
        println("\n1. Generating data with Holling Type II predation...")

        X_true = generate_true_data()
        P_true = [X * X' for X in X_true]

        println("   Nodes: " * string(N_PRED) * " predators, " *
                string(N_PREY) * " prey, " * string(N_RES) * " resources")
        println("   True dynamics include:")
        println("     - Holling Type II: κ_PY(p) = " * @sprintf("%.2f", HOLLING_ALPHA) *
                "·p / (1 + " * @sprintf("%.1f", HOLLING_BETA) * "·p)")
        println("     - Linear: κ_YP(p) = -0.04·p (prey flee)")
        println("     - Constant: κ_PP = -0.008 (predator repulsion)")

        println("\n   Generating adjacency samples (K=10 per timestep)...")
        A_obs = generate_adjacencies(X_true; K=10)

        println("   Running DUASE estimation (window=5 for smoothing)...")
        X_est = duase_estimate(A_obs, D_EMBED; window=5)
        P_est = [X * X' for X in X_est]

        # Save data
        save_data(X_true, A_obs, X_est)
    end

    T_train = Int(floor(TRAIN_FRAC * T_TOTAL))
    X_train = X_est[1:T_train]

    println("   Training: t=1-" * string(T_train) * ", Validation: t=" *
            string(T_train+1) * "-" * string(T_TOTAL))

    # =========================================================================
    # 1b. Preliminary Trajectory Visualization (skip if data was loaded)
    # =========================================================================
    if regenerate_data || !isfile(joinpath(OUTPUT_DIR, "trajectories.png"))
        println("\n1b. Visualizing trajectories (checking gauge consistency)...")
        fig_traj = visualize_trajectories(X_true, X_est, P_true, P_est)
        CM.save(joinpath(OUTPUT_DIR, "trajectories.png"), fig_traj; px_per_unit=2)
        println("   Saved: " * joinpath(OUTPUT_DIR, "trajectories.png"))

        # Report trajectory statistics
        X_flat_true = vcat([X_true[t][:] for t in 1:length(X_true)]...)
        X_flat_est = vcat([X_est[t][:] for t in 1:length(X_est)]...)
        println("   True X range: [" * @sprintf("%.3f", minimum(X_flat_true)) * ", " *
                @sprintf("%.3f", maximum(X_flat_true)) * "]")
        println("   Est  X̂ range: [" * @sprintf("%.3f", minimum(X_flat_est)) * ", " *
                @sprintf("%.3f", maximum(X_flat_est)) * "]")
        println("   Mean P-error (DUASE): " * @sprintf("%.2f%%", 100*mean(compute_P_error(P_est, P_true))))
    else
        println("\n1b. Skipping visualization (trajectories.png exists)")
    end

    # =========================================================================
    # 2. Train Kernel NN
    # =========================================================================
    println("\n2. Training kernel NN (UDE: knows types + self-rates, learns κ)...")
    println("   Known self-rates (not learned):")
    type_names = ["Pred", "Prey", "Res"]
    for ti in 1:K_TYPES
        println("     a_" * type_names[ti] * " = " * @sprintf("%.4f", KNOWN_SELF_RATES[ti]))
    end

    ps_learned, nn, st = train_kernel_nn(X_train; epochs=500, lr=0.01, epochs_lbfgs=200)

    # =========================================================================
    # 3. Symbolic Regression (using SymbolicRegression.jl)
    # =========================================================================
    println("\n3. Symbolic Regression: Discovering kernel equations...")

    samples = sample_kernel_for_symreg(nn, ps_learned, st)

    # Run actual symbolic regression to discover equations
    # More iterations + larger maxsize for Holling Type II discovery
    sr_results = run_symbolic_regression(samples;
                                          niterations=100,
                                          maxsize=20,
                                          populations=30,
                                          timeout_seconds=120)

    # Display results
    format_sr_results(sr_results)

    # Check if anything resembling Holling was discovered for P→Y
    py_result = sr_results[(TYPE_P, TYPE_Y)]
    eq_str = py_result.equation
    has_division = occursin("/", eq_str)

    if has_division
        println("\n   ✓ Division-based form discovered for P→Y!")
        println("     True:      κ(p) = " * @sprintf("%.3f", HOLLING_ALPHA) *
                "·p / (1 + " * @sprintf("%.1f", HOLLING_BETA) * "·p)")
        println("     Discovered: " * replace(eq_str, "x1" => "p"))
    else
        println("\n   Note: No division-based form for P→Y (likely linear/constant)")
        println("     This may indicate insufficient training epochs.")
    end

    # Save model and SR results
    save_model(ps_learned, nn, st, samples, sr_results)

    # =========================================================================
    # 4. Evaluate
    # =========================================================================
    println("\n4. Evaluation (P is gauge-invariant, so starting point shouldn't matter)...")

    # Predict from true X(0)
    P_pred, X_pred = predict_P_trajectory(ps_learned, nn, st, X_true[1], T_TOTAL)
    err_pred = compute_P_error(P_pred, P_true)

    # DUASE baseline
    err_duase = compute_P_error(P_est, P_true)

    println("\n   P-error (UDE prediction vs true P):")
    println("     Training (t=1-" * string(T_train) * "):     " * @sprintf("%.2f%%", 100*mean(err_pred[1:T_train])))
    println("     Extrapolation (t=" * string(T_train+1) * "-" * string(T_TOTAL) * "): " * @sprintf("%.2f%%", 100*mean(err_pred[T_train+1:end])))
    println("     DUASE baseline:        " * @sprintf("%.2f%%", 100*mean(err_duase)))

    # Save evaluation results
    save_evaluation(err_pred, err_duase, P_pred, P_true)

    # =========================================================================
    # 5. Visualization
    # =========================================================================
    println("\n5. Generating visualizations...")

    fig = CM.Figure(size=(1400, 1000))

    # Row 1: True P (ground truth)
    # Row 2: UDE Predicted P
    for (col, t) in enumerate([1, T_train, T_TOTAL])
        ax1 = CM.Axis(fig[1, col]; aspect=1, title="t=" * string(t-1))
        CM.heatmap!(ax1, P_true[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax1)
        if col == 1
            CM.Label(fig[1, 0], "True P", rotation=pi/2, fontsize=14)
        end

        ax2 = CM.Axis(fig[2, col]; aspect=1)
        CM.heatmap!(ax2, P_pred[t]; colorrange=(0,1), colormap=:viridis)
        CM.hidedecorations!(ax2)
        if col == 1
            CM.Label(fig[2, 0], "UDE Predicted", rotation=pi/2, fontsize=14)
        end
    end

    CM.Colorbar(fig[1:2, 4]; colorrange=(0,1), colormap=:viridis, label="P(i,j)")

    # Row 3: P-error over time
    ax_err = CM.Axis(fig[3, 1:2]; xlabel="Time", ylabel="Relative P-error",
               title="UDE Prediction Error")
    CM.vspan!(ax_err, [0], [T_train-1]; color=(:green, 0.1))
    CM.lines!(ax_err, 0:T_TOTAL-1, err_duase; color=:coral, linestyle=:dash, linewidth=2, label="DUASE baseline")
    CM.lines!(ax_err, 0:T_TOTAL-1, err_pred; color=:blue, linewidth=2, label="UDE")
    CM.axislegend(ax_err; position=:lt)

    # Row 3 col 3: Kernel fits visualization
    ax_kern = CM.Axis(fig[3, 3]; xlabel="P_ij", ylabel="κ(P_ij)",
               title="Discovered Kernels (Predator)")

    p_range = range(0.1, 0.9, length=100)

    # True and learned κ_PY (should be Holling)
    κ_true_py = [κ_true(TYPE_P, TYPE_Y, p) for p in p_range]
    CM.lines!(ax_kern, p_range, κ_true_py; color=:red, linewidth=2, label="True κ_P→Y")

    data_py = samples[(TYPE_P, TYPE_Y)]
    CM.scatter!(ax_kern, data_py.p, data_py.κ; color=:red, alpha=0.3, markersize=5, label="Learned")

    # True and learned κ_PP (constant)
    κ_true_pp = [κ_true(TYPE_P, TYPE_P, p) for p in p_range]
    CM.lines!(ax_kern, p_range, κ_true_pp; color=:purple, linewidth=2, label="True κ_P→P")

    data_pp = samples[(TYPE_P, TYPE_P)]
    CM.scatter!(ax_kern, data_pp.p, data_pp.κ; color=:purple, alpha=0.3, markersize=5)

    CM.axislegend(ax_kern; position=:rt)

    # Save
    CM.save(joinpath(OUTPUT_DIR, "results.png"), fig; px_per_unit=2)
    println("   Saved: results/example4_type_kernel/results.png")

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("Summary")
    println("=" ^ 70)

    println("\n1. UDE STRUCTURE")
    println("   Known physics:")
    println("     - Node type membership (predator/prey/resource)")
    println("     - Self-rates a_P, a_Y, a_R (decay per type)")
    println("   Learned: 9 message kernels κ(P_ij, type_i, type_j) via NN")

    println("\n2. SYMBOLIC REGRESSION (SymbolicRegression.jl)")
    n_division = count(r -> occursin("/", r.equation), values(sr_results))
    n_simple = 9 - n_division
    println("   Equations with division (potential Holling): " * string(n_division))
    println("   Simple (linear/constant): " * string(n_simple))

    println("\n3. KEY RESULT")
    if has_division
        println("   Division-based form discovered for predator-prey!")
        println("   This may represent Holling Type II saturation.")
    else
        println("   No Holling-like form found yet.")
        println("   Try more training epochs or larger population to improve NN fit.")
    end

    return (
        ps_learned = ps_learned,
        nn = nn,
        st = st,
        sr_results = sr_results,
        samples = samples,
        errors = (pred=err_pred, duase=err_duase)
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_example4()
end
