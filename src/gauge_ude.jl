"""
    Gauge-consistent UDE architecture for RDPG dynamics.

    Implements the theoretically-motivated form: Ẋ = N(P)X
    where N(P) is a symmetric matrix depending on P = XX'.

    This architecture automatically eliminates gauge freedom since
    symmetric N cannot produce invisible dynamics (rotation around origin).
"""

using Lux
using OrdinaryDiffEq
using Optimization
using OptimizationOptimisers
using SciMLSensitivity
using ComponentArrays
using LinearAlgebra
using Random
using Zygote
using ProgressMeter
using Dates: now, Second, format

export SymmetricNConfig, PolynomialNConfig, KernelNConfig
export build_polynomial_N_ode, build_kernel_N_ode, build_symmetric_nn_ode
export train_gauge_ude, predict_gauge_trajectory

# ============================================================================
# Configuration Types
# ============================================================================

"""
    PolynomialNConfig

Configuration for polynomial N(P) = α₀I + α₁P + α₂P² + ...

This is the most parsimonious parameterization with k+1 learnable scalars.
"""
Base.@kwdef struct PolynomialNConfig
    n::Int                          # Number of nodes
    d::Int                          # Embedding dimension
    degree::Int = 2                 # Polynomial degree (0 = identity only)
    datasize::Int = 15              # Training timesteps
    learning_rate::Float64 = 0.01
    epochs::Int = 500
    constraint_weight::Float64 = 1.0
    seed::Int = 1254
end

"""
    KernelNConfig

Configuration for kernel N(P) where N_ij = κ(P_ij) for a learned function κ.

Uses a small neural network to learn the pairwise kernel.
"""
Base.@kwdef struct KernelNConfig
    n::Int                          # Number of nodes
    d::Int                          # Embedding dimension
    kernel_hidden::Vector{Int} = [16, 16]  # Hidden layers for kernel NN
    datasize::Int = 15
    learning_rate::Float64 = 0.01
    epochs::Int = 500
    constraint_weight::Float64 = 1.0
    seed::Int = 1254
end

"""
    SymmetricNConfig

Configuration for general symmetric N(P) with NN outputting upper triangle.

More expressive but less parsimonious than polynomial or kernel forms.
"""
Base.@kwdef struct SymmetricNConfig
    n::Int
    d::Int
    hidden_sizes::Vector{Int} = [64, 64]
    datasize::Int = 15
    learning_rate::Float64 = 0.01
    epochs::Int = 500
    constraint_weight::Float64 = 1.0
    seed::Int = 1254
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    upper_tri_to_symmetric(v::AbstractVector, n::Int) -> Matrix

Convert a vector of upper triangular elements to a symmetric matrix.
Vector length should be n(n+1)/2.
"""
function upper_tri_to_symmetric(v::AbstractVector{T}, n::Int) where T
    N = zeros(T, n, n)
    k = 1
    for i in 1:n
        for j in i:n
            N[i, j] = v[k]
            N[j, i] = v[k]
            k += 1
        end
    end
    return N
end

"""
    compute_P(X::AbstractMatrix) -> Matrix

Compute probability matrix P = XX' from embedding X (n × d).
"""
function compute_P(X::AbstractMatrix)
    return X * X'
end

# ============================================================================
# Polynomial N(P) Architecture
# ============================================================================

"""
    build_polynomial_N_dynamics(config::PolynomialNConfig)

Build the polynomial N(P)X dynamics.

Returns a function (X, α) -> Ẋ where α are the polynomial coefficients.
"""
function build_polynomial_N_ode(config::PolynomialNConfig)
    rng = Xoshiro(config.seed)
    n, d = config.n, config.d
    degree = config.degree

    # Initialize coefficients: α₀, α₁, ..., α_degree
    # Start with small values, α₀ slightly negative for stability
    α_init = Float32.(randn(rng, degree + 1) * 0.1)
    α_init[1] = -0.1f0  # Slight contraction by default

    params = ComponentArray(α=α_init)

    # Dynamics function: Ẋ = N(P)X where N = Σ αₖ Pᵏ
    function dudt(u, p, t)
        # Reshape flattened u to X matrix (n × d)
        X = reshape(u, d, n)'  # Now n × d
        P = X * X'  # n × n probability matrix

        # Build N = α₀I + α₁P + α₂P² + ...
        N = p.α[1] * I(n)
        Pk = P
        for k in 2:(degree + 1)
            N = N + p.α[k] * Pk
            Pk = Pk * P
        end

        # Compute Ẋ = NX
        dX = N * X  # n × d

        # Flatten back (transpose to match input layout)
        return vec(dX')
    end

    # Initial condition placeholder
    u0 = zeros(Float32, n * d)
    tspan = (0.0f0, Float32(config.datasize - 1))

    prob = ODEProblem{false}(dudt, u0, tspan, params)

    return prob, params
end

# ============================================================================
# Kernel N(P) Architecture
# ============================================================================

"""
    build_kernel_N_ode(config::KernelNConfig)

Build the kernel N(P)X dynamics where N_ij = κ(P_ij).

The kernel κ is a small neural network mapping scalars to scalars.
"""
function build_kernel_N_ode(config::KernelNConfig)
    rng = Xoshiro(config.seed)
    n, d = config.n, config.d

    # Build kernel network: κ: ℝ → ℝ
    layers = Vector{Any}()
    prev_size = 1
    for hidden_size in config.kernel_hidden
        push!(layers, Dense(prev_size, hidden_size, tanh))
        prev_size = hidden_size
    end
    push!(layers, Dense(prev_size, 1))

    kernel_chain = Chain(layers...)
    kernel_params, kernel_state = Lux.setup(rng, kernel_chain)

    # Also learn diagonal separately: h(P_ii)
    diag_chain = Chain(Dense(1, 8, tanh), Dense(8, 1))
    diag_params, diag_state = Lux.setup(rng, diag_chain)

    params = ComponentArray(kernel=ComponentArray(kernel_params),
                           diag=ComponentArray(diag_params))

    # Dynamics function
    function dudt(u, p, t)
        X = reshape(u, d, n)'  # n × d
        P = X * X'  # n × n

        # Build N matrix using kernel
        # N_ij = κ(P_ij) for i ≠ j
        # N_ii = h(P_ii)
        N = similar(P)

        for i in 1:n
            # Diagonal: use separate network
            P_ii = reshape([P[i, i]], 1, 1)
            h_val, _ = diag_chain(P_ii, p.diag, diag_state)
            N[i, i] = h_val[1]

            for j in (i+1):n
                # Off-diagonal: use kernel
                P_ij = reshape([P[i, j]], 1, 1)
                κ_val, _ = kernel_chain(P_ij, p.kernel, kernel_state)
                N[i, j] = κ_val[1]
                N[j, i] = κ_val[1]  # Symmetric
            end
        end

        dX = N * X
        return vec(dX')
    end

    u0 = zeros(Float32, n * d)
    tspan = (0.0f0, Float32(config.datasize - 1))

    prob = ODEProblem{false}(dudt, u0, tspan, params)

    return prob, params, (kernel_chain, diag_chain, kernel_state, diag_state)
end

# ============================================================================
# General Symmetric NN Architecture
# ============================================================================

"""
    build_symmetric_nn_ode(config::SymmetricNConfig)

Build general symmetric N(P) where NN outputs upper triangle of N.

More expressive than polynomial/kernel but less interpretable.
"""
function build_symmetric_nn_ode(config::SymmetricNConfig)
    rng = Xoshiro(config.seed)
    n, d = config.n, config.d

    # Input: flattened P (n² values, or just upper triangle n(n+1)/2)
    # Output: upper triangle of N (n(n+1)/2 values)
    input_dim = div(n * (n + 1), 2)  # Upper triangle of P
    output_dim = div(n * (n + 1), 2)  # Upper triangle of N

    layers = Vector{Any}()
    prev_size = input_dim
    for hidden_size in config.hidden_sizes
        push!(layers, Dense(prev_size, hidden_size, celu))
        prev_size = hidden_size
    end
    push!(layers, Dense(prev_size, output_dim))

    chain = Chain(layers...)
    nn_params, nn_state = Lux.setup(rng, chain)
    params = ComponentArray(nn=ComponentArray(nn_params))

    # Helper to extract upper triangle as vector
    function upper_tri_vec(M::AbstractMatrix{T}) where T
        n = size(M, 1)
        v = Vector{T}(undef, div(n * (n + 1), 2))
        k = 1
        for i in 1:n
            for j in i:n
                v[k] = M[i, j]
                k += 1
            end
        end
        return v
    end

    function dudt(u, p, t)
        X = reshape(u, d, n)'  # n × d
        P = X * X'  # n × n

        # Extract upper triangle of P as input
        P_upper = upper_tri_vec(P)

        # Get N upper triangle from NN
        N_upper, _ = chain(P_upper, p.nn, nn_state)

        # Reconstruct symmetric N
        N = upper_tri_to_symmetric(N_upper, n)

        dX = N * X
        return vec(dX')
    end

    u0 = zeros(Float32, n * d)
    tspan = (0.0f0, Float32(config.datasize - 1))

    prob = ODEProblem{false}(dudt, u0, tspan, params)

    return prob, params, (chain, nn_state)
end

# ============================================================================
# Training Function
# ============================================================================

"""
    train_gauge_ude(L_data::AbstractVector, prob::ODEProblem, params_init;
                    config, verbose::Bool=true) -> ComponentArray

Train a gauge-consistent UDE model.

# Arguments
- `L_data`: Vector of embedding matrices (n × d per timestep)
- `prob`: ODE problem from build_*_ode functions
- `params_init`: Initial parameters
- `config`: Configuration struct (any of the N configs)
- `verbose`: Print progress

# Returns
- Trained parameters
"""
function train_gauge_ude(L_data::AbstractVector, prob::ODEProblem, params_init;
                         config, verbose::Bool=true)
    n, d = config.n, config.d
    datasize = min(config.datasize, length(L_data))

    # Flatten embeddings (transpose so node dims are contiguous)
    u = Float32.(hcat([vec(L') for L in L_data[1:datasize]]...))
    u0 = u[:, 1]

    tspan = (0.0f0, Float32(datasize - 1))
    tsteps = range(tspan[1], tspan[2]; length=datasize)

    # Update problem with actual u0
    prob = remake(prob; u0=u0, tspan=tspan)

    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())

    # Loss function with probability constraints
    function loss_fn(params, _)
        pred = Array(solve(prob, Tsit5(); p=params, saveat=tsteps, sensealg=sensealg))

        mse = sum(abs2, u .- pred)

        # Constraint loss on probability matrices
        constraint = zero(eltype(pred))
        for col in eachcol(pred)
            X = reshape(col, d, n)'
            P = X * X'
            # Penalize P_ij < 0 or P_ij > 1
            constraint += sum(max.(-P, 0)) + sum(max.(P .- 1, 0))
        end

        return mse + config.constraint_weight * constraint
    end

    # Optimization setup
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(loss_fn, adtype)
    optprob = Optimization.OptimizationProblem(optf, params_init, nothing)

    # Progress callback
    iter_count = Ref(0)
    best_loss = Ref(Inf)

    function callback(opt_state, loss)
        iter_count[] += 1
        if loss < best_loss[]
            best_loss[] = loss
        end

        if verbose && iter_count[] % 50 == 0
            println("  iter " * string(iter_count[]) * ": loss=" * string(round(loss; digits=4)))
        end
        return false
    end

    if verbose
        println("\nTraining gauge-consistent UDE")
        println("  Architecture: " * string(typeof(config).name.name))
        println("  n=" * string(n) * ", d=" * string(d))
        println("  Epochs: " * string(config.epochs))
    end

    result = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(config.learning_rate);
        callback=callback,
        maxiters=config.epochs
    )

    if verbose
        println("  Final loss: " * string(round(best_loss[]; digits=6)))
    end

    return result.u
end

"""
    predict_gauge_trajectory(prob::ODEProblem, params, u0, timesteps::Int)

Predict trajectory using trained gauge-consistent model.
"""
function predict_gauge_trajectory(prob::ODEProblem, params, u0, timesteps::Int)
    tspan = (0.0f0, Float32(timesteps - 1))
    tsteps = range(tspan[1], tspan[2]; length=timesteps)

    prob_remade = remake(prob; u0=u0, tspan=tspan, p=params)
    sol = solve(prob_remade, Tsit5(); saveat=tsteps)

    return Array(sol)
end
