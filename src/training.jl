"""
    Neural ODE training pipeline for RDPG dynamics.

    Provides configuration structs and training functions for learning
    continuous dynamics of temporal network embeddings.
"""

using Lux
using OrdinaryDiffEq
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using SciMLSensitivity
using ComponentArrays
using MLUtils
using Random
using Zygote
using ProgressMeter
using Dates: now, Second, format

export RDPGConfig, build_neural_ode, train_rdpg_node, predict_trajectory
export build_single_target_ode, train_single_target, predict_single_target
export SingleTargetConfig

"""
    RDPGConfig

Configuration for RDPG Neural ODE training.

# Fields
- `n::Int`: Number of nodes in the network
- `d::Int`: Embedding dimension
- `datasize::Int`: Number of training timesteps (default: 15)
- `hidden_sizes::Vector{Int}`: Hidden layer sizes (default: [128, 128, 64])
- `activation::Function`: Activation function (default: celu)
- `learning_rate::Float64`: Initial learning rate for Adam (default: 0.01)
- `epochs_adam::Int`: Epochs for Adam optimizer (default: 500)
- `epochs_lion::Int`: Epochs for Lion optimizer (default: 300)
- `lr_lion::Float64`: Learning rate for Lion (default: 5e-5)
- `use_bfgs::Bool`: Whether to refine with BFGS (default: false)
- `batch_size::Int`: Batch size for training (default: 3)
- `constraint_weight::Float64`: Weight for probability constraints (default: 1.0)
- `seed::Int`: Random seed for reproducibility (default: 1254)
"""
Base.@kwdef struct RDPGConfig
    n::Int
    d::Int
    datasize::Int = 15
    hidden_sizes::Vector{Int} = [128, 128, 64]
    activation::Function = celu
    learning_rate::Float64 = 0.01
    epochs_adam::Int = 500
    epochs_lion::Int = 300
    lr_lion::Float64 = 5e-5
    use_bfgs::Bool = false
    batch_size::Int = 3
    constraint_weight::Float64 = 1.0
    seed::Int = 1254
end

"""
    build_neural_ode(config::RDPGConfig) -> Tuple{Chain, ODEProblem, ComponentArray, NamedTuple}

Build the neural network and ODE problem for RDPG dynamics.

# Arguments
- `config`: Training configuration

# Returns
- `chain`: Lux neural network chain
- `prob`: ODEProblem ready for solving
- `params`: Initial parameters as ComponentArray
- `state`: Lux state (for stateful layers)
"""
function build_neural_ode(config::RDPGConfig)
    rng = Xoshiro(config.seed)

    input_dim = config.n * config.d
    output_dim = config.n * config.d

    # Build chain: input -> hidden layers -> output
    layers = Vector{Any}()
    prev_size = input_dim

    for hidden_size in config.hidden_sizes
        push!(layers, Dense(prev_size, hidden_size, config.activation))
        prev_size = hidden_size
    end
    push!(layers, Dense(prev_size, output_dim))

    chain = Chain(layers...)

    # Setup parameters
    params, state = Lux.setup(rng, chain)
    params_ca = ComponentArray(params)

    # Placeholder initial condition and timespan (will be overwritten)
    u0 = zeros(Float32, input_dim)
    tspan = (0.0f0, Float32(config.datasize - 1))

    # Create ODE function - state is passed via closure
    # Using StatefulLuxLayer pattern for better Zygote compatibility
    function dudt(u, p, t)
        y, _ = chain(u, p, state)
        return y
    end

    prob = ODEProblem{false}(dudt, u0, tspan, params_ca)

    return chain, prob, params_ca, state
end

"""
    predict_trajectory(prob::ODEProblem, params, u0, tspan, tsteps; sensealg=nothing)

Solve the Neural ODE and return predicted trajectory.

# Arguments
- `prob`: ODE problem
- `params`: Neural network parameters
- `u0`: Initial condition
- `tspan`: Time span tuple
- `tsteps`: Time points to save at
- `sensealg`: Sensitivity algorithm for gradients (default: InterpolatingAdjoint)

# Returns
- Array of predictions at each timestep
"""
function predict_trajectory(prob::ODEProblem, params, u0, tspan, tsteps;
                            sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    prob_remade = remake(prob; u0=u0, tspan=tspan, p=params)
    sol = solve(prob_remade, Tsit5(); saveat=tsteps, sensealg=sensealg)
    return Array(sol)
end

"""
    embedding_to_probability(L::AbstractMatrix, R_translation::AbstractMatrix) -> Matrix

Convert embedding L to probability matrix P = L * (L * R_translation)'.

# Arguments
- `L`: Left embedding matrix (n × d)
- `R_translation`: Translation matrix for L→R alignment

# Returns
- Probability matrix P (n × n)
"""
function embedding_to_probability(L::AbstractMatrix, R_translation::AbstractMatrix)
    return L * (L * R_translation)'
end

"""
    compute_loss(predictions, targets, translations, config::RDPGConfig)

Compute combined MSE + constraint loss.

# Arguments
- `predictions`: Predicted embeddings (d*n × timesteps)
- `targets`: Target embeddings (d*n × timesteps)
- `translations`: L→R translation matrices for each timestep
- `config`: Training configuration

# Returns
- Total loss value
"""
function compute_loss(predictions::AbstractMatrix, targets::AbstractMatrix,
                      translations::AbstractVector, config::RDPGConfig)
    # MSE loss on embeddings
    mse_loss = sum(abs2, targets .- predictions)

    # Constraint loss on probability matrices
    n, d = config.n, config.d
    constraint_loss = zero(eltype(predictions))

    for (i, col) in enumerate(eachcol(predictions))
        L = reshape(col, n, d)
        if i <= length(translations)
            P = embedding_to_probability(L, translations[i])
            constraint_loss += probability_constraint_loss(P)
        end
    end

    return mse_loss + config.constraint_weight * constraint_loss
end

"""
    train_rdpg_node(L_data::AbstractVector, config::RDPGConfig;
                    translations=nothing, verbose::Bool=true) -> ComponentArray

Train a Neural ODE to model RDPG embedding dynamics.

# Arguments
- `L_data`: Vector of left embedding matrices (one per timestep)
- `config`: Training configuration
- `translations`: Optional L→R translation matrices (computed if not provided)
- `verbose`: Print training progress (default: true)

# Returns
- Trained parameters as ComponentArray
"""
function train_rdpg_node(L_data::AbstractVector, config::RDPGConfig;
                         translations=nothing, verbose::Bool=true)
    # Prepare data
    n, d = config.n, config.d
    datasize = min(config.datasize, length(L_data))

    # Flatten embeddings to vectors (transpose so node dims are contiguous)
    u = Float32.(hcat([vec(L') for L in L_data[1:datasize]]...))
    u0 = u[:, 1]

    tspan = (0.0f0, Float32(datasize - 1))
    tsteps = range(tspan[1], tspan[2]; length=datasize)

    # Build model
    chain, prob, params_init, state = build_neural_ode(config)

    # Update problem with actual initial condition
    prob = remake(prob; u0=u0, tspan=tspan)

    # Sensitivity algorithm for efficient gradients
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())

    # Loss function for optimizer (full trajectory)
    function loss_fn(params, _)
        pred = Array(solve(prob, Tsit5(); p=params, saveat=tsteps, sensealg=sensealg))
        return sum(abs2, u .- pred)
    end

    # Setup optimization
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(loss_fn, adtype)
    optprob = Optimization.OptimizationProblem(optf, params_init, nothing)

    # Create progress callback factory
    function make_progress_callback(stage_name::String, max_iters::Int; show_progress::Bool=true)
        iter_count = Ref(0)
        best_loss = Ref(Inf)
        start_time = Ref(time())

        prog = show_progress ? Progress(max_iters;
            desc=stage_name * " ",
            showspeed=true,
            barlen=30,
            output=stderr) : nothing

        function callback(opt_state, loss)
            iter_count[] += 1
            if loss < best_loss[]
                best_loss[] = loss
            end

            if show_progress && !isnothing(prog)
                elapsed = time() - start_time[]
                avg_time = elapsed / iter_count[]
                remaining = (max_iters - iter_count[]) * avg_time

                ProgressMeter.update!(prog, iter_count[];
                    showvalues = [
                        (:loss, round(loss; digits=6)),
                        (:best, round(best_loss[]; digits=6)),
                        (:eta, format(now() + Second(round(Int, remaining)), "HH:MM:SS"))
                    ])
            end
            return false  # Don't stop
        end

        return callback, best_loss
    end

    # Stage 1: Adam
    if verbose
        println("\n" * "=" ^ 50)
        println("Stage 1: Adam optimizer")
        println("  Learning rate: " * string(config.learning_rate))
        println("  Iterations: " * string(config.epochs_adam))
        println("=" ^ 50)
    end

    adam_callback, adam_best = make_progress_callback("Adam", config.epochs_adam; show_progress=verbose)
    result = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(config.learning_rate);
        callback=adam_callback,
        maxiters=config.epochs_adam
    )

    if verbose
        println("\nAdam complete. Best loss: " * string(round(adam_best[]; digits=6)))
    end

    # Stage 2: Lion (fine-tuning with smaller learning rate)
    if config.epochs_lion > 0
        if verbose
            println("\n" * "=" ^ 50)
            println("Stage 2: Lion optimizer (fine-tuning)")
            println("  Learning rate: " * string(config.lr_lion))
            println("  Iterations: " * string(config.epochs_lion))
            println("=" ^ 50)
        end

        optprob2 = Optimization.OptimizationProblem(optf, result.u, nothing)
        lion_callback, lion_best = make_progress_callback("Lion", config.epochs_lion; show_progress=verbose)
        result = Optimization.solve(
            optprob2,
            OptimizationOptimisers.Lion(config.lr_lion, (0.9, 0.999));
            callback=lion_callback,
            maxiters=config.epochs_lion
        )
        if verbose
            println("\nLion complete. Best loss: " * string(round(lion_best[]; digits=6)))
        end
    end

    # Stage 3: BFGS (optional, for final refinement)
    if config.use_bfgs
        if verbose
            println("\n" * "=" ^ 50)
            println("Stage 3: BFGS refinement")
            println("  Iterations: 100")
            println("=" ^ 50)
        end

        optprob3 = Optimization.OptimizationProblem(optf, result.u, nothing)
        bfgs_callback, bfgs_best = make_progress_callback("BFGS", 100; show_progress=verbose)
        result = Optimization.solve(
            optprob3,
            Optim.BFGS(; initial_stepnorm=0.01);
            callback=bfgs_callback,
            maxiters=100
        )
        if verbose
            println("\nBFGS complete. Final loss: " * string(round(bfgs_best[]; digits=6)))
        end
    end

    return result.u
end

# ============================================================================
# Single-Target Node Prediction (Original Thesis Approach)
# ============================================================================
#
# The core insight: predict ONE node's trajectory given the rest of the network.
# This is a much more tractable problem than learning all dynamics at once.
#
# Input: (n-1)*d context dimensions (other nodes)
# Output: d dimensions (target node)
# ============================================================================

"""
    SingleTargetConfig

Configuration for single-target node prediction.

# Fields
- `n::Int`: Total number of nodes (before removing target)
- `d::Int`: Embedding dimension
- `target_node::Int`: Index of node to predict (1 to n)
- `datasize::Int`: Number of training timesteps
- `hidden_sizes::Vector{Int}`: Hidden layer sizes
- `activation::Function`: Activation function
- `learning_rate::Float64`: Learning rate for Adam
- `epochs::Int`: Total training epochs
- `seed::Int`: Random seed
"""
Base.@kwdef struct SingleTargetConfig
    n::Int
    d::Int
    target_node::Int = 1
    datasize::Int = 15
    hidden_sizes::Vector{Int} = [64, 64, 32]  # Smaller network for single target
    activation::Function = celu
    learning_rate::Float64 = 0.01
    epochs::Int = 800
    seed::Int = 1254
end

"""
    build_single_target_ode(config::SingleTargetConfig)

Build Neural ODE for single-target prediction.

Input: concatenation of context (other nodes) + current target state
Output: derivative of target node embedding

This allows the target node dynamics to depend on the rest of the network.
"""
function build_single_target_ode(config::SingleTargetConfig)
    rng = Xoshiro(config.seed)

    # Input: context nodes + target node = (n-1)*d + d = n*d
    # But we condition on the CURRENT context, so input is just target (d dims)
    # with context passed through closure
    # Actually, let's make input = (n-1)*d + d = n*d total
    # The network learns: f(context, target) -> d(target)/dt

    context_dim = (config.n - 1) * config.d
    target_dim = config.d
    input_dim = context_dim + target_dim
    output_dim = target_dim  # Only predict target's derivative

    layers = Vector{Any}()
    prev_size = input_dim

    for hidden_size in config.hidden_sizes
        push!(layers, Dense(prev_size, hidden_size, config.activation))
        prev_size = hidden_size
    end
    push!(layers, Dense(prev_size, output_dim))

    chain = Chain(layers...)
    params, state = Lux.setup(rng, chain)
    params_ca = ComponentArray(params)

    return chain, params_ca, state
end

"""
    train_single_target(L_data::AbstractVector, config::SingleTargetConfig;
                        verbose::Bool=true) -> NamedTuple

Train a Neural ODE to predict a single target node's trajectory.

# Arguments
- `L_data`: Vector of left embedding matrices (n × d), one per timestep
- `config`: Training configuration with target_node specified

# Returns
- NamedTuple with :params (trained parameters), :chain, :state
"""
function train_single_target(L_data::AbstractVector, config::SingleTargetConfig;
                             verbose::Bool=true)
    n, d = config.n, config.d
    target_idx = config.target_node
    datasize = min(config.datasize, length(L_data))

    # Separate target and context for each timestep
    # L_data[t] is n × d matrix
    target_traj = Float32.(hcat([vec(L_data[t][target_idx, :]) for t in 1:datasize]...))  # d × T
    context_traj = Float32.(hcat([
        vec(L_data[t][[i for i in 1:n if i != target_idx], :]')
        for t in 1:datasize
    ]...))  # (n-1)*d × T

    # Build model
    chain, params_init, state = build_single_target_ode(config)

    # Initial condition and timespan
    u0 = target_traj[:, 1]  # d-dimensional
    tspan = (0.0f0, Float32(datasize - 1))
    tsteps = range(tspan[1], tspan[2]; length=datasize)

    # ODE function that interpolates context
    function dudt(u, p, t)
        # Interpolate context at time t
        t_idx = min(max(1, Int(floor(t)) + 1), datasize)
        context = context_traj[:, t_idx]

        # Concatenate context and current state
        input = vcat(context, u)
        y, _ = chain(input, p, state)
        return y
    end

    prob = ODEProblem{false}(dudt, u0, tspan, params_init)

    # Sensitivity algorithm
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())

    # Loss function
    function loss_fn(params, _)
        pred = Array(solve(prob, Tsit5(); p=params, saveat=tsteps, sensealg=sensealg))
        return sum(abs2, target_traj .- pred)
    end

    # Optimization setup
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(loss_fn, adtype)
    optprob = Optimization.OptimizationProblem(optf, params_init, nothing)

    # Progress tracking
    iter_count = Ref(0)
    best_loss = Ref(Inf)
    start_time = Ref(time())

    prog = verbose ? Progress(config.epochs;
        desc="Training target " * string(target_idx) * " ",
        showspeed=true,
        barlen=30,
        output=stderr) : nothing

    function callback(opt_state, loss)
        iter_count[] += 1
        if loss < best_loss[]
            best_loss[] = loss
        end

        if verbose && !isnothing(prog)
            elapsed = time() - start_time[]
            avg_time = elapsed / iter_count[]
            remaining = (config.epochs - iter_count[]) * avg_time

            ProgressMeter.update!(prog, iter_count[];
                showvalues = [
                    (:loss, round(loss; digits=4)),
                    (:best, round(best_loss[]; digits=4)),
                    (:eta, format(now() + Second(round(Int, remaining)), "HH:MM:SS"))
                ])
        end
        return false
    end

    if verbose
        println("\nTraining single-target prediction for node " * string(target_idx))
        println("  Context: " * string(n - 1) * " nodes × " * string(d) * " dims = " * string((n-1)*d) * " context dims")
        println("  Target: " * string(d) * " dims")
        println("  Timesteps: " * string(datasize))
    end

    result = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(config.learning_rate);
        callback=callback,
        maxiters=config.epochs
    )

    if verbose
        println("\nTraining complete. Best loss: " * string(round(best_loss[]; digits=4)))
    end

    return (params=result.u, chain=chain, state=state,
            context_traj=context_traj, target_traj=target_traj,
            config=config)
end

"""
    predict_single_target(trained::NamedTuple, L_data::AbstractVector,
                          timesteps::Int=30) -> Matrix

Generate predictions for the target node using trained model.

# Returns
- Predicted trajectory matrix (d × timesteps)
"""
function predict_single_target(trained::NamedTuple, L_data::AbstractVector, timesteps::Int=30)
    config = trained.config
    n, d = config.n, config.d
    target_idx = config.target_node

    # Get context trajectory (all nodes except target)
    context_traj = Float32.(hcat([
        vec(L_data[t][[i for i in 1:n if i != target_idx], :]')
        for t in 1:min(timesteps, length(L_data))
    ]...))

    # Initial condition
    u0 = Float32.(L_data[1][target_idx, :])

    tspan = (0.0f0, Float32(timesteps - 1))
    tsteps = range(tspan[1], tspan[2]; length=timesteps)

    # ODE with interpolated context
    function dudt(u, p, t)
        t_idx = min(max(1, Int(floor(t)) + 1), size(context_traj, 2))
        context = context_traj[:, t_idx]
        input = vcat(context, u)
        y, _ = trained.chain(input, p, trained.state)
        return y
    end

    prob = ODEProblem{false}(dudt, u0, tspan, trained.params)
    sol = solve(prob, Tsit5(); saveat=tsteps)

    return Array(sol)
end
