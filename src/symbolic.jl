"""
    Symbolic regression for extracting interpretable equations from Neural ODEs.

    After training a Neural ODE, use symbolic regression to find closed-form
    expressions that approximate the learned dynamics.
"""

import SymbolicRegression: Options, equation_search, calculate_pareto_frontier, eval_tree_array
using OrdinaryDiffEq

export extract_symbolic_dynamics, predict_symbolic, SymbolicDynamics

"""
    SymbolicDynamics

Container for symbolic regression results.

# Fields
- `trees`: Vector of symbolic expression trees (one per output dimension)
- `options`: SymbolicRegression options used
- `pareto_fronts`: Pareto-optimal solutions for each dimension
- `complexity`: Complexity of chosen expressions
"""
struct SymbolicDynamics
    trees::Vector
    options
    pareto_fronts::Vector
    complexity::Vector{Int}
end

"""
    extract_symbolic_dynamics(trained::NamedTuple, L_data::AbstractVector;
                               niterations::Int=40,
                               max_complexity::Int=20,
                               binary_operators=[+, -, *],
                               unary_operators=[cos, sin],
                               verbose::Bool=true) -> SymbolicDynamics

Extract symbolic equations from trained Neural ODE.

# Arguments
- `trained`: Output from `train_single_target`
- `L_data`: Training data (for generating derivative samples)
- `niterations`: Number of symbolic regression iterations
- `max_complexity`: Maximum expression complexity
- `binary_operators`: Allowed binary operators
- `unary_operators`: Allowed unary operators (cos, sin for periodic dynamics)

# Returns
- SymbolicDynamics with extracted equations
"""
function extract_symbolic_dynamics(trained::NamedTuple, L_data::AbstractVector;
                                    niterations::Int=40,
                                    max_complexity::Int=20,
                                    binary_operators=[+, -, *],
                                    unary_operators=[cos, sin],
                                    verbose::Bool=true)
    config = trained.config
    n, d = config.n, config.d
    target_idx = config.target_node
    datasize = size(trained.context_traj, 2)

    if verbose
        println("\nExtracting symbolic dynamics for node " * string(target_idx))
        println("  Generating training data from Neural ODE...")
    end

    # Generate input-output pairs from the trained neural network
    # Input: [context, target_state]
    # Output: d(target)/dt from the neural network
    X_list = Vector{Vector{Float32}}()
    Y_list = Vector{Vector{Float32}}()

    for t in 1:datasize
        context = trained.context_traj[:, t]
        target_state = trained.target_traj[:, t]
        input = vcat(context, target_state)

        # Get derivative from neural network
        derivative, _ = trained.chain(input, trained.params, trained.state)

        push!(X_list, input)
        push!(Y_list, derivative)
    end

    X = Float64.(hcat(X_list...))  # (context_dim + d) × timesteps
    Y = Float64.(hcat(Y_list...))  # d × timesteps

    if verbose
        println("  Input dimension: " * string(size(X, 1)))
        println("  Output dimension: " * string(size(Y, 1)))
        println("  Samples: " * string(size(X, 2)))
        println("\nRunning symbolic regression...")
    end

    # Configure symbolic regression (simpler config for compatibility)
    options = Options(
        populations=25,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        should_optimize_constants=true,
        should_simplify=true
    )

    # Run symbolic regression
    hall_of_fame = equation_search(
        X, Y;
        niterations=niterations,
        options=options,
        parallelism=:multithreading
    )

    # Get Pareto-optimal solutions
    pareto_fronts = calculate_pareto_frontier.(hall_of_fame)

    # Select best expressions (balance accuracy vs complexity)
    # Choose the expression with best loss among those with complexity <= max_complexity
    trees = Vector{Any}()
    complexities = Vector{Int}()

    for (dim, front) in enumerate(pareto_fronts)
        if isempty(front)
            error("No symbolic expressions found for dimension " * string(dim))
        end

        # Select the best (lowest loss) within reasonable complexity
        best_idx = 1
        best_loss = Inf

        for (idx, entry) in enumerate(front)
            if entry.complexity <= max_complexity && entry.loss < best_loss
                best_loss = entry.loss
                best_idx = idx
            end
        end

        push!(trees, front[best_idx].tree)
        push!(complexities, front[best_idx].complexity)

        if verbose
            println("  Dim " * string(dim) * ": complexity=" * string(complexities[end]) *
                    ", loss=" * string(round(best_loss; digits=6)))
        end
    end

    if verbose
        println("\nSymbolic extraction complete.")
    end

    return SymbolicDynamics(trees, options, pareto_fronts, complexities)
end

"""
    predict_symbolic(symbolic::SymbolicDynamics, L_data::AbstractVector,
                     target_idx::Int, n::Int, d::Int,
                     timesteps::Int=30) -> Matrix

Generate predictions using extracted symbolic dynamics.

# Returns
- Predicted trajectory (d × timesteps)
"""
function predict_symbolic(symbolic::SymbolicDynamics, L_data::AbstractVector,
                          target_idx::Int, n::Int, d::Int;
                          timesteps::Int=30)
    # Get context trajectory
    context_traj = Float64.(hcat([
        vec(L_data[t][[i for i in 1:n if i != target_idx], :]')
        for t in 1:min(timesteps, length(L_data))
    ]...))

    # Initial condition
    u0 = Float64.(L_data[1][target_idx, :])

    tspan = (0.0, Float64(timesteps - 1))
    tsteps = range(tspan[1], tspan[2]; length=timesteps)

    # Create ODE function from symbolic expressions
    function dudt_symbolic(u, p, t)
        t_idx = min(max(1, Int(floor(t)) + 1), size(context_traj, 2))
        context = context_traj[:, t_idx]
        input = vcat(context, u)

        # Evaluate each symbolic tree
        derivatives = zeros(d)
        for (dim, tree) in enumerate(symbolic.trees)
            result = eval_tree_array(tree, reshape(input, :, 1), symbolic.options)
            derivatives[dim] = result[1][1]
        end

        return derivatives
    end

    prob = ODEProblem(dudt_symbolic, u0, tspan)
    sol = solve(prob, Tsit5(); saveat=tsteps)

    return Array(sol)
end

"""
    print_symbolic_equations(symbolic::SymbolicDynamics; variable_names=nothing)

Pretty-print the extracted symbolic equations.
"""
function print_symbolic_equations(symbolic::SymbolicDynamics; variable_names=nothing)
    println("\nExtracted Symbolic Equations:")
    println("=" ^ 50)

    for (dim, tree) in enumerate(symbolic.trees)
        println("d(target_" * string(dim) * ")/dt = " * string(tree))
    end

    println("=" ^ 50)
end
