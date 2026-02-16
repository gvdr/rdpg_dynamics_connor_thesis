#!/usr/bin/env -S julia --project
"""
Example 2: Homogeneous Dynamics (4 Options)

All nodes follow the SAME dynamics rule - testing which formulation works best.
This determines the dynamics for Examples 3a/3b.

Dynamics Options:
A) pairwise      - Pairwise attraction/repulsion (Lennard-Jones style)
B) cohesion      - Cohesion to centroid + short-range repulsion
C) radial        - Radial equilibrium (nodes find optimal distance from center)
D) kuramoto      - Kuramoto-like angular + radial dynamics

For each dynamics, we train:
- Pure NN
- UDE (with known structure)

Usage:
    julia --project scripts/example2_homogeneous_v2.jl generate <dynamics>
    julia --project scripts/example2_homogeneous_v2.jl estimate <dynamics>
    julia --project scripts/example2_homogeneous_v2.jl train <dynamics> <model>
    julia --project scripts/example2_homogeneous_v2.jl evaluate <dynamics>
    julia --project scripts/example2_homogeneous_v2.jl visualize <dynamics>
    julia --project scripts/example2_homogeneous_v2.jl all <dynamics>
    julia --project scripts/example2_homogeneous_v2.jl compare   # Compare all dynamics

Where <dynamics> is: pairwise, cohesion, radial, kuramoto
      <model> is: pure_nn, ude, or all
"""

using RDPGDynamics

# Additional imports needed for dynamics definitions
using LinearAlgebra: norm, diagind
using Statistics: mean
using Random
using Lux
using ComponentArrays: ComponentArray

# =============================================================================
# Configuration
# =============================================================================

const N_NODES = 60
const D = 2

# Time parameters
const T_END = 40.0
const DT = 1.0

# Training
const K_SAMPLES = 100
const TRAIN_FRACTION = 0.75
const EPOCHS_ADAM = 600
const EPOCHS_BFGS = 100

# =============================================================================
# Dynamics A: Pairwise Attraction/Repulsion (Lennard-Jones style)
# =============================================================================

module DynamicsPairwise
    using LinearAlgebra
    using Statistics
    using Random
    using RDPGDynamics: project_embedding_to_Bd_plus

    const N = Main.N_NODES
    const D = Main.D

    # Parameters (Lennard-Jones like: -a/r + b/r³)
    const A_ATTRACT = 0.02      # Long-range attraction
    const B_REPEL = 0.0008      # Short-range repulsion
    const R_SOFT = 0.05         # Soft cutoff for numerical stability

    # B^d_+ barrier parameters
    const BOUND_K = 0.5         # Barrier strength
    const BOUND_SCALE = 0.02    # Barrier smoothness

    # Soft B^d_+ barrier force (matches library constraints: x≥0, ||x||≤1)
    function bdplus_barrier!(dX, X, n, d)
        for i in 1:n
            # Positive orthant barrier: exponential repulsion from 0
            for dim in 1:d
                dX[i, dim] += BOUND_K * exp(-X[i, dim] / BOUND_SCALE)
            end
            # Unit ball barrier: exponential repulsion from ||x||=1
            norm_i = norm(X[i, :])
            if norm_i > 0.01
                dX[i, :] .-= BOUND_K * exp((norm_i - 1.0) / BOUND_SCALE) .* X[i, :] ./ norm_i
            end
        end
    end

    function true_dynamics!(du, u, p, t)
        X = reshape(u, N, D)

        # Vectorized pairwise distances
        sq_norms = sum(X .^ 2, dims=2)
        D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
        D_mat = sqrt.(max.(D_sq, R_SOFT^2))

        # Force magnitudes: f(d) = -a/d + b/d³
        F_mag = -A_ATTRACT ./ D_mat .+ B_REPEL ./ (D_mat .^ 3)
        F_mag[diagind(F_mag)] .= 0.0

        dX = zeros(N, D)
        for dim in 1:D
            diff_dim = X[:, dim] .- X[:, dim]'
            dX[:, dim] = sum(F_mag .* diff_dim ./ D_mat, dims=2)
        end

        bdplus_barrier!(dX, X, N, D)
        du .= vec(dX)
    end

    function initial_positions(seed::Int)
        rng = Random.MersenneTwister(seed)
        X0 = 0.3 .+ 0.4 .* rand(rng, N, D)
        return project_embedding_to_Bd_plus(X0)
    end

    # UDE: mean-field known structure (O(n)), NN learns pairwise corrections
    function make_ude_dynamics(nn, st, n, d)
        nd = n * d

        function dynamics(u::AbstractVector{T}, p, t) where T
            X = reshape(u, n, d)

            # Known: mean-field cohesion (O(n))
            X_bar = mean(X, dims=1)
            dX_known = -T(0.05) .* (X .- X_bar)

            # NN learns pairwise corrections
            u_reshape = reshape(u, nd, 1)
            correction, _ = nn(u_reshape, p, st)
            return vec(dX_known) .+ vec(correction)
        end
        return dynamics
    end
end

# =============================================================================
# Dynamics B: Cohesion + Repulsion (Swarming)
# =============================================================================

module DynamicsCohesion
    using LinearAlgebra
    using Statistics
    using Random
    using RDPGDynamics: project_embedding_to_Bd_plus

    const N = Main.N_NODES
    const D = Main.D

    const K_COHESION = 0.1      # Attraction to centroid
    const K_REPEL = 0.002       # Short-range repulsion
    const R_SOFT = 0.05         # Soft cutoff
    const BOUND_K = 0.5
    const BOUND_SCALE = 0.02

    # Soft B^d_+ barrier (x≥0, ||x||≤1)
    function bdplus_barrier!(dX, X, n, d)
        for i in 1:n
            for dim in 1:d
                dX[i, dim] += BOUND_K * exp(-X[i, dim] / BOUND_SCALE)
            end
            norm_i = norm(X[i, :])
            if norm_i > 0.01
                dX[i, :] .-= BOUND_K * exp((norm_i - 1.0) / BOUND_SCALE) .* X[i, :] ./ norm_i
            end
        end
    end

    function true_dynamics!(du, u, p, t)
        X = reshape(u, N, D)

        # Cohesion: attraction to centroid (O(n))
        X_bar = mean(X, dims=1)
        dX = -K_COHESION .* (X .- X_bar)

        # Vectorized pairwise repulsion
        sq_norms = sum(X .^ 2, dims=2)
        D_sq = sq_norms .+ sq_norms' .- 2 .* (X * X')
        D_mat = sqrt.(max.(D_sq, R_SOFT^2))

        F_repel = K_REPEL ./ (D_mat .^ 2)
        F_repel[diagind(F_repel)] .= 0.0

        for dim in 1:D
            diff_dim = X[:, dim] .- X[:, dim]'
            dX[:, dim] .+= vec(sum(F_repel .* diff_dim ./ D_mat, dims=2))
        end

        bdplus_barrier!(dX, X, N, D)
        du .= vec(dX)
    end

    function initial_positions(seed::Int)
        rng = Random.MersenneTwister(seed)
        X0 = 0.2 .+ 0.6 .* rand(rng, N, D)
        return project_embedding_to_Bd_plus(X0)
    end

    # UDE: known cohesion (O(n)), NN learns repulsion
    function make_ude_dynamics(nn, st, n, d)
        nd = n * d

        function dynamics(u::AbstractVector{T}, p, t) where T
            X = reshape(u, n, d)
            X_bar = mean(X, dims=1)
            dX_known = -T(K_COHESION) .* (X .- X_bar)

            u_reshape = reshape(u, nd, 1)
            correction, _ = nn(u_reshape, p, st)
            return vec(dX_known) .+ vec(correction)
        end
        return dynamics
    end
end

# =============================================================================
# Dynamics C: Radial Equilibrium with Circulation
# Nodes spiral around centroid while maintaining target radius
# =============================================================================

module DynamicsRadial
    using LinearAlgebra
    using Statistics
    using Random
    using RDPGDynamics: project_embedding_to_Bd_plus

    const N = Main.N_NODES
    const D = Main.D

    const K_RADIAL = 0.08       # Radial spring constant
    const R_TARGET = 0.25       # Target distance from centroid (smaller to stay in B^d_+)
    const K_ANGULAR = 0.03      # Angular velocity (circulation around centroid)
    const R_SOFT = 0.01
    const BOUND_K = 0.5
    const BOUND_SCALE = 0.02

    # Soft B^d_+ barrier (x≥0, ||x||≤1)
    function bdplus_barrier!(dX, X, n, d)
        for i in 1:n
            for dim in 1:d
                dX[i, dim] += BOUND_K * exp(-X[i, dim] / BOUND_SCALE)
            end
            norm_i = norm(X[i, :])
            if norm_i > 0.01
                dX[i, :] .-= BOUND_K * exp((norm_i - 1.0) / BOUND_SCALE) .* X[i, :] ./ norm_i
            end
        end
    end

    function true_dynamics!(du, u, p, t)
        X = reshape(u, N, D)

        X_bar = mean(X, dims=1)
        delta = X .- X_bar

        r_sq = sum(delta .^ 2, dims=2)
        r_norm = sqrt.(max.(r_sq, R_SOFT^2))

        # Radial force: spring toward target radius
        radial_factor = -K_RADIAL .* (r_norm .- R_TARGET) ./ r_norm
        dX = radial_factor .* delta

        # Angular force (2D): circulation around centroid
        if D == 2
            perp = hcat(-delta[:, 2], delta[:, 1]) ./ r_norm
            dX .+= K_ANGULAR .* perp
        end

        bdplus_barrier!(dX, X, N, D)
        du .= vec(dX)
    end

    function initial_positions(seed::Int)
        rng = Random.MersenneTwister(seed)
        # Start centered in B^d_+ with room for circulation
        X0 = 0.4 .+ 0.2 .* rand(rng, N, D)
        return project_embedding_to_Bd_plus(X0)
    end

    # UDE: known radial structure, NN learns angular + corrections
    function make_ude_dynamics(nn, st, n, d)
        nd = n * d

        function dynamics(u::AbstractVector{T}, p, t) where T
            X = reshape(u, n, d)
            X_bar = mean(X, dims=1)
            delta = X .- X_bar

            r_sq = sum(delta .^ 2, dims=2)
            r_norm = sqrt.(max.(r_sq, T(R_SOFT)^2))
            radial_factor = -T(K_RADIAL * 0.5) .* (r_norm .- T(R_TARGET)) ./ r_norm
            dX_known = radial_factor .* delta

            u_reshape = reshape(u, nd, 1)
            correction, _ = nn(u_reshape, p, st)
            return vec(dX_known) .+ vec(correction)
        end
        return dynamics
    end
end

# =============================================================================
# Dynamics D: Kuramoto-like (Angular coupling + Radial equilibrium)
# Nodes synchronize their angular phase while maintaining radial distance
# =============================================================================

module DynamicsKuramoto
    using LinearAlgebra
    using Statistics
    using Random
    using RDPGDynamics: project_embedding_to_Bd_plus

    const N = Main.N_NODES
    const D = Main.D

    const OMEGA_BASE = 0.04     # Base angular velocity
    const K_COUPLING = 0.02     # Kuramoto coupling strength
    const K_RADIAL = 0.05       # Radial spring
    const R_TARGET = 0.25       # Target radius (smaller to stay in B^d_+)
    const R_SOFT = 0.01
    const BOUND_K = 0.5
    const BOUND_SCALE = 0.02

    # Soft B^d_+ barrier (x≥0, ||x||≤1)
    function bdplus_barrier!(dX, X, n, d)
        for i in 1:n
            for dim in 1:d
                dX[i, dim] += BOUND_K * exp(-X[i, dim] / BOUND_SCALE)
            end
            norm_i = norm(X[i, :])
            if norm_i > 0.01
                dX[i, :] .-= BOUND_K * exp((norm_i - 1.0) / BOUND_SCALE) .* X[i, :] ./ norm_i
            end
        end
    end

    function true_dynamics!(du, u, p, t)
        X = reshape(u, N, D)

        X_bar = mean(X, dims=1)
        delta = X .- X_bar

        r_sq = sum(delta .^ 2, dims=2)
        r_norm = sqrt.(max.(r_sq, R_SOFT^2))

        # Radial force: spring toward target
        radial_factor = -K_RADIAL .* (r_norm .- R_TARGET) ./ r_norm
        dX = radial_factor .* delta

        # Kuramoto angular coupling (2D)
        if D == 2
            angles = atan.(delta[:, 2], delta[:, 1])
            sin_angles = sin.(angles)
            cos_angles = cos.(angles)
            sum_sin = sum(sin_angles)
            sum_cos = sum(cos_angles)

            # O(n) Kuramoto: sum_{j≠i} sin(θ_j - θ_i)
            angular_coupling = (sum_sin .* cos_angles .- sum_cos .* sin_angles)
            angular_force = OMEGA_BASE .+ (K_COUPLING / N) .* angular_coupling

            # Convert to Cartesian (perpendicular to radial)
            perp_x = -delta[:, 2] ./ vec(r_norm)
            perp_y = delta[:, 1] ./ vec(r_norm)
            dX[:, 1] .+= angular_force .* perp_x
            dX[:, 2] .+= angular_force .* perp_y
        end

        bdplus_barrier!(dX, X, N, D)
        du .= vec(dX)
    end

    function initial_positions(seed::Int)
        rng = Random.MersenneTwister(seed)
        # Start centered in B^d_+
        X0 = 0.4 .+ 0.2 .* rand(rng, N, D)
        return project_embedding_to_Bd_plus(X0)
    end

    # UDE: known radial structure, NN learns Kuramoto coupling
    function make_ude_dynamics(nn, st, n, d)
        nd = n * d

        function dynamics(u::AbstractVector{T}, p, t) where T
            X = reshape(u, n, d)
            X_bar = mean(X, dims=1)
            delta = X .- X_bar

            r_sq = sum(delta .^ 2, dims=2)
            r_norm = sqrt.(max.(r_sq, T(R_SOFT)^2))

            # Known: partial radial attraction
            radial_factor = -T(K_RADIAL * 0.5) .* (r_norm .- T(R_TARGET)) ./ r_norm
            dX_known = radial_factor .* delta

            # NN learns Kuramoto coupling and corrections
            u_reshape = reshape(u, nd, 1)
            correction, _ = nn(u_reshape, p, st)
            return vec(dX_known) .+ vec(correction)
        end
        return dynamics
    end
end

# =============================================================================
# Dynamics Dispatcher
# =============================================================================

const DYNAMICS_MODULES = Dict(
    "pairwise" => DynamicsPairwise,
    "cohesion" => DynamicsCohesion,
    "radial" => DynamicsRadial,
    "kuramoto" => DynamicsKuramoto
)

function get_dynamics_module(name::String)
    if !haskey(DYNAMICS_MODULES, name)
        error("Unknown dynamics: " * name * ". Use: pairwise, cohesion, radial, kuramoto")
    end
    return DYNAMICS_MODULES[name]
end

function example_name(dynamics::String)
    return "example2_" * dynamics
end

# =============================================================================
# Pure NN Dynamics (same for all)
# =============================================================================

function make_pure_nn_dynamics(nn, st, n::Int, d::Int)
    nd = n * d

    function dynamics(u::AbstractVector{T}, p, t) where T
        u_reshape = reshape(u, nd, 1)
        out, _ = nn(u_reshape, p, st)
        return vec(out)
    end
    return dynamics
end

# =============================================================================
# Phase Implementations
# =============================================================================

function run_generate(dynamics::String)
    mod = get_dynamics_module(dynamics)
    X0 = mod.initial_positions(42)

    metadata = Dict(
        "dynamics_type" => dynamics,
        "communities" => [collect(1:N_NODES)]  # All one community
    )

    phase_generate(example_name(dynamics), mod.true_dynamics!, X0;
                   T_end=T_END, dt=DT, metadata=metadata, seed=42)
end

function run_estimate(dynamics::String)
    phase_estimate(example_name(dynamics); K=K_SAMPLES)
end

function run_train(dynamics::String, model_name::String)
    mod = get_dynamics_module(dynamics)
    ename = example_name(dynamics)

    true_data = load_data(ename, FILES.true_dynamics)
    n, d = true_data["n"], true_data["d"]
    nd = n * d

    rng = Random.Xoshiro(42)

    if model_name == "pure_nn"
        nn = build_nn(nd, [64, 64, 32], nd; rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = make_pure_nn_dynamics(nn, st, n, d)

    elseif model_name == "ude"
        nn = build_correction_nn(nd, nd; hidden_size=48, rng=rng)
        ps, st = Lux.setup(rng, nn)
        ps_init = ComponentArray{F}(ps)
        dynamics_fn = mod.make_ude_dynamics(nn, st, n, d)

    else
        error("Unknown model: " * model_name * ". Use: pure_nn, ude")
    end

    phase_train(ename, model_name, dynamics_fn, ps_init;
                train_fraction=TRAIN_FRACTION,
                epochs_adam=EPOCHS_ADAM, epochs_bfgs=EPOCHS_BFGS,
                lr=0.01f0, dt=F(DT))
end

function run_evaluate(dynamics::String)
    phase_evaluate(example_name(dynamics), ["pure_nn", "ude"])
end

function run_visualize(dynamics::String)
    phase_visualize(example_name(dynamics), ["pure_nn", "ude"])
end

function run_compare()
    println("\n" * "=" ^ 70)
    println("COMPARISON OF ALL DYNAMICS")
    println("=" ^ 70)

    results = Dict{String, Any}()

    for dyn in ["pairwise", "cohesion", "radial", "kuramoto"]
        ename = example_name(dyn)
        eval_file = "evaluation.jls"

        if !data_exists(ename, eval_file)
            println("\n  " * dyn * ": Not evaluated yet")
            continue
        end

        eval_data = load_data(ename, eval_file)
        metrics_dict = eval_data["metrics_dict"]
        T_train = eval_data["T_train"]

        results[dyn] = Dict()

        println("\n  " * uppercase(dyn) * ":")

        for model in ["pure_nn", "ude"]
            if haskey(metrics_dict, model)
                m = metrics_dict[model]
                baseline = metrics_dict["Baseline"]

                D_corr = mean(m.D_corr)
                D_val = mean(m.D_corr[T_train+1:end])
                P_err = mean(m.P_err)

                D_ratio = D_corr / mean(baseline.D_corr)

                results[dyn][model] = (D_corr=D_corr, D_val=D_val, P_err=P_err, D_ratio=D_ratio)

                println("    " * model * ": D_corr=" * string(round(D_corr, digits=3)) *
                        " (" * string(round(100*D_ratio, digits=1)) * "%), Val D=" *
                        string(round(D_val, digits=3)))
            end
        end
    end

    # Find best
    if !isempty(results)
        println("\n" * "=" ^ 70)
        println("RANKING (by validation D correlation)")
        println("=" ^ 70)

        ranking = []
        for (dyn, models) in results
            for (model, metrics) in models
                push!(ranking, (dyn=dyn, model=model, D_val=metrics.D_val, D_ratio=metrics.D_ratio))
            end
        end

        sort!(ranking, by=x -> x.D_val, rev=true)

        for (i, r) in enumerate(ranking)
            println("  " * string(i) * ". " * r.dyn * "/" * r.model *
                    ": Val D=" * string(round(r.D_val, digits=3)) *
                    " (" * string(round(100*r.D_ratio, digits=1)) * "% of baseline)")
        end

        println("\nRecommendation: Use '" * ranking[1].dyn * "' dynamics for Examples 3a/3b")
    end
end

# =============================================================================
# Main Entry Point (Julia 1.11+)
# =============================================================================

function print_usage()
    println("Usage: julia --project scripts/example2_homogeneous_v2.jl <command> [dynamics] [model]")
    println("\nCommands:")
    println("  generate <dynamics>  - Generate true dynamics")
    println("  estimate <dynamics>  - RDPG estimation")
    println("  train <dynamics> <model>  - Train model (pure_nn, ude, or all)")
    println("  evaluate <dynamics>  - Evaluate models")
    println("  visualize <dynamics> - Generate plots")
    println("  all <dynamics>       - Run full pipeline")
    println("  compare              - Compare all dynamics")
    println("  status [dynamics]    - Show status")
    println("\nDynamics: pairwise, cohesion, radial, kuramoto")
end

function (@main)(args)
    if isempty(args)
        print_usage()
        return 0
    end

    command = args[1]

    if command == "compare"
        run_compare()
        return 0
    end

    if command == "status"
        if length(args) >= 2
            print_status(example_name(args[2]))
        else
            for dyn in ["pairwise", "cohesion", "radial", "kuramoto"]
                print_status(example_name(dyn))
            end
        end
        return 0
    end

    # Other commands require dynamics argument
    if length(args) < 2
        println("Specify dynamics: pairwise, cohesion, radial, kuramoto")
        return 1
    end

    dynamics = args[2]

    if command == "generate"
        run_generate(dynamics)

    elseif command == "estimate"
        run_estimate(dynamics)

    elseif command == "train"
        if length(args) < 3
            println("Specify model: pure_nn, ude, or all")
            return 1
        end
        model = args[3]
        if model == "all"
            run_train(dynamics, "pure_nn")
            run_train(dynamics, "ude")
        else
            run_train(dynamics, model)
        end

    elseif command == "evaluate"
        run_evaluate(dynamics)

    elseif command == "visualize"
        run_visualize(dynamics)

    elseif command == "all"
        run_generate(dynamics)
        run_estimate(dynamics)
        run_train(dynamics, "pure_nn")
        run_train(dynamics, "ude")
        run_evaluate(dynamics)
        run_visualize(dynamics)

    else
        println("Unknown command: " * command)
        return 1
    end

    return 0
end
