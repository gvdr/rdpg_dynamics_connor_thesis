#!/usr/bin/env -S julia --project
"""
UDE Pipeline Experiment — Phase 2: Alignment + UDE Training + Symbolic Regression

For each rep × condition (anchor / sequential / unaligned):
1. Build UDE with known linear damping + unknown NN residual
2. Train via adjoint sensitivity (learn γ̂ and NN weights)
3. Extract NN residual samples
4. Run SymbolicRegression per output dimension
5. Save results

True residual (what SymReg should discover):
  f_u(δ) = β r² δ + ω J δ

Outputs: results/ude_experiment/ude_results.jls
"""

using Random
using LinearAlgebra
using Statistics
using Serialization
using OrdinaryDiffEq
using SciMLSensitivity
using Lux
using ComponentArrays
using Zygote
using Optimization
using OptimizationOptimisers
using SymbolicRegression

include(joinpath(@__DIR__, "..", "alg4", "alg4_utils.jl"))
include(joinpath(@__DIR__, "..", "alg4", "alg4_alignment_utils.jl"))

using .Alg4Utils
using .Alg4AlignmentUtils

# =============================================================================
# Constants
# =============================================================================

const DATA_DIR = joinpath("data", "ude_experiment")
const RESULTS_DIR = joinpath("results", "ude_experiment")

const J_MATRIX = Float32.((1.0 / sqrt(3.0)) .* [
    0.0  -1.0   1.0;
    1.0   0.0  -1.0;
   -1.0   1.0   0.0
])

# Training config
const N_EPOCHS = 500
const LEARNING_RATE = 0.01f0
const NN_REG_LAMBDA = Float32(1e-3)  # L2 regularization on NN weights
const N_SR_SAMPLES = 2000
const SR_MAXSIZE = 20
const SR_POPULATIONS = 25

# =============================================================================
# UDE construction
# =============================================================================

"""
    build_spiral_ude(config, community, anchor_mask)

Build the UDE components: neural network, initial parameters, and ODE function.

Returns (chain, ps, st, dudt, n, d, mu_all_f32, non_anchor_f32)
"""
function build_spiral_ude(config::Dict, community::Vector{Int}, anchor_mask::BitVector)
    n = config["n"]
    d = config["d"]
    centroids = config["centroids"]

    # Neural network: 3 -> 16 -> 16 -> 3
    rng_lux = Lux.replicate(Random.default_rng())
    chain = Lux.Chain(
        Lux.Dense(d, 16, tanh),
        Lux.Dense(16, 16, tanh),
        Lux.Dense(16, d)
    )
    ps_nn, st = Lux.setup(rng_lux, chain)

    # Precompute centroid matrix (n x d) and non-anchor mask
    mu_all = zeros(Float64, n, d)
    for i in 1:n
        k = community[i]
        mu_all[i, :] = centroids[k]
    end
    mu_all_f32 = Float32.(mu_all)

    non_anchor_f32 = Float32.(.!anchor_mask)  # 1.0 for mobile, 0.0 for anchor

    # Initial parameters: learnable gamma + NN weights
    ps = ComponentArray(gamma=Float32[0.1], nn=ComponentArray(ps_nn))

    return chain, ps, st, mu_all_f32, non_anchor_f32
end

"""
    make_ude_ode(chain, st, mu_all_f32, non_anchor_f32, n, d)

Create an out-of-place ODE function compatible with Zygote AD.
Returns a function dudt(u, p, t).
"""
function make_ude_ode(chain, st, mu_all_f32::Matrix{Float32},
                      non_anchor_f32::Vector{Float32},
                      n::Int, d::Int)
    function dudt(u, p, t)
        X = reshape(u, n, d)
        delta = X .- mu_all_f32           # n x d offsets from centroids

        # Known part: -γ δ
        f_known = -p.gamma[1] .* delta

        # Unknown part: NN(δ')  where δ' is d x n for Lux convention
        f_nn_out, _ = Lux.apply(chain, delta', p.nn, st)  # d x n output
        f_nn = f_nn_out'                                    # n x d

        # Combine and mask (zero out anchor derivatives)
        dX = (f_known .+ f_nn) .* non_anchor_f32

        return vec(dX)
    end
    return dudt
end

# =============================================================================
# Training
# =============================================================================

"""
    train_spiral_ude(X_data_series, config, chain, ps0, st, mu_all_f32, non_anchor_f32)

Train the UDE via Adam optimizer with adjoint sensitivity.
Returns trained parameters.
"""
function train_spiral_ude(X_data_series::Vector{<:AbstractMatrix},
                           config::Dict,
                           chain, ps0, st,
                           mu_all_f32::Matrix{Float32},
                           non_anchor_f32::Vector{Float32})
    n = config["n"]
    d = config["d"]
    dt = Float32(config["dt"])
    T = length(X_data_series)

    # Flatten data: each column is vec(X(t))
    u_data = Float32.(hcat([vec(X) for X in X_data_series]...))
    tspan = (0.0f0, dt * Float32(T - 1))
    tsteps = range(tspan[1], tspan[2]; length=T)

    # Build ODE
    dudt = make_ude_ode(chain, st, mu_all_f32, non_anchor_f32, n, d)

    u0 = u_data[:, 1]
    prob = ODEProblem(dudt, u0, tspan, ps0)

    # Loss: MSE over trajectory + L2 regularization on NN weights
    # The regularization discourages the NN from absorbing the linear
    # damping term, improving identifiability of gamma.
    function loss_fn(p, _)
        sol = solve(prob, Tsit5();
            p=p, saveat=tsteps,
            sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
            abstol=1e-5, reltol=1e-5,
            maxiters=5000)

        u_pred = Array(sol)
        mse = mean((u_pred .- u_data) .^ 2)
        nn_reg = sum(abs2, p.nn)
        return mse + NN_REG_LAMBDA * nn_reg
    end

    # Callback
    iter_count = Ref(0)
    function callback(state, loss_val)
        iter_count[] += 1
        if iter_count[] % 50 == 0
            println("      Epoch " * string(iter_count[]) * ": loss = " *
                    string(round(Float64(loss_val), digits=6)))
        end
        return false
    end

    # Optimization
    opt_f = OptimizationFunction(loss_fn, Optimization.AutoZygote())
    opt_prob = OptimizationProblem(opt_f, ps0)
    result = solve(opt_prob, OptimizationOptimisers.Adam(LEARNING_RATE);
                   maxiters=N_EPOCHS, callback=callback)

    return result.u
end

# =============================================================================
# NN residual extraction
# =============================================================================

"""
    extract_nn_residual(chain, ps_trained, st, n_samples; delta_range=0.3)

Sample the trained NN on random δ vectors.
Returns (delta_samples, f_nn_samples) both as n_samples x d matrices.
"""
function extract_nn_residual(chain, ps_trained, st;
                              n_samples::Int=N_SR_SAMPLES,
                              delta_range::Float64=0.3)
    rng = Random.MersenneTwister(42)
    # Random δ in [-delta_range, delta_range]^d
    d = 3
    delta_samples = Float32.((2.0 * delta_range) .* rand(rng, n_samples, d) .- delta_range)

    # Evaluate NN: input is d x n_samples
    f_nn_out, _ = Lux.apply(chain, delta_samples', ps_trained.nn, st)  # d x n_samples
    f_nn_samples = Matrix(f_nn_out')  # n_samples x d

    return delta_samples, f_nn_samples
end

"""
    compute_true_residual(delta_samples; beta, omega)

Compute the true f_u(δ) = β r² δ + ω J δ for comparison.
"""
function compute_true_residual(delta_samples::AbstractMatrix;
                                beta::Float64=-0.5, omega::Float64=1.0)
    n_samples, d = size(delta_samples)
    f_true = zeros(Float32, n_samples, d)

    J = Float32.((1.0 / sqrt(3.0)) .* [0.0 -1.0 1.0; 1.0 0.0 -1.0; -1.0 1.0 0.0])

    for i in 1:n_samples
        delta = delta_samples[i, :]
        r2 = dot(delta, delta)
        f_true[i, :] = Float32(beta) .* r2 .* delta .+ Float32(omega) .* (J * delta)
    end
    return f_true
end

# =============================================================================
# Symbolic Regression
# =============================================================================

"""
    run_symreg(delta_samples, f_nn_samples)

Run SymbolicRegression on each output dimension of the NN residual.
Returns vector of Pareto front dicts (one per dimension), each containing
vectors of (complexity, loss, equation_string) for serialization safety.
"""
function run_symreg(delta_samples::AbstractMatrix, f_nn_samples::AbstractMatrix)
    d = size(f_nn_samples, 2)
    X_sr = Float64.(delta_samples')  # d x n_samples for SymbolicRegression

    options = Options(;
        binary_operators=[+, -, *],
        maxsize=SR_MAXSIZE,
        populations=SR_POPULATIONS,
        progress=true,
        save_to_file=false
    )

    results = []
    for dim in 1:d
        println("      SymReg dimension " * string(dim) * "/" * string(d))
        y_dim = Float64.(f_nn_samples[:, dim])
        hof = equation_search(X_sr, y_dim; niterations=40, options=options)

        # Extract Pareto front into plain types for safe serialization
        valid_idx = findall(hof.exists)
        pareto = Dict(
            "complexities" => Float64[compute_complexity(hof.members[i], options) for i in valid_idx],
            "losses" => Float64[hof.members[i].loss for i in valid_idx],
            "equations" => String[string(hof.members[i].tree) for i in valid_idx]
        )
        push!(results, pareto)
    end

    return results
end

# =============================================================================
# Main loop
# =============================================================================

function process_rep(rep::Int)
    println("\n  Rep " * string(rep))
    data_path = joinpath(DATA_DIR, "rep" * string(rep) * ".jls")
    if !isfile(data_path)
        println("    WARNING: data file not found: " * data_path)
        return nothing
    end

    data = deserialize(data_path)
    config = data["config"]
    community = data["community"]
    anchor_mask = data["anchor_mask"]

    conditions = Dict(
        "anchor" => data["X_anchor_aligned"],
        "sequential" => data["X_seq_aligned"],
        "unaligned" => data["X_unaligned"]
    )

    rep_results = Dict{String, Any}()

    for (cond_name, X_data_series) in conditions
        println("    Condition: " * cond_name)

        # Build UDE
        chain, ps0, st, mu_all_f32, non_anchor_f32 = build_spiral_ude(
            config, community, anchor_mask)

        # Train
        println("      Training UDE (" * string(N_EPOCHS) * " epochs)...")
        ps_trained = train_spiral_ude(
            X_data_series, config, chain, ps0, st, mu_all_f32, non_anchor_f32)

        gamma_hat = Float64(ps_trained.gamma[1])
        println("      Learned gamma = " * string(round(gamma_hat, digits=4)) *
                " (true = " * string(config["gamma"]) * ")")

        # Extract NN residual
        println("      Extracting NN residual samples...")
        delta_samples, f_nn_samples = extract_nn_residual(chain, ps_trained, st)

        # True residual for comparison
        f_true_samples = compute_true_residual(delta_samples;
            beta=config["beta"], omega=config["omega"])

        # Residual accuracy (NN-only)
        residual_mse = mean((f_nn_samples .- f_true_samples) .^ 2)
        println("      NN residual MSE vs true: " *
                string(round(Float64(residual_mse), digits=6)))

        # Total dynamics accuracy: f_learned(δ) = -γ̂ δ + NN(δ) vs f_true(δ) = -γ δ + βr²δ + ωJδ
        f_total_learned = -Float32(gamma_hat) .* delta_samples .+ f_nn_samples
        f_total_true = -Float32(config["gamma"]) .* delta_samples .+ f_true_samples
        total_dynamics_mse = mean((f_total_learned .- f_total_true) .^ 2)
        println("      Total dynamics MSE: " *
                string(round(Float64(total_dynamics_mse), digits=6)))

        # Symbolic Regression
        println("      Running SymbolicRegression...")
        sr_results = run_symreg(delta_samples, f_nn_samples)

        # Collect best equations from extracted Pareto front dicts
        best_eqs = String[]
        for (dim, pareto) in enumerate(sr_results)
            if !isempty(pareto["losses"])
                best_idx = argmin(pareto["losses"])
                eq_str = pareto["equations"][best_idx]
                loss_val = pareto["losses"][best_idx]
                push!(best_eqs, "dim" * string(dim) * ": " * eq_str)
                println("        Best dim" * string(dim) * ": " * eq_str *
                        " (loss=" * string(round(loss_val, digits=6)) * ")")
            else
                push!(best_eqs, "dim" * string(dim) * ": (no result)")
            end
        end

        rep_results[cond_name] = Dict(
            "gamma_hat" => gamma_hat,
            "delta_samples" => delta_samples,
            "f_nn_samples" => f_nn_samples,
            "f_true_samples" => f_true_samples,
            "residual_mse" => Float64(residual_mse),
            "total_dynamics_mse" => Float64(total_dynamics_mse),
            "sr_results" => sr_results,
            "best_equations" => best_eqs
        )
    end

    return rep_results
end

function main()
    println("UDE Pipeline Experiment — Training & Symbolic Regression")
    mkpath(RESULTS_DIR)

    all_results = Dict{Int, Any}()

    for rep in 1:5
        result = process_rep(rep)
        if !isnothing(result)
            all_results[rep] = result
        end
    end

    # Save all results
    out_path = joinpath(RESULTS_DIR, "ude_results.jls")
    serialize(out_path, all_results)
    println("\n" * "="^60)
    println("All results saved to: " * out_path)
    println("="^60)

    # Print summary
    println("\nSummary:")
    for rep in sort(collect(keys(all_results)))
        println("  Rep " * string(rep) * ":")
        for cond in ["anchor", "sequential", "unaligned"]
            if haskey(all_results[rep], cond)
                r = all_results[rep][cond]
                println("    " * cond * ": gamma_hat=" *
                        string(round(r["gamma_hat"], digits=4)) *
                        ", residual_MSE=" *
                        string(round(r["residual_mse"], digits=6)) *
                        ", total_dyn_MSE=" *
                        string(round(r["total_dynamics_mse"], digits=6)))
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
