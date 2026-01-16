#!/usr/bin/env julia
"""
Reproduce all results from the paper:
"Modelling Temporal Networks with Scientific Machine Learning"

Usage:
    julia --project scripts/reproduce_paper.jl [options]

Options:
    --retrain       Retrain models instead of using saved weights
    --dataset NAME  Run only for specific dataset (1_community_oscillation,
                    2_communities_joining, long_tail)
    --output DIR    Output directory for figures (default: results/)
"""

using RDPGDynamics
using JSON
using Serialization
using CairoMakie

# Dataset configurations (n and d are auto-detected from data)
# Original small datasets (n=5-36)
const DATASETS_SMALL = [
    (
        name = "1_community_oscillation",
        file = "data/1_community_oscillation.json",
        description = "Single community with oscillating connection probabilities (n=5)"
    ),
    (
        name = "2_communities_joining",
        file = "data/2_communities_joining.json",
        description = "Two communities gradually merging (n=11)"
    ),
    (
        name = "long_tail",
        file = "data/long_tail.json",
        description = "Long-tailed degree distribution (n=36)"
    )
]

# New larger datasets (n=50-75)
const DATASETS_LARGE = [
    (
        name = "oscillating_community_n50",
        file = "data/oscillating_community_n50.json",
        description = "Oscillating community (n=50)"
    ),
    (
        name = "merging_communities_n60",
        file = "data/merging_communities_n60.json",
        description = "Merging communities (n=60)"
    ),
    (
        name = "long_tail_n75",
        file = "data/long_tail_n75.json",
        description = "Long-tail dynamics (n=75)"
    )
]

# Default to large datasets
const DATASETS = DATASETS_LARGE

"""
    load_dataset(dataset_config) -> NamedTuple

Load L and R series from JSON data file using RDPGDynamics loader.
"""
function load_dataset(config)
    return load_rdpg_json(config.file)
end

"""
    train_or_load_model(data::NamedTuple, config; retrain=false) -> params

Train a new model or load existing saved parameters.
"""
function train_or_load_model(data::NamedTuple, config; retrain::Bool=false)
    model_path = "models/" * config.name * "/trained_params.jls"

    if !retrain && isfile(model_path)
        println("Loading saved model from " * model_path)
        return deserialize(model_path)
    end

    println("Training model for " * config.name * "...")

    train_config = RDPGConfig(
        n = data.n,
        d = data.d,
        datasize = min(15, data.timesteps),
        epochs_adam = 500,
        epochs_lion = 300
    )

    params = train_rdpg_node(data.L, train_config; verbose=true)

    # Save trained model
    mkpath(dirname(model_path))
    serialize(model_path, params)
    println("Saved model to " * model_path)

    return params
end

"""
    plot_embedding_trajectory(L_data, predictions, name; n, d, output_dir="results/")

Create trajectory plots comparing ground truth and predictions.
"""
function plot_embedding_trajectory(L_data, predictions, name::String; n::Int, d::Int, output_dir::String="results/")
    mkpath(output_dir)

    timesteps = size(predictions, 2)

    # Plot each embedding dimension
    for dim in 1:d
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1, 1],
            xlabel="Time",
            ylabel="Embedding dimension " * string(dim),
            title=name * " - Dimension " * string(dim)
        )

        # Ground truth for each node
        for node in 1:n
            ground_truth = [L_data[t][node, dim] for t in 1:min(timesteps, length(L_data))]
            lines!(ax, 1:length(ground_truth), ground_truth,
                   color=(:blue, 0.3), linewidth=1)
        end

        # Predictions for each node
        for node in 1:n
            pred_series = [predictions[(node-1)*d + dim, t] for t in 1:timesteps]
            lines!(ax, 1:timesteps, pred_series,
                   color=(:red, 0.3), linewidth=1, linestyle=:dash)
        end

        # Legend
        Legend(fig[1, 2], [LineElement(color=:blue), LineElement(color=:red, linestyle=:dash)],
               ["Ground Truth", "Prediction"])

        save(output_dir * "/" * name * "_d" * string(dim) * ".pdf", fig)
        println("Saved: " * output_dir * "/" * name * "_d" * string(dim) * ".pdf")
    end
end

"""
    compute_metrics(L_data, predictions; n, d) -> NamedTuple

Compute MSE and constraint violation metrics.
"""
function compute_metrics(L_data, predictions; n::Int, d::Int)
    timesteps = min(size(predictions, 2), length(L_data))

    # Flatten ground truth
    ground_truth = hcat([vec(L_data[t]') for t in 1:timesteps]...)

    # MSE
    mse = sum(abs2, ground_truth .- predictions[:, 1:timesteps]) / length(ground_truth)

    return (mse=mse,)
end

"""
    main(; retrain=false, dataset=nothing, output_dir="results/")

Run the full reproduction pipeline.
"""
function main(; retrain::Bool=false, dataset::Union{String,Nothing}=nothing, output_dir::String="results/")
    println("=" ^ 60)
    println("RDPG Dynamics - Paper Reproduction")
    println("=" ^ 60)

    datasets_to_run = if isnothing(dataset)
        DATASETS
    else
        filter(d -> d.name == dataset, DATASETS)
    end

    if isempty(datasets_to_run)
        error("Unknown dataset: " * dataset)
    end

    results = []

    for config in datasets_to_run
        println("\n" * "-" ^ 40)
        println("Dataset: " * config.name)
        println(config.description)
        println("-" ^ 40)

        # Check if data file exists
        if !isfile(config.file)
            println("WARNING: Data file not found: " * config.file)
            println("Skipping this dataset.")
            continue
        end

        # Load data
        data = load_dataset(config)
        println("Loaded " * string(data.timesteps) * " timesteps (n=" * string(data.n) * ", d=" * string(data.d) * ")")

        # Train or load model
        params = train_or_load_model(data, config; retrain=retrain)

        # Generate predictions
        println("Generating predictions...")
        train_config = RDPGConfig(n=data.n, d=data.d)
        _, prob, _, _ = build_neural_ode(train_config)

        u0 = Float32.(vec(data.L[1]'))
        tspan = (0.0f0, Float32(data.timesteps - 1))
        tsteps = range(tspan[1], tspan[2]; length=data.timesteps)

        predictions = predict_trajectory(prob, params, u0, tspan, tsteps)

        # Compute metrics
        metrics = compute_metrics(data.L, predictions; n=data.n, d=data.d)
        println("MSE: " * string(round(metrics.mse; digits=6)))

        push!(results, (name=config.name, metrics=metrics))

        # Generate diagnostic plots (phase portrait, network snapshots, probability heatmaps)
        plot_all_diagnostics(data.L, predictions, config.name; n=data.n, d=data.d, output_dir=output_dir)
    end

    # Summary table
    println("\n" * "=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    println("Dataset                    | MSE")
    println("-" ^ 60)
    for r in results
        println(rpad(r.name, 26) * " | " * string(round(r.metrics.mse; digits=6)))
    end
    println("=" ^ 60)

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    retrain = "--retrain" in ARGS
    dataset = nothing
    output_dir = "results/"

    for (i, arg) in enumerate(ARGS)
        if arg == "--dataset" && i < length(ARGS)
            global dataset = ARGS[i+1]
        elseif arg == "--output" && i < length(ARGS)
            global output_dir = ARGS[i+1]
        end
    end

    main(; retrain=retrain, dataset=dataset, output_dir=output_dir)
end
