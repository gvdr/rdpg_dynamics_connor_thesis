"""
    RDPGDynamics

A Julia package for modeling temporal network dynamics using Random Dot Product Graphs (RDPG)
and Neural Ordinary Differential Equations (Neural ODEs).

This package implements the methodology from:
"Modelling Temporal Networks with Scientific Machine Learning"

## Main Components

- **Embedding**: SVD-based RDPG embedding with Procrustes alignment
- **Types**: `TemporalNetworkEmbedding` for storing temporal embeddings
- **Constraints**: Probability constraint losses for training
- **Training**: Neural ODE training pipeline with multi-stage optimization

## Quick Start

```julia
using RDPGDynamics
using JSON

# Load data
data = JSON.parsefile("data/2_communities_joining.json")
L_data = [[L[1] L[2]] for L in data["L_series"]]

# Configure and train
config = RDPGConfig(n=11, d=2, epochs_adam=500)
params = train_rdpg_node(L_data, config)
```

## Exports

### Embedding Functions
- `svd_embedding`: Compute RDPG embedding from adjacency matrix
- `ortho_procrustes_RM`: Orthogonal Procrustes alignment
- `truncated_svd`: Truncated SVD using Arpack

### Types
- `TemporalNetworkEmbedding`: Container for temporal L/R embeddings

### Training
- `RDPGConfig`: Configuration struct for training
- `train_rdpg_node`: Train Neural ODE model
- `build_neural_ode`: Construct neural network and ODE problem

### Constraints
- `probability_constraint_loss`: Penalize probabilities outside [0,1]
"""
module RDPGDynamics

# Core dependencies
using LinearAlgebra
using Random
using Zygote

# Include submodules in dependency order
include("embedding.jl")
include("types.jl")
include("constraints.jl")
include("training.jl")
include("gauge_ude.jl")  # Gauge-consistent N(P)X architecture
include("visualization.jl")
include("symbolic.jl")

# Pipeline utilities for examples
include("pipeline.jl")

# Re-export key functions
export svd_embedding, ortho_procrustes_RM, truncated_svd
export project_to_Bd_plus, in_Bd_plus, align_to_Bd_plus, svd_embedding_Bd_plus
export embed_temporal_network_Bd_plus, embed_temporal_network_smoothed
export TemporalNetworkEmbedding, without_node, target_node
export probability_constraint_loss, below_zero_penalty, above_one_penalty
export RDPGConfig, build_neural_ode, train_rdpg_node, predict_trajectory
export SingleTargetConfig, train_single_target, predict_single_target
export load_rdpg_json
export plot_phase_portrait, plot_network_snapshots, plot_probability_heatmaps, plot_all_diagnostics
export plot_single_target_trajectory
export extract_symbolic_dynamics, predict_symbolic, SymbolicDynamics
# Gauge-consistent N(P)X architecture
export PolynomialNConfig, KernelNConfig, SymmetricNConfig
export build_polynomial_N_ode, build_kernel_N_ode, build_symmetric_nn_ode
export train_gauge_ude, predict_gauge_trajectory

"""
    load_rdpg_json(filepath::String) -> NamedTuple

Load RDPG embedding data from JSON file.

The JSON format stores L_series and R_series as d × n arrays.
This function converts them to vectors of n × d matrices.

# Returns
- Named tuple with fields: L, R, n, d, timesteps
"""
function load_rdpg_json(filepath::String)
    # Import JSON here to avoid adding it as a hard dependency
    JSON = Base.require(Base.PkgId(Base.UUID("682c06a0-de6a-54ab-a142-c8b1cf79cde6"), "JSON"))

    data = JSON.parsefile(filepath)
    L_raw = data["L_series"]
    R_raw = data["R_series"]

    # Data is stored as d × n, convert to vector of n × d matrices
    d = length(L_raw[1])
    n = length(L_raw[1][1])
    timesteps = length(L_raw)

    L_data = Vector{Matrix{Float64}}(undef, timesteps)
    R_data = Vector{Matrix{Float64}}(undef, timesteps)

    for t in 1:timesteps
        # Stack dimensions as columns to get n × d
        L_mat = hcat([Float64.(L_raw[t][dim]) for dim in 1:d]...)
        R_mat = hcat([Float64.(R_raw[t][dim]) for dim in 1:d]...)
        L_data[t] = L_mat
        R_data[t] = R_mat
    end

    return (L=L_data, R=R_data, n=n, d=d, timesteps=timesteps)
end

end # module
