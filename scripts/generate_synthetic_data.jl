#!/usr/bin/env julia
"""
Generate synthetic temporal network data with larger network sizes (n=50-100).

The key insight is to generate networks where we KNOW the latent dynamics,
then sample adjacency matrices from the resulting probability matrices.

Usage:
    julia --project scripts/generate_synthetic_data.jl
"""

using LinearAlgebra
using Random
using JSON
using Statistics

"""
    project_to_Bd_plus(x)

Project a point onto B^d_+ (non-negative unit ball).
All coordinates ≥ 0 and ||x|| ≤ 1.
"""
function project_to_Bd_plus(x::Vector{Float64})
    # Step 1: Clamp to non-negative
    x_pos = max.(x, 0.0)
    # Step 2: Scale if norm > 1
    n = norm(x_pos)
    return n > 1.0 ? x_pos ./ n : x_pos
end

"""
    oscillating_bridge_node(; n1=25, n2=24, d=2, timesteps=30, omega=0.3, seed=1234)

Generate two stable communities with ONE bridge node that oscillates between them.

**IMPORTANT: All positions live in B^d_+ (non-negative unit ball)**
This ensures L_i · L_j ∈ [0, 1] automatically (valid probabilities).

Scenario in 2D:
- Community 1 (n1 nodes): Clustered near [0.85, 0.25] (high dim1, low dim2)
- Community 2 (n2 nodes): Clustered near [0.25, 0.85] (low dim1, high dim2)
- Bridge node: Oscillates along the arc between these positions

Connection probabilities:
- Within community 1: ~0.72 + 0.06 = 0.78 (high)
- Within community 2: ~0.06 + 0.72 = 0.78 (high)
- Between communities: ~0.21 + 0.21 = 0.42 (moderate)
- Bridge ↔ community depends on bridge position (the interesting dynamics!)

Ground truth dynamics for bridge node:
    θ(t) = θ_min + (θ_max - θ_min) * (1 + cos(ω*t))/2
    L_bridge(t) = r * [cos(θ(t)), sin(θ(t))]
where θ oscillates between community 1's angle and community 2's angle.
"""
function oscillating_bridge_node(; n1::Int=25, n2::Int=24, d::Int=2, timesteps::Int=40,
                                   omega::Float64=0.2, seed::Int=1234)
    rng = Random.MersenneTwister(seed)
    n = n1 + n2 + 1  # +1 for the bridge node

    # Community centers in B^d_+ (all positive, norm ≤ 1)
    # Use angles to parameterize positions on the unit circle (positive quadrant)
    r = 0.9  # Radius (< 1 to stay inside unit ball)

    θ1 = π/8   # Community 1: ~22.5 degrees (high dim1)
    θ2 = 3π/8  # Community 2: ~67.5 degrees (high dim2)

    center1 = r * [cos(θ1), sin(θ1)]  # ≈ [0.83, 0.34]
    center2 = r * [cos(θ2), sin(θ2)]  # ≈ [0.34, 0.83]

    # Generate fixed positions for community nodes (with small perturbations)
    # Use angular perturbation to stay on/near the arc
    community1_positions = Vector{Vector{Float64}}(undef, n1)
    community2_positions = Vector{Vector{Float64}}(undef, n2)

    for i in 1:n1
        δθ = 0.08 * randn(rng)  # Angular perturbation
        δr = 0.05 * abs(randn(rng))  # Radial perturbation (keep positive)
        ri = r - δr
        community1_positions[i] = ri * [cos(θ1 + δθ), sin(θ1 + δθ)]
    end

    for i in 1:n2
        δθ = 0.08 * randn(rng)
        δr = 0.05 * abs(randn(rng))
        ri = r - δr
        community2_positions[i] = ri * [cos(θ2 + δθ), sin(θ2 + δθ)]
    end

    L_series = Vector{Matrix{Float64}}(undef, timesteps)
    R_series = Vector{Matrix{Float64}}(undef, timesteps)

    for t in 1:timesteps
        L_t = zeros(d, n)

        # Community 1: static (with tiny jitter for realism)
        for i in 1:n1
            pos = community1_positions[i] .+ 0.005 * randn(rng, d)
            L_t[:, i] = project_to_Bd_plus(pos)
        end

        # Community 2: static (with tiny jitter for realism)
        for i in 1:n2
            pos = community2_positions[i] .+ 0.005 * randn(rng, d)
            L_t[:, n1 + i] = project_to_Bd_plus(pos)
        end

        # Bridge node: oscillates along the arc between θ1 and θ2
        # At t=1: at community 1 (θ1)
        # Oscillates with period 2π/omega
        phase = omega * (t - 1)
        alpha = (1 + cos(phase)) / 2  # Ranges from 1 to 0 to 1 ...

        θ_bridge = alpha * θ1 + (1 - alpha) * θ2
        bridge_position = r * [cos(θ_bridge), sin(θ_bridge)]
        L_t[:, n] = bridge_position  # Last node is the bridge

        L_series[t] = L_t
        R_series[t] = L_t  # For undirected: R = L
    end

    # Verify all positions are in B^d_+
    for t in 1:timesteps
        for i in 1:n
            @assert all(L_series[t][:, i] .>= 0) "Negative coordinate at t=$t, node=$i"
            @assert norm(L_series[t][:, i]) <= 1.0 + 1e-6 "Norm > 1 at t=$t, node=$i"
        end
    end

    return (
        L_series = L_series,
        R_series = R_series,
        n = n,
        d = d,
        timesteps = timesteps,
        bridge_node_idx = n,  # The bridge node is the LAST node
        description = "Two communities with oscillating bridge node in B^d_+ (n=" * string(n) * ", bridge=node " * string(n) * ")"
    )
end

"""
    oscillating_community(; n=50, d=2, timesteps=30, omega=0.2, seed=1234)

DEPRECATED: Use oscillating_bridge_node instead.

Generate a single community where ALL embedding positions oscillate sinusoidally.
(All nodes have the same dynamics - less interesting for prediction)
"""
function oscillating_community(; n::Int=50, d::Int=2, timesteps::Int=30,
                                omega::Float64=0.2, seed::Int=1234)
    rng = Random.MersenneTwister(seed)

    # Initial positions: nodes clustered around origin with some spread
    # All nodes in same community, slightly perturbed
    base_L = [0.5, 0.3]  # Initial community center
    L0 = [base_L .+ 0.05 * randn(rng, d) for _ in 1:n]

    # Generate L series by rotating each node's position
    L_series = Vector{Matrix{Float64}}(undef, timesteps)
    R_series = Vector{Matrix{Float64}}(undef, timesteps)

    for t in 1:timesteps
        theta = omega * (t - 1)
        rotation = [cos(theta) -sin(theta); sin(theta) cos(theta)]

        L_t = zeros(d, n)
        for i in 1:n
            L_t[:, i] = rotation * L0[i]
        end

        L_series[t] = L_t
        # For undirected graphs, R = L * diag(1, -1) approximately
        R_series[t] = L_t .* [1.0; -1.0]
    end

    return (
        L_series = L_series,
        R_series = R_series,
        n = n,
        d = d,
        timesteps = timesteps,
        description = "Oscillating community (n=" * string(n) * ", omega=" * string(omega) * ")"
    )
end

"""
    merging_communities(; n1=25, n2=25, d=2, timesteps=30, merge_rate=0.1, seed=1234)

Generate two communities that gradually merge over time.

Initially: Two separate clusters in embedding space
Finally: Single merged cluster

Ground truth: Linear interpolation between separated and merged states
"""
function merging_communities(; n1::Int=25, n2::Int=25, d::Int=2, timesteps::Int=30,
                              merge_rate::Float64=0.1, seed::Int=1234)
    rng = Random.MersenneTwister(seed)
    n = n1 + n2

    # Initial positions: two clusters
    center1 = [0.6, 0.2]
    center2 = [-0.4, -0.3]

    # Final positions: merged cluster
    center_merged = [0.1, -0.05]

    L_series = Vector{Matrix{Float64}}(undef, timesteps)
    R_series = Vector{Matrix{Float64}}(undef, timesteps)

    # Generate node-specific perturbations (fixed across time)
    perturbations = [0.08 * randn(rng, d) for _ in 1:n]

    for t in 1:timesteps
        # Interpolation factor (0 = separated, 1 = merged)
        alpha = min(1.0, merge_rate * (t - 1))

        L_t = zeros(d, n)

        # Community 1
        current_center1 = (1 - alpha) * center1 + alpha * center_merged
        for i in 1:n1
            L_t[:, i] = current_center1 + perturbations[i]
        end

        # Community 2
        current_center2 = (1 - alpha) * center2 + alpha * center_merged
        for i in (n1+1):n
            L_t[:, i] = current_center2 + perturbations[i]
        end

        L_series[t] = L_t
        R_series[t] = L_t .* [1.0; -1.0]
    end

    return (
        L_series = L_series,
        R_series = R_series,
        n = n,
        d = d,
        timesteps = timesteps,
        description = "Merging communities (n1=" * string(n1) * ", n2=" * string(n2) * ")"
    )
end

"""
    long_tail_dynamics(; n=75, d=2, timesteps=30, hub_fraction=0.1, seed=1234)

Generate a network with long-tailed degree distribution where hubs and
peripheral nodes have different dynamics.

Hubs: High connectivity, slow dynamics (stable)
Periphery: Lower connectivity, faster dynamics (more variable)
"""
function long_tail_dynamics(; n::Int=75, d::Int=2, timesteps::Int=30,
                             hub_fraction::Float64=0.1, seed::Int=1234)
    rng = Random.MersenneTwister(seed)

    n_hubs = max(3, round(Int, n * hub_fraction))
    n_periphery = n - n_hubs

    L_series = Vector{Matrix{Float64}}(undef, timesteps)
    R_series = Vector{Matrix{Float64}}(undef, timesteps)

    # Hubs have high L values (high connectivity)
    hub_base = [0.7, 0.5]
    # Periphery has lower L values
    periphery_base = [0.3, 0.2]

    # Individual perturbations
    hub_perturbations = [0.05 * randn(rng, d) for _ in 1:n_hubs]
    periphery_perturbations = [0.1 * randn(rng, d) for _ in 1:n_periphery]

    for t in 1:timesteps
        L_t = zeros(d, n)

        # Hubs: slow oscillation
        theta_hub = 0.05 * (t - 1)
        hub_center = hub_base .+ 0.1 * [cos(theta_hub), sin(theta_hub)]
        for i in 1:n_hubs
            L_t[:, i] = hub_center + hub_perturbations[i]
        end

        # Periphery: faster, more erratic dynamics
        theta_periph = 0.15 * (t - 1)
        periph_center = periphery_base .+ 0.2 * [sin(theta_periph), cos(2*theta_periph)]
        for i in 1:n_periphery
            L_t[:, n_hubs + i] = periph_center + periphery_perturbations[i] * (1 + 0.1*sin(0.3*t))
        end

        L_series[t] = L_t
        R_series[t] = L_t .* [1.0; -1.0]
    end

    return (
        L_series = L_series,
        R_series = R_series,
        n = n,
        d = d,
        timesteps = timesteps,
        description = "Long-tail dynamics (n=" * string(n) * ", hubs=" * string(n_hubs) * ")"
    )
end

"""
    sample_adjacency(L, R; threshold=0.5, seed=nothing)

Sample a binary adjacency matrix from probability matrix P = L * R'.
"""
function sample_adjacency(L::Matrix, R::Matrix; threshold::Float64=0.5, seed=nothing)
    P = L' * R  # n × n probability matrix
    n = size(P, 1)

    # Clamp to [0, 1]
    P = clamp.(P, 0.0, 1.0)

    # Sample binary edges
    rng = isnothing(seed) ? Random.GLOBAL_RNG : Random.MersenneTwister(seed)
    A = zeros(Int, n, n)
    for i in 1:n
        for j in (i+1):n  # Upper triangle only (undirected)
            if rand(rng) < P[i, j]
                A[i, j] = 1
                A[j, i] = 1
            end
        end
    end

    return A
end

"""
    save_dataset(data, filename)

Save dataset to JSON file in the format expected by RDPGDynamics.
"""
function save_dataset(data, filename::String)
    # Convert L_series and R_series to the expected format (d × n arrays as lists)
    L_json = [[data.L_series[t][dim, :] for dim in 1:data.d] for t in 1:data.timesteps]
    R_json = [[data.R_series[t][dim, :] for dim in 1:data.d] for t in 1:data.timesteps]

    # Also sample adjacency matrices for visualization
    graph_series = [sample_adjacency(data.L_series[t], data.R_series[t]; seed=t)
                    for t in 1:data.timesteps]

    dict = Dict(
        "L_series" => L_json,
        "R_series" => R_json,
        "graph_series" => graph_series,
        "n" => data.n,
        "d" => data.d,
        "timesteps" => data.timesteps,
        "description" => data.description
    )

    mkpath(dirname(filename))
    open(filename, "w") do f
        JSON.print(f, dict, 2)
    end

    println("Saved: " * filename)
    println("  n=" * string(data.n) * ", d=" * string(data.d) * ", timesteps=" * string(data.timesteps))
end

function main()
    println("=" ^ 60)
    println("Generating Synthetic Temporal Network Data")
    println("=" ^ 60)

    # Generate larger versions of each dataset
    println("\n1. Oscillating Community (n=50)")
    data1 = oscillating_community(n=50, timesteps=30, omega=0.2)
    save_dataset(data1, "data/oscillating_community_n50.json")

    println("\n2. Merging Communities (n=60)")
    data2 = merging_communities(n1=30, n2=30, timesteps=30, merge_rate=0.05)
    save_dataset(data2, "data/merging_communities_n60.json")

    println("\n3. Long-Tail Dynamics (n=75)")
    data3 = long_tail_dynamics(n=75, timesteps=30, hub_fraction=0.1)
    save_dataset(data3, "data/long_tail_n75.json")

    println("\n" * "=" ^ 60)
    println("Data generation complete!")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
