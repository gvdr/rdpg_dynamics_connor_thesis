#!/usr/bin/env -S julia --project
#=
GB-DASE Gibbs Sampler - Julia Implementation
Following the reference: https://github.com/joshloyal/DynamicRDPG

Key algorithm:
1. Initialize X via ASE + Procrustes
2. For each iteration:
   a. For each node i: sample x_i from Gaussian conditional
   b. Sample σ from Inverse-Gamma (half-Cauchy via auxiliary ν)
   c. Optionally sample λ
3. Post-process: align all samples to reference

Reference: arXiv:2509.19748
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using CairoMakie
using ProgressMeter

using RDPGDynamics  # For svd_embedding, ortho_procrustes_RM

#=============================================================================
Adjacency Spectral Embedding (ASE) - matches reference ase()
=============================================================================#

"""
ASE via eigendecomposition of adjacency matrix.
Returns n × d embedding.
"""
function ase(A::AbstractMatrix, d::Int)
    # Use eigen on symmetric matrix (like scipy.sparse.linalg.eigsh)
    F = eigen(Symmetric(Matrix(A)))

    # Get top d eigenvalues by magnitude (sorted ascending, so take last d)
    idx = sortperm(abs.(F.values), rev=true)[1:d]
    eigvals = F.values[idx]
    eigvecs = F.vectors[:, idx]

    # X = eigvecs * sqrt(|eigvals|)
    return eigvecs .* sqrt.(abs.(eigvals))'
end

#=============================================================================
Procrustes alignment - matches reference smooth_positions_procrustes()
=============================================================================#

"""
Sequentially align X[t] to X[t-1] via Procrustes.
X is (m, n, d) array.
"""
function smooth_positions_procrustes!(X::Array{Float64,3})
    m = size(X, 1)
    for t in 2:m
        # orthogonal_procrustes finds R minimizing ||X[t] @ R - X[t-1]||
        R = ortho_procrustes_RM(X[t, :, :]', X[t-1, :, :]')
        X[t, :, :] = X[t, :, :] * R
    end
    return X
end

#=============================================================================
Difference matrix and precision matrix construction
=============================================================================#

"""
r-th order difference matrix D such that D @ x gives r-th differences.
Returns (m-r) × m matrix.
"""
function difference_matrix(m::Int, r::Int)
    D = Matrix{Float64}(I, m, m)
    for _ in 1:r
        D = diff(D, dims=1)
    end
    return D
end

"""
Precision matrix K = kron(D'D, I_d) for r-th order random walk prior.
Returns (m*d) × (m*d) matrix.
"""
function rw_precision_matrix(m::Int, d::Int, r::Int)
    D = difference_matrix(m, r)
    DtD = D' * D
    return kron(DtD, Matrix{Float64}(I, d, d))
end

"""
Initial position prior matrix K_init.
Adds 1/prior_std² to first r diagonal blocks.
"""
function initial_position_prior(m::Int, d::Int, r::Int, prior_std::Float64)
    K_init_time = zeros(m, m)
    for s in 1:r
        K_init_time[s, s] = 1.0 / (prior_std^2)
    end
    return kron(K_init_time, Matrix{Float64}(I, d, d))
end

#=============================================================================
GB-DASE Gibbs Sampler
=============================================================================#

"""
    gbdase_gibbs(A_list, d; kwargs...)

GB-DASE Gibbs sampler for dynamic RDPG.

# Arguments
- `A_list`: Vector of T adjacency matrices (can be sparse or dense)
- `d`: Embedding dimension

# Keyword Arguments
- `rw_order`: Order of random walk prior (default: 2)
- `prior_std`: Std for initial position prior (default: 10.0)
- `n_burnin`: Burn-in iterations (default: 500)
- `n_samples`: Post-burnin samples (default: 1000)
- `sample_scale`: Whether to sample λ (default: false)
- `scale`: Fixed scale λ, or "auto" (default: "auto")
- `seed`: Random seed (default: 42)
- `verbose`: Print progress (default: true)

# Returns
Named tuple with:
- `X_mean`: Posterior mean embedding (m, n, d)
- `sigma_mean`: Posterior mean σ (n,)
- `X_samples`: All samples (n_samples, m, n, d)
- `sigma_samples`: All σ samples (n_samples, n)
"""
function gbdase_gibbs(A_list::Vector{<:AbstractMatrix}, d::Int;
                      rw_order::Int=2,
                      prior_std::Float64=10.0,
                      n_burnin::Int=500,
                      n_samples::Int=1000,
                      sample_scale::Bool=false,
                      scale::Union{Float64,String}="auto",
                      seed::Int=42,
                      verbose::Bool=true)

    rng = Xoshiro(seed)

    m = length(A_list)  # number of time points
    n = size(A_list[1], 1)  # number of nodes
    r = rw_order

    verbose && println("GB-DASE Gibbs: n=" * string(n) * " nodes, d=" * string(d) *
                       " dims, m=" * string(m) * " timesteps, r=" * string(r))
    verbose && println("Running " * string(n_burnin) * " burn-in + " * string(n_samples) * " samples")

    # Convert to sparse CSC for efficient column access
    Y = [sparse(A_list[t]) for t in 1:m]

    # Vectorize lower triangular for scale computation
    n_dyads = div(n * (n - 1), 2)
    y_vec = zeros(m, n_dyads)
    idx = 1
    for i in 2:n
        for j in 1:(i-1)
            for t in 1:m
                y_vec[t, idx] = A_list[t][i, j]
            end
            idx += 1
        end
    end

    # Auto-scale: λ = 1 / var(y_vec)
    if scale == "auto"
        scale_val = 1.0 / var(y_vec[:])
        verbose && println("Auto scale λ = " * string(round(scale_val, digits=4)))
    else
        scale_val = scale
    end

    #=========================================================================
    # Initialization
    =========================================================================#

    # ASE per timestep
    X = zeros(m, n, d)
    for t in 1:m
        X[t, :, :] = ase(A_list[t], d)
    end

    # Procrustes alignment
    smooth_positions_procrustes!(X)

    # Transpose to (n, d, m) for efficient node-wise updates
    # X_internal[i, :, t] = position of node i at time t
    X_internal = permutedims(X, (2, 3, 1))

    # Initial σ² from empirical variance of ORDER-1 differences (velocity)
    # NOTE: Reference uses order-1 for init, then order-r in sampling!
    sigma = zeros(n)
    for i in 1:n
        diffs = diff(X_internal[i, :, :], dims=2)  # Order-1 differences
        sigma[i] = mean(diffs.^2)
    end
    sigma = max.(sigma, 1e-6)  # Avoid zeros

    # Auxiliary variable for half-Cauchy
    nu = ones(n)

    #=========================================================================
    # Precompute precision matrices
    =========================================================================#

    # Random walk precision K (md × md)
    K = rw_precision_matrix(m, d, r)

    # Initial position prior K_init (md × md)
    K_init = initial_position_prior(m, d, r, prior_std)

    #=========================================================================
    # Initialize running statistics XtX
    =========================================================================#

    # XtX[t] = Σ_i x_i(t) x_i(t)' is d × d
    XtX = [zeros(d, d) for _ in 1:m]
    for t in 1:m
        for i in 1:n
            xi_t = X_internal[i, :, t]
            XtX[t] .+= xi_t * xi_t'
        end
    end

    #=========================================================================
    # Storage for samples
    =========================================================================#

    X_samples = zeros(n_samples, m, n, d)
    sigma_samples = zeros(n_samples, n)
    scale_samples = sample_scale ? zeros(n_samples) : nothing

    #=========================================================================
    # Diagnostic: check initialization quality
    =========================================================================#

    function compute_p_error_internal(X_int)
        # X_int is (n, d, m)
        err = 0.0
        for t in 1:m
            Xt = X_int[:, :, t]  # (n, d)
            P_est = Xt * Xt'
            # Need P_true from outside... skip for now
        end
    end

    # Check initial P reconstruction
    P_init_err = 0.0
    P_init_norm = 0.0
    for t in 1:m
        Xt = X_internal[:, :, t]  # (n, d) - no transpose needed!
        P_init = Xt * Xt'  # (n, n)
        P_obs = A_list[t]
        P_init_err += norm(P_init - P_obs)^2
        P_init_norm += norm(P_obs)^2
    end
    verbose && println("Initial P error (vs observed A): " *
                       string(round(100*sqrt(P_init_err/P_init_norm), digits=1)) * "%")

    # Check scale of initial positions
    X_init_mean = mean(abs.(X_internal))
    X_init_std = std(X_internal)
    verbose && println("Initial X: mean(|X|)=" * string(round(X_init_mean, digits=3)) *
                       ", std(X)=" * string(round(X_init_std, digits=3)))

    #=========================================================================
    # Gibbs sampling loop
    =========================================================================#

    total_iters = n_burnin + n_samples

    @showprogress desc="Gibbs sampling" for iter in 1:total_iters

        #=====================================================================
        # Sample X node by node
        =====================================================================#

        for i in 1:n
            # Step 1: Remove node i from XtX, compute XtY
            XtY = zeros(d, m)  # d × m (will be reshaped to md)

            for t in 1:m
                xi_t = X_internal[i, :, t]
                XtX[t] .-= xi_t * xi_t'

                # XtY[:, t] = Σ_j A_ij * x_j(t)
                # For binary A: this is sum of neighbor positions
                # For continuous A: this is weighted sum
                # Use full matrix-vector product for correctness with any A
                for j in 1:n
                    if j != i
                        XtY[:, t] .+= A_list[t][i, j] * X_internal[j, :, t]
                    end
                end
            end

            # Step 2: Build precision matrix P (md × md)
            # P = (1/σ_i) * K + K_init + (λ/2) * I + λ * block_diag(XtX)

            P = (1.0 / sigma[i]) * K + K_init

            # Add λ/2 to diagonal
            for k in 1:(m*d)
                P[k, k] += 0.5 * scale_val
            end

            # Add λ * block_diag(XtX)
            for t in 1:m
                block_start = (t-1)*d + 1
                block_end = t*d
                P[block_start:block_end, block_start:block_end] .+= scale_val * XtX[t]
            end

            # Step 3: Cholesky decomposition
            # P should be positive definite
            P_sym = Symmetric(P)
            L = cholesky(P_sym)

            # Step 4: Solve for mean: P @ X_hat = λ * XtY
            XtY_vec = vec(XtY)  # Column-major: [XtY[:,1]; XtY[:,2]; ...]
            X_hat = L \ (scale_val * XtY_vec)

            # Step 5: Sample from N(X_hat, P^{-1})
            # z ~ N(0, I), then X_hat + L^{-T} @ z ~ N(X_hat, P^{-1})
            z = randn(rng, m * d)
            X_new_vec = X_hat + L.U \ z

            # Reshape back to (d, m)
            X_new = reshape(X_new_vec, d, m)
            X_internal[i, :, :] = X_new

            # Step 6: Add node i back to XtX
            for t in 1:m
                xi_t = X_internal[i, :, t]
                XtX[t] .+= xi_t * xi_t'
            end
        end

        #=====================================================================
        # Sample σ from Inverse-Gamma (half-Cauchy via auxiliary ν)
        =====================================================================#

        shape_sigma = 0.5 * ((m - r) * d + 1)

        for i in 1:n
            # Compute sum of squared r-th differences
            # diff(X_internal[i, :, :], dims=2) gives d × (m-1) for r=1
            diffs = X_internal[i, :, :]
            for _ in 1:r
                diffs = diff(diffs, dims=2)
            end
            sum_sq = sum(diffs.^2)

            scale_sigma = 0.5 * sum_sq + 1.0 / nu[i]

            # Sample σ² ~ InverseGamma(shape, scale)
            # NOTE: sigma stores VARIANCE (σ²), not std dev (σ)
            sigma[i] = rand(rng, InverseGamma(shape_sigma, scale_sigma))

            # Update auxiliary ν ~ InverseGamma(1, 1 + 1/σ²)
            nu[i] = rand(rng, InverseGamma(1.0, 1.0 + 1.0 / sigma[i]))
        end

        #=====================================================================
        # Optionally sample scale λ
        =====================================================================#

        if sample_scale
            # Compute reconstruction error on lower triangular
            x_out = permutedims(X_internal, (3, 1, 2))  # (m, n, d)

            err_sum = 0.0
            norm_sum = 0.0
            for t in 1:m
                Xt = x_out[t, :, :]
                XXt = Xt * Xt'
                for i in 2:n
                    for j in 1:(i-1)
                        err_sum += (y_vec[t, div((i-1)*(i-2), 2) + j] - XXt[i,j])^2
                    end
                end
                norm_sum += sum(Xt.^2)
            end

            a = 1e-3 + 0.25 * n * (n + 1) * m
            b = 1e-3 + 0.5 * err_sum + 0.25 * norm_sum
            scale_val = rand(rng, Gamma(a, 1.0 / b))
        end

        #=====================================================================
        # Periodic diagnostics
        =====================================================================#

        if verbose && iter % 200 == 0
            # Compute current P error vs observed A
            curr_err = 0.0
            curr_norm = 0.0
            for t in 1:m
                Xt = X_internal[:, :, t]  # (n, d) - no transpose needed!
                P_curr = Xt * Xt'  # (n, n)
                curr_err += norm(P_curr - A_list[t])^2
                curr_norm += norm(A_list[t])^2
            end
            curr_p_err = sqrt(curr_err / curr_norm)
            X_curr_mean = mean(abs.(X_internal))
            println("  Iter " * string(iter) * ": P_err=" *
                    string(round(100*curr_p_err, digits=1)) * "%, mean(|X|)=" *
                    string(round(X_curr_mean, digits=3)) *
                    ", σ_mean=" * string(round(mean(sigma), digits=4)))
        end

        #=====================================================================
        # Store sample after burn-in
        =====================================================================#

        if iter > n_burnin
            sample_idx = iter - n_burnin
            X_samples[sample_idx, :, :, :] = permutedims(X_internal, (3, 1, 2))
            sigma_samples[sample_idx, :] = sigma
            if sample_scale
                scale_samples[sample_idx] = scale_val
            end
        end
    end

    #=========================================================================
    # Post-processing: align all samples to reference
    =========================================================================#

    verbose && println("Post-processing: aligning samples...")

    # Smooth the last sample as reference
    X_ref = copy(X_samples[end, :, :, :])
    smooth_positions_procrustes!(X_ref)

    # Align all samples to this reference
    for s in 1:n_samples
        for t in 1:m
            R = ortho_procrustes_RM(X_samples[s, t, :, :]', X_ref[t, :, :]')
            X_samples[s, t, :, :] = X_samples[s, t, :, :] * R
        end
    end

    # Compute posterior means
    X_mean = dropdims(mean(X_samples, dims=1), dims=1)
    sigma_mean = dropdims(mean(sigma_samples, dims=1), dims=1)

    # sigma_mean stores variance (σ²), report std dev (σ) for interpretability
    verbose && println("Done. σ_mean (std dev) range: [" * string(round(sqrt(minimum(sigma_mean)), digits=4)) *
                       ", " * string(round(sqrt(maximum(sigma_mean)), digits=4)) * "]")

    return (
        X_mean = X_mean,
        sigma_mean = sigma_mean,
        X_samples = X_samples,
        sigma_samples = sigma_samples,
        scale_samples = scale_samples
    )
end

#=============================================================================
Test on synthetic data
=============================================================================#

function main()
    println("=" ^ 70)
    println("GB-DASE GIBBS SAMPLER TEST")
    println("=" ^ 70)

    # Generate simple test data: random walk in 2D
    Random.seed!(42)
    n = 30  # nodes
    d = 2   # dimensions
    m = 20  # timesteps
    K_avg = 50  # number of samples to average for less noisy P_hat

    # True positions: smooth random walk
    X_true = zeros(m, n, d)
    X_true[1, :, :] = 0.4 .+ 0.15 * randn(n, d)
    X_true[1, :, :] = clamp.(X_true[1, :, :], 0.15, 0.85)

    for t in 2:m
        X_true[t, :, :] = X_true[t-1, :, :] + 0.015 * randn(n, d)
        X_true[t, :, :] = clamp.(X_true[t, :, :], 0.1, 0.9)
    end

    # Generate probability matrices
    P_true = Vector{Matrix{Float64}}(undef, m)
    for t in 1:m
        P_true[t] = clamp.(X_true[t, :, :] * X_true[t, :, :]', 0.0, 1.0)
    end

    # Generate AVERAGED adjacencies (less noisy - for testing)
    A_avg = Vector{Matrix{Float64}}(undef, m)
    for t in 1:m
        A_sum = zeros(n, n)
        for _ in 1:K_avg
            A_sample = Float64.(rand(n, n) .< P_true[t])
            A_sample = (A_sample + A_sample') / 2
            A_sample[diagind(A_sample)] .= 0.0
            A_sample = Float64.(A_sample .> 0.5)
            A_sum .+= A_sample
        end
        A_avg[t] = A_sum / K_avg
    end

    # Generate SINGLE BINARY adjacencies (very noisy - real use case)
    A_binary = Vector{Matrix{Float64}}(undef, m)
    for t in 1:m
        A_binary[t] = Float64.(rand(n, n) .< P_true[t])
        A_binary[t] = (A_binary[t] + A_binary[t]') / 2
        A_binary[t][diagind(A_binary[t])] .= 0.0
        A_binary[t] = Float64.(A_binary[t] .> 0.5)
    end

    println("\nGenerated data: n=" * string(n) * ", d=" * string(d) * ", m=" * string(m))
    avg_noise = mean([norm(A_avg[t] - P_true[t]) / norm(P_true[t]) for t in 1:m])
    binary_noise = mean([norm(A_binary[t] - P_true[t]) / norm(P_true[t]) for t in 1:m])
    println("Averaged A noise (K=" * string(K_avg) * "): " * string(round(100*avg_noise, digits=1)) * "%")
    println("Binary A noise: " * string(round(100*binary_noise, digits=1)) * "%")

    # Test both: averaged (less noisy) and binary (very noisy)
    # First run on averaged to verify algorithm works when SNR > 1
    println("\n" * "=" ^ 70)
    println("TEST 1: AVERAGED ADJACENCIES (SNR > 1)")
    println("=" ^ 70)

    println("\n--- Running GB-DASE Gibbs sampler on AVERAGED A ---")
    result_avg = gbdase_gibbs(A_avg, d;
                          rw_order=2,
                          n_burnin=300,
                          n_samples=500,
                          scale="auto",
                          verbose=true)

    # Compute P-error for averaged case
    function p_error(X_est)
        err = 0.0
        norm_true = 0.0
        for t in 1:m
            P_est = X_est[t, :, :] * X_est[t, :, :]'
            err += norm(P_est - P_true[t])^2
            norm_true += norm(P_true[t])^2
        end
        return sqrt(err / norm_true)
    end

    println("GB-DASE on averaged A: P-error = " * string(round(100 * p_error(result_avg.X_mean), digits=1)) * "%")

    println("\n" * "=" ^ 70)
    println("TEST 2: BINARY ADJACENCIES (SNR < 1)")
    println("=" ^ 70)

    # Now test on binary
    A_test = A_binary

    println("\n--- Running GB-DASE Gibbs sampler on BINARY A ---")
    println("Using 2500 burn-in + 2500 samples (matching reference)")
    result = gbdase_gibbs(A_test, d;
                          rw_order=2,
                          n_burnin=2500,
                          n_samples=2500,
                          scale="auto",
                          verbose=true)

    # TWO ALIGNMENT STRATEGIES:

    # 1. GLOBAL alignment: ONE rotation for all timesteps (what we need for dynamics)
    function align_global(X_est)
        X_aligned = copy(X_est)
        # Find R using first timestep only
        F = svd(X_est[1, :, :]' * X_true[1, :, :])
        R = F.U * F.Vt
        for t in 1:m
            X_aligned[t, :, :] = X_est[t, :, :] * R
        end
        return X_aligned
    end

    # 2. ORACLE per-timestep alignment: separate R_t for each timestep (their evaluation)
    # This DESTROYS trajectory structure but makes pointwise accuracy look good
    function align_oracle_per_timestep(X_est)
        X_aligned = copy(X_est)
        for t in 1:m
            F = svd(X_est[t, :, :]' * X_true[t, :, :])
            R_t = F.U * F.Vt
            X_aligned[t, :, :] = X_est[t, :, :] * R_t
        end
        return X_aligned
    end

    # Pointwise error: average distance from estimated to true position
    function pointwise_error(X_est)
        total = 0.0
        for t in 1:m
            for i in 1:n
                total += norm(X_est[t, i, :] - X_true[t, i, :])
            end
        end
        return total / (m * n)
    end

    X_gbdase_global = align_global(result.X_mean)
    X_gbdase_oracle = align_oracle_per_timestep(result.X_mean)

    # Compare with DUASE
    println("\n--- Running DUASE for comparison ---")
    function duase_embed(A_input, d)
        m_loc = length(A_input)
        Unfolded = hcat(A_input...)
        U, S, V = svd(Unfolded)
        G = U[:, 1:d]

        X_duase = zeros(m_loc, size(A_input[1], 1), d)
        for t in 1:m_loc
            Qt = G' * A_input[t] * G
            Qt_sym = (Qt + Qt') / 2
            eig = eigen(Symmetric(Qt_sym))
            sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
            X_duase[t, :, :] = G * sqrt_Q
        end
        return X_duase
    end

    X_duase_raw = duase_embed(A_test, d)
    X_duase_global = align_global(X_duase_raw)
    X_duase_oracle = align_oracle_per_timestep(X_duase_raw)

    # SVD + Procrustes
    println("\n--- Running SVD + Procrustes for comparison ---")
    X_svd = zeros(m, n, d)
    for t in 1:m
        emb = svd_embedding(A_test[t], d)
        X_svd[t, :, :] = emb.L_hat
    end
    for t in 2:m
        R = ortho_procrustes_RM(X_svd[t, :, :]', X_svd[t-1, :, :]')
        X_svd[t, :, :] = X_svd[t, :, :] * R
    end
    X_svd_global = align_global(X_svd)
    X_svd_oracle = align_oracle_per_timestep(X_svd)

    # Results
    println("\n" * "=" ^ 70)
    println("RESULTS: P-error (gauge-invariant)")
    println("=" ^ 70)
    println("Method                     P-error")
    println("-" ^ 50)
    println("SVD + Procrustes           " * string(round(100 * p_error(X_svd_global), digits=1)) * "%")
    println("DUASE                      " * string(round(100 * p_error(X_duase_global), digits=1)) * "%")
    println("GB-DASE Gibbs              " * string(round(100 * p_error(X_gbdase_global), digits=1)) * "%")

    println("\n" * "=" ^ 70)
    println("COMPARISON: Global vs Oracle Per-Timestep Alignment")
    println("=" ^ 70)
    println("Method                  Pointwise Error (Global)    Pointwise Error (Oracle)")
    println("-" ^ 75)
    println("SVD + Procrustes        " *
            string(round(pointwise_error(X_svd_global), digits=4)) * "                       " *
            string(round(pointwise_error(X_svd_oracle), digits=4)))
    println("DUASE                   " *
            string(round(pointwise_error(X_duase_global), digits=4)) * "                       " *
            string(round(pointwise_error(X_duase_oracle), digits=4)))
    println("GB-DASE Gibbs           " *
            string(round(pointwise_error(X_gbdase_global), digits=4)) * "                       " *
            string(round(pointwise_error(X_gbdase_oracle), digits=4)))

    # Trajectory smoothness comparison
    function mean_velocity_magnitude(X)
        total = 0.0
        count = 0
        for t in 1:(m-1)
            for i in 1:n
                v = X[t+1, i, :] - X[t, i, :]
                total += norm(v)
                count += 1
            end
        end
        return total / count
    end

    println("\nMean velocity magnitude (using GLOBAL alignment):")
    println("True                       " * string(round(mean_velocity_magnitude(X_true), digits=4)))
    println("SVD + Procrustes           " * string(round(mean_velocity_magnitude(X_svd_global), digits=4)))
    println("DUASE                      " * string(round(mean_velocity_magnitude(X_duase_global), digits=4)))
    println("GB-DASE Gibbs              " * string(round(mean_velocity_magnitude(X_gbdase_global), digits=4)))

    println("\nMean velocity magnitude (using ORACLE per-timestep - MEANINGLESS for dynamics!):")
    println("True                       " * string(round(mean_velocity_magnitude(X_true), digits=4)))
    println("SVD + Procrustes           " * string(round(mean_velocity_magnitude(X_svd_oracle), digits=4)))
    println("DUASE                      " * string(round(mean_velocity_magnitude(X_duase_oracle), digits=4)))
    println("GB-DASE Gibbs              " * string(round(mean_velocity_magnitude(X_gbdase_oracle), digits=4)))

    # Plot
    println("\n" * "=" ^ 70)
    println("GENERATING PLOTS")
    println("=" ^ 70)

    fig = Figure(size=(2000, 1000))

    function plot_trajectories!(ax, X, title_str)
        for i in 1:n
            traj = X[:, i, :]
            lines!(ax, traj[:, 1], traj[:, 2], color=(:blue, 0.5), linewidth=1)
            scatter!(ax, [traj[1, 1]], [traj[1, 2]], color=:green, markersize=5)
            scatter!(ax, [traj[end, 1]], [traj[end, 2]], color=:red, markersize=5)
        end
        ax.title = title_str
    end

    # Row 1: Global alignment (what we need for dynamics)
    Label(fig[1, 1:4], "GLOBAL ALIGNMENT (single R for all timesteps)", fontsize=16, tellwidth=false)

    ax1 = Axis(fig[2, 1], xlabel="x₁", ylabel="x₂", aspect=1)
    ax2 = Axis(fig[2, 2], xlabel="x₁", ylabel="x₂", aspect=1)
    ax3 = Axis(fig[2, 3], xlabel="x₁", ylabel="x₂", aspect=1)
    ax4 = Axis(fig[2, 4], xlabel="x₁", ylabel="x₂", aspect=1)

    plot_trajectories!(ax1, X_true, "True")
    plot_trajectories!(ax2, X_svd_global, "SVD+Proc (Global)\nP-err=" * string(round(100*p_error(X_svd_global), digits=1)) * "%")
    plot_trajectories!(ax3, X_duase_global, "DUASE (Global)\nP-err=" * string(round(100*p_error(X_duase_global), digits=1)) * "%")
    plot_trajectories!(ax4, X_gbdase_global, "GB-DASE (Global)\nP-err=" * string(round(100*p_error(X_gbdase_global), digits=1)) * "%")

    # Row 2: Oracle per-timestep alignment (their evaluation - destroys trajectories)
    Label(fig[3, 1:4], "ORACLE PER-TIMESTEP ALIGNMENT (separate R_t - destroys trajectory!)", fontsize=16, tellwidth=false)

    ax5 = Axis(fig[4, 1], xlabel="x₁", ylabel="x₂", aspect=1)
    ax6 = Axis(fig[4, 2], xlabel="x₁", ylabel="x₂", aspect=1)
    ax7 = Axis(fig[4, 3], xlabel="x₁", ylabel="x₂", aspect=1)
    ax8 = Axis(fig[4, 4], xlabel="x₁", ylabel="x₂", aspect=1)

    plot_trajectories!(ax5, X_true, "True")
    plot_trajectories!(ax6, X_svd_oracle, "SVD+Proc (Oracle)")
    plot_trajectories!(ax7, X_duase_oracle, "DUASE (Oracle)")
    plot_trajectories!(ax8, X_gbdase_oracle, "GB-DASE (Oracle)")

    save("results/gbdase_gibbs_test.png", fig)
    println("\nSaved: results/gbdase_gibbs_test.png")

    return result
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
