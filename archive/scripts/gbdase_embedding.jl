#!/usr/bin/env -S julia --project
#=
GB-DASE style embedding with r-th order random walk priors

Key insight: r-th order random walk prior penalizes r-th derivative
- r=1: penalizes velocity → linearizes (what we had)
- r=2: penalizes acceleration → allows curvature!
- r=3: penalizes jerk → allows smooth acceleration

Reference: arxiv.org/abs/2509.19748

=============================================================================
METHODOLOGICAL COMPARISON: Gibbs Sampling vs MAP Estimation
=============================================================================

The original GB-DASE paper uses Gibbs sampling (MCMC):
1. Sample X | σ, Y from Gaussian conditional
2. Sample σ | X from Inverse-Gamma
3. Repeat for burn-in + samples
4. Average over posterior samples

Our implementation uses MAP estimation (optimization):
1. Optimize X given fixed σ using LBFGS
2. Update σ analytically (MLE given X)
3. Repeat until convergence
4. Return single point estimate

Key differences:
- Gibbs: Full posterior distribution, uncertainty quantification
- MAP: Mode of posterior, faster but no uncertainty
- Both should converge to similar point estimates for large data

The objective being optimized (MAP) / sampled from (Gibbs):
  log π(X|Y) ∝ -λ_P * Σ_t ||X(t)X(t)' - P(t)||²_F  [likelihood]
              - Σ_i (1/σᵢ²) * x_i' K x_i            [prior]

where K = D'D is the r-th order difference precision matrix.
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using LinearAlgebra
using Optim
using Statistics
using Random
using CairoMakie
using OrdinaryDiffEq

using RDPGDynamics

#=============================================================================
Random Walk Prior Structure
=============================================================================#

"""
    difference_matrix(T, r)

Create r-th order difference matrix D such that D*x gives r-th differences.
- r=1: D[t,:] gives x[t+1] - x[t]
- r=2: D[t,:] gives x[t+2] - 2x[t+1] + x[t]
"""
function difference_matrix(T::Int, r::Int)
    D = Matrix{Float64}(I, T, T)
    for _ in 1:r
        D = diff(D, dims=1)
    end
    return D
end

"""
    rw_precision_matrix(T, r)

Precision matrix K = D'D for r-th order random walk prior.
This is a banded matrix with bandwidth 2r.
"""
function rw_precision_matrix(T::Int, r::Int)
    D = difference_matrix(T, r)
    return D' * D
end

#=============================================================================
GB-DASE MAP Estimation (mode of posterior)
=============================================================================#

"""
    gbdase_embed(P_list, d; rw_order=2, λ_P=1.0, node_specific_sigma=true, ...)

GB-DASE style embedding using r-th order random walk prior.

Arguments:
- P_list: Vector of T probability matrices
- d: embedding dimension
- rw_order: order of random walk prior (2 = penalize acceleration, not velocity!)
- λ_P: weight on P reconstruction loss
- node_specific_sigma: if true, learn σᵢ per node; otherwise single σ

Returns named tuple with X_list, sigma values, and diagnostics.
"""
function gbdase_embed(P_list::Vector{Matrix{Float64}}, d::Int;
                      rw_order::Int=2,
                      λ_P::Float64=1.0,
                      node_specific_sigma::Bool=true,
                      estimate_sigma::Bool=true,
                      sigma_init::Float64=0.1,
                      max_iter::Int=100,
                      X_init::Union{Nothing, Vector{Matrix{Float64}}}=nothing,
                      use_procrustes_init::Bool=true,
                      verbose::Bool=true)

    T = length(P_list)
    n = size(P_list[1], 1)

    # Build precision matrix for random walk prior
    K = rw_precision_matrix(T, rw_order)

    verbose && println("Using r=" * string(rw_order) * " random walk prior (penalizes derivative order " * string(rw_order) * ")")

    # Initialize: use provided X_init, or compute from SVD
    if X_init === nothing
        X_init = Vector{Matrix{Float64}}(undef, T)
        for t in 1:T
            emb = svd_embedding(P_list[t], d)
            X_init[t] = emb.L_hat
        end
        if use_procrustes_init
            for t in 2:T
                Omega = ortho_procrustes_RM(X_init[t]', X_init[t-1]')
                X_init[t] = X_init[t] * Omega
            end
            verbose && println("Initialized with SVD + Procrustes")
        else
            verbose && println("Initialized with raw SVD (NO Procrustes)")
        end
    else
        verbose && println("Using provided initialization")
    end

    # Initialize sigma (per-node or global)
    n_sigma = node_specific_sigma ? n : 1
    sigma = fill(sigma_init, n_sigma)

    # Pack X into vector for optimization
    function pack_X(X_list)
        return vcat([vec(X) for X in X_list]...)
    end

    function unpack_X(x_vec)
        X_list = Vector{Matrix{Float64}}(undef, T)
        offset = 0
        for t in 1:T
            X_list[t] = reshape(x_vec[offset+1:offset+n*d], n, d)
            offset += n * d
        end
        return X_list
    end

    # Objective: P reconstruction + r-th order random walk prior
    # Use FULL matrix including diagonal - diagonal P[i,i] = ||x_i||² carries norm info!
    function objective(x_vec, sigma)
        X_list = unpack_X(x_vec)

        # P reconstruction loss - FULL MATRIX (diagonal has norm information)
        loss_P = 0.0
        for t in 1:T
            P_hat = X_list[t] * X_list[t]'
            loss_P += norm(P_hat - P_list[t])^2
        end

        # Random walk prior loss: sum over nodes of x_i' * K * x_i / sigma_i^2
        loss_rw = 0.0
        for i in 1:n
            # Extract trajectory of node i: T x d matrix
            traj_i = vcat([X_list[t][i:i, :] for t in 1:T]...)  # T x d

            sigma_i = node_specific_sigma ? sigma[i] : sigma[1]

            # For each dimension, compute x' * K * x
            for dim in 1:d
                x_dim = traj_i[:, dim]  # T-vector
                loss_rw += (x_dim' * K * x_dim) / (sigma_i^2)
            end
        end

        return λ_P * loss_P + loss_rw
    end

    # Analytic gradient for X (given fixed sigma)
    function gradient_X!(G, x_vec, sigma)
        X_list = unpack_X(x_vec)
        G .= 0.0

        offset = 0
        for t in 1:T
            # Gradient from P reconstruction: 4 * (XX' - P) * X
            P_hat = X_list[t] * X_list[t]'
            dL_dX_P = 4.0 * λ_P * (P_hat - P_list[t]) * X_list[t]

            G[offset+1:offset+n*d] .= vec(dL_dX_P)
            offset += n * d
        end

        # Gradient from random walk prior
        # For node i, dimension j: d/dx_{it,j} of (x_j' K x_j / σᵢ²)
        # = 2 * K[t,:] * x_j / σᵢ²
        for i in 1:n
            sigma_i = node_specific_sigma ? sigma[i] : sigma[1]

            for dim in 1:d
                # Extract trajectory for node i, dimension dim
                traj = [X_list[t][i, dim] for t in 1:T]

                # Gradient: 2 * K * traj / sigma_i^2
                grad_traj = 2.0 * K * traj / (sigma_i^2)

                # Add to G
                for t in 1:T
                    idx = (t-1) * n * d + (i-1) * d + dim
                    G[idx] += grad_traj[t]
                end
            end
        end
    end

    # Estimate sigma from current X (closed form for inverse-gamma posterior)
    function update_sigma(X_list)
        new_sigma = similar(sigma)

        for i in 1:n
            # Extract trajectory
            traj_i = vcat([X_list[t][i:i, :] for t in 1:T]...)  # T x d

            # Compute sum of x' * K * x over dimensions
            sum_quad = 0.0
            for dim in 1:d
                x_dim = traj_i[:, dim]
                sum_quad += x_dim' * K * x_dim
            end

            # MLE for sigma_i^2: sum_quad / (d * (T - rw_order))
            # (T - rw_order) is effective number of observations
            df = d * (T - rw_order)
            sigma_sq = sum_quad / df

            if node_specific_sigma
                new_sigma[i] = sqrt(max(sigma_sq, 1e-6))
            else
                # Accumulate for global sigma
                new_sigma[1] = get(new_sigma, 1, 0.0) + sum_quad
            end
        end

        if !node_specific_sigma
            df_total = n * d * (T - rw_order)
            new_sigma[1] = sqrt(max(new_sigma[1] / df_total, 1e-6))
        end

        return new_sigma
    end

    # Alternating optimization: X given sigma, then sigma given X
    x_vec = pack_X(X_init)

    for iter in 1:max_iter
        # Optimize X given sigma
        result = optimize(
            x -> objective(x, sigma),
            (G, x) -> gradient_X!(G, x, sigma),
            x_vec,
            LBFGS(),
            Optim.Options(iterations=100, g_tol=1e-6, show_trace=false)
        )
        x_vec = Optim.minimizer(result)

        # Update sigma given X
        if estimate_sigma
            X_list = unpack_X(x_vec)
            sigma_new = update_sigma(X_list)

            sigma_change = norm(sigma_new - sigma) / (norm(sigma) + 1e-10)
            sigma = sigma_new

            if verbose && iter % 10 == 0
                loss = Optim.minimum(result)
                println("Iter " * string(iter) * ": loss=" * string(round(loss, digits=2)) *
                       ", sigma range=[" * string(round(minimum(sigma), digits=4)) * ", " *
                       string(round(maximum(sigma), digits=4)) * "]")
            end

            # Convergence check
            if sigma_change < 1e-4
                verbose && println("Converged at iteration " * string(iter))
                break
            end
        else
            break  # No sigma update, single optimization is enough
        end
    end

    X_final = unpack_X(x_vec)

    # Compute final P error
    P_errors = [norm(X_final[t] * X_final[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T]
    mean_P_error = mean(P_errors)

    return (X_list=X_final, sigma=sigma, P_error=mean_P_error, K=K)
end

#=============================================================================
GIBBS SAMPLING Implementation (for comparison with MAP)
=============================================================================#
#
# Key difference from MAP:
# - σ is SAMPLED from Inverse-Gamma, not point-estimated
# - X is sampled via Metropolis-Hastings (P = XX' is nonlinear)
#
# In MAP: σ² = x'Kx/df → makes prior term constant!
# In Gibbs: σ² ~ Inverse-Gamma(α + df/2, β + x'Kx/2) → has variance
#

using Distributions

function gbdase_gibbs(P_list::Vector{Matrix{Float64}}, d::Int;
                      rw_order::Int=2,
                      λ_P::Float64=1.0,
                      node_specific_sigma::Bool=true,
                      n_samples::Int=1000,
                      n_burnin::Int=500,
                      proposal_std::Float64=0.01,
                      sigma_prior_alpha::Float64=2.0,
                      sigma_prior_beta::Float64=0.01,
                      X_init::Union{Nothing, Vector{Matrix{Float64}}}=nothing,
                      use_procrustes_init::Bool=true,
                      verbose::Bool=true)

    T = length(P_list)
    n = size(P_list[1], 1)

    # Build precision matrix for random walk prior
    K = rw_precision_matrix(T, rw_order)

    verbose && println("Gibbs sampler: r=" * string(rw_order) * ", " * string(n_burnin) * " burn-in + " * string(n_samples) * " samples")

    # Initialize: use provided X_init, or compute from SVD
    if X_init === nothing
        X_current = Vector{Matrix{Float64}}(undef, T)
        for t in 1:T
            emb = svd_embedding(P_list[t], d)
            X_current[t] = emb.L_hat
        end
        if use_procrustes_init
            for t in 2:T
                Omega = ortho_procrustes_RM(X_current[t]', X_current[t-1]')
                X_current[t] = X_current[t] * Omega
            end
            verbose && println("Initialized with SVD + Procrustes")
        else
            verbose && println("Initialized with raw SVD (NO Procrustes)")
        end
    else
        X_current = [copy(X_init[t]) for t in 1:T]
        verbose && println("Using provided initialization")
    end

    # Initialize sigma
    n_sigma = node_specific_sigma ? n : 1
    sigma_current = fill(0.1, n_sigma)

    # Log-posterior function (up to constant)
    function log_posterior(X_list, sigma)
        # P reconstruction term
        log_p = 0.0
        for t in 1:T
            P_hat = X_list[t] * X_list[t]'
            log_p -= λ_P * norm(P_hat - P_list[t])^2
        end

        # Random walk prior term
        for i in 1:n
            traj_i = vcat([X_list[t][i:i, :] for t in 1:T]...)
            sigma_i = node_specific_sigma ? sigma[i] : sigma[1]

            for dim in 1:d
                x_dim = traj_i[:, dim]
                log_p -= (x_dim' * K * x_dim) / (2 * sigma_i^2)
                log_p -= T * log(sigma_i)  # normalizing constant
            end
        end

        return log_p
    end

    # Compute quadratic form x'Kx for sigma update
    function compute_quad_form(X_list, i)
        traj_i = vcat([X_list[t][i:i, :] for t in 1:T]...)
        sum_quad = 0.0
        for dim in 1:d
            x_dim = traj_i[:, dim]
            sum_quad += x_dim' * K * x_dim
        end
        return sum_quad
    end

    # Storage for samples
    X_samples = Vector{Vector{Matrix{Float64}}}()
    sigma_samples = Vector{Vector{Float64}}()

    n_accept = 0
    n_total = 0

    for iter in 1:(n_burnin + n_samples)
        # --- Step 1: Metropolis-Hastings for X ---
        # Propose small perturbation to each node's trajectory
        X_proposed = [copy(X_current[t]) for t in 1:T]

        for i in 1:n
            for t in 1:T
                X_proposed[t][i, :] .+= proposal_std * randn(d)
            end
        end

        # Compute acceptance probability
        log_p_current = log_posterior(X_current, sigma_current)
        log_p_proposed = log_posterior(X_proposed, sigma_current)

        log_alpha = log_p_proposed - log_p_current
        n_total += 1

        if log(rand()) < log_alpha
            X_current = X_proposed
            n_accept += 1
        end

        # --- Step 2: Gibbs update for sigma (conjugate Inverse-Gamma) ---
        # Prior: σ² ~ Inverse-Gamma(α₀, β₀)
        # Posterior: σ² | X ~ Inverse-Gamma(α₀ + df/2, β₀ + x'Kx/2)
        df = d * (T - rw_order)

        if node_specific_sigma
            for i in 1:n
                quad_form = compute_quad_form(X_current, i)
                alpha_post = sigma_prior_alpha + df / 2
                beta_post = sigma_prior_beta + quad_form / 2

                # Sample σ² from Inverse-Gamma
                sigma_sq = rand(InverseGamma(alpha_post, beta_post))
                sigma_current[i] = sqrt(sigma_sq)
            end
        else
            total_quad = sum(compute_quad_form(X_current, i) for i in 1:n)
            alpha_post = sigma_prior_alpha + n * df / 2
            beta_post = sigma_prior_beta + total_quad / 2

            sigma_sq = rand(InverseGamma(alpha_post, beta_post))
            sigma_current[1] = sqrt(sigma_sq)
        end

        # Store sample after burn-in
        if iter > n_burnin
            push!(X_samples, [copy(X_current[t]) for t in 1:T])
            push!(sigma_samples, copy(sigma_current))
        end

        # Progress
        if verbose && iter % 200 == 0
            accept_rate = n_accept / n_total
            println("Iter " * string(iter) * "/" * string(n_burnin + n_samples) *
                   ": accept_rate=" * string(round(100*accept_rate, digits=1)) * "%" *
                   ", σ_mean=" * string(round(mean(sigma_current), digits=4)))
        end
    end

    # Posterior mean estimate
    X_mean = [zeros(n, d) for t in 1:T]
    for sample in X_samples
        for t in 1:T
            X_mean[t] .+= sample[t]
        end
    end
    for t in 1:T
        X_mean[t] ./= length(X_samples)
    end

    sigma_mean = mean(hcat(sigma_samples...), dims=2)[:]

    # Compute P error for posterior mean
    P_errors = [norm(X_mean[t] * X_mean[t]' - P_list[t]) / norm(P_list[t]) for t in 1:T]
    mean_P_error = mean(P_errors)

    verbose && println("Final accept rate: " * string(round(100*n_accept/n_total, digits=1)) * "%")
    verbose && println("Posterior σ mean: " * string(round(mean(sigma_mean), digits=4)))

    return (X_list=X_mean, sigma=sigma_mean, P_error=mean_P_error,
            X_samples=X_samples, sigma_samples=sigma_samples)
end

#=============================================================================
Curvature Computation
=============================================================================#

function compute_curvature(X_list::Vector{Matrix{Float64}})
    T = length(X_list)
    n = size(X_list[1], 1)

    # Compute velocities
    V = [X_list[t+1] - X_list[t] for t in 1:(T-1)]

    # Compute angles between consecutive velocities for each node
    angles = Float64[]
    for i in 1:n
        for t in 1:(T-2)
            v1 = V[t][i, :]
            v2 = V[t+1][i, :]

            n1, n2 = norm(v1), norm(v2)
            if n1 > 1e-10 && n2 > 1e-10
                cos_angle = clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                push!(angles, acos(cos_angle))
            end
        end
    end

    return rad2deg(mean(angles))
end

function compute_node_curvature(X_list::Vector{Matrix{Float64}}, node_idx::Int)
    T = length(X_list)

    # Compute velocities for this node
    V = [X_list[t+1][node_idx, :] - X_list[t][node_idx, :] for t in 1:(T-1)]

    # Compute angles between consecutive velocities
    angles = Float64[]
    for t in 1:(T-2)
        v1 = V[t]
        v2 = V[t+1]

        n1, n2 = norm(v1), norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            cos_angle = clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            push!(angles, acos(cos_angle))
        end
    end

    return isempty(angles) ? 0.0 : rad2deg(mean(angles))
end

#=============================================================================
GAUGE-INVARIANT Derivative-Based Evaluation Metrics
=============================================================================#
#
# Key insight: P = XX' is invariant to orthogonal transformations X → XQ
# Our metrics must also be gauge-invariant!
#
# Gauge-invariant quantities:
# - Speed |v_i| (magnitude, not direction)
# - Velocity Gram matrix VV' (inner products between node velocities)
# - P-derivative: dP/dt = VX' + XV'
# - Relative angles between node velocities: cos(v_i, v_j)
#

"""
Compute velocity from trajectory.
Returns vector of (T-1) velocity matrices, each n×d.
"""
function compute_velocities(X_list::Vector{Matrix{Float64}})
    return [X_list[t+1] - X_list[t] for t in 1:(length(X_list)-1)]
end

"""
Compute P-derivative: dP/dt = VX' + XV' (GAUGE-INVARIANT)
Returns vector of (T-1) matrices, each n×n.
"""
function compute_P_derivatives(X_list::Vector{Matrix{Float64}})
    V = compute_velocities(X_list)
    T = length(V)
    dP = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        # Use average X between t and t+1 for better approximation
        X_mid = 0.5 * (X_list[t] + X_list[t+1])
        dP[t] = V[t] * X_mid' + X_mid * V[t]'
    end
    return dP
end

"""
Speed comparison (GAUGE-INVARIANT): Compare |v| magnitudes.
Returns (mean_log_ratio, correlation).
- mean_log_ratio: mean |log(|v_est|/|v_true|)|, 0 = perfect
- correlation: Pearson correlation of speeds
"""
function speed_comparison(X_true::Vector{Matrix{Float64}},
                          X_est::Vector{Matrix{Float64}})
    V_true = compute_velocities(X_true)
    V_est = compute_velocities(X_est)

    T = length(V_true)
    n = size(V_true[1], 1)

    speeds_true = Float64[]
    speeds_est = Float64[]

    for t in 1:T
        for i in 1:n
            s_t = norm(V_true[t][i, :])
            s_e = norm(V_est[t][i, :])
            push!(speeds_true, s_t)
            push!(speeds_est, s_e)
        end
    end

    # Log ratio (avoiding zeros)
    log_ratios = Float64[]
    for (s_t, s_e) in zip(speeds_true, speeds_est)
        if s_t > 1e-10 && s_e > 1e-10
            push!(log_ratios, abs(log(s_e / s_t)))
        end
    end
    mean_log_ratio = isempty(log_ratios) ? Inf : mean(log_ratios)

    # Correlation
    μ_t, μ_e = mean(speeds_true), mean(speeds_est)
    σ_t = sqrt(mean((speeds_true .- μ_t).^2))
    σ_e = sqrt(mean((speeds_est .- μ_e).^2))
    if σ_t > 1e-10 && σ_e > 1e-10
        corr = mean((speeds_true .- μ_t) .* (speeds_est .- μ_e)) / (σ_t * σ_e)
    else
        corr = 0.0
    end

    return (mean_log_ratio=mean_log_ratio, correlation=corr)
end

"""
P-derivative error (GAUGE-INVARIANT): Compare dP/dt.
Returns normalized Frobenius error: ||dP_est - dP_true||_F / ||dP_true||_F
"""
function P_derivative_error(X_true::Vector{Matrix{Float64}},
                            X_est::Vector{Matrix{Float64}})
    dP_true = compute_P_derivatives(X_true)
    dP_est = compute_P_derivatives(X_est)

    error_sum = 0.0
    norm_sum = 0.0

    for t in 1:length(dP_true)
        error_sum += norm(dP_est[t] - dP_true[t])^2
        norm_sum += norm(dP_true[t])^2
    end

    return sqrt(error_sum) / sqrt(max(norm_sum, 1e-10))
end

"""
Velocity Gram matrix error (GAUGE-INVARIANT): Compare VV'.
VV' captures the inner products between all node velocities.
Returns normalized error: ||VV'_est - VV'_true||_F / ||VV'_true||_F
"""
function velocity_gram_error(X_true::Vector{Matrix{Float64}},
                             X_est::Vector{Matrix{Float64}})
    V_true = compute_velocities(X_true)
    V_est = compute_velocities(X_est)

    error_sum = 0.0
    norm_sum = 0.0

    for t in 1:length(V_true)
        G_true = V_true[t] * V_true[t]'
        G_est = V_est[t] * V_est[t]'
        error_sum += norm(G_est - G_true)^2
        norm_sum += norm(G_true)^2
    end

    return sqrt(error_sum) / sqrt(max(norm_sum, 1e-10))
end

"""
Relative velocity structure (GAUGE-INVARIANT): Compare pairwise velocity angles.
For each pair (i,j), compute cos(v_i, v_j) and compare between true and estimated.
"""
function relative_velocity_structure_error(X_true::Vector{Matrix{Float64}},
                                            X_est::Vector{Matrix{Float64}})
    V_true = compute_velocities(X_true)
    V_est = compute_velocities(X_est)

    T = length(V_true)
    n = size(V_true[1], 1)

    errors = Float64[]

    for t in 1:T
        for i in 1:n
            for j in (i+1):n
                v_i_t = V_true[t][i, :]
                v_j_t = V_true[t][j, :]
                v_i_e = V_est[t][i, :]
                v_j_e = V_est[t][j, :]

                n_it, n_jt = norm(v_i_t), norm(v_j_t)
                n_ie, n_je = norm(v_i_e), norm(v_j_e)

                if n_it > 1e-10 && n_jt > 1e-10 && n_ie > 1e-10 && n_je > 1e-10
                    cos_true = dot(v_i_t, v_j_t) / (n_it * n_jt)
                    cos_est = dot(v_i_e, v_j_e) / (n_ie * n_je)
                    push!(errors, (cos_est - cos_true)^2)
                end
            end
        end
    end

    return isempty(errors) ? 0.0 : sqrt(mean(errors))
end

"""
Compute all GAUGE-INVARIANT derivative metrics.
"""
function derivative_metrics(X_true::Vector{Matrix{Float64}},
                           X_est::Vector{Matrix{Float64}})
    speed = speed_comparison(X_true, X_est)
    return (
        speed_log_ratio = speed.mean_log_ratio,
        speed_corr = speed.correlation,
        P_deriv_error = P_derivative_error(X_true, X_est),
        vel_gram_error = velocity_gram_error(X_true, X_est),
        rel_vel_error = relative_velocity_structure_error(X_true, X_est)
    )
end

#=============================================================================
DUASE Embedding (proper implementation from variational_embedding.jl)
=============================================================================#

"""
Proper DUASE: Unfold all P matrices, find common basis, project per-timestep.
"""
function duase_embed(P_list::Vector{Matrix{Float64}}, d::Int)
    T = length(P_list)

    # Unfold: stack all P matrices horizontally
    Unfolded = hcat(P_list...)

    # SVD to get common basis G
    U, S, V = svd(Unfolded)
    G = U[:, 1:d]

    # Per-timestep projection
    X_duase = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        Qt = G' * P_list[t] * G
        Qt_sym = (Qt + Qt') / 2
        eig = eigen(Symmetric(Qt_sym))
        sqrt_Q = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0))) * eig.vectors'
        X_duase[t] = G * sqrt_Q
    end

    return X_duase
end

#=============================================================================
Data Generation: Type-specific dynamics with Holling Type II (from example4_type_kernel.jl)
=============================================================================#
#
# Heterogeneous dynamics: ẋ = N(P)·x where N_ij = κ_{type(i), type(j)}(P_ij)
#
# Key feature: Predator-Prey interaction uses Holling Type II (saturating)
#   κ_PY(p) = α·p / (1 + β·p)
#

const N_PRED = 12    # Predators
const N_PREY = 15    # Prey
const N_RES = 10     # Resources
const N_TOTAL = N_PRED + N_PREY + N_RES  # 37 total
const D_EMBED = 2
const T_TOTAL = 25
const SEED = 42

# GB-DASE uses RAW BINARY adjacency matrices, not averaged P_hat!
# The random walk prior is designed to smooth NOISY binary observations.

# Type indices
const IDX_PRED = 1:N_PRED
const IDX_PREY = (N_PRED+1):(N_PRED+N_PREY)
const IDX_RES = (N_PRED+N_PREY+1):N_TOTAL

# Type labels
const TYPE_P = 1  # Predator
const TYPE_Y = 2  # Prey
const TYPE_R = 3  # Resource

# Type assignment for each node
const NODE_TYPES = vcat(
    fill(TYPE_P, N_PRED),
    fill(TYPE_Y, N_PREY),
    fill(TYPE_R, N_RES)
)

# Self-rates (KNOWN PHYSICS)
const KNOWN_SELF_RATES = Dict(
    TYPE_P => -0.002,   # Predator: small decay
    TYPE_Y => -0.001,   # Prey: minimal decay
    TYPE_R =>  0.000    # Resource: stable
)

# Holling Type II parameters
const HOLLING_ALPHA = 0.025
const HOLLING_BETA = 2.0

"""Type-specific message kernel κ_ab(p)."""
function κ_true(type_i::Int, type_j::Int, p::Real)
    if type_i == TYPE_P && type_j == TYPE_P
        return -0.004  # Predator-Predator: mild repulsion
    elseif type_i == TYPE_P && type_j == TYPE_Y
        return HOLLING_ALPHA * p / (1 + HOLLING_BETA * p)  # HOLLING TYPE II
    elseif type_i == TYPE_P && type_j == TYPE_R
        return 0.0
    elseif type_i == TYPE_Y && type_j == TYPE_P
        return -0.02 * p  # Prey flees predator
    elseif type_i == TYPE_Y && type_j == TYPE_Y
        return 0.003  # Prey herding
    elseif type_i == TYPE_Y && type_j == TYPE_R
        return 0.012 * p  # Prey attracted to resource
    elseif type_i == TYPE_R && type_j == TYPE_P
        return 0.0
    elseif type_i == TYPE_R && type_j == TYPE_Y
        return -0.006 * p  # Resource depleted by prey
    elseif type_i == TYPE_R && type_j == TYPE_R
        return 0.005  # Resource cohesion
    else
        return 0.0
    end
end

"""Compute N matrix from true type-specific kernels."""
function compute_N_true(P::Matrix{Float64})
    n = size(P, 1)
    N = zeros(n, n)
    for i in 1:n
        ti = NODE_TYPES[i]
        N[i,i] = KNOWN_SELF_RATES[ti]
        for j in 1:n
            if j != i
                tj = NODE_TYPES[j]
                κ_ij = κ_true(ti, tj, P[i,j])
                N[i,j] = κ_ij
                N[i,i] -= κ_ij  # Message-passing form
            end
        end
    end
    return N
end

"""True dynamics: ẋ = N(P)·x with type-specific kernels."""
function true_dynamics!(dX::Matrix{Float64}, X::Matrix{Float64}, p, t)
    P = X * X'
    N = compute_N_true(P)
    dX .= N * X
end

function true_dynamics_vec!(du::Vector{Float64}, u::Vector{Float64}, p, t)
    X = collect(transpose(reshape(u, D_EMBED, N_TOTAL)))
    dX = similar(X)
    true_dynamics!(dX, X, p, t)
    du .= vec(transpose(dX))
end

"""Generate initial positions with type clusters in B^d_+."""
function generate_initial_X(seed::Int)
    rng = Xoshiro(seed)
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
    return X0
end

#=============================================================================
Observation Model: RAW BINARY adjacency matrices (NOT averaged!)
=============================================================================#
#
# CRITICAL: GB-DASE is designed for raw binary observations.
# The paper uses single Bernoulli samples per edge per timestep.
# Averaging K samples PRE-SMOOTHS the data, defeating the purpose of the prior!
#

"""
Generate a SINGLE raw binary adjacency matrix from P (per timestep).
This is what GB-DASE expects - noisy binary observations, not averaged P_hat.
INCLUDES DIAGONAL: P[i,i] = ||x_i||² carries norm information!
"""
function generate_binary_adjacency(P::Matrix{Float64}, rng::AbstractRNG)
    n = size(P, 1)
    A = zeros(n, n)

    # Sample ALL entries including diagonal
    # Diagonal P[i,i] = ||x_i||² tells us about node norms
    for i in 1:n
        # Diagonal (self-loop)
        p_ii = clamp(P[i, i], 0.0, 1.0)
        if rand(rng) < p_ii
            A[i, i] = 1.0
        end
        # Off-diagonal (edges)
        for j in 1:(i-1)
            p_ij = clamp(P[i, j], 0.0, 1.0)
            if rand(rng) < p_ij
                A[i, j] = 1.0
                A[j, i] = 1.0
            end
        end
    end

    return A
end

"""
Generate raw binary adjacency matrices for all timesteps.
"""
function generate_binary_adjacencies(X_true::Vector{Matrix{Float64}}, rng::AbstractRNG)
    T = length(X_true)
    A_obs = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        P = clamp.(X_true[t] * X_true[t]', 0.0, 1.0)
        A_obs[t] = generate_binary_adjacency(P, rng)
    end

    return A_obs
end

"""
For comparison: generate averaged P_hat (what we were doing before - WRONG for GB-DASE).
"""
function generate_averaged_adjacencies(X_true::Vector{Matrix{Float64}}, K::Int, rng::AbstractRNG)
    T = length(X_true)
    n = size(X_true[1], 1)
    A_obs = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        P = clamp.(X_true[t] * X_true[t]', 0.0, 1.0)
        A_sum = zeros(n, n)

        for _ in 1:K
            A_sum .+= generate_binary_adjacency(P, rng)
        end

        A_obs[t] = A_sum / K
    end

    return A_obs
end

#=============================================================================
Main Experiment: Focus on SVD vs SVD+Procrustes initialization
=============================================================================#

function main()
    println("=" ^ 70)
    println("GB-DASE: CORRECT OBSERVATION MODEL (RAW BINARY A)")
    println("=" ^ 70)
    println("\nKey insight from reference implementation (joshloyal/DynamicRDPG):")
    println("  - GB-DASE uses RAW BINARY adjacency matrices")
    println("  - NOT averaged P_hat!")
    println("  - The random walk prior smooths the noisy binary observations")

    # Generate data (type-specific dynamics with Holling Type II)
    X0 = generate_initial_X(SEED)
    u0 = vec(collect(transpose(X0)))
    tspan = (0.0, Float64(T_TOTAL - 1))

    prob = ODEProblem(true_dynamics_vec!, u0, tspan)
    sol = solve(prob, Tsit5(); saveat=1.0)

    X_true = [collect(transpose(reshape(sol.u[i], D_EMBED, N_TOTAL))) for i in 1:length(sol.t)]
    P_true = [X_true[t] * X_true[t]' for t in 1:length(X_true)]

    n = N_TOTAL
    d = D_EMBED
    T = length(X_true)

    println("\nn=" * string(n) * " nodes, d=" * string(d) * " dimensions, T=" * string(T) * " timesteps")

    # =========================================================================
    # Generate RAW BINARY adjacency matrices (CORRECT for GB-DASE)
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("GENERATING OBSERVATIONS")
    println("=" ^ 70)

    rng = Xoshiro(123)

    # RAW BINARY - what GB-DASE actually expects
    A_binary = generate_binary_adjacencies(X_true, rng)
    binary_noise = mean([norm(A_binary[t] - P_true[t]) / norm(P_true[t]) for t in 1:T])
    println("\nRaw binary A noise (||A - P_true|| / ||P_true||): " * string(round(100*binary_noise, digits=1)) * "%")

    # Averaged P_hat - what we were doing before (WRONG for GB-DASE)
    rng2 = Xoshiro(456)
    P_hat_50 = generate_averaged_adjacencies(X_true, 50, rng2)
    avg_noise = mean([norm(P_hat_50[t] - P_true[t]) / norm(P_true[t]) for t in 1:T])
    println("Averaged P_hat (K=50) noise: " * string(round(100*avg_noise, digits=1)) * "%")

    # Use RAW BINARY for GB-DASE (correct!)
    A_list = A_binary

    # Alignment helper (global alignment to true for visualization)
    function align_to_true(X_est)
        F = svd(X_est[1]' * X_true[1])
        [X_est[t] * (F.U * F.Vt) for t in 1:T]
    end

    # P-error against TRUE probability matrix (what we're trying to recover)
    function p_error(X_est)
        mean([norm(X_est[t] * X_est[t]' - P_true[t]) / norm(P_true[t]) for t in 1:T])
    end

    # Store all results for plotting
    all_results = Dict{String, NamedTuple}()

    # =========================================================================
    # PART 1: Methods on RAW BINARY A (correct for GB-DASE)
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("PART 1: METHODS ON RAW BINARY ADJACENCY MATRICES")
    println("(This is what GB-DASE was designed for)")
    println("=" ^ 70)

    # 1a. Raw SVD on binary A
    println("\n--- Raw SVD on binary A ---")
    X_svd_raw = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        emb = svd_embedding(A_list[t], d)
        X_svd_raw[t] = emb.L_hat
    end
    X_svd_raw_al = align_to_true(X_svd_raw)
    all_results["SVD raw (A)"] = (X=X_svd_raw_al, P_err=p_error(X_svd_raw_al))
    println("P error: " * string(round(100*all_results["SVD raw (A)"].P_err, digits=1)) * "%")

    # 1b. SVD + Procrustes on binary A
    println("\n--- SVD + Procrustes on binary A ---")
    X_svd_proc = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        emb = svd_embedding(A_list[t], d)
        X_svd_proc[t] = emb.L_hat
    end
    for t in 2:T
        Omega = ortho_procrustes_RM(X_svd_proc[t]', X_svd_proc[t-1]')
        X_svd_proc[t] = X_svd_proc[t] * Omega
    end
    X_svd_proc_al = align_to_true(X_svd_proc)
    all_results["SVD+Proc (A)"] = (X=X_svd_proc_al, P_err=p_error(X_svd_proc_al))
    println("P error: " * string(round(100*all_results["SVD+Proc (A)"].P_err, digits=1)) * "%")

    # 1c. DUASE on binary A
    println("\n--- DUASE on binary A ---")
    X_duase = duase_embed(A_list, d)
    X_duase_al = align_to_true(X_duase)
    all_results["DUASE (A)"] = (X=X_duase_al, P_err=p_error(X_duase_al))
    println("P error: " * string(round(100*all_results["DUASE (A)"].P_err, digits=1)) * "%")

    # 1d. GB-DASE on binary A with FIXED σ (to see if prior helps when not cancelled)
    println("\n--- GB-DASE r=2 on binary A (σ estimated - prior gets cancelled!) ---")
    result_gbd_est = gbdase_embed(A_list, d;
                                   rw_order=2, λ_P=1.0,
                                   use_procrustes_init=true,
                                   estimate_sigma=true, max_iter=50, verbose=false)
    X_gbd_est_al = align_to_true(result_gbd_est.X_list)
    all_results["GB-DASE σ_est (A)"] = (X=X_gbd_est_al, P_err=p_error(X_gbd_est_al),
                                         sigma=result_gbd_est.sigma)
    println("P error: " * string(round(100*all_results["GB-DASE σ_est (A)"].P_err, digits=1)) * "%")
    println("σ range: [" * string(round(minimum(result_gbd_est.sigma), digits=4)) * ", " *
           string(round(maximum(result_gbd_est.sigma), digits=4)) * "]")

    # 1e. GB-DASE on binary A with FIXED small σ (strong prior)
    println("\n--- GB-DASE r=2 on binary A (σ=0.01 FIXED - strong prior) ---")
    result_gbd_fix = gbdase_embed(A_list, d;
                                   rw_order=2, λ_P=1.0,
                                   use_procrustes_init=true,
                                   estimate_sigma=false, sigma_init=0.01,
                                   max_iter=50, verbose=false)
    X_gbd_fix_al = align_to_true(result_gbd_fix.X_list)
    all_results["GB-DASE σ=0.01 (A)"] = (X=X_gbd_fix_al, P_err=p_error(X_gbd_fix_al))
    println("P error: " * string(round(100*all_results["GB-DASE σ=0.01 (A)"].P_err, digits=1)) * "%")

    # =========================================================================
    # PART 2: Methods on AVERAGED P_hat (for comparison - WRONG for GB-DASE)
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("PART 2: METHODS ON AVERAGED P_hat (K=50)")
    println("(Pre-smoothed data - NOT what GB-DASE was designed for)")
    println("=" ^ 70)

    # 2a. DUASE on averaged P_hat
    println("\n--- DUASE on averaged P_hat ---")
    X_duase_avg = duase_embed(P_hat_50, d)
    X_duase_avg_al = align_to_true(X_duase_avg)
    all_results["DUASE (P_hat)"] = (X=X_duase_avg_al, P_err=p_error(X_duase_avg_al))
    println("P error: " * string(round(100*all_results["DUASE (P_hat)"].P_err, digits=1)) * "%")

    # 2b. GB-DASE on averaged P_hat (WRONG - for comparison)
    println("\n--- GB-DASE r=2 on averaged P_hat (WRONG for GB-DASE) ---")
    result_gbd_avg = gbdase_embed(P_hat_50, d;
                                   rw_order=2, λ_P=1.0,
                                   use_procrustes_init=true,
                                   estimate_sigma=true, max_iter=50, verbose=false)
    X_gbd_avg_al = align_to_true(result_gbd_avg.X_list)
    all_results["GB-DASE (P_hat)"] = (X=X_gbd_avg_al, P_err=p_error(X_gbd_avg_al),
                                       sigma=result_gbd_avg.sigma)
    println("P error: " * string(round(100*all_results["GB-DASE (P_hat)"].P_err, digits=1)) * "%")
    println("σ range: [" * string(round(minimum(result_gbd_avg.sigma), digits=4)) * ", " *
           string(round(maximum(result_gbd_avg.sigma), digits=4)) * "]")

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    println("\nPART 1: Raw Binary A (correct for GB-DASE)")
    println("-" ^ 50)
    println("Method                          P-error")
    for name in ["SVD raw (A)", "SVD+Proc (A)", "DUASE (A)", "GB-DASE (A)"]
        if haskey(all_results, name)
            println(rpad(name, 32) * string(round(100*all_results[name].P_err, digits=1)) * "%")
        end
    end

    println("\nPART 2: Averaged P_hat K=50 (wrong for GB-DASE)")
    println("-" ^ 50)
    for name in ["DUASE (P_hat)", "GB-DASE (P_hat)"]
        if haskey(all_results, name)
            println(rpad(name, 32) * string(round(100*all_results[name].P_err, digits=1)) * "%")
        end
    end

    println("\n" * "=" ^ 70)
    println("KEY FINDING: GB-DASE on raw binary A should outperform DUASE")
    println("because the RW prior smooths noisy observations.")
    println("If it doesn't, there's still an implementation issue.")
    println("=" ^ 70)

    # =========================================================================
    # VISUALIZATION: Focus on trajectories
    # =========================================================================
    println("\n" * "=" ^ 70)
    println("GENERATING TRAJECTORY PLOTS")
    println("=" ^ 70)

    fig = Figure(size=(2000, 1000))

    colors_plot = [:red, :blue, :green]

    # Helper to plot trajectories
    function plot_trajectories!(ax, X_list, title_str)
        for i in 1:n
            type_i = NODE_TYPES[i]
            traj = hcat([X_list[t][i, :] for t in 1:T]...)'
            lines!(ax, traj[:, 1], traj[:, 2], color=(colors_plot[type_i], 0.6), linewidth=1.5)
            scatter!(ax, [traj[1, 1]], [traj[1, 2]], color=colors_plot[type_i], markersize=6)
            scatter!(ax, [traj[end, 1]], [traj[end, 2]], color=colors_plot[type_i], marker=:star5, markersize=8)
        end
    end

    # Row 1: True, then methods on raw binary A
    ax1 = Axis(fig[1, 1], title="True Trajectories", xlabel="x₁", ylabel="x₂", aspect=1)
    ax2 = Axis(fig[1, 2], title="SVD+Proc (A)\nP-err=" * string(round(100*all_results["SVD+Proc (A)"].P_err, digits=1)) * "%",
               xlabel="x₁", ylabel="x₂", aspect=1)
    ax3 = Axis(fig[1, 3], title="DUASE (A)\nP-err=" * string(round(100*all_results["DUASE (A)"].P_err, digits=1)) * "%",
               xlabel="x₁", ylabel="x₂", aspect=1)
    ax4 = Axis(fig[1, 4], title="GB-DASE r=2 (A) ✓\nP-err=" * string(round(100*all_results["GB-DASE (A)"].P_err, digits=1)) * "%",
               xlabel="x₁", ylabel="x₂", aspect=1)

    plot_trajectories!(ax1, X_true, "True")
    plot_trajectories!(ax2, all_results["SVD+Proc (A)"].X, "SVD+Proc")
    plot_trajectories!(ax3, all_results["DUASE (A)"].X, "DUASE")
    plot_trajectories!(ax4, all_results["GB-DASE (A)"].X, "GB-DASE")

    # Row 2: Comparison - same methods on averaged P_hat
    ax5 = Axis(fig[2, 1], title="(Reference: True)", xlabel="x₁", ylabel="x₂", aspect=1)
    ax6 = Axis(fig[2, 2], title="DUASE (P_hat K=50)\nP-err=" * string(round(100*all_results["DUASE (P_hat)"].P_err, digits=1)) * "%",
               xlabel="x₁", ylabel="x₂", aspect=1)
    ax7 = Axis(fig[2, 3], title="GB-DASE (P_hat) ✗ WRONG\nP-err=" * string(round(100*all_results["GB-DASE (P_hat)"].P_err, digits=1)) * "%",
               xlabel="x₁", ylabel="x₂", aspect=1)

    plot_trajectories!(ax5, X_true, "True")
    plot_trajectories!(ax6, all_results["DUASE (P_hat)"].X, "DUASE P_hat")
    plot_trajectories!(ax7, all_results["GB-DASE (P_hat)"].X, "GB-DASE P_hat")

    # Row 2 col 4: P-error bar chart comparing both approaches
    ax8 = Axis(fig[2, 4], title="P-error: Raw A vs Averaged P_hat", xlabel="Method", ylabel="P-error (%)")
    methods = ["SVD+Proc\n(A)", "DUASE\n(A)", "GB-DASE\n(A)", "DUASE\n(P_hat)", "GB-DASE\n(P_hat)"]
    p_errors = [100*all_results["SVD+Proc (A)"].P_err,
                100*all_results["DUASE (A)"].P_err,
                100*all_results["GB-DASE (A)"].P_err,
                100*all_results["DUASE (P_hat)"].P_err,
                100*all_results["GB-DASE (P_hat)"].P_err]
    colors_bar = [:orange, :gray, :blue, :lightgray, :lightblue]
    barplot!(ax8, 1:5, p_errors, color=colors_bar)
    ax8.xticks = (1:5, methods)

    # Legend
    Legend(fig[3, 1:2], [LineElement(color=c, linewidth=3) for c in colors_plot],
           ["Predator (P)", "Prey (Y)", "Resource (R)"], "Node Type",
           orientation=:horizontal, tellwidth=false, tellheight=true)

    # Notes text
    notes = """
    KEY INSIGHT: GB-DASE is designed for RAW BINARY adjacency matrices!

    Row 1: Methods on raw binary A(t) ~ Bernoulli(P(t))
           - High noise (~38%), GB-DASE prior should help smooth

    Row 2: Methods on averaged P̂(t) = (1/K)Σₖ A(t)
           - Pre-smoothed (~14% noise), prior has nothing to do

    Reference: joshloyal/DynamicRDPG uses single Bernoulli per edge.
    """
    Label(fig[3, 3:4], notes, fontsize=12, halign=:left, valign=:top)

    save("results/gbdase_binary_vs_averaged.png", fig)
    println("\nSaved: results/gbdase_binary_vs_averaged.png")

    return all_results
end

# Run
all_results = main()
