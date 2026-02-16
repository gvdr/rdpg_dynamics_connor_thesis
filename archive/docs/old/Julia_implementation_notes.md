# Julia Implementation Notes for RDPG Dynamics

This document captures best practices for RDPG estimation learned from graspologic,
the literature, and our experiments.

## The Proper RDPG Pipeline

The correct pipeline for learning dynamics from temporal networks:

```
True X(t) ∈ B^d_+ → A(t) ~ Bernoulli(XX') → [K samples, average] → SVD → Procrustes chain → B^d_+ alignment → X̂(t) → Train Neural ODE
```

### Key Components

1. **Generate in B^d_+**: True positions must satisfy x ≥ 0, ||x|| ≤ 1
2. **Sample adjacency**: Generate K samples, average to reduce noise
3. **SVD embedding**: Recovers positions up to arbitrary rotation
4. **Temporal Procrustes**: Align each timestep to previous (smooth trajectory)
5. **B^d_+ alignment**: Find global rotation Q to map back into positive orthant
6. **Projection**: Clamp any remaining negatives, scale if ||x|| > 1
7. **Train on estimated**: Neural ODE learns from X̂(t), never sees true X(t)

### Using RDPGDynamics Library

```julia
using RDPGDynamics

# Step 1-2: Sample adjacencies from true positions
A_series = [sample_adjacency_repeated(X_true[t], K) for t in 1:T]

# Step 3-6: Embed with full B^d_+ pipeline
L_series = embed_temporal_network_Bd_plus(A_series, d)

# Step 7: Convert to Float32 for training
X_train = [Float32.(L) for L in L_series]
```

### Experimental Results

With proper B^d_+ alignment, both Pure NN and UDE **exceed baseline** RDPG estimation:

| Model | D corr | vs Baseline |
|-------|--------|-------------|
| Baseline (RDPG) | 0.912 | 100% |
| Pure NN | 0.941 | **103%** |
| UDE | 0.966 | **106%** |

**Key insight**: Without B^d_+ alignment, SVD outputs can be in negative quadrants,
making learning impossible. With proper alignment, the dynamics stay in B^d_+ naturally.

---

## ⚠️ CRITICAL UPDATE: Gauge-Equivariant Learning (January 2026)

**The B^d_+ projection approach described above has significant drawbacks.** Based on Example 1 experiments, we now recommend a simpler, more principled approach.

### The Problem with B^d_+ Projection

B^d_+ projection (clamping negatives, scaling to unit ball) **distorts geometry**:
- Changes pairwise distances
- Breaks temporal consistency
- Introduces artifacts at boundaries

### The Solution: Gauge-Equivariant Dynamics

**Key insight**: Dynamics of the form **Ẋ = N(P)X** where P = XX' are **gauge-equivariant**. The learned parameters are scalars that work in ANY coordinate system.

This means:
1. **Learn in DUASE space**: Train on whatever coordinates DUASE naturally produces (no projection!)
2. **Apply to TRUE initial conditions**: Use learned parameters with true X(0)
3. **Direct comparison**: Recovered trajectories are directly comparable to ground truth - NO Procrustes needed

### Why This Works

DUASE provides embeddings that are:
- **Temporally consistent** via shared basis G
- Related to true positions by unknown orthogonal transformation Q: X̂ = XQ

Since N(P)X dynamics only depend on P = XX' (which is rotation-invariant), the learned parameters (β₀, β₁, etc.) are the same regardless of which coordinate system we use.

**Example**: Message-passing dynamics
```
Ẋᵢ = β₀·Xᵢ + β₁·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ)
```
The scalars β₀, β₁ are gauge-invariant. Learn them in DUASE space, apply them to true X(0).

### Updated Pipeline

```
A(t) ~ Bernoulli(XX') → [K samples, average] → DUASE embedding → X̂(t) → Train N(P)X dynamics
                                                                              ↓
                                                          Apply learned params to TRUE X(0)
                                                                              ↓
                                                              Compare P_recovered vs P_true
```

### P(t) is the True Test

**Do NOT compare X positions** (gauge-dependent). Compare **P = XX'** (rotation-invariant):
- P_error = mean(|P_pred - P_true|)
- This is the honest test of dynamics recovery

### Experimental Validation (Example 1)

| Model | Parameters | Validation Error (vs True) |
|-------|------------|---------------------------|
| MsgPass | 2 | 0.48 |
| Poly P² | 3 | 0.56 |
| Pure NN | 7832 | 8.24 |

**Parsimonious physics-informed models (2-3 params) vastly outperform black-box NN (7832 params)** on extrapolation. The N(P)X structure provides massive inductive bias.

### Misspecification Robustness (Surprising Finding)

In Example 1, the true dynamics were message-passing form:
```
Ẋᵢ = α₀·Xᵢ + α₁·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ)  [equivalent to N = α₀I + α₁(P - D)]
```

But the Polynomial architecture `Ẋ = (β₀I + β₁P + β₂P²)X` (which does NOT have the degree matrix subtraction) still captured the dynamics well.

**Implication**: N(P)X architectures are flexible enough to approximate related dynamics even when not exactly matched. The polynomial form can absorb some of the "missing" structure through its coefficients. This is encouraging for real applications where the true dynamics form is unknown.

### When to Use B^d_+ Projection

B^d_+ projection may still be useful when:
- You need strict probability constraints (P_ij ∈ [0,1])
- You're using dynamics that are NOT gauge-equivariant
- Visualization requires a canonical coordinate system

But for learning N(P)X dynamics, **skip the projection**.

### DUASE Estimates vs Dynamics Predictions

**Important distinction**:
- **DUASE**: Estimates X from observed adjacency matrices (can't extrapolate)
- **Dynamics models**: Predict future X from initial conditions (can extrapolate)

When evaluating extrapolation, only dynamics models are making true predictions. DUASE "extrapolation" would require seeing future data.

---

## Legacy Warning

**NOTE**: Older example scripts (example1, example2, example4) may train directly
on true latent positions X(t), bypassing the RDPG observation model. This is
invalid for real applications. Always use the proper pipeline above.

---

## RDPG Estimation Best Practices

### 0. Self-Loops (CRITICAL!)

**Why**: The diagonal of P = XX' contains P_ii = ||X_i||², which is essential information
for SVD recovery. If you zero out the diagonal when sampling adjacency matrices, the
averaged A will NOT converge to P, and SVD will fail to recover X.

**The problem**: Standard network analysis often excludes self-loops (A_ii = 0). But for
RDPG estimation, this loses critical norm information.

**Empirical result** (from our experiments):
- Without self-loops: embedding error stays ~0.9 even with K=1000 samples
- With self-loops: embedding error → 0 as K → ∞

**Solution**: When sampling A from P = XX', sample ALL entries including diagonal:
```julia
function sample_adjacency(X; symmetric=true, self_loops=true)
    P = X * X'
    P = clamp.(P, 0.0, 1.0)
    A = Float64.(rand(n, n) .< P)  # Bernoulli for ALL i,j including i=i

    if symmetric
        # Mirror upper to lower, keep diagonal as sampled
        for i in 1:n, j in 1:i-1
            A[i, j] = A[j, i]
        end
    end
    return A  # Diagonal is NOT zeroed!
end
```

**Impact on results**: With this fix, UDE improvement went from 7.9% to 39.7%.

### 1. Repeated Sampling

**Why**: A single binary adjacency matrix is extremely noisy. Each edge is a Bernoulli
sample from P(i,j) = X_i · X_j, so we get high variance.

**Solution**: Generate K independent samples at each time point and average:
```julia
A_avg(t) = (1/K) * sum(A_k(t) for k in 1:K)
```

**Recommended**: K = 10 samples per time point

### 2. Temporal Windowing

**Why**: Even with repeated sampling, adjacent time points have similar structure.
Averaging over a window improves the eigenvalue gap.

**Solution**: Sliding window averaging before SVD:
```julia
A_smooth(t) = mean(A_avg(s) for s in (t-W/2):(t+W/2))
```

**Recommended**: window W = 3 (balance smoothness vs. temporal resolution)

### 3. Folded/Unfolded Embedding

From graspologic, two main approaches for multiple graphs:

#### Omnibus Embedding
Block matrix where M_ij = (A_i + A_j)/2:
```
    [ A_1      (A_1+A_2)/2  (A_1+A_3)/2  ... ]
M = [ (A_1+A_2)/2  A_2     (A_2+A_3)/2  ... ]
    [ ...                                    ]
```
Embed M with SVD, then reshape. Guarantees common latent space.

#### Folded Embedding (Horizontal Stacking)
Simpler approach - stack adjacency matrices horizontally:
```
A_folded = [A_1 | A_2 | A_3 | ... | A_T]  # n × (n*T) matrix
```
Then SVD on A_folded. Faster but may not preserve temporal structure as well.

**Our choice**: Folded embedding with windowed averaging
- Stack K=10 samples × W=3 windows = 30 matrices per effective time point
- Provides good noise reduction while remaining computationally tractable

### 4. Procrustes Alignment

SVD embedding is only identified up to orthogonal rotation. For temporal networks:

1. **Temporal chain alignment**: Align X̂(t) to X̂(t-1) using Procrustes
2. **Global alignment to B^d_+**: Find single Q mapping all embeddings into valid space

```julia
function ortho_procrustes_RM(A, B)
    U, _, V = svd(A * B')
    return V * U'
end
```

### 5. B^d_+ Constraint Handling

For valid RDPG probabilities (P(i,j) ∈ [0,1]), latent positions should be in B^d_+
(non-negative unit ball): `x ≥ 0` and `||x|| ≤ 1`.

#### The Problem

SVD identifies latent positions X only up to orthogonal transformation Q:
- If true X is in B^d_+, the SVD output X̂ = XQ may not be
- We need to find Q⁻¹ that maps X̂ back into B^d_+

#### Sign Flips vs Rotations

An orthogonal matrix Q can be decomposed as Q = S · R where:
- **S** is a sign/reflection matrix (diagonal with ±1, det = ±1)
- **R** is a proper rotation (det = +1)

Both preserve inner products (important for RDPG), but they're different:
- Sign flips can instantly fix entire columns of negative entries
- Rotations provide fine-grained adjustment within the positive orthant

#### Algorithm: `find_global_orthogonal_to_Bd_plus`

For a temporal series of embeddings, find ONE common transformation Q:

```
1. Stack all T matrices: L_stacked = vcat(L(1), ..., L(T))

2. Sign flips (S):
   For each column j:
     If more than half the entries are negative, flip sign

3. Rotation optimization (R):
   - d=2: Grid search over angle θ (1 parameter)
   - d=3: Joint grid search over 3 angles + refinement
   - d=4: Joint grid search over 6 angles + refinement
   - d>4: Iterative Givens rotations with multiple passes until convergence

4. Return Q = S · R
```

**Key**: For d≤4, we do **joint optimization** over all d(d-1)/2 rotation parameters.
This finds rotations that mix all dimensions optimally.

**Violation score**: `sum(|negative entries|) + sum(max(0, ||x|| - 1))`

#### Handling Remaining Violations

After applying Q, some points may still violate B^d_+ constraints. Two options:

| Method | Description | Preserves |
|--------|-------------|-----------|
| `:project` | Per-point: clamp negatives to 0, scale if ‖x‖ > 1 | Nothing (distorts geometry) |
| `:rescale` | Global: clamp negatives, scale ALL points by same factor s | Relative distances |

**Recommendation**:
- Use `:rescale` if you want to preserve relative geometry (important for dynamics)
- Use `:project` if you need strict B^d_+ membership and can tolerate distortion

#### Usage

```julia
# Full pipeline for temporal series
result = align_series_to_Bd_plus(L_series; method=:rescale)

L_aligned = result.L_aligned  # All matrices now in B^d_+
Q = result.Q                   # The orthogonal transformation used
scale = result.scale           # Global scale factor (if method=:rescale)

# Check statistics
println("Negatives: ", result.stats[:neg_before], " → ", result.stats[:neg_after])
println("Outside ball: ", result.stats[:outside_ball_before], " → ", result.stats[:outside_ball_after])
```

#### Example Results

Test with n=10 nodes, d=2 dimensions, T=5 time points:
```
Sign matrix S: diag([-1, 1])  # One column flipped
Rotation R: θ ≈ 34.6°         # Fine-tuned angle
det(Q) = -1                   # Overall reflection

Violations:
  Before Q:        21.9
  After Q:         17.5
  After projection: 0.0

All points in B^d_+: true
```

---

## Implementation Status

### Implemented in `src/embedding.jl`:
- [x] `svd_embedding` - basic truncated SVD
- [x] `ortho_procrustes_RM` - Procrustes rotation
- [x] `project_to_Bd_plus` - projection onto B^d_+
- [x] `embed_temporal_network_Bd_plus` - temporal alignment
- [x] `embed_temporal_network_smoothed` - sliding window
- [x] `sample_adjacency`, `sample_adjacency_repeated` - RDPG observation model
- [x] `duase_embedding`, `embed_temporal_duase` - DUASE method
- [x] `omni_embedding`, `embed_temporal_omni` - Omnibus method
- [x] `find_global_orthogonal_to_Bd_plus` - sign flips + rotation
- [x] `align_series_to_Bd_plus` - full B^d_+ alignment pipeline

### TODO:
- [ ] Dimension selection (elbow method on singular values)

---

## Recommended Parameters

Based on our experiments and graspologic defaults:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Repeated samples (K) | 10 | Good noise reduction, tractable computation |
| Window size (W) | 3 | Smooth trajectories, preserves fast dynamics |
| Embedding dimension (d) | 2-4 | Application dependent, use elbow method |
| SVD algorithm | Arpack truncated | Fast for large sparse matrices |

---

## Key Insight: Rotation Identifiability

**RDPG embeds are invariant under orthonormal rotation**: If X is the true latent
positions, then QX for any orthogonal Q gives the same edge probabilities.

### What This Means Precisely

The RDPG rotational invariance is about **rotation around the origin (0,0)**:
- If all nodes rotate together around the origin by angle θ, P = XX' is unchanged
- This is absorbed by Procrustes alignment and is fundamentally unidentifiable

### What IS Learnable: Circulation Around Centroid

**Circulation dynamics** (nodes rotating around the swarm centroid) **ARE learnable**:

```
RDPG invariance:  X → QX           (rotation around origin)
Circulation:       X → R(X - c) + c  (rotation around centroid c)
```

When c ≠ 0, these are **different transformations**:
- Procrustes alignment finds Q minimizing ||X̂(t+1)Q - X̂(t)||
- Circulation produces real relative motion that Procrustes preserves
- This motion IS the signal we want to learn

### Summary: What's Learnable vs Unlearnable

| Dynamics Type | Around | Learnable? | Why |
|---------------|--------|------------|-----|
| Global rotation | Origin | ❌ No | Absorbed by Procrustes: P = (XQ)(XQ)' = XX' |
| Circulation/swarming | Centroid | ✅ Yes | Changes relative positions over time |
| Radial expansion/contraction | Any point | ✅ Yes | Changes pairwise distances |
| Pairwise attraction/repulsion | N/A | ✅ Yes | Changes distances |
| Uniform translation | N/A | ✅ Yes | Changes P: (X+d)(X+d)' = XX' + cross terms ≠ XX' |

### Corrected Understanding (2026-01)

Earlier notes incorrectly stated that "Example 2 (rotation) was fundamentally flawed."
This was based on conflating rotation-around-origin (unlearnable) with
circulation-around-centroid (learnable). The radial and Kuramoto dynamics with
angular velocity around the centroid ARE valid and learnable dynamics.

---

## References

1. graspologic: https://github.com/microsoft/graspologic
   - OmnibusEmbed: block matrix approach for multiple graphs
   - MASE: two-stage SVD (individual then shared)

2. Athreya et al. (2017/2018) - Statistical inference on RDPGs
3. Sanna Passino et al. (2021) - Link prediction in dynamic networks using RDPGs
4. UASE literature - Unfolded Adjacency Spectral Embedding

---

## Code Patterns

### Generating adjacency matrices from latent positions
```julia
function sample_adjacency(X::Matrix{Float64}; symmetric::Bool=true)
    n = size(X, 1)
    P = X * X'  # Probability matrix
    P = clamp.(P, 0.0, 1.0)  # Ensure valid probabilities

    A = rand(n, n) .< P  # Bernoulli sampling

    if symmetric
        A = Float64.(triu(A, 1))
        A = A + A'  # Make symmetric, zero diagonal
    end

    return A
end
```

### Folded embedding with repeated samples
```julia
function folded_embedding(X_true::Matrix, d::Int; K::Int=10, W::Int=3, T::Int=50)
    n = size(X_true, 1)

    # For each effective time point, collect K samples over W window
    A_folded = Matrix{Float64}(undef, n, n * K * W)

    for t in 1:T
        # ... generate and stack samples
    end

    # SVD on folded matrix
    U, S, V = svd(A_folded)
    X_hat = U[:, 1:d] .* sqrt.(S[1:d])'

    return X_hat
end
```

---

## Neural ODE and UDE Training

### UDE vs Full NN: When Does Prior Knowledge Help?

**Universal Differential Equations (UDE)**: dx/dt = f_known(x) + NN(x)
**Full NN**: dx/dt = NN(x)

UDE should outperform Full NN when:
1. Known structure captures significant portion of dynamics
2. Unknown part is "corrections" or "residuals" - smaller magnitude
3. Training data is limited (prior knowledge acts as regularization)

UDE may underperform when:
1. Known structure is wrong or misleading
2. Dynamics are simple enough that Full NN learns easily
3. Known/unknown decomposition doesn't match true physics

### Zygote Compatibility (Critical!)

Zygote (Julia's AD) cannot differentiate through in-place mutations. This causes
cryptic `_throw_mutation_error` failures during training.

**FORBIDDEN patterns:**
```julia
# These will FAIL during gradient computation:
dx .+= correction           # In-place addition
dx[i] = value              # Array element assignment
array[idx] = something     # Any indexed assignment
push!(array, x)            # Growing arrays
```

**SAFE patterns:**
```julia
# Use these instead:
dx = dx .+ correction       # Reassignment (creates new array)
dx = vcat(dx[1:i-1], [value], dx[i+1:end])  # Rebuild array

# For constant lookups, use tuples:
const LOOKUP = (val1, val2, val3)  # Tuple, not array
result = LOOKUP[idx]               # Safe indexing
```

### Neural Network Architecture

For learning residual dynamics in embedding space:

```julia
nn = Lux.Chain(
    Lux.Dense(input_dim, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, output_dim)
)
```

**Guidelines:**
- `tanh` activation: bounded output, smooth gradients, works well for dynamics
- 2 hidden layers usually sufficient for smooth dynamics
- 32-64 neurons per layer for d=2-4 embedding dimensions
- Match output dimension to state dimension
- Use `Lux.jl` (not Flux) for better Zygote compatibility

### Training Strategy: Two-Stage Optimization

**Stage 1: ADAM** - Global exploration
```julia
result_adam = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.01),
    maxiters=300-500,
    callback=callback
)
```
- Learning rate 0.01 is good starting point
- 300-500 iterations for initial convergence
- Escapes bad local minima

**Stage 2: BFGS** - Local refinement
```julia
result_bfgs = Optimization.solve(
    OptimizationProblem(optf, result_adam.u),
    OptimizationOptimJL.BFGS(initial_stepnorm=0.01),
    maxiters=100-150,
    allow_f_increases=false
)
```
- Start from ADAM solution
- Faster convergence near optimum
- May fail on non-smooth loss - wrap in try/catch

### Loss Function Design

**Trajectory-based loss** (preferred for dynamics):
```julia
function loss(p)
    total = 0f0
    for i in train_nodes
        u0 = data[i][:, 1]
        pred = solve_ode(u0, p, dynamics)  # Integrate full trajectory
        true_traj = data[i]
        total += sum(abs2, true_traj .- pred)
    end
    return total / (n_nodes * n_timesteps)
end
```

**Key considerations:**
- Normalize by number of nodes AND timesteps
- Use `Float32` for GPU compatibility and speed
- Consider weighting later timesteps more (error accumulates)

### Sensitivity Analysis (Adjoint Methods)

For differentiating through ODE solutions:
```julia
sol = solve(prob, Tsit5(), saveat=tsteps,
    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol=1f-5, reltol=1f-5)
```

- `InterpolatingAdjoint`: memory-efficient, recommended for long trajectories
- `ZygoteVJP()`: use Zygote for vector-Jacobian products
- Looser tolerances (1e-5 vs 1e-7) during training for speed; tighten for final evaluation

---

## Symbolic Regression

### Purpose

After training Neural ODE/UDE, extract interpretable equations from the learned NN.
This reveals the underlying physics in human-readable form.

### Approach with SymbolicRegression.jl

```julia
using SymbolicRegression

# Generate input-output pairs from trained NN
X_samples = generate_state_samples(bounds, n_samples)
Y_samples = [trained_nn(x) for x in X_samples]

# Run symbolic regression
options = Options(
    binary_operators=[+, -, *, /],
    unary_operators=[sin, cos, exp, log, sqrt],
    npopulations=20,
    maxsize=30,
    parsimony=0.001
)

hall_of_fame = EquationSearch(X_samples, Y_samples, options=options)
```

### Best Practices

1. **Sample densely** in regions where dynamics are interesting
2. **Include boundary regions** - dynamics often have boundary terms
3. **Start simple**: limit operator set, increase if needed
4. **Parsimony matters**: prefer simpler equations (lower complexity penalty)
5. **Cross-validate**: test discovered equations on held-out trajectories

### Interpreting Results

The goal is equations like:
```
dr/dt = -k(r - r_target) + α*exp(-r/ε)  # Radial relaxation + boundary
```

Rather than opaque neural network weights.

**Validation**: Symbolic equation should:
1. Achieve similar loss to trained NN
2. Generalize to unseen initial conditions
3. Make physical sense (units, signs, limiting behavior)

---

## Fundamental Limitation: Position vs Trajectory Recovery

### The Core Problem

RDPG identifies latent positions X only up to orthogonal transformation. This has critical implications for dynamics learning:

**What IS recovered well:**
| Quantity | Recovery Quality | Notes |
|----------|------------------|-------|
| P = XX' (edge probabilities) | ~5-8% error | This is what RDPG guarantees |
| Pairwise distances | ~0.97 correlation | Shape of point cloud preserved |
| Community structure | Qualitative | Relative positions preserved |

**What is NOT recovered:**
| Quantity | Recovery Quality | Notes |
|----------|------------------|-------|
| Absolute positions | ~55% relative error | Even with Procrustes alignment |
| Velocities (dx/dt) | ~0.05 correlation | Essentially random |
| Trajectory shape | Poor | Rotation drift corrupts signal |

### Why Trajectories Fail

The rotation Q that aligns X̂(t) to X(t) **drifts over time**:

```
t=1:  rotation drift = 0°   (aligned here)
t=10: rotation drift = 17°
t=20: rotation drift = 29°
t=30: rotation drift = 90°
```

This drift is not due to alignment error—even with **per-timestep oracle Procrustes**, position error remains ~35%. The rotation is fundamentally unidentifiable from network data.

### Experimental Evidence (n=50, d=3, K=30, T=30)

```
Per-timestep alignment error: 0.35 (oracle - best possible)
Global alignment error:       0.35
Gap (alignment drift):        0.01  ← Drift is NOT the main issue!

Velocity correlation:         0.045  ← Dynamics are lost
Distance correlation:         0.97   ← Shape is preserved
```

### Implications for Neural ODE Training

**Problem**: If we train Neural ODE on recovered trajectories X̂(t), the velocity signal is essentially noise.

**Potential solutions:**

1. **Distance-based dynamics**: Learn dynamics in terms of pairwise distances rather than coordinates
   ```
   d/dt D_ij = f(D)  instead of  d/dt X = f(X)
   ```

2. **Rotation-invariant features**: Use features that don't depend on coordinate system
   - Gram matrix G = XX'
   - Distance matrix D_ij = ||x_i - x_j||
   - Angles between position vectors

3. **Joint embedding + dynamics**: Constrain the embedding to have smooth rotations
   - Add temporal smoothness penalty on Q(t)
   - Use sliding window DUASE with overlapping windows

4. **Learn on P directly**: Model dynamics of the probability matrix
   ```
   d/dt P = f(P)  where P = XX'
   ```

### When Position Recovery Works Better

Recovery improves with:
- **Lower dimension d**: d=2 has less rotation freedom than d=3
- **More samples K**: K=100 gives ~2.6% P-error vs 13% for K=5
- **Denser networks**: Higher edge probabilities = better signal
- **Slower dynamics**: Less change per timestep = less drift

### Two Strategies for Rotation-Invariant Dynamics Learning

Since positions are not recoverable but pairwise distances are (0.97 correlation), we have two viable strategies:

#### Strategy 1: Direct Distance Dynamics

Model the evolution of the distance matrix D directly:

```
d/dt D_ij = f(D)

where D_ij = ||x_i - x_j||
```

**State space**: Upper triangle of D, dimension n(n-1)/2

**Pros:**
- Directly models what we can observe
- No rotation ambiguity at all
- Loss is straightforward: ||D_true - D_predicted||

**Cons:**
- Higher dimensional: n(n-1)/2 vs n·d (for n=50, d=3: 1225 vs 150)
- Constraints: D must satisfy triangle inequality, be embeddable
- Less interpretable: "distance dynamics" vs "position dynamics"

**Implementation:**
```julia
function distance_dynamics!(dD, D, p, t)
    # D is flattened upper triangle, length n(n-1)/2
    # f is a neural network or symbolic expression
    dD .= f(D, p)
end

# Loss compares distance matrices directly
loss = sum((D_predicted .- D_true).^2)
```

#### Strategy 2: Position Dynamics with Distance-Based Loss

Keep dynamics in position space X, but compute loss on distances:

```
d/dt X = f(X)    ← Neural ODE / UDE / Symbolic

Loss = ||D(X_pred) - D(X_true)||²

where D(X)_ij = ||X_i - X_j||
```

**State space**: Positions X, dimension n·d

**Critical requirement**: f(X) must be **rotation-equivariant**:
```
f(QX) = Q f(X)   for any rotation Q
```

This means: if positions rotate, velocities rotate the same way. Otherwise, the dynamics would depend on the arbitrary coordinate system.

**Pros:**
- Lower dimensional state space
- More interpretable (nodes moving in latent space)
- Standard Neural ODE machinery applies
- Symbolic regression gives position-based equations

**Cons:**
- Need differentiable distance computation
- Predicted positions are arbitrary (up to rotation), only distances matter
- f must be equivariant (constrains the function class)

**Implementation:**
```julia
function position_dynamics!(dX, X, p, t)
    # X is n×d matrix (flattened)
    # f is neural network or symbolic
    dX .= f(X, p)
end

# Distance matrix from positions
function compute_distances(X)
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        D[i,j] = norm(X[i,:] - X[j,:])
        D[j,i] = D[i,j]
    end
    return D
end

# Loss on distances, not positions
loss = sum((compute_distances(X_pred) .- compute_distances(X_true)).^2)
```

#### Comparison

| Aspect | Strategy 1 (Distance) | Strategy 2 (Position + Distance Loss) |
|--------|----------------------|---------------------------------------|
| State dimension | n(n-1)/2 | n·d |
| For n=50, d=3 | 1225 | 150 |
| Rotation handling | Eliminated | In loss function |
| Interpretability | Distance dynamics | Position dynamics |
| Constraints | Triangle inequality | None |
| Symbolic regression | On distances | On positions |

#### Recommendation

**Start with Strategy 2** (position dynamics + distance loss):
- Lower dimensional, faster training
- More interpretable symbolic equations
- Standard Neural ODE tools work directly

**Use Strategy 1** if:
- n is small (distance dimension manageable)
- You want to avoid any rotation-related issues in optimization
- Distance-based equations are the scientific target

### Rotation-Equivariant Dynamics

For Strategy 2 to work, f(X) must satisfy f(QX) = Qf(X). Here are classes of equivariant dynamics:

#### Building Blocks for Equivariant f(X)

The velocity of node i can only depend on:
1. **Relative positions**: (X_i - X_j)
2. **Distances**: d_ij = ||X_i - X_j||
3. **Position relative to centroid**: (X_i - X̄)
4. **Scalar functions of the above**

**General form:**
```
dX_i/dt = Σⱼ g(d_ij) · (X_i - X_j)  +  h(||X_i - X̄||) · (X_i - X̄)
          \_____________________/     \________________________/
           pairwise interactions        centroid interaction
```

where g(·) and h(·) are scalar functions (can be learned).

#### Interesting Equivariant Dynamics for Networks

**1. Attraction-Repulsion (Lennard-Jones style)**
```julia
# Nodes attract at long range, repel at short range
dXᵢ/dt = Σⱼ [ -a/dᵢⱼ² + b/dᵢⱼ⁴ ] · (Xᵢ - Xⱼ)/dᵢⱼ

# Equilibrium distance: d* = (2b/a)^(1/2)
```
Models: communities finding optimal separation

**2. Community Cohesion**
```julia
# Attraction to own community centroid, repulsion from others
dXᵢ/dt = -k_in · (Xᵢ - X̄_own) + k_out · Σ_other (Xᵢ - X̄_other)/||Xᵢ - X̄_other||
```
Models: communities maintaining cohesion while separating

**3. Distance-Dependent Coupling**
```julia
# Interaction strength depends on current connection probability
dXᵢ/dt = Σⱼ w(Pᵢⱼ) · (Xⱼ - Xᵢ)

where Pᵢⱼ = Xᵢ · Xⱼ (dot product = edge probability)
      w(p) = p · (1-p)  # Strongest at p=0.5
```
Models: uncertain edges driving dynamics

**4. Radial + Angular Dynamics**
```julia
# Nodes move radially based on degree, angularly based on neighbors
rᵢ = ||Xᵢ||
drᵢ/dt = α(r_target - rᵢ) + β·mean_neighbor_r

θᵢ = angle(Xᵢ)  # Only meaningful relative to neighbors
dθᵢ/dt = γ · Σⱼ sin(θⱼ - θᵢ) / dᵢⱼ  # Kuramoto-like
```
Models: degree evolution + community alignment

**5. Gradient Flow on Energy**
```julia
# Minimize energy E = Σᵢⱼ V(dᵢⱼ) + Σᵢ U(||Xᵢ||)
dXᵢ/dt = -∇ₓᵢ E = -Σⱼ V'(dᵢⱼ)·(Xᵢ-Xⱼ)/dᵢⱼ - U'(||Xᵢ||)·Xᵢ/||Xᵢ||
```
Models: network relaxing to equilibrium configuration

#### UDE Structure for Equivariant Dynamics

For Universal Differential Equations, encode known structure + learn corrections:

```julia
function equivariant_ude(X, p_known, p_nn)
    n, d = size(X)
    dX = zeros(n, d)

    # Known: attraction to centroid
    X_bar = mean(X, dims=1)
    dX .+= -p_known.k_centroid .* (X .- X_bar)

    # Known: pairwise repulsion at short range
    for i in 1:n, j in i+1:n
        r_ij = X[i,:] - X[j,:]
        d_ij = norm(r_ij)
        if d_ij > 0.01
            repulsion = p_known.k_repel / d_ij^2
            dX[i,:] .+= repulsion .* r_ij ./ d_ij
            dX[j,:] .-= repulsion .* r_ij ./ d_ij
        end
    end

    # Learned: distance-dependent correction (must be equivariant!)
    for i in 1:n, j in i+1:n
        r_ij = X[i,:] - X[j,:]
        d_ij = norm(r_ij)
        # NN takes scalar distance, outputs scalar coefficient
        coef = nn_scalar(d_ij, p_nn)  # NN: ℝ → ℝ
        dX[i,:] .+= coef .* r_ij ./ d_ij
        dX[j,:] .-= coef .* r_ij ./ d_ij
    end

    return dX
end
```

**Key insight**: The NN only outputs scalars (coefficients), which multiply equivariant vectors (relative positions). This preserves equivariance.

#### Symbolic Regression Target

After training, extract symbolic form of g(d):

```julia
# If NN learned: g(d) ≈ -0.5/d + 0.1/d³
# Then dynamics are:
dXᵢ/dt = Σⱼ [-0.5/dᵢⱼ + 0.1/dᵢⱼ³] · (Xᵢ - Xⱼ)/dᵢⱼ

# = attraction at 1/d² + repulsion at 1/d⁴
# Equilibrium at d* where g(d*) = 0
```

#### Verified Properties

Tested with Lennard-Jones style dynamics g(d) = -a/d + b/d³:

| Property | Verification |
|----------|--------------|
| f(QX) = Qf(X) | Error = 2×10⁻¹² ✓ |
| D(X(t)) = D((QX)(t)) | Identical distance trajectories ✓ |
| Loss(X_rot, X_true) ≈ 0 | Rotation-invariant loss ✓ |
| Equilibrium d* = √(b/a) | System converges correctly ✓ |

---

## Equivariant Dynamics for Case Studies

This section provides **rotation-equivariant** formulations for each of the 4 main examples.
These can be learned from RDPG observations because they only depend on pairwise distances.

### Example 1: Bridge Node Oscillation (Equivariant Version)

**Original dynamics**: Node oscillates between fixed attractors A₁, A₂.
**Problem**: Fixed points in absolute coordinates are NOT rotation-equivariant.

**Equivariant reformulation**: Use other nodes as implicit attractors.

```
Setup:
- n nodes total: {community_1} ∪ {community_2} ∪ {bridge}
- X̄₁ = mean(X_i for i ∈ community_1)  # centroid of community 1
- X̄₂ = mean(X_i for i ∈ community_2)  # centroid of community 2
- d₁ = ||X_bridge - X̄₁||, d₂ = ||X_bridge - X̄₂||

Equivariant dynamics:
  dX_bridge/dt = σ(d₁ - d₂) · (X̄₂ - X_bridge) + σ(d₂ - d₁) · (X̄₁ - X_bridge)
               + soft_boundary_repulsion(X_bridge)

where σ(x) = 1/(1 + exp(-kx)) is a sigmoid switch.
```

**Why equivariant**: Centroids X̄₁, X̄₂ transform as QX̄, so the dynamics satisfy f(QX) = Qf(X).

**Julia implementation**:
```julia
function bridge_node_dynamics(X, i_bridge, community_1, community_2; k_switch=10.0, k_attract=0.5)
    n, d = size(X)
    X_bar1 = mean(X[community_1, :], dims=1)[1,:]
    X_bar2 = mean(X[community_2, :], dims=1)[1,:]

    d1 = norm(X[i_bridge, :] - X_bar1)
    d2 = norm(X[i_bridge, :] - X_bar2)

    sigma1 = 1 / (1 + exp(-k_switch * (d1 - d2)))  # High when closer to C1
    sigma2 = 1 - sigma1

    # When close to C1, get pushed to C2, and vice versa
    dx = k_attract * (sigma1 * (X_bar2 - X[i_bridge, :]) +
                      sigma2 * (X_bar1 - X[i_bridge, :]))
    return dx
end
```

**Learnable parameters**: k_switch (switching sharpness), k_attract (attraction strength)

---

### Example 2: Circulation (Equivariant Version)

**Original dynamics**: ω[-x₂, x₁] rotation around origin.
**Problem**: The origin is a fixed point; rotation around it is NOT equivariant.

**Equivariant reformulation**: Circulation around the **centroid**.

```
Setup:
- X̄ = mean(X_i for all i)  # system centroid
- r_i = X_i - X̄            # position relative to centroid

For d=2:
  dX_i/dt = ω · J · (X_i - X̄)  where J = [0, -1; 1, 0] is rotation by 90°

For general d:
  dX_i/dt = Σⱼ ω_jk · P_jk · (X_i - X̄)
  where P_jk rotates in the (j,k) plane
```

**Why equivariant**: The centroid X̄ transforms as QX̄, so (X_i - X̄) transforms covariantly.

**Key insight**: Pure circulation (no radial component) preserves ||X_i - X̄||, hence preserves
inter-node distances. This means:
- **Circulation is observable!** Even though positions rotate, the distances stay fixed.
- In fact, if ALL nodes circulate with the same ω, this is **invisible** in RDPG (pure rotation).
- Observable dynamics: different nodes have different ω, or there's radial drift.

**Mixed observable dynamics**:
```julia
function circulation_with_drift(X; omega_base=0.1, k_radial=0.05, r_target=0.5)
    n, d = size(X)
    X_bar = mean(X, dims=1)[1,:]

    dX = zeros(n, d)
    for i in 1:n
        r_i = X[i, :] - X_bar
        r_norm = norm(r_i)

        # Circulation around centroid (only observable if ω varies)
        if d == 2
            dX[i, :] = omega_base * [-r_i[2], r_i[1]]
        end

        # Radial drift toward target radius (observable!)
        if r_norm > 1e-6
            radial_force = k_radial * (r_target - r_norm)
            dX[i, :] += radial_force * r_i / r_norm
        end
    end
    return dX
end
```

**What's learnable from RDPG**:
- k_radial (radial drift strength) - ✓ Observable
- r_target (equilibrium radius) - ✓ Observable
- omega_base (circulation speed) - ✗ Unobservable if uniform (rotation-invariant)
- ω variations between nodes - ✓ Observable (changes relative angles)

---

### Example 3: Two Communities Joining

**Original dynamics**: Two communities move toward each other in time.
**Naturally equivariant**: This dynamics is already equivariant!

```
Setup:
- Nodes partitioned: community_1, community_2
- X̄₁ = centroid of community 1
- X̄₂ = centroid of community 2

Equivariant dynamics:
  For i ∈ community_1:
    dX_i/dt = -k_in · (X_i - X̄₁) + k_out · (X̄₂ - X_i)
            = -k_in · (X_i - X̄₁) + k_out · (X̄₂ - X_i)

  For i ∈ community_2:
    dX_i/dt = -k_in · (X_i - X̄₂) + k_out · (X̄₁ - X_i)

Simplified:
  dX_i/dt = -k_in · (X_i - X̄_own) + k_out · (X̄_other - X_i)
```

**Observable effects**:
- Intra-community distances shrink if k_in > 0 (cohesion)
- Inter-community distances shrink if k_out > 0 (joining)
- Eventually communities merge if k_out > 0

**Julia implementation**:
```julia
function communities_joining_dynamics(X, community_1, community_2; k_in=0.1, k_out=0.05)
    n, d = size(X)
    X_bar1 = mean(X[community_1, :], dims=1)[1,:]
    X_bar2 = mean(X[community_2, :], dims=1)[1,:]

    dX = zeros(n, d)
    for i in 1:n
        if i in community_1
            X_bar_own, X_bar_other = X_bar1, X_bar2
        else
            X_bar_own, X_bar_other = X_bar2, X_bar1
        end

        # Cohesion: attract to own centroid
        dX[i, :] = -k_in * (X[i, :] - X_bar_own)
        # Joining: attract to other centroid
        dX[i, :] += k_out * (X_bar_other - X[i, :])
    end
    return dX
end
```

**Learnable parameters**: k_in (cohesion), k_out (joining rate)

**Equilibrium**: When k_in > 0 and k_out > 0, communities converge to a merged state where
X̄₁ = X̄₂ and all nodes cluster around the global centroid.

---

### Example 4: Food Web (Predator-Prey-Resource)

**Original dynamics**: Type-specific fixed attractors + interactions.
**Problem**: Fixed target positions are NOT equivariant.

**Equivariant reformulation**: Replace fixed targets with type-centroid interactions.

```
Setup:
- Three node types: Predator (P), Prey (Y), Resource (R)
- X̄_P = mean(X_i for i ∈ Predator)
- X̄_Y = mean(X_i for i ∈ Prey)
- X̄_R = mean(X_i for i ∈ Resource)

Equivariant dynamics:

For Predator i:
  dX_i/dt = -k_P · (X_i - X̄_P)           # Cohesion with other predators
          + α_hunt · (X̄_Y - X_i)          # Hunt: attracted to prey centroid

For Prey i:
  dX_i/dt = -k_Y · (X_i - X̄_Y)           # Cohesion with other prey
          - α_flee · (X̄_P - X_i)          # Flee: repelled from predator centroid
          + α_feed · (X̄_R - X_i)          # Feed: attracted to resource centroid

For Resource i:
  dX_i/dt = -k_R · (X_i - X̄_R)           # Cohesion with other resources
          - α_deplete · Σ_{j∈Prey} 1/||X_i - X_j||² · (X_i - X_j)  # Depletion pressure
```

**Why equivariant**: All interactions are relative (X_i - X̄ or X_i - X_j).

**Julia implementation**:
```julia
function food_web_dynamics(X, predators, prey, resources;
                           k_cohesion=0.1, alpha_hunt=0.08,
                           alpha_flee=0.1, alpha_feed=0.05)
    n, d = size(X)

    X_bar_P = mean(X[predators, :], dims=1)[1,:]
    X_bar_Y = mean(X[prey, :], dims=1)[1,:]
    X_bar_R = mean(X[resources, :], dims=1)[1,:]

    dX = zeros(n, d)

    for i in 1:n
        if i in predators
            # Cohesion + hunting
            dX[i, :] = -k_cohesion * (X[i, :] - X_bar_P)
            dX[i, :] += alpha_hunt * (X_bar_Y - X[i, :])

        elseif i in prey
            # Cohesion + fleeing + feeding
            dX[i, :] = -k_cohesion * (X[i, :] - X_bar_Y)
            dX[i, :] -= alpha_flee * (X_bar_P - X[i, :])  # Note: repulsion
            dX[i, :] += alpha_feed * (X_bar_R - X[i, :])

        else  # Resource
            # Cohesion only (could add depletion dynamics)
            dX[i, :] = -k_cohesion * (X[i, :] - X_bar_R)
        end
    end
    return dX
end
```

**Observable patterns**:
- Predator-prey distances oscillate (chase-flee cycles)
- Prey-resource distances correlate with feeding
- Type centroids have characteristic separations

**Learnable parameters**: k_cohesion, alpha_hunt, alpha_flee, alpha_feed

---

### Summary: Equivariant Dynamics for All Examples

| Example | Original Issue | Equivariant Fix | Key Observable |
|---------|---------------|-----------------|----------------|
| 1. Bridge Node | Fixed attractors A₁, A₂ | Use community centroids | Bridge-community distances |
| 2. Circulation | Fixed origin | Centroid-relative + radial drift | Radial distances to centroid |
| 3. Communities | Already equivariant | N/A | Inter/intra-community distances |
| 4. Food Web | Fixed type targets | Type centroid interactions | Type-centroid separations |

**Key principle**: Replace any fixed point with a data-dependent centroid or relative position.

---

## Evaluation Framework: What We Actually Care About

### The Key Insight

We don't need to recover absolute positions X(t) - we care about **rotation-invariant quantities**:
- **Pairwise distances** D_ij = ||X_i - X_j||
- **Probability matrix** P = XX'
- **Functional form** of the dynamics (via symbolic regression)

This means rotation ambiguity in RDPG estimation might not matter, as long as we evaluate correctly.

### Experimental Setup: 2 × 4 Examples

We run **8 experiments** (2 dynamics formulations × 4 scenarios):

**Formulation A**: Original position-based dynamics
- Train on estimated positions X̂(t)
- Loss: ||X_pred(t) - X_target(t)||²

**Formulation B**: Rotation-equivariant dynamics
- Same data, but use equivariant architecture
- OR use distance-based loss: ||D(X_pred) - D(X_target)||²

**4 Scenarios**:
1. Bridge Node Oscillation
2. Circulation
3. Two Communities Joining
4. Food Web (Predator-Prey-Resource)

### Three Evaluation Comparisons

For ALL experiments, evaluate with these metrics:

```
┌─────────────────┬──────────────────────┬─────────────────────┐
│   Comparison    │      Metrics         │     Measures        │
├─────────────────┼──────────────────────┼─────────────────────┤
│ True ↔ Estimated│ D correlation, P err │ RDPG estimation     │
│                 │                      │ quality (baseline)  │
├─────────────────┼──────────────────────┼─────────────────────┤
│ True ↔ Recovered│ D correlation, P err │ Dynamics learning   │
│                 │                      │ quality (our goal)  │
├─────────────────┼──────────────────────┼─────────────────────┤
│ Est ↔ Recovered │ Position RMSE, D corr│ Training fit        │
│                 │                      │ (sanity check)      │
└─────────────────┴──────────────────────┴─────────────────────┘
```

### Metric Definitions

```julia
# Distance correlation
D_corr(X1, X2) = cor(upper_tri(D(X1)), upper_tri(D(X2)))

# P reconstruction error (relative)
P_err(X1, X2) = ||X1*X1' - X2*X2'||_F / ||X2*X2'||_F

# Position RMSE (only meaningful for Est ↔ Rec)
Pos_RMSE(X1, X2) = sqrt(mean((X1 - X2).^2))
```

### Success Criteria

**Good RDPG estimation** (True ↔ Estimated):
- D correlation > 0.95
- P error < 0.10

**Good dynamics learning** (True ↔ Recovered):
- D correlation > 0.90 × (True ↔ Estimated correlation)
- P error < 1.5 × (True ↔ Estimated error)

The dynamics learning should be nearly as good as the RDPG estimation baseline.
If it's much worse, the learned dynamics don't generalize.

### Validation Metrics

For both strategies, evaluate on:

1. **Distance prediction error**: ||D_pred - D_true|| / ||D_true||
2. **P reconstruction**: ||X_pred X_pred' - X_true X_true'|| (for Strategy 2)
3. **Community distance preservation**: Are inter/intra-community distances captured?

---

## RDPG Evaluation Metrics

### Important: P-reconstruction vs Position Error

RDPG only identifies latent positions X up to orthogonal rotation. This has implications for evaluation:

**Correct metric: P reconstruction error**
```julia
function P_error(L_hat_series, X_true_series)
    T = length(L_hat_series)
    err = 0.0
    for t in 1:T
        P_true = X_true_series[t] * X_true_series[t]'
        P_hat = L_hat_series[t] * L_hat_series[t]'
        err += norm(P_true - P_hat) / norm(P_true)
    end
    return err / T
end
```

This measures what RDPG actually cares about: recovering edge probabilities.

**Misleading metric: Per-node position error**
```julia
# This can be misleading because rotations differ at each time!
err = norm(L_hat[t] - X_true[t]) / n  # BAD
```

Even with Procrustes alignment, position error can be high because:
1. RDPG is rotation-invariant
2. Different rotations at different times break trajectory smoothness
3. Noise in alignment accumulates

### Experimental Results (DUASE vs OMNI vs Folded)

| Method | P-reconstruction | Position error |
|--------|------------------|----------------|
| DUASE  | **0.049**        | 0.30           |
| OMNI   | 0.072            | 0.30           |
| Folded | 0.083            | 0.23           |

- **DUASE** gives best P-reconstruction (the correct metric)
- Position error is similar across methods because of rotation ambiguity
- Folded has lower position error but higher P-error (misleading)

### Effect of Repeated Samples K

| K | P-reconstruction error |
|---|------------------------|
| 5 | 0.122 |
| 10 | 0.082 |
| 30 | 0.057 |
| 50 | 0.045 |
| 100 | 0.032 |

More samples → better P reconstruction, as expected.

---

## Common Pitfalls and Solutions

### 1. "Mutating arrays not supported" error
**Cause**: In-place operations in dynamics function
**Fix**: Replace `.+=` with `= ... .+`, avoid indexed assignment

### 2. ODE solver hits maxiters
**Cause**: Dynamics produce NaN/Inf, or very stiff system
**Fix**: Add bounds checking, use softer boundary terms, check NN output scale

### 3. Loss doesn't decrease
**Cause**: Learning rate too high/low, bad initialization, wrong loss scale
**Fix**: Try different LR (0.001, 0.01, 0.1), check initial loss magnitude

### 4. UDE performs worse than Full NN
**Cause**: Known structure is misleading or wrong
**Fix**: Verify known dynamics are actually correct, simplify known part

### 5. Training is very slow
**Cause**: Tight ODE tolerances, large network, many trajectories
**Fix**: Use 1e-5 tolerances during training, batch trajectories, smaller network

---

## Recommended Packages

```julia
# Core
using Lux                    # Neural networks (preferred over Flux for SciML)
using OrdinaryDiffEq         # ODE solvers
using SciMLSensitivity       # Adjoint sensitivity
using Optimization           # Unified optimization interface
using OptimizationOptimisers # ADAM, etc.
using OptimizationOptimJL    # BFGS, L-BFGS

# Utilities
using ComponentArrays       # Named parameter arrays
using Zygote               # Automatic differentiation
using LinearAlgebra        # SVD, norm, etc.
using Arpack               # Truncated SVD for large matrices

# Symbolic Regression
using SymbolicRegression   # Equation discovery

# Visualization
using CairoMakie           # Publication-quality plots
```

---

## Fundamental Limitation: RDPG Collapse Degeneracy

### The Problem

In RDPG, connection probabilities are given by P_ij = x_i · x_j^T. When all nodes collapse to the same position (x_i = x_j = x for all i, j):

```
P_ij = x · x^T = ||x||²  (constant for all i, j)
```

This means:
1. **Only magnitude is identifiable**: We can recover ||x||, but the direction of x is completely lost
2. **Distances provide no signal**: D_ij = ||x_i - x_j|| = 0 for all pairs
3. **Positions are pure noise**: SVD embedding can return x pointing in ANY direction, constrained only by ||x̂|| ≈ ||x||

### Why This Makes Learning Hard

When the true dynamics cause nodes to cluster tightly:
- **D correlation becomes meaningless**: True distances have near-zero variance, so correlation is dominated by noise
- **P error can still be low**: A constant matrix is easy to approximate (even with wrong positions)
- **Position estimates are arbitrary**: The embedding rotation is unconstrained

### Diagnostic Signs

If you see:
- D correlation ≈ 0 but P error is low
- D std → 0 over time
- Position spread → 0 (all nodes at same location)

Then your dynamics have a **collapse degeneracy**. The RDPG observation model cannot distinguish positions when nodes are co-located.

### Solution

Use dynamics that maintain structural diversity:
- Oscillating dynamics around fixed points
- Repulsive forces to prevent collapse
- Multiple stable equilibria (communities that stay separated)

---

## Position vs Distance Loss: Key Experimental Finding

### Background

A natural question for rotation-invariant problems: should we use a rotation-invariant loss
function (e.g., distance-based loss) instead of position-based loss?

Our goals ARE rotation-invariant:
- **D correlation**: Pairwise distances (invariant to rotation)
- **P reconstruction**: P = XX' (invariant to joint rotation of L, R)

### Experiment: `scripts/experiment_template.jl`

Compared Neural ODE training with:

1. **Position loss**: ||X_pred - X_target||² (standard MSE)
2. **Distance loss**: ||D(X_pred) - D(X_target)||² (rotation-invariant)

### Results

| Metric | Position Loss | Distance Loss | Baseline (RDPG) |
|--------|---------------|---------------|-----------------|
| D corr (True↔Rec) | 0.106 | 0.030 | 0.131 |
| P error (True↔Rec) | 0.093 | 13.1 | 0.077 |

**Key finding**: Position-based training achieves rotation-invariant goals BETTER than
distance-based training!

### Why Distance Loss Fails

1. **Weaker gradient signal**: Distance loss doesn't tell positions WHERE to go, only that
   their relative distances should match. The Neural ODE slowly drifts.

2. **Degenerate solutions**: The model can minimize distance loss while having completely
   wrong absolute positions (scale drift, degenerate configurations).

3. **Training converged but evaluation failed**: Distance loss achieved 0.047 (lower than
   position loss's 0.40!) but produced catastrophic evaluation results.

### Implication for RDPG Dynamics

**You don't need equivariant architectures or rotation-invariant losses.**

Standard position-based training on RDPG-estimated positions works well for achieving
rotation-invariant goals. The stronger gradient signal from position loss keeps the
Neural ODE on track, even though different runs of RDPG estimation produce different
rotations.

This simplifies the approach: train on estimated positions, evaluate on rotation-invariant
metrics (D correlation, P reconstruction).

---

## Performance Optimization for Neural ODE / UDE Training

Based on research from the [SciML documentation](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/),
[SciMLSensitivity benchmarks](https://docs.sciml.ai/SciMLSensitivity/dev/Benchmark/), and
[Julia Discourse discussions](https://discourse.julialang.org/t/universal-differential-equations-ude-with-modern-sciml-stack-lux-enzyme/120869).

### 1. Use Float32 Throughout (Critical!)

Float32 provides ~2x speedup over Float64 with minimal accuracy loss for neural networks.

**Pattern:**
```julia
# Initial conditions
u0 = Float32[0.5, 0.5]

# Time spans (note the f0 suffix!)
tspan = (0.0f0, 30.0f0)
tsteps = range(0.0f0, 30.0f0, length=31)

# Data
X_data = Float32.(X_data)

# Neural network setup (Lux respects types correctly)
rng = Random.default_rng()
nn = Lux.Chain(Lux.Dense(32, 64, tanh), Lux.Dense(64, 32))
ps, st = Lux.setup(rng, nn)
ps_f32 = ComponentArray{Float32}(ps)  # Explicit Float32 parameters
```

**Why Lux over Flux**: Flux Chain does not respect Julia's type promotion rules. Restructuring
a Flux neural network will silently downgrade Float64 → Float32. Lux handles types correctly.

### 2. Sensitivity Algorithm Selection

The choice of sensitivity algorithm dramatically affects training speed.

| Problem Type | Recommended Algorithm | VJP Choice |
|--------------|----------------------|------------|
| **Non-stiff Neural ODE** | `BacksolveAdjoint` | `ZygoteVJP()` |
| **Stiff problems** | `InterpolatingAdjoint(checkpointing=true)` | `ReverseDiffVJP(true)` |
| **Large networks** | `GaussAdjoint` | `ZygoteVJP()` |
| **Small params (<100)** | `ForwardDiffSensitivity` | N/A |

**For our RDPG dynamics (non-stiff, medium network):**
```julia
sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP())

# Or for more stability:
sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
```

### 3. VJP (Vector-Jacobian Product) Hierarchy

Performance ranking (fastest to slowest):
1. **`EnzymeVJP()`** - Fastest but limited to non-allocating code
2. **`ReverseDiffVJP(true)`** - Fast for mutation-based code, requires compilation
3. **`ZygoteVJP()`** - Best for vectorized, non-mutating functions (recommended default)
4. **`TrackerVJP()`** - Fallback only

**Note**: `ReverseDiffVJP(true)` only works if f has no branches (if/while statements).

### 4. In-Place vs Out-of-Place Dynamics

For small systems (<20 variables), out-of-place with StaticArrays is fastest.
For larger systems, in-place functions reduce allocations.

**Out-of-place (our case, n*d ≈ 24-32):**
```julia
function dynamics(u, p, t)
    # Non-mutating, returns new array
    u_reshape = reshape(u, n * d, 1)
    out, _ = nn(u_reshape, p, st)
    return vec(out)
end
```

**In-place (for larger systems):**
```julia
function dynamics!(du, u, p, t)
    u_reshape = reshape(u, n * d, 1)
    out, _ = nn(u_reshape, p, st)
    du .= vec(out)  # Mutate in place
    return nothing
end
```

### 5. Solver Settings

**Non-stiff (our oscillatory dynamics):**
```julia
solver = Tsit5()  # Fast explicit Runge-Kutta
abstol = 1.0f-5   # Float32 tolerances
reltol = 1.0f-5
```

**If stiff:**
```julia
solver = TRBDF2()  # Or Rodas5 for small systems
```

### 6. Two-Stage Optimizer Strategy

The [DiffEqFlux documentation](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/)
recommends ADAM → BFGS for fastest convergence:

```julia
# Stage 1: ADAM for rapid initial convergence
result_adam = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.01f0),  # Float32 LR
    maxiters = 500
)

# Stage 2: BFGS for fine-tuning
result_bfgs = Optimization.solve(
    optprob_bfgs,
    Optim.BFGS(initial_stepnorm = 0.01f0),
    maxiters = 100,
    allow_f_increases = false
)
```

"By using the two together, we can fit the neural ODE in 9 seconds!"

### 7. Type Stability Checklist

- [ ] All initial conditions are `Float32`
- [ ] Time spans use `f0` suffix: `(0.0f0, 30.0f0)`
- [ ] Neural network parameters are `ComponentArray{Float32}`
- [ ] Training data is `Float32`
- [ ] Solver tolerances are `Float32`: `abstol=1f-5`
- [ ] No global variables in dynamics function
- [ ] Learning rate is `Float32`: `Adam(0.01f0)`

### 8. Memory Optimization

**Avoid allocations in dynamics:**
```julia
# BAD: allocates new array each call
function dynamics(u, p, t)
    dX = zeros(n, d)  # Allocation!
    ...
end

# GOOD: vectorized operations
function dynamics(u, p, t)
    X = reshape(u, n, d)
    delta = X .- centers  # Broadcasts, minimal allocation
    ...
end
```

**Use `@.` for broadcast fusion:**
```julia
@. du = -omega * delta_y - k * delta_x
```

### 9. Complete Optimized Setup

```julia
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization
using OptimizationOptimisers, OptimizationOptimJL, ComponentArrays

# Float32 everything
const T = Float32
u0 = T[0.5, 0.5, ...]
tspan = (zero(T), T(30))
tsteps = range(zero(T), T(30), length=31)

# Neural network
nn = Lux.Chain(Lux.Dense(n*d, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, n*d))
ps, st = Lux.setup(rng, nn)
ps = ComponentArray{T}(ps)

# Dynamics (non-mutating for Zygote)
function dynamics(u, p, t)
    out, _ = nn(reshape(u, n*d, 1), p, st)
    return vec(out)
end

# ODE Problem
prob = ODEProblem(dynamics, u0, tspan, ps)

# Solve with optimized settings
sol = solve(prob, Tsit5(),
    saveat = tsteps,
    sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP()),
    abstol = T(1e-5),
    reltol = T(1e-5)
)
```

### 10. Future: Even Faster with SimpleChains + Enzyme

For production code, the [SciML recommendation](https://docs.sciml.ai/SciMLSensitivity/dev/manual/differential_equation_sensitivities/)
is to prototype with Lux + Zygote, then move to SimpleChains.jl + EnzymeVJP for
"an order of magnitude improvement (or more)."

```julia
# After prototyping works:
using SimpleChains, Enzyme
nn_fast = SimpleChain(...)
sensealg = GaussAdjoint(autojacvec=EnzymeVJP())
```

---

## B^d_+ (Non-Negative Unit Ball) Alignment Functions

### The Rotation Ambiguity Problem

SVD embedding recovers latent positions **only up to orthogonal rotation**. If the true positions
lie in B^d_+ = {x ∈ R^d : x ≥ 0, ||x|| ≤ 1}, the SVD output might be arbitrarily rotated into
any orthant.

**Mathematical statement**: If A = LR' and Q is orthogonal, then A = (LQ)(RQ)' = L'R''. SVD cannot
distinguish between (L, R) and (LQ, RQ) for any orthogonal Q.

**Why B^d_+ matters**: For valid RDPG probabilities, we need P_ij = L_i · L_j ∈ [0, 1]. If L ∈ B^d_+,
then ||L_i|| ≤ 1 and L_i ≥ 0, guaranteeing L_i · L_j ∈ [0, 1].

### Available Functions (src/embedding.jl)

#### 1. Point-Level Operations

```julia
# Check if a point is in B^d_+
in_Bd_plus(x::AbstractVector; tol=1e-10) -> Bool
# Returns true if all(x .≥ -tol) && norm(x) ≤ 1 + tol

# Project a single point onto B^d_+
project_to_Bd_plus(x::AbstractVector) -> Vector
# Two-step: (1) clamp negatives to 0, (2) scale if norm > 1

# Project all rows of embedding matrix
project_embedding_to_Bd_plus(L::AbstractMatrix) -> Matrix
# Applies project_to_Bd_plus to each row
```

#### 2. Alignment with Known Reference (Oracle Mode)

```julia
# Procrustes alignment to known ground truth
align_to_reference(L_hat::AbstractMatrix, L_ref::AbstractMatrix) -> Matrix
# Finds Q minimizing ||L_hat * Q - L_ref||_F via SVD of L_hat' * L_ref
# Returns L_hat * Q

# Use case: Synthetic experiments where we know true positions
L_aligned = align_to_reference(L_svd, L_true)
```

#### 3. Heuristic Alignment (No Ground Truth)

```julia
# Single matrix alignment
align_to_Bd_plus(L::AbstractMatrix; max_iters=100) -> (L_aligned, Q)
# Algorithm:
#   1. Sign flips: If majority of column j is negative, flip sign
#   2. Column swaps (d=2): Try swapping if it reduces negatives
#   3. Final projection to ensure constraints
# Returns (aligned matrix, orthogonal transformation Q)
```

#### 4. Global Alignment for Temporal Series

```julia
# Find SINGLE rotation Q for entire temporal series
find_global_orthogonal_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                                   max_iters=100, n_angles=36) -> (Q, S)
# Algorithm:
#   1. Stack all T matrices into (n*T) × d super-matrix
#   2. Sign flips (S): Flip column signs to maximize positives
#   3. Rotation search (R):
#      - d=2: Grid search over θ ∈ [0, 2π] + local refinement
#      - d=3,4: Joint optimization over d(d-1)/2 angles
#      - d>4: Iterative Givens rotations until convergence
# Returns Q = S * R where S is sign matrix, R is proper rotation

# Full pipeline with statistics
align_series_to_Bd_plus(L_series::Vector{<:AbstractMatrix};
                         method=:project,
                         Q=nothing) -> NamedTuple
# Methods for handling remaining violations:
#   :project - Per-point projection (clamp + scale)
#   :rescale - Global scaling to fit largest point in unit ball
#   :none    - Only apply rotation, no projection
# Returns: (L_aligned, Q, scale, stats)
```

### Optimization Compatibility

**Current status**: These functions are designed for **preprocessing**, not training.

| Aspect | Status | Notes |
|--------|--------|-------|
| Float32 support | ❌ | Hardcoded Float64 throughout |
| Zygote compatible | ❌ | Uses in-place operations (`.*=`) |
| Type-stable | ⚠️ | Mostly stable but some dynamic allocation |

**Why not AD-compatible**: The B^d_+ alignment problem involves discrete decisions (which column
to flip, which rotation angle to choose) that cannot be differentiated through. The functions
use grid search and iterative refinement, not gradient descent.

**Recommended usage pattern**:

```julia
# 1. Generate/load data (Float64 is fine here)
X_true_series = generate_true_dynamics(...)

# 2. RDPG estimation with alignment (preprocessing, Float64)
X_est_raw = [svd_embed(A_avg[t], d) for t in 1:T]
result = align_series_to_Bd_plus(X_est_raw; method=:project)
X_est_aligned = result.L_aligned

# 3. Convert to Float32 for training
X_train = [Float32.(X) for X in X_est_aligned]

# 4. Train Neural ODE (Float32, AD-compatible code only here)
loss = train_node(X_train, ...)
```

### When to Use Each Function

| Scenario | Function | Notes |
|----------|----------|-------|
| Synthetic data with known truth | `align_to_reference` | Optimal - uses Procrustes |
| Real data, single graph | `align_to_Bd_plus` | Heuristic sign flips + rotation |
| Real data, temporal series | `align_series_to_Bd_plus` | Global Q across all timesteps |
| Quick check | `in_Bd_plus` | Verify if projection needed |

### The Violation Score

The `_compute_Bd_plus_violation(L)` function measures how far L is from B^d_+:

```
violation = Σᵢⱼ |min(Lᵢⱼ, 0)| + Σᵢ max(||Lᵢ|| - 1, 0)
```

- First term: penalty for negative entries
- Second term: penalty for points outside unit ball
- Lower is better; 0 means all points are in B^d_+

### Example: Full Embedding Pipeline

```julia
using RDPGDynamics

# Generate true dynamics
X_true = [evolve(X0, t) for t in 1:30]

# Sample adjacency matrices (K repetitions)
A_series = [sample_adjacency_repeated(X_true[t], 100) for t in 1:30]

# SVD embedding
L_raw = [svd_embedding(A, 2).L_hat for A in A_series]

# Temporal Procrustes chain (align to previous timestep)
L_aligned = Vector{Matrix{Float64}}(undef, 30)
L_aligned[1] = L_raw[1]
for t in 2:30
    L_aligned[t] = align_to_reference(L_raw[t], L_aligned[t-1])
end

# Global B^d_+ alignment
result = align_series_to_Bd_plus(L_aligned; method=:project)
L_final = result.L_aligned

# Check alignment quality
println("Negative values before: ", result.stats[:neg_before])
println("Negative values after:  ", result.stats[:neg_after])
println("Max norm after:         ", result.stats[:max_norm_after])

# Convert to Float32 for training
L_train = [Float32.(L) for L in L_final]
```

### Future: AD-Compatible B^d_+ Projection

If you need to enforce B^d_+ constraints **during training** (as a soft constraint in the loss),
use the vectorized, type-stable version:

```julia
# AD-compatible soft constraint loss
function bd_plus_penalty(L::AbstractMatrix{T}) where T
    # Penalty for negatives (smooth approximation)
    neg_penalty = sum(softplus.(-L))  # softplus(x) = log(1 + exp(x))

    # Penalty for points outside unit ball
    row_norms = sqrt.(sum(L.^2, dims=2))
    norm_penalty = sum(softplus.(row_norms .- one(T)))

    return neg_penalty + norm_penalty
end
```

This can be added to the training loss with a weight λ:
```julia
total_loss = mse_loss + λ * bd_plus_penalty(L_predicted)
```

---

*Last updated: 2026-01-16*
