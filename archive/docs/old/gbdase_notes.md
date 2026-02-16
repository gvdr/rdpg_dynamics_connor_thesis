# GB-DASE: Generalized Bayesian Dynamic Adjacency Spectral Embedding

## Reference

- **Paper**: "Generalized Bayesian Inference for Dynamic Random Dot Product Graphs" (arXiv:2509.19748)
- **Implementation**: https://github.com/joshloyal/DynamicRDPG

---

## 1. Problem Setting

We observe a sequence of adjacency matrices A(1), ..., A(m) from a dynamic network.

**RDPG Model**:
- Each node i has a latent position x_i(t) in R^d
- Edge probability: P_ij(t) = x_i(t)' x_j(t)
- Observation: A_ij(t) ~ Bernoulli(P_ij(t))

**Goal**: Estimate latent positions X(t) = [x_1(t), ..., x_n(t)]' from noisy binary observations.

---

## 2. Model Specification

### 2.1 Observation Model (Gibbs Posterior)

GB-DASE uses a **Gibbs posterior** (generalized Bayesian) with least-squares loss:

```
π(X | A) ∝ exp(-λ Σ_t ||A(t) - X(t)X(t)'||²_F) × π(X)
```

where λ > 0 is a "learning rate" or "scale" parameter.

**CRITICAL**: The loss is computed on the **lower triangular** part only (unique dyads, excluding diagonal):
```python
subdiag = np.tril_indices(n_nodes, k=-1)
y[t] = A[t][subdiag]  # Only lower triangular!
```

This means:
- NO diagonal entries (no self-loops)
- NO redundant upper triangular (since A is symmetric)
- n_dyads = n(n-1)/2 observations per timestep

**BUT**: The full adjacency matrix A is used for:
- SVD/ASE initialization (needs full symmetric structure)
- Computing XtX and XtY in Gibbs updates

The lower triangular restriction is ONLY for the loss computation (avoiding double-counting).

### 2.2 Random Walk Prior

The prior on X uses an **r-th order Gaussian random walk**:

```
Δ^r x_i(t) = σ_i * w_i(t),   w_i(t) ~ N(0, I_d)
```

where Δ^r is the r-th order difference operator:
- r=1: x(t) - x(t-1) ~ N(0, σ²I)  (penalizes velocity)
- r=2: x(t) - 2x(t-1) + x(t-2) ~ N(0, σ²I)  (penalizes acceleration)

**Precision matrix construction**:
```python
D = np.diff(np.eye(m), r, axis=0)  # (m-r) × m difference matrix
K = np.kron(D.T @ D, np.eye(d))     # md × md precision matrix
```

The prior on the full trajectory vec(x_i) is:
```
vec(x_i,1:m) ~ N(0, σ_i² (D'D ⊗ I_d)^{-1})
```

**Initial position prior** (optional, default prior_std=10):
```python
K_init = np.zeros((m, m))
for s in range(r):
    K_init[s, s] = 1 / prior_std²
K_init = np.kron(K_init, np.eye(d))
```
This weakly anchors the first r positions.

### 2.3 Transition Variance Prior (Half-Cauchy)

The paper uses a **half-Cauchy** prior on σ_i, implemented via auxiliary variable:

```
σ_i² | ν_i ~ Inverse-Gamma(1/2, 1/ν_i)
ν_i ~ Inverse-Gamma(1/2, 1)
```

Marginalizing ν gives σ ~ Half-Cauchy(0, 1).

In code (lines 222-230):
```python
shape_sigma = 0.5 * ((n_time_points - self.rw_order) * self.n_features + 1)
scale_sigma = 0.5 * np.sum(np.diff(X, self.rw_order, axis=2) ** 2, axis=(1, 2)) + (1. / nu)
sigma = stats.invgamma.rvs(shape_sigma, scale=scale_sigma)
nu = stats.invgamma.rvs(1, 1 + (1. / sigma))
```

### 2.4 Scale Parameter λ

Default: `scale = 1 / var(y_vec)` where y_vec is the vectorized lower-triangular adjacencies.

Optionally sampled (lines 232-240):
```python
a = 1e-3 + 0.25 * n_nodes * (n_nodes + 1) * n_time_points
b = 1e-3 + 0.5 * np.sum((self.y_vec_ - XXt) ** 2) + 0.25 * np.sum(x ** 2)
scale = stats.gamma.rvs(a, scale=1./b)
```

---

## 3. Gibbs Sampling Algorithm (Complete Details)

### 3.1 Initialization (lines 129-143)

```python
# Per-timestep ASE using eigendecomposition
X = np.zeros((n_time_points, n_nodes, n_features))
for t in range(n_time_points):
    eigvals, eigvec = sp.linalg.eigsh(A[t], k=d)  # FULL matrix A
    X[t] = eigvec[:, ::-1] * np.sqrt(np.abs(eigvals)[::-1])

# Procrustes alignment to smooth
for t in range(1, n_time_points):
    R, _ = orthogonal_procrustes(X[t], X[t-1])
    X[t] = X[t] @ R

# Transpose to (n_nodes, n_features, n_time_points) for efficient updates
X = X.transpose((1, 2, 0))

# Initial σ from empirical variance of differences
sigma = np.mean(np.diff(X, axis=2) ** 2, axis=(1, 2))
nu = np.ones(n_nodes)
```

### 3.2 Running Statistics XtX (lines 145-150)

Precompute and maintain XtX[t] = Σ_i x_i(t) x_i(t)':
```python
XtX = []
for t in range(n_time_points):
    XtX[t] = np.zeros((n_features, n_features))
    for i in range(n_nodes):
        XtX[t] += X[i, :, t][:, None] @ X[i, :, t][:, None].T
```

### 3.3 Sample X | σ, λ, A (lines 166-219)

For each node i:

**Step 1**: Remove node i from XtX and compute XtY
```python
for t in range(n_time_points):
    # Remove node i's contribution
    XtX[t] -= X[i, :, t][:, None] @ X[i, :, t][:, None].T

    # Get neighbors of i at time t (sparse lookup!)
    indices = Y[t].indices[Y[t].indptr[i]:Y[t].indptr[i+1]]

    # Sum positions of neighbors (binary case)
    XtY[t] = np.sum(X[indices, :, t], axis=0)
```

**Step 2**: Build precision matrix (md × md, banded)
```python
# Prior: (1/σ_i) * K + K_init + (λ/2) * I
precision = (1. / sigma[i]) * K
precision.diagonal() += 0.5 * scale
precision += K_init

# Likelihood: λ * block_diag(XtX[0], XtX[1], ..., XtX[m-1])
P = precision + scale * block_diag(XtX)
```

**Step 3**: Banded Cholesky and solve for mean
```python
# Convert to banded format and decompose
L = cholesky_banded(P)

# Solve P @ X_hat = λ * XtY
X_hat = cho_solve_banded(L, scale * np.ravel(XtY))
```

**Step 4**: Sample from N(X_hat, P^{-1})
```python
# Sample z ~ N(0, P^{-1}) via L^{-T} @ randn
z = spsolve_triangular(L, randn(m*d), lower=False)

# New sample
X[i] = (X_hat + z).reshape(n_time_points, n_features).T
```

**Step 5**: Add node i back to XtX
```python
for t in range(n_time_points):
    XtX[t] += X[i, :, t][:, None] @ X[i, :, t][:, None].T
```

### 3.4 Sample σ | X (lines 222-230)

```python
shape_sigma = 0.5 * ((m - r) * d + 1)
scale_sigma = 0.5 * np.sum(np.diff(X, r, axis=2) ** 2, axis=(1, 2)) + 1./nu
sigma = invgamma.rvs(shape_sigma, scale=scale_sigma)

# Update auxiliary variable for half-Cauchy
nu = invgamma.rvs(1, scale=1 + 1./sigma)
```

### 3.5 Sample λ | X, A (optional, lines 232-240)

```python
XXt = np.einsum('tid,tjd->tij', x, x)[..., subdiag[0], subdiag[1]]  # lower tri only
a = 1e-3 + 0.25 * n * (n+1) * m
b = 1e-3 + 0.5 * np.sum((y_vec - XXt) ** 2) + 0.25 * np.sum(x ** 2)
scale = gamma.rvs(a, scale=1./b)
```

### 3.6 Post-Processing (lines 248-255)

After all sampling iterations:
```python
# Procrustes-smooth the last sample as reference
samples['X'][-1] = smooth_positions_procrustes(samples['X'][-1])

# Align ALL samples to this reference
for idx in range(n_samples):
    for t in range(n_time_points):
        R, _ = orthogonal_procrustes(samples['X'][idx][t], samples['X'][-1][t])
        samples['X'][idx][t] = samples['X'][idx][t] @ R

# Posterior mean
X_mean = samples['X'].mean(axis=0)
```

---

## 4. Key Implementation Details

### 4.1 Data Layout

- Input Y: list of sparse CSR matrices, or (m, n, n) dense array
- Internal X: (n_nodes, n_features, n_time_points) during sampling
- Output X: (n_time_points, n_nodes, n_features)

### 4.2 Sparse Neighbor Lookup

For CSR matrix Y[t]:
```python
# Row i's nonzero column indices (neighbors)
indices = Y[t].indices[Y[t].indptr[i]:Y[t].indptr[i+1]]
```

This is O(degree) not O(n).

### 4.3 Banded Matrix Structure

The precision P is banded with bandwidth 2*r*d due to Kronecker structure of K.
Using banded Cholesky: O(m*d*(r*d)²) vs O((m*d)³) for dense.

### 4.4 The 0.5*λ Regularization Term

Line 185: `precision.data[diag_loc] += 0.5 * scale`

This adds λ/2 to the diagonal. This comes from the x_i'x_i term in the likelihood expansion:
```
||A - XX'||² contains terms (x_i'x_j)² which expand to x_i'(x_j x_j')x_i
```
The diagonal contribution is λ/2 * I (accounting for symmetry).

---

## 5. Critical Evaluation Findings

### 5.1 Oracle vs Global Alignment

The reference implementation evaluates trajectory recovery using **per-timestep oracle alignment**:

```python
for t in range(n_time_points):
    R, _ = orthogonal_procrustes(X_est[t], X_true[t])  # Oracle!
    X_est[t] = X_est[t] @ R
```

This aligns each timestep **independently** to truth. This:
- **Destroys trajectory structure** (each timestep in different gauge)
- **Makes pointwise accuracy look good** but is meaningless for dynamics
- **Cannot be done in practice** (we don't have X_true)

For dynamics learning, we need **global alignment** (single R for all timesteps).

### 5.2 Binary Adjacency Noise Problem

With n=30 nodes, m=20 timesteps, density ~0.3:
- **Binary A noise: 107%** (SNR < 1)
- **Averaged A noise: 63%** (with K=50 samples)

When SNR < 1, **more iterations make P-error worse**:
- 500+1000 iterations: P-error = 77.6%
- 2500+2500 iterations: P-error = 92.1%

The random walk prior over-smooths, fitting noise rather than recovering P_true.

### 5.3 AUC vs Frobenius Error

The reference uses **AUC** (ranking metric), not Frobenius P-error:
- AUC: How well do predicted P_ij rank edges vs non-edges?
- Frobenius: How close is ||XX' - P_true||_F?

AUC is more forgiving - P̂ = 2×P_true has perfect AUC but 100% Frobenius error.
Their reported AUC ~0.9 doesn't contradict our 77-92% Frobenius error.

---

## 5. Experimental Results (January 2026)

### 5.1 Test Setup

**Synthetic data**:
- n = 30 nodes, d = 2 dimensions, m = 20 timesteps
- True positions: Random walk with step size 0.015, clamped to [0.1, 0.9]
- P_true(t) = X_true(t) @ X_true(t)', clamped to [0, 1]
- Binary adjacencies: A_ij ~ Bernoulli(P_ij), symmetrized

**Noise levels**:
- Binary A noise: ||A - P_true|| / ||P_true|| = **107%** (SNR < 1)
- Averaged A noise (K=50): **63%**

### 5.2 Frobenius P-Error Results

P-error = √(Σ_t ||X_est(t) X_est(t)' - P_true(t)||²) / √(Σ_t ||P_true(t)||²)

| Method | Data | Iterations | P-error |
|--------|------|------------|---------|
| GB-DASE Gibbs | Binary A | 500+1000 | 77.6% |
| GB-DASE Gibbs | Binary A | 2500+2500 | 92.1% (worse!) |
| GB-DASE Gibbs | Averaged A (K=50) | 300+500 | 67.6% |
| DUASE | Binary A | - | 65.5% |
| SVD+Procrustes | Binary A | - | 78.0% |

**Key finding**: More iterations made P-error WORSE (77.6% → 92.1%) due to over-smoothing.
The random walk prior dominates when SNR < 1.

### 5.3 AUC Results (Reference Metric)

AUC measures ranking quality: can predicted P_ij distinguish edges from non-edges?

| Method | AUC | vs Oracle |
|--------|-----|-----------|
| P_true (oracle) | 0.7192 | - |
| GB-DASE Gibbs | 0.7194 | -0.03% (matches!) |
| DUASE | 0.7180 | 0.16% below |
| SVD+Procrustes | 0.5735 | 20.25% below |

**Key finding**: GB-DASE and DUASE both achieve near-optimal AUC!
The "oracle" AUC is only 0.72 because even P_true can't perfectly predict noisy binary A.

### 5.4 Why Frobenius and AUC Disagree

- **AUC**: Only cares about ranking. P̂ = 0.5 × P_true has perfect AUC.
- **Frobenius**: Cares about absolute values. P̂ = 0.5 × P_true has 50% error.

GB-DASE correctly ranks edge probabilities but gets the **scale wrong**.
This is a calibration issue, not a fundamental failure.

### 5.5 Trajectory Recovery Results

**Velocity magnitude** (smoothness measure):
| Method | Velocity | True = 0.0184 |
|--------|----------|---------------|
| GB-DASE | 0.018 | Matches ✓ |
| DUASE | 0.060 | 3x too fast |
| SVD+Procrustes | 0.458 | 25x too fast |

**Pointwise error** (after global alignment):
| Method | Global Align | Oracle Per-Timestep |
|--------|--------------|---------------------|
| GB-DASE | 0.344 | 0.339 |
| DUASE | 0.285 | 0.282 |
| SVD+Procrustes | 0.464 | 0.432 |

GB-DASE produces smooth trajectories but in wrong region of space.
DUASE has better pointwise accuracy.

### 5.6 Oracle Evaluation in Reference

The reference evaluates trajectory recovery using **per-timestep oracle alignment**:

```python
for t in range(n_time_points):
    R, _ = orthogonal_procrustes(X_est[t], X_true[t])  # Uses TRUE X!
    X_est[t] = X_est[t] @ R
```

This aligns each timestep independently to truth:
- Uses oracle knowledge (X_true) unavailable in practice
- For GB-DASE, global vs oracle alignment gives similar results (~2% difference)
- This suggests GB-DASE maintains internally consistent gauge (due to RW prior)

**Important**: Oracle alignment is only for visualization, NOT used during sampling or forecasting.

### 5.7 Implications for Dynamics Learning

For gauge-equivariant dynamics Ẋ = N(P)X, we need:
1. **Correct P values** - GB-DASE has wrong scale (77% Frobenius error)
2. **Consistent trajectories** - GB-DASE has correct smoothness but wrong shape

**Conclusions**:
- GB-DASE achieves optimal AUC (ranking) but poor Frobenius (absolute values)
- The scale calibration issue may be addressable with post-processing
- For dynamics learning, DUASE may be preferable (better Frobenius, worse smoothness)
- Neither method reliably recovers true trajectory shapes with SNR < 1

---

## 6. Why Node-by-Node Gibbs Works

The joint posterior over all X is NOT Gaussian (XX' is nonlinear).

But conditional on all other nodes, the posterior for node i IS Gaussian:
- The term x_i'x_j is linear in x_i when x_j is fixed
- So (A_ij - x_i'x_j)² is quadratic in x_i
- Combined with Gaussian prior → Gaussian conditional

This enables exact sampling without Metropolis-Hastings.

---

## 7. Forecasting (lines 310-356)

For r=2 (second-order random walk), forecasting uses:
```python
# x(m+1) = 2*x(m) - x(m-1) + σ*ε
samples[0] = 2 * X[-1] - X[-2] + sigma * randn()

# x(m+2) = 2*x(m+1) - x(m) + σ*ε
samples[1] = 2 * samples[0] - X[-1] + sigma * randn()

# Continue...
for h in range(2, k_steps):
    samples[h] = 2 * samples[h-1] - samples[h-2] + sigma * randn()
```

This extrapolates the trajectory with the same random walk dynamics used in the prior.

---

## 8. Summary and Recommendations

### What GB-DASE Does Well
1. **Optimal AUC** - Correctly ranks edge probabilities (matches oracle)
2. **Smooth trajectories** - Velocity magnitude matches true dynamics
3. **Internally consistent gauge** - Random walk prior maintains temporal coherence
4. **Uncertainty quantification** - Provides posterior samples, not just point estimates

### What GB-DASE Does Poorly
1. **P-value calibration** - 77% Frobenius error (wrong scale)
2. **Trajectory shape recovery** - Embeddings in different region of space
3. **Low SNR performance** - Over-smooths when noise > signal
4. **Computational cost** - Node-by-node Gibbs is slow for large networks

### AUC is a Very Generous Metric
The reference uses AUC which only measures ranking ability:
- P̂ = 0.5 × P_true has **perfect AUC** but **50% Frobenius error**
- P̂ = 2 × P_true has **perfect AUC** but **100% Frobenius error**

For link prediction (their use case), AUC is appropriate.
For dynamics learning (our use case), we need actual P values → Frobenius matters.

### Recommendations for Dynamics Learning

1. **Use DUASE over GB-DASE** when P-accuracy matters (65% vs 77% Frobenius)
2. **Average multiple edge observations** if possible (reduces noise from 107% to 63%)
3. **Don't over-iterate GB-DASE** - more iterations can increase error
4. **Consider scale recalibration** - GB-DASE rankings are good, just miscalibrated
5. **Focus on gauge-equivariant dynamics** - these are robust to embedding gauge issues

### Open Questions
1. Can GB-DASE scale be recalibrated post-hoc to improve Frobenius error?
2. How do these methods perform on denser networks (higher SNR)?
3. Is there a hybrid approach combining DUASE stability with GB-DASE smoothness?
