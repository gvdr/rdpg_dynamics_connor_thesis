# RDPG Estimation Methods for Temporal Networks

This document summarizes spectral embedding methods for estimating latent positions from temporal network data, based on the literature and our experiments.

## Problem Setting

We observe a sequence of adjacency matrices A(1), A(2), ..., A(T) where each A(t) is generated from an RDPG:

```
A(t) ~ Bernoulli(P(t))  where  P(t) = X(t) X(t)'
```

**Goal**: Recover the latent positions X(t) from the observed adjacency matrices.

**Challenges**:
1. Each A(t) is a noisy binary sample from P(t)
2. SVD identifies X only up to orthogonal rotation
3. We need temporal alignment across time points

---

## Method 1: Simple Per-Time SVD

**Algorithm**:
1. For each time t, compute SVD of A(t): `A(t) = U Σ V'`
2. Extract embedding: `X̂(t) = U[:, 1:d] · √Σ[1:d]`
3. Align temporally via Procrustes chain

**Pros**: Simple, direct
**Cons**: High variance, no information sharing across time points

---

## Method 2: UASE (Unfolded Adjacency Spectral Embedding)

**Reference**: Gallagher et al. (2021), NeurIPS

**Algorithm**:
1. **Unfold**: Stack horizontally `A = [A(1) | A(2) | ... | A(T)]` (size n × nT)
2. **SVD**: `A = U Σ V'`
3. **Extract**: Partition V into T blocks of n rows each
   - `V(t) = V[(t-1)n+1 : tn, :]`
4. **Embedding**: `X̂(t) = V(t) · √Σ`

**Key insight**: The RIGHT singular vectors V (partitioned by time) give per-time embeddings.

**Pros**: Guaranteed stability properties (cross-sectional and longitudinal)
**Cons**: In our experiments, performed worse than OMNI and DUASE for RDPG recovery

---

## Method 3: OMNI (Omnibus Embedding)

**Reference**: Levin et al. (2017), IEEE ICDMW

**Algorithm**:
1. **Construct omnibus matrix** M of size (nT × nT):
   ```
   M = [ A(1)           (A(1)+A(2))/2   ...  (A(1)+A(T))/2  ]
       [ (A(2)+A(1))/2  A(2)            ...  (A(2)+A(T))/2  ]
       [ ...            ...             ...  ...            ]
       [ (A(T)+A(1))/2  (A(T)+A(2))/2   ...  A(T)           ]
   ```
   - Diagonal blocks: `M[ii] = A(i)`
   - Off-diagonal blocks: `M[ij] = (A(i) + A(j)) / 2`

2. **SVD**: `M = U Σ V'`

3. **Extract per-time embeddings** from U (rows):
   - `X̂(t) = U[(t-1)n+1 : tn, 1:d] · √Σ[1:d]`

**Pros**: Joint embedding ensures alignment, good empirical performance
**Cons**: O((nT)²) memory for omnibus matrix

---

## Method 4: DUASE (Doubly Unfolded ASE) for Single-Layer Temporal Networks

**Reference**: Baum, Sanna Passino & Gandy (2024), arXiv:2410.09810

For single-layer temporal networks, DUASE simplifies to:

**Model**: `A(t) ≈ G Q(t) G'` where:
- G is shared node embedding (n × d)
- Q(t) is time-specific score matrix (d × d)

**Algorithm**:
1. **Unfold**: `A = [A(1) | A(2) | ... | A(T)]` (size n × nT)
2. **SVD**: `A = U Σ V'`
3. **Extract shared basis**: `G = U[:, 1:d]` (NOT scaled by √Σ)
4. **For each time t**:
   - Project: `Q(t) = G' A(t) G`
   - Eigendecompose: `Q(t) = V_q Λ_q V_q'`
   - Embedding: `X̂(t) = G · V_q · √Λ_q · V_q'`

   Or equivalently: `X̂(t) = G · √Q(t)` where √Q(t) is the matrix square root.

**Pros**: Best empirical performance in our tests, separates shared structure from temporal variation
**Cons**: Requires eigendecomposition per time point

---

## Method 5: MASE (Multiple Adjacency Spectral Embedding)

**Reference**: Arroyo et al. (2021)

**Model**: `A(i) ≈ V R(i) V'` where:
- V is shared latent positions (n × d)
- R(i) is graph-specific score matrix (d × d)

**Algorithm**:
1. Embed each graph individually: `X(i) = ASE(A(i))`
2. Concatenate: `[X(1) | X(2) | ... | X(M)]`
3. SVD to get shared `V̂`
4. Score matrices: `R̂(i) = V̂' A(i) V̂`

**Note**: Similar in spirit to DUASE but different construction.

---

## Experimental Comparison

Test setup: n=16 nodes, T=5 time points, K=30 samples per time, d=2 dimensions

| Method | Mean Error (per-node) |
|--------|----------------------|
| Simple SVD | 0.162 |
| UASE | 0.245 |
| OMNI | 0.107 |
| DUASE | **0.107** |

**Conclusion**: OMNI and DUASE perform best for RDPG recovery from temporal networks.

---

## Combining Repeated Sampling with Temporal Methods

When we have K repeated samples at each time point, how should we combine them with methods like DUASE?

### Two Approaches

**Approach A: Average First, Then DUASE**
```
1. At each time t: A_avg(t) = (1/K) Σ_k A_k(t)
2. Unfold: [A_avg(1) | ... | A_avg(T)]
3. SVD → G, then Q(t) = G' A_avg(t) G
```

**Approach B: Unfold All K·T Samples**
```
1. Unfold everything: [A_1(1)|...|A_K(1)|A_1(2)|...|A_K(T)]  (n × n·K·T)
2. SVD → G from all samples
3. Q(t) = G' A_avg(t) G (still use average for Q)
```

### Experimental Results (n=16, T=5, 10 trials)

| K | Avg-first | Unfold-all | Better |
|---|-----------|------------|--------|
| 5 | 0.181 | **0.170** | Unfold-all |
| 10 | **0.152** | 0.158 | ~Tie |
| 20 | **0.146** | 0.149 | ~Tie |
| 50 | **0.099** | 0.139 | Avg-first |

### Recommendations

- **K < 10**: Consider unfolding all samples (more data for G estimation helps)
- **K ≥ 20**: Average first (cleaner signal beats more data)
- **K ≈ 10-20**: Either approach works similarly

**Note**: Unfold-all has lower variance (more stable), but Avg-first has better asymptotic performance as K → ∞.

### Why Averaging Wins for Large K

In the DUASE model `A(t) ≈ G Q(t) G'`:
- All K samples at time t share the same Q(t)
- Averaging gives A_avg(t) → P(t) = X(t)X(t)' as K → ∞
- Clean signal for G estimation beats noisy K-fold data

---

## Windowed DUASE

Instead of unfolding all T time points, we can use a sliding window of size W centered at each time t.

### Algorithm: Windowed DUASE

For each time t:
1. Define window: `[max(1, t-W÷2), min(T, t+W÷2)]`
2. Unfold only matrices in window: `A_window = [A_avg(t-W÷2) | ... | A_avg(t+W÷2)]`
3. SVD → `G_t` (window-specific shared basis)
4. `Q(t) = G_t' A_avg(t) G_t`
5. `X̂(t) = G_t · √Q(t)`

### Experimental Results (n=16, d=2, K=30)

**T = 10 time points:**
| Window | 3 | 5 | 7 | Full (10) |
|--------|---|---|---|-----------|
| Error | 0.090 | 0.081 | 0.071 | 0.073 |

**T = 20 time points:**
| Window | 3 | 5 | 7 | 11 | Full (20) |
|--------|---|---|---|----|----|
| Error | 0.094 | 0.084 | 0.078 | 0.068 | 0.067 |

**T = 40 time points:**
| Window | 3 | 5 | 7 | 11 | 21 | Full (40) |
|--------|---|---|---|----|----|-----------|
| Error | 0.106 | 0.095 | 0.091 | 0.078 | 0.071 | 0.068 |

### Recommendations

- **Generally**: Larger windows perform better (more data for shared basis estimation)
- **Plateau effect**: Beyond ~50% of T, diminishing returns
- **For stationary dynamics**: Use full window (all T)
- **For non-stationary dynamics**: Moderate window (T/2 to T/4) may be better if dynamics change significantly over time

### When Windowing Might Help

Windowing is useful when:
1. **Dynamics change character over time**: The shared subspace G itself evolves
2. **Memory constraints**: Full unfolding is O(n × nT), windowing reduces to O(n × nW)
3. **Streaming data**: Can update embeddings incrementally

For our RDPG dynamics experiments with smooth dynamics, full unfolding or large windows work best.

---

## Implementation Notes

### Self-Loops (Critical!)

When sampling adjacency matrices from P = XX', you MUST include diagonal entries:

```julia
A[i,j] ~ Bernoulli(P[i,j])  for ALL i,j including i=j
```

The diagonal P[ii] = ||X[i]||² contains essential norm information. Without it, the averaged adjacency does NOT converge to P, and SVD fails to recover X.

### Repeated Sampling

Generate K independent samples at each time point and average:
```
A_avg(t) = (1/K) Σ_k A_k(t)
```

As K → ∞, A_avg(t) → P(t). Recommended: K = 10-50.

### Temporal Windowing

For smoother embeddings, use a sliding window W:
```
A_smooth(t) = mean(A_avg(s) for s in [t-W/2, t+W/2])
```

Recommended: W = 3-5.

### Procrustes Alignment

SVD embeddings are identified only up to orthogonal rotation. Use Procrustes to align:

```julia
function procrustes(A, B)
    U, _, V = svd(A' * B)
    Q = U * V'
    return A * Q  # A rotated to align with B
end
```

### B^d_+ Alignment for Temporal Series

For valid RDPG probabilities, latent positions should be in B^d_+ (non-negative unit ball).
After embedding, we need to find ONE common orthogonal Q for all time points.

**Algorithm**: Sign flips + Rotation

```
1. Stack all matrices: L_all = vcat(L(1), ..., L(T))
2. Sign flips (S): Flip column j if >50% entries are negative
3. Rotation (R): Minimize violation = sum(|negatives|) + sum(excess norms)
   - d=2: Grid search over angle θ
   - d=3,4: Joint optimization over all d(d-1)/2 angles
   - d>4: Iterative Givens with multiple passes
4. Apply Q = S·R to all matrices
5. Handle remaining violations:
   - :project  → per-point clamping (may distort geometry)
   - :rescale  → global scaling (preserves relative distances)
```

**Usage**:
```julia
result = align_series_to_Bd_plus(L_series; method=:rescale)
L_aligned = result.L_aligned  # All in B^d_+
```

**Recommendation**: Use `:rescale` for dynamics learning (preserves geometry).

---

## Fundamental Limitation: Trajectory Recovery

### What RDPG Can and Cannot Recover

| Quantity | Recovery | Typical Error |
|----------|----------|---------------|
| P = XX' (probabilities) | ✓ Good | 5-8% |
| Pairwise distances | ✓ Good | 0.97 correlation |
| Community structure | ✓ Good | Qualitative |
| **Absolute positions** | ✗ Poor | 55% relative error |
| **Velocities (dx/dt)** | ✗ Poor | 0.05 correlation |

### Why Trajectories Fail

The orthogonal rotation Q that maps X̂(t) → X(t) **drifts over time** (up to 90° over 30 timesteps). This is not an alignment bug—it's fundamental to RDPG identifiability.

Even with **oracle per-timestep Procrustes alignment**, position error remains ~35%.

### Implications for Dynamics Learning

Training Neural ODE on recovered positions X̂(t) is problematic because velocity information is lost.

**Two strategies for rotation-invariant dynamics:**

| Strategy | State Space | Loss Function |
|----------|-------------|---------------|
| **1. Direct distance dynamics** | D_ij (n(n-1)/2 dim) | ||D_pred - D_true||² |
| **2. Position dynamics + distance loss** | X (n·d dim) | ||D(X_pred) - D(X_true)||² |

**Strategy 2 recommended**: Lower dimensional, more interpretable, standard Neural ODE tools apply.

See `Julia_implementation_notes.md` for detailed analysis, implementation code, and recommendations.

---

## References

1. Gallagher, I., Jones, A., & Sherlock, C. (2021). Spectral embedding for dynamic networks with stability guarantees. NeurIPS.

2. Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017). A central limit theorem for an omnibus embedding of multiple random dot product graphs. IEEE ICDMW.

3. Baum, M., Sanna Passino, F., & Gandy, A. (2024). Doubly unfolded adjacency spectral embedding of dynamic multiplex graphs. arXiv:2410.09810.

4. Arroyo, J., Athreya, A., Cape, J., Chen, G., Priebe, C. E., & Vogelstein, J. T. (2021). Inference for multiple heterogeneous networks with a common invariant subspace. JMLR.

5. graspologic documentation: https://graspologic.readthedocs.io/

---

*Last updated: 2026-01-15*
