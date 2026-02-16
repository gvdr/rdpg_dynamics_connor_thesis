# UDE Pipeline Experiment — Detailed Results

## Experimental Setup

- **n** = 200 nodes, **d** = 3 latent dimensions
- **Communities**: K = 3, balanced (~67 nodes each), centroids at (0.7,0.2,0.2), (0.2,0.7,0.2), (0.2,0.2,0.7)
- **Anchors**: n_a = 100 (frozen), **Mobile**: 100 nodes
- **Dynamics**: damped spiral, non-gauge-equivariant
  - True ODE: Xdot_i = (-gamma + beta ||x_i - mu_k||^2)(x_i - mu_k) + omega J(x_i - mu_k)
  - J = (1/sqrt(3))[0 -1 1; 1 0 -1; -1 1 0] (rotation around (1,1,1)/sqrt(3))
- **True parameters**: gamma = 0.3, beta = -0.5, omega = 1.0
- **Bernoulli samples per frame**: K = 10
- **Time steps**: T = 50, dt = 0.1 (total time = 4.9)
- **Monte Carlo repetitions**: 5
- **Base seed**: 2025 (rep i uses seed 2025 + i)

### UDE Architecture

- **Known part**: f_k(delta) = -gamma * delta (learnable scalar gamma, initialized at 0.1)
- **Unknown part**: NN(delta; theta), architecture: Dense(3,16,tanh) -> Dense(16,16,tanh) -> Dense(16,3)
- **Total parameters**: ~388 (1 scalar gamma + 387 NN weights)
- **Optimizer**: Adam(0.01), 500 epochs
- **L2 regularization**: lambda = 1e-3 on NN weights (discourages NN from absorbing the linear damping term)
- **ODE solver**: Tsit5, sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())

### Alignment Conditions

1. **Anchor-aligned**: each ASE frame aligned to reference (t=1) via anchor-node Procrustes, then globally aligned to true X frame
2. **Sequential Procrustes**: each frame aligned to previous via all-node Procrustes, then globally aligned to true X frame
3. **Unaligned**: raw ASE with only a single global Procrustes at t=1; subsequent frames remain in arbitrary rotations

---

## Per-Rep Results

### Recovered gamma (true = 0.3)

| Rep | Anchor | Sequential | Unaligned |
|-----|--------|------------|-----------|
| 1   | 0.3976 | 0.2047     | 0.1698    |
| 2   | 0.2046 | 0.2297     | 0.2402    |
| 3   | 0.2616 | 0.2543     | 0.2447    |
| 4   | 0.2135 | 0.1966     | 0.2357    |
| 5   | 0.1980 | 0.1947     | 0.2059    |
| **Mean +/- std** | **0.255 +/- 0.081** | **0.216 +/- 0.026** | **0.219 +/- 0.031** |

### NN Residual MSE vs True f_u

| Rep | Anchor | Sequential | Unaligned |
|-----|--------|------------|-----------|
| 1   | 0.000307 | 0.006892 | 0.228927 |
| 2   | 0.000856 | 0.007355 | 0.891297 |
| 3   | 0.000459 | 0.007740 | 0.246158 |
| 4   | 0.000712 | 0.007534 | 0.765747 |
| 5   | 0.001913 | 0.007004 | 0.076130 |
| **Mean +/- std** | **0.0009 +/- 0.0006** | **0.0073 +/- 0.0003** | **0.442 +/- 0.353** |

### Total Dynamics MSE (primary metric)

This compares the complete learned dynamics f_learned(delta) = -gamma_hat * delta + NN(delta) against the true dynamics f_true(delta) = -gamma * delta + beta r^2 delta + omega J delta, evaluated on 2000 random delta samples. It sidesteps the gamma identifiability issue because the total dynamics can be correct even if the gamma/NN split is wrong.

| Rep | Anchor | Sequential | Unaligned |
|-----|--------|------------|-----------|
| 1   | 0.000333 | 0.007339 | 0.227742 |
| 2   | 0.000573 | 0.007740 | 0.890041 |
| 3   | 0.000395 | 0.008005 | 0.245828 |
| 4   | 0.000468 | 0.007853 | 0.765257 |
| 5   | 0.001281 | 0.007328 | 0.075895 |
| **Mean +/- std** | **0.0006 +/- 0.0004** | **0.0076 +/- 0.0003** | **0.44 +/- 0.35** |

---

## Key Observations

### 1. Total Dynamics Recovery (the main story)

The total dynamics MSE is the definitive metric, as it measures how well the complete learned ODE matches the true ODE regardless of how the gamma/NN split is apportioned:
- **Anchor-aligned**: 0.0006 +/- 0.0004 — excellent dynamics recovery
- **Sequential Procrustes**: 0.0076 +/- 0.0003 — ~13x worse
- **Unaligned**: 0.44 +/- 0.35 — ~700x worse, highly variable

This confirms that anchor-based alignment is essential for non-gauge-equivariant dynamics recovery.

### 2. NN Residual Recovery

The NN residual MSE (comparing only the unknown part NN(delta) vs true f_u(delta)) shows an even cleaner separation:
- **Anchor**: ~0.001 — NN accurately learns the nonlinear + rotation residual
- **Sequential**: ~0.007 — partial recovery, systematic errors from gauge drift
- **Unaligned**: ~0.44 — NN learns noise, not structure

### 3. Gamma Recovery (identifiability caveat)

With L2 regularization (lambda=1e-3), gamma recovery improved compared to the unregularized run (previous mean ~0.19):
- Anchor mean gamma_hat = 0.255 (closer to true 0.3, though still underestimated on average)
- Rep 1 anchor achieved gamma_hat = 0.398 (overshooting slightly)
- Sequential and unaligned remain at ~0.22

The remaining bias reflects the fundamental identifiability issue in additive UDE decompositions: the NN can still absorb part of the linear term. However, the total dynamics MSE shows this doesn't matter for the complete dynamics — the anchor condition correctly recovers the total dynamics function regardless of the gamma/NN split.

### 4. Symbolic Regression

The anchor condition produces Pareto-optimal expressions with losses 1-2 orders of magnitude below sequential and unaligned at each complexity level (visible in panel (c) of the figure). The discovered expressions for the anchor condition contain the expected structural terms (products of coordinates, cross-terms consistent with the rotation matrix J).

### 5. Effect of L2 Regularization

Comparing with the previous unregularized run:
- Anchor gamma improved from ~0.196 to ~0.255 (closer to true 0.3)
- The regularization penalty pushes NN weights toward zero, forcing the explicit gamma parameter to capture more of the linear damping
- Total dynamics MSE was not previously computed, but NN residual MSE improved from ~0.021 to ~0.0009 (23x improvement for anchor condition)

---

## Data Validity

All true trajectories remain within B+^3:
- P = XX' in [0, 1] for all entries at all timesteps across all reps.

---

## File Locations

- **Data**: `data/ude_experiment/rep{1-5}.jls`
- **Aggregated results**: `results/ude_experiment/ude_results.jls`
- **Figure**: `paper/plots/ude-pipeline-results.png`
- **Scripts**: `scripts/ude_experiment/{generate_spiral_data,run_spiral_ude,plot_spiral_results}.jl`
