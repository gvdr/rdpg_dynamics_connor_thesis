# Example 1: Parsimonious UDE Architectures

## Overview

This example demonstrates that parsimonious physics-informed architectures (2-3 parameters)
significantly outperform black-box neural networks (7832 parameters) for learning RDPG dynamics.

## Experimental Setup

- **Network size**: N = 60 nodes, d = 2 dimensions
- **True dynamics**: Message-passing with repulsion
  - Ẋᵢ = α₀·Xᵢ + α₁·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ)
  - α₀ = -0.01 (decay toward origin)
  - α₁ = -0.002 (neighbor repulsion)
- **Time horizon**: T = 21 timesteps (14 training, 7 validation)
- **RDPG estimation**: K = 40 averaged adjacency samples

## Models Compared

| Model | Parameters | Form |
|-------|------------|------|
| MsgPass | 2 | Ẋᵢ = β₀·Xᵢ + β₁·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ) |
| Poly P² | 3 | Ẋ = (β₀I + β₁P + β₂P²)X |
| Pure NN | 7832 | Ẋ = NN(X) |

## Results

### Prediction Error (vs True Trajectories)

| Model | Training | Validation |
|-------|----------|------------|
| MsgPass (2p) | 0.2 | 0.48 |
| Poly P² (3p) | 0.17 | 0.56 |
| Pure NN | 5.31 | 8.24 |

**Key finding**: Parsimonious models (MsgPass, Poly) vastly outperform Pure NN on extrapolation.

### Parameter Recovery (Message-Passing)

| Coefficient | True | Recovered | Error |
|-------------|------|-----------|-------|
| β₀ (self) | -0.01 | -0.0175 | 74.6% |
| β₁ (neighbor) | -0.002 | -0.0026 | 29.7% |

The correct signs are recovered, demonstrating that the model structure enables interpretable parameter estimation.

## Figures

1. **main_comparison.pdf**: 2×2 panel showing true dynamics, model predictions, and learning curves
2. **coefficient_recovery.pdf**: Bar chart comparing true vs recovered parameters
3. **trajectory_comparison.pdf**: True vs DUASE-estimated vs MsgPass-recovered trajectories
4. **P_heatmaps.pdf**: Probability matrix P(t) = XX' heatmaps comparing True vs MsgPass vs Poly at t=0, train end, and extrapolation. Note: Shows dynamics models (not DUASE) since only dynamics can genuinely extrapolate.
5. **metrics_over_time.pdf**: P-error and distance correlation over time. DUASE (dashed) shows estimates from observed data; dynamics models (solid) show predictions from initial conditions only.

## Key Insights

### 1. Gauge Equivariance: Learn Anywhere, Apply Everywhere

**Critical finding**: We should NOT force estimated embeddings back to the canonical B^d_+ space. The B^d_+ projection distorts the geometry and breaks temporal consistency.

Instead, we learn dynamics in whatever coordinate system DUASE naturally produces:
- DUASE provides embeddings that are **temporally consistent** via shared basis G
- These embeddings are related to true positions by an unknown orthogonal transformation Q
- Since our dynamics have the form Ẋ = N(P)X where P = XX', they are **gauge-equivariant**

**Key consequence**: The learned parameters (β₀, β₁) are scalars that don't depend on the coordinate system. We can:
1. Learn in DUASE space (from noisy spectral estimates)
2. Apply the learned parameters to TRUE initial conditions
3. Get trajectories directly comparable to ground truth - NO Procrustes alignment needed!

This is gauge-equivariance in action: learn the physics in one frame, apply it in any frame.

### 2. Parsimonious Beats Black-Box

The success of the 2-parameter MsgPass model over the 7832-parameter Pure NN demonstrates that incorporating correct physics (gauge equivariance via N(P)X form) provides massive inductive bias that compensates for limited data and noisy observations.

### 3. P(t) is the True Test

Since P = XX' is rotation-invariant, comparing P matrices is the honest test of dynamics recovery, not comparing X positions (which are gauge-dependent).

## Interpretation

The message-passing dynamics model repulsive interactions where nodes push away from highly-connected neighbors while decaying toward the origin. This creates stable, non-collapsing dynamics that maintain the eigenvalue structure needed for accurate RDPG embedding.
