# UDE Examples Plan for Paper

## Overview

This document outlines the synthetic examples demonstrating the Neural ODE + UDE + Symbolic Regression pipeline for learning RDPG dynamics.

**⚠️ Key Finding from Example 1 (January 2026):**
The B^d_+ constraint (non-negative unit ball) is **NOT required** for numerical learning.
- Soft barrier losses maintain valid probabilities without geometric distortion
- Projection onto B^d_+ can actually impede learning
- B^d_+ remains useful for mathematical analysis, not for numerics

**Gauge-Equivariant Architecture**: Use Ẋ = N(P)X with symmetric N:
- Automatically gauge-consistent (symmetric N cannot produce invisible dynamics)
- Scalar parameters are gauge-invariant when expressed in terms of P
- **"Learn anywhere, apply everywhere"**: Parameters learned in DUASE coordinates transfer to true coordinates

**Primary Evaluation Metric**: P(t) = X(t)X(t)' (gauge-invariant), not X positions.

---

## Pipeline for Each Example

```
Generate True Data
    → Estimate via DUASE (introduces ~35% position error, ~5% P error)
    → Train N(P)X dynamics on DUASE estimates
    → Apply learned parameters to TRUE initial conditions
    → Evaluate in P-space (compare P_pred to P_true)
```

**Outputs per example:**
- Loss curves (training progress)
- P(t) heatmaps: True vs Estimated vs Predicted
- Trajectory plots (true vs predicted, noting gauge ambiguity)
- Symbolic equations discovered (gauge-invariant scalars)
- Error metrics: P-error (primary), X-error (secondary, gauge-dependent)

---

## Example 1: Parsimonious UDE Showcase (COMPLETED ✅)

**Story**: Demonstrate that 2-3 parameter gauge-equivariant architectures can learn RDPG dynamics as well as or better than black-box NNs with thousands of parameters.

**Status**: ✅ COMPLETED (January 2026) - validates core methodology

**True Dynamics**: Message-passing repulsion
```julia
Ẋᵢ = α₀·Xᵢ + α₁·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ)
# α₀ = -0.01 (decay), α₁ = -0.002 (repulsion)
```
Equivalent to N(P)X with N = α₀I + α₁(P - D) where D = diag(P·1).

**Architectures tested**:
1. **Polynomial (P²)**: Ẋ = (β₀I + β₁P + β₂P²)X [3 params]
2. **Message-Passing**: Ẋᵢ = a·Xᵢ + m·Σⱼ Pᵢⱼ(Xⱼ - Xᵢ) [2 params]
3. **Pure NN baseline**: [~thousands params]

**Key findings**:
1. **B^d_+ projection is unnecessary** - soft barrier loss works better
2. **"Learn anywhere, apply everywhere"** - train on DUASE, apply to TRUE
3. **P(t) is the correct evaluation metric** - gauge-invariant
4. **Scalar parameters are gauge-invariant** when expressed in terms of P
5. **DUASE vs Dynamics distinction** - DUASE estimates, dynamics extrapolates
6. **Implicit misspecification robustness** - Polynomial N(P)X captured message-passing dynamics well, even though they're structurally different (polynomial has no degree matrix subtraction). This suggests N(P)X architectures are flexible enough to approximate related dynamics even when not exactly matched.

**Configuration**: n=60 nodes, d=2, T=21 timesteps, 70% train / 30% extrapolation

**Script**: `scripts/example1_bridge_node_v2.jl`

**NOTE**: Original "bridge node oscillation" plan was replaced with this more fundamental demonstration of the gauge-equivariant methodology.

---

## Example 2: Homogeneous Circulation in B^d_+

**Story**: All nodes follow same physics - demonstrates data pooling efficiency.

**Challenge**: Pure rotation leaves B^d_+. Need B^d_+-compatible dynamics.

**Dynamics** (autonomous, stays in B^d_+):
```julia
# Circulation with soft boundary confinement
# In Cartesian coordinates:

v_circulation = ω * [-x₂, x₁]           # Tangent to circles (KNOWN)

# Soft repulsion from boundaries (UNKNOWN):
repel_x1 = α * exp(-x₁/ε)               # Repel from x₁=0 edge
repel_x2 = α * exp(-x₂/ε)               # Repel from x₂=0 edge
repel_arc = α * exp((r-1)/ε)            # Repel from unit circle arc

# Radial equilibrium (UNKNOWN):
radial_attraction = -k * (r - r_target) * [x₁, x₂] / r

dx/dt = v_circulation + [repel_x1, 0] + [0, repel_x2]
        - (x/r)*repel_arc + radial_attraction
```

**Why autonomous?** The vector field depends only on position x, not time t.
This allows static visualization of the learned dynamics as arrow plots.

**Known part**: Circulation structure `ω * [-x₂, x₁]`
**Unknown part**: Boundary confinement + radial equilibrium

**Key demonstrations**:
- Data efficiency: 1 NN learns from ALL node trajectories (pooled)
- Visualize learned vector field vs true vector field
- Symbolic regression attempts to recover confinement terms

**Dimensions**: d=2, clean data

---

## Example 3a: Wrong Known Part (Clean Data)

**Story**: What if our physics assumption is wrong? Does encoding incorrect
structure help or hurt?

**Hypothesis**: UDE might be *worse* than Full NN. The rigid incorrect structure
constrains the NN to learn corrections from a wrong baseline. Full NN has
freedom to find the right solution without that handicap.

**Setup**:
- True dynamics: Circulation with ω_true
- Assumed known (UDE): Circulation with ω_wrong ≠ ω_true
- Full NN: No assumptions, learns everything

**Misspecification levels**:
- Mild: ω_wrong = 1.2 * ω_true (20% off)
- Severe: ω_wrong = 2.0 * ω_true (100% off)

**Key demonstrations**:
- Compare UDE vs Full NN at different misspecification levels
- Show when rigidity helps vs hurts
- Quantify: at what error level does Full NN overtake UDE?

**Dimensions**: d=2, clean data

---

## Example 3b: Correct Known Part (Noisy Data)

**Story**: Real data has measurement noise. Does encoding correct structure
help with regularization?

**Hypothesis**: UDE might be *better* than Full NN. The correct known structure
acts as regularization, preventing overfitting to noise. Full NN might chase
the noise without the structural constraint.

**Setup**:
- True dynamics: Same circulation as Example 2
- Known part (UDE): Correct circulation with ω_true
- Full NN: No assumptions
- Noise: Gaussian noise on observed trajectories

**Noise levels**:
- Low: σ = 0.01 (1% of typical magnitude)
- Medium: σ = 0.05 (5%)
- High: σ = 0.10 (10%)

**Key demonstrations**:
- Compare UDE vs Full NN at different noise levels
- Show regularization benefit of correct structure
- Robustness of symbolic regression under noise

**Dimensions**: d=2, noisy data

**Contrast with 3a**: Together, Examples 3a and 3b show:
- Wrong structure + clean data → Full NN might win
- Correct structure + noisy data → UDE might win

---

## Example 4: Heterogeneous Dynamics with Type-Specific Kernels + Symbolic Regression

**Story**: Different node types follow different interaction rules. Can we:
1. Use domain knowledge (types, self-rates) to structure the UDE
2. Let an NN learn the unknown interaction kernels flexibly
3. Use SymReg to extract interpretable equations from the NN

**Key idea**: **Kernel-by-type** with NN learning
```
N_ij = κ_{type(i), type(j)}(P_ij)  ← learned by NN, then interpreted by SymReg
```

### UDE Structure

**Known physics (not learned):**
- Node types: K=3 (predator, prey, resource)
- Self-rates: a_P, a_Y, a_R (decay rates per type)

**Unknown (learned by NN):**
- 9 message kernels: κ(P_ij, type_i, type_j)
- Input: (P_ij, type_i_onehot, type_j_onehot) = 7 features
- Output: 9 values [κ_PP, κ_PY, κ_PR, κ_YP, κ_YY, κ_YR, κ_RP, κ_RY, κ_RR]

### True Dynamics (for testing)

Includes **Holling Type II** for predator-prey (nonlinear saturation):
```julia
κ_PY(p) = α·p / (1 + β·p)    # Saturating predation (to be discovered!)
κ_YP(p) = -0.04·p            # Linear fleeing
κ_PP    = -0.008             # Constant repulsion
# ... etc
```

### Pipeline

1. **Generate data** from true dynamics (includes Holling Type II)
2. **DUASE estimation** (introduces ~35% position error)
3. **Train NN kernel** on DUASE estimates
4. **Symbolic Regression**: For each of 9 kernel outputs, fit:
   - Constant: κ(p) = c
   - Linear: κ(p) = a + b·p
   - Holling Type II: κ(p) = α·p / (1 + β·p)
5. **Evaluate** in P-space on true initial conditions

### Key Demonstrations

1. **SymReg discovers Holling Type II** for predator-prey interaction
   - The nonlinear ecological law emerges from flexible learning
2. **Domain knowledge helps**: Types + self-rates structure the problem
3. **Interpretability**: Each discovered κ has biological meaning
4. **Extrapolation**: Dynamics model predicts beyond training window

### Why This Is Interesting

1. **Full pipeline**: Data → UDE → NN → SymReg → Interpretable equations
2. **Realistic structure**: Self-rates as domain knowledge is common in ecology
3. **Nonlinearity discovery**: Holling Type II is a classic functional response
4. **Practical relevance**: Most real networks have node types

**Script**: `scripts/example4_type_kernel.jl`

**Dimensions**: n=25 (8 predators, 10 prey, 7 resources), d=2, K=3 types

---

## Example 5: Parameter Recovery Test

**Story**: Test whether polynomial N(P)X can recover EXACT parameters when the true dynamics match the architecture perfectly.
**Relationship to Example 1**: Example 1 validated the methodology with message-passing repulsion. Example 5 generates data from polynomial dynamics Ẋ = (αI + βP)X to test exact parameter recovery.

**Motivation**: The RDPG embedding has gauge freedom - X and XQ give identical probabilities P = XX' for any orthogonal Q. Standard NNs can waste capacity learning "invisible" dynamics. The N(P)X form with symmetric N eliminates this ambiguity.

### Validated Architectures (from Example 1)

1. **Polynomial N(P)X**: Ẋ = (β₀I + β₁P + β₂P²)X
   - 3 learnable scalars
   - Interpretation: β₀ = self-dynamics, β₁ = direct neighbor, β₂ = two-hop
   - ✅ Works: Parameters are gauge-invariant, transfer from DUASE to true coords

2. **Message-Passing N(P)X**: ẋᵢ = a·xᵢ + m·Σⱼ Pᵢⱼ(xⱼ - xᵢ)
   - 2 learnable scalars (a = intrinsic rate, m = message strength)
   - Equivalent symmetric N: N_ii = a - m·Σⱼ Pᵢⱼ, N_ij = m·Pᵢⱼ
   - ✅ Works: Natural interpretation as attraction/repulsion

3. **Standard NN baseline**: Ẋ = f(X) with MLP
   - ~10,000 parameters
   - No gauge consistency guarantees
   - Compare to show efficiency gain

### Updated Protocol (based on Example 1 learnings)

**Data generation**:
- Generate TRUE trajectories from known dynamics (e.g., Ẋ = (αI + βP)X)
- Create DUASE estimates (SVD + Procrustes alignment)
- Split: 70% training, 30% validation (extrapolation test)

**Training**:
- Train on DUASE estimates (X_est)
- Use soft barrier loss for probability constraints (NO B^d_+ projection)
- Adam → LBFGS optimization

**Evaluation** (the key insight):
- Apply learned parameters to TRUE initial conditions X_true(0)
- Integrate to get X_pred(t)
- Compare P_pred(t) = X_pred·X_pred' to P_true(t)
- **Primary metric**: ||P_pred - P_true||_F (gauge-invariant)
- Secondary: X-trajectory comparison (noting gauge ambiguity)

### Key Demonstrations

1. **"Learn anywhere, apply everywhere"**:
   - Train on noisy DUASE estimates
   - Apply to clean true coordinates
   - Show P-error is low despite X-coordinate differences

2. **Parameter efficiency**:
   - Polynomial: 3 params vs MsgPass: 2 params vs NN: ~10,000 params
   - All achieve comparable P-error

3. **Parameter recovery**:
   - If true dynamics are Ẋ = (αI + βP)X, does polynomial recover α, β?
   - Compare learned vs true coefficients

4. **Extrapolation**:
   - DUASE can only estimate where we have observations
   - Dynamics models predict into the future
   - Show P(t) predictions beyond training window

5. **DUASE vs Dynamics distinction**:
   - DUASE = estimation method (dashed lines in plots)
   - Poly/MsgPass = dynamics models (solid lines)
   - Only dynamics can extrapolate

### Comparison Table to Generate

| Architecture | Params | P-Error (train) | P-Error (extrap) | Params Recovered? |
|--------------|--------|-----------------|------------------|-------------------|
| Polynomial   | 3      | ?               | ?                | Yes/No            |
| MsgPass      | 2      | ?               | ?                | Yes/No            |
| Standard NN  | ~10k   | ?               | ?                | N/A               |
| DUASE only   | 0      | baseline        | cannot           | N/A               |

**Script**: `scripts/example5_gauge_comparison.jl`

**Dimensions**: d=2, n=30 (or n=11 like Example 1), clean data

---

## Technical Notes

### Probability Constraint Strategies (Updated January 2026)

**⚠️ Key finding**: B^d_+ projection is NOT recommended for numerical learning.

**Recommended approach** (validated in Example 1):
1. **Soft barrier loss**: Add penalty for P_ij outside [0,1]
   ```julia
   prob_loss = sum(max.(-P, 0).^2) + sum(max.(P .- 1, 0).^2)
   ```
2. Standard Euclidean ODE solver (Tsit5, etc.)
3. No projection needed - barrier loss is sufficient

**Why NOT B^d_+ projection**:
- Distorts the natural geometry of embedding space
- Can introduce artifacts that complicate learning
- Unnecessary when soft constraints work well

**When B^d_+ IS useful**:
- Theoretical analysis of constraint satisfaction
- Mathematical proofs about probability bounds
- NOT for practical optimization

### Windowing for SVD Estimation

- Window size affects SVD stability
- Smaller window (3): More responsive, noisier
- Larger window (5): Smoother, may miss fast dynamics
- Compare in Example 1

### Script Structure (for each example)

```julia
# example_N.jl
# Usage:
#   julia --project scripts/example_N.jl           # Full run
#   julia --project scripts/example_N.jl --viz     # Visualization only

# 1. Generate/load data (respecting B^d_+)
# 2. Setup NNs and dynamics functions
# 3. Train Full NN vs UDE (if not --viz)
# 4. Symbolic regression (if not --viz)
# 5. Save results
# 6. Visualize (from saved data)
```

---

## TODO (Updated January 2026)

### Completed
- [x] Example 1: Parsimonious UDE Showcase - **DONE** (validated gauge-equivariant N(P)X)
- [x] Validate polynomial N(P)X architecture
- [x] Validate message-passing N(P)X architecture
- [x] Demonstrate "learn anywhere, apply everywhere" principle
- [x] Show P(t) as primary evaluation metric
- [x] Confirm B^d_+ projection is unnecessary
- [x] Document misspecification robustness (poly captures msgpass)
- [x] Example 4: Type-specific kernels - **SCRIPT CREATED** (`example4_type_kernel.jl`)

### Ready to Run
- [ ] **Example 4**: Run and validate type-specific N(P)X
  - 9 parameters for K=3 types
  - Test parameter recovery
  - Compare to homogeneous case

### Lower Priority / May Skip
- [ ] Example 2 (homogeneous circulation) - redundant with Example 1
- [ ] Example 3a (wrong known part) - already have implicit misspec in Ex1
- [ ] Example 3b (noisy data) - could be interesting for robustness
- [ ] Example 5 (parameter recovery) - redundant with Examples 1 & 4
- [ ] Symbolic regression step - nice-to-have
