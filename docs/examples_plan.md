# UDE Examples Plan for Paper

## Overview

This document outlines the synthetic examples demonstrating the Neural ODE + UDE + Symbolic Regression pipeline for learning RDPG dynamics.

**Key constraint**: All embeddings must remain in B^d_+ (non-negative unit ball):
- All coordinates ≥ 0
- ||L|| ≤ 1

This ensures valid RDPG probabilities: P(edge i,j) = L_i · L_j ∈ [0,1].

---

## Pipeline for Each Example

```
Generate Data (in B^d_+)
    → Train Full NN vs UDE (trajectory-based loss)
    → Symbolic Regression on learned NN
    → Compare & Visualize
```

**Outputs per example:**
- Loss curves (training progress)
- Trajectory plots (true vs predicted)
- Symbolic equations discovered
- Error metrics table (training/validation)

---

## Example 1: Bridge Node Oscillation (Lead Example)

**Story**: A "bridge" node oscillates between two community attractors within B^d_+.

**Dynamics** (in B^d_+):
```
dL/dt = -k₁(L - A₁) * sigmoid(sin(ωt)) - k₂(L - A₂) * sigmoid(-sin(ωt))
```
where A₁, A₂ are attractor points in first quadrant.

Alternative (simpler): Interpolation along arc in first quadrant.

**Known part**: Oscillatory structure (switching between attractors)
**Unknown part**: Attractor locations, frequency, transition sharpness

**Key demonstrations**:
- Role of windowing in SVD estimation (window=3 vs window=5)
- Clean trajectory - easy to see fit quality
- Loss curves showing training progress
- Symbolic regression recovers attractor locations

**Dimensions**: d=2, clean data

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

## Example 4: Food Web Heterogeneous Dynamics

**Story**: Different node types follow different dynamics (inspired by RDPG food web).

**Setup**:
- Multiple node types (e.g., predator, prey, resource)
- Each type has different ODE parameters
- All must stay in B^d_+

**Known part**: General attraction/repulsion structure
**Unknown part**: Type-specific parameters

**Key demonstrations**:
- Handles real-world complexity
- Scalability to higher dimensions
- Conditional UDE (node type as input)

**Dimensions**: d=4, clean data

---

## Technical Notes

### B^d_+ Constraint Strategies

1. **Bounded angular dynamics**: θ ∈ [0, π/2] keeps points in first quadrant
2. **Reflecting boundaries**: Bounce off boundaries of B^d_+
3. **Soft constraints**: Add penalty for leaving B^d_+ during training
4. **Projection**: Project predictions back to B^d_+ after each step

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

## TODO

- [x] Fix Example 2 dynamics to stay in B^d_+ (circulation approach)
- [ ] Implement Example 2 (circulation) - validates the approach
- [ ] Implement Example 1 (bridge node)
- [ ] Implement Example 3a (wrong ω)
- [ ] Implement Example 3b (noisy data)
- [ ] Look at food web dynamics in RDPG repo for Example 4
- [ ] Add symbolic regression step to pipeline
