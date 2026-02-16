# Examples Implementation Notes

This document describes the example scripts for comparing Neural ODE and UDE approaches on RDPG dynamics. For general implementation details (RDPG estimation, B^d_+ alignment, training optimization), see the main [Julia Implementation Notes](../Julia_implementation_notes.md).

---

## Julia 1.11+ Script Best Practices

All scripts follow modern Julia conventions:

### Shebang with `--project`

```bash
#!/usr/bin/env -S julia --project
```

The `-S` flag allows passing arguments to julia. This enables direct execution:
```bash
chmod +x scripts/example1_bridge_node_v2.jl
./scripts/example1_bridge_node_v2.jl generate
```

### Entry Point with `@main`

Julia 1.11 introduced the `@main` macro for proper script entry points:

```julia
function (@main)(args)
    # args is the command-line arguments (like ARGS but passed explicitly)
    if isempty(args)
        print_usage()
        return 0  # Exit code
    end

    # ... handle commands ...

    return 0  # Success
end
```

Benefits over the old `if abspath(PROGRAM_FILE) == @__FILE__` pattern:
- Cleaner syntax
- Explicit return codes (0 = success, non-zero = error)
- Arguments passed as function parameter
- Works correctly with precompilation

### No `Pkg.activate(".")`

With `--project` in the shebang/command, manual activation is unnecessary.

---

## Modular Pipeline Architecture

Each example uses a phase-based pipeline that saves intermediate data, allowing you to:
- Re-run training without regenerating data
- Tweak visualization without retraining
- Compare different models on identical data

### Pipeline Phases

```
Phase 1: generate   → data/<example>/true_dynamics.jls
Phase 2: estimate   → data/<example>/estimated.jls
Phase 3: train      → data/<example>/model_<name>.jls
Phase 4: evaluate   → data/<example>/evaluation.jls
Phase 5: visualize  → results/<example>/*.pdf
```

### Shared Module

All examples use `shared_pipeline.jl` which provides:
- `phase_generate()` - Generate true dynamics, save positions
- `phase_estimate()` - RDPG estimation with B^d_+ alignment
- `phase_train()` - Train with BacksolveAdjoint + ADAM/BFGS
- `phase_evaluate()` - Compute D correlation, P error metrics
- `phase_visualize()` - Generate comparison plots

Key functions from RDPGDynamics library:
- `sample_adjacency_repeated(X, K)` - Generate K adjacency samples
- `embed_temporal_network_Bd_plus(A_series, d)` - SVD + Procrustes + B^d_+ alignment

---

## Example 1: Bridge Node Between Communities

**File**: `example1_bridge_node_v2.jl`

**Purpose**: Test whether absolute (fixed points) vs equivariant (centroid-based) dynamics matter when evaluated with rotation-invariant metrics.

### Setup
- Two communities (35 nodes each) clustered around centers A₁, A₂
- One "bridge" node oscillating between communities
- n = 71 nodes, d = 2

### Models Compared

| Model | Known Structure | What NN Learns |
|-------|-----------------|----------------|
| `pure_nn` | None | Everything |
| `ude_absolute` | Cohesion to fixed A₁, A₂ | Switching, boundary |
| `ude_equivariant` | Cohesion to centroids X̄₁, X̄₂ | Switching, boundary |

### Key Question

The preliminary results from `full_pipeline_comparison.jl` showed that even non-equivariant dynamics can achieve good rotation-invariant metrics. This example tests:

> Does formulating dynamics in terms of fixed points (absolute) vs data-dependent centroids (equivariant) make a practical difference when we evaluate using D correlation and P error?

### Usage

```bash
julia --project scripts/example1_bridge_node_v2.jl generate
julia --project scripts/example1_bridge_node_v2.jl estimate
julia --project scripts/example1_bridge_node_v2.jl train all
julia --project scripts/example1_bridge_node_v2.jl evaluate
julia --project scripts/example1_bridge_node_v2.jl visualize
```

---

## Example 2: Homogeneous Dynamics (4 Options)

**File**: `example2_homogeneous_v2.jl`

**Purpose**: Find the best homogeneous dynamics formulation for Examples 3a/3b. All nodes follow the same rule, enabling pooled training.

### Setup
- n = 60 nodes, d = 2
- All nodes follow identical dynamics (different initial conditions)

### Dynamics Options

| Option | Description | Observable? | UDE Known Part |
|--------|-------------|-------------|----------------|
| `pairwise` | Lennard-Jones: -a/r + b/r³ | ✓ Distances change | Pairwise structure |
| `cohesion` | Cohesion to centroid + repulsion | ✓ Distances change | Centroid attraction |
| `radial` | Spring toward target radius | ✓ Radial distances | Radial form |
| `kuramoto` | Angular coupling + radial | ✓ Relative angles | Radial spring |

### Selection Criteria

The best dynamics for Examples 3a/3b should:
1. Produce observable distance changes (learnable from RDPG)
2. Have clear known/unknown decomposition (for UDE comparison)
3. Be simple enough to interpret results

### Usage

```bash
# Run all 4 dynamics
julia --project scripts/example2_homogeneous_v2.jl all pairwise
julia --project scripts/example2_homogeneous_v2.jl all cohesion
julia --project scripts/example2_homogeneous_v2.jl all radial
julia --project scripts/example2_homogeneous_v2.jl all kuramoto

# Compare results
julia --project scripts/example2_homogeneous_v2.jl compare
```

The `compare` command ranks all dynamics/model combinations by validation D correlation and recommends the best one for Examples 3a/3b.

---

## Example 3a/3b: Wrong Known Structure / Noisy Data

**Files**: To be created after Example 2 comparison

**Purpose**:
- 3a: Test UDE when the "known" structure is wrong
- 3b: Test noise robustness (varying K samples instead of Gaussian noise)

### Design Decision

These examples will use the **winning dynamics from Example 2**. The shared modular pipeline makes this easy:

```julia
# Copy the dynamics module from example2
const DYNAMICS = Example2Winner.dynamics_module

# 3a: Use wrong parameters in UDE known part
const K_COHESION_WRONG = 0.05  # True is 0.1

# 3b: Vary K samples (real noise source)
for K in [10, 30, 50, 100]
    phase_estimate(example_name; K=K)
    # ... train and compare
end
```

### Why RDPG Sampling Noise (not Gaussian)

The original `example3b_noisy_data.jl` added Gaussian noise to positions:
```julia
noisy = clean .+ randn() * sigma  # Synthetic noise
```

This is unrealistic. In the real pipeline, noise comes from:
1. **Finite samples**: Only K adjacency matrices per timestep
2. **SVD estimation error**: Eigenvalue gaps, rotation ambiguity
3. **B^d_+ projection**: Clamping/scaling distortions

The v2 approach tests the **actual noise source** by varying K:
- K = 10: High noise (few samples)
- K = 100: Low noise (many samples)

---

## Example 4: Food Web (Heterogeneous Types)

**File**: `example4_foodweb_v2.jl`

**Purpose**: Test higher-dimensional embedding (d=4) with heterogeneous node types.

### Setup
- 3 node types: Predator (15), Prey (25), Resource (20)
- n = 60 nodes, d = 4
- Equivariant dynamics using type centroids

### Dynamics

| Type | Dynamics | Equivariant? |
|------|----------|--------------|
| Predator | Cohesion + hunt (→ prey centroid) | ✓ |
| Prey | Cohesion + flee (← pred centroid) + feed (→ res centroid) | ✓ |
| Resource | Cohesion + regrowth | ✓ |

### Why Equivariant Only

For d=4, rotation ambiguity is severe (6 rotation parameters vs 1 for d=2). Using fixed targets would likely fail completely. The equivariant formulation using type centroids is:
- Naturally interpretable (predators track prey, prey flee predators)
- Rotation-invariant by construction
- Matches ecological intuition

### Visualization

Since d=4, we show 2D projections:
- Dims 1-2: Often captures predator-prey separation
- Dims 3-4: Often captures resource positioning

### Usage

```bash
julia --project scripts/example4_foodweb_v2.jl all
```

---

## Network Sizes

| Example | n (nodes) | d (dims) | Communities/Types |
|---------|-----------|----------|-------------------|
| 1 - Bridge | 71 | 2 | 35 + 35 + 1 bridge |
| 2 - Homogeneous | 60 | 2 | 1 (all same) |
| 4 - Food Web | 60 | 4 | 15 + 25 + 20 |

Chosen to be large enough (50-100 nodes) for meaningful RDPG estimation while remaining computationally tractable.

---

## Evaluation Metrics

All examples use rotation-invariant metrics (see main implementation notes):

| Metric | Formula | Measures |
|--------|---------|----------|
| D correlation | cor(D_true, D_recovered) | Pairwise distance preservation |
| P error | ‖P_true - P_rec‖ / ‖P_true‖ | Probability matrix reconstruction |

### Three Comparisons

1. **True ↔ Estimated**: RDPG estimation quality (baseline)
2. **True ↔ Recovered**: Dynamics learning quality (our goal)
3. **Train vs Val**: Generalization (overfitting check)

Success criterion: Recovered should be close to or exceed Estimated baseline.

---

## Running the Full Comparison

```bash
# 1. Example 1: Absolute vs Equivariant
julia --project scripts/example1_bridge_node_v2.jl all

# 2. Example 2: Find best homogeneous dynamics
for dyn in pairwise cohesion radial kuramoto; do
    julia --project scripts/example2_homogeneous_v2.jl all $dyn
done
julia --project scripts/example2_homogeneous_v2.jl compare

# 3. Example 4: Food web (d=4)
julia --project scripts/example4_foodweb_v2.jl all

# 4. Create Examples 3a/3b using winner from step 2
# (scripts to be created based on comparison results)
```

---

## Key Insights from Preliminary Results

From `full_pipeline_comparison.jl`:

1. **B^d_+ alignment is critical**: Without proper alignment, SVD outputs can be in negative quadrants, making learning impossible.

2. **Both models can exceed baseline**: With proper pipeline:
   - Pure NN: 103% of baseline D correlation
   - UDE: 106% of baseline D correlation

3. **Position loss works for rotation-invariant goals**: Despite rotation ambiguity, training with position MSE achieves good D correlation and P error.

4. **UDE advantage on validation**: UDE shows better generalization (Val D: 0.937 vs 0.847 for Pure NN in preliminary results).

---

## File Structure

```
scripts/
├── shared_pipeline.jl          # Common functions, phase runners
├── example1_bridge_node_v2.jl  # Bridge node (abs + equiv)
├── example2_homogeneous_v2.jl  # 4 homogeneous dynamics
├── example4_foodweb_v2.jl      # Food web (d=4, equiv)
├── EXAMPLES_NOTES.md           # This file
│
├── full_pipeline_comparison.jl # Original optimized pipeline
└── (legacy example*.jl files)  # Old versions, for reference

data/
├── example1_bridge/
│   ├── true_dynamics.jls
│   ├── estimated.jls
│   ├── model_pure_nn.jls
│   ├── model_ude_absolute.jls
│   └── model_ude_equivariant.jls
├── example2_pairwise/
├── example2_cohesion/
├── example2_radial/
├── example2_kuramoto/
└── example4_foodweb/

results/
├── example1_bridge/
│   ├── evolution_true.pdf
│   ├── comparison_*.pdf
│   ├── metrics.pdf
│   └── learning_curves.pdf
└── ...
```

---

## TODO: Symbolic Regression Phase

The current pipeline ends at Neural ODE/UDE training and evaluation. The full pipeline for the paper requires an additional phase:

### Phase 6: Symbolic Regression

After training, extract interpretable equations from the learned dynamics:

```
Phase 6: symbolic_regression → data/<example>/symbolic_model.jls
                             → results/<example>/equations.pdf
```

**Purpose**: Convert the black-box NN dynamics into closed-form equations that can be:
- Interpreted scientifically
- Compared to known ground truth
- Published in the paper

**Approach** (from main implementation notes):
1. Sample (u, du/dt) pairs from trained Neural ODE
2. Use SymbolicRegression.jl to find best-fit equations
3. Validate symbolic model against held-out data
4. Compare recovered equations to true dynamics

**Key questions to address**:
- How many sample points needed?
- What complexity constraints for symbolic search?
- How to handle multi-dimensional outputs (n×d)?
- Equation simplification and interpretation

**Reference**: See `Julia_implementation_notes.md` section on symbolic regression (currently marked as TODO in the paper draft).

---

*Last updated: 2026-01-16*
