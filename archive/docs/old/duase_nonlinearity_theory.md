# Nonlinear Extensions to DUASE for Curvature Preservation

## 1. The Problem

We observe temporal networks A(1), A(2), ..., A(T) generated from latent positions:
```
A(t) ~ Bernoulli(P(t))  where  P(t) = X(t) X(t)'
```

The true dynamics are:
```
dX/dt = N(P) · X
```
where N depends on P (and thus on X) in potentially nonlinear ways (e.g., Holling Type II).

**Goal**: Estimate X(t) from A(t) such that the estimated trajectories preserve the curvature needed to learn the dynamics.

**Current problem**: DUASE estimates linearize trajectories, losing the curvature signal.

---

## 2. DUASE Decomposition

### 2.1 The Algorithm

Given adjacency matrices A(1), ..., A(T):

1. **Unfold horizontally**: Form Ã = [A(1) | A(2) | ... | A(T)], size n × (nT)

2. **SVD**: Ã = U Σ V', take top d components to get G = U[:,1:d]

3. **Project each timestep**: Q(t) = G' A(t) G, size d × d

4. **Extract positions**: X̂(t) = G · √Q(t)

### 2.2 The Constraint

The key equation is:
```
X̂(t) = G · R(t)    where R(t) = √Q(t)
```

- **G** is n × d, fixed across all times
- **R(t)** is d × d, varies with time
- Each node i has position: X̂(t)[i,:] = G[i,:] · R(t)

### 2.3 Why This Linearizes

The velocity of node i is:
```
dX̂[i,:]/dt = G[i,:] · dR/dt
```

**Critical observation**: The matrix dR/dt is the SAME for all nodes.

This means:
- All nodes undergo the same linear transformation at each instant
- Nodes with similar G rows move in parallel directions
- Types cannot curve differently from each other

---

## 3. Decomposing the Motion

Let's write the velocity more explicitly. If R(t) evolves as:
```
R(t + δt) = R(t) + δt · Ṙ(t)
```

Then:
```
X̂(t + δt)[i,:] = G[i,:] · R(t + δt)
                = G[i,:] · R(t) + δt · G[i,:] · Ṙ(t)
                = X̂(t)[i,:] + δt · G[i,:] · Ṙ(t)
```

The velocity direction for node i is determined by G[i,:] · Ṙ(t).

**For two nodes i and j**:
```
v_i = G[i,:] · Ṙ(t)
v_j = G[j,:] · Ṙ(t)
```

If G[i,:] ≈ G[j,:] (same type), then v_i ≈ v_j (parallel motion).

---

## 4. Where Can Nonlinearity Enter?

### 4.1 Option A: Node-Specific R(t)

Replace:
```
X̂(t)[i,:] = G[i,:] · R(t)           [DUASE]
```
With:
```
X̂(t)[i,:] = G[i,:] · R_i(t)         [Node-specific]
```

**Problem**: Loses all temporal alignment. Back to independent ASE with its noise issues.

### 4.2 Option B: Type-Specific R(t)

Replace:
```
X̂(t)[i,:] = G[i,:] · R(t)           [DUASE]
```
With:
```
X̂(t)[i,:] = G[i,:] · R_type(i)(t)   [Type-specific]
```

**Analysis**:
- k types → k different d×d matrices R_type(t)
- Each type can move in different directions
- Still provides alignment within types
- Need to constrain: X̂(t) X̂(t)' ≈ P(t)

**Constraint satisfaction**:
```
P̂[i,j] = X̂[i,:] · X̂[j,:]'
       = G[i,:] · R_type(i) · R_type(j)' · G[j,:]'
```

For same-type pairs (type(i) = type(j) = τ):
```
P̂[i,j] = G[i,:] · R_τ · R_τ' · G[j,:]' = G[i,:] · Q_τ · G[j,:]'
```

For cross-type pairs (type(i) = τ, type(j) = σ):
```
P̂[i,j] = G[i,:] · R_τ · R_σ' · G[j,:]'
```

The cross-type term R_τ · R_σ' is NOT necessarily symmetric or equal to any Q!

### 4.3 Option C: Additive Correction

Replace:
```
X̂(t) = G · R(t)                     [DUASE]
```
With:
```
X̂(t) = G · R(t) + Δ(t)              [With correction]
```

Where Δ(t) is n × d capturing deviations from DUASE.

**Analysis**:
```
P̂ = X̂ X̂' = (GR + Δ)(GR + Δ)'
   = GR R'G' + GR Δ' + Δ R'G' + Δ Δ'
   = G Q G' + [cross terms] + Δ Δ'
```

The cross terms GR Δ' + Δ R'G' allow node-specific adjustments.

**Constraint**: Δ(t) must be smooth and preserve P.

### 4.4 Option D: Nonlinear Transformation of Q

Replace:
```
X̂(t) = G · √Q(t)                    [DUASE]
```
With:
```
X̂(t) = G · f(Q(t))                  [Nonlinear f]
```

Where f: ℝ^(d×d) → ℝ^(d×d) is a nonlinear function.

**Problem**: f acts the same on all nodes, so still no heterogeneous motion.

### 4.5 Option E: Type-Dependent Nonlinear f

Replace:
```
X̂(t)[i,:] = G[i,:] · √Q(t)          [DUASE]
```
With:
```
X̂(t)[i,:] = G[i,:] · f_type(i)(Q(t))  [Type-specific nonlinear]
```

**Analysis**: This combines the benefits of type-specificity with potential nonlinearity.

Example: f_τ(Q) = √Q · Ω_τ where Ω_τ is a type-specific rotation.

---

## 5. The Type-Augmented DUASE (TA-DUASE)

### 5.1 Formulation

Let there be k node types. For each type τ ∈ {1, ..., k}, define a time-varying orthogonal matrix Ω_τ(t) ∈ O(d).

```
X̂(t)[i,:] = G[i,:] · √Q(t) · Ω_type(i)(t)
```

### 5.2 Properties

**Within-type pairs** (both type τ):
```
P̂[i,j] = G[i,:] · √Q · Ω_τ · Ω_τ' · √Q' · G[j,:]'
       = G[i,:] · Q · G[j,:]'              [since Ω_τ Ω_τ' = I]
```
Same as DUASE! Within-type P is preserved exactly.

**Cross-type pairs** (type τ and σ):
```
P̂[i,j] = G[i,:] · √Q · Ω_τ · Ω_σ' · √Q' · G[j,:]'
       = G[i,:] · √Q · Ω_τσ · √Q' · G[j,:]'    [where Ω_τσ = Ω_τ Ω_σ']
```

The cross-type interaction depends on the RELATIVE rotation Ω_τσ = Ω_τ · Ω_σ'.

### 5.3 Degrees of Freedom

- DUASE: Q(t) has d² degrees of freedom per timestep
- TA-DUASE: Q(t) + k rotations Ω_τ(t), each with d(d-1)/2 DoF
- For d=2, k=3: 4 + 3×1 = 7 DoF per timestep (vs 4 for DUASE)

### 5.4 Optimization Problem

```
minimize    Σ_t ||X̂(t) X̂(t)' - A(t)||²_F + λ Σ_t,τ ||Ω_τ(t+1) - Ω_τ(t)||²_F

subject to  X̂(t)[i,:] = G[i,:] · √Q(t) · Ω_type(i)(t)
            Ω_τ(t) ∈ O(d)  for all τ, t
```

The smoothness penalty on Ω ensures types don't jump around arbitrarily.

---

## 6. Alternative: Direct Nonlinear Decomposition

### 6.1 Neural Network Parameterization

Instead of √Q, use a learned nonlinear function:
```
X̂(t) = G · NN_θ(Q(t), t)
```

Where NN_θ: ℝ^(d×d) × ℝ → ℝ^(d×d) is a neural network.

**Constraint**: Must reconstruct P:
```
X̂(t) X̂(t)' ≈ P(t)  ⟺  G · NN(Q,t) · NN(Q,t)' · G' ≈ A(t)
```

### 6.2 Type-Conditioned Neural Decomposition

```
X̂(t)[i,:] = G[i,:] · NN_θ(Q(t), type(i), t)
```

The network takes type as input, allowing type-specific transformations.

---

## 7. Theoretical Questions

1. **Identifiability**: Given P(t), can we uniquely determine the Ω_τ(t)? Or is there remaining gauge freedom?

2. **Consistency**: If the true dynamics are dX/dt = N(P)X with type-specific N, does TA-DUASE recover trajectories with the correct curvature?

3. **Stability**: Does adding type-specific rotations amplify noise, or does the smoothness constraint stabilize it?

4. **Rank constraints**: DUASE assumes rank-d structure. Does TA-DUASE change the effective rank?

---

## 8. Relation to Gauge Theory

The RDPG gauge group is O(d) - orthogonal transformations that leave P invariant:
```
P = X X' = (X Q)(X Q)'  for any Q ∈ O(d)
```

DUASE implicitly fixes a gauge by choosing G from the SVD.

TA-DUASE partially relaxes this: different types can live in different gauges, related by Ω_τ.

**Key insight**: The gauge is NOT arbitrary - it affects the TRAJECTORY, not just the instantaneous P. The "correct" gauge is the one where trajectories have physically meaningful curvature.

---

## 9. Next Steps

1. **Implement TA-DUASE** with type-specific Ω_τ(t)
2. **Test on synthetic data** where true types have different curvatures
3. **Compare** curvature preservation vs standard DUASE
4. **Evaluate** whether UDE can learn nonlinear dynamics from TA-DUASE trajectories

---

## 10. Open Questions

1. Can we derive Ω_τ(t) directly from the observed A(t) without optimization?

2. Is there a "canonical" type-specific gauge that maximizes curvature preservation?

3. How does the number of types k affect the reconstruction quality?

4. Can this be extended to continuous "type" (soft clustering)?

---

## 11. Deeper Analysis: Why P Doesn't Determine Curvature

### 11.1 The Gauge-Curvature Relationship

Key insight: **P(t) determines X(t) only up to rotation, but the curvature depends on which rotation we choose.**

Given P(t) = X(t)X(t)', any X̃(t) = X(t)Ω for Ω ∈ O(d) gives the same P(t).

But the velocities are different:
```
dX/dt = N(P)X       →   v = NX
dX̃/dt = N(P)X̃ = NXΩ  →   ṽ = NXΩ = vΩ
```

The velocity DIRECTIONS differ by rotation Ω. So the curvature (angle between consecutive velocities) depends on Ω!

### 11.2 The Fundamental Incompatibility

**True dynamics**: X evolves freely in ℝ^(n×d) according to dX/dt = N(P)X.

**DUASE constraint**: X̂(t) must lie in the subspace {G·R : R ∈ ℝ^(d×d)}.

These are incompatible unless the true dynamics happen to preserve the G-structure.

More precisely: DUASE velocity is
```
dX̂/dt = G · d(√Q)/dt
```
which is determined ENTIRELY by Q(t) and Q̇(t).

But true velocity is
```
dX/dt = N(XX')X
```
which depends on X, not just on P = XX'.

**Critical observation**: Different X with the same P have different velocities!

### 11.3 Information in Curvature

The curvature contains information BEYOND P:
- P(t) tells us the "fiber" of valid X(t) (a coset of O(d))
- Curvature tells us which point on that fiber we're at, and how we're moving on it

When we estimate X from A(t), we're recovering P(t) but losing the curvature information.

### 11.4 Using Higher Derivatives of P

From P = XX' and dX/dt = NX, we can derive:
```
dP/dt = dX/dt · X' + X · dX'/dt
      = NXX' + XX'N'
      = NP + PN'
```

This is a **Lyapunov-like equation** relating dP/dt to N.

**Key idea**: If we observe dP/dt (from P(t+1) - P(t)), we might be able to infer N!

For type-structured N where N_{ij} = κ(type_i, type_j, P_{ij}):
- We have n(n+1)/2 equations (symmetric part of dP/dt = NP + PN')
- We have k² type-pair interaction functions κ_{τσ}(·)

If κ is parameterized (e.g., linear, Holling), this is an overdetermined system.

### 11.5 The Circular Problem

To find the "correct" gauge (embedding), we need to know the dynamics N.
To learn the dynamics N, we need the correct embedding X.

**Possible resolutions**:

**A. Joint optimization**: Parameterize both the embedding X(t;φ) and dynamics N(P;θ), optimize both.

**B. P-space dynamics**: Skip X entirely, learn dynamics directly on P:
```
dP/dt = f(P; θ)   where f must satisfy f(P) = NP + PN' for some N
```

**C. Type constraints**: Use the type structure to reduce the gauge freedom.

**D. Smoothness prior**: Choose the gauge that makes trajectories "simplest" (minimum acceleration, etc.)

---

## 12. Learning Dynamics Directly from P

### 12.1 The P-Space Approach

Instead of embedding to X, learn dynamics directly on P:
```
minimize  Σ_t ||P(t+1) - P(t) - Δt·f(P(t);θ)||²
```

where f must have the form f(P) = N(P)P + PN(P)' for consistency.

### 12.2 Parameterizing N(P)

For type-structured interactions:
```
N_{ij} = δ_{ij} · r_{type(i)} + (1-δ_{ij}) · κ(type(i), type(j), P_{ij})
```

where:
- r_τ is the self-interaction rate for type τ
- κ(τ, σ, p) is the interaction function

Examples:
- Linear: κ(τ,σ,p) = α_{τσ} · p
- Holling II: κ(τ,σ,p) = α_{τσ} · p / (1 + β_{τσ} · p)

### 12.3 Advantages of P-Space

1. **No gauge ambiguity**: P is invariant, no rotation to choose
2. **Direct observability**: A(t) → P(t) is straightforward
3. **Symmetric structure**: P is symmetric, reducing dimensionality

### 12.4 Disadvantages of P-Space

1. **Higher dimensionality**: P is n×n vs X is n×d
2. **Nonlinear constraint**: P must be PSD with entries in [0,1]
3. **No geometric intuition**: Harder to visualize than X trajectories

---

## 13. The Type-Specific Gauge Revisited

### 13.1 Why Types Help

Within a type, nodes should move "coherently" - they have similar interactions with other types.

DUASE enforces too much coherence: ALL nodes (across types) undergo the same transformation.

Type-specific gauges relax this: types can move differently, but nodes within a type move coherently.

### 13.2 Mathematical Formulation

Let X_DUASE(t) = G · √Q(t) be the standard DUASE embedding.

Define type-specific embedding:
```
X_τ(t) = X_DUASE(t)|_{type τ} · Ω_τ(t)
```

where Ω_τ(t) ∈ O(d) is the type-τ gauge.

The full embedding is:
```
X̂(t)[i,:] = X_DUASE(t)[i,:] · Ω_{type(i)}(t)
```

### 13.3 What Ω_τ Can and Cannot Do

**Can do**:
- Change the velocity direction for type τ
- Allow different types to curve differently
- Preserve within-type P exactly (since Ω_τ Ω_τ' = I)

**Cannot do**:
- Change within-type relative positions (G[i,:] vs G[j,:] for same type)
- Add non-rotational deformations
- Change the speed (only direction)

### 13.4 Constraint from Cross-Type P

For i of type τ and j of type σ:
```
P̂_{ij} = X̂[i,:] · X̂[j,:]'
       = (G[i,:] √Q Ω_τ) · (G[j,:] √Q Ω_σ)'
       = G[i,:] √Q Ω_τ Ω_σ' √Q' G[j,:]'
       = G[i,:] √Q Ω_{τσ} √Q' G[j,:]'
```

where Ω_{τσ} = Ω_τ Ω_σ' is the relative rotation.

For P̂ ≈ P, we need:
```
√Q Ω_{τσ} √Q' ≈ Q   for all type pairs τ,σ
```

In 2D, Ω_{τσ} is parameterized by a single angle θ_{τσ} = θ_τ - θ_σ.

The constraint becomes:
```
R(θ_{τσ}) ≈ √Q^{-1} Q √Q'^{-1} = I
```

So Ω_{τσ} ≈ I, meaning θ_τ ≈ θ_σ for all τ,σ.

**This is too restrictive!** It forces all types to have nearly the same rotation.

### 13.5 Relaxing the Constraint

The issue: we're trying to preserve P exactly while adding rotation freedom.

Alternative: accept some P error in exchange for better curvature.

New objective:
```
min  λ_P · ||P̂ - P||² + λ_curv · CurvatureLoss + λ_smooth · SmoothnessLoss
```

where CurvatureLoss encourages trajectories to match some target curvature pattern.

But what target curvature? We don't know the true curvature!

### 13.6 A Different Approach: Curvature from dP/dt

From Section 11.4, we know dP/dt = NP + PN'.

If we can estimate N from the Lyapunov equation, we can compute:
```
dX/dt = NX
d²X/dt² = dN/dt · X + N · dX/dt = (Ṅ + N²)X
```

The curvature depends on d²X/dt², which we can express in terms of N and its derivatives.

**Idea**: Use the structure dP/dt = NP + PN' to constrain the embedding X such that its curvature is consistent with some N.

---

## 14. Extracting Velocity from ΔP: The Sylvester Approach

### 14.1 The Key Equation

From dP/dt = VX' + XV', if we have:
- X̂ from embedding P
- ΔP ≈ dP/dt from consecutive observations

We can solve for V̂ via the **Sylvester-like equation**:
```
V̂X̂' + X̂V̂' = ΔP
```

### 14.2 Solution via Eigendecomposition

For ASE embedding, X̂ = U·Λ^{1/2} where U (n×d) are top eigenvectors of P and Λ = diag(λ₁,...,λ_d).

Project V onto the column space of U: let A = U'V (d×d matrix).

Substituting V ≈ UA into V̂X̂' + X̂V̂' = ΔP and multiplying by U' on left and U on right:
```
U'·(UA·X̂' + X̂·A'U')·U = U'·ΔP·U
A·(X̂'U) + (U'X̂)·A' = S    where S = U'·ΔP·U
```

Since X̂ = U·Λ^{1/2}, we have U'X̂ = Λ^{1/2} and X̂'U = Λ^{1/2}. So:
```
A·Λ^{1/2} + Λ^{1/2}·A' = S
```

This is a **Sylvester equation** in A!

### 14.2.1 Solving the Sylvester Equation

The equation A·Λ^{1/2} + Λ^{1/2}·A' = S (with S symmetric) has solutions:

**For d=2**: Let Λ^{1/2} = diag(σ₁, σ₂). Then:
```
[a₁₁  a₁₂] [σ₁  0 ]   [σ₁  0 ] [a₁₁  a₂₁]   [s₁₁  s₁₂]
[a₂₁  a₂₂] [0   σ₂] + [0   σ₂] [a₁₂  a₂₂] = [s₁₂  s₂₂]
```

Expanding:
- (1,1): 2σ₁·a₁₁ = s₁₁  →  a₁₁ = s₁₁/(2σ₁)
- (2,2): 2σ₂·a₂₂ = s₂₂  →  a₂₂ = s₂₂/(2σ₂)
- (1,2): σ₂·a₁₂ + σ₁·a₂₁ = s₁₂
- (2,1): σ₁·a₁₂ + σ₂·a₂₁ = s₁₂  [same equation!]

The off-diagonal gives ONE equation for TWO unknowns (a₁₂, a₂₁):
```
σ₂·a₁₂ + σ₁·a₂₁ = s₁₂
```

**The antisymmetric part (a₁₂ - a₂₁) is FREE!**

Canonical choice: A symmetric, so a₁₂ = a₂₁ = s₁₂/(σ₁ + σ₂).

### 14.2.2 The Complete Algorithm

```
Input: P(t), ΔP(t) = P(t+1) - P(t)
Output: X̂(t), V̂(t)

1. Eigendecompose P(t) = U·Λ·U' (top d)
2. Position: X̂ = U·Λ^{1/2}
3. Compute S = U'·ΔP·U (d×d symmetric)
4. Solve Sylvester: A·Λ^{1/2} + Λ^{1/2}·A' = S
   - Diagonal: a_{ii} = s_{ii}/(2√λᵢ)
   - Off-diagonal (symmetric choice): a_{ij} = s_{ij}/(√λᵢ + √λⱼ)
5. Velocity (projected): V̂ = U·A
```

### 14.2.3 Residual Check

The projected velocity V̂ = UA satisfies:
```
V̂X̂' + X̂V̂' = U·A·Λ^{1/2}·U' + U·Λ^{1/2}·A'·U' = U·S·U'
```

This equals ΔP only if ΔP is in the span of UU'. The residual:
```
ΔP_⊥ = ΔP - U·S·U'
```

represents velocity components orthogonal to the current embedding subspace.

### 14.3 The Freedom in the Solution

The solution is unique up to adding X̂R for antisymmetric R ∈ so(d).

This corresponds to the **gauge freedom**: changing the time-derivative of the rotation.

Choosing D symmetric gives the "canonical" velocity that doesn't introduce arbitrary rotation.

### 14.4 Why This Recovers Curvature

DUASE computes X̂(t) = G·√Q(t), which determines velocity implicitly via:
```
V̂_DUASE(t) = X̂(t+1) - X̂(t) = G·(√Q(t+1) - √Q(t))
```

This velocity is constrained by the G structure.

The Sylvester approach computes V̂ directly from ΔP:
```
V̂_Sylvester satisfies V̂X̂' + X̂V̂' = ΔP
```

This velocity is NOT constrained by G - it's whatever velocity is consistent with the observed ΔP!

### 14.5 The Algorithm: Position-Velocity Embedding

```
Input: A(1), ..., A(T) adjacency matrices
Output: X̂(t), V̂(t) for t = 1, ..., T

1. Estimate P(t) ≈ mean of A samples (or use A directly)
2. Compute ΔP(t) = P(t+1) - P(t) for t = 1, ..., T-1

3. For each t:
   a. SVD of P(t): P ≈ X̂X̂' where X̂ = U·√Λ (top d eigenvectors)
   b. Compute S = U'·ΔP(t)·U
   c. Set D = S/2
   d. Compute C = ΔP(t)·U - U·D
   e. Compute V̂(t) = C·Λ^{-1/2}·U'·X̂ ... [need to work out details]

4. Align X̂(t) across time via Procrustes

5. Return X̂(t), V̂(t)
```

### 14.6 Gauge Consistency Check

If X̂ = XQ for some rotation Q, does V̂ = VQ?

From V̂X̂' + X̂V̂' = ΔP:
```
V̂(XQ)' + (XQ)V̂' = ΔP
V̂Q'X' + XQV̂' = ΔP
```

For this to equal VX' + XV' = ΔP, we need V̂Q' = V, i.e., V̂ = VQ. ✓

**The Sylvester solution respects gauge transformations!**

### 14.7 What Information Does V̂ Contain?

V̂ satisfies V̂X̂' + X̂V̂' = ΔP, which means:
- V̂ is consistent with the observed change in P
- V̂ captures the "true" velocity direction (up to gauge)
- The curvature of the V̂ trajectory reflects the true dynamics

Unlike DUASE velocity (which is G·Ṙ for universal G), Sylvester velocity V̂ is derived from ΔP and can have node-specific directions.

### 14.8 Combining with DUASE

One approach: use DUASE for position, Sylvester for velocity.

```
X̂(t) = G·√Q(t)           [DUASE position]
V̂(t) from Sylvester       [Velocity from ΔP]
```

But there's a consistency issue: V̂(t) should ≈ X̂(t+1) - X̂(t).

Alternative: use Sylvester V̂ to CORRECT the DUASE trajectory.

```
X̂_corrected(t+1) = X̂(t) + dt·V̂(t)
```

This uses the ΔP-derived velocity instead of the DUASE-implied velocity.

### 14.9 Gauge Freedom vs Physical Rotation

**Critical clarification**: There are two types of "rotation" to distinguish:

#### Rotation around the Origin (Gauge Freedom)

If X → XQ for orthogonal Q, then P = XX' is unchanged. This is the RDPG gauge freedom.

- **Unobservable** from P or any of its derivatives
- **We don't care** about this - it's coordinate choice
- The antisymmetric freedom in Sylvester solution corresponds to this
- Choosing A symmetric removes this and only this

#### Rotation around a Centroid c ≠ 0 (Physical Dynamics)

If nodes circulate around a centroid c ≠ 0:
```
X(t) = c + R(θ(t))·(X₀ - c)
```

Velocity:
```
V = dX/dt = θ̇·J·(X - c)    where J = [0 -1; 1 0] (2D rotation generator)
```

Then:
```
VX' + XV' = θ̇·(J(X-c)X' + X(X-c)'J')
          = θ̇·(JXX' - JcX' + XX'J' - Xc'J')
          = θ̇·(JP - PJ) - θ̇·(JcX' + Xc'J')
```

The commutator [J,P] = JP - PJ is **nonzero** for general P, and there are additional terms from c.

**This changes P!** So circulation around non-origin centroids IS observable from ΔP.

#### Implication for Sylvester Approach

The Sylvester solution V̂ satisfying V̂X̂' + X̂V̂' = ΔP captures:
- ✓ Radial expansion/contraction
- ✓ Circulation around centroids (c ≠ 0)
- ✓ All curvature from P-changing dynamics
- ✗ Rotation around origin (gauge - but we don't care!)

**Choosing A symmetric removes only the unobservable gauge rotation, preserving all physical dynamics.**

### 14.10 The Propagation Algorithm

Given the gauge clarification, here's the refined algorithm:

```
Input: P(1), P(2), ..., P(T)
Output: X̂(1), ..., X̂(T) with consistent gauge and correct curvature

1. Initialize:
   - Embed P(1) → X̂(1) = U₁·Λ₁^{1/2}  [arbitrary initial gauge]

2. For t = 1 to T-1:
   a. Compute ΔP(t) = P(t+1) - P(t)

   b. Solve Sylvester for velocity:
      - Let X̂(t) = U·Λ^{1/2} (current embedding)
      - Compute S = U'·ΔP(t)·U
      - Solve A·Λ^{1/2} + Λ^{1/2}·A' = S with A symmetric
      - V̂(t) = U·A

   c. Propagate position:
      - X̂(t+1) = X̂(t) + V̂(t)

   d. [Optional] Project X̂(t+1) to satisfy P(t+1) exactly:
      - Procrustes: find Q minimizing ||X̂(t+1)·Q - X̃(t+1)||
        where X̃(t+1) is ASE of P(t+1)
      - Or: re-embed but align to X̂(t+1)

3. Return X̂(1), ..., X̂(T)
```

### 14.11 Why This Preserves Curvature

**DUASE problem**: X̂(t) = G·√Q(t) with fixed G. Velocity is G·d(√Q)/dt - same transformation for all nodes.

**Sylvester solution**: V̂(t) comes from ΔP(t), which contains:
- First-order information about how P changes
- Implicitly encodes N(P) through dP/dt = NP + PN'

The curvature (change in velocity direction) depends on V̂(t+1) - V̂(t), which reflects:
- Change in ΔP (i.e., d²P/dt²)
- Change in X̂ (propagated with V̂)

Since d²P/dt² contains acceleration information, and we're extracting velocity consistently at each step, the curvature should be preserved.

### 14.12 Open Questions

1. How does noise in ΔP affect V̂? (ΔP is noisier than P)

2. Should we project back to P-consistent manifold at each step, or let errors accumulate?

3. How does the propagation error grow over time?

4. Can we use higher-order integration (Runge-Kutta style) for better accuracy?

---

## 15. Worked Example: Two Nodes Circulating Around a Centroid

### 15.1 Setup

Consider n=2 nodes circulating around centroid c = (c₁, c₂) with radius r and angular velocity ω:

```
X₁(t) = c + r·(cos(ωt), sin(ωt))
X₂(t) = c - r·(cos(ωt), sin(ωt))    [opposite side]
```

So:
```
X(t) = [c₁ + r·cos(ωt)   c₂ + r·sin(ωt)]
       [c₁ - r·cos(ωt)   c₂ - r·sin(ωt)]
```

### 15.2 The P Matrix

```
P(t) = X(t)·X(t)'
```

Computing each entry:
```
P₁₁ = (c₁ + r·cos)² + (c₂ + r·sin)² = |c|² + r² + 2r(c₁·cos + c₂·sin)
P₂₂ = (c₁ - r·cos)² + (c₂ - r·sin)² = |c|² + r² - 2r(c₁·cos + c₂·sin)
P₁₂ = (c₁² - r²cos²) + (c₂² - r²sin²) = |c|² - r²
```

**Key observation**: P₁₂ is constant! Only the diagonal elements change.

Let β(t) = c₁·cos(ωt) + c₂·sin(ωt). Then:
```
P(t) = [|c|² + r² + 2rβ    |c|² - r²        ]
       [|c|² - r²          |c|² + r² - 2rβ  ]
```

### 15.3 The Velocity and ΔP

True velocity:
```
V(t) = dX/dt = [−rω·sin(ωt)    rω·cos(ωt)]
               [ rω·sin(ωt)   −rω·cos(ωt)]
```

Change in P:
```
dP/dt = [2rω(−c₁·sin + c₂·cos)    0                        ]
        [0                         −2rω(−c₁·sin + c₂·cos)  ]
```

Let α(t) = rω(−c₁·sin(ωt) + c₂·cos(ωt)). Then:
```
ΔP ≈ dP/dt·dt = [2α    0 ]·dt
                [0    −2α]
```

### 15.4 Verification: VX' + XV' = dP/dt

Let's verify the fundamental equation. After computation:
```
VX' = rω(−c₁·sin + c₂·cos)·[ 1   1]
                            [−1  −1]

XV' = rω(−c₁·sin + c₂·cos)·[ 1  −1]
                            [ 1  −1]

VX' + XV' = rω(−c₁·sin + c₂·cos)·[2   0]  =  [2α   0 ]
                                  [0  −2]     [0   −2α]
```

This equals dP/dt. ✓

**The circulation velocity IS encoded in ΔP!**

### 15.5 Numerical Example

Let c = (0.5, 0.5), r = 0.3, ω = 1, t = 0.

**True positions:**
```
X(0) = [0.8   0.5]
       [0.2   0.5]
```

**P matrix:**
```
P(0) = [0.89   0.41]
       [0.41   0.29]
```

**ASE embedding** (eigendecomposition of P):
- λ₁ ≈ 1.098, λ₂ ≈ 0.082
- Eigenvectors form U

```
X̂(0) = U·Λ^{1/2} ≈ [0.935    0.129]
                    [0.474   −0.255]
```

Note: X̂ ≠ X because they're in different gauges. But X̂·X̂' ≈ P. ✓

**ΔP at t=0:**

At t=0: sin(0)=0, cos(0)=1, so α = rω·c₂ = 0.3·1·0.5 = 0.15
```
ΔP = [0.3    0  ]·dt
     [0    −0.3]
```

### 15.6 Sylvester Solution for Velocity

Compute S = U'·ΔP·U (project ΔP onto eigenbasis):
```
S ≈ [ 0.178    0.242]·dt
    [ 0.242   −0.178]
```

Solve A·Λ^{1/2} + Λ^{1/2}·A' = S with symmetric A:
```
a₁₁ = s₁₁/(2σ₁) = 0.178/(2·1.048) ≈ 0.085
a₂₂ = s₂₂/(2σ₂) = −0.178/(2·0.286) ≈ −0.311
a₁₂ = s₁₂/(σ₁+σ₂) = 0.242/(1.334) ≈ 0.181
```

So:
```
A ≈ [ 0.085    0.181]·dt
    [ 0.181   −0.311]
```

**Recovered velocity:**
```
V̂ = U·A
```

### 15.7 Key Verification

By construction, V̂ satisfies:
```
V̂·X̂' + X̂·V̂' = U·(A·Λ^{1/2} + Λ^{1/2}·A')·U' = U·S·U' = ΔP  ✓
```

(In 2D with d=2, U spans ℝ², so no projection loss.)

### 15.8 Gauge Relationship

X̂ and X are related by a rotation Q: X̂ ≈ X·Q

The true velocity in the X̂ gauge is V·Q.

**Claim**: V̂ ≈ V·Q (same gauge transformation).

This follows from the gauge covariance of the Sylvester equation (Section 14.6).

### 15.9 Curvature Check

The trajectory has constant angular velocity ω, so the curvature is constant.

**True curvature**: The angle between V(t) and V(t+dt) is ω·dt (rotation rate).

**Recovered curvature**: Since V̂ = V·Q with constant Q, the angle between V̂(t) and V̂(t+dt) is also ω·dt.

**The Sylvester approach recovers the correct curvature!**

### 15.10 Why This Worked

1. Circulation around c ≠ 0 changes P (diagonal elements oscillate)
2. ΔP encodes the velocity information
3. Sylvester extracts V̂ consistent with ΔP
4. The symmetric A choice removes only origin-rotation (gauge), not centroid-rotation (physical)
5. Propagating with V̂ maintains gauge consistency
6. Curvature is preserved because V̂ ∝ V (up to gauge)

### 15.11 What Would Fail?

If c = 0 (rotation around origin):
- P would be constant! (P₁₁ = P₂₂ = r², P₁₂ = −r²)
- ΔP = 0
- V̂ = 0 (no velocity recovered)
- The rotation is pure gauge and unobservable

This confirms: origin-rotation is gauge (unrecoverable), centroid-rotation is physical (recoverable).

---

## 16. Summary of Mathematical Insights

1. **P doesn't determine X uniquely**: X is determined up to rotation (gauge freedom O(d)).

2. **But ΔP contains velocity information**: From dP/dt = VX' + XV', we can extract V given X.

3. **Two types of rotation**:
   - Rotation around **origin**: gauge freedom, unobservable from P, we don't care
   - Rotation around **centroid c≠0**: physical dynamics, changes P, recoverable from ΔP

4. **DUASE linearizes** because X̂ = G·√Q(t) with fixed G constrains all nodes to the same transformation.

5. **Sylvester approach**: Given X̂ from P and ΔP, solve V̂X̂' + X̂V̂' = ΔP for velocity V̂.

6. **The solution is unique up to gauge**: The antisymmetric freedom in A corresponds to origin-rotation only.

7. **Propagation preserves P**: X̂(t) + dt·V̂(t) satisfies P(t+1) to first order.

8. **Curvature is preserved**: Because V̂ captures all P-changing dynamics, including centroid-rotation.

9. **The worked example confirms**: Circulation around c≠0 is fully recoverable from ΔP via Sylvester.

---

---

## 17. Three-Point Stencil: Coupling Velocities via Acceleration

### 17.1 The Problem with Two-Point Sylvester

The simple Sylvester approach computes V(t) independently at each timestep from ΔP(t).

Issues:
- Velocities at consecutive times are not coupled
- Gauge freedom (antisymmetric part of A) is chosen arbitrarily at each step
- No acceleration/curvature information is directly used

### 17.2 Second Derivative of P

From P = XX' and dX/dt = V:
```
dP/dt = VX' + XV'
```

Differentiating again:
```
d²P/dt² = d(VX' + XV')/dt
        = V̇X' + VẊ' + ẊV' + XV̇'
        = AX' + VV' + VV' + XA'
        = AX' + XA' + 2VV'
```

where A = dV/dt = V̇ is the **acceleration**.

Rearranging:
```
AX' + XA' = d²P/dt² - 2VV'
```

This is another **Sylvester equation for A**!

### 17.3 Three-Point Discretization

Using the standard second-derivative stencil:
```
d²P/dt² ≈ (P(t+1) - 2P(t) + P(t-1)) / dt²
```

And the first derivative (centered):
```
dP/dt ≈ (P(t+1) - P(t-1)) / (2dt)
```

### 17.4 The Coupled Algorithm

```
Input: P(1), P(2), ..., P(T)
Output: X(t), V(t), A(t) for all t

1. Initialize at t=1:
   - X(1) from ASE of P(1)
   - V(1) from Sylvester: V·X' + X·V' = P(2) - P(1)

2. For t = 2 to T-1:
   a. Compute second derivative:
      d²P = P(t+1) - 2P(t) + P(t-1)

   b. Solve for acceleration A(t-1):
      A·X(t-1)' + X(t-1)·A' = d²P - 2V(t-1)·V(t-1)'

   c. Update velocity:
      V(t) = V(t-1) + A(t-1)

   d. Propagate position:
      X(t+1) = X(t) + V(t)

3. Return X, V, A
```

### 17.5 Why This Couples Velocities

In the simple Sylvester approach:
- V(t) and V(t+1) are computed independently
- No constraint relates them

In the three-point approach:
- V(t) = V(t-1) + A(t-1)
- A(t-1) is derived from d²P, which involves P(t-1), P(t), P(t+1)
- **Velocities are coupled through the acceleration!**

### 17.6 Solving the Acceleration Sylvester Equation

The equation AX' + XA' = M where M = d²P - 2VV' has the same structure as the velocity Sylvester.

Project onto eigenbasis U of XX':
```
B·Λ^{1/2} + Λ^{1/2}·B' = U'·M·U
```

where B = U'·A and A = U·B.

The solution (choosing B symmetric):
```
b_ii = m_ii / (2σ_i)
b_ij = m_ij / (σ_i + σ_j)
```

### 17.7 Curvature is Built In

The curvature (change in velocity direction) is directly related to acceleration A.

By computing A from d²P, we're using the **actual curvature information** from the P trajectory, not just inferring it from consecutive velocities.

### 17.8 The Gauge Constraint

The three-point approach also helps with gauge:
- V(t) = V(t-1) + A
- If V(t-1) has a certain gauge, V(t) inherits it (plus the A contribution)
- The gauge evolves smoothly rather than being chosen independently at each step

### 17.9 Higher-Order Extensions

We could go further:
- **Four points**: d³P/dt³ for jerk (derivative of acceleration)
- **Smoothing splines**: fit a smooth curve through P(t) and take derivatives

The three-point method is a good balance between using temporal information and computational simplicity.

---

## 18. Experimental Results: Forward Propagation Fails

### 18.1 Test Setup

We tested the Sylvester approaches on a predator-prey-resource system:
- n = 37 nodes (12 predators, 15 prey, 10 resources)
- d = 2 embedding dimension
- T = 25 timesteps
- Holling Type II interactions (nonlinear)
- True curvature ≈ 5.2° (mean angle between consecutive velocities)

### 18.2 Methods Compared

1. **DUASE**: Standard algorithm with global Procrustes alignment
2. **Two-Point Sylvester**: Solve VX' + XV' = ΔP at each step, propagate X(t+1) = X(t) + V(t)
3. **Three-Point Sylvester**: Use d²P to couple velocities via acceleration
4. **Three-Point Anchored**: Re-anchor to ASE of P every 5 steps

### 18.3 Results

| Method | Curvature | P-error (mean) | P-error at T=25 |
|--------|-----------|----------------|-----------------|
| True | 5.2° | 0% | 0% |
| DUASE | 1.8° (0.35×) | 1.7% | 2.1% |
| Two-Point Sylvester | 10.8° (2.1×) | 41.7% | 67% |
| Three-Point Sylvester | 13.3° (2.6×) | 42% | 68% |
| Three-Point Anchored | — | exploded | 10^15% |

### 18.4 Analysis: Why Forward Propagation Fails

**The fundamental issue**: Forward propagation accumulates errors with no feedback mechanism.

1. **Error amplification in velocity**:
   - Small errors in V(t) propagate to X(t+1)
   - X(t+1) error affects the eigenbasis U for the next Sylvester solve
   - Errors compound exponentially

2. **Three-point is WORSE because of 2VV' term**:
   - The acceleration equation: AX' + XA' = d²P - 2VV'
   - Errors in V get **squared** in the 2VV' term
   - This amplifies velocity errors rather than correcting them

3. **Anchoring doesn't help**:
   - Re-anchoring introduces discontinuities in V
   - The velocity "jumps" when we reset to ASE
   - These jumps accumulate and destabilize the trajectory

4. **Low-rank approximation loses information**:
   - With d=2 for n=37 nodes, we're projecting to a 2D subspace
   - True ΔP has rank up to 37, but we only capture rank-2 component
   - The discarded components contain dynamics information

### 18.5 The Projection Residual

When solving V̂X̂' + X̂V̂' = ΔP with X̂ = UΛ^{1/2} (rank d), the best we can do is:

```
V̂X̂' + X̂V̂' = U·S·U'  where S = U'·ΔP·U
```

The residual ΔP_⊥ = ΔP - U·S·U' is **lost information**. In our tests:
- ||ΔP_⊥|| / ||ΔP|| ≈ 30-50% at each step
- This error accumulates through propagation

### 18.6 Curvature Over-Estimation

Why does the Sylvester approach give 2-3× the true curvature?

1. The propagated trajectory drifts from the true P manifold
2. As X̂(t) drifts, P̂ = X̂X̂' diverges from P(t)
3. The "velocity" V̂ is now solving a different problem (matching ΔP with wrong X̂)
4. This creates spurious curvature from the correction attempts

### 18.7 Fundamental Limitation

**Forward propagation cannot work for this problem** because:

1. We only observe P, not X directly
2. P constrains X to a manifold (orbit of O(d))
3. The dynamics dX/dt = NX live on this manifold
4. But forward propagation leaves the manifold immediately
5. Once off-manifold, there's no way back without re-anchoring
6. Re-anchoring disrupts the velocity continuity

This is analogous to integrating on a sphere: small Euler steps leave the sphere, and projection back creates numerical issues.

---

## 19. Alternative Approaches

Given that forward propagation fails, what are the alternatives?

### 19.1 Global Optimization (Variational)

Instead of propagating forward, solve for the entire trajectory X(1), ..., X(T) simultaneously:

```
minimize  Σ_t ||X(t)X(t)' - P(t)||² + λ·Σ_t ||X(t+1) - X(t) - V(t)||²
                                    + μ·Σ_t ||V(t)X(t)' + X(t)V(t)' - ΔP(t)||²
subject to: gauge constraints (e.g., fix X(1))
```

**Advantages**:
- All timesteps coupled simultaneously
- No error accumulation
- Can enforce Sylvester constraint as soft penalty

**Challenges**:
- Large optimization problem (n×d×T variables)
- Non-convex (P constraint is quadratic)
- Need good initialization

### 19.2 P-Space Dynamics

Skip X entirely and learn dynamics directly on P:

```
dP/dt = f(P; θ)   subject to: f(P) = NP + PN' for some N
```

**Advantages**:
- No gauge ambiguity
- P is directly observable
- Constraints are linear in N

**Challenges**:
- Higher dimensionality (n² vs nd)
- PSD constraint on P
- Less geometric intuition

### 19.3 Type-Parameterized Embedding

Use the type structure to constrain the embedding:

```
X[nodes of type τ, :] = G_τ · R_τ(t)
```

where G_τ is type-specific and R_τ(t) is type-specific time evolution.

**Advantages**:
- Reduces degrees of freedom
- Types move coherently
- Cross-type P constrains relative R_τ

**Challenges**:
- Requires knowing types a priori
- Still needs to fit observed P
- May be too restrictive for some dynamics

### 19.4 Manifold Integration

Use geometric integration methods designed for manifold-valued dynamics:

- **Lie group integrators** for the rotation component
- **Projection methods** that maintain P = XX' exactly
- **Symplectic methods** if there's a Hamiltonian structure

**Advantages**:
- Designed for manifold constraints
- Better long-time behavior

**Challenges**:
- Complex to implement
- May not fit RDPG structure exactly

---

## 20. Current Conclusions

1. **DUASE linearizes trajectories** (curvature 0.35× true) but maintains P consistency (1.7% error).

2. **Sylvester forward propagation fails**: curvature is over-estimated (2-3×) and P error grows unboundedly (40-70%+).

3. **Three-point stencil makes it worse**: the 2VV' term amplifies velocity errors.

4. **The fundamental issue is error accumulation**: without feedback to the P manifold, trajectories drift.

5. **Possible paths forward**:
   - Global optimization (variational formulation)
   - P-space dynamics (skip X entirely)
   - Type-parameterized embedding
   - Manifold integration methods

---

## 21. Variational Embedding: Global Optimization Approach

### 21.1 Motivation

Since forward propagation fails due to error accumulation, we try **global optimization**: solve for the entire trajectory X(1), ..., X(T) simultaneously.

### 21.2 Basic Formulation

Variables: X(2), ..., X(T) (with X(1) fixed to ASE of P(1))

Objective:
```
L(X) = λ_P · Σ_t ||X(t)X(t)' - P(t)||²_F     [P reconstruction]
     + λ_s · Σ_t ||X(t+1) - X(t)||²_F        [smoothness]
     + λ_a · Σ_t ||X(t+1) - 2X(t) + X(t-1)||²_F  [acceleration penalty]
```

Implementation: LBFGS with analytic gradients.

### 21.3 Results: Basic Variational

| Method | Curvature | P-error |
|--------|-----------|---------|
| True | 5.17° | 0% |
| DUASE | 4.56° (0.88×) | 12.8% |
| Variational (λ_s=0.1, λ_a=0.01) | 10.68° (2.07×) | 1.8% |
| Variational (λ_s=0.01, λ_a=0.001) | 11.13° (2.15×) | 1.8% |
| Variational (λ_s=1.0, λ_a=0.1) | 10.51° (2.03×) | 1.9% |

**Key observation**: Variational achieves excellent P reconstruction (1.8% vs 12.8%) but **over-estimates curvature by ~2×**.

The smoothness parameter λ_s has minimal effect on curvature (10.5-11.1° across 100× range).

### 21.4 Diagnosis: Gauge Drift

The variational approach over-estimates curvature because of **gauge drift**:
- The optimizer can rotate each X(t) slightly to fit P(t) better
- These rotations accumulate as spurious "curvature"
- The smoothness penalty doesn't prevent gauge drift because small rotations are smooth

### 21.5 Failed Remedy: Sylvester Consistency

Adding a Sylvester consistency term: ||VX' + XV' - ΔP||²

| Method | Curvature | P-error |
|--------|-----------|---------|
| Variational + Sylvester (λ=0.5) | 14.89° (2.88×) | 1.9% |

**Result**: Made curvature WORSE (14.89° vs 10.68°). The Sylvester term doesn't prevent gauge drift.

### 21.6 Failed Remedy: Procrustes Penalty

Penalize ||X(t+1) - X(t)*Q||² where Q is the Procrustes alignment from X(t) to X(t+1).

| Method | Curvature | P-error |
|--------|-----------|---------|
| Variational + Procrustes (λ=1) | 10.65° (2.06×) | 1.9% |

**Result**: No improvement. The Procrustes penalty measures the residual AFTER optimal alignment, which is already small.

### 21.7 Failed Remedy: Antisymmetric Penalty

Penalize ||antisym(X'V)||² to discourage rotational velocity.

| Method | Curvature | P-error |
|--------|-----------|---------|
| Variational + Antisym (λ=1) | 16.17° (3.13×) | 5.2% |
| Variational + Antisym (λ=10) | 11.97° (2.32×) | 9.8% |

**Result**: Made things WORSE.

**Critical insight**: Centroid rotation (which is physical) produces antisymmetric X'V. Penalizing it removes real curvature and forces P error up.

### 21.8 Summary of Variational Results

| Method | Curvature | P-error | Notes |
|--------|-----------|---------|-------|
| True | 5.17° | 0% | - |
| DUASE | 4.56° | 12.8% | Under-estimates curvature, high P error |
| Variational | 10.68° | 1.8% | Over-estimates curvature, low P error |
| + Sylvester | 14.89° | 1.9% | Worse curvature |
| + Procrustes | 10.65° | 1.9% | No change |
| + Antisym (λ=1) | 16.17° | 5.2% | Much worse |
| + Antisym (λ=10) | 11.97° | 9.8% | Still worse, P error up |
| DUASE-reg (λ=0.1) | 10.96° | 1.8% | No change from variational |
| DUASE-reg (λ=1) | 10.81° | 2.7% | Slight reduction |
| DUASE-reg (λ=10) | 8.89° | 7.4% | **Best**: curvature 1.72× (down from 2×) |

### 21.9 The Fundamental Trade-off

There appears to be a trade-off:
- **DUASE**: Constrains gauge tightly → under-estimates curvature, high P error
- **Variational**: Fits P well → over-estimates curvature due to gauge drift

No penalty we tried can reduce variational curvature toward the true value while maintaining P accuracy.

### 21.10 Why Is This Hard?

The core problem: **curvature and gauge rotation look the same locally**.

Both produce changes in X that are "perpendicular" to the radial direction. The difference is:
- True curvature: changes P (centroid rotation, type-specific motion)
- Gauge drift: preserves P (rotation around origin)

But in the optimization, both contribute to reducing ||XX' - P||. The optimizer finds gauge rotations that help fit P, even though they're physically meaningless.

---

## 22. Open Questions and Future Directions

### 22.1 Is There a "Right" Curvature?

The experiments show:
- DUASE: 4.56° (under)
- Variational: 10.68° (over)
- True: 5.17°

The true curvature is between them. Is there a principled way to interpolate?

### 22.2 DUASE Regularization (Tested)

Using DUASE as a regularizer:
```
L = ||XX' - P||² + λ_duase ||X - X_DUASE||²
```

Results:

| λ_duase | Curvature | P-error |
|---------|-----------|---------|
| 0 (pure variational) | 10.68° | 1.8% |
| 0.1 | 10.96° | 1.8% |
| 1.0 | 10.81° | 2.7% |
| 10.0 | 8.89° | 7.4% |
| ∞ (pure DUASE) | 4.56° | 12.8% |

**Observation**: DUASE regularization creates a smooth trade-off between:
- Low P error, high curvature (variational limit)
- High P error, low curvature (DUASE limit)

With λ_duase=10, curvature drops to 8.89° (closer to 5.17° true) but P error rises to 7.4%.

**The true curvature (5.17°) would require even stronger regularization**, but that pushes P error toward DUASE levels (12.8%).

### 22.3 P-Space Dynamics

Perhaps the cleanest solution is to skip X entirely:
- Learn dP/dt = NP + PN' directly
- No gauge ambiguity (P is invariant)
- N can be parameterized by type

### 22.4 The Role of Type Structure

All our methods treat nodes equally. But the dynamics have type structure:
- Predators, prey, resources move differently
- Cross-type interactions drive the dynamics

Could we constrain the embedding to respect types?
```
X[type τ nodes, :] = centroid_τ(t) + deviation_τ(t)
```

### 22.5 What's Actually Learnable?

Given just P(t), what can we learn about the dynamics?
- dP/dt = NP + PN' is directly observable
- But N is not unique (many N give the same dP/dt)
- The type structure constrains N

This suggests: learn N (parameterized by type), not X.

---

## 23. Conclusions

### What Works:
1. **Mathematical theory**: Sylvester equation VX' + XV' = ΔP correctly extracts velocity from P changes
2. **Gauge theory**: Rotation around origin is unobservable; rotation around centroid is physical
3. **Variational approach**: Can achieve excellent P reconstruction (1.8% error)

### What Doesn't Work:
1. **Forward Sylvester propagation**: Error accumulation leads to P drift (40-70% error)
2. **Three-point stencil**: The 2VV' term amplifies velocity errors
3. **Variational for curvature**: Over-estimates by 2× due to gauge drift
4. **Gauge-drift penalties** (Procrustes, antisymmetric): Either no effect or make things worse

### The Fundamental Problem:
P(t) determines X(t) only up to gauge (O(d) rotation). Different gauge choices give different curvatures. Neither DUASE's constraint (all nodes share G) nor variational's freedom (each timestep optimized) gives the "true" curvature.

### Promising Directions:
1. **P-space dynamics**: Work directly with P, avoid gauge entirely
2. **Type-parameterized embedding**: Use problem structure to constrain gauge
3. **Learn N instead of X**: Parameterize interactions by type, fit to dP/dt

---

## 24. Joint Embedding + Dynamics Optimization

### 24.1 The Idea

Instead of learning embedding and dynamics separately, optimize them jointly:
- The dynamics model acts as a **physics prior** constraining the gauge
- The "right" embedding is one where the dynamics have a simple, physically-meaningful form

### 24.2 Formulation

Variables:
- X(t) for t=1,...,T (embeddings)
- θ (dynamics parameters)

Objective:
```
L = λ_P · Σ_t ||X(t)X(t)' - P(t)||²     [P reconstruction]
  + λ_dyn · Σ_t ||X(t+1) - X(t) - f(X(t);θ)||²  [dynamics consistency]
  + λ_reg · ||θ||²                        [regularization]
```

Dynamics model (message-passing):
```
Ẋᵢ = β₀[type(i)] · Xᵢ + Σⱼ α[type(i),type(j)] · Pᵢⱼ · (Xⱼ - Xᵢ)
```

### 24.3 Model Variants Tested

1. **LinearMsgPass**: Learn β₀ (3 self-rates) + α (9 interactions) = 12 params
2. **FixedSelfMsgPass**: Fix β₀ to known values, learn only α = 9 params
3. **SparseMsgPass**: Fix β₀, only learn 4 specific interactions (P←Y, Y←P, Y←R, R←Y)

### 24.4 Results

| Method | Curvature | P-error | Notes |
|--------|-----------|---------|-------|
| True | 5.17° | 0% | - |
| DUASE | 4.56° | 12.8% | Under-estimates |
| Joint Linear (λ_dyn=1) | 12.69° | 1.9% | 2.45× over |
| Joint FixedSelf (λ_dyn=1) | 16.13° | 1.9% | 3.12× over - worse! |
| **Joint Sparse (λ_dyn=1)** | 10.65° | 1.9% | 2.06× over |
| **Joint Sparse (λ_dyn=10)** | **9.03°** | 2.4% | **1.75× over - best!** |

### 24.5 Key Findings

1. **More constrained models help**: Sparse (4 params) < Linear (12 params) < FixedSelf (9 params) in curvature over-estimation. Counter-intuitively, FixedSelf is worst - fixing self-rates without fixing interaction structure doesn't help.

2. **Sparse + strong dynamics weight is best**: λ_dyn=10 with 4-parameter sparse model gives 9.03° (1.75× true), the best result so far.

3. **But extrapolation fails**: Even the best sparse model has 91% P-error when extrapolating from true X(1). The learned parameters are close to true values but the embedding-dynamics combination doesn't transfer.

### 24.6 Learned vs True Parameters (Sparse λ_dyn=1)

| Parameter | Learned | True | Notes |
|-----------|---------|------|-------|
| α_PY (Predator←Prey) | 0.0021 | ~0.0125* | Under-estimated |
| α_YP (Prey←Predator) | -0.0174 | -0.02 | Close! |
| α_YR (Prey←Resource) | 0.0209 | 0.012 | Over-estimated |
| α_RY (Resource←Prey) | -0.0127 | -0.006 | Over-estimated |

*True α_PY is Holling: 0.025·p/(1+2p) ≈ 0.0125 at p=0.5

### 24.7 Why Extrapolation Fails

The embedding X learned jointly with dynamics θ is optimized for:
1. Reconstructing P over the training time window
2. Being consistent with the learned dynamics f(X;θ)

But this doesn't mean:
- X matches the true embedding (up to gauge)
- θ matches the true dynamics
- The combination transfers outside the training window

The joint optimization finds a **self-consistent** (X, θ) pair, not necessarily the **true** one.

### 24.8 Comparison: All Methods So Far

| Method | Curvature | P-error | Extrapolation |
|--------|-----------|---------|---------------|
| True | 5.17° | 0% | 0% |
| DUASE | 4.56° (0.88×) | 12.8% | N/A |
| Variational | 10.68° (2.07×) | 1.8% | N/A |
| DUASE-reg λ=10 | 8.89° (1.72×) | 7.4% | N/A |
| **Joint Sparse λ=10** | **9.03° (1.75×)** | 2.4% | 91% error |

The two best methods (DUASE-reg λ=10 and Joint Sparse λ=10) achieve similar curvature (~1.7-1.8× true) with different trade-offs:
- DUASE-reg: higher P-error (7.4%) but simpler (no dynamics model)
- Joint Sparse: lower P-error (2.4%) but extrapolation fails

### 24.9 Implications

1. **Curvature ~1.7× true seems to be a limit** for methods that fit P well
2. **Lower curvature requires accepting higher P-error** (DUASE direction)
3. **Joint optimization doesn't automatically give transferable dynamics**

The fundamental gauge freedom means that even with a correct dynamics model structure, we can't uniquely recover the true embedding without additional constraints.
