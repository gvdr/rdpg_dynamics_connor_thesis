# Observable Dynamics in RDPG: A Gauge-Theoretic Perspective

## 1. The Observable and Its Parameterization

### 1.1 What We Observe

The **observable** is a sequence of graphs, or equivalently, the probability matrix:

$$P \in \mathbb{R}^{n \times n}, \quad P_{ij} \in [0,1]$$

This is all that exists physically—graphs are sampled from $P$, and $P$ is what we can (noisily) estimate.

### 1.2 The Parameterization

We **choose** to parameterize $P$ via latent positions:

$$P = XX^\top$$

This is a useful fiction. The matrix $X$ is not observable; only $P$ is.

### 1.3 The Constraint: Valid Configurations

For $P_{ij} \in [0,1]$, $X$ cannot be arbitrary. Define:

$$B^d_+ = \{x \in \mathbb{R}^d : x \geq 0, \|x\| \leq 1\}$$

the positive orthant of the unit ball.

**Proposition.** If $X_i \in B^d_+$ for all $i$, then $P_{ij} \in [0,1]$ for all $i, j$.

*Proof.* Non-negativity: $P_{ij} = X_i \cdot X_j = \sum_k X_{ik} X_{jk} \geq 0$ since all coordinates are non-negative. Upper bound: $P_{ij} \leq \|X_i\| \|X_j\| \leq 1$ by Cauchy-Schwarz and the norm constraint. $\square$

### 1.4 The Gauge Freedom

The parameterization is not unique. For any $Q \in O(d)$ (the orthogonal group):

$$(XQ)(XQ)^\top = XQQ^\top X^\top = XX^\top = P$$

So $X$ and $XQ$ represent the **same observable** $P$. The equivalence class:

$$[X] = \{XQ : Q \in O(d)\}$$

is what corresponds to a single probability matrix.

### 1.5 The State Space

The **valid configuration space** is:

$$\mathcal{X} = \{X \in \mathbb{R}^{n \times d} : (XX^\top)_{ij} \in [0,1] \text{ for all } i, j\}$$

**Caution on fundamental domains:** One might hope that $(B^d_+)^n$ serves as a fundamental domain, i.e., that every $O(d)$-orbit in $\mathcal{X}$ intersects $(B^d_+)^n$ exactly once (up to discrete symmetry). This is **not generally true**.

**The problem:** For a single vector $x \in \mathbb{R}^d$, we can always find $Q \in O(d)$ such that $xQ \in B^d_+$ (rotate to align with the positive orthant). But for $n$ vectors simultaneously, we need ONE $Q$ that works for ALL rows. This may be impossible.

**Example ($n = 2$, $d = 2$):** Let $X_1 = (1, 0)$ and $X_2 = (0, 1)$. Then $P_{12} = 0$. The orbit $\{(X_1 Q, X_2 Q) : Q \in O(2)\}$ includes configurations where both rows are in $B^2_+$, e.g., $Q = I$. But consider $X_1 = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$ and $X_2 = (\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}})$. Here $P_{12} = 0$, but no rotation $Q$ can make both coordinates of both rows non-negative simultaneously (they're orthogonal, so any rotation that fixes one will rotate the other).

**Resolution:** The constraint $P_{ij} \in [0,1]$ does NOT require $X \in (B^d_+)^n$ up to rotation. It requires a weaker condition:

$$X_i \cdot X_j \geq 0 \text{ and } \|X_i\| \leq 1 \text{ for all } i, j$$

The set $\mathcal{X}$ is larger than $\bigcup_{Q \in O(d)} (B^d_+)^n \cdot Q$.

**Practical approach:** When working with estimated $\hat{X}$ from ASE/SVD:
1. Project each row to unit ball: $\hat{X}_i \leftarrow \hat{X}_i / \max(1, \|\hat{X}_i\|)$
2. Check $P_{ij} = \hat{X}_i \cdot \hat{X}_j \in [0,1]$—may require additional adjustment if some $P_{ij} < 0$
3. Accept that exact $(B^d_+)^n$ membership may not be achievable

### 1.6 Two Perspectives on Dynamics

**Gauge-fixed:** Work in $(B^d_+)^n$, ensure dynamics preserve this set.

**Gauge-free:** Work in $\mathcal{X}$, allow gauge drift within orbits. The observable $P$ is still well-defined.

For UDE learning, the gauge-free perspective is cleaner mathematically, but implementation typically uses a gauge-fixed representative.

---

## 2. Dynamics: Observable vs Invisible

### 2.1 Dynamics on the Parameterization

Suppose we model latent position dynamics:

$$\frac{dX}{dt} = f(X)$$

This induces dynamics on the observable:

$$\frac{dP}{dt} = \dot{X}X^\top + X\dot{X}^\top = f(X)X^\top + Xf(X)^\top$$

### 2.2 Key Definitions

**Definition 1 (Observable dynamics).** A vector field $f$ produces *observable dynamics* if $\dot{P} \neq 0$. Otherwise, the dynamics are *invisible*—the graph sequence is static even though the parameterization moves.

**Definition 2 (Gauge equivalence).** Two vector fields $f$ and $\tilde{f}$ are *gauge equivalent* if they induce the same $\dot{P}$ for all $X$.

---

## 3. Characterization of Invisible Dynamics

**Theorem 1 (Invisible dynamics).** Let $X \in \mathbb{R}^{n \times d}$ have full column rank. A vector field $f$ produces invisible dynamics ($\dot{P} = 0$) if and only if:

$$f(X) = XA$$

for some skew-symmetric matrix $A \in \mathfrak{so}(d)$, i.e., $A^\top = -A$.

*Proof.* 

$(\Leftarrow)$ Suppose $f(X) = XA$ with $A^\top = -A$. Then:

$$\dot{P} = XAX^\top + X(XA)^\top = XAX^\top + XA^\top X^\top = X(A + A^\top)X^\top = 0$$

$(\Rightarrow)$ Suppose $\dot{P} = f(X)X^\top + Xf(X)^\top = 0$.

Decompose $f(X) = XA + W$ where $A = (X^\top X)^{-1}X^\top f(X)$ and $X^\top W = 0$.

Substituting:
$$0 = X(A + A^\top)X^\top + WX^\top + XW^\top$$

For generic full-rank $X$, the constraint $X^\top W = 0$ combined with the symmetry condition forces $W = 0$.

With $W = 0$: $X(A + A^\top)X^\top = 0$. Since $X$ has full column rank, this implies $A + A^\top = 0$. $\square$

**Corollary 1.** The space of invisible dynamics is $\mathfrak{so}(d)$, with dimension $\frac{d(d-1)}{2}$.

**Interpretation.** The invisible dynamics $\dot{X} = XA$ generate continuous motion along $O(d)$ orbits—the parameterization changes, but the observable $P$ stays fixed. These are the infinitesimal gauge transformations.

*Remark.* Reflections ($\det Q = -1$) also preserve $P$, but they are not connected to the identity, so they don't arise from continuous dynamics. The continuous gauge freedom is generated by $\mathfrak{so}(d)$.

---

## 4. Gauge Equivalence of Vector Fields

**Theorem 2 (Gauge equivalence).** Two vector fields $f$ and $\tilde{f}$ are gauge equivalent if and only if:

$$f(X) - \tilde{f}(X) = XA(X)$$

for some $\mathfrak{so}(d)$-valued function $A(X)$.

*Proof.* Apply Theorem 1 to $h = f - \tilde{f}$. $\square$

**Corollary 2 (Canonical decomposition).** Any vector field decomposes uniquely as:
$$f(X) = f_{\text{phys}}(X) + XA(X)$$
where $f_{\text{phys}}$ determines $\dot{P}$ and $XA$ is pure gauge.

**What this means for learning:** If we learn $f$ from observations of $P(t)$, we can only determine $f$ up to the gauge freedom $XA$. The "physical" content—the part that affects the observable—is uniquely determined.

---

## 5. Rotation Around Origin vs Circulation Around Centroid

**Theorem 3 (Rotation around origin is invisible).** The dynamics:
$$\dot{X}_i = X_i A, \quad A \in \mathfrak{so}(d)$$
(all nodes rotating uniformly around the origin) satisfy $\dot{P} = 0$.

*Proof.* This is exactly the form of Theorem 1. $\square$

**Theorem 4 (Circulation around nonzero centroid is observable).** Let $\bar{X} = \frac{1}{n}\sum_i X_i$ be the centroid. The dynamics:
$$\dot{X}_i = (X_i - \bar{X})A, \quad A \in \mathfrak{so}(d)$$
produce $\dot{P} \neq 0$ whenever $\bar{X} \neq 0$.

*Proof.* Rewrite: $\dot{X}_i = X_i A - \bar{X}A$. In matrix form with $\mathbf{1} \in \mathbb{R}^n$ the all-ones vector:
$$\dot{X} = XA - \mathbf{1}\bar{X}^\top A$$

Then:
$$\dot{P} = \dot{X}X^\top + X\dot{X}^\top$$
$$= (XA - \mathbf{1}\bar{X}^\top A)X^\top + X(XA - \mathbf{1}\bar{X}^\top A)^\top$$
$$= XAX^\top + XA^\top X^\top - \mathbf{1}\bar{X}^\top A X^\top - X A^\top \bar{X}\mathbf{1}^\top$$

The first two terms cancel since $A + A^\top = 0$. Let $v = A^\top \bar{X} = -A\bar{X}$:
$$\dot{P} = \mathbf{1}v^\top X^\top + Xv\mathbf{1}^\top$$

Entry-wise:
$$\dot{P}_{ij} = v^\top X_j + X_i^\top v = v \cdot X_j + X_i \cdot v$$

This vanishes for all $i,j$ only if $v = 0$, i.e., $A\bar{X} = 0$. For generic $\bar{X} \neq 0$ and $A \neq 0$, we have $\dot{P} \neq 0$. $\square$

**Interpretation.** Circulation around the centroid decomposes as:
$$\dot{X}_i = \underbrace{X_i A}_{\text{invisible}} - \underbrace{\bar{X}A}_{\text{shared drift}}$$

The first term is pure gauge (rotation around origin). The second term is a constant velocity $-\bar{X}A$ applied to all nodes—this creates observable changes in $P$ because it shifts all dot products.

---

## 6. Differential Rotation Rates

**Theorem 5 (Differential rotation is observable).** If nodes have different rotation rates:
$$\dot{X}_i = X_i A_i, \quad A_i \in \mathfrak{so}(d)$$
then:
$$\dot{P}_{ij} = X_i (A_i - A_j) X_j^\top$$

This is generically nonzero when $A_i \neq A_j$.

*Proof.* 
$$\dot{P}_{ij} = \dot{X}_i X_j^\top + X_i \dot{X}_j^\top = (X_i A_i) X_j^\top + X_i (X_j A_j)^\top$$
$$= X_i A_i X_j^\top + X_i A_j^\top X_j^\top$$

Since $A_j^\top = -A_j$:
$$\dot{P}_{ij} = X_i A_i X_j^\top - X_i A_j X_j^\top = X_i (A_i - A_j) X_j^\top$$

This is a scalar (a $1 \times d$ times $d \times d$ times $d \times 1$ product). It's nonzero when $A_i \neq A_j$ and $X_i, X_j$ are generic. $\square$

---

## 7. Summary: What Changes the Observable?

| Dynamics | $\dot{P} = 0$? | Observable? |
|----------|---------------|-------------|
| $\dot{X} = XA$ (uniform rotation around origin) | Yes | **No** |
| $\dot{X}_i = (X_i - \bar{X})A$ with $\bar{X} \neq 0$ | No | **Yes** |
| $\dot{X}_i = X_i A_i$ with $A_i \neq A_j$ | No | **Yes** |
| $\dot{X}_i = \alpha X_i$ (radial scaling) | No | **Yes** |
| $\dot{X}_i = \sum_j w_{ij}(X_j - X_i)$ (attraction/repulsion) | No | **Yes** |

**The Fundamental Theorem.** The kernel of the map $f \mapsto \dot{P}$ consists exactly of vector fields of the form $f(X) = XA$ with $A \in \mathfrak{so}(d)$. Two vector fields produce identical observable dynamics iff they differ by such a term.

---

## 8. Implications for Learning

### 8.1 What Neural ODEs Learn

When we train a Neural ODE on estimated positions $\hat{X}(t)$:
- We learn some $f$ in a particular gauge (determined by the embedding method)
- The learned $f$ determines $\dot{P}$ correctly
- A different gauge choice would give $f + XA$—same physics, different parameterization

**On $B^d_+$ projection:** The constraint $X \in (B^d_+)^n$ is convenient for mathematical analysis (ensuring $P_{ij} \in [0,1]$), but is **not required for numerical learning**. In fact, projecting estimated embeddings to $B^d_+$ can distort geometry and break temporal consistency. For dynamics of the form $\dot{X} = N(P)X$, learning in whatever coordinate system the embedding method naturally produces (e.g., DUASE) works well.

### 8.2 Why Position Loss Works

Position loss $\|\hat{X}_{pred} - \hat{X}_{target}\|^2$ implicitly captures dot product structure because:
- Temporal embedding methods (DUASE, Omnibus) provide consistent gauges across time
- In that gauge, position errors translate to probability errors
- The Neural ODE learns to match positions, which matches $P$

### 8.3 Why Distance Loss Fails

**Proposition.** Circulation around a nonzero centroid preserves all pairwise distances but changes $P$.

*Proof.* The map $X_i \mapsto R(X_i - \bar{X}) + \bar{X}$ is a rigid rotation around $\bar{X}$, preserving $\|X_i - X_j\|$. But by Theorem 4, $\dot{P} \neq 0$. $\square$

Distance-based objectives are blind to dynamics that change $P$. Since the observable is $P$ (dot products), not distances, distance loss optimizes the wrong thing.

### 8.4 Symbolic Regression and Gauge Dependence

**Equations expressed in terms of $X$ directly are gauge-dependent.**

**Critical point:** If you write dynamics as $\dot{X}_i = f(X_1, X_2, \ldots)$ with explicit coordinate dependence, the symbolic form depends on the arbitrary choice of coordinate system (gauge). In a different basis, $\dot{X}_1 = aX_2$ might appear as $\dot{X}_1 = bX_1 + cX_2$. Different researchers using different embedding/alignment procedures may recover **different-looking equations from the same data**—all equally valid.

**Example:** Suppose true dynamics are harmonic oscillation. In one gauge:
$$\dot{X}_1 = \omega X_2, \quad \dot{X}_2 = -\omega X_1$$

After rotation by $\pi/4$:
$$\dot{Y}_1 = 0, \quad \dot{Y}_2 = -\omega Y_1 + \omega Y_2 \quad \text{(different-looking!)}$$

Both produce **identical** $P(t)$.

**What IS gauge-invariant (for X-based equations):**
- Eigenvalues of the linearization (frequencies, decay rates)
- Equilibrium structure (existence, stability type)
- Qualitative dynamics (oscillatory, stable, chaotic, etc.)
- Topological properties (number of equilibria, limit cycles)

### 8.5 Gauge-Invariant Scalars in $N(P)X$ Dynamics

**Key insight:** When dynamics are parameterized as $\dot{X} = N(P)X$ with $N$ depending only on $P = XX^\top$, the **scalar parameters in $N$ are gauge-invariant**.

**Example (Message-Passing):**
$$\dot{X}_i = \beta_0 X_i + \beta_1 \sum_j P_{ij}(X_j - X_i)$$

This is equivalent to $\dot{X} = N(P)X$ where $N = \beta_0 I + \beta_1(P - D)$ with $D = \text{diag}(P\mathbf{1})$.

The scalars $\beta_0, \beta_1$ are **gauge-invariant** because:
1. $P = XX^\top$ is invariant under $X \mapsto XQ$
2. $N(P)$ therefore doesn't change under gauge transformation
3. The scalars $\beta_0, \beta_1$ parameterize $N$, not $X$ directly

**Practical consequence: Learn Anywhere, Apply Everywhere**

Because $\beta_0, \beta_1$ are gauge-invariant:
1. **Learn** in DUASE space (from noisy spectral estimates $\hat{X}(t)$)
2. **Apply** the learned $\beta_0, \beta_1$ to TRUE initial conditions $X(0)$
3. **Compare** recovered trajectories directly to ground truth—no Procrustes alignment needed!

This was verified experimentally (Example 1): parameters learned from DUASE embeddings, when applied to true $X(0)$, produced trajectories with validation error 0.48 (MsgPass) vs 8.24 (black-box NN).

**Contrast with X-based equations:** If we had learned $\dot{X}_1 = aX_1 + bX_2$ directly, the coefficients $a, b$ would be gauge-dependent and could not be transferred between coordinate systems.

### 8.6 The Primary Evaluation Metric: $P(t)$

Since $P = XX^\top$ is rotation-invariant, comparing probability matrices is the honest test of dynamics recovery:

$$\text{P-error}(t) = \|P_{pred}(t) - P_{true}(t)\|$$

**Do NOT compare $X$ positions directly** (gauge-dependent). Even with Procrustes alignment, position comparison conflates dynamics errors with alignment errors.

**For truly gauge-invariant equations:** Regress on $\dot{P}_{ij}$ as functions of $P$:
$$\dot{P}_{ij} = \phi(P)$$

This directly models observable evolution without coordinate dependence. The tradeoff: higher dimensionality ($\frac{n(n+1)}{2}$ vs $nd$) and must respect tangent space constraints.

---

## 9. Obstructions to Learning

We have shown that invisible dynamics form a $\frac{d(d-1)}{2}$-dimensional space (Theorem 1). A natural question: can we learn *all* non-invisible dynamics, or are there further obstructions?

### 9.1 Theoretical Identifiability

**Theorem 6 (Identifiability modulo gauge).** Let $X \in \mathbb{R}^{n \times d}$ have full column rank. Given $\dot{P}$ and $X$, the vector field $f(X)$ is uniquely determined up to gauge:

$$f(X) = F + XA$$

where $F$ is any solution to $\dot{P} = FX^\top + XF^\top$ and $A \in \mathfrak{so}(d)$ is arbitrary.

*Proof.* The equation $\dot{P} = FX^\top + XF^\top$ is linear in $F$. Since $X$ has full column rank, the map $F \mapsto FX^\top + XF^\top$ has kernel exactly $\{XA : A \in \mathfrak{so}(d)\}$ (by Theorem 1). Thus solutions exist and are unique modulo this kernel. $\square$

**Corollary 3.** There is no theoretical obstruction beyond gauge freedom. Every non-invisible dynamics can be recovered from $\dot{P}$.

**Important caveats.** Theorem 6 assumes idealized conditions:
1. **Continuous observation** of $P(t)$—in practice, we have discrete samples
2. **Exact $P$**—in practice, we observe graphs sampled from $P$, then estimate $\hat{P}$
3. **Known $X$**—in practice, $X$ is estimated with error (~35% relative error typical)
4. **Full rank** $X(t)$ for all $t$—may fail near equilibria or degenerate configurations
5. **Infinite time horizon**—finite observations may not distinguish some dynamics

In practice, identifiability is limited by:
- Temporal resolution (discrete sampling rate)
- Estimation error in $\hat{P}$ from sampled graphs
- Noise in $\hat{X}$ from spectral embedding
- Numerical precision in computing $\dot{P}$ from finite differences

### 9.2 Constraint: Low-Rank Tangent Space

Not every symmetric matrix $\dot{P}$ can arise from RDPG dynamics.

**Proposition.** The image of $F \mapsto FX^\top + XF^\top$ has dimension $nd - \frac{d(d-1)}{2}$.

For $n = 10, d = 2$: image has dimension $19$, but symmetric $n \times n$ matrices have dimension $55$.

**Interpretation.** RDPG dynamics are constrained to the tangent space of the rank-$d$ positive semidefinite cone. This is not an obstruction to learning—it's a structural property. If $P(t) = X(t)X(t)^\top$ by construction, then $\dot{P}(t)$ automatically lies in this tangent space.

This constraint is actually useful: it means RDPG dynamics preserve low-rank structure, providing a strong inductive bias.

### 9.3 Standard Identifiability Issues

**Single trajectory.** From one trajectory $P(t)$ starting at $P(0)$, we only learn $f$ along that trajectory—not globally. Two vector fields $f, \tilde{f}$ agreeing on the trajectory but differing elsewhere are indistinguishable.

**Resolution:** Multiple initial conditions, parametric assumptions on $f$, or regularization. This is standard for dynamical systems learning, not specific to RDPG.

### 9.4 Practical Obstructions

In practice, several factors impede learning even theoretically identifiable dynamics:

**1. Estimation noise in $\hat{X}$.** SVD embedding introduces error. Even with repeated sampling, $\hat{X}$ has ~35% position error relative to truth (though $\hat{P}$ has only ~5% error).

**2. Gauge alignment artifacts.** Sequential Procrustes alignment can:
- Introduce spurious motion (if the alignment drifts)
- Remove real motion (if it resembles global rotation)
- Accumulate errors over long sequences

**3. Discrete, noisy observations.** We observe graphs sampled from $P$ at discrete times $t_1, \ldots, t_T$, not continuous $P(t)$. Estimating $\dot{P}$ from discrete $\hat{P}(t_k)$ introduces discretization error.

**4. Optimization landscape.** Neural ODEs may have difficult loss landscapes. Some dynamics may be hard to learn even if theoretically expressible.

**5. Finite network capacity.** Universal approximation is asymptotic. Finite networks may not capture all dynamics equally well.

### 9.5 The Position vs Probability Tradeoff

An interesting decoupling emerges from empirical observations:

| Quantity | Recovery quality |
|----------|------------------|
| Positions $X$ | ~35% relative error |
| Probability matrix $P$ | ~5% relative error |

This suggests two learning strategies:

**Strategy A: Learn $f: X \to \dot{X}$ (lift to parameterization)**
- Pros: Lower dimensional ($nd$ vs $\frac{n(n+1)}{2}$), standard Neural ODE
- Cons: Contaminated by gauge noise, coordinate-dependent

**Strategy B: Learn $\Phi: P \to \dot{P}$ (directly on observable)**
- Pros: Gauge-free, no alignment artifacts
- Cons: Higher dimensional, must respect PSD tangent space constraint

The practical success of Strategy A (as in your experiments) suggests that despite the gauge noise, the Neural ODE learns to predict $P$ correctly by implicitly respecting the constraint $\dot{P} = f(X)X^\top + Xf(X)^\top$.

### 9.6 Summary of Obstructions

| Obstruction | Type | Resolution |
|-------------|------|------------|
| Gauge freedom $\mathfrak{so}(d)$ | Theoretical | Quotient by gauge; learn equivalence classes |
| Single trajectory | Theoretical | Multiple initial conditions |
| Low-rank tangent space | Structural | Not an obstruction—a useful constraint |
| Estimation noise | Practical | More samples, better estimators |
| Procrustes artifacts | Practical | Joint embedding (UASE/Omnibus), careful alignment |
| Discrete observations | Practical | Finer time resolution, smoothing |
| Optimization difficulty | Practical | Architecture choice, regularization |

**Bottom line:** Theoretically, all non-invisible dynamics are identifiable. Practically, the main challenges are noise in $\hat{X}$ and alignment artifacts—both addressable with careful methodology.

---

## 10. Constraints on Realizable Dynamics

There are two independent sources of constraints on $\dot{P}$:

1. **Algebraic (rank preservation):** Follows from the factorization $P = XX^\top$
2. **Geometric (probability bounds):** Follows from requiring $P_{ij} \in [0,1]$

### 10.1 The State Space

**Valid configurations:**
$$\mathcal{X} = \{X \in \mathbb{R}^{n \times d} : (XX^\top)_{ij} \in [0,1] \text{ for all } i,j\}$$

This is the proper state space—not all of $\mathbb{R}^{n \times d}$.

**Interior vs boundary:** In the interior of $\mathcal{X}$, only algebraic constraints apply. On the boundary (where some $P_{ij} = 0$ or $P_{ij} = 1$), geometric constraints also apply.

### 10.2 Algebraic Constraint: Rank Preservation

The factorization $P = XX^\top$ with $X \in \mathbb{R}^{n \times d}$ means $\text{rank}(P) \leq d$. Since rank is preserved under continuous dynamics, $\dot{P}$ must be tangent to the rank-$d$ manifold.

**Key decomposition.** Let $V \in \mathbb{R}^{n \times d}$ be orthonormal columns spanning $\text{col}(P) = \text{col}(X) \subset \mathbb{R}^n$, and let $V_\perp \in \mathbb{R}^{n \times (n-d)}$ span the orthogonal complement in $\mathbb{R}^n$.

Note: This decomposition is in **node space** $\mathbb{R}^n$, not latent space $\mathbb{R}^d$. The column space of $P$ is a $d$-dimensional subspace of $\mathbb{R}^n$.

**Theorem 7 (Tangent space characterization).** The algebraically realizable $\dot{P}$ are exactly those satisfying:
$$V_\perp^\top \dot{P} \, V_\perp = 0$$

*Proof.* Any realizable $\dot{P} = FX^\top + XF^\top$ for some $F \in \mathbb{R}^{n \times d}$. Since $\text{col}(X) = \text{col}(V)$, we have $X = VR$ for invertible $R \in \mathbb{R}^{d \times d}$. Then:
$$V_\perp^\top \dot{P} \, V_\perp = V_\perp^\top F R^\top V^\top V_\perp + V_\perp^\top V R F^\top V_\perp = 0 + 0 = 0$$
using $V^\top V_\perp = 0$. Conversely, any symmetric $\dot{P}$ with $V_\perp^\top \dot{P} V_\perp = 0$ can be written in this form. $\square$

### 10.3 Understanding the Block Decomposition

Any symmetric $n \times n$ matrix $M$ decomposes as:
$$M = \underbrace{V A V^\top}_{\text{range-range}} + \underbrace{V B V_\perp^\top + V_\perp B^\top V^\top}_{\text{range-null cross}} + \underbrace{V_\perp C V_\perp^\top}_{\text{null-null}}$$

where $A \in \mathbb{R}^{d \times d}$ symmetric, $B \in \mathbb{R}^{d \times (n-d)}$, $C \in \mathbb{R}^{(n-d) \times (n-d)}$ symmetric.

For $\dot{P} = FX^\top + XF^\top$:
- $A$ and $B$ can be arbitrary (achieved by choosing $F$)
- $C = 0$ always (the algebraic constraint)

**Interpretation:**

| Block | Meaning | Realizable? |
|-------|---------|-------------|
| $VAV^\top$ | $\text{col}(P)$ unchanged, eigenvalues within it change | ✓ Yes |
| $VBV_\perp^\top + V_\perp B^\top V^\top$ | $\text{col}(P)$ rotates in $\mathbb{R}^n$ (still $d$-dimensional) | ✓ Yes |
| $V_\perp C V_\perp^\top$ | Structure in $\text{col}(P)^\perp$, would increase rank | ✗ No |

**The latent dimension $d$ never changes.** What can change is *which* $d$-dimensional subspace of $\mathbb{R}^n$ equals $\text{col}(P)$.

### 10.4 Concrete Example

$n = 3$ nodes, $d = 1$ factor. So $X = (x_1, x_2, x_3)^\top \in \mathbb{R}^3$ and $P = xx^\top$ is rank-1.

$\text{col}(P) = \text{span}\{x\} \subset \mathbb{R}^3$ — a line in node space.

**Case 1: Range-range ($\dot{x} \parallel x$)**

$\dot{x} = \alpha x$ gives $\dot{P} = 2\alpha xx^\top$. The line $\text{col}(P)$ doesn't move; magnitude changes.

**Case 2: Range-null cross ($\dot{x} \perp x$)**

$\dot{x} = v$ where $v \perp x$ gives $\dot{P} = vx^\top + xv^\top$.

After time $\delta t$: $x(\delta t) \approx x + v \delta t$, so $\text{col}(P(\delta t)) = \text{span}\{x + v \delta t\}$.

The line **rotates** in $\mathbb{R}^3$. Still rank-1 (still $d = 1$), but pointing differently.

**Case 3: Null-null (impossible)**

$\dot{P} = ww^\top$ where $w \perp x$. This would add rank, making $P + \dot{P} \cdot \delta t$ have rank 2.

No choice of $\dot{x}$ can produce this, because $\dot{P} = \dot{x}x^\top + x\dot{x}^\top$ always has $x$ in every term.

### 10.5 Dimension Count

**Corollary 4.** 
$$\dim(T_P\mathcal{M}_d) = nd - \frac{d(d-1)}{2}$$

| Space | Dimension | Interpretation |
|-------|-----------|----------------|
| Symmetric $n \times n$ | $\frac{n(n+1)}{2}$ | All possible $\dot{P}$ |
| Tangent space | $nd - \frac{d(d-1)}{2}$ | Algebraically realizable $\dot{P}$ |
| Null-null block | $\frac{(n-d)(n-d+1)}{2}$ | Algebraically unrealizable |

**Example ($n = 10$, $d = 2$):**
- All symmetric: 55 dimensions
- Tangent space: $20 - 1 = 19$ dimensions
- Unrealizable: 36 dimensions

Most symmetric perturbations of $P$ are not achievable by RDPG dynamics!

### 10.6 Geometric Constraint: Probability Bounds

On the boundary of $\mathcal{X}$, additional constraints apply:

**At $P_{ij} = 0$:** Require $\dot{P}_{ij} \geq 0$ (can't go negative)

**At $P_{ij} = 1$:** Require $\dot{P}_{ij} \leq 0$ (can't exceed 1)

These are **independent** of the algebraic constraint. A $\dot{P}$ can satisfy the tangent space condition ($C = 0$) but still be geometrically forbidden because it would push some $P_{ij}$ outside $[0,1]$.

### 10.7 Summary: Two Sources of Constraint

| Source | Constraint | When active | Nature |
|--------|-----------|-------------|--------|
| Algebraic | $V_\perp^\top \dot{P} V_\perp = 0$ | Always | Rank preservation |
| Geometric | $\dot{P}_{ij} \geq 0$ when $P_{ij} = 0$ | At lower boundary | Probability bound |
| Geometric | $\dot{P}_{ij} \leq 0$ when $P_{ij} = 1$ | At upper boundary | Probability bound |

**In the interior of $\mathcal{X}$:** Only the algebraic constraint matters.

**On the boundary of $\mathcal{X}$:** Both constraints apply.

### 10.8 Gauge-Free Decomposition of $\dot{X}$

Any velocity $\dot{X}$ decomposes uniquely as:
$$\dot{X} = X \cdot A + W$$
where $A = (X^\top X)^{-1}X^\top \dot{X} \in \mathbb{R}^{d \times d}$ and $W \perp \text{col}(X)$.

Further decompose $A = A_{\text{sym}} + A_{\text{skew}}$:

| Component | Contributes to $\dot{P}$? | Interpretation |
|-----------|---------------------------|----------------|
| $X \cdot A_{\text{sym}}$ | Yes | Radial/stretching dynamics |
| $X \cdot A_{\text{skew}}$ | No | Pure rotation (gauge) |
| $W$ | Yes | Rotates $\text{col}(P)$ in $\mathbb{R}^n$ |

**The observable content of $\dot{X}$ is $(A_{\text{sym}}, W)$.**

### 10.9 Remark: Algebraic Constraint and Dimensional Changes

The algebraic constraint $V_\perp^\top \dot{P} V_\perp = 0$ has a natural interpretation: violation would indicate that the latent dimension is increasing (a new factor is emerging). However, the practical utility of this observation is limited.

**The circularity.** To compute $V_\perp$, we need to know $d$—we must eigendecompose $P$ and decide which eigenvectors span the "signal" subspace ($V$) versus the "null" subspace ($V_\perp$). This requires either:
- Assuming $d$ is known, or
- Estimating $d$ from the data (e.g., via eigenvalue gap, profile likelihood, cross-validation)

But if we can estimate $d$ reliably enough to construct $\hat{V}_\perp$, we could simply use static model selection methods directly. The dynamic diagnostic doesn't provide information beyond what's already needed to set it up.

**What remains.** The observation is mathematically correct: if the true dynamics have $V_\perp^\top \dot{P} V_\perp \neq 0$, then rank is increasing and no fixed-$d$ RDPG model can capture the dynamics. This is a consistency check rather than a discovery tool:

- If we fit a $d$-dimensional model and find systematic residuals in the null-null block of $\dot{\hat{P}}$, this suggests model inadequacy
- But detecting this requires already having estimated $d$

**Possible future directions.** There might be settings where the dynamic signature is detectable before or more robustly than static dimensional estimates—for instance, if a new dimension is "emerging" gradually and affects $\dot{P}$ before it substantially affects $P$ itself. This would require careful study of the relative statistical power of dynamic versus static tests, which we leave to future work.

For now, we note this as a theoretical consistency condition rather than a practical diagnostic tool.

---

## 11. UDE Architecture for RDPG Dynamics

### 11.1 The Main Theorem

**Theorem 8 (General equivariant dynamics).** Let $X \in \mathcal{X}$ have full column rank. Any $O(d)$-equivariant vector field $f: \mathcal{X} \to \mathbb{R}^{n \times d}$ has the form:

$$\boxed{f(X) = N(P) \cdot X}$$

where $N: \{P \in \mathbb{R}^{n \times n} : P = XX^\top, X \in \mathcal{X}\} \to \mathbb{R}^{n \times n}$ is a matrix-valued function.

*Proof.* 

**Existence of $N$:** For full-rank $X$, define $N(X) := f(X) X^\dagger$ where $X^\dagger = (X^\top X)^{-1}X^\top$ is the Moore-Penrose right pseudoinverse. Then $f(X) = N(X) \cdot X$ since $X^\dagger X = I_d$.

**$N$ is constant on orbits:** For any $Q \in O(d)$:
$$N(XQ) = f(XQ)(XQ)^\dagger = f(X)Q \cdot Q^\top X^\dagger = f(X)X^\dagger = N(X)$$

using equivariance $f(XQ) = f(X)Q$ and $(XQ)^\dagger = Q^\top X^\dagger$.

**$N$ depends only on $P$:** Since $N(XQ) = N(X)$ for all $Q \in O(d)$, and orbits $\{XQ : Q \in O(d)\}$ are indexed by $P = XX^\top$, we have $N = N(P)$.

**Equivariance verification:** $f(XQ) = N(P) \cdot XQ = N(P) X \cdot Q = f(X) \cdot Q$. ✓ $\square$

### 11.2 Gauge Elimination via Symmetry

**Theorem 9 (Gauge is not symmetric).** The gauge dynamics $\dot{X} = XA$ with $A \in \mathfrak{so}(d)$ correspond to $N = XAX^\dagger$. For generic full-rank $X$ and nonzero $A$, this $N$ is **not symmetric**.

*Proof.* Consider first $n = d$ and $X = I_d$. Then $X^\dagger = I_d$ and $N = A$. Since $A$ is skew-symmetric ($A^\top = -A$), $N$ is not symmetric unless $A = 0$.

For general full-rank $X$, let $X = U\Sigma V^\top$ be the thin SVD with $U \in \mathbb{R}^{n \times d}$, $\Sigma \in \mathbb{R}^{d \times d}$ invertible, $V \in O(d)$. Then $X^\dagger = V\Sigma^{-1}U^\top$ and:
$$N = U\Sigma V^\top A V \Sigma^{-1} U^\top = U B U^\top$$

where $B = \Sigma (V^\top A V) \Sigma^{-1}$. Since $V^\top A V$ is skew-symmetric, we have:
$$B^\top = (\Sigma^{-1})^\top (V^\top A V)^\top \Sigma^\top = \Sigma^{-1} (V^\top A^\top V) \Sigma = -\Sigma^{-1} (V^\top A V) \Sigma$$

For $B = B^\top$, we need $\Sigma (V^\top A V) \Sigma^{-1} = -\Sigma^{-1} (V^\top A V) \Sigma$, i.e., $\Sigma^2 (V^\top A V) = -(V^\top A V) \Sigma^2$.

This says $V^\top A V$ anti-commutes with $\Sigma^2$. For generic $\Sigma$ (distinct singular values), the only matrix anti-commuting with $\Sigma^2$ is zero. Hence $V^\top A V = 0$, implying $A = 0$. $\square$

**Theorem 10 (Symmetric $N$ implies observable).** Let $N = N^\top$ be symmetric. If $NX \neq 0$, then $\dot{P} = NP + PN \neq 0$.

*Proof.* Suppose $N = N^\top$ and $\dot{P} = NP + PN = 0$.

Let $P = V\Lambda V^\top$ be the spectral decomposition with $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d, 0, \ldots, 0)$ and $\lambda_i > 0$ for $i \leq d$.

Define $\tilde{N} = V^\top N V$ (symmetric since $N$ is). The condition $NP + PN = 0$ transforms to:
$$\tilde{N}\Lambda + \Lambda\tilde{N} = 0$$

Entry-wise: $\tilde{N}_{ij}(\lambda_i + \lambda_j) = 0$.

**Case analysis:**
- For $i, j \leq d$: $\lambda_i + \lambda_j > 0$, so $\tilde{N}_{ij} = 0$.
- For $i \leq d$, $j > d$: $\lambda_i + 0 = \lambda_i > 0$, so $\tilde{N}_{ij} = 0$.
- For $i > d$, $j \leq d$: by symmetry $\tilde{N}_{ij} = \tilde{N}_{ji} = 0$.
- For $i, j > d$: the constraint $0 \cdot \tilde{N}_{ij} = 0$ is vacuous.

Therefore $\tilde{N} = \begin{pmatrix} 0 & 0 \\ 0 & \tilde{N}_{22} \end{pmatrix}$ where $\tilde{N}_{22} \in \mathbb{R}^{(n-d) \times (n-d)}$ is arbitrary symmetric.

Since $\text{col}(X) = \text{col}(V_1)$ where $V_1$ comprises the first $d$ columns of $V$, we can write $X = V_1 R$ for invertible $R$. Then:
$$NX = V\tilde{N}V^\top V_1 R = V\tilde{N} \begin{pmatrix} I_d \\ 0 \end{pmatrix} R = V \begin{pmatrix} 0 \\ 0 \end{pmatrix} = 0$$

**Contrapositive:** If $N = N^\top$ and $NX \neq 0$, then $NP + PN \neq 0$, so $\dot{P} \neq 0$. $\square$

**Corollary 5.** Constraining $N(P) = N(P)^\top$ (symmetric) eliminates all non-trivial gauge freedom. Any symmetric $N$ with $NX \neq 0$ produces observable dynamics.

### 11.3 Symmetric $N$ Spans the Tangent Space

**Theorem 10 (Tangent space parameterization).** The map 
$$N \mapsto \dot{P} = NP + PN$$
from symmetric $N \in \mathbb{R}^{n \times n}$ to symmetric $\dot{P}$ has:

1. **Kernel:** Symmetric matrices supported on $\text{null}(P)$, dimension $\frac{(n-d)(n-d+1)}{2}$
2. **Image:** The full tangent space $T_P\mathcal{M}_d$, dimension $nd - \frac{d(d-1)}{2}$

*Proof.* In the eigenbasis $P = V\Lambda V^\top$ with $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d, 0, \ldots, 0)$:

The equation $\tilde{N}\Lambda + \Lambda\tilde{N} = 0$ (where $N = V\tilde{N}V^\top$) gives $\tilde{N}_{ij}(\lambda_i + \lambda_j) = 0$.

- For $i, j \leq d$: $\lambda_i + \lambda_j > 0$, forcing $\tilde{N}_{ij} = 0$
- For $i \leq d, j > d$: $\lambda_i > 0$, forcing $\tilde{N}_{ij} = 0$
- For $i, j > d$: constraint is $0 = 0$, so $\tilde{N}_{ij}$ is free

Kernel dimension: $\frac{(n-d)(n-d+1)}{2}$.

**Crucially:** If $N$ is supported on $\text{null}(P)$, then $NX = 0$ since $\text{col}(X) \subseteq \text{col}(P)$. So the kernel produces **zero dynamics**, not invisible dynamics.

Image dimension: $\frac{n(n+1)}{2} - \frac{(n-d)(n-d+1)}{2} = nd - \frac{d(d-1)}{2}$. ✓ $\square$

### 11.4 Rank Preservation

**Proposition.** The dynamics $\dot{X} = N(P)X$ with symmetric $N$ preserve the rank of $X$.

*Proof.* The induced $\dot{P} = NP + PN$ lies in the tangent space to the rank-$d$ manifold, since $\dot{P} = FX^\top + XF^\top$ with $F = NX$. $\square$

### 11.5 The Complete UDE Structure

For RDPG dynamics:

$$\boxed{\dot{X} = N(P) \cdot X, \quad N(P) = N(P)^\top}$$

**Properties:**
| Property | How enforced |
|----------|--------------|
| $O(d)$-equivariance | $N$ depends only on $P$ |
| Gauge-free | $N$ symmetric |
| Rank-preserving | Automatic (tangent to manifold) |
| Tangent space coverage | Full (Theorem 10) |

**UDE decomposition:**
$$N(P) = N_{\text{known}}(P) + N_{\text{NN}}(P)$$

where both terms are symmetric.

### 11.6 Concrete Parameterizations for $N(P)$

Since $N$ must be symmetric and depend on $P$, natural building blocks include:

**Polynomial in $P$:**
$$N(P) = \alpha_0 I + \alpha_1 P + \alpha_2 P^2 + \cdots$$

Always symmetric since $P$ is symmetric.

**Spectral functions:**
$$N(P) = g(P) = V \, g(\Lambda) \, V^\top$$

where $P = V\Lambda V^\top$ and $g$ acts on eigenvalues.

**Degree-based:**
$$N(P) = \text{diag}(h(P\mathbf{1}))$$

where $h$ is a scalar function and $P\mathbf{1}$ is the degree vector.

**Laplacian-inspired:**
$$N(P) = D^{-1/2} P D^{-1/2} - I$$

where $D = \text{diag}(P\mathbf{1})$.

**Pairwise interaction:**
$$N_{ij}(P) = g(P_{ij}) \quad \text{for } i \neq j, \quad N_{ii}(P) = h(P_{ii})$$

where $g, h$ are scalar functions (can be neural networks).

**General neural network:**

The NN takes $P$ (or features derived from $P$) as input and outputs $\frac{n(n+1)}{2}$ values representing the upper triangle of symmetric $N$.

### 11.7 Physical Interpretation

The dynamics $\dot{X} = NX$ have a clear interpretation:

- **Row $i$:** $\dot{X}_i = \sum_j N_{ij} X_j$

Node $i$'s velocity is a weighted combination of all nodes' positions, with weights $N_{ij}$.

- **$N_{ij} > 0$:** Node $i$'s position moves toward node $j$'s position direction
- **$N_{ij} < 0$:** Node $i$'s position moves away from node $j$'s position direction
- **$N_{ii}$:** Self-interaction (radial expansion/contraction)

The symmetry $N_{ij} = N_{ji}$ means the interaction is **reciprocal**: the influence of $j$ on $i$ equals the influence of $i$ on $j$.

### 11.8 Induced Dynamics on $P$

The probability matrix evolves as:
$$\dot{P} = NP + PN$$

This is a **Lyapunov equation**. Properties:

- If $N$ has all negative eigenvalues: $P \to 0$ (network dissolves)
- If $N$ has all positive eigenvalues: $P$ grows (network densifies)
- Mixed eigenvalues: complex dynamics

**Equilibria:** $\dot{P} = 0$ requires $NP + PN = 0$. For symmetric $N$ and $P$, this means $NP$ is skew-symmetric, which generically requires $N = 0$ or special structure.

### 11.9 Implementation

```julia
function rdpg_ude(X, p_known, p_nn)
    P = X * X'
    n = size(X, 1)
    
    # Known physics: polynomial in P
    N_known = p_known.α₀ * I(n) + p_known.α₁ * P
    
    # Learned: NN outputs upper triangle of symmetric matrix
    upper_tri = nn(vec(P), p_nn)  # length n(n+1)/2
    N_nn = upper_triangular_to_symmetric(upper_tri, n)
    
    N = N_known + N_nn
    return N * X
end

function upper_triangular_to_symmetric(v, n)
    N = zeros(eltype(v), n, n)
    k = 1
    for i in 1:n
        for j in i:n
            N[i,j] = v[k]
            N[j,i] = v[k]
            k += 1
        end
    end
    return N
end
```

### 11.10 Connection to Previous Formulations

The formulation $\dot{X} = N(P) X$ is equivalent to but simpler than the decomposition $\dot{X} = X S(X) + W(X)$ used earlier in the literature.

**Relationship:** If $\dot{X} = N X$, we can decompose:
- $S = X^\dagger N X = (X^\top X)^{-1} X^\top N X$ (the $d \times d$ "coefficient matrix")
- $W = N X - X S = (I - X X^\dagger) N X = \Pi_X^\perp N X$ (the null-space component)

The constraint that $N$ be symmetric ensures:
- $S$ is symmetric (no gauge waste)
- The dynamics respect the tangent space structure

But the $N(P) X$ formulation is more direct: **$N$ is simply a symmetric $n \times n$ matrix depending on $P$**.

---

## 12. Extension to Directed Graphs

For directed graphs, the probability matrix factors as $P = LR^\top$ where $L, R \in \mathbb{R}^{n \times d}$ are **left** (source) and **right** (target) embeddings.

### 12.1 Gauge Group

The gauge transformation is $(L, R) \mapsto (LQ, RQ)$ for $Q \in O(d)$:
$$(LQ)(RQ)^\top = LQQ^\top R^\top = LR^\top = P$$

**Crucially:** Both embeddings rotate by the **same** $Q$. This is because $P_{ij} = L_i \cdot R_j$ must be preserved, and $L_i \cdot R_j = (L_i Q) \cdot (R_j Q)$ only if the same rotation acts on both.

### 12.2 Gauge-Invariant Quantities

Under $(L, R) \mapsto (LQ, RQ)$:

| Quantity | Transformation | Invariant? |
|----------|----------------|------------|
| $P = LR^\top$ | $LQQ^\top R^\top = LR^\top$ | ✓ Yes |
| $G_L = LL^\top$ | $LQQ^\top L^\top = LL^\top$ | ✓ Yes |
| $G_R = RR^\top$ | $RQQ^\top R^\top = RR^\top$ | ✓ Yes |
| $L^\top L$ | $Q^\top(L^\top L)Q$ | ✗ Conjugation |
| $R^\top R$ | $Q^\top(R^\top R)Q$ | ✗ Conjugation |
| $L^\top R$ | $Q^\top(L^\top R)Q$ | ✗ Conjugation |

The **gauge-invariant data** is $(P, G_L, G_R)$—three $n \times n$ matrices.

*Note:* $P$ is generally **not symmetric** for directed graphs. The diagonal $P_{ii} = L_i \cdot R_i$ represents self-loop probability.

### 12.3 Equivariant Dynamics

**Theorem 11 (Directed equivariant dynamics).** Let $L, R \in \mathbb{R}^{n \times d}$ both have full column rank. Any $O(d)$-equivariant vector field $(f_L, f_R)$ on $(L, R)$ has the form:

$$\dot{L} = N_L(P, G_L, G_R) \cdot L, \quad \dot{R} = N_R(P, G_L, G_R) \cdot R$$

where $N_L, N_R: \mathbb{R}^{n \times n} \times \mathbb{R}^{n \times n} \times \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$.

*Proof.* Apply Theorem 8 separately to $L$ and $R$. For $L$: define $N_L := f_L(L,R) L^\dagger$. Then:
$$N_L(LQ, RQ) = f_L(LQ, RQ)(LQ)^\dagger = f_L(L,R)Q \cdot Q^\top L^\dagger = f_L(L,R)L^\dagger = N_L(L,R)$$

So $N_L$ is constant on $O(d)$-orbits. The orbit $(L,R) \mapsto (LQ, RQ)$ is characterized by the invariants $(P, G_L, G_R)$, so $N_L = N_L(P, G_L, G_R)$. Similarly for $N_R$. $\square$

### 12.4 Induced Dynamics on $P$

$$\dot{P} = \dot{L}R^\top + L\dot{R}^\top = N_L LR^\top + LR^\top N_R^\top = N_L P + P N_R^\top$$

**Contrast with undirected case:** 
- Undirected: $\dot{P} = NP + PN^\top = NP + PN$ (one symmetric $N$)
- Directed: $\dot{P} = N_L P + P N_R^\top$ (two independent matrices)

### 12.5 Gauge Characterization

The invisible dynamics are $\dot{L} = LA$, $\dot{R} = RA$ for $A \in \mathfrak{so}(d)$ (same $A$ for both).

**Verification:**
$$\dot{P} = LAR^\top + LR^\top A^\top = LAR^\top + L(-A)R^\top = LAR^\top - LAR^\top = 0$$ ✓

The corresponding $N$ matrices are $N_L = LAL^\dagger$ and $N_R = RAR^\dagger$.

**Theorem 12 (Directed gauge is not symmetric).** For generic full-rank $L, R$ and nonzero $A \in \mathfrak{so}(d)$:
- $N_L = LAL^\dagger$ is not symmetric
- $N_R = RAR^\dagger$ is not symmetric

*Proof.* Same spectral argument as Theorem 9, applied to $L$ and $R$ separately. $\square$

### 12.6 Symmetric Coupling Eliminates Gauge

**Theorem 13 (Strongest gauge elimination).** Suppose $N_L = N_R = N$ (same transformation for both) and $N = N^\top$ (symmetric). If $NL \neq 0$ or $NR \neq 0$, then $\dot{P} \neq 0$.

*Proof.* With $N_L = N_R = N$ symmetric:
$$\dot{P} = NP + PN^\top = NP + PN$$

Suppose $\dot{P} = 0$, so $NP = -PN$, i.e., $N$ and $P$ anti-commute.

Taking transpose: $(NP)^\top = (-PN)^\top$, so $P^\top N = -N P^\top$.

Now $N$ anti-commutes with both $P$ and $P^\top$. Let $P = U\Sigma V^\top$ be the SVD with $\Sigma$ having $d$ nonzero singular values.

From $NP = -PN$: $NU\Sigma V^\top = -U\Sigma V^\top N$, so $U^\top N U \Sigma = -\Sigma V^\top N V$.

From $NP^\top = -P^\top N$: $NV\Sigma^\top U^\top = -V\Sigma^\top U^\top N$, so $V^\top N V \Sigma^\top = -\Sigma^\top U^\top N U$.

For generic $P$ (distinct singular values), these constraints force $N$ to have support only on the null spaces of both $P$ and $P^\top$. Since $\text{col}(L) \subseteq \text{col}(P^\top)$ and $\text{col}(R) \subseteq \text{col}(P)$, this implies $NL = NR = 0$.

**Contrapositive:** If $N = N^\top$ with $N_L = N_R = N$, and $NL \neq 0$ or $NR \neq 0$, then $\dot{P} \neq 0$. $\square$

**Corollary 6.** For directed graphs, requiring $N_L = N_R = N$ with $N$ symmetric reduces the dynamics to the undirected form and eliminates gauge.

### 12.7 Asymmetric Directed Dynamics

For truly directed dynamics, we may want $N_L \neq N_R$:

$$\dot{L} = N_L(P, G_L, G_R) \cdot L, \quad \dot{R} = N_R(P, G_L, G_R) \cdot R$$

**Gauge elimination (weaker form):** Require $N_L = N_L^\top$ and $N_R = N_R^\top$ individually. Since $LAL^\dagger$ and $RAR^\dagger$ are not symmetric for generic $L, R$ and nonzero $A$, this eliminates gauge.

**Physical interpretation:**
- $N_L$ controls **source** (out-link) propensity evolution
- $N_R$ controls **target** (in-link) propensity evolution

Example: In a citation network:
- Citing papers (sources) may cite more aggressively early, then slow: $N_L$ has time-dependent decay
- Highly-cited papers (targets) continue accumulating citations: $N_R$ has positive diagonal

### 12.8 Induced Dynamics on $P$ (Asymmetric Case)

With general $N_L \neq N_R$:
$$\dot{P} = N_L P + P N_R^\top$$

This is a **Sylvester equation** in disguise. The dynamics can produce:
- Asymmetric growth patterns
- Directed community formation
- Source-target differentiation

### 12.9 Comparison: Undirected vs Directed

| Aspect | Undirected | Directed |
|--------|------------|----------|
| State | $X \in \mathbb{R}^{n \times d}$ | $(L, R) \in \mathbb{R}^{n \times d} \times \mathbb{R}^{n \times d}$ |
| Probability | $P = XX^\top$ (symmetric) | $P = LR^\top$ (asymmetric) |
| Gauge group | $O(d)$ via $X \mapsto XQ$ | $O(d)$ via $(L,R) \mapsto (LQ, RQ)$ |
| Invariants | $P$ alone | $(P, G_L, G_R)$ |
| Dynamics | $\dot{X} = N(P)X$ | $\dot{L} = N_L(\cdot)L$, $\dot{R} = N_R(\cdot)R$ |
| Gauge-free constraint | $N = N^\top$ | $N_L = N_L^\top$, $N_R = N_R^\top$ |
| Induced $\dot{P}$ | $NP + PN$ | $N_L P + P N_R^\top$ |
| Free parameters | $\frac{n(n+1)}{2}$ | $n(n+1)$ (two symmetric matrices) |

### 12.10 The Directed UDE

$$\boxed{\dot{L} = N_L(P, G_L, G_R) \cdot L, \quad \dot{R} = N_R(P, G_L, G_R) \cdot R}$$

with $N_L = N_L^\top$ and $N_R = N_R^\top$ to eliminate gauge.

**Special cases:**
- **Coupled:** $N_L = N_R = N(P)$ — reduces to undirected-like structure
- **Decoupled:** $N_L = N_L(G_L)$, $N_R = N_R(G_R)$ — independent evolution
- **Cross-coupled:** $N_L$ depends on $G_R$, $N_R$ depends on $G_L$ — sources respond to targets and vice versa

---

## 13. The Complete Picture

```
Graphs (observable)
    ↑ sample from
Probability matrix P (observable)
    ↑ P = XX^T (undirected) or P = LR^T (directed)
Latent positions X or (L,R) (parameterization, not unique)
    ↑ equivalence class
[X] = {XQ : Q ∈ O(d)} (the "true" object)
```

### 13.1 Summary of Main Results

**For undirected graphs ($P = XX^\top$):**

| Result | Statement |
|--------|-----------|
| Theorem 8 | Most general equivariant dynamics: $\dot{X} = N(P) X$ |
| Theorem 9 | Gauge $N = XAX^\dagger$ is not symmetric for generic $X$, nonzero $A \in \mathfrak{so}(d)$ |
| Theorem 10 | Symmetric $N$ with $NX \neq 0$ implies $\dot{P} \neq 0$ (observable) |
| Corollary 5 | Symmetric $N$ eliminates all non-trivial gauge |

**For directed graphs ($P = LR^\top$):**

| Result | Statement |
|--------|-----------|
| Theorem 11 | Most general equivariant dynamics: $\dot{L} = N_L(\cdot)L$, $\dot{R} = N_R(\cdot)R$ |
| Theorem 12 | Gauge $N_L = LAL^\dagger$, $N_R = RAR^\dagger$ not symmetric generically |
| Theorem 13 | $N_L = N_R = N$ symmetric with $NL \neq 0$ or $NR \neq 0$ implies $\dot{P} \neq 0$ |
| Corollary 6 | Coupled symmetric dynamics reduces to undirected form |

### 13.2 The UDE Recipe

1. **Choose state space:** 
   - Undirected: $X \in \mathbb{R}^{n \times d}$
   - Directed: $(L, R) \in \mathbb{R}^{n \times d} \times \mathbb{R}^{n \times d}$

2. **Identify invariants:**
   - Undirected: $P = XX^\top$
   - Directed: $(P, G_L, G_R) = (LR^\top, LL^\top, RR^\top)$

3. **Parameterize $N$:** Symmetric $n \times n$ matrix depending on invariants

4. **Decompose:** $N = N_{\text{known}} + N_{\text{NN}}$
   - Known physics: Polynomial in $P$, degree-based, Laplacian, etc.
   - Learned part: NN outputs $\frac{n(n+1)}{2}$ values (upper triangle)

5. **Write dynamics:**
   - Undirected: $\dot{X} = N(P) X$
   - Directed: $\dot{L} = N_L(\cdot) L$, $\dot{R} = N_R(\cdot) R$

6. **Verify:**
   - Gauge-free: $N$ (or $N_L$, $N_R$) symmetric
   - Observable: $NX \neq 0$ (or $N_LL \neq 0$, $N_RR \neq 0$)

### 13.3 What We Can and Cannot Learn

**Learnable from $P(t)$:**
- The symmetric matrix $N(P)$ (up to the null-space kernel)
- Eigenvalues and qualitative behavior
- Equilibrium structure

**Not learnable (gauge freedom):**
- Skew-symmetric component of any putative $\dot{X}/X$ decomposition
- Absolute orientation in latent space
- Coordinate-dependent form of equations (only invariant structure)

### 13.4 Connection to Practice

**On gauge choices:** Temporal embedding methods (DUASE, Omnibus, Procrustes chains) select consistent representatives from equivalence classes, enabling coordinate-based Neural ODE training with position-based loss functions.

**On $B^d_+$:** The constraint $X \in (B^d_+)^n$ is convenient for mathematical analysis (ensuring $P_{ij} \in [0,1]$), but is **not required for numerical learning**. Projecting to $B^d_+$ can distort geometry. For $N(P)X$ dynamics, learning in the embedding method's natural coordinate system works well.

**Gauge-invariant scalars:** For dynamics $\dot{X} = N(P)X$, scalar parameters in $N$ (e.g., $\beta_0, \beta_1$ in message-passing) are gauge-invariant. This enables:
- **Learn anywhere**: Train on estimated embeddings $\hat{X}(t)$
- **Apply everywhere**: Use learned parameters with true initial conditions $X(0)$
- **Direct comparison**: No Procrustes alignment needed for evaluation

The theory guarantees that any dynamics learned with symmetric $N$ will:
- Produce correct predictions of $P(t)$
- Have gauge-invariant scalar parameters
- Cover the full tangent space of achievable $\dot{P}$
