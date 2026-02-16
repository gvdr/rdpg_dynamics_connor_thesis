# Parsimonious UDE Parameterizations for RDPG Dynamics

## 1. The Setup

### 1.1 State Space

**Latent positions:** $X \in \mathbb{R}^{n \times d}$ for $n$ nodes in $d$ dimensions.

**Observable:** $P = XX^\top$ (probability matrix), requiring $P_{ij} \in [0,1]$.

**The constraint:** Not all $X \in \mathbb{R}^{n \times d}$ are valid. Define:

$$B^d_+ = \{x \in \mathbb{R}^d : x \geq 0, \|x\| \leq 1\}$$

If $X_i \in B^d_+$ for all $i$, then $P_{ij} \in [0,1]$ (by non-negativity of coordinates and Cauchy-Schwarz).

**Valid configuration space:**

$$\mathcal{X} = \{X \in \mathbb{R}^{n \times d} : (XX^\top)_{ij} \in [0,1] \text{ for all } i,j\}$$

**Caution:** $(B^d_+)^n$ is NOT a fundamental domain in general. For $n$ vectors, we need one $Q \in O(d)$ that works for ALL rows simultaneously—this may be impossible (e.g., two orthogonal vectors in $d=2$ cannot both have non-negative coordinates after any rotation).

### 1.2 Gauge Freedom

For $Q \in O(d)$: $(XQ)(XQ)^\top = P$, so $X$ and $XQ$ are equivalent.

**Gauge-free dynamics:** $\dot{X} = N(P)X$ with $N = N^\top$ (symmetric).

### 1.3 The UDE Goal

Parameterize $N(P)$ such that:
- Symmetric (automatic gauge elimination)
- Parsimonious (far fewer than $n^2$ parameters)
- Expressive enough for relevant dynamics
- Can encode qualitative priors

**Output dimension:** We're learning $\dot{X} \in \mathbb{R}^{n \times d}$, i.e., $nd$ values—not $n^2$.

**On $B^d_+$ constraints:** The constraint $X \in (B^d_+)^n$ ensures $P_{ij} \in [0,1]$ and is convenient for mathematical analysis. However, it is **not required for numerical learning**. Projecting estimated embeddings to $B^d_+$ can distort geometry and break temporal consistency. For $N(P)X$ dynamics, learning in whatever coordinate system the embedding method naturally produces (e.g., DUASE) works well. Constraint enforcement via barrier losses (Section 9) is optional and only needed if probability bounds are violated.

### 1.4 Gauge-Invariant Scalars

**Key insight:** Scalar parameters in $N(P)$ are gauge-invariant. For example, in message-passing dynamics:
$$\dot{X}_i = \beta_0 X_i + \beta_1 \sum_j P_{ij}(X_j - X_i)$$

The scalars $\beta_0, \beta_1$ do not depend on the coordinate system because $N$ depends only on $P = XX^\top$, which is rotation-invariant.

**Practical consequence:** Learn parameters from noisy estimates $\hat{X}(t)$, then apply them to true initial conditions $X(0)$ for direct comparison—no Procrustes alignment needed.

---

## 2. Taxonomy of Parameterizations

### 2.1 By Homogeneity

| Type | Description | Parameters |
|------|-------------|------------|
| **Homogeneous** | All node pairs follow same rule | $O(1)$ functions |
| **Type-based** | Nodes grouped into $K$ types | $O(K^2)$ functions |
| **Node-specific** | Each node has own parameters | $O(n)$ scalars or functions |
| **Fully heterogeneous** | Each pair has own parameter | $O(n^2)$ — avoid! |

### 2.2 By Structure

| Structure | Form of $N_{ij}$ | Symmetry |
|-----------|------------------|----------|
| **Diagonal** | $N_{ij} = \delta_{ij} h_i$ | Automatic |
| **Kernel** | $N_{ij} = \kappa(P_{ij}, P_{ii}, P_{jj})$ | If $\kappa$ symmetric in $i \leftrightarrow j$ |
| **Spectral** | $N = f(P)$ via eigendecomposition | Automatic |
| **Low-rank** | $N = \sum_k u_k u_k^\top$ | Automatic |
| **Graph-based** | $N_{ij} = g(P_{ij}) \cdot \mathbf{1}[P_{ij} > \epsilon]$ | If $g$ symmetric |

---

## 3. Homogeneous Parameterizations

All nodes and pairs follow the same rules. Most parsimonious.

### 3.1 Scalar × Identity

$$N(P) = \alpha(P) \cdot I_n$$

**Dynamics:** $\dot{X}_i = \alpha(P) X_i$ — uniform radial scaling.

**Parameters:** One scalar function $\alpha: \mathbb{R}^{n \times n} \to \mathbb{R}$.

**Simplest:** $\alpha = \alpha_0$ constant, or $\alpha = \alpha_0 + \alpha_1 \bar{P}$ where $\bar{P} = \frac{1}{n^2}\sum_{ij} P_{ij}$.

**Behavior:** 
- $\alpha < 0$: contraction (network sparsifies)
- $\alpha > 0$: expansion (network densifies)

### 3.2 Polynomial in $P$

$$N(P) = \alpha_0 I + \alpha_1 P + \alpha_2 P^2 + \cdots + \alpha_k P^k$$

**Parameters:** $k+1$ scalars.

**Dynamics:** Node $i$'s velocity depends on neighbors (via $P$), neighbors-of-neighbors (via $P^2$), etc.

**Interpretation:**
- $\alpha_0 I$: self-dynamics
- $\alpha_1 P$: direct neighbor influence
- $\alpha_2 P^2$: two-hop influence

**UDE form:** 
$$N(P) = \underbrace{(\alpha_0 I + \alpha_1 P)}_{\text{known: local}} + \underbrace{\theta_{\text{NN}} \cdot P^2}_{\text{learned: nonlocal correction}}$$

### 3.3 Spectral Function

$$N(P) = V f(\Lambda) V^\top$$

where $P = V \Lambda V^\top$ is the eigendecomposition and $f$ acts on eigenvalues.

**Parameters:** The function $f: \mathbb{R}_{\geq 0} \to \mathbb{R}$.

**Examples:**
- $f(\lambda) = \alpha \lambda$: recovers $N = \alpha P$
- $f(\lambda) = \alpha / (1 + \lambda)$: regularized inverse
- $f(\lambda) = \alpha (\lambda - \bar{\lambda})$: mean-centered

**Powerful but expensive:** Requires eigendecomposition at each step.

### 3.4 Laplacian-Based

$$N(P) = \alpha (D^{-1/2} P D^{-1/2} - I) = \alpha (\mathcal{L}_{\text{sym}} - I)$$

where $D = \text{diag}(P \mathbf{1})$ is the degree matrix.

**Parameters:** One scalar $\alpha$.

**Dynamics:** Normalized diffusion on the graph.

**Interpretation:** Nodes move toward the (degree-weighted) average of their neighbors.

### 3.5 Pairwise Kernel

$$N_{ij}(P) = \kappa(P_{ij})$$

with $\kappa: [0,1] \to \mathbb{R}$ a scalar function.

**Symmetry:** Automatic since $P_{ij} = P_{ji}$.

**Parameters:** The function $\kappa$ (can be a small NN, or parametric like $\kappa(p) = a + bp + cp^2$).

**Interpretation:** 
- $\kappa(p) > 0$ for large $p$: strongly connected nodes attract
- $\kappa(p) < 0$ for small $p$: weakly connected nodes repel

**Diagonal handling:** Often want $N_{ii} = h(P_{ii})$ separate from off-diagonal:
$$N_{ij} = \begin{cases} h(P_{ii}) & i = j \\ \kappa(P_{ij}) & i \neq j \end{cases}$$

### 3.6 Attraction-Repulsion Kernel

$$N_{ij} = \kappa_+(P_{ij}) - \kappa_-(P_{ij})$$

where $\kappa_+$ is attractive (positive, increasing in $P_{ij}$) and $\kappa_-$ is repulsive.

**Classic form (Lennard-Jones inspired):**
$$\kappa(p) = \frac{a}{p + \epsilon} - \frac{b}{(p + \epsilon)^2}$$

**Parameters:** $(a, b, \epsilon)$.

**Equilibrium:** Nodes settle at distance where attraction balances repulsion.

---

## 4. Type-Based Parameterizations

Nodes belong to types $\tau: \{1, \ldots, n\} \to \{1, \ldots, K\}$. Interactions depend on type pairs.

### 4.1 Block Kernel

$$N_{ij} = \kappa_{\tau(i), \tau(j)}(P_{ij})$$

**Symmetry:** Require $\kappa_{ab} = \kappa_{ba}$.

**Parameters:** $\frac{K(K+1)}{2}$ functions.

**Use case:** Community structure with different within/between dynamics.

### 4.2 Stochastic Block Model Prior

$$N_{ij} = \alpha_{\tau(i), \tau(j)} + \beta \cdot P_{ij}$$

**Parameters:** $\frac{K(K+1)}{2}$ scalars $\alpha_{ab}$ plus one shared $\beta$.

**Interpretation:** Base rate depends on community pair, plus universal connection-strength effect.

### 4.3 Source-Sink Types

For directed intuition even in undirected setting:
- Type A: "sources" (high out-degree in some sense)
- Type B: "sinks" (high in-degree)

$$N_{ij} = \begin{cases}
\alpha_{AA} & \tau(i) = \tau(j) = A \\
\alpha_{BB} & \tau(i) = \tau(j) = B \\
\alpha_{AB} & \text{otherwise}
\end{cases}$$

---

## 5. Node-Specific Parameterizations

Each node has individual parameters, but interactions follow shared rules.

### 5.1 Diagonal + Shared Off-Diagonal

$$N_{ij} = \begin{cases} h_i & i = j \\ \kappa(P_{ij}) & i \neq j \end{cases}$$

**Parameters:** 
- $n$ scalars $h_i$ (node-specific self-rates)
- 1 function $\kappa$ (shared interaction)

**UDE form:**
$$N = \underbrace{\text{diag}(h_{\text{known}}(P))}_{\text{known: e.g., degree-based}} + \underbrace{\text{diag}(h_{\text{NN}}(P)) + \kappa_{\text{NN}}(P) \odot (\mathbf{1}\mathbf{1}^\top - I)}_{\text{learned}}$$

### 5.2 Node Features Determine Rate

$$h_i = g(\phi_i)$$

where $\phi_i = (P_{ii}, \sum_j P_{ij}, \max_j P_{ij}, \ldots)$ are node-level features extracted from $P$.

**Parameters:** One function $g: \mathbb{R}^m \to \mathbb{R}$ (small NN).

**Output:** $n$ scalars, but generated by shared $g$.

### 5.3 Attention-Style

$$N_{ij} = \frac{\exp(s(P_{i\cdot}, P_{j\cdot}))}{\sum_k \exp(s(P_{i\cdot}, P_{k\cdot}))} \cdot w(P_{ij})$$

where $s$ is a similarity score and $w$ is an edge weight.

**Parameters:** Functions $s$ and $w$.

**Interpretation:** Node $i$ attends to nodes with similar connectivity patterns.

---

## 6. Low-Rank Parameterizations

Constrain $N$ to have low rank, dramatically reducing parameters.

### 6.1 Rank-1

$$N = \alpha \cdot u u^\top$$

where $u \in \mathbb{R}^n$ and $\alpha \in \mathbb{R}$.

**Parameters:** $n + 1$.

**Dynamics:** $\dot{X} = \alpha (u u^\top) X = \alpha u (u^\top X)$

All nodes move in direction $u$, with magnitude proportional to $u^\top X$.

### 6.2 Rank-$r$ Symmetric

$$N = \sum_{k=1}^r \alpha_k u_k u_k^\top = U \text{diag}(\alpha) U^\top$$

where $U \in \mathbb{R}^{n \times r}$ has orthonormal columns.

**Parameters:** $nr + r$ (but with orthogonality constraints).

**Practical form:** $N = U \Lambda U^\top$ where $U, \Lambda$ learned.

### 6.3 Data-Derived Basis

$$N = \sum_{k=1}^r \alpha_k v_k v_k^\top$$

where $v_k$ are the top eigenvectors of $P$ itself.

**Parameters:** $r$ scalars $\alpha_k$.

**Interpretation:** Dynamics aligned with principal modes of the network.

---

## 7. Message-Passing Formulations

Write dynamics in terms of "messages" between nodes—natural for GNN-style thinking.

### 7.1 Basic Message Passing

$$\dot{X}_i = a(P_{ii}) X_i + \sum_{j \neq i} m(P_{ij}) (X_j - X_i)$$

**Equivalent $N$:**
$$N_{ij} = \begin{cases} a(P_{ii}) - \sum_{k \neq i} m(P_{ik}) & i = j \\ m(P_{ij}) & i \neq j \end{cases}$$

**Symmetry:** If $m(P_{ij}) = m(P_{ji})$ (true since $P$ symmetric), then check:
$$N_{ij} = m(P_{ij}) = m(P_{ji}) = N_{ji}$$ ✓

**Parameters:** Two functions $a, m$.

**Interpretation:**
- $a(P_{ii})$: intrinsic rate (self-loop strength)
- $m(P_{ij})$: attraction to neighbor $j$

### 7.2 Aggregation + Transform

$$\dot{X}_i = f\left(X_i, \sum_j P_{ij} X_j, \sum_j P_{ij}\right)$$

**Not directly in $N(P)X$ form!** This is more general.

But if $f$ is linear in $X$:
$$f(X_i, M_i, d_i) = \alpha(d_i) X_i + \beta(d_i) M_i$$

then:
$$\dot{X}_i = \alpha(d_i) X_i + \beta(d_i) \sum_j P_{ij} X_j = \sum_j N_{ij} X_j$$

with $N_{ij} = \alpha(d_i) \delta_{ij} + \beta(d_i) P_{ij}$.

**Symmetry issue:** This $N$ is **not symmetric** unless $\alpha, \beta$ are constant or depend symmetrically on $(d_i, d_j)$.

**Fix:** Use $N_{ij} = \frac{1}{2}(\beta(d_i) + \beta(d_j)) P_{ij}$ for off-diagonal.

### 7.3 Edge-Conditioned Messages

$$\dot{X}_i = \sum_j m(P_{ij}, P_{ii}, P_{jj}) (X_j - X_i)$$

**Symmetry:** Need $m(p, a, b) = m(p, b, a)$.

**Parameterization:** 
$$m(p, a, b) = \kappa(p) \cdot \psi(a + b) + \phi(p) \cdot \omega(|a - b|)$$

Symmetric by construction.

---

## 8. Encoding Qualitative Priors

### 8.1 Stability / Contraction

**Prior:** Network should stabilize (nodes don't explode).

**Encoding:** Ensure $N$ has non-positive eigenvalues.

**Parameterization:**
$$N = -\exp(M) \text{ where } M = M^\top$$

This guarantees $N$ is negative definite.

**Softer version:** 
$$N = N_{\text{raw}} - \lambda_{\max}(N_{\text{raw}}) \cdot I$$

Shift spectrum to be non-positive.

### 8.2 Conservation / Volume Preservation

**Prior:** Total "mass" $\sum_i \|X_i\|^2 = \text{tr}(P)$ is conserved.

**Constraint:** $\frac{d}{dt}\text{tr}(P) = \text{tr}(\dot{P}) = \text{tr}(NP + PN) = 2\text{tr}(NP) = 0$.

**Encoding:** Require $\text{tr}(NP) = 0$.

**Parameterization:** 
$$N = N_{\text{raw}} - \frac{\text{tr}(N_{\text{raw}} P)}{\text{tr}(P)} I$$

Project to zero-trace-product subspace.

### 8.3 Equilibrium Structure

**Prior:** System has known equilibrium $P^* = X^* {X^*}^\top$.

**Encoding:** $N(P^*) X^* = 0$.

**Parameterization:**
$$N(P) = (P - P^*) M(P)$$

for some matrix function $M$. At equilibrium, $N(P^*) = 0$.

### 8.4 Community Preservation

**Prior:** Community structure should be maintained (within-community distances stable).

**Encoding:** For nodes $i, j$ in same community, $\dot{P}_{ij} \approx 0$.

**Parameterization:** Let $C$ be community indicator matrix.
$$N = N_{\text{base}} - \gamma C \odot N_{\text{base}} C$$

Dampens within-community dynamics.

### 8.5 Sparsity Preservation

**Prior:** If $P_{ij} \approx 0$, it should stay $\approx 0$.

**Encoding:** $N_{ij} \to 0$ as $P_{ij} \to 0$.

**Parameterization:**
$$N_{ij} = P_{ij} \cdot \kappa(P_{ij})$$

The $P_{ij}$ factor ensures sparse regions stay decoupled.

---

## 9. Constraint Preservation

**Practical note:** For well-behaved dynamics learned via $N(P)X$ parameterizations, explicit constraint enforcement is often unnecessary. In experiments (Example 1), parsimonious models trained without barrier losses or projection maintained valid probability bounds throughout. The analysis below is provided for completeness and for cases where constraint violations occur.

There are **two independent types** of constraints on realizable $\dot{P}$:

1. **Algebraic (rank preservation):** From the factorization $P = XX^\top$
2. **Geometric (probability bounds):** From requiring $P_{ij} \in [0,1]$

### 9.1 Algebraic Constraint: Rank Preservation

The factorization $P = XX^\top$ with $X \in \mathbb{R}^{n \times d}$ means $\text{rank}(P) \leq d$. Any realizable $\dot{P}$ must be tangent to the rank-$d$ manifold.

**The constraint:** Let $V \in \mathbb{R}^{n \times d}$ span $\text{col}(P) \subset \mathbb{R}^n$ and $V_\perp \in \mathbb{R}^{n \times (n-d)}$ span its orthogonal complement. Then:

$$V_\perp^\top \dot{P} \, V_\perp = 0 \quad \text{(always)}$$

**What this means:** 
- The decomposition is in **node space** $\mathbb{R}^n$, not latent space $\mathbb{R}^d$
- $\text{col}(P)$ is a $d$-dimensional subspace of $\mathbb{R}^n$
- The latent dimension $d$ never changes
- What CAN change is *which* $d$-dimensional subspace equals $\text{col}(P)$ (it can rotate in $\mathbb{R}^n$)

**In practice:** This constraint is automatically satisfied by dynamics of the form $\dot{X} = N(P)X$, since $\dot{P} = NP + PN$ has the required structure. You don't need to enforce it explicitly.

### 9.2 Geometric Constraints: Probability Bounds

**Non-negativity:** $P_{ij} \geq 0$

**Upper bound:** $P_{ij} \leq 1$

These only matter at the **boundary** of the valid configuration space $\mathcal{X}$.

### 9.3 Boundary Analysis

From $\dot{P} = NP + PN$:
$$\dot{P}_{ij} = \sum_k N_{ik} P_{kj} + \sum_k P_{ik} N_{kj} = (NP)_{ij} + (PN)_{ij}$$

**At lower boundary ($P_{ij} = 0$):** Need $\dot{P}_{ij} \geq 0$.

This requires:
$$\sum_k N_{ik} P_{kj} + \sum_k P_{ik} N_{kj} \geq 0$$

**Caution:** This is NOT simply a Metzler condition on $N$. The condition involves the entire structure of $P$, not just local properties of $N$.

**At upper boundary ($P_{ij} = 1$):** Need $\dot{P}_{ij} \leq 0$.

Since $P_{ij} = X_i \cdot X_j \leq \|X_i\| \|X_j\|$, we have $P_{ij} = 1$ only if $X_i = X_j$ with $\|X_i\| = 1$ (i.e., $i = j$ on the diagonal, or two identical nodes).

For diagonal: $P_{ii} = \|X_i\|^2 = 1$ means $X_i$ is on the unit sphere.

### 9.4 Why Simple Conditions Fail

**Incorrect claim (DO NOT USE):** "If $N$ has non-negative off-diagonal entries, then $P_{ij} \geq 0$ is preserved."

**Why it fails:** For the linear system $\dot{y} = Ay$, Metzler $A$ (non-negative off-diagonal) preserves the positive orthant. But $\dot{P} = NP + PN$ is not of this form—it's a Lyapunov equation in $\dot{P}$, and the "state" $P$ has matrix structure.

**Correct analysis:** The mapping $P \mapsto NP + PN$ can take $P$ with $P_{ij} = 0$ to $\dot{P}_{ij} < 0$ even for "nice" $N$, because the sum $\sum_k N_{ik}P_{kj} + \sum_k P_{ik}N_{kj}$ depends on the full row/column structure.

### 9.5 Practical Constraint Enforcement

**Option A: Barrier in Loss Function (RECOMMENDED)**

Do NOT modify $N$. Instead, add to the training loss:
$$\mathcal{L}_{\text{barrier}} = \gamma \sum_{i,j} \left[\phi(-P_{ij}) + \phi(P_{ij} - 1)\right]$$

where $\phi(z) = \max(0, z)^2$ (quadratic penalty) or $\phi(z) = \log(1 + e^{\alpha z})$ (soft barrier).

This encourages the learned dynamics to stay in the valid region without breaking the symmetric $N$ structure.

**Option B: Projection After Integration**

After each ODE step, project:
$$P_{ij} \leftarrow \text{clamp}(P_{ij}, 0, 1)$$

Or equivalently, adjust $X$ to ensure constraints.

**Drawback:** Breaks ODE structure; introduces discontinuities.

**Option C: Reparameterization**

Work with unconstrained $Y \in \mathbb{R}^{n \times d}$ and define:
$$X = \sigma(Y) \cdot \frac{Y}{\|Y\|_{\text{row}}}$$

where $\sigma$ is a sigmoid-like function ensuring non-negativity and norm bounds.

**Drawback:** Complicates the dynamics and Jacobian computation.

### 9.6 Intrinsic Preservation: Special Cases

Some specific $N$ structures do preserve constraints:

**Pure contraction:** $N = -\alpha I$ with $\alpha > 0$ gives $\dot{X} = -\alpha X$, so $X \to 0$ and $P \to 0$. Constraints preserved (trivially).

**Normalized dynamics:** If $N(P)$ is designed such that $\text{tr}(\dot{P}) \leq 0$ when $\text{tr}(P)$ is large, boundedness is encouraged (but not guaranteed entry-wise).

### 9.7 Monitoring and Diagnostics

In practice:
1. **Initialize in interior:** Start with $\|X_i\| \ll 1$ and $X_i \cdot X_j > \epsilon$
2. **Monitor during training:** Track $\min_{ij} P_{ij}$ and $\max_{ij} P_{ij}$
3. **Interpret violations:** If learned dynamics frequently violate constraints, this suggests:
   - The true dynamics may not preserve constraints (model misspecification)
   - The parameterization is too flexible (regularize more)
   - Numerical issues (reduce step size)

### 9.8 Summary: Two Types of Constraints

| Type | Constraint | When Active | Enforcement |
|------|-----------|-------------|-------------|
| Algebraic | $V_\perp^\top \dot{P} V_\perp = 0$ | Always | Automatic from $\dot{X} = N(P)X$ |
| Geometric | $\dot{P}_{ij} \geq 0$ when $P_{ij} = 0$ | At lower boundary | Barrier in loss |
| Geometric | $\dot{P}_{ij} \leq 0$ when $P_{ij} = 1$ | At upper boundary | Barrier in loss |

**Key point:** In the interior of $\mathcal{X}$, only the algebraic constraint matters—and it's automatically satisfied. Geometric constraints only matter at boundaries.

---

## 10. Oscillatory Dynamics

**Challenge:** Oscillations typically require complex eigenvalues in the linearization. For a linear system $\dot{X} = NX$ with symmetric $N$, eigenvalues are real—no oscillations. How can symmetric $N(P)$ produce oscillations?

### 10.1 The Key Distinction: Linear vs Nonlinear

**Linear system:** $\dot{X} = NX$ with constant symmetric $N$ has solutions $X(t) = e^{Nt}X_0$. Since $N$ is symmetric with real eigenvalues, solutions are sums of exponentials—no oscillations.

**Our system:** $\dot{X} = N(P)X = N(XX^\top)X$ is **nonlinear** because $N$ depends on $X$ through $P = XX^\top$.

### 10.2 Linearization Around Equilibrium

At equilibrium $X^*$ with $N(P^*)X^* = 0$, the linearization is:
$$\delta\dot{X} = N(P^*)\delta X + \left[\frac{\partial N}{\partial P}\bigg|_{P^*} \cdot (\delta X \cdot X^{*\top} + X^* \cdot \delta X^\top)\right] X^*$$

The Jacobian of this system (as a linear operator on $\delta X \in \mathbb{R}^{n \times d}$) is **NOT** simply $N(P^*)$. The second term involves derivatives of $N$ and creates coupling that can produce complex eigenvalues.

**Conclusion:** Even with symmetric $N(P)$ at each instant, the linearized dynamics can have complex eigenvalues, enabling oscillations near equilibrium.

### 10.3 Mechanisms for Oscillation

**Mechanism 1: Hopf Bifurcation**

As parameters vary, eigenvalues of the Jacobian can cross the imaginary axis, creating limit cycles. This is generic for nonlinear systems and doesn't require non-symmetric $N$.

**Mechanism 2: Amplitude-Phase Coupling (for $d = 2$)**

Write $X_i = r_i(\cos\theta_i, \sin\theta_i)$. Then:
- $P_{ij} = r_i r_j \cos(\theta_i - \theta_j)$
- Phase differences $\theta_i - \theta_j$ affect connection probabilities
- Connection probabilities affect dynamics of phases

This feedback loop can create oscillatory phase dynamics even with symmetric $N(P)$.

**Mechanism 3: Multi-Scale Interaction**

$$N(P) = \alpha_1 P - \alpha_2 P^2$$

- $\alpha_1 P$: local attraction (neighbors pull together)
- $-\alpha_2 P^2$: nonlocal repulsion (neighbors-of-neighbors push apart)

The competition between scales can create oscillatory approach to equilibrium.

### 10.4 What Symmetric $N$ Cannot Do

Symmetric $N(P)$ cannot produce:
- **Rotation around origin in latent space** (this is gauge/invisible anyway)
- **Oscillations in the linear approximation with constant $N$**

But it CAN produce:
- **Damped oscillations** approaching equilibrium (via nonlinear Jacobian)
- **Limit cycles** via Hopf bifurcation
- **Quasi-periodic motion** in higher dimensions

### 10.5 Example: Two-Node System

Consider $n = 2$, $d = 1$ (scalar positions $x_1, x_2$). Then $P = \begin{pmatrix} x_1^2 & x_1 x_2 \\ x_1 x_2 & x_2^2 \end{pmatrix}$.

Let $N(P) = \alpha I + \beta P$. The dynamics:
$$\dot{x}_i = \alpha x_i + \beta x_i(x_1^2 + x_2^2) = (\alpha + \beta r^2) x_i$$

where $r^2 = x_1^2 + x_2^2$. This is radially symmetric with no oscillations.

For oscillations, need $d \geq 2$ or $n \geq 3$ to have sufficient degrees of freedom for amplitude-phase coupling.

### 10.6 Practical Oscillation Parameterizations

**Option A: Competing scales**
$$N(P) = \alpha_1 P - \alpha_2 P^2 + \alpha_3 P^3$$

**Option B: Phase-sensitive coupling (for $d = 2$)**
Design $N(P)$ so that $N_{ij}$ depends on whether $P_{ij}$ is increasing or decreasing relative to some reference.

**Option C: Auxiliary state**
Extend state to $(X, Y)$ where $Y$ captures "velocity" or "momentum," enabling second-order dynamics.

---

## 11. The UDE Template

### 11.1 General Structure

$$\dot{X} = N(P) X = [N_{\text{known}}(P) + N_{\text{learned}}(P)] X$$

where:
- $N_{\text{known}}$: encodes physics priors (structure chosen by modeler)
- $N_{\text{learned}}$: residual correction (parsimonious, learned from data)

Both must be symmetric.

### 11.2 Choosing $N_{\text{known}}$

| Prior Knowledge | Suggested $N_{\text{known}}$ |
|-----------------|------------------------------|
| Local diffusion | $\alpha(D^{-1}P - I)$ |
| Global coupling | $\alpha(P - \bar{P}I)$ |
| Degree-driven | $\text{diag}(h(P\mathbf{1}))$ |
| Community structure | Block-diagonal with community rates |
| Equilibrium $P^*$ | $(P - P^*)M$ |
| Conservation | Trace-zero projection |
| Stability | Negative definite form |

### 11.3 Choosing $N_{\text{learned}}$

| Complexity Budget | Suggested $N_{\text{learned}}$ |
|-------------------|--------------------------------|
| Minimal (1-5 params) | $\theta_1 I + \theta_2 P$ |
| Low (~$n$ params) | $\text{diag}(h_{\text{NN}}(P))$ |
| Medium (~$n$ params) | Kernel $\kappa_{\text{NN}}(P_{ij})$ |
| Higher (~$nr$ params) | Low-rank $U\Lambda U^\top$ |

### 11.4 Concrete Example: Community Dynamics UDE

**Setup:** Two communities, unknown cohesion/separation rates.

**Known:** 
- Nodes within same community attract
- Nodes in different communities may repel

**$N_{\text{known}}$:**
$$N^{\text{known}}_{ij} = \begin{cases} \alpha_{\text{in}} P_{ij} & \text{same community} \\ \alpha_{\text{out}} P_{ij} & \text{different community} \end{cases}$$

**$N_{\text{learned}}$:** Diagonal correction for node-specific rates:
$$N^{\text{learned}}_{ii} = h_{\text{NN}}(P_{i\cdot})$$

**Parameters:** 
- Known: $\alpha_{\text{in}}, \alpha_{\text{out}}$ (2 scalars, could be fixed or learned)
- Learned: $h_{\text{NN}}$ (small NN outputting $n$ values from node features)

### 11.5 Concrete Example: Oscillatory Dynamics UDE

**Setup:** Nodes oscillate around equilibrium configuration.

**Known:**
- There's an equilibrium $P^*$ 
- Dynamics should be oscillatory near equilibrium

**$N_{\text{known}}$:**
$$N^{\text{known}} = \alpha (P - P^*) + \beta (P - P^*)^2$$

The linear term drives toward equilibrium; quadratic term can induce oscillation.

**$N_{\text{learned}}$:** Polynomial correction:
$$N^{\text{learned}} = \gamma_1 P + \gamma_2 P^2$$

**Parameters:** $\alpha, \beta$ (known physics), $\gamma_1, \gamma_2$ (learned).

---

## 12. Implementation Patterns

### 12.1 Ensuring Symmetry

```julia
function symmetrize(M)
    return (M + M') / 2
end
```

### 12.2 Polynomial $N$

```julia
function polynomial_N(P, α)
    # α = [α₀, α₁, α₂, ...]
    n = size(P, 1)
    N = α[1] * I(n)
    Pk = P
    for k in 2:length(α)
        N += α[k] * Pk
        Pk = Pk * P
    end
    return N
end
```

### 12.3 Kernel $N$

```julia
function kernel_N(P, κ, h)
    # κ: off-diagonal kernel function
    # h: diagonal function
    n = size(P, 1)
    N = similar(P)
    for i in 1:n
        N[i,i] = h(P[i,i], sum(P[i,:]))
        for j in i+1:n
            N[i,j] = κ(P[i,j])
            N[j,i] = N[i,j]
        end
    end
    return N
end
```

### 12.4 Message-Passing Dynamics

```julia
function message_passing_dX(X, P, a, m)
    # Note: P is symmetric (P[i,j] = P[j,i]), so m(P[i,j]) = m(P[j,i])
    # automatically, ensuring the corresponding N matrix is symmetric.
    n, d = size(X)
    dX = zeros(n, d)
    for i in 1:n
        dX[i, :] = a(P[i,i]) * X[i, :]
        for j in 1:n
            if j != i
                dX[i, :] += m(P[i,j]) * (X[j, :] - X[i, :])
            end
        end
    end
    return dX
end
```

### 12.5 Low-Rank $N$

```julia
function lowrank_N(P, r, nn_U, nn_α, params)
    n = size(P, 1)
    # NN outputs n×r matrix U and r coefficients α
    U = reshape(nn_U(vec(P), params.U), n, r)
    α = nn_α(vec(P), params.α)
    # Orthogonalize U (optional, for numerical stability)
    U, _ = qr(U)
    U = Matrix(U)[:, 1:r]
    return U * Diagonal(α) * U'
end
```

---

## 13. Summary: Decision Tree

```
What do you know about the dynamics?
│
├─► Nothing specific
│   └─► N = θ₀I + θ₁P + diag(h_NN(node_features))
│
├─► Community structure exists
│   └─► Block kernel: N_ij = κ_{τ(i),τ(j)}(P_ij)
│
├─► Should stabilize to equilibrium P*
│   └─► N = α(P - P*) + N_learned
│
├─► Conservation law
│   └─► Project N to tr(NP) = 0
│
├─► Local interactions dominate
│   └─► Sparse kernel: N_ij = κ(P_ij)·𝟙[P_ij > ε]
│
├─► Oscillatory behavior expected
│   └─► Multi-scale: N = α₁P - α₂P² + N_learned
│
├─► Heterogeneous node behaviors
│   └─► diag(h_NN(P)) + shared kernel
│
└─► Want maximal flexibility with budget $B$
    ├─► B ~ O(1): polynomial coefficients
    ├─► B ~ O(n): diagonal + scalar kernel
    └─► B ~ O(nr): low-rank
```

---

## 14. Open Questions

1. **Automatic structure selection:** Can we learn which parameterization is appropriate?

2. **Spectral constraints:** Efficient enforcement of eigenvalue bounds during training?

3. **Directed extension:** Parsimonious $N_L, N_R$ for $P = LR^\top$?

4. **Time-varying $N$:** Explicit time dependence vs. implicit through $P(t)$?

5. **Stochastic extension:** SDE versions with state-dependent noise?

6. **Identifiability:** Given finite observations of $P(t)$, which parameterizations are distinguishable?
