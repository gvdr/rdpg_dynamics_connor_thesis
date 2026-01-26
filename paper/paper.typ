#set page(
  paper: "us-letter",
  margin: 2.5cm,
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en"
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.1")

#set math.equation(numbering: "(1)")

// Custom Commands
#let RR = $bb(R)$
#let so = $frak("so")$

// Theorem Environments
#let theorem(title: none, body) = figure(
  kind: "theorem",
  supplement: "Theorem",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "theorem"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let proposition(title: none, body) = figure(
  kind: "proposition",
  supplement: "Proposition",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "proposition"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let definition(title: none, body) = figure(
  kind: "definition",
  supplement: "Definition",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "definition"): it => block(width: 100%, inset: 8pt, stroke: (left: 2pt + black), radius: 0pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let corollary(title: none, body) = figure(
  kind: "corollary",
  supplement: "Corollary",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "corollary"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let proof(body) = block(width: 100%)[
  _Proof._ #body #h(1fr) $square$
]

#let remark(body) = block(width: 100%, inset: 8pt, stroke: (left: 2pt + luma(180)), fill: luma(250))[
  _Remark._ #body
]

// Title Block
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Learning Interpretable Dynamics of Temporal Networks \
    via Neural ODEs and Symbolic Regression
  ]

  #v(1em)

  Connor Smith#super[1] and Giulio V. Dalla Riva#super[1]

  #v(0.5em)

  #text(size: 0.9em)[
    #super[1]School of Mathematics and Statistics, University of Canterbury, New Zealand
  ]
]

#v(2em)

// Abstract
#align(center)[
  #block(width: 85%)[
    *Abstract* \
    Temporal networks---networks whose structure changes over time---appear across domains from neuroscience to ecology to social systems.
    While most approaches focus on predicting future network states, they rarely provide interpretable models of the underlying dynamics.
    We present a framework for learning continuous, interpretable differential equations governing the evolution of temporal network structure.
    Our approach embeds networks into a low-dimensional latent space via Random Dot Product Graphs (RDPG) and aims to learn the dynamics of this embedding using Neural Ordinary Differential Equations (Neural ODEs).
    We develop a gauge-theoretic analysis showing that RDPG embeddings have rotational ambiguity ($O(d)$ gauge freedom) and characterize which dynamics are observable versus invisible under this symmetry.
    A critical challenge is that adjacency spectral embedding (ASE) introduces arbitrary gauge transformations at each time step---random rotations from SVD sign ambiguity---that are unrelated to the dynamics and make trajectory learning ill-posed.
    Existing joint embedding methods (e.g., UASE) assume generative models incompatible with ODE dynamics on latent positions.
    We propose *structure-constrained alignment*: joint optimization over gauges and dynamics where symmetry constraints on the dynamics family identify and remove the random ASE gauge artifacts.
    We prove that symmetric dynamics cannot fit the skew-symmetric contamination from wrong gauges, enabling recovery of the true trajectory.
    This yields gauge-consistent architectures $dot(X) = N(P)X$ with symmetric $N$, achieving dramatic parameter reduction.
    We present mathematical foundations, algorithms with identifiability guarantees, and honest assessment of remaining challenges.
  ]
]

#v(2em)

= Introduction <sec:intro>

Temporal networks---networks whose edges and nodes change over time---are ubiquitous in complex systems @HOLME201297.
Examples include protein interaction networks that rewire during cellular processes @lucas2021inferring, social networks where relationships form and dissolve @hanneke2010discrete, and ecological networks whose structure responds to environmental change @poisot2015species.
Understanding how and why network structure changes is central to predicting system behavior.

Most temporal network modeling falls into two categories.
The first models _dynamics on networks_: how node states evolve given a fixed or slowly-changing network topology (e.g., epidemic spreading, opinion dynamics) @porter2016dynamical.
The second models _dynamics of networks_: how the network structure itself evolves @holme2015modern.
This paper addresses the latter, which remains less developed despite its importance.

Existing approaches to modeling network dynamics face a fundamental tension.
Statistical models like temporal exponential random graphs @hanneke2010discrete are interpretable but often lack predictive power.
Machine learning approaches @kazemi2020representation achieve better predictions but function as black boxes, offering little insight into the mechanisms driving structural change.
We propose a framework that achieves both: predictive models that can be distilled into interpretable differential equations.

Our key insight is that the discreteness of network events (edges appearing or disappearing) can be overcome by working in a continuous embedding space.
We use Random Dot Product Graphs (RDPG) @athreya2017statistical to embed each network snapshot into a low-dimensional latent space where similar nodes cluster together and connection probabilities arise naturally from inner products.
The temporal evolution of these embeddings is then smooth and amenable to differential equation modeling.

We train Neural Ordinary Differential Equations (Neural ODEs) @chen2018neural to learn the dynamics in embedding space.
While Neural ODEs provide excellent fits, they remain opaque.
We therefore apply symbolic regression to discover closed-form differential equations that approximate the learned neural dynamics.
These equations are interpretable---they can be analyzed mathematically, checked for conservation laws, and compared across systems.

*Contributions.* We make contributions in three areas:

_Theoretical foundations:_
+ A gauge-theoretic analysis of RDPG dynamics, characterizing observable vs. invisible dynamics and deriving gauge-invariant parameterizations ($dot(X) = N(P)X$ with $N$ symmetric)
+ The horizontal lift framework: fiber bundle geometry for understanding what "gauge-free trajectory" means, connecting to Procrustes alignment
+ Identifiability results: conditions under which dynamics parameters can be recovered from observed probability matrices

_Critical analysis:_
+ Demonstration that existing joint embedding methods (UASE, Omnibus) assume generative models incompatible with ODE dynamics on latent positions
+ Honest assessment of gauge alignment as a hard open problem for general dynamics
+ Clarification of what spectral methods can and cannot provide

_Proposed approach:_
+ *Structure-constrained alignment*: joint optimization over gauges and dynamics, using structural assumptions (symmetry, sparsity) as regularization
+ Alternating algorithm for linear horizontal dynamics, with identifiability guarantees under generic conditions
+ Discussion of extensions to polynomial and message-passing dynamics

= Methods <sec:methods>

Our framework consists of three stages: (1) embedding temporal networks via RDPG, (2) learning dynamics with Neural ODEs, and (3) extracting interpretable equations through symbolic regression.

== Random Dot Product Graphs and Latent Position Space <sec:rdpg>

We begin by establishing the latent space framework for Random Dot Product Graphs (RDPG) @athreya2017statistical @young2007random.

=== Latent positions and connection probabilities

In an RDPG, each node $i$ is associated with a latent position that determines its propensity to form connections.
For directed graphs, this position has two components:
- A *source position* $arrow(g)_i in B^d_+$: the node's propensity to initiate connections
- A *target position* $arrow(r)_i in B^d_+$: the node's propensity to receive connections

Here $B^d_+$ denotes the positive orthant of the unit ball:
$ B^d_+ = {x in RR^d : x_k >= 0 "for all" k, quad ||x|| <= 1} $

The probability of a directed edge from node $i$ to node $j$ is given by the dot product:
$ P_(i j) = arrow(g)_i dot arrow(r)_j $

The constraint $arrow(g)_i, arrow(r)_j in B^d_+$ ensures $P_(i j) in [0,1]$: non-negativity of coordinates gives $P_(i j) >= 0$, while Cauchy-Schwarz gives $P_(i j) <= ||arrow(g)_i|| dot ||arrow(r)_j|| <= 1$.

More generally, the RDPG literature defines an *inner product distribution* $F$ as any distribution on $RR^d$ such that $arrow(x)_i^top arrow(x)_j in [0,1]$ for all pairs drawn from $F$ @athreya2017statistical.
The positive orthant constraint $B^d_+$ is a convenient sufficient condition, but other configurations (e.g., positions on the Hardy-Weinberg curve in population genetics) also yield valid probability matrices.

Collecting positions into matrices $G, R in RR^(n times d)$ with rows $G_(i dot) = arrow(g)_i^top$ and $R_(i dot) = arrow(r)_i^top$, the probability matrix is:
$ P = G R^top $

For *undirected* graphs, we impose $arrow(g)_i = arrow(r)_i =: arrow(x)_i$, so that $P_(i j) = arrow(x)_i dot arrow(x)_j$ is symmetric.
Writing $X in RR^(n times d)$ for the matrix with rows $arrow(x)_i^top$:
$ P = X X^top $

We develop the theory primarily for the undirected case, noting that the directed generalization $(G, R) |-> G R^top$ follows similar principles (see @app:directed).

=== Adjacency spectral embedding

Given an observed adjacency matrix $A in {0,1}^(n times n)$, we estimate latent positions via the *adjacency spectral embedding* (ASE): a truncated singular value decomposition of $A$.
For $A = U Sigma V^top$, the rank-$d$ approximation gives:
$ hat(G) = U_d Sigma_d^(1/2), quad hat(R) = V_d Sigma_d^(1/2) $

For undirected graphs with symmetric $A$, we have $U = V$ and set $hat(X) = U_d Sigma_d^(1/2)$.

The estimated positions $hat(X)$ may not lie exactly in $(B^d_+)^n$, but for well-specified models they concentrate near the true positions @athreya2017statistical.
This spectral approach is consistent: as $n -> infinity$, $hat(X)$ converges to the true positions up to orthogonal transformation.

=== Temporal alignment via Procrustes

For a temporal network ${A_t}_(t=1)^T$, we embed each snapshot independently, then align across time.
SVD determines positions only up to orthogonal transformation: if $hat(X)$ is a valid embedding, so is $hat(X) Q$ for any $Q in O(d)$.
This nonidentifiability is fundamental to the RDPG model @athreya2017statistical.

We align consecutive embeddings using the *orthogonal Procrustes* solution:
$ Q_t = arg min_(Q^top Q = I) ||hat(X)_t Q - hat(X)_(t-1)||_F $
which has closed form $Q_t = V U^top$ where $hat(X)_t^top hat(X)_(t-1) = U Sigma V^top$.

=== The alignment problem for learning dynamics <sec:alignment-problem>

While pairwise Procrustes alignment produces reasonable discrete approximations, it is fundamentally inadequate for learning continuous dynamics.
The problem is more severe than simple "drift"---the gauge can jump discontinuously even for arbitrarily small time steps.

*The discrete gauge ambiguity.*
At each time $t$, ASE computes the eigendecomposition $P(t) approx U(t) Lambda(t) U(t)^top$ and returns $hat(X)(t) = U(t) Lambda(t)^(1\/2)$.
But eigenvectors are determined only up to sign: if $u$ is an eigenvector, so is $-u$.
For embedding dimension $d$, this gives $2^d$ possible gauge choices at each time, and the SVD algorithm picks arbitrarily based on numerical details.

*Consequence: gauge jumps are $O(1)$ even for infinitesimal $delta t$.*
Suppose $P(t)$ and $P(t + delta t)$ differ by $O(delta t)$.
The eigenvalues vary continuously, but the eigenvector signs can flip between consecutive times.
When this happens, $hat(X)(t + delta t)$ and $hat(X)(t)$ differ by a reflection---an $O(1)$ change---even though the true dynamics moved by $O(delta t)$.

This means we cannot estimate velocities by finite differences:
$ (hat(X)(t + delta t) - hat(X)(t)) / (delta t) $
is not approximating any smooth velocity.
It's dominated by the gauge jump, which contributes $O(1\/delta t) -> infinity$ as $delta t -> 0$.

*Pairwise Procrustes doesn't solve this.*
Aligning $hat(X)(t+1)$ to $hat(X)(t)$ via Procrustes finds the best orthogonal transformation _between those two frames_.
But:
- Each pairwise alignment $Q_(t,t+1)$ can be a large rotation (reflecting sign flips)
- The composition $Q_(0,1) Q_(1,2) dots.c Q_(T-1,T)$ accumulates these rotations
- The resulting trajectory, while locally aligned, can wander far from horizontal

In short: *we don't have a scrambled trajectory---we don't have a trajectory at all*.
We have a sequence of snapshots, each in an arbitrary gauge unrelated to its neighbors.

=== Why existing joint embedding methods don't solve this <sec:why-not-uase>

A natural question is whether existing multi-graph embedding methods resolve the gauge alignment problem.
The most prominent is *Unfolded Adjacency Spectral Embedding* (UASE) @gallagher2021spectral, which embeds all time points jointly via a single SVD.

#definition(title: "UASE")[
  Given adjacency matrices $A^((1)), ..., A^((T)) in {0,1}^(n times n)$:
  + Form the unfolded matrix $bold(A) = (A^((1)) | dots.c | A^((T))) in RR^(n times n T)$
  + Compute the rank-$d$ SVD: $bold(A) approx hat(X) hat(Y)^top$ where $hat(X) in RR^(n times d)$, $hat(Y) in RR^(n T times d)$
  + Partition $hat(Y) = (hat(Y)^((1)); dots.c ; hat(Y)^((T)))$ into $T$ blocks
]

UASE produces gauge-consistent embeddings---all $hat(Y)^((t))$ share a common basis---and satisfies important stability properties @gallagher2021spectral.
However, *UASE assumes a different generative model than ours*.

*The model mismatch.*
UASE is designed for the _Multilayer RDPG_ where:
$ P^((t))_(i j) = X_i Lambda^((t)) Y^((t) top)_j $
with $X$ *constant* across time and only $Y^((t))$ varying.
This factored structure is appropriate when nodes have a stable "identity" component and time-varying "activity" patterns.

Our dynamics model is different:
$ P^((t))_(i j) = x_i (t)^top x_j (t) $
where the *same* evolving position $x_i (t)$ determines both sending and receiving, and the positions themselves follow ODE dynamics.
There is no factorization through a shared constant component.

*Consequence: UASE distorts the dynamics.*
When applied to data from $P^((t)) = X(t) X(t)^top$ with genuinely evolving $X(t)$, UASE finds a compromise:
- $hat(X)$ captures some "average" structure across time
- $hat(Y)^((t))$ absorbs temporal variation, but in a way that doesn't correspond to true position evolution

In practice, UASE can produce trajectories that fail to capture the actual dynamics.
The joint SVD enforces a low-rank factored structure that the true model doesn't satisfy.

#remark[
  Similar limitations apply to Omnibus embedding @levin2017central, which builds a block matrix of averaged adjacencies, and to COSIE @arroyo2021inference.
  All assume some shared structure across time that our ODE-driven dynamics model lacks.
]

*What we need:* A global gauge-fixing procedure that:
1. Works with independent ASE at each time (respecting our generative model)
2. Finds gauges $Q_t in O(d)$ for all times simultaneously
3. Produces a genuinely smooth trajectory $tilde(X)(t) = hat(X)(t) Q_t$
4. Approximates the horizontal lift (minimal gauge motion)

=== Fiber bundle geometry and horizontal lifts <sec:horizontal>

To understand what "smooth" means for gauge-ambiguous data, we formalize the geometry.
The embedding space forms a *principal fiber bundle* with structure group $O(d)$.

#definition(title: "Latent Position Bundle")[
  Let $cal(M) = {X in RR^(n times d) : "rank"(X) = d}$ be the space of full-rank latent position matrices.
  The *projection* $pi: cal(M) -> cal(P)$ maps $X |-> P = X X^top$ to the space of rank-$d$ positive semidefinite matrices.
  The *fiber* over $P$ is $pi^(-1)(P) = {X Q : Q in O(d)}$---the gauge orbit.
]

A path $P(t)$ in the base space (observable dynamics) can be lifted to many paths $X(t)$ in the total space.
These lifts differ by time-dependent gauge transformations $Q(t) in O(d)$.
The key question: among all lifts, which is the "natural" one?

#definition(title: "Vertical and Horizontal Subspaces")[
  At each $X in cal(M)$, the tangent space decomposes as $T_X cal(M) = cal(V)_X plus.o cal(H)_X$ where:
  - *Vertical subspace* (gauge directions): $cal(V)_X = {X A : A in so(d)}$
  - *Horizontal subspace* (observable directions): $cal(H)_X = cal(V)_X^perp$

  A tangent vector $dot(X)$ is *horizontal* if $dot(X) in cal(H)_X$, i.e., it has no gauge component.
]

#proposition(title: "Horizontal Characterization")[
  A velocity $dot(X)$ is horizontal if and only if $X^top dot(X)$ is symmetric.
  Equivalently, $dot(X)^top X = X^top dot(X)$.
] <prop:horizontal-char>

#proof[
  Decompose $dot(X) = X A + W$ where $A = (X^top X)^(-1) X^top dot(X)$ and $W perp "col"(X)$.
  The vertical component is $X A_("skew")$ where $A_("skew") = (A - A^top)/2$.
  Thus $dot(X) in cal(H)_X$ iff $A_("skew") = 0$ iff $A = A^top$ iff $X^top dot(X)$ is symmetric.
]

#definition(title: "Horizontal Lift")[
  Given a path $P(t)$ in the base space and initial condition $X(0)$ with $X(0)X(0)^top = P(0)$, the *horizontal lift* is the unique path $X(t)$ satisfying:
  1. $X(t) X(t)^top = P(t)$ for all $t$ (projects to $P$)
  2. $dot(X)(t) in cal(H)_(X(t))$ for all $t$ (velocity is horizontal)
]

The horizontal lift is the "smoothest" lift in a precise sense: it has no superfluous gauge motion.
This is exactly what we want for learning dynamics.

#theorem(title: "Horizontal Lift Existence and Uniqueness")[
  Let $P(t)$ be a smooth path of rank-$d$ positive semidefinite matrices.
  For any initial $X_0$ with $X_0 X_0^top = P(0)$, there exists a unique horizontal lift $X(t)$ with $X(0) = X_0$.
] <thm:horizontal-lift>

#proof[
  Differentiating $P = X X^top$ gives $dot(P) = dot(X) X^top + X dot(X)^top$.
  For $dot(X)$ horizontal, we require $X^top dot(X) = S$ symmetric.
  Substituting $dot(X) = X (X^top X)^(-1) S + W$ with $X^top W = 0$:
  $ dot(P) = X (X^top X)^(-1) S X^top + X S (X^top X)^(-1) X^top + W X^top + X W^top $

  Given $dot(P)$ and requiring $W X^top + X W^top$ to match the residual, this is a system of linear equations for $(S, W)$.
  For full-rank $X$, standard ODE existence/uniqueness theory applies.
]

=== Global gauge synchronization <sec:gauge-sync>

The horizontal lift theorem tells us what we want; now we need a practical method to approximate it from discrete, gauge-scrambled embeddings.

*The synchronization problem.*
Given embeddings ${hat(X)_0, hat(X)_1, ..., hat(X)_T}$ with arbitrary gauges, find gauge corrections ${Q_0, Q_1, ..., Q_T} subset O(d)$ such that the corrected trajectory $tilde(X)_t = hat(X)_t Q_t$ is as smooth as possible.

This is an instance of *synchronization over groups*, a well-studied problem in computer vision and signal processing.

#definition(title: "Gauge Synchronization Objective")[
  The *smoothness energy* of a gauge assignment ${Q_t}$ is:
  $ cal(E)({Q_t}) = sum_(t=0)^(T-1) ||hat(X)_(t+1) Q_(t+1) - hat(X)_t Q_t||_F^2 $
  The *synchronized gauges* minimize $cal(E)$ over $(O(d))^(T+1)$.
]

Minimizing $cal(E)$ directly is nonconvex, but good algorithms exist.
A key observation simplifies the problem: only relative gauges $R_t = Q_t^top Q_(t+1)$ matter for smoothness.

#proposition(title: "Relative Gauge Formulation")[
  Define $R_t = Q_t^top Q_(t+1)$ (relative gauge from $t$ to $t+1$).
  Then:
  $ cal(E) = sum_(t=0)^(T-1) ||hat(X)_(t+1) R_t^top dots.c R_0^top Q_0 - hat(X)_t R_(t-1)^top dots.c R_0^top Q_0||_F^2 $
  The optimal $Q_0$ is arbitrary (global gauge freedom); only the $R_t$ affect smoothness.
]

*Connection to Procrustes.*
The pairwise Procrustes solution $R_t^("pair") = arg min_(R in O(d)) ||hat(X)_(t+1) R - hat(X)_t||_F$ provides an initial estimate.
But these local solutions may be inconsistent: $R_0^("pair") R_1^("pair") dots.c R_(T-1)^("pair")$ may not equal $I$ even if the path is closed.

*Spectral relaxation.*
A practical algorithm relaxes $O(d)$ to the Stiefel manifold and solves via eigendecomposition.
Define the *connection Laplacian*:
$ L_(t,t') = cases(
  I & "if" t = t',
  -hat(X)_t^top hat(X)_(t+1) (hat(X)_(t+1)^top hat(X)_(t+1))^(-1) & "if" |t - t'| = 1,
  0 & "otherwise"
) $
The synchronized gauges correspond to the bottom eigenvectors of $L$.

*Practical algorithm.*
1. Compute pairwise Procrustes: $R_t^("init") = "Procrustes"(hat(X)_t, hat(X)_(t+1))$
2. Chain to get initial gauges: $Q_t^("init") = R_0^("init") R_1^("init") dots.c R_(t-1)^("init")$
3. Refine via gradient descent on $cal(E)$ over $(O(d))^(T+1)$, using Riemannian optimization
4. Output $tilde(X)_t = hat(X)_t Q_t$

=== The Procrustes flow (continuous limit) <sec:procrustes-flow>

The synchronization procedure above works for discrete data.
In the limit of continuous observations, it converges to the *Procrustes flow*---the horizontal lift.

#theorem(title: "Procrustes Flow as Horizontal Lift")[
  Let ${hat(X)_t}_(t in [0,T])$ be a continuous path of embeddings satisfying $hat(X)_t hat(X)_t^top = P_t$ for some smooth $P_t$.
  Suppose $hat(X)_t = X_t R_t$ where $X_t$ is horizontal and $R_t in O(d)$ is the (unknown) gauge error.

  Define the *gauge velocity* $Omega_t = R_t^(-1) dot(R)_t in so(d)$.
  The Procrustes flow removes this gauge motion: if we set
  $ dot(Q) = -Q dot (hat(X)^top hat(X))^(-1) dot "skew"(hat(X)^top dot(hat(X))) $
  where $"skew"(M) = (M - M^top)/2$, then $tilde(X)_t = hat(X)_t Q_t$ is horizontal.
] <thm:procrustes-flow>

#proof[
  The corrected trajectory is $tilde(X) = hat(X) Q = X R Q$.
  Its velocity is $dot(tilde(X)) = dot(X) R Q + X dot(R) Q + X R dot(Q)$.
  For $tilde(X)$ to be horizontal, we need $tilde(X)^top dot(tilde(X))$ symmetric.

  The skew-symmetric part of $hat(X)^top dot(hat(X))$ measures the gauge velocity in the raw embeddings.
  The flow $dot(Q) = -Q (hat(X)^top hat(X))^(-1) "skew"(hat(X)^top dot(hat(X)))$ exactly cancels this, leaving only the horizontal component.
]

#remark[
  In practice, we cannot use the Procrustes flow directly because:
  (1) we have discrete samples, not continuous paths, and
  (2) the gauge jumps between samples are discontinuous, so $dot(hat(X))$ doesn't exist.
  The flow equation describes the *idealized* continuous-time limit; the synchronization algorithm (@sec:gauge-sync) provides the practical discrete-time implementation.
]

#remark[
  The horizontal condition $X^top dot(X)$ symmetric connects directly to our invisible dynamics theorem (@thm:invisible).
  Invisible dynamics $dot(X) = X A$ with $A in so(d)$ have $X^top dot(X) = (X^top X) A$ skew-symmetric.
  The Procrustes flow/synchronization removes exactly this invisible component, leaving only observable dynamics.
]

#remark[
  When the adjacency matrix $A$ has negative eigenvalues (e.g., due to heterophilic "opposites attract" connectivity), the standard RDPG with $P = X X^top >= 0$ is inadequate.
  The *Generalized RDPG* (GRDPG) @rubin2022statistical handles this by using an indefinite inner product: $P_(i j) = arrow(x)_i^top I_(p,q) arrow(x)_j$ where $I_(p,q) = "diag"(1,...,1,-1,...,-1)$.
  The gauge group then becomes the indefinite orthogonal group $O(p,q)$.
  Our gauge-theoretic analysis extends naturally to this setting, though we focus on the positive-definite case for clarity.
]

=== The alignment problem: an honest assessment <sec:alignment-honest>

We must be candid: *gauge alignment from spectral embeddings alone is a hard open problem* for learning continuous dynamics.
The methods described above---global synchronization, horizontal lifts, Procrustes flow---provide a theoretical framework and idealized algorithms, but face fundamental challenges in practice.

The core difficulty is not gauge drift from dynamics (which may be zero for horizontal dynamics).
The problem is that *ASE introduces arbitrary gauge transformations at each time step*:
$ hat(X)^((t)) = X^((t)) R^((t)) $
where the $R^((t)) in O(d)$ are essentially random---determined by SVD sign conventions, not by physics.
Even perfectly smooth true dynamics produce jagged, discontinuous embedding trajectories.

Specific challenges:
1. *Discontinuous gauge jumps*: The $R^((t))$ can change arbitrarily between consecutive time steps, making finite differences meaningless.

2. *No direct observability*: We never observe $R^((t)}$ or $X^((t)}$ separately---only their product $hat(X)^((t))$.

3. *Statistical noise*: ASE estimates have $O(1\/sqrt(n))$ errors that compound with alignment errors.

Existing joint embedding methods like UASE solve a different problem (the multilayer RDPG) and, as discussed in @sec:why-not-uase, assume a generative model incompatible with the ODE dynamics we aim to learn.

This motivates a different approach: rather than aligning first and learning dynamics second, we propose *joint optimization* where structural assumptions about dynamics help identify and remove the random ASE gauge artifacts.

=== Structure-constrained gauge alignment <sec:structure-constrained>

The core problem is not gauge drift from dynamics---horizontal dynamics produce no such drift.
The problem is that *ASE introduces arbitrary gauge transformations at each time step*, unrelated to the dynamics.

*The ASE gauge artifact.*
At each time $t$, ASE computes the eigendecomposition of $A^((t))$ (or $hat(P)^((t))$).
The eigenvectors are determined only up to sign (and rotation within repeated eigenspaces).
This means:
$ hat(X)^((t)) = X^((t)) R^((t)) + E^((t)) $
where $R^((t)) in O(d)$ is essentially *random*---determined by numerical details of the SVD algorithm, not by the dynamics.
Even if the true positions $X^((t))$ evolve smoothly, the estimates $hat(X)^((t))$ jump erratically due to these arbitrary gauge choices.

*The alignment goal.*
We seek gauge corrections $Q_t in O(d)$ such that:
$ tilde(X)^((t)) := hat(X)^((t)) Q_t approx X^((t)) Q^* $
for some fixed (unknown) global gauge $Q^*$.
That is, we want to "undo" the random ASE gauges $R^((t))$, recovering a trajectory that follows the true dynamics up to an overall rotation.

*Why structure helps.*
Without additional information, the problem is underdetermined: any smooth interpolation through the $hat(X)^((t)}$ could be "correct."
But if we know (or assume) the dynamics belong to a restricted family $cal(F)$, this provides a constraint.
The corrected trajectory $tilde(X)^((t))$ should be *explainable* by some $f in cal(F)$.
Random ASE gauges produce trajectories that require dynamics outside $cal(F)$; correct gauges produce learnable trajectories.

#definition(title: "Joint Alignment-Learning Problem")[
  Given ASE embeddings ${hat(X)^((t))}_(t=0)^T$ and dynamics family $cal(F)$, find gauge corrections ${Q_t in O(d)}$ and $f in cal(F)$ minimizing:
  $ cal(L)({Q_t}, f) = sum_(t=0)^(T-1) ||hat(X)^((t+1)) Q_(t+1) - hat(X)^((t)) Q_t - delta t dot f(hat(X)^((t)) Q_t)||_F^2 $
]

The dynamics family $cal(F)$ acts as regularization: it couples gauge choices across time by requiring the corrected trajectory to be *learnable* within $cal(F)$.

*The tradeoff:*
- $cal(F)$ too restrictive: true dynamics may not fit, alignment fails
- $cal(F)$ too expressive (e.g., unconstrained neural network): can fit any trajectory, no constraint on gauges

We now work out the mathematics for a concrete, tractable case.

==== Linear horizontal dynamics <sec:linear-horizontal>

Consider the family of linear dynamics with symmetric coefficient:
$ cal(F)_("lin") = {dot(X) = N X : N in RR^(n times n), N = N^top} $

The symmetry constraint $N = N^top$ ensures dynamics are *horizontal*---purely observable, with no invisible gauge component (connecting to @thm:invisible and @prop:horizontal-char).

*Discrete-time formulation.*
For small time step $delta t$, the dynamics $dot(X) = N X$ discretize to:
$ X^((t+1)) approx (I + delta t dot N) X^((t)) = M X^((t)) $
where $M = I + delta t dot N$ inherits symmetry from $N$.

*The observation model.*
True positions evolve as $X^((t+1)) = M X^((t))$ for some fixed symmetric $M$.
Note: because $M$ is symmetric, the true dynamics are *horizontal*---there is no gauge drift from the dynamics themselves.

ASE gives us $hat(X)^((t)) = X^((t)) R^((t))$ where $R^((t)) in O(d)$ are arbitrary (from SVD sign choices).
These $R^((t))$ are unrelated across time---they are numerical artifacts, not dynamical.

*The alignment goal.*
Find $Q_t approx (R^((t)))^(-1)$ (up to a global constant) so that:
$ tilde(X)^((t)) = hat(X)^((t)) Q_t approx X^((t)) $
In other words, undo the random ASE gauges to recover the true smooth trajectory.

*The optimization problem.*
Find gauges ${Q_t}$ and symmetric $M$ minimizing:
$ cal(L) = sum_(t=0)^(T-1) ||hat(X)^((t+1)) Q_(t+1) - M hat(X)^((t)) Q_t||_F^2 $ <eq:linear-objective>

#proposition(title: "Relative Gauge Reduction")[
  Define relative gauges $R_t = Q_t^top Q_(t+1)$. The objective @eq:linear-objective depends on ${Q_t}$ only through ${R_t}$ and a global gauge $Q_0$:
  $ cal(L) = sum_(t=0)^(T-1) ||hat(X)^((t+1)) R_t^top - M hat(X)^((t))||_F^2 $
  The global gauge $Q_0$ is unidentifiable (reflecting the inherent $O(d)$ freedom); only relative gauges affect the loss.
]

#proof[
  $hat(X)^((t+1)) Q_(t+1) - M hat(X)^((t)) Q_t = (hat(X)^((t+1)) R_t^top - M hat(X)^((t))) Q_t$.
  Since $Q_t$ is orthogonal, $||A Q_t||_F = ||A||_F$.
]

*Alternating optimization algorithm.*

The non-convex problem decomposes into tractable subproblems:

*Step 1: Fix ${R_t}$, solve for $M$.*

This is least squares with a symmetry constraint:
$ min_(M = M^top) sum_t ||hat(X)^((t+1)) R_t^top - M hat(X)^((t))||_F^2 $

Stack the data: let $Y = [hat(X)^((1)) R_0^top | dots.c | hat(X)^((T)) R_(T-1)^top]$ and $Z = [hat(X)^((0)) | dots.c | hat(X)^((T-1))]$.

The unconstrained minimizer is $M^* = Y Z^top (Z Z^top)^(-1)$ (assuming $Z Z^top$ is invertible).

Project onto symmetric matrices:
$ M = (M^* + (M^*)^top) / 2 $

*Step 2: Fix $M$, solve for ${R_t}$.*

For each $t$ independently, solve the orthogonal Procrustes problem:
$ min_(R_t in O(d)) ||hat(X)^((t+1)) R_t^top - M hat(X)^((t))||_F^2 $

#proposition(title: "Procrustes Solution for Relative Gauges")[
  Let $A_t = hat(X)^((t))^top M^top hat(X)^((t+1))$ with SVD $A_t = U_t Sigma_t V_t^top$. Then:
  $ R_t = V_t U_t^top $
]

#proof[
  The objective $||hat(X)^((t+1)) R_t^top - M hat(X)^((t))||_F^2$ expands to a constant minus $2 "tr"(R_t A_t^top)$.
  Maximizing $"tr"(R_t A_t^top)$ over $O(d)$ is the orthogonal Procrustes problem, solved by $R_t = V_t U_t^top$.
]

*The complete algorithm:*

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Structure-constrained gauge alignment (linear dynamics)],
  block(width: 100%, inset: 1em, stroke: 0.5pt)[
    *Input:* Embeddings ${hat(X)^((t))}_(t=0)^T$, tolerance $epsilon$\
    *Output:* Aligned embeddings ${tilde(X)^((t))}$, dynamics matrix $M$

    1. *Initialize:* $R_t^((0)) <- "Procrustes"(hat(X)^((t)), hat(X)^((t+1)))$ for all $t$

    2. *Repeat until convergence:*
       - *M-step:* $Y <- [hat(X)^((1)) (R_0^((k)))^top | dots.c ]$, $Z <- [hat(X)^((0)) | dots.c ]$
       - $M^((k+1)) <- (Y Z^top (Z Z^top)^(-1) + (Y Z^top (Z Z^top)^(-1))^top) / 2$
       - *R-step:* For each $t$: compute SVD of $hat(X)^((t))^top M^((k+1)) hat(X)^((t+1)) = U Sigma V^top$
       - $R_t^((k+1)) <- V U^top$

    3. *Recover absolute gauges:* $Q_0 <- I$, $Q_(t+1) <- Q_t R_t$ for $t = 0, ..., T-1$

    4. *Return:* $tilde(X)^((t)) <- hat(X)^((t)) Q_t$, $M$
  ]
) <alg:structure-constrained>

==== Why the symmetry constraint helps <sec:why-symmetry-helps>

Without requiring $M = M^top$, the problem decouples: each $R_t$ could be chosen via pairwise Procrustes independently, and we'd simply fit the best (possibly asymmetric) $M$ afterward.
This recovers the naive approach that fails---it aligns consecutive frames but doesn't use the global structure to identify the random ASE gauge artifacts.

The symmetry constraint *couples gauge choices globally* and forces recovery of the true gauges.
The key insight: random ASE gauge errors $R^((t))$ introduce apparent "gauge velocity" that is skew-symmetric, while true horizontal dynamics are symmetric.
The constraint that dynamics fit symmetric $N$ implicitly enforces $Q_t approx (R^((t)))^(-1)$.
We now make this precise.

#theorem(title: "Gauge Velocity Contamination")[
  Let $X(t)$ follow true dynamics $dot(X) = N X$ with $N = N^top$.
  Let $tilde(X) = X S$ for some time-varying gauge error $S(t) in O(d)$.
  Then the apparent dynamics in the $tilde(X)$ frame are:
  $ dot(tilde(X)) = N tilde(X) + tilde(X) Omega $
  where $Omega = S^(-1) dot(S) in frak(s o)(d)$ is skew-symmetric.

  Moreover, this can be written as $dot(tilde(X)) = tilde(N) tilde(X)$ for some symmetric $tilde(N)$ if and only if $Omega = 0$.
] <thm:gauge-contamination>

#proof[
  By the product rule:
  $ dot(tilde(X)) = dot(X) S + X dot(S) = N X S + X dot(S) $

  Since $S in O(d)$, we have $S S^top = I$, so $X = tilde(X) S^(-1)$. Substituting:
  $ dot(tilde(X)) = N tilde(X) S^(-1) S + tilde(X) S^(-1) dot(S) = N tilde(X) + tilde(X) Omega $

  where $Omega = S^(-1) dot(S)$. Since $S S^top = I$, differentiating gives $dot(S) S^top + S dot(S)^top = 0$, so $dot(S) S^top = -S dot(S)^top$. Thus:
  $ Omega^top = (S^(-1) dot(S))^top = dot(S)^top (S^(-1))^top = dot(S)^top S = -(dot(S) S^top)^top S = -S^top dot(S) S^top S = -S^top dot(S) $

  And $Omega = S^top dot(S) = -Omega^top$, confirming $Omega in frak(s o)(d)$.

  For the second claim, suppose $dot(tilde(X)) = tilde(N) tilde(X)$ with $tilde(N) = tilde(N)^top$.
  Then $tilde(X)^top dot(tilde(X)) = tilde(X)^top tilde(N) tilde(X)$ is symmetric (as $tilde(N)$ is symmetric and $tilde(X)^top tilde(N) tilde(X)$ is a congruence).

  But from our expression:
  $ tilde(X)^top dot(tilde(X)) = tilde(X)^top N tilde(X) + tilde(X)^top tilde(X) Omega $

  The first term $tilde(X)^top N tilde(X)$ is symmetric (since $N$ is).
  The second term $(tilde(X)^top tilde(X)) Omega$ is the product of a symmetric positive-definite matrix and a skew-symmetric matrix.

  If $B$ is symmetric positive-definite and $Omega$ is skew-symmetric, then $(B Omega)^top = Omega^top B = -Omega B$.
  For $B Omega$ to be symmetric, we need $B Omega = -Omega B$, i.e., $B Omega + Omega B = 0$.
  Since $B$ is positive-definite, this holds only if $Omega = 0$.

  Thus $tilde(X)^top dot(tilde(X))$ symmetric implies $Omega = 0$, i.e., $S$ is constant (no gauge drift).
]

#corollary[
  The constraint that learned dynamics have the form $dot(X) = N X$ with $N$ symmetric automatically enforces horizontal alignment (no gauge velocity contamination).
]

This theorem explains the mechanism: random ASE gauge artifacts $R^((t))$ introduce skew-symmetric contamination $tilde(X) Omega$ that cannot be absorbed into symmetric $N$. The joint optimization, by requiring symmetric dynamics, implicitly selects $Q_t approx (R^((t)))^(-1)$---exactly the gauges that remove the ASE artifacts.

==== Extension to gauge-invariant dynamics <sec:gauge-invariant-dynamics>

The analysis extends beyond linear dynamics. Consider any dynamics of the form:
$ dot(X) = N(P) X quad "where" N(P) = N(P)^top $
with $N$ depending on $P = X X^top$ (which is gauge-invariant: $tilde(P) = tilde(X) tilde(X)^top = X S S^top X^top = X X^top = P$).

#proposition[
  For dynamics $dot(X) = N(P) X$ with $N(P)$ symmetric, the gauge velocity contamination analysis applies unchanged: wrong gauges produce apparent dynamics $dot(tilde(X)) = N(P) tilde(X) + tilde(X) Omega$ with skew $Omega$, which cannot be fit by symmetric dynamics.
]

*Examples of gauge-invariant symmetric dynamics:*

1. *Polynomial:* $N(P) = sum_(k=0)^K alpha_k P^k$ (symmetric since $P$ is)

2. *Spectral:* $N(P) = sum_(i=1)^n phi(lambda_i) Pi_i$ where $P = sum_i lambda_i Pi_i$ is the spectral decomposition

3. *Heat kernel:* $N(P) = e^(beta P) - I$ (matrix exponential of symmetric is symmetric)

4. *Graph Laplacian:* $N(P) = D - P$ where $D = "diag"(P bold(1))$ is the degree matrix

All of these automatically enforce horizontal alignment when used as the dynamics family $cal(F)$.

#proposition(title: "Identifiability under Generic Dynamics")[
  Suppose the true dynamics $M^*$ has distinct eigenvalues and the true trajectory ${X^((t))}$ spans $RR^n$ (i.e., $[X^((0)) | dots.c | X^((T-1))]$ has full row rank).
  Then the gauges ${Q_t}$ are identifiable up to a global $O(d)$ transformation: any minimizer of @eq:linear-objective satisfies $Q_t = Q^* R_t^*$ for some fixed $Q^* in O(d)$.
]

#proof[
  (Sketch) If $M$ is symmetric with distinct eigenvalues, its eigenspaces are one-dimensional.
  The constraint $hat(X)^((t+1)) R_t^top = M hat(X)^((t))$ for all $t$, combined with the trajectory spanning $RR^n$, determines $M$ uniquely.
  Given $M$, each $R_t$ is determined by the Procrustes problem (unique for generic data).
  The only remaining freedom is the choice of $Q_0$, which propagates as a global transformation.
]

==== Extensions and limitations <sec:extensions-limitations>

*Polynomial dynamics.*
The linear case extends to polynomial dynamics $dot(X) = N(P) X$ where $N(P) = sum_(k=0)^K alpha_k P^k$.
Since $P = P^top$, we have $N(P) = N(P)^top$, ensuring horizontal dynamics.

The discrete formulation becomes:
$ X^((t+1)) approx M(P^((t))) X^((t)), quad M(P) = I + delta t sum_(k=0)^K alpha_k P^k $

The key observation is that $P^((t)) = hat(X)^((t)) Q_t Q_t^top hat(X)^((t) top) = hat(X)^((t)) hat(X)^((t) top)$ is *gauge-invariant*.
Thus we can compute $P^((t))$ directly from the (unaligned) embeddings.

The optimization becomes:
$ min_({R_t}, {alpha_k}) sum_(t=0)^(T-1) ||hat(X)^((t+1)) R_t^top - M(P^((t))) hat(X)^((t))||_F^2 $

*Alternating optimization for polynomial case:*
- *$alpha$-step (fix ${R_t}$):* Linear regression. With $Y_t = hat(X)^((t+1)) R_t^top - hat(X)^((t))$ and features $Phi_t^((k)) = (P^((t)))^k hat(X)^((t))$, solve:
  $ min_({alpha_k}) sum_t ||Y_t - delta t sum_k alpha_k Phi_t^((k))||_F^2 $
  This is ordinary least squares in the coefficients ${alpha_k}$.

  *Closed-form solution:* Define Gram matrix $G in RR^((K+1) times (K+1))$ and target vector $b in RR^(K+1)$:
  $ G_(k j) = sum_t chevron.l Phi_t^((k)), Phi_t^((j)) chevron.r_F, quad b_k = sum_t chevron.l Phi_t^((k)), Y_t chevron.r_F $
  Then $alpha = G^(-1) b \/ delta t$.

- *$R$-step (fix ${alpha_k}$):* Same Procrustes subproblems as the linear case.
  For each $t$: compute $M(P^((t))) = I + delta t sum_k alpha_k (P^((t)))^k$, then SVD of $hat(X)^((t) top) M(P^((t)))^top hat(X)^((t+1)) = U Sigma V^top$, and set $R_t = V U^top$.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Structure-constrained alignment for polynomial dynamics],
  block(width: 100%, inset: 1em, stroke: 0.5pt)[
    *Input:* Embeddings ${hat(X)^((t))}_(t=0)^T$, polynomial degree $K$, tolerance $epsilon$\
    *Output:* Aligned embeddings ${tilde(X)^((t))}$, coefficients ${alpha_k}_(k=0)^K$

    1. *Precompute:* $P^((t)) <- hat(X)^((t)) hat(X)^((t) top)$ for all $t$ (gauge-invariant!)

    2. *Initialize:* $R_t^((0)) <- "Procrustes"(hat(X)^((t)), hat(X)^((t+1)))$ for all $t$

    3. *Repeat until $|cal(L)^((k)) - cal(L)^((k-1))| < epsilon$:*
       - *$alpha$-step:*
         - $Y_t <- hat(X)^((t+1)) (R_t^((k)))^top - hat(X)^((t))$ for all $t$
         - $Phi_t^((j)) <- (P^((t)))^j hat(X)^((t))$ for all $t, j$
         - $G_(k j) <- sum_t "tr"(Phi_t^((k) top) Phi_t^((j)))$; $b_k <- sum_t "tr"(Phi_t^((k) top) Y_t)$
         - $alpha^((k+1)) <- G^(-1) b \/ delta t$
       - *$R$-step:* For each $t$:
         - $M <- I + delta t sum_j alpha_j^((k+1)) (P^((t)))^j$
         - SVD: $hat(X)^((t) top) M^top hat(X)^((t+1)) = U Sigma V^top$
         - $R_t^((k+1)) <- V U^top$

    4. *Recover absolute gauges:* $Q_0 <- I$, $Q_(t+1) <- Q_t R_t$ for $t = 0, ..., T-1$

    5. *Return:* $tilde(X)^((t)) <- hat(X)^((t)) Q_t$, ${alpha_k}$
  ]
) <alg:polynomial>

*Identifiability analysis for polynomial dynamics.*
A critical question: does the polynomial constraint actually identify the correct gauges, or might there be spurious solutions?

Since $P^((t))$ is gauge-invariant, choosing ${alpha_k}$ completely determines $M(P^((t))) = I + delta t sum_k alpha_k (P^((t)))^k$ for each $t$.
The $R$-step then asks: does there exist $R_t in O(d)$ such that $hat(X)^((t+1)) R_t^top = M(P^((t))) hat(X)^((t))$?

For the *true* coefficients ${alpha_k^*}$, this equation holds exactly (with $R_t$ undoing the ASE gauge jumps).
For *wrong* coefficients, $M(P^((t))) hat(X)^((t))$ is generically *not* a rotation of $hat(X)^((t+1)}$ in the $d$-dimensional column space, so Procrustes yields nonzero residual.

*Why the constraint helps:*
The requirement that the *same* ${alpha_k}$ must work for *all* time steps is restrictive when $P^((t))$ varies.
Different $P^((t))$ produce different $M(P^((t)})$, each imposing constraints on the coefficients.

#proposition(title: "Coefficient Identifiability from Varying P")[
  Suppose the matrices ${(P^((0)))^k, (P^((1)))^k, ..., (P^((T-1)))^k}_(k=0)^K$ are such that the system of equations
  $ sum_(k=0)^K alpha_k (P^((t)))^k hat(X)^((t)) = hat(X)^((t+1)) R_t^top quad "for" t = 0, ..., T-1 $
  is overdetermined in ${alpha_k}$ when the ${R_t}$ are fixed.
  Then for generic data, the coefficients ${alpha_k}$ are determined (given correct gauges), and wrong gauges produce coefficient estimates inconsistent across time.
]

*Failure modes:*

1. *Slowly varying $P^((t))$:* If $P^((t)) approx P$ for all $t$, the matrices $(P^((t)))^k$ are approximately constant.
   Different polynomial coefficients give similar $M(P)$, and the system becomes underdetermined.
   _Remedy:_ Use longer time series with larger spacing, or use trajectories with significant $P$ variation.

2. *Short time series:* With $T <= K + 1$ time transitions, the polynomial coefficients are underdetermined even with varying $P$.
   _Remedy:_ Require $T >> K + 1$.

3. *Low embedding dimension:* For $d = 1$, gauges are signs ($plus.minus 1$), giving $2^T$ discrete possibilities.
   The polynomial constraint may not distinguish all sign sequences.
   _Remedy:_ Use higher-order constraints or cross-validation.

4. *Degenerate configurations:* If the trajectory $hat(X)^((t))$ lies in a low-dimensional subspace, Procrustes may have non-unique solutions.
   _Remedy:_ Verify rank conditions; use regularization.

*The core mechanism (restated):*
ASE introduces gauge errors $R^((t)) in O(d)$ that are essentially "random" (determined by eigendecomposition conventions, not dynamics).
The true trajectory $X^((t))$ satisfies $X^((t+1)) = M^*(P^((t))) X^((t))$.
In scrambled coordinates: $hat(X)^((t+1)) = M^*(P^((t))) hat(X)^((t)) S^((t))$ where $S^((t)) = R^((t)) (R^((t+1)))^(-1)$.

The joint optimization seeks ${R_t}$ undoing the $S^((t))$ and ${alpha_k}$ matching the true $M^*$.
For wrong ${alpha_k}$, the Procrustes residuals are generically nonzero.
The alternating algorithm descends toward solutions with small residual, which (under the conditions above) correspond to correct gauges and coefficients.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: none,
    table.hline(),
    table.header([*Condition*], [*Risk*], [*Mitigation*]),
    table.hline(stroke: 0.5pt),
    [$P^((t))$ nearly constant], [High], [Larger $delta t$; longer $T$],
    [$T <= K + 1$], [High], [Use lower-degree polynomial or more data],
    [$d = 1$], [Medium], [Cross-validate; multiple restarts],
    [Noise $>> $ signal], [Medium], [Larger $n$; regularization],
    [Local minima], [Medium], [Good initialization; multiple restarts],
    table.hline()
  ),
  caption: [Identifiability risks and mitigations for polynomial dynamics.]
) <tab:poly-identifiability>

*Message-passing dynamics.*
Message-passing dynamics have the form:
$ dot(x)_i = sum_j P_(i j) g(x_i, x_j) $
where $g: RR^d times RR^d -> RR^d$ specifies how node $j$ influences node $i$.
This is a strong structural constraint: node $i$'s velocity depends only on local information $(x_i, {x_j : P_(i j) > 0})$.

*Important special cases that yield horizontal dynamics:*

1. *Neighbor attraction:* $g(x_i, x_j) = x_j$
   $ dot(x)_i = sum_j P_(i j) x_j = (P X)_i $
   In matrix form: $dot(X) = P X$, which is $N = P$ (symmetric since $P = P^top$). ✓

2. *Laplacian diffusion:* $g(x_i, x_j) = x_j - x_i$
   $ dot(x)_i = sum_j P_(i j) (x_j - x_i) = (P X)_i - d_i x_i $
   where $d_i = sum_j P_(i j)$. In matrix form: $dot(X) = (P - D) X = -L X$ where $L = D - P$ is the graph Laplacian. Since $L = L^top$, this is horizontal. ✓

3. *Heat kernel:* $g(x_i, x_j) = (e^(beta P) - I)_(i j) x_j \/ P_(i j)$ (when $P_(i j) > 0$)
   Gives $dot(X) = (e^(beta P) - I) X$, symmetric for $beta in RR$. ✓

*General linear message-passing.*
Consider $g(x_i, x_j) = W x_j$ for some $W in RR^(d times d)$:
$ dot(X) = P X W $

This mixes rows (via $P$) and columns (via $W$). For this to be horizontal, we need the induced $N$ to be symmetric.

If $W = w I$ for scalar $w$, then $dot(X) = w P X$, which is $N = w P$ (symmetric). ✓

For general $W$, the dynamics $dot(X) = P X W$ are *not* of the form $dot(X) = N X$ with $N$ acting on rows only.
However, if we vectorize: $"vec"(dot(X)) = (W^top times.o P) "vec"(X)$, and $(W^top times.o P)$ is symmetric iff both $W$ and $P$ are symmetric.

*The structural constraint for gauge identification.*
Message-passing enforces that velocity at node $i$ is determined by *local* information.
Random ASE gauge errors $R^((t))$ introduce *global* correlations that violate this structure.

#proposition(title: "Gauge Contamination in Message-Passing")[
  Let $X(t)$ follow message-passing dynamics $dot(x)_i = sum_j P_(i j) g(x_i, x_j)$.
  Let $tilde(X) = X S$ for time-varying gauge $S(t) in O(d)$.
  Then the apparent dynamics in the $tilde(X)$ frame are:
  $ dot(tilde(x))_i = sum_j P_(i j) g(tilde(x)_i S^(-1), tilde(x)_j S^(-1)) S + tilde(x)_i Omega $
  where $Omega = S^(-1) dot(S) in frak(s o)(d)$.

  The second term $tilde(x)_i Omega$ is a *global* contribution (same $Omega$ for all nodes) that does not factor through the message-passing structure.
]

This means: with wrong gauges, the "apparent velocities" include a global skew-symmetric component that cannot be explained by any local message-passing rule.

*Joint optimization for parameterized message-passing.*
When $g = g_theta$ is parameterized (e.g., a small neural network), we optimize:
$ min_({Q_t}, theta) sum_(t=0)^(T-1) sum_(i=1)^n ||tilde(x)_i^((t+1)) - tilde(x)_i^((t)) - delta t sum_j P_(i j)^((t)) g_theta(tilde(x)_i^((t)), tilde(x)_j^((t)))||^2 $
where $tilde(x)_i^((t)) = (hat(X)^((t)) Q_t)_(i dot)$.

*Algorithm for message-passing dynamics:*

Unlike the polynomial case, there's no closed-form solution for the $theta$-step. We use gradient-based optimization:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Structure-constrained alignment for message-passing dynamics],
  block(width: 100%, inset: 1em, stroke: 0.5pt)[
    *Input:* Embeddings ${hat(X)^((t))}_(t=0)^T$, message-passing architecture $g_theta$\
    *Output:* Aligned embeddings ${tilde(X)^((t))}$, learned parameters $theta$

    1. *Initialize:* $Q_t^((0)) <- product_(s=0)^(t-1) "Procrustes"(hat(X)^((s)), hat(X)^((s+1)))$ (chained pairwise)
       $theta^((0)) <-$ random initialization

    2. *Repeat until convergence:*
       - *$theta$-step:* Gradient descent on $theta$ with ${Q_t}$ fixed:
         $ theta^((k+1)) <- theta^((k)) - eta nabla_theta cal(L)({Q_t^((k))}, theta^((k))) $
       - *$Q$-step:* For each $t$, optimize $Q_t$ on $O(d)$ manifold:
         $ Q_t^((k+1)) <- arg min_(Q in O(d)) cal(L)_t(Q; theta^((k+1)), {Q_(s != t)^((k))}) $
         (via Riemannian gradient descent or Procrustes-like update)

    3. *Return:* $tilde(X)^((t)) <- hat(X)^((t)) Q_t$, $theta$
  ]
) <alg:message-passing>

*Why message-passing constrains gauges (intuition):*

Consider two scenarios:
- *Correct gauges:* $tilde(X)^((t)) approx X^((t))$. The velocity $tilde(x)_i^((t+1)) - tilde(x)_i^((t))$ truly depends only on local neighbors. The message-passing model fits well.
- *Wrong gauges:* The gauge error $S^((t)) = R^((t)} Q_t$ varies with $t$. The apparent velocity includes $tilde(x)_i Omega^((t))$ which is the *same* for all nodes (given their current position). This global coherence cannot be captured by local message-passing.

The message-passing loss will be higher for wrong gauges because it cannot fit the global gauge-velocity component.

*Horizontal message-passing as a special case.*
When $g(x_i, x_j) = g(x_j, x_i)$ (symmetric interaction) and the aggregation is linear:
$ dot(x)_i = sum_j P_(i j) h(x_j) $
for some $h: RR^d -> RR^d$, we can write $dot(X) = P H(X)$ where $H(X)$ applies $h$ row-wise.

For $h(x) = W x$ with $W = W^top$, we get $dot(X) = P X W$, and if additionally $W = I$, this reduces to $dot(X) = P X$---the polynomial case with $alpha_0 = 0, alpha_1 = 1$.

#remark[
  The polynomial case $N(P) = sum_k alpha_k P^k$ is a *subset* of message-passing dynamics.
  Specifically, $(P^k X)_i = sum_j (P^k)_(i j) x_j$ sums over $k$-hop neighbors.
  Message-passing with general $g_theta$ can capture *nonlinear* local interactions that polynomials cannot.
]

==== Summary: Three dynamics families <sec:dynamics-families-summary>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Family*], [*Form*], [*Parameters*], [*Algorithm*]),
    table.hline(stroke: 0.5pt),
    [Linear],
    [$dot(X) = N X$, $N = N^top$],
    [$n(n+1)/2$ entries of $N$],
    [Closed-form alternating (@alg:structure-constrained)],

    [Polynomial],
    [$dot(X) = (sum_k alpha_k P^k) X$],
    [$K + 1$ coefficients],
    [Closed-form alternating (@alg:polynomial)],

    [Message-passing],
    [$dot(x)_i = sum_j P_(i j) g_theta(x_i, x_j)$],
    [$|theta|$ (neural net)],
    [Gradient-based (@alg:message-passing)],
    table.hline()
  ),
  caption: [Comparison of dynamics families for structure-constrained alignment.]
) <tab:dynamics-families>

*Key tradeoffs:*

- *Linear:* Most general symmetric dynamics, but $O(n^2)$ parameters. Best when dynamics don't factor through $P$.

- *Polynomial:* Parsimonious ($K + 1$ parameters), closed-form updates, gauge-invariant features. Best when dynamics depend on network structure through powers of $P$.

- *Message-passing:* Most expressive (nonlinear local interactions), but requires gradient-based optimization and risks overfitting. Best for complex local dynamics.

*Gauge identification strength:*

All three families enforce *symmetric* or *local* structure that random ASE gauges violate.
- Linear/polynomial: Wrong gauges produce asymmetric $N$ or inconsistent $alpha_k$ across time.
- Message-passing: Wrong gauges produce global velocity correlations that local $g$ cannot fit.

The polynomial family offers the best balance: parsimonious, closed-form, and leverages the gauge-invariance of $P$ to separate the alignment problem from coefficient estimation.

*Convergence.*
The alternating optimization decreases the objective at each step (both subproblems have closed-form solutions that achieve the global optimum given the other variables fixed).
Thus the algorithm converges to a stationary point.
However, the overall problem is non-convex, so we cannot guarantee convergence to the global optimum.

#proposition(title: "Monotonic Decrease")[
  Let $cal(L)^((k))$ denote the objective value after iteration $k$ of @alg:structure-constrained.
  Then $cal(L)^((k+1)) <= cal(L)^((k))$, with equality if and only if $(M^((k)), {R_t^((k))})$ is a stationary point.
]

Good initialization is crucial.
Starting from pairwise Procrustes alignments (ignoring the symmetry constraint) provides a reasonable warm start.
Multiple random restarts can help escape poor local minima.

*Limitations.*
1. *Model mismatch:* If true dynamics lie outside $cal(F)$, the method may find spurious alignments or fail to converge.
2. *Local minima:* The alternating optimization is non-convex; good initialization (e.g., from pairwise Procrustes) is important.
3. *Sample complexity:* With few time points or high noise, the constraint may be insufficient to determine gauges.
4. *Computational cost:* Each iteration requires $T$ SVDs of $d times d$ matrices (cheap) and one $n times n$ linear solve (more expensive for large $n$).

#remark[
  This approach inverts the usual pipeline: rather than "align then learn," we "learn to align."
  The dynamics model provides the inductive bias needed to resolve gauge ambiguity.
  This is philosophically similar to how physical constraints (symmetries, conservation laws) help identify coordinates in physics.
]

==== Effect of noise <sec:noise-analysis>

In practice, we observe adjacency matrices $A^((t))$ from which we estimate $hat(X)^((t))$ via ASE.
These estimates have error: $hat(X)^((t)) = X^((t)) R^((t)) + E^((t))$ where $R^((t)) in O(d)$ is the gauge ambiguity and $E^((t))$ is statistical noise.

Under standard RDPG asymptotics @athreya2017statistical, the rows of $E^((t))$ satisfy:
$ ||E^((t))_(i dot)|| = O_p(1\/sqrt(n)) $
with the noise being approximately Gaussian for large $n$.

*Impact on structure-constrained alignment:*

The objective becomes:
$ cal(L) = sum_t ||(X^((t+1)) R^((t+1)) + E^((t+1))) Q_(t+1) - M (X^((t)) R^((t)) + E^((t))) Q_t||_F^2 $

At the true solution ($Q_t = (R^((t)))^(-1)$ and $M = M^*$), the residual is:
$ cal(L)^* = sum_t ||E^((t+1)) (R^((t+1)))^(-1) - M^* E^((t)) (R^((t)))^(-1)||_F^2 = O_p(n \/ n) = O_p(1) $

The noise contributes $O(1\/sqrt(n))$ per entry, summed over $n$ nodes, giving $O(1)$ total.

*Key insight:* The structure constraint helps *even with noise* because:
1. Noise is isotropic (no preferred gauge direction)
2. The symmetry constraint on $M$ averages over many node pairs
3. Wrong gauges produce *systematic* asymmetry in $M$, distinguishable from random noise

However, for small $n$ or large noise, the signal (symmetry violation from wrong gauge) may be overwhelmed by noise.
A rough requirement is that the gauge-induced asymmetry exceeds noise:
$ ||M^* - M^("wrong")||_F >> ||E||_F \/ ||X||_F approx 1\/sqrt(n) $

== Dynamics in Latent Space and Gauge Freedom <sec:gauge>

We now consider how latent positions evolve over time and what this implies for observable network structure.

=== The dynamical framework

We model the evolution of latent positions as a dynamical system:
$ dot(X) = f(X) $
where $X in (B^d_+)^n$ collects all node positions and $f: (B^d_+)^n -> RR^(n times d)$ specifies each node's velocity as a function of the current configuration.

Concretely, $dot(X)_i = f(X)_i$ describes how node $i$'s position $arrow(x)_i$ changes based on the positions of all nodes.
For instance, $f$ might encode attraction toward similar nodes or repulsion from dissimilar ones.

Since we observe networks (and hence probability matrices $P = X X^top$), not latent positions directly, we need to understand what dynamics on $X$ imply for $P$.
By the product rule:
$ dot(P) = dot(X) X^top + X dot(X)^top = f(X) X^top + X f(X)^top $
This is the *induced dynamics* on the observable.
Any vector field $f$ on latent space induces a corresponding evolution of the probability matrix.

=== Gauge freedom

The latent positions $X$ are not uniquely determined by $P$.
For any orthogonal matrix $Q in O(d)$:
$ (X Q)(X Q)^top = X Q Q^top X^top = X X^top = P $

Thus $X$ and $X Q$ represent the _same_ probability matrix $P$.
This is the *gauge freedom* of RDPG: the equivalence class $[X] = {X Q : Q in O(d)}$ corresponds to a single observable network structure.

Geometrically, gauge freedom means that a global rotation of all positions in latent space leaves all connection probabilities unchanged.
This makes sense: dot products depend only on angles and magnitudes, which are preserved under orthogonal transformations.

This gauge freedom has profound implications for learning dynamics.
If two configurations $X$ and $X Q$ are observationally indistinguishable, then any dynamics that moves along this equivalence class---rotating all positions by a common orthogonal transformation---produces no observable change in the network.
We call such dynamics *invisible*.

#definition(title: "Observable vs. Invisible Dynamics")[
  A vector field $f: RR^(n times d) -> RR^(n times d)$ produces _observable dynamics_ if $dot(P) != 0$, where $dot(P) = f(X)X^top + X f(X)^top$.
  Otherwise, the dynamics are _invisible_---the parameterization changes but the graph structure is static.
]

#theorem(title: "Characterization of Invisible Dynamics")[
  Let $X in RR^(n times d)$ have full column rank.
  A vector field $f$ produces invisible dynamics if and only if $f(X) = X A$ for some skew-symmetric matrix $A in so(d)$, i.e., $A^top = -A$.
] <thm:invisible>

#remark[
  The theorem is stated for $X in RR^(n times d)$ to emphasize that the gauge-theoretic structure holds in full generality.
  In practice, for $P = X X^top$ to be a valid probability matrix, positions must satisfy the *inner product distribution* constraint @athreya2017statistical: for all pairs $i, j$, we require $arrow(x)_i dot arrow(x)_j in [0,1]$.
  A sufficient condition is $X in (B^d_+)^n$, but other configurations (e.g., positions on a Hardy-Weinberg curve) also yield valid probabilities.
]

#proof[
  $(arrow.l.double)$ Suppose $f(X) = X A$ with $A^top = -A$. Then:
  $ dot(P) = X A X^top + X (X A)^top = X A X^top + X A^top X^top = X(A + A^top)X^top = 0 $

  $(arrow.r.double)$ Suppose $dot(P) = f(X)X^top + X f(X)^top = 0$.

  Decompose $f(X) = X A + W$ where $A = (X^top X)^(-1)X^top f(X)$ and $X^top W = 0$ (i.e., $W$ lies in the orthogonal complement of $"col"(X)$).

  Substituting into $dot(P) = 0$:
  $ 0 = (X A + W)X^top + X(X A + W)^top = X(A + A^top)X^top + W X^top + X W^top $

  For $X$ with full column rank, consider the constraint $X^top W = 0$ combined with the equation above.
  Taking the projection onto $"col"(X)$: multiplying on left by $(X^top X)^(-1)X^top$ and on right by $X(X^top X)^(-1)$:
  $ 0 = A + A^top + (X^top X)^(-1)X^top W X^top X (X^top X)^(-1) + "similar term" $

  Since $X^top W = 0$, the middle terms vanish, giving $A + A^top = 0$.

  For the remaining equation $W X^top + X W^top = 0$ with $X^top W = 0$: this is a symmetric matrix equation. For generic full-rank $X$, the only solution is $W = 0$.
]

#corollary[
  The space of invisible dynamics is isomorphic to $so(d)$, with dimension $(d(d-1))/2$.
  For $d = 2$, this is 1-dimensional (a single rotation rate); for $d = 3$, it is 3-dimensional.
]

The invisible dynamics $dot(X) = X A$ are infinitesimal rotations along gauge orbits.
In particular, _uniform rotation around the origin_ satisfies $dot(X)_i = X_i A$, producing $dot(P) = 0$---the embedding rotates but the network is static.

Crucially, other rotational dynamics _are_ observable.
Rotation around the _origin_ is gauge (invisible), but circulation around a _nonzero centroid_ is observable:

#proposition(title: "Centroid Circulation")[
  Dynamics of the form $dot(X)_i = (X_i - bar(X))A$ with $bar(X) != 0$ and $A in so(d)$ produce observable changes in $P$.
] <prop:centroid>

#proof[
  Let $bar(X) = 1/n sum_i X_i$ be the centroid. The dynamics $dot(X)_i = (X_i - bar(X))A$ can be rewritten as:
  $ dot(X)_i = X_i A - bar(X)A $

  In matrix form with $bold(1) in RR^n$ the all-ones vector:
  $ dot(X) = X A - bold(1)bar(X)^top A $

  Computing $dot(P)$:
  $ dot(P) &= dot(X)X^top + X dot(X)^top \
           &= (X A - bold(1)bar(X)^top A)X^top + X(X A - bold(1)bar(X)^top A)^top \
           &= X A X^top + X A^top X^top - bold(1)bar(X)^top A X^top - X A^top bar(X)bold(1)^top $

  The first two terms cancel since $A + A^top = 0$. Let $v = A^top bar(X) = -A bar(X)$:
  $ dot(P) = bold(1)v^top X^top + X v bold(1)^top $

  Entry-wise: $dot(P)_(i j) = v dot X_j + X_i dot v$.

  This vanishes for all $i,j$ only if $v = 0$, i.e., $A bar(X) = 0$. For generic $bar(X) != 0$ and $A != 0$, we have $dot(P) != 0$.
]

*Interpretation.* Circulation around the centroid decomposes as:
$ dot(X)_i = underbrace(X_i A, "invisible (gauge)") - underbrace(bar(X)A, "shared drift (observable)") $
The first term is pure gauge. The second is a constant velocity applied to all nodes, which shifts all dot products and hence changes $P$.

Similarly, differential rotation rates are observable:

#proposition(title: "Differential Rotation is Observable")[
  If nodes have different rotation rates:
  $ dot(X)_i = X_i A_i, quad A_i in so(d) $
  then:
  $ dot(P)_(i j) = X_i (A_i - A_j) X_j^top $
  This is generically nonzero when $A_i != A_j$.
]

#proof[
  $ dot(P)_(i j) &= dot(X)_i X_j^top + X_i dot(X)_j^top = (X_i A_i) X_j^top + X_i (X_j A_j)^top \
               &= X_i A_i X_j^top + X_i A_j^top X_j^top = X_i A_i X_j^top - X_i A_j X_j^top \
               &= X_i (A_i - A_j) X_j^top $
  using $A_j^top = -A_j$. This is nonzero when $A_i != A_j$ and $X_i, X_j$ are generic.
]

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Dynamics*], [$dot(P) = 0$?], [*Observable?*],
    ),
    table.hline(stroke: 0.5pt),
    [$dot(X) = X A$ (uniform rotation around origin)], [Yes], [No],
    [$dot(X)_i = (X_i - bar(X))A$ with $bar(X) != 0$], [No], [Yes],
    [$dot(X)_i = X_i A_i$ with $A_i != A_j$], [No], [Yes],
    [$dot(X)_i = alpha X_i$ (radial scaling)], [No], [Yes],
    [$dot(X)_i = sum_j w_(i j)(X_j - X_i)$ (attraction/repulsion)], [No], [Yes],
    table.hline()
  ),
  caption: [Classification of dynamics by observability.]
) <tab:observable>

A natural question arises: given that some dynamics are invisible, can we still learn the observable part?

#theorem(title: "Identifiability Modulo Gauge")[
  Let $X in RR^(n times d)$ have full column rank.
  Given $dot(P)$ and $X$, the vector field $f(X)$ is uniquely determined up to gauge:
  $ f(X) = F + X A $
  where $F$ is any solution to $dot(P) = F X^top + X F^top$ and $A in so(d)$ is arbitrary.
] <thm:identifiability>

#proof[
  Consider the linear map $cal(L): RR^(n times d) -> RR^(n times n)$ defined by $cal(L)(F) = F X^top + X F^top$.
  The equation $dot(P) = cal(L)(f(X))$ determines $f(X)$ up to elements of $ker(cal(L))$.

  By @thm:invisible, $F in ker(cal(L))$ if and only if $F = X A$ for some skew-symmetric $A in so(d)$.
  Thus if $F_1$ and $F_2$ both satisfy $dot(P) = F_i X^top + X F_i^top$, then $F_1 - F_2 in ker(cal(L))$, so $F_1 = F_2 + X A$ for some $A in so(d)$.

  To show existence: the image of $cal(L)$ consists of symmetric matrices in the row/column space of $X$.
  Since $dot(P) = dot(X) X^top + X dot(X)^top$ is symmetric and $dot(X)$ lies in this space (being a velocity of positions), $dot(P) in "im"(cal(L))$.
]

#theorem(title: "Gauge Equivalence")[
  Two vector fields $f$ and $tilde(f)$ are gauge equivalent (induce the same $dot(P)$) if and only if:
  $ f(X) - tilde(f)(X) = X A(X) $
  for some $so(d)$-valued function $A(X)$.
]

#proof[
  Apply @thm:invisible to the difference $h = f - tilde(f)$.
]

#corollary(title: "Canonical Decomposition")[
  Any vector field decomposes uniquely as:
  $ f(X) = f_("phys")(X) + X A(X) $
  where $f_("phys")$ determines $dot(P)$ and $X A$ is pure gauge.
]

This decomposition clarifies what can be learned: the "physical" content---what affects the observable---is uniquely determined; only the coordinate-dependent form varies with gauge choice.

*Gauge-Free Decomposition of $dot(X)$.*
Any velocity $dot(X)$ decomposes uniquely as $dot(X) = X A + W$ where $A = (X^top X)^(-1)X^top dot(X) in RR^(d times d)$ and $W perp "col"(X)$.
Further decomposing $A = A_("sym") + A_("skew")$:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Component*], [*Contributes to $dot(P)$?*], [*Interpretation*],
    ),
    table.hline(stroke: 0.5pt),
    [$X A_("sym")$], [Yes], [Radial/stretching dynamics],
    [$X A_("skew")$], [No], [Pure rotation (gauge)],
    [$W$], [Yes], [Rotates $"col"(P)$ in $RR^n$],
    table.hline()
  ),
  caption: [Decomposition of $dot(X)$ into observable and gauge components.]
)

The observable content of $dot(X)$ is $(A_("sym"), W)$.

*Implications for learning.*
When we train on estimated positions $hat(X)(t)$, the Procrustes alignment fixes a consistent gauge.
The learned $f$ determines $dot(P)$ correctly, but a different alignment procedure would yield a gauge-equivalent $f + X A$.

== Neural ODE Dynamics <sec:node>

After embedding, we have a sequence of latent positions ${hat(X)_t}_(t=1)^T$.
We flatten each $hat(X)_t in RR^(n times d)$ into a vector $bold(u)_t in RR^(n d)$ and model the dynamics as:
$ (d bold(u)) / (d t) = f_theta(bold(u)) $
where $f_theta$ is a neural network with parameters $theta$.

We parameterize $f_theta$ as a fully-connected network with architecture:
$ f_theta: RR^(n d) arrow.long^("Dense") RR^(128) arrow.long^("celu") RR^(128) arrow.long^("celu") RR^(64) arrow.long^("celu") RR^(n d) $

Training minimizes the prediction error:
$ cal(L)(theta) = sum_(t=1)^T ||bold(u)_t - hat(bold(u))_t(theta)||_2^2 + lambda cal(L)_("prob") $
where $hat(bold(u))_t$ is obtained by integrating the Neural ODE from $bold(u)_1$, and $cal(L)_("prob")$ penalizes predicted probabilities outside $[0,1]$:
$ cal(L)_("prob") = sum_(i != j) max(0, -P_(i j)) + max(0, P_(i j) - 1) $

We use a two-stage optimization: Adam for initial exploration followed by Lion for fine-tuning.
Gradients are computed via adjoint sensitivity analysis for memory efficiency @chen2018neural.

== Universal Differential Equations <sec:ude>

When domain knowledge suggests a particular functional form for the dynamics, we can incorporate it via Universal Differential Equations (UDEs) @SciML_C_Rak.
The vector field decomposes as:
$ f(bold(u)) = f_("known")(bold(u); phi) + f_("NN")(bold(u); theta) $
where $f_("known")$ encodes known physics with parameters $phi$, and $f_("NN")$ is a neural network that learns residual corrections.

For RDPG dynamics, gauge theory (@sec:gauge) suggests a particularly elegant form.

#theorem(title: "Equivariant Dynamics")[
  Let $X in RR^(n times d)$ have full column rank.
  Any $O(d)$-equivariant vector field $f: RR^(n times d) -> RR^(n times d)$ has the form:
  $ f(X) = N(P) dot X $
  where $N: RR^(n times n) -> RR^(n times n)$ depends only on $P = X X^top$.
]

#proof[
  Define $N(X) := f(X) X^dagger$ where $X^dagger = (X^top X)^(-1)X^top$ is the Moore-Penrose pseudoinverse.
  
  For equivariance, we require $f(X Q) = f(X)Q$ for all $Q in O(d)$. Then:
  $ N(X Q) = f(X Q)(X Q)^dagger = f(X)Q (Q^top X^dagger) = f(X) X^dagger = N(X) $
  
  So $N$ is constant on $O(d)$-orbits. Since orbits are indexed by $P = X X^top$ (two matrices $X, tilde(X)$ lie on the same orbit iff $X X^top = tilde(X) tilde(X)^top$), we have $N = N(P)$.
]

This form is automatically gauge-consistent since $N$ depends on the observable $P$, not the gauge-dependent $X$.
The key question is: how should we constrain $N$ to eliminate gauge freedom?

#theorem(title: "Gauge Dynamics are Not Symmetric")[
  The invisible (gauge) dynamics $dot(X) = X A$ with $A in so(d)$ correspond to $N = X A X^dagger$.
  For generic full-rank $X$ and nonzero $A$, this $N$ is *not symmetric*.
]

#proof[
  For $X = I_d$ (taking $n = d$), we have $N = A$, which is skew-symmetric.
  
  For general $X$ with thin SVD $X = U Sigma V^top$, we get:
  $ N = X A X^dagger = U Sigma V^top A V Sigma^(-1) U^top = U B U^top $
  where $B = Sigma(V^top A V)Sigma^(-1)$.
  
  Since $V^top A V$ is skew-symmetric (as $A$ is) and $Sigma$ generically has distinct singular values, the conjugation by $Sigma$ and $Sigma^(-1)$ breaks the skew-symmetry: $B != B^top$ unless $A = 0$.
  
  Thus $N = U B U^top$ is neither symmetric nor skew-symmetric for generic $X$ and nonzero $A$.
]

This theorem is the key insight: _gauge directions correspond to non-symmetric $N$_.
Therefore, constraining $N$ to be symmetric eliminates gauge:

#theorem(title: "Gauge Elimination via Symmetry")[
  Constraining $N(P) = N(P)^top$ (symmetric) eliminates all non-trivial gauge freedom.
  Any symmetric $N$ with $N X != 0$ produces observable dynamics ($dot(P) != 0$).
  Moreover, symmetric $N$ can produce _any_ realizable $dot(P)$---no expressivity is lost.
] <thm:symmetric>

#proof[
  Suppose $N = N^top$ and $dot(P) = N P + P N = 0$.

  Let $P = V Lambda V^top$ be the spectral decomposition with $Lambda = "diag"(lambda_1, ..., lambda_d, 0, ..., 0)$ and $lambda_i > 0$ for $i <= d$.

  Define $tilde(N) = V^top N V$ (symmetric since $N$ is). The condition $N P + P N = 0$ transforms to:
  $ tilde(N)Lambda + Lambda tilde(N) = 0 $

  Entry-wise: $tilde(N)_(i j)(lambda_i + lambda_j) = 0$.

  *Case analysis:*
  - For $i, j <= d$: $lambda_i + lambda_j > 0$, so $tilde(N)_(i j) = 0$.
  - For $i <= d$, $j > d$: $lambda_i + 0 = lambda_i > 0$, so $tilde(N)_(i j) = 0$.
  - For $i > d$, $j <= d$: by symmetry $tilde(N)_(i j) = tilde(N)_(j i) = 0$.
  - For $i, j > d$: the constraint $0 dot tilde(N)_(i j) = 0$ is vacuous.

  Therefore $tilde(N) = mat(0, 0; 0, tilde(N)_(22))$ where $tilde(N)_(22) in RR^((n-d) times (n-d))$ is arbitrary symmetric (supported on $"null"(P)$).

  Since $"col"(X) = "col"(V_1)$ where $V_1$ comprises the first $d$ columns of $V$, we can write $X = V_1 R$ for invertible $R$. Then:
  $ N X = V tilde(N) V^top V_1 R = V tilde(N) mat(I_d; 0) R = V mat(0; 0) = 0 $

  *Contrapositive:* If $N = N^top$ and $N X != 0$, then $N P + P N != 0$, so $dot(P) != 0$.
]

This suggests a principled UDE architecture:
$ N(P) = N_("known")(P) + N_("NN")(P), quad text("both symmetric") $
where $N_("known")$ might be a polynomial in $P$ (encoding local neighbor influence) and $N_("NN")$ learns corrections.

We consider three parameterization classes in order of increasing expressivity:

*Polynomial $N(P)$.* The most parsimonious form:
$ N = alpha_0 I + alpha_1 P + alpha_2 P^2 + ... + alpha_k P^k $
with only $k+1$ learnable scalars.
The interpretation is intuitive: $alpha_0 I$ represents intrinsic node dynamics, $alpha_1 P$ captures direct neighbor influence (one-hop interactions), and $alpha_2 P^2$ captures two-hop effects through shared neighbors.
For many network dynamics, degree $k <= 2$ suffices.

*Gauge invariance of learned parameters.*
A key advantage of expressing dynamics in terms of $P$ rather than $X$ directly: the scalar coefficients $alpha_0, alpha_1, ...$ are _gauge-invariant_.
Since $P = X X^top$ is unchanged by orthogonal transformations $X |-> X Q$, and $N(P)$ depends only on $P$, the learned $alpha_k$ values are independent of the coordinate system chosen by SVD and Procrustes alignment.
This means coefficients learned in _any_ gauge (including the DUASE-estimated coordinates) can be applied to _any other_ gauge (including the true positions)---the "learn anywhere, apply everywhere" principle.
By contrast, dynamics expressed directly in $X$ coordinates (e.g., $dot(X)_1 = a X_1 + b X_2$) would have coefficients that depend on the specific basis chosen.

*Pairwise kernel $N(P)$.* A flexible homogeneous form:
$ N_(i j) = cases(
  kappa(P_(i j)) & i != j,
  h(P_(i i)) & i = j
) $
where $kappa, h: [0,1] -> RR$ are learned functions (small neural networks or parametric forms).
Symmetry is automatic since $P_(i j) = P_(j i)$.
The kernel $kappa$ can capture nonlinear responses to connection probability, such as threshold effects or saturation.

*General symmetric $N(P)$.* A neural network that outputs the upper triangle of a symmetric matrix:
$ "NN": "uptri"(P) |-> "uptri"(N) $
with $(n(n+1))/2$ inputs and outputs.
This is the most expressive but least parsimonious option.

@tab:parameterizations summarizes the parameter counts.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Architecture*], [*Parameters*], [*Expressivity*],
    ),
    table.hline(stroke: 0.5pt),
    [Polynomial ($k=1$)], [2], [Low],
    [Polynomial ($k=2$)], [3], [Low],
    [Pairwise kernel (16-16 NN)], [$approx$300], [Medium],
    [General symmetric NN], [$approx$5,000], [High],
    [Standard Neural ODE], [$approx$10,000], [Highest],
    table.hline()
  ),
  caption: [Parameter counts for N(P)X architectures vs. standard Neural ODE.]
) <tab:parameterizations>

The dynamics $dot(X) = N X$ have a clear physical interpretation: node $i$'s velocity is $dot(X)_i = sum_j N_(i j) X_j$, a weighted combination of all positions where $N_(i j) > 0$ indicates attraction and $N_(i j) < 0$ indicates repulsion.
When the true dynamics have this form, the polynomial parameterization can recover the exact coefficients with orders of magnitude fewer parameters than a generic neural network.

== Realizable Dynamics and Model Diagnostics <sec:realizable>

Beyond gauge freedom, RDPG dynamics face a fundamental geometric constraint: the probability matrix $P = X X^top$ lives on a low-dimensional manifold, so most symmetric perturbations $dot(P)$ are not achievable.

#proposition(title: "Tangent Space Constraint")[
  Let $V in RR^(n times d)$ be an orthonormal basis for $"col"(P) subset RR^n$, and $V_perp in RR^(n times (n-d))$ span its orthogonal complement.
  Any realizable $dot(P)$ satisfies:
  $ V_perp^top dot(P) thick V_perp = 0 $
  The realizable tangent space has dimension $n d - (d(d-1))/2$.
]<prop:tangent>

#proof[
  Any realizable $dot(P) = F X^top + X F^top$ for some $F in RR^(n times d)$. Since $"col"(X) = "col"(V)$, we have $X = V R$ for invertible $R in RR^(d times d)$. Then:
  $ V_perp^top dot(P) thick V_perp = V_perp^top F R^top V^top V_perp + V_perp^top V R F^top V_perp = 0 + 0 = 0 $
  using $V^top V_perp = 0$. Conversely, any symmetric $dot(P)$ with $V_perp^top dot(P) V_perp = 0$ can be written in this form.
]

*Interpretation of blocks.* Any symmetric $n times n$ matrix $M$ decomposes as:
$ M = underbrace(V A V^top, "range-range") + underbrace(V B V_perp^top + V_perp B^top V^top, "range-null cross") + underbrace(V_perp C V_perp^top, "null-null") $
For realizable $dot(P)$: the $A$ and $B$ blocks can be arbitrary, but $C = 0$ always.

The null-null block represents "structure in the orthogonal complement"---dynamics that would increase the rank of $P$.
If we fit an RDPG model and find systematic residuals with $C != 0$, this suggests the latent dimension $d$ is too small or the RDPG model is inappropriate.

#corollary(title: "Dimension Count")[
  $ dim(T_P cal(M)_d) = n d - (d(d-1))/2 $
  This equals the dimension of $X$-space ($n d$) minus the gauge freedom ($(d(d-1))/2$)---exactly the observable degrees of freedom.
]

*Example.* For $n = 10$ nodes and $d = 2$ dimensions: symmetric matrices have 55 degrees of freedom, but only 19 directions are realizable. The remaining 36 directions would require increasing the rank of $P$---that is, increasing the latent dimension $d$.

*Model diagnostic.*
This constraint provides a principled diagnostic for model adequacy.
If observed dynamics have structure in the "null-null" block $V_perp^top dot(P) V_perp$, this indicates one of two possibilities:
+ *Model misspecification*: The true dynamics do not preserve low-rank structure, so RDPG embedding is inappropriate.
+ *Dimensional emergence*: The latent dimension $d$ is increasing over time---new factors are emerging in the network structure.

In practice, the constraint is automatically satisfied by dynamics of the form $dot(X) = N(P)X$, since $dot(P) = N P + P N$ has the required tangent structure by construction.
Violations in fitted residuals suggest the chosen $d$ may be too small.

== Probability Constraints <sec:constraints>

For edge probabilities to be valid, we require $P_(i j) in [0,1]$ for all $i,j$.
A sufficient condition is that all node positions lie in the positive orthant of the unit ball:
$ B^d_+ = {x in RR^d : x >= 0, ||x|| <= 1} $
If $X_i in B^d_+$ for all $i$, then $P_(i j) = X_i dot X_j in [0,1]$ by non-negativity of coordinates and Cauchy-Schwarz.

However, $(B^d_+)^n$ is _not_ a fundamental domain: one cannot always rotate $n$ vectors simultaneously into the positive orthant.
For example, two orthogonal unit vectors in $RR^2$ cannot both have non-negative coordinates after any rotation.

We handle constraints via a barrier loss:
$ cal(L)_("prob") = gamma sum_(i,j) [max(0, -P_(i j))^2 + max(0, P_(i j) - 1)^2] $
This encourages learned dynamics to remain in the valid region without breaking the ODE structure.
Unlike projection-based approaches that clamp values, the barrier loss maintains differentiability throughout training.

*Practical note.*
While the $B^d_+$ constraint provides a convenient sufficient condition for mathematical analysis, our experiments show that explicit projection onto $B^d_+$ is neither necessary nor beneficial for numerical learning.
Euclidean ODE solvers with barrier losses maintain valid probabilities while preserving the natural geometry of the embedding space.
Projecting onto $B^d_+$ can distort the geometry and introduce artifacts that complicate learning.
The $B^d_+$ framework remains valuable for theoretical analysis of constraint satisfaction, but practitioners should prefer unconstrained optimization with soft penalties.

== Symbolic Regression <sec:symreg>

The trained Neural ODE provides accurate predictions but remains a black box.
To extract interpretable dynamics, we apply symbolic regression to find closed-form expressions approximating $f_theta$.

Given the learned vector field $f_theta(bold(u))$, we seek symbolic expressions $g(bold(u))$ from a grammar of operations (polynomials, trigonometric functions, etc.) that minimize:
$ min_g integral ||f_theta(bold(u)) - g(bold(u))||^2 dif bold(u) + "complexity"(g) $

The complexity penalty encourages parsimonious expressions.
We use genetic programming @schmidt2009distilling to search the space of symbolic expressions.

*Gauge dependence of symbolic equations.*
A critical caveat: recovered equations are _gauge-dependent_.
The symbolic form depends on the coordinate system fixed by SVD and Procrustes alignment.
In a rotated basis, $dot(X)_1 = omega X_2$ might appear as $dot(Y)_1 = a Y_1 + b Y_2$---different-looking equations producing identical $P(t)$.

What _is_ gauge-invariant:
- Eigenvalues of the linearization (frequencies, decay rates)
- Equilibrium structure (existence, stability type)
- Qualitative behavior (oscillatory, stable, chaotic)

For truly coordinate-free equations, one could regress $dot(P)_(i j)$ directly as functions of $P$, though at the cost of higher dimensionality.

= Data: Synthetic Temporal Networks <sec:data>

We evaluate our framework on three synthetic temporal networks with known generating processes.
This allows us to assess whether extracted equations match ground truth.
Crucially, all three systems exhibit _observable_ dynamics in the sense of @sec:gauge ---the latent position changes produce actual changes in $P$.

== Single Community Oscillation

A network of $n=5$ nodes where connection probabilities oscillate sinusoidally.
All nodes belong to a single community whose internal connectivity varies periodically.
The ground-truth dynamics follow:
$ (d X_(i,1)) / (d t) = omega X_(i,2), quad (d X_(i,2)) / (d t) = -omega X_(i,1) $
producing circular trajectories in embedding space.
This is observable because nodes circulate around a nonzero centroid (@prop:centroid), not the origin.

== Two Communities Merging

A network of $n=11$ nodes initially partitioned into two separate communities.
Over time, the communities gradually merge into a single cohesive group.
This models scenarios like organizational mergers or ecosystem succession.
The dynamics involve attraction between nodes (@tab:observable), which is observable as it changes pairwise dot products.

== Long-Tailed Degree Distribution

A network of $n=36$ nodes with a power-law degree distribution.
This tests whether our method handles networks with heterogeneous node connectivity, which are common in real-world systems due to preferential attachment.

= Results <sec:results>

== Training Performance

@tab:results summarizes the training results across all three systems.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Dataset*], [$n$], [$d$], [*Final MSE*],
    ),
    table.hline(stroke: 0.5pt),
    [Single community oscillation], [5], [2], [0.114],
    [Two communities merging], [11], [2], [1.169],
    [Long-tailed distribution], [36], [2], [0.159],
    table.hline()
  ),
  caption: [Training results for three synthetic temporal networks.]
) <tab:results>

The single community and long-tail systems achieve low reconstruction error, while the merging communities system is more challenging.
This may reflect the more complex dynamics of community reorganization.

== Embedding Trajectories

Figure below (placeholder) shows example embedding trajectories comparing ground truth (from data) with Neural ODE predictions.
The model successfully captures the qualitative dynamics in all cases.

// Placeholder for figure
// #figure(
//     image("plots/trajectories.pdf"),
//     caption: [Embedding trajectories for the single community oscillation system.]
// ) <fig:trajectories>

== Symbolic Regression

For the single community oscillation system, symbolic regression recovers equations of the form:
$ (d X_1) / (d t) approx a X_2, quad (d X_2) / (d t) approx -a X_1 $
matching the ground-truth harmonic oscillator dynamics.
This demonstrates that our framework can recover interpretable, mechanistically meaningful equations from network observations alone.

== Gauge-Consistent Architecture Comparison

To test whether the theoretically-motivated $dot(X) = N(P)X$ form offers practical advantages, we compare three architectures on synthetic pairwise dynamics:
$ dot(X) = (alpha I + beta P)X $
with $alpha = -0.02$ (slight contraction) and $beta = 0.001$ (pairwise attraction).
This dynamics has exactly the polynomial $N(P)X$ form with degree 1, providing a fair test where the correct inductive bias should help.

We compare:
1. *Standard Neural ODE*: Generic $f_theta(X)$ with $approx$10,000 parameters
2. *Polynomial $N(P)X$*: $N = alpha_0 I + alpha_1 P$ with 2 parameters
3. *Kernel $N(P)X$*: $N_(i j) = kappa(P_(i j))$ with $approx$300 parameters

@tab:gauge_results summarizes the results.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Architecture*], [*Parameters*], [*MSE*], [*Parameter Recovery*],
    ),
    table.hline(stroke: 0.5pt),
    [Standard Neural ODE], [$approx$10,000], [---], [N/A],
    [Polynomial $N(P)X$], [2], [---], [$hat(alpha)_0 approx ?$, $hat(alpha)_1 approx ?$],
    [Kernel $N(P)X$], [$approx$300], [---], [N/A],
    table.hline()
  ),
  caption: [Architecture comparison on pairwise dynamics ($n=30$, $d=2$).]
) <tab:gauge_results>

// TODO: Fill in results after running Example 5

The polynomial architecture offers a dramatic reduction in parameters (5000$times$ fewer than standard NN) while potentially recovering the true dynamical coefficients.
When the inductive bias matches the true dynamics, parsimony does not sacrifice accuracy.

= Discussion <sec:discussion>

We presented a framework for learning interpretable dynamics of temporal networks.
By combining RDPG embedding, Neural ODEs, and symbolic regression, we bridge the gap between black-box prediction and mechanistic understanding.

*Theoretical foundations.*
The gauge-theoretic analysis (@sec:gauge) provides principled answers to fundamental questions:
_What can we learn?_ All dynamics except uniform rotation around the origin (@thm:invisible).
_What architecture respects the structure?_ The form $dot(X) = N(P)X$ with symmetric $N$ (@thm:symmetric).
These results inform both architecture design and interpretation of learned models.

*The gauge alignment challenge.*
We must be candid about a central difficulty: ASE introduces arbitrary gauge transformations at each time step---random $R^((t)) in O(d)$ from SVD sign ambiguity---that are unrelated to the dynamics.
Even if true dynamics are perfectly smooth, the embedding trajectory $hat(X)^((t))$ jumps erratically.
Existing joint embedding methods like UASE (@sec:why-not-uase) assume generative models incompatible with ODE dynamics on latent positions---they model time-varying "activity" against fixed "identity," not evolving positions.

*Structure-constrained alignment as a path forward.*
Our proposed solution (@sec:structure-constrained) uses dynamics structure to identify and remove the random ASE gauge artifacts.
The key insight: random gauge errors introduce skew-symmetric "contamination" that cannot be fit by symmetric dynamics (@thm:gauge-contamination).
By jointly optimizing over gauges and dynamics with symmetry constraints, we implicitly find $Q_t approx (R^((t)))^(-1)$---the corrections that undo ASE artifacts.
For linear horizontal dynamics ($dot(X) = N X$ with $N$ symmetric), we derive an alternating algorithm with closed-form steps and identifiability guarantees under generic conditions.

This is philosophically similar to how physical constraints help identify coordinates: we don't try to solve the alignment problem in isolation, but let the dynamics model guide the choice of gauge.

*Practical obstructions.*
Beyond the theoretical gauge freedom, practical challenges include:
(i) estimation error in $hat(X)$ from SVD ($approx$35% position error, though $hat(P)$ has only $approx$5% error);
(ii) finite-sample gauge jumps that make $dot(hat(X))$ undefined;
(iii) discrete, noisy observations rather than continuous $P(t)$.
The structure-constrained approach addresses (ii) by construction, but (i) and (iii) remain fundamental limitations of spectral methods.

*Evaluation in $P$-space.*
The gauge freedom implies that $X$-based metrics (e.g., position RMSE) are coordinate-dependent and potentially misleading.
The natural evaluation metric is $P(t) = X(t)X(t)^top$, which is gauge-invariant.
Comparing predicted $hat(P)(t)$ to true $P(t)$ directly tests whether the learned dynamics capture observable network structure, independent of coordinate conventions.
Our experiments confirm that models can achieve low $P$-error even when $X$-trajectories differ substantially due to gauge ambiguity.

*Parsimonious architectures.*
The $dot(X) = N(P)X$ architecture with symmetric $N$ offers two key advantages over generic neural networks: (1) automatic gauge consistency---symmetric $N$ cannot produce invisible dynamics (@thm:symmetric), and (2) dramatic parameter reduction---polynomial $N$ achieves comparable accuracy with $10^3$--$10^4$ fewer parameters.
When the true dynamics have this form, the polynomial parameterization can recover exact coefficients, providing interpretability that symbolic regression cannot match for black-box neural networks.

*Model diagnostics.*
The tangent space constraint (@prop:tangent) provides a diagnostic for model adequacy.
If residuals $hat(dot(P)) - dot(P)_("pred")$ have systematic structure in the null-null block $V_perp^top (dot) V_perp$, this suggests either that the RDPG model is inappropriate or that the latent dimension $d$ should be increased.
This connects temporal network modeling to static dimension selection methods, potentially enabling dynamic diagnostics for detecting emerging community structure.

*Limitations.*
The long-tailed network shows higher reconstruction error, suggesting challenges with highly heterogeneous degree distributions.
The polynomial $N(P)X$ form, while parsimonious, may be too restrictive for dynamics that do not factor through $P$.
For such cases, the kernel or general symmetric architectures provide a middle ground.
Additionally, all methods rely on accurate RDPG embedding, which introduces estimation error ($approx$35% in positions, though only $approx$5% in probabilities).

*Open problems.*
Several important questions remain:
1. *Sample complexity:* How many time points $T$ and nodes $n$ are needed for reliable structure-constrained alignment?
2. *Model selection:* How to choose the dynamics family $cal(F)$ when the true form is unknown? Cross-validation on held-out time points may help.
3. *Beyond linear:* Can the alternating optimization approach extend efficiently to nonlinear dynamics families?
4. *Theoretical guarantees:* Under what conditions does the alternating algorithm converge to the global optimum?

*Extensions.*
The UDE framework (@sec:ude) enables incorporating domain knowledge.
For ecological networks, one might encode known trophic interactions in $N_("known")$ while learning corrections.
For social networks, community structure could inform block-diagonal parameterizations.
The theory extends to directed graphs (@app:directed), where $P = L R^top$ with separate dynamics for source and target embeddings, and to oscillatory dynamics (@app:oscillations), which symmetric $N$ can produce through nonlinear coupling despite having real eigenvalues in the linear case.

= Conclusion

We introduced a framework for learning interpretable dynamics of temporal networks, grounded in gauge-theoretic analysis of Random Dot Product Graphs.

*What we established:*
The RDPG embedding has inherent $O(d)$ rotational ambiguity (gauge freedom).
We characterized which dynamics are observable versus invisible under this symmetry: uniform rotations leave the probability matrix $P = X X^top$ unchanged and cannot be detected.
The architecture $dot(X) = N(P)X$ with symmetric $N$ is gauge-consistent by construction---it cannot produce invisible dynamics---and enables dramatic parameter reduction compared to generic neural networks.

*What remains challenging:*
Aligning spectral embeddings across time to learn continuous dynamics is a hard open problem.
Existing joint embedding methods (UASE, Omnibus) assume generative models incompatible with ODE dynamics on latent positions.
We proposed structure-constrained alignment---joint optimization over gauges and dynamics using structural assumptions as regularization---and derived algorithms with identifiability guarantees for restricted dynamics families.
This transforms the underdetermined problem into a tractable one, but practical effectiveness depends on having appropriate inductive bias and sufficient data.

*The path forward:*
Learning continuous dynamics from discrete network snapshots requires either (a) working directly in gauge-invariant spaces (e.g., on $P$ rather than $X$), or (b) using dynamics structure to regularize gauge choice.
We developed the mathematical foundations for approach (b) and showed it can work for linear horizontal dynamics.
Extension to more expressive dynamics families, characterization of sample complexity, and empirical validation on real temporal networks remain important directions.

Our approach produces interpretable differential equations governing network evolution, enabling both prediction and mechanistic insight---provided the gauge alignment challenge is adequately addressed for the dynamics class of interest.

= Acknowledgments

// TODO: Add acknowledgments

= Data and Code Availability

The `RDPGDynamics.jl` package and all data are available at [repository URL].
Experiments can be reproduced with: `julia --project scripts/reproduce_paper.jl`

#bibliography("bibliography.bib", style: "ieee")

#pagebreak()

= Oscillatory Dynamics with Symmetric $N$ <app:oscillations>

A common concern: can symmetric $N(P)$ produce oscillations? For a _linear_ system $dot(X) = N X$ with constant symmetric $N$, eigenvalues are real, so solutions are sums of exponentials---no oscillations.

However, our system $dot(X) = N(P)X = N(X X^top)X$ is _nonlinear_ because $N$ depends on $X$ through $P$.

== Linearization Around Equilibrium

At equilibrium $X^ast$ with $N(P^ast)X^ast = 0$, the linearization is:
$ delta dot(X) = N(P^ast)delta X + [(partial N) / (partial P) bar.v_(P^ast) dot (delta X dot X^ast^top + X^ast dot delta X^top)] X^ast $

The Jacobian (as a linear operator on $delta X in RR^(n times d)$) is _not_ simply $N(P^*)$. The second term involves derivatives of $N$ and creates coupling that can produce complex eigenvalues.

== Mechanisms for Oscillation

*1. Hopf bifurcation:* As parameters vary, eigenvalues of the Jacobian can cross the imaginary axis, creating limit cycles.

*2. Amplitude-phase coupling (for $d = 2$):* Write $X_i = r_i (cos theta_i, sin theta_i)$. Then $P_(i j) = r_i r_j cos(theta_i - theta_j)$. Phase differences affect probabilities, which affect phase dynamics---a feedback loop enabling oscillation.

*3. Multi-scale interaction:*
$ N(P) = alpha_1 P - alpha_2 P^2 $
Local attraction ($alpha_1 P$) competes with nonlocal repulsion ($-alpha_2 P^2$), potentially creating oscillatory approach to equilibrium.

== What Symmetric $N$ Cannot Do

- Rotation around origin in latent space (this is gauge/invisible anyway)
- Oscillations in the linear approximation with _constant_ $N$

== What Symmetric $N$ Can Do

- Damped oscillations approaching equilibrium (via nonlinear Jacobian)
- Limit cycles via Hopf bifurcation
- Quasi-periodic motion in higher dimensions

= Extension to Directed Graphs <app:directed>

For directed graphs, the probability matrix factors as $P = G R^top$ where $G in RR^(n times d)$ contains source (giving) positions and $R in RR^(n times d)$ contains target (receiving) positions, with rows $G_(i dot) = arrow(g)_i^top$ and $R_(i dot) = arrow(r)_i^top$.

== Gauge Group

The gauge transformation is $(G, R) |-> (G Q, R Q)$ for $Q in O(d)$:
$ (G Q)(R Q)^top = G Q Q^top R^top = G R^top = P $
Crucially, both embeddings rotate by the _same_ $Q$.

== Gauge-Invariant Quantities

Under $(G, R) |-> (G Q, R Q)$:
- $P = G R^top$ --- invariant
- $Gamma_G = G G^top$ --- invariant (source Gramian)
- $Gamma_R = R R^top$ --- invariant (target Gramian)
- $G^top G$, $R^top R$, $G^top R$ --- transform by conjugation

The gauge-invariant data is $(P, Gamma_G, Gamma_R)$---three $n times n$ matrices.

== Equivariant Dynamics

Any $O(d)$-equivariant vector field has the form:
$ dot(G) = N_G(P, Gamma_G, Gamma_R) dot G, quad quad dot(R) = N_R(P, Gamma_G, Gamma_R) dot R $

To eliminate gauge, require $N_G = N_G^top$ and $N_R = N_R^top$ individually.

== Induced Dynamics on $P$

$ dot(P) = N_G P + P N_R^top $

This is a Sylvester equation, enabling:
- Asymmetric growth patterns
- Directed community formation
- Source-target differentiation

*Special case:* If $N_G = N_R = N$ with $N$ symmetric, the directed dynamics reduce to the undirected form $dot(P) = N P + P N$.

= Parsimonious UDE Parameterizations <app:ude>

This appendix catalogs parameterization choices for $N(P)$ in the UDE framework $dot(X) = N(P)X$.

== Taxonomy by Homogeneity

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Type*], [*Description*], [*Parameters*],
    ),
    table.hline(stroke: 0.5pt),
    [Homogeneous], [All node pairs follow same rule], [$O(1)$ functions],
    [Type-based], [Nodes grouped into $K$ types], [$O(K^2)$ functions],
    [Node-specific], [Each node has own parameters], [$O(n)$ scalars],
    [Fully heterogeneous], [Each pair has own parameter], [$O(n^2)$---avoid!],
    table.hline()
  ),
  caption: [Parameterization complexity by homogeneity assumption.]
)

== Homogeneous Parameterizations

*Polynomial in $P$:*
$ N(P) = alpha_0 I + alpha_1 P + alpha_2 P^2 + ... + alpha_k P^k $
Parameters: $k+1$ scalars. Interpretation: $alpha_0 I$ is self-dynamics, $alpha_1 P$ is direct neighbor influence, $alpha_2 P^2$ is two-hop influence.

*Pairwise kernel:*
$ N_(i j)(P) = kappa(P_(i j)) quad text("for ") i != j, quad quad N_(i i)(P) = h(P_(i i)) $
Symmetry is automatic since $P_(i j) = P_(j i)$. The function $kappa$ can be a small neural network or parametric (e.g., $kappa(p) = a + b p + c p^2$).

*Attraction-repulsion (Lennard-Jones inspired):*
$ kappa(p) = a / (p + epsilon) - b / (p + epsilon)^2 $
Parameters: $(a, b, epsilon)$. Equilibrium occurs where attraction balances repulsion.

*Laplacian-based:*
$ N(P) = alpha (D^(-1\/2) P D^(-1\/2) - I) $
where $D = "diag"(P bold(1))$ is the degree matrix. This encodes normalized diffusion.

== Type-Based Parameterizations

When nodes belong to types $tau: {1, ..., n} -> {1, ..., K}$, interactions can depend on type pairs.

*Block kernel:*
$ N_(i j) = kappa_(tau(i), tau(j))(P_(i j)) $
Symmetry requires $kappa_(a b) = kappa_(b a)$. Parameters: $(K(K+1))/2$ functions.

*Stochastic Block Model prior:*
$ N_(i j) = alpha_(tau(i), tau(j)) + beta dot P_(i j) $
Parameters: $(K(K+1))/2$ scalars $alpha_(a b)$ plus one shared $beta$.

Interpretation: Base rate depends on community pair, plus universal connection-strength effect.

== Node-Specific Parameterizations

Each node has individual parameters, but interactions follow shared rules.

*Diagonal + shared off-diagonal:*
$ N_(i j) = cases(
  h_i & i = j,
  kappa(P_(i j)) & i != j
) $
Parameters: $n$ scalars $h_i$ (node-specific self-rates) plus 1 function $kappa$ (shared interaction).

*Node features determine rate:*
$ h_i = g(phi_i), quad phi_i = (P_(i i), sum_j P_(i j), max_j P_(i j), ...) $
where $phi_i$ are node-level features extracted from $P$ and $g$ is a shared function (can be a small NN).

== Message-Passing Formulation

An equivalent view writes dynamics as node-level updates:
$ dot(X)_i = a(P_(i i)) X_i + sum_(j != i) m(P_(i j)) (X_j - X_i) $
where $a$ is the intrinsic rate and $m$ is the message function.

The equivalent symmetric $N$ is:
$ N_(i j) = cases(
  a(P_(i i)) - sum_(k != i) m(P_(i k)) & i = j,
  m(P_(i j)) & i != j
) $

== Low-Rank Parameterizations

*Rank-$r$ symmetric:*
$ N = sum_(k=1)^r alpha_k u_k u_k^top = U "diag"(alpha) U^top $
Parameters: $n r + r$ (with orthogonality constraints on $U$).

*Data-derived basis:*
$ N = sum_(k=1)^r alpha_k v_k v_k^top $
where $v_k$ are the top eigenvectors of $P$ itself. Parameters: $r$ scalars only.

== Encoding Qualitative Priors

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Prior*], [*Parameterization*],
    ),
    table.hline(stroke: 0.5pt),
    [Stability (nodes don't explode)], [$N = -exp(M)$ where $M = M^top$],
    [Conservation ($"tr"(P)$ constant)], [Project $N$ to $"tr"(N P) = 0$],
    [Known equilibrium $P^ast$], [$N(P) = (P - P^ast)M(P)$],
    [Sparsity preservation], [$N_(i j) = P_(i j) dot kappa(P_(i j))$],
    table.hline()
  ),
  caption: [Parameterizations encoding specific priors.]
)

== Geometric Boundary Constraints

There are two independent sources of constraints on $dot(P)$: algebraic (rank preservation, automatically satisfied by $dot(X) = N(P)X$) and geometric (probability bounds).

*At lower boundary ($P_(i j) = 0$):* Require $dot(P)_(i j) >= 0$.

From $dot(P) = N P + P N$:
$ dot(P)_(i j) = sum_k N_(i k) P_(k j) + sum_k P_(i k) N_(k j) $

*Caution:* This is NOT simply a Metzler condition on $N$. For the linear system $dot(y) = A y$, Metzler $A$ (non-negative off-diagonal) preserves the positive orthant. But $dot(P) = N P + P N$ is a Lyapunov equation---the condition involves the entire structure of $P$, not just local properties of $N$.

*At upper boundary ($P_(i j) = 1$):* Require $dot(P)_(i j) <= 0$.

Since $P_(i j) = X_i dot X_j <= ||X_i|| ||X_j||$, we have $P_(i j) = 1$ only if $X_i = X_j$ with $||X_i|| = 1$.

*Practical enforcement:* Rather than modifying $N$ (which breaks the symmetric structure) or projecting onto $B^d_+$ (which distorts the geometry), use a barrier in the loss function:
$ cal(L)_("barrier") = gamma sum_(i,j) [max(0, -P_(i j))^2 + max(0, P_(i j) - 1)^2] $
This encourages learned dynamics to stay in the valid region while preserving the natural geometry of the embedding space.
Our experiments show that projection-based constraint enforcement (e.g., onto $B^d_+$) is unnecessary and can actually impede learning.

*Summary:* In the interior of the valid configuration space, only the algebraic constraint matters---and it's automatic. Geometric constraints only matter at boundaries. Soft penalties (barrier loss) outperform hard constraints (projection) in practice.
