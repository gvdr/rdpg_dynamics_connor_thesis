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
#let od = $frak("o")$

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
    Random Dot Product Graphs as Dynamical Systems: \
    Limitations and Opportunities
  ]

  #v(1em)

  Connor Smith#super[1] and Giulio V. Dalla Riva#super[1]

  #v(0.5em)

  #text(size: 0.9em)[
    #super[1] Baffelan.com
  ]
]

#v(2em)

// Abstract
#align(center)[
  #block(width: 85%)[
    *Abstract* \
    Many phenomena across disparate fields as ecology, history, economics, social behaviors, and others, can be described in terms of networks whose edges changes in time. Temporal networks are, hence, ubiquitous in modern data science. They are commonly studied in terms of time series, with the goal of making predictions about future network states. Here, instead, we turn our attention to the task of understanding the evolution of temporal networks from the point of view of dynamical (complex) systems. That is, when the goal is to undestand why networks evolve in a certain way: we investigate the problem of learning differential equations that govern temporal networks.
    We focus on Random Dot Product Graphs (RDPG), where each network snapshot is generated from latent positions $X(t)$ via an edge probability given by $P_(i j) (t) = X_i (t) X_j (t)^top$. Here, our goal becomes that of recovering the dynamics $dot(X) = f(X)$ from samples of observed graphs.
    Here we provide for the first time theoretical limitations of what we can hope to learn, and what is beyond our reach. We identify three fundamental obstructions: (1) gauge freedom: the $O(d)$ rotational ambiguity in latent positions makes some dynamics invisible; (2) realizability constraints: the manifold structure of $P = X X^top$ restricts which perturbations are achievable; and (3) recovering trajectories from spectral embeddings, which introduce arbitrary gauge transformations at each time step.
    We develop a rigorous geometric framework using principal fiber bundles, deriving explicit formulas for the Ehresmann connection, curvature, and holonomy that govern gauge transformations along trajectories.
    We show that existing joint embedding methods (UASE, Omnibus) assume generative models incompatible with ODE dynamics, and that Bayesian smoothing approaches, while producing smooth trajectories, lack dynamical consistency as they interpolate but don't enforce that velocity is state-dependent.
    We attempt a constructive solution, we propose *structure-constrained alignment*: joint optimization over gauge corrections and dynamics parameters, where structural assumptions on the dynamics family identify and remove random embedding artifacts.
    We sketch identifiability results for linear, polynomial, and message-passing dynamics families, and derive closed-form alternating algorithms, and analyze their limits.
  ]
]

#v(2em)

= Introduction <sec:intro>

Temporal networks are networks whose edges and nodes change over time. They appear throughout science, from ecological food webs that vary across space and time @poisot2015species to neural connectomes that rewire during development.
RDPGs have proven particularly useful for such networks: Dalla Riva and Stouffer @dallariva2016exploring first applied RDPG to ecological networks, showing that latent positions capture evolutionary signatures in food webs; subsequent work has used RDPG embeddings to predict trophic interactions @strydom2022food, while the Athreya et al. survey @athreya2017statistical demonstrates applications to connectome analysis and social network inference.
A fundamental question is: *what differential equations govern the evolution of network structure?*

This paper investigates that question within the framework of Random Dot Product Graphs (RDPGs) @athreya2017statistical @young2007random.
In an RDPG, each node $i$ has a latent position $x_i in RR^d$, and the probability of an edge between nodes $i$ and $j$ is $P_(i j) = x_i^top x_j$.
When these positions evolve according to some dynamics $dot(X) = f(X)$, the network structure changes accordingly.
Our goal is to recover $f$ from observations.

This approach is appealing for several reasons.
First, the latent space provides a continuous representation where dynamics can be modeled with standard tools from dynamical systems theory.
Second, interpretable dynamics in $X$-space (e.g., attraction, repulsion, diffusion) translate to interpretable changes in network structure.
Third, the RDPG framework has well-developed statistical theory connecting latent positions to spectral properties of observed adjacency matrices.

However, we identify *three fundamental obstructions* to learning dynamics from RDPG observations:

+ *Gauge freedom* (@sec:gauge-freedom): The latent positions are determined only up to orthogonal transformation. Indeed, $X$ and $X Q$ produce identical networks for any $Q in O(d)$. This means some dynamics are *invisible*: they change $X$ without changing observable network structure.

+ *Realizability constraints* (@sec:realizable): The probability matrix $P = X X^top$ lives on a low-dimensional manifold. Not every symmetric perturbation $dot(P)$ is achievable; those that would increase the rank of $P$ are forbidden.

+ *Recovering trajectories from embeddings* (@sec:trajectory-problem): Even when true dynamics are smooth, spectral embedding produces trajectories that jump erratically due to arbitrary gauge choices at each time step. This is not a statistical issue but a fundamental ambiguity in eigendecomposition.

We show that existing approaches fail to address these obstructions.
Joint embedding methods like UASE @gallagher2021spectral and Omnibus @levin2017central assume generative models incompatible with ODE dynamics on latent positions. In fact, they model time-varying "activity" against fixed "identity," not genuinely evolving positions.
Bayesian approaches with hierarchical smoothness priors @loyal2025 produce smooth trajectories but lack *dynamical consistency*: they interpolate well but don't enforce that velocity depends on state according to $dot(X) = f(X)$.
Pairwise Procrustes alignment is local and cannot ensure global consistency.

Our main contribution is *structure-constrained alignment* (@sec:structure-constrained): a joint optimization framework where assumptions about dynamics structure (e.g., symmetry, polynomial form) regularize the otherwise underdetermined alignment problem.
We prove that random gauge artifacts introduce skew-symmetric contamination that cannot be absorbed by symmetric dynamics, enabling identification of correct gauges.
For linear, polynomial, and message-passing dynamics families, we derive algorithms with closed-form steps and prove identifiability under generic conditions.

*Paper outline.*
@sec:rdpg reviews RDPG fundamentals.
@sec:obstructions presents the obstructions with mathematical characterizations, including a detailed treatment of fiber bundle geometry.
@sec:approach describes our proposed pipeline and structure-constrained alignment algorithms.
@sec:experiments demonstrates the approach on synthetic systems.
@sec:discussion reflects on achievements and open problems.

*Notation.*
$X in RR^(n times d)$ denotes the matrix of latent positions with rows $x_i^top$.
$P = X X^top$ is the probability matrix.
$O(d)$ is the orthogonal group; $so(d)$ is its Lie algebra of skew-symmetric matrices.
$hat(X)$ denotes an estimate (e.g., from spectral embedding).
$||dot||_F$ is the Frobenius norm.
$"Sym"(d)$ denotes $d times d$ symmetric matrices; $"Skew"(d)$ denotes skew-symmetric matrices.


= Random Dot Product Graphs <sec:rdpg>

We begin with the RDPG framework for undirected networks.
The extension to directed graphs appears in @app:directed.

== Latent positions and connection probabilities

In a Random Dot Product Graph, each node $i in {1, ..., n}$ is associated with a latent position $x_i in RR^d$.
The probability of an edge between nodes $i$ and $j$ is:
$ P_(i j) = x_i^top x_j $

For this to be a valid probability, we need $P_(i j) in [0, 1]$ for all pairs.
A sufficient condition is that all positions lie in the positive orthant of the unit ball:
$ B_+^d = {x in RR^d : x_k >= 0 "for all" k, quad ||x|| <= 1} $
The non-negativity ensures $P_(i j) >= 0$, and Cauchy-Schwarz gives $P_(i j) <= ||x_i|| ||x_j|| <= 1$.

Collecting positions into a matrix $X in RR^(n times d)$ with rows $x_i^top$, the probability matrix is:
$ P = X X^top $
This is symmetric and positive semidefinite with rank at most $d$.

Given latent positions, edges are drawn independently:
$ A_(i j) tilde.op "Bernoulli"(P_(i j)) quad "for" i < j $
with $A_(j i) = A_(i j)$ (undirected) and $A_(i i) = 0$ (no self-loops).
The expected adjacency matrix is $EE[A] = P - "diag"(P)$.

== Adjacency spectral embedding <sec:ase>

Given an observed adjacency matrix $A$, we estimate latent positions via *adjacency spectral embedding* (ASE) @athreya2017statistical.

Since $A$ is symmetric, it has an eigendecomposition $A = U Lambda U^top$ with real eigenvalues $lambda_1 >= lambda_2 >= ... >= lambda_n$ and orthonormal eigenvectors.
The rank-$d$ ASE is:
$ hat(X) = U_d |Lambda_d|^(1\/2) $
where $U_d in RR^(n times d)$ contains the $d$ leading eigenvectors (by eigenvalue magnitude) and $Lambda_d = "diag"(lambda_1, ..., lambda_d)$.
We take absolute values because $A$ can have negative eigenvalues due to sampling noise, even though the true $P$ is positive semidefinite.

#remark[
  For symmetric matrices, eigendecomposition and SVD are closely related: if $A = U Lambda U^top$ (eigen) and $A = tilde(U) Sigma tilde(V)^top$ (SVD), then $tilde(U) = tilde(V) = U$ (up to signs) and $sigma_i = |lambda_i|$.
  We use eigendecomposition for ASE (the standard in RDPG literature) but SVD for Procrustes alignment (the standard for that problem).
]

*Statistical properties.*
Under mild conditions, ASE is consistent: as $n -> infinity$, the rows of $hat(X)$ converge to the true latent positions (up to orthogonal transformation) at rate $O(1\/sqrt(n))$ @athreya2017statistical.
Specifically, there exists $Q in O(d)$ such that:
$ max_i ||hat(x)_i - x_i Q|| = O_p (1 / sqrt(n)) $

The orthogonal matrix $Q$ reflects the fundamental gauge ambiguity: eigenvectors are determined only up to sign, and rotations within repeated eigenspaces are arbitrary.
This ambiguity is unavoidable and is the source of the trajectory estimation problem discussed in @sec:trajectory-problem.


= Obstructions to Learning Dynamics <sec:obstructions>

We now analyze the fundamental obstructions to learning dynamics from RDPG observations.
Here we characterize what is and isn't learnable from a theoretical perspective, without yet proposing solutions.

== What dynamics are possible? <sec:gauge-freedom>

=== Gauge freedom and observability

The latent positions $X$ are not uniquely determined by the probability matrix $P$.
For any orthogonal matrix $Q in O(d)$:
$ (X Q)(X Q)^top = X Q Q^top X^top = X X^top = P $

Thus $X$ and $X Q$ produce identical connection probabilities.
This $O(d)$ symmetry is the *gauge freedom* of RDPG: the equivalence class $[X] = {X Q : Q in O(d)}$ corresponds to a single observable network structure.

Geometrically, a global rotation of all positions in latent space leaves all pairwise angles and magnitudes unchanged, hence all dot products are preserved.

This gauge freedom has profound implications for learning dynamics.
If $X$ and $X Q$ are observationally indistinguishable, then any dynamics moving along the equivalence class, that is, rotating all positions by a common time-varying orthogonal transformation, produces *no observable change* in the network.

Consider dynamics $dot(X) = f(X)$ on the latent positions.
The induced dynamics on the probability matrix $P = X X^top$ follow from the product rule:
$ dot(P) = dot(X) X^top + X dot(X)^top = f(X) X^top + X f(X)^top $

#definition(title: "Observable and Invisible Dynamics")[
  A vector field $f: RR^(n times d) -> RR^(n times d)$ produces *observable dynamics* if $dot(P) != 0$.
  Otherwise, the dynamics are *invisible*: the latent positions change but the network structure remains static.
]

The central question is: which dynamics are invisible?

#theorem(title: "Characterization of Invisible Dynamics")[
  A vector field $f$ produces invisible dynamics ($dot(P) = 0$) if and only if $f(X) = X A$ for some skew-symmetric matrix $A in so(d)$.
] <thm:invisible>

#proof[
  ($arrow.l.double$) If $f(X) = X A$ with $A^top = -A$, then:
  $ dot(P) = f(X) X^top + X f(X)^top = X A X^top + X A^top X^top = X A X^top - X A X^top = 0 $

  ($arrow.r.double$) Suppose $dot(P) = f(X) X^top + X f(X)^top = 0$.
  Write $f(X) = X B + X_perp C$ where $X_perp$ spans the orthogonal complement of $"col"(X)$.
  Then:
  $ 0 = X B X^top + X B^top X^top + X_perp C X^top + X C^top X_perp^top $
  
  The cross terms $X_perp C X^top$ and $X C^top X_perp^top$ are in different subspaces from $X B X^top$.
  For the equation to hold, we need $C = 0$ (so $f(X) = X B$) and $B + B^top = 0$ (so $B in so(d)$).
]

*Interpretation.*
Invisible dynamics are exactly uniform rotations around the origin in latent space.
All other dynamics (attraction, repulsion, non-uniform rotation, drift) produce observable changes in network structure.

This is good news: the class of invisible dynamics is small (dimension $d(d-1)\/2$, the dimension of $so(d)$), while observable dynamics span a much larger space.


=== Realizable dynamics <sec:realizable>

Beyond gauge freedom, RDPG dynamics face a geometric constraint: the probability matrix $P = X X^top$ lives on a low-dimensional manifold, so most symmetric perturbations $dot(P)$ are not achievable.

#proposition(title: "Tangent Space Constraint")[
  Let $V in RR^(n times d)$ be an orthonormal basis for $"col"(P)$, and $V_perp in RR^(n times (n-d))$ span its orthogonal complement.
  Any realizable $dot(P)$ (i.e., $dot(P) = dot(X) X^top + X dot(X)^top$ for some $dot(X)$) satisfies:
  $ V_perp^top dot(P) V_perp = 0 $
  The realizable tangent space has dimension $n d - d(d-1)\/2$.
] <prop:tangent>

#proof[
  Any realizable $dot(P) = F X^top + X F^top$ for some $F in RR^(n times d)$.
  Since $"col"(X) = "col"(V)$, we have $X = V R$ for invertible $R$.
  Then:
  $ V_perp^top dot(P) V_perp = V_perp^top F R^top V^top V_perp + V_perp^top V R F^top V_perp = 0 $
  using $V^top V_perp = 0$.
]

*Block decomposition.*
Any symmetric matrix $M$ decomposes into blocks relative to $(V, V_perp)$:
$ M = underbrace(V A V^top, "range-range") + underbrace(V B V_perp^top + V_perp B^top V^top, "cross terms") + underbrace(V_perp C V_perp^top, "null-null") $

For realizable $dot(P)$: the $A$ and $B$ blocks can be arbitrary, but $C = 0$ always.
The null-null block represents directions that would increase the rank of $P$. Movements in these directions are forbidden for *fixed* latent dimension $d$ (but they might be achieavable for networks that start rank deficient and grow in dimension).

#corollary(title: "Dimension Count")[
  $ dim(T_P cal(B)) = n d - d(d-1)\/2 $
  This equals the dimension of $X$-space ($n d$) minus the gauge freedom ($d(d-1)\/2$).
]

*Model diagnostic.*
If observed dynamics have nonzero structure in the null-null block $V_perp^top dot(P) V_perp$, this indicates either:
+ *Model misspecification*: The true dynamics don't preserve low-rank structure
+ *Dimensional emergence*: The latent dimension $d$ is increasing, suggesting that new interaction dimensions are emerging


=== The fiber bundle perspective <sec:fiber-bundle>

The gauge freedom in RDPGs has a natural geometric structure that helps us understand what can and cannot be learned from network observations.
We now develop this structure carefully, introducing the necessary concepts with motivation before formal definitions.

==== Why fiber bundles?

The core issue is that multiple latent configurations $X$ produce the same observable $P = X X^top$.
Specifically, $X$ and $X Q$ are indistinguishable for any $Q in O(d)$.
We want a framework that:
+ Clearly separates "observable" from "gauge" degrees of freedom
+ Tells us which directions of motion in $X$-space produce observable changes
+ Tracks how gauge ambiguity accumulates along trajectories

Fiber bundle theory provides exactly this.
The intuition is:
- The *base space* $cal(B)$ consists of all possible observables $P$
- The *total space* $cal(E)$ consists of all latent configurations $X$
- The *projection* $pi: X |-> X X^top$ maps latents to observables
- The *fiber* over each $P$ is the set of all $X$ that map to it: the space where gauge freedom lives

==== The probability constraint

Before defining the bundle, we must address a constraint glossed over earlier: for $P$ to represent connection probabilities, we need $P_(i j) in [0, 1]$ for all pairs $i, j$.

Not every $X in RR^(n times d)$ satisfies this.
The constraint $(X X^top)_(i j) = x_i^top x_j in [0, 1]$ defines a semi-algebraic subset of $RR^(n times d)$.

#definition(title: "Valid Latent Positions")[
  The *valid configuration space* is:
  $ cal(E) = {X in RR^(n times d) : "rank"(X) = d, quad 0 <= x_i^top x_j <= 1 "for all" i, j} $
  
  The *valid probability space* is:
  $ cal(B) = {P in RR^(n times n) : P = X X^top "for some" X in cal(E)} $
]

A sufficient condition for $X in cal(E)$ is that all rows lie in the *positive orthant of the unit ball*:
$ x_i in B_+^d = {x in RR^d : x_k >= 0 "for all" k, quad ||x|| <= 1} $
This ensures $x_i^top x_j >= 0$ (non-negative entries) and $x_i^top x_j <= ||x_i|| ||x_j|| <= 1$ (Cauchy-Schwarz).
However, $(B_+^d)^n$ is sufficient but not necessary, infinite valid configurations exist outside it.

*Interior and boundary:*
The interior of $cal(E)$ consists of configurations where all constraints are strict: $0 < x_i^top x_j < 1$.
The boundary includes configurations where some $P_(i j) = 0$ (nodes with orthogonal positions, hence no connection probability) or $P_(i j) = 1$ (nodes with aligned unit-length positions, hence certain connection).

The fiber bundle structure is cleanest on the interior; boundary behavior requires additional care.
For most of our analysis, we work in the interior where the geometry is smooth.

==== Principal bundle structure

#definition(title: "RDPG Principal Bundle")[
  On the interior of the valid spaces, $(cal(E), cal(B), pi, O(d))$ forms a principal fiber bundle:
  - *Total space* $cal(E)$: valid latent configurations (interior)
  - *Base space* $cal(B)$: valid probability matrices (interior)
  - *Projection* $pi: cal(E) -> cal(B)$: sends $X |-> X X^top$
  - *Structure group* $O(d)$: acts on $cal(E)$ by $X dot Q = X Q$
  
  The *fiber* over $P$ is $pi^(-1)(P) = {X Q : Q in O(d)} tilde.equiv O(d)$.
]

This is a *principal bundle* because $O(d)$ acts freely (no $X$ is fixed by any non-identity $Q$) and transitively on each fiber (any two lifts of $P$ differ by some $Q$).

*What the bundle tells us:*
- The base $cal(B)$ has dimension $n d - d(d-1)\/2$: this is the "true" number of degrees of freedom in network structure
- Each fiber has dimension $d(d-1)\/2$: the gauge degrees of freedom
- Together: $dim(cal(E)) = dim(cal(B)) + dim(O(d)) = n d$

==== Decomposing motion: vertical and horizontal

At each $X in cal(E)$, we can ask: which directions of motion change $P$, and which don't?

The *vertical subspace* $cal(V)_X$ consists of directions along the fiber. Motion in these directions changes $X$ but not $P = X X^top$:
$ cal(V)_X = ker(d pi_X) = {X Omega : Omega in so(d)} $
These are exactly the *invisible dynamics* from @thm:invisible: infinitesimal rotations $dot(X) = X Omega$ with skew-symmetric $Omega$.

The *horizontal subspace* $cal(H)_X$ consists of directions transverse to the fiber. Motion in these directions do change $P$ (and $X$):
$ cal(H)_X = {Z in T_X cal(E) : X^top Z in "Sym"(d)} $

Every tangent vector decomposes uniquely into vertical and horizontal parts:
$ T_X cal(E) = cal(V)_X plus.circle cal(H)_X $

#proposition(title: "Horizontal Characterization")[
  A tangent vector $dot(X) in T_X cal(E)$ is horizontal if and only if $X^top dot(X)$ is symmetric.
] <prop:horizontal>

#proof[
  Write $dot(X) = X Omega + H$ with $Omega in so(d)$ (vertical part) and $H in cal(H)_X$ (horizontal part).
  Then $X^top dot(X) = X^top X Omega + X^top H$.
  The term $X^top H$ is symmetric by definition of $cal(H)_X$.
  The term $(X^top X) Omega$ is symmetric only if $Omega = 0$, since the product of a positive definite symmetric matrix with a nonzero skew-symmetric matrix is never symmetric.
  Thus $X^top dot(X)$ is symmetric iff $Omega = 0$ iff $dot(X)$ is purely horizontal.
]

*Interpretation:* The horizontal condition $X^top dot(X) in "Sym"(d)$ is a *gauge-fixing condition*.
It picks out, among all ways to move in $cal(E)$, the ones with no "wasted motion" along the fiber.


==== The connection 1-form: extracting gauge components

The choice of horizontal subspaces $cal(H)_X$ varying smoothly over $cal(E)$ is called an *Ehresmann connection*.
It provides a consistent way to separate "observable" from "gauge" directions throughout the bundle.
Once we have a connection, we can define parallel transport (moving along the base while staying horizontal) and curvature (measuring how the connection twists).

The Ehresmann connection can be encoded by a *connection 1-form* $omega$ that extracts the vertical (gauge) component of any motion:

#definition(title: "Connection 1-Form")[
  The connection 1-form $omega: T cal(E) -> so(d)$ is defined by:
  $ omega_X (Z) = Omega quad "where" quad (X^top X) Omega + Omega (X^top X) = X^top Z - Z^top X $
  That is, $omega_X(Z)$ is the unique skew-symmetric matrix $Omega$ solving this Lyapunov equation.
]

*Derivation:* Any tangent vector decomposes as $Z = X Omega + H$ with $Omega in so(d)$ and $X^top H$ symmetric.
Computing $X^top Z - Z^top X$: since $Omega^top = -Omega$, we have $Z^top X = -Omega (X^top X) + H^top X$.
Thus $X^top Z - Z^top X = (X^top X) Omega + Omega (X^top X) + (X^top H - H^top X)$.
Since $X^top H$ is symmetric, $(X^top H - H^top X) = 0$, giving the Lyapunov equation.
Uniqueness holds because the Lyapunov operator $Omega |-> G Omega + Omega G$ is invertible when $G = X^top X$ is positive definite: in the eigenbasis of $G$, this operator acts diagonally on skew-symmetric matrices with eigenvalues $lambda_i + lambda_j > 0$.

*Why this connection?*
The horizontal space $cal(H)_X = {Z : X^top Z "symmetric"}$ is not arbitrary: it is the *metric connection* induced by the Frobenius inner product.
Equivalently, $omega_X (Z)$ minimizes $||Z - X Omega||_F^2$ over $Omega in so(d)$: it extracts the gauge component with minimal kinetic energy.
This is the standard Riemannian connection on quotient manifolds @absil2008optimization, ensuring that horizontal lifts are geodesics when projected appropriately.

*What $omega$ tells us:*
- $omega_X (dot(X))$ is the "instantaneous rotation rate" of the motion $dot(X)$
- If $omega_X (dot(X)) = 0$, the motion is purely horizontal (no gauge component)
- If $omega_X (dot(X)) = Omega$, then $dot(X)$ includes a rotation at rate $Omega$

The connection satisfies three key properties:
+ $ker(omega_X) = cal(H)_X$: horizontal vectors have zero gauge component
+ $omega_X(X Omega) = Omega$: vertical vectors are identified with their rotation rate
+ Equivariance: $omega$ transforms appropriately under gauge changes

==== Horizontal lifts and parallel transport

Suppose we observe a trajectory $P(t)$ in the base space (observable dynamics).
We want to "lift" this to a trajectory $X(t)$ in the total space.
But there are infinitely many lifts, so which one should we choose?

The *horizontal lift* is the unique lift with no gauge drift:
$ pi(X(t)) = P(t) quad "and" quad omega_(X(t))(dot(X)(t)) = 0 $

The second condition says the velocity $dot(X)(t)$ has no vertical component at any time: the trajectory moves purely horizontally.

*Existence and uniqueness:*
Given $P(t)$ and an initial lift $X(0)$ with $pi(X(0)) = P(0)$, the horizontal lift exists and is unique @kobayashi1963foundations.
The horizontal condition $omega = 0$ defines an ODE on $cal(E)$ whose solutions project to $P(t)$.

*Why horizontal lifts matter:*
Among all trajectories $X(t)$ projecting to $P(t)$, the horizontal lift is the "gauge-canonical" one that tracks the observable dynamics without introducing spurious rotation.
If we could compute horizontal lifts from data, we would solve the gauge problem.
The difficulty is that we observe noisy snapshots $hat(X)^((t))$, not the continuous trajectory $P(t)$.

*Computing horizontal lifts.*
The horizontal lift formula connects directly to the block decomposition of @prop:tangent.
Let $P = V Lambda V^top$ where $V in RR^(n times d)$ has orthonormal columns and $Lambda = "diag"(lambda_1, ..., lambda_d)$ with $lambda_i > 0$.
A natural lift is $X = V Lambda^(1\/2)$, giving $X^top X = Lambda$.

Given realizable $dot(P)$, the block decomposition yields:
$ dot(P) = V A V^top + V B V_perp^top + V_perp B^top V^top $
where $A = V^top dot(P) V$ (symmetric, $d times d$) and $B = V^top dot(P) V_perp$ (the cross term).

We seek the horizontal lift $dot(X) = V S + V_perp T$ satisfying:
+ *Projection*: $dot(P) = dot(X) X^top + X dot(X)^top$
+ *Horizontality*: $X^top dot(X) = Lambda^(1\/2) S$ is symmetric

From the projection condition, matching blocks:
- Range-range: $A = S Lambda^(1\/2) + Lambda^(1\/2) S^top$
- Cross terms: $B = T^top Lambda^(1\/2)$, giving $T = B^top Lambda^(-1\/2)$

The horizontality condition requires $S = Lambda^(-1\/2) Sigma$ for some symmetric $Sigma$.
Substituting into the range-range equation and setting $tilde(Sigma) = Lambda^(-1\/2) Sigma Lambda^(-1\/2)$:
$ Lambda tilde(Sigma) + tilde(Sigma) Lambda = Lambda^(-1\/2) A Lambda^(-1\/2) $ <eq:horizontal-lift-lyapunov>

This is the same Lyapunov structure as in the connection 1-form above. And not coincidentally: both of them involve separating gauge from observable components.

#proposition(title: "Horizontal Lift Formula")[
  Given $P = V Lambda V^top$ and realizable $dot(P)$ with blocks $A = V^top dot(P) V$ and $B = V^top dot(P) V_perp$, the horizontal lift at $X = V Lambda^(1\/2)$ is:
  $ dot(X) = V tilde(Sigma) Lambda^(1\/2) + V_perp B^top Lambda^(-1\/2) $
  where $tilde(Sigma)$ is the symmetric solution to @eq:horizontal-lift-lyapunov, given elementwise by:
  $ tilde(Sigma)_(i j) = ((Lambda^(-1\/2) A Lambda^(-1\/2))_(i j)) / (lambda_i + lambda_j) $
]

The formula reveals two contributions to horizontal motion:
- The first term $V tilde(Sigma) Lambda^(1\/2)$ handles motion within the current column space of $X$, determined by the range-range block $A$ via a Lyapunov equation
- The second term $V_perp B^top Lambda^(-1\/2)$ handles motion expanding into new directions, determined directly by the cross block $B$

*Practical considerations:*
In our discrete setting, we observe noisy snapshots $hat(P)^((t))$ rather than continuous $P(t)$.
Computing horizontal lifts would require interpolating between snapshots and integrating the ODE.
Instead, our methods in @sec:structure-constrained use discrete Procrustes alignment, which approximates horizontal transport without explicit interpolation.

==== Curvature and holonomy

A subtle issue: even horizontal lifts can accumulate gauge drift over closed loops.

The *curvature* of the connection measures how horizontal directions fail to commute:

#definition(title: "Curvature 2-Form")[
  For horizontal vector fields $H_1, H_2$ on $cal(E)$:
  $ Omega(H_1, H_2) = -omega([H_1, H_2]) $
  where $[dot, dot]$ is the Lie bracket.
]

When curvature is nonzero, traveling horizontally in direction $H_1$ then $H_2$ doesn't end up at the same point as $H_2$ then $H_1$. So, there's a vertical (gauge) discrepancy.

This has a striking consequence for closed loops:

#definition(title: "Holonomy")[
  Let $gamma: [0,1] -> cal(B)$ be a closed curve with $gamma(0) = gamma(1) = P$.
  The horizontal lift starting at $X$ ends at $X Q_gamma$ for some $Q_gamma in O(d)$.
  The *holonomy* of $gamma$ is this accumulated rotation $Q_gamma$.
]

#theorem(title: "Holonomy Obstruction")[
  If the bundle has nontrivial curvature, there exist closed paths in $cal(B)$ such that no globally consistent gauge exists: any lift satisfies $X(1) = X(0) Q$ for some $Q != I$ @kobayashi1963foundations.
]

*Implication:* If the true dynamics $P(t)$ trace a closed loop (periodic network behavior), the underlying $X(t)$ may not close on itself. Instead, it returns rotated by the holonomy.
This is a fundamental obstruction: even perfect local alignment accumulates global gauge drift over cycles.

*Connection to spectral properties:*
The curvature of $cal(B)$ depends on the eigenvalues of $P = X X^top$.
Since the nonzero eigenvalues of $P$ are exactly those of $X^top X$ (the Gram matrix of the latent positions), small $lambda_d$ arises when:

- *Latent positions cluster in a lower-dimensional subspace*: if nodes' positions are nearly coplanar in $RR^d$, the columns of $X$ become nearly linearly dependent.

- *Stochastic block models with weak community structure*: for SBM with $X = Z B^(1\/2)$ (membership $Z$, block matrix $B$), the spectral gap of $P$ reflects that of $B$. When communities are hard to distinguish, $lambda_d$ is small.

- *Sparse networks*: if all entries of $X$ scale as $rho_n -> 0$ (sparsity parameter), then $lambda_d tilde rho_n^2 -> 0$.

This matters doubly: not only does curvature increase as $lambda_d -> 0$, but ASE convergence rates also deteriorate (indeed, they scale as $O(sqrt(log n \/ lambda_d))$ @athreya2017statistical).
Configurations with small spectral gap are thus problematic both statistically (harder to estimate) and geometrically (harder to track gauges).

#proposition(title: "Curvature and Spectral Gap")[
  The quotient manifold $cal(B) = RR_*^(n times d) \/ O(d)$ with Procrustes metric has nontrivial sectional curvature @massart2019curvature.
  For $P = X X^top$ with eigenvalues $lambda_1 >= ... >= lambda_d > 0$, the curvature increases as $lambda_d -> 0$.
]

==== Riemannian structure

The quotient $cal(B) tilde.equiv cal(E) \/ O(d)$ inherits a natural metric:

#definition(title: "Quotient Metric")[
  The Riemannian distance between $[X], [Y] in cal(B)$ is the *Procrustes distance*:
  $ d_cal(B)([X], [Y]) = min_(Q in O(d)) ||X - Y Q||_F $
]

This connects the geometry to computation: Procrustes alignment computes geodesic distance on the base space.

The *injectivity radius* at a point $[X] in cal(B)$ is the largest $r$ such that geodesics from $[X]$ of length less than $r$ remain length-minimizing and don't intersect.
Massart and Absil @massart2020quotient compute this explicitly for $cal(B)$ with the Procrustes metric: the injectivity radius at $[X]$ equals $sqrt(lambda_d)$, where $lambda_d$ is the smallest nonzero eigenvalue of $P = X X^top$.
The geometric intuition is that directions of small eigenvalue correspond to "thin" directions in the embedding. Perturbations along these directions bring the matrix closer to rank-deficiency, where the quotient structure becomes singular.
Beyond the injectivity radius, multiple geodesics can connect the same points, making interpolation non-unique.
The global injectivity radius (infimum over all points) is zero, reflecting both the nontrivial topology of $cal(B)$ and the fact that $lambda_d$ can be arbitrarily small.


== Families of RDPG dynamics <sec:dynamics-families>

Having characterized invisible dynamics (@thm:invisible) and horizontal subspaces (@prop:horizontal), we now catalog concrete families of dynamics on RDPG latent positions.
These families will later serve as inductive bias for the alignment problem (@sec:structure-constrained).

=== Linear dynamics

The simplest family has constant velocity field:
$ dot(X) = N X, quad N in RR^(n times n) $

Each node's velocity is a linear combination of all positions.
The matrix $N$ encodes interaction strengths: $N_(i j)$ determines how node $j$'s position affects node $i$'s velocity.

*Horizontal condition:* By @prop:horizontal, $dot(X) = N X$ is horizontal iff $X^top N X$ is symmetric.
A sufficient condition is $N = N^top$ (symmetric $N$), which ensures horizontality for all $X$.

#proposition(title: "Symmetric N gives horizontal dynamics")[
  If $N = N^top$, then $dot(X) = N X$ is horizontal for all $X$ with full column rank.
]

#proof[
  $X^top dot(X) = X^top N X$. Since $N$ is symmetric, $X^top N X$ is symmetric.
]

*Induced dynamics on P:*
$ dot(P) = N X X^top + X X^top N^top = N P + P N^top $
For symmetric $N$: $dot(P) = N P + P N$, a Lyapunov equation.

=== Polynomial dynamics in P

A parsimonious family with strong theoretical properties:
$ dot(X) = N(P) X, quad N(P) = sum_(k=0)^K alpha_k P^k $

Since $P = P^top$, we have $N(P) = N(P)^top$ automatically. Thus, these dynamics are *always horizontal*.

*Key property:* $P = X X^top$ is *gauge-invariant*.
This means $N(P)$ can be computed from observed data without solving the alignment problem first.

*Examples:*
- $K = 0$: $dot(X) = alpha_0 X$ (uniform expansion/contraction)
- $K = 1$: $dot(X) = (alpha_0 I + alpha_1 P) X$ (self-dynamics plus neighbor attraction)
- $K = 2$: includes second-order neighbor effects

*Interpretation of terms:*
- $P^0 = I$: self-dynamics (decay, growth)
- $P^1 = P$: direct neighbor influence (attraction if $alpha_1 > 0$)
- $P^k$: $k$-hop neighbor influence ($(P^k)_(i j)$ counts weighted paths of length $k$)

#proposition(title: "Polynomial dynamics parameter count")[
  Polynomial dynamics of degree $K$ have $K + 1$ parameters, independent of network size $n$.
]

This dramatic reduction (from $n^2$ for general $N$ to $K + 1$) is key for identifiability and interpretability.

=== Graph Laplacian dynamics

A special case with physical interpretation:
$ dot(X) = -L X, quad L = D - P $
where $D = "diag"(P bold(1))$ is the degree matrix.

This is *diffusion on the graph*: each node moves toward the weighted average of its neighbors.
Since $L = L^top$, this is horizontal.

Entry-wise: $dot(x)_i = sum_j P_(i j) (x_j - x_i)$ (move toward neighbors).

=== Message-passing dynamics

A more expressive family where velocity depends on local interactions:
$ dot(x)_i = sum_j P_(i j) g(x_i, x_j) $
for some function $g: RR^d times RR^d -> RR^d$.

*Special cases that are horizontal:*
- $g(x_i, x_j) = x_j$: gives $dot(X) = P X$ (neighbor attraction)
- $g(x_i, x_j) = x_j - x_i$: gives $dot(X) = -L X$ (Laplacian diffusion)
- $g(x_i, x_j) = W x_j$ with $W = W^top$: gives $dot(X) = P X W$

*General case:*
For arbitrary $g$, the dynamics may not be horizontal.
However, the *locality constraint* (velocity depends only on neighbors) is a strong structural assumption useful for alignment.

=== Observable despite rotation

Several dynamics that might seem "rotational" are in fact observable.
These serve as important examples of dynamics in the families above.

#proposition(title: "Centroid Circulation")[
  Dynamics of the form $dot(x)_i = (x_i - macron(x)) A$ with $macron(x) = 1/n sum_i x_i != 0$ and $A in so(d)$ produce observable changes in $P$.
] <prop:centroid>

#proof[
  The dynamics can be written as $dot(X) = X A - bold(1) macron(x)^top A$ where $bold(1)$ is the all-ones vector.
  Computing $dot(P)$:
  $ dot(P) = (X A - bold(1) macron(x)^top A) X^top + X (X A - bold(1) macron(x)^top A)^top $
  
  The $X A X^top$ terms cancel (as in the invisible case), leaving:
  $ dot(P) = -bold(1) macron(x)^top A X^top - X A^top macron(x) bold(1)^top = bold(1) v^top X^top + X v bold(1)^top $
  where $v = -A^top macron(x) = A macron(x)$.
  
  This vanishes for all pairs only if $v = 0$, i.e., $A macron(x) = 0$.
  For generic nonzero centroid $macron(x)$ and $A != 0$, we have $dot(P) != 0$.
]

*Interpretation.*
Circulation around the centroid decomposes as:
$ dot(x)_i = underbrace(x_i A, "invisible") - underbrace(macron(x) A, "shared drift") $
The first term is pure gauge; the second is a constant velocity applied to all nodes, which shifts dot products.

*Observable but not horizontal.*
While centroid circulation produces observable changes in $P$, the dynamics are _not_ horizontal.
Writing $dot(X) = (I - 1/n bold(1) bold(1)^top) X A$, we compute:
$ X^top dot(X) = C A $
where $C = X^top (I - 1/n bold(1) bold(1)^top) X$ is the sample covariance matrix of the latent positions.
The product of a symmetric matrix and a skew-symmetric matrix is symmetric only if they commute, which generically fails.
Thus the true trajectory, spiralling through the fibers, aquires vertical (gauge) component at each instant.
If structure-constrained alignment (@sec:structure-constrained) assumes horizontal dynamics, it will recover the _projection_ onto the horizontal subspace, not the exact trajectory $X(t)$.

#proposition(title: "Differential Rotation")[
  If nodes have different rotation rates $dot(x)_i = x_i A_i$ with $A_i in so(d)$, then:
  $ dot(P)_(i j) = x_i (A_i - A_j) x_j^top $
  This is generically nonzero when $A_i != A_j$.
]

#proof[
  Direct computation:
  $ dot(P)_(i j) = dot(x)_i^top x_j + x_i^top dot(x)_j = x_i A_i x_j^top + x_i A_j^top x_j^top = x_i (A_i - A_j) x_j^top $
  using $A_j^top = -A_j$.
]

=== Summary: horizontal families

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, center, left),
    stroke: none,
    table.hline(),
    table.header([*Family*], [*Form*], [*Horizontal?*], [*Parameters*]),
    table.hline(stroke: 0.5pt),
    [Linear symmetric], [$dot(X) = N X$, $N = N^top$], [Always], [$n(n+1)\/2$],
    [Polynomial], [$dot(X) = (sum_k alpha_k P^k) X$], [Always], [$K + 1$],
    [Laplacian], [$dot(X) = -L X$], [Always], [0 (fixed)],
    [Message-passing], [$dot(x)_i = sum_j P_(i j) g(x_i, x_j)$], [If $g$ symmetric], [$|theta|$],
    [Centroid circulation], [$dot(x)_i = (x_i - macron(x)) A$], [No], [$d(d-1)\/2$],
    [General linear], [$dot(X) = N X$], [If $X^top N X$ sym.], [$n^2$],
    table.hline()
  ),
  caption: [Families of RDPG dynamics and their horizontality properties. Centroid circulation is _observable_ but not horizontal, as the trajectory spirals through fibers.]
) <tab:dynamics-families>

The polynomial family stands out: it is *always horizontal*, has *few parameters* ($K + 1$ vs $n^2$), and $N(P)$ is *gauge-invariant* (computable without alignment).
These properties make it ideal for structure-constrained alignment.


== Recovering trajectories from spectral embeddings <sec:trajectory-problem>

The obstructions above concern what dynamics are *theoretically* possible.
We now turn to the *practical* problem: even when dynamics are observable and realizable, recovering trajectories from data is hard because spectral embedding introduces arbitrary gauge transformations at each time step.

=== ASE introduces random gauge transformations

At each time $t$, ASE computes the eigendecomposition of $A^((t))$.
The eigenvectors are determined only up to sign (and rotation within repeated eigenspaces).
This means:
$ hat(X)^((t)) = X^((t)) R^((t)) + E^((t)) $
where $R^((t)) in O(d)$ is determined by numerical details of the eigenvalue solver, not by the dynamics, and hence is essentially *random*. Thus, $E^((t))$ is statistical noise.

*The key point*: Even if the true positions $X^((t))$ evolve smoothly, the estimates $hat(X)^((t))$ jump erratically because the $R^((t))$ are unrelated across time.

=== Finite differences fail

Consider estimating velocity via finite differences:
$ hat(dot(X))^((t)) = (hat(X)^((t + delta t)) - hat(X)^((t))) / (delta t) $

Substituting the gauge-contaminated estimates:
$ hat(dot(X))^((t)) = (X^((t+delta t)) R^((t+delta t)) - X^((t)) R^((t))) / (delta t) + O(E) $

Even ignoring noise $E$, this is dominated by the gauge jump $R^((t+delta t)) - R^((t))$, which is $O(1)$ regardless of $delta t$.
As the "velocity" diverges as $delta t -> 0$, we're measuring gauge jumps, not dynamics.

=== Pairwise Procrustes is insufficient

One might try aligning consecutive embeddings via Procrustes:
$ Q^((t)) = arg min_(Q in O(d)) ||hat(X)^((t+1)) - hat(X)^((t)) Q||_F $

This finds the best rotation to match adjacent frames.
However:
- The solution is *local*: it doesn't ensure *global* consistency across the full trajectory
- Errors accumulate: small misalignments at each step compound
- There's no guarantee the aligned trajectory corresponds to *any* consistent dynamics

=== Pairwise alignment and error accumulation <sec:alignment-accumulation>

One might try aligning consecutive embeddings via Procrustes:
$ Q^((t)) = arg min_(Q in O(d)) ||hat(X)^((t+1)) - hat(X)^((t)) Q||_F $

This has a closed-form solution via SVD and finds the best rotation to match adjacent frames.
However, sequential pairwise alignment suffers from fundamental limitations:

+ *Local, not global:* Each alignment minimizes error between adjacent frames but doesn't ensure consistency across the full trajectory.

+ *Error accumulation:* Small misalignments at each step compound. After $T$ steps, the accumulated rotation error can be $O(sqrt(T) sigma)$ where $sigma$ is the per-step noise level.

+ *No dynamical constraint:* There's no guarantee the aligned trajectory corresponds to *any* consistent dynamics because the alignment is purely geometric.

*Noise in spectral embeddings:*
RDPG spectral embeddings have estimation error $||hat(X) - X Q||_F = O_p(sqrt(n))$ for some $Q in O(d)$, giving per-node error $O_p(1\/sqrt(n))$ @athreya2017statistical.
For pairwise Procrustes between consecutive frames, this translates to rotation estimation error that depends on both $n$ and the separation between frames.

When the true trajectory moves slowly (small $||X^((t+1)) - X^((t))||$), the signal-to-noise ratio for alignment degrades. In these cases, we're trying to detect small true rotations against a background of estimation noise.

=== Why existing joint embedding methods don't help <sec:why-not-uase>

Joint embedding methods like UASE @gallagher2021spectral embed all time points simultaneously, producing gauge-consistent estimates.
However, they assume a different generative model.

*UASE model (Multilayer RDPG):*
$ P^((t))_(i j) = x_i Lambda^((t)) y_j^((t) top) $
where $x_i$ is a *fixed* "identity" and $y_j^((t))$ is a time-varying "activity."

*Our model (temporal RDPG with ODE dynamics):*
$ P^((t))_(i j) = x_i^((t) top) x_j^((t)) $
where the *same* position $x_i^((t))$ evolves according to $dot(x)_i = f(x_i, ...)$.

The mismatch is fundamental: UASE assumes a factored structure with fixed left-embedding, while ODE dynamics on $X$ don't preserve such structure.
Applying UASE to data from our model forces the wrong decomposition, distorting the recovered trajectory.

Similar issues affect Omnibus embedding @levin2017central and COSIE @arroyo2021inference.

=== Why Bayesian smoothing approaches are insufficient <sec:bayesian-smoothing>

Recent work @loyal2025 proposes Bayesian inference for dynamic RDPGs using hierarchical priors on latent positions.
By placing priors on successive differences (or equivalently, using Gaussian processes with smooth kernels), the resulting trajectories are smooth: velocities and accelerations are well-defined and continuous.

However, *smoothness is necessary but not sufficient for dynamical consistency*.

*The fundamental distinction:*
An ODE $dot(X) = f(X)$ constrains the velocity to be a *function of the current state*.
A smoothness prior constrains velocities to vary continuously, but does not require that $dot(X)(t)$ is determined by $X(t)$.

#definition(title: "Dynamical Consistency")[
  A trajectory $X(t)$ is *dynamically consistent* with respect to a function class $cal(F)$ if there exists $f in cal(F)$ such that $dot(X)(t) = f(X(t))$ for all $t$.
]

Hierarchical Bayesian priors enforce:
- $X(t)$ is smooth #h(1em) ✓
- $dot(X)(t)$ is smooth #h(1em) ✓
- $accent(X, dot.double)(t)$ is smooth #h(1em) ✓

But they do *not* enforce:
- $dot(X)(t) = f(X(t))$ for some $f$ #h(1em) ✗

*Prior support and the ODE solution manifold:*
For a given initial condition $X(0) = X_0$ and dynamics $dot(X) = f(X)$, the solution $X(t)$ traces a *unique* curve: there is no uncertainty in the trajectory given $(f, X_0)$.
The space of all ODE solutions (over all $f in cal(F)$ and $X_0$) forms a *finite-dimensional manifold* in the infinite-dimensional space of smooth paths.

A hierarchical smoothness prior assigns positive probability to a much larger set: all smooth paths, not just those satisfying some ODE.
The ODE solution manifold has *measure zero* under such priors.

#proposition(title: "Measure Zero")[
  Let $mu$ be any Gaussian measure on smooth paths $C^k([0,T], RR^(n times d))$ with full support.
  Let $cal(M)_cal(F) = {X(dot) : dot(X) = f(X) "for some" f in cal(F)}$ be the manifold of ODE solutions.
  Then $mu(cal(M)_cal(F)) = 0$.
]

*Consequence:* The posterior under a smoothness prior concentrates on trajectories that:
+ Are smooth (from the prior)
+ Explain observed edges well (from the likelihood)
+ But do *not* satisfy $dot(X) = f(X)$ for any reasonable $f$

This is the distinction between *interpolation* and *dynamics learning*.
Both produce smooth curves through the data; only the latter respects the constraint that velocity is state-dependent.

*Empirical signature:*
In our experiments with the Loyal (2025) approach on data generated from known ODE dynamics, the recovered trajectories are indeed smooth, but:
- The inferred velocities $dot(hat(X))(t)$ do not match $f(hat(X)(t))$ for the true $f$
- The trajectories "cut corners" through state space rather than following the flow

*Diagnostics for detecting dynamical inconsistency:*

+ *State-dependence test:* For autonomous dynamics $dot(X) = f(X)$, if $X(t_1) approx X(t_2)$ at two different times, then $dot(X)(t_1) approx dot(X)(t_2)$ (same position implies same velocity).
  Smoothing priors do not enforce this.
  Find times where the trajectory passes through similar positions and compare the inferred velocities: large discrepancies indicate dynamical inconsistency.

+ *ODE residual test:* Fit candidate dynamics $hat(f)$ to the smoothed trajectory, then compute residuals $r(t) = dot(hat(X))(t) - hat(f)(hat(X)(t))$.
  For dynamically consistent trajectories, $||r(t)||$ should be small.
  For smooth-but-wrong trajectories, residuals will be systematically large.

+ *Flow consistency:* Integrate the fitted $hat(f)$ forward from various initial conditions along the trajectory.
  Dynamically consistent trajectories will track the integrated flow; interpolated trajectories will diverge.

+ *Medium-to-long-term forecasting:* Both approaches can make short-term predictions: smoothing uses Taylor extrapolation ($X(T + delta t) approx X(T) + delta t dot(X)(T) + ...$), while ODE-based methods integrate $dot(X) = f(X)$.
  The distinction emerges over longer horizons: kinematic extrapolation assumes velocity/acceleration persist, while dynamic extrapolation adapts velocity to the current state via $f(X)$.
  When the trajectory enters new regions of state space, only the latter remains accurate.

*Alternative approaches with dynamical consistency:*

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Approach*], [*Smoothness*], [*Dynamical Consistency*]),
    table.hline(stroke: 0.5pt),
    [Hierarchical GP prior], [Yes ($C^k$)], [No---interpolates only],
    [SDE $d X = f(X) d t + sigma d W$], [No (Hölder $< 1\/2$)], [Yes, as $sigma -> 0$],
    [Neural ODE], [Yes ($C^k$)], [Yes (by construction)],
    [GP-ODE (NPODE) @heinonen2018learning], [Yes ($C^1$)], [Yes---learns $f$ as GP],
    [Structure-constrained (ours)], [Yes], [Yes---family $cal(F)$ enforced],
    table.hline()
  ),
  caption: [Comparison of approaches: smoothness vs dynamical consistency.]
)

The SDE formulation $d X = f(X) d t + sigma d W$ provides a principled bridge: Freidlin-Wentzell theory @freidlin1998random shows that as $sigma -> 0$, solutions concentrate around ODE solutions with rate function $J_T(phi) = 1/2 integral_0^T ||dot(phi)(t) - f(phi(t))||^2 d t$.
This rate function is precisely the *dynamical consistency penalty*, which measures how far a path deviates from being an ODE solution.

=== Honest assessment

We must be candid: *aligning spectral embeddings to recover continuous-time trajectories is a hard open problem*.
The methods described above address related but different problems.
There is no existing method that provably recovers trajectories from ODE dynamics on RDPG latent positions.

Error accumulation in sequential alignment (@sec:alignment-accumulation) compounds over long trajectories.
Dynamical consistency considerations (@sec:bayesian-smoothing) distinguish interpolation from true dynamics learning.
Holonomy (@sec:fiber-bundle) implies that even perfect local alignment may accumulate global gauge drift.

This motivates our approach in the next section: using *structure of the dynamics themselves* to constrain the alignment problem.


= Proposed Approach <sec:approach>

We now present our constructive solution to the trajectory recovery problem.

== Pipeline overview

Our approach consists of four stages:

#figure(
  align(center)[
    #block(inset: 1em, stroke: 0.5pt, radius: 4pt)[
      $A^((1)), ..., A^((T))$ #h(0.5em) $arrow.r$ #h(0.5em)
      *ASE* #h(0.5em) $arrow.r$ #h(0.5em)
      $hat(X)^((1)), ..., hat(X)^((T))$ #h(0.5em) $arrow.r$ #h(0.5em)
      *Alignment* #h(0.5em) $arrow.r$ #h(0.5em)
      $tilde(X)^((1)), ..., tilde(X)^((T))$ #h(0.5em) $arrow.r$ #h(0.5em)
      *UDE* #h(0.5em) $arrow.r$ #h(0.5em)
      $dot(X) = f(X)$
    ]
  ],
  caption: [Pipeline from observed adjacencies to learned dynamics.]
)

+ *Spectral embedding*: Apply ASE to each $A^((t))$ independently, obtaining $hat(X)^((t))$
+ *Structure-constrained alignment*: Jointly optimize gauge corrections ${Q_t}$ and dynamics parameters, using dynamics structure as regularization
+ *Dynamics learning*: Fit a Universal Differential Equation (UDE) to the aligned trajectory
+ *Symbolic regression* (optional): Extract closed-form equations from the learned dynamics

The key innovation is step 2: we don't try to align first and learn second, but solve both problems jointly.


== Probability constraints <sec:constraints>

As discussed in @sec:fiber-bundle, valid RDPG configurations require $P_(i j) in [0, 1]$ for all pairs.
The valid configuration space $cal(E)$ is a proper subset of $RR^(n times d)$, defined by these polynomial constraints.

While the condition $X in (B_+^d)^n$ (positive orthant of the unit ball) is sufficient, it is not necessary, and infinite valid configurations exist outside this region.
Moreover, $(B_+^d)^n$ is not invariant under $O(d)$: one cannot always rotate $n$ arbitrary vectors into the positive orthant simultaneously.

For learning dynamics, we use a *barrier loss* to softly enforce probability constraints:
$ cal(L)_("prob") = gamma sum_(i,j) [max(0, -P_(i j))^2 + max(0, P_(i j) - 1)^2] $

This maintains differentiability while encouraging learned dynamics to stay in the interior of $cal(E)$.
In practice, starting from valid initial conditions and learning smooth dynamics usually maintains validity without explicit projection.


== Structure-constrained alignment <sec:structure-constrained>

=== Core insight

The alignment problem is underdetermined: without additional information, many gauge choices produce plausible trajectories.
Our insight is that *inductive bias about dynamics structure* constrains which trajectories are learnable.

We assume dynamics belong to one of the families characterized in @sec:dynamics-families, for instance, polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum_k alpha_k P^k$.
These families have two crucial properties:
+ They are *horizontal* (symmetric $N$), so the true dynamics produce no gauge drift
+ They have *restricted structure* that random gauge errors violate

If dynamics belong to family $cal(F)$, then:
- *Correct gauges* produce trajectories that can be fit by some $f in cal(F)$
- *Wrong gauges* produce trajectories requiring dynamics outside $cal(F)$

The dynamics family acts as regularization on the gauge choice.

To formalize this, we discretize the continuous dynamics $dot(X) = f(X)$ using forward Euler:
$ X(t + delta t) approx X(t) + delta t dot f(X(t)) $

Rearranging, the *residual* that should vanish for a trajectory consistent with $f$ is:
$ X(t + delta t) - X(t) - delta t dot f(X(t)) approx 0 $

The three terms have distinct roles:
- $X(t + delta t)$: where the trajectory actually ends up
- $X(t)$: where the trajectory started
- $delta t dot f(X(t))$: how far the dynamics predict we should move (velocity $times$ time)

Applying this to gauge-corrected embeddings:

#definition(title: "Joint Alignment-Learning Problem")[
  Given ASE embeddings ${hat(X)^((t))}_(t=0)^T$ and dynamics family $cal(F)$, find gauge corrections ${Q_t in O(d)}$ and $f in cal(F)$ minimizing:
  $ cal(L)({Q_t}, f) = sum_(t=0)^(T-1) ||underbrace(hat(X)^((t+1)) Q_(t+1), "aligned final") - underbrace(hat(X)^((t)) Q_t, "aligned initial") - underbrace(delta t dot f(hat(X)^((t)) Q_t), "predicted displacement")||_F^2 $
]

The objective measures how well the aligned trajectory ${hat(X)^((t)) Q_t}$ is explained by dynamics $f$: if the gauges are correct and $f$ captures the true dynamics, each step's displacement should match the predicted velocity.

*Tradeoff:*
- $cal(F)$ too restrictive: true dynamics may not fit
- $cal(F)$ too expressive (e.g., generic neural net): can fit any trajectory, no constraint on gauges

=== Why symmetric dynamics identify gauges

Consider dynamics $dot(X) = N X$ with $N = N^top$ (symmetric).
These are *horizontal*: they produce no gauge drift by construction (@prop:horizontal).

Now suppose we have the wrong gauges.
The apparent dynamics in the gauge-contaminated frame are:

#theorem(title: "Gauge Velocity Contamination")[
  Let $X(t)$ follow true dynamics $dot(X) = N X$ with $N = N^top$.
  Let $tilde(X) = X S$ for time-varying gauge error $S(t) in O(d)$.
  Then the apparent dynamics are:
  $ dot(tilde(X)) = N tilde(X) + tilde(X) Omega $
  where $Omega = S^(-1) dot(S) in so(d)$ is skew-symmetric.
  
  Moreover, $dot(tilde(X)) = tilde(N) tilde(X)$ for some symmetric $tilde(N)$ if and only if $Omega = 0$.
] <thm:gauge-contamination>

#proof[
  By the product rule: $dot(tilde(X)) = dot(X) S + X dot(S) = N X S + X dot(S)$.
  Since $X = tilde(X) S^(-1)$: $dot(tilde(X)) = N tilde(X) + tilde(X) S^(-1) dot(S) = N tilde(X) + tilde(X) Omega$.
  
  For this to equal $tilde(N) tilde(X)$ with $tilde(N)$ symmetric, we need $tilde(X)^top dot(tilde(X))$ symmetric.
  But $tilde(X)^top dot(tilde(X)) = tilde(X)^top N tilde(X) + tilde(X)^top tilde(X) Omega$.
  The first term is symmetric; the second is $(tilde(X)^top tilde(X)) Omega$.
  For positive definite $tilde(X)^top tilde(X)$ and skew $Omega$, this is symmetric only if $Omega = 0$.
]

*Mechanism:* Random ASE gauge errors $R^((t))$ introduce skew-symmetric contamination that *cannot be absorbed* by symmetric $N$.
Requiring symmetric dynamics implicitly selects gauges where the contamination vanishes, i.e., $Q_t approx (R^((t)))^(-1)$.

=== Algorithm for linear dynamics

For the family $cal(F)_("lin") = {dot(X) = N X : N = N^top}$, we derive a closed-form alternating algorithm.

*Discrete formulation:*
$ X^((t+1)) approx M X^((t)), quad M = I + delta t N quad (M = M^top) $

*Optimization:*
$ min_({R_t}, M = M^top) sum_(t=0)^(T-1) ||hat(X)^((t+1)) R_t^top - M hat(X)^((t))||_F^2 $ <eq:linear-objective>
where $R_t = Q_t^top Q_(t+1)$ are relative gauges.

*Alternating steps:*

+ *M-step* (fix ${R_t}$): Least squares with symmetry constraint
  $ M = (M^* + (M^*)^top) / 2, quad M^* = Y Z^top (Z Z^top)^(-1) $
  where $Y = [hat(X)^((1)) R_0^top | ... | hat(X)^((T)) R_(T-1)^top]$ and $Z = [hat(X)^((0)) | ... | hat(X)^((T-1))]$.

+ *R-step* (fix $M$): Orthogonal Procrustes for each $t$ (solved via SVD)
  $ R_t = V U^top quad "where" quad hat(X)^((t) top) M^top hat(X)^((t+1)) = U Sigma V^top quad "(SVD)" $

#proposition(title: "Identifiability")[
  If $M^*$ has distinct eigenvalues and the trajectory ${X^((t))}$ spans $RR^n$, then the gauges are identifiable up to a global $O(d)$ transformation.
]

=== Algorithm for polynomial dynamics

For $cal(F)_("poly") = {dot(X) = N(P) X : N(P) = sum_(k=0)^K alpha_k P^k}$, we exploit a key observation: *$P$ is gauge-invariant*.
$ P^((t)) = hat(X)^((t)) hat(X)^((t) top) = X^((t)) R^((t)) R^((t) top) X^((t) top) = X^((t)) X^((t) top) $

So we can compute $P^((t))$ directly from unaligned embeddings!

*Optimization:*
$ min_({R_t}, {alpha_k}) sum_t ||hat(X)^((t+1)) R_t^top - M(P^((t))) hat(X)^((t))||_F^2 $
where $M(P) = I + delta t sum_k alpha_k P^k$.

*Alternating steps:*

+ *$alpha$-step* (fix ${R_t}$): Linear regression
  $ alpha = G^(-1) b / delta t $
  where $G_(k j) = sum_t angle.l Phi_t^((k)), Phi_t^((j)) angle.r_F$, $b_k = sum_t angle.l Phi_t^((k)), Y_t angle.r_F$, and $Phi_t^((k)) = (P^((t)))^k hat(X)^((t))$.

+ *R-step* (fix ${alpha_k}$): Procrustes as before, with $M = M(P^((t)))$.

=== Algorithm for message-passing dynamics

For $cal(F)_("MP") = {dot(x)_i = sum_j P_(i j) g_theta (x_i, x_j)}$ with parameterized $g_theta$:

*Special horizontal cases:*
- $g(x_i, x_j) = x_j$ gives $dot(X) = P X$ (attraction to neighbors)
- $g(x_i, x_j) = x_j - x_i$ gives $dot(X) = -L X$ (Laplacian diffusion)

*General case requires gradient-based optimization:*
+ *$theta$-step*: Gradient descent on $theta$
+ *$Q$-step*: Riemannian gradient descent on $O(d)$

The structural constraint (locality) means wrong gauges introduce global correlations that local $g$ cannot fit.

=== Summary: alignment algorithms

The families from @sec:dynamics-families yield different alignment algorithms:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Family*], [*Parameters*], [*$alpha$/Dynamics step*], [*Gauge step*]),
    table.hline(stroke: 0.5pt),
    [Linear], [$n(n+1)\/2$], [Least squares + symmetrize], [Procrustes],
    [Polynomial], [$K + 1$], [Linear regression], [Procrustes],
    [Message-passing], [$|theta|$], [Gradient descent], [Riemannian GD],
    table.hline()
  ),
  caption: [Alignment algorithms for each dynamics family. Linear and polynomial have closed-form steps; message-passing requires iterative optimization.]
)


== Learning dynamics with UDEs <sec:ude>

After alignment, we learn dynamics using *Universal Differential Equations* (UDEs) @rackauckas2020universal: a framework combining mechanistic structure with neural network flexibility.

=== General UDE formulation

The general form is:
$ dot(X) = g(f_("known")(X, phi), f_("NN")(X, theta)) $

where $g$ specifies how known structure and learned components combine.
This is more general than additive UDEs:

*Additive:* $dot(X) = f_("known")(X, phi) + f_("NN")(X, theta)$
  - NN corrects known dynamics
  - Example: $dot(X) = -L X + f_theta (X)$ (Laplacian plus correction)

*Structural:* NN learns coefficients, structure imposed
  - Example: $dot(X) = (sum_k alpha_k (theta) P^k) X$ where $alpha_k$ are NN outputs
  - Structure (polynomial in $P$ from @sec:dynamics-families) is fixed; NN learns the coefficients
  - This is particularly powerful: the NN can learn time-varying or state-dependent coefficients while maintaining the horizontal property

*Multiplicative:* $dot(X) = f_("known")(X) dot.op sigma(f_("NN")(X))$
  - NN modulates known dynamics
  - Example: Activity-dependent scaling of Laplacian diffusion

=== Connecting to dynamics families

For RDPG dynamics, we use families from @sec:dynamics-families as the structural backbone:

*Polynomial UDE:*
$ dot(X) = (sum_(k=0)^K alpha_k (theta, X) P^k) X $
The coefficients $alpha_k$ can be constants (pure polynomial), functions of global state (adaptive), or NN outputs (fully flexible).
The polynomial structure ensures horizontality regardless of how $alpha_k$ varies.

*Message-passing UDE:*
$ dot(x)_i = sum_j P_(i j) g_theta (x_i, x_j) $
The NN $g_theta$ learns the local interaction function.
Locality is structural; the form of interaction is learned.

=== Training

Given aligned trajectory $tilde(X)^((0)), ..., tilde(X)^((T))$, minimize:
$ cal(L)(theta) = sum_t ||tilde(X)^((t)) - hat(X)(t; theta)||_F^2 + cal(L)_("prob") $
where $hat(X)(t; theta)$ solves the ODE $dot(X) = f_theta (X)$ from $tilde(X)^((0))$.

Gradients flow through the ODE solver via adjoint sensitivity methods, enabling efficient optimization.

#remark[
  When alignment and learning are performed jointly (@sec:structure-constrained), the UDE parameters $theta$ and gauge corrections $Q_t$ are optimized simultaneously.
  The UDE structure constrains the gauges; the gauges enable learning the UDE.
]


== Symbolic regression <sec:symreg>

The trained UDE provides accurate predictions but remains a black box.
To extract interpretable equations, we apply *symbolic regression*. This consists in canvassing the space of mathematical expressions for formulas that fit the learned dynamics.

Given the trained $f_theta$, we:
1. Sample state-velocity pairs $(X, f_theta (X))$ from the learned model
2. Apply symbolic regression (e.g., SymbolicRegression.jl @symbolicregression) to find closed-form $f^* approx f_theta$
3. Validate: compare trajectories from $dot(X) = f^*(X)$ to original data

For gauge-consistent architectures, we search for expressions of the form $N(P) X$ with $N$ a simple function of $P$.


= Experiments <sec:experiments>

We validate our approach on synthetic systems with known ground-truth dynamics.

== Synthetic systems

*Single community oscillation.*
$n = 5$ nodes in $d = 2$ dimensions.
Dynamics: $dot(x)_i = (x_i - macron(x)) A$ with $A = mat(0, -omega; omega, 0)$.
Nodes circulate around a nonzero centroid---observable despite being rotational (@prop:centroid).

*Two communities merging.*
$n = 11$ nodes initially in two clusters.
Dynamics: attraction toward global centroid, causing communities to merge.
Tests recovery of attraction dynamics.

*Long-tailed degree distribution.*
$n = 36$ nodes with heterogeneous initial positions following a long-tailed distribution.
Tests robustness to realistic degree heterogeneity.

== Results summary

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*System*], [*$n$*], [*True $|$params$|$*], [*Recovered?*]),
    table.hline(stroke: 0.5pt),
    [Oscillation], [5], [1 ($omega$)], [Yes],
    [Merging], [11], [1 (attraction rate)], [Yes],
    [Long-tailed], [36], [varies], [Partial],
    table.hline()
  ),
  caption: [Summary of experimental results.]
)

The polynomial architecture with $K = 1$ (2 parameters: $alpha_0, alpha_1$) successfully recovers the dynamics for simple systems, achieving comparable accuracy to generic neural networks with 1000$times$ fewer parameters.


= Discussion <sec:discussion>

*What we achieved.*
We provided a rigorous geometric framework for understanding dynamics on RDPGs.
The fiber bundle perspective (@sec:fiber-bundle) formalizes gauge freedom via principal bundles, with explicit formulas for the connection 1-form, curvature, and holonomy.
We showed that existing methods, such as joint embeddings assuming the wrong generative model (@sec:why-not-uase), Bayesian smoothing approaches that interpolate rather than learn dynamics (@sec:bayesian-smoothing), fail to address the fundamental obstructions.
Structure-constrained alignment exploits the geometry: symmetric dynamics cannot absorb skew-symmetric gauge contamination.

*What remains challenging.*
The structure-constrained approach requires choosing an appropriate dynamics family $cal(F)$.
If the true dynamics lie outside $cal(F)$, the method may fail or produce misleading results.
Model selection (be it choosing among polynomial degrees, message-passing architectures, etc.) remains an open problem.

Holonomy poses a fundamental challenge: for dynamics tracing closed loops in observable space, gauge drift may accumulate regardless of alignment quality.
Understanding which dynamics families have trivial holonomy is an important theoretical question.

*Open problems.*
1. What are the sample complexity requirements for dynamics recovery as a function of $n$, $d$, $T$, and the dynamics family?
2. What is the relationship between curvature of $cal(B)$ and alignment difficulty in practice?
3. Can neural SDE approaches bridge the gap between smooth interpolation and dynamically correct inference?
4. How should dimension selection be performed when $d$ may vary over time?

*Broader impact.*
Learning interpretable dynamics from network data could enable mechanistic understanding in domains from neuroscience to ecology.
However, the RDPG assumption---that connection probabilities arise from latent position dot products---is strong.
Real networks may violate this assumption, and learned dynamics should be validated against domain knowledge.


= Conclusion

We investigated the problem of learning differential equations governing time-evolving Random Dot Product Graphs, developing a rigorous geometric framework based on principal fiber bundles.

We identified three fundamental obstructions: gauge freedom (formalized via the connection 1-form and characterized by @thm:invisible), realizability constraints (the tangent space of $cal(B)$), and recovering trajectories from spectral embeddings (complicated by holonomy and information-theoretic limits).
We proved that existing joint embedding methods assume generative models incompatible with ODE dynamics, and that Bayesian random walk approaches suffer from fundamental path regularity mismatches.

The key insight is that dynamics structure regularizes the alignment problem: symmetric dynamics cannot absorb the skew-symmetric contamination from wrong gauges, enabling identification of correct gauge corrections.
For linear, polynomial, and message-passing dynamics families, we derived algorithms with closed-form steps and proved identifiability under generic conditions.

This work establishes mathematical foundations for learning interpretable dynamics from temporal network data.
Practical application requires careful model selection and validation, but the framework provides a principled path from observed adjacencies to differential equations.


= Acknowledgments

// TODO

= Data and Code Availability

Code and data are available at [repository URL].


#bibliography("bibliography.bib", style: "ieee")


#pagebreak()

// APPENDICES

= Oscillatory Dynamics with Symmetric $N$ <app:oscillations>

Can symmetric $N(P)$ produce oscillations?
For linear $dot(X) = N X$ with constant symmetric $N$, eigenvalues are real, so no oscillations.

However, $dot(X) = N(P) X$ with $P = X X^top$ is *nonlinear*.
Near an equilibrium $X^*$, linearization can yield complex eigenvalues through the coupling.

*Example:* Two nodes with $x_1, x_2 in RR^2$ and $N(P) = alpha I + beta P$.
The equilibrium and Jacobian analysis shows oscillations are possible for appropriate $alpha, beta$.


= Extension to Directed Graphs <app:directed>

For directed graphs, each node has source position $g_i$ and target position $r_i$.
The probability matrix is $P = G R^top$ (not symmetric).

*Gauge group:* $(G, R) ~ (G Q, R Q)$ for $Q in O(d)$.

*Invisible dynamics:* $dot(G) = G A$, $dot(R) = R A$ with $A in so(d)$.

The gauge-consistent architecture becomes:
$ dot(G) = N_G (P) G, quad dot(R) = N_R (P) R $
with appropriate symmetry constraints on $(N_G, N_R)$.


= Riemannian Optimization on the Quotient Manifold <app:riemannian>

This appendix provides computational details for optimization on the quotient $cal(B) = RR_*^(n times d) \/ O(d)$.

== Riemannian gradient

For a function $f: cal(B) -> RR$, the Riemannian gradient at $[X]$ is the horizontal lift of the Euclidean gradient:
$ "grad"_cal(B) f = "proj"_(cal(H)_X) (nabla f) $
where $"proj"_(cal(H)_X)(Z) = Z - X omega_X(Z)$ projects onto the horizontal space.

== Retractions

A retraction $R_X: T_X cal(B) -> cal(B)$ approximates the exponential map. Common choices:

*QR-based:* For tangent vector $Z$, compute $X + Z = Q R$ (thin QR), return $[Q]$.

*Metric projection:* Return $[U_d]$ where $X + Z = U_d Sigma V^top$ (truncated SVD).

== Vector transport

Approximate parallel transport for optimization: $T_(X -> Y)(Z) = "proj"_(cal(H)_Y)(Z)$.

== Conjugate gradient on quotient

Given $f: cal(B) -> RR$, the Riemannian CG iteration:
+ Compute gradient $g_k = "grad" f(X_k)$
+ Compute search direction $d_k = -g_k + beta_k T_(X_(k-1) -> X_k)(d_(k-1))$
+ Line search: $alpha_k = arg min_alpha f(R_(X_k)(alpha d_k))$
+ Update: $X_(k+1) = R_(X_k)(alpha_k d_k)$

The coefficient $beta_k$ uses a transported version of Fletcher-Reeves or Polak-Ribière.
