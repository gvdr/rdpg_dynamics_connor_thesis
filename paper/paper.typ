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

#let conjecture(title: none, body) = figure(
  kind: "conjecture",
  supplement: "Conjecture",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "conjecture"): it => block(width: 100%, inset: 8pt, fill: rgb("#fff3e0"), radius: 4pt, stroke: (left: 2pt + rgb("#ff9800")))[
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
    Many phenomena across disparate fields, such as ecology, history, economics, social behavior, can be described in terms of networks whose edges change in time. Temporal networks are ubiquitous in modern data science, commonly studied as time series with the goal of predicting future network states. Here, instead, we investigate the problem from the perspective of dynamical systems: when the goal is to understand _why_ networks evolve as they do, can we learn the differential equations that govern them?
    We focus on Random Dot Product Graphs (RDPG), where each network snapshot is generated from latent positions $X(t)$ via edge probabilities $P_(i j) (t) = X_i (t)^top X_j (t)$. Our goal is to recover the dynamics $dot(X) = f(X)$ from observed graphs.
    We identify three fundamental obstructions: (1) gauge freedom: the $O(d)$ rotational ambiguity in latent positions makes some dynamics invisible; (2) realizability constraints: the manifold structure of $P = X X^top$ restricts which perturbations of the network are achievable; and (3) the trajectory recovery problem: spectral embeddings introduce arbitrary gauge transformations at each time step, destroying continuity.
    We develop a rigorous geometric framework using principal fiber bundles, deriving explicit formulas for the Ehresmann connection, curvature, and holonomy that govern gauge transformations along trajectories. We catalog families of RDPG dynamics (linear, polynomial, Laplacian, message-passing) and characterize their horizontality, observability, and parameter counts.
    We establish a sharp holonomy dichotomy: polynomial dynamics have commuting generators and trivial holonomy, making gauge alignment a purely statistical problem; Laplacian dynamics generically produce full $"SO"(d)$ holonomy, making alignment simultaneously a statistical and topological challenge. We derive Cramér-Rao lower bounds showing that the same spectral gap controlling geometric difficulty also controls statistical difficulty, revealing an inextricable statistical-geometric duality.
    We show that existing joint embedding methods (UASE, Omnibus) assume generative models incompatible with ODE dynamics, and that Bayesian smoothing approaches, while producing smooth trajectories, lack dynamical consistency: they interpolate but do not enforce that velocity is state-dependent.
    We prove that symmetric dynamics structure can in principle identify correct gauges, as random gauge artifacts introduce skew-symmetric contamination that horizontal dynamics families cannot absorb, but show that this identifiability result faces significant practical obstacles: finite-sample bias and the expressiveness of natural dynamics families limit what can be recovered from realistic data. We frame the constructive problem as an open challenge and discuss directions for progress.
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

Our main contributions are theoretical.
We develop a geometric framework based on principal fiber bundles that formalizes gauge freedom, realizability, and trajectory recovery as distinct but interrelated obstructions.
We prove that random gauge artifacts introduce skew-symmetric contamination that cannot be absorbed by symmetric dynamics (@thm:gauge-contamination), establishing an identifiability principle.
We establish a holonomy dichotomy among horizontal dynamics families: polynomial dynamics have commuting generators and trivial holonomy, so gauge alignment is purely a statistical problem; Laplacian dynamics generically produce full $"SO"(d)$ holonomy, adding a topological obstruction on top of the statistical one.
We derive Cramér-Rao lower bounds showing that the spectral gap controlling curvature simultaneously controls Fisher information, so geometric and statistical difficulty are inextricable.
However, we show that exploiting these structures in practice faces fundamental difficulties: the bias induced by finite samples and the expressiveness of natural dynamics families make the constructive problem substantially harder than the identifiability theory suggests.
We frame the gap between identifiability and practical recovery as an open problem and discuss directions for progress.

*Paper outline.*
@sec:rdpg reviews RDPG fundamentals.
@sec:obstructions develops the geometric framework: gauge freedom, realizability, and the fiber bundle perspective with connections, curvature, and holonomy.
@sec:dynamics catalogs concrete dynamics families, analyzes their observable consequences through the Lyapunov equation, establishes the holonomy dichotomy, and derives information-theoretic lower bounds.
@sec:trajectory-problem addresses the practical problem of recovering trajectories from spectral embeddings, showing why existing methods fail.
@sec:constructive discusses the constructive problem: the identifiability principle, its algorithmic implications, and the practical obstacles that remain.
@sec:discussion reflects on achievements and open problems.

*Notation.*
$X in RR^(n times d)$ denotes the matrix of latent positions with rows $x_i^top$.
$P = X X^top$ is the probability matrix.
$O(d)$ is the orthogonal group; $so(d)$ is its Lie algebra of skew-symmetric matrices.
$hat(X)$ denotes an estimate (e.g., from spectral embedding).
$||dot||_F$ is the Frobenius norm.
$"Sym"(d)$ denotes $d times d$ symmetric matrices; $"Skew"(d)$ denotes skew-symmetric matrices.
Indices $i, j$ refer to nodes; indices $iota, gamma$ refer to eigenvector directions when working in a spectral basis.


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
$ A_(i j) tilde.op "Bernoulli"(P_(i j)) quad "for" i =< j $
with $A_(j i) = A_(i j)$ (undirected). It is common to disregard self-links, setting $A_(i i) = 0$, because in various applications these are not observable. This choice is not without drawbacks: we'll see that this loss of information bias the estimation of the latent positions.
The expected adjacency matrix is $EE[A] = P$ (or $EE[A] = P - "diag"(P)$ in case we disregard self-links).

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
The statistical properties of RDPG are well studied, and here we remind some fundamental results from the literature. Under mild conditions, ASE is consistent: as $n -> infinity$, the rows of $hat(X)$ converge to the true latent positions (up to orthogonal transformation) at rate $O(1\/sqrt(n))$ @athreya2016limit @athreya2017statistical.
Specifically, there exists $Q in O(d)$ such that $max_i ||hat(x)_i - x_i Q|| = O_p (1 / sqrt(n))$, and conditionally on each latent position, $sqrt(n)(hat(x)_i - Q x_i)$ converges to a Gaussian whose covariance depends on the edge variance profile @athreya2016limit @rubindelanchy2022statistical.
The perturbation expansion $hat(X) = X W + (A - P) X (X^top X)^(-1) W + R$, where $W in O(d)$ and $||R||_(2 -> infinity) = o(n^(-1\/2))$, shows that the leading error term is linear in $A - P$ and has conditional mean zero @cape2019two.
ASE is asymptotically unbiased but not fully efficient for individual latent positions; a one-step Newton correction achieves the semiparametric efficiency bound @xie2021efficient.
For stochastic blockmodel parameters, the spectral estimator is itself asymptotically efficient @tang2022efficient.
The eigenvalues of $A$ are also asymptotically normal about those of $P$, with an $O(1)$ bias from the Bernoulli variance and the hollow diagonal that is negligible relative to the $O(n)$ eigenvalue magnitudes @tang2018eigenvalues.

The orthogonal matrix $Q$ reflects the fundamental gauge ambiguity: eigenvectors are determined only up to sign, and rotations within repeated eigenspaces are arbitrary.
This ambiguity is unavoidable and is the source of the trajectory estimation problem discussed in @sec:trajectory-problem.


= The Gauge Obstruction <sec:obstructions>

We analyze the fundamental obstructions to learning dynamics from RDPG observations, characterizing what is and is not learnable from a theoretical perspective.

== Gauge freedom and observability <sec:gauge-freedom>

The latent positions $X$ are not uniquely determined by the probability matrix $P$.
For any orthogonal matrix $Q in O(d)$:
$ (X Q)(X Q)^top = X Q Q^top X^top = X X^top = P $

Thus $X$ and $X Q$ produce identical connection probabilities.
This $O(d)$ symmetry is the *gauge freedom* of RDPG: the equivalence class $[X] = {X Q : Q in O(d)}$ corresponds to a single observable network structure.

From a geometric point of view, a global rotation of all positions in latent space leaves all pairwise angles and magnitudes unchanged, hence all dot products are preserved.

This gauge freedom has profound implications for learning dynamics.
If $X$ and $X Q$ are observationally indistinguishable, then any dynamics moving along the equivalence class, that is, rotating all positions by a common time-varying orthogonal transformation, produces *no observable change* in the network.

Consider dynamics $dot(X) = f(X)$ on the latent positions.
The induced dynamics on the probability matrix $P = X X^top$ follow from the product rule:
$ dot(P) = dot(X) X^top + X dot(X)^top = f(X) X^top + X f(X)^top $

#definition(title: "Observable and Invisible Dynamics")[
  A vector field $f: RR^(n times d) -> RR^(n times d)$ produces *observable dynamics* if $dot(P) != 0$.
  Otherwise, the dynamics are *invisible*: the latent positions change but the network structure remains static.
]

The definition calls for a natural question: which dynamics are invisible?

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

We can readily interpret the result: invisible dynamics are exactly uniform rotations around the origin in latent space; all other dynamics (attraction, repulsion, non-uniform rotation, drift, ...) produce observable changes in network structure.

This implies that the class of invisible dynamics is small (dimension $d(d-1)\/2$, the dimension of $so(d)$), while observable dynamics span a much larger space.


== Realizable dynamics <sec:realizable>

Beyond gauge freedom, RDPG dynamics face a geometric constraint: the probability matrix $P = X X^top$ lives on a low-dimensional manifold, so most (symmetric) perturbations $dot(P)$ are not achievable by dynamics of $X$.

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

We can better understand the dynamics by considering a *block decomposition* of the generating matrices. Observe that any symmetric matrix $M$ decomposes into blocks relative to $(V, V_perp)$:
$ M = underbrace(V A V^top, "range-range") + underbrace(V B V_perp^top + V_perp B^top V^top, "cross terms") + underbrace(V_perp C V_perp^top, "null-null") $

For realizable $dot(P)$: the $A$ and $B$ blocks can be arbitrary, but $C = 0$ always.
The null-null block represents directions that would increase the rank of $P$. Movements in these directions are forbidden for *fixed* latent dimension $d$ (but they might be achieavable for networks that start rank deficient and grow in dimension).

#corollary(title: "Dimension Count")[
  $ dim(T_P cal(B)) = n d - d(d-1)\/2 $
  This equals the dimension of $X$-space ($n d$) minus the gauge freedom ($d(d-1)\/2$).
]

The block decomposition suggest an immediate diagnostic for the model: if observed dynamics have nonzero structure in the null-null block $V_perp^top dot(P) V_perp$, this indicates either:
+ *Model misspecification*: The true dynamics don't preserve low-rank structure
+ *Dimensional emergence*: The latent dimension $d$ is increasing, suggesting that new interaction dimensions are emerging


== The fiber bundle perspective <sec:fiber-bundle>

The gauge freedom in RDPGs has a natural geometric structure that helps us understand what can and cannot be learned from network observations.
The quotient geometry of $RR_*^(n times d) \/ O(d)$ has been developed extensively in the optimization literature, particularly by Absil, Mahony, and Sepulchre @absil2008optimization, Journée, Bach, Absil, and Sepulchre @journee2010low, and Massart and Absil @massart2020quotient @massart2019curvature.
We recall this geometry here, adapting it to the RDPG setting where the connections, curvature, and holonomy structures acquire concrete statistical meaning as obstructions to dynamics estimation.

=== Why fiber bundles?

Remember the fundamental unidentifiability for RDPG: that multiple latent configurations $X$ produce the same observable $P = X X^top$: $X$ and $X Q$ are indistinguishable for any $Q in O(d)$.
Hence, we seek to establish a framework that:
+ Clearly separates "observable" from "gauge" degrees of freedom
+ Tells us which directions of motion in $X$-space produce observable changes
+ Tracks how gauge ambiguity accumulates along trajectories

The geometric theory of fiber bundles provides exactly this.
The intuition is:
- The *base space* $cal(B)$ consists of all possible observables $P$
- The *total space* $cal(E)$ consists of all latent configurations $X$
- The *projection* $pi: X |-> X X^top$ maps latents to observables
- The *fiber* over each $P$ is the set of all $X$ that map to it: the space where gauge freedom lives

=== The probability constraint

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

*Interior and boundary.*
The interior of $cal(E)$ consists of configurations where all constraints are strict: $0 < x_i^top x_j < 1$.
The boundary includes configurations where some $P_(i j) = 0$ (nodes with orthogonal positions, hence no connection probability) or $P_(i j) = 1$ (nodes with aligned unit-length positions, hence certain connection).

The fiber bundle structure requires a smooth total space, which holds on the interior: there, $cal(E)$ is an open subset of the smooth manifold ${X in RR^(n times d) : "rank"(X) = d}$, and all the machinery of connections, curvature, and holonomy applies without caveat.

At the boundary, three issues arise.
First, the feasible set ${X : 0 <= x_i^top x_j <= 1 "for all" i, j}$ is defined by polynomial inequalities, and has *corners* where multiple constraints are active simultaneously. Hence, it is a manifold with corners, not a smooth manifold.
Second, at boundary points the tangent space is replaced by a *tangent cone*: not all directions are available, because some would push $P_(i j)$ outside $[0, 1]$.
The horizontal-vertical decomposition of @prop:horizontal may not respect feasibility: the horizontal component of a feasible velocity can point outside the constraint set.
Third, if a trajectory reaches $P_(i j) = 0$ or $P_(i j) = 1$, the unconstrained ODE $dot(X) = f(X)$ may attempt to leave the feasible region, requiring constraint handling (projection, reflection, or barrier terms as discussed in @sec:constructive).

In practice, the boundary issue is rarely binding for temporal networks of interest: edge probabilities that are exactly 0 or 1 correspond to nodes that never interact, which are typically excluded from temporal network analysis (as they are "undetectable"), or always interact (which mean they don't have stochastic variability in the network structure).
For the remainder of this paper, we work on the interior of $cal(E)$, where the geometry is smooth and the bundle structure is clean.

=== Principal bundle structure

#definition(title: "RDPG Principal Bundle")[
  On the interior of the valid spaces, $(cal(E), cal(B), pi, O(d))$ forms a principal fiber bundle:
  - *Total space* $cal(E)$: valid latent configurations (interior)
  - *Base space* $cal(B)$: valid probability matrices (interior)
  - *Projection* $pi: cal(E) -> cal(B)$: sends $X |-> X X^top$
  - *Structure group* $O(d)$: acts on $cal(E)$ by $X dot Q = X Q$

  The *fiber* over $P$ is $pi^(-1)(P) = {X Q : Q in O(d)} tilde.equiv O(d)$.
]

This is a *principal bundle* because $O(d)$ acts freely (no $X$ is fixed by any non-identity $Q$) and transitively on each fiber (any two lifts of $P$ differ by some $Q$).

The fiber bundle confirm our dimensional understanding. The base $cal(B)$ has dimension $n d - d(d-1)\/2$: this is the "true" number of degrees of freedom in network structure. Each fiber has dimension $d(d-1)\/2$: the gauge degrees of freedom. Together, the the base and the bundle recover the full dimension of the space: $dim(cal(E)) = dim(cal(B)) + dim(O(d)) = n d$

=== Decomposing motion: vertical and horizontal

At each $X in cal(E)$, we can ask: which directions of motion change $P$, and which don't?

The *vertical subspace* $cal(V)_X$ consists of directions along the fiber. Motion in these directions changes $X$ but not $P = X X^top$:
$ cal(V)_X = ker(d pi_X) = {X Omega : Omega in so(d)} $
These are exactly the *invisible dynamics* from @thm:invisible: infinitesimal rotations $dot(X) = X Omega$ with skew-symmetric $Omega$.

The *horizontal subspace* $cal(H)_X$ consists of directions transverse to the fiber. Motion in these directions do change $P$ (and $X$):
$ cal(H)_X = {Z in T_X cal(E) : X^top Z in "Sym"(d)} $

Every tangent vector decomposes uniquely into vertical and horizontal parts:
$ T_X cal(E) = cal(V)_X plus.o cal(H)_X $

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

The horizontal condition $X^top dot(X) in "Sym"(d)$ is a *gauge-fixing condition*: it picks out, among all ways to move in $cal(E)$, the ones with no "wasted motion" along the fiber.


=== The connection 1-form: extracting gauge components

The choice of horizontal subspaces $cal(H)_X$ varying smoothly over $cal(E)$ is called an *Ehresmann connection*.
It provides a consistent way to separate "observable" from "gauge" directions throughout the bundle.
Once we have a connection, we can define parallel transport (moving along the base while staying horizontal) and curvature (measuring how the connection twists).

The Ehresmann connection can be encoded by a *connection 1-form* $omega$ that extracts the vertical (gauge) component of any motion:

#definition(title: "Connection 1-Form")[
  The connection 1-form $omega: T cal(E) -> so(d)$ is defined by:
  $ omega_X (Z) = Omega quad "where" quad (X^top X) Omega + Omega (X^top X) = X^top Z - Z^top X $
  That is, $omega_X(Z)$ is the unique skew-symmetric matrix $Omega$ solving this Lyapunov equation.
]

The horizontal space $cal(H)_X = {Z : X^top Z "symmetric"}$ is not an arbitrary object: it is the *metric connection* induced by the Frobenius inner product.
Equivalently, $omega_X (Z)$ minimizes $||Z - X Omega||_F^2$ over $Omega in so(d)$: it extracts the gauge component with minimal kinetic energy.
This is the standard Riemannian connection on quotient manifolds @absil2008optimization @massart2020quotient, ensuring that horizontal lifts are geodesics when projected appropriately.

To derive its expression, remember that any tangent vector decomposes as $Z = X Omega + H$ with $Omega in so(d)$ and $X^top H$ symmetric.
Hence, computing $X^top Z - Z^top X$: since $Omega^top = -Omega$, we have $Z^top X = -Omega (X^top X) + H^top X$.
Thus $X^top Z - Z^top X = (X^top X) Omega + Omega (X^top X) + (X^top H - H^top X)$.
Since $X^top H$ is symmetric, $(X^top H - H^top X) = 0$, giving the Lyapunov equation.
Uniqueness holds because the Lyapunov operator $Omega |-> G Omega + Omega G$ is invertible when $G = X^top X$ is positive definite: in the eigenbasis of $G$, this operator acts diagonally on skew-symmetric matrices with eigenvalues $lambda_iota + lambda_gamma > 0$.

The connection 1-form $omega_X (dot(X))$ is the "instantaneous rotation rate" of the motion $dot(X)$: if $omega_X (dot(X)) = 0$, the motion is purely horizontal (no gauge component); if $omega_X (dot(X)) = Omega$, then $dot(X)$ includes a rotation at rate $Omega$

The connection satisfies three key properties:
+ $ker(omega_X) = cal(H)_X$: horizontal vectors have zero gauge component
+ $omega_X(X Omega) = Omega$: vertical vectors are identified with their rotation rate
+ Equivariance: $omega$ transforms appropriately under gauge changes

=== Horizontal lifts and parallel transport

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
Let $P = V Lambda V^top$ where $V in RR^(n times d)$ has orthonormal columns and $Lambda = "diag"(lambda_1, ..., lambda_d)$ with $lambda_iota > 0$.
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

This is the same Lyapunov structure as in the connection 1-form above. And not coincidentally: both of them involve separating gauge from observable components. The following formula is essentially the horizontal projection of @absil2008optimization and @massart2020quotient, specialized to the spectral decomposition of $P$.

#proposition(title: "Horizontal Lift Formula")[
  Given $P = V Lambda V^top$ and realizable $dot(P)$ with blocks $A = V^top dot(P) V$ and $B = V^top dot(P) V_perp$, the horizontal lift at $X = V Lambda^(1\/2)$ is:
  $ dot(X) = V tilde(Sigma) Lambda^(1\/2) + V_perp B^top Lambda^(-1\/2) $
  where $tilde(Sigma)$ is the symmetric solution to @eq:horizontal-lift-lyapunov, given elementwise by:
  $ tilde(Sigma)_(iota gamma) = ((Lambda^(-1\/2) A Lambda^(-1\/2))_(iota gamma)) / (lambda_iota + lambda_gamma) $
]

The formula reveals two contributions to horizontal motion:
- The first term $V tilde(Sigma) Lambda^(1\/2)$ handles motion within the current column space of $X$, determined by the range-range block $A$ via a Lyapunov equation
- The second term $V_perp B^top Lambda^(-1\/2)$ handles motion expanding into new directions, determined directly by the cross block $B$

Some practical considerations temper our theoretical result: in our discrete setting, we observe noisy snapshots $hat(P)^((t))$ rather than continuous $P(t)$.
Computing horizontal lifts would require interpolating between snapshots and integrating the ODE.
Instead, we discuss in @sec:constructive how discrete Procrustes alignment approximates horizontal transport without explicit interpolation.

=== Curvature and holonomy

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

A classic result (see for example @kobayashi1963foundations) provides a precise characterization of the holonomy obstruction.

#theorem(title: "Holonomy Obstruction")[
  If the bundle has nontrivial curvature, there exist closed paths in $cal(B)$ such that no globally consistent gauge exists: any lift satisfies $X(1) = X(0) Q$ for some $Q != I$.
]

The results implies that, if the true dynamics $P(t)$ trace a closed loop (periodic network behavior), the underlying $X(t)$ may not close on itself. Instead, it returns rotated by the holonomy.
This is a fundamental obstruction: even perfect local alignment accumulates global gauge drift over cycles.

It is interesting to notice that the holonomy obstructions provides a *connection to spectral properties* of the (observed) graphs: in fact, the curvature of $cal(B)$ depends on the eigenvalues of $P = X X^top$. Since the nonzero eigenvalues of $P$ are exactly those of $X^top X$ (the Gram matrix of the latent positions), small $lambda_d$ arises when, for example:

- *Latent positions cluster in a lower-dimensional subspace*: if nodes' positions are nearly coplanar in $RR^d$, the columns of $X$ become nearly linearly dependent.

- *Stochastic block models with weak community structure*: for SBM with $X = Z B^(1\/2)$ (membership $Z$, block matrix $B$), the spectral gap of $P$ reflects that of $B$. When communities are hard to distinguish, $lambda_d$ is small.

- *Sparse networks*: if all entries of $X$ scale as $rho_n -> 0$ (sparsity parameter), then $lambda_d tilde rho_n^2 -> 0$.

This matters doubly: not only does curvature increase as $lambda_d -> 0$, but ASE convergence rates also deteriorate (indeed, they scale as $O(sqrt(log n \/ lambda_d))$ @cape2019two).
Configurations with small spectral gap are thus problematic both statistically (harder to estimate) and geometrically (harder to track gauges).

#proposition(title: "Curvature and Spectral Gap")[
  The quotient manifold $cal(B) = RR_*^(n times d) \/ O(d)$ with the Procrustes metric has sectional curvature given by O'Neill's formula for Riemannian submersions @oneill1966fundamental: for orthonormal horizontal vectors $overline(xi), overline(eta) in cal(H)_X$,
  $ K(xi, eta) = 3 ||[overline(xi), overline(eta)]^cal(V)||^2 $
  where $[overline(xi), overline(eta)]^cal(V)$ denotes the vertical component of the Lie bracket.

  Massart, Hendrickx, and Absil @massart2019curvature compute the sectional curvature explicitly for this quotient:
  - If $d = 1$, the sectional curvature is identically zero (the base space is flat).
  - If $d >= 2$, the minimum sectional curvature is zero, achieved when the horizontal fields commute.
  - The maximum sectional curvature at $[X]$ diverges as $lambda_d -> 0$: specifically, it blows up when the two smallest eigenvalues $lambda_(d-1), lambda_d$ of $P = X X^top$ approach zero simultaneously.
]<prop:curv-spectral-gap>

We can read this result from the point of view of the dynamics estimation. In fact, the curvature formula reveals why alignment becomes harder near rank-deficiency.
The vertical component $[overline(xi), overline(eta)]^cal(V)$ measures how much two horizontal motions "twist" relative to each other, in other words, how much gauge drift accumulates when traversing an infinitesimal parallelogram in $cal(B)$.
Large curvature means that even short paths in the base space can produce substantial holonomy, making local Procrustes alignment inconsistent over moderate distances.
The $d = 1$ case is degenerate in a useful way: $O(1) = {plus.minus 1}$ is discrete, so the only gauge ambiguity is a global sign flip, and the curvature vanishes because there is no continuous rotation to accumulate.

=== Riemannian structure

The quotient $cal(B) tilde.equiv cal(E) \/ O(d)$ inherits a natural metric:

#definition(title: "Quotient Metric")[
  The Riemannian distance between $[X], [Y] in cal(B)$ is the *Procrustes distance*:
  $ d_cal(B)([X], [Y]) = min_(Q in O(d)) ||X - Y Q||_F $
]

This connects the geometry to computation: Procrustes alignment computes geodesic distance on the base space @massart2020quotient.

The *injectivity radius* at a point $[X] in cal(B)$ is the largest $r$ such that geodesics from $[X]$ of length less than $r$ remain length-minimizing and don't intersect.
Massart and Absil @massart2020quotient compute this explicitly for $cal(B)$ with the Procrustes metric: the injectivity radius at $[X]$ equals $sqrt(lambda_d)$, where $lambda_d$ is the smallest nonzero eigenvalue of $P = X X^top$.
The geometric intuition is that directions of small eigenvalue correspond to "thin" directions in the embedding. Perturbations along these directions bring the matrix closer to rank-deficiency, where the quotient structure becomes singular.
Beyond the injectivity radius, multiple geodesics can connect the same points, making interpolation non-unique.
The global injectivity radius (infimum over all points) is zero, reflecting both the nontrivial topology of $cal(B)$ and the fact that $lambda_d$ can be arbitrarily small.


= Dynamics on RDPGs <sec:dynamics>

The geometric framework of the previous section characterizes the abstract structure of gauge freedom, curvature, and holonomy.
We now turn to concrete dynamics families, their observable consequences, and the interplay between geometric and statistical obstructions to estimation.

== Families of RDPG dynamics <sec:dynamics-families>

We now catalog concrete families of dynamics on RDPG latent positions.
These families will later serve as inductive bias for the alignment problem (@sec:constructive); the holonomy and statistical analyses that follow depend on their specific structure.

*Linear dynamics.*
The simplest family is $dot(X) = N X$ with $N in RR^(n times n)$, where $N_(i j)$ determines how node $j$'s position affects node $i$'s velocity.
By @prop:horizontal, this is horizontal if and only if $X^top N X$ is symmetric; a sufficient condition is $N = N^top$, which gives $X^top N X$ symmetric for all $X$.
The induced $P$-dynamics are $dot(P) = N P + P N^top$, which reduces to the Lyapunov equation $dot(P) = N P + P N$ when $N$ is symmetric.
Symmetric linear dynamics have $n(n+1)/2$ free parameters.

*Polynomial dynamics in $P$.*
A far more parsimonious family replaces the constant $N$ with a polynomial in the probability matrix:
$ dot(X) = N(P) X, quad N(P) = sum_(k=0)^K alpha_k P^k $
Since $P$ is symmetric, $N(P)$ is automatically symmetric and these dynamics are *always horizontal*, regardless of the coefficients $alpha_k$.
The terms have a clean interpretation: $P^0 = I$ gives self-dynamics (decay or growth), $P^1 = P$ encodes direct neighbor influence, and $P^k$ captures $k$-hop effects through weighted path counts $(P^k)_(i j)$.
Crucially, $P = X X^top$ is gauge-invariant, so $N(P)$ can be computed from observations without solving the alignment problem.
The family has only $K + 1$ parameters independent of $n$, a dramatic reduction from $n(n+1)/2$ for general symmetric $N$ that is key for identifiability.

*Graph Laplacian dynamics.*
Setting $dot(X) = -L X$ with $L = D - P$ and $D = "diag"(P bold(1))$ gives diffusion on the graph: entry-wise, $dot(x)_i = sum_j P_(i j)(x_j - x_i)$, so each node moves toward the weighted average of its neighbors.
Since $L$ is symmetric, this is horizontal.
Unlike polynomial dynamics, the Laplacian is *not* a polynomial in $P$: the degree matrix $D$ depends on row sums, which mix eigenvector information that polynomials in $P$ cannot access.
This distinction will prove crucial for holonomy (@sec:holonomy-dynamics).

*Message-passing dynamics.*
More generally, $dot(x)_i = sum_j P_(i j) g(x_i, x_j)$ for some interaction function $g: RR^d times RR^d -> RR^d$.
Special cases include neighbor attraction ($g(x_i, x_j) = x_j$, giving $dot(X) = P X$) and Laplacian diffusion ($g(x_i, x_j) = x_j - x_i$).
Horizontality depends on $g$; the locality constraint (velocity depends only on neighbors weighted by $P$) is itself a useful structural assumption.

*Observable non-horizontal dynamics.*
Not all observable dynamics are horizontal.
Centroid circulation $dot(x)_i = (x_i - macron(x)) A$ with $A in so(d)$ decomposes as $dot(x)_i = x_i A - macron(x) A$: the first term is pure gauge (@thm:invisible), but the shared drift $-macron(x) A$ shifts dot products, making $dot(P) != 0$ generically.
However, $X^top dot(X) = C A$ where $C$ is the sample covariance of positions, and the product of a symmetric and skew-symmetric matrix is generically not symmetric, so the dynamics are not horizontal: the trajectory spirals through the fibers.
Similarly, differential rotation $dot(x)_i = x_i A_i$ with node-specific $A_i in so(d)$ gives $dot(P)_(i j) = x_i (A_i - A_j) x_j^top$, which is observable whenever rotation rates differ.

*Summary.*

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
  caption: [Families of RDPG dynamics. The polynomial family is always horizontal, has few parameters ($K + 1$ independent of $n$), and is gauge-invariant ($N(P)$ computable without alignment). Centroid circulation is observable but not horizontal.]
) <tab:dynamics-families>


== Observable dynamics and the Lyapunov equation <sec:p-dynamics>

@thm:invisible characterizes what is _lost_ to gauge freedom.
We now characterize what is _preserved_: the observable content of RDPG dynamics, projected to the base space $cal(B)$ of probability matrices.

The key observation relies on two assumptions jointly.
First, the RDPG model provides the factorization $P = X X^top$, so that $P$ inherits smoothness from $X$ and the product rule applies.
Second, considering for example the linear symmetric regime, the dynamical model $dot(X) = N X$ specifies how positions evolve.
Together, these give a closed equation for $dot(P)$:
$ dot(P) = dot(X) X^top + X dot(X)^top = N X X^top + X X^top N^top = N P + P N^top $ <eq:p-dynamics>
When $N$ is symmetric: $dot(P) = N P + P N$, a Lyapunov equation.

This is _not_ a general fact about temporal networks.
Without the RDPG factorization, $P(t)$ is merely a time-varying matrix of edge probabilities whose entries could evolve independently, follow a completely different matrix ODE, or fail to be differentiable.
@eq:p-dynamics is the *observable footprint* of RDPG dynamics: it is what the gauge-invariant content of $dot(X) = N X$ looks like when projected to $cal(B)$.

The Lyapunov operator $cal(L)_P: "Sym"(n) -> "Sym"(n)$ defined by $cal(L)_P (N) = N P + P N$ is a linear map on symmetric matrices.
To analyze it, we pass to the eigenbasis of $P$.
Let $P = U Lambda U^top$ be the eigendecomposition, and write $tilde(N) = U^top N U$ and $tilde(dot(P)) = U^top dot(P) U$ for the representations in this basis, with indices $iota, gamma in {1, ..., n}$ labeling eigenvector directions (not nodes).
In this basis, $cal(L)_P$ acts diagonally: $(cal(L)_P (N))_(iota gamma) = (lambda_iota + lambda_gamma) tilde(N)_(iota gamma)$.

For the RDPG setting, where $P = X X^top$ with $X in RR^(n times d)$, the matrix $P$ has rank $d$ with eigenvalues $lambda_1 >= ... >= lambda_d > 0$ and $lambda_(d+1) = ... = lambda_n = 0$.
The Lyapunov operator is therefore *not invertible* on all of $"Sym"(n)$: when both $iota, gamma > d$, the equation $(lambda_iota + lambda_gamma) tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma)$ reduces to $0 dot tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma)$.
The realizability constraint (@prop:tangent) forces $tilde(dot(P))_(iota gamma) = 0$ for this block, so the equation says nothing about $tilde(N)_(iota gamma)$.
For $n = 100$ and $d = 3$, this leaves $(n - d)(n - d + 1)\/2 = 4753$ out of $n(n+1)\/2 = 5050$ entries of $tilde(N)$ completely unconstrained by the data.

However, the resolution is clean: the undetermined part of $N$ is exactly the part that does not affect the dynamics.

#proposition(title: "Partial Lyapunov Identifiability")[
  Let $P = X X^top$ with $"rank"(P) = d < n$, and let $V in RR^(n times d)$ and $V_perp in RR^(n times (n-d))$ span the range and null space of $P$ respectively.
  In the eigenbasis of $P$ (indices $iota, gamma$), the Lyapunov equation $dot(P) = N P + P N$ uniquely determines the following blocks of $tilde(N) = U^top N U$:

  - *Range-range block* ($iota, gamma <= d$): $tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma) \/ (lambda_iota + lambda_gamma)$, giving $d(d+1)\/2$ entries.
  - *Cross block* ($iota <= d, gamma > d$ or vice versa): $tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma) \/ lambda_iota$, giving $d(n - d)$ entries.

  The *null-null block* ($iota, gamma > d$): $(n-d)(n-d+1)\/2$ entries remain unconstrained.

  The total number of determined entries is $n d - d(d-1)\/2$, matching the dimension of the realizable tangent space.
] <prop:lyapunov-invert>

#proof[
  In the eigenbasis, the Lyapunov equation gives $(lambda_iota + lambda_gamma) tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma)$ entry-wise.
  For $iota, gamma <= d$: $lambda_iota + lambda_gamma > 0$, so $tilde(N)_(iota gamma)$ is uniquely determined.
  For $iota <= d, gamma > d$: $lambda_iota + lambda_gamma = lambda_iota > 0$, so $tilde(N)_(iota gamma)$ is uniquely determined.
  For $iota, gamma > d$: $lambda_iota + lambda_gamma = 0$, and realizability forces $tilde(dot(P))_(iota gamma) = 0$, leaving $tilde(N)_(iota gamma)$ free.
  Counting: $d(d+1)\/2 + d(n-d) = n d - d(d-1)\/2$.
]

The undetermined null-null block governs how $N$ acts on vectors orthogonal to $"col"(X)$.
Since $X$ has no component in this subspace, $N X$ does not depend on this block: the dynamics $dot(X) = N X$ are insensitive to it.
The Lyapunov equation determines $N$ on exactly the subspace that matters for the dynamics, and leaves unconstrained exactly the subspace that is dynamically irrelevant.

In summary symmetric linear dynamics are identifiable from $(P, dot(P))$ up to the dynamically irrelevant null-null block, without ever choosing a gauge.
This extends to polynomial dynamics: $dot(X) = N(P) X$ with $N(P) = sum_k alpha_k P^k$ gives $dot(P) = N(P) P + P N(P)$, and the $alpha_k$ can be recovered by projecting $dot(P)$ onto the basis ${P^k P + P P^k}_(k=0)^K$ in the space of symmetric matrices.
The polynomial structure is particularly natural here, as $N(P) = sum_k alpha_k P^k$ automatically has its null-null block determined by the $alpha_k$ and the eigenvalues of $P$, eliminating the underdetermined component entirely.

However, two fundamental obstacles prevent direct application.

*We do not observe $P(t)$.*
In practice, we observe adjacency matrices $A^((t))$ sampled as $A_(i j)^((t)) tilde.op "Bernoulli"(P_(i j)(t))$.
Entry-wise, $A_(i j)^((t))$ is an unbiased estimate of $P_(i j)(t)$ for $i != j$, with variance $P_(i j)(1 - P_(i j))$.
When $m$ independent samples are available at each time $t$, averaging $macron(A)^((t)) = 1/m sum_(ell=1)^m A_ell^((t))$ reduces this per-entry variance by a factor of $m$, independently of the network size $n$.

However, the Lyapunov inversion in @prop:lyapunov-invert requires the _spectral decomposition_ of $P$, not merely its entries, and spectral accuracy depends on _both_ $n$ and $m$.
Spectral estimation introduces one or two complications.
First, if self-links are disregarded, $A$ has a hollow diagonal ($A_(i i) = 0$) while $P_(i i) = ||x_i||^2 > 0$, introducing a rank-$n$ perturbation of spectral norm at most 1, which is $O(1\/n)$ relative to $||P||_("op") tilde O(n)$ but comparable to the random fluctuation $||macron(A) - P||_("op") = O(sqrt(n\/m))$.
Second, eigendecomposition is nonlinear, so even without the diagonal issue, the eigenvalues and eigenvectors of $macron(A)$ are biased estimates of those of $P$.
The bias is asymptotically negligible: the leading-order perturbation $(macron(A) - P) X (X^top X)^(-1)$ has conditional mean zero (up to the diagonal discrepancy), and the residual bias is $o(1\/sqrt(n m))$ @athreya2016limit @cape2019two.
But the approximation in @prop:lyapunov-invert requires dividing by $lambda_iota + lambda_gamma$, and this division amplifies noise precisely where the spectral gap is small.
This is the same $1 / lambda_d$ sensitivity that appears in the geometric analysis of @sec:fiber-bundle, now manifesting as ill-conditioning of the Lyapunov inverse.

*The parameter count of $N$ on the relevant subspace.*
Even restricting to the dynamically relevant part of $N$, the number of free parameters can be large.
The range-range and cross blocks together have $n d - d(d-1)\/2$ entries.
When $N$ is constant, a single well-estimated $(P, dot(P))$ pair provides exactly this many scalar constraints, so the system is determined.
But when $N$ varies with time or state (for example in polynomial dynamics $N(P) = sum_k alpha_k P^k$, where the $alpha_k$ are constant but $N(P(t))$ changes because $P$ does) recovering the $alpha_k$ requires sufficient variation in $P(t)$ across time points to separate the contributions of different powers $P^k$.
The parsimonious polynomial family, with only $K + 1$ parameters independent of $n$, is attractive precisely because it sidesteps the parameter-counting issue.
For richer families, the estimation problem becomes underdetermined without strong structural assumptions or regularization.

#remark[
  The $P$-dynamics perspective and the fiber bundle perspective address different aspects of the same problem.
  @prop:lyapunov-invert says that the dynamics are identifiable _in principle_ from the gauge-invariant observable $P$, up to the dynamically irrelevant null-null block.
  The fiber bundle theory (@sec:fiber-bundle) addresses the harder question: what happens when we _must_ work with $X$ (e.g., because we need latent positions for interpretation or for the UDE pipeline), and how the geometry of the quotient space governs the difficulty of doing so.
  The two perspectives are complementary: the algebraic result tells us what information is available; the geometric framework tells us what it costs to extract it.
]


== Holonomy for horizontal dynamics families <sec:holonomy-dynamics>

The holonomy obstruction (@sec:fiber-bundle) raises a concrete question: for which horizontal dynamics families is holonomy trivial, and for which is it non-trivial?
This distinction has direct practical consequences: trivial holonomy means global gauge consistency is achievable in principle; non-trivial holonomy means that even perfect local alignment must accumulate global drift.

We show that the answer depends sharply on whether the dynamics generator commutes with itself along the trajectory: a condition satisfied by polynomial dynamics but violated by Laplacian and message-passing dynamics (cf. @sec:dynamics-families).

*Curvature from non-commuting generators.*
Consider horizontal dynamics $dot(X) = M(X) X$ with $M(X)$ symmetric, so that $X^top dot(X) = X^top M(X) X$ is symmetric and the dynamics are horizontal.
At two distinct times $t_1, t_2$, the generators $M_1 = M(X(t_1))$ and $M_2 = M(X(t_2))$ are both symmetric.
The commutator $[M_1, M_2] = M_1 M_2 - M_2 M_1$ is skew-symmetric (since transposing reverses the order).

The Lie bracket of the corresponding horizontal vector fields $overline(xi)_i (X) = M_i X$ is:
$ [overline(xi)_1, overline(xi)_2](X) = (M_2 M_1 - M_1 M_2) X = -[M_1, M_2] X $

Since $[M_1, M_2]$ is skew-symmetric, we have $X^top [M_1, M_2] X in so(d)$, which means this bracket is *vertical*: it lies in $cal(V)_X = {X Omega : Omega in so(d)}$.
Remembering @prop:curv-spectral-gap, by O'Neill's formula @oneill1966fundamental we get that the sectional curvature of the 2-plane spanned by the projections of $overline(xi)_1, overline(xi)_2$ in $cal(B)$ is: $K(xi_1, xi_2) = 3 ||cal(A)_(overline(xi)_1) overline(xi)_2||^2$ where $cal(A)_(overline(xi)_1) overline(xi)_2 = 1/2 [overline(xi)_1, overline(xi)_2]^cal(V)$ is the $A$-tensor of the submersion.

Since $RR_*^(n times d)$ is flat (it is an open subset of Euclidean space), the base curvature arises *entirely* from the vertical bracket above.

#proposition(title: "Curvature Criterion for Horizontal Dynamics")[
  For horizontal dynamics $dot(X) = M(X) X$ with $M$ symmetric, the sectional curvature along the trajectory in $cal(B)$ vanishes if and only if the generators at different times commute: $[M(X(t_1)), M(X(t_2))] = 0$ for all $t_1, t_2$ along the trajectory.
  When the generators fail to commute, the curvature is strictly positive and proportional to $||X^top [M_1, M_2] X||^2$.
] <prop:curvature-criterion>

*Polynomial dynamics: trivial holonomy.*
For polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum_(k=0)^K alpha_k P^k$, the induced $P$-dynamics are:
$ dot(P) = N(P) P + P N(P) = 2 sum_(k=0)^K alpha_k P^(k+1) $
This is a polynomial in $P$ alone. Writing $P = U Lambda U^top$ in its eigendecomposition, every power $P^k = U Lambda^k U^top$ shares the same eigenvectors $U$.
Therefore $dot(P) = U (2 sum alpha_k Lambda^(k+1)) U^top$ also has eigenvectors $U$: the eigenvectors of $P$ are stationary under polynomial dynamics.

The trajectory $P(t)$ lies on a $d$-dimensional submanifold of $cal(B)$ parameterized by the eigenvalues $(lambda_1(t), ..., lambda_d(t))$ alone, with each eigenvalue evolving independently:
$ dot(lambda)_iota = 2 sum_(k=0)^K alpha_k lambda_iota^(k+1) $

Two consequences follow immediately.

#proposition(title: "Polynomial Dynamics: Trivial Holonomy")[
  Polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum alpha_k P^k$ have trivial holonomy. Specifically:
  + The generators $N(P(t_1))$ and $N(P(t_2))$ commute at all times (they share eigenvectors), so the curvature along the trajectory vanishes by @prop:curvature-criterion.
  + The eigenvalue ODEs are autonomous and one-dimensional, hence admit no periodic orbits. Closed loops in $cal(B)$ are therefore impossible under polynomial dynamics.
] <prop:poly-trivial-holonomy>

#proof[
  For (1): since the eigenvectors of $P$ are stationary, $N(P(t)) = U (sum alpha_k Lambda(t)^k) U^top$ for all $t$, with the same $U$. Any two diagonal matrices commute, so $[N(P(t_1)), N(P(t_2))] = U [sum alpha_k Lambda(t_1)^k, sum alpha_k Lambda(t_2)^k] U^top = 0$.

  For (2): each $lambda_iota (t)$ satisfies the autonomous ODE $dot(lambda)_iota = f(lambda_iota)$ with $f(x) = 2 sum alpha_k x^(k+1)$. By the uniqueness theorem for ODEs, if $lambda_iota (t_0) = lambda_iota (t_1)$ for $t_0 != t_1$, then $lambda_iota$ is constant. Non-constant orbits are monotone, so no closed loop in eigenvalue space, and hence in $cal(B)$,is possible.
]

*What polynomial dynamics look like for nodes.*
The statement that eigenvectors of $P$ are stationary deserves both rigorous justification and concrete interpretation.

_Why eigenvectors are fixed._
Write $P(t) = U(t) Lambda(t) U(t)^top$ and let $Omega = U^top dot(U) in so(d)$ be the eigenvector rotation rate.
Differentiating $P = U Lambda U^top$ and projecting into the eigenbasis gives $(U^top dot(P) U)_(iota gamma) = Omega_(iota gamma)(lambda_gamma - lambda_iota)$ for $iota != gamma$.
Since $dot(P) = 2 sum alpha_k P^(k+1)$ is diagonal in the eigenbasis, the left side vanishes, so $Omega_(iota gamma) (lambda_gamma - lambda_iota) = 0$ for all $iota != gamma$.
It remains to show that eigenvalues remain distinct.
All $d$ eigenvalues satisfy the same autonomous ODE $dot(lambda) = f(lambda)$ with $f(x) = 2 sum alpha_k x^(k+1)$, differing only in initial conditions.
If $lambda_iota (0) != lambda_gamma (0)$, then by uniqueness of ODE solutions $lambda_iota (t) != lambda_gamma (t)$ for all $t$: if they ever coincided, they would be the same solution, contradicting their distinct initial conditions.
Therefore $lambda_gamma - lambda_iota != 0$, which forces $Omega = 0$: the eigenvectors of $P$ are genuinely stationary, not merely stationary to first order.

_What this means for nodes._
In the canonical gauge $X = U Lambda^(1\/2)$, node $i$ has latent position $x_i (t) = (U_(i 1) sqrt(lambda_1(t)), ..., U_(i d) sqrt(lambda_d(t)))$.
The loading $U_(i iota)$ (which can be read as the "membership weight" of node $i$ in eigenvector direction $iota$) is constant; what evolves is the scale $sqrt(lambda_iota (t))$ of each direction.

Since the eigenvalue ODE $dot(lambda) = f(lambda)$ is nonlinear for $K >= 1$, eigenvalues with different magnitudes evolve at different rates, and the ratios $lambda_iota (t) \/ lambda_gamma (t)$ change over time.
The effect on the node positions is an *anisotropic rescaling* of the latent space along fixed axes: in $d = 2$, the point cloud deforms as if inscribed in an ellipse whose axes do not rotate but whose eccentricity changes.
Node directions in $RR^d$ do change (they are not merely rescaled in magnitude), but the change is tightly constrained: all deformation is captured by the $d$ scalar functions $lambda_iota (t)$.

The exception is linear dynamics ($K = 0$, $f(lambda) = 2 alpha_0 lambda$), where $lambda_iota (t) = lambda_iota (0) e^(2 alpha_0 t)$ and all eigenvalues grow or decay at the same exponential rate.
In this case the ratios are constant, the point cloud is rescaled isotropically, and node angular positions in $RR^d$ are truly fixed.
For quadratic or higher dynamics ($K >= 1$), the nonlinearity of $f$ causes larger eigenvalues to evolve differently from smaller ones, producing genuine reshaping of the point cloud's geometry while preserving its combinatorial community structure.

*Laplacian dynamics: non-trivial holonomy.*
Graph Laplacian dynamics $dot(X) = -L X$ with $L = D - P$, where $D = "diag"(P bold(1))$, provide a natural example of horizontal dynamics with non-trivial holonomy.
The generator $-L = P - D$ is symmetric, so the dynamics are horizontal.
But $L$ is *not* a polynomial in $P$: the degree matrix $D$ depends on the row sums of $P$, which mix eigenvector information that a polynomial in $P$ cannot access.

The $P$-dynamics under Laplacian flow are:
$ dot(P) = -L P - P L = -(D - P) P - P (D - P) = 2 P^2 - D P - P D $
The term $2P^2$ preserves eigenvectors of $P$ (it is a polynomial in $P$), but $D P + P D$ does not: in the eigenbasis $P = U Lambda U^top$, the degree matrix $D = "diag"(U Lambda U^top bold(1))$ is diagonal in the *node* basis, not the eigenbasis.
Thus $U^top D U$ is generally a full matrix, and $dot(P)$ has eigenvector components transverse to those of $P$.

*Consequence:* the eigenvectors of $P$ rotate under Laplacian dynamics, the trajectory sweeps out area in $cal(B)$ transverse to the eigenvalue-parameterized submanifold, and @prop:curvature-criterion guarantees positive curvature along any portion of the trajectory where $[L(P(t_1)), L(P(t_2))] != 0$.

*What Laplacian dynamics look like for nodes.*
The contrast with polynomial dynamics is sharp at the node level.
Under Laplacian diffusion, each node's velocity $dot(x)_i = -sum_j L_(i j) x_j$ is a weighted average of the positions of its neighbors (with weights from $P$), minus a restoring term from the node's own degree.
This acts like heat diffusion on the latent positions: high-degree nodes experience stronger mean-reversion, while the diffusion couples all positions through the network structure.
Because the coupling depends on the _row sums_ of $P$, mixing contributions from all eigenvector directions, the eigenvectors of $P$ rotate over time.
In the ellipse analogy for $d = 2$: not only does the eccentricity change, but the axes themselves rotate.
The community structure is no longer static; nodes' membership weights across latent dimensions genuinely evolve, creating the richer dynamics that produce non-trivial holonomy.

#proposition(title: "Laplacian Dynamics: Non-Trivial Holonomy")[
  For $d >= 2$ and generic initial conditions, Laplacian dynamics $dot(X) = -L(P) X$ produce trajectories in $cal(B)$ with strictly positive sectional curvature.
  If the trajectory visits states $P(t_1), P(t_2)$ such that $L(P(t_1))$ and $L(P(t_2))$ do not commute, the holonomy around any loop enclosing the corresponding portion of the trajectory is non-trivial.
] <prop:laplacian-holonomy>

#proof[
  It suffices to show that $L(P(t_1))$ and $L(P(t_2))$ generically fail to commute.
  Write $L_i = D_i - P_i$ for $i = 1, 2$.
  Then $[L_1, L_2] = [D_1, P_2] - [D_2, P_1] + [P_1, P_2] - [D_1, D_2]$.
  Since $D_1, D_2$ are both diagonal, $[D_1, D_2] = 0$. And $[P_1, P_2] = 0$ when $P_1, P_2$ share eigenvectors, but under Laplacian dynamics they do not, so generically $[P_1, P_2] != 0$.
  Moreover, $[D_1, P_2]$ is generically nonzero since the row sums encoded in $D_1$ depend on the full matrix $P_1$, not just its eigenvalues.
  By @prop:curvature-criterion, the curvature along the trajectory is positive, and by the Ambrose-Singer theorem, the holonomy group is generated by the curvature 2-form evaluated at such pairs.
]

For $d = 2$, the result is particularly sharp.

#corollary(title: "Full Holonomy for Laplacian Dynamics in Dimension 2")[
  For $d = 2$ and generic initial conditions, the restricted holonomy group of Laplacian dynamics is $"SO"(2)$, the largest possible connected subgroup of $O(2)$.
  Concretely: for any target rotation angle $phi in [0, 2 pi)$, there exists a loop in $cal(B)$ traversable by Laplacian dynamics whose holonomy is rotation by $phi$.
]

#proof[
  The Lie algebra $frak(s o)(2)$ is one-dimensional, spanned by the single generator $J = mat(0, -1; 1, 0)$.
  By @prop:laplacian-holonomy, the curvature along a generic Laplacian trajectory is strictly positive, so the curvature 2-form produces a nonzero element of $frak(s o)(2)$.
  A single nonzero element spans the one-dimensional Lie algebra, so by the Ambrose-Singer theorem, the holonomy algebra is all of $frak(s o)(2)$, and the restricted holonomy group (holonomy of contractible loops) is $"SO"(2)$.
]

This means that for rank-2 latent spaces under Laplacian dynamics, the gauge ambiguity is not merely a discrete sign flip (as for polynomial dynamics) but a continuous rotation by an arbitrary angle---the worst possible case for alignment.

*Quantitative estimate.* For $d = 2$, the holonomy around a loop $gamma$ in $cal(B)$ is a rotation by angle:
$ phi(gamma) = integral.double_Sigma K thin d A $
where $Sigma$ is a surface bounded by $gamma$, $K$ is the sectional curvature, and $d A$ is the area element.
By the curvature formula of @massart2019curvature, regions where $lambda_2$ is small contribute large curvature $K tilde 1\/lambda_2$, so even short Laplacian trajectories that pass near rank-deficient states can accumulate substantial holonomy.

*Higher dimensions ($d >= 2$).*
For general $d$, the Lie algebra $frak(s o)(d)$ has dimension $d(d-1)\/2$.
The curvature 2-form at each point produces elements $X^top [M_1, M_2] X in frak(s o)(d)$, and as the trajectory visits different states, these elements span an increasing subspace.
For generic Laplacian dynamics with $n >> d$, the degree matrices at different times are sufficiently varied that we expect the curvature elements to span all of $frak(s o)(d)$, yielding holonomy group $"SO"(d)$.
We state this as a conjecture.

#conjecture(title: "Full Holonomy in Higher Dimensions")[
  For $d >= 2$, $n > d$, and generic initial conditions, the restricted holonomy group of Laplacian dynamics $dot(X) = -L(P) X$ is $"SO"(d)$.
]

A proof would require showing that the commutators $X^top [L(P(t_1)), L(P(t_2))] X$, as $t_1, t_2$ vary along the trajectory, span $frak(s o)(d)$.
For $d = 2$ this is automatic (one nonzero element spans a one-dimensional Lie algebra); for $d >= 3$ it requires a transversality argument exploiting the fact that the degree matrix $D = "diag"(P bold(1))$ mixes all $d$ eigenvector components.

The same argument applies to message-passing dynamics $dot(x)_i = sum_j P_(i j) g(x_i, x_j)$ with symmetric generators that are not polynomials in $P$.

*Summary:* the holonomy obstruction creates a sharp dichotomy among horizontal dynamics families.
Polynomial dynamics, which operate on the spectral structure of $P$ through commuting generators, preserve eigenvectors and have trivial holonomy: global gauge consistency is achievable without topological obstruction.
Laplacian and message-passing dynamics, which mix spectral and spatial structure through non-commuting generators, rotate eigenvectors and produce non-trivial holonomy: gauge drift accumulates even along horizontal trajectories, and no local alignment procedure can achieve global consistency over cycles.

This distinction has practical implications: for polynomial dynamics, the constructive alignment problem (@sec:constructive) is purely a statistical challenge (overcoming Bernoulli noise); for Laplacian dynamics, it is simultaneously a statistical _and_ topological challenge.


== Information-theoretic limits <sec:info-theoretic>

The geometric obstructions of @sec:obstructions (gauge freedom, curvature, holonomy) and the dynamics analysis above constrain what is learnable in principle.
We now complement these with *statistical* obstructions: given $T$ noisy adjacency matrices, how accurately can we estimate the parameters governing the dynamics, regardless of the estimation method?

The answer reveals a duality: the same spectral gap $lambda_d$ that controls geometric difficulty also controls statistical difficulty, so the two obstructions reinforce each other.

=== Fisher information for dynamics parameters

Consider a parametric dynamics family with parameter $theta in RR^k$ generating a trajectory $P(t; theta)$.
We observe $T$ adjacency matrices $A^((t)) tilde "Bernoulli"(P(t; theta))$ at times $t_1, ..., t_T$, with all entries conditionally independent given $P(t; theta)$.
The log-likelihood factorizes:
$ ell(theta) = sum_(t=1)^T sum_(i < j) [A_(i j)^((t)) log P_(i j)(t; theta) + (1 - A_(i j)^((t))) log(1 - P_(i j)(t; theta))] $

The Fisher information matrix $cal(I)(theta) in RR^(k times k)$ has entries:
$ cal(I)(theta)_(a b) = sum_(t=1)^T sum_(i < j) (partial P_(i j))/(partial theta_a) (partial P_(i j))/(partial theta_b) dot.c 1/(P_(i j)(1 - P_(i j))) $ <eq:fisher>
where $P_(i j) = P_(i j)(t; theta)$ and the derivatives $partial P_(i j) \/ partial theta_a$ capture how perturbations in the dynamics parameters propagate to the observations.

=== Sensitivity propagation through the Lyapunov equation

The key quantity in @eq:fisher is the sensitivity $partial P_(i j) \/ partial theta_a$.
For polynomial dynamics $N(theta) = sum_(k=0)^K theta_k P^k$, the $P$-dynamics are:
$ dot(P) = 2 sum_(k=0)^K theta_k P^(k+1) $
A crucial simplification comes from @prop:poly-trivial-holonomy: the eigenvectors of $P$ are stationary under polynomial dynamics.
Writing $P = U Lambda U^top$ with $Lambda = "diag"(lambda_1, ..., lambda_d)$ and $U in RR^(n times d)$ fixed, the sensitivity $partial P \/ partial theta_a = U (partial Lambda \/ partial theta_a) U^top$ is diagonal in the eigenbasis.

Each eigenvalue satisfies $dot(lambda)_iota = 2 sum_(k=0)^K theta_k lambda_iota^(k+1)$, so:
$ partial/(partial t) (partial lambda_iota)/(partial theta_a) = 2 lambda_iota^(a+1) + (partial dot(lambda)_iota)/(partial lambda_iota) dot (partial lambda_iota)/(partial theta_a) $
where $partial dot(lambda)_iota \/ partial lambda_iota = 2 sum_(k=0)^K theta_k (k+1) lambda_iota^k$ is the linearized eigenvalue dynamics.
This is a scalar linear ODE for each pair $(iota, a)$, with forcing $2 lambda_iota^(a+1)$ and feedback through the linearized dynamics.

The resulting sensitivity of the observed probabilities is:
$ (partial P_(i j))/(partial theta_a) = sum_(iota=1)^d U_(i iota) U_(j iota) (partial lambda_iota)/(partial theta_a) $ <eq:poly-sensitivity>
Each node pair $(i,j)$ receives signal from all $d$ eigenvalue sensitivities, weighted by the eigenvector loadings $U_(i iota) U_(j iota)$.

*Example: linear dynamics ($K = 0$).*
For $dot(X) = alpha_0 X$ (single parameter), the eigenvalue dynamics are $dot(lambda)_iota = 2 alpha_0 lambda_iota$, giving $lambda_iota (t) = lambda_iota (0) e^(2 alpha_0 t)$.
The sensitivity is $partial lambda_iota \/ partial alpha_0 = 2 t lambda_iota (t)$, and therefore:
$ (partial P_(i j))/(partial alpha_0) = 2 t sum_iota U_(i iota) lambda_iota (t) U_(j iota) = 2 t P_(i j)(t) $
The Fisher information at a single snapshot $t$ is:
$ cal(I)_t (alpha_0) = 4 t^2 sum_(i < j) P_(i j)(t)^2 / (P_(i j)(t)(1 - P_(i j)(t))) = 4 t^2 sum_(i < j) P_(i j)(t) / (1 - P_(i j)(t)) $
Summing over $T$ equally-spaced snapshots gives $cal(I)(alpha_0) = 4 sum_(t=1)^T t^2 sum_(i<j) P_(i j)(t) \/ (1 - P_(i j)(t))$, which scales as $O(T^3 dot n^2)$ when the edge probabilities are bounded away from 0 and 1.

*Higher-degree terms and the spectral gap.*
For $a >= 1$, the forcing $2 lambda_iota^(a+1)$ is small for eigenvalues near the spectral gap: $2 delta^(a+1)$ versus $2 lambda_1^(a+1)$.
This means the sensitivity $partial lambda_d \/ partial theta_a$ is small relative to $partial lambda_1 \/ partial theta_a$, so the Fisher information for higher-degree parameters $theta_a$ receives proportionally less signal from the small-eigenvalue directions.
Concretely, the contribution of the $lambda_d$ direction to $cal(I)(theta)_(a b)$ scales as $delta^(a+b+2)$, while the $lambda_1$ direction contributes $lambda_1^(a+b+2)$.

=== General dynamics and the Lyapunov amplification

The polynomial case is clean because eigenvectors are fixed, confining all sensitivity to the diagonal of the eigenbasis.
For general symmetric dynamics $dot(X) = M(P) X$ with $M$ symmetric but not a polynomial in $P$, eigenvectors rotate and the sensitivity $partial P \/ partial theta$ has both diagonal and off-diagonal components in the eigenbasis.

In this setting, the Lyapunov equation $dot(P) = M P + P M$ must be inverted to recover $M$ from $dot(P)$.
In the eigenbasis of $P$, the inversion acts componentwise: $M_(iota gamma) = dot(P)_(iota gamma) \/ (lambda_iota + lambda_gamma)$.
The factor $1 \/ (lambda_iota + lambda_gamma)$ amplifies noise in the off-diagonal components, and is the same factor that appears in the connection 1-form (@sec:fiber-bundle) and governs curvature.
This yields a direct correspondence: directions in parameter space that are hard geometrically (small $lambda_iota + lambda_gamma$ produces large curvature and large connection coefficients) are also hard statistically (small $lambda_iota + lambda_gamma$ amplifies estimation noise).

For Laplacian dynamics, the situation is compounded by holonomy.
The eigenvector rotation contributes additional degrees of freedom to the sensitivity, but this information is entangled with gauge degrees of freedom that the holonomy obstruction (@prop:laplacian-holonomy) prevents from being resolved locally.

=== The statistical-geometric duality

Combining the geometric and statistical pictures yields a unified obstruction.
We first state the result for polynomial dynamics, then explain the broader duality.

#proposition(title: "Cramér-Rao Bound for Polynomial Dynamics")[
  For polynomial dynamics of degree $K$ with parameter $theta = (theta_0, ..., theta_K) in RR^(k)$ ($k = K + 1$) on an RDPG with $n$ nodes, $d$ latent dimensions, and spectral gap $delta = lambda_d$, any unbiased estimator $hat(theta)$ satisfies:
  $ EE[||hat(theta) - theta||^2] >= "tr"(cal(I)(theta)^(-1)) $
  where the Fisher information matrix has the structure:
  $ cal(I)(theta)_(a b) = sum_(t=1)^T sum_(iota=1)^d (partial lambda_iota)/(partial theta_a) (partial lambda_iota)/(partial theta_b) dot.c cal(W)_iota (t) + R_(a b) $
  with $cal(W)_iota (t) = sum_(i < j) U_(i iota)^2 U_(j iota)^2 \/ [P_(i j)(t)(1 - P_(i j)(t))]$ measuring the information content of eigenvalue direction $iota$ at time $t$, and $R_(a b)$ collecting the cross-eigenvalue terms that are subdominant for $n >> d$.
] <prop:cramer-rao>

#proof[
  By @eq:poly-sensitivity, the sensitivity of $P_(i j)$ decomposes as $partial P_(i j) \/ partial theta_a = sum_iota U_(i iota) U_(j iota) (partial lambda_iota \/ partial theta_a)$.
  Substituting into @eq:fisher:
  $ cal(I)(theta)_(a b) = sum_t sum_(i < j) [sum_iota U_(i iota) U_(j iota) (partial lambda_iota)/(partial theta_a)] [sum_gamma U_(i gamma) U_(j gamma) (partial lambda_gamma)/(partial theta_b)] dot 1/(P_(i j)(1 - P_(i j))) $
  Expanding the product over $iota, gamma$ and separating the diagonal ($iota = gamma$) and cross ($iota != gamma$) contributions yields the stated structure plus cross terms of the form $(partial lambda_iota \/ partial theta_a)(partial lambda_gamma \/ partial theta_b) sum_(i<j) U_(i iota) U_(j iota) U_(i gamma) U_(j gamma) \/ [P_(i j)(1-P_(i j))]$.
  These cross terms are generically nonzero, but subdominant: the eigenvectors of a rank-$d$ matrix in $RR^(n times n)$ satisfy $sum_i U_(i iota) U_(i gamma) = delta_(iota gamma)$, and this approximate orthogonality suppresses the cross-sums relative to the diagonal sums when $n >> d$.
  The stated formula is exact when cross terms are retained; the diagonal approximation clarifies the structure.

  For the spectral gap dependence: the eigenvalue sensitivity $partial lambda_iota \/ partial theta_a$ is determined by the scalar ODE $dot(lambda)_iota = 2 sum_k theta_k lambda_iota^(k+1)$.
  The forcing term $2 lambda_iota^(a+1)$ scales as $delta^(a+1)$ for $iota = d$, so the $lambda_d$ direction contributes $O(delta^(a+b+2))$ to $cal(I)_(a b)$.
  When $delta -> 0$, the Fisher information matrix becomes ill-conditioned: the row and column corresponding to the highest-degree parameter $theta_K$ scale as $delta^(2K+2)$, making $"tr"(cal(I)^(-1))$ diverge.
]

The bound reveals three scaling regimes.
Each snapshot provides $O(n^2)$ conditionally independent Bernoulli observations, so the Fisher information grows as $n^2$.
Information accumulates linearly in $T$, assuming the dynamics produce sufficient variation in $P$ across time.
The spectral gap $delta$ controls estimation difficulty through the eigenvalue sensitivities: the parameter $theta_a$ receives signal proportional to $delta^(a+1)$ from the smallest eigenvalue direction.
For the constant term $theta_0$ (linear dynamics), the $delta$-direction contributes $O(delta^2)$ to the Fisher information; for the highest-degree term $theta_K$, the contribution degrades to $O(delta^(2K+2))$.

#corollary(title: "Linear Dynamics: Explicit Fisher Information")[
  For linear dynamics $dot(X) = alpha_0 X$ with $T$ snapshots at times $t_1, ..., t_T$, the Fisher information is:
  $ cal(I)(alpha_0) = 4 sum_(t=1)^T t^2 sum_(i < j) P_(i j)(t) / (1 - P_(i j)(t)) $
  This scales as $Theta(T^3 dot n^2)$ when edge probabilities are bounded away from 0 and 1, giving a Cramér-Rao lower bound of $Omega(1\/(T^3 n^2))$.
]

The cubic dependence on $T$ (rather than linear) reflects the fact that the sensitivity $partial P_(i j) \/ partial alpha_0 = 2 t P_(i j)$ grows with time: later snapshots are more informative because the perturbation has had longer to propagate.
This is a general feature of dynamics estimation, in contrast to i.i.d. settings where information accumulates linearly.

*The duality.*
The statistical-geometric correspondence is sharpest for general (non-polynomial) symmetric dynamics, where the full Lyapunov structure appears.
The connection 1-form involves factors $1 \/ (lambda_iota + lambda_gamma)$ controlling gauge sensitivity (@sec:fiber-bundle); the sectional curvature is proportional to $||X^top [M_1, M_2] X||^2$ which diverges as $lambda_d -> 0$ (@prop:curvature-criterion); and the Fisher information for estimating $M_(iota gamma)$ from the Lyapunov equation $dot(P)_(iota gamma) = (lambda_iota + lambda_gamma) M_(iota gamma)$ involves the same factor $(lambda_iota + lambda_gamma)^2$ that appears in the curvature denominator.
This means there is no regime in which the geometric and statistical problems decouple: the same spectral gap that inflates curvature and holonomy also degrades Fisher information.
Networks near rank-deficiency ($delta -> 0$) are simultaneously harder to align (curvature $tilde 1\/delta$), harder to interpolate (injectivity radius $tilde sqrt(delta)$), and harder to estimate statistically (Fisher information for the weakest direction $tilde delta^2$).

#remark[
  For non-polynomial dynamics (e.g., Laplacian), the sensitivity propagation is more complex because the eigenvectors of $P$ also evolve.
  The additional eigenvector sensitivity contributes positively to the Fisher information (there are more directions in which observations carry signal), but also introduces the holonomy obstruction: the extra information is entangled with gauge degrees of freedom that @prop:laplacian-holonomy shows cannot be resolved locally.
  A complete minimax theory for dynamics estimation in the presence of holonomy remains an important open problem.
]

#remark[
  *Relationship to ASE estimation theory.*
  @prop:cramer-rao is derived from the Bernoulli likelihood of the observed adjacency matrices and applies to _any_ estimator of $theta$, not to spectral methods specifically.
  The bound is therefore conceptually distinct from the CLT for adjacency spectral embedding @athreya2016limit @athreya2017statistical, the two-to-infinity perturbation theory @cape2019two, and the efficiency results for spectral estimators of static network parameters @tang2022efficient @xie2021efficient.
  Those results characterize the accuracy of estimating _latent positions_ $X$ or _block probability matrices_ $B$ from a _single_ graph (or a fixed collection of graphs sharing the same $P$).
  ASE is asymptotically unbiased with per-vertex error $O(1\/sqrt(n))$ @athreya2016limit, and the spectral estimator of SBM block probabilities achieves asymptotic efficiency @tang2022efficient; for individual latent positions, a one-step Newton correction is needed @xie2021efficient.
  The minimax rate for latent position estimation under Frobenius loss is $Theta(d\/n)$ per vertex @xie2020optimal, with two-to-infinity rates depending on the spectral gap @agterberg2023minimax.

  By contrast, @prop:cramer-rao concerns estimation of the _dynamics parameters_ $theta$ from a _time series_ of graphs, a setting for which no prior efficiency theory exists.
  The estimation target is qualitatively different: not the $O(n d)$ latent position coordinates, but the $O(1)$ parameters governing their temporal evolution.
  The eigenvalue-direction decomposition of the Fisher information in @prop:cramer-rao, with weights $cal(W)_iota (t)$ involving eigenvector loadings, has no counterpart in the static theory.
  Extending the existing minimax framework to parametric temporal models is an open problem that our Fisher information structure could help resolve.

  The "any unbiased estimator" qualification deserves comment.
  The CRB is a bound on the Bernoulli likelihood, which is well-defined for any $n$, so the bound itself does not require asymptotic arguments.
  However, whether useful estimators of $theta$ are unbiased at finite $n$ is a separate question.
  Any estimator that first recovers $P$ spectrally and then fits $theta$ inherits the finite-sample bias of eigendecomposition: the hollow diagonal of $A$ (since $A_(i i) = 0$ but $P_(i i) = ||x_i||^2 > 0$) and the nonlinearity of the spectral map both contribute, but the resulting bias is $o(1\/sqrt(n))$ @cape2019two and does not affect the asymptotic bound.
  For finite-sample inference, the biased Cramér-Rao variant $"Var"(hat(theta)) >= (1 + b'(theta))^2 \/ cal(I)(theta)$ provides the appropriate generalization.
]


= Recovering trajectories from spectral embeddings <sec:trajectory-problem>

The obstructions above concern what dynamics are *theoretically* possible.
We now turn to the *practical* problem: even when dynamics are observable and realizable, recovering trajectories from data is hard because spectral embedding introduces arbitrary gauge transformations at each time step.

== ASE introduces random gauge transformations

At each time $t$, ASE computes the eigendecomposition of $A^((t))$.
The eigenvectors are determined only up to sign (and rotation within repeated eigenspaces).
This means:
$ hat(X)^((t)) = X^((t)) R^((t)) + E^((t)) $
where $R^((t)) in O(d)$ is determined by numerical details of the eigenvalue solver, not by the dynamics, and hence is essentially *random*. Thus, $E^((t))$ is statistical noise.

*The key point*: Even if the true positions $X^((t))$ evolve smoothly, the estimates $hat(X)^((t))$ jump erratically because the $R^((t))$ are unrelated across time.

== Finite differences fail

Consider estimating velocity via finite differences:
$ hat(dot(X))^((t)) = (hat(X)^((t + delta t)) - hat(X)^((t))) / (delta t) $

Substituting the gauge-contaminated estimates:
$ hat(dot(X))^((t)) = (X^((t+delta t)) R^((t+delta t)) - X^((t)) R^((t))) / (delta t) + O(E) $

Even ignoring noise $E$, this is dominated by the gauge jump $R^((t+delta t)) - R^((t))$, which is $O(1)$ regardless of $delta t$.
As the "velocity" diverges as $delta t -> 0$, we're measuring gauge jumps, not dynamics.

== Pairwise Procrustes is insufficient

One might try aligning consecutive embeddings via Procrustes:
$ Q^((t)) = arg min_(Q in O(d)) ||hat(X)^((t+1)) - hat(X)^((t)) Q||_F $

This finds the best rotation to match adjacent frames.
However:
- The solution is *local*: it doesn't ensure *global* consistency across the full trajectory
- Errors accumulate: small misalignments at each step compound
- There's no guarantee the aligned trajectory corresponds to *any* consistent dynamics

== Pairwise alignment and error accumulation <sec:alignment-accumulation>

One might try aligning consecutive embeddings via Procrustes:
$ Q^((t)) = arg min_(Q in O(d)) ||hat(X)^((t+1)) - hat(X)^((t)) Q||_F $

This has a closed-form solution via SVD and finds the best rotation to match adjacent frames.
However, sequential pairwise alignment suffers from fundamental limitations:

+ *Local, not global:* Each alignment minimizes error between adjacent frames but doesn't ensure consistency across the full trajectory.

+ *Error accumulation:* Small misalignments at each step compound. After $T$ steps, the accumulated rotation error can be $O(sqrt(T) sigma)$ where $sigma$ is the per-step noise level.

+ *No dynamical constraint:* There's no guarantee the aligned trajectory corresponds to *any* consistent dynamics because the alignment is purely geometric.

*Noise in spectral embeddings:*
RDPG spectral embeddings have estimation error $||hat(X) - X Q||_F = O_p(sqrt(n))$ for some $Q in O(d)$, giving per-node error $O_p(1\/sqrt(n))$ @athreya2017statistical @cape2019two.
For pairwise Procrustes between consecutive frames, this translates to rotation estimation error that depends on both $n$ and the separation between frames.

When the true trajectory moves slowly (small $||X^((t+1)) - X^((t))||$), the signal-to-noise ratio for alignment degrades. In these cases, we're trying to detect small true rotations against a background of estimation noise.

== Why existing joint embedding methods don't help <sec:why-not-uase>

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

== Why Bayesian smoothing approaches are insufficient <sec:bayesian-smoothing>

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
    [Hierarchical GP prior], [Yes ($C^k$)], [No: it interpolates only],
    [SDE $d X = f(X) d t + sigma d W$], [No (Hölder $< 1\/2$)], [Yes, as $sigma -> 0$],
    [Neural ODE], [Yes ($C^k$)], [Yes (by construction)],
    [GP-ODE (NPODE) @heinonen2018learning], [Yes ($C^1$)], [Yes: it learns $f$ as GP],
    [Structure-constrained (ours)], [Yes], [Yes: the family $cal(F)$ is enforced],
    table.hline()
  ),
  caption: [Comparison of approaches: smoothness vs dynamical consistency.]
)

The SDE formulation $d X = f(X) d t + sigma d W$ provides a principled bridge: Freidlin-Wentzell theory @freidlin1998random shows that as $sigma -> 0$, solutions concentrate around ODE solutions with rate function $J_T(phi) = 1/2 integral_0^T ||dot(phi)(t) - f(phi(t))||^2 d t$.
This rate function is precisely the *dynamical consistency penalty*, which measures how far a path deviates from being an ODE solution.

== Honest assessment

We must be candid: *aligning spectral embeddings to recover continuous-time trajectories is a hard open problem*.
The methods described above address related but different problems.
There is no existing method that provably recovers trajectories from ODE dynamics on RDPG latent positions.

Error accumulation in sequential alignment (@sec:alignment-accumulation) compounds over long trajectories.
Dynamical consistency considerations (@sec:bayesian-smoothing) distinguish interpolation from true dynamics learning.
Holonomy (@sec:fiber-bundle) implies that even perfect local alignment may accumulate global gauge drift.

This motivates investigating whether *structure of the dynamics themselves* can constrain the alignment problem. We explore this idea in the next section, where we find that the theoretical principle is sound but practical realization faces fundamental difficulties.


= The Constructive Problem: Identifiability and Its Limits <sec:constructive>

The obstructions in the previous sections characterize what is and isn't observable in principle.
We now turn to the constructive question: given that observable dynamics exist, can we _recover_ them from spectral embeddings?
We establish a theoretical identifiability result showing that dynamics structure can, in principle, resolve gauge ambiguity.
We then discuss why this identifiability does not straightforwardly translate into a practical algorithm.

== Structure-constrained alignment

The alignment problem is underdetermined: without additional information, many gauge choices produce plausible trajectories.
A natural idea is to use *inductive bias about dynamics structure* to constrain which trajectories are admissible.

Suppose dynamics belong to a family $cal(F)$ from @sec:dynamics-families. For instance, polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum_k alpha_k P^k$.
These families have two properties that suggest they could regularize alignment:
they are *horizontal* (symmetric $N$ ensures no gauge drift), and they have *restricted structure* that random gauge errors would violate.

If the true dynamics belong to $cal(F)$, then correct gauges produce trajectories fittable by some $f in cal(F)$, while wrong gauges produce trajectories requiring dynamics outside $cal(F)$.
The dynamics family acts as regularization on the gauge choice.

This leads to a joint optimization formulation:

#definition(title: "Joint Alignment-Learning Problem")[
  Given ASE embeddings ${hat(X)^((t))}_(t=0)^T$ and dynamics family $cal(F)$, find gauge corrections ${Q_t in O(d)}$ and $f in cal(F)$ minimizing:
  $ cal(L)({Q_t}, f) = sum_(t=0)^(T-1) ||hat(X)^((t+1)) Q_(t+1) - hat(X)^((t)) Q_t - delta t dot f(hat(X)^((t)) Q_t)||_F^2 $
]

The objective measures how well the aligned trajectory ${hat(X)^((t)) Q_t}$ is explained by dynamics $f$: if the gauges are correct and $f$ captures the true dynamics, each step's displacement should match the predicted velocity.


== The identifiability principle

The theoretical case for structure-constrained alignment rests on a clean separation between gauge artifacts and horizontal dynamics.

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

The mechanism is elegant: random ASE gauge errors $R^((t))$ introduce skew-symmetric contamination that *cannot be absorbed* by symmetric dynamics.
Requiring symmetric dynamics implicitly selects gauges where the contamination vanishes.

For specific families, this yields concrete algorithms.
For linear symmetric dynamics, one obtains an alternating scheme: fix gauges and solve a least-squares problem with symmetry constraint for the dynamics matrix $M$; fix $M$ and solve orthogonal Procrustes for each gauge.
For polynomial dynamics $dot(X) = (sum_k alpha_k P^k) X$, the gauge invariance of $P = hat(X) hat(X)^top$ simplifies the dynamics step to linear regression in the $alpha_k$ coefficients.
Both have closed-form updates per step.


== The downstream pipeline: from trajectories to equations

Suppose, optimistically, that the trajectory recovery problem were solved, that is, some method (structure-constrained alignment, a future $P$-level estimator, or domain-specific prior information) produced gauge-consistent estimates $tilde(X)^((t))$ of the latent positions, up to noise.
When multiple independent network samples $A_1^((t)), ..., A_m^((t))$ are available at each time $t$, averaging $macron(A)^((t)) = 1/m sum_i A_i^((t))$ before embedding reduces per-entry variance by a factor of $m$ and per-vertex ASE error by $sqrt(m)$, potentially making the trajectory recovery feasible even when individual snapshots are noisy.

Given such a recovered trajectory, the remaining steps are well-established.
*Universal Differential Equations* (UDEs) @rackauckas2020universal provide a framework for learning dynamics that combine known mechanistic structure with neural network flexibility:
$ dot(X) = g(f_("known")(X, phi), f_("NN")(X, theta)) $
For RDPG dynamics, the known structure comes from the families in @sec:dynamics-families. For example, the polynomial architecture $dot(X) = (sum_k alpha_k (theta, X) P^k) X$ ensures horizontality by construction while allowing the coefficients to be learned.
Gradients flow through the ODE solver via adjoint sensitivity methods, and the gauge-consistent architecture constrains the hypothesis space.

Once a UDE is trained, *symbolic regression* @symbolicregression can extract interpretable closed-form equations.
By sampling state-velocity pairs $(X, f_theta(X))$ from the trained model and searching for symbolic expressions of the form $N(P) X$, one can recover explicit dynamics equations completing the path from observed adjacency matrices to differential equations.

The bottleneck is step 2: obtaining $tilde(X)^((t))$.
The UDE and symbolic regression machinery is mature; the trajectory recovery problem is not.


== Why the constructive problem remains open

Despite the clean identifiability theory, translating @thm:gauge-contamination into a reliable recovery method faces three interacting difficulties.

*Finite-sample bias.*
Each spectral embedding $hat(X)^((t))$ estimates the true positions with error $O_p(1\/sqrt(n))$.
When the true dynamics are slow (small $||X^((t+1)) - X^((t))||$), the signal-to-noise ratio for alignment degrades: we are trying to distinguish small true displacements from estimation noise of comparable magnitude.
The alternating optimization inherits this: the dynamics step fits a combination of true signal and noise, while the gauge step aligns to a noisy target.
The two sources of error reinforce rather than cancel.

*Expressiveness of dynamics families.*
The identifiability argument requires that $cal(F)$ be restrictive enough that wrong gauges produce trajectories outside $cal(F)$.
But in practice, even low-degree polynomial families can be surprisingly expressive.
With $K + 1$ free parameters and $T$ time steps, the polynomial family can fit trajectories contaminated by moderate gauge errors, absorbing the skew-symmetric artifact into the coefficients rather than rejecting it.
The problem is sharpest for small $d$: when $d = 2$, the skew-symmetric space $so(2)$ is one-dimensional, and a single-parameter gauge drift can be partially absorbed by adjusting the dynamics coefficients.
Making $cal(F)$ more restrictive risks excluding the true dynamics; making it more expressive loses the regularization.

*Holonomy and global consistency.*
Even if local alignment succeeds (each consecutive pair of frames is well-aligned), holonomy (@sec:fiber-bundle) implies that gauge drift can accumulate over the trajectory.
For dynamics tracing closed or nearly-closed loops in the base space $cal(B)$, no sequence of local corrections can achieve global consistency: the horizontal lift of a closed curve need not close.
This is not a statistical issue but a topological one, and no amount of data resolves it.

*The fundamental tradeoff.*
The core difficulty is a tension between two requirements.
The dynamics family must be restrictive enough to reject gauge-contaminated trajectories (identifiability), but expressive enough to capture the true dynamics (model adequacy).
For the problem to be well-posed, the "gap" between what $cal(F)$ can fit and what gauge contamination produces must exceed the noise level.
Characterizing when this gap is sufficient (as a function of $n$, $d$, $T$, the spectral gap of $P$, and the dynamics family) is the central open problem.

#remark[
  The difficulty is not that the identifiability principle fails theoretically.
  @thm:gauge-contamination is sharp: in the continuous-time, infinite-data limit, symmetric dynamics _do_ identify gauges.
  The difficulty is that the finite-sample, discrete-time setting introduces errors that interact with the gauge-dynamics coupling in ways that the asymptotic theory does not capture.
  Progress likely requires either stronger structural assumptions (e.g., working directly on $P$-dynamics to bypass the gauge problem entirely) or fundamentally different estimation strategies (e.g., information-theoretic approaches that characterize the minimax rate for dynamics recovery).
]


= Discussion <sec:discussion>

*What we achieved.*
We provided a rigorous geometric framework for understanding dynamics on RDPGs.
The fiber bundle perspective (@sec:fiber-bundle) formalizes gauge freedom via principal bundles, with explicit formulas for the connection 1-form, curvature, and holonomy.
We established a sharp holonomy dichotomy (@sec:holonomy-dynamics): polynomial dynamics have trivial holonomy through commuting generators, while Laplacian dynamics generically produce full $"SO"(d)$ holonomy, which constitutes the worst case for alignment.
We derived Cramér-Rao lower bounds (@sec:info-theoretic) revealing that geometric and statistical difficulty are controlled by the same spectral gap, an inextricable duality.
We cataloged concrete families of dynamics on RDPG latent positions, characterizing their horizontality, observability, and parameter complexity.
We showed that existing methods, such as joint embeddings assuming the wrong generative model (@sec:why-not-uase), Bayesian smoothing approaches that interpolate rather than learn dynamics (@sec:bayesian-smoothing), fail to address the fundamental obstructions.
We established the identifiability principle (@thm:gauge-contamination): symmetric dynamics cannot absorb skew-symmetric gauge contamination, providing a theoretical foundation for structure-constrained alignment.

*What remains open.*
The gap between identifiability and practical recovery is the central challenge.
@thm:gauge-contamination guarantees that, in the continuous-time limit with perfect data, dynamics structure resolves gauge ambiguity.
But the discrete, finite-sample setting introduces errors that interact with the gauge-dynamics coupling in ways not captured by the asymptotic theory.
The expressiveness-restrictiveness tradeoff for dynamics families, the finite-sample signal-to-noise ratio for alignment, and holonomy over long trajectories all remain unresolved.

*Directions for progress.*
Several avenues seem promising.
First, the $P$-dynamics approach (@sec:p-dynamics) bypasses the gauge problem in principle, but the estimation theory for recovering $N$ from noisy Bernoulli samples $A^((t))$ remains undeveloped.
The Lyapunov inverse amplifies noise at rate $1\/(lambda_iota + lambda_gamma)$, suggesting that the minimax rate for estimating $N$ from $T$ adjacency matrices will depend critically on the spectral gap of $P$.
Characterizing this rate, and understanding how it interacts with the choice of dynamics family (polynomial vs. general symmetric $N$), would clarify whether the $P$-level approach is practically viable or faces the same fundamental barriers as $X$-level methods.

Second, the holonomy characterization (@sec:holonomy-dynamics) establishes that polynomial dynamics have trivial holonomy while Laplacian dynamics generically have full $"SO"(d)$ holonomy, but several questions remain.
Proving the full holonomy conjecture for $d >= 3$ requires a transversality argument that the commutators $X^top [L_1, L_2] X$ span $frak(s o)(d)$ along generic trajectories.
More practically, quantifying the *rate* at which holonomy accumulates along specific trajectories, and not just its generic nontriviality, would connect the abstract topological obstruction to concrete alignment error budgets.

Third, the Cramér-Rao bound of @prop:cramer-rao establishes the Fisher information structure and its spectral gap dependence, but leaves open whether the implied rates are achievable.
For polynomial dynamics, where holonomy is trivial and the $P$-dynamics are explicit, a matching upper bound via maximum likelihood on the $P$-trajectory seems within reach.
For Laplacian dynamics, the entanglement of Fisher information with gauge degrees of freedom (@sec:info-theoretic) suggests that achievability may require estimators that account for holonomy, potentially through the $P$-level approach of @sec:p-dynamics.

Fourth, the geometry changes qualitatively in the sparse regime $P = rho_n X X^top$ with $rho_n -> 0$: the injectivity radius shrinks as $sqrt(rho_n lambda_d)$, curvature blows up, and ASE convergence rates deteriorate.
The interplay between sparsity, curvature, and dynamics recovery in this regime is largely unexplored.

*Broader impact.*
Learning interpretable dynamics from network data could enable mechanistic understanding in domains from neuroscience to ecology.
However, the RDPG assumption, that connection probabilities arise from latent position dot products, is strong.
Real networks may violate this assumption, and any learned dynamics should be validated against domain knowledge.


= Conclusion

We investigated the problem of learning differential equations governing time-evolving Random Dot Product Graphs, developing a rigorous geometric framework based on principal fiber bundles.

We identified three fundamental obstructions: gauge freedom (formalized via the connection 1-form and characterized by @thm:invisible), realizability constraints (the tangent space of $cal(B)$), and recovering trajectories from spectral embeddings (complicated by holonomy and information-theoretic limits).
We established a sharp dichotomy: polynomial dynamics have trivial holonomy and commuting generators (@prop:poly-trivial-holonomy), making alignment a purely statistical problem, while Laplacian dynamics generically produce full $"SO"(d)$ holonomy (@prop:laplacian-holonomy), making alignment simultaneously a statistical and topological challenge.
We derived Cramér-Rao lower bounds (@prop:cramer-rao) revealing a statistical-geometric duality: the same spectral gap that controls curvature also controls Fisher information, so geometric and statistical difficulty are inextricable.
We proved that existing joint embedding methods assume generative models incompatible with ODE dynamics, and that Bayesian smoothing approaches lack the dynamical consistency required for learning state-dependent velocity fields.

We established the identifiability principle that dynamics structure can resolve gauge ambiguity: symmetric dynamics cannot absorb the skew-symmetric contamination from wrong gauges (@thm:gauge-contamination).
However, we showed that this theoretical identifiability faces significant practical obstacles in the finite-sample, discrete-time setting, and framed the constructive recovery problem as an open challenge.

This work establishes mathematical foundations for the study of dynamics on temporal network data.
The characterization of invisible dynamics, the horizontal lift formula, the holonomy dichotomy between polynomial and Laplacian families, and the statistical-geometric duality, provides both a geometrical language and the concrete results needed to formulate and attack the open problems that remain.
Progress on these problems, whether through $P$-level dynamics, tighter information-theoretic bounds, or estimation strategies that account for holonomy, would open the door to mechanistic understanding of how and why networks evolve.


= Acknowledgments

// TODO

= Data and Code Availability

This paper is primarily theoretical. No datasets were generated or analyzed.


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
