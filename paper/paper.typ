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
    Across ecology, history, economics, social behavior, cultural dynamics, and more, many phenomena can be described in terms of entities establishing and disestablishing interactions with each other. These scenarios are commonly represented mathematically as temporal networks, and the time evolution of these objects is studied as a time series with the goal of predicting future network states. Here, instead, we take a dynamical systems perspective: when the goal is to understand _why_ networks evolve as they do, can we learn the differential equations that govern them?
    We investigate this question within the framework of Random Dot Product Graphs (RDPGs), where each network snapshot is generated from latent positions evolving under unknown dynamics. We identify three fundamental obstructions to recovering these dynamics: gauge freedom from the rotational ambiguity in latent positions, realizability constraints from the manifold structure of the probability matrix, and the trajectory recovery problem arising from spectral embedding artifacts.
    We develop a geometric framework based on principal fiber bundles that formalizes these obstructions and reveals their interplay. A sharp holonomy dichotomy emerges: polynomial dynamics have trivial holonomy, making gauge alignment purely statistical, while Laplacian dynamics generically produce non-trivial holonomy (proven to be full $"SO"(2)$ for $d = 2$, conjectured $"SO"(d)$ for $d >= 3$), adding a topological obstruction. Cramér-Rao lower bounds show that the spectral gap controlling geometric difficulty simultaneously controls statistical difficulty, an inextricable duality.
    We establish an identifiability principle stating that symmetric dynamics cannot absorb skew-symmetric gauge contamination. Yet, we identify and present significant practical obstacles remain in finite samples. We frame the gap between identifiability and practical recovery as an open challenge and discuss directions for progress.
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

+ *Recovering trajectories from embeddings* (@sec:trajectory-problem): In the RDPG pipeline, we do not observe $X(t)$ or $P(t)$. We observe adjacency matrices and then estimate $X(t)$ via ASE. This estimation step necessarily introduces a time-dependent gauge choice. Even when the underlying $X(t)$ is smooth, the estimated embeddings can jump erratically because the eigendecomposition picks a representative of the $O(d)$ fiber at each time step. This is not something one can fix by taking smaller time steps. It is a structural artifact of eigenvector indeterminacy coupled with estimation from noisy graphs.

We show that existing approaches fail to address these obstructions.
Joint embedding methods like UASE @gallagher2021spectral and Omnibus @levin2017central assume generative models incompatible with ODE dynamics on latent positions. In fact, they model time-varying "activity" against fixed "identity," not genuinely evolving positions.
Bayesian approaches with hierarchical smoothness priors @loyal2025 produce smooth trajectories but lack *dynamical consistency*: they interpolate well but don't enforce that velocity depends on state according to $dot(X) = f(X)$.
Pairwise Procrustes alignment is local and cannot ensure global consistency.

Our main contributions are theoretical.
We develop a geometric framework based on principal fiber bundles that formalizes gauge freedom, realizability, and trajectory recovery as distinct but interrelated obstructions.
We prove that random gauge artifacts introduce skew-symmetric contamination that cannot be absorbed by symmetric dynamics (@thm:gauge-contamination), establishing an identifiability principle.
We establish a holonomy dichotomy among horizontal dynamics families: polynomial dynamics have commuting generators and trivial holonomy, so gauge alignment is purely a statistical problem; Laplacian dynamics generically produce non-trivial holonomy (full $"SO"(2)$ for $d = 2$; conjectured $"SO"(d)$ for $d >= 3$), adding a topological obstruction on top of the statistical one.
We derive Cramér-Rao lower bounds showing that the spectral gap controlling curvature simultaneously controls Fisher information, so geometric and statistical difficulty are inextricable.
However, we show that exploiting these structures in practice faces fundamental difficulties: the bias induced by finite samples and the expressiveness of natural dynamics families make the constructive problem substantially harder than the identifiability theory suggests.
We frame the gap between identifiability and practical recovery as an open problem and discuss directions for progress.

*Paper outline.*
@sec:rdpg reviews RDPG fundamentals.
@sec:obstructions develops the geometric framework: gauge freedom, realizability, and the fiber bundle perspective with connections, curvature, and holonomy.
@sec:trajectory-problem addresses the practical problem of recovering trajectories from spectral embeddings, showing why existing methods fail and motivating the need for dynamics-aware approaches.
@sec:dynamics catalogs concrete dynamics families, analyzes their observable consequences through the Lyapunov equation, and establishes the holonomy dichotomy.
@sec:info-theoretic derives information-theoretic lower bounds revealing the statistical-geometric duality.
@sec:constructive discusses the constructive problem: the identifiability principle, its algorithmic implications, the practical obstacles that remain, and a tractable special case using anchor nodes, illustrated with numerical experiments.
@sec:discussion reflects on achievements and open problems.

*Notation.*
$X in RR^(n times d)$ denotes the matrix of latent positions with rows $x_i^top$.
$RR_*^(n times d) = {X in RR^(n times d) : "rank"(X) = d}$ denotes the set of full-rank $n times d$ matrices.
$P = X X^top$ is the probability matrix.
$O(d)$ is the orthogonal group; $so(d)$ is its Lie algebra of skew-symmetric matrices.
$hat(X)$ denotes an estimate (e.g., from spectral embedding).
$||dot||_F$ is the Frobenius norm; $||M||_(2 -> infinity) = max_i ||e_i^top M||_2$ is the two-to-infinity norm, measuring the largest row norm of $M$.
$O_p(dot)$ denotes stochastic order: $Y_n = O_p(a_n)$ means $Y_n \/ a_n$ is bounded in probability.
$"Sym"(d)$ denotes $d times d$ symmetric matrices.
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
Throughout this paper, we restrict attention to configurations where all edge probabilities lie in the open interval: $0 < P_(i j) < 1$ for all $i, j$.
This excludes boundary cases (deterministic edges or guaranteed non-edges) that would create singularities in the Fisher information and complications in the differential geometry; see @sec:fiber-bundle for details.

Given latent positions, edges are drawn independently:
$ A_(i j) tilde.op "Bernoulli"(P_(i j)) quad "for" i =< j $
with $A_(j i) = A_(i j)$ (undirected) and $A_(i i) tilde.op "Bernoulli"(P_(i i))$, so $EE[A] = P$ exactly.
In some applications it is common to disregard self-links by setting $A_(i i) = 0$, yielding $EE[A] = P - "diag"(P)$.
This hollows the diagonal and introduces a systematic off-manifold bias that complicates the spectral theory (the perturbation $"diag"(P)$ has spectral norm at most 1, comparable to sampling noise for moderate $n$).
Our geometric and statistical arguments extend to this setting by treating the missing/zeroed diagonal as an additional structured perturbation term; we avoid it in the main text to keep focus and keep $EE[A]$ exactly on the rank-$d$ manifold ${X X^top}$.
(If desired, the hollow-diagonal extension can be carried out in an appendix by explicitly tracking the extra diagonal perturbation through the embedding and alignment steps.)

== Adjacency spectral embedding <sec:ase>

Given an observed adjacency matrix $A$, we estimate latent positions via *adjacency spectral embedding* (ASE) @athreya2017statistical.

Since $A$ is symmetric, it has an eigendecomposition $A = U Lambda U^top$ with real eigenvalues $lambda_1 >= lambda_2 >= ... >= lambda_n$ and orthonormal eigenvectors.
The rank-$d$ ASE is:
$ hat(X) = U_d |Lambda_d|^(1\/2) $
where $U_d in RR^(n times d)$ contains the $d$ leading eigenvectors (by eigenvalue magnitude) and $Lambda_d = "diag"(lambda_1, ..., lambda_d)$.
We take absolute values because $A$ can have negative eigenvalues due to sampling noise, even though the true $P$ is positive semidefinite.
In the dense regime where $lambda_d (P) = Omega(sqrt(n))$, Weyl's inequality guarantees that the top $d$ eigenvalues of $A$ are positive with high probability, so the absolute value has no effect asymptotically.
We retain it for definiteness at finite $n$.

#remark[
  For symmetric matrices, eigendecomposition and SVD are closely related: if $A = U Lambda U^top$ (eigen) and $A = tilde(U) Sigma tilde(V)^top$ (SVD), then $tilde(U) = tilde(V) = U$ (up to signs) and $sigma_i = |lambda_i|$.
  We use eigendecomposition for ASE (the standard in RDPG literature) but SVD for Procrustes alignment (the standard for that problem).
]

*Statistical properties.*
The statistical properties of RDPG are well studied, and here we remind some fundamental results from the literature. Under mild conditions, ASE is consistent: as $n -> infinity$, the rows of $hat(X)$ converge to the true latent positions (up to orthogonal transformation) at rate $O(1\/sqrt(n))$ @athreya2016limit @athreya2017statistical.
Specifically, there exists $Q in O(d)$ such that $max_i ||hat(x)_i - x_i Q|| = O_p (1 / sqrt(n))$, and conditionally on each latent position, $sqrt(n)(hat(x)_i - Q x_i)$ converges to a Gaussian whose covariance depends on the edge variance profile @athreya2016limit @rubindelanchy2022statistical.
The perturbation expansion $hat(X) = X W + (A - P) X (X^top X)^(-1) W + R$, where $W in O(d)$ and $||R||_(2 -> infinity) = o(n^(-1\/2))$, shows that the leading error term is linear in $A - P$ and has conditional mean zero @cape2019two.
ASE is asymptotically unbiased but not fully efficient for individual latent positions; Xie and Xu's one-step procedure achieves what they term _local efficiency_: for each vertex, the estimator's asymptotic covariance matches that of the oracle MLE that knows all other latent positions, up to an orthogonal transformation @xie2021efficient.
For stochastic blockmodel parameters, the spectral estimator is itself asymptotically efficient @tang2022efficient.
The eigenvalues of $A$ are also asymptotically normal about those of $P$, with an $O(1)$ bias from the Bernoulli variance @tang2018eigenvalues.
This bias is negligible relative to the leading eigenvalue ($lambda_1 tilde n$), but can be comparable to the spectral gap $delta = lambda_d$ when the gap is small, and is of the same order as the eigenvalue fluctuations ($O(sqrt(n))$); for dynamics estimation, what matters is the bias relative to the *changes* in eigenvalues across time steps, a point we revisit in @sec:info-theoretic.

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
  For $X in RR_*^(n times d)$ (i.e., $"rank"(X) = d$), a vector field $f$ produces invisible dynamics ($dot(P) = 0$) if and only if $f(X) = X A$ for some skew-symmetric matrix $A in so(d)$.
] <thm:invisible>

#proof[
  ($arrow.l.double$) If $f(X) = X A$ with $A^top = -A$, then:
  $ dot(P) = f(X) X^top + X f(X)^top = X A X^top + X A^top X^top = X A X^top - X A X^top = 0 $

  ($arrow.r.double$) Suppose $dot(P) = f(X) X^top + X f(X)^top = 0$.
  Let $G = X^top X$, which is invertible since $"rank"(X)=d$.
  Left-multiply the identity by $(I - X G^(-1) X^top)$ to project onto $"col"(X)^perp$:
  $ 0 = (I - X G^(-1) X^top) f(X) X^top $
  Since $X^top$ has full row rank, this implies
  $ (I - X G^(-1) X^top) f(X) = 0 $
  hence $f(X) in "col"(X)$ and there exists $B in RR^(d times d)$ such that $f(X) = X B$.

  Substituting back into $dot(P)=0$ gives
  $ 0 = X(B + B^top) X^top $
  and since $X$ has full column rank, this forces $B + B^top = 0$, i.e. $B in so(d)$.
  Taking $A = B$ completes the proof.
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
The null-null block represents directions supported entirely on $"col"(P)^perp$, hence directions that cannot arise from $dot(P) = F X^top + X F^top$ and would generically increase the rank of $P$.
Movements in these directions are forbidden for *fixed* latent dimension $d$ (but they might be achievable for networks that start rank deficient and grow in dimension).

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

In a dynamic RDPG setting we have a latent structure $X$ that evolves in time following a certain regime. Yet, we observe only a function of $X$ which leaks information: we sample graphs from the probability of interaction matrix $P(t) = X(t) X(t)^top$, and we know $P(t)$ is invariant under $O(d)$ rotations.

We are in a similar situation to trying to observe the flight of birds from the ground. You can easily track the birds' motion across the sky (left/right/forward/back), but it's much harder to infer how they are moving *toward* or *away from* the ground without additional cues. Many different 3D trajectories can cast the same 2D "shadow" when viewed from a fixed vantage point.

In our case, the observable is $P = X X^top$ (or, in practice, noisy graphs drawn from it), and the latent configuration is $X$ itself.
The latent configuration contains the information we need to talk about velocities and accelerations in latent space, but it cannot be directly measured.
Replacing $X$ by a rotated version $X Q$ does not change $P$. It is a different representative of the same "shadow." (This is only an analogy. The gauge direction is not literally a spatial vertical, but the feeling is the same. There is a hidden direction you cannot see from the base space.)

A *fiber bundle* formalizes this structure and allows us to study the relationship between the base space and the total (latent) space.
We call the space of all possible observables, e.g. $P(t)$, the *base space* $cal(B)$.
Above each observable $P$, there is a "fiber" of equivalent latent configurations ${X Q : Q in O(d)}$, all producing the same $P$.
We call the full space of latent configurations $X$ the *total space* $cal(E)$. We are endowed with a projection $pi(X) = X X^top$ dropping from the total space to the base space.

In a fiber bundle we can consider *lifts* that take trajectories from the base to the total space: given a path $P(t)$ of evolving observables in the base space, we "lift" it to a path $X(t)$ in the total space.
Alas, there are infinitely many possible choices of what $X(t)$ to pick for any $P(t)$, differing by time-varying gauge choices $Q(t)$.
The _connection_ and _curvature_ of the bundle tell us which lifts are "natural" (no spurious rotation) and whether consistent lifting is possible at all.
When it is not, and the bundle is inherently curved, we get *holonomy*. This is gauge drift that accumulates even along the most careful trajectory.

We now make this precise.

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
However, $X in B_+^d$ is sufficient but not necessary: many valid configurations exist with some entries outside the positive orthant, as long as all pairwise dot products remain in $[0,1]$.

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
For the remainder of this paper, we work on a connected component of the interior of $cal(E)$, where the geometry is smooth and the bundle structure is clean.
This ensures that trajectories $X(t)$ do not cross rank-deficient strata where the fiber bundle structure degenerates.

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
Uniqueness holds because the Lyapunov operator $Omega |-> G Omega + Omega G$ is invertible when $G = X^top X$ is positive definite: in the eigenbasis of $G$ (with eigenvalues $lambda_1, ..., lambda_d$), this operator maps each entry $Omega_(iota gamma)$ to $(lambda_iota + lambda_gamma) Omega_(iota gamma)$ independently, so it acts as a diagonal operator on the vector space of skew-symmetric matrices, with all eigenvalues $lambda_iota + lambda_gamma > 0$.
At rank-deficient points ($"rank"(X) < d$), $G$ has a zero eigenvalue, the Lyapunov operator acquires a kernel, and the horizontal-vertical decomposition ceases to be unique.
This is the algebraic counterpart of the geometric fact that the fiber bundle structure degenerates at the boundary of $cal(E)$, and foreshadows the role of the spectral gap in @prop:curv-spectral-gap.

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
The difficulty is that in practice we observe noisy adjacency matrices, not $P(t)$ directly (see @sec:p-dynamics for a detailed treatment of this estimation problem).

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
$ Lambda tilde(Sigma) + tilde(Sigma) Lambda = A $ <eq:horizontal-lift-lyapunov>

A small notational point: $A = V^top dot(P) V$ is already the range-range block expressed in the eigenbasis of $P = V Lambda V^top$. That is why the right-hand side is simply $A$, with no additional conjugation by powers of $Lambda$.

This is the same Lyapunov structure as in the connection 1-form above. And not coincidentally: both of them involve separating gauge from observable components. The following formula is essentially the horizontal projection of @absil2008optimization and @massart2020quotient, specialized to the spectral decomposition of $P$.

#proposition(title: "Horizontal Lift Formula")[
  Given $P = V Lambda V^top$ and realizable $dot(P)$ with blocks $A = V^top dot(P) V$ and $B = V^top dot(P) V_perp$, the horizontal lift at $X = V Lambda^(1\/2)$ is:
  $ dot(X) = V tilde(Sigma) Lambda^(1\/2) + V_perp B^top Lambda^(-1\/2) $
  where $tilde(Sigma)$ is the symmetric solution to @eq:horizontal-lift-lyapunov, given elementwise by:
  $ tilde(Sigma)_(iota gamma) = A_(iota gamma) / (lambda_iota + lambda_gamma) $
]

#proof[
  The derivation above (@eq:horizontal-lift-lyapunov and the preceding block algebra) constructs $dot(X)$ satisfying both the projection condition $dot(P) = dot(X) X^top + X dot(X)^top$ and the horizontality condition $X^top dot(X)$ symmetric.
  The Lyapunov equation has a unique symmetric solution because $lambda_iota + lambda_gamma > 0$ for all $iota, gamma$.
]

The formula reveals two contributions to horizontal motion:
- The first term $V tilde(Sigma) Lambda^(1\/2)$ handles motion within the current column space of $X$, determined by the range-range block $A$ via a Lyapunov equation
- The second term $V_perp B^top Lambda^(-1\/2)$ handles motion expanding into new directions, determined directly by the cross block $B$

In practice, computing horizontal lifts from discrete noisy observations requires interpolation and ODE integration.
We discuss in @sec:constructive how discrete Procrustes alignment approximates horizontal transport.

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

A classic result (see for example @kobayashi1963foundations) provides a precise qualitative characterization of the holonomy obstruction. (We will later return to concrete holonomy behavior for specific RDPG dynamics families.)

#theorem(title: "Holonomy Obstruction")[
  If the bundle has nontrivial curvature, there exist closed paths in $cal(B)$ such that no globally consistent gauge exists: any lift satisfies $X(1) = X(0) Q$ for some $Q != I$.
]

The result implies that, if the true dynamics $P(t)$ trace a closed loop (periodic network behavior), the underlying $X(t)$ may not close on itself. Instead, it returns rotated by the holonomy.
This is a fundamental obstruction: even perfect local alignment accumulates global gauge drift over cycles.

It is interesting to notice that the holonomy obstruction provides a *connection to spectral properties* of the (observed) graphs: in fact, the curvature of $cal(B)$ depends on the eigenvalues of $P = X X^top$. Since the nonzero eigenvalues of $P$ are exactly those of $X^top X$ (the Gram matrix of the latent positions), small $lambda_d$ arises when, for example:

- *Latent positions cluster in a lower-dimensional subspace*: if nodes' positions are nearly coplanar in $RR^d$, the columns of $X$ become nearly linearly dependent.

- *Stochastic block models with weak community structure*: for SBM with $X = Z B^(1\/2)$ (membership $Z$, block matrix $B$), the spectral gap of $P$ reflects that of $B$. When communities are hard to distinguish, $lambda_d$ is small.

- *Sparse networks*: if all entries of $X$ scale as $rho_n -> 0$ (sparsity parameter), then $lambda_d tilde rho_n^2 -> 0$.

This matters doubly: not only do geometric obstructions worsen as $lambda_d -> 0$ (the injectivity radius vanishes as $sqrt(lambda_d)$, and curvature blows up when $lambda_(d-1)$ is also small), but ASE convergence rates also deteriorate (they scale as $O(sqrt(log n \/ lambda_d))$ @cape2019two).
Configurations with small spectral gap are thus problematic both statistically (harder to estimate) and geometrically (harder to track gauges). We emphasize this expectation-setting point now because it will reappear later: in @sec:dynamics and @sec:info-theoretic, the same spectral quantities controlling curvature and injectivity also control estimation error and Fisher information.

#proposition(title: "Curvature and Spectral Gap")[
  The quotient manifold $cal(B) = RR_*^(n times d) \/ O(d)$ with the Procrustes metric has *sectional curvature* $K$ (a scalar measuring the Gaussian curvature of 2-dimensional sections, distinct from the Lie algebra-valued curvature 2-form $Omega$ above) given by O'Neill's formula for Riemannian submersions @oneill1966fundamental: for orthonormal horizontal vectors $overline(xi), overline(eta) in cal(H)_X$,
  $ K(xi, eta) = 3 ||cal(A)_(overline(xi)) overline(eta)||^2 = 3/4 ||[overline(xi), overline(eta)]^cal(V)||^2 $
  where $cal(A)_(overline(xi)) overline(eta) = 1/2 [overline(xi), overline(eta)]^cal(V)$ is the O'Neill $A$-tensor and $[overline(xi), overline(eta)]^cal(V)$ denotes the vertical component of the Lie bracket. Since the total space $RR_*^(n times d)$ is flat, all base curvature arises from the $A$-tensor.

  Massart, Hendrickx, and Absil @massart2019curvature compute the sectional curvature explicitly for this quotient:
  - If $d = 1$, the sectional curvature is identically zero (the base space is flat).
  - If $d >= 2$, the minimum sectional curvature is zero, achieved when the horizontal fields commute.
  - The maximum sectional curvature at $[X]$ diverges when the two smallest eigenvalues $lambda_(d-1), lambda_d$ of $P = X X^top$ approach zero simultaneously.
]<prop:curv-spectral-gap>

Two related but distinct geometric obstructions emerge from the spectral gap.
The *curvature* depends on the $A$-tensor, which involves the connection coefficients $1\/(lambda_iota + lambda_gamma)$ for $iota != gamma$.
Since the connection is skew-symmetric, the worst case is $1\/(lambda_(d-1) + lambda_d)$: curvature blows up only when *both* of the two smallest eigenvalues approach zero.
If $lambda_d -> 0$ but $lambda_(d-1)$ remains bounded, the curvature stays finite (the manifold looks locally like a cylinder in the collapsing direction).
By contrast, the *injectivity radius* at $[X]$ equals $sqrt(lambda_d)$ @massart2020quotient and vanishes whenever $lambda_d -> 0$ alone: beyond this radius, geodesics cease to be length-minimizing and multiple paths can connect the same points.

For dynamics estimation, both phenomena matter.
Large curvature means that even short paths produce substantial holonomy, making local Procrustes alignment inconsistent over moderate distances.
Small injectivity radius means that interpolation between time steps is non-unique, even without curvature.
The $d = 1$ case is degenerate in a useful way: $O(1) = {plus.minus 1}$ is discrete, so the only gauge ambiguity is a global sign flip, and the curvature vanishes because there is no continuous rotation to accumulate.

=== Riemannian structure

The quotient $cal(B) tilde.equiv cal(E) \/ O(d)$ inherits a natural metric:

#definition(title: "Quotient Metric")[
  The Riemannian distance between $[X], [Y] in cal(B)$ is the *Procrustes distance*:
  $ d_cal(B)([X], [Y]) = min_(Q in O(d)) ||X - Y Q||_F $
]

This connects the geometry to computation: Procrustes alignment computes geodesic distance on the base space @massart2020quotient.
The global injectivity radius of $cal(B)$ is zero (since $lambda_d$ can be arbitrarily small), reflecting the nontrivial topology of the quotient.



= Recovering trajectories from spectral embeddings <sec:trajectory-problem>

The obstructions above concern what dynamics are *theoretically* possible.
We now turn to the *practical* problem: even when dynamics are observable and realizable, recovering trajectories from data is hard because spectral embedding introduces arbitrary gauge transformations at each time step.

== ASE introduces arbitrary gauge transformations

At each time $t$, ASE computes the eigendecomposition of $A^((t))$.
The eigenvectors are determined only up to sign (and rotation within repeated eigenspaces).
This means:
$ hat(X)^((t)) = X^((t)) R^((t)) + E^((t)) $
where $E^((t))$ is statistical noise and $R^((t)) in O(d)$ is the gauge factor.
Although $R^((t))$ is deterministic (fixed by the eigensolver's conventions, e.g. sign of the first nonzero entry), it is *discontinuous* in $t$: standard eigensolvers enforce conventions that jump whenever an entry crosses zero or eigenvalues change ordering.
These jumps are uncorrelated with the dynamics, so $R^((t))$ behaves effectively like an arbitrary selection from the fiber $O(d)$ at each time step.

*The key point*: Even if the true positions $X^((t))$ evolve smoothly, the estimates $hat(X)^((t))$ jump erratically because the $R^((t))$ are unrelated across time.
In principle, a smooth choice of eigenvectors exists when eigenvalues are simple and vary analytically, but standard numerical eigensolvers do not enforce this choice.

== Finite differences fail

Consider estimating velocity via finite differences:
$ hat(dot(X))^((t)) = (hat(X)^((t + delta t)) - hat(X)^((t))) / (delta t) $

Substituting the gauge-contaminated estimates:
$ hat(dot(X))^((t)) = (X^((t+delta t)) R^((t+delta t)) - X^((t)) R^((t))) / (delta t) + O(E) $

Even ignoring noise $E$, this is dominated by the gauge jump $R^((t+delta t)) - R^((t))$, which is $O(1)$ regardless of $delta t$.
As the "velocity" diverges as $delta t -> 0$, we're measuring gauge jumps, not dynamics.



== Pairwise alignment and error accumulation <sec:alignment-accumulation>

One might try aligning consecutive embeddings via Procrustes:
$ Q^((t)) = arg min_(Q in O(d)) ||hat(X)^((t+1)) - hat(X)^((t)) Q||_F $

This has a closed-form solution via SVD and finds the best rotation to match adjacent frames.
However, sequential pairwise alignment suffers from fundamental limitations:

+ *Local, not global:* Each alignment minimizes error between adjacent frames but doesn't ensure consistency across the full trajectory.

+ *Error accumulation:* The ASE gauge $R^((t))$ behaves like an arbitrary $O(1)$ rotation at each time step, unrelated across time. Pairwise Procrustes targets the relative gauge $R^((t))^(-1) R^((t+1))$ between consecutive frames, with error controlled by the ASE noise $E^((t))$. When these noise-induced errors are roughly independent across steps, the accumulated rotation error after aligning $T$ frames behaves diffusively, scaling like $O(sqrt(T) sigma)$ where $sigma$ is the per-step noise level.

+ *Holonomy is a separate obstruction:* The $O(sqrt(T) sigma)$ diffusive error is the *statistical* difficulty of alignment. The holonomy obstruction (@sec:fiber-bundle) is *geometric/topological* and exists even with perfect data ($sigma = 0$): for dynamics with nontrivial holonomy, no globally consistent gauge exists over closed loops, regardless of how accurately each pairwise alignment is performed. The two obstructions interact but should not be conflated.

+ *No dynamical constraint:* Even a perfectly aligned sequence need not correspond to a dynamically meaningful lift: Procrustes alignment is purely geometric and does not enforce that $dot(X) = f(X)$ for any structured family.

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

The mismatch is fundamental and can be stated precisely in terms of the rank of the concatenated data.
UASE assumes that the concatenated matrix $[A^((1)); A^((2)); ...; A^((T))]$ has a *common invariant column space* of dimension $d$: all time slices share the same left-singular vectors.
Under ODE dynamics on latent positions, the column space of $P^((t)) = X^((t)) X^((t))^top$ generically evolves (especially for Laplacian dynamics, where eigenvectors rotate), so the rank of the concatenated probability matrix grows with $T$.
UASE fits a static average subspace to what is inherently an evolving one, compressing the dynamical signal into a deformation of "activity scores" within a fixed basis.
Applying UASE to data from our model distorts the recovered trajectory, attributing genuine positional evolution to time-varying activities against fixed identities.

Similar issues affect Omnibus embedding @levin2017central and COSIE @arroyo2021inference: all assume a shared subspace structure that ODE dynamics do not preserve.

== Why Bayesian smoothing approaches are insufficient <sec:bayesian-smoothing>

Recent work @loyal2025 proposes Bayesian inference for dynamic RDPGs using hierarchical priors on latent positions.
By placing priors on successive differences (or equivalently, using Gaussian processes with smooth kernels), the resulting trajectories are smooth: velocities and accelerations are well-defined and continuous.

However, smoothness is necessary but not sufficient for dynamical consistency.
An ODE $dot(X) = f(X)$ constrains the velocity to be a *function of the current state*; a smoothness prior constrains velocities to vary continuously, but does not require that $dot(X)(t)$ is determined by $X(t)$.

To be clear about scope: the critique in this section is aimed at *smoothing-only* models that place priors directly on trajectories (or their increments) without including $f$ and the ODE constraint in the generative model. Bayesian inference with an explicit ODE/SDE structure is a different object entirely (see the comparison table below).

#definition(title: "Dynamical Consistency")[
  A trajectory $X(t)$ is *dynamically consistent* with respect to a function class $cal(F)$ if there exists $f in cal(F)$ such that $dot(X)(t) = f(X(t))$ for all $t$.
]

Hierarchical Bayesian priors ensure that $X(t)$, $dot(X)(t)$, and higher derivatives are all smooth, but they do not enforce the state-dependence constraint $dot(X)(t) = f(X(t))$ for any $f$.
This distinction is the gap between kinematic regularity and dynamical structure.

*Prior support and the ODE solution manifold.*
For a given initial condition $X(0) = X_0$ and dynamics $dot(X) = f(X)$, the solution $X(t)$ traces a *unique* curve: there is no uncertainty in the trajectory given $(f, X_0)$.
The space of all ODE solutions (over all $f in cal(F)$ and $X_0$) forms a *finite-dimensional manifold* in the infinite-dimensional space of smooth paths.
A hierarchical smoothness prior assigns positive probability to all smooth paths, a much larger set than just those satisfying some ODE. So, the ODE solution manifold has measure zero under such priors.

#proposition(title: "Measure Zero")[
  Let $mu$ be any Gaussian measure on smooth paths $C^k([0,T], RR^(n times d))$ with full support.
  Let $cal(M)_cal(F) = {X(dot) : dot(X) = f(X) "for some" f in cal(F)}$ be the manifold of ODE solutions.
  Then $mu(cal(M)_cal(F)) = 0$.
] <prop:measure-zero>

#proof[
  See @app:deferred-proofs.
]

The substantive implication concerns *posterior behavior*: Gaussian process posteriors with standard kernels (RBF, Matérn) cannot concentrate on $cal(M)_cal(F)$ as the number of observations grows, because the RKHS of these kernels spans the full path space and has no mechanism to enforce the ODE constraint.
The posterior will converge to a smooth path that best fits the data in the RKHS norm, which is generically not an ODE solution.

The posterior under a smoothness prior therefore concentrates on trajectories that are smooth (from the prior) and explain observed edges well (from the likelihood), but do not satisfy $dot(X) = f(X)$ for any reasonable $f$.
This is the distinction between *interpolation* and *dynamics learning*: both produce smooth curves through the data, but only the latter respects the constraint that velocity is state-dependent.

*Diagnostics for detecting dynamical inconsistency.*
Several tests can distinguish smooth-but-dynamically-inconsistent trajectories from genuine ODE solutions.
First, a *state-dependence test*: for autonomous dynamics $dot(X) = f(X)$, if the trajectory passes through similar positions at two different times ($X(t_1) approx X(t_2)$), then the inferred velocities must also be similar ($dot(X)(t_1) approx dot(X)(t_2)$); smoothing priors do not enforce this, and large velocity discrepancies at revisited states indicate dynamical inconsistency.
Second, an *ODE residual test*: fit candidate dynamics $hat(f)$ to the smoothed trajectory and compute residuals $r(t) = dot(hat(X))(t) - hat(f)(hat(X)(t))$; for dynamically consistent trajectories $||r(t)||$ should be small, while for smooth-but-wrong trajectories residuals will be systematically large.
Third, *flow consistency*: integrate the fitted $hat(f)$ forward from various initial conditions along the trajectory; dynamically consistent trajectories will track the integrated flow, while interpolated trajectories will diverge.
Finally, *medium-to-long-term forecasting* discriminates the two approaches: kinematic extrapolation assumes velocity and acceleration persist, while dynamic extrapolation adapts velocity to the current state via $f(X)$, and only the latter remains accurate when the trajectory enters new regions of state space.

*Alternative approaches with dynamical consistency.*

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Approach*], [*Smoothness*], [*Dynamical Consistency*]),
    table.hline(stroke: 0.5pt),
    [Hierarchical GP prior], [Yes ($C^k$)], [No: interpolation only],
    [SDE $d X = f(X) d t + sigma d W$], [No (Hölder $< 1\/2$)], [Yes, as $sigma -> 0$],
    [Neural ODE], [Yes ($C^k$)], [Yes (by construction)],
    [GP-ODE (NPODE) @heinonen2018learning], [Yes ($C^1$)], [Yes: learns $f$ as GP],
    [Structure-constrained (ours)], [Yes], [Yes: family $cal(F)$ enforced],
    table.hline()
  ),
  caption: [Comparison of approaches: smoothness vs dynamical consistency.]
)

The SDE formulation $d X = f(X) d t + sigma d W$ provides a principled bridge: Freidlin-Wentzell theory @freidlin1998random shows that as $sigma -> 0$, solutions concentrate around ODE solutions with rate function $J_T(phi) = 1/2 integral_0^T ||dot(phi)(t) - f(phi(t))||^2 d t$.
This rate function is precisely the *dynamical consistency penalty*, measuring how far a path deviates from being an ODE solution.

== Honest assessment

We must be candid: *aligning spectral embeddings to recover continuous-time trajectories is a hard open problem*.
The methods described above address related but different problems.
There is no existing method that provably recovers trajectories from ODE dynamics on RDPG latent positions.

Error accumulation in sequential alignment (@sec:alignment-accumulation) compounds over long trajectories.
Dynamical consistency considerations (@sec:bayesian-smoothing) distinguish interpolation from true dynamics learning.
Holonomy (@sec:fiber-bundle) implies that even perfect local alignment may accumulate global gauge drift.

This motivates investigating whether *structure of the dynamics themselves* can constrain the alignment problem, a question we return to in @sec:constructive after analyzing the dynamics families in @sec:dynamics.



= Dynamics on RDPGs <sec:dynamics>

The geometric framework of @sec:obstructions characterizes the abstract structure of gauge freedom, curvature, and holonomy, and @sec:trajectory-problem demonstrates the practical difficulties of recovering trajectories from spectral embeddings.
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
We now characterize what is _preserved_: the gauge-invariant footprint of latent-space dynamics, i.e. what survives after projection to the base space $cal(B)$ of probability matrices.

The key observation uses two ingredients together.
First, the RDPG model gives the factorization $P = X X^top$, so $P$ inherits smoothness from $X$ and the product rule applies.
Second, a dynamics model (start with the clean case $dot(X) = N X$) specifies how positions evolve.
Together, these imply a closed equation for $dot(P)$:
$ dot(P) = dot(X) X^top + X dot(X)^top = N X X^top + X X^top N^top = N P + P N^top $ <eq:p-dynamics>
When $N$ is symmetric, this becomes the Lyapunov equation $dot(P) = N P + P N$.

This is *not* a generic statement about temporal networks.
Without the low-rank factorization, $P(t)$ is just an arbitrary time-varying probability matrix whose entries could evolve independently, follow a different matrix ODE, or fail to be differentiable at all.
Equation @eq:p-dynamics is specific to the RDPG-as-dynamical-system viewpoint: it is what the latent ODE looks like once you throw away gauge.

The Lyapunov operator $cal(L)_P: "Sym"(n) -> "Sym"(n)$ defined by $cal(L)_P (N) = N P + P N$ is a linear map on symmetric matrices.
To analyze it, we pass to the eigenbasis of $P$.
Let $P = U Lambda U^top$ be the eigendecomposition, and write $tilde(N) = U^top N U$ and $tilde(dot(P)) = U^top dot(P) U$ for the representations in this basis, with indices $iota, gamma in {1, ..., n}$ labeling eigenvector directions (not nodes).
In this basis, $cal(L)_P$ acts diagonally: $(cal(L)_P (N))_(iota gamma) = (lambda_iota + lambda_gamma) tilde(N)_(iota gamma)$.

For the RDPG setting, where $P = X X^top$ with $X in RR^(n times d)$, the matrix $P$ has rank $d$ with eigenvalues $lambda_1 >= ... >= lambda_d > 0$ and $lambda_(d+1) = ... = lambda_n = 0$.
The Lyapunov operator is therefore *not invertible* on all of $"Sym"(n)$: when both $iota, gamma > d$, the equation $(lambda_iota + lambda_gamma) tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma)$ reduces to $0 dot tilde(N)_(iota gamma) = tilde(dot(P))_(iota gamma)$.
The realizability constraint (@prop:tangent) forces $tilde(dot(P))_(iota gamma) = 0$ for this block, so the equation says nothing about $tilde(N)_(iota gamma)$.
For $n = 100$ and $d = 3$, this leaves $(n - d)(n - d + 1)\/2 = 4753$ out of $n(n+1)\/2 = 5050$ entries of $tilde(N)$ completely unconstrained by the data.

However, the resolution is clean: the undetermined part of $N$ is exactly the part that does not affect $dot(X) = N X$ for the given $X$.

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

In summary, symmetric linear dynamics are identifiable from $(P, dot(P))$ up to the dynamically irrelevant null-null block, without ever choosing a gauge.
This perspective extends naturally to polynomial dynamics: if $dot(X) = N(P) X$ with $N(P) = sum_k alpha_k P^k$, then
$ dot(P) = N(P) P + P N(P) $
and the coefficients $alpha_k$ can, in principle, be recovered by fitting $dot(P)$ in the span of the symmetric matrices ${P^k P + P P^k}_(k=0)^K$.
Here the polynomial structure is doing real work: it pins down the otherwise-underdetermined action of $N$ off $"col"(P)$, eliminating the free null-null block by construction.

However, two fundamental obstacles prevent direct application.

*We do not observe $P(t)$.*
In practice, we observe adjacency matrices $A^((t))$ with $A_(i j)^((t)) tilde.op "Bernoulli"(P_(i j)(t))$.
Entry-wise, $A_(i j)^((t))$ is an unbiased estimate of $P_(i j)(t)$, with variance $P_(i j)(1 - P_(i j))$.
If we have $m$ independent graphs per time point, averaging $macron(A)^((t)) = 1/m sum_(ell=1)^m A_ell^((t))$ reduces this per-entry variance by a factor $m$.

But Lyapunov inversion (and most geometry) depends on the *spectrum* of $P$, not just its entries, and spectral accuracy depends on both $n$ and $m$.
Because eigendecomposition is nonlinear, the eigenvalues/eigenvectors of $macron(A)$ are biased at finite sample size.
Asymptotically, the leading perturbation has conditional mean zero and the residual bias is small @athreya2016limit @cape2019two, but the inverse map in @prop:lyapunov-invert divides by $(lambda_iota + lambda_gamma)$.
That division amplifies exactly the directions that are already ill-conditioned when the spectral gap is small. This is the same $1/lambda_d$ sensitivity that showed up geometrically in @sec:fiber-bundle, now wearing an algebraic hat.

*The parameter count of $N$ on the relevant subspace.*
Even after throwing away the dynamically irrelevant null-null block, the remaining degrees of freedom can still be large: the range-range and cross blocks together have $n d - d(d-1)\/2$ entries.

If $N$ is constant and you could somehow access a well-estimated pair $(P, dot(P))$, the constraint count matches the unknown count.
But as soon as $N$ varies with time/state, you need enough *excitation* in the trajectory to separate parameters.
Even in polynomial dynamics, where the coefficients $alpha_k$ are constant, identifying them requires that $P(t)$ moves through enough of the base space to distinguish the different powers $P^k$.
This is why the polynomial family is attractive: $K+1$ parameters (independent of $n$) is not just parsimonious, it is structurally compatible with what the data can hope to constrain.
Richer families quickly become underdetermined without strong structural assumptions or regularization.

#remark[
  The $P$-dynamics perspective and the fiber bundle perspective address different aspects of the same problem.
  @prop:lyapunov-invert says that the dynamics are identifiable _in principle_ from the gauge-invariant observable $P$, up to the dynamically irrelevant null-null block.
  The fiber bundle theory (@sec:fiber-bundle) addresses the harder question: what happens when we _must_ work with $X$ (e.g., because we need latent positions for interpretation or for the UDE pipeline), and how the geometry of the quotient space governs the difficulty of doing so.
  The two perspectives are complementary: the algebraic result tells us what information is available; the geometric framework tells us what it costs to extract it.
]


== Holonomy for horizontal dynamics families <sec:holonomy-dynamics>

The holonomy obstruction (@sec:fiber-bundle) raises a concrete question: for which horizontal dynamics families is holonomy trivial, and for which is it non-trivial?
This distinction has practical bite: trivial holonomy means a globally consistent gauge is achievable *in principle* (put differently: alignment is "only" statistical); non-trivial holonomy means that even perfect local alignment can accumulate global drift over cycles.

In this section we make the mechanism explicit for two dynamics families.
The punchline is a sharp commuting/non-commuting dichotomy:
polynomial dynamics use generators that commute along the trajectory (hence no curvature along that path, and no loops to begin with), while Laplacian-type generators generically do not commute, which produces curvature and allows non-trivial holonomy.

*Curvature from non-commuting generators.*
Consider horizontal dynamics $dot(X) = M(X) X$ with $M(X)$ symmetric, so that $X^top dot(X) = X^top M(X) X$ is symmetric and the dynamics are horizontal.
At two distinct times $t_1, t_2$, the generators $M_1 = M(X(t_1))$ and $M_2 = M(X(t_2))$ are both symmetric.
The commutator $[M_1, M_2] = M_1 M_2 - M_2 M_1$ is skew-symmetric (since transposing reverses the order).

The Lie bracket of the corresponding horizontal vector fields $overline(xi)_i (X) = M_i X$ (constant-coefficient extensions, horizontal since each $M_i$ is symmetric) is:
$ [overline(xi)_1, overline(xi)_2](X) = (M_2 M_1 - M_1 M_2) X = -[M_1, M_2] X $

Writing $S = -[M_1, M_2]$ (skew-symmetric), the bracket is $[overline(xi)_1, overline(xi)_2](X) = S X$.
This is *not* automatically vertical: vertical directions are right-multiplications $X Omega$, while $S X$ is a left-multiplication. They only coincide in special cases (roughly, when $S$ preserves $"col"(X)$).

But for curvature we only need the *vertical component* $[overline(xi)_1, overline(xi)_2]^cal(V)$.
Projecting $Z = S X$ onto $cal(V)_X = {X Omega : Omega in so(d)}$ gives $X Omega^*$ where $Omega^*$ solves the Lyapunov equation
$ G Omega^* + Omega^* G = 2 "skew"(X^top Z) = 2 X^top S X $
with $G = X^top X$ (see @app:vertical-projection).
Since $G$ is positive definite, this has a unique solution. In the eigenbasis of $G$:
$ Omega^*_(iota gamma) = 2(X^top S X)_(iota gamma) / (lambda_iota + lambda_gamma) $

So the vertical component vanishes iff the *projected commutator* vanishes:
$ X^top [M_1, M_2] X = 0. $
This is weaker than $[M_1, M_2]=0$, but for generic trajectories with $n >> d$ a nonzero commutator will typically have a nonzero projection as well.

Remembering @prop:curv-spectral-gap, by O'Neill's formula @oneill1966fundamental the sectional curvature of the 2-plane spanned by the projections of $overline(xi)_1, overline(xi)_2$ in $cal(B)$ is: $K(xi_1, xi_2) = 3/4 ||[overline(xi)_1, overline(xi)_2]^cal(V)||^2 slash (||overline(xi)_1||^2 ||overline(xi)_2||^2 - chevron.l overline(xi)_1, overline(xi)_2 chevron.r^2)$.
The $1 \/ (lambda_iota + lambda_gamma)$ factors in $Omega^*$ connect the curvature directly to the connection coefficients of the fiber bundle: the same denominators that amplify gauge sensitivity also amplify the vertical bracket.

Since $RR_*^(n times d)$ is flat (it is an open subset of Euclidean space), the base curvature arises *entirely* from the vertical bracket.

#proposition(title: "Curvature Criterion for Horizontal Dynamics")[
  For horizontal dynamics $dot(X) = M(X) X$ with $M$ symmetric, the sectional curvature along the trajectory in $cal(B)$ vanishes if and only if the projected commutator vanishes: $X^top [M(X(t_1)), M(X(t_2))] X = 0$ for all $t_1, t_2$ along the trajectory.
  A sufficient condition is that the full generators commute: $[M(X(t_1)), M(X(t_2))] = 0$.
  When the projected commutator is nonzero, the curvature is strictly positive, with:
  $ ||[overline(xi)_1, overline(xi)_2]^cal(V)||^2 = 4 sum_(iota < gamma) (lambda_iota lambda_gamma) / (lambda_iota + lambda_gamma) [(U^top [M_1, M_2] U)_(iota gamma)]^2 $
  where $P = U Lambda U^top$ is the eigendecomposition. The $1 \/ (lambda_iota + lambda_gamma)$ factors link the curvature directly to the connection coefficients of the fiber bundle.
] <prop:curvature-criterion>

#proof[
  The Lie bracket of the constant-coefficient horizontal extensions $overline(xi)_i (Y) = M_i Y$ is $[overline(xi)_1, overline(xi)_2](X) = -[M_1, M_2] X$ (standard commutator of linear vector fields).
  The vertical projection of $S X$ (with $S = -[M_1, M_2]$ skew-symmetric) is $X Omega^*$ where $Omega^*$ solves $G Omega^* + Omega^* G = 2 X^top S X$ (@app:vertical-projection).
  This vanishes iff $X^top [M_1, M_2] X = 0$.
  The norm formula follows from $||X Omega^*||^2 = "tr"(Omega^(*top) G Omega^*)$ evaluated in the eigenbasis of $G$ (see @app:vertical-norm for the computation).
]

*Polynomial dynamics: trivial holonomy.*
For polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum_(k=0)^K alpha_k P^k$, the induced $P$-dynamics are:
$ dot(P) = N(P) P + P N(P) = 2 sum_(k=0)^K alpha_k P^(k+1) $
This is a polynomial in $P$ alone. Writing $P = U Lambda U^top$ in its eigendecomposition, every power $P^k = U Lambda^k U^top$ shares the same eigenvectors $U$.
Therefore $dot(P) = U (2 sum alpha_k Lambda^(k+1)) U^top$ also has eigenvectors $U$: the eigenvectors of $P$ are stationary under polynomial dynamics, provided the spectrum of $P(0)$ is simple (all eigenvalues distinct).

The trajectory $P(t)$ lies on a $d$-dimensional submanifold of $cal(B)$ parameterized by the eigenvalues $(lambda_1(t), ..., lambda_d(t))$ alone, with each eigenvalue evolving independently:
$ dot(lambda)_iota = 2 sum_(k=0)^K alpha_k lambda_iota^(k+1) $

Two consequences follow immediately.

#proposition(title: "Polynomial Dynamics: Trivial Holonomy")[
  Polynomial dynamics $dot(X) = N(P) X$ with $N(P) = sum alpha_k P^k$ and simple initial spectrum ($lambda_1(0) > lambda_2(0) > ... > lambda_d (0) > 0$) have trivial holonomy, in the following concrete sense:
  + The generators $N(P(t_1))$ and $N(P(t_2))$ commute for all $t_1, t_2$ (they share eigenvectors), so the curvature *along this trajectory* vanishes by @prop:curvature-criterion.
  + Each eigenvalue satisfies a 1D autonomous ODE, so non-constant eigenvalue trajectories are monotone. In particular, the induced $P(t)$ cannot trace a nontrivial closed loop in $cal(B)$.
] <prop:poly-trivial-holonomy>

#proof[
  For (1): since the eigenvectors of $P$ are stationary, $N(P(t)) = U (sum alpha_k Lambda(t)^k) U^top$ for all $t$, with the same $U$. Any two diagonal matrices commute, so $[N(P(t_1)), N(P(t_2))] = U [sum alpha_k Lambda(t_1)^k, sum alpha_k Lambda(t_2)^k] U^top = 0$.

  For (2): each $lambda_iota (t)$ satisfies the autonomous ODE $dot(lambda)_iota = f(lambda_iota)$ with $f(x) = 2 sum alpha_k x^(k+1)$.
  By uniqueness of ODE solutions, if $lambda_iota (t_0) = lambda_iota (t_1)$ for $t_0 != t_1$, then $lambda_iota$ must be constant.
  Non-constant solutions are therefore monotone, so no closed loop in eigenvalue space is possible. As a consequence, no nontrivial closed loop in $cal(B)$ is possible.
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

The simple spectrum assumption is essential: if $lambda_iota(0) = lambda_gamma(0)$ for some $iota != gamma$, the eigenbasis within the repeated eigenspace is not unique, and polynomial dynamics preserve this degeneracy (the repeated eigenvalues evolve identically, maintaining the degeneracy for all time).
In this case, eigenvector "stationarity" is ill-defined because the basis itself is non-unique.
For block models with equal block sizes, for instance, repeated eigenvalues arise from the model symmetry.
This is a measure-zero condition in the space of initial conditions.

_What this means for nodes._
In the canonical gauge $X = U Lambda^(1\/2)$, node $i$ has latent position $x_i (t) = (U_(i 1) sqrt(lambda_1(t)), ..., U_(i d) sqrt(lambda_d(t)))$.
The loading $U_(i iota)$ (which can be read as the "membership weight" of node $i$ in eigenvector direction $iota$) is constant; what evolves is the scale $sqrt(lambda_iota (t))$ of each direction.

Since the eigenvalue ODE $dot(lambda) = f(lambda)$ is nonlinear for $K >= 1$, eigenvalues with different magnitudes evolve at different rates, and the ratios $lambda_iota (t) \/ lambda_gamma (t)$ change over time.
The effect on the node positions is an *anisotropic rescaling* of the latent space along fixed axes: in $d = 2$, the point cloud deforms as if inscribed in an ellipse whose axes do not rotate but whose eccentricity changes.
Node directions in $RR^d$ do change (they are not merely rescaled in magnitude), but the change is tightly constrained: all deformation is captured by the $d$ scalar functions $lambda_iota (t)$.

The exception is linear dynamics ($K = 0$, $f(lambda) = 2 alpha_0 lambda$), where $lambda_iota (t) = lambda_iota (0) e^(2 alpha_0 t)$ and all eigenvalues grow or decay at the same exponential rate.
In this case the ratios are constant, the point cloud is rescaled isotropically, and node angular positions in $RR^d$ are truly fixed.
For quadratic or higher dynamics ($K >= 1$), the nonlinearity of $f$ causes larger eigenvalues to evolve differently from smaller ones, producing genuine reshaping of the point cloud's geometry while preserving the eigenvector axes that define community alignment.
Note, however, that preserving the axes does not guarantee preserving community *separability*: if eigenvalue ratios diverge (e.g., $lambda_d (t) -> 0$ while $lambda_1$ grows), the point cloud collapses onto a lower-dimensional subspace, and communities that were distinguishable along the $lambda_d$ direction may become indistinguishable.

*Laplacian dynamics: non-trivial holonomy.*
Graph Laplacian dynamics $dot(X) = -L X$ with $L = D - P$, where $D = "diag"(P bold(1))$, provide a natural example of horizontal dynamics with non-trivial holonomy.
The generator $-L = P - D$ is symmetric, so the dynamics are horizontal.
But $L$ is *not* a polynomial in $P$: the degree matrix $D$ depends on the row sums of $P$, which mix eigenvector information that a polynomial in $P$ cannot access.

There is a notable exception: if $D$ is a scalar multiple of the identity ($D = c I$, i.e., the graph is *regular* with all node degrees equal), then $L = c I - P$ is affine in $P$, and the Laplacian dynamics reduce to the polynomial family.
Regular graphs (and more generally, vertex-transitive latent configurations where all rows of $X$ have the same norm) thus have trivial holonomy under Laplacian dynamics.
The non-trivial holonomy results below require that $D$ is *not* scalar: structural heterogeneity in the graph is the source of holonomy.

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
  For $d >= 2$ and initial conditions such that $D(P(0))$ is not a scalar multiple of the identity (i.e., the graph is not regular), Laplacian dynamics $dot(X) = -L(P) X$ generically produce trajectories in $cal(B)$ with strictly positive sectional curvature.

  Moreover, if the trajectory visits two states $P(t_1), P(t_2)$ for which the projected commutator is nonzero,
  $ X^top [L(P(t_1)), L(P(t_2))] X != 0, $
  then the connection has nonzero curvature along the trajectory segment spanning those states. In particular, any closed loop in $cal(B)$ that encloses a region of nonzero curvature yields nontrivial holonomy (a non-identity gauge rotation after horizontal transport).
] <prop:laplacian-holonomy>

#proof[
  It suffices to show that $X^top [L(P(t_1)), L(P(t_2))] X != 0$ generically.
  Write $L_i = D_i - P_i$ for $i = 1, 2$.
  Then
  $ [L_1, L_2] = [D_1 - P_1, D_2 - P_2] = [D_1, D_2] - [D_1, P_2] + [D_2, P_1] + [P_1, P_2]. $
  Since $D_1, D_2$ are both diagonal, $[D_1, D_2] = 0$. Also $[P_1, P_2] = 0$ whenever $P_1$ and $P_2$ share eigenvectors. Along a Laplacian trajectory, for distinct times $t_1 != t_2$ the eigenvectors generically rotate, so $P_1 = P(t_1)$ and $P_2 = P(t_2)$ generically fail to commute and $[P_1, P_2] != 0$.
  Moreover, $[D_1, P_2]$ is generically nonzero since the row sums encoded in $D_1$ depend on the full matrix $P_1$, not just its eigenvalues.
  For $n >> d$, a nonzero $n times n$ commutator $[L_1, L_2]$ generically projects to a nonzero $d times d$ matrix $X^top [L_1, L_2] X$. If $X^top [L_1, L_2] X = 0$, then $[L_1, L_2]$ maps $"col"(X)$ into $"col"(X)^perp$. This is a non-generic algebraic constraint on the entries of $[L_1, L_2]$ relative to $X$ (it requires a whole family of bilinear forms to vanish), so we do not expect it to hold except on a measure-zero set of trajectories or initial conditions.
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

This means that for rank-2 latent spaces under Laplacian dynamics, the obstruction is not merely a discrete sign flip (as in the simplest cases) but a continuous rotation by an arbitrary angle: you can get *any* amount of gauge drift, depending on the loop. From an alignment point of view, this is about as bad as it gets.

*Quantitative estimate.* For $d = 2$, the holonomy around a loop $gamma$ in $cal(B)$ is a rotation by angle:
$ phi(gamma) = integral.double_Sigma K thin d A $
where $Sigma$ is a surface bounded by $gamma$, $K$ is the sectional curvature, and $d A$ is the area element.
By the curvature formula of @massart2019curvature, regions where the smallest eigenvalues of $P$ are small contribute large curvature, so even short Laplacian trajectories that pass near rank-deficient states can accumulate substantial holonomy.

*Higher dimensions ($d >= 2$).*
For general $d$, the Lie algebra $frak(s o)(d)$ has dimension $d(d-1)\/2$.
The curvature 2-form produces holonomy-algebra elements $Omega^* in frak(s o)(d)$ with
$ Omega^*_(iota gamma) = 2(X^top [M_1, M_2] X)_(iota gamma) \/ (lambda_iota + lambda_gamma). $
As the trajectory visits different states, the *direction* of these elements in $frak(s o)(d)$ can change, and the span can grow.

For generic Laplacian dynamics with $n >> d$, the degree matrices at different times are sufficiently varied that we expect these curvature-generated directions to span all of $frak(s o)(d)$, giving holonomy group $"SO"(d)$.
We state this as a conjecture.

#conjecture(title: "Full Holonomy in Higher Dimensions")[
  For $d >= 2$, $n > d$, and initial conditions with $D(P(0))$ not scalar (non-regular graph), the restricted holonomy group of Laplacian dynamics $dot(X) = -L(P) X$ is $"SO"(d)$.
] <conj:full-holonomy>

A proof would require showing that the holonomy algebra elements $Omega^*_(iota gamma) = 2(X^top [L(P(t_1)), L(P(t_2))] X)_(iota gamma) \/ (lambda_iota + lambda_gamma)$, as $t_1, t_2$ vary along the trajectory, span $frak(s o)(d)$.
Since the Lyapunov weights $1\/(lambda_iota + lambda_gamma)$ are positive scalars that do not change rank, this is equivalent to requiring that the projected commutators $X^top [L(P(t_1)), L(P(t_2))] X$ span $frak(s o)(d)$.
For $d = 2$ this is automatic (one nonzero element spans a one-dimensional Lie algebra); for $d >= 3$ it requires a transversality argument exploiting the fact that the degree matrix $D = "diag"(P bold(1))$ mixes all $d$ eigenvector components.
The potential obstruction to full holonomy would be an *invariant subspace* of the degree operator: if the map $P |-> D(P) = "diag"(P bold(1))$ always preserved some subspace of $frak(s o)(d)$ aligned with the eigenvector structure, the holonomy group would be confined to a proper subgroup.
Generically, the degree operator mixes all eigenvector components, precluding such invariant subspaces; the non-generic exceptions are configurations with symmetries that force $D$ to commute with $P$ (regular graphs being the extreme case identified above).

The same argument applies to message-passing dynamics $dot(x)_i = sum_j P_(i j) g(x_i, x_j)$ with symmetric generators that are not polynomials in $P$.

*Summary:* the holonomy obstruction creates a sharp dichotomy among horizontal dynamics families.
Polynomial dynamics, which operate on the spectral structure of $P$ through commuting generators, preserve eigenvectors and have trivial holonomy: global gauge consistency is achievable without topological obstruction.
Laplacian and message-passing dynamics, which mix spectral and spatial structure through non-commuting generators, rotate eigenvectors and produce non-trivial holonomy: gauge drift accumulates even along horizontal trajectories, and no local alignment procedure can achieve global consistency over cycles.

This distinction has practical implications: for polynomial dynamics, the constructive alignment problem (@sec:constructive) is purely a statistical challenge (overcoming Bernoulli noise); for Laplacian dynamics, it is simultaneously a statistical _and_ topological challenge.

#remark[
  *Ambrose-Singer perspective (a quick intuition pump).*
  The relationship between local curvature and global holonomy is formalized by the Ambrose-Singer theorem @kobayashi1963foundations: the Lie algebra $frak(h o l)_p$ of the restricted holonomy group at $p$ is generated by curvature endomorphisms $Omega_q (u, v)$ evaluated at points $q$ reachable from $p$ by horizontal curves, after parallel-transporting them back to $p$.

  For $d = 2$, $frak(s o)(2)$ has no proper nonzero subalgebras, so a single nonzero curvature element forces the full restricted holonomy $"Hol"_p^0 = "SO"(2)$ (as used above).

  For $d >= 3$, one curvature value gives only one direction inside a much larger algebra.
  Full holonomy requires a bracket-generating phenomenon: curvature directions encountered along the trajectory must not all live inside a commuting subalgebra.
  Since the degree matrix $D(X)$ depends nonlinearly on the latent positions, the orientation of the induced curvature elements typically rotates as the trajectory evolves, which is exactly the mixing mechanism behind @conj:full-holonomy.

  Finally, the spectral gap appears again: as shown in @prop:curvature-criterion, the curvature magnitude is weighted by $1\/(lambda_iota + lambda_gamma)$.
  So a small spectral gap not only amplifies statistical noise (via ASE error bounds) but also makes holonomy effects larger, because the curvature-generated gauge rotations blow up in the ill-conditioned directions.
]




= Information-theoretic limits <sec:info-theoretic>

The geometric obstructions of @sec:obstructions (gauge freedom, curvature, holonomy), the trajectory recovery difficulties of @sec:trajectory-problem, and the dynamics analysis of @sec:dynamics constrain what is learnable in principle.
We now complement these with *statistical* obstructions: given $T$ noisy adjacency matrices, how accurately can we estimate the parameters governing the dynamics, regardless of the estimation method?

The answer reveals a duality: the same spectral gap $lambda_d$ that controls geometric difficulty also controls statistical difficulty, so the two obstructions reinforce each other.

== Fisher information for dynamics parameters

Consider a parametric dynamics family with parameter $theta in RR^k$ generating a trajectory $P(t; theta)$.
We observe $T$ adjacency matrices $A^((t)) tilde "Bernoulli"(P(t; theta))$ at times $t_1, ..., t_T$, with all entries conditionally independent given $P(t; theta)$.
The log-likelihood factorizes:
$ ell(theta) = sum_(t=1)^T sum_(i < j) [A_(i j)^((t)) log P_(i j)(t; theta) + (1 - A_(i j)^((t))) log(1 - P_(i j)(t; theta))] $

The Fisher information matrix $cal(I)(theta) in RR^(k times k)$ has entries:
$ cal(I)(theta)_(a b) = sum_(t=1)^T sum_(i < j) (partial P_(i j))/(partial theta_a) (partial P_(i j))/(partial theta_b) dot.c 1/(P_(i j)(1 - P_(i j))) $ <eq:fisher>
where $P_(i j) = P_(i j)(t; theta)$ and the derivatives $partial P_(i j) \/ partial theta_a$ capture how perturbations in the dynamics parameters propagate to the observations.

== Sensitivity propagation through the Lyapunov equation

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
$ cal(I)_t (alpha_0) = 4 t^2 sum_(i < j) P_(i j)(t)^2 / (P_(i j)(t)(1 - P_(i j)(t))) = 4 t^2 sum_(i < j) (P_(i j)(t)) / (1 - P_(i j)(t)) $
Summing over $T$ equally-spaced snapshots gives $cal(I)(alpha_0) = 4 sum_(t=1)^T t^2 sum_(i<j) P_(i j)(t) \/ (1 - P_(i j)(t))$, which scales as $O(T^3 dot n^2)$ when the edge probabilities are bounded away from 0 and 1.

*Higher-degree terms and the spectral gap.*
For $a >= 1$, the forcing $2 lambda_iota^(a+1)$ is small for eigenvalues near the spectral gap: $2 delta^(a+1)$ versus $2 lambda_1^(a+1)$.
This means the sensitivity $partial lambda_d \/ partial theta_a$ is small relative to $partial lambda_1 \/ partial theta_a$, so the Fisher information for higher-degree parameters $theta_a$ receives proportionally less signal from the small-eigenvalue directions.
Concretely, the contribution of the $lambda_d$ direction to $cal(I)(theta)_(a b)$ scales as $delta^(a+b+2)$, while the $lambda_1$ direction contributes $lambda_1^(a+b+2)$.

== General dynamics and the Lyapunov amplification

The polynomial case is clean because eigenvectors are fixed, confining all sensitivity to the diagonal of the eigenbasis.
For general symmetric dynamics $dot(X) = M(P) X$ with $M$ symmetric but not a polynomial in $P$, eigenvectors rotate and the sensitivity $partial P \/ partial theta$ has both diagonal and off-diagonal components in the eigenbasis.

In this setting, the Lyapunov equation $dot(P) = M P + P M$ must be inverted to recover $M$ from $dot(P)$.
In the eigenbasis of $P$, the inversion acts componentwise: $M_(iota gamma) = dot(P)_(iota gamma) \/ (lambda_iota + lambda_gamma)$.
The factor $1 \/ (lambda_iota + lambda_gamma)$ amplifies noise in the off-diagonal components, and is the same factor that appears in the connection 1-form (@sec:fiber-bundle) and governs curvature.
This yields a direct correspondence: directions in parameter space that are hard geometrically (small $lambda_iota + lambda_gamma$ produces large curvature and large connection coefficients) are also hard statistically (small $lambda_iota + lambda_gamma$ amplifies estimation noise).

For Laplacian dynamics, the situation is compounded by holonomy.
The eigenvector rotation contributes additional degrees of freedom to the sensitivity, but this information is entangled with gauge degrees of freedom that the holonomy obstruction (@prop:laplacian-holonomy) prevents from being resolved locally.

== The statistical-geometric duality

Combining the geometric and statistical pictures yields a unified obstruction.
We first state the result for polynomial dynamics, then explain the broader duality.

#proposition(title: "Cramér-Rao Bound for Polynomial Dynamics")[
  For polynomial dynamics of degree $K$ with parameter $theta = (theta_0, ..., theta_K) in RR^(k)$ ($k = K + 1$) on an RDPG with $n$ nodes, $d$ latent dimensions, and spectral gap $delta = lambda_d$, any unbiased estimator $hat(theta)$ satisfies:
  $ EE[||hat(theta) - theta||^2] >= "tr"(cal(I)(theta)^(-1)) $
  The Fisher information matrix $cal(I)(theta)$ has entries:
  $ cal(I)(theta)_(a b) = sum_(t=1)^T sum_(iota, gamma = 1)^d (partial lambda_iota)/(partial theta_a) (partial lambda_gamma)/(partial theta_b) dot.c C_(iota gamma)(t) $
  where $C_(iota gamma)(t) = sum_(i < j) U_(i iota) U_(j iota) U_(i gamma) U_(j gamma) \/ [P_(i j)(t)(1 - P_(i j)(t))]$.
  The diagonal entries $C_(iota iota)(t) = cal(W)_iota (t)$ measure the information content of eigenvalue direction $iota$ at time $t$; the off-diagonal entries $C_(iota gamma)(t)$ for $iota != gamma$ capture cross-eigenvalue correlations.
  A baseline lower bound on each parameter's variance follows from the matrix inequality $(cal(I)^(-1))_(a a) >= 1 \/ cal(I)_(a a)$:
  $ "Var"(hat(theta)_a) >= 1 / cal(I)_(a a) = 1 / (sum_t sum_iota ((partial lambda_iota) / (partial theta_a))^2 cal(W)_iota (t) + "cross terms") $
  Non-zero cross-eigenvalue terms arising from structured graphs (where Fisher weights correlate with eigenvector components) can only _increase_ the true Cramér-Rao bound above this baseline.
] <prop:cramer-rao>

#proof[
  By @eq:poly-sensitivity, the sensitivity of $P_(i j)$ decomposes as $partial P_(i j) \/ partial theta_a = sum_iota U_(i iota) U_(j iota) (partial lambda_iota \/ partial theta_a)$.
  Substituting into @eq:fisher:
  $ cal(I)(theta)_(a b) = sum_t sum_(i < j) [sum_iota U_(i iota) U_(j iota) (partial lambda_iota)/(partial theta_a)] [sum_gamma U_(i gamma) U_(j gamma) (partial lambda_gamma)/(partial theta_b)] dot 1/(P_(i j)(1 - P_(i j))) $
  Expanding the product over $iota, gamma$ yields:
  $ cal(I)(theta)_(a b) = sum_t sum_(iota, gamma) (partial lambda_iota)/(partial theta_a) (partial lambda_gamma)/(partial theta_b) underbrace(sum_(i<j) (U_(i iota) U_(j iota) U_(i gamma) U_(j gamma))/(P_(i j)(1 - P_(i j))), C_(iota gamma)(t)) $

  *On the cross-eigenvalue terms.*
  Column orthogonality of $U$ gives $sum_i U_(i iota) U_(i gamma) = delta_(iota gamma)$, but the Fisher weights $1\/[P_(i j)(1-P_(i j))]$ are not uniform.
  In structured RDPGs (e.g., stochastic block models with equal-sized blocks), these weights are constant within blocks and the blocks are defined by the eigenvectors themselves, so the weights correlate with eigenvector components and the cross terms $C_(iota gamma)$ for $iota != gamma$ are generically nonzero.
  The Fisher information matrix is therefore *not* diagonal in the eigenvalue basis for structured graphs.

  However, the spectral gap scaling is unaffected: the eigenvalue sensitivity $partial lambda_iota \/ partial theta_a$ is determined by the scalar ODE $dot(lambda)_iota = 2 sum_k theta_k lambda_iota^(k+1)$.
  The forcing term $2 lambda_iota^(a+1)$ scales as $delta^(a+1)$ for $iota = d$, so the $lambda_d$ direction contributes $O(delta^(a+b+2))$ to $cal(I)_(a b)$ regardless of whether this contribution arises from diagonal or cross terms.
  When $delta -> 0$, the Fisher information matrix becomes ill-conditioned: the rows and columns involving high powers of $delta$ shrink, making $"tr"(cal(I)^(-1))$ diverge.

  The baseline bound $(cal(I)^(-1))_(a a) >= 1\/cal(I)_(a a)$ follows from the Schur complement: for any positive definite matrix, the diagonal of the inverse is at least the reciprocal of the diagonal.
  This bound is tight when $cal(I)$ is diagonal and loose when off-diagonal terms are large (i.e., when parameter correlations from structured graphs make estimation strictly harder than the diagonal analysis suggests).
]

The bound reveals three scaling regimes.
Each snapshot provides $O(n^2)$ conditionally independent Bernoulli observations, so the Fisher information grows as $n^2$.
Information accumulates linearly in $T$, assuming the dynamics produce sufficient variation in $P$ across time.
The spectral gap $delta$ controls estimation difficulty through the eigenvalue sensitivities: the parameter $theta_a$ receives signal proportional to $delta^(a+1)$ from the smallest eigenvalue direction.
For the constant term $theta_0$ (linear dynamics), the $delta$-direction contributes $O(delta^2)$ to the Fisher information; for the highest-degree term $theta_K$, the contribution degrades to $O(delta^(2K+2))$.
These scalings hold for both the diagonal entries and the full matrix; the cross-eigenvalue terms share the same $delta$-dependence.

For linear dynamics (single scalar parameter $alpha_0$), the Fisher information is exact with no cross-term ambiguity:

#corollary(title: "Linear Dynamics: Explicit Fisher Information")[
  For linear dynamics $dot(X) = alpha_0 X$ with $T$ snapshots at times $t_1, ..., t_T$, the Fisher information is:
  $ cal(I)(alpha_0) = 4 sum_(t=1)^T t^2 sum_(i < j) (P_(i j)(t)) / (1 - P_(i j)(t)) $
  This scales as $Theta(T^3 dot n^2)$ when edge probabilities remain bounded away from 0 and 1 throughout the trajectory, giving a Cramér-Rao lower bound of $Omega(1\/(T^3 n^2))$.
] <cor:linear-fisher>

#proof[
  See @app:deferred-proofs.
]

The cubic dependence on $T$ (rather than linear) reflects the fact that the sensitivity $partial P_(i j) \/ partial alpha_0 = 2 t P_(i j)$ grows with time: later snapshots are more informative because the perturbation has had longer to propagate.
This is a general feature of dynamics estimation, in contrast to i.i.d. settings where information accumulates linearly.

The scaling $Theta(T^3 dot n^2)$ holds only while the trajectory remains in the interior of the probability space.
Linear dynamics produce exponential growth or decay of eigenvalues: $lambda_iota (t) = lambda_iota (0) e^(2 alpha_0 t)$.
For $alpha_0 > 0$, the probabilities $P_(i j)$ grow toward 1, and the Fisher weights $P_(i j)\/(1 - P_(i j))$ diverge; but the model becomes invalid as probabilities saturate, so the growth of $cal(I)$ eventually halts.
For $alpha_0 < 0$, probabilities decay toward 0, the sensitivities $partial P_(i j) \/ partial alpha_0 = 2 t P_(i j)(t)$ shrink exponentially, and the Fisher information per snapshot saturates.
In both cases, the $T^3$ scaling describes a *short-time regime* before boundary effects dominate; the effective time horizon is $T_("eff") tilde 1\/|alpha_0|$.

*The duality.*
The statistical-geometric correspondence is sharpest for general (non-polynomial) symmetric dynamics, where the full Lyapunov structure appears.
The connection 1-form involves factors $1 \/ (lambda_iota + lambda_gamma)$ controlling gauge sensitivity (@sec:fiber-bundle); the vertical bracket norm in the sectional curvature involves the same factors, with $||[overline(xi)_1, overline(xi)_2]^cal(V)||^2 tilde sum_(iota < gamma) lambda_iota lambda_gamma \/ (lambda_iota + lambda_gamma) dot [(U^top [M_1, M_2] U)_(iota gamma)]^2$ (@prop:curvature-criterion), which diverges as $lambda_d -> 0$ for fixed commutator; and the Fisher information for estimating $M_(iota gamma)$ from the Lyapunov equation $dot(P)_(iota gamma) = (lambda_iota + lambda_gamma) M_(iota gamma)$ involves the same factor $(lambda_iota + lambda_gamma)^2$ that appears in the curvature denominator.

An important qualification: this duality applies to *full operator recovery* (estimating the action of $M$ on all eigenvector directions, including the weak subspace near $lambda_d$).
For *parsimonious parameterizations* (e.g., a single scalar parameter $alpha_0$ shared across all eigenvalue directions), estimation can proceed primarily from the dominant modes: the contribution of $lambda_1$ alone may suffice to identify $alpha_0$, bypassing the ill-conditioned $lambda_d$ direction entirely.
The geometric-statistical coupling bites when the parameter of interest specifically governs the weak subspace, or when the full operator $M$ must be recovered.

With this qualification, the correspondence remains tight: for any component of the dynamics that depends on the spectral gap, the same $delta$ controls both curvature and Fisher information.
Networks near rank-deficiency ($delta -> 0$, with $lambda_(d-1)$ also small) are simultaneously harder to align (curvature $tilde 1\/(lambda_(d-1) + delta)$), harder to interpolate (injectivity radius $tilde sqrt(delta)$), and harder to estimate statistically (Fisher information for the weakest direction $tilde delta^2$).

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
  ASE is asymptotically unbiased with per-vertex error $O(1\/sqrt(n))$ @athreya2016limit, and the spectral estimator of SBM block probabilities achieves asymptotic efficiency @tang2022efficient; for individual latent positions, a one-step procedure achieves local efficiency (matching the oracle MLE covariance, up to orthogonal transformation) @xie2021efficient.
  The minimax rate for latent position estimation under Frobenius loss is $Theta(d\/n)$ per vertex @xie2020optimal, with two-to-infinity rates depending on the spectral gap @agterberg2023minimax.

  By contrast, @prop:cramer-rao concerns estimation of the _dynamics parameters_ $theta$ from a _time series_ of graphs, a setting for which no prior efficiency theory exists.
  The estimation target is qualitatively different: not the $O(n d)$ latent position coordinates, but the $O(1)$ parameters governing their temporal evolution.
  The eigenvalue-direction decomposition of the Fisher information in @prop:cramer-rao, with weights $cal(W)_iota (t)$ involving eigenvector loadings, has no counterpart in the static theory.
  Extending the existing minimax framework to parametric temporal models is an open problem that our Fisher information structure could help resolve.

  The "any unbiased estimator" qualification deserves comment.
  The CRB is a bound on the Bernoulli likelihood, which is well-defined for any $n$, so the bound itself does not require asymptotic arguments.
  However, whether useful estimators of $theta$ are unbiased at finite $n$ is a separate question.
  Any estimator that first recovers $P$ spectrally and then fits $theta$ inherits the finite-sample bias of eigendecomposition: the nonlinearity of the spectral map introduces bias, but this is $o(1\/sqrt(n))$ @cape2019two and does not affect the asymptotic bound.
  Near rank-deficiency ($delta -> 0$), the situation is more severe: the ASE collapses the weak eigenvalue direction, and the resulting bias in $hat(lambda)_d$ can dominate the signal $lambda_d$ itself.
  In this regime, the *mean squared error* for parameters that depend on the $lambda_d$ direction is likely dominated by bias rather than variance, and the CRB (which bounds variance alone) understates the true difficulty.
  For finite-sample inference, the biased Cramér-Rao variant $"Var"(hat(theta)) >= (1 + b'(theta))^2 \/ cal(I)(theta)$ provides the appropriate generalization, but characterizing the bias $b(theta)$ as a function of $delta$ remains open.
]



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

#definition(title: "Joint Alignment and Learning Problem")[
  Given ASE embeddings ${hat(X)^((t))}_(t=0)^T$ and a dynamics family $cal(F)$, find gauge corrections ${Q_t in O(d)}$ and $f in cal(F)$ minimizing:
  $ cal(L)({Q_t}, f) = sum_(t=0)^(T-1) ||hat(X)^((t+1)) Q_(t+1) - hat(X)^((t)) Q_t - delta t f(hat(X)^((t)) Q_t)||_F^2 $
]

The objective measures how well the aligned discrete trajectory ${hat(X)^((t)) Q_t}$ is explained by the dynamics $f$: if the gauges are correct and $f$ captures the true dynamics, each step's displacement should match the predicted velocity.




== The identifiability principle

The theoretical case for structure-constrained alignment rests on a clean separation between gauge artifacts and horizontal dynamics.

#theorem(title: "Gauge Velocity Contamination")[
  Let $X(t)$ follow true dynamics $dot(X) = N X$ with $N = N^top$.
  Let $S(t) in O(d)$ be a time-varying (mis)gauge and define the gauged trajectory $tilde(X)(t) = X(t) S(t)$.
  Then
  $ dot(tilde(X)) = N tilde(X) + tilde(X) Omega $
  where $Omega(t) = S(t)^top dot(S)(t) in so(d)$ is skew-symmetric.

  Moreover, if $tilde(X)(t)$ has full column rank for all $t$ in an interval, then the following are equivalent on that interval:
  + there exists a symmetric matrix function $tilde(N)(t) = tilde(N)(t)^top$ such that $dot(tilde(X)) = tilde(N)(t) tilde(X)$,
  + $Omega(t) = 0$ (i.e., $S(t)$ is constant in time).
] <thm:gauge-contamination>

#proof[
  By the product rule,
  $ dot(tilde(X)) = dot(X) S + X dot(S) = N X S + X dot(S). $
  Using $X = tilde(X) S^top$ (since $S in O(d)$), this becomes
  $ dot(tilde(X)) = N tilde(X) + tilde(X) (S^top dot(S)) = N tilde(X) + tilde(X) Omega, $
  where $Omega = S^top dot(S) in so(d)$ because differentiating $S^top S = I$ gives $dot(S)^top S + S^top dot(S) = 0$.

  Now suppose $dot(tilde(X)) = tilde(N) tilde(X)$ for some symmetric (possibly time-varying) $tilde(N)$ and that $tilde(X)$ has full column rank, so $G = tilde(X)^top tilde(X)$ is positive definite.
  Left-multiplying by $tilde(X)^top$ gives
  $ tilde(X)^top dot(tilde(X)) = tilde(X)^top tilde(N) tilde(X), $
  and the right-hand side is symmetric because $tilde(N)$ is symmetric.

  On the other hand, from the decomposition above,
  $ tilde(X)^top dot(tilde(X)) = tilde(X)^top N tilde(X) + G Omega. $
  The term $tilde(X)^top N tilde(X)$ is symmetric, hence $G Omega$ must be symmetric as well.
  But if $G$ is symmetric positive definite and $Omega$ is skew-symmetric, then $G Omega$ is symmetric iff $Omega = 0$ (equivalently, $G Omega + Omega G = 0$ implies $Omega = 0$).
  Therefore $Omega = 0$, which means $S$ is constant in time.
  The converse direction is immediate: if $Omega = 0$ then $dot(tilde(X)) = N tilde(X)$ is already of the required form with symmetric generator $N$.
]

The mechanism is elegant: random ASE gauge errors $R^((t))$ introduce skew-symmetric contamination that *cannot be absorbed* by symmetric dynamics.
Requiring symmetric dynamics implicitly selects gauges where the contamination vanishes.

The reader may notice an affinity with @prop:horizontal, which states that $dot(X)$ is horizontal if and only if $X^top dot(X)$ is symmetric.
The two results use the same algebraic fact (a positive definite matrix times a nonzero skew-symmetric matrix is never symmetric) but answer different questions.
@prop:horizontal characterizes directions at a single point: given a velocity $dot(X)$, does it have a gauge component?
@thm:gauge-contamination characterizes trajectories over time: given an entire trajectory $tilde(X)(t)$ observed in the wrong gauge, can any symmetric dynamics explain it?
The former is a pointwise decomposition; the latter is an identifiability statement about the dynamics-gauge coupling along a path.

For specific families, this yields concrete algorithms.
For linear symmetric dynamics, one obtains an alternating scheme: fix gauges and solve a least-squares problem with symmetry constraint for the dynamics matrix $M$; fix $M$ and solve orthogonal Procrustes for each gauge.
For polynomial dynamics $dot(X) = (sum_k alpha_k P^k) X$, the gauge invariance of $P = hat(X) hat(X)^top$ simplifies the dynamics step to linear regression in the $alpha_k$ coefficients.
Both have closed-form updates per step.


== The downstream pipeline: from trajectories to equations

Suppose, optimistically, that the trajectory recovery problem were solved, that is, some method (structure-constrained alignment, a future $P$-level estimator, or domain-specific prior information) produced gauge-consistent estimates $tilde(X)^((t))$ of the latent positions, up to noise.
When multiple independent network samples $A_1^((t)), ..., A_m^((t))$ are available at each time $t$, averaging $macron(A)^((t)) = 1/m sum_i A_i^((t))$ before embedding reduces per-entry variance by a factor of $m$ and (heuristically, in standard dense regimes) reduces per-vertex embedding error by about $sqrt(m)$, potentially making trajectory recovery feasible even when individual snapshots are noisy.

Given such a recovered trajectory, the remaining steps are well-established.
*Universal Differential Equations* (UDEs) @rackauckas2020universal provide a framework for learning dynamics that combine known mechanistic structure with neural network flexibility:
$ dot(X) = g(f_("known")(X, phi), f_("NN")(X, theta)) $
For RDPG dynamics, the known structure comes from the families in @sec:dynamics-families. For example, the polynomial architecture $dot(X) = (sum_k alpha_k (theta, X) P^k) X$ ensures horizontality by construction while allowing the coefficients to be learned.
Gradients flow through the ODE solver via adjoint sensitivity methods, and the gauge-consistent architecture constrains the hypothesis space.

Once a UDE is trained, *symbolic regression* @symbolicregression can extract interpretable closed-form equations.
By sampling state-velocity pairs $(X, f_theta(X))$ from the trained model and searching for symbolic expressions of the form $N(P) X$, one can recover explicit dynamics equations completing the path from observed adjacency matrices to differential equations.
A subtlety: symbolic regression on latent variables $X$ is gauge-dependent.
Unless the UDE output is pre-aligned to a canonical frame (e.g., the eigenbasis of $P$), the regression may fail to find sparse coefficients because $X$ in the UDE gauge is a rotated version of the "natural" coordinates.
Working with *gauge-invariant features* (eigenvalues of $P$, or the $P$-dynamics $dot(P) = N P + P N$ directly) avoids this issue, and is natural for polynomial families where the dynamics separate by eigenvalue.

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
The identifiability argument needs $cal(F)$ to be restrictive enough that wrong gauges force you outside the family.
But in finite samples and discrete time, surprisingly flexible behavior can sneak in.

One mechanism is numerical rather than philosophical: even for polynomial families, the regression problem for $(alpha_0, ..., alpha_K)$ can become ill-conditioned.
The matrices $I, P, P^2, ...$ increasingly align with the top eigenspace of $P$ (power-iteration style), so the design becomes Vandermonde-like in the eigenvalues and can be badly conditioned for $K >= 2$.
In that regime, modest gauge-induced artifacts (or plain ASE noise) can be fitted by large, canceling coefficients in higher-order terms, making it hard to distinguish "true dynamics" from "dynamics that explain the mis-gauged data".

A second mechanism is geometric: for small $d$, the gauge degrees of freedom are low-dimensional (e.g., $so(2)$ is one-dimensional), so a slowly varying mis-gauge can produce a coherent-looking drift that is not obviously nonsense at the discrete-time resolution.
The moral is the same: making $cal(F)$ more restrictive risks excluding the truth; making it more expressive risks letting gauge artifacts (and noise) slip through.

*Holonomy and global consistency.*
Even if local alignment succeeds (each consecutive pair of frames is well-aligned), holonomy (@sec:fiber-bundle) means gauge drift can accumulate over long trajectories.
For dynamics whose *base-space* path $P(t)$ forms (or nearly forms) loops in $cal(B)$, a globally consistent choice of gauge along the entire loop may be impossible: the horizontal lift of a closed curve need not close.

This is not something you fix with more data; it is a geometric obstruction of the model class.
What more data *can* do is reduce the statistical wobble on top of it. That is important, but it is a different issue.

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


== Anchor-based alignment: a tractable special case <sec:anchor-alignment>

The difficulties above are genuine in the general case.
However, there is a natural special case in which the gauge problem admits a clean solution: when a subset of nodes is known (or assumed) to be stationary.

*The anchor principle.*
Suppose a subset $S subset {1, ..., n}$ of nodes has $dot(x)_i = 0$ for $i in S$. These are "anchor" points whose latent positions do not move.
At each time $t$, ASE produces $hat(X)^((t)) = X^((t)) R^((t)) + E^((t))$ with a gauge factor $R^((t)) in O(d)$ (coming from the eigendecomposition choice) and estimation noise $E^((t))$.
For anchor nodes $i in S$, we have $x_i^((t)) = x_i^((0))$ for all $t$, so the anchor rows satisfy
$ hat(X)_S^((t)) = X_S R^((t)) + E_S^((t)) $
where $X_S in RR^(|S| times d)$ is constant.

Align the anchor rows to a reference frame (say $t = 0$) via Procrustes:
$ hat(Q)^((t)) = arg min_(Q in O(d)) ||hat(X)_S^((t)) Q - hat(X)_S^((0))||_F. $
In the noiseless limit this recovers the relative gauge $(R^((t)))^(-1) R^((0))$ directly, without using any dynamics model and without ever propagating a gauge sequentially.
Applying $hat(Q)^((t))$ to the full embedding $hat(X)^((t))$ then aligns all nodes to the $t=0$ gauge.

*Why this works.*
The anchor nodes provide a fixed reference frame that "pins" the gauge at each time step.
The alignment is *global* (all frames aligned to $t = 0$, not sequentially) so there is no error accumulation.
The method requires no knowledge of the dynamics family $cal(F)$: it works for polynomial, Laplacian, or any other dynamics, as long as the anchors are truly stationary.
Holonomy is irrelevant because we never attempt to propagate a gauge along the trajectory; each frame is independently aligned to the fixed reference.

*Conditions and limitations.*
The anchor principle requires:
+ *Well-conditioned anchors:* the anchor position matrix $X_S in RR^(|S| times d)$ must have full column rank for the Procrustes problem to be determined, and should be well-conditioned ($sigma_d (X_S)$ not too small) for robustness to noise. Counting anchors alone is insufficient: $|S| >> d$ anchors that are nearly collinear or clustered around a single point leave the rotation poorly constrained in orthogonal directions. This is the same stability story as in the classical orthogonal Procrustes problem: the rotation estimate becomes sensitive when the reference configuration is nearly rank-deficient. (We do not reproduce a full perturbation bound here; the key point for our purposes is the conditioning dependence. See, e.g., standard treatments of Procrustes perturbation theory, or interpret it directly through the SVD formula for Procrustes and Weyl/Wedin-type eigen/singular vector perturbation bounds.)
+ *Known anchor identity:* we must know which nodes are stationary. In practice this could come from domain knowledge (e.g., established species in an ecological network, institutional nodes in a social network) or from a preliminary analysis identifying nodes with low temporal variance.
+ *Approximately stationary anchors:* if anchor nodes drift slowly (with velocity $||dot(x)_i|| = epsilon$ for $i in S$), the alignment incurs a systematic bias that grows with the time horizon. A crude but useful rule of thumb is that anchor drift should remain small compared to the embedding noise accumulated over the window of interest; otherwise, the "reference frame" itself becomes moving.

The third condition explains a phenomenon we observed in preliminary experiments: networks with a large block of slowly-moving nodes could be aligned successfully by naive Procrustes, while networks where all nodes moved at comparable rates could not.
The slow nodes were acting as _de facto_ anchors, stabilizing the gauge without our explicitly recognizing it.

*When anchors are realistic.*
Several application domains naturally feature nodes with heterogeneous dynamics rates.
In ecological food webs, basal species (primary producers) often have stable trophic positions while higher-level consumers undergo rapid changes.
In social networks, institutional actors (organizations, permanent positions) may persist while individuals fluctuate.
In neural connectomes, structural hub regions may be stable on timescales over which peripheral connections rewire.
More generally, any system with a separation of timescales is a natural candidate: it has a slowly evolving "backbone" and a rapidly evolving periphery.

The anchor approach does not solve the general gauge problem.
It replaces a hard geometric-statistical problem with a domain knowledge requirement: identifying stationary nodes.
But when such knowledge is available, it provides a clean and computationally trivial path to gauge-consistent trajectories, enabling the downstream UDE pipeline of @sec:constructive.


== Numerical illustration <sec:numerics>

We illustrate the theory with two controlled experiments on synthetic RDPG data.
The first demonstrates anchor-based alignment on gauge-equivariant polynomial dynamics, where alignment quality can be assessed independently of dynamics recovery.
The second demonstrates the full downstream pipeline: UDE training and symbolic regression; to do this more visibily, we move to a non-gauge-equivariant dynamics, where alignment quality directly impacts dynamics recovery.

=== Experiment 1: Anchor-based alignment <sec:numerics-anchor>

*Setup.*
We generate an RDPG with $n = 200$ nodes in $d = 2$ latent dimensions.
The initial positions $X(0)$ are drawn uniformly from $B_+^2 = {x in RR^2 : x_1, x_2 >= 0, ||x|| <= 1}$, with a designated anchor set $S$ of size $n_a$.
The non-anchor nodes evolve under polynomial dynamics $dot(X) = (alpha_0 I + alpha_1 P) X$ with $(alpha_0, alpha_1) = (-0.3, 0.003)$, while anchor nodes remain fixed: $dot(x)_i = 0$ for $i in S$.
We integrate the ODE to produce a trajectory $X(t)$ at $T$ equally-spaced times, generate $m = 3$ independent adjacency matrices per time step $A_k^((t)) tilde "Bernoulli"(X(t) X(t)^top)$ (and average them to reduce noise), compute ASE embeddings $hat(X)^((t))$, and compare anchor-based alignment to sequential Procrustes.

Unless otherwise stated, we use $T = 50$ steps with $delta t = 0.05$ (total time $2.45$), and report means and standard deviations over 20 Monte Carlo repetitions.

*Metrics.*
We measure alignment quality by the mean Procrustes error:
$ "err"(t) = 1/n ||hat(X)^((t)) hat(Q)^((t)) - X^((t)) Q^*||_F $
where $Q^*$ is the best global alignment to the true trajectory (accounting for the residual gauge at $t = 0$).
We sweep over four experimental conditions: anchor count $n_a$, trajectory length $T$, anchor drift rate $epsilon$, and initial embedding norm scale (controlling signal-to-noise ratio).

#figure(
  image("plots/anchor-main-results.png", width: 95%),
  caption: [
    Anchor-based alignment experiment ($n = 200$, $d = 2$, polynomial dynamics $dot(X) = (alpha_0 I + alpha_1 P) X$ with $m = 3$ Bernoulli samples per time step).
    *(a)* Alignment error vs.\ number of anchor nodes: below $n_a = d = 2$ (dashed vertical line) Procrustes is underdetermined and alignment fails; with sufficient anchors, error stabilizes at the ASE noise floor.
    *(b)* Error accumulation over trajectory length: anchor-based alignment (blue) remains bounded while sequential Procrustes (coral, dashed) grows with $T$, consistent with $O(sqrt(T))$ drift accumulation.
    *(c)* Effect of anchor drift rate $epsilon$: with larger $epsilon$, systematic bias from drifting anchors becomes visible.
    *(d)* Per-timestep alignment error for the $T = 200$ trajectory: the anchor-based error (blue) remains approximately flat, while sequential Procrustes (coral) accumulates drift, with the gap widening toward later time steps.
    Shaded bands show $plus.minus 1$ standard deviation across 20 Monte Carlo repetitions.
  ],
) <fig:anchor-main>

#figure(
  image("plots/anchor-spectral-portrait.png", width: 95%),
  caption: [
    *(a)* Alignment error as a function of the initial embedding norm scale (proxy for signal strength): smaller norms yield weaker signals and less informative Bernoulli observations, increasing ASE noise and degrading alignment for both methods. Note that uniform scaling preserves the condition number $sigma_1 \/ sigma_2$ (constant at $approx 2.2$); this experiment varies signal _magnitude_ rather than the spectral gap ratio.
    *(b)* Phase portrait with $n_a = 15$ anchor nodes (red diamonds, stationary) and non-anchor nodes (gray trajectories evolving under polynomial dynamics) in $B_+^2$.
  ],
) <fig:anchor-spectral>

*Results.*
The experiment matches the basic theoretical story, and also shows where the "easy" intuitions break.

- *Anchor count.* With too few anchors the Procrustes problem is underdetermined: for $d=2$, $n_a=1$ is a worst case and alignment fails, while $n_a=2$ is a knife-edge case with high variance (sometimes it works, sometimes it doesn't, depending on anchor geometry). Once $n_a$ is moderately larger (empirically $n_a >= 5$ here), the anchor-based error stabilizes near the ASE noise floor (about $0.005$ in our setting), consistent with the idea that the anchors now robustly span the latent plane.

- *Trajectory length and drift accumulation.* For short trajectories ($T <= 50$), sequential Procrustes is marginally better (about $2%$ in our sweep), because it uses all $n=200$ nodes at each step while anchor-based Procrustes uses only the $n_a = 15$ anchors. But sequential alignment accumulates drift: the crossover occurs around $T approx 100$, and by $T = 200$ sequential Procrustes is about $13%$ worse (ratio $approx 1.13$), consistent with the expected $O(sqrt(T))$ growth of accumulated rotation error.

- *Drifting anchors.* Over $T=50$ steps (total time $2.45$), modest anchor drift rates have little effect on the mean alignment error (because the metric averages over all 200 nodes and only 15 are anchors). The effect is more visible in downstream variability: at $epsilon = 0.1$ the variability of recovered parameters increases noticeably, consistent with systematic bias entering through a slowly moving reference frame.

*Dynamics recovery is (mostly) gauge-free here.*
A useful sanity check is that, for polynomial dynamics, the coefficients $(alpha_0, alpha_1)$ are estimable directly from the gauge-invariant trajectory $P(t) = X(t) X(t)^top$.
In our sweep, the standard errors of $hat(alpha)_0$ are essentially unchanged across anchor counts (around $0.008$), confirming that this particular parameter recovery problem does not fundamentally depend on aligning $X$.

However, the *point estimates* do shift with the anchor count (e.g., $hat(alpha)_0$ moves from about $-0.282$ at $n_a=0$ to about $-0.250$ at $n_a=15$), because freezing a large subset of nodes changes the actual $P(t)$ trajectory. This is a data-generating change, not a gauge artifact.
This motivates the next question: when the dynamics depend on $X$-space coordinates (i.e., are not gauge-equivariant), does alignment quality directly control dynamics recovery?
Experiment 2 is designed to make that dependence unavoidable.


=== Experiment 2: UDE pipeline with non-gauge-equivariant dynamics <sec:numerics-ude>

*Motivation.*
The polynomial dynamics in Experiment 1 are gauge-equivariant: $P = X X^top$ is rotation-invariant, so the dynamics parameters live in $P$-space and can be recovered without alignment.
To exercise the full trajectory-recovery $\to$ UDE pipeline, we now design dynamics that depend on $X$-space coordinates directly, so that alignment quality becomes a genuine bottleneck.

*Setup.*
We generate an RDPG with $n = 200$ nodes in $d = 3$ latent dimensions, with $n_a = 100$ anchor nodes organized into $K_c = 3$ communities near the vertices of $B_+^3$ (at positions $(0.7, 0.2, 0.2)$, $(0.2, 0.7, 0.2)$, $(0.2, 0.2, 0.7)$ plus Gaussian noise).
The 100 non-anchor nodes evolve under damped spiral dynamics around their community centroids $mu_k$:
$ dot(x)_i = (-gamma + beta ||x_i - mu_k||^2)(x_i - mu_k) + omega J(x_i - mu_k) $
where $J = 1/sqrt(3) mat(0, -1, 1; 1, 0, -1; -1, 1, 0)$ is the rotation generator around the $(1,1,1)/sqrt(3)$ axis, and $(gamma, beta, omega) = (0.3, -0.5, 1.0)$.
These dynamics are _not_ gauge-equivariant: the centroids $mu_k$ and rotation axis live in $X$-space coordinates, so a misaligned trajectory $tilde(X)(t)$ produces different offsets $tilde(x)_i - mu_k != x_i - mu_k$, corrupting the dynamics structure.

We observe $m = 10$ independent adjacency matrices per time step over $T = 50$ steps at $delta t = 0.1$, compute ASE embeddings, and apply three alignment conditions: anchor-based Procrustes, sequential Procrustes, and no alignment (raw ASE with random per-frame rotations).

*UDE architecture.*
Following the framework of @sec:constructive, we decompose the dynamics into known and unknown components:
$ dot(x)_i = underbrace(-hat(gamma) (x_i - mu_k), f_"known") + underbrace(f_theta (x_i - mu_k), f_"NN") $
where $hat(gamma)$ is a learnable scalar parameter and $f_theta: RR^3 -> RR^3$ is a small neural network (architecture: $3 -> 16 -> 16 -> 3$ with $tanh$ activations, $approx 390$ parameters total).
The known part encodes the structural assumption that non-anchor nodes are attracted toward their community centroids; the unknown part captures whatever additional dynamics exist.
L2 regularization on the network weights ($lambda = 10^(-3)$) discourages the network from absorbing the linear damping term, improving the identifiability of the known-unknown decomposition.
The UDE is trained by solving the neural ODE forward from the initial condition and minimizing the trajectory MSE via adjoint sensitivity methods.

*Primary metric: total dynamics MSE (the main story).*
Because the NN can absorb part of the linear damping term, the additive decomposition
$ f_"learned"(delta) = -hat(gamma) delta + f_theta(delta) $
has an inherent identifiability caveat: different $(hat(gamma), f_theta)$ pairs can yield essentially the same total vector field.
So we evaluate what we actually care about, namely the *total learned dynamics*, against the true dynamics
$ f_"true"(delta) = (-gamma + beta ||delta||^2) delta + omega J delta $
on a held-out cloud of random test inputs $delta in [-0.3, 0.3]^3$ (2000 samples in our implementation).
This metric is intentionally invariant to how the model splits responsibility between $hat(gamma)$ and the NN.

#figure(
  image("plots/ude-pipeline-results.png", width: 95%),
  caption: [
    UDE pipeline experiment ($n = 200$, $d = 3$, damped spiral dynamics with rotation around $(1,1,1)/sqrt(3)$, $m = 10$ Bernoulli samples per frame).
    *(a)* True trajectories (gray) and anchor-aligned ASE (blue) in $B_+^3$; anchor nodes shown as red diamonds.
    *(b)* Learned NN residual $f_theta(delta)$ vs.\ true residual $f_u(delta)$: anchor-aligned data (left) produces tight agreement along the diagonal; sequential Procrustes (center) and unaligned data (right) degrade progressively.
    *(c)* Symbolic regression Pareto fronts (loss vs.\ expression complexity): anchor-aligned data achieves $1$--$2$ orders of magnitude lower loss at each complexity level.
    *(d)* Total dynamics MSE (log scale) by alignment condition: anchor alignment achieves MSE $approx 6 times 10^(-4)$, sequential Procrustes $approx 8 times 10^(-3)$ ($13 times$ worse), and unaligned $approx 0.44$ ($approx 700 times$ worse).
    Error bars show $plus.minus 1$ standard deviation across 5 repetitions.
  ],
) <fig:ude-pipeline>

*Results.*
The total dynamics MSE separates the three alignment conditions extremely clearly (@fig:ude-pipeline\d), across 5 Monte Carlo repetitions:

- *Anchor-aligned:* mean MSE $approx 6 times 10^(-4)$ with standard deviation $approx 4 times 10^(-4)$, i.e., excellent recovery of the full vector field.
- *Sequential Procrustes:* mean MSE $approx 7.6 times 10^(-3)$ with standard deviation $approx 3 times 10^(-4)$, about $13x$ worse.
- *Unaligned:* mean MSE $approx 0.44$ with standard deviation $approx 0.35$, about $700x$ worse and highly variable.

The NN-residual MSE shows the same ordering even more starkly: anchor alignment yields NN residual MSE around $9 times 10^(-4)$, sequential around $7.3 times 10^(-3)$, and unaligned around $0.44$.
This is the mechanism in plain sight: when frames are coherently aligned, the NN learns the intended nonlinear+rotational residual; when frames are incoherently rotated, it mostly learns noise.

*Symbolic regression.*
Symbolic regression on the trained network's input-output pairs (@fig:ude-pipeline\c) achieves Pareto-optimal expressions with losses $1$--$2$ orders of magnitude lower under anchor alignment than under the other conditions at each complexity level.
The anchor-aligned expressions contain the expected structural terms (products of coordinates and cross-terms consistent with the rotation generator $J$).
As noted in @sec:constructive, symbolic regression requires working with gauge-invariant features or a canonically aligned frame; the anchor alignment provides the latter.

*The identifiability caveat (and why it doesn't undercut the result).*
The learned damping parameter $hat(gamma)$ is not particularly stable across repetitions (anchor-aligned mean $approx 0.255$ with std $approx 0.081$ versus true $gamma=0.3$), because the NN can still absorb part of the linear term even with L2 regularization.
This is a real limitation of the additive UDE decomposition, not an alignment issue.
The reason it does not undermine the main claim is that the *total dynamics* are nevertheless recovered accurately under anchor alignment, and that is exactly what the total-dynamics MSE measures.

*Takeaway.*
When dynamics are gauge-equivariant ($dot(X) = N(P) X$), alignment quality is irrelevant for parameter recovery (Experiment 1).
When dynamics depend on $X$-space coordinates, alignment quality directly controls the fidelity of the learned dynamics (Experiment 2).
Anchor-based alignment provides the gauge-consistent trajectories that the downstream UDE pipeline requires.


= Discussion <sec:discussion>

*What this paper buys you (and what it doesn't).*
The thesis of the paper is simple to state: treating an RDPG as a dynamical system is appealing, but there are geometric and statistical reasons it is genuinely hard.
The payoff is a vocabulary and a set of concrete "failure modes" that let you diagnose where the difficulty is coming from.

On the geometry side, the fiber bundle perspective (@sec:fiber-bundle) formalizes the gauge freedom as a principal bundle, and makes the connection/curvature/holonomy machinery explicit in the RDPG setting.
On the dynamics side, we identified a sharp holonomy dichotomy (@sec:holonomy-dynamics): polynomial dynamics act through commuting generators (trivial holonomy along trajectories), while Laplacian dynamics generically introduce non-commuting generators and hence nontrivial holonomy (full $"SO"(2)$ for $d=2$).
On the statistics side, the information-theoretic analysis (@sec:info-theoretic) shows a stubborn duality: the same spectral quantities that control curvature and injectivity also control Fisher information and ill-conditioning.

*Where the optimism lives.*
The identifiability principle (@thm:gauge-contamination) says that structure can, in principle, resolve gauge ambiguity: symmetric dynamics cannot absorb skew-symmetric gauge contamination.
And in a tractable special case with anchor nodes, alignment becomes straightforward (@sec:anchor-alignment), which makes the downstream UDE pipeline feasible in practice (as the numerical results illustrate).

*What remains open (the constructive gap).*
The discrete, finite-sample setting is where theory meets its limits.
Finite-sample spectral bias and noise interact with alignment; expressive dynamics families can fit artifacts; and holonomy can make global gauge consistency impossible over loops even with perfect local alignment.
Bridging this gap seems to be the central open problem. In particular, we want to characterize when structure-constrained alignment is stable, and when it is doomed.

*Concrete directions that seem worth the effort.*
1. *Work in $P$-space when you can.* The $P$-dynamics viewpoint (@sec:p-dynamics) bypasses gauge in principle, but we lack an estimation theory for recovering $dot(P)$ and inverting the Lyapunov structure from Bernoulli samples. The $1\/(lambda_iota + lambda_gamma)$ amplification suggests minimax rates will depend sharply on the spectral gap; pinning this down would tell us whether $P$-level inference is practically viable.
2. *Quantify holonomy, not just its existence.* Beyond generic nontriviality, we want trajectory-dependent "holonomy budgets": how much gauge drift should we expect over a cycle as a function of curvature and how close the trajectory gets to rank-deficiency?
3. *Achievability of the CRB.* For polynomial dynamics (trivial holonomy, explicit eigenvalue ODEs), a matching upper bound via likelihood-based estimators on the $P$-trajectory looks plausible. For Laplacian dynamics, holonomy suggests that achieving the CRB may require estimators that explicitly account for gauge transport.
4. *Sparsity changes everything.* In the sparse regime $P = rho_n X X^top$ with $rho_n -> 0$, injectivity shrinks and curvature and ASE error deteriorate. The combined effect on dynamics recovery is largely unexplored.

*Broader impact (and a reminder).*
Learning interpretable dynamics from network data could be powerful across neuroscience, ecology, and social systems.
But RDPG is a strong structural assumption, and learned dynamics should be treated as hypotheses to be stress-tested against domain knowledge and model misspecification.


= Conclusion

Treating an RDPG as a dynamical system is an attractive idea: it promises mechanistic explanations for *why* networks evolve, not just short-term prediction.
But it also comes with a very specific set of obstacles, and this paper's main contribution is to make those obstacles crisp enough to reason about.

We identified three core obstructions.
Gauge freedom makes a whole subspace of latent motion invisible (@thm:invisible) and forces any latent-space method to confront alignment.
Realizability constrains what probability-matrix dynamics are even possible at fixed latent dimension.
And the trajectory-recovery problem shows that, in the standard RDPG pipeline where we must estimate latent positions via ASE, naive time differencing measures gauge jitter rather than dynamics.

The geometric framework of @sec:fiber-bundle ties these together.
The connection 1-form formalizes "gauge velocity," and curvature/holonomy explain why local alignment can fail to globalize.
This yields a sharp dichotomy: polynomial dynamics act through commuting generators and have trivial holonomy along trajectories (@prop:poly-trivial-holonomy), while Laplacian dynamics generically induce nontrivial holonomy (@prop:laplacian-holonomy, with full $"SO"(2)$ proved for $d=2$), adding a genuine global obstruction.

We complemented the geometry with information-theoretic limits.
Cramér-Rao bounds show a statistical and geometric duality. The same spectral quantities that control curvature and injectivity also control Fisher information and ill-conditioning in dynamics estimation.

Finally, we showed where the optimism lives.
Structure can, in principle, constrain gauge: symmetric dynamics cannot absorb skew-symmetric gauge contamination (@thm:gauge-contamination).
And in a practical special case with stationary anchor nodes, the gauge can be pinned directly. This enables end-to-end dynamics learning (UDE + symbolic regression) for non-gauge-equivariant dynamics.

The central open problem is constructive: turning these identifiability and geometric insights into stable, finite-sample estimators.
Progress likely requires a combination of $P$-level inference (bypassing gauge when possible), quantitative holonomy estimates (to budget unavoidable drift), and estimation strategies that explicitly account for curvature rather than fighting it implicitly.


= Acknowledgments

// TODO

= Data and Code Availability

// TODO: code for numerical experiments


#bibliography("bibliography.bib", style: "ieee")


#pagebreak()

// APPENDICES

= Deferred Proofs <app:deferred-proofs>

*Proof of @prop:measure-zero (Measure Zero).*
Assume $cal(F)$ is a finite-dimensional family with $p$ parameters (e.g., polynomial dynamics with $p = K+1$ coefficients), and restrict attention to a parameter domain on which solutions exist on $[0,T]$ and depend smoothly on parameters and initial conditions (standard under local Lipschitz conditions in $X$ and smoothness in parameters).
For each $(f, X_0) in cal(F) times RR^(n times d)$, the Picard-Lindelöf theorem yields a unique solution $X(dot; f, X_0) in C^k([0,T], RR^(n times d))$.
The solution map $Phi: (f, X_0) |-> X(dot; f, X_0)$ is smooth, so $cal(M)_cal(F) = "im"(Phi)$ is the image of a smooth map from an $m$-dimensional manifold (with $m = p + n d$) into the path space $cal(X) = C^k([0,T], RR^(n times d))$.

Choose $m + 1$ continuous linear functionals $ell_1, ..., ell_(m+1)$ on $cal(X)$ (for example, evaluations of specific coordinates at distinct times) and define $L = (ell_1, ..., ell_(m+1)): cal(X) -> RR^(m+1)$.
Choose these functionals so that the pushforward $L_* mu$ is non-degenerate (equivalently, has full-rank covariance), hence absolutely continuous with respect to Lebesgue measure on $RR^(m+1)$.
Then $L circle.small Phi: RR^m -> RR^(m+1)$ is a smooth map between finite-dimensional spaces with $m < m+1$.
A standard consequence of Sard's theorem (equivalently, the area formula) is that the image of such a map has Lebesgue measure zero in $RR^(m+1)$.
Thus $L(cal(M)_cal(F)) subset.eq "im"(L circle.small Phi)$ also has Lebesgue measure zero, and therefore
$ mu(cal(M)_cal(F)) <= (L_* mu)(L(cal(M)_cal(F))) = 0$. $square$

*Proof of @cor:linear-fisher (Linear Dynamics: Explicit Fisher Information).*
For linear dynamics $dot(X) = alpha_0 X$, the solution is $X(t) = e^(alpha_0 t) X(0)$, giving $P(t) = e^(2 alpha_0 t) P(0)$.
The sensitivity is $partial P_(i j)(t) \/ partial alpha_0 = 2t e^(2 alpha_0 t) P_(i j)(0) = 2t P_(i j)(t)$.
Since there is a single scalar parameter, the Fisher information is:
$ cal(I)(alpha_0) = sum_(t=1)^T sum_(i < j) ((partial P_(i j)(t)) / (partial alpha_0))^2 dot 1 / (P_(i j)(t)(1 - P_(i j)(t))) = 4 sum_(t=1)^T t^2 sum_(i < j) (P_(i j)(t)) / (1 - P_(i j)(t)) $
For the scaling claim: when $P_(i j)(t) in [epsilon, 1 - epsilon]$ for some $epsilon > 0$, the inner sum $sum_(i < j) P_(i j) \/ (1 - P_(i j))$ is $Theta(n^2)$ (there are $binom(n, 2)$ terms, each bounded away from zero and infinity).
The outer sum $sum_(t=1)^T t^2 = T(T+1)(2T+1)\/6 = Theta(T^3)$.
Therefore $cal(I)(alpha_0) = Theta(T^3 dot n^2)$, giving a Cramér-Rao lower bound $"Var"(hat(alpha)_0) >= cal(I)^(-1) = Omega(1\/(T^3 n^2))$. $square$


= Vertical Projection onto the Fiber <app:vertical-projection>

We derive the formula for projecting a tangent vector $Z in T_X RR_*^(n times d)$ onto the vertical subspace $cal(V)_X = {X Omega : Omega in frak(s o)(d)}$.

The vertical component $Z^cal(V) = X Omega^*$ is defined by the orthogonality condition $Z - X Omega^* in cal(H)_X$ (the remainder is horizontal).
By @prop:horizontal, a vector $W$ at $X$ is horizontal if and only if $X^top W$ is symmetric.
Applying this to $W = Z - X Omega^*$:
$ X^top (Z - X Omega^*) = X^top Z - G Omega^* quad "must be symmetric" $
where $G = X^top X$ is positive definite.
Decomposing $X^top Z$ into symmetric and skew-symmetric parts, $X^top Z = "sym"(X^top Z) + "skew"(X^top Z)$, the condition becomes:
$ "skew"(G Omega^*) = "skew"(X^top Z) $

Since $Omega^*$ is skew-symmetric and $G$ is symmetric, the skew-symmetric part of $G Omega^*$ is:
$ "skew"(G Omega^*) = 1/2 (G Omega^* - (G Omega^*)^top) = 1/2 (G Omega^* + Omega^* G) $
where the last step uses $Omega^(*top) = -Omega^*$ and $G^top = G$.
Equating:
$ G Omega^* + Omega^* G = 2 "skew"(X^top Z) $

This is a Lyapunov equation in $Omega^*$.
Since $G$ is positive definite, all sums $lambda_iota + lambda_gamma > 0$, so the equation has a unique solution.
In the eigenbasis of $G = "diag"(lambda_1, ..., lambda_d)$, the solution is elementwise:
$ Omega^*_(iota gamma) = (2 "skew"(X^top Z)_(iota gamma)) / (lambda_iota + lambda_gamma) $

For the application in @prop:curvature-criterion, $Z = S X$ with $S$ skew-symmetric.
Then $X^top Z = X^top S X$, which is already skew-symmetric (since $(X^top S X)^top = X^top S^top X = -X^top S X$), so $"skew"(X^top S X) = X^top S X$ and the Lyapunov equation becomes $G Omega^* + Omega^* G = 2 X^top S X$.


= Norm of the Vertical Bracket Component <app:vertical-norm>

We derive the explicit formula for $||[overline(xi)_1, overline(xi)_2]^cal(V)||^2$ stated in @prop:curvature-criterion.

The vertical component is $X Omega^*$ with $Omega^*_(iota gamma) = 2(X^top S X)_(iota gamma) \/ (lambda_iota + lambda_gamma)$ where $S = -[M_1, M_2]$ (@app:vertical-projection).
Its squared norm in the ambient Euclidean metric is:
$ ||X Omega^*||^2 = "tr"((X Omega^*)^top X Omega^*) = "tr"(Omega^(*top) G Omega^*) $

Working in the eigenbasis of $G = "diag"(lambda_1, ..., lambda_d)$:
$ "tr"(Omega^(*top) G Omega^*) = sum_(iota, gamma) lambda_iota (Omega^*_(iota gamma))^2 $
(using $"tr"(A^top B A) = sum_a b_a sum_b A_(a b)^2$ when $B = "diag"(b_1, ..., b_d)$; diagonal terms vanish since $Omega^*_(iota iota) = 0$).
Collecting the pair $(iota, gamma)$ and $( gamma, iota)$ for $iota < gamma$ and using skew-symmetry $(Omega^*_(gamma iota))^2 = (Omega^*_(iota gamma))^2$:
$ = sum_(iota < gamma) (lambda_iota + lambda_gamma)(Omega^*_(iota gamma))^2 $

Substituting $Omega^*_(iota gamma) = 2(X^top S X)_(iota gamma) \/ (lambda_iota + lambda_gamma)$:
$ = sum_(iota < gamma) (lambda_iota + lambda_gamma) dot (4(X^top S X)_(iota gamma)^2) / (lambda_iota + lambda_gamma)^2 = 4 sum_(iota < gamma) ((X^top S X)_(iota gamma)^2) / (lambda_iota + lambda_gamma) $

It remains to express $(X^top S X)_(iota gamma)$ in terms of $(U^top [M_1, M_2] U)_(iota gamma)$.
Writing $X = U Lambda^(1\/2)$ in the canonical gauge (where $U$ has orthonormal columns spanning $"col"(X)$ and $Lambda = "diag"(lambda_1, ..., lambda_d)$):
$ X^top S X = Lambda^(1\/2) U^top S U Lambda^(1\/2) $
so $(X^top S X)_(iota gamma) = sqrt(lambda_iota) (U^top S U)_(iota gamma) sqrt(lambda_gamma)$.
Since $S = -[M_1, M_2]$:
$ (X^top S X)_(iota gamma)^2 = lambda_iota lambda_gamma (U^top [M_1, M_2] U)_(iota gamma)^2 $

Substituting:
$ ||X Omega^*||^2 = 4 sum_(iota < gamma) (lambda_iota lambda_gamma) / (lambda_iota + lambda_gamma) [(U^top [M_1, M_2] U)_(iota gamma)]^2 $

This is the formula stated in @prop:curvature-criterion.
The $1 \/ (lambda_iota + lambda_gamma)$ weighting shows that curvature is amplified when both eigenvalues in a pair are small, matching the connection coefficient structure of the fiber bundle.


= Extension to Directed Graphs <app:directed>

This appendix is a sketch intended to indicate how the framework adapts to directed graphs. Details are omitted.

For directed graphs, each node has source position $g_i$ and target position $r_i$.
The probability matrix is $P = G R^top$ (not symmetric).

The gauge group becomes $(G, R) tilde (G Q, R Q)$ for $Q in O(d)$, and the invisible dynamics are $dot(G) = G A$, $dot(R) = R A$ with $A in so(d)$.

The gauge-consistent architecture generalizes to:
$ dot(G) = N_G (P) G, quad dot(R) = N_R (P) R $
with appropriate symmetry constraints on $(N_G, N_R)$ ensuring that the induced dynamics on $P = G R^top$ are well-defined.
The fiber bundle framework carries over: the total space is pairs $(G, R) in RR_*^(n times d) times RR_*^(n times d)$, the structure group is $O(d)$ acting diagonally, and the connection, curvature, and holonomy theory applies with the same Lyapunov-type structures, though the asymmetry of $P$ introduces additional subtleties in the spectral decomposition.
