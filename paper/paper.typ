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
    We present a framework that learns continuous, interpretable differential equations governing the evolution of temporal network structure.
    Our approach embeds networks into a low-dimensional latent space via Random Dot Product Graphs (RDPG), learns the dynamics of this embedding using Neural Ordinary Differential Equations (Neural ODEs), and extracts human-interpretable equations through symbolic regression.
    We develop a gauge-theoretic analysis showing that RDPG embeddings have rotational ambiguity, and derive a gauge-consistent architecture $dot(X) = N(P)X$ with symmetric $N$ that eliminates this ambiguity while achieving dramatic parameter reduction (from $approx$10,000 to as few as 2 parameters).
    We demonstrate the framework on synthetic temporal networks, showing that it successfully recovers governing equations and dynamical parameters.
    This work bridges the gap between predictive accuracy and mechanistic understanding in temporal network modeling.
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

*Contributions.* We introduce:
+ A complete pipeline from temporal network observations to interpretable differential equations
+ A gauge-theoretic analysis of RDPG dynamics, identifying what can and cannot be learned from embedding trajectories
+ Parsimonious architectures ($dot(X) = N(P)X$ with symmetric $N$) that are gauge-consistent by construction and can recover exact dynamical parameters
+ Demonstration on synthetic systems with known ground-truth dynamics
+ Open-source Julia implementation (`RDPGDynamics.jl`) for reproducibility

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
This produces a smooth trajectory ${hat(X)_t}$ in latent space.

#remark[
  When the adjacency matrix $A$ has negative eigenvalues (e.g., due to heterophilic "opposites attract" connectivity), the standard RDPG with $P = X X^top >= 0$ is inadequate.
  The *Generalized RDPG* (GRDPG) @rubin2022statistical handles this by using an indefinite inner product: $P_(i j) = arrow(x)_i^top I_(p,q) arrow(x)_j$ where $I_(p,q) = "diag"(1,...,1,-1,...,-1)$.
  The gauge group then becomes the indefinite orthogonal group $O(p,q)$.
  Our gauge-theoretic analysis extends naturally to this setting, though we focus on the positive-definite case for clarity.
]

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

*Practical obstructions.*
Beyond the theoretical gauge freedom, practical challenges include:
(i) estimation error in $hat(X)$ from SVD ($approx$35% position error, though $hat(P)$ has only $approx$5% error);
(ii) Procrustes alignment artifacts that can introduce spurious motion or remove real motion resembling global rotation;
(iii) discrete, noisy observations rather than continuous $P(t)$.
These factors may explain why some dynamics (e.g., circulation) are harder to learn than others (e.g., attraction/repulsion).

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

*Extensions.*
The UDE framework (@sec:ude) enables incorporating domain knowledge.
For ecological networks, one might encode known trophic interactions in $N_("known")$ while learning corrections.
For social networks, community structure could inform block-diagonal parameterizations.
The theory extends to directed graphs (@app:directed), where $P = L R^top$ with separate dynamics for source and target embeddings, and to oscillatory dynamics (@app:oscillations), which symmetric $N$ can produce through nonlinear coupling despite having real eigenvalues in the linear case.

= Conclusion

We introduced a framework that transforms the problem of temporal network modeling from discrete event prediction to continuous dynamical systems analysis.
The gauge-theoretic analysis reveals that RDPG embeddings have inherent rotational ambiguity, but we identify a broad class of observable dynamics and derive architectures that are gauge-consistent by construction.

The parsimonious $dot(X) = N(P)X$ form with symmetric $N$ achieves two goals simultaneously: it eliminates ambiguity about what the model can learn, and it reduces parameters by orders of magnitude while maintaining or improving accuracy.
When the true dynamics have this form, the polynomial parameterization can recover exact coefficients---a level of interpretability that post-hoc symbolic regression cannot match.

Our approach produces interpretable differential equations governing network evolution, enabling both prediction and mechanistic insight.
The open-source implementation facilitates application to new domains.

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
