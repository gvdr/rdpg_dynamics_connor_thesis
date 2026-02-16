// ============================================================================
// Section 7 update: Numerical illustration
// Replaces everything from "== Numerical illustration <sec:numerics>"
// up to (but not including) "= Discussion <sec:discussion>"
// ============================================================================

== Numerical illustration <sec:numerics>

We illustrate the theory with two controlled experiments on synthetic RDPG data.
The first demonstrates anchor-based alignment on gauge-equivariant polynomial dynamics, where alignment quality can be assessed independently of dynamics recovery.
The second demonstrates the full downstream pipeline --- UDE training and symbolic regression --- on non-gauge-equivariant dynamics, where alignment quality directly impacts dynamics recovery.

=== Experiment 1: Anchor-based alignment <sec:numerics-anchor>

*Setup.*
We generate an RDPG with $n = 200$ nodes in $d = 2$ latent dimensions.
The initial positions $X(0)$ are drawn uniformly from $B_+^2 = {x in RR^2 : x_1, x_2 >= 0, ||x|| <= 1}$, with a designated anchor set $S$ of $n_a$ nodes.
The non-anchor nodes evolve under polynomial dynamics $dot(X) = (alpha_0 I + alpha_1 P) X$ with $(alpha_0, alpha_1) = (-0.3, 0.003)$, while anchor nodes remain fixed: $dot(x)_i = 0$ for $i in S$.
We integrate the ODE to produce a trajectory $X(t)$ at $T$ equally-spaced times, generate $K = 3$ independent adjacency matrices $A_k^((t)) tilde "Bernoulli"(X(t) X(t)^top)$ at each time (averaging to reduce noise), compute ASE embeddings $hat(X)^((t))$, and compare anchor-based alignment to sequential Procrustes.

*Metrics.*
We measure alignment quality by the mean Procrustes error:
$ "err"(t) = 1/n ||hat(X)^((t)) hat(Q)^((t)) - X^((t)) Q^*||_F $
where $Q^*$ is the best global alignment to the true trajectory (accounting for the residual gauge at $t = 0$).
We sweep over four experimental conditions: anchor count $n_a$, trajectory length $T$, anchor drift rate $epsilon$, and initial embedding scale (as proxy for spectral gap).

#figure(
  image("plots/anchor-main-results.png", width: 95%),
  caption: [
    Anchor-based alignment experiment ($n = 200$, $d = 2$, polynomial dynamics $dot(X) = (alpha_0 I + alpha_1 P) X$ with $K = 3$ Bernoulli samples per time step).
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
    *(a)* Alignment error as a function of the initial embedding norm scale (proxy for signal strength): smaller norms yield weaker signals, increasing ASE noise and degrading alignment for both methods.
    *(b)* Phase portrait with $n_a = 15$ anchor nodes (red diamonds, stationary) and non-anchor nodes (gray trajectories evolving under polynomial dynamics) in $B_+^2$.
  ],
) <fig:anchor-spectral>

*Results.*
The experiment confirms the theoretical predictions:
+ *Sufficient anchors ($n_a >= d$):* alignment error stabilizes at the ASE noise floor (@fig:anchor-main\a). Below $n_a = d = 2$, Procrustes is underdetermined and fails catastrophically.
+ *Bounded vs.\ growing error:* anchor-based alignment maintains $O(1)$ error independent of $T$, while sequential Procrustes accumulates $O(sqrt(T))$ drift. At $T = 200$, the sequential method is approximately 13% worse (@fig:anchor-main\b, d).
+ *Drifting anchors:* a systematic bias linear in $epsilon$ becomes visible for $epsilon >= 0.01$ (@fig:anchor-main\c), consistent with the short-time validity condition $epsilon T << 1\/sqrt(n)$.

*Dynamics recovery is gauge-free.*
A notable observation: for polynomial dynamics, the coefficients $(alpha_0, alpha_1)$ can be estimated from the $P$-space trajectory via the gauge-free estimator of @sec:constructive, since $P = X X^top$ is rotation-invariant.
With $n_a = 15$, we recover $(hat(alpha)_0, hat(alpha)_1) = (-0.250 plus.minus 0.008, 0.0024 plus.minus 0.0001)$ against true values $(-0.3, 0.003)$; with $n_a = 0$ the estimates are $(-0.282 plus.minus 0.008, 0.0028 plus.minus 0.0001)$.
The similar quality regardless of anchor count confirms that the $P$-based estimator does not require alignment.
This motivates a natural question: when the dynamics _cannot_ be recovered from $P$ alone, does alignment quality directly impact dynamics recovery?
The second experiment addresses this.


=== Experiment 2: UDE pipeline with non-gauge-equivariant dynamics <sec:numerics-ude>

*Motivation.*
The polynomial dynamics in Experiment 1 are gauge-equivariant: $P = X X^top$ is rotation-invariant, so the dynamics parameters live in $P$-space and can be recovered without alignment.
To demonstrate the full UDE pipeline and the necessity of anchor-based alignment for dynamics recovery, we design dynamics that depend on the coordinates of $X$ directly --- making alignment quality a bottleneck.

*Setup.*
We generate an RDPG with $n = 200$ nodes in $d = 3$ latent dimensions, with $n_a = 100$ anchor nodes organized into 3 communities near the vertices of $B_+^3$ (at positions $(0.7, 0.2, 0.2)$, $(0.2, 0.7, 0.2)$, $(0.2, 0.2, 0.7)$ plus Gaussian noise).
The 100 non-anchor nodes evolve under damped spiral dynamics around their community centroids $mu_k$:
$ dot(x)_i = (-gamma + beta ||x_i - mu_k||^2)(x_i - mu_k) + omega J(x_i - mu_k) $
where $J = 1/sqrt(3) mat(0, -1, 1; 1, 0, -1; -1, 1, 0)$ is the rotation generator around the $(1,1,1)/sqrt(3)$ axis, and $(gamma, beta, omega) = (0.3, -0.5, 1.0)$.
These dynamics are _not_ gauge-equivariant: the centroids $mu_k$ and rotation axis live in $X$-space coordinates, so a misaligned trajectory $tilde(X)(t)$ produces different offsets $tilde(x)_i - mu_k != x_i - mu_k$, corrupting the dynamics structure.

We observe $K = 10$ independent adjacency matrices per time step over $T = 50$ steps at $delta t = 0.1$, compute ASE embeddings, and apply three alignment conditions: anchor-based Procrustes, sequential Procrustes, and no alignment (raw ASE with random per-frame rotations).

*UDE architecture.*
Following the framework of @sec:constructive, we decompose the dynamics into known and unknown components:
$ dot(x)_i = underbrace(-hat(gamma) (x_i - mu_k), f_"known") + underbrace(f_theta (x_i - mu_k), f_"NN") $
where $hat(gamma)$ is a learnable scalar parameter and $f_theta: RR^3 -> RR^3$ is a small neural network (architecture: $3 -> 16 -> 16 -> 3$ with $tanh$ activations, $approx 390$ parameters total).
The known part encodes the structural assumption that non-anchor nodes are attracted toward their community centroids; the unknown part captures whatever additional dynamics exist.
L2 regularization on the network weights ($lambda = 10^(-3)$) discourages the network from absorbing the linear damping term, improving the identifiability of the known-unknown decomposition.
The UDE is trained by solving the neural ODE forward from the initial condition and minimizing the trajectory MSE via adjoint sensitivity methods.

*Primary metric: total dynamics MSE.*
Since the additive UDE decomposition has an inherent identifiability issue --- the network can absorb part of the linear damping, shifting $hat(gamma)$ --- we evaluate the _total_ learned dynamics function $f_"learned"(delta) = -hat(gamma) delta + f_theta (delta)$ against the true dynamics $f_"true"(delta) = (-gamma + beta ||delta||^2) delta + omega J delta$, evaluated on random test inputs $delta in [-0.3, 0.3]^3$.
This metric is invariant to the gamma--network split.

#figure(
  image("plots/ude-pipeline-results.png", width: 95%),
  caption: [
    UDE pipeline experiment ($n = 200$, $d = 3$, damped spiral dynamics with rotation around $(1,1,1)/sqrt(3)$, $K = 10$ samples).
    *(a)* True trajectories (gray) and anchor-aligned ASE (blue) in $B_+^3$; anchor nodes shown as red diamonds.
    *(b)* Learned NN residual $f_theta(delta)$ vs.\ true residual $f_u(delta)$: anchor-aligned data (left) produces tight agreement along the diagonal; sequential Procrustes (center) and unaligned data (right) degrade progressively.
    *(c)* Symbolic regression Pareto fronts (loss vs.\ expression complexity): anchor-aligned data achieves $1$--$2$ orders of magnitude lower loss at each complexity level.
    *(d)* Total dynamics MSE (log scale) by alignment condition: anchor alignment achieves MSE $approx 6 times 10^(-4)$, sequential Procrustes $approx 8 times 10^(-3)$ ($13 times$ worse), and unaligned $approx 0.44$ ($approx 700 times$ worse).
    Error bars show $plus.minus 1$ standard deviation across 5 repetitions.
  ],
) <fig:ude-pipeline>

*Results.*
The total dynamics MSE cleanly separates the three alignment conditions (@fig:ude-pipeline\d):
- *Anchor-aligned:* MSE $= 6 times 10^(-4) plus.minus 4 times 10^(-4)$ --- the UDE accurately recovers the complete dynamics function.
- *Sequential Procrustes:* MSE $= 7.6 times 10^(-3) plus.minus 3 times 10^(-4)$ --- approximately $13 times$ worse, reflecting systematic gauge drift that corrupts the coordinate-dependent dynamics.
- *Unaligned:* MSE $= 0.44 plus.minus 0.35$ --- approximately $700 times$ worse, with high variance; the network cannot learn coherent dynamics from incoherently-rotated frames.

The NN residual accuracy (@fig:ude-pipeline\b) shows the mechanism: with anchor alignment, the learned $f_theta$ closely matches the true nonlinear residual $beta ||delta||^2 delta + omega J delta$; without alignment, the network fits noise rather than structure.

*Symbolic regression.*
Symbolic regression on the trained network's input-output pairs (@fig:ude-pipeline\c) achieves Pareto-optimal expressions with losses $1$--$2$ orders of magnitude lower under anchor alignment than under the other conditions at each complexity level.
The anchor-aligned expressions contain the expected structural terms (products of coordinates and cross-terms consistent with the rotation generator $J$), though exact closed-form recovery requires careful tuning of the search parameters.

*The identifiability caveat.*
The learned damping parameter $hat(gamma)$ shows high variance across repetitions ($0.26 plus.minus 0.08$ for anchor-aligned, vs.\ true $gamma = 0.3$), reflecting the fundamental identifiability issue in additive UDE decompositions: the network can absorb part of the linear term.
This is a known limitation, and the total dynamics MSE confirms it is benign --- the complete dynamics function $f_"known" + f_"NN"$ is well-recovered regardless of the $hat(gamma)$-network split.

*Takeaway.*
When dynamics are gauge-equivariant ($dot(X) = N(P) X$), alignment quality is irrelevant for parameter recovery (Experiment 1).
When dynamics depend on $X$-space coordinates, alignment quality directly controls the fidelity of the learned dynamics (Experiment 2).
Anchor-based alignment provides the gauge-consistent trajectories that the downstream UDE pipeline requires.
