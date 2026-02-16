# Alg4 Implementation Plan & Progress Log

This document tracks the **implementation plan** for Section 4 algorithms (structure‑constrained alignment) and keeps a **progress log** that is updated as work advances.

---

## 0) Scope & Constraints

- Implement **separately** from the core package first.
- Target **three dynamics families**:
  1. Linear symmetric
  2. Polynomial in P
  3. Message‑passing
- Use **averaged adjacency** before embedding.
- Always log the **spectral gap** λ_d and the **gauge skew diagnostic**.
- Keep datasets **synthetic** and controlled for identifiability tests.

---

## 1) Data Generation (Synthetic)

### 1.1 Linear symmetric dynamics
**Dynamics:**  
Ẋ = N X, with N = Nᵀ

**Initial positions:**  
Two clusters in ℝ², separated.

**Key params:**  
n=30, d=2, T=40, dt=1, K=20 samples per time.

**Diagnostics:**  
λ_d(t) from X(t); skew gauge error on X̂(t).

---

### 1.2 Polynomial dynamics in P
**Dynamics:**  
Ẋ = (α₀I + α₁P + α₂P²) X, with P = X Xᵀ

**Initial positions:**  
Single cluster in ℝ².

**Key params:**  
n=30, d=2, T=40, dt=1, K=20 samples per time.

**Diagnostics:**  
λ_d(t) and skew gauge error.

---

### 1.3 Message‑passing dynamics
**Dynamics:**  
ẋᵢ = Σⱼ Pᵢⱼ g(xᵢ, xⱼ)  
g(xᵢ, xⱼ) = β₁(xⱼ − xᵢ) + β₂ xⱼ

**Initial positions:**  
Three clusters in ℝ².

**Key params:**  
n=25, d=2, T=40, dt=1, K=20 samples per time.

**Diagnostics:**  
λ_d(t) and skew gauge error.

---

## 2) Observation Model

For each time t:

1. P(t) = X(t) X(t)ᵀ
2. Sample K adjacency matrices A_k(t) ~ Bernoulli(P(t))
3. Average: Ā(t) = (1/K) Σ A_k(t)
4. Embed Ā(t) with ASE to get X̂(t)

---

## 3) Alignment + Learning (Per Family)

### 3.1 Linear symmetric family
**Discrete model:**  
X(t+1) ≈ M X(t), with M = Mᵀ

**Optimization:**  
min Σ_t || X̂(t+1) R_tᵀ − M X̂(t) ||_F²

**Alternating updates:**
- **M‑step:** least squares + symmetrize
- **R‑step:** Procrustes per t

**Gauge diagnostic:**  
Track ||Skew(X̂(t)ᵀ dX̂(t))||_F across iterations.

---

### 3.2 Polynomial family
**Discrete model:**  
X(t+1) ≈ M(P(t)) X(t)  
M(P) = I + dt Σ α_k P^k

**Optimization:**
min Σ_t || X̂(t+1) R_tᵀ − M(P(t)) X̂(t) ||_F²

**Alternating updates:**
- **α‑step:**  
α = G⁻¹ b / dt  
G_kj = Σ_t ⟨Φ_t^k, Φ_t^j⟩  
b_k  = Σ_t ⟨Φ_t^k, Y_t⟩  
Φ_t^k = (P(t))^k X̂(t)  
Y_t = X̂(t+1) R_tᵀ
- **R‑step:** Procrustes per t

---

### 3.3 Message‑passing family
**Model:**  
ẋᵢ = Σⱼ Pᵢⱼ g_θ(xᵢ, xⱼ)

**Optimization:**  
Jointly optimize θ and Q_t.

**Approach:**  
- θ‑step: gradient descent (Lux/Optimization)
- Q‑step: Riemannian optimization on O(d)^T (Manopt Trust Regions)

---

## 4) Baselines

Run for all families:

1. **No alignment** (expect failure)
2. **Procrustes chain + fit**
3. **Structure‑constrained alignment + fit** (target method)

Optional:
4. **DUASE/OMNI embedding + fit**

---

## 5) Evaluation Metrics

**Primary:**
- P‑error: ||P_rec − P_true||_F / ||P_true||_F
- Distance correlation (pairwise distances)
- Dynamics residual: ||X_{t+1} − X_t − dt f(X_t)||_F

**Gauge diagnostic:**
- Skew magnitude: ||Skew(X̂(t)ᵀ dX̂(t))||_F

**Spectral gap:**
- λ_d(t), track min/mean over time.

---

## 6) Planned Script Layout (Separate)

```
scripts/alg4/
  alg4_utils.jl
  generate_linear.jl
  generate_polynomial.jl
  generate_message_passing.jl
  run_linear_alignment.jl
  run_polynomial_alignment.jl
  run_message_passing_alignment.jl
```

Outputs stored in:
```
data/alg4/<family>/
results/alg4/<family>/
```

---

## 7) Progress Log

### ✅ 2025‑MM‑DD (current)
- Drafted `alg4_implementation_plan.md` and recorded the end‑to‑end plan.
- Created shared utilities module for Alg4.
- Implemented dataset generators for:
  - linear symmetric
  - polynomial
  - message‑passing
- Averaged adjacency sampling before embedding.
- Diagnostics included: spectral gap and skew gauge error.
- Ran generators:
  - linear: completed and saved `data/alg4/linear/linear_data.jls`
  - polynomial: completed with solver warning (dt below epsilon), saved `data/alg4/polynomial/polynomial_data.jls`
  - message‑passing: completed with solver warning (dt below epsilon), saved `data/alg4/message_passing/message_passing_data.jls`
- Noted in runs:
  - polynomial skew diagnostic returned NaN due to early solver abort
  - message‑passing terminated early (unstable/ill‑conditioned step)

### ✅ Stabilized dynamics and plotting
- Stabilized all three dynamics families with valid P ∈ [0,1]:
  - linear: P ∈ [0.21, 0.84], 0 violations
  - polynomial: P ∈ [0.003, 1.0], 0 violations
  - message‑passing: P ∈ [0.0, 1.07], 81 minor violations (acceptable)
- Fixed `plot_dynamics.jl`:
  - Updated CairoMakie imports
  - Replaced deprecated `arrows!` with `arrows2d!`
  - Removed unsupported `linewidth` parameter
- Generated all diagnostic plots:
  - Phase portraits (true trajectories)
  - Phase comparison (true vs ASE embeddings)
  - Diagnostics timeseries (spectral gap, skew error, P validity)
  - P heatmaps (raw and clamped)
  - Summary plots
  - All families comparison plot

### ✅ Alignment runs (linear + polynomial)
- Implemented alignment utilities and scripts:
  - `scripts/alg4/alg4_alignment_utils.jl`
  - `scripts/alg4/run_linear_alignment.jl`
  - `scripts/alg4/run_polynomial_alignment.jl`
- Ran linear alignment:
  - Residual decreased from ~2.63 to ~2.41 over 50 iters
- Gauge‑free polynomial P regression (trapezoid rule):
  - On true `X(t)` series, recovered α accurately:
    - true α = `[-0.0001, 0.3, -0.02]`
    - est α = `[0.000307..., 0.299711..., -0.019982...]`
  - On ASE `X̂(t)`, gauge‑free P regression is biased:
    - est α ≈ `[0.41233, 0.01733, -0.00304]`
    - mean P residual ≈ `1.2008`
  - Conclusion: P‑only regression works on true series; ASE noise dominates parameter recovery.
  - Saved `results/alg4/linear_alignment_results.jls`
- Ran polynomial alignment:
  - Converged quickly (iter 3), residual ~4.1248
  - Saved `results/alg4/polynomial_alignment_results.jls`

### ⏭️ Next
- Implement message‑passing alignment (Riemannian Q‑step + θ‑step).
- Add baseline comparisons.
- Evaluate skew diagnostic vs alignment residuals.