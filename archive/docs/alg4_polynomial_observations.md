# Alg4 Polynomial (Gauge-Free) — Observations & Interpretations

## Summary
We explored a gauge-free estimation route for polynomial dynamics by fitting directly in `P = X X'` space. This removes the gauge feedback loop entirely. The key result is that **P-only regression recovers the true parameters when given the true latent series**, but **fails on ASE embeddings** due to noise in `X̂` (and thus in `P̂`).

## What we implemented
- **Gauge-free alpha estimation** in `P` space using the trapezoid rule:
  - Model: `dot(P) = 2 N(P) P`, with `N(P) = Σ_k α_k P^k`
  - Discrete regression:  
    `P_{t+1} - P_t ≈ dt Σ_k α_k (P_t^(k+1) + P_{t+1}^(k+1))`
- **No R-step** and **no gauge alignment** for the polynomial family when running in gauge-free mode.
- Residuals computed **directly in P-space**.

## Key empirical findings
1. **True series recovery works**  
   On the true latent trajectory `X(t)`, the gauge-free P regression recovers the coefficients accurately:
   - True `α = [-0.0001, 0.3, -0.02]`
   - Estimated `α ≈ [0.000307, 0.299711, -0.019983]`

2. **ASE series recovery fails**  
   On the ASE embedding `X̂(t)` (from averaged adjacency), the same gauge-free regression is strongly biased:
   - Estimated `α ≈ [0.41233, 0.01733, -0.00304]`
   - Mean P residual ≈ `1.2008`

3. **A_avg vs ASE fidelity to P_true**  
   Directly comparing `P̂ = X̂ X̂'` and `A_avg` to `P_true`:
   - Mean relative P error: `P̂` vs true ≈ `0.0570`
   - Mean relative P error: `A_avg` vs true ≈ `0.1565`
   - Interpretation: ASE produces a closer `P̂` than raw `A_avg` in this setting.

4. **A_avg-based regression (no ASE)**  
   Using `A_avg` directly in the trapezoid regression (no regularization):
   - Estimated `α ≈ [0.12919, 0.10766, -0.00779]`
   - Still biased relative to truth; direct P regression on `A_avg` did not recover parameters.

5. **A_avg regression with scaling + ridge**  
   Feature scaling (per-basis norm) + ridge (`l2 = 1e-2`) yields:
   - Estimated `α ≈ [0.12931, -0.00358, -0.00041]`
   - Shrinkage does not restore the correct parameter balance.

6. **ASE (P̂) regression with scaling + ridge**  
   Feature scaling (per-basis norm) + ridge (`l2 = 1e-2`) yields:
   - Estimated `α ≈ [0.18885, -0.00559, -0.00054]`
   - Shrinkage does not restore the correct parameter balance.

7. **ASE (P̂) regression without scaling + ridge**  
   Ridge only (`l2 = 1e-2`) yields:
   - Estimated `α ≈ [0.27715, 0.07384, -0.00619]`
   - Still biased relative to truth.

8. **Interpretation**  
   - The **gauge-free P regression is correct** in principle and performs well on clean trajectories.
   - The **dominant failure mode is embedding/sampling noise** in `P̂` and `A_avg`, which overwhelms parameter recovery even without gauge entanglement.
   - Regularization and scaling help numerical stability but **do not remove bias** from noisy `P` estimates.
   - Direct P regression on `A_avg` (with and without regularization/rescaling) still fails to recover the true coefficients.

## Implications
- Gauge freedom is **not** the limiting factor for the polynomial model when working purely in `P` space.
- The bottleneck is the **quality of `P̂`**, which depends on sampling noise and ASE distortion.
- Improving `P̂` quality (e.g., larger `K_samples`, alternative embeddings, or denoising) is likely more impactful than further gauge handling for polynomial dynamics.

## Next directions
- Explore **denoising / variance-aware regression** for `P̂` (e.g., weighted least squares using entrywise variance).
- Try **alternative embeddings or direct P estimators** that reduce bias relative to `P_true`.
- Test **bias-correction or calibration** for `A_avg` before regression (since raw `A_avg` underperforms).