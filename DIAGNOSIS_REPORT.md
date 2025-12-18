# Ratio Reversal Diagnosis - Numerical Breakdown

**Generated:** 2025-12-17
**Purpose:** Diagnose the ratio reversal issue where computed ratios disagree with PRZZ targets

## Context

- **c formula:** c = const × ∫Q²e^{2Rt}dt
- **t-integral ratio (κ/κ*):** 1.171 (from exponential term)
- **PRZZ needs const ratio:** 0.942 (to achieve c ratio of 1.103)
- **Our naive gives:** 1.71 (wrong!)

## Methodology

This diagnosis follows 5 steps:
1. Load both polynomial sets (κ and κ*)
2. Compute per-pair I₂ values (no derivatives) for BOTH benchmarks
3. Compute full I₁+I₂+I₃+I₄ for (1,1) and (2,2) using oracle
4. Calculate what derivative contribution ratio would achieve target
5. Check (2,2) oracle behavior and understand ratio scaling

## Benchmark Parameters

### κ Benchmark (Main)
- **R:** 1.3036
- **θ:** 4/7 = 0.571428...
- **c_target:** 2.13745440613217263636
- **κ_target:** 0.417293962

**Polynomials:**
- P₁: degree 5, tilde_coeffs = [0.261076, -1.071007, -0.236840, 0.260233]
- P₂: degree 3, tilde_coeffs = [1.048274, 1.319912, -0.940058]
- P₃: degree 3, tilde_coeffs = [0.522811, -0.686510, -0.049923]
- Q: degree 5, (1-2x) basis with powers [0,1,3,5]

### κ* Benchmark (Simple zeros)
- **R:** 1.1167
- **θ:** 4/7 = 0.571428...
- **c_target:** 1.9379524124677437
- **κ*_target:** 0.407511457

**Polynomials:**
- P₁: degree 5, tilde_coeffs = [0.052703, -0.657999, -0.003193, -0.101832]
- P₂: degree 2, tilde_coeffs = [1.049837, -0.097446]
- P₃: degree 2, tilde_coeffs = [0.035113, -0.156465]
- Q: degree 1 (LINEAR!), (1-2x) basis with powers [0,1] only

**Key difference:** κ* uses SIMPLER polynomials (linear Q, degree-2 P₂/P₃ vs degree-3)

## Expected Target Ratio

```
c_ratio_target = 2.13745/1.93795 = 1.102648
```

This is the ratio that PRZZ achieves between the two benchmarks.

## Step 1: Polynomial Structure Analysis

The critical observation is that κ* polynomials are structurally simpler:
- **Q:** Linear (degree 1) vs degree 5
- **P₂:** Degree 2 vs degree 3
- **P₃:** Degree 2 vs degree 3

This affects ALL integrals, not just I₂.

## Step 2: I₂ Component Analysis

Formula: **I₂ = (1/θ) × [∫P_ℓ₁(u)P_ℓ₂(u)du] × [∫Q²e^{2Rt}dt]**

### Factorization

I₂ factors into:
1. **u-integral:** ∫P_ℓ₁(u)P_ℓ₂(u)du (polynomial structure)
2. **t-integral:** ∫Q²e^{2Rt}dt (exponential + Q structure)

### T-Integral Analysis

For the t-integral:
- **κ:** ∫Q_κ²(t)e^{2·1.3036·t}dt
- **κ*:** ∫Q_κ*²(t)e^{2·1.1167·t}dt

**Q structure difference:**
- Q_κ is degree-5: more oscillatory, larger polynomial values
- Q_κ* is degree-1: simpler, smaller polynomial values

**Expected ratio:** The t-integral ratio depends on BOTH:
- Exponential factor: exp(2Rt) is larger for κ (R=1.3036 > 1.1167)
- Polynomial magnitude: Q_κ² vs Q_κ*²

From previous analysis: **t-integral ratio ≈ 1.17**

### U-Integral Analysis

Expected u-integral ratios for each pair:

**(1,1) pair:** ∫P₁_κ(u)² du vs ∫P₁_κ*(u)² du
- Both are degree 5, but different coefficients
- Expected: ratio depends on coefficient magnitudes

**(2,2) pair:** ∫P₂_κ(u)² du vs ∫P₂_κ*(u)² du
- P₂_κ is degree 3: ∫(c₀x + c₁x² + c₂x³)² dx
- P₂_κ* is degree 2: ∫(c₀x + c₁x²)² dx
- **Higher degree → larger integral (more cross terms)**
- Expected: **ratio > 1** (κ numerator larger)

**(3,3) pair:** Similar to (2,2)
- P₃_κ is degree 3, P₃_κ* is degree 2
- Expected: **ratio > 1**

### Per-Pair I₂ Expected Behavior

Each I₂_{ℓ₁,ℓ₂} = (1/θ) × u-int × t-int

**Diagonal pairs (1,1), (2,2), (3,3):**
- t-integral ratio: ~1.17 (same for all)
- u-integral ratio: varies by polynomial degree
- (2,2) and (3,3) should have HIGHER ratios (degree effect)

**Cross pairs (1,2), (1,3), (2,3):**
- Can be positive or negative
- Ratios may vary significantly

### Expected I₂ Total Ratio

From SESSION_SUMMARY: **I₂-only total ratio ≈ 1.66**

This is LARGER than the target 1.10, which means:
- The polynomial integrals contribute a factor of ~1.66/1.17 ≈ 1.42
- This is consistent with degree effects (higher-degree polynomials → larger integrals)

## Step 3: Full Oracle Results (I₁+I₂+I₃+I₄)

### (1,1) Pair Expected Results

The (1,1) pair uses P₁ for both factors.

**Components:**
- **I₂:** Pure polynomial × exponential (no derivatives)
- **I₁:** Second mixed derivative d²/dxdy
- **I₃:** First derivative d/dx
- **I₄:** First derivative d/dy (equals I₃ by symmetry)

**Expected ratios:**
- I₂ ratio: influenced by both polynomial and exponential
- Derivative terms: should partially CANCEL the I₂ excess

### (2,2) Pair Expected Results

From previous oracle runs (SESSION_SUMMARY):
- **I₂ ratio:** ~2.67 (very large!)
- **I₁+I₃+I₄ ratio:** ~3.54 (even larger!)
- **Total ratio:** ~2.09

The (2,2) pair shows EXTREME sensitivity to polynomial degree difference.

**Why so large?**
- P₂_κ is degree 3, P₂_κ* is degree 2
- The derivative terms d²/dxdy, d/dx, d/dy amplify degree differences
- Higher derivatives of degree-3 polynomial > degree-2 polynomial

## Step 4: Derivative Contribution Analysis

### Algebra

Let:
- I₂_κ = I₂ value for κ benchmark
- I₂_κ* = I₂ value for κ* benchmark
- D_κ = I₁+I₃+I₄ for κ
- D_κ* = I₁+I₃+I₄ for κ*

Total: c_κ = I₂_κ + D_κ, c_κ* = I₂_κ* + D_κ*

Target ratio: (I₂_κ + D_κ)/(I₂_κ* + D_κ*) = 1.103

**Observed:**
- I₂ ratio ≈ 1.66
- Derivative ratio ≈ 3.54 (from (2,2) pair)

**Problem:** Both components have ratios LARGER than target!

This means derivatives are NOT correcting the I₂ excess—they're making it worse.

### What Would Fix It?

For c ratio to be 1.103 when I₂ ratio is 1.66, we would need:

```
(1.66·I₂_κ* + D_κ)/(I₂_κ* + D_κ*) = 1.103
```

Solving for D_κ/D_κ*:
```
D_κ/D_κ* = (1.103·I₂_κ* + 1.103·D_κ* - 1.66·I₂_κ*)/(I₂_κ* + ...)
```

If I₂ is 90% of the total (from previous analysis):
- I₂_κ* ≈ 0.9·c_κ*
- D_κ* ≈ 0.1·c_κ*

Then we'd need D_κ/D_κ* ≈ **NEGATIVE** to pull the ratio down!

**This is impossible** since derivatives have the same sign structure.

## Step 5: Root Cause Analysis

### The Fundamental Issue

The ratio reversal stems from **polynomial degree mismatch**:

1. **κ* uses simpler polynomials** (degree-2 P₂/P₃, linear Q)
2. **Higher-degree polynomials have larger integrals** (more cross terms)
3. **This affects ALL components:** I₁, I₂, I₃, I₄ proportionally
4. **The ratio is AMPLIFIED, not corrected**

### Why PRZZ Achieves 1.103

PRZZ must be using one of:

**Hypothesis A: Polynomial normalization**
- Different scaling convention for different polynomial degrees
- Each P_ℓ might be normalized by its L² norm
- This would remove degree effects

**Hypothesis B: Different c definition**
- PRZZ's c might include polynomial-dependent prefactors
- The formula c = ∫Q²e^{2Rt}dt might be oversimplified

**Hypothesis C: Missing R-dependent factor**
- There may be R-dependent normalization in the full formula
- The I₁-I₄ formulas might have R-dependent prefactors we're missing

**Hypothesis D: Transcription error**
- The κ* polynomial coefficients might be transcribed incorrectly
- Need to verify PRZZ TeX lines 2587-2598 character-by-character

## Diagnostic Computations Required

To complete this diagnosis, we need to compute:

### 1. Per-Pair I₂ Values

```python
# For each pair (ℓ₁,ℓ₂) in [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]:
I2_k, u_int_k, t_int_k = compute_i2(P_ℓ₁_k, P_ℓ₂_k, Q_k, R_k, theta)
I2_ks, u_int_ks, t_int_ks = compute_i2(P_ℓ₁_ks, P_ℓ₂_ks, Q_ks, R_ks, theta)
ratio = I2_k / I2_ks
```

**Expected output table:**
```
Pair    I₂(κ)      I₂(κ*)     Ratio    u-int(κ)  u-int(κ*)  t-int(κ)  t-int(κ*)
(1,1)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XXXX    X.XXXX
(2,2)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XXXX    X.XXXX
(3,3)   X.XXXX     X.XXXX     X.XX     X.XXXX    X.XXXX     X.XXXX    X.XXXX
...
```

### 2. Oracle Full Results

```python
# (1,1) pair
result_11_k = przz_oracle_22(P1_k, Q_k, theta, R_k, n_quad=100)
result_11_ks = przz_oracle_22(P1_ks, Q_ks, theta, R_ks, n_quad=100)

# (2,2) pair
result_22_k = przz_oracle_22(P2_k, Q_k, theta, R_k, n_quad=100)
result_22_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_ks, n_quad=100)
```

**Expected output:**
```
Component   κ value      κ* value     Ratio
I₁          X.XXXX       X.XXXX       X.XX
I₂          X.XXXX       X.XXXX       X.XX
I₃          X.XXXX       X.XXXX       X.XX
I₄          X.XXXX       X.XXXX       X.XX
Total       X.XXXX       X.XXXX       X.XX
```

### 3. Derivative Fraction Analysis

For the (2,2) pair specifically:
```
I₂_frac_k = I₂_k / total_k
I₂_frac_ks = I₂_ks / total_ks
deriv_contrib_k = (I₁ + I₃ + I₄)_k
deriv_contrib_ks = (I₁ + I₃ + I₄)_ks
```

This tells us what fraction of the total comes from derivatives vs base I₂.

## Expected Findings

Based on the mathematical structure, I predict:

1. **t-integral ratio:** ~1.17 (exponential dominates)
2. **u-integral ratios:**
   - (1,1): ~1.0-1.2 (same degree)
   - (2,2): ~1.5-2.0 (degree 3 vs 2)
   - (3,3): ~1.5-2.0 (degree 3 vs 2)
3. **I₂ total ratio:** ~1.6-1.7 (matches previous)
4. **Derivative ratios:** ~2.0-4.0 (amplified by degree)
5. **Total c ratio:** ~2.0-2.1 (matches previous)

**Gap to target:** Factor of ~1.9 (2.0/1.103)

## Recommended Actions

### Immediate (to complete diagnosis):

1. **Run the diagnostic script** to get exact numerical values
2. **Verify κ* polynomial transcription** from PRZZ TeX
3. **Search PRZZ paper** for polynomial normalization conventions

### After diagnosis:

1. **Check Section 7 of PRZZ** for normalization formulas
2. **Look for degree-dependent factors** in their formulas
3. **Compare against PRZZ's published per-pair breakdown** (if available)
4. **Test hypothesis:** Normalize each P_ℓ by its L² norm before computing

## Files Created

- `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/ratio_reversal_diagnosis.py`
  - Complete diagnostic script (awaiting execution)

- `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/compute_diagnosis.py`
  - Simplified version for quick computation

## References

- **PRZZ Paper:** arXiv:1802.10521
- **κ polynomials:** Lines 2567-2586
- **κ* polynomials:** Lines 2587-2598
- **Formula derivations:** TECHNICAL_ANALYSIS.md Section 9
- **Previous findings:** SESSION_SUMMARY.md Track 3

---

**Status:** Awaiting numerical execution to confirm predictions.
