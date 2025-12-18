# Ratio Reversal Diagnosis - Computation Checklist

**Purpose:** Exact numerical values needed to complete the diagnosis

## Quick Reference Values

### Benchmark Parameters
```
κ:  R=1.3036, θ=4/7, c_target=2.13745, κ_target=0.417293962
κ*: R=1.1167, θ=4/7, c_target=1.93795, κ*_target=0.407511457

Target c ratio: 2.13745/1.93795 = 1.1026
```

### Polynomial Structure
```
κ:  P₁(deg 5), P₂(deg 3), P₃(deg 3), Q(deg 5, 4 terms)
κ*: P₁(deg 5), P₂(deg 2), P₃(deg 2), Q(deg 1, 2 terms)
```

## Computation Tasks

### Task 1: T-Integral (Common Factor)

**Compute:**
```python
t_int_k  = ∫₀¹ Q_κ²(t) exp(2×1.3036×t) dt
t_int_ks = ∫₀¹ Q_κ*²(t) exp(2×1.1167×t) dt
t_ratio  = t_int_k / t_int_ks
```

**Expected:** t_ratio ≈ 1.17

**Significance:** This is the "exponential × Q structure" component that's common to all pairs.

---

### Task 2: U-Integrals (Per-Pair Polynomial Structure)

**Diagonal Pairs:**

#### (1,1): ∫₀¹ P₁²(u) du
```python
u_11_k  = ∫ P₁_κ² du
u_11_ks = ∫ P₁_κ*² du
ratio_11 = u_11_k / u_11_ks
```
**Expected:** ~0.8-1.2 (same degree, different coeffs)

#### (2,2): ∫₀¹ P₂²(u) du
```python
u_22_k  = ∫ P₂_κ² du
u_22_ks = ∫ P₂_κ*² du
ratio_22 = u_22_k / u_22_ks
```
**Expected:** ~3.5-4.0 (degree 3 vs 2 - CRITICAL!)

**Hand calculation check:**
```
P₂_κ  = x(1.048 + 1.320x - 0.940x²)
P₂_κ* = x(1.050 - 0.097x)

∫ P₂_κ² du ≈ 1.16 (from analytical expansion)
∫ P₂_κ*² du ≈ 0.32 (from analytical expansion)
Ratio ≈ 3.63
```

#### (3,3): ∫₀¹ P₃²(u) du
```python
u_33_k  = ∫ P₃_κ² du
u_33_ks = ∫ P₃_κ*² du
ratio_33 = u_33_k / u_33_ks
```
**Expected:** ~2-3 (degree 3 vs 2)

**Cross Pairs:**

#### (1,2): ∫₀¹ P₁(u)P₂(u) du
```python
u_12_k  = ∫ P₁_κ(u)P₂_κ(u) du
u_12_ks = ∫ P₁_κ*(u)P₂_κ*(u) du
ratio_12 = u_12_k / u_12_ks
```
**Expected:** ~1-2 (mixed degrees)

#### (1,3): ∫₀¹ P₁(u)P₃(u) du
```python
u_13_k  = ∫ P₁_κ(u)P₃_κ(u) du
u_13_ks = ∫ P₁_κ*(u)P₃_κ*(u) du
ratio_13 = u_13_k / u_13_ks
```
**Expected:** ~1-2 (mixed degrees)

#### (2,3): ∫₀¹ P₂(u)P₃(u) du
```python
u_23_k  = ∫ P₂_κ(u)P₃_κ(u) du
u_23_ks = ∫ P₂_κ*(u)P₃_κ*(u) du
ratio_23 = u_23_k / u_23_ks
```
**Expected:** ~2-3 (both degree 3 vs 2)

---

### Task 3: Per-Pair I₂ Values

**Formula:** I₂ = (7/4) × u-integral × t-integral

**Compute for each pair:**
```python
I2_k  = (7/4) × u_k × t_int_k
I2_ks = (7/4) × u_ks × t_int_ks
I2_ratio = I2_k / I2_ks
```

**Expected total I₂ ratio:** ~1.66

**Critical pair:** (2,2) should show ratio ~4-5

---

### Task 4: Oracle Full Results

**For (1,1) pair:**
```python
result_11_k = przz_oracle_22(P₁_κ, Q_κ, 4/7, 1.3036, n_quad=100)
result_11_ks = przz_oracle_22(P₁_κ*, Q_κ*, 4/7, 1.1167, n_quad=100)

Output:
  I₁_κ, I₂_κ, I₃_κ, I₄_κ, total_κ
  I₁_κ*, I₂_κ*, I₃_κ*, I₄_κ*, total_κ*
  Ratios for each
```

**For (2,2) pair:**
```python
result_22_k = przz_oracle_22(P₂_κ, Q_κ, 4/7, 1.3036, n_quad=100)
result_22_ks = przz_oracle_22(P₂_κ*, Q_κ*, 4/7, 1.1167, n_quad=100)

Output:
  I₁_κ, I₂_κ, I₃_κ, I₄_κ, total_κ
  I₁_κ*, I₂_κ*, I₃_κ*, I₄_κ*, total_κ*
  Ratios for each
```

**Expected from SESSION_SUMMARY:**
```
(2,2) κ:  I₁≈1.17, I₂≈0.91, I₃≈-0.54, I₄≈-0.54, total≈1.00
(2,2) κ*: I₁≈0.48, I₂≈0.34, I₃≈-0.15, I₄≈-0.15, total≈0.52
Ratios:   2.44,    2.68,    3.60,     3.60,     1.92
```

---

### Task 5: Derivative Analysis

**For (2,2) pair specifically:**
```python
deriv_k = I₁_κ + I₃_κ + I₄_κ
deriv_ks = I₁_κ* + I₃_κ* + I₄_κ*
deriv_ratio = deriv_k / deriv_ks

I2_frac_k = I₂_κ / total_κ
I2_frac_ks = I₂_κ* / total_κ*
```

**Output:**
- What fraction of total comes from I₂ vs derivatives?
- Are derivatives helping or hurting the ratio?

**Expected:** Derivatives make it worse (ratio ~3.5 vs I₂ ratio ~2.7)

---

## Execution Commands

### Option 1: Full Diagnostic
```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
python ratio_reversal_diagnosis.py
```

**Output:** Complete tables for all tasks

### Option 2: Quick Check
```bash
python compute_diagnosis.py
```

**Output:** Summary of key ratios only

---

## Validation Checks

### Check 1: T-Integral Consistency
All pairs should show the SAME t-integral values (t_int_k and t_int_ks).
If they differ → bug in integration.

### Check 2: Symmetry
For diagonal pairs: I₃ should equal I₄ (by x↔y symmetry).
If not → oracle bug.

### Check 3: Quadrature Convergence
Re-run with n_quad = 60, 80, 100, 120.
Ratios should be stable to 3-4 digits.

### Check 4: Total Consistency
```
Sum of all I₂_{ℓ₁,ℓ₂} should equal total I₂ contribution
```

---

## Critical Numbers to Extract

**Most important 5 numbers:**

1. **t_ratio:** Should be ~1.17
2. **u_ratio_22:** Should be ~3.5-4.0
3. **I2_total_ratio:** Should be ~1.66
4. **oracle_total_ratio_22:** Should be ~1.9-2.0
5. **deriv_ratio_22:** Should be ~3.5

If these 5 match predictions → **polynomial degree is the root cause**.

If any differ significantly → **need to investigate that component**.

---

## Post-Execution Analysis

### If All Predictions Match

**Conclusion:** Polynomial degree mismatch confirmed.

**Next step:** Test normalization hypothesis:
```python
norm_k = sqrt(∫ P₂_κ² du)   # ≈ 1.08
norm_ks = sqrt(∫ P₂_κ*² du) # ≈ 0.56

P₂_κ_normalized = P₂_κ / norm_k
P₂_κ*_normalized = P₂_κ* / norm_ks

# Recompute with normalized polynomials
# Check if ratio becomes ~1.10
```

### If Predictions Don't Match

**Investigate:**
- Which component differs most?
- Is it u-integral, t-integral, or derivative?
- Does the difference point to a specific formula error?

---

## Success Metric

**Diagnosis complete when:**
We can point to a specific formula element and say:
"This normalization/factor/convention, when applied, reduces the ratio from 2.0 to 1.10."

---

**Ready to execute:** All scripts prepared, predictions documented, validation criteria defined.
