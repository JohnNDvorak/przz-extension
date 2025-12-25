# Handoff Document: Session 3 - December 17, 2025

## Executive Summary

Multi-agent investigation resolved key aspects of the ratio reversal mystery. **The ω-dependent factorial normalization is NOT the answer** - the issue is primarily **polynomial L² norm differences** between κ and κ* benchmarks.

## Critical Findings

### 1. ω-Dependent Normalization Does NOT Explain Ratio Reversal

From PRZZ TeX analysis, Case C (ω > 0) has suppression factor 1/(ω-1)!. However:

- **ω depends only on piece index ℓ, NOT polynomial degree**
- For d=1: ω = l₁ - 1 where l₁ = ℓ - 1 (number of Λ-convolutions)
- This gives the SAME suppression factors for both κ and κ* benchmarks:
  - Piece ℓ=1: ω = -1 → Case A
  - Piece ℓ=2: ω = 0 → Case B
  - Piece ℓ=3: ω = 1 → Case C with 1/0! = 1

**Conclusion:** Factorial normalization is identical for both benchmarks.

### 2. Root Cause: Polynomial L² Norm Differences

The (2,2) pair analysis revealed the ratio breakdown:

| Component | κ/κ* Ratio | Contribution |
|-----------|------------|--------------|
| ∫P₂² du | **2.28** | 85% of ratio |
| ∫Q²e^{2Rt}dt | 1.17 | 15% of ratio |
| Combined I₂ | 2.67 | Base integral |
| With derivatives | 2.43 | Total |

**Key Insight:** κ has **much larger polynomial integrals** than κ*:
- κ: ∫P₂² du = 0.725 (degree-3 polynomial)
- κ*: ∫P₂² du = 0.318 (degree-2 polynomial)
- Ratio: 2.28x

### 3. Derivatives Actually HELP, Not Hurt

Contrary to the previous handoff statement ("derivatives make ratio worse"):

- I₂ alone ratio: 2.67
- With I₁+I₃+I₄: 2.43
- **Derivatives reduce ratio by 9%**

The derivative sum (I₁+I₃+I₄) has ratio = **1.20**, very close to target 1.10!

### 4. All I-Terms Scale Similarly

| Term | κ Value | κ* Value | Ratio |
|------|---------|----------|-------|
| I₁ | 1.169 | 0.490 | 2.39 |
| I₂ | 0.909 | 0.341 | 2.67 |
| I₃ | -0.544 | -0.212 | 2.57 |
| I₄ | -0.544 | -0.212 | 2.57 |
| **Total** | **0.989** | **0.407** | **2.43** |

This uniformity suggests a **global scaling issue**, not term-specific errors.

### 5. PRZZ ω-Parameter Structure

For d=1 (K=3 mollifier), the ω parameter is:
```
ω(d,l) = 1×l₁ + 2×l₂ + ⋯ + d×l_d - 1

For d=1: ω = l₁ - 1 where l₁ = ℓ - 1 (Λ-convolution count)
```

Three cases:
- **Case A (ω = -1):** Piece ℓ=1, suppression ~ 1/log N
- **Case B (ω = 0):** Piece ℓ=2, no suppression
- **Case C (ω > 0):** Piece ℓ=3, suppression 1/(ω-1)!

## What We Can Rule Out

1. **ω-dependent factorial normalization** - Same for both benchmarks
2. **Derivative extraction errors** - All terms scale consistently by ~2.4x
3. **Sign convention issues** - Would affect absolute values, not ratios
4. **I₅ missing** - Lower-order term, forbidden in main mode
5. **DSL variable structure** - Oracle also shows 2.43x ratio

## What Remains Plausible

1. **Polynomial-dependent normalization** - PRZZ may normalize by ∫P² or similar
2. **Missing prefactor** - Something like 1/∫P_{ℓ}² for each polynomial
3. **Different interpretation of "const"** - In c = const × ∫Q²e^{2Rt}dt

## Numerical Summary

### Oracle c Values
- κ (R=1.3036): c = 2.137
- κ* (R=1.1167): c = 1.938
- Target ratio: **1.103**

### Computed Values (Our Code)
- κ total (all pairs): ~2.42
- κ* total (all pairs): ~1.21
- Computed ratio: **2.01** (wrong direction!)

### Gap Analysis
```
Target ratio: 1.103
Our ratio: 2.01
Gap factor: 0.55 (we're ~1.8x too high in ratio)
```

## Code Locations

| File | Purpose |
|------|---------|
| `src/przz_22_exact_oracle.py` | Oracle for (2,2) I-terms |
| `src/compare_term_structure.py` | I-term ratio comparison |
| `src/check_polynomial_integrals.py` | Polynomial integral analysis |
| `TERM_STRUCTURE_ANALYSIS.md` | Detailed analysis notes |
| `RATIO_BREAKDOWN_ANALYSIS.md` | Complete ratio decomposition |

## Next Session Priorities

### Priority 1: Search for Polynomial Normalization in PRZZ
Search PRZZ TeX for:
- Division by ∫P² or ∫P_ℓ²
- "Normalized polynomial" phrases
- L² norm mentions

### Priority 2: Check PRZZ Section 7 Prefactors
Re-read PRZZ Section 7 line-by-line for:
- Prefactors on I₂ formula (line 1548)
- R-dependent normalization
- Polynomial-dependent normalization

### Priority 3: Test Polynomial Normalization Hypothesis
Try computing c with:
```python
I2_normalized = I2 / integral_P2_squared
```
If ratio improves toward 1.10, this confirms the hypothesis.

### Priority 4: Verify the "const" Interpretation
The handoff mentioned:
```
c = const × ∫Q²e^{2Rt}dt
const ratio needed: 0.94
Our const ratio: 1.71 (wrong direction!)
```
Need to understand what "const" actually represents in PRZZ.

## Key Equations

**PRZZ I₂ structure:**
```
I₂ = (1/θ) × [∫₀¹ P_ℓ₁(u)P_ℓ₂(u) du] × [∫₀¹ Q(t)² e^{2Rt} dt]
```

**Ratio decomposition:**
```
I₂(κ)/I₂(κ*) = [∫P²(κ)/∫P²(κ*)] × [∫Q²e^{2Rt}(κ)/∫Q²e^{2Rt}(κ*)]
             = 2.28 × 1.17
             = 2.67
```

## Files Created This Session

- `docs/HANDOFF_2025_12_17_SESSION3.md` (this file)
- `TERM_STRUCTURE_ANALYSIS.md`
- `RATIO_BREAKDOWN_ANALYSIS.md`
- `analyze_kappa_ratio.py`
- `detailed_ratio_analysis.py`
- `src/compare_term_structure.py`
- `src/check_polynomial_integrals.py`

## Agent Results Summary

| Agent | Finding |
|-------|---------|
| PRZZ TeX | ω-dependent normalization same for both benchmarks |
| Ratio Analysis | Polynomial L² norm is 2.28x different |
| I-term Decomposition | All terms scale by ~2.4x consistently |
| Polynomial Degree | κ has higher degree (3 vs 2) polynomials |
| Normalization Search | Current code has 1/(ℓ₁!ℓ₂!) but NOT polynomial-norm-dependent |

## Test Commands

```bash
# Run I-term comparison
cd przz-extension && python3 src/compare_term_structure.py

# Run polynomial integral analysis
cd przz-extension && python3 src/check_polynomial_integrals.py

# Run ratio analysis
cd przz-extension && python3 analyze_kappa_ratio.py

# Run all tests
cd przz-extension && python -m pytest tests/ -v
```

## Conclusion

The ratio reversal mystery is now better understood:
1. **NOT from ω-dependent factorial suppression** (same for both benchmarks)
2. **Primarily from polynomial L² norm differences** (2.28x factor)
3. **Secondarily from R/Q differences** (1.17x factor)
4. **Derivatives help, not hurt** (reduce ratio by 9%)

The missing piece is likely a **polynomial-dependent normalization factor** that we haven't found in PRZZ's formulas yet. Next session should focus on searching PRZZ TeX for how polynomials are normalized.
