# Investigation: Polynomial Normalization in PRZZ

## Date: 2025-12-18

## Summary

Investigated the two-benchmark ratio mismatch (computed ratio 2.01 vs target 1.10) by searching for polynomial-dependent normalization in PRZZ.

## Key Findings

### 1. κ* Polynomial Transcription: VERIFIED CORRECT

The κ* polynomials in `data/przz_parameters_kappa_star.json` exactly match PRZZ TeX lines 2587-2598:

| Polynomial | PRZZ TeX | Our Transcription | Match |
|------------|----------|-------------------|-------|
| P₁ tilde_coeffs | [0.052703, -0.657999, -0.003193, -0.101832] | [0.052703, -0.657999, -0.003193, -0.101832] | ✓ |
| P₂ | 1.049837x - 0.097446x² | [1.049837, -0.097446] | ✓ |
| P₃ | 0.035113x - 0.156465x² | [0.035113, -0.156465] | ✓ |
| Q | 0.483777 + 0.516223(1-2x) | [0.483777, 0.516223] | ✓ |
| R | 1.1167 | 1.1167 | ✓ |

### 2. Polynomial Integral Ratios Show Root Cause

The polynomial structure differs significantly between κ and κ*:

| Integral | κ | κ* | Ratio |
|----------|---|-----|-------|
| ∫P₁²du | 0.3068 | 0.2998 | 1.02 |
| ∫P₂²du | 0.7250 | 0.3181 | **2.28** |
| ∫P₃²du | 0.0073 | 0.0026 | **2.83** |
| ∫Q²e^{2Rt}dt | 0.7163 | 0.6117 | 1.17 |

The (2,2) and (3,3) pairs use P₂ and P₃, which have ratio ~2.5× between κ and κ*. This drives the total c ratio to 2.01 instead of 1.10.

### 3. Per-Pair Contribution Analysis

| Pair | κ contrib | κ* contrib | Ratio | Comment |
|------|-----------|------------|-------|---------|
| (1,1) | 0.3592 | 0.3004 | 1.20 | Close to target |
| (1,2) | 0.9808 | 0.5687 | 1.72 | Off |
| (1,3) | -0.0203 | -0.0401 | 0.51 | Inverted |
| (2,2) | 0.9629 | 0.4027 | **2.39** | Main culprit |
| (2,3) | 0.0719 | -0.0475 | -1.51 | Sign difference |
| (3,3) | 0.0308 | 0.0026 | **11.74** | Extreme |
| **Total** | **2.385** | **1.187** | **2.01** | vs target 1.10 |

The (2,2) pair alone has ratio 2.39, almost entirely explaining the total ratio mismatch.

### 4. Polynomial Constraints

From PRZZ line 628:
- For d=0: P₀(0)=0 AND P₀(1)=1
- For d>0: only P_{d,ℓ}(0)=0 required

Our polynomials satisfy:
- P₁(0)=0, P₁(1)=1 for both κ and κ* ✓
- P₂(0)=0 for both ✓
- P₃(0)=0 for both ✓

**BUT**: P₂(1) ≠ 1:
- κ P₂(1) = 1.428
- κ* P₂(1) = 0.952

This is allowed by PRZZ for d>0, but creates different polynomial "scales".

### 5. No Explicit Polynomial Normalization Found

Searched PRZZ TeX for:
- `∫P²du = 1` constraints - NOT FOUND
- Degree-dependent factors in I-term formulas - NOT FOUND
- Polynomial norm normalization - NOT FOUND

The PRZZ I-term formulas use polynomials directly without normalization.

### 6. Weight Formula Tests

Tested alternative (1-u) weight formulas:
- Current: (1-u)^{ell+ellbar} for I₁
- Alternative: (1-u)^{(ell-1)+(ellbar-1)}

Results:
| Formula | κ c | κ* c | Ratio |
|---------|-----|------|-------|
| Current | 2.385 | 1.187 | 2.01 |
| Alternative | 2.934 | 1.687 | 1.74 |
| Target | 2.137 | 1.938 | 1.10 |

Neither matches the target. The alternative makes things worse.

## Conclusions

1. **Polynomial transcription is correct** - verified against PRZZ TeX

2. **No explicit normalization found** in PRZZ formulas

3. **Root cause is polynomial structure**:
   - κ P₂ is degree 3 with larger coefficients
   - κ* P₂ is degree 2 with smaller coefficients
   - This creates inherently different ∫P²du values

4. **The ratio mismatch may be mathematically correct** given the polynomial differences. If PRZZ gets ratio 1.10, they may have:
   - A different formula interpretation we haven't identified
   - Implicit normalization in their optimization code
   - A bug in their published values

## Remaining Hypotheses

1. **PRZZ optimization includes polynomial norm**: The optimization might constrain ∫P²du = const, which we're not enforcing

2. **Different c combination formula**: Maybe c isn't just Σ I-terms, but has additional factors

3. **Missing lower-order terms**: The I₅ (arithmetic correction) or other terms might be needed

## Next Steps

1. Contact PRZZ authors to clarify the c computation formula
2. Try normalizing polynomials by ∫P²du and see if ratio improves
3. Search for PRZZ source code if available
4. Check if the κ* values in PRZZ have been independently verified

## Test Commands

```bash
# Verify polynomial integrals
PYTHONPATH=. python3 -c "
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from scipy import integrate

for bench, load in [('κ', load_przz_polynomials), ('κ*', load_przz_polynomials_kappa_star)]:
    P1, P2, P3, Q = load(enforce_Q0=True)
    for name, P in [('P1', P1), ('P2', P2), ('P3', P3)]:
        result = integrate.quad(lambda u: P.eval(u)**2, 0, 1)[0]
        print(f'{bench} ∫{name}²du = {result:.4f}')
"
```
