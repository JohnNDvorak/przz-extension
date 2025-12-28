# Phase 28 Findings: S12 Backend Canonicalization

**Date:** 2025-12-26
**Status:** COMPLETE - Mirror multiplier is empirical, not derived

---

## Executive Summary

Phase 28 discovered the root cause of the Phase 27 m_eff discrepancy:

**`unified_general` ≡ `raw` kernel regime**
**`compute_c_paper_with_mirror` ≡ `paper` kernel regime**

These are **mathematically different** due to Case C kernel attenuation in paper regime.

---

## The Discovery

### Kernel Regime Comparison

Testing the OLD DSL term makers with different kernel regimes:

| Pair | `raw` regime | `paper` regime | unified_general | Ratio (paper/raw) |
|------|-------------|----------------|-----------------|-------------------|
| (2,2) | 3.884 | 0.917 | 3.884 ✓ | 0.24 |
| (1,3) | -0.582 | +0.072 | -0.582 ✓ | **sign flip** |
| (2,3) | 3.571 | -0.173 | 3.571 ✓ | **sign flip** |

### Key Insight

The OLD DSL with `kernel_regime=None` or `"raw"` gives **identical values** to Phase 26B's `unified_general`.

The OLD DSL with `kernel_regime="paper"` gives the values used in `compute_c_paper_with_mirror`.

### Why This Matters

1. **Phase 26B validation was against raw regime** - explaining why unified_general matched OLD DSL
2. **Production evaluator uses paper regime** - which has different values
3. **Empirical m = exp(R) + 5 was calibrated for paper regime** - not raw!
4. **Phase 27's derived mirror used raw regime** - comparing apples to oranges!

---

## The Two Regimes

### Raw Regime (unified_general)

Uses simple exponential/log factors:
- exp(2Rt + Rθ(2t-1)(x+y))
- Q(A_α) × Q(A_β) with A = t + θ(t-1)x + θty

Good for mathematical derivation, matches PRZZ TeX structure directly.

### Paper Regime (compute_c_paper_with_mirror)

Uses Case C kernel attenuation for P₂/P₃ terms:
- Different exponential structure for higher pieces
- Sign flips for (1,3), (2,3) pairs
- Magnitude reduction ~4× for (2,2)

Achieves ~2% accuracy on c targets.

---

## Implications for Mirror Derivation

### Phase 27's Error

Phase 27 computed:
- S12_direct using unified_general (raw regime)
- Mirror transform using raw regime structure
- Compared m_eff to empirical m = exp(R) + 5 (calibrated for paper regime)

This comparison was **meaningless** because the two S12 objects are not the same.

### Correct Approach

To derive the mirror transform correctly:
1. Either implement mirror transform for **paper regime** kernels
2. Or calibrate a new empirical m for **raw regime**

The existing m = exp(R) + 5 is **not** the correct multiplier for raw regime S12.

---

## Recommended Next Steps

### Option A: Implement Paper Regime in Unified Engine

Create `compute_I1_unified_paper()` that:
1. Uses the same bivariate coefficient extraction as unified_general
2. But applies Case C kernel attenuation for P₂/P₃ terms
3. This would let us test derived mirror against the actual production baseline

### Option B: Derive m for Raw Regime

1. Compute c using raw regime S12
2. Find what m value makes: c = S12_raw(+R) + m × S12_raw(-R) + S34(+R)
3. This would give the "correct" m for raw regime (likely NOT exp(R) + 5)

### Option C: Use Term DSL with Paper Regime

1. Compute derived mirror using term DSL (paper regime)
2. Compare to term DSL proxy(-R)
3. This stays within consistent paper regime

---

## Files Created in Phase 28

| File | Purpose |
|------|---------|
| `src/evaluator/s12_spec.py` | Canonical S12 representation with regime/convention tracking |
| `scripts/run_phase28_s12_matrix_compare.py` | Matrix comparison diagnostic |
| `scripts/run_phase28_paper_regime_mirror.py` | Paper regime mirror analysis |
| `tests/test_phase28_regime_equivalence.py` | Regime equivalence tests (10 passing) |
| `docs/PHASE_28_FINDINGS.md` | This summary |

---

## Conclusion

**Phase 28 succeeded in identifying the root cause.**

The unified_general backend is mathematically correct for raw regime.
The paper regime (used in production) applies additional kernel attenuation.
These are intentionally different - paper regime models the PRZZ "Case C" structure.

The mirror derivation must be redone using paper regime S12, not raw regime.

---

## Extended Analysis: Paper Regime Mirror Relationship

### Paper Regime S12 Values (with factorial normalization)

| R | S12(+R) | S12(-R) | Ratio |
|---|---------|---------|-------|
| 1.3036 | 0.797 | 0.220 | 3.62 |

### Critical Discovery: m is NOT a ratio

The empirical formula uses m = exp(R) + 5 = 8.68.

But **S12(+R) / S12(-R) = 3.62**, not 8.68.

This proves that the mirror multiplier m is **empirically calibrated**, not derived from
a direct/proxy ratio relationship.

### How the c Formula Works

```
c = S12(+R) + m × S12(-R) + S34(+R)
  = 0.797 + 8.68 × 0.220 + (-0.6)
  = 0.797 + 1.91 - 0.6
  = 2.11
```

This matches the target c ≈ 2.138 within 1.4%.

The formula works because m was **calibrated to make it work**, not because m represents
the ratio of direct to mirror contributions.

### Per-Pair Ratios

| Pair | I1(+R)/I1(-R) | % of m=8.68 |
|------|---------------|-------------|
| (1,1) | 2.38 | 27% |
| (2,2) | 3.56 | 41% |
| (3,3) | 4.29 | 49% |
| (1,2) | 2.97 | 34% |
| (1,3) | 3.76 | 43% |
| (2,3) | 4.10 | 47% |

The per-pair ratios range from 2.4 to 4.3, all far below m = 8.68.

### Implications

1. **Cannot derive m from first principles** using the ratio approach
2. **m = exp(R) + 5 is a calibration formula** that happens to work well
3. **Phase 27's approach was fundamentally flawed** - trying to derive an empirical constant
4. **The production evaluator is correct** - it uses empirical m and achieves ~2% accuracy

### Why Does m = exp(R) + 5 Work?

This remains an open question. The formula was discovered empirically during
development of `compute_c_paper_with_mirror`. It may encode:

- Asymptotic behavior of the mirror transform
- Compensation for Case C kernel attenuation
- A happy coincidence that produces accurate results

Without deriving the PRZZ mirror structure from first principles (including Case C
handling), we cannot explain why this particular formula works.
