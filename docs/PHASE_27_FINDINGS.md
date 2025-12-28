# Phase 27 Findings: Derived Mirror Transform Analysis

**Date:** 2025-12-26
**Status:** Key findings documented, investigation ongoing

---

## Executive Summary

Phase 27 implemented a derived mirror transform based on PRZZ TeX structure, attempting to replace the empirical `m = exp(R) + 5` formula with a first-principles derivation. The investigation revealed significant discrepancies between:

1. **Phase 26B unified_general backend** (validated against OLD DSL)
2. **Existing evaluate.py infrastructure** (achieves ~2% accuracy on c)

The derived mirror transform gives m_eff ≈ 3.94, not matching the empirical m = 8.68.

---

## Key Findings

### Finding 1: I1/I2 Value Discrepancy

Phase 26B's `unified_general` backend gives different I1/I2 values than the existing `evaluate.py` infrastructure:

| Pair | Phase 26B I1 | evaluate.py I1 | Ratio |
|------|--------------|----------------|-------|
| (1,1) | 0.4135 | 0.4135 | 1.00 |
| (2,2) | 3.8839 | 0.9169 | 4.24 |
| (3,3) | 2.8610 | 0.0503 | 56.9 |
| (1,2) | -0.5650 | -0.5543 | 1.02 |
| (1,3) | -0.5816 | +0.0715 | **sign flip!** |
| (2,3) | 3.5715 | -0.1727 | **sign flip!** |

**Key observation:** For higher pairs (3,3) and cross-terms (1,3), (2,3), the values differ dramatically, including sign flips.

### Finding 2: S12 Totals

| Source | S12_direct(+R) |
|--------|----------------|
| Phase 26B s12_backend | 2.494 |
| Existing evaluate.py | 0.797 |

The Phase 26B value is 3.13× larger due to:
1. Factorial normalization (l1!×l2!) included in I1 values
2. Different I2 computation path

### Finding 3: Mirror m_eff Values

Testing derived mirror with different T_prefactor modes:

| Mode | m_eff | vs Empirical (8.68) |
|------|-------|---------------------|
| "absorbed" | 0.29 | 3.3% |
| "none" | 3.94 | 45.4% |
| "exp_2R" | 53.4 | 615% |

None match the empirical m = exp(R) + 5 = 8.68.

### Finding 4: Existing c Computation Works

The existing `compute_c_paper_with_mirror` achieves:
- c = 2.108 (target: 2.138, gap: -1.4%)
- κ = 0.428 (target: 0.417, gap: +2.5%)

This uses:
- S12_plus = 0.797
- S12_minus = 0.220
- S34 = -0.600
- m = 8.68
- c = 0.797 + 8.68×0.220 - 0.600 = 2.108

---

## Files Created in Phase 27

| File | Purpose |
|------|---------|
| `src/evaluator/s12_backend.py` | Backend abstraction for S12 computation |
| `src/mirror_transform_derived.py` | Derived mirror transform implementation |
| `src/evaluator/c_assembly_tex_mirror.py` | c assembly with derived mirror (partial) |
| `scripts/run_phase27_meff_report.py` | m_eff diagnostic script |
| `tests/test_s12_backend_equivalence.py` | Backend validation tests |
| `tests/test_mirror_transform_microcases.py` | Mirror transform tests |

---

## Open Questions

### Q1: Why do Phase 26B and evaluate.py give different I1 values?

Phase 26B unified_general was validated against OLD DSL structure. But evaluate.py uses a different computation path via `make_all_terms_k3()` and `evaluate_term()`. These should be mathematically equivalent but produce different results.

**Hypothesis:** The factorial normalization (l1!×l2!) is applied differently:
- Phase 26B: Applied inside compute_I1_unified_general
- evaluate.py: Not applied in I1 values, only during assembly

But this doesn't explain the sign flips.

### Q2: Why is m_eff ≈ 3.94 instead of 8.68?

The derived mirror uses:
1. Sign flip on (x+y) coefficient in exp factor
2. Q eigenvalue swap (x↔y coupling)
3. Same t-integration structure as direct

This gives m_eff = 3.94 / 1.0 (proxy ratio). The gap 8.68/3.94 ≈ 2.2 suggests a missing factor.

**Possibilities:**
- Missing factor of 2 from symmetry
- Incorrect eigenvalue swap direction
- The empirical formula encodes additional structure beyond T^{-(α+β)}

### Q3: Which I1/I2 values are "correct"?

Both claim to implement PRZZ structure:
- Phase 26B: Matches OLD DSL (validated)
- evaluate.py: Achieves ~2% accuracy on c (empirically validated)

They can't both be correct. Investigation needed to understand the discrepancy.

---

## Recommended Next Steps

### Path A: Reconcile I1/I2 discrepancy
1. Trace evaluate.py I1 computation path completely
2. Compare to Phase 26B unified_general derivation
3. Identify mathematical source of difference

### Path B: Investigate m_eff gap
1. Test different T_prefactor formulas: exp(R), exp(R/2), exp(R)/2, etc.
2. Check if Q eigenvalue swap is correct
3. Re-derive mirror structure from PRZZ TeX more carefully

### Path C: Hybrid approach
1. Use existing evaluate.py infrastructure (achieves ~2% accuracy)
2. Replace only the mirror multiplier with derived value
3. Measure accuracy improvement

---

## Conclusion

Phase 27 revealed that the mirror transform derivation is more complex than initially understood. The key blocker is the I1/I2 discrepancy between Phase 26B and evaluate.py. Until this is resolved, the derived mirror transform cannot be properly validated.

The empirical m = exp(R) + 5 formula encodes relationships not yet fully understood at the operator level.
