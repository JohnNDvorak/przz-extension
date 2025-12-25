# Phase 7 Planning Summary

**Date:** 2025-12-22
**Status:** Planning Complete, Approved for Implementation

---

## What Was Accomplished

### 1. Codebase Analysis

Conducted comprehensive exploration of the PRZZ κ-optimization codebase:

- **54 Python source files** in `src/`
- **76 test files** with 64+ tests passing
- **102 documentation files**
- Key modules analyzed:
  - `m1_policy.py` — Current empirical m₁ implementation (exp(R)+5 for K=3)
  - `operator_post_identity.py` — Post-identity operator approach (Phase 5 validated)
  - `combined_identity_regularized.py` — U-regularized path (machine precision match)
  - `evaluate.py` — Main computation engine (~230KB)
  - `polynomials.py` — P₁, P₂, P₃, Q polynomial classes

### 2. GPT Guidance Integration

Analyzed and integrated GPT's Phase 7 guidance into implementation plan:

**Key Insight:** The Q-shift ratio (85-127×) differs dramatically from empirical m₁ (~8.68). This proves "m₁ as a single scalar" is the **wrong derivation target**.

**Reframing:**
- The combined identity (TeX 1502-1511) already INTERNALIZES the mirror contribution
- There is no external scalar m₁ in the TeX
- Current m₁ is a projection weight from our decomposition — not a TeX constant

### 3. Phase 7 Plan Created

Created `PLAN_PHASE7_TEX_EXACT.md` with four sub-phases:

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| **7A** | TeX-exact evaluator (no m₁) | `src/tex_exact_k3.py` |
| **7B** | Prefactor/normalization audit | `tests/test_logN_logT_consistency.py` |
| **7C** | Channel projection diagnostic | `run_channel_projection_diagnostics.py` |
| **7D** | Polynomial regression anchor | `tests/test_tex_polynomials_match_paper.py` |

**Implementation order:** 7D → 7B → 7A → 7C (polynomial verification first, then normalization, then main implementation, finally diagnostics)

### 4. Files Created

| File | Purpose |
|------|---------|
| `PLAN_PHASE7_TEX_EXACT.md` | Detailed implementation plan (~350 lines) |
| `docs/PHASE7_PLANNING_SUMMARY.md` | This summary document |

---

## Technical Analysis Summary

### Current State (Phase 5/6)

- **Post-identity operator approach** validated to machine precision (1e-16)
- **U-regularized path** matches post-identity exactly
- **Q-shift identity** mathematically correct but doesn't yield scalar m₁
- **Empirical m₁ = exp(R)+5** works for K=3 benchmarks (~1-3% accuracy)

### Phase 6 Key Finding

The operator shift identity Q(D_α)(T^{-s}F) = T^{-s}Q(1+D_α)F produces:
- Q-shift ratios: 85-127× (κ benchmark), 5× (κ* benchmark)
- Empirical m₁: ~8.68 (κ), ~8.05 (κ*)
- **Dramatic mismatch proves scalar m₁ is not directly derivable from Q-shift**

### Why This Matters

**Before Phase 7:** m₁ is treated as empirical calibration, blocking K>3 extension.

**After Phase 7:** Either:
- (a) TeX-exact path works → m₁ is non-fundamental, clean K>3 extension possible
- (b) TeX-exact path has gaps → we understand exactly what m₁ is compensating for

---

## Estimated Effort

| Phase | New Files | New Tests | Lines (est) |
|-------|-----------|-----------|-------------|
| 7D | 1 | ~15 | 150-200 |
| 7B | 2 | ~15 | 180-250 |
| 7A | 2 | ~20 | 600-800 |
| 7C | 1 | ~5 | 150-200 |
| **Total** | **6** | **~55** | **1080-1450** |

---

## Key TeX References

| Lines | Content | Relevance |
|-------|---------|-----------|
| 1502-1511 | Combined identity (difference quotient → integral) | Core of TeX-exact I₁ |
| 1514-1517 | Q(...)Q(...) operator structure | Eigenvalue application |
| 1521-1523 | Mirror combination at α=β=-R/L | Evaluation point |
| 1529-1533 | Final I₁ formula | Post-identity form |
| 2587-2598 | κ* polynomial coefficients | Polynomial anchor |

---

## Next Steps

1. **Implement Phase 7D** — Polynomial regression tests (low effort, high value)
2. **Implement Phase 7B** — Normalization audit (catches silent mismatches)
3. **Implement Phase 7A** — TeX-exact evaluator (main implementation)
4. **Implement Phase 7C** — Channel diagnostics (answers "is scalar m₁ exact?")

---

## Success Criteria

Phase 7 is **SUCCESSFUL** when we can definitively state:

> "Scalar m₁ is [fundamental/non-fundamental] to TeX-exact evaluation because [evidence]."

This positions the codebase for clean K=4 extension by either:
- Using the TeX-exact path (if it works), OR
- Understanding exactly what calibration the scalar m₁ provides

---

## References

- `PLAN_PHASE7_TEX_EXACT.md` — Full implementation plan
- `docs/DECISIONS.md` — Decisions 1-6 (all locked/implemented)
- `docs/TEX_MIRROR_OPERATOR_SHIFT.md` — Phase 6 mathematical derivation
- `docs/TRUTH_SPEC.md` — TeX interpretation authority
