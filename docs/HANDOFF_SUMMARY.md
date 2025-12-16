# PRZZ Reproduction Audit — Complete Handoff Summary

**Date:** 2025-12-16
**Status:** Structural mismatch identified; term-level validation complete
**Tests:** 445 passing

---

## TL;DR for Fresh Session

We have a mathematically validated pipeline that computes **a different object** than PRZZ's published main-term constant. Our c ≈ 1.95 vs PRZZ c ≈ 2.14 (at R=1.3036). The gap is **R-dependent** (different correction factors at different R values), which rules out simple normalization fixes. The computed κ values should **NOT** be interpreted as zeta-zero proportion bounds until equivalence is proven.

---

## 1. Project Goal

Reproduce and extend PRZZ (2019) computation of κ, the proportion of Riemann zeta zeros on the critical line, using Levinson/Feng-style mollifiers.

**The bound:**
```
κ ≥ 1 - (1/R)·log(c)
```

where c is the main-term constant from the mollified mean square.

**Targets:**
- Benchmark 1: R = 1.3036, κ = 0.417293962, c = 2.13745440613217
- Benchmark 2: R = 1.1167, κ* = 0.407511457, c = 1.93795241

---

## 2. Current Numerical State

### Computed vs Target (using PRZZ polynomials)

| Benchmark | R | Our c | PRZZ c | Factor Needed | Our κ | PRZZ κ |
|-----------|------|-------|--------|---------------|-------|--------|
| 1 | 1.3036 | 1.9501 | 2.1375 | 1.096 | 0.488 | 0.417 |
| 2 | 1.1167 | 1.6424 | 1.9380 | 1.180 | 0.556 | 0.408 |

### The Critical Finding

**The correction factor is R-dependent (1.096 vs 1.180).** This definitively rules out any global multiplicative fix and proves we are computing a structurally different object.

### Per-Pair Breakdown (R = 1.3036)

| Pair | Contribution |
|------|-------------|
| c₁₁ | +0.4419 |
| c₁₂ | -0.2009 |
| c₁₃ | -0.2179 |
| c₂₂ | +1.2611 |
| c₂₃ | +0.5861 |
| c₃₃ | +0.0798 |
| **Total** | **1.9501** |

---

## 3. What Has Been Validated (LOCKED)

### 3.1 I₅ Is Lower-Order (Do Not Use for Matching)

PRZZ explicitly bounds I₅ ≪ T/L (lines 1621-1628). It is an error term, not part of the leading constant.

**Implementation:** `mode="main"` excludes I₅; using I₅ to hit targets masks bugs.

### 3.2 I₃/I₄ Prefactor Is -1/θ

PRZZ formula has (1+θx)/θ, which at x=0 gives 1/θ. Combined with the negative sign structure, the prefactor is **-1/θ**.

**Validation:** Finite-difference oracle confirmed to ~1e-13 relative error.

### 3.3 Q-Operator Substitution Is Correct

The Q arguments are:
- arg_α = t + θt·x + θ(t-1)·y
- arg_β = t + θ(t-1)·x + θt·y

**Validation:** Q-operator oracle confirmed symmetry and structure.

### 3.4 (1-u) Powers Match PRZZ

| Term | Power |
|------|-------|
| I₁ | (1-u)^{ℓ₁+ℓ₂} |
| I₂ | none |
| I₃ | (1-u)^{ℓ₁} |
| I₄ | (1-u)^{ℓ₂} |

### 3.5 Multi-Variable Derivative Extraction Is Correct

The DSL correctly handles:
- (1,1): 2 vars (x₁, y₁), derivatives ∂²/∂x₁∂y₁
- (2,2): 4 vars (x₁, x₂, y₁, y₂), derivatives ∂⁴/∂x₁∂x₂∂y₁∂y₂
- Cross-terms from algebraic prefactors

**Critical methodological point:** Any oracle for ℓ>1 pairs MUST preserve the full multi-variable structure. A simplified 2-variable oracle for (2,2) is **invalid**.

### 3.6 Mirror Combination Structure

The algebraic prefactor (1+θ(x+y))/θ is included and contributes cross-terms under differentiation. The R-dependence from the mirror combination is captured through the t-integral: ∫₀¹ exp(2Rt)dt.

---

## 4. Disproven Hypotheses (Do Not Revisit)

### 4.1 Global Factor (1+θ/6) — DEAD

Matched Benchmark 1 but failed Benchmark 2. The two-benchmark test is now a permanent falsification gate.

### 4.2 Q Substitution Error — DEAD

Oracle validated.

### 4.3 I₅ Calibration — DEAD

Architecturally wrong; I₅ is lower-order.

### 4.4 Naive Case C Kernel Replacement — DEAD

Replacing P(u) with F_d kernel in our structure makes c ≈ 0.57 (much worse). The kernel must be integrated at the correct stage, not as a post-hoc multiplicative correction.

### 4.5 Sign Convention (-1)^ℓ/θ for I₃/I₄ — DEAD

Testing this hypothesis made c even smaller (≈1.11). The current -1/θ for all I₃/I₄ is correct.

---

## 5. The Core Problem: Stage/Assembly Mismatch

### The Paradox

Our c < PRZZ c implies our κ > PRZZ κ. But PRZZ **optimized** their polynomials to maximize κ. If we could achieve higher κ with their polynomials, they would have found it.

**Conclusion:** We are computing a different mathematical object, most likely:
1. A subcomponent of PRZZ's full constant
2. The same formulas evaluated at a different asymptotic stage
3. Missing additional main-term families

### Evidence for Stage Mismatch

1. **R-dependent gap** — Not fixable by any constant factor
2. **R-sensitivity mismatch** — Our c ratio is 1.187, PRZZ is 1.103 (we have ~8% excess R-sensitivity)
3. **Gap decreases with R** — At R=1.4, factor needed is only 1.056; at R=0.8, it's 1.263

### What This Means for κ Interpretation

**WARNING:** The computed κ values (0.488, 0.556) are **NOT** valid zeta-zero proportion bounds. They are outputs of a surrogate objective that has not been proven equivalent to PRZZ's mean-square constant.

---

## 6. Case C Status (Not Dead, But Complex)

PRZZ uses Case C machinery (auxiliary a-integral, F_d kernel) for ω>0 pieces. Our attempts to patch this failed because:

1. The oracle used for F_d comparisons was structurally invalid (wrong variable count, wrong derivative order)
2. Multiplicative post-corrections don't work; the kernel must be integrated at the correct stage

**Current position:** Case C investigation is not closed, but requires proper multi-variable implementation, not simplified oracles.

---

## 7. Key PRZZ TeX References

| Concept | Lines | Status |
|---------|-------|--------|
| κ/c definition | 286-289 | Used |
| Mirror combination | 1502-1511 | Validated |
| Q-arguments | 1514-1517 | Validated |
| I₃ formula | 1551-1564 | Implemented |
| I₄ formula | 1577-1602 | Implemented |
| I₅ error bound | 1621-1628 | Confirmed lower-order |
| "Same process" for ℓ>1 | 1726 | Key ambiguity |
| ω definition | 2301-2310 | Implemented |
| Case C a-integral | 2369-2384 | Not fully integrated |
| Variable rescaling x→x log N | 2309 | Needs verification |
| "Matched Feng's code" | 2566 | Key clue |

---

## 8. Repository Structure

```
przz-extension/
├── CLAUDE.md                  # Project guidelines
├── src/
│   ├── polynomials.py         # PRZZ polynomial loading
│   ├── quadrature.py          # Gauss-Legendre integration
│   ├── series.py              # Multi-variable Taylor series
│   ├── term_dsl.py            # Term data model
│   ├── terms_k3_d1.py         # K=3 term definitions
│   ├── evaluate.py            # Main computation (mode="main" vs "with_error_terms")
│   ├── fd_oracle.py           # Finite-difference validation
│   └── przz_22_oracle.py      # (2,2) pair oracle (educational, structurally limited)
├── tests/                     # 445 tests
├── docs/
│   ├── HANDOFF_SUMMARY.md     # This file
│   ├── TRUTH_SPEC.md          # Benchmark parameters and targets
│   ├── STRUCTURE_COMPARISON.md # Implementation vs PRZZ mapping
│   ├── GAP_ANALYSIS_SUMMARY.md # R-sensitivity analysis
│   └── DEEP_INVESTIGATION_2025_12_16.md # Latest findings
└── data/
    └── przz_parameters.json   # Published PRZZ polynomial coefficients
```

---

## 9. Recommended Next Steps

### Track 1: Surrogate Optimization (If Goal Is Exploration)

Proceed with optimization but:
- **Label results as "surrogate objective optimization"** — not κ bounds
- Maintain two-benchmark gate as regression check
- Keep `mode="main"` (I₅ excluded)
- Do not advertise κ as zeta-zero proportion

### Track 2: Reconciliation (If Goal Is PRZZ Equivalence)

**Priority order:**

#### 2a. Finite-T Numerical Validation (Highest Value, Hardest)
Numerically compute (1/T)∫|Vψ|² dt for moderate T using actual zeta evaluations. Compare trend vs asymptotic constant as T grows. This would definitively answer "are we computing the right object?"

#### 2b. Assembly Order Investigation (Most Tractable)
Re-derive from α,β level, checking:
1. When exactly does PRZZ substitute α=β=-R/L?
2. Does analytic combination before substitution produce cross-terms we miss?
3. Variable rescaling: does "x → x log N" (line 2309) affect our setup?
4. What are the "lot more" terms mentioned at line 1726?

#### 2c. External Information
- Search for Feng's original code or detailed numerical procedure
- Check if PRZZ authors have published per-pair breakdowns
- Consider direct contact with PRZZ authors

---

## 10. Key Methodological Rules (For Future Sessions)

1. **Two-benchmark gate is mandatory** — Any proposed fix must improve BOTH R=1.3036 AND R=1.1167
2. **I₅ is forbidden in main mode** — Using it to match targets masks bugs
3. **Multi-variable structure must be preserved** — No collapsing (2,2) to 2 variables
4. **Do not claim κ as zeta-zero bound** — Until equivalence is proven
5. **Finite-difference oracles must match DSL variable/derivative structure**

---

## 11. Commands for Quick Verification

```bash
# Run all tests
python3 -m pytest

# Compute current c values
python3 -c "
from src.evaluate import evaluate_c_full, compute_kappa
from src.polynomials import load_przz_polynomials
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
result = evaluate_c_full(4/7, 1.3036, 60, polys, mode='main')
print(f'c = {result.total:.10f}')
print(f'kappa = {compute_kappa(result.total, 1.3036):.10f}')
"

# Run two-benchmark test
python3 src/second_benchmark.py
```

---

## 12. Summary Statement for New Session

> The PRZZ reproduction pipeline is **internally consistent and term-level validated**, but computes a **different object** than PRZZ's published main-term constant. The gap (~10% in c) is **R-dependent**, ruling out simple fixes. The resulting κ values should **not** be interpreted as zeta-zero bounds. Next steps are either (a) proceed with surrogate optimization under explicit limitations, or (b) investigate assembly-order differences to achieve true equivalence.
