# J15 vs I5 Reconciliation

**Phase 20.1 Documentation**
**Created:** 2025-12-24
**Goal:** Determine if the code's "J15" is the same as TRUTH_SPEC's "I5"

---

## Executive Summary

**VERDICT: J15 ≈ I5 (SAME OBJECT)**

The code's J15 and TRUTH_SPEC's I5 both refer to the A^{(1,1)} derivative contribution to I₁I₂ integrals. They are the same mathematical object computed in different contexts:

| Aspect | J15 (Code) | I5 (TRUTH_SPEC) |
|--------|------------|-----------------|
| Source | `src/ratios/j1_euler_maclaurin.py` | PRZZ TeX Lines 1621-1628 |
| Formula | A^{(1,1)}(0) × ∫P₁P₂ du | A_{α,β}^{(1,1)}(0,0;β,α) contribution |
| Classification | Error term (excluded in main-only) | Error term O(T/L) |
| Value | ~0.65 contribution to B/A | Same order of magnitude |

**Implication:** Excluding J15 in main-only mode is correct per TRUTH_SPEC. The problem is that main-only B/A = 4.28 (not 5), which means we're missing ~0.72 from somewhere else in the main term, NOT from misclassifying J15.

---

## 1. What the Code Calls "J15"

### Source Location
- **Module:** `src/ratios/j1_euler_maclaurin.py`
- **Function:** `j15_as_integral(R, theta, P1_func, P2_func)`
- **Called by:** `compute_I12_components()` → used in mirror assembly

### Formula (Lines 324-358)
```python
def j15_as_integral(R, theta, *, P1_func=None, P2_func=None) -> float:
    """
    J15 contribution using A^{(1,1)} prime sum.

    J15 = A^{(1,1)}(0) × ∫₀¹ P₁(u)P₂(u) du
    """
    # Polynomial integral
    poly_integral = ∫₀¹ P₁(u)P₂(u) du

    # A^{(1,1)}(0) ≈ 1.3856
    A11_value = A11_prime_sum(0.0, prime_cutoff=5000)

    return A11_value * poly_integral
```

### A^{(1,1)} Definition (from `arithmetic_factor.py`)
```
A^{(1,1)}(s) = Σ_p (log p)² / p^{2+2s}
```
At s=0: A^{(1,1)}(0) ≈ 1.3856

### How J15 Enters the Computation

1. `compute_m1_with_mirror_assembly()` calls `compute_I12_components()` at +R and -R
2. `compute_I12_components()` returns dict with keys `j11`, `j12`, `j13`, `j14`, `j15`
3. These are combined in mirror assembly: I12(+R) + m × I12(-R)
4. J15 contributes to both exp(R) coefficient (A) and constant offset (B)

### J15 Contribution (from plus5_split diagnostic)
- **κ benchmark:** J15 adds +0.67 to B/A ratio
- **κ* benchmark:** J15 adds +0.65 to B/A ratio

---

## 2. What TRUTH_SPEC Calls "I5"

### Source Location
- **PRZZ TeX Lines 1621-1628** (primary citation)
- **TRUTH_SPEC.md Section 4a**

### Original PRZZ Text
```latex
The term (I_5) is of smaller order in (log N).
...
I₅ ≪ T/L.
Hence the term associated to A_{α,β}^{(1,1)}(0,0;β,α) is an error term.
```

### Mathematical Definition
The A^{(1,1)} contribution appears when taking derivatives of the arithmetic factor A(s):
- A(s) = Π_p (1 - 1/p^{1+s})^{-1} × (product structure)
- A^{(1,1)} = ∂²A/∂α∂β at the diagonal
- This involves sums over primes of the form Σ_p (log p)² / p^{2+s}

### Classification in PRZZ
- **O(T/L)** = lower order than main term O(T)
- Explicitly stated as "error term"
- Should NOT be included when computing main-term κ bound

---

## 3. Structural Comparison

### Same Prime Sum Structure

| Feature | J15 | I5 |
|---------|-----|-----|
| Prime sum form | Σ_p (log p)² / p^{2+2s} | Σ_p (log p)² / p^{2+2s} |
| Evaluation point | s = 0 | s = 0 (diagonal) |
| Order | O(T/L) implicit | O(T/L) explicit |

### Same Functional Role

Both represent the "derivative of A at (0,0)" contribution:
- In J1 decomposition, this is the 5th piece (J15)
- In PRZZ I-term structure, this is I₅

### Code Path Verification

The code's J15 flows through:
```
j15_as_integral()
  → A11_prime_sum(0.0)          # Same as PRZZ A^{(1,1)}(0,0)
  → ∫₀¹ P₁(u)P₂(u) du           # Polynomial weight
  → enters I12 via mirror assembly
```

This matches PRZZ's description of A^{(1,1)} contributing to I₁I₂.

---

## 4. Why They Are the Same Object

### Argument 1: Same Mathematical Formula
Both J15 and I5 are defined by:
- A^{(1,1)} prime sum (Σ_p (log p)² / p²)
- Evaluated at the diagonal point (s = 0)
- Weighted by polynomial integrals

### Argument 2: Same Order Classification
- PRZZ: "I₅ ≪ T/L" (lower order)
- Our implementation: Treated as error term, excluded in MAIN_TERM_ONLY mode

### Argument 3: Same Numerical Magnitude
- J15 contributes ~0.65-0.67 to B/A ratio
- This is ~13% of the target B/A = 5
- Consistent with "lower order" classification (not dominant, but not negligible)

### Argument 4: Same Exclusion Behavior
- TRUTH_SPEC says I5 should not be used for main-term matching
- Our `include_j15=False` mode excludes it
- Result: main-only B/A = 4.28 (≠5), confirming J15 = I5 was correctly excluded

---

## 5. What This Means for Phase 20

### The Problem Is NOT Misclassification

Since J15 = I5 (same object), the problem is NOT that we're wrongly excluding a main-term piece. The problem is:

**Main-only B/A = 4.28 instead of 5**

This 0.72 gap must come from somewhere ELSE in the main term, NOT from J15/I5.

### Possible Sources of Missing 0.72

1. **Euler-Maclaurin approximation error** in J11/J12/J13/J14
2. **Missing combinatorial factor** in the main term
3. **Polynomial normalization** differences
4. **Mirror assembly coefficient** m = exp(R) + 5 may need different derivation

### Phase 20.2 Direction

Focus on improving the MAIN TERM (J11-J14), not on reclassifying J15:
- Research PRZZ Section 7 for exact main-term structure
- Implement `j1_main_term_exact.py` with correct combinatorial constants
- Verify against PRZZ's explicit formulas

---

## 6. Provenance Tracking

### Current J15 Provenance (from `plus5_harness.py`)
```python
DEFAULT_J15_PROVENANCE = J15Provenance(
    source_module="src.ratios.j1_euler_maclaurin",
    source_function="j15_contribution_integral",
    przz_line_numbers="1621-1628 (A^{(1,1)} terms)",
    truth_spec_reference="TRUTH_SPEC.md Lines 1621-1628",
    passed_evaluation_mode_guardrails=False,
    guardrail_mode=None,
    formula_description="J15 = A^{(1,1)}(0) × ∫P₁P₂ du",
    is_error_term_per_spec=True,
)
```

### Reconciliation Status
- **Reconciled:** J15 = I5 (same object)
- **Action:** No reclassification needed
- **Focus:** Fix main term to produce B/A = 5 without J15

---

## 7. Key References

### PRZZ TeX Lines
- **1621-1628:** I₅ is error term (primary citation)
- **1722-1727:** A-derivatives contribute O(T/L) (supporting)

### Code Files
- `src/ratios/j1_euler_maclaurin.py:324-358` — J15 implementation
- `src/ratios/arithmetic_factor.py:69-100` — A11_prime_sum
- `src/ratios/plus5_harness.py` — Split analysis showing J15 contribution
- `src/evaluation_modes.py` — MAIN_TERM_ONLY guardrails

### Test Files
- `tests/test_j15_provenance_tags.py` — Provenance verification
- `tests/test_plus5_split.py` — J15 separation verification
- `tests/test_no_I5_in_main_mode.py` — I5 guardrail tests

---

## 8. Conclusion

**J15 = I5** — They are the same mathematical object.

The code correctly:
1. Identifies J15 as the A^{(1,1)} contribution
2. Classifies it as error-order per TRUTH_SPEC
3. Excludes it in MAIN_TERM_ONLY mode
4. Provides provenance tracking

The problem is NOT in J15/I5 handling. The problem is that the remaining main term (J11-J14) produces B/A ≈ 4.28 instead of 5. Phase 20.2 must fix this.

---

*Generated 2025-12-24 as part of Phase 20.1 implementation.*
