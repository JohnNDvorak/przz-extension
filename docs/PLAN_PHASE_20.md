# Phase 20 Implementation Plan: Derive Main Term Without Error Contamination

**Created:** 2025-12-24
**Goal:** Make MAIN_TERM_ONLY produce B/A = 2K-1 without J₁,₅ compensation

## Executive Summary

Phase 19 revealed a **critical finding**: the +5 signature requires J₁,₅ (error term) to achieve B/A ≈ 5. Main-only gives B/A ≈ 4.28, with J₁,₅ adding +0.67. This means we're either:

1. Missing a main-term contribution
2. Misclassifying J₁,₅
3. Using an unfaithful Euler-Maclaurin approximation

Phase 20 systematically investigates and fixes this.

---

## Phase 20.0: Enhanced Delta Tracking

### Goal
One command tells whether we're closer to derived main term.

### Files to Create/Modify

**Extend:** `scripts/run_delta_report.py`
- Add `--plus5-split` flag for detailed main vs error breakdown
- Print per-piece contributions (j11-j15) to both A and B
- Show zeta mode used (Laurent vs numeric)
- Include precision metrics

**New Test:** `tests/test_plus5_delta_track_smoke.py`
```python
# Tests:
# 1. Script returns dict with required keys
# 2. MAIN_TERM_ONLY forbids J15 access
# 3. Per-piece breakdown sums correctly
# 4. Both benchmarks produce results
```

### Definition of Done
- `python scripts/run_delta_report.py --plus5-split` prints comprehensive breakdown
- Test file passes with 5+ tests

---

## Phase 20.1: J₁,₅ vs I₅ Reconciliation

### Goal
Determine if the object we forbid (I₅) is the same as J₁,₅ contributing +0.67.

### Files to Create

**New:** `docs/J15_VS_I5_RECONCILIATION.md`
- What `plus5_harness` calls "J15" (code path, formula)
- What `TRUTH_SPEC` calls "I5" (Lines 1621-1628)
- Symbolic/structural comparison
- Verdict: same object or different?

**Modify:** `src/ratios/plus5_harness.py`
- Add provenance tags to split output:
  - Module + function that produced J15 contribution
  - Whether it passed through evaluation_modes guardrails
  - Source formula reference (PRZZ line numbers)

**New Test:** `tests/test_j15_provenance_tags.py`
```python
# Tests:
# 1. Result includes 'j15_provenance' field
# 2. Provenance includes module name
# 3. Provenance includes function name
# 4. Provenance stable across runs
```

### Definition of Done
- Clear documentation stating whether J15 == I5
- Provenance metadata in Plus5SplitResult
- Test file passes

---

## Phase 20.2: Fix Main Term to Produce B/A = 2K-1

### Goal
MAIN_TERM_ONLY should hit B/A ≈ 5 (K=3) without J₁,₅.

### Hypothesis
The Euler-Maclaurin extraction in `j1_euler_maclaurin.py` is missing part of the true main-term asymptotics. The combinatorial constant is undercounted, and J₁,₅ compensates.

### Files to Create

**New:** `src/ratios/j1_main_term_exact.py`
```python
"""
Exact main-term asymptotic extraction following PRZZ structure.

Unlike j1_euler_maclaurin.py (simplified approximation), this implements
the residue/series + leading asymptotics as the paper uses.

Key difference: captures the full combinatorial constant that produces
the "+5" factor without needing J15.
"""

def j11_main_term_exact(R, theta, polys) -> float:
    """J11 via exact main-term extraction."""

def j12_main_term_exact(R, theta, polys, laurent_mode) -> float:
    """J12 via exact main-term extraction."""

def j13_main_term_exact(R, theta, polys, laurent_mode) -> float:
    """J13 via exact main-term extraction."""

def j14_main_term_exact(R, theta, polys, laurent_mode) -> float:
    """J14 via exact main-term extraction."""

def compute_I12_main_term_exact(R, theta, polys, laurent_mode) -> Dict:
    """I12 components via exact extraction (no J15)."""

def compute_m1_exact_assembly(theta, R, polys, K, laurent_mode) -> Dict:
    """Mirror assembly using exact main-term extraction."""
```

**Modify:** `src/ratios/plus5_harness.py`
- Add `extraction_mode` parameter: "euler_maclaurin" (legacy) or "exact" (new)
- Wire to j1_main_term_exact.py when mode="exact"

### Tests to Create (Initially xfail)

**New:** `tests/test_plus5_main_only_gate.py`
```python
@pytest.mark.xfail(reason="Phase 20.2: exact main-term not yet implemented")
class TestMainOnlyGate:
    def test_kappa_main_only_hits_5(self):
        """κ: B/A main-only should be within 1% of 5."""
        result = compute_plus5_signature_split("kappa", extraction_mode="exact")
        assert abs(result.B_over_A_main_only - 5.0) / 5.0 < 0.01

    def test_kappa_star_main_only_hits_5(self):
        """κ*: B/A main-only should be within 1% of 5."""
        result = compute_plus5_signature_split("kappa_star", extraction_mode="exact")
        assert abs(result.B_over_A_main_only - 5.0) / 5.0 < 0.01
```

**New:** `tests/test_plus5_independence_from_benchmark.py`
```python
class TestBenchmarkIndependence:
    def test_main_only_similar_across_benchmarks(self):
        """B/A main-only should be benchmark-independent (structural constant)."""
        kappa = compute_plus5_signature_split("kappa", extraction_mode="exact")
        kappa_star = compute_plus5_signature_split("kappa_star", extraction_mode="exact")

        # Should be within 5% of each other
        ratio = kappa.B_over_A_main_only / kappa_star.B_over_A_main_only
        assert 0.95 < ratio < 1.05
```

### Research Required

Before implementing `j1_main_term_exact.py`, need to:

1. **Re-read PRZZ Section 7** (main-term extraction)
   - Identify exact residue structure
   - Find where combinatorial constants arise

2. **Compare with current j1_euler_maclaurin.py**
   - What approximations were made?
   - What terms were dropped?

3. **Check TRUTH_SPEC Lines 1621-1628**
   - Exact definition of A^{(1,1)} error term
   - Why it's classified as O(T/L)

### Definition of Done
- `j1_main_term_exact.py` exists with full implementation
- `test_plus5_main_only_gate.py` passes (remove xfail)
- Main-only B/A within 1% of 5 for both benchmarks

---

## Phase 20.3: Revisit exp(R) Coefficient (After +5 Derived)

### Goal
Once constant term is correct, investigate remaining exp(R) mismatch.

### Prerequisites
- Phase 20.2 complete (main-only B/A ≈ 5)

### Files to Modify

**Extend:** `scripts/run_delta_report.py`
- Report implied weights comparison
- Show residual amplitude spans (κ vs κ*)
- Identify if single global factor missing

**New:** `src/ratios/amplitude_analysis.py`
```python
def analyze_exp_coefficient_residual(benchmark) -> Dict:
    """Analyze why exp(R) coefficient differs from derived expectation."""

def compare_amplitude_across_benchmarks() -> Dict:
    """Compare residual patterns between κ and κ*."""
```

### Definition of Done
- Clear documentation of exp(R) coefficient status
- Diagnostic showing whether single factor explains both benchmarks

---

## Phase 20.4: Safe evaluate.py Refactoring

### Goal
Start modularizing the 6,700-line file while maintaining correctness.

### Approach
Move **pure helpers** first, keep evaluate.py as thin façade.

### Files to Create

**New Package:** `src/evaluator/`
```
src/evaluator/
├── __init__.py
├── result_types.py      # EvaluationResult, TermResult dataclasses
├── solver_utils.py      # solve_two_weight_operator, etc.
├── diagnostics.py       # Reporting utilities
└── facade.py            # Re-exports for backwards compatibility
```

### Refactoring Rules

1. **Every move must pass `tests/test_evaluate_snapshots.py`**
2. Keep original function signatures in evaluate.py
3. Add deprecation warnings for direct evaluate.py imports
4. Document each extracted module

### Definition of Done
- `src/evaluator/` package exists
- At least 3 helper modules extracted
- All snapshot tests still pass
- No functionality changes

---

## Implementation Order

```
Phase 20.0 (1-2 hours)
    ├── Extend run_delta_report.py
    └── Add smoke tests

Phase 20.1 (2-3 hours)
    ├── Write reconciliation doc
    ├── Add provenance tags
    └── Add provenance tests

Phase 20.2 (4-8 hours) ← CRITICAL PATH
    ├── Research PRZZ main-term structure
    ├── Implement j1_main_term_exact.py
    ├── Wire to plus5_harness.py
    └── Pass main-only gate tests

Phase 20.3 (2-3 hours, after 20.2)
    ├── Extend delta report
    └── Analyze residual

Phase 20.4 (parallel, ongoing)
    ├── Create evaluator package
    └── Extract helpers incrementally
```

---

## Success Criteria

### Minimum Viable Phase 20
- [ ] Phase 20.0: Enhanced tracking operational
- [ ] Phase 20.1: J15/I5 reconciliation documented
- [ ] Phase 20.2: Main-only B/A within 5% of target

### Full Phase 20 Success
- [ ] Main-only B/A within 1% of 5 for both benchmarks
- [ ] J₁,₅ no longer needed for +5 signature
- [ ] exp(R) coefficient residual characterized
- [ ] At least one evaluate.py module extracted

---

## Risk Assessment

### High Risk
- **Phase 20.2 may require significant PRZZ re-reading** - The exact main-term structure isn't fully understood yet

### Medium Risk
- **J15 may actually be necessary** - If reconciliation shows J15 is NOT the same as forbidden I5, the approach changes

### Low Risk
- **Refactoring (20.4)** - Snapshot protection makes this safe
- **Tracking (20.0)** - Purely additive, no behavioral changes

---

## Files Summary

### New Files (7)
1. `tests/test_plus5_delta_track_smoke.py`
2. `tests/test_j15_provenance_tags.py`
3. `tests/test_plus5_main_only_gate.py`
4. `tests/test_plus5_independence_from_benchmark.py`
5. `docs/J15_VS_I5_RECONCILIATION.md`
6. `src/ratios/j1_main_term_exact.py`
7. `src/ratios/amplitude_analysis.py`

### Modified Files (2)
1. `scripts/run_delta_report.py`
2. `src/ratios/plus5_harness.py`

### New Package (1)
1. `src/evaluator/` (4-5 modules)

---

## The North Star Experiment

Before writing more math, run this diagnostic:

```python
from src.ratios.plus5_harness import compute_plus5_signature_split

for bench in ["kappa", "kappa_star"]:
    result = compute_plus5_signature_split(bench)
    print(f"\n{bench.upper()}:")
    print(f"  Main-only B/A: {result.B_over_A_main_only:.4f}")
    print(f"  With-error B/A: {result.B_over_A_with_error:.4f}")
    print(f"  J15 delta: {result.j15_contribution_ratio:+.4f}")
    print(f"  Gap from 5: {(result.B_over_A_main_only - 5)/5 * 100:+.2f}%")
```

**Goal:** Replace the missing ~0.72 in main-only with the correct main-term contribution, so J15 becomes unnecessary and can remain forbidden.
