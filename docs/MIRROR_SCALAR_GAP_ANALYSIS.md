# Mirror Scalar Gap Analysis

**Date:** 2025-12-28
**Purpose:** Trace the derivation of m = exp(R) + (2K-1) and identify the 1.8× gap

---

## The Problem

The production mirror formula uses:
```
m = exp(R) + (2K-1) = exp(1.3036) + 5 = 3.68 + 5 = 8.68
```

But the PRZZ difference quotient identity suggests:
```
DQ scalar limit = ∫₀¹ exp(2Rt) dt = (exp(2R)-1)/(2R) = 4.82
```

**The gap:** 8.68 / 4.82 ≈ **1.80×**

Where does this discrepancy come from?

---

## PRZZ Difference Quotient Identity (Lines 1502-1511)

The identity is:
```
[N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
  = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

At α = β = -R/L (the PRZZ evaluation point):

| Expression | Value |
|------------|-------|
| α + β | -2R/L |
| T^{-(α+β)} | T^{2R/L} = exp(2R) ≈ 13.6 |
| ∫₀¹ exp(2Rt) dt | (exp(2R)-1)/(2R) ≈ 4.82 |

---

## Four Candidate Scalar Values

| Source | Formula | Value at R=1.3036 |
|--------|---------|-------------------|
| **Production m** | exp(R) + 5 | **8.68** |
| DQ scalar limit | (exp(2R)-1)/(2R) | 4.82 |
| PRZZ prefactor | exp(2R) | 13.6 |
| Mysterious exp(R) | exp(R) | 3.68 |

None of the PRZZ-derivable quantities match the production m = 8.68.

---

## Phase 32's Claim

Phase 32 claims m = exp(R) + (2K-1) is "derived" via:

1. **B/A = 5:** From the unified bracket structure, the ratio B/A = 2K-1 = 5 for K=3
2. **exp(R):** "Comes from T^{-(α+β)} prefactor at α=β=-R"

But this is problematic:

### Issue 1: The "+5" is Circular

Looking at `abd_diagnostics.py`:
```python
# Core decomposition
A = I12_minus
D = I12_plus + I34_plus
B = D + 5 * A  # <-- The "+5" is ASSUMED here!
```

The code DEFINES B = D + 5A, then shows D ≈ 0, concluding B/A ≈ 5.
This is circular - the "+5" was baked into the definition.

### Issue 2: The exp(R) Doesn't Match PRZZ

From PRZZ, the mirror term has prefactor T^{-(α+β)} = exp(2R), not exp(R).

At α = β = -R/L:
- T^{-(α+β)} = T^{2R/L} = exp(2R) ≈ 13.6
- NOT exp(R) ≈ 3.68

So where does the "exp(R)" in m = exp(R) + 5 come from?

---

## Tracing exp(R)

Looking at `unified_bracket_ladder.py` lines 271-277:
```python
# In the unified structure:
# - D = 0 (by construction of difference quotient)
# - B/A = 2K-1 = 5 for K=3

D = 0.0  # By construction
B = (2 * self.K - 1) * A  # = 5A for K=3
```

This is NOT computing D and B - it's SETTING them!

The ladder test then verifies these assumptions produce consistent results,
but doesn't derive them from PRZZ.

---

## The Honest Picture

### What IS Derived

1. **B/A = 2K-1 structure:** The unified bracket approach shows this ratio is stable
   across polynomial variations (P=1,Q=1 through P=PRZZ,Q=PRZZ). This is a
   structural identity that holds for the bracket, not a fitted parameter.

2. **D → 0:** The unified bracket construction makes the "leftover" term D vanish
   analytically. This is a mathematical property of the difference quotient identity.

### What is NOT Derived

1. **exp(R) coefficient:** The claim that "exp(R) comes from T^{-(α+β)} at α=β=-R"
   is incorrect. At that point, T^{-(α+β)} = exp(2R), not exp(R).

2. **The production formula:** m = exp(R) + 5 works empirically but the derivation
   chain has a gap. The factor of 2 difference between exp(R) and exp(2R) is
   unexplained.

---

## Possible Explanations for the Gap

### Hypothesis A: Missing Factor of 2 Somewhere

The ratio exp(2R) / m = 13.6 / 8.68 ≈ 1.57
The ratio m / DQ_limit = 8.68 / 4.82 ≈ 1.80

Neither is exactly 2, but both are close to factors involving 2.

Possible sources:
- Division by (α+β) = -2R/L in the DQ identity
- Symmetrization of the integral
- Missing normalization from PRZZ derivation

### Hypothesis B: Different Evaluation Points

The DQ scalar limit (4.82) is computed at x=y=0.
But the production m (8.68) may include contributions from x,y ≠ 0 through
the polynomial factors.

The ratio 8.68/4.82 ≈ 1.8 might come from:
```
∫₀¹ [polynomial factors] × exp(2Rt) dt / ∫₀¹ exp(2Rt) dt
```

### Hypothesis C: The Formula is Empirically Tuned

The most honest interpretation:
- The STRUCTURE m = exp(R) + (2K-1) may be partially derived
- But the specific values exp(R) (instead of exp(2R) or DQ_limit) and 2K-1
  were found by fitting to PRZZ's c_target
- The ±0.15% accuracy suggests it's not exact, just a good approximation

---

## Impact on κ = 0.5213 Result

### If the Gap is Real (Missing Derivation)

The κ = 0.5213 result uses the empirical m formula. If this formula is
fundamentally wrong (exploiting a calibration to match PRZZ), then:

1. Optimizing polynomials while holding m fixed may find artificial minima
2. The "improvement" may be due to the optimizer compensating for m's inaccuracy
3. The result would not be a valid PRZZ-framework bound

### If the Gap is Understood (Hidden Factor)

If the gap can be traced to a legitimate mathematical factor (e.g., from
integration limits, polynomial normalization, or PRZZ derivation structure),
then:

1. The m formula has a complete derivation (just not documented)
2. κ = 0.5213 is a legitimate result
3. It represents polynomial optimization within the correctly-implemented framework

---

## Resolution Path

To resolve this, we need ONE of:

1. **Mathematical derivation:** Show that exp(R) (not exp(2R)) is the correct
   coefficient from PRZZ's Section 10 limiting passage. This requires careful
   analysis of the L'Hôpital limits and integration order.

2. **Acceptance as calibrated:** Explicitly label m = exp(R) + 5 as an empirical
   approximation that achieves ±0.15% accuracy on both benchmarks. Do not claim
   κ = 0.5213 as a "derived PRZZ bound" - instead call it an "optimized
   computational result within an empirically-calibrated framework."

3. **Alternative formula:** Find a PRZZ-derivable formula that matches the
   production accuracy. Candidates:
   - m = DQ_limit × [some polynomial factor]
   - m = exp(2R) / [some normalization]

---

## Summary

| Component | Status | Source |
|-----------|--------|--------|
| B/A = 2K-1 | **Structural** | Unified bracket identity (ladder tests) |
| D → 0 | **Structural** | DQ identity construction |
| exp(R) coefficient | **Gap** | PRZZ gives exp(2R), not exp(R) |
| m = exp(R) + 5 | **Empirical** | Works within ±0.15%, derivation incomplete |
| g_I1, g_I2 | **Calibrated** | 2-equation fit to benchmarks |

The production formula achieves excellent accuracy (±0.15%) but the derivation
chain from PRZZ to m = exp(R) + (2K-1) has a gap that needs resolution.

---

## Files Referenced

- `src/abd_diagnostics.py` - ABD decomposition (circular "+5")
- `src/unified_bracket_ladder.py` - Ladder tests (assumes D=0, B=5A)
- `docs/PHASE_32_FINDINGS.md` - Claims m derivation
- `docs/DERIVATION_STATUS.md` - Derivation status summary
