# Deep Investigation Summary (2025-12-16)

## The Core Concern

**User's valid concern:** If we're computing a "whole new object," our κ values could be meaningless as bounds on the proportion of zeros on the critical line.

**Key insight:** Our c < PRZZ c implies our κ > PRZZ κ. Since PRZZ optimized polynomials to maximize κ, getting a BETTER κ with their polynomials suggests we're computing something different - and likely **underestimating** the true main term constant.

---

## What We Investigated

### 1. (1,1) Pair R-Sensitivity

**Finding:** (1,1) pair has +13.4% R-sensitivity vs PRZZ target +10.3%.

**Root cause:**
- Raw exp(2Rt) factor: +29% R-sensitivity
- Q² polynomial weighting reduces this to +13%
- No bug found - this is intrinsic formula behavior

### 2. Mirror Combination

**Finding:** Correctly implemented.
- Algebraic prefactor (1+θ(x+y))/θ included
- Contributes ~28% extra to I1 terms
- R-dependence captured through t-integral

### 3. Sign Convention (-1)^{ℓ₁+ℓ₂}

**Finding:** Applied to I1 terms only, not I2/I3/I4.

Pattern in code:
| Pair | I1 prefactor | I2/I3/I4 prefactors |
|------|--------------|---------------------|
| (1,1) | +1 | +1/θ, -1/θ, -1/θ |
| (1,2) | **-1** | +1/θ, -1/θ, -1/θ |
| (2,2) | +1 | +1/θ, -1/θ, -1/θ |

**Tested hypothesis:** Maybe I3/I4 should have (-1)^{ℓ}/θ instead of always -1/θ.
**Result:** Applying this makes c = 1.11 (even smaller), so hypothesis is WRONG.

### 4. Gap as Function of R

**Critical finding:** Gap DECREASES as R increases.

| R | Our c | Factor needed | log(factor)/R |
|---|-------|---------------|---------------|
| 0.8 | 1.26 | 1.263 | 0.291 |
| 1.0 | 1.48 | 1.206 | 0.188 |
| 1.3 | 1.95 | 1.096 | 0.070 |
| 1.4 | 2.14 | 1.056 | 0.039 |

**Implication:** We're missing something that contributes more at small R.

### 5. Per-Pair R-Sensitivity

| Pair | Our change (R: 0.9→1.3) | Contribution to Δc |
|------|-------------------------|-------------------|
| (1,1) | +30.1% | +0.102 |
| (2,2) | +34.8% | +0.326 |
| (2,3) | +29.5% | +0.267 |
| (1,2) | +19.7% | +0.099 |
| (1,3) | -24.2% | -0.085 |
| (3,3) | +23.6% | +0.015 |

**Our total R-sensitivity:** 1.427 (R: 0.9→1.3)
**PRZZ R-sensitivity:** 1.265

We have ~13% excess R-sensitivity.

---

## Hypotheses for the Gap

### Most Likely: Missing Positive Terms

We underestimate c by ~10%, meaning we're missing positive contributions. The gap's R-dependence (larger at small R) suggests the missing term:
- Has positive sign
- Contributes ~0.19 to c at R=1.3
- Has LESS R-dependence than our current terms
- Possibly related to Case C structure but not as simple F_d replacement

### Possible Sources

1. **Additional main term family** - PRZZ may include terms beyond I1-I4 that we haven't identified

2. **Cross-terms from mirror expansion** - When PRZZ combines I(α,β) + T^{-α-β}I(-β,-α) analytically before α=β=-R/L substitution, there may be cross-terms we miss

3. **Different asymptotic stage** - We may be computing at a different point in the asymptotic expansion, including subleading terms with different R-behavior

4. **Feng's specific implementation** - PRZZ line 2566 references matching "Feng's code" - there may be numerical techniques or terms specific to that implementation

---

## Implications for Phase 1

### The Honest Assessment

**If our κ values are invalid (too optimistic):**
- Optimization would maximize the wrong quantity
- Results wouldn't translate to actual zeros on critical line
- Scientific value would be limited

**However:**
- Our implementation passes extensive validation
- Term-level structure matches PRZZ formulas
- The gap is systematic, not random noise

### Recommendation

**Do NOT proceed with Phase 1 optimization** until we either:

1. **Identify and fix the gap source** - Find the missing positive terms or correct computation error

2. **Prove our formula is a valid bound** - Derive from first principles that our computation provides a valid lower bound on κ (just different from PRZZ's)

3. **Accept a documented limitation** - Proceed with optimization but clearly state that results may not match PRZZ's mathematical object

---

## What Would Help

1. **Access to Feng's code or detailed numerical description** - Would allow direct comparison of computation structure

2. **PRZZ per-pair breakdown** - If available, would pinpoint which pairs differ

3. **Mathematical derivation review** - Expert check of whether our I1-I4 structure captures all main terms

4. **Alternative validation** - Independent implementation or comparison with other published results

---

## Files Created

| File | Purpose |
|------|---------|
| `docs/DEEP_INVESTIGATION_2025_12_16.md` | This document |

## Conclusion

The user's concern is valid. We cannot confidently claim our κ values are meaningful bounds without understanding the gap source. The systematic nature of the gap (10% in c, R-dependent) suggests a structural issue rather than implementation bugs.

The prudent path is to pause optimization efforts and focus on understanding the discrepancy.
