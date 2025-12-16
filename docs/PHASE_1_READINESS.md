# Phase 1 Readiness Assessment

**Date:** 2025-12-16
**Status:** Ready with caveats

---

## Summary of Phase 0 Investigation

### What Works

1. **Term-level implementation is validated**
   - First-principles oracle matches DSL for (1,1) pair
   - I₃/I₄ prefactor (-1/θ) confirmed via finite differences
   - Q-operator arguments match PRZZ TeX lines 1514-1517
   - Polynomial values match PRZZ published coefficients

2. **Multi-variable structure is correctly implemented**
   - (ℓ₁, ℓ₂) pairs use ℓ₁ + ℓ₂ formal variables
   - Derivative extraction via multi-variable Taylor series
   - 439 tests passing

3. **Mirror combination correctly handled**
   - Algebraic prefactor (1 + θ(x+y))/θ included
   - R-dependence comes from exp(2Rt) factor through t-integral
   - Q² factor reduces R-sensitivity from +29% to +13%

### The Core Gap

| Metric | Our Value | PRZZ Value | Difference |
|--------|-----------|------------|------------|
| c (R=1.3036) | 1.950 | 2.137 | -8.8% |
| κ (R=1.3036) | 0.488 | 0.417 | +17.0% |
| R-sensitivity | 18.7% | 10.3% | +81% excess |

### The Paradox

**Our c is LOWER than PRZZ's, giving HIGHER κ with the same polynomials.**

Since κ = 1 - log(c)/R, smaller c means larger κ. If both computations are valid lower bounds on the proportion of zeros on the critical line, ours would be BETTER.

But PRZZ optimized polynomials specifically to maximize their κ. How could we achieve better κ with their polynomials?

### Most Likely Explanation

We are computing a **related but different mathematical object** than PRZZ's published main-term constant. This could be because:

1. **Stage mismatch**: We evaluate at α=β=-R/L already substituted; PRZZ may combine terms analytically before this substitution
2. **Missing positive terms**: PRZZ may include additional main-term families we haven't identified
3. **Feng's specific code path**: PRZZ line 2566 references matching "Feng's code" specifically

---

## Phase 1 Options

### Option A: Accept and Optimize

**Proceed with Phase 1 optimization treating our formula as a valid (different) κ bound.**

**Pros:**
- Our κ = 0.488 is already beyond the 0.42 goal
- Optimization could potentially push κ higher
- Infrastructure is ready

**Cons:**
- Unknown if our object is a valid κ bound
- Cannot claim to reproduce PRZZ
- Mathematical interpretation unclear

### Option B: Continue Investigation

**Defer Phase 1 until we understand the gap source.**

**Pros:**
- Would ensure mathematical rigor
- Could potentially match PRZZ exactly

**Cons:**
- May require significant effort (Case C multi-variable integration)
- May never fully resolve (Feng's code not available)

### Option C: Document and Publish Current State

**Write up findings as independent contribution.**

**Pros:**
- Valuable technical documentation
- Could be useful to other researchers
- No false claims about PRZZ reproduction

**Cons:**
- Doesn't advance the 0.42 goal
- Leaves open questions

---

## Recommendation

**Option A with transparency:** Proceed with Phase 1 optimization while clearly documenting that we compute a different object than PRZZ.

**Rationale:**
1. Our κ = 0.488 suggests potential value in optimization
2. The infrastructure is complete and tested
3. We can note the gap and continue investigation in parallel
4. If our formula IS valid, optimizing it is scientifically interesting

**Key Caveat:**
Any results should be presented as "optimizing the Levinson-type bound computed by our specific term assembly" rather than "reproducing PRZZ optimization."

---

## Technical Readiness Checklist

### Ready for Phase 1

- [x] Polynomial module with constraint enforcement
- [x] Multi-variable series engine
- [x] All K=3 pair evaluations (I1-I4 for 6 pairs)
- [x] Quadrature convergence validated
- [x] Per-pair breakdown available

### Known Limitations

- [ ] Gap vs PRZZ unresolved (c=1.95 vs 2.14)
- [ ] R-sensitivity higher than PRZZ (18.7% vs 10.3%)
- [ ] Mathematical interpretation of our object uncertain

### Optimization Infrastructure Needed

1. **Parameter space definition**
   - P1, P2, P3 polynomial coefficients
   - Q polynomial coefficients
   - Constraint enforcement (Q(0)=1, P(1)=1, etc.)

2. **Optimization algorithm**
   - Gradient-based (SciPy L-BFGS-B with bounds)
   - Multiple random restarts
   - Alternating minimization option

3. **R-sweep capability**
   - Evaluate κ(R) for R ∈ [1.2, 1.4]
   - Find optimal R for current polynomials

---

## Files Summary

| Document | Purpose |
|----------|---------|
| `docs/PHASE_1_READINESS.md` | This assessment |
| `docs/HANDOFF_SUMMARY.md` | Technical investigation summary |
| `docs/ORACLE_INVESTIGATION_2025_12_16.md` | F_d kernel analysis |
| `CLAUDE.md` | Project guidelines (updated status) |
