# PRZZ vs Our Implementation: Rigorous Audit

**Date:** 2025-12-29
**Phase:** 59.3 (Critical Gap Analysis)
**Triggered by:** Realization that we never truly implemented PRZZ's method

---

## The Problem Statement

We claim to have "reproduced PRZZ's κ = 0.417" but:
- Our method (split-channel assembly) differs from PRZZ's method (DQ identity)
- Phase 59 showed 85% error between unified and split-channel
- The DQ identity doesn't even hold numerically in our tests (ratio 1.5x, not 1.0)

**We achieved a numerical coincidence, not a methodological reproduction.**

---

## PRZZ's Computational Flow (from TeX references)

### Step 1: Define the Mollified Mean Square
**TeX lines ~200-300**

```
M(s) = ∑_{m,n} mollifier_coeffs × (m/n)^s × ...
```

The goal is to compute the asymptotic expansion of the mollified mean square.

### Step 2: Integral Decomposition
**TeX lines ~1400-1500**

PRZZ decomposes the computation into integrals I₁, I₂, I₃, I₄, I₅ where:
- I₁, I₂: Main terms requiring "mirror" handling
- I₃, I₄: Cross terms (no mirror)
- I₅: Error term (I₅ ≪ T/L, negligible)

### Step 3: The Mirror Identity (CRITICAL)
**TeX lines 1502-1511**

```
[N^{αx+βy} - T^{-α-β}N^{-βx-αy}] / (α+β)
    = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

At α = β = -R/L = -Rθ:
```
LHS = [Direct term - exp(2R) × Mirror term] / (-2Rθ)
RHS = bracket (unified t-integral)
```

**Key question: How does PRZZ use this identity to compute c?**

### Step 4: Asymptotic Extraction
**TeX lines ~1600-1800**

PRZZ extracts the leading constant c from the asymptotic expansion.

**We don't know the exact procedure here.**

### Step 5: κ Computation
**TeX lines 286-289**

```
κ = 1 - log(c) / R
```

This part we understand and implement correctly.

---

## Our Computational Flow

### Step 1: Load Polynomials
- Load P₁, P₂, P₃, Q from JSON (coefficients from PRZZ)
- ✓ Matches PRZZ

### Step 2: Compute Split-Channel Integrals
```python
S12_plus = compute_S12(+R)   # "Direct" term
S12_minus = compute_S12(-R)  # "Mirror" term
S34_plus = compute_S34(+R)   # Cross terms
```

**Question: Does S12(±R) match what PRZZ computes?**

### Step 3: Mirror Assembly (EMPIRICAL)
```python
m = exp(R) + 5  # Where does this come from?
c = S12_plus + m × S12_minus + S34_plus
```

**This is NOT derived from PRZZ's mirror identity!**

### Step 4: κ Computation
```python
kappa = 1 - log(c) / R
```
- ✓ Matches PRZZ formula

---

## The Divergence Point

| Step | PRZZ | Ours | Match? |
|------|------|------|--------|
| Polynomials | P₁,P₂,P₃,Q | Same | ✓ |
| κ formula | 1 - log(c)/R | Same | ✓ |
| I₁/I₂ definition | TeX 1400+ | ? | **UNKNOWN** |
| Mirror handling | DQ identity | m × assembly | **DIFFERENT** |
| c extraction | Asymptotic | Direct sum | **UNKNOWN** |

---

## Critical Unknown: How Does PRZZ Go From Mirror Identity to c?

The mirror identity is:
```
[Direct - exp(2R)×Mirror] / (-2Rθ) = bracket
```

This is a SUBTRACTION divided by a factor.

But c is supposed to be ~2.14 (a positive sum).

**Question: What transformation converts the bracket into c?**

Possibilities:
1. PRZZ computes bracket, then applies some formula to get c
2. PRZZ uses the identity to simplify, but ultimately does a different assembly
3. Our S12(±R) don't match what PRZZ calls "Direct" and "Mirror"

---

## Numerical Evidence of Divergence

### Phase 59 Results

| Quantity | κ benchmark | κ* benchmark |
|----------|-------------|--------------|
| S12_split | ~2.14 | ~1.94 |
| S12_unified | ~0.88 | ~0.67 |
| Relative error | **85%** | **85%** |

### DQ Identity Check

| Quantity | κ | κ* |
|----------|---|-----|
| DQ LHS | 1.47 | 1.10 |
| bracket (RHS) | 0.995 | 0.681 |
| Ratio | 1.48 | 1.62 |

**The DQ identity doesn't even hold numerically!**

---

## What We Need to Investigate

### Investigation 1: What ARE S12_plus and S12_minus?

Our code computes them as:
```python
S12 = I1 + I2  # Sum of integrals
```

**Question: Is this what PRZZ means by the terms in the mirror identity?**

### Investigation 2: PRZZ's c Extraction Procedure

Find in PRZZ TeX:
- Where do they write "c = ..."?
- What is the exact formula for c in terms of their integrals?
- How does the mirror identity feed into this?

### Investigation 3: Normalization Conventions

- Does PRZZ use normalized or unnormalized integrals?
- Are there factors of θ, R, L that we're missing?
- Does PRZZ work per-pair or sum all pairs first?

---

## The Audit Checklist

| Question | Status | Where to Look |
|----------|--------|---------------|
| Does our I₁ match PRZZ's I₁? | UNKNOWN | TeX 1400-1500 |
| Does our I₂ match PRZZ's I₂? | UNKNOWN | TeX 1400-1500 |
| How does PRZZ extract c? | UNKNOWN | TeX 1600-1800 |
| What is PRZZ's exact assembly formula? | UNKNOWN | TeX ~1700 |
| Does PRZZ use m = exp(R)+5? | NO (our invention) | - |
| Does PRZZ use g-corrections? | NO (our invention) | - |
| Why does our formula work? | UNKNOWN | - |

---

## The Honest Assessment

### What We Know
1. Our formula produces κ ≈ 0.428 (2.5% from PRZZ's 0.417)
2. With g-corrections, we match to <0.0003%
3. The formula transfers across polynomial sets (0.12% stability)

### What We DON'T Know
1. Whether our S12(±R) match PRZZ's Direct/Mirror terms
2. How PRZZ extracts c from the mirror identity
3. Why m = exp(R)+5 works
4. Whether our method is equivalent to PRZZ's method

### The Gap
**We reverse-engineered a formula that matches PRZZ's output, but we don't understand PRZZ's actual method.**

---

## Recommended Next Steps

### Priority 1: Deep PRZZ TeX Dive
- Find exact c extraction formula in PRZZ
- Trace from mirror identity to c step by step
- Document every intermediate expression

### Priority 2: Compute PRZZ Intermediate Values
- If PRZZ reports any intermediate integral values, compare to ours
- Even if they don't, derive what their intermediate values should be

### Priority 3: Test Hypotheses
- Hypothesis A: Our S12 ≠ PRZZ's Direct/Mirror (different definitions)
- Hypothesis B: Missing normalization factor connects them
- Hypothesis C: They're genuinely different methods that happen to agree

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/audit_przz_i1_definition.py` | Compare our I₁ to PRZZ definition |
| `scripts/audit_przz_c_extraction.py` | Trace PRZZ's c formula |
| `scripts/audit_normalization.py` | Check all normalization conventions |

---

## Phase 59.3 Audit Results (2025-12-29)

### Numerical Findings

**For κ benchmark (R=1.3036):**
| Quantity | Value |
|----------|-------|
| S12(+R) | 0.7975 |
| S12(-R) | 0.2201 |
| S34(+R) | -0.6002 |
| m_needed | 8.814 |
| exp(2R) | 13.56 |
| exp(R)+5 | 8.68 |
| F = m_needed/exp(2R) | 0.650 |

**For κ* benchmark (R=1.1167):**
| Quantity | Value |
|----------|-------|
| S12(+R) | 0.6146 |
| S12(-R) | 0.2164 |
| S34(+R) | -0.4434 |
| m_needed | 8.163 |
| exp(2R) | 9.33 |
| exp(R)+5 | 8.05 |
| F = m_needed/exp(2R) | 0.875 |

### The Central Mystery

PRZZ TeX says the mirror weight is T^{-α-β} = exp(2R), but:
- Using exp(2R) gives c = 3.18, κ = 0.11 (WRONG)
- Using exp(R)+5 gives c = 2.11, κ = 0.43 (CORRECT)

**The ratio F = m_needed / exp(2R) is R-dependent (0.65 vs 0.87).**

### Possible Explanations

1. **Our S12(-R) ≠ PRZZ's mirror integral**
   - Different normalization conventions
   - F represents this normalization mismatch

2. **PRZZ extracts c from the bracket, not Direct+Mirror**
   - The DQ identity produces a bracket
   - c might be derived from bracket via different formula

3. **The DQ division factor (-2Rθ) enters the c formula**
   - PRZZ: bracket = [Direct - exp(2R)×Mirror] / (-2Rθ)
   - Maybe: c = bracket × (some factor) ≠ Direct + exp(2R)×Mirror

4. **We're computing a different object than PRZZ**
   - Our "c" matches their numerical result
   - But computed via completely different method

### Key Open Question

**How does PRZZ go from the bracket (DQ identity output) to the final c = 2.137?**

The bracket value is ~1.0, but c is ~2.14. There must be a transformation we don't understand.

---

## Bottom Line

**We need to read PRZZ like we're implementing it for the first time, not like we're validating our existing implementation.**

Our current implementation was built to "match the number", not to "follow the method". These are different goals, and only now do we realize the distinction matters.

The fact that our formula works (within ~1.5%) but PRZZ's stated formula (exp(2R)) gives nonsense results suggests either:
1. We're using different definitions/normalizations
2. We're extracting c via a different (but equivalent?) method
3. We got lucky and reverse-engineered a formula that happens to work

**Until we trace PRZZ's exact c extraction procedure, we cannot claim methodological reproduction.**
