# PRZZ Reproduction Audit — Complete Handoff Summary

**Date:** 2025-12-16 (Updated)
**Status:** Structural mismatch CONFIRMED; DSL hypersensitivity identified
**Tests:** 445 passing

---

## TL;DR for Fresh Session

We have a mathematically validated pipeline that computes **a fundamentally different object** than PRZZ's published main-term constant. The discrepancy is **NOT** a simple normalization issue—our DSL is **hypersensitive to polynomial changes** while PRZZ's formula is stable.

**Critical evidence:**
- With κ polynomials (R=1.3036): Our c=1.95 vs PRZZ c=2.14 (factor 1.10)
- With κ* polynomials (R=1.1167): Our c=0.82 vs PRZZ c=1.94 (factor **2.35**)
- PRZZ c ratio: 2.14/1.94 = 1.10 (stable)
- Our c ratio: 1.95/0.82 = **2.38** (unstable)

The **derivative extraction mechanism** in our DSL creates artificial cancellations that don't exist in PRZZ's formula.

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
- Benchmark 2: R = 1.1167, κ* = 0.407511457, c = 1.93795241 (different polynomials!)

---

## 2. Current Numerical State

### Computed vs Target (CORRECT polynomials for each benchmark)

| Benchmark | R | Our c | PRZZ c | Factor Needed | Our κ | PRZZ κ |
|-----------|------|-------|--------|---------------|-------|--------|
| 1 (κ polys) | 1.3036 | 1.9501 | 2.1375 | 1.096 | 0.488 | 0.417 |
| 2 (κ* polys) | 1.1167 | **0.8231** | 1.9380 | **2.355** | 1.174 | 0.408 |

### The Critical Finding

**The DSL is hypersensitive to polynomial changes.** PRZZ optimized two different polynomial sets for maximum κ and κ*. Both should produce large c values (~2). Our DSL gives:
- κ polynomials: c ≈ 1.95 (reasonable)
- κ* polynomials: c ≈ 0.82 (catastrophically wrong)

This proves we are computing a structurally different object that happens to give a reasonable answer for one polynomial set but fails completely for another.

### Per-Pair Breakdown Comparison

| Pair | κ (R=1.30) | κ* (R=1.12) | Ratio |
|------|------------|-------------|-------|
| c₁₁ | +0.4419 | +0.3742 | 1.18 |
| c₁₂ | -0.2009 | **-0.0016** | **129** |
| c₁₃ | -0.2179 | -0.0380 | 5.73 |
| c₂₂ | +1.2611 | +0.4190 | 3.01 |
| c₂₃ | +0.5861 | +0.0649 | 9.04 |
| c₃₃ | +0.0798 | +0.0046 | 17.4 |
| **Total** | **1.9501** | **0.8231** | **2.37** |

**The (1,2) pair has ratio 129!** With κ* polynomials, this pair nearly vanishes due to catastrophic cancellation between I₁, I₂, I₃, I₄ terms.

### Root Cause: Derivative Extraction Instability

Analysis of the (1,2) pair with κ* polynomials:
- I₁_12: -0.198
- I₂_12: +0.329
- I₃_12: -0.184
- I₄_12: +0.052
- Sum of positives: 0.380
- Sum of negatives: -0.382
- **Net: -0.0016** (near-perfect cancellation!)

With κ polynomials, the same pair has |neg|/|pos| ≈ 1.30 (less cancellation).
With κ* polynomials: |neg|/|pos| ≈ 1.004 (extreme cancellation).

This cancellation is an artifact of our DSL structure, not a feature of PRZZ's formula.

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

### 4.6 Using κ* Polynomials to Validate — REVEALED THE BUG

GPT identified that prior two-benchmark tests used κ polynomials for BOTH benchmarks. The κ* benchmark requires DIFFERENT polynomials (PRZZ TeX lines 2587-2598). Testing with correct polynomials made the gap **WORSE** (factor 2.35), revealing the DSL hypersensitivity.

---

## 5. The Core Problem: Derivative Extraction Architecture

### The Smoking Gun

Testing with correct κ* polynomials revealed the fundamental issue:
- **PRZZ's c values are stable** across different optimized polynomial sets (~2 for both κ and κ*)
- **Our DSL's c values are unstable** (1.95 for κ, 0.82 for κ*)

This hypersensitivity is caused by our derivative extraction mechanism.

### DSL Architecture vs PRZZ

**Our DSL approach:**
1. Build series with formal variables (x₁, x₂, y₁, y₂, ...)
2. Multiply all polynomial/exponential factors as series
3. Extract derivatives via series coefficient lookup

**PRZZ Section 7 approach:**
1. Enumerate Faà-di-Bruno partitions for derivative structure
2. Compute F_d factors which already encode derivatives
3. Apply Euler-Maclaurin for n-sum → integral
4. Integrate with explicit formulas

In PRZZ, derivatives are "pre-computed" into F_d factors. In our DSL, we try to recover them post-hoc via series operations, which creates artificial sign cancellations.

### Evidence Summary

1. **Polynomial sensitivity**: Our c ratio (κ/κ*) = 2.38, PRZZ ratio = 1.10
2. **Per-pair breakdown**: (1,2) ratio = 129× due to extreme cancellation
3. **Simplified integrals**: P²×Q² integrals have stable ratio ~2.2, not 129
4. **I₂ terms only**: Moderately stable (no derivatives extracted)
5. **Full DSL with I₁,I₃,I₄**: Extremely unstable

### What This Means

**The DSL computes a different mathematical object.** For κ polynomials, it happens to produce a reasonable c value. For κ* polynomials, the derivative extraction creates cancellations that destroy the answer.

**WARNING:** The computed κ values should **NOT** be interpreted as zeta-zero proportion bounds. The DSL is computing a surrogate objective.

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
│   ├── polynomials.py         # PRZZ polynomial loading (κ AND κ*)
│   ├── quadrature.py          # Gauss-Legendre integration
│   ├── series.py              # Multi-variable Taylor series
│   ├── term_dsl.py            # Term data model
│   ├── terms_k3_d1.py         # K=3 term definitions
│   ├── evaluate.py            # Main computation (mode="main" vs "with_error_terms")
│   ├── fd_oracle.py           # Finite-difference validation
│   ├── mollifier_profiles.py  # Case B/C profile generators
│   ├── przz_section7_oracle.py # NEW: PRZZ Section 7 oracle (in progress)
│   └── przz_22_oracle.py      # (2,2) pair oracle (educational, structurally limited)
├── tests/                     # 445 tests
├── docs/
│   ├── HANDOFF_SUMMARY.md     # This file
│   ├── TRUTH_SPEC.md          # Benchmark parameters and targets
│   ├── STRUCTURE_COMPARISON.md # Implementation vs PRZZ mapping
│   ├── GAP_ANALYSIS_SUMMARY.md # R-sensitivity analysis
│   └── DEEP_INVESTIGATION_2025_12_16.md # Latest findings
└── data/
    ├── przz_parameters.json       # κ benchmark polynomials (R=1.3036)
    └── przz_parameters_kappa_star.json  # NEW: κ* benchmark polynomials (R=1.1167)
```

---

## 9. Recommended Next Steps

### Track 1: Complete the PRZZ Section 7 Oracle (HIGHEST PRIORITY)

The oracle in `src/przz_section7_oracle.py` is partially implemented. To complete it:

1. **Implement the n-sum → integral conversion properly**
   - Use Euler-Maclaurin with correct (1-u) powers
   - Handle arithmetic coefficients correctly

2. **Add the S(z) zeta-ratio function**
   - This creates the t-integral structure
   - Must match PRZZ's pole structure at α+β

3. **Combine F_d factors with proper derivative handling**
   - PRZZ pre-computes derivatives in F_d via Faà-di-Bruno
   - Do NOT use post-hoc series extraction

4. **Test on both benchmarks**
   - κ polynomials @ R=1.3036 should give c ≈ 2.14
   - κ* polynomials @ R=1.1167 should give c ≈ 1.94

### Track 2: Surrogate Optimization (With Explicit Limitations)

If the goal is exploration rather than PRZZ equivalence:
- **Label all results as "surrogate objective"**
- Use only the κ polynomial set (relatively stable)
- Do NOT claim κ values as zeta-zero bounds
- Document that κ* polynomials break the DSL

### Track 3: External Information

- Search for Feng's original code (PRZZ line 2566: "matched Feng's")
- Check if PRZZ authors have per-pair breakdowns
- Contact PRZZ authors for clarification

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

## 12. ROOT CAUSE IDENTIFIED (2025-12-16)

### The Smoking Gun: I₂ Matches, I₁/I₃/I₄ Don't

Building an exact oracle for the (2,2) pair using PRZZ formulas (lines 1530-1570) revealed:

| Term | Oracle κ | DSL κ | Match? |
|------|----------|-------|--------|
| I₂   | 0.9088   | 0.9088 | ✓ Perfect |
| I₁   | 1.1686   | 3.8839 | ✗ DSL 3.3× larger |
| I₃   | -0.5444  | +0.1259 | ✗ Wrong sign! |
| I₄   | -0.5444  | +0.1259 | ✗ Wrong sign! |

**I₂ has NO derivatives** (evaluated at x=y=0). This proves polynomials, Q, exp, and integration are correct.

The bug is specifically in **derivative extraction** for I₁, I₃, I₄.

### Root Cause: Variable Count Mismatch

**PRZZ (ℓ₁=ℓ₂=1 section, lines 1530-1570):**
- Uses **2 variables**: single x, single y
- P factors: P(x+u), P(y+u)
- Derivative: **d²/dxdy** (2nd order total)

**DSL (2,2) pair:**
- Uses **4 variables**: x₁, x₂, y₁, y₂
- P factors: P(x₁+x₂+u), P(y₁+y₂+u)
- Derivative: **d⁴/dx₁dx₂dy₁dy₂** (4th order total!)

These are **mathematically different objects**:
- PRZZ extracts P'(u)×P'(u) + cross-terms
- DSL extracts P''(u)×P''(u) + different cross-terms

### Why Was Multi-Variable Structure Chosen?

CLAUDE.md states: "For pair (ℓ₁,ℓ₂) we must keep distinct formal variables x₁,...,x_{ℓ₁} and y₁,...,y_{ℓ₂}".

This was a **misinterpretation**. PRZZ's explicit formulas for ℓ₁=ℓ₂=1 use single x, y, not multiple variables. The number of formal variables should NOT equal the piece index ℓ.

### The Fix

The DSL needs to be rewritten to use **single x, y variables** for all pairs, matching PRZZ's explicit formulas:

- I₁: d²/dxdy[P(x+u)P(y+u)Q(arg_α)Q(arg_β)exp...]|_{x=y=0}
- I₃: d/dx[(1+θx)/θ × P(x+u)P(u)...]|_{x=0}
- I₄: d/dy[(1+θy)/θ × P(u)P(y+u)...]|_{y=0}

NOT the current multi-variable structure.

### Verification

The oracle in `src/przz_22_exact_oracle.py` implements the correct single-variable formulas and produces stable results for both κ and κ* polynomial sets.

## 13. Summary Statement for New Session (OUTDATED - SEE SECTION 14)

> The PRZZ reproduction pipeline has a **fundamental structural bug**: the DSL uses wrong variable counts.
>
> **Evidence:**
> - I₂ (no derivatives): Oracle = DSL = 0.9088 ✓
> - I₁, I₃, I₄ (with derivatives): Oracle ≠ DSL (completely different values!)
>
> **Root cause:** For (2,2) pair, DSL uses d⁴/dx₁dx₂dy₁dy₂ with 4 variables. PRZZ uses d²/dxdy with 2 variables. These compute **different mathematical objects**.
>
> **Fix:** Rewrite the DSL to use single x, y variables matching PRZZ formulas.
>
> **Do NOT use current DSL for optimization.** The derivative extraction is fundamentally wrong.

---

## 14. TRUE ROOT CAUSE: Missing Ψ Combinatorial Structure (2025-12-17)

### The "I₁-I₄" Structure is Only Valid for (1,1)

GPT identified the fundamental issue: the "I₁-I₄" decomposition from PRZZ Section 7 is **ONLY derived for ℓ=1 pairs**. For ℓ>1, PRZZ requires the full Ψ combinatorial expansion.

### The Ψ Formula

For pair (ℓ, ℓ̄), the main-term structure is:

```
Ψ_{ℓ,ℓ̄}(A,B,C,D) = Σ_{p=0}^{min(ℓ,ℓ̄)} C(ℓ,p)C(ℓ̄,p)p! × (D-C²)^p × (A-C)^{ℓ-p} × (B-C)^{ℓ̄-p}
```

Where:
- A = ∂/∂z (z-derivative piece)
- B = ∂/∂w (w-derivative piece)
- C = log ξ(s₀) (no-derivative piece)
- D = ∂²/∂z∂w (mixed derivative piece)

### Monomial Counts

| Pair | DSL terms | Ψ monomials | DSL Coverage |
|------|-----------|-------------|--------------|
| (1,1) | 4 | **4** | 100% ✓ |
| (2,2) | 4 | **12** | 33% ✗ |
| (3,3) | 4 | **27** | 15% ✗ |
| (1,2) | 4 | 7 | 57% ✗ |
| (1,3) | 4 | 10 | 40% ✗ |
| (2,3) | 4 | 18 | 22% ✗ |

### Detailed (1,1) Expansion (Correct)

```
Ψ_{1,1} = AB - AC - BC + D
```

This corresponds exactly to:
- AB → I₁ (mixed derivative ∂z∂w)
- D → I₂ (no derivatives, base integral)
- -AC → I₃ (∂z only)
- -BC → I₄ (∂w only)

### Detailed (2,2) Expansion (12 Monomials)

```
D terms (4):
  +4 × C⁰D¹A¹B¹
  +2 × C⁰D²A⁰B⁰
  -4 × C¹D¹A⁰B¹
  -4 × C¹D¹A¹B⁰

Mixed A×B (3):
  +1 × C⁰D⁰A²B²
  -2 × C¹D⁰A¹B²
  -2 × C¹D⁰A²B¹

A-only (2):
  +1 × C²D⁰A²B⁰
  +2 × C³D⁰A¹B⁰

B-only (2):
  +1 × C²D⁰A⁰B²
  +2 × C³D⁰A⁰B¹

Pure C (1):
  -1 × C⁴D⁰A⁰B⁰
```

### Why This Explains the Two-Benchmark Failure

1. **DSL only captures ~25-33% of terms for (2,2) and (3,3)**
2. **The missing terms have different polynomial sensitivities**
3. **κ vs κ* polynomials have different structures** (degrees 3 vs 2)
4. **Missing terms' contributions change differently across benchmarks**

### Path Forward

1. **Implement the Ψ formula** as a config-driven term generator
2. **Each monomial (k₁,k₂,ℓ₁,m₁)** becomes a distinct integral term
3. **Build new oracles** that match the full Ψ expansion
4. **Validate (1,1) unchanged**, then extend to (2,2), (3,3)

### Ψ Monomial Sum Validated for (1,1) ✓

**Key numerical validation** (2025-12-17):
```
Oracle I-terms:
  I₁ = +0.426028
  I₂ = +0.384629
  I₃ = -0.225749
  I₄ = -0.225749
  Total = 0.359159

Ψ monomial sum:
  +1 × D    = +0.3846
  -1 × BC   = -0.2257
  -1 × AC   = -0.2257
  +1 × AB   = +0.4260
  Sum = 0.359159

Difference = 5.55e-17 ✓ PERFECT MATCH
```

This validates the entire monomial pipeline:
1. p-configs → monomials expansion ✓
2. Monomial (a,b,c,d) → I-term mapping ✓
3. Ψ combinatorial structure is correct ✓

### New Files

- `src/psi_combinatorial.py` — Validates Ψ monomial counts
- `src/psi_block_configs.py` — p-sum representation (16 configs for K=3)
- `src/psi_monomial_expansion.py` — Expands p-configs to (a,b,c,d) vectors
- `src/psi_monomial_evaluator.py` — Maps monomials to I-term evaluations

### Key Reference

PRZZ Section 7 formulas are for ℓ=1 only. The full (ℓ,ℓ̄) structure requires:
- Faà-di-Bruno partitions for derivative organization
- F_d factors that encode higher derivatives
- Euler-Maclaurin for n-sum conversion

The DSL attempted to generalize the ℓ=1 structure to all pairs, which is mathematically incorrect.
