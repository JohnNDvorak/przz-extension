# TRUTH_SPEC.md — Single Source of Truth for PRZZ c/κ Computation

**Purpose:** Pin the exact mathematical object we're computing, with explicit PRZZ TeX line references.

**Document Status:** AUTHORITATIVE — No code changes without matching this spec.

**TeX Source:** All line numbers refer to `RMS_PRZZ.tex` (1-indexed)

**Verification command:**
```bash
nl -ba RMS_PRZZ.tex | sed -n 'START,ENDp'
```

---

## 1. κ Definition

**PRZZ TeX Lines 286–289** (core definition: σ₀, c, and κ):
```latex
κ ≥ 1 - (1/R) log((1/T) ∫₁ᵀ |Vψ(σ₀+it)|² dt) + o(1)
```

**Implication:**
- `c` is the **asymptotic main term constant** from the mean square integral
- `c` does NOT include lower-order terms like I₅ (unless PRZZ explicitly keeps them)
- κ = 1 - log(c)/R

---

## 2. Published Target Values

```
θ = 4/7 ≈ 0.5714285714285714
R = 1.3036
c_target = 2.13745440613217263636
κ_target = 0.417293962
```

**Verification:**
```python
import math
kappa = 1 - math.log(2.13745440613217263636) / 1.3036
# kappa ≈ 0.4172939615...  ✓
```

---

## 3. Mollifier Definition

**PRZZ TeX Lines 542-545:**
```latex
ψ_{d=1}(s) = Σ_{n≤y₁} μ(n)n^{σ₀-1/2}/n^s Σ_{k=2}^K Σ_{p₁...p_k|n}
             (log p₁...log p_k)/(log^k y₁) P_{1,k}(log(y₁/n)/log(y₁))
```

**PRZZ TeX Line 548:**
> "Feng has set the convention of starting at K=2."

### Critical Indexing

| Our Poly | Our Index | PRZZ k | Λ convolutions | Description |
|----------|-----------|--------|----------------|-------------|
| P₁ | 1 | 2 | 1 | μ⋆Λ piece |
| P₂ | 2 | 3 | 2 | μ⋆Λ⋆Λ piece |
| P₃ | 3 | 4 | 3 | μ⋆Λ⋆Λ⋆Λ piece |

**Our code:** k starts at 1 (P₁, P₂, P₃)
**PRZZ paper:** k starts at 2 (k=2, 3, 4)
**Mapping:** Our index i corresponds to PRZZ k = i+1

---

## 4. I₅ is an Error Term

### 4a. Primary Citation: I₅ ≪ T/L

**PRZZ TeX Lines 1621–1628:**
```latex
The term (I_5) is of smaller order in (log N).
...
I₅ ≪ T/L.
Hence the term associated to A_{α,β}^{(1,1)}(0,0;β,α) is an error term.
```

This is the **canonical citation** for "I₅ is error term".

### 4b. Earlier Statement (same theme)

**PRZZ TeX Lines 1490–1499:**
```latex
I_{1,5} … prime sum …
This is of smaller order in log N.
```

### 4c. General Architecture Rule

**PRZZ TeX Lines 1714–1727:**

Any nonzero derivatives of A contribute only to the error term O(T/L).

Tightest slice for "derivatives → error" is **Lines 1722–1727**.

### Implication
- I₅ is **O(T/L)** — lower order than main term **O(T)**
- I₅ should NOT be used to calibrate/match the published κ
- If we need I₅ to hit κ, we're computing the **wrong main term**
- All A-derivative contributions are absorbed into error terms

### Enforcement
- `mode="main"`: I₅ forbidden, not even imported
- `mode="with_error_terms"`: Diagnostic only, prints warning

---

## 5. Mirror Combination Identity

### 5a. Headline Identity

**PRZZ TeX Lines 1502–1504:**

The main object is written as base term plus mirror term with T^{-α-β}.

### 5b. Difference Quotient → Integral Representation

**PRZZ TeX Lines 1502–1511:**
```latex
(N^{αx+βy} - T^{-α-β}N^{-βx-αy})/(α+β)
= N^{αx+βy} log(N^{x+y}T) ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

This removes the singularity at α+β=0 analytically.

### 5c. Q(...) Operator Structure

**PRZZ TeX Lines 1514–1517:**

The `Q(...)Q(...)` arguments appearing in the I_{1,1}(α,β) line.

### 5d. Evaluated at α=β=-R/L

**PRZZ TeX Lines 1521–1523:**
```latex
= log(T^{θ(x+y)+1}) ∫₀¹ Q(...)Q(...) e^{R[...]} e^{R[...]} dt
```

### Implication
- PRZZ does analytic combination **BEFORE** extracting constants
- The difference quotient 1/(α+β) becomes an integral representation
- This removes the singularity at α+β=0
- Expanding separately and adding later → **wrong constant term**

---

## 6. I₃/I₄ Prefactor

### 6a. I₃ Formula

**PRZZ TeX Lines 1551–1564:**
```latex
I₃ = -TΦ̂(0) × (1+θx)/θ × d/dx[∫∫ (1-u)P₁(x+u)P₂(u) e^{R[...]} Q(...)Q(...) dtdu]|_{x=0}
```

At x=0: (1+θx)/θ = 1/θ

**Prefactor:** -1/θ (NOT -1)

### 6b. I₄ Formula

**PRZZ TeX Lines 1566–1570:**
```latex
I₄ = -TΦ̂(0) × (1+θy)/θ × d/dy[∫∫ (1-u)P₁(u)P₂(y+u) e^{R[...]} Q(...)Q(...) dtdu]|_{y=0}
```

Same prefactor structure by symmetry.

### Variable Stage Note
**PRZZ TeX Line 2309:**
> "by the change of variable x → x log N"

The variable x in the I₃/I₄ formula may be at a different scaling stage.
FD oracle must operate at the same stage as the paper formula.

---

## 7. ω-Case Classification

### 7a. ω Definition and Case A + Variable Rescaling

**PRZZ TeX Lines 2301–2310:**
```latex
ω(d,l) = 1×l₁ + 2×l₂ + ... + d×l_d - 1
```

For d=1: ω = l₁ - 1

**Critical:** Line 2309 contains "by the change of variable x → x log N"

### 7b. Case B and C Definitions

**PRZZ TeX Lines 2320–2344:**

Contains the U/V/W indicator definitions and case split structure.

### Cases

| Case | Condition | Structure |
|------|-----------|-----------|
| A | ω = -1 | Simple derivative |
| B | ω = 0 | Direct polynomial evaluation |
| C | ω > 0 | Auxiliary a-integral required |

### Mapping to Our Polynomials

| Our Poly | PRZZ k | ω = k-2 | Case |
|----------|--------|---------|------|
| P₁ | k=2 | 0 | B |
| P₂ | k=3 | 1 | C |
| P₃ | k=4 | 2 | C |

**Warning:** P₂ and P₃ pairs are Case C and may need auxiliary a-integral!

---

## 8. Case C Auxiliary Integral (CRITICAL FOR P₂/P₃)

### 8.1 Case C Derivation and Structure

**PRZZ TeX Lines 2336–2362** (Case C derivation):

The Case C derivation shows how ω > 0 introduces an auxiliary a-integral
that is **not present** in Case B (ω = 0).

**PRZZ TeX Lines 2364–2368** (Product F_d × F_d structure):

For pair (ℓ₁, ℓ₂), the I₁ integral has product structure F_{d,ℓ₁} × F_{d,ℓ₂}.
This means:
- **B×B** (like (1,1)): No auxiliary integrals (both factors are Case B)
- **B×C** (like (1,2), (1,3)): ONE auxiliary a-integral (from C factor)
- **C×C** (like (2,2), (2,3), (3,3)): TWO auxiliary a-integrals (from both C factors)

### 8.2 Case C Integral Formula

**PRZZ TeX Lines 2369–2384** (Case C a-integral, especially line 2374):
```latex
∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da
```

This block contains both g_d(k,α,n) and F_d(k,α,n) structures.

This is a Beta-type integral that appears in Case C (ω > 0).

For P₂-involving pairs: ω = 1, so integral is ∫₀¹ (1-a)^i (N/n)^{-αa} da
For P₃-involving pairs: ω = 2, so integral is ∫₀¹ (1-a)^i a (N/n)^{-αa} da

### 8.3 Cross-Term Bookkeeping

**PRZZ TeX Lines 2387–2388** (Cross-term structure):

Cross-terms have asymmetric Case structure. For pair (ℓ₁, ℓ₂):
- If ℓ₁ is Case B and ℓ₂ is Case C: one a-integral from ℓ₂
- If both are Case C: two a-integrals (product)

### 8.4 Implications for Our Implementation

| Pair | Cases | a-integrals | Term DSL | J1x Diagnostic |
|------|-------|-------------|----------|----------------|
| (1,1) | B×B | 0 | ✓ Correct | ✓ Correct |
| (1,2) | B×C | 1 | ✓ Implemented | ⚠️ Simplified |
| (1,3) | B×C | 1 | ✓ Implemented | ⚠️ Simplified |
| (2,2) | C×C | 2 | ✓ Implemented | ⚠️ Simplified |
| (2,3) | C×C | 2 | ✓ Implemented | ⚠️ Simplified |
| (3,3) | C×C | 2 | ✓ Implemented | ⚠️ Simplified |

**Phase 17 Update (2025-12-24):**

The **Term DSL** (`PolyFactor.evaluate()`) now correctly dispatches to Case C kernel
via `case_c_taylor_coeffs()` when `kernel_regime="paper"` and omega > 0. This is used
by the production `compute_c_paper_with_mirror()` pipeline.

The **J1x diagnostic pipeline** (`j1_euler_maclaurin.py`) uses a simplified polynomial
approach that doesn't implement Case C structure. This diagnostic is for validating
the m₁ = exp(R) + 5 formula, not for production κ computation.

**Current accuracy with Case C in production pipeline:**
- κ (R=1.3036): c gap = -1.35%
- κ* (R=1.1167): c gap = -1.21%

### 8.5 Case C Kernel Implementation Architecture

The Case C kernel K_ω(u; R) is implemented in **two files** with different conventions:

| File | Function | Formula | Includes Prefactor? |
|------|----------|---------|---------------------|
| `case_c_kernel.py` | `compute_case_c_kernel` | K_ω(u; R) = u^ω/(ω-1)! × ∫... | ✓ Yes |
| `case_c_exact.py` | `compute_case_c_kernel_vectorized` | ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθau) da | ✗ No |

**Relationship:**
```python
K_full = (u ** omega) / factorial(omega - 1) * K_integral
```

**Why Two Implementations?**
- `case_c_exact.py` returns just the integral for use in series expansion
- `case_c_kernel.py` returns the full kernel for direct I-term evaluation

**Validation:**
- Gate K1 tests (`test_case_c_kernel_exact.py`) verify both implementations match
- Gate K2 tests (`test_case_c_kernel_series.py`) verify Taylor coefficient extraction

**Derivative Implementation:**
- `compute_case_c_kernel_derivative()` computes ∂K_ω/∂u analytically
- Uses chain rule: derivative of ∫P((1-a)u)... gives ∫(1-a)P'((1-a)u)...
- Validated against finite differences in Gate K1

---

## 8b. Prime Sum / S(0) Value

**PRZZ TeX Lines 1377–1389:**

Contains the prime-sum expression (over p) coming from A_{α,β}^{(1,1)} evaluated at zero.

Numeric approximation: **S(0) ≈ 1.385603705**

This is what we call `S_AT_ZERO` in the code (`src/arithmetic_constants.py`).

For wider context including the definition, use **Lines 1372–1389**.

---

## 8c. N = T^θ / y_d Definition

**PRZZ TeX Lines 624–635:**

Contains the statement where `y_d = N = T^{θ_d}` appears and the "θ = 4/7 − ε" context.

This defines the normalization relationship between T, N, and θ.

---

## 9. Normalizations

### Φ̂(0) (Test Function)
Assumed normalized: Φ̂(0) = 1

### Factorial Normalization
Pairs have factor 1/(ℓ₁! × ℓ₂!) from bracket combinatorics.

| Pair | Factor |
|------|--------|
| (1,1) | 1/1 = 1 |
| (2,2) | 1/4 |
| (3,3) | 1/36 |
| (1,2) | 1/2 |
| (1,3) | 1/6 |
| (2,3) | 1/12 |

### Ordered vs Triangle Assembly

**CORRECTED (2025-12-22):** PRZZ uses **triangle convention** for ALL terms.

The individual I₃/I₄ terms are NOT swap-symmetric (I₃(1,2) ≠ I₃(2,1), measured
Δ_S34 ≈ 0.54). However, this does NOT mean PRZZ sums over ordered pairs.

PRZZ sums over **ℓ₁ ≤ ℓ₂** with symmetry factor 2 for off-diagonal pairs:
- c = Σ_{ℓ₁ ≤ ℓ₂} symmetry_factor × c_{ℓ₁,ℓ₂}
- symmetry_factor = 1 (diagonal), 2 (off-diagonal)

The ×2 factor comes from the structure of the mollifier mean square, not from
term-level symmetry. PRZZ never evaluates both (1,2) and (2,1) — only (1,2).

**Numerical verification (2025-12-22):**
- Triangle×2 for S34: c = 2.109, gap = **-1.35%**
- Ordered (9 pairs) for S34: c = 2.371, gap = **+10.91%**

**Code policy (corrected):**
- Use triangle×2 for BOTH S12 and S34 (PRZZ convention)
- The "ordered pairs required" conclusion from earlier was a misinterpretation
- `compute_c_paper_ordered()` now uses triangle×2 for S34

### Q(0) Constraint
Q is a polynomial with Q(0) = 1 (endpoint normalization).

---

## 10. Target Formula Identification

**The exact PRZZ formula producing published c:**

From Section 6.2.1, after combining:
- I₁(α,β) + T^{-α-β}I₁(-β,-α) (mirror terms)
- I₂(α,β) + T^{-α-β}I₂(-β,-α)
- I₃(α,β) and I₄(α,β)

Then evaluating at α=β=-R/L and extracting:
- d²/dxdy at x=y=0 for I₁
- No derivatives for I₂
- d/dx at x=0 for I₃
- d/dy at y=0 for I₄

**Main constant is:** c = Σ pairs (with normalizations)

**"Main" mode means:** This sum, without I₅.

---

## 11. Verification Checklist

Before claiming "matched PRZZ":

1. [ ] FD oracle confirms DSL prefactor = -1/θ
2. [ ] Gap attribution shows which pairs contribute deficit
3. [ ] Piece ↔ ω mapping proven from mollifier definition
4. [ ] Case C auxiliary integral included if needed
5. [ ] Mirror combination done correctly
6. [ ] mode="main" produces c ≈ 2.137 WITHOUT I₅
7. [ ] Every factor has PRZZ equation reference

---

## References

All line numbers refer to: `RMS_PRZZ.tex` (1-indexed)

### Must-Quote Shortlist (Highest Load-Bearing Citations)

| Line(s) | Claim |
|---------|-------|
| 286–289 | κ / c definition |
| 1621–1628 | I₅ is error term (primary) |
| 1722–1726 | A-derivatives are error terms O(T/L) |
| 1502–1504 | Mirror term structure |
| 1551–1564 | I₃ displayed formula |
| 1566–1570 | I₄ displayed formula |
| 2301–2310 | ω-case split + x → x log N |

### Complete Reference Table

| Line(s) | Content |
|---------|---------|
| 286–289 | κ definition (core: σ₀, c, and κ ≥ 1 − (1/R) log c) |
| 542–545 | Mollifier definition |
| 548 | "Feng convention starting at K=2" |
| 624–635 | N = T^θ / y_d definition |
| 1377–1389 | Prime sum S(0) ≈ 1.385603705 |
| 1490–1499 | I_{1,5} smaller order statement |
| 1502–1504 | Mirror term headline identity |
| 1502–1511 | Difference quotient → integral representation |
| 1514–1517 | Q(...)Q(...) operator structure |
| 1521–1523 | Mirror combination at α=β=-R/L |
| 1551–1564 | I₃ formula with (1+θx)/θ prefactor |
| 1566–1570 | I₄ formula with (1+θy)/θ prefactor |
| 1621–1628 | I₅ ≪ T/L (explicit, canonical citation) |
| 1714–1727 | A-derivatives contribute only to O(T/L) error |
| 1722–1726 | Tightest slice for "derivatives → error" |
| 2301–2310 | ω definition + Case A + "x → x log N" |
| 2320–2344 | Case B + Case C definitions (U/V/W indicators) |
| 2336–2362 | Case C derivation (auxiliary a-integral introduction) |
| 2364–2368 | Product F_d × F_d structure (B×C vs C×C) |
| 2369–2384 | Case C auxiliary a-integral structure (esp. line 2374) |
| 2387–2388 | Cross-term bookkeeping for B×C and C×C |
