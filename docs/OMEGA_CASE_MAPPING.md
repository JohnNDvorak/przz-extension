# OMEGA_CASE_MAPPING.md — Piece ↔ ℓ ↔ ω Mapping Proof

**Purpose:** Prove the mapping between our polynomial indices and PRZZ's definitions.

**Status:** DOCUMENTED with OPEN QUESTIONS

---

## 1. PRZZ Mollifier Structure (TeX lines 542-545)

The PRZZ d=1 mollifier is:
```latex
ψ_{d=1}(s) = Σ_{n≤y₁} μ(n)n^{σ₀-1/2}/n^s Σ_{k=2}^K Σ_{p₁...p_k|n}
             (log p₁...log p_k)/(log^k y₁) P_{1,k}(log(y₁/n)/log(y₁))
```

**Key observations:**
- k starts at 2 (Feng convention, line 548)
- Each piece k has polynomial P_{1,k}
- The number of log factors = k-1 (from Λ^{*(k-1)} convolution)

---

## 2. Piece Index Mapping

| Our Index | Our Poly | PRZZ k | Λ convolutions | Variable count (ℓ) |
|-----------|----------|--------|----------------|-------------------|
| 1 | P₁ | k=2 | 1 | ℓ=1 |
| 2 | P₂ | k=3 | 2 | ℓ=2 |
| 3 | P₃ | k=4 | 3 | ℓ=3 |

**Relationship:**
- Our index i → PRZZ k = i+1
- Variable count ℓ = k-1 = i

---

## 3. Cross-Term (ℓ₁, ℓ₂) Indexing

In PRZZ, a cross term (ℓ₁, ℓ₂) means:
- ℓ₁ z-variables (z₁,...,z_{ℓ₁})
- ℓ₂ w-variables (w₁,...,w_{ℓ₂})
- Derivative: d^{ℓ₁+ℓ₂}/dz₁...dz_{ℓ₁}dw₁...dw_{ℓ₂}

| Our Pair | PRZZ (ℓ₁,ℓ₂) | Total vars | Derivative order |
|----------|--------------|------------|------------------|
| (1,1) | (1,1) | 2 | d²/dx₁dy₁ |
| (2,2) | (2,2) | 4 | d⁴/dx₁dx₂dy₁dy₂ |
| (3,3) | (3,3) | 6 | d⁶/dx₁dx₂dx₃dy₁dy₂dy₃ |
| (1,2) | (1,2) | 3 | d³/dx₁dy₁dy₂ |
| (1,3) | (1,3) | 4 | d⁴/dx₁dy₁dy₂dy₃ |
| (2,3) | (2,3) | 5 | d⁵/dx₁dx₂dy₁dy₂dy₃ |

**Our implementation matches this structure.**

---

## 4. The ω-Case Classification (TeX lines 2302-2304)

**Definition:**
```latex
ω(d,l) = 1×l₁ + 2×l₂ + ... + d×l_d - 1
```

**For d=1:** ω = l₁ - 1

**Cases:**
- **Case A (ω = -1):** Simple derivative d/dx
- **Case B (ω = 0):** Direct polynomial evaluation
- **Case C (ω > 0):** Auxiliary a-integral required

**Critical insight from PRZZ TeX lines 2350-2357:**
Case C formula includes:
```latex
∫₀¹ (1-a)^i a^{ω-1} (N/n)^{-αa} da
```

---

## 5. Where ω-Cases Apply

**Important:** The ω-case classification applies to the **coefficient sum** evaluation (Υ function), NOT to the main integrals I₁-I₄.

The I₁-I₄ structure comes from Section 6.2.1 and is **independent** of ω-cases.

However, the ω-cases affect the **arithmetic factor A** contributions that feed into the constant extraction.

---

## 6. OPEN QUESTIONS

### Q1: Variable Argument Structure

Our (2,2) implementation uses **summed** polynomial arguments:
```python
P_arg_left = x1 + x2 + u    # P₂(x1+x2+u)
P_arg_right = y1 + y2 + u   # P₂(y1+y2+u)
```

**Question:** Is this correct, or should the structure be different?

PRZZ (ℓ₁=2, ℓ₂=2) case (line 1636+) has variables z₁, z₂, w₁, w₂ but the polynomial argument structure isn't immediately obvious from the contour integral form.

### Q2: Arithmetic Factor A Derivatives

For (ℓ₁=2, ℓ₂=2), PRZZ computes:
```
A_{α,β}^{(i,j,k,l)}(0,0,0,0;β,α)
```
with various derivative combinations. These feed into the constant.

**Question:** Are we correctly accounting for these arithmetic derivatives?

### Q3: Does Case C Structure Affect Our Implementation?

If the ω-case analysis applies somewhere in our pipeline, we might be missing auxiliary integrals for ℓ ≥ 2 pieces.

---

## 7. Current Implementation Status

| Aspect | Status |
|--------|--------|
| Piece ↔ k mapping | ✓ Verified |
| (ℓ₁,ℓ₂) → variable count | ✓ Verified |
| I₁-I₄ structure | ✓ Matches PRZZ Section 6.2.1 for (1,1) |
| FD oracle validation | ✓ Prefactor -1/θ confirmed |
| Summed argument assumption | ⚠️ OPEN - needs verification |
| Arithmetic factor derivatives | ⚠️ OPEN - needs verification |

---

## 8. Next Steps to Resolve

1. **Check (2,2) I₁ structure against PRZZ Section 6.2.1**
   - Line 1636+: "The case ℓ₁=ℓ₂=2"
   - Verify polynomial argument structure

2. **Check arithmetic factor contributions**
   - Lines 1649-1652: Cross derivatives of A
   - Verify we include all non-zero terms

3. **Verify I₃/I₄ for (2,2)**
   - Build FD oracle for I3_22
   - Confirm prefactor structure for higher ℓ

---

## 9. References

| Line | Content |
|------|---------|
| 542-545 | Mollifier definition (k=2,3,...,K) |
| 548 | "Feng convention starting at K=2" |
| 1252-1256 | General I₁ formula |
| 1304 | "We will start with ℓ₁=ℓ₂=1, and ℓ₁=ℓ₂=2" |
| 1308 | "specialize to ℓ₁=ℓ₂=1" |
| 1636 | "The case ℓ₁=ℓ₂=2" |
| 2302-2304 | ω definition |
| 2305-2335 | Cases A, B, C definitions |
| 2350-2357 | Case C auxiliary a-integral |
