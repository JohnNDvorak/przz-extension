# The Combinatorial Wall: Documentation Roadmap for PRZZ Paper

## Why This Matters

As PRZZ notes (line 139): "Even the simplest examples require sophisticated and very long analysis... At one point in Levinson's original paper there are **twenty four cancellations going on simultaneously!**"

The "combinatorial wall" has stopped major mathematicians from pursuing mollifiers further. To claim κ improvements, we must show the ugly derivations that prove our results.

---

## PRZZ.tex Section Structure

| Section | Lines | Topic | Combinatorial Content |
|---------|-------|-------|----------------------|
| §5 | 510-761 | Constructing a mollifier | Faà di Bruno's formula, Bell polynomials |
| §5.1 | 568+ | Quadratic case d=2 | Trick to convert ζ''/ζ |
| §6 | 762-911 | Main result for moment integral | Heath-Brown identity, Kloosterman sums |
| §7 | 912-1103 | Square-free terms & Feng's conjecture | Möbius convolution structure |
| §8 | 1104-1727 | Specializing coefficients (d=1) | **THE CORE**: Cases A, B, C |
| §9 | 1728-2529 | General case d≥0 | Extension machinery |
| §10 | 2530-2599 | Numerical aspects | Polynomial optimization |

---

## The Core Combinatorial Structure (Section 8)

### The ω Classification

**Definition (line 2303)**:
```
ω(d, l) := 1×l₁ + 2×l₂ + ... + d×l_d - 1
```

This determines which case applies:
- **Case A**: ω = -1 (derivative terms)
- **Case B**: ω = 0 (no attenuation)
- **Case C**: ω > 0 (auxiliary integral)

### Case A: ω = -1 (Lines 2305-2323)

```latex
Υ_A(d,l) = U(d,l) × (1/i!) × (d/dx) e^{αx} × (x + log(N/n)/log N)^i |_{x=0}
```

Where:
```
U(d,l) = 1{ω=-1} × (1!(-1)¹)^{l₁} × (2!(-1)²)^{l₂} × ... × (d!(-1)^d)^{l_d}
```

**Key property**: Produces derivatives of polynomial terms.

### Case B: ω = 0 (Lines 2324-2335)

```latex
Υ_B(d,l) = -V(d,l) × (log^i N / i!) × (log(N/n)/log N)^i
```

Where:
```
V(d,l) = 1{ω=0} × (1!(-1)¹)^{l₁} × (2!(-1)²)^{l₂} × ... × (d!(-1)^d)^{l_d}
```

**Key property**: Pure polynomial evaluation, no derivatives.

### Case C: ω > 0 (Lines 2336-2362)

The most complicated case. Uses the integral identity (line 2347):
```latex
∫_{1/q}^1 t^{α+s-1} log^τ t dt = (-1)^τ τ! / (α+s)^{τ+1} - q^{-α-s}/(α+s)^{τ+1} × P(s,α,log q)
```

Leading to:
```latex
Υ_C(d,l) = W(d,l) × (-1)^{1-ω} / ((ω-1)!) × (log N)^ω × (log(N/n)/log N)^ω
           × ∫_0^1 P_{d,ℓ}((1-a) × log(N/n)/log N) × a^{ω-1} × (N/n)^{-αa} da
```

**Key property**: Introduces auxiliary integration variable `a`.

---

## The 9 Cross-Terms → 6 by Symmetry (Line 2387)

When combining F_d(l,α,n) × F_d(k,β,n), we get 3×3 = 9 cases:

| ω(l) | ω(k) | Case Type | Count |
|------|------|-----------|-------|
| -1 | -1 | A×A | 1 |
| -1 | 0 | A×B | 2 (symmetric with B×A) |
| -1 | >0 | A×C | 2 (symmetric with C×A) |
| 0 | 0 | B×B | 1 |
| 0 | >0 | B×C | 2 (symmetric with C×B) |
| >0 | >0 | C×C | 1 |

By symmetry: **9 → 6 distinct cases**

---

## For K=3, d=1: The Six Pair Types

| Pair (ℓ₁,ℓ₂) | ω(ℓ₁) | ω(ℓ₂) | Case Type | PRZZ Contribution |
|--------------|-------|-------|-----------|-------------------|
| (1,1) | -1 | -1 | A×A | I₁ term |
| (1,2) | -1 | 0 | A×B | I₁ + I₃ term |
| (1,3) | -1 | 1 | A×C | I₁ + I₃ term |
| (2,2) | 0 | 0 | B×B | I₂ term |
| (2,3) | 0 | 1 | B×C | I₂ + I₄ term |
| (3,3) | 1 | 1 | C×C | I₁ + I₂ + I₃ + I₄ term |

**Critical insight**: The Case C terms (involving P₃) contribute the auxiliary integral structure that allows negative pair contributions.

---

## The Mirror Assembly (Lines 1500-1533)

The key formula:
```latex
I₁(α,β) = I_{1,1}(α,β) + T^{-α-β} I_{1,1}(-β,-α) + O(T/L)
```

This reduces to (line 1508-1511):
```latex
(N^{αx+βy} - T^{-α-β} N^{-βx-αy}) / (α+β)
= N^{αx+βy} × log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-t(α+β)} dt
```

At α = β = -R/L, this becomes the exponential kernel:
```
e^{R[θt(x+y)-θy+t]} × e^{R[θt(x+y)-θx+t]}
```

---

## What Our Paper Must Show

### Appendix A: Complete I₁ Derivation
- The difference quotient identity (lines 1502-1511)
- Q operator action (lines 1512-1518)
- Explicit form at α=β=-R/L (lines 1519-1533)
- All cancellations tracked

### Appendix B: Case A/B/C Kernel Derivations
- ω classification (line 2303)
- Each case's integral form
- Why Case C produces auxiliary integrals

### Appendix C: The Six Pair Contributions
- Table of (ℓ₁,ℓ₂) → contribution to c
- Explicit integrals for each pair
- Numerical verification

### Appendix D: Why Negative Pair Contributions Are Valid
- Pair (1,3) and (2,3) can be negative
- The TOTAL c must be positive, but individual pairs can cancel
- Mathematical proof from PRZZ structure

### Appendix E: Mirror Term Assembly
- Why S12 needs mirror but S34 doesn't
- The m = exp(R) + (2K-1) formula origin
- g_I1, g_I2 correction factors

---

## Code-to-Math Mapping

| PRZZ TeX | Code Location | What It Computes |
|----------|---------------|------------------|
| §8 I₁ formula | `unified_s12_evaluator_v3.py` | S12 pair contributions |
| Case A/B/C | `terms_k3_d1.py` | Term structure per pair |
| Mirror assembly | `kappa_engine.py:compute_c_from_integrals` | c = S12(+R) + m×S12(-R) + S34 |
| Q operator | `polynomials.py:QPolynomial` | Q(t) evaluation |
| P_ℓ polynomials | `polynomials.py:PellPolynomial` | P_ℓ(x) = x × P_tilde(x) |

---

## Verification Requirements for Publication

1. **Symbolic consistency**: Every formula in code matches PRZZ TeX
2. **Numerical agreement**: κ matches PRZZ published value to 0.001%
3. **Quadrature stability**: Results stable under n=40→100 refinement
4. **R-sweep validity**: Improvement persists across R values
5. **Sign change validity**: Negative pair contributions are theoretically allowed

**Current status**: All 5 requirements PASSED for the α=61 perturbation.

---

## Timeline for Documentation

1. **Phase 1** (immediate): Lock validation results in JSON
2. **Phase 2** (1-2 weeks): Write Appendix A (I₁ derivation)
3. **Phase 3** (1-2 weeks): Write Appendix B-C (Case structure)
4. **Phase 4** (1 week): Write Appendix D-E (validity proofs)
5. **Phase 5** (ongoing): Cross-reference with code

The combinatorial wall is the barrier to entry. Breaking through it with documented derivations is what makes the κ improvement publishable.
