# Code-to-PRZZ Mapping for Paper Documentation

## Purpose

This document maps every critical formula in our implementation to the corresponding lines in RMS_PRZZ.tex, enabling verification that our code correctly implements the theory.

---

## Core Formula Mappings

### 1. Difference Quotient Identity

**PRZZ TeX Lines 1502-1511**:
```latex
\frac{N^{\alpha x + \beta y}-T^{-\alpha-\beta}N^{-\beta x - \alpha y}}{\alpha+\beta}
= N^{\alpha x + \beta y} \log(N^{x+y}T) \int_0^1 (N^{x+y}T)^{-t(\alpha+\beta)}dt
```

**Code**: `src/difference_quotient.py:build_bracket_exp_series()`
**Also**: `src/unified_s12_evaluator_v3.py` lines 25-33

**At α = β = -R/L** (Line 1522):
```latex
e^{R[\theta t(x+y)-\theta y+t]} e^{R[\theta t(x+y)-\theta x+t]}
```

### 2. I₁ Main Term Formula

**PRZZ TeX Lines 1529-1533**:
```latex
I_1 = T\widehat{\Phi}(0) \frac{d^2}{dxdy} \frac{\theta(x+y)+1}{\theta}
      \int_0^1 \int_0^1 (1-u)^2 P_1(x+u) P_2(y+u)
      \times e^{R[\theta t(x+y)-\theta y+t]} e^{R[\theta t(x+y)-\theta x+t]}
      \times Q(\theta t(x+y)-\theta y+t) Q(\theta t(x+y)-\theta x+t) |_{x=y=0} du dt
```

**Code**: `src/unified_s12_evaluator_v3.py:compute_unified_I1_v3()`

**Key components**:
| TeX Element | Code Location |
|-------------|---------------|
| (1-u)² | `(1-u)**2` weight in integrand |
| P₁(x+u)P₂(y+u) | `compute_P1_factor()`, `compute_Pell_factor()` |
| Q(θt(x+y)-θy+t) | `compute_Q_factor()` |
| d²/dxdy | `series.get_coeff((1,1))` |

### 3. I₂ Term Formula

**PRZZ TeX Lines 1548**:
```latex
I_2 = T\frac{\widehat{\Phi}(0)}{\theta} \int_0^1 \int_0^1 Q(t)^2 e^{2Rt} P_1(u)P_2(u) dt du
```

**Code**: `src/unified_s12_evaluator_v3.py` (pair (2,2) computation)

**Note**: I₂ is the "no derivative" case (Case B in ω classification).

### 4. I₃ and I₄ Term Formulas

**PRZZ TeX Lines 1562-1570**:
```latex
I_3 = -T\widehat{\Phi}(0) \frac{1+\theta x}{\theta} \frac{d}{dx}
      \int_0^1\int_0^1 (1-u)P_1(x+u)P_2(u) e^{R[2t+2\theta xt-\theta x]}
      \times Q(t+\theta xt) Q(-\theta x+t+\theta xt) dtdu |_{x=0}
```

**Code**: `src/unified_s12_evaluator_v3.py` (pairs involving ℓ₁ ≠ ℓ₂)

### 5. Case A/B/C Classification

**PRZZ TeX Line 2303**:
```latex
\omega(d,\mathbf{l}) := 1 \times l_1 + 2 \times l_2 + \cdots + d \times l_d - 1
```

**For K=3, d=1**:
| Pair | ω(ℓ₁) | ω(ℓ₂) | Case |
|------|-------|-------|------|
| (1,1) | -1 | -1 | A×A |
| (1,2) | -1 | 0 | A×B |
| (1,3) | -1 | 1 | A×C |
| (2,2) | 0 | 0 | B×B |
| (2,3) | 0 | 1 | B×C |
| (3,3) | 1 | 1 | C×C |

**Code**: `src/terms_k3_d1.py` (term structure per pair)

### 6. The U, V, W Constants

**PRZZ TeX Lines 2311, 2330, 2343**:
```latex
\mathcal{U}(d,\mathbf{l}) = \mathbf{1}\{\omega=-1\} (1!(-1)^1)^{l_1}(2!(-1)^2)^{l_2} \cdots
\mathcal{V}(d,\mathbf{l}) = \mathbf{1}\{\omega=0\} (1!(-1)^1)^{l_1}(2!(-1)^2)^{l_2} \cdots
\mathcal{W}(d,\mathbf{l}) = \mathbf{1}\{\omega>0\} (1!(-1)^1)^{l_1}(2!(-1)^2)^{l_2} \cdots
```

**Code**: Sign factors are tracked in `src/term_dsl.py` and pair-specific terms.

### 7. Euler-Maclaurin Lemma

**PRZZ TeX Lines 2391-2414**:
```latex
\sum_{n \le z} \frac{g(n)}{n^{1+s}} F(\frac{\log(x/n)}{\log x}) H(\frac{\log(z/n)}{\log z})
= \frac{c_g \log^{k_g} z}{z^s} \int_0^1 (1-u)^{k_g-1} F(1-(1-u)\frac{\log z}{\log x}) H(u) z^{us} du
```

**Code**: This is the theoretical basis for replacing sums with integrals.
The (1-u)^{k-1} weight appears in the integrand structure.

---

## Mirror Assembly

### 8. Mirror Term Formula

**PRZZ TeX Line 1502**:
```latex
I_1(\alpha,\beta) = I_{1,1}(\alpha,\beta) + T^{-\alpha-\beta}I_{1,1}(-\beta,-\alpha) + O(T/L)
```

**Code**: `src/kappa_engine.py:compute_c_from_integrals()`
```python
c = S12_plus + m * S12_minus + S34_plus
```

**The m factor** (empirical but validated):
```python
m = (f_I1 * g_I1 + (1 - f_I1) * g_I2) * base
base = exp(R) + (2*K - 1)  # = exp(R) + 5 for K=3
```

### 9. Correction Factors g_I1, g_I2

**Code**: `src/kappa_engine.py:compute_g_I1()`, `compute_g_I2()`

**First-principles formulas**:
```python
g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)
g_I2 = 1 + θ(2-θ) / (2K(2K+1))
```

**Note**: These correct for the difference between unified-bracket and separated-R computation.

---

## Polynomial Representations

### 10. P₁ Polynomial

**PRZZ TeX numerical section (line ~2570)**:
```latex
P_1(x) = x + 0.261076 x(1-x) - 1.071007 x(1-x)^2 - 0.236840 x(1-x)^3 + 0.260233 x(1-x)^4
```

**Code**: `src/polynomials.py:P1Polynomial`
```python
P1(x) = x + x(1-x) * P_tilde(1-x)
# where P_tilde uses tilde_coeffs = [0.261076, -1.071007, -0.23684, 0.260233]
```

### 11. P₂, P₃ Polynomials (Pell form)

**PRZZ TeX**:
```latex
P_2(x) = 1.048274 x + 1.319912 x^2 - 0.940058 x^3
P_3(x) = 0.522811 x - 0.686510 x^2 - 0.049923 x^3
```

**Code**: `src/polynomials.py:PellPolynomial`
```python
P_ell(x) = x * P_tilde(x)
# P_tilde is in monomial basis
```

### 12. Q Polynomial

**PRZZ TeX**:
```latex
Q(x) = 0.490464 + 0.636851(1-2x) - 0.159327(1-2x)^3 + 0.032011(1-2x)^5
```

**Code**: `src/polynomials.py:QPolynomial`
```python
# Uses (1-2x)^k basis
Q(0) = sum of coefficients = 1 (enforced)
```

---

## Validation Checkpoints

| Checkpoint | PRZZ Reference | Code Test |
|------------|----------------|-----------|
| κ = 0.417293962 | Line 2573 | `test_golden_regression.py` |
| R = 1.3036, θ = 4/7 | Lines 2569-2570 | `data/przz_parameters.json` |
| Polynomial constraints | Implicit | `test_polynomials.py` |
| Quadrature convergence | Implicit | `test_out_of_sample_smoke.py` |
| No calibrated constants | N/A | `test_production_guards.py` |

---

## Gaps to Fill in Paper

1. **Line 2138 confession**: "A way to automate this process would be most welcome"
   - We have automated it with the unified evaluator

2. **9 cross-terms → 6 by symmetry** (Line 2387)
   - Need explicit table in paper appendix

3. **Why negative pair contributions are valid**
   - PRZZ doesn't explicitly state this
   - Need mathematical argument

4. **The "5" in m = exp(R) + 5**
   - Origin is (2K-1) for K=3 pieces
   - Need derivation from PRZZ mirror structure

---

## Files Cross-Reference

| Purpose | Primary File | PRZZ Section |
|---------|--------------|--------------|
| S12 evaluation | `unified_s12_evaluator_v3.py` | §8 |
| κ computation | `kappa_engine.py` | §10 |
| Polynomials | `polynomials.py` | §5, §10 |
| Term structure | `terms_k3_d1.py` | §8 |
| Quadrature | `quadrature.py` | N/A (numerical) |
| Series algebra | `series.py` | §8 (derivatives) |
