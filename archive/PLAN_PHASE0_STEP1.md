# Plan: Phase 0, Step 1 — polynomials.py + tests (Final)

## Overview

**Objective:** Implement the polynomial module that represents and evaluates the P₁, P₂, P₃, Q polynomials used in PRZZ, with built-in constraint enforcement.

**Files to create:**
- `src/polynomials.py` — Core polynomial implementation
- `tests/test_polynomials.py` — Unit and validation tests

**Acceptance criteria:**
1. Constraints automatically enforced in parameterized mode; paper-literal mode exists for reproducing printed coefficients
2. PRZZ polynomials exactly reproduced from stored parameters
3. Vectorized evaluation and derivative computation (no finite differences)
4. All constrained derivatives computed via cached monomial conversion (no manual product rule)
5. All tests passing, including JSON self-consistency check

---

## Mathematical Specification

### Constraint Summary

| Polynomial | Constraints | Parameterization | Free Parameters |
|------------|-------------|------------------|-----------------|
| P₁ | P₁(0)=0, P₁(1)=1 | P₁(x) = x + x(1-x)·P̃(x) | Coefficients of P̃ |
| P₂ | P₂(0)=0 | P₂(x) = x·P̃₂(x) | Coefficients of P̃₂ |
| P₃ | P₃(0)=0 | P₃(x) = x·P̃₃(x) | Coefficients of P̃₃ |
| Q | Q(0)=1 | c₀ = 1 - Σₖ cₖ (computed, not stored) | Non-constant coefficients c₁, c₃, c₅, ... |

### Key Design Decision: Q(0)=1 Enforcement

**Problem:** PRZZ printed coefficients give Q(0) ≈ 0.999999, not exactly 1.0. This is coefficient rounding, but can cause ~1e-5 discrepancies in c if not handled.

**Solution:** Two modes:

1. **Constraint mode (default for optimization):**
   - Store only non-constant coefficients: `{1: c₁, 3: c₃, 5: c₅, ...}`
   - Compute c₀ = 1 - Σₖ cₖ internally
   - Guarantees Q(0) = 1 exactly

2. **Paper-literal mode (for PRZZ reproduction):**
   - Store all coefficients including c₀ exactly as printed
   - Accept Q(0) ≈ 1 (not exactly 1)

**Implementation:** Boolean flag `enforce_Q0=True` in constructor.

---

### PRZZ Polynomial Values (from `przz_parameters.json`)

**P₁:** Constrained form with P̃ in (1-x) powers
```
P₁(x) = x + x(1-x)·P̃(x)
P̃(x) = 0.261076 - 1.071007(1-x) - 0.236840(1-x)² + 0.260233(1-x)³
```
Free parameters: `p_tilde_coeffs = [0.261076, -1.071007, -0.236840, 0.260233]`

**P₂:** x·P̃₂(x) form
```
P₂(x) = x·(1.048274 + 1.319912x - 0.940058x²)
```
Free parameters: `tilde_coeffs = [1.048274, 1.319912, -0.940058]`

**P₃:** x·P̃₃(x) form
```
P₃(x) = x·(0.522811 - 0.686510x - 0.049923x²)
```
Free parameters: `tilde_coeffs = [0.522811, -0.686510, -0.049923]`

**Q:** PRZZ (1-2x) basis
```
Q(x) = c₀ + c₁(1-2x) + c₃(1-2x)³ + c₅(1-2x)⁵
```
PRZZ printed values:
- c₀ = 0.490464
- c₁ = 0.636851
- c₃ = -0.159327
- c₅ = 0.032011

Sum: 0.490464 + 0.636851 - 0.159327 + 0.032011 = 0.999999 (not exactly 1.0 due to rounding)

---

## Implementation Design

### Class Structure

```python
class Polynomial:
    """
    Base polynomial in monomial basis: P(x) = Σₖ cₖ xᵏ

    This is the workhorse class. All constrained polynomials convert to this
    for evaluation and derivatives, avoiding manual product/chain rule.
    """
    coeffs: np.ndarray  # [c₀, c₁, ..., cₙ]

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate using Horner's method."""

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluate k-th derivative analytically."""

    @property
    def degree(self) -> int:
        """Polynomial degree."""


class P1Polynomial:
    """
    P₁(x) = x + x(1-x)·P̃(x)

    Constraints: P₁(0)=0, P₁(1)=1 (automatically enforced)
    Free parameters: coefficients of P̃ in (1-x) powers
    """
    tilde_coeffs: np.ndarray  # P̃ coefficients
    _monomial: Polynomial     # Cached monomial expansion

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate via cached monomial."""
        return self._monomial.eval(x)

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Derivative via cached monomial."""
        return self._monomial.eval_deriv(x, k)

    def to_monomial(self) -> Polynomial:
        """Return cached monomial representation."""
        return self._monomial

    def _build_monomial(self) -> Polynomial:
        """Expand x + x(1-x)·P̃(x) to monomial form."""


class PellPolynomial:
    """
    P_ℓ(x) = x·P̃(x) for ℓ ≥ 2

    Constraint: P(0)=0 (automatically enforced by x factor)
    Free parameters: coefficients of P̃ in monomial basis
    """
    tilde_coeffs: np.ndarray  # P̃ = [c₀, c₁, c₂, ...] where P̃(x) = Σ cₖxᵏ
    _monomial: Polynomial     # Cached: x * P̃(x)

    def eval(self, x: np.ndarray) -> np.ndarray
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray
    def to_monomial(self) -> Polynomial


class QPolynomial:
    """
    Q(x) = Σₖ cₖ (1-2x)ᵏ in PRZZ basis

    Constraint: Q(0)=1

    Two modes:
    - enforce_Q0=True: c₀ computed as 1 - Σₖ>₀ cₖ (default for optimization)
    - enforce_Q0=False: c₀ stored exactly (for paper-literal reproduction)

    Supports arbitrary powers k (typically odd: 1, 3, 5, ..., plus k=0)
    """
    basis_coeffs: Dict[int, float]  # k -> cₖ for (1-2x)ᵏ
    enforce_Q0: bool
    _monomial: Polynomial  # Cached monomial expansion

    def eval(self, x: np.ndarray) -> np.ndarray
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray
    def to_monomial(self) -> Polynomial
    def Q_at_zero(self) -> float:
        """Return Q(0) - should be 1.0 or very close."""

    def _build_monomial(self) -> Polynomial:
        """Expand Σ cₖ(1-2x)ᵏ via binomial theorem."""
```

### Factory Functions

```python
def load_przz_polynomials(enforce_Q0: bool = False) -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial, QPolynomial]:
    """
    Load PRZZ polynomials from przz_parameters.json.

    Args:
        enforce_Q0: If True, recompute c₀ so Q(0)=1 exactly.
                    If False, use printed c₀ value.
    """

def make_P1_from_tilde(tilde_coeffs: List[float]) -> P1Polynomial:
    """Create P₁ from P̃ coefficients in (1-x) basis."""

def make_Pell_from_tilde(tilde_coeffs: List[float]) -> PellPolynomial:
    """Create P_ℓ from P̃ monomial coefficients. P(x) = x·P̃(x)."""

def make_Q_from_basis(basis_coeffs: Dict[int, float], enforce_Q0: bool = True) -> QPolynomial:
    """
    Create Q from (1-2x)ᵏ basis coefficients.

    Args:
        basis_coeffs: {k: cₖ} for (1-2x)ᵏ terms
        enforce_Q0: If True, ignore any provided c₀ and compute it as 1 - Σₖ>₀ cₖ
    """
```

### Derivative Computation (All via Monomial Form)

**Key principle:** All constrained polynomial classes convert to monomial form once at construction, then delegate eval/eval_deriv to the base `Polynomial` class.

**Benefits:**
- No manual product rule / chain rule implementation
- Single, well-tested derivative code path
- Tests naturally verify correctness via roundtrip

```python
def falling_factorial(n: int, k: int) -> int:
    """
    Compute n(n-1)...(n-k+1) = n!/(n-k)!

    Returns 0 if k > n (safe guardrail).
    Returns 1 if k == 0.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k > n:
        return 0
    if k == 0:
        return 1
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


class Polynomial:
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Evaluate k-th derivative at points x.

        For P(x) = Σⱼ cⱼ xʲ:
        P⁽ᵏ⁾(x) = Σⱼ≥ₖ cⱼ · falling_factorial(j,k) · xʲ⁻ᵏ
        """
        if k > self.degree:
            return np.zeros_like(x, dtype=float)

        deriv_coeffs = np.zeros(len(self.coeffs) - k)
        for j in range(k, len(self.coeffs)):
            deriv_coeffs[j - k] = self.coeffs[j] * falling_factorial(j, k)

        return Polynomial(deriv_coeffs).eval(x)
```

### Monomial Expansion for Constrained Forms

**P₁:**
```python
def _build_monomial(self) -> Polynomial:
    """
    Expand P₁(x) = x + x(1-x)·P̃(x) where P̃(y) = Σₘ aₘ yᵐ with y=(1-x).

    P̃(1-x) = Σₘ aₘ (1-x)ᵐ
    x(1-x)·P̃(1-x) = x(1-x)·Σₘ aₘ (1-x)ᵐ = Σₘ aₘ · x(1-x)ᵐ⁺¹

    (1-x)ᵐ = Σₖ C(m,k)(-x)ᵏ = Σₖ C(m,k)(-1)ᵏ xᵏ
    x(1-x)ᵐ = Σₖ C(m,k)(-1)ᵏ xᵏ⁺¹

    Final: P₁(x) = x + Σₘ aₘ · [Σₖ C(m+1,k)(-1)ᵏ xᵏ⁺¹]
    """
```

**P_ℓ:**
```python
def _build_monomial(self) -> Polynomial:
    """P_ℓ(x) = x·P̃(x) = x·Σₖ cₖxᵏ = Σₖ cₖxᵏ⁺¹"""
    # Just shift coefficients by 1 degree
    return Polynomial(np.concatenate([[0], self.tilde_coeffs]))
```

**Q:**
```python
def _build_monomial(self) -> Polynomial:
    """
    Expand Q(x) = Σₖ cₖ(1-2x)ᵏ.

    (1-2x)ᵏ = Σⱼ C(k,j)(-2x)ʲ = Σⱼ C(k,j)(-2)ʲ xʲ

    Collect all terms, combine like powers.
    """
```

---

## Test Specification

### test_polynomials.py

#### 0. JSON Self-Consistency Test

```python
def test_json_self_consistency():
    """Verify przz_parameters.json internal consistency."""
    import json
    from pathlib import Path

    json_path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
    with open(json_path) as f:
        data = json.load(f)

    R = data["configuration"]["R"]
    c = data["targets"]["c"]
    kappa = data["targets"]["kappa"]

    # Check κ = 1 - log(c)/R
    computed_kappa = 1 - np.log(c) / R
    assert abs(computed_kappa - kappa) < 1e-8, f"κ mismatch: {computed_kappa} vs {kappa}"

    # Check c = exp(R(1-κ))
    computed_c = np.exp(R * (1 - kappa))
    assert abs(computed_c - c) < 1e-10, f"c mismatch: {computed_c} vs {c}"
```

#### 1. Constraint Enforcement Tests (Automatic by Design)

```python
def test_P1_constraints_automatic():
    """P₁(0)=0 and P₁(1)=1 for ANY P̃ coefficients."""
    rng = np.random.default_rng(42)  # Deterministic
    x = np.array([0.0, 1.0])

    for _ in range(20):
        random_tilde = rng.standard_normal(5).tolist()
        p1 = make_P1_from_tilde(random_tilde)
        vals = p1.eval(x)
        assert abs(vals[0]) < 1e-14, "P₁(0) should be 0"
        assert abs(vals[1] - 1.0) < 1e-14, "P₁(1) should be 1"

def test_Pell_zero_at_zero_automatic():
    """P_ℓ(0)=0 for ANY P̃ coefficients."""
    rng = np.random.default_rng(123)

    for _ in range(20):
        random_tilde = rng.standard_normal(4).tolist()
        p = make_Pell_from_tilde(random_tilde)
        assert abs(p.eval(np.array([0.0]))[0]) < 1e-14

def test_Q_one_at_zero_enforced():
    """Q(0)=1 exactly when enforce_Q0=True."""
    # Use coefficients that don't naturally sum to 1
    q = make_Q_from_basis({1: 0.5, 3: 0.3, 5: -0.1}, enforce_Q0=True)
    assert abs(q.eval(np.array([0.0]))[0] - 1.0) < 1e-14

def test_Q_paper_literal_mode():
    """Q(0) uses stored c₀ when enforce_Q0=False."""
    coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
    q = make_Q_from_basis(coeffs, enforce_Q0=False)
    expected_Q0 = sum(coeffs.values())  # ≈ 0.999999
    assert abs(q.eval(np.array([0.0]))[0] - expected_Q0) < 1e-10
```

#### 2. PRZZ Polynomial Reproduction (Grid Comparison)

```python
def test_przz_P1_grid():
    """P₁ matches expected values across [0,1]."""
    P1, _, _, _ = load_przz_polynomials()
    x = np.linspace(0, 1, 21)

    # Compute expected from explicit formula
    tilde_coeffs = [0.261076, -1.071007, -0.236840, 0.260233]
    def p_tilde(x):
        y = 1 - x
        return sum(c * y**i for i, c in enumerate(tilde_coeffs))
    expected = x + x * (1 - x) * p_tilde(x)

    assert np.allclose(P1.eval(x), expected, rtol=1e-12)

def test_przz_P2_grid():
    """P₂ matches expected values across [0,1]."""
    _, P2, _, _ = load_przz_polynomials()
    x = np.linspace(0, 1, 21)
    expected = 1.048274*x + 1.319912*x**2 - 0.940058*x**3
    assert np.allclose(P2.eval(x), expected, rtol=1e-12)

def test_przz_P3_grid():
    """P₃ matches expected values across [0,1]."""
    _, _, P3, _ = load_przz_polynomials()
    x = np.linspace(0, 1, 21)
    expected = 0.522811*x - 0.686510*x**2 - 0.049923*x**3
    assert np.allclose(P3.eval(x), expected, rtol=1e-12)

def test_przz_Q_grid():
    """Q matches expected values across [0,1]."""
    _, _, _, Q = load_przz_polynomials(enforce_Q0=False)
    x = np.linspace(0, 1, 21)

    u = 1 - 2*x  # basis variable
    expected = 0.490464 + 0.636851*u - 0.159327*u**3 + 0.032011*u**5
    assert np.allclose(Q.eval(x), expected, rtol=1e-10)
```

#### 3. Endpoint Behavior Tests

```python
def test_endpoint_values():
    """Key endpoint values for all polynomials."""
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)

    # P₁(0)=0, P₁(1)=1
    assert abs(P1.eval(np.array([0.0]))[0]) < 1e-14
    assert abs(P1.eval(np.array([1.0]))[0] - 1.0) < 1e-14

    # P₂(0)=0, P₃(0)=0
    assert abs(P2.eval(np.array([0.0]))[0]) < 1e-14
    assert abs(P3.eval(np.array([0.0]))[0]) < 1e-14

    # Q(1/2): since (1-2x)=0 at x=1/2, Q(1/2) = c₀
    assert abs(Q.eval(np.array([0.5]))[0] - 0.490464) < 1e-10
```

#### 4. Derivative Tests (for ALL polynomial types)

```python
def test_polynomial_derivative_analytic():
    """Base Polynomial derivative matches analytic formula."""
    # P(x) = 1 + 2x + 3x² + 4x³
    p = Polynomial(np.array([1.0, 2.0, 3.0, 4.0]))
    x = np.linspace(0.1, 0.9, 20)

    # P'(x) = 2 + 6x + 12x²
    assert np.allclose(p.eval_deriv(x, 1), 2 + 6*x + 12*x**2, rtol=1e-12)
    # P''(x) = 6 + 24x
    assert np.allclose(p.eval_deriv(x, 2), 6 + 24*x, rtol=1e-12)
    # P'''(x) = 24
    assert np.allclose(p.eval_deriv(x, 3), np.full_like(x, 24.0), rtol=1e-12)
    # P''''(x) = 0
    assert np.allclose(p.eval_deriv(x, 4), np.zeros_like(x), rtol=1e-12)

def test_P1_derivative_via_monomial():
    """P₁ derivative matches monomial conversion."""
    P1, _, _, _ = load_przz_polynomials()
    x = np.linspace(0.05, 0.95, 20)

    p1_mono = P1.to_monomial()
    for k in range(1, 5):
        assert np.allclose(P1.eval_deriv(x, k), p1_mono.eval_deriv(x, k), rtol=1e-12)

def test_P2_derivative_via_monomial():
    """P₂ derivative matches monomial conversion."""
    _, P2, _, _ = load_przz_polynomials()
    x = np.linspace(0.05, 0.95, 20)

    p2_mono = P2.to_monomial()
    for k in range(1, 4):
        assert np.allclose(P2.eval_deriv(x, k), p2_mono.eval_deriv(x, k), rtol=1e-12)

def test_P3_derivative_via_monomial():
    """P₃ derivative matches monomial conversion."""
    _, _, P3, _ = load_przz_polynomials()
    x = np.linspace(0.05, 0.95, 20)

    p3_mono = P3.to_monomial()
    for k in range(1, 4):
        assert np.allclose(P3.eval_deriv(x, k), p3_mono.eval_deriv(x, k), rtol=1e-12)

def test_Q_derivative_via_monomial():
    """Q derivative matches monomial conversion."""
    _, _, _, Q = load_przz_polynomials()
    x = np.linspace(0.05, 0.95, 20)

    q_mono = Q.to_monomial()
    for k in range(1, 4):
        assert np.allclose(Q.eval_deriv(x, k), q_mono.eval_deriv(x, k), rtol=1e-10)
```

#### 5. Vectorization Tests

```python
def test_vectorized_1d():
    """Evaluation works on 1D arrays."""
    p = Polynomial(np.array([1.0, 2.0, 3.0]))
    x = np.linspace(0, 1, 100)
    assert p.eval(x).shape == (100,)
    assert p.eval_deriv(x, 1).shape == (100,)

def test_vectorized_2d():
    """Evaluation works on 2D arrays (quadrature grids)."""
    p = Polynomial(np.array([1.0, 2.0, 3.0]))
    x = np.random.rand(50, 50)
    assert p.eval(x).shape == (50, 50)
    assert p.eval_deriv(x, 1).shape == (50, 50)
```

#### 6. Monomial Conversion Roundtrip Tests

```python
def test_P1_to_monomial_roundtrip():
    """P₁ -> monomial -> evaluate gives same result."""
    P1, _, _, _ = load_przz_polynomials()
    x = np.linspace(0, 1, 50)

    direct = P1.eval(x)
    via_monomial = P1.to_monomial().eval(x)
    assert np.allclose(direct, via_monomial, rtol=1e-12)

def test_Pell_to_monomial_roundtrip():
    """P_ℓ -> monomial -> evaluate gives same result."""
    _, P2, P3, _ = load_przz_polynomials()
    x = np.linspace(0, 1, 50)

    for P in [P2, P3]:
        direct = P.eval(x)
        via_monomial = P.to_monomial().eval(x)
        assert np.allclose(direct, via_monomial, rtol=1e-12)

def test_Q_to_monomial_roundtrip():
    """Q -> monomial -> evaluate gives same result."""
    _, _, _, Q = load_przz_polynomials()
    x = np.linspace(0, 1, 50)

    direct = Q.eval(x)
    via_monomial = Q.to_monomial().eval(x)
    assert np.allclose(direct, via_monomial, rtol=1e-10)
```

#### 7. Falling Factorial and Edge Cases

```python
def test_falling_factorial():
    """falling_factorial matches expected values."""
    assert falling_factorial(5, 0) == 1
    assert falling_factorial(5, 1) == 5
    assert falling_factorial(5, 2) == 20  # 5*4
    assert falling_factorial(5, 3) == 60  # 5*4*3
    assert falling_factorial(5, 5) == 120  # 5!

def test_falling_factorial_k_greater_than_n():
    """falling_factorial returns 0 when k > n."""
    assert falling_factorial(3, 4) == 0
    assert falling_factorial(0, 1) == 0
    assert falling_factorial(5, 10) == 0

def test_derivative_higher_than_degree():
    """Derivative order > degree returns zero."""
    p = Polynomial(np.array([1.0, 2.0, 3.0]))  # degree 2
    x = np.linspace(0, 1, 10)
    assert np.allclose(p.eval_deriv(x, 3), 0.0)
    assert np.allclose(p.eval_deriv(x, 10), 0.0)
```

---

## Implementation Steps

### Step 1: Implement `falling_factorial` helper
- Handle k=0 (returns 1)
- Handle k>n (returns 0, safe guardrail)
- Clean implementation

### Step 2: Implement `Polynomial` base class
- Monomial representation: `coeffs[k]` = coefficient of xᵏ
- Horner evaluation
- Analytic derivative via falling factorial
- `degree` property

### Step 3: Implement `P1Polynomial`
- Constructor: accept P̃ coefficients in (1-x) basis
- `_build_monomial()`: expand x + x(1-x)·P̃(1-x) via binomial theorem
- Cache monomial form at construction
- Delegate `eval`/`eval_deriv` to cached monomial

### Step 4: Implement `PellPolynomial`
- Constructor: accept P̃ coefficients in monomial basis
- `_build_monomial()`: simply prepend 0 to shift degrees by 1
- Cache and delegate

### Step 5: Implement `QPolynomial`
- Constructor: accept basis_coeffs dict and enforce_Q0 flag
- If enforce_Q0: compute c₀ = 1 - Σₖ>₀ cₖ
- `_build_monomial()`: expand each (1-2x)ᵏ via binomial and sum
- Cache and delegate

### Step 6: Implement factory functions
- `load_przz_polynomials()`: read from JSON, support enforce_Q0 flag
- `make_P1_from_tilde()`, `make_Pell_from_tilde()`, `make_Q_from_basis()`

### Step 7: Write all tests
- Run tests incrementally as classes are implemented

---

## Numerical Considerations

1. **Horner's method** for stable polynomial evaluation
2. **Analytic derivatives only** (no finite differences per Rule A)
3. **Binomial coefficients** computed via math.comb (exact integers)
4. **Float64 precision** sufficient; tolerances:
   - Constraint enforcement: 1e-14
   - PRZZ reproduction: 1e-12 for P, 1e-10 for Q
   - Derivative roundtrips: 1e-10 to 1e-12
5. **falling_factorial(k>n) = 0** as safe guardrail

---

## Validation Checkpoints

1. [ ] JSON self-consistency: κ = 1 - log(c)/R verified
2. [ ] `falling_factorial` helper: correct values, edge cases (k=0, k>n) handled
3. [ ] `Polynomial` class: eval and eval_deriv match analytic formulas
4. [ ] `P1Polynomial`: P₁(0)=0, P₁(1)=1 for any P̃ (20 random tests)
5. [ ] `PellPolynomial`: P(0)=0 for any P̃ (20 random tests)
6. [ ] `QPolynomial`: Q(0)=1 exactly when enforce_Q0=True
7. [ ] `QPolynomial`: Q(0)=stored when enforce_Q0=False
8. [ ] PRZZ polynomials loaded and match expected values on grid
9. [ ] All derivatives computed analytically via cached monomial
10. [ ] Monomial conversion roundtrips correctly for all types
11. [ ] All tests passing

---

## Dependencies

- `numpy` for array operations
- `math` for `comb` (binomial coefficients)
- `json` for loading parameters
- `pytest` for testing

No external dependencies beyond standard scientific Python.

---

## Estimated Complexity

- `polynomials.py`: ~300-350 lines
- `test_polynomials.py`: ~250-300 lines
- Straightforward implementation using binomial expansion

This is the foundation for all subsequent modules.

---

## Future-Proofing Notes

1. **Q supports arbitrary powers:** Dict[int, float] allows any k, not just 0,1,3,5
2. **All constrained classes are parameterization-based:** Ready for optimization
3. **Cached monomial form:** Evaluation and derivatives are fast for repeated calls
4. **Two modes for Q:** Supports both exact reproduction and constrained optimization
