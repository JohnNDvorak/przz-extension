# VALIDATION.md — Test Plan and Checkpoints

## Philosophy

We cannot trust any κ improvement unless it survives:
- Quadrature refinement
- Regression against PRZZ
- Multiple internal invariants

This file defines required tests before advancing phases.

---

## Unit Tests

### test_polynomials.py

**Constraint tests:**
```python
def test_P1_constraints():
    """P₁(0)=0 and P₁(1)=1 enforced by parameterization."""
    p1 = make_P1_from_params(random_params)
    assert abs(p1.eval(0.0)) < 1e-14
    assert abs(p1.eval(1.0) - 1.0) < 1e-14

def test_Pell_zero_at_zero():
    """P_ℓ(0)=0 enforced for ℓ≥2."""
    for ell in [2, 3, 4]:
        p = make_Pell_from_params(ell, random_params)
        assert abs(p.eval(0.0)) < 1e-14

def test_Q_one_at_zero():
    """Q(0)=1 enforced by parameterization."""
    q = make_Q_from_params(random_params)
    assert abs(q.eval(0.0) - 1.0) < 1e-14
```

**Derivative tests:**
```python
def test_polynomial_derivatives():
    """eval_deriv matches finite-difference (sanity check on low degree)."""
    p = Polynomial([1, 2, 3, 4])  # 1 + 2x + 3x² + 4x³
    x = np.linspace(0.1, 0.9, 10)
    
    # Analytic: P'(x) = 2 + 6x + 12x²
    analytic = 2 + 6*x + 12*x**2
    computed = p.eval_deriv(x, 1)
    assert np.allclose(analytic, computed)
```

**PRZZ polynomial reproduction:**
```python
def test_przz_polynomials_exact():
    """Verify we can reproduce PRZZ's published polynomials."""
    P1, P2, P3, Q = load_przz_polynomials()
    
    # Test at specific points
    assert abs(P1.eval(0.5) - expected_P1_at_half) < 1e-10
    # ... etc
```

---

### test_quadrature.py

**1D accuracy:**
```python
def test_1d_monomials():
    """∫₀¹ x^k dx = 1/(k+1)."""
    for n in [20, 40, 60]:
        nodes, weights = gauss_legendre_01(n)
        for k in range(2*n - 1):  # Exact up to degree 2n-1
            integral = np.sum(weights * nodes**k)
            expected = 1.0 / (k + 1)
            assert abs(integral - expected) < 1e-12

def test_1d_smooth_function():
    """Test on known smooth integral."""
    # ∫₀¹ exp(-x²) dx = √π/2 * erf(1) ≈ 0.7468...
    exact = 0.7468241328124271
    for n in [20, 40, 60]:
        nodes, weights = gauss_legendre_01(n)
        integral = np.sum(weights * np.exp(-nodes**2))
        assert abs(integral - exact) < 1e-10
```

**2D accuracy:**
```python
def test_2d_separable():
    """∫∫ u^a t^b du dt = 1/((a+1)(b+1))."""
    U, T, W = tensor_grid_2d(40)
    for a in range(5):
        for b in range(5):
            integral = np.sum(W * U**a * T**b)
            expected = 1.0 / ((a+1) * (b+1))
            assert abs(integral - expected) < 1e-12

def test_2d_convergence():
    """Verify convergence as n increases."""
    def f(u, t):
        return np.exp(-u*t) * np.sin(np.pi * u)
    
    results = []
    for n in [40, 60, 80, 100]:
        U, T, W = tensor_grid_2d(n)
        results.append(np.sum(W * f(U, T)))
    
    # Should converge: |r[i] - r[i+1]| decreasing
    diffs = [abs(results[i] - results[i+1]) for i in range(3)]
    assert diffs[1] < diffs[0]
    assert diffs[2] < diffs[1]
```

---

### test_series.py

**Bitset operations:**
```python
def test_mask_multiplication():
    """Overlapping masks should vanish."""
    s1 = TruncatedSeries(("x", "y", "z"))
    s1.coeffs[0b001] = np.array([1.0])  # x term
    s1.coeffs[0b010] = np.array([2.0])  # y term
    
    s2 = TruncatedSeries(("x", "y", "z"))
    s2.coeffs[0b001] = np.array([3.0])  # x term
    s2.coeffs[0b100] = np.array([4.0])  # z term
    
    result = s1.multiply(s2)
    
    # x*x should vanish (overlap)
    # y*x = xy should exist: 2*3 = 6
    # y*z = yz should exist: 2*4 = 8
    assert 0b011 in result.coeffs  # xy
    assert abs(result.coeffs[0b011][0] - 6.0) < 1e-14
    assert 0b110 in result.coeffs  # yz
    assert abs(result.coeffs[0b110][0] - 8.0) < 1e-14
```

**Coefficient extraction:**
```python
def test_derivative_extraction():
    """Extract ∂²/∂x∂y coefficient correctly."""
    # Build series for (1 + x)(1 + 2y) = 1 + x + 2y + 2xy
    s = TruncatedSeries(("x", "y"))
    s.coeffs[0b00] = np.array([1.0])  # constant
    s.coeffs[0b01] = np.array([1.0])  # x
    s.coeffs[0b10] = np.array([2.0])  # y
    s.coeffs[0b11] = np.array([2.0])  # xy
    
    # ∂²/∂x∂y at 0 should be 2 (coefficient of xy times 1!*1!)
    result = s.extract_derivative({"x": 1, "y": 1})
    assert abs(result[0] - 2.0) < 1e-14
```

**Exponential expansion:**
```python
def test_exp_expansion():
    """exp(R(a + bx + cy)) with x²=y²=0."""
    # exp(1 + 2x + 3y) = e * (1 + 2x) * (1 + 3y)
    #                  = e * (1 + 2x + 3y + 6xy)
    R = 1.0
    a0, bx, cy = 1.0, 2.0, 3.0
    
    s = expand_exponential(R, a0, {"x": bx, "y": cy}, vars=("x", "y"))
    
    e = np.exp(1.0)
    assert abs(s.coeffs[0b00][0] - e) < 1e-12        # constant
    assert abs(s.coeffs[0b01][0] - 2*e) < 1e-12     # x
    assert abs(s.coeffs[0b10][0] - 3*e) < 1e-12     # y
    assert abs(s.coeffs[0b11][0] - 6*e) < 1e-12     # xy
```

**Polynomial expansion:**
```python
def test_polynomial_expansion():
    """P(u + ax + by) expansion."""
    # P(z) = z + z² = z(1+z)
    # P(u + x) = (u+x)(1+u+x) with x²=0
    #          = u(1+u) + x(1+u) + ux + x(1+u)  ... need to be careful
    
    # Simpler test: P(z) = z
    # P(u + x) = u + x
    p = Polynomial([0, 1])  # P(z) = z
    
    # At u=0.5: P(0.5 + x) = 0.5 + x
    s = expand_polynomial_at(p, u_val=0.5, var_coeffs={"x": 1.0}, vars=("x",))
    assert abs(s.coeffs[0b0][0] - 0.5) < 1e-14  # constant
    assert abs(s.coeffs[0b1][0] - 1.0) < 1e-14  # x coefficient
```

---

## Integration Tests

### Stage 1: (1,1) only

```python
def test_pair_11_stability():
    """(1,1) contribution is stable under quadrature refinement."""
    params = load_przz_params()
    
    results = {}
    for n in [60, 80, 100]:
        grid = make_grid(n)
        results[n] = compute_pair(1, 1, params, grid)
    
    # Should be stable to ~1e-6 or better
    assert abs(results[80] - results[100]) < 1e-5
    assert abs(results[60] - results[100]) < 1e-4

def test_pair_11_symmetry():
    """Swapping P₁↔P₂ and x↔y should give same result."""
    # This is actually c_{11}, not c_{12}, so symmetry is trivial
    # But we can test that our implementation respects it
    pass
```

### Stage 2: Add pairs incrementally

```python
def test_pair_12_equals_21():
    """c_{12} should equal c_{21}."""
    params = load_przz_params()
    grid = make_grid(80)
    
    c12 = compute_pair(1, 2, params, grid)
    c21 = compute_pair(2, 1, params, grid)
    
    assert abs(c12 - c21) < 1e-10

def test_all_pairs_stable():
    """All pairs stable under quadrature refinement."""
    params = load_przz_params()
    
    for ell1 in range(1, 4):
        for ell2 in range(ell1, 4):
            results = {}
            for n in [60, 80, 100]:
                grid = make_grid(n)
                results[n] = compute_pair(ell1, ell2, params, grid)
            
            # Log the values for debugging
            print(f"c_{ell1}{ell2}: n=60: {results[60]:.10f}, "
                  f"n=80: {results[80]:.10f}, n=100: {results[100]:.10f}")
            
            assert abs(results[80] - results[100]) < 1e-5
```

### Stage 3: Full PRZZ regression

```python
def test_przz_full_regression():
    """Full K=3 assembly matches PRZZ published values."""
    params = load_przz_params()
    grid = make_grid(100)
    
    c_total, breakdown = compute_c(K=3, params=params, grid=grid)
    kappa = compute_kappa(c_total, params["R"])
    
    # Target values
    c_target = 2.13745440613217263636
    kappa_target = 0.417293962
    
    # Tolerances
    assert abs(c_total - c_target) < 1e-5
    assert abs(kappa - kappa_target) < 1e-6
    
    # Log breakdown for reference
    print("Per-pair breakdown:")
    for key, val in sorted(breakdown.items()):
        print(f"  {key}: {val:.10f}")
```

---

## Quadrature Convergence Gate

**Required for any claimed improvement:**

```python
def validate_improvement(old_kappa, new_params):
    """
    Validate that a claimed improvement is real, not quadrature noise.
    """
    results = {}
    for n in [60, 80, 100]:
        grid = make_grid(n)
        c, _ = compute_c(K=3, params=new_params, grid=grid)
        results[n] = compute_kappa(c, new_params["R"])
    
    # Must be consistent across quadrature levels
    spread = max(results.values()) - min(results.values())
    if spread > 2e-6:
        raise ValueError(f"Quadrature unstable: spread = {spread}")
    
    # Must be a genuine improvement
    improvement = results[100] - old_kappa
    if improvement < 1e-6:
        raise ValueError(f"Improvement {improvement} is within noise")
    
    return results[100]
```

---

## Invariants / Sanity Checks

### 1. Symmetry
```python
def check_symmetry_invariant(K, params, grid):
    """c_{ij} = c_{ji} for all pairs."""
    for i in range(1, K+1):
        for j in range(i+1, K+1):
            c_ij = compute_pair(i, j, params, grid)
            c_ji = compute_pair(j, i, params, grid)
            if abs(c_ij - c_ji) > 1e-10:
                raise AssertionError(f"Symmetry violated: c_{i}{j}={c_ij}, c_{j}{i}={c_ji}")
```

### 2. Degeneration tests
```python
def test_one_piece_degeneration():
    """Setting P₂=P₃=0 should recover one-piece behavior."""
    params = load_przz_params()
    params["P2"] = zero_polynomial()
    params["P3"] = zero_polynomial()
    
    c, breakdown = compute_c(K=3, params=params, grid=make_grid(80))
    
    # Only c_{11} should be nonzero
    assert abs(breakdown["c_22"]) < 1e-14
    assert abs(breakdown["c_33"]) < 1e-14
    assert abs(breakdown["c_12"]) < 1e-14
    assert abs(breakdown["c_13"]) < 1e-14
    assert abs(breakdown["c_23"]) < 1e-14

def test_Q_equals_one():
    """Setting Q≡1 should give different (lower) κ."""
    params_original = load_przz_params()
    params_Q1 = params_original.copy()
    params_Q1["Q"] = constant_polynomial(1.0)
    
    c_original, _ = compute_c(K=3, params=params_original, grid=make_grid(80))
    c_Q1, _ = compute_c(K=3, params=params_Q1, grid=make_grid(80))
    
    kappa_original = compute_kappa(c_original, params_original["R"])
    kappa_Q1 = compute_kappa(c_Q1, params_Q1["R"])
    
    # Q optimization should improve κ
    assert kappa_original > kappa_Q1
```

---

## High-Precision Audit

```python
def high_precision_spot_check():
    """
    Verify float64 computation against mpmath.
    Run on 2-3 random parameter sets.
    """
    import mpmath
    mpmath.mp.dps = 50  # 50 decimal places
    
    for seed in [42, 123, 456]:
        params = random_params(seed)
        
        # Compute in float64
        c_float64, _ = compute_c(K=3, params=params, grid=make_grid(60))
        
        # Compute in mpmath (slower, but exact)
        c_mpmath = compute_c_mpmath(K=3, params=params, n=60)
        
        # Should agree to ~1e-10
        relative_error = abs(float(c_mpmath) - c_float64) / c_float64
        assert relative_error < 1e-8, f"Precision issue: {relative_error}"
```

---

## Decision Check: A-factor/I5-type Terms

```python
def audit_arithmetic_corrections():
    """
    Implement A-correction terms and verify they are lower-order.
    """
    params = load_przz_params()
    grid = make_grid(80)
    
    # Compute without A-corrections
    c_main, _ = compute_c(K=3, params=params, grid=grid, include_A_corrections=False)
    
    # Compute with A-corrections
    c_full, _ = compute_c(K=3, params=params, grid=grid, include_A_corrections=True)
    
    # A-corrections should be small
    correction = abs(c_full - c_main) / c_main
    print(f"A-correction relative magnitude: {correction:.2e}")
    
    # Typically should be O(1/log²N) ~ O(0.01) or smaller
    assert correction < 0.05, "A-corrections unexpectedly large"
```

---

## Success Criteria Summary

| Checkpoint | Requirement |
|------------|-------------|
| Unit tests pass | All polynomial, quadrature, series tests green |
| (1,1) stable | Variation < 1e-5 across n=60/80/100 |
| Symmetry holds | c_{ij} = c_{ji} to 1e-10 |
| Full regression | κ = 0.417293962 ± 1e-6 |
| Quadrature converged | n=80 vs n=100 differ by < 1e-6 |
| Breakdown logged | All 6 c_{ij} values recorded |
