# OPTIMIZATION.md — Strategy for Improving κ Beyond PRZZ

## Goal

We want to improve κ = 1 - log(c)/R by reducing c and choosing R appropriately.

**Important:** c depends on R, so R must be optimized jointly (outer-loop).

**Key insight:** Achieving κ = 0.42 from the current 0.41729 requires only a **0.35% reduction** in c.

---

## Parameter Count (PRZZ Baseline)

Using the published K=3, d=1 setup:

| Component | Free Parameters | Notes |
|-----------|-----------------|-------|
| P₁ | 4 | Constrained form: x + x(1-x)·P̃ |
| P₂ | 3 | Constrained form: x·P̃ |
| P₃ | 3 | Constrained form: x·P̃ |
| Q | 3 | In PRZZ's (1-2x) basis with Q(0)=1 |
| R | 1 | Shift parameter |
| **Total** | **14** | Small enough for global optimization |

This is a tractable optimization problem.

---

## Structure of the Objective

### c as a function of parameters

For fixed (R, θ, polynomial degrees):
```
c(R, P₁, P₂, P₃, Q) = Σᵢ cᵢᵢ(R, Pᵢ, Q) + 2·Σᵢ<ⱼ cᵢⱼ(R, Pᵢ, Pⱼ, Q)
```

### Blockwise convexity

**Key observation:** The mean value integral is quadratic in polynomial coefficients when other polynomials are fixed.

- **Fix Q**: c is quadratic in P coefficients → convex optimization
- **Fix P's**: c is quadratic in Q coefficients → convex optimization
- **Joint (P, Q)**: Nonconvex due to bilinear composition in |Vψ|²

### Consequence

Use **alternating minimization**: solve convex subproblems iteratively.

---

## Recommended Optimization Structure

### Algorithm: Alternating Minimization with R Scan

```python
def optimize_kappa(initial_params, R_grid, max_iters=100, tol=1e-8):
    """
    Find parameters that maximize κ.
    
    Returns: best_params, best_kappa
    """
    best_kappa = -np.inf
    best_params = None
    
    for R in R_grid:
        # Initialize from PRZZ or random
        params = initialize_params(initial_params, R)
        
        # Alternating minimization
        for iteration in range(max_iters):
            c_old = compute_c(params)
            
            # Step 1: Optimize P's with Q fixed
            params = optimize_P_given_Q(params)
            
            # Step 2: Optimize Q with P's fixed
            params = optimize_Q_given_P(params)
            
            c_new = compute_c(params)
            
            # Check convergence
            if abs(c_new - c_old) / c_old < tol:
                break
        
        kappa = compute_kappa(c_new, R)
        if kappa > best_kappa:
            best_kappa = kappa
            best_params = params.copy()
    
    return best_params, best_kappa
```

### Subproblem: Optimize P's Given Q

Since c is quadratic in P coefficients:
```
c = p^T A p + b^T p + const
```
where p is the vector of all P coefficients.

This is a **quadratic program** (QP). If A is positive definite, the minimum is:
```
p* = -A⁻¹ b / 2
```

**Implementation:**
1. Precompute the Hessian matrix A and gradient vector b
2. These depend on integrals that can be cached
3. Solve via linear algebra (np.linalg.solve)

### Subproblem: Optimize Q Given P's

Same structure: c is quadratic in Q coefficients.

### Handling Constraints

- P₁(0)=0, P₁(1)=1: Use parameterization P₁(x) = x + x(1-x)·P̃(x)
- P_ℓ(0)=0: Use parameterization P_ℓ(x) = x·P̃_ℓ(x)
- Q(0)=1: Use parameterization that enforces this

These parameterizations make the constraints **implicit**, so the optimization is unconstrained.

---

## Random Restarts

To mitigate local minima in the joint (P, Q) space:

```python
def optimize_with_restarts(n_restarts=10):
    results = []
    
    # Always include PRZZ as a starting point
    results.append(optimize_kappa(przz_params))
    
    # Random restarts
    for i in range(n_restarts - 1):
        perturbed = perturb_params(przz_params, scale=0.1)
        results.append(optimize_kappa(perturbed))
    
    return max(results, key=lambda x: x[1])  # Best κ
```

---

## Sensitivity Analysis

### Sensitivity to c

```
∂κ/∂c = -1/(R·c)

At PRZZ point (R=1.3036, c=2.1375):
∂κ/∂c ≈ -0.359
```

This means: **1% reduction in c ≈ 0.0077 increase in κ**

### Sensitivity to R (with c varying)

At an R-optimum for fixed polynomial family:
```
∂κ/∂R = 0  implies  ∂c/∂R = (c/R)·log(c)
```

At the PRZZ point: ∂c/∂R ≈ 1.245

### Using Sensitivities

For gradient-based refinement:
```python
def compute_gradient(params, epsilon=1e-6):
    """Finite-difference gradient of κ w.r.t. parameters."""
    grad = {}
    kappa_0 = compute_kappa_from_params(params)
    
    for key in params:
        params_plus = params.copy()
        params_plus[key] += epsilon
        kappa_plus = compute_kappa_from_params(params_plus)
        grad[key] = (kappa_plus - kappa_0) / epsilon
    
    return grad
```

---

## Phase Roadmap

### Phase 1: K=3, d=1 Polishing

**Goal:** Extract maximum κ from current framework.

**Steps:**
1. Increase polynomial degrees modestly:
   - P₂, P₃: try degree 4-5 instead of 3
   - Q: try degree 7-9 instead of 5
2. R scan: evaluate R ∈ [1.2, 1.4] with step 0.01
3. Multiple restarts: 10-20 random perturbations
4. Record best κ and corresponding parameters

**Expected gain:** O(10⁻³) improvement in κ

### Phase 2: K=4, d=1

**Goal:** Demonstrate whether K=4 can push κ > 0.42.

**Steps:**
1. Add P₄ polynomial (start with degree 2-3)
2. Implement 4 new pairs: (1,4), (2,4), (3,4), (4,4)
3. Re-optimize all parameters including P₄
4. Compare best K=4 κ against best K=3 κ

**Complexity increase:**
- Pairs: 6 → 10
- Max variables: 6 → 8
- Parameters: ~14 → ~17-18

**Expected gain:** Plausibly O(10⁻³) to O(10⁻²), potentially crossing 0.42

### Phase 3: Hybrid/Different-Shaped Pieces

**Motivation:** Correlation limits (Čech–Matomäki 2025) suggest adding "different-shaped" 
pieces could outperform simply increasing K.

**Idea:** Instead of P₄ being another Feng-style piece, use a Bui-Conrey-Young style piece.

**Status:** Requires additional mathematical extraction. Consider only if K=4 saturates.

### Phase 4: d=2

**Goal:** Higher derivative depth for potentially larger gains.

**Challenges:**
- Bell diagram combinatorics explode
- Implementation complexity increases significantly
- Many more terms per pair

**Status:** Consider only if K=4 and hybrid approaches saturate.

---

## Quadrature and "False Improvements"

### The Danger

Small κ improvements (0.1-0.5%) are within the range of **quadrature artifacts** if not controlled.

### The Solution

Every candidate improvement must pass the **quadrature convergence gate**:

```python
def validate_improvement(candidate_params, baseline_kappa):
    """Verify improvement survives quadrature refinement."""
    
    results = {}
    for n in [60, 80, 100]:
        grid = make_grid(n)
        c, _ = compute_c(K=3, params=candidate_params, grid=grid)
        results[n] = compute_kappa(c, candidate_params["R"])
    
    # Spread check
    spread = max(results.values()) - min(results.values())
    if spread > 2e-6:
        print(f"WARNING: Quadrature unstable, spread = {spread:.2e}")
        return False
    
    # Improvement check
    improvement = results[100] - baseline_kappa
    if improvement < spread:
        print(f"WARNING: Improvement {improvement:.2e} is within noise {spread:.2e}")
        return False
    
    return True
```

---

## Outputs to Record

For any "best result" claim, record:

```python
result = {
    # Configuration
    "R": 1.3036,
    "theta": 4/7,
    "K": 3,
    "d": 1,
    
    # Polynomials (full coefficient vectors)
    "P1_coeffs": [...],
    "P2_coeffs": [...],
    "P3_coeffs": [...],
    "Q_coeffs": [...],
    "polynomial_basis": "monomial",  # or "constrained"
    
    # Quadrature settings
    "quadrature_n": 100,
    "quadrature_type": "gauss-legendre",
    
    # Results
    "c": 2.1374544...,
    "kappa": 0.417293962...,
    
    # Convergence evidence
    "kappa_n60": ...,
    "kappa_n80": ...,
    "kappa_n100": ...,
    
    # Breakdown
    "c_breakdown": {
        "c_11": ...,
        "c_22": ...,
        ...
    },
    
    # Metadata
    "timestamp": "...",
    "optimization_method": "alternating_minimization",
    "n_restarts": 10,
}
```

---

## Theoretical Limits

### What We Know

- **No unconditional hard ceiling** like "Levinson methods can't exceed X%"
- **θ is the super-knob**: θ > 4/7 would enable larger gains than K↑ or d↑
- **Farmer θ=∞ philosophy**: Arbitrarily long mollifiers would push κ → 1

### Practical Limits

- **Correlation saturation**: Adding similar pieces has diminishing returns
- **θ = 4/7 barrier**: Current exponential sum technology limit
- **Computational limits**: d=3 or K>6 becomes very expensive

### Our Target

κ > 0.42 is realistic with K=4 because:
- Required c improvement: 0.35%
- K=4 adds 4 new "directions" (pairs)
- Each direction can contribute to c reduction

---

## Implementation Checklist

Before starting optimization (Phase 1):

- [ ] Phase 0 complete: κ = 0.417293962 reproduced
- [ ] Gradient computation verified (finite-difference test)
- [ ] Alternating minimization converges on PRZZ params
- [ ] R scan infrastructure working
- [ ] Quadrature convergence gate implemented
- [ ] Result logging infrastructure ready

Before claiming an improvement:

- [ ] Passes quadrature convergence gate
- [ ] Multiple restarts give consistent result
- [ ] R has been scanned (not just single point)
- [ ] Polynomial degrees have been varied
- [ ] Full result record saved
