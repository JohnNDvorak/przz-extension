# Session Transition Document

**Date:** 2025-12-14
**Project:** PRZZ Extension (κ Optimization)
**Location:** `/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/`

---

## Final State

| Item | Status |
|------|--------|
| GitHub repo | https://github.com/JohnNDvorak/przz-extension (private) |
| Phase 0, Step 1 | ✅ Complete (32 tests) |
| Phase 0, Step 2 | ✅ Complete (46 tests) |
| Phase 0, Step 3 | ✅ Complete (64 tests) |
| Composition module | ✅ Complete (20 tests) |
| Phase 0, Step 4 | ✅ Complete (47 tests) |
| Phase 0, Step 5 (I₁-I₄) | ✅ Complete (60 tests) |
| Phase 0, Step 6 (evaluate.py) | ✅ Complete (22 tests) |
| Phase 0, Step 7 (PRZZ c₁₁) | ✅ Complete (16 tests) |
| Total tests | **307 passing** |
| Local commits | Pending (push via GitHub Desktop) |

---

## Next Session Instructions

```bash
cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
```

1. Read `CLAUDE.md` first
2. Read this file for handoff context
3. Push pending commits if not already done
4. Proceed with **Phase 0, Step 8: Remaining pairs** (add (1,2), (1,3), (2,2), (2,3), (3,3))
5. Then validate full c and κ against PRZZ targets

---

## What Was Accomplished

### Phase 0, Step 1: polynomials.py — COMPLETE
- `src/polynomials.py` (486 lines)
- `tests/test_polynomials.py` (32 tests)
- Constraint enforcement by parameterization
- PRZZ polynomial reproduction verified

### Phase 0, Step 2: quadrature.py — COMPLETE
- `src/quadrature.py` (107 lines)
- `tests/test_quadrature.py` (46 tests)

**Key Features:**
- `gauss_legendre_01(n)`: 1D GL on [0,1], cached with read-only arrays
- `tensor_grid_2d(n)`: 2D grid with `indexing="ij"`
- `tensor_grid_3d(n)`: 3D grid for future use
- Reference-based 2D convergence test (analytically reduced 1D integral)

### Phase 0, Step 3: series.py — COMPLETE
- `src/series.py` (215 lines core, ~330 with docstrings)
- `tests/test_series.py` (57 tests)

**Key Features:**
- `TruncatedSeries` class with bitset representation
- Nilpotent multiplication: `x * x = 0` (overlapping bits vanish)
- Full arithmetic: `__add__`, `__sub__`, `__mul__`, `__neg__`
- `exp()` for computing `exp(c + N)` where N is nilpotent
- `extract(deriv_vars)` to get derivative coefficients
- Supports numpy array coefficients for vectorized grid evaluation

**Defensive Invariants (added after review):**
- `var_names` uniqueness validation in `__init__`
- `_assert_same_vars()` check on all binary operations between series
- `extract()` returns 0 for duplicate variables (x² = 0)
- 12 tests for pitfall guards

### Composition Module — COMPLETE
- `src/composition.py` (~205 lines)
- `tests/test_composition.py` (20 tests)

**Key Features:**
- `compose_polynomial_on_affine()`: Core Taylor expansion P(u + δ)
- `compose_exp_on_affine()`: Convenience wrapper for exp(R*(u + δ))
- `PolyLike` protocol for polynomial-like objects
- `_get_poly_degree()`: Robust degree detection with fallback
- Validates lin keys are in var_names (both functions)
- Active variable truncation: uses `len(lin)` not `len(var_names)`
- Full docstrings with mathematical derivations

**Key validations:**
- Coefficient of x₁x₂...xₖ = (∏ aᵢ) · P⁽ᵏ⁾(u) verified
- Array broadcasting for grid-shaped u validated
- Full pipeline test: compose → scale → exp → extract
- PRZZ polynomial wrappers (P1, Pell, Q) tested directly
- Minimal poly-like protocol fallback tested

**ChatGPT Feedback Integration (2025-12-13):**
- Fixed: Active variable count for truncation (efficiency)
- Fixed: Lin-key validation in compose_exp_on_affine (fail-fast)
- Verified: series.py defensive invariants correct
- Verified: TERM_DSL.md α/β structure is mathematically correct

### Phase 0, Step 4: term_dsl.py — COMPLETE
- `src/term_dsl.py` (~340 lines)
- `tests/test_term_dsl.py` (47 tests)

**Key Classes:**
- `SeriesContext`: Canonical var_names ownership, creates zero/scalar/variable series
- `AffineExpr`: Represents a₀(u,t) + Σᵢ aᵢ(u,t)·varᵢ with dtype-safe lifting
- `PolyFactor`: Polynomial factor P(argument)^power with evaluate() method
- `ExpFactor`: Exponential factor exp(scale·argument) via compose_exp_on_affine()
- `Term`: Complete integral term specification with strict invariant validation

**Key Features:**
- `_eval_gridfunc()`: Centralized grid evaluation with shape validation
- Zero-pruning: to_u0_lin() and to_series() skip identically-zero coefficients
- Shape invariants: wrong-shape callables raise ValueError immediately
- Dtype preservation: complex scalars preserve dtype via np.result_type
- Composition delegation: all Taylor work through composition.py
- Var validation: to_u0_lin/to_series raise if var not in ctx.var_names
- d=1 enforcement: __post_init__ validates all derivative orders ≤ 1
- vars/deriv_orders consistency: keys must match exactly
- `deriv_tuple()` helper: returns vars for series.extract()
- GridFunc accepts float, complex, np.ndarray, or callable

**Test Coverage (47 tests):**
- SeriesContext creation and series factories (6 tests)
- AffineExpr evaluation with shape/dtype invariants (7 tests)
- AffineExpr to series conversion (3 tests)
- Factor structure (3 tests)
- Term structure and helpers (3 tests)
- Factor evaluation via composition (3 tests)
- Missing mask integration (2 tests)
- d=1 enforcement and Term.create_context (5 tests)
- Failure modes: wrong shape, bad var (4 tests)
- Zero-pruning behavior (4 tests)
- vars/deriv_orders consistency (3 tests)
- deriv_tuple() helper (4 tests)

### Phase 0, Step 5 (I₁ only): terms_k3_d1.py — COMPLETE
- `src/terms_k3_d1.py` (~220 lines)
- `tests/test_terms_k3_d1.py` (26 tests)

**Implemented:**
- `make_P_argument(var)`: Build P argument var + u
- `make_Q_arg_alpha(theta, x_vars, y_vars)`: Build Arg_α = t + θt·x1 + θ(t-1)·y1
- `make_Q_arg_beta(theta, x_vars, y_vars)`: Build Arg_β = t + θ(t-1)·x1 + θt·y1
- `make_algebraic_prefactor_11(theta)`: Build (θS+1)/θ = 1/θ + x1 + y1
- `make_poly_prefactor_11()`: Build (1-u)²
- `make_I1_11(theta, R)`: Build complete I₁ term

**I₁ Term Structure:**
- vars = ("x1", "y1"), deriv_orders = {"x1": 1, "y1": 1}
- numeric_prefactor = 1.0 (pair sign (-1)^{1+1} = +1)
- algebraic_prefactor = 1/θ + x1 + y1
- poly_prefactors = [(1-u)²]
- poly_factors = [P₁(x1+u), P₁(y1+u), Q(Arg_α), Q(Arg_β)]
- exp_factors = [exp(R·Arg_α), exp(R·Arg_β)]

**CRITICAL: α ≠ β**
- Arg_α = t + θt·x1 + θ(t-1)·y1 (x-coefficient is θt)
- Arg_β = t + θ(t-1)·x1 + θt·y1 (x-coefficient is θ(t-1))
- These are NOT equal! Cannot collapse Q(Arg_α)·Q(Arg_β) into Q(...)²

**Test Coverage (60 tests):**
- I₁: Term structure (8), α/β distinct (4), prefactors (6), symbolic sanity (3), shape/quadrature (4)
- I₂: Term structure (9), evaluation (2)
- I₃: Term structure (7), α/β distinct (2), evaluation (1)
- I₄: Term structure (7), α/β distinct (2), evaluation (1)
- make_all_terms_11 (3)

**I₂ Structure (Decoupled term):**
- vars = (), deriv_orders = {}
- numeric_prefactor = 1/θ
- poly_factors = [P₁(u), P₁(u), Q(t)²]  (Q with power=2)
- exp_factors = [exp(2R·t)]  (note: 2R, not R!)

**I₃ Structure (Single x derivative):**
- vars = ("x1",), deriv_orders = {"x1": 1}
- numeric_prefactor = -1.0
- poly_prefactors = [(1-u)]  (single power)
- poly_factors = [P₁(x1+u), P₁(u), Q(Arg_α|_{y=0}), Q(Arg_β|_{y=0})]
- exp_factors = [exp(R·Arg_α|_{y=0}), exp(R·Arg_β|_{y=0})]

**I₄ Structure (Single y derivative):**
- vars = ("y1",), deriv_orders = {"y1": 1}
- numeric_prefactor = -1.0
- poly_prefactors = [(1-u)]
- poly_factors = [P₁(u), P₁(y1+u), Q(Arg_α|_{x=0}), Q(Arg_β|_{x=0})]
- exp_factors = [exp(R·Arg_α|_{x=0}), exp(R·Arg_β|_{x=0})]

---

## Repository Structure

```
przz-extension/
├── CLAUDE.md                  # Project rules — READ FIRST
├── SESSION_TRANSITION.md      # This file
├── TECHNICAL_ANALYSIS.md      # Mathematical derivations
├── ARCHITECTURE.md            # Implementation design
├── VALIDATION.md              # Test plan and checkpoints
├── TERM_DSL.md                # Term data model spec (for Step 4)
├── OPTIMIZATION.md            # κ improvement strategy
├── src/
│   ├── __init__.py
│   ├── polynomials.py         # ✅ Complete
│   ├── quadrature.py          # ✅ Complete
│   ├── series.py              # ✅ Complete
│   ├── composition.py         # ✅ Complete
│   ├── term_dsl.py            # ✅ Complete
│   ├── terms_k3_d1.py         # ✅ Complete (I₁-I₄ for (1,1))
│   └── evaluate.py            # ✅ Complete
├── tests/
│   ├── __init__.py
│   ├── test_polynomials.py       # 32 tests ✅
│   ├── test_quadrature.py        # 46 tests ✅
│   ├── test_series.py            # 64 tests ✅
│   ├── test_composition.py       # 20 tests ✅
│   ├── test_term_dsl.py          # 47 tests ✅
│   ├── test_terms_k3_d1.py       # 60 tests ✅
│   ├── test_evaluate.py          # 22 tests ✅
│   └── test_przz_integration.py  # 16 tests ✅
├── data/
│   └── przz_parameters.json   # PRZZ values (schema v1)
└── archive/
    └── PLAN_PHASE0_STEP1.md
```

---

## Phase 0, Step 7: PRZZ c₁₁ Integration — COMPLETE

**c₁₁ computed with PRZZ polynomials:**

```
c₁₁ = 0.594576269278
```

**Per-term breakdown (n=100):**
```
I1_11: +0.413474102447
I2_11: +0.384629463444
I3_11: -0.101763648306
I4_11: -0.101763648306
Total:  0.594576269278
```

**Quadrature convergence:** Essentially perfect by n=40 (errors ~1e-16, machine precision).

**Key validations passed:**
- I₂ separable: 2D = product of 1D ✅
- I₁ P=1,Q=1: matches analytic formula ✅
- All terms evaluate with PRZZ polynomials ✅
- Per-term breakdown sums correctly ✅
- Convergence verified: n=40,60,80,100 all give same result ✅

---

## Next Step: Phase 0, Step 8 — Remaining Pairs

**c₁₁ is validated.** Now implement remaining pairs.

### Task: Add (1,2), (1,3), (2,2), (2,3), (3,3) terms

For each pair (ℓ₁, ℓ₂):
1. Build terms in terms_k3_d1.py (using appropriate variable counts)
2. Add tests
3. Compute c_{ℓ₁ℓ₂} with PRZZ polynomials

### Full c computation

```
c = c₁₁ + c₂₂ + c₃₃ + 2*c₁₂ + 2*c₁₃ + 2*c₂₃
```

### Targets

```
c_target = 2.13745440613217263636
κ_target = 0.417293962
```

After all pairs implemented:
- Sum pair contributions (with factors of 2 for off-diagonal)
- Compute κ = 1 - log(c)/R
- Validate against PRZZ targets

---

## Commands Reference

```bash
# Run all tests
python3 -m pytest tests/ -v

# Quick summary
python3 -m pytest tests/ -q

# Single module
python3 -m pytest tests/test_quadrature.py -v
```

---

## Key Regression Targets

```
R = 1.3036
θ = 4/7 ≈ 0.5714285714285714
κ_target = 0.417293962
c_target = 2.13745440613217263636
```

Relationship: `κ = 1 - log(c)/R`

---

## Dependencies

- Python 3.9.6
- numpy 2.0.2
- pytest 8.4.2

---

## Git State

```
Branch: main
Commits ahead of origin: 2

17a839f Phase 0, Step 2 complete: quadrature module
9e35995 Update session transition with GitHub remote info
```

**To push:** Use GitHub Desktop or configure SSH authentication.
