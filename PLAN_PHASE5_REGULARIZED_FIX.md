# Plan: Phase 5 - Make Regularized Path TeX-Identical

## Executive Summary

The Phase 4.4 u-regularized implementation is mathematically promising but has **three implementation bugs** that explain the I1/L → 0.522 mismatch (vs expected ~1.923):

1. **Finite differences for mixed partial** (violates CLAUDE.md Rule A)
2. **Prefactor double-count** (L×(1+θ(x+y)) AND (1/θ+x+y) both present)
3. **Phantom t-integral** (t_nodes loop where t is never used)

The fix is to make the regularized path compute the **exact same object** as `operator_post_identity.py`, using nilpotent series extraction.

---

## Root Cause Analysis (From GPT + TeX Review)

### TeX Lines 1508-1533: The Bracket Identity

The PRZZ bracket rewrite (TeX 1508-1511):
```
B = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
  = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t(α+β)} dt
```

The log factor is `log(N^{x+y}T) = L(1 + θ(x+y))`.

**But**: The original I₁ has a `1/log N = 1/(θL)` prefactor. So:
```
I₁ = (1/log N) × ... × log(N^{x+y}T) × ...
   = (1/θL) × L(1+θ(x+y)) × ...
   = (1+θ(x+y))/θ × ...
   = (1/θ + x + y) × ...
```

**The L cancels!** The final TeX formula (line 1529-1533) has only `(1/θ + x + y)` as the algebraic prefactor.

### Bug 1: Prefactor Double-Count

Current `combined_identity_regularized.py` has:
- Line 214: `prefactor = L * (1 + theta * (x + y))` inside QQB_at_xy
- Lines 409-411: `alg_prefactor = 1/θ + x + y` as a TruncatedSeries

Both are being multiplied, but TeX says only ONE should exist (the already-cancelled form).

### Bug 2: Finite Differences

Lines 453-481 use 4-point stencil with `eps=1e-6` to compute d²/dxdy of QQB. This:
- Violates CLAUDE.md Rule A ("No finite differences for derivatives at 0")
- Produces "stable but wrong by O(1) factor" answers at large L
- Is unnecessary - the kernel is a pure exponential with affine eigenvalues

### Bug 3: Phantom t-Integral

Lines 382 loop over `t_nodes`, but `t` is never used in the regularized kernel computation. The u-regularization variable `u_reg` is the combined-identity t from TeX. There should NOT be two independent integration variables.

---

## Implementation Plan

### Phase 5.1: Refactor to Nilpotent Series Extraction

**File**: `src/combined_identity_regularized.py`

**Change**: Replace finite-difference d²/dxdy with TruncatedSeries algebra.

The key insight: For the regularized kernel E(α,β;x,y,u):
```
E = exp(-Ls(1-u) - θL(βx + αy - us(x+y)))
```

The eigenvalues A_α(u,x,y) and A_β(u,x,y) are **affine in x,y**:
```
A_α = (1-u) + θ((1-u)y - ux)
A_β = (1-u) + θ((1-u)x - uy)
```

So we can:
1. Build A_α, A_β as affine forms
2. Compose Q(A_α), Q(A_β) using `compose_polynomial_on_affine`
3. Build exp factor as series using `compose_exp_on_affine`
4. Multiply all series
5. Extract ("x","y") coefficient directly - NO FINITE DIFFERENCES

**Implementation Steps**:

```python
def compute_QQE_series_at_u(
    Q_poly,
    u: float,
    theta: float,
    R: float,
    L: float,
    var_names: Tuple[str, ...] = ("x", "y")
) -> TruncatedSeries:
    """
    Build Q(A_α)Q(A_β)E as a TruncatedSeries at fixed u.

    Returns series with coefficients {1, x, y, xy} extractable directly.
    """
    from src.composition import compose_polynomial_on_affine, compose_exp_on_affine

    alpha = beta = -R / L
    s = alpha + beta  # = -2R/L

    # Eigenvalue A_α = (1-u) + θ((1-u)y - ux) = (1-u) + θ(1-u)y - θux
    u0_alpha = 1 - u
    lin_alpha = {"x": -theta * u, "y": theta * (1 - u)}

    # Eigenvalue A_β = (1-u) + θ((1-u)x - uy)
    u0_beta = 1 - u
    lin_beta = {"x": theta * (1 - u), "y": -theta * u}

    # Compose Q on each
    Q_alpha = compose_polynomial_on_affine(Q_poly, u0_alpha, lin_alpha, var_names)
    Q_beta = compose_polynomial_on_affine(Q_poly, u0_beta, lin_beta, var_names)

    # Exp factor: E(α,β,x,y,u) at α=β=-R/L
    # Exponent = -Ls(1-u) - θL(βx + αy - us(x+y))
    #          = 2R(1-u) + θR(x + y - u(-2R/L)(x+y)L/R)  ... [simplify at α=β=-R/L]
    # Actually easier: evaluate exp_u0 and exp_lin from kernel structure

    exp_u0 = -L * s * (1 - u)  # = 2R(1-u)
    exp_lin_x = -theta * L * (beta - u * s)  # β = -R/L, s = -2R/L
    exp_lin_y = -theta * L * (alpha - u * s)

    exp_series = compose_exp_on_affine(1.0, exp_u0,
                                        {"x": exp_lin_x, "y": exp_lin_y},
                                        var_names)

    return Q_alpha * Q_beta * exp_series
```

### Phase 5.2: Fix Prefactor Normalization

**Change**: Use TeX-final normalization (no L in prefactor).

The TeX I₁ formula after cancellation is:
```
I₁ = d²/dxdy [(1/θ + x + y) × ∫∫ (1-u)² P(x+u)P(y+u) × QQE du dt]_{x=y=0}
```

The `L(1+θ(x+y))` should NOT appear. Instead:
- Integrate QQE over u (the regularization/combined-identity variable)
- Multiply by profiles P₁(x+u)P₂(y+u)
- Multiply by algebraic prefactor (1/θ + x + y)
- Extract xy coefficient

**Delete**: The `prefactor = L * (1 + theta * (x + y))` in QQB_at_xy.

### Phase 5.3: Remove Phantom t-Integral

The current code has TWO integration variables:
- `t_nodes` from the outer DSL structure
- `u_reg_nodes` for the regularization

But the u-regularization IS the TeX t-integral (just reparameterized). Looking at TeX 1517:
```
∫₀¹ Q(θt(x+y) - θy + t) Q(θt(x+y) - θx + t) × exp(...) dt
```

This t IS the regularization parameter. The current code should either:
1. Rename `u_reg` → `t` and delete the `t_nodes` loop, OR
2. Wire t into the regularization if there's a genuine 2D structure

From TeX analysis: There is only ONE combined-identity parameter. **Delete the unused t_nodes loop**.

### Phase 5.4: Add Gate Test

**New File**: `tests/test_regularized_matches_post_identity.py`

This gate test verifies that the regularized path computes the SAME I₁ as the post-identity operator path.

```python
@pytest.mark.parametrize("ell1,ell2", [
    (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)
])
@pytest.mark.parametrize("L", [20, 50, 200])
def test_regularized_equals_post_identity(polys_kappa, ell1, ell2, L):
    """Gate: regularized I₁ must match post-identity I₁ to 1e-10."""
    I1_reg = compute_I1_combined_regularized_at_L(
        theta=4/7, R=1.3036, L=L, n=30,
        polynomials=polys_kappa, ell1=ell1, ell2=ell2
    )

    I1_post = compute_I1_operator_post_identity_pair(
        theta=4/7, R=1.3036, ell1=ell1, ell2=ell2, n=30,
        polynomials=polys_kappa
    )

    rel_error = abs(I1_reg.I1_combined - I1_post.I1_value) / (abs(I1_post.I1_value) + 1e-100)

    assert rel_error < 1e-10, \
        f"Gate failed for ({ell1},{ell2}) at L={L}: rel_error={rel_error:.2e}"
```

Also test that the regularized value is **L-invariant** (not scaling with L):

```python
def test_regularized_L_invariance(polys_kappa):
    """Regularized I₁ should NOT scale with L."""
    L_values = [20, 50, 100, 200]
    results = []

    for L in L_values:
        I1 = compute_I1_combined_regularized_at_L(
            theta=4/7, R=1.3036, L=L, n=30,
            polynomials=polys_kappa
        )
        results.append(I1.I1_combined)

    # Check variance is small (should converge to constant, not scale with L)
    mean_val = sum(results) / len(results)
    max_dev = max(abs(r - mean_val) / (abs(mean_val) + 1e-100) for r in results)

    assert max_dev < 0.01, \
        f"L-dependence detected: max deviation = {max_dev:.2%}"
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/combined_identity_regularized.py` | **Major Refactor** | Replace FD with TruncatedSeries, fix prefactor, delete phantom t |
| `tests/test_combined_identity_regularized.py` | Update | Adjust tests for new implementation |
| `tests/test_regularized_matches_post_identity.py` | **New** | Gate test comparing regularized to post-identity |

---

## Acceptance Criteria

1. **No finite differences** in `combined_identity_regularized.py`
2. **Prefactor = (1/θ + x + y)** only (no L factor)
3. **Single integration variable** (u or t, not both)
4. **Gate passes**: regularized I₁ = post-identity I₁ to 1e-10 relative
5. **L-invariance**: regularized I₁ does NOT scale with L
6. **Both benchmarks**: passes for κ (R=1.3036) and κ* (R=1.1167)

---

## Expected Outcome

After this fix:
- I₁_regularized should match I₁_post_identity for all pairs
- The 0.522 vs 1.923 mismatch disappears (they should agree)
- The regularized path becomes a **second derivation** of the same TeX I₁
- This provides redundancy for K>3 safety

---

## Phase 6 Preview (After Gate Passes)

Once regularized = post-identity, proceed to:

**Phase 6: Derive m₁ from operator structure**

The mirror term contains T^{-(α+β)}. Under D_α:
```
D_α(T^{-(α+β)}F) = T^{-(α+β)} × (1 + D_α)F
```

So for polynomial Q:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

This is the structural reason the mirror isn't just "evaluate at -R" - it's "evaluate with shifted operators."

Deliverables:
1. Derivation note: `docs/TEX_MIRROR_OPERATOR_SHIFT.md`
2. Code implementing mirror as exact T^{-s} weight + operator shift
3. Decomposition gate: I₁_combined = I₁_direct + I₁_mirror_exact

This eliminates empirical m₁ entirely.
