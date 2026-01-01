# PRZZ Exact Replication Specification

**Goal:** Computationally replicate PRZZ's κ = 0.417293962 and κ* = 0.407511457

**Status:** SPECIFICATION (not yet implemented)

---

## Source: PRZZ TeX Lines 1500-1570

PRZZ computes the main-term constant c from four integral types: I₁, I₂, I₃, I₄.

The key equation (line 287):
```
κ ≥ 1 - (1/R) log(c)
```

where c is the asymptotic value of the mollified mean square.

---

## The Four Integral Types (for pair (1,1), K=3, d=1)

### I₁ Formula (lines 1530-1532)

```
I₁ = T·Φ̂(0) × (d²/dxdy) × [(θ(x+y)+1)/θ]
     × ∫₀¹ ∫₀¹ (1-u)² P₁(x+u) P₂(y+u)
     × exp(R[θt(x+y)-θy+t]) × exp(R[θt(x+y)-θx+t])
     × Q(θt(x+y)-θy+t) × Q(θt(x+y)-θx+t)
     |_{x=y=0} du dt + O(T/L)
```

**Key insight:** The Q arguments are position-dependent:
- Q₁ = Q(θt(x+y) - θy + t)
- Q₂ = Q(θt(x+y) - θx + t)

At x=y=0: Q₁ = Q₂ = Q(t), but the derivative d²/dxdy picks up terms from Q'(t).

### I₂ Formula (lines 1548)

```
I₂ = T·Φ̂(0)/θ × ∫₀¹ ∫₀¹ Q(t)² e^{2Rt} P₁(u) P₂(u) dt du + O(T/L)
```

**Key insight:** Q(t)² is a FROZEN SCALAR (line 1544):
```
Q(-1/logT · ∂/∂α) Q(-1/logT · ∂/∂β) T^{-tα-tβ} |_{α=β=-R/L} = Q(t)² e^{2Rt}
```

### I₃ Formula (lines 1562-1563)

```
I₃ = -T·Φ̂(0) × (1+θx)/θ × (d/dx)
     × ∫₀¹ ∫₀¹ (1-u) P₁(x+u) P₂(u)
     × e^{R[t+θxt]} × e^{R[-θx+t+θxt]}
     × Q(t+θxt) × Q(-θx+t+θxt) dt du |_{x=0} + O(T/L)
```

### I₄ Formula (lines 1568-1569)

```
I₄ = -T·Φ̂(0) × (1+θy)/θ × (d/dy)
     × ∫₀¹ ∫₀¹ (1-u) P₁(u) P₂(y+u)
     × e^{R[t+θyt]} × e^{R[-θy+t+θyt]}
     × Q(t+θyt) × Q(-θy+t+θyt) dt du |_{y=0} + O(T/L)
```

---

## Normalization and Assembly

For the asymptotic as T→∞, the T factors cancel:
- Each I term is T × (something)
- The mean square is (1/T) × Σ I terms
- So c = Φ̂(0) × Σ (normalized I terms)

### Normalization per pair

For pair (ℓ₁, ℓ₂), there are factors:
- 1/(ℓ₁! × ℓ₂!) - factorial normalization
- 2 for off-diagonal pairs (ℓ₁ ≠ ℓ₂) - symmetry factor
- 1/(log N)^{ℓ₁+ℓ₂} - absorbed into polynomial definitions

### Sum over pairs

For K=3:
```
c = Φ̂(0) × Σ_{ℓ₁,ℓ₂ ∈ {1,2,3}, ℓ₁≤ℓ₂} w_{ℓ₁,ℓ₂} × (I₁ + I₂ + I₃ + I₄)_{ℓ₁,ℓ₂}
```

where w_{ℓ₁,ℓ₂} = 2/(ℓ₁!ℓ₂!) if ℓ₁<ℓ₂, else 1/(ℓ₁!ℓ₂!)

---

## Critical Differences from Our Implementation

### 1. I₁ vs I₂ Q treatment

| Integral | Q treatment | Our code |
|----------|-------------|----------|
| I₁ | Position-dependent Q(arg_α)×Q(arg_β) | ❌ Mixed with I₂ |
| I₂ | Frozen Q(t)² scalar | ❌ Mixed with I₁ |

**Problem:** Our unified evaluator mixes I₁ and I₂ together using either:
- Legacy mode: all Q is position-dependent (wrong for I₂)
- Frozen mode: all Q is frozen scalar (wrong for I₁)

**Solution:** Compute I₁ and I₂ SEPARATELY with correct Q treatment.

### 2. Derivative structure

| Integral | Derivative | Our code |
|----------|------------|----------|
| I₁ | d²/dxdy with (1-u)² P(x+u)P(y+u) | ❌ Uses same structure for I₂ |
| I₂ | NO derivative, P(u)P(u) | ❌ Mixed with I₁ |
| I₃ | d/dx with (1-u) P(x+u)P(u) | ✓ Computed separately |
| I₄ | d/dy with (1-u) P(u)P(y+u) | ✓ Computed separately |

### 3. (1-u) powers

| Integral | Power of (1-u) |
|----------|----------------|
| I₁ | (1-u)^{ℓ₁+ℓ₂} = (1-u)² for (1,1) |
| I₂ | None |
| I₃ | (1-u)^{ℓ₁} = (1-u) for (1,1) |
| I₄ | (1-u)^{ℓ₂} = (1-u) for (1,1) |

---

## Implementation Plan

### Step 1: Create separate I₁ evaluator

```python
def compute_I1_przz(u, t, theta, R, P1, P2, Q):
    """
    I₁ integrand: d²/dxdy of bracket at x=y=0.

    Uses position-dependent Q eigenvalues.
    """
    # Q arguments at (t, x, y):
    # arg_α = θt(x+y) - θy + t
    # arg_β = θt(x+y) - θx + t
    # At x=y=0: both = t
    # d/dx: changes arg_α by θt, arg_β by θt-θ
    # d/dy: changes arg_α by θt-θ, arg_β by θt

    # Build series and extract d²/dxdy coefficient
    ...
```

### Step 2: Create separate I₂ evaluator

```python
def compute_I2_przz(u, t, theta, R, P1, P2, Q):
    """
    I₂ integrand: Q(t)² e^{2Rt} P₁(u) P₂(u).

    NO derivatives, frozen Q(t)² scalar.
    """
    Q_t = Q.eval(t)
    return Q_t**2 * np.exp(2*R*t) * P1.eval(u) * P2.eval(u)
```

### Step 3: Create I₃, I₄ evaluators (similar to existing)

### Step 4: Assemble c

```python
def compute_c_przz_exact(theta, R, P1, P2, P3, Q, n_quad=80):
    """
    Compute c using PRZZ's exact method.

    Sum over pairs with correct normalization.
    """
    c_total = 0.0

    for ell1 in [1, 2, 3]:
        for ell2 in range(ell1, 4):
            P_ell1 = [P1, P2, P3][ell1-1]
            P_ell2 = [P1, P2, P3][ell2-1]

            I1 = integrate_I1(theta, R, ell1, ell2, P_ell1, P_ell2, Q, n_quad)
            I2 = integrate_I2(theta, R, P_ell1, P_ell2, Q, n_quad)
            I3 = integrate_I3(theta, R, ell1, P_ell1, P_ell2, Q, n_quad)
            I4 = integrate_I4(theta, R, ell2, P_ell1, P_ell2, Q, n_quad)

            # Normalization
            norm = 1.0 / (math.factorial(ell1) * math.factorial(ell2))
            symmetry = 2.0 if ell1 < ell2 else 1.0

            c_total += norm * symmetry * (I1 + I2 + I3 + I4)

    return c_total
```

---

## Validation Targets

| Benchmark | R | θ | c target | κ target |
|-----------|---|----|----------|----------|
| κ | 1.3036 | 4/7 | 2.137 | 0.417293962 |
| κ* | 1.1167 | 4/7 | 1.938 | 0.407511457 |

Note: c target = exp(R×(1-κ))

---

## Key Questions to Resolve

1. **Φ̂(0) value:** What is the normalization constant? (Likely = 1 for smooth weight)

2. **General pair formulas:** Lines 1530-1569 show (1,1). What changes for (ℓ₁, ℓ₂)?
   - (1-u) power changes to (1-u)^{ℓ₁+ℓ₂} for I₁
   - P polynomial indexing changes

3. **I₅ term:** PRZZ mentions I₅ = O(T/L) is negligible. Confirm we don't need it.

4. **Log factor:** The (θ(x+y)+1)/θ = 1/θ + x + y term in I₁. How does this contribute?

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/przz_exact_i1.py` | I₁ evaluator with position-dependent Q |
| `src/przz_exact_i2.py` | I₂ evaluator with frozen Q(t)² |
| `src/przz_exact_assembly.py` | Assembly of all terms |
| `scripts/test_przz_exact_replication.py` | Validation against κ = 0.417 |
