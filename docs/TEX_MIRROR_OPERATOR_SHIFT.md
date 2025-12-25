# TeX Mirror Operator Shift Derivation

**Date:** 2025-12-22
**Status:** Phase 6 Foundation
**Purpose:** Derive the exact mirror contribution via operator shift Q→Q(1+·)

---

## 1. Operator Definitions

The PRZZ framework uses logarithmic derivative operators:

```
D_α = -1/L × ∂/∂α
D_β = -1/L × ∂/∂β
```

where L = log T is the asymptotic parameter.

These operators have the key property that for any polynomial Q:
```
Q(D_α)F = Σₖ qₖ(-1/L)ᵏ × ∂ᵏF/∂αᵏ
```

---

## 2. The Fundamental Shift Identity

**Theorem (Operator Shift):**
For the T-weight factor T^{-s} where s = α + β:
```
D_α(T^{-s}F) = T^{-s} × (1 + D_α)F
```

**Proof:**

Let T^{-s} = exp(-sL). Then:
```
D_α(T^{-s}F) = -1/L × ∂/∂α [exp(-sL) × F]
             = -1/L × [exp(-sL) × (-L) × F + exp(-sL) × ∂F/∂α]
             = -1/L × exp(-sL) × [-L × F + ∂F/∂α]
             = exp(-sL) × [F - 1/L × ∂F/∂α]
             = T^{-s} × [F + D_α F]
             = T^{-s} × (1 + D_α)F  ✓
```

**Corollary (Power Extension):**
By induction:
```
D_α^n(T^{-s}F) = T^{-s} × (1 + D_α)^n F
```

**Proof:** Apply the identity n times:
```
D_α^n(T^{-s}F) = D_α^{n-1}(T^{-s}(1+D_α)F)
               = T^{-s}(1+D_α)^{n-1}(1+D_α)F  [by induction hypothesis]
               = T^{-s}(1+D_α)^n F  ✓
```

**Corollary (Polynomial Extension):**
For any polynomial Q(z) = Σₖ qₖz^k:
```
Q(D_α)(T^{-s}F) = T^{-s} × Q(1 + D_α)F
```

**Proof:** Linear combination of the power identity.

---

## 3. Application to PRZZ Bracket

### 3.1 The Bracket Structure

The PRZZ bracket (TeX lines 1499-1501) is:
```
B(α,β;x,y) = [N^{αx+βy} - T^{-(α+β)}N^{-βx-αy}] / (α+β)
```

where N = T^θ.

This decomposes as:
```
B = (B_direct - B_mirror) / s
```

where:
- B_direct = N^{αx+βy} = exp(θL(αx+βy))
- B_mirror = T^{-s} × N^{-βx-αy} = exp(-sL) × exp(-θL(βx+αy))
- s = α + β

### 3.2 Operator Application to Direct Term

The direct term has no T^{-s} factor, so operators apply normally:
```
Q(D_α)Q(D_β)B_direct = Q(A_α^dir)Q(A_β^dir) × B_direct
```

where the eigenvalues are:
```
A_α^dir = D_α log(B_direct) = D_α[θL(αx+βy)] = -θx
A_β^dir = D_β log(B_direct) = D_β[θL(αx+βy)] = -θy
```

Wait—this gives purely linear eigenvalues. But the post-identity approach has more structure because it operates AFTER the combined identity transformation. Let me reconsider.

### 3.3 The Combined Identity Transformation (TeX 1508-1511)

The TeX uses the identity:
```
[A - B]/s = A × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-t×s} dt
```

This rewrites the bracket as an integral, where the integration variable t enters the eigenvalues.

**Post-identity eigenvalues** (from operator_post_identity.py):
```
A_α = t + θ(t-1)x + θt·y
A_β = t + θt·x + θ(t-1)·y
```

These are the eigenvalues for the POST-combined-identity exponential core.

### 3.4 Decomposing the Combined Identity

The combined identity integral can be written as:
```
∫₀¹ (N^{x+y}T)^{-ts} dt = ∫₀¹ N^{-ts(x+y)} × T^{-ts} dt
```

The T^{-ts} factor inside the integral means the operator shift applies at each t-value:
```
Q(D_α)[T^{-ts}F(t)] = T^{-ts} × Q(1+D_α)F(t)
```

But wait—here s appears in the exponent as well as in (1+D_α). This needs careful treatment.

---

## 4. Correct Decomposition Analysis

### 4.1 The Issue with Direct/Mirror Separation

The combined identity COMBINES direct and mirror into a single integral. The t-integral representation:
```
B = prefactor × ∫₀¹ E(α,β;x,y,t) dt
```

does NOT separate cleanly into "direct + mirror" contributions.

However, we can analyze the ORIGINAL bracket before the combined identity:
```
B = (N^{αx+βy} - T^{-s}N^{-βx-αy}) / s
```

### 4.2 Pre-Combined-Identity Analysis

For the mirror term T^{-s}N^{-βx-αy}:
```
Q(D_α)Q(D_β)[T^{-s}N^{-βx-αy}]
= T^{-s} × Q(1+D_α)Q(1+D_β)[N^{-βx-αy}]
```

Now N^{-βx-αy} = exp(-θL(βx+αy)), so:
```
D_α[N^{-βx-αy}] = -1/L × ∂/∂α[exp(-θL(βx+αy))]
                = -1/L × exp(-θL(βx+αy)) × (-θL×y)
                = θy × N^{-βx-αy}
```

Similarly:
```
D_β[N^{-βx-αy}] = θx × N^{-βx-αy}
```

So the mirror eigenvalues are:
```
A_α^{mirror} = θy
A_β^{mirror} = θx
```

These are SWAPPED and FLIPPED compared to direct!

### 4.3 The Shifted Polynomial

For the mirror term, we apply Q(1+D_α)Q(1+D_β), which means:
```
Q(1+D_α)[N^{-βx-αy}] = Q(1+A_α^{mirror}) × N^{-βx-αy}
                     = Q(1+θy) × N^{-βx-αy}
```

The polynomial Q(1+θy) is Q SHIFTED and then evaluated at the mirror eigenvalue.

### 4.4 The Complete Mirror Contribution

```
Q(D_α)Q(D_β)[T^{-s}N^{-βx-αy}]
= T^{-s} × Q(1+θy) × Q(1+θx) × N^{-βx-αy}
```

At the evaluation point α = β = -R/L:
- s = α + β = -2R/L
- T^{-s} = T^{2R/L} = exp(2R)
- N^{-βx-αy} = exp(-θL(-R/L)(x+y)) = exp(θR(x+y))

So the mirror term evaluates to:
```
exp(2R) × Q(1+θy) × Q(1+θx) × exp(θR(x+y))
```

---

## 5. The Shifted Polynomial Definition

Define the **shifted polynomial**:
```
Q_shifted(z) = Q(1 + z)
```

If Q(z) = Σₖ qₖ z^k in monomial form, then:
```
Q(1+z) = Σₖ qₖ (1+z)^k
       = Σₖ qₖ Σⱼ C(k,j) z^j
       = Σⱼ [Σₖ qₖ C(k,j)] z^j
```

So the shifted polynomial coefficients are:
```
q_shifted[j] = Σₖ≥ⱼ qₖ × C(k,j)
```

### 5.1 Example: Q in (1-2x)^k basis

For PRZZ Q with basis_coeffs = {0: c₀, 1: c₁, 3: c₃, 5: c₅}:
```
Q(z) = Σₖ cₖ (1-2z)^k
Q(1+z) = Σₖ cₖ (1-2(1+z))^k
       = Σₖ cₖ (-1-2z)^k
       = Σₖ cₖ (-1)^k (1+2z)^k
```

So the shifted Q is Q evaluated at 1+z, which changes sign for odd powers of (1-2x).

---

## 6. Implementation Strategy

### 6.1 Direct vs Mirror Decomposition

The key insight is that after operator application:

**Direct contribution:**
```
I₁_direct ∝ ∫∫ P(x+u)P(y+u) × Q(A_α^dir)Q(A_β^dir) × exp(direct) × prefactors du dt
```

**Mirror contribution:**
```
I₁_mirror ∝ ∫∫ P(x+u)P(y+u) × Q(1+A_α^mir)Q(1+A_β^mir) × exp(2R) × exp(θR(x+y)) × prefactors du dt
```

### 6.2 Why Scalar m1 Is Insufficient

The scalar approximation treats:
```
I₁_mirror ≈ m1 × I₁(-R)
```

But the true mirror has:
1. **Shifted polynomial:** Q(1+·) instead of Q(·)
2. **Swapped eigenvalues:** θy for α, θx for β
3. **Different exp factor:** exp(θR(x+y)) vs exp(-θR(x+y))

The m1 = exp(R) + 5 empirical value is trying to capture all these effects as a single scalar, which is too coarse.

### 6.3 The Decomposition Gate

The combined identity computes:
```
I₁_combined = (I₁_direct_contributions - I₁_mirror_contributions) / s
```

After we implement exact mirror, we verify:
```
I₁_combined ≈ I₁_direct_exact + I₁_mirror_exact
```

where the exact computations use the proper shifted polynomials and swapped eigenvalues.

---

## 7. Connection to Regularized Path

The Phase 5 u-regularized path uses the t-integral representation which ALREADY combines direct and mirror. The eigenvalues A_α(u,x,y) and A_β(u,x,y) in the regularized path are:
```
A_α = (1-u) + θ((1-u)y - ux)
A_β = (1-u) + θ((1-u)x - uy)
```

These are the POST-combined-identity eigenvalues that naturally include both contributions.

To decompose, we need to:
1. Separate the integral into direct-like and mirror-like regions, OR
2. Analytically extract the direct and mirror contributions from the combined kernel

The second approach is cleaner: recognize that the combined kernel E(u) interpolates between:
- u=1: Mirror endpoint (T^{-s} contribution dominates)
- u=0: Direct endpoint (N^{αx+βy} contribution dominates)

---

## 8. References

- PRZZ TeX lines 1499-1501: Bracket definition
- PRZZ TeX lines 1508-1511: Combined identity
- PRZZ TeX lines 1529-1533: Final I₁ formula
- `src/operator_post_identity.py`: Post-identity eigenvalues
- `src/combined_identity_regularized.py`: U-regularized path
- Phase 5 documentation: `docs/PHASE5_REGULARIZED_FIX_COMPLETE.md`

---

## 9. Summary

The mirror contribution to I₁ can be computed exactly using:

1. **Operator shift identity:** Q(D_α)(T^{-s}F) = T^{-s}Q(1+D_α)F
2. **Shifted polynomial:** Q_shifted(z) = Q(1+z)
3. **Swapped mirror eigenvalues:** A_α^{mirror} = θy, A_β^{mirror} = θx
4. **Mirror exp factor:** exp(2R) × exp(θR(x+y))

This replaces the empirical m1 = exp(R) + 5 with an exact computation that properly accounts for polynomial shift and eigenvalue swap.

The decomposition gate I₁_combined = I₁_direct + I₁_mirror verifies correctness.
