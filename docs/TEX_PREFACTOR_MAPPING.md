# TeX Prefactor Mapping (Run 17A0)

## Summary

**Correct Prefactor**: `exp(2R)` (NOT `exp(2R/θ)`)

## Derivation

From TeX line 1502:
```
I₁(α,β) = I_{1,1}(α,β) + T^{-α-β}·I_{1,1}(-β,-α) + O(T/L)
```

At the evaluation point α = β = -R/L where **L = log T**:
```
-α - β = 2R/L = 2R/log T
T^{-α-β} = T^{2R/log T} = exp(log T × 2R/log T) = exp(2R)
```

## Key Values

| Benchmark | R | exp(2R) | exp(2R/θ) | tex_mirror m1 |
|-----------|------|---------|-----------|---------------|
| κ | 1.3036 | 14.11 | 95.83 | 6.22 |
| κ* | 1.1167 | 9.37 | 49.82 | 6.14 |

## Why Run 14 Was Wrong

Run 14 used `exp(2R/θ)` ≈ 95.83 based on misinterpreting L as log N = θ log T.

The correct interpretation is L = log T, giving `exp(2R)` ≈ 14.11.

This is a 6.8x difference.

## Why tex_mirror Uses Different Values

tex_mirror uses m ≈ 6-8, which is neither exp(2R) ≈ 14 nor exp(2R/θ) ≈ 96.

This is because tex_mirror's shape×amplitude factorization captures a
TRANSFORMED quantity. The I1 terms have derivative structure (d²/dxdy)
that modifies how the mirror contribution enters.

The amplitude formula `A = exp(R) + K-1 + ε` is not the raw prefactor,
but rather a calibrated surrogate that works after shape factorization.

## Implications for Subsequent Agents

- **Agent 17A (I1 Combined)**: Use `exp(2R)` as the mirror prefactor
- **Agent 17B (S34 Mirror)**: Use `exp(2R)` as the mirror prefactor
- The gap between `exp(2R)` and tex_mirror's m values represents the
  derivative structure modification

## Python Reference

```python
def tex_mirror_prefactor(R: float) -> float:
    '''Correct TeX mirror prefactor at α=β=-R/L.'''
    return np.exp(2 * R)  # NOT exp(2 * R / theta)
```
