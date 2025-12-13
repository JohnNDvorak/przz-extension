# ARCHITECTURE.md — Implementation Design for PRZZ κ Engine

## Overview

We compute c (and thus κ) by evaluating a finite set of "pair terms" c_{ℓ₁ℓ₂},
each expressed as derivatives at 0 of integrals over (u,t,...) with factors:
- P-polynomials evaluated at affine expressions in the formal vars and integration vars
- Q-polynomials evaluated at affine expressions
- exponentials exp(R·affine(...))
- algebraic prefactors like (1−u)^m and (θ·sum(vars)+1)/θ

**Core challenges:**
1. Correct residue derivatives without finite differences
2. Efficient, stable quadrature
3. Modular term tables for each (ℓ₁,ℓ₂), scalable to K=4

---

## Data Flow

```
Term definition (DSL)
  → build factor series on quadrature grid
  → multiply series (truncate var exponents to 0/1)
  → extract coefficient for derivative multi-index
  → integrate over domain
  → accumulate into c_{ℓ₁ℓ₂} and final c
  → κ = 1 − log(c)/R
```

---

## Core Modules

### polynomials.py

**Responsibilities:**
- Polynomial class storing coefficients in a chosen basis
- Fast vectorized evaluation and derivatives

**Key methods:**
```python
class Polynomial:
    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at points x (vectorized)."""
        
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluate k-th derivative at points x (vectorized)."""
```

**Constraint parameterizations:**
```python
# P₁: P₁(0)=0, P₁(1)=1
# Parameterize as: P₁(x) = x + x(1−x)·P̃(x)
# where P̃ is a free polynomial

# P_ℓ (ℓ≥2): P_ℓ(0)=0
# Parameterize as: P_ℓ(x) = x·P̃_ℓ(x)

# Q: Q(0)=1
# Parameterize to enforce this automatically
```

---

### quadrature.py

**Responsibilities:**
- Gauss-Legendre nodes/weights on [0,1]
- Tensor-product grids for 2D, 3D
- Optional domain transforms

**Key functions:**
```python
def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) for n-point GL quadrature on [0,1]."""

def tensor_grid_2d(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (U, T, W) arrays for 2D quadrature on [0,1]²."""
    # U[i,j] = u_i, T[i,j] = t_j, W[i,j] = w_i * w_j
```

**Caching:**
- Nodes/weights for each n
- 2D/3D grid arrays

---

### series.py

**Core insight:** Since each formal variable has derivative order 1, we never need 
powers ≥ 2. This means v² = 0 for all variables, and monomials are products of 
**distinct** variables.

**Representation:** Monomials as bitsets
```python
# vars = (v0, v1, ..., v_{m-1})
# monomial key = integer mask in [0, 2^m)
# e.g., for vars=(x,y1,y2): x*y2 -> mask = 0b101 = 5
```

**Data structure:**
```python
class TruncatedSeries:
    """Multi-variable series with per-variable max order 1."""
    
    def __init__(self, var_names: Tuple[str, ...]):
        self.vars = var_names
        self.n_vars = len(var_names)
        # coeff[mask] = numpy array on quadrature grid
        self.coeffs: Dict[int, np.ndarray] = {}
```

**Operations:**
```python
def multiply(self, other: TruncatedSeries) -> TruncatedSeries:
    """Multiply two series. Key rule: overlapping masks vanish."""
    result = TruncatedSeries(self.vars)
    for mask_a, coeff_a in self.coeffs.items():
        for mask_b, coeff_b in other.coeffs.items():
            if (mask_a & mask_b) == 0:  # No overlap
                mask_out = mask_a | mask_b
                if mask_out not in result.coeffs:
                    result.coeffs[mask_out] = np.zeros_like(coeff_a)
                result.coeffs[mask_out] += coeff_a * coeff_b
    return result
```

**Factor expansions:**

*Exponential:*
```python
# exp(R(a0 + Σ ai·vi)) = exp(R·a0) · Π_i (1 + R·ai·vi)
# because vi² = 0
```

*Polynomial:*
```python
# P(u + Σ ai·vi) = Σ_{k=0}^m P^{(k)}(u)/k! · (Σ ai·vi)^k
# where (Σ ai·vi)^k expands via subset enumeration
```

---

### term_dsl.py

**Purpose:** Define the Term structure and AffineExpr for specifying integrands.

```python
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
import numpy as np

@dataclass(frozen=True)
class AffineExpr:
    """
    Represents: a0(u,t) + Σ ai(u,t)·var_i
    
    Coefficients depend on integration variables (arrays on grid).
    """
    # Base term: callable(U, T) -> array, or just array
    a0: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Var coefficients: var_name -> callable(U, T) -> array
    var_coeffs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]

@dataclass
class PolyFactor:
    """A polynomial factor P(arg) or Q(arg)."""
    poly_name: str      # "P1", "P2", "P3", "Q"
    argument: AffineExpr

@dataclass  
class ExpFactor:
    """An exponential factor exp(R * arg)."""
    R: float
    argument: AffineExpr

@dataclass
class Term:
    """Complete specification of one integral term."""
    name: str
    pair: Tuple[int, int]
    
    # Formal variables and their derivative orders
    vars: Tuple[str, ...]
    deriv_orders: Dict[str, int]  # Typically all 1's
    
    # Integration domain
    domain: str  # "[0,1]^2", "[0,1]^3", "special"
    
    # Numeric prefactor (constant)
    numeric_prefactor: float
    
    # Prefactor expressions depending on (u,t) and vars
    # e.g., (1-u)^3, (θ*S + 1)/θ where S = sum of vars
    prefactor_callables: List[Callable]
    
    # Factors to multiply
    poly_factors: List[PolyFactor]
    exp_factors: List[ExpFactor]
```

---

### terms_k3_d1.py

**Purpose:** Manual term tables for K=3, d=1.

**Strategy:**
- Start with (1,1) fully implemented and validated
- Add pairs incrementally: (1,2), (1,3), (2,2), (2,3), (3,3)

**Example (1,1) I₁ term:**
```python
def make_term_11_I1(theta: float, R: float) -> Term:
    """Create the I₁ term for pair (1,1)."""
    
    # Affine expressions for Q arguments
    # Arg_α = θ*t*(x+y) - θ*y + t
    def arg_alpha_a0(U, T):
        return T  # constant part
    def arg_alpha_x(U, T):
        return theta * T  # coefficient of x
    def arg_alpha_y(U, T):
        return theta * T - theta  # coefficient of y
    
    arg_alpha = AffineExpr(
        a0=arg_alpha_a0,
        var_coeffs={"x": arg_alpha_x, "y": arg_alpha_y}
    )
    
    # Similar for arg_beta...
    
    return Term(
        name="I1_11",
        pair=(1, 1),
        vars=("x", "y"),
        deriv_orders={"x": 1, "y": 1},
        domain="[0,1]^2",
        numeric_prefactor=1.0,  # T*Φ̂(0) handled separately
        prefactor_callables=[
            lambda U, T, S: (theta * S + 1) / theta,  # algebraic
            lambda U, T, S: (1 - U) ** 2,  # polynomial
        ],
        poly_factors=[
            PolyFactor("P1", AffineExpr(...)),  # P₁(x+u)
            PolyFactor("P2", AffineExpr(...)),  # P₂(y+u)
            PolyFactor("Q", arg_alpha),
            PolyFactor("Q", arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, arg_alpha),
            ExpFactor(R, arg_beta),
        ]
    )
```

---

### evaluate.py

**Purpose:** Compute c and κ from terms.

```python
def evaluate_term(term: Term, params: Dict, grid: QuadGrid) -> float:
    """
    Evaluate a single term contribution.
    
    1. Build series for each factor
    2. Multiply all series
    3. Extract target derivative coefficient
    4. Integrate over domain
    """
    pass

def compute_pair(ell1: int, ell2: int, params: Dict, grid: QuadGrid) -> float:
    """Compute c_{ℓ₁ℓ₂} by summing all terms for this pair."""
    pass

def compute_c(K: int, params: Dict, grid: QuadGrid) -> Tuple[float, Dict]:
    """
    Compute total c and return breakdown.
    
    Returns: (c_total, {"c_11": ..., "c_12": ..., ...})
    """
    pass

def compute_kappa(c: float, R: float) -> float:
    """κ = 1 - log(c)/R"""
    return 1.0 - np.log(c) / R
```

---

### optimize.py

See OPTIMIZATION.md for strategy details.

---

## Caching Strategy

### Layer 1: Always cache (grid-dependent only)
- Quadrature nodes/weights for each n
- 2D/3D grid arrays U, T and weights W

### Layer 2: Cache per (R, θ)
- Base exponential arrays: exp(R * base_arg(U, T))
- Argument coefficient arrays that depend only on θ

### Layer 3: Cache per polynomial degrees
- Power arrays: for base argument A(U,T), cache A^0, A^1, ..., A^{deg_max}
- Then P(A) = Σ c_k * A^k is a dot product

### Layer 4: Cannot cache (coefficient-dependent)
- Actual P and Q evaluations depend on coefficients being optimized
- Must recompute each optimization iteration

---

## Performance Notes

### Bitset series is fast
- For 6 vars (K=3 max): 64 monomials
- For 8 vars (K=4 max): 256 monomials
- Multiplication is O(M²) where M is number of nonzero monomials
- In practice, many monomials are zero, so much faster

### Vectorization
- Store all coeff[mask] as arrays of shape (n_u, n_t) or flattened
- Integration is a dot product with weight array

### Memory
- For n=100 quadrature: 10,000 points × 64 monomials × 8 bytes ≈ 5 MB per series
- Acceptable for single-threaded computation

---

## Numerical Robustness

### Avoiding singularities
- Gauss-Legendre nodes avoid endpoints → no 1/(1-u) issues
- All our integrands are smooth on (0,1)²

### Verifying convergence
- Always test n = 60, 80, 100
- Require relative stability better than target precision

### High-precision audits
- For any claimed improvement, spot-check with mpmath (50 digits)
- Ensures float64 isn't hiding errors

---

## Extension Path to K=4

Adding K=4 requires:
1. New pairs: (1,4), (2,4), (3,4), (4,4)
2. Maximum variables increases to 8
3. Bracket formulas for new pairs (provided in TECHNICAL_ANALYSIS.md)
4. New polynomial P₄ with constraint P₄(0)=0

**The architecture supports this without redesign** because:
- Bitset series handles up to ~20 variables easily
- Term DSL is parameterized by (ℓ₁, ℓ₂)
- evaluate.py loops over all pairs
