# TERM_DSL.md — Term Data Model for PRZZ Pair Computations

## Purpose

PRZZ reduces the mollified moment to a finite sum of structurally similar terms:
- Each term is a derivative-at-0 of a multi-dimensional integral
- Derivatives correspond to residue extraction variables (z_i, w_j)
- Integrands are products of factors: P, Q, exp, algebraic prefactors

A Term DSL makes:
- Correctness review possible
- Incremental additions (pair-by-pair)
- Scaling to K=4 feasible

---

## Core Concepts

### Formal Variables

For pair (ℓ₁, ℓ₂):
- x_vars: x₁, ..., x_{ℓ₁}
- y_vars: y₁, ..., y_{ℓ₂}

All derivative orders are 1 per variable (d=1 residue structure).

**Canonical naming convention:**
- Always use numbered variables: `x1`, `y1` even when ℓ=1
- This ensures consistent handling across all pair types
- `Term.vars` is the single source of truth for variable ordering

**Examples:**
```
(1,1): vars = ("x1", "y1")
(1,2): vars = ("x1", "y1", "y2")
(2,2): vars = ("x1", "x2", "y1", "y2")
(3,3): vars = ("x1", "x2", "x3", "y1", "y2", "y3")
```

### Affine Expressions

Q arguments and exponential arguments have the form:
```
Arg = a₀(u,t) + Σᵢ aᵢ(u,t) · varᵢ
```

where the coefficients aᵢ depend on the integration variables but NOT on the formal variables.

**Example for (1,1):**
```
Arg_α = θ·t·(x+y) - θ·y + t
      = t + (θ·t)·x + (θ·t - θ)·y
      
So: a₀ = t, a_x = θ·t, a_y = θ·t - θ
```

### Polynomial Factors

P factors evaluate P_ℓ at arguments like "sum_of_x_vars + u":
```
P₁(x + u)           for (1,1)
P₂(y₁ + y₂ + u)     for (1,2)
```

Q factors evaluate Q at the Arg_α or Arg_β expressions.

### Exponential Factors

Exponentials are exp(R · Arg) where Arg is an affine expression.

---

## Python Data Model

### AffineExpr

```python
from dataclasses import dataclass
from typing import Dict, Callable, Union
import numpy as np

# Type alias: either a constant or a function of grid points
GridFunc = Union[float, Callable[[np.ndarray, np.ndarray], np.ndarray]]

@dataclass(frozen=True)
class AffineExpr:
    """
    Represents: a₀(u,t) + Σᵢ aᵢ(u,t) · varᵢ
    
    Attributes:
        a0: Base term - constant or callable(U, T) -> array
        var_coeffs: Dict mapping variable name to its coefficient
                    Each coefficient is constant or callable(U, T) -> array
    """
    a0: GridFunc
    var_coeffs: Dict[str, GridFunc]
    
    def evaluate_a0(self, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Evaluate the base term on grid. Preserves dtype for complex."""
        if callable(self.a0):
            return self.a0(U, T)
        # Dtype-safe lifting: preserves complex if scalar is complex
        return np.full(U.shape, self.a0, dtype=np.result_type(U, self.a0))

    def evaluate_coeff(self, var: str, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Evaluate coefficient of var on grid. Preserves dtype for complex."""
        c = self.var_coeffs.get(var, 0.0)
        if callable(c):
            return c(U, T)
        # Dtype-safe lifting: preserves complex if scalar is complex
        return np.full(U.shape, c, dtype=np.result_type(U, c))
```

### PolyFactor and ExpFactor

```python
@dataclass(frozen=True)
class PolyFactor:
    """A polynomial factor: P_name(argument)^power or Q(argument)^power."""
    poly_name: str      # "P1", "P2", "P3", "Q"
    argument: AffineExpr
    power: int = 1      # For Q² cases, use power=2 instead of repeating factor

@dataclass(frozen=True)
class ExpFactor:
    """An exponential factor: exp(scale · argument)."""
    scale: float        # For exp(2R·arg), use scale=2*R
    argument: AffineExpr
```

### Term

```python
from typing import Tuple, List, Optional

@dataclass
class Term:
    """Complete specification of one integral term."""
    
    # Identification
    name: str                           # e.g., "I1_11", "I2_12"
    pair: Tuple[int, int]               # (ℓ₁, ℓ₂)
    przz_reference: Optional[str]       # e.g., "Section 6.2.1, I₁"
    
    # Formal variables
    vars: Tuple[str, ...]               # ("x1", "y1") or ("x1", "y1", "y2") etc.
    deriv_orders: Dict[str, int]        # Typically all 1's for d=1
    
    # Integration
    domain: str                         # "[0,1]^2", "[0,1]^3", "special"
    
    # Prefactors
    numeric_prefactor: float            # Constant multiplier
    algebraic_prefactor: Optional[AffineExpr]  # e.g., (θS + 1)/θ where S = sum(vars)
    poly_prefactors: List[GridFunc]     # e.g., [(1-u)^3]
    
    # Factors to expand and multiply
    poly_factors: List[PolyFactor]      # P and Q factors
    exp_factors: List[ExpFactor]        # Exponential factors
    
    def total_vars(self) -> int:
        """Number of formal variables."""
        return len(self.vars)
    
    def target_mask(self) -> int:
        """Bitmask for the target derivative (all 1's for d=1)."""
        return (1 << len(self.vars)) - 1
```

---

## Constructing Terms

### Helper Functions

```python
def make_sum_expr(var_names: Tuple[str, ...]) -> AffineExpr:
    """Create expression for sum of variables: Σ vars."""
    return AffineExpr(
        a0=0.0,
        var_coeffs={v: 1.0 for v in var_names}
    )

def make_P_argument(x_vars: Tuple[str, ...]) -> AffineExpr:
    """P argument: sum(x_vars) + u."""
    return AffineExpr(
        a0=lambda U, T: U,  # The 'u' part
        var_coeffs={v: 1.0 for v in x_vars}
    )

def make_Q_arg_alpha(theta: float, all_vars: Tuple[str, ...], 
                     y_vars: Tuple[str, ...]) -> AffineExpr:
    """
    Q argument for alpha side: θ·t·S - θ·Y + t
    where S = sum(all_vars), Y = sum(y_vars)
    """
    var_coeffs = {}
    for v in all_vars:
        if v in y_vars:
            # Coefficient: θ·t - θ
            var_coeffs[v] = lambda U, T, th=theta: th * T - th
        else:
            # Coefficient: θ·t
            var_coeffs[v] = lambda U, T, th=theta: th * T
    
    return AffineExpr(
        a0=lambda U, T: T,  # The '+t' part
        var_coeffs=var_coeffs
    )

def make_Q_arg_beta(theta: float, all_vars: Tuple[str, ...],
                    x_vars: Tuple[str, ...]) -> AffineExpr:
    """
    Q argument for beta side: θ·t·S - θ·X + t
    where S = sum(all_vars), X = sum(x_vars)
    """
    var_coeffs = {}
    for v in all_vars:
        if v in x_vars:
            # Coefficient: θ·t - θ
            var_coeffs[v] = lambda U, T, th=theta: th * T - th
        else:
            # Coefficient: θ·t
            var_coeffs[v] = lambda U, T, th=theta: th * T
    
    return AffineExpr(
        a0=lambda U, T: T,
        var_coeffs=var_coeffs
    )
```

### Example: Building (1,1) Terms

```python
def make_terms_11(theta: float, R: float) -> List[Term]:
    """Create all terms for pair (1,1)."""
    
    x_vars = ("x1",)
    y_vars = ("y1",)
    all_vars = ("x1", "y1")

    # Common expressions
    P1_arg = make_P_argument(x_vars)  # x1 + u
    P2_arg = make_P_argument(y_vars)  # y1 + u
    Q_arg_alpha = make_Q_arg_alpha(theta, all_vars, y_vars)
    Q_arg_beta = make_Q_arg_beta(theta, all_vars, x_vars)
    
    terms = []
    
    # I₁: Main coupled term (∂²/∂x∂y)
    terms.append(Term(
        name="I1_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₁",
        vars=all_vars,
        deriv_orders={"x1": 1, "y1": 1},
        domain="[0,1]^2",
        numeric_prefactor=1.0,
        algebraic_prefactor=AffineExpr(
            # (θ(x1+y1) + 1)/θ = 1/θ + (x1+y1)
            a0=lambda U, T, th=theta: 1.0/th,
            var_coeffs={"x1": 1.0, "y1": 1.0}
        ),
        poly_prefactors=[lambda U, T: (1 - U)**2],
        poly_factors=[
            PolyFactor("P1", P1_arg),
            PolyFactor("P2", P2_arg),
            PolyFactor("Q", Q_arg_alpha),
            PolyFactor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    ))
    
    # I₂: Decoupled term (no derivatives)
    terms.append(Term(
        name="I2_11",
        pair=(1, 1),
        przz_reference="Section 6.2.1, I₂",
        vars=(),  # No formal variables
        deriv_orders={},
        domain="[0,1]^2",
        numeric_prefactor=1.0 / theta,
        algebraic_prefactor=None,
        poly_prefactors=[],
        poly_factors=[
            PolyFactor("P1", AffineExpr(a0=lambda U, T: U, var_coeffs={})),
            PolyFactor("P2", AffineExpr(a0=lambda U, T: U, var_coeffs={})),
            PolyFactor("Q", AffineExpr(a0=lambda U, T: T, var_coeffs={})),
            PolyFactor("Q", AffineExpr(a0=lambda U, T: T, var_coeffs={})),
        ],
        exp_factors=[
            ExpFactor(2*R, AffineExpr(a0=lambda U, T: T, var_coeffs={})),
        ]
    ))
    
    # I₃ and I₄: Single derivative terms (with negative sign)
    # ... similar construction ...
    
    return terms
```

### Example: Building (1,2) Terms

```python
def make_terms_12(theta: float, R: float) -> List[Term]:
    """Create all terms for pair (1,2)."""
    
    x_vars = ("x1",)
    y_vars = ("y1", "y2")
    all_vars = ("x1", "y1", "y2")

    # P arguments
    P1_arg = make_P_argument(x_vars)      # x1 + u
    P2_arg = make_P_argument(y_vars)      # y1 + y2 + u
    
    # Q arguments
    Q_arg_alpha = make_Q_arg_alpha(theta, all_vars, y_vars)
    Q_arg_beta = make_Q_arg_beta(theta, all_vars, x_vars)
    
    terms = []
    
    # Main term (∂³/∂x∂y1∂y2)
    terms.append(Term(
        name="main_12",
        pair=(1, 2),
        przz_reference="Generalization of Section 6.2.1",
        vars=all_vars,
        deriv_orders={"x1": 1, "y1": 1, "y2": 1},
        domain="[0,1]^2",
        numeric_prefactor=1.0,
        algebraic_prefactor=AffineExpr(
            # (θ·S + 1)/θ where S = x1 + y1 + y2
            a0=lambda U, T, th=theta: 1.0/th,
            var_coeffs={"x1": 1.0, "y1": 1.0, "y2": 1.0}
        ),
        poly_prefactors=[lambda U, T: (1 - U)**3],  # (1-u)^{ℓ₁+ℓ₂}
        poly_factors=[
            PolyFactor("P1", P1_arg),
            PolyFactor("P2", P2_arg),
            PolyFactor("Q", Q_arg_alpha),
            PolyFactor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    ))
    
    # Additional terms (I2, I3, I4 analogs)...
    
    return terms
```

---

## Generator Pattern

For systematic generation of all pairs:

```python
def generate_pair_terms(ell1: int, ell2: int, theta: float, R: float) -> List[Term]:
    """
    Generate all terms for pair (ℓ₁, ℓ₂).
    
    This follows the structural pattern:
    - x_vars has ℓ₁ variables
    - y_vars has ℓ₂ variables
    - Poly prefactor is (1-u)^{ℓ₁+ℓ₂}
    - P arguments sum their respective vars + u
    - Q arguments follow the Arg_α/Arg_β pattern
    """
    # Variable naming - always use numbered form for consistency
    x_vars = tuple(f"x{i}" for i in range(1, ell1 + 1))
    y_vars = tuple(f"y{j}" for j in range(1, ell2 + 1))
    
    all_vars = x_vars + y_vars
    
    # Build arguments
    P_left_arg = make_P_argument(x_vars)
    P_right_arg = make_P_argument(y_vars)
    Q_arg_alpha = make_Q_arg_alpha(theta, all_vars, y_vars)
    Q_arg_beta = make_Q_arg_beta(theta, all_vars, x_vars)
    
    # Main term
    main_term = Term(
        name=f"main_{ell1}{ell2}",
        pair=(ell1, ell2),
        vars=all_vars,
        deriv_orders={v: 1 for v in all_vars},
        domain="[0,1]^2",
        numeric_prefactor=1.0,
        algebraic_prefactor=AffineExpr(
            a0=lambda U, T, th=theta: 1.0/th,
            var_coeffs={v: 1.0 for v in all_vars}
        ),
        poly_prefactors=[lambda U, T, m=ell1+ell2: (1 - U)**m],
        poly_factors=[
            PolyFactor(f"P{ell1}", P_left_arg),
            PolyFactor(f"P{ell2}", P_right_arg),
            PolyFactor("Q", Q_arg_alpha),
            PolyFactor("Q", Q_arg_beta),
        ],
        exp_factors=[
            ExpFactor(R, Q_arg_alpha),
            ExpFactor(R, Q_arg_beta),
        ]
    )
    
    # TODO: Add additional terms (I2, I3, I4 analogs)
    # These require careful extraction from PRZZ
    
    return [main_term]
```

---

## Series Engine Contract

Given a Term and a quadrature grid:

1. **Expand each factor** into a truncated multi-var series
   - P factors: use polynomial derivative expansion
   - Q factors: same treatment
   - Exp factors: use exp(a₀)·Π(1 + aᵢ·vᵢ) identity
   - Prefactors: evaluate on grid, multiply into constant term

2. **Multiply all factor series** using bitset multiplication

3. **Extract coefficient** for target multi-index (the full mask for d=1)

4. **Integrate** the coefficient array over the domain

5. **Return** the term's contribution to c

---

## Writing Terms: Practical Guidance

1. **Keep terms small and explicit**: PRZZ I1/I2/I3/I4/I5 should each be separate Term records

2. **Include PRZZ reference**: Every Term should document which equation it comes from

3. **Log per-term contributions**: In Phase 0, print each term's value to debug sign/prefactor errors

4. **Test incrementally**: Validate each term before adding the next

5. **Watch for sign conventions**: The pair sign (−1)^{ℓ₁+ℓ₂} and term signs must be tracked carefully
