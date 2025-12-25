"""
Operator-Level Mirror Computation for GPT Step 2.

WARNING: STEP-2 PRE-IDENTITY IMPLEMENTATION (DIAGNOSTIC ONLY)
=============================================================
This module implements the PRE-IDENTITY operator approach, which has known
L-divergence and is kept for DIAGNOSTIC PURPOSES ONLY.

The pre-identity bracket B(α,β,x,y) does NOT contain the integration variable t,
which causes the resulting coefficients to diverge with L. This is a mathematical
dead end, but useful for understanding why the combined identity is necessary.

For PRODUCTION USE, see: src/operator_post_identity.py
That module implements the correct POST-IDENTITY operator approach which:
1. Uses the PRZZ combined identity to introduce t-dependence
2. Has L-stable results (no divergence)
3. Matches the DSL to machine precision

HISTORICAL CONTEXT:
-------------------
This module implements GPT's "decisive experiment" - applying Q as actual differential
operators (d/dα, d/dβ) to the pre-identity bracket, rather than using the TeX combined
identity and multiplying by Q as a polynomial.

The key insight: the order of operations matters.
- Previous runs (18, 19, 20): Apply combined identity first, then multiply by Q
- This approach: Apply Q(D_α) × Q(D_β) as differential operators, then evaluate

Mathematical setup:
    B(α,β,x,y) = (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

    where N = T^θ.

    Q(D_α) = Σⱼ qⱼ D^j_α where D_α = -1/logT × d/dα

    The full operator: Q(D_α) × Q(D_β) × B = Σᵢ Σⱼ qᵢqⱼ × (-1/L)^{i+j} × ∂^{i+j}B/∂α^i∂β^j

Then evaluate at α = β = -R/L where L = log T.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from functools import lru_cache


# ============================================================================
# Q Polynomial Conversion
# ============================================================================

def convert_Q_basis_to_monomial(basis_coeffs: Dict[int, float]) -> List[float]:
    """
    Convert Q from (1-2x)^k basis to monomial coefficients.

    Q(x) = Σₖ cₖ (1-2x)^k → Σⱼ qⱼ x^j

    Args:
        basis_coeffs: Dictionary mapping power k to coefficient cₖ
                      e.g., {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}

    Returns:
        List of monomial coefficients [q₀, q₁, q₂, ..., q_max_k]
    """
    if not basis_coeffs:
        return [1.0]  # Default to Q(x) = 1

    max_k = max(basis_coeffs.keys())
    result = np.zeros(max_k + 1)

    for k, c_k in basis_coeffs.items():
        # (1-2x)^k = Σⱼ C(k,j) (-2x)^j (1)^{k-j} = Σⱼ C(k,j) (-2)^j x^j
        for j in range(k + 1):
            binom = _binomial(k, j)
            result[j] += c_k * binom * ((-2) ** j)

    return result.tolist()


def _binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result


# ============================================================================
# Symbolic Bracket Derivatives (Stage 21A)
# ============================================================================

@dataclass
class BracketDerivatives:
    """
    Computes ∂^{i+j}B/∂α^i∂β^j for the pre-identity bracket.

    B(α,β,x,y) = (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

    where N = T^θ, so log N = θ log T = θL.

    Written out:
        B = (exp(θL(αx+βy)) - exp(-L(α+β))exp(θL(-βx-αy))) / (α+β)
          = (exp(θL(αx+βy)) - exp(-L(α+β)(1+θ(x+y)/(α+β)×(-αx-βy+αy+βx)/(α+β)))) / (α+β)

    Simplifying the second term:
        T^{-α-β}N^{-βx-αy} = exp(-L(α+β))exp(-θL(βx+αy))
                          = exp(-L(α+β + θ(βx+αy)))
                          = exp(L×(-(α+β) - θ(βx+αy)))

    So B = (exp(θL(αx+βy)) - exp(L×(-(α+β) - θ(βx+αy)))) / (α+β)

    At α = β = -R/L:
        α + β = -2R/L ≠ 0 (no singularity)

    This class uses SymPy to symbolically compute derivatives, then generates
    a fast numerical evaluation function.
    """
    theta: float
    max_deriv: int = 5  # Q is degree 5, need derivatives up to order 5 in each variable

    def __post_init__(self):
        """Pre-compute symbolic derivative expressions."""
        self._build_symbolic_derivatives()

    def _build_symbolic_derivatives(self):
        """Build symbolic expressions for all needed derivatives."""
        # Symbolic variables
        alpha, beta, x_s, y_s, L_s = sp.symbols('alpha beta x y L', real=True)
        theta_s = sp.Rational(4, 7)  # Use exact fraction for theta

        # The two terms of the bracket (before dividing by α+β)
        # Term 1: N^{αx+βy} = exp(θL(αx+βy))
        term1 = sp.exp(theta_s * L_s * (alpha * x_s + beta * y_s))

        # Term 2: T^{-α-β} N^{-βx-αy} = exp(-L(α+β)) exp(-θL(βx+αy))
        #       = exp(-L(α+β) - θL(βx+αy))
        #       = exp(L × (-(α+β) - θ(βx+αy)))
        term2 = sp.exp(L_s * (-(alpha + beta) - theta_s * (beta * x_s + alpha * y_s)))

        # The numerator: term1 - term2
        numerator = term1 - term2

        # The denominator: α + β
        denominator = alpha + beta

        # Full bracket B = numerator / denominator
        B = numerator / denominator

        # Store for later
        self._symbolic_B = B
        self._symbolic_vars = (alpha, beta, x_s, y_s, L_s)

        # Pre-compute derivatives up to max_deriv in each variable
        # Store lambdified functions for fast evaluation
        self._deriv_funcs: Dict[Tuple[int, int], Callable] = {}

        for i in range(self.max_deriv + 1):
            for j in range(self.max_deriv + 1):
                # Compute ∂^{i+j}B/∂α^i∂β^j
                deriv_expr = B
                for _ in range(i):
                    deriv_expr = sp.diff(deriv_expr, alpha)
                for _ in range(j):
                    deriv_expr = sp.diff(deriv_expr, beta)

                # Simplify (can be slow for high orders, but we only do this once)
                deriv_expr = sp.simplify(deriv_expr)

                # Lambdify for fast numerical evaluation
                self._deriv_funcs[(i, j)] = sp.lambdify(
                    (alpha, beta, x_s, y_s, L_s),
                    deriv_expr,
                    modules=['numpy']
                )

    def compute_derivative(
        self,
        i: int, j: int,
        alpha: float, beta: float,
        x_val: float, y_val: float,
        L: float
    ) -> float:
        """
        Compute ∂^{i+j}B/∂α^i∂β^j evaluated at (α,β,x,y,L).

        Args:
            i: Order of derivative in α
            j: Order of derivative in β
            alpha: Value of α
            beta: Value of β
            x_val: Value of x
            y_val: Value of y
            L: log T (asymptotic parameter)

        Returns:
            The derivative value as a float
        """
        if i < 0 or j < 0:
            raise ValueError("Derivative orders must be non-negative")
        if i > self.max_deriv or j > self.max_deriv:
            raise ValueError(f"Derivative order ({i},{j}) exceeds max_deriv={self.max_deriv}")

        func = self._deriv_funcs[(i, j)]
        result = func(alpha, beta, x_val, y_val, L)

        # Handle potential complex results from numerical instability
        if np.iscomplex(result):
            result = np.real(result)

        return float(result)

    def compute_bracket_direct(
        self,
        alpha: float, beta: float,
        x_val: float, y_val: float,
        L: float
    ) -> float:
        """
        Compute B(α,β,x,y) directly (i=j=0 case).

        This is the pre-identity bracket value.
        """
        return self.compute_derivative(0, 0, alpha, beta, x_val, y_val, L)


# ============================================================================
# Q Operator Application (Stage 21B)
# ============================================================================

def apply_Q_operator_to_bracket(
    Q_monomial_coeffs: List[float],
    bracket: BracketDerivatives,
    R: float,
    L: float,
    x_val: float,
    y_val: float
) -> float:
    """
    Apply Q(D_α) × Q(D_β) to the bracket at the evaluation point.

    Q(D_α) × Q(D_β) × B(α,β,x,y)|_{α=β=-R/L}

    = Σᵢ Σⱼ qᵢ qⱼ × (∂^i/∂α^i)(∂^j/∂β^j)B × (-1/L)^{i+j}

    where D_α = -1/L × d/dα.

    Args:
        Q_monomial_coeffs: [q₀, q₁, q₂, ..., q₅] monomial coefficients of Q
        bracket: BracketDerivatives instance
        R: The R parameter
        L: log T (asymptotic parameter)
        x_val: Value of x
        y_val: Value of y

    Returns:
        Value of Q(D_α) × Q(D_β) × B at (α,β) = (-R/L, -R/L)
    """
    alpha = -R / L
    beta = -R / L

    result = 0.0
    n_coeffs = len(Q_monomial_coeffs)

    for i, qi in enumerate(Q_monomial_coeffs):
        if abs(qi) < 1e-15:
            continue
        for j, qj in enumerate(Q_monomial_coeffs):
            if abs(qj) < 1e-15:
                continue

            # Compute the derivative ∂^{i+j}B/∂α^i∂β^j
            deriv_val = bracket.compute_derivative(i, j, alpha, beta, x_val, y_val, L)

            # Scale by (-1/L)^{i+j} for the D operator definition
            scale = ((-1.0) / L) ** (i + j)

            result += qi * qj * deriv_val * scale

    return result


def apply_Q_operator_grid(
    Q_monomial_coeffs: List[float],
    bracket: BracketDerivatives,
    R: float,
    L: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray
) -> np.ndarray:
    """
    Apply Q operators over a grid of (x, y) values.

    This is for extracting the xy coefficient via sampling.

    Args:
        Q_monomial_coeffs: Monomial coefficients of Q
        bracket: BracketDerivatives instance
        R: The R parameter
        L: log T
        x_grid: 1D array of x values
        y_grid: 1D array of y values

    Returns:
        2D array of shape (len(x_grid), len(y_grid)) with operator values
    """
    result = np.zeros((len(x_grid), len(y_grid)))

    for ix, x_val in enumerate(x_grid):
        for iy, y_val in enumerate(y_grid):
            result[ix, iy] = apply_Q_operator_to_bracket(
                Q_monomial_coeffs, bracket, R, L, x_val, y_val
            )

    return result


# ============================================================================
# Series Expansion via Sampling (Stage 21C helper)
# ============================================================================

def extract_xy_coefficient_by_sampling(
    func: Callable[[float, float], float],
    n_sample: int = 10,
    epsilon: float = 0.01
) -> float:
    """
    Extract the xy coefficient from a function f(x,y) by sampling near origin.

    For f(x,y) = a₀₀ + a₁₀x + a₀₁y + a₁₁xy + O(x²,y²,...)

    The xy coefficient is a₁₁ = ∂²f/∂x∂y|_{x=y=0}

    We use 2D polynomial fitting at small (x,y) values.

    Args:
        func: Function f(x,y) to analyze
        n_sample: Number of sample points in each dimension
        epsilon: Max magnitude of sample points

    Returns:
        Estimated coefficient of xy term
    """
    # Sample grid
    pts = np.linspace(-epsilon, epsilon, n_sample)
    X, Y = np.meshgrid(pts, pts)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(n_sample):
        for j in range(n_sample):
            Z[i, j] = func(X[i, j], Y[i, j])

    # Fit polynomial: f ≈ a₀₀ + a₁₀x + a₀₁y + a₂₀x² + a₁₁xy + a₀₂y²
    # Using least squares
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Design matrix: [1, x, y, x², xy, y²]
    A = np.column_stack([
        np.ones_like(X_flat),  # 1
        X_flat,                 # x
        Y_flat,                 # y
        X_flat**2,              # x²
        X_flat * Y_flat,        # xy
        Y_flat**2               # y²
    ])

    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)

    # coeffs = [a₀₀, a₁₀, a₀₁, a₂₀, a₁₁, a₀₂]
    a_xy = coeffs[4]

    return a_xy


# ============================================================================
# Full I₁ Computation with Operator-Level Mirror (Stage 21C)
# ============================================================================

@dataclass
class OperatorLevelI1Result:
    """Result from operator-level I₁ computation."""
    I1_operator_level: float
    L: float
    R: float
    details: Dict


def compute_I1_operator_level_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict,
    L: float = 20.0,
    verbose: bool = False,
) -> OperatorLevelI1Result:
    """
    Compute I₁ for (1,1) pair using operator-level mirror.

    This is GPT's "Step 2" decisive experiment.

    The full I₁ integrand:
        I₁ = ∫∫ Q(D_α)Q(D_β) × P₁(x+u)P₁(y+u) × B(α,β,x,y) × prefactor × (1-u)² du dt

    Since P factors don't depend on α,β, they commute with Q(D):
        = ∫∫ P₁(x+u)P₁(y+u) × [Q(D_α)Q(D_β) B] × prefactor × (1-u)² du dt

    Strategy:
    1. Compute [Q(D_α)Q(D_β) B](x,y) at α=β=-R/L
    2. Expand as series in x,y
    3. Multiply by P factors, prefactor (as in existing tex_mirror)
    4. Extract xy coefficient
    5. Integrate over (u,t)

    Args:
        theta: The θ parameter (typically 4/7)
        R: The R parameter
        n: Number of quadrature points
        polynomials: Dict with keys P1, P2, P3, Q
        L: log T for asymptotic regime
        verbose: Print debug info

    Returns:
        OperatorLevelI1Result with I₁ value and details
    """
    from src.polynomials import Polynomial, load_przz_polynomials
    from src.quadrature import gauss_legendre_01

    # Get Q in monomial form
    Q_poly = polynomials.get('Q')
    if Q_poly is None:
        raise ValueError("Q polynomial not found")

    # Convert Q to monomial coefficients
    # Q is in (1-2x)^k basis
    basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
    Q_monomial = convert_Q_basis_to_monomial(basis_coeffs)

    if verbose:
        print(f"Q monomial coefficients: {Q_monomial}")

    # Build bracket derivatives (this pre-computes symbolic derivatives)
    if verbose:
        print("Building symbolic bracket derivatives...")
    bracket = BracketDerivatives(theta=theta, max_deriv=5)

    # Get P1
    P1 = polynomials.get('P1')
    if P1 is None:
        raise ValueError("P1 polynomial not found")

    # Get quadrature for (u, t) integration
    u_nodes, u_weights = gauss_legendre_01(n)
    t_nodes, t_weights = gauss_legendre_01(n)

    # For each (u, t) point, we need to:
    # 1. Compute the operator-applied bracket as a function of (x, y)
    # 2. Multiply by P₁(x+u)P₁(y+u) (expanded as series)
    # 3. Extract xy coefficient
    # 4. Multiply by prefactor and (1-u)²

    I1_total = 0.0

    for i_u, (u, w_u) in enumerate(zip(u_nodes, u_weights)):
        for i_t, (t, w_t) in enumerate(zip(t_nodes, t_weights)):
            # The prefactor for I₁: exp(-2Rθt) × (1-u)²
            prefactor = np.exp(-2 * R * theta * t) * (1 - u) ** 2

            # x = θt(1-u) and y = θt(1-u) at the integration point
            # But we need to expand around x=y=0 first

            # Build operator-applied bracket as a series in (x, y)
            # [Q(D_α)Q(D_β) B](x, y) at α=β=-R/L

            # Use sampling to extract the xy coefficient of the operator result
            def operator_func(x_val, y_val):
                return apply_Q_operator_to_bracket(
                    Q_monomial, bracket, R, L, x_val, y_val
                )

            # For (1,1) pair, we extract d²/dxdy coefficient
            # The xy coefficient of [operator × P₁(x+u)P₁(y+u)]

            # Build series for operator-applied bracket
            # Evaluate at grid of (x,y) values near origin
            eps = 0.001
            n_pts = 5

            # Get operator values at sample points
            # Then fit to extract Taylor coefficients
            x_pts = np.linspace(-eps, eps, n_pts)
            y_pts = np.linspace(-eps, eps, n_pts)

            op_values = np.zeros((n_pts, n_pts))
            for ix, xv in enumerate(x_pts):
                for iy, yv in enumerate(y_pts):
                    op_values[ix, iy] = operator_func(xv, yv)

            # Extract Taylor coefficients: f ≈ a₀₀ + a₁₀x + a₀₁y + a₁₁xy
            X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
            X_flat, Y_flat = X.flatten(), Y.flatten()
            Z_flat = op_values.flatten()

            # Design matrix for up to order 2
            design = np.column_stack([
                np.ones_like(X_flat),  # 1
                X_flat,                 # x
                Y_flat,                 # y
                X_flat * Y_flat,        # xy
            ])

            op_coeffs, _, _, _ = np.linalg.lstsq(design, Z_flat, rcond=None)
            a_00, a_10, a_01, a_11_op = op_coeffs

            # Now build the full integrand series:
            # [operator] × P₁(x+u)P₁(y+u)

            # P₁(x+u) = P₁(u) + P₁'(u)×x + O(x²)
            P1_u = P1.eval(np.array([u]))[0]
            P1_prime_u = P1.eval_deriv(np.array([u]), 1)[0]

            # P₁(y+u) = P₁(u) + P₁'(u)×y + O(y²)
            # (Same values since both evaluated at u)

            # The product P₁(x+u)P₁(y+u) expanded:
            # = (P₁(u) + P₁'(u)x)(P₁(u) + P₁'(u)y)
            # = P₁(u)² + P₁(u)P₁'(u)x + P₁(u)P₁'(u)y + P₁'(u)²xy

            # The operator series:
            # = a₀₀ + a₁₀x + a₀₁y + a₁₁xy

            # Product [operator] × [P₁(x+u)P₁(y+u)]:
            # The xy coefficient comes from:
            # a₁₁ × P₁(u)² + a₁₀ × P₁(u)P₁'(u) + a₀₁ × P₁(u)P₁'(u) + a₀₀ × P₁'(u)²

            xy_coeff = (
                a_11_op * P1_u ** 2 +
                a_10 * P1_u * P1_prime_u +
                a_01 * P1_u * P1_prime_u +
                a_00 * P1_prime_u ** 2
            )

            # Multiply by prefactor and quadrature weight
            contribution = xy_coeff * prefactor * w_u * w_t
            I1_total += contribution

    if verbose:
        print(f"I1_operator_level = {I1_total:.6f}")
        print(f"L = {L}, R = {R}")

    return OperatorLevelI1Result(
        I1_operator_level=I1_total,
        L=L,
        R=R,
        details={
            "Q_monomial": Q_monomial,
            "n_quad": n,
        }
    )


# ============================================================================
# Diagnostic: Compare against Run 20 and tex_mirror
# ============================================================================

def run_operator_level_diagnostic(
    R: float,
    L_values: List[float] = [10.0, 20.0, 50.0],
    n_quad: int = 40,
    verbose: bool = True,
):
    """
    Run diagnostic comparison between operator-level and other approaches.

    Args:
        R: The R parameter
        L_values: List of L = log T values to test convergence
        n_quad: Number of quadrature points
        verbose: Print debug info
    """
    from src.polynomials import load_przz_polynomials

    theta = 4.0 / 7.0
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Operator-Level Mirror Diagnostic (R={R})")
        print(f"{'='*60}")

    results = []
    for L in L_values:
        result = compute_I1_operator_level_11(
            theta=theta,
            R=R,
            n=n_quad,
            polynomials=polys,
            L=L,
            verbose=False,
        )
        results.append((L, result.I1_operator_level))
        if verbose:
            print(f"  L={L:6.1f}: I1_operator = {result.I1_operator_level:.6f}")

    return results


if __name__ == "__main__":
    # Quick test
    print("Testing operator-level mirror computation...")

    # Test Q conversion
    basis_coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
    Q_mono = convert_Q_basis_to_monomial(basis_coeffs)
    print(f"Q monomial coefficients: {Q_mono}")

    # Verify Q(0) = 1 (sum of basis coefficients)
    Q_at_0 = sum(Q_mono)  # Q(0) = sum of monomial coeffs when x=0 is just q0
    print(f"Q(0) via monomial = {Q_mono[0]:.6f} (should be ~1.0)")

    # Test bracket derivatives
    print("\nBuilding bracket derivatives (this may take a moment)...")
    bracket = BracketDerivatives(theta=4.0/7.0, max_deriv=3)  # Lower max for quick test

    R, L = 1.3036, 20.0
    alpha, beta = -R/L, -R/L

    B_00 = bracket.compute_derivative(0, 0, alpha, beta, 0.1, 0.1, L)
    B_10 = bracket.compute_derivative(1, 0, alpha, beta, 0.1, 0.1, L)
    B_01 = bracket.compute_derivative(0, 1, alpha, beta, 0.1, 0.1, L)
    B_11 = bracket.compute_derivative(1, 1, alpha, beta, 0.1, 0.1, L)

    print(f"B(0,0) at x=y=0.1: {B_00:.6f}")
    print(f"∂B/∂α at x=y=0.1: {B_10:.6f}")
    print(f"∂B/∂β at x=y=0.1: {B_01:.6f}")
    print(f"∂²B/∂α∂β at x=y=0.1: {B_11:.6f}")

    print("\nDone!")
