"""
Independent Bivariate Taylor Reference Engine (Phase 2.1)

This module provides a MATHEMATICALLY INDEPENDENT implementation for computing
bivariate truncated Taylor series. It is used to PROVE the normalization mapping
between the multi-variable DSL pipeline and the scalar bivariate formulation.

CRITICAL: This module must NOT import or use any code from series.py.
All coefficient formulas are computed directly from mathematical definitions.

Key Design:
- Series represented as 2D coefficient table C[p,q] for x^p * y^q
- Multiplication via 2D convolution
- Total degree truncation: p + q <= max_order
- Maximum supported order: 6 (for (3,3) pair with 6 derivatives total)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from math import factorial
from functools import lru_cache


@lru_cache(maxsize=64)
def _binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n,k) = n! / (k!(n-k)!)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


@dataclass
class BivariateSeries:
    """
    Truncated bivariate Taylor series in (x, y).

    Represents: sum_{p,q: p+q <= max_order} C[p,q] * x^p * y^q

    The coefficient C[p,q] corresponds to the term x^p y^q / (p! q!),
    i.e., C[p,q] = (∂^p_x ∂^q_y f)(0,0) / (p! q!)

    Actually, we store the actual polynomial coefficient (not divided by factorial),
    so C[p,q] is the coefficient of x^p y^q in the expansion.
    """
    coeffs: np.ndarray  # Shape (max_order+1, max_order+1), C[p,q] = coeff of x^p y^q
    max_order: int

    @classmethod
    def zero(cls, max_order: int = 6) -> 'BivariateSeries':
        """Create zero series."""
        coeffs = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
        return cls(coeffs=coeffs, max_order=max_order)

    @classmethod
    def constant(cls, c: float, max_order: int = 6) -> 'BivariateSeries':
        """Create constant series f(x,y) = c."""
        coeffs = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
        coeffs[0, 0] = c
        return cls(coeffs=coeffs, max_order=max_order)

    @classmethod
    def from_x(cls, max_order: int = 6) -> 'BivariateSeries':
        """Create series f(x,y) = x."""
        coeffs = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
        coeffs[1, 0] = 1.0
        return cls(coeffs=coeffs, max_order=max_order)

    @classmethod
    def from_y(cls, max_order: int = 6) -> 'BivariateSeries':
        """Create series f(x,y) = y."""
        coeffs = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
        coeffs[0, 1] = 1.0
        return cls(coeffs=coeffs, max_order=max_order)

    def __add__(self, other: 'BivariateSeries') -> 'BivariateSeries':
        """Add two series."""
        assert self.max_order == other.max_order
        return BivariateSeries(
            coeffs=self.coeffs + other.coeffs,
            max_order=self.max_order
        )

    def __sub__(self, other: 'BivariateSeries') -> 'BivariateSeries':
        """Subtract two series."""
        assert self.max_order == other.max_order
        return BivariateSeries(
            coeffs=self.coeffs - other.coeffs,
            max_order=self.max_order
        )

    def __mul__(self, other: 'BivariateSeries') -> 'BivariateSeries':
        """
        Multiply two series via 2D convolution with total degree truncation.

        (sum_p,q A[p,q] x^p y^q) * (sum_r,s B[r,s] x^r y^s)
        = sum_{p,q,r,s} A[p,q] B[r,s] x^{p+r} y^{q+s}

        Coefficient of x^m y^n is sum_{p+r=m, q+s=n} A[p,q] B[r,s]
        with truncation: m + n <= max_order
        """
        assert self.max_order == other.max_order
        max_ord = self.max_order
        result = np.zeros((max_ord + 1, max_ord + 1), dtype=np.float64)

        for p in range(max_ord + 1):
            for q in range(max_ord + 1 - p):  # p + q <= max_ord
                if self.coeffs[p, q] == 0:
                    continue
                for r in range(max_ord + 1 - p):
                    for s in range(max_ord + 1 - p - q - r):  # p+q+r+s <= max_ord
                        if other.coeffs[r, s] == 0:
                            continue
                        m, n = p + r, q + s
                        if m + n <= max_ord:
                            result[m, n] += self.coeffs[p, q] * other.coeffs[r, s]

        return BivariateSeries(coeffs=result, max_order=max_ord)

    def scale(self, c: float) -> 'BivariateSeries':
        """Multiply by scalar."""
        return BivariateSeries(
            coeffs=self.coeffs * c,
            max_order=self.max_order
        )

    def get_coeff(self, p: int, q: int) -> float:
        """Get coefficient of x^p y^q."""
        if p < 0 or q < 0 or p + q > self.max_order:
            return 0.0
        return self.coeffs[p, q]

    def extract_mixed_coeff(self, px: int, qy: int) -> float:
        """
        Extract coefficient of x^{px} y^{qy} term.

        This is the key extraction that maps to the DSL's multi-derivative extraction.
        """
        return self.get_coeff(px, qy)


def polynomial_taylor_coeffs(poly_coeffs: np.ndarray, u: float, max_deriv: int) -> np.ndarray:
    """
    Compute Taylor coefficients of P(u + z) around z=0.

    P(u + z) = sum_{k=0}^{max_deriv} P^{(k)}(u) / k! * z^k

    Returns array of length max_deriv+1 where result[k] = P^{(k)}(u) / k!

    Parameters:
        poly_coeffs: Coefficients [a_0, a_1, ..., a_d] for P(x) = sum a_i x^i
        u: Expansion point
        max_deriv: Maximum derivative order needed

    Returns:
        Array where taylor[k] = coeff of z^k in expansion P(u+z)
    """
    d = len(poly_coeffs) - 1  # Polynomial degree
    taylor = np.zeros(max_deriv + 1, dtype=np.float64)

    # P(u+z) = sum_{i=0}^d a_i (u+z)^i
    #        = sum_{i=0}^d a_i sum_{k=0}^i C(i,k) u^{i-k} z^k
    #        = sum_{k=0}^d z^k sum_{i=k}^d a_i C(i,k) u^{i-k}

    for k in range(min(d + 1, max_deriv + 1)):
        coeff_k = 0.0
        for i in range(k, d + 1):
            coeff_k += poly_coeffs[i] * _binomial(i, k) * (u ** (i - k))
        taylor[k] = coeff_k

    return taylor


def exp_taylor_coeffs(lam: float, max_deriv: int) -> np.ndarray:
    """
    Compute Taylor coefficients of exp(λz) around z=0.

    exp(λz) = sum_{k=0}^{max_deriv} λ^k / k! * z^k

    Returns array where result[k] = λ^k / k!
    """
    taylor = np.zeros(max_deriv + 1, dtype=np.float64)
    lam_power = 1.0
    factorial_k = 1.0
    for k in range(max_deriv + 1):
        if k > 0:
            lam_power *= lam
            factorial_k *= k
        taylor[k] = lam_power / factorial_k
    return taylor


def compose_polynomial_bivariate(
    poly_coeffs: np.ndarray,
    u: float,
    coeff_x: float,
    coeff_y: float,
    max_order: int = 6
) -> BivariateSeries:
    """
    Compute bivariate Taylor expansion of P(u + a*x + b*y).

    P(u + a*x + b*y) = sum_k P^{(k)}(u)/k! * (a*x + b*y)^k
                     = sum_k (P^{(k)}(u)/k!) * sum_{j=0}^k C(k,j) a^j b^{k-j} x^j y^{k-j}

    Coefficient of x^p y^q (where p+q=k):
        = (P^{(k)}(u)/k!) * C(k,p) * a^p * b^q
        = P^{(k)}(u) / (p! q!) * a^p * b^q

    Parameters:
        poly_coeffs: Polynomial coefficients [a_0, a_1, ..., a_d]
        u: Base point
        coeff_x: Coefficient 'a' of x in argument
        coeff_y: Coefficient 'b' of y in argument
        max_order: Maximum total degree

    Returns:
        BivariateSeries representing P(u + a*x + b*y)
    """
    # Get Taylor coefficients P^{(k)}(u) for k = 0, ..., max_order
    taylor = polynomial_taylor_coeffs(poly_coeffs, u, max_order)

    result = BivariateSeries.zero(max_order)

    for k in range(max_order + 1):
        P_k_over_k_fact = taylor[k]  # This is already P^{(k)}(u)/k! from Taylor expansion
        if abs(P_k_over_k_fact) < 1e-300:
            continue

        # (a*x + b*y)^k = sum_{j=0}^k C(k,j) a^j b^{k-j} x^j y^{k-j}
        for j in range(k + 1):
            p, q = j, k - j
            if p + q > max_order:
                continue
            # Coefficient of x^p y^q from (ax + by)^k is C(k,p) * a^p * b^q
            binom = _binomial(k, p)
            coeff = P_k_over_k_fact * binom * (coeff_x ** p) * (coeff_y ** q)
            result.coeffs[p, q] += coeff

    return result


def compose_exp_bivariate(
    lam: float,
    t: float,
    coeff_x: float,
    coeff_y: float,
    max_order: int = 6
) -> BivariateSeries:
    """
    Compute bivariate Taylor expansion of exp(λ(t + a*x + b*y)).

    exp(λ(t + ax + by)) = exp(λt) * exp(λ(ax + by))
                        = exp(λt) * sum_k (λ^k/k!) * (ax + by)^k

    Coefficient of x^p y^q (where p+q=k):
        = exp(λt) * (λ^k/k!) * C(k,p) * a^p * b^q

    Parameters:
        lam: Scale factor λ
        t: Base point
        coeff_x: Coefficient 'a' of x
        coeff_y: Coefficient 'b' of y
        max_order: Maximum total degree

    Returns:
        BivariateSeries representing exp(λ(t + ax + by))
    """
    exp_lam_t = np.exp(lam * t)

    result = BivariateSeries.zero(max_order)

    lam_k = 1.0  # λ^k
    fact_k = 1.0  # k!
    for k in range(max_order + 1):
        if k > 0:
            lam_k *= lam
            fact_k *= k

        lam_k_over_k_fact = lam_k / fact_k
        if abs(lam_k_over_k_fact) < 1e-300:
            continue

        # (ax + by)^k = sum_{j=0}^k C(k,j) a^j b^{k-j} x^j y^{k-j}
        for j in range(k + 1):
            p, q = j, k - j
            if p + q > max_order:
                continue
            binom = _binomial(k, p)
            coeff = exp_lam_t * lam_k_over_k_fact * binom * (coeff_x ** p) * (coeff_y ** q)
            result.coeffs[p, q] += coeff

    return result


def compose_Q_bivariate(
    Q_coeffs: np.ndarray,
    t: float,
    coeff_x: float,
    coeff_y: float,
    max_order: int = 6
) -> BivariateSeries:
    """
    Compute bivariate Taylor expansion of Q(t + a*x + b*y).

    This is just compose_polynomial_bivariate with Q coefficients.

    Parameters:
        Q_coeffs: Q polynomial coefficients
        t: Base point
        coeff_x: Coefficient 'a' of x
        coeff_y: Coefficient 'b' of y
        max_order: Maximum total degree

    Returns:
        BivariateSeries representing Q(t + ax + by)
    """
    return compose_polynomial_bivariate(Q_coeffs, t, coeff_x, coeff_y, max_order)


def linear_bivariate(
    const: float,
    coeff_x: float,
    coeff_y: float,
    max_order: int = 6
) -> BivariateSeries:
    """
    Create bivariate series for linear function c + a*x + b*y.

    This is for the algebraic prefactor (1/θ + Σxᵢ + Σyⱼ) reduced to (1/θ + ℓ₁x + ℓ₂y).
    """
    result = BivariateSeries.zero(max_order)
    result.coeffs[0, 0] = const
    result.coeffs[1, 0] = coeff_x
    result.coeffs[0, 1] = coeff_y
    return result


def polynomial_prefactor_bivariate(
    u: float,
    power: int,
    max_order: int = 6
) -> BivariateSeries:
    """
    Create bivariate series for (1-u)^power.

    This is constant in x,y, so it's just a scalar multiple.
    """
    return BivariateSeries.constant((1.0 - u) ** power, max_order)


class ReferenceBivariateEngine:
    """
    Reference engine for computing term contributions using bivariate series.

    This engine computes the same mathematical quantity as the DSL pipeline,
    but using an independent bivariate representation. The mapping between
    the multi-variable DSL coefficient and the bivariate coefficient should be:

        DSL coeff (x₁x₂...y₁y₂...) = Bivariate coeff (x^{ℓ₁} y^{ℓ₂}) × M_{ℓ₁,ℓ₂}

    where M_{ℓ₁,ℓ₂} is the normalization mapping factor to be determined empirically.

    Expected: M_{ℓ₁,ℓ₂} = ℓ₁! × ℓ₂! (factorial representation mapping)
    """

    def __init__(self, max_order: int = 6):
        self.max_order = max_order

    def compute_product_series(
        self,
        P_coeffs: np.ndarray,
        Q_coeffs: np.ndarray,
        u: float,
        t: float,
        theta: float,
        R: float,
        l1: int,
        l2: int,
        include_prefactor: bool = True,
        include_exp: bool = True
    ) -> BivariateSeries:
        """
        Compute the product of all factors for a term at given (u, t) point.

        For I₁-type terms with pair (ℓ₁, ℓ₂), the integrand is roughly:
            [algebraic prefactor] × P(u + ℓ₁x) × P(u + ℓ₂y) × Q(α)Q(β) × exp(...)

        Where the Q arguments and exp arguments depend on the specific term.

        For this basic version, we compute:
            prefactor × P_left(u + x) × P_right(u + y) × Q(t + ax + by) × Q(...) × exp(...)

        This method is for validation/mapping derivation, not full term computation.

        Parameters:
            P_coeffs: P polynomial coefficients
            Q_coeffs: Q polynomial coefficients
            u: First integration variable
            t: Second integration variable
            theta: θ parameter
            R: R parameter
            l1: Left piece index ℓ₁
            l2: Right piece index ℓ₂
            include_prefactor: Whether to include algebraic prefactor
            include_exp: Whether to include exponential factor

        Returns:
            BivariateSeries representing the product
        """
        result = BivariateSeries.constant(1.0, self.max_order)

        # P_left = P(u + x) where x represents the sum of ℓ₁ x-variables
        # In bivariate reduction: x coeff is 1, y coeff is 0
        P_left = compose_polynomial_bivariate(P_coeffs, u, 1.0, 0.0, self.max_order)
        result = result * P_left

        # P_right = P(u + y) where y represents the sum of ℓ₂ y-variables
        P_right = compose_polynomial_bivariate(P_coeffs, u, 0.0, 1.0, self.max_order)
        result = result * P_right

        # For now, we can add Q and exp factors as needed for specific term types
        # This is a simplified version for mapping validation

        if include_prefactor:
            # Algebraic prefactor: (1/θ + X + Y) - same for ALL pairs!
            # NOTE: The prefactor is always (1/θ + X + Y), NOT (1/θ + ℓ₁X + ℓ₂Y).
            # This matches the DSL structure where the prefactor doesn't depend on pair indices.
            prefactor = linear_bivariate(1.0 / theta, 1.0, 1.0, self.max_order)
            result = result * prefactor

        return result

    def extract_coefficient(
        self,
        series: BivariateSeries,
        l1: int,
        l2: int
    ) -> float:
        """
        Extract the coefficient of x^{ℓ₁} y^{ℓ₂} from a bivariate series.

        This is the bivariate analog of extracting the x₁x₂...y₁y₂... coefficient
        in the multi-variable DSL.
        """
        return series.extract_mixed_coeff(l1, l2)


# =============================================================================
# Validation utilities
# =============================================================================

def validate_polynomial_taylor_coeffs():
    """Validate polynomial Taylor expansion against known formula."""
    # Test: P(x) = x^3
    # P(u+z) = (u+z)^3 = u^3 + 3u^2 z + 3u z^2 + z^3
    coeffs = np.array([0.0, 0.0, 0.0, 1.0])  # x^3
    u = 2.0
    taylor = polynomial_taylor_coeffs(coeffs, u, 4)

    expected = np.array([
        u**3,           # k=0: u^3
        3 * u**2,       # k=1: 3u^2
        3 * u,          # k=2: 3u
        1.0,            # k=3: 1
        0.0             # k=4: 0
    ])

    assert np.allclose(taylor, expected), f"Taylor mismatch: {taylor} vs {expected}"
    return True


def validate_exp_taylor_coeffs():
    """Validate exponential Taylor coefficients."""
    # exp(λz) = 1 + λz + λ²z²/2 + λ³z³/6 + ...
    lam = 2.0
    taylor = exp_taylor_coeffs(lam, 4)

    expected = np.array([
        1.0,            # k=0
        lam,            # k=1
        lam**2 / 2,     # k=2
        lam**3 / 6,     # k=3
        lam**4 / 24     # k=4
    ])

    assert np.allclose(taylor, expected), f"Exp Taylor mismatch: {taylor} vs {expected}"
    return True


def validate_bivariate_multiplication():
    """Validate bivariate series multiplication."""
    # (1 + x)(1 + y) = 1 + x + y + xy
    A = BivariateSeries.constant(1.0, 4) + BivariateSeries.from_x(4)  # 1 + x
    B = BivariateSeries.constant(1.0, 4) + BivariateSeries.from_y(4)  # 1 + y
    C = A * B

    assert abs(C.get_coeff(0, 0) - 1.0) < 1e-14  # const term
    assert abs(C.get_coeff(1, 0) - 1.0) < 1e-14  # x term
    assert abs(C.get_coeff(0, 1) - 1.0) < 1e-14  # y term
    assert abs(C.get_coeff(1, 1) - 1.0) < 1e-14  # xy term
    assert abs(C.get_coeff(2, 0)) < 1e-14        # x² should be 0

    return True


def validate_compose_polynomial():
    """Validate polynomial composition to bivariate series."""
    # P(x) = x^2
    # P(u + ax + by) = (u + ax + by)^2 = u^2 + 2u(ax+by) + (ax+by)^2
    #                = u^2 + 2au·x + 2bu·y + a²x² + 2ab·xy + b²y²
    poly = np.array([0.0, 0.0, 1.0])  # x^2
    u, a, b = 3.0, 2.0, 5.0

    series = compose_polynomial_bivariate(poly, u, a, b, 4)

    assert abs(series.get_coeff(0, 0) - u**2) < 1e-12
    assert abs(series.get_coeff(1, 0) - 2*a*u) < 1e-12
    assert abs(series.get_coeff(0, 1) - 2*b*u) < 1e-12
    assert abs(series.get_coeff(2, 0) - a**2) < 1e-12
    assert abs(series.get_coeff(1, 1) - 2*a*b) < 1e-12
    assert abs(series.get_coeff(0, 2) - b**2) < 1e-12

    return True


def run_all_validations():
    """Run all reference engine validations."""
    print("Validating polynomial Taylor coefficients...")
    validate_polynomial_taylor_coeffs()
    print("  PASSED")

    print("Validating exponential Taylor coefficients...")
    validate_exp_taylor_coeffs()
    print("  PASSED")

    print("Validating bivariate multiplication...")
    validate_bivariate_multiplication()
    print("  PASSED")

    print("Validating polynomial composition...")
    validate_compose_polynomial()
    print("  PASSED")

    print("\nAll reference engine validations PASSED!")


if __name__ == "__main__":
    run_all_validations()
