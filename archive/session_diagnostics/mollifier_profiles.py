"""
Mollifier profile generators for PRZZ omega-case handling.

This module implements the correct Case B/C Taylor coefficient extraction
as specified in PRZZ TeX lines 2302-2377.

Key insight from GPT:
- Case C is NOT a multiplier - it changes the polynomial factor BEFORE derivatives
- The kernel K_omega(arg;R) replaces P(arg) inside the series expansion
- The (1-a)^j factor appears from differentiating under the integral

For d=1:
    omega = ell - 1

    Case A (omega=-1, ell=0): Conrey piece with derivative structure
    Case B (omega=0, ell=1):  Standard P(u+X) evaluation
    Case C (omega>0, ell>1):  Kernel with auxiliary a-integral

Reference: PRZZ arXiv:1802.10521 Section 7.1-7.3
"""

from __future__ import annotations
import numpy as np
from math import factorial
from typing import Protocol, Tuple, Optional
from dataclasses import dataclass


class PolyLike(Protocol):
    """Protocol for polynomial objects."""
    def eval(self, x: np.ndarray) -> np.ndarray: ...
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray: ...


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes and weights on [0,1].

    Args:
        n: Number of quadrature points

    Returns:
        (nodes, weights) tuple, both shape (n,)
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)  # Transform from [-1,1] to [0,1]
    w = 0.5 * w
    return x, w


def case_b_taylor_coeffs(
    P: PolyLike,
    u: float,
    max_order: int
) -> np.ndarray:
    """
    Case B (omega=0) Taylor coefficients.

    For omega=0, the profile is just P(u+X), so:
        coeff[j] = P^{(j)}(u)

    This is the "factorial basis" representation where coeff[j] is the
    j-th derivative (not divided by j!).

    Args:
        P: Polynomial object with eval_deriv method
        u: Base point (scalar)
        max_order: Maximum derivative order

    Returns:
        Array of shape (max_order+1,) with coeff[j] = P^{(j)}(u)
    """
    u_arr = np.array([u])
    return np.array([
        float(P.eval_deriv(u_arr, j)[0])
        for j in range(max_order + 1)
    ], dtype=float)


def case_c_taylor_coeffs(
    P: PolyLike,
    u: float,
    omega: int,
    R: float,
    theta: float,
    max_order: int,
    n_quad_a: int = 40
) -> np.ndarray:
    """
    Case C (omega>0) Taylor coefficients with correct kernel structure.

    For omega>0, the profile is the kernel:
        K_omega(arg;R) = arg^omega/(omega-1)! * integral_0^1
                         a^{omega-1} * P((1-a)*arg) * exp(R*theta*a*arg) da

    The j-th derivative at arg=u is:
        F^{(j)}(u) = u^omega/(omega-1)! * integral_0^1
                     a^{omega-1} * exp(R*theta*a*u) * (1-a)^j * P^{(j)}((1-a)*u) da

    KEY: The (1-a)^j factor appears from differentiating under the integral.
    This is what the "multiply by a-integral" approximation misses!

    Args:
        P: Polynomial object with eval_deriv method
        u: Base point (scalar)
        omega: Omega value (must be > 0)
        R: R parameter
        theta: Theta parameter (typically 4/7)
        max_order: Maximum derivative order
        n_quad_a: Number of quadrature points for a-integral

    Returns:
        Array of shape (max_order+1,) with coeff[j] = F^{(j)}(u)
    """
    assert omega > 0, f"Case C requires omega > 0, got {omega}"

    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # Prefactor: u^omega / (omega-1)!
    if abs(u) < 1e-15:
        # At u=0, the kernel vanishes for omega > 0
        return np.zeros(max_order + 1)

    pref = (u ** omega) / factorial(omega - 1)

    # Common part: a^{omega-1} * exp(R * theta * u * a)
    base = (a_nodes ** (omega - 1)) * np.exp(R * theta * u * a_nodes)

    coeff = np.zeros(max_order + 1)
    for j in range(max_order + 1):
        # Argument for polynomial: (1-a)*u
        arg = (1.0 - a_nodes) * u

        # P^{(j)} at the shifted argument
        Pj = P.eval_deriv(arg, j)

        # Full integrand: base * (1-a)^j * P^{(j)}((1-a)*u)
        integrand = base * ((1.0 - a_nodes) ** j) * Pj

        coeff[j] = pref * float(np.sum(a_weights * integrand))

    return coeff


def case_a_taylor_coeffs(
    P: PolyLike,
    u: float,
    alpha: float,
    logN: float,
    max_order: int
) -> np.ndarray:
    """
    Case A (omega=-1) Taylor coefficients.

    For omega=-1 (ell=0, Conrey piece), PRZZ defines:
        F_A = (1/logN) * d/dX [ N^{alpha*X} * P(u+X) ] at X=0

    Expanding:
        F_A = alpha * P(u) + P'(u) / logN

    This is a special case that doesn't follow the standard kernel pattern.

    NOTE: This produces O(1/L^2) contributions that are lower order.
    Only implement if your mollifier truly includes an ell=0 piece.

    Args:
        P: Polynomial object with eval_deriv method
        u: Base point (scalar)
        alpha: Alpha parameter (typically -R/L)
        logN: log N = theta * log T
        max_order: Maximum derivative order (usually 0 for Case A)

    Returns:
        Array of shape (max_order+1,) with coefficients
    """
    # Case A produces a single-term series (the derivative is already taken)
    # For proper implementation, need to expand N^{alpha*X} * P(u+X) as a
    # bivariate series and differentiate.

    # Simplified version for order 0:
    u_arr = np.array([u])
    P_u = float(P.eval(u_arr)[0])
    P_prime_u = float(P.eval_deriv(u_arr, 1)[0])

    # F_A at X=0 = alpha * P(u) + P'(u) / logN
    F_A = alpha * P_u + P_prime_u / logN

    coeff = np.zeros(max_order + 1)
    coeff[0] = F_A

    # Higher derivatives would require more careful expansion
    # For now, we only support max_order=0 for Case A
    if max_order > 0:
        # Would need to compute d/dX[F_A(u+X)] at X=0
        # This involves second derivatives of P
        for j in range(1, max_order + 1):
            # d^j/dX^j [alpha*P(u+X) + P'(u+X)/logN] at X=0
            # = alpha*P^{(j)}(u) + P^{(j+1)}(u)/logN
            Pj = float(P.eval_deriv(u_arr, j)[0])
            Pj1 = float(P.eval_deriv(u_arr, j + 1)[0])
            coeff[j] = alpha * Pj + Pj1 / logN

    return coeff


@dataclass
class MollifierProfile:
    """
    Represents a mollifier piece with its omega-case.

    This class encapsulates the logic for choosing the correct
    Taylor coefficient generator based on omega.

    Attributes:
        poly_name: Name of the polynomial ("P1", "P2", "P3")
        ell: The ell index (number of Λ convolutions + 1)
        omega: Computed as ell - 1 for d=1
    """
    poly_name: str
    ell: int

    @property
    def omega(self) -> int:
        """Omega = ell - 1 for d=1."""
        return self.ell - 1

    @property
    def case(self) -> str:
        """Return the omega-case label."""
        if self.omega == -1:
            return "A"
        elif self.omega == 0:
            return "B"
        else:
            return "C"

    def get_taylor_coeffs(
        self,
        P: PolyLike,
        u: float,
        max_order: int,
        R: float,
        theta: float,
        alpha: Optional[float] = None,
        logN: Optional[float] = None,
        n_quad_a: int = 40
    ) -> np.ndarray:
        """
        Get Taylor coefficients for this profile at point u.

        Args:
            P: Polynomial object
            u: Base point
            max_order: Maximum derivative order
            R: R parameter
            theta: Theta parameter
            alpha: Alpha for Case A (default: -R/logN)
            logN: log N for Case A (default: theta * some large value)
            n_quad_a: Quadrature points for Case C a-integral

        Returns:
            Array of Taylor coefficients in factorial basis
        """
        if self.omega == -1:
            # Case A: Conrey piece
            if alpha is None:
                # Default: asymptotic limit value
                # In practice, this contributes O(1/L^2)
                logT = 100.0  # Large value for asymptotic
                logN_val = theta * logT if logN is None else logN
                alpha_val = -R / logT
            else:
                alpha_val = alpha
                logN_val = logN if logN is not None else theta * 100.0
            return case_a_taylor_coeffs(P, u, alpha_val, logN_val, max_order)

        elif self.omega == 0:
            # Case B: Standard polynomial
            return case_b_taylor_coeffs(P, u, max_order)

        else:
            # Case C: Kernel with a-integral
            return case_c_taylor_coeffs(
                P, u, self.omega, R, theta, max_order, n_quad_a
            )


# =============================================================================
# Precomputation cache for efficiency
# =============================================================================

class ProfileCache:
    """
    Cache for precomputed Taylor coefficients on u-grid.

    Case C coefficients depend on (piece, u, R, theta, order) but NOT on t.
    So we precompute them once per u-node and reuse across all t-nodes.
    """

    def __init__(self):
        self._cache = {}

    def get_coeffs(
        self,
        profile: MollifierProfile,
        P: PolyLike,
        u_nodes: np.ndarray,
        max_order: int,
        R: float,
        theta: float,
        n_quad_a: int = 40
    ) -> np.ndarray:
        """
        Get cached Taylor coefficients for all u-nodes.

        Args:
            profile: MollifierProfile object
            P: Polynomial object
            u_nodes: Array of u values
            max_order: Maximum derivative order
            R: R parameter
            theta: Theta parameter
            n_quad_a: Quadrature points for Case C

        Returns:
            Array of shape (len(u_nodes), max_order+1)
        """
        key = (profile.poly_name, profile.ell, max_order, R, theta, len(u_nodes))

        if key not in self._cache:
            coeffs = np.zeros((len(u_nodes), max_order + 1))
            for i, u in enumerate(u_nodes):
                coeffs[i] = profile.get_taylor_coeffs(
                    P, u, max_order, R, theta, n_quad_a=n_quad_a
                )
            self._cache[key] = coeffs

        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache = {}


# Global cache instance
_profile_cache = ProfileCache()


def get_profile_coeffs(
    profile: MollifierProfile,
    P: PolyLike,
    u_nodes: np.ndarray,
    max_order: int,
    R: float,
    theta: float,
    n_quad_a: int = 40
) -> np.ndarray:
    """
    Convenience function to get cached profile coefficients.

    Args:
        profile: MollifierProfile object
        P: Polynomial object
        u_nodes: Array of u values
        max_order: Maximum derivative order
        R: R parameter
        theta: Theta parameter
        n_quad_a: Quadrature points for Case C

    Returns:
        Array of shape (len(u_nodes), max_order+1)
    """
    return _profile_cache.get_coeffs(
        profile, P, u_nodes, max_order, R, theta, n_quad_a
    )


def clear_profile_cache():
    """Clear the global profile cache."""
    _profile_cache.clear()
