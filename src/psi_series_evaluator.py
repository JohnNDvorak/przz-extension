"""
src/psi_series_evaluator.py
Ψ-based Evaluator using Series Coefficient Extraction

This evaluator computes pair contributions using the CORRECT approach:
1. Build F(x,y) as a product of Taylor series
2. Extract coefficients g_{a,b} from (1/θ + x + y) × F(x,y)
3. Combine with Ψ monomial coefficients using MONOMIAL weights (a+b)

KEY INSIGHT (from GPT guidance):
- The identity: g_{a,b} = (1/θ)f_{a,b} + f_{a-1,b} + f_{a,b-1}
- No manual partial derivatives needed
- Uses series multiplication instead

FACTOR STRUCTURE:
- P_L(u-x): polynomial in x only
- P_R(u-y): polynomial in y only
- Q(α_arg): where α_arg = t + θ(t·x + (t-1)·y)
- Q(β_arg): where β_arg = t + θ((t-1)·x + t·y)
- exp(R(α_arg + β_arg)): exponential of linear form

WEIGHT CONVENTION:
- Use MONOMIAL weights (1-u)^{a+b} from Ψ expansion
- NOT pair-based weights (which are GenEval's collapsed form)
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from math import factorial

from src.psi_expansion import expand_psi, MonomialTwoC
from src.mollifier_profiles import case_b_taylor_coeffs, case_c_taylor_coeffs


def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class PsiPairResult:
    """Result from evaluating a pair using Ψ expansion."""
    ell: int
    ellbar: int
    total: float
    monomial_contributions: Dict[Tuple[int, int, int, int, int], float]


class PsiSeriesEvaluator:
    """
    Evaluator using bivariate Taylor coefficient extraction.

    For each (u, t) quadrature point:
    1. Build F(x,y) = P_L(u-x) × P_R(u-y) × Q(α) × Q(β) × exp(R(α+β))
    2. Compute Taylor coefficients f_{i,j} via series multiplication
    3. Convert to g_{i,j} = [x^i y^j] of (1/θ + x + y)×F
    4. Integrate over (u,t) with weight (1-u)^{a+b}
    """

    def __init__(
        self,
        P_ell,
        P_ellbar,
        Q,
        R: float,
        theta: float,
        max_order: int = 3,
        n_quad: int = 60,
        omega_left: int = 0,
        omega_right: int = 0,
        n_quad_a: int = 40,
    ):
        """
        Initialize the Ψ-series evaluator.

        Args:
            P_ell: Left polynomial
            P_ellbar: Right polynomial
            Q: Q polynomial
            R: PRZZ R parameter
            theta: θ parameter (typically 4/7)
            max_order: Maximum derivative order to compute (3 for K=3)
            n_quad: Quadrature points
        """
        self.P_ell = P_ell
        self.P_ellbar = P_ellbar
        self.Q = Q
        self.R = R
        self.theta = theta
        self.max_order = int(max_order)
        self.n_quad = int(n_quad)
        self.omega_left = int(omega_left)
        self.omega_right = int(omega_right)
        self.n_quad_a = int(n_quad_a)

        # Quadrature
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)
        self.t_nodes, self.t_weights = gauss_legendre_01(n_quad)

        # Precompute Q derivatives (need up to max_order for Taylor expansion)
        self._precompute_Q_derivs()
        self._precompute_profile_derivs()

    def _precompute_Q_derivs(self):
        """Precompute Q and its derivatives at t nodes up to 2*max_order."""
        self.Q_derivs = []
        max_k = 2 * self.max_order
        for k in range(max_k + 1):
            self.Q_derivs.append(self.Q.eval_deriv(self.t_nodes, k))

    def _precompute_profile_derivs(self) -> None:
        """Precompute left/right profile derivatives on the u quadrature grid.

        For omega==0: derivatives are P^{(j)}(u).
        For omega>0: derivatives are K_omega^{(j)}(u;R,theta) from the Case C
        auxiliary integral (see `src.mollifier_profiles`).
        """

        n = self.max_order + 1
        n_u = len(self.u_nodes)

        self._left_profile_derivs = np.zeros((n_u, n), dtype=float)
        self._right_profile_derivs = np.zeros((n_u, n), dtype=float)

        # Case B is vectorizable; Case C currently computed pointwise.
        if self.omega_left <= 0:
            for j in range(n):
                self._left_profile_derivs[:, j] = self.P_ell.eval_deriv(self.u_nodes, j)
        else:
            for iu, u in enumerate(self.u_nodes):
                self._left_profile_derivs[iu, :] = case_c_taylor_coeffs(
                    self.P_ell,
                    u=float(u),
                    omega=self.omega_left,
                    R=self.R,
                    theta=self.theta,
                    max_order=self.max_order,
                    n_quad_a=self.n_quad_a,
                )

        if self.omega_right <= 0:
            for j in range(n):
                self._right_profile_derivs[:, j] = self.P_ellbar.eval_deriv(self.u_nodes, j)
        else:
            for iu, u in enumerate(self.u_nodes):
                self._right_profile_derivs[iu, :] = case_c_taylor_coeffs(
                    self.P_ellbar,
                    u=float(u),
                    omega=self.omega_right,
                    R=self.R,
                    theta=self.theta,
                    max_order=self.max_order,
                    n_quad_a=self.n_quad_a,
                )

    def _profile_derivs_at_u(
        self,
        *,
        side: str,
        u: float,
        u_idx: Optional[int],
    ) -> np.ndarray:
        """Return profile derivatives at u in the factorial basis.

        Args:
            side: "left" or "right"
            u: u value (float)
            u_idx: optional u-grid index for fast lookup

        Returns:
            ndarray of shape (max_order+1,) with derivs[j] = F^{(j)}(u)
            where F is either P (omega=0) or K_omega (omega>0).
        """

        if side == "left":
            omega = self.omega_left
            poly = self.P_ell
            if u_idx is not None:
                return self._left_profile_derivs[u_idx]
        elif side == "right":
            omega = self.omega_right
            poly = self.P_ellbar
            if u_idx is not None:
                return self._right_profile_derivs[u_idx]
        else:
            raise ValueError(f"Unknown side: {side!r}")

        if omega <= 0:
            return case_b_taylor_coeffs(poly, u=float(u), max_order=self.max_order)

        return case_c_taylor_coeffs(
            poly,
            u=float(u),
            omega=omega,
            R=self.R,
            theta=self.theta,
            max_order=self.max_order,
            n_quad_a=self.n_quad_a,
        )

    def _build_F_coefficients(
        self, u: float, t: float, *, u_idx: Optional[int] = None, t_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Build 2D Taylor coefficients f_{i,j} for F(x,y) at given (u,t).

        F(x,y) = P_L(u-x) × P_R(u-y) × Q(α_arg) × Q(β_arg) × exp(R(α_arg+β_arg))

        Args:
            u: u quadrature point
            t: t quadrature point
            u_idx: Optional index into the u quadrature grid
            t_idx: Optional index into precomputed Q_derivs for efficiency

        Returns:
            2D array f[i,j] for i,j in [0, max_order]
        """
        n = self.max_order + 1

        # Initialize coefficient arrays (n x n)
        # f[i,j] = coefficient of x^i y^j

        # Factor 1: "left profile" evaluated at u with an x-shift.
        # In raw regime (omega_left=0): profile is P_ell.
        # In paper regime (omega_left>0): profile is K_omega (Case C).
        #
        # CONVENTION: Match existing Ψ-series convention (no sign alternation).
        # Taylor: f1[i,0] = F^{(i)}(u)/i!
        f1 = np.zeros((n, n))
        left_derivs = self._profile_derivs_at_u(side="left", u=u, u_idx=u_idx)
        for i in range(n):
            f1[i, 0] = left_derivs[i] / factorial(i)

        # Factor 2: "right profile" evaluated at u with a y-shift.
        # Taylor: f2[0,j] = F^{(j)}(u)/j! (no sign alternation)
        f2 = np.zeros((n, n))
        right_derivs = self._profile_derivs_at_u(side="right", u=u, u_idx=u_idx)
        for j in range(n):
            f2[0, j] = right_derivs[j] / factorial(j)

        # Factor 3: Q(α_arg) where α_arg = t + θ(t·x + (t-1)·y)
        # At x=y=0, α_arg = t
        # d(α_arg)/dx = θt, d(α_arg)/dy = θ(t-1)
        # Q(α_arg) = sum_k Q^{(k)}(t)/k! × (θt·x + θ(t-1)·y)^k
        dα_dx = self.theta * t
        dα_dy = self.theta * (t - 1)
        f3 = self._expand_Q_at_t(t, dα_dx, dα_dy, t_idx)

        # Factor 4: Q(β_arg) where β_arg = t + θ((t-1)·x + t·y)
        # d(β_arg)/dx = θ(t-1), d(β_arg)/dy = θt
        dβ_dx = self.theta * (t - 1)
        dβ_dy = self.theta * t
        f4 = self._expand_Q_at_t(t, dβ_dx, dβ_dy, t_idx)

        # Factor 5a: exp(R × α_arg)
        # At x=y=0: exp(Rt)
        # Linear coefficients: R × θt (x), R × θ(t-1) (y)
        exp_α_base = np.exp(self.R * t)
        c_α_x = self.R * dα_dx
        c_α_y = self.R * dα_dy
        f5a = self._expand_exp_at_t(exp_α_base, c_α_x, c_α_y)

        # Factor 5b: exp(R × β_arg)
        # At x=y=0: exp(Rt)
        # Linear coefficients: R × θ(t-1) (x), R × θt (y)
        exp_β_base = np.exp(self.R * t)
        c_β_x = self.R * dβ_dx
        c_β_y = self.R * dβ_dy
        f5b = self._expand_exp_at_t(exp_β_base, c_β_x, c_β_y)

        # Multiply all factors via 2D convolution
        # Start with f1
        result = f1.copy()

        # Multiply by f2
        result = self._multiply_2d(result, f2)

        # Multiply by f3
        result = self._multiply_2d(result, f3)

        # Multiply by f4
        result = self._multiply_2d(result, f4)

        # Multiply by f5a (exp(R×α))
        result = self._multiply_2d(result, f5a)

        # Multiply by f5b (exp(R×β))
        result = self._multiply_2d(result, f5b)

        return result

    def _expand_exp_at_t(self, exp_base: float, c_x: float, c_y: float) -> np.ndarray:
        """
        Expand exp(base + c_x·x + c_y·y) as 2D Taylor series.

        exp(base + c_x·x + c_y·y) = exp_base × exp(c_x·x + c_y·y)
                                  = exp_base × Σ (c_x·x + c_y·y)^k / k!

        The coefficient of x^i y^j is:
            exp_base × c_x^i × c_y^j / (i! × j!)
        """
        n = self.max_order + 1
        result = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                result[i, j] = exp_base * (c_x ** i) * (c_y ** j) / (factorial(i) * factorial(j))

        return result

    def _expand_Q_at_t(self, t: float, a: float, b: float, t_idx: int = None) -> np.ndarray:
        """
        Expand Q(t + a·x + b·y) as 2D Taylor series using CLOSED-FORM formula.

        For linear form δ = a·x + b·y:
            Q(t + δ) = Σ_k Q^{(k)}(t)/k! × (a·x + b·y)^k

        The coefficient of x^i y^j is (closed-form):
            [x^i y^j] Q(t+δ) = Q^{(i+j)}(t) / (i+j)! × C(i+j, i) × a^i × b^j
                             = Q^{(i+j)}(t) × a^i × b^j / (i! × j!)

        This formula is exact and avoids numerical binomial expansion errors.

        Args:
            t: Base point for Q evaluation (used for fallback if t_idx not provided)
            a: Coefficient of x in the linear perturbation (e.g., θt for α)
            b: Coefficient of y in the linear perturbation (e.g., θ(t-1) for α)
            t_idx: Optional index into precomputed Q_derivs array

        Returns:
            2D array of coefficients q[i,j]
        """
        n = self.max_order + 1
        result = np.zeros((n, n))

        # Use precomputed Q derivatives if t_idx provided, else compute fresh
        if t_idx is not None:
            for i in range(n):
                for j in range(n):
                    k = i + j
                    Qk = self.Q_derivs[k][t_idx]  # Q^{(k)}(t)
                    result[i, j] = Qk * (a ** i) * (b ** j) / (factorial(i) * factorial(j))
        else:
            # Fallback: compute Q derivatives locally
            max_deriv = 2 * self.max_order
            Q_derivs = [self.Q.eval_deriv(np.array([t]), k)[0] for k in range(max_deriv + 1)]
            for i in range(n):
                for j in range(n):
                    k = i + j
                    Qk = Q_derivs[k]
                    result[i, j] = Qk * (a ** i) * (b ** j) / (factorial(i) * factorial(j))

        return result

    def _multiply_2d(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two 2D polynomial coefficient arrays (convolution).

        Truncates to max_order + 1 in each dimension.
        """
        n = self.max_order + 1
        result = np.zeros((n, n))

        for i1 in range(n):
            for j1 in range(n):
                if a[i1, j1] == 0:
                    continue
                for i2 in range(n - i1):
                    for j2 in range(n - j1):
                        result[i1 + i2, j1 + j2] += a[i1, j1] * b[i2, j2]

        return result

    def _prefactor_transform(self, f: np.ndarray) -> np.ndarray:
        """
        Apply prefactor transformation: G = (1/θ + x + y) × F

        g_{a,b} = (1/θ)f_{a,b} + f_{a-1,b} + f_{a,b-1}
        """
        n = self.max_order + 1
        g = np.zeros((n, n))

        for a in range(n):
            for b in range(n):
                # (1/θ) × f_{a,b}
                g[a, b] = f[a, b] / self.theta

                # + f_{a-1,b} (if a >= 1)
                if a >= 1:
                    g[a, b] += f[a - 1, b]

                # + f_{a,b-1} (if b >= 1)
                if b >= 1:
                    g[a, b] += f[a, b - 1]

        return g

    def compute_integral_grid(self) -> Dict[Tuple[int, int, int], float]:
        """
        Compute all integrals I[a, b, weight] needed for Ψ evaluation.

        I[a, b, w] = ∫∫ g_{a,b}(u,t) × (1-u)^w du dt

        where g_{a,b} is the coefficient of x^a y^b in (1/θ + x + y)×F.

        Returns:
            Dict mapping (a, b, weight_exp) -> integral value
        """
        n = self.max_order + 1

        # Accumulate integrals for each (a, b, weight) combination
        # We need weights 0 through 2*max_order (for a+b up to 2*max_order)
        max_weight = 2 * self.max_order

        # Initialize accumulators: integral_accum[a, b, w]
        integral_accum = {}
        for a in range(n):
            for b in range(n):
                for w in range(max_weight + 1):
                    integral_accum[(a, b, w)] = 0.0

        # Integrate over (u, t) grid
        for iu, u in enumerate(self.u_nodes):
            wu = self.u_weights[iu]

            # Precompute weight factors
            weight_factors = [(1.0 - u) ** w for w in range(max_weight + 1)]

            for it, t in enumerate(self.t_nodes):
                wt = self.t_weights[it]

                # Build F coefficients at this (u, t), using precomputed Q derivs
                f_coeffs = self._build_F_coefficients(u, t, u_idx=iu, t_idx=it)

                # Apply prefactor transformation
                g_coeffs = self._prefactor_transform(f_coeffs)

                # Accumulate to integrals
                for a in range(n):
                    for b in range(n):
                        g_ab = g_coeffs[a, b]
                        for w in range(max_weight + 1):
                            integral_accum[(a, b, w)] += wu * wt * g_ab * weight_factors[w]

        return integral_accum

    def eval_pair(self, ell: int, ellbar: int, verbose: bool = False) -> PsiPairResult:
        """
        Evaluate pair (ℓ, ℓ̄) using Ψ expansion with series coefficients.

        Uses MONOMIAL weights (1-u)^{a+b} - NOT pair-based weights!
        """
        # Get Ψ monomials
        monomials = expand_psi(ell, ellbar)

        if verbose:
            print(f"\n=== PsiSeriesEvaluator ({ell},{ellbar}): {len(monomials)} monomials ===")

        # Compute integral grid
        integral_grid = self.compute_integral_grid()

        # Sum over monomials
        total = 0.0
        contributions = {}

        for mono in monomials:
            a, b = mono.a, mono.b
            weight_exp = mono.weight_exponent  # = a + b (MONOMIAL weight!)
            psi_coeff = mono.coeff

            # Look up integral
            key = (a, b, weight_exp)
            if key in integral_grid:
                integral_val = integral_grid[key]
            else:
                integral_val = 0.0
                if verbose:
                    print(f"  WARNING: No integral for {key}")

            contrib = psi_coeff * integral_val
            total += contrib
            contributions[mono.key()] = contrib

            if verbose:
                print(f"  {mono}: weight=(1-u)^{weight_exp}, "
                      f"integral={integral_val:.6f}, contrib={contrib:.6f}")

        if verbose:
            print(f"  TOTAL = {total:.6f}")

        return PsiPairResult(
            ell=ell,
            ellbar=ellbar,
            total=total,
            monomial_contributions=contributions
        )


def compute_c_psi_series(P1, P2, P3, Q, R: float, theta: float = 4/7,
                          n_quad: int = 60, verbose: bool = False,
                          kernel_regime: str = "raw",
                          n_quad_a: int = 40) -> float:
    """
    Compute total c using PsiSeriesEvaluator.
    """
    from src.kernel_registry import kernel_spec_for_piece

    poly_map = {1: P1, 2: P2, 3: P3}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    total = 0.0
    for ell, ellbar in pairs:
        P_ell = poly_map[ell]
        P_ellbar = poly_map[ellbar]

        spec_left = kernel_spec_for_piece(ell, regime=kernel_regime, d=1)
        spec_right = kernel_spec_for_piece(ellbar, regime=kernel_regime, d=1)

        max_order = max(ell, ellbar)
        evaluator = PsiSeriesEvaluator(
            P_ell,
            P_ellbar,
            Q,
            R,
            theta,
            max_order=max_order,
            n_quad=n_quad,
            omega_left=spec_left.omega,
            omega_right=spec_right.omega,
            n_quad_a=n_quad_a,
        )
        result = evaluator.eval_pair(ell, ellbar, verbose=verbose)

        sym = 1 if ell == ellbar else 2
        total += sym * result.total

        if verbose:
            print(f"  ({ell},{ellbar}) × {sym} = {sym * result.total:.6f}")

    if verbose:
        print(f"\nTotal c = {total:.6f}")

    return total


# =============================================================================
# TESTING
# =============================================================================

def test_11_oracle():
    """Test (1,1) against the known oracle."""
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    evaluator = PsiSeriesEvaluator(P1, P1, Q, R=1.3036, theta=4/7, max_order=1, n_quad=60)
    result = evaluator.eval_pair(1, 1, verbose=True)

    oracle = 0.359159
    print(f"\n(1,1) Result: {result.total:.6f}")
    print(f"Oracle: {oracle:.6f}")
    print(f"Error: {abs(result.total - oracle) / oracle * 100:.2f}%")
    print(f"Match: {abs(result.total - oracle) < 1e-3}")


if __name__ == "__main__":
    print("=" * 70)
    print("PSI SERIES EVALUATOR TEST")
    print("=" * 70)

    test_11_oracle()
