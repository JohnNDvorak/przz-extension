"""
PRZZ Section 7 Oracle Implementation.

This module implements the PRZZ pipeline for computing the main-term constant c
directly from the paper's formulas (arXiv:1802.10521 Section 7), WITHOUT using
the I₁-I₄ DSL intermediate representation.

The goal is to build a reference implementation that we can diff against our
DSL-based evaluator to identify where they diverge.

Key formulas from PRZZ TeX:
- Line 2303: ω(d,l) := 1×l₁ + 2×l₂ + ... + d×l_d - 1
- Lines 2305-2385: Case A/B/C structure
- Lines 2370-2385: F_d factor definitions
- Line 2391-2399: Euler-Maclaurin summation

For d=1, K=3:
- Piece 1 (P₁, μ): Λ-count = 0
- Piece 2 (P₂, μ⋆Λ): Λ-count = 1
- Piece 3 (P₃, μ⋆Λ⋆Λ): Λ-count = 2
"""

from __future__ import annotations
import numpy as np
from math import factorial, log, exp
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from numpy.polynomial.legendre import leggauss


# =============================================================================
# Gauss-Legendre quadrature
# =============================================================================

def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0,1]."""
    x, w = leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


# =============================================================================
# ω computation and Case classification
# =============================================================================

def compute_omega_d1(l1: int) -> int:
    """
    Compute ω for d=1.

    ω(1, (l₁,)) = 1×l₁ - 1 = l₁ - 1

    Args:
        l1: The Faà-di-Bruno index for d=1

    Returns:
        ω value: -1 (Case A), 0 (Case B), or >0 (Case C)
    """
    return l1 - 1


def get_omega_case(omega: int) -> str:
    """Return the case label based on ω value."""
    if omega == -1:
        return "A"
    elif omega == 0:
        return "B"
    else:
        return "C"


# =============================================================================
# Weight functions U, V, W (d=1 specialization)
# =============================================================================

def weight_U_d1(l1: int) -> float:
    """
    Case A weight for d=1.

    U(1,l) = 1{ω=-1} × (1!)^{l₁} × (-1)^{l₁}

    For ω = l₁ - 1 = -1, need l₁ = 0.
    Then U = (1!)^0 × (-1)^0 = 1
    """
    omega = compute_omega_d1(l1)
    if omega != -1:
        return 0.0
    # l1 = 0 for Case A
    sign = (-1) ** l1
    fact_power = factorial(1) ** l1
    return float(sign * fact_power)


def weight_V_d1(l1: int) -> float:
    """
    Case B weight for d=1.

    V(1,l) = 1{ω=0} × (1!)^{l₁} × (-1)^{l₁}

    For ω = l₁ - 1 = 0, need l₁ = 1.
    Then V = (1!)^1 × (-1)^1 = -1
    """
    omega = compute_omega_d1(l1)
    if omega != 0:
        return 0.0
    sign = (-1) ** l1
    fact_power = factorial(1) ** l1
    return float(sign * fact_power)


def weight_W_d1(l1: int) -> float:
    """
    Case C weight for d=1.

    W(1,l) = 1{ω>0} × (1!)^{l₁} × (-1)^{l₁}

    For ω = l₁ - 1 > 0, need l₁ > 1.
    Then W = (1!)^{l₁} × (-1)^{l₁} = (-1)^{l₁}
    """
    omega = compute_omega_d1(l1)
    if omega <= 0:
        return 0.0
    sign = (-1) ** l1
    fact_power = factorial(1) ** l1
    return float(sign * fact_power)


# =============================================================================
# F_d factor computation (PRZZ lines 2370-2385)
# =============================================================================

@dataclass
class FdFactor:
    """Represents an F_d factor evaluation."""
    case: str
    l1: int
    omega: int
    value: float


class PolynomialEvaluator:
    """
    Wrapper for polynomial evaluation that matches PRZZ conventions.

    PRZZ uses P_{d,ℓ}(u) where u = log(N/n)/log(N).
    Our polynomials use the same argument convention.
    """

    def __init__(self, poly):
        """
        Args:
            poly: A polynomial object with eval() and eval_deriv() methods
        """
        self.poly = poly

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """Evaluate P(u)."""
        return self.poly.eval(u)

    def deriv(self, u: np.ndarray, k: int = 1) -> np.ndarray:
        """Evaluate P^{(k)}(u)."""
        return self.poly.eval_deriv(u, k)


def compute_Fd_case_A(
    P: PolynomialEvaluator,
    u: float,
    alpha: float,
    logN: float,
    l1: int = 0
) -> float:
    """
    Case A (ω = -1): Derivative structure.

    From PRZZ line 2322:
    F_d(l,α,n) = U(d,l)/logN × d/dx[N^{αx} × P(u + x)]|_{x=0}

    Where u = log(N/n)/logN.

    Expanding the derivative:
    d/dx[N^{αx} × P(u + x)] = N^{αx} × [α×logN × P(u+x) + P'(u+x)]
    At x=0: α×logN × P(u) + P'(u)

    So: F_d = U(d,l)/logN × [α×logN × P(u) + P'(u)]
            = U(d,l) × [α × P(u) + P'(u)/logN]

    Args:
        P: Polynomial evaluator
        u: Argument u = log(N/n)/logN ∈ [0,1]
        alpha: α parameter (typically -R/logT)
        logN: log(N) = θ × logT
        l1: Faà-di-Bruno index (must be 0 for Case A)

    Returns:
        F_d factor value
    """
    if compute_omega_d1(l1) != -1:
        return 0.0

    U = weight_U_d1(l1)
    u_arr = np.array([u])
    P_u = float(P(u_arr)[0])
    Pprime_u = float(P.deriv(u_arr, 1)[0])

    # F_d = U × [α × P(u) + P'(u)/logN]
    return U * (alpha * P_u + Pprime_u / logN)


def compute_Fd_case_B(
    P: PolynomialEvaluator,
    u: float,
    l1: int = 1
) -> float:
    """
    Case B (ω = 0): Direct polynomial evaluation.

    From PRZZ line 2334:
    F_d(l,α,n) = V(d,l) × P(log(N/n)/logN) = V(d,l) × P(u)

    Args:
        P: Polynomial evaluator
        u: Argument u = log(N/n)/logN ∈ [0,1]
        l1: Faà-di-Bruno index (must be 1 for Case B with d=1)

    Returns:
        F_d factor value
    """
    if compute_omega_d1(l1) != 0:
        return 0.0

    V = weight_V_d1(l1)
    u_arr = np.array([u])
    P_u = float(P(u_arr)[0])

    return V * P_u


def compute_Fd_case_C(
    P: PolynomialEvaluator,
    u: float,
    alpha: float,
    logN: float,
    n_over_N: float,
    l1: int,
    n_quad: int = 60
) -> float:
    """
    Case C (ω > 0): Auxiliary a-integral.

    From PRZZ lines 2360-2361:
    F_d(l,α,n) = W(d,l) × (-1)^{1-ω}/(ω-1)! × (logN)^ω × u^ω
                × ∫₀¹ P((1-a)×u) × a^{ω-1} × (N/n)^{-αa} da

    Where u = log(N/n)/logN and (N/n)^{-αa} = exp(-α × logN × u × a).

    Args:
        P: Polynomial evaluator
        u: Argument u = log(N/n)/logN ∈ [0,1]
        alpha: α parameter
        logN: log(N)
        n_over_N: n/N ratio (for the exponential factor)
        l1: Faà-di-Bruno index (must be > 1 for Case C with d=1)
        n_quad: Quadrature points for a-integral

    Returns:
        F_d factor value
    """
    omega = compute_omega_d1(l1)
    if omega <= 0:
        return 0.0

    W = weight_W_d1(l1)

    # Prefactors
    sign_factor = (-1) ** (1 - omega)
    factorial_factor = 1.0 / factorial(omega - 1)
    logN_power = logN ** omega
    u_power = u ** omega

    # Handle u = 0 case (kernel vanishes)
    if abs(u) < 1e-15:
        return 0.0

    # Compute the a-integral via quadrature
    a_nodes, a_weights = gauss_legendre_01(n_quad)

    # P((1-a) × u)
    args = (1.0 - a_nodes) * u
    P_vals = P(args)

    # a^{ω-1}
    a_power = a_nodes ** (omega - 1)

    # (N/n)^{-α×a} = exp(-α × log(N/n) × a) = exp(-α × logN × u × a)
    # Note: log(N/n) = logN × u
    exp_factor = np.exp(-alpha * logN * u * a_nodes)

    # Full integrand
    integrand = P_vals * a_power * exp_factor

    integral = float(np.sum(a_weights * integrand))

    return W * sign_factor * factorial_factor * logN_power * u_power * integral


def compute_Fd(
    P: PolynomialEvaluator,
    u: float,
    alpha: float,
    logN: float,
    n_over_N: float,
    l1: int,
    n_quad: int = 60
) -> FdFactor:
    """
    Compute F_d factor for any case.

    Dispatches to Case A, B, or C based on ω = l1 - 1.

    Args:
        P: Polynomial evaluator
        u: Argument u = log(N/n)/logN ∈ [0,1]
        alpha: α parameter
        logN: log(N)
        n_over_N: n/N ratio
        l1: Faà-di-Bruno index
        n_quad: Quadrature points for Case C

    Returns:
        FdFactor with case label and value
    """
    omega = compute_omega_d1(l1)
    case = get_omega_case(omega)

    if case == "A":
        value = compute_Fd_case_A(P, u, alpha, logN, l1)
    elif case == "B":
        value = compute_Fd_case_B(P, u, l1)
    else:  # Case C
        value = compute_Fd_case_C(P, u, alpha, logN, n_over_N, l1, n_quad)

    return FdFactor(case=case, l1=l1, omega=omega, value=value)


# =============================================================================
# Euler-Maclaurin integral conversion (PRZZ Lemma 7.2)
# =============================================================================

def euler_maclaurin_exponent(L_count: int, d: int = 1) -> int:
    """
    Compute the (1-u) exponent for Euler-Maclaurin summation.

    From PRZZ line 2395:
    The sum Σ_{n≤N} (1⋆Λ₁^{k₁}⋆...)(n)/n × F × H converts to:
    ∫₀¹ (1-u)^{k + 1×k₁ + ... - 1} × F(1-(1-u)×logz/logx) × H(u) du

    For our setup with piece using L_count Λ-convolutions:
    - k = 1 (from the divisor function)
    - k₁ = L_count (for d=1)

    Exponent = 1 + L_count - 1 = L_count

    Args:
        L_count: Number of Λ convolutions in this piece
        d: Degree of Q (default 1)

    Returns:
        Exponent for (1-u) in the integral
    """
    # k = 1 (divisor function d_1)
    # For d=1: k₁ = L_count, no higher k's
    return 1 + L_count - 1  # = L_count


def euler_maclaurin_coefficient(L_count: int, d: int = 1) -> float:
    """
    Compute the coefficient for Euler-Maclaurin conversion.

    From PRZZ line 2395:
    Coefficient = 1^{k₁} × (2!)^{k₂} × ... × (logz)^{k+Σrk_r} / (k+Σrk_r-1)!

    For d=1 with L_count Λ's:
    - k = 1
    - k₁ = L_count
    - Numerator: 1^{L_count} × (logz)^{1+L_count}
    - Denominator: (1 + L_count - 1)! = L_count!

    We omit the (logz)^{...} factor as it appears in both numerator and
    denominator when combining with the F_d factors.

    Args:
        L_count: Number of Λ convolutions
        d: Degree of Q (default 1)

    Returns:
        Coefficient (excluding log powers)
    """
    exponent = 1 + L_count
    return 1.0 / factorial(L_count)


# =============================================================================
# Single-pair contribution computation
# =============================================================================

@dataclass
class PairContribution:
    """Result of computing a single pair (ℓ₁, ℓ₂) contribution."""
    ell1: int
    ell2: int
    value: float
    case_combination: str  # e.g., "BB", "BC", "CC"
    debug_info: Optional[Dict] = None


def compute_pair_contribution_oracle(
    P1: PolynomialEvaluator,
    P2: PolynomialEvaluator,
    Q: PolynomialEvaluator,
    ell1: int,
    ell2: int,
    theta: float,
    R: float,
    n_quad: int = 60,
    logT: float = 100.0,
    debug: bool = False
) -> PairContribution:
    """
    Compute the contribution from a single pair (ℓ₁, ℓ₂) using PRZZ oracle.

    This follows the PRZZ pipeline:
    1. Determine ω-cases for each piece
    2. Compute F_d factors
    3. Apply Euler-Maclaurin to convert n-sum to integral
    4. Evaluate the resulting double integral

    Args:
        P1: Polynomial for piece ℓ₁
        P2: Polynomial for piece ℓ₂
        Q: Q polynomial
        ell1: Piece index 1 (1, 2, or 3)
        ell2: Piece index 2 (1, 2, or 3)
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        logT: Asymptotic log(T) value
        debug: Whether to include debug info

    Returns:
        PairContribution with the computed value
    """
    logN = theta * logT
    alpha = -R / logT
    beta = -R / logT

    # Λ-convolution counts for each piece
    # Piece 1: μ → L_count = 0
    # Piece 2: μ⋆Λ → L_count = 1
    # Piece 3: μ⋆Λ⋆Λ → L_count = 2
    L1_count = ell1 - 1
    L2_count = ell2 - 1

    # For d=1, the Faà-di-Bruno index l₁ for a piece with L_count Λ's
    # ranges based on the derivative structure.
    # The dominant contribution comes from l₁ = L_count + 1 for the
    # "matched" case where all derivatives hit the Λ-powers.
    #
    # For simplicity, we start with the leading term: l₁ = L_count + 1
    # which gives ω = L_count.

    l1_index = L1_count + 1  # l₁ for first piece
    l2_index = L2_count + 1  # l₁ for second piece

    omega1 = compute_omega_d1(l1_index)
    omega2 = compute_omega_d1(l2_index)
    case1 = get_omega_case(omega1)
    case2 = get_omega_case(omega2)
    case_combo = case1 + case2

    # Set up quadrature for the u-integral (from Euler-Maclaurin)
    u_nodes, u_weights = gauss_legendre_01(n_quad)

    # The Euler-Maclaurin conversion gives us an integral over u ∈ [0,1]
    # where u corresponds to log(N/n)/logN.
    #
    # For piece 1 with L₁ Λ's, the (1-u) exponent is L₁.
    # For piece 2 with L₂ Λ's, the (1-u) exponent is L₂.
    #
    # When combining two pieces, the total exponent is L₁ + L₂.
    # But we also need to handle the cross-term structure carefully.

    exp1 = euler_maclaurin_exponent(L1_count)
    exp2 = euler_maclaurin_exponent(L2_count)
    coef1 = euler_maclaurin_coefficient(L1_count)
    coef2 = euler_maclaurin_coefficient(L2_count)

    # Actually, for the cross-term (ℓ₁, ℓ₂), both pieces share the same n
    # in the original sum. So we get a single u-integral with combined exponent.
    #
    # The structure is:
    # Σ_n (arithmetic)(n)/n × F_d(ℓ₁) × F_d(ℓ₂)
    #
    # With arithmetic function (1 ⋆ Λ^{L₁+L₂}).
    # The Euler-Maclaurin exponent is L₁ + L₂.

    total_L = L1_count + L2_count
    total_exp = euler_maclaurin_exponent(total_L)
    total_coef = euler_maclaurin_coefficient(total_L)

    # Compute the integrand at each u-node
    integrand = np.zeros(len(u_nodes))

    for i, u in enumerate(u_nodes):
        if u < 1e-12:
            # Handle u → 0 limit
            continue

        n_over_N = np.exp(-u * logN)

        # Compute F_d factors for both pieces
        Fd1 = compute_Fd(P1, u, alpha, logN, n_over_N, l1_index, n_quad)
        Fd2 = compute_Fd(P2, u, beta, logN, n_over_N, l2_index, n_quad)

        # Q(u) factor
        Q_u = float(Q(np.array([u]))[0])

        # The (1-u)^{total_exp} factor
        one_minus_u_factor = (1.0 - u) ** total_exp if total_exp > 0 else 1.0

        # Combine
        integrand[i] = Fd1.value * Fd2.value * Q_u * one_minus_u_factor

    # Integrate
    integral = total_coef * float(np.sum(u_weights * integrand))

    # The 1/(α+β) pole factor from PRZZ line 2366
    # At α = β = -R/logT, we have α+β = -2R/logT
    # This contributes a factor of logT/(2R) in the limit
    # But this is part of the T/L structure that PRZZ handles separately.
    #
    # For now, we include the leading-order structure without the T/L factor.

    # Symmetry factor: 2 for off-diagonal pairs
    sym_factor = 1 if ell1 == ell2 else 2

    value = sym_factor * integral

    debug_info = None
    if debug:
        debug_info = {
            "L1_count": L1_count,
            "L2_count": L2_count,
            "l1_index": l1_index,
            "l2_index": l2_index,
            "omega1": omega1,
            "omega2": omega2,
            "total_L": total_L,
            "total_exp": total_exp,
            "total_coef": total_coef,
            "raw_integral": integral,
        }

    return PairContribution(
        ell1=ell1,
        ell2=ell2,
        value=value,
        case_combination=case_combo,
        debug_info=debug_info
    )


# =============================================================================
# Full K=3 oracle evaluation
# =============================================================================

@dataclass
class OracleResult:
    """Result of full oracle evaluation."""
    c: float
    kappa: float
    per_pair: Dict[Tuple[int, int], float]
    total: float
    R: float
    theta: float
    debug_info: Optional[Dict] = None


def evaluate_oracle_k3_d1(
    P1, P2, P3, Q,
    theta: float,
    R: float,
    n_quad: int = 60,
    logT: float = 100.0,
    debug: bool = False
) -> OracleResult:
    """
    Full oracle evaluation for K=3, d=1.

    Computes contributions from all 6 pairs:
    (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)

    Args:
        P1, P2, P3: Polynomials for pieces 1, 2, 3
        Q: Q polynomial
        theta: θ = 4/7
        R: R parameter
        n_quad: Quadrature points
        logT: Asymptotic log(T)
        debug: Include debug info

    Returns:
        OracleResult with c, κ, and per-pair breakdown
    """
    # Wrap polynomials
    P1_eval = PolynomialEvaluator(P1)
    P2_eval = PolynomialEvaluator(P2)
    P3_eval = PolynomialEvaluator(P3)
    Q_eval = PolynomialEvaluator(Q)

    polys = {1: P1_eval, 2: P2_eval, 3: P3_eval}

    per_pair = {}
    total = 0.0

    # All pairs (ℓ₁, ℓ₂) with ℓ₁ ≤ ℓ₂
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    for ell1, ell2 in pairs:
        contrib = compute_pair_contribution_oracle(
            polys[ell1], polys[ell2], Q_eval,
            ell1, ell2,
            theta, R, n_quad, logT, debug
        )
        per_pair[(ell1, ell2)] = contrib.value
        total += contrib.value

        if debug:
            print(f"Pair ({ell1},{ell2}): case={contrib.case_combination}, value={contrib.value:.6f}")

    # c is the main-term constant
    c = total

    # κ = 1 - log(c)/R
    kappa = 1.0 - log(c) / R if c > 0 else float('nan')

    return OracleResult(
        c=c,
        kappa=kappa,
        per_pair=per_pair,
        total=total,
        R=R,
        theta=theta,
        debug_info={"logT": logT} if debug else None
    )


# =============================================================================
# Testing utilities
# =============================================================================

def compare_oracle_vs_dsl(
    oracle_result: OracleResult,
    dsl_result,  # EvaluationResult from evaluate.py
) -> Dict:
    """
    Compare oracle results against DSL evaluator.

    Returns dict with comparison metrics.
    """
    comparison = {
        "oracle_c": oracle_result.c,
        "dsl_c": dsl_result.total,
        "c_ratio": oracle_result.c / dsl_result.total if dsl_result.total != 0 else float('nan'),
        "c_diff": oracle_result.c - dsl_result.total,
        "c_rel_diff": (oracle_result.c - dsl_result.total) / dsl_result.total if dsl_result.total != 0 else float('nan'),
        "oracle_kappa": oracle_result.kappa,
        "dsl_kappa": dsl_result.kappa,
    }

    return comparison


if __name__ == "__main__":
    # Quick test with mock polynomials
    print("PRZZ Section 7 Oracle - Test Run")
    print("=" * 50)

    # Test ω computation
    for l1 in range(5):
        omega = compute_omega_d1(l1)
        case = get_omega_case(omega)
        print(f"l1={l1}: ω={omega}, Case {case}")

    print()

    # Test weight functions
    print("Weight functions:")
    for l1 in range(4):
        U = weight_U_d1(l1)
        V = weight_V_d1(l1)
        W = weight_W_d1(l1)
        print(f"l1={l1}: U={U}, V={V}, W={W}")
