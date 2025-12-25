"""
src/case_c_integral.py
Case C auxiliary a-integral implementation.

PRZZ TeX References:
- Lines 2336-2362: Case C derivation
- Lines 2364-2368: Product F_d x F_d structure
- Lines 2369-2384: Case C a-integral formula (esp. line 2374)
- Lines 2387-2388: Cross-term bookkeeping

The Case C auxiliary integral (TeX 2374):
    integral_0^1 (1-a)^i a^{omega-1} (N/n)^{-alpha*a} da

For d=1:
- P_1 (k=2): omega = k-2 = 0 -> Case B (no a-integral)
- P_2 (k=3): omega = k-2 = 1 -> Case C with a^{0} = 1
- P_3 (k=4): omega = k-2 = 2 -> Case C with a^{1} = a

Cross-term structure:
- B x B: No a-integrals (e.g., (1,1))
- B x C: One a-integral from C factor (e.g., (1,2), (1,3))
- C x C: Two a-integrals (product) (e.g., (2,2), (2,3), (3,3))
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any
import math


def case_c_weight(omega: int, a: np.ndarray, i: int = 0) -> np.ndarray:
    """
    Compute Case C weight function: (1-a)^i * a^{omega-1}

    For omega=1 (P_2): a^{0} = 1, so weight = (1-a)^i
    For omega=2 (P_3): a^{1} = a, so weight = (1-a)^i * a

    Args:
        omega: Case C parameter (omega = k-2 where k is PRZZ mollifier index)
        a: Grid of a-values in [0,1]
        i: Power of (1-a), typically 0 for leading term

    Returns:
        Weight array: (1-a)^i * a^{omega-1}
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got omega={omega}")

    # (1-a)^i factor
    one_minus_a_power = (1 - a) ** i if i > 0 else np.ones_like(a)

    # a^{omega-1} factor
    # For omega=1: a^0 = 1
    # For omega=2: a^1 = a
    a_power = a ** (omega - 1) if omega > 1 else np.ones_like(a)

    return one_minus_a_power * a_power


def case_c_exp_factor(
    a: np.ndarray,
    R: float,
    theta: float,
    x_sum: float = 0.0,
    y_sum: float = 0.0
) -> np.ndarray:
    """
    Compute the exponential factor from (N/n)^{-alpha*a} in Case C.

    At alpha = -R/L (where L = log T), and N = T^theta:
        (N/n)^{-alpha*a} -> exp(R * a * something)

    The exact form depends on how PRZZ handles the N/n ratio.
    For the leading-order term where n ~ N:
        (N/n)^{-alpha*a} ~ 1 (at alpha = -R/L, n = N)

    For subleading terms, there's coupling through theta*x_sum*a and theta*y_sum*a.

    This is a SIMPLIFIED model - the exact formula needs verification against PRZZ.

    Args:
        a: Grid of a-values in [0,1]
        R: Shift parameter
        theta: Mollifier exponent
        x_sum: Sum of x-variables at evaluation point
        y_sum: Sum of y-variables at evaluation point

    Returns:
        Exponential factor exp(R * theta * a * (x_sum or y_sum))
    """
    # Simplified model: the a-integral contributes an R-dependent factor
    # through the coupling theta * a * (x + y)
    # This is a HYPOTHESIS based on the structure at TeX 2369-2384

    # For the leading-order contribution at x=y=0, the factor is just 1
    # The R-dependence comes from the coupling with formal variables

    return np.exp(R * theta * a * (x_sum + y_sum))


def compute_case_c_normalization(omega: int, i: int = 0) -> float:
    """
    Compute the normalization integral for Case C (Beta function).

    integral_0^1 (1-a)^i * a^{omega-1} da = B(i+1, omega) = Gamma(i+1)*Gamma(omega)/Gamma(i+omega+1)

    For omega=1 (P_2): B(i+1, 1) = 1/(i+1)
    For omega=2 (P_3): B(i+1, 2) = 1/((i+1)(i+2))

    Args:
        omega: Case C parameter
        i: Power of (1-a)

    Returns:
        Beta function value B(i+1, omega)
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got omega={omega}")

    # B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    return math.gamma(i + 1) * math.gamma(omega) / math.gamma(i + omega + 1)


def get_omega_for_piece(piece_index: int) -> int:
    """
    Get omega value for a given piece index.

    PRZZ convention (TeX line 548): k starts at 2
    Our P_1 -> k=2 -> omega = k-2 = 0 (Case B)
    Our P_2 -> k=3 -> omega = k-2 = 1 (Case C)
    Our P_3 -> k=4 -> omega = k-2 = 2 (Case C)

    Args:
        piece_index: Our polynomial index (1, 2, or 3)

    Returns:
        omega value
    """
    przz_k = piece_index + 1  # Our index -> PRZZ k
    omega = przz_k - 2
    return omega


def get_case_for_piece(piece_index: int) -> str:
    """
    Get case classification for a given piece index.

    Args:
        piece_index: Our polynomial index (1, 2, or 3)

    Returns:
        "A", "B", or "C"
    """
    omega = get_omega_for_piece(piece_index)
    if omega == -1:
        return "A"
    elif omega == 0:
        return "B"
    else:
        return "C"


def get_pair_case_structure(ell1: int, ell2: int) -> Tuple[str, str, int]:
    """
    Get case structure for a pair (ell_1, ell_2).

    Args:
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)

    Returns:
        Tuple of (case1, case2, num_a_integrals)
    """
    case1 = get_case_for_piece(ell1)
    case2 = get_case_for_piece(ell2)

    # Count number of a-integrals needed
    num_a = (1 if case1 == "C" else 0) + (1 if case2 == "C" else 0)

    return case1, case2, num_a


def case_c_correction_factor_12(
    R: float,
    theta: float,
    n_quad: int = 60
) -> float:
    """
    Compute the Case C correction factor for (1,2) pair.

    The (1,2) pair is B x C, requiring ONE auxiliary a-integral.

    P_2 has omega = 1, so the weight is (1-a)^i * a^0 = (1-a)^i

    For the leading term (i=0), this is just integral_0^1 da = 1.
    But there's an R-dependent exponential factor from (N/n)^{-alpha*a}.

    This is a DIAGNOSTIC function to test if Case C correction
    improves R-sensitivity matching.

    Args:
        R: Shift parameter
        theta: Mollifier exponent
        n_quad: Number of quadrature points for a-integral

    Returns:
        Case C correction factor for (1,2)
    """
    from src.quadrature import gauss_legendre_01

    nodes, weights = gauss_legendre_01(n_quad)

    # For (1,2): B x C with omega_2 = 1
    omega = 1
    i = 0  # Leading term

    # Weight: (1-a)^0 * a^{1-1} = 1 (trivial for omega=1, i=0)
    weight = case_c_weight(omega, nodes, i)

    # The key question: what is the exponential factor?
    # From PRZZ TeX 2369-2384, there's (N/n)^{-alpha*a}
    # At alpha = -R/L, n ~ N, this gives exp(R * something * a)
    #
    # HYPOTHESIS: The missing factor is of the form integral_0^1 f(a, R) da
    # where f involves the R-dependence through the (N/n)^{-alpha*a} term.
    #
    # For a first test, let's try:
    # factor = integral_0^1 exp(R * theta * a) * (1-a)^i da

    # Test hypothesis: exp(R * theta * a) factor
    exp_factor = np.exp(R * theta * nodes)

    # Integrate
    integrand = weight * exp_factor
    correction = np.sum(weights * integrand)

    return correction


def diagnose_case_c_structure(verbose: bool = True) -> Dict[str, Any]:
    """
    Diagnostic function showing Case C structure for all K=3 pairs.

    Returns:
        Dictionary with case structure for each pair
    """
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    results = {}
    for ell1, ell2 in pairs:
        pair_key = f"{ell1}{ell2}"
        case1, case2, num_a = get_pair_case_structure(ell1, ell2)
        omega1 = get_omega_for_piece(ell1)
        omega2 = get_omega_for_piece(ell2)

        results[pair_key] = {
            "ell1": ell1,
            "ell2": ell2,
            "case1": case1,
            "case2": case2,
            "omega1": omega1,
            "omega2": omega2,
            "num_a_integrals": num_a,
            "structure": f"{case1}x{case2}",
        }

    if verbose:
        print("\n" + "=" * 60)
        print("CASE C STRUCTURE FOR K=3 PAIRS")
        print("=" * 60)
        print()
        print(f"{'Pair':<8} {'Cases':<8} {'omega1':>8} {'omega2':>8} {'#a-int':>8}")
        print("-" * 60)

        for pair_key, info in results.items():
            print(f"({info['ell1']},{info['ell2']})    "
                  f"{info['structure']:<8} "
                  f"{info['omega1']:>8} "
                  f"{info['omega2']:>8} "
                  f"{info['num_a_integrals']:>8}")

        print()
        print("Legend:")
        print("  B = Case B (omega=0): No auxiliary integral")
        print("  C = Case C (omega>0): Needs auxiliary a-integral")
        print()
        print("Missing a-integrals:")
        for pair_key, info in results.items():
            if info['num_a_integrals'] > 0:
                print(f"  ({info['ell1']},{info['ell2']}): {info['num_a_integrals']} a-integral(s) MISSING")

        print("=" * 60)

    return results


def test_case_c_r_dependence(
    R_values: list = None,
    theta: float = 4/7,
    n_quad: int = 60,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test R-dependence of Case C correction factor.

    This diagnostic tests whether the Case C auxiliary integral
    introduces R-dependent corrections that could explain the gap.

    Args:
        R_values: List of R values to test
        theta: Mollifier exponent
        n_quad: Quadrature points
        verbose: Print results

    Returns:
        Dictionary with R-dependence analysis
    """
    if R_values is None:
        R_values = [1.0, 1.1, 1.1167, 1.2, 1.3036, 1.4]

    results = []
    for R in R_values:
        factor = case_c_correction_factor_12(R, theta, n_quad)
        results.append({
            "R": R,
            "correction_factor": factor,
        })

    # Compute relative change from first to last R
    if len(results) >= 2:
        first = results[0]["correction_factor"]
        for r in results:
            r["relative_to_first"] = (r["correction_factor"] - first) / first * 100

    if verbose:
        print("\n" + "=" * 60)
        print("CASE C CORRECTION R-DEPENDENCE TEST")
        print("=" * 60)
        print()
        print(f"Hypothesis: integral_0^1 exp(R*theta*a) da")
        print(f"theta = {theta:.6f}")
        print()
        print(f"{'R':>10} | {'Factor':>15} | {'Rel. Change':>12}")
        print("-" * 45)

        for r in results:
            rel = r.get("relative_to_first", 0)
            print(f"{r['R']:>10.4f} | {r['correction_factor']:>15.10f} | {rel:>+10.2f}%")

        # Compare with our observed gap
        print()
        print("Observed R-sensitivity in (1,2) pair: +32.32%")
        print("This test shows whether Case C integral could explain it.")
        print("=" * 60)

    return {"r_scan": results}


if __name__ == "__main__":
    diagnose_case_c_structure(verbose=True)
    test_case_c_r_dependence(verbose=True)
