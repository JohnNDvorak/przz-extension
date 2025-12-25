"""
src/verify_przz_polynomials.py
Verify PRZZ polynomial transcription against exact TeX values.

This script compares our stored polynomials against the EXACT values
from PRZZ TeX lines 2567-2598.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PRZZPolynomials:
    """Exact PRZZ polynomials from TeX."""
    P1_coeffs: List[float]  # Standard powers: [c0, c1, c2, ...]
    P2_coeffs: List[float]
    P3_coeffs: List[float]
    Q_coeffs: List[float]
    R: float
    target_c: float
    target_kappa: float

# ============================================================================
# EXACT PRZZ POLYNOMIALS FROM TEX (Section 8, lines 2567-2586)
# ============================================================================

# Benchmark A: κ (R=1.3036) - CORRECT polynomials with DEGREE-5 Q
# These are computed from the factorized PRZZ forms:
#   P1(x) = x + 0.261076 x(1-x) - 1.071007 x(1-x)^2 - 0.236840 x(1-x)^3 + 0.260233 x(1-x)^4
#   P2(x) = 1.048274 x + 1.319912 x^2 - 0.940058 x^3
#   P3(x) = 0.522811 x - 0.686510 x^2 - 0.049923 x^3
#   Q(x) = 0.490464 + 0.636851(1-2x) - 0.159327(1-2x)^3 + 0.032011(1-2x)^5

def _expand_p1_kappa():
    """Expand P1 for κ benchmark from factorized form."""
    tilde = [0.261076, -1.071007, -0.236840, 0.260233]
    return expand_factorized_p1(tilde)

def _expand_q_kappa():
    """Expand Q for κ benchmark from (1-2x)^k basis."""
    # Q = 0.490464 + 0.636851(1-2x) - 0.159327(1-2x)^3 + 0.032011(1-2x)^5
    from math import comb
    result = np.zeros(6)  # degree 5
    terms = [(0, 0.490464), (1, 0.636851), (3, -0.159327), (5, 0.032011)]
    for k, c_k in terms:
        for j in range(k + 1):
            result[j] += c_k * comb(k, j) * ((-2) ** j)
    return result

# Note: These will be computed at module load time after the functions are defined
# For now, use placeholders that will be populated in main()
PRZZ_KAPPA = PRZZPolynomials(
    P1_coeffs=[],  # Will be computed dynamically
    P2_coeffs=[0.0, 1.048274, 1.319912, -0.940058],
    P3_coeffs=[0.0, 0.522811, -0.686510, -0.049923],
    Q_coeffs=[],   # Will be computed dynamically (degree-5)
    R=1.3036,
    target_c=2.137,
    target_kappa=0.417293962
)

# Benchmark B: κ* (R=1.1167)
# Expanded into standard powers of x
PRZZ_KAPPA_STAR = PRZZPolynomials(
    # P1(x) = -0.101832x^5 + 0.410521x^4 - 1.278570x^3 + 1.680202x^2 + 0.289679x
    P1_coeffs=[0.0, 0.289679, 1.680202, -1.278570, 0.410521, -0.101832],
    # P2(x) = -0.097446x^2 + 1.049837x
    P2_coeffs=[0.0, 1.049837, -0.097446],
    # P3(x) = -0.156465x^2 + 0.035113x
    P3_coeffs=[0.0, 0.035113, -0.156465],
    # Q(x) = 1 - 1.032446x
    Q_coeffs=[1.0, -1.032446],
    R=1.1167,
    target_c=1.939,
    target_kappa=0.407511457
)

def expand_factorized_p1(tilde_coeffs: List[float]) -> np.ndarray:
    """
    Expand P1(x) = x + sum_k c_k * x(1-x)^k to standard form.

    tilde_coeffs = [c_1, c_2, c_3, ...] for x(1-x)^1, x(1-x)^2, etc.
    """
    # Start with x term
    result = np.array([0.0, 1.0])  # x = 0 + 1*x

    for k, c_k in enumerate(tilde_coeffs, start=1):
        # x(1-x)^k = x * (1-x)^k
        # (1-x)^k expanded
        one_minus_x_k = np.zeros(k + 1)
        for j in range(k + 1):
            # (1-x)^k = sum_j C(k,j) * (-x)^j = sum_j C(k,j) * (-1)^j * x^j
            from math import comb
            one_minus_x_k[j] = comb(k, j) * ((-1) ** j)

        # Multiply by x (shift by 1)
        x_times = np.zeros(len(one_minus_x_k) + 1)
        x_times[1:] = one_minus_x_k

        # Add to result with coefficient c_k
        if len(x_times) > len(result):
            result = np.pad(result, (0, len(x_times) - len(result)))
        result[:len(x_times)] += c_k * x_times

    return result

def expand_przz_q(coeffs_in_basis: Dict) -> np.ndarray:
    """
    Expand Q(x) from (1-2x)^k basis to standard powers.

    Q(x) = sum_k c_k * (1-2x)^k
    """
    result = np.array([0.0])

    for k_str, c_k in coeffs_in_basis.items():
        if k_str == "constant":
            k = 0
        elif k_str.startswith("power_"):
            k = int(k_str.split("_")[1])
        else:
            continue

        # (1-2x)^k expanded
        term = np.zeros(k + 1)
        for j in range(k + 1):
            from math import comb
            term[j] = comb(k, j) * ((-2) ** j)

        if len(term) > len(result):
            result = np.pad(result, (0, len(term) - len(result)))
        result[:len(term)] += c_k * term

    return result

def verify_boundary_conditions(name: str, P_coeffs: List[float],
                               P0_expected: float, P1_expected: float = None):
    """Verify P(0) and optionally P(1)."""
    P = np.array(P_coeffs)
    P0 = P[0] if len(P) > 0 else 0.0
    P1 = sum(P)

    print(f"  {name}(0) = {P0:.6f} (expected: {P0_expected})")
    if P1_expected is not None:
        print(f"  {name}(1) = {P1:.6f} (expected: {P1_expected})")

    ok = abs(P0 - P0_expected) < 1e-6
    if P1_expected is not None:
        ok = ok and abs(P1 - P1_expected) < 1e-6
    return ok

def compare_polynomials(name: str, our_coeffs: np.ndarray, przz_coeffs: List[float]):
    """Compare our polynomial coefficients against PRZZ exact values."""
    przz = np.array(przz_coeffs)

    # Pad to same length
    max_len = max(len(our_coeffs), len(przz))
    ours_padded = np.pad(our_coeffs, (0, max_len - len(our_coeffs)))
    przz_padded = np.pad(przz, (0, max_len - len(przz)))

    print(f"\n{name}:")
    print(f"  Degree: ours={len(our_coeffs)-1}, PRZZ={len(przz)-1}")
    print(f"  Our coeffs:  {ours_padded}")
    print(f"  PRZZ coeffs: {przz_padded}")

    diff = np.abs(ours_padded - przz_padded)
    max_diff = np.max(diff)
    print(f"  Max difference: {max_diff:.6e}")

    if max_diff < 1e-4:
        print(f"  STATUS: MATCH ✓")
        return True
    else:
        print(f"  STATUS: MISMATCH ✗")
        for i, (o, p, d) in enumerate(zip(ours_padded, przz_padded, diff)):
            if d > 1e-6:
                print(f"    x^{i}: ours={o:.6f}, PRZZ={p:.6f}, diff={d:.6f}")
        return False

def load_our_polynomials():
    """Load our stored polynomials from JSON."""
    import json

    # Load kappa benchmark
    with open("data/przz_parameters.json") as f:
        kappa_data = json.load(f)

    # Load kappa* benchmark
    with open("data/przz_parameters_kappa_star.json") as f:
        kappa_star_data = json.load(f)

    return kappa_data, kappa_star_data

def main():
    print("=" * 70)
    print("PRZZ POLYNOMIAL TRANSCRIPTION VERIFICATION")
    print("=" * 70)

    # Load our stored polynomials
    kappa_data, kappa_star_data = load_our_polynomials()

    # Compute reference P1 and Q for κ benchmark dynamically
    ref_P1_kappa = _expand_p1_kappa()
    ref_Q_kappa = _expand_q_kappa()

    # =========================================================================
    # BENCHMARK A: κ (R=1.3036)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK A: κ (R=1.3036)")
    print("=" * 70)

    # Expand our P1 from factorized form
    our_P1 = expand_factorized_p1(kappa_data["polynomials"]["P1"]["tilde_coeffs"])
    our_P2 = np.array(kappa_data["polynomials"]["P2"]["coeffs"])
    our_P3 = np.array(kappa_data["polynomials"]["P3"]["coeffs"])
    our_Q = expand_przz_q(kappa_data["polynomials"]["Q"]["coeffs_in_basis"])

    print("\n--- Boundary Conditions ---")
    verify_boundary_conditions("P1", our_P1.tolist(), 0.0, 1.0)
    verify_boundary_conditions("P2", our_P2.tolist(), 0.0)
    verify_boundary_conditions("P3", our_P3.tolist(), 0.0)
    verify_boundary_conditions("Q", our_Q.tolist(), 1.0)

    print("\n--- Coefficient Comparison ---")
    # Use dynamically computed reference values
    match_P1 = compare_polynomials("P1", our_P1, ref_P1_kappa.tolist())
    match_P2 = compare_polynomials("P2", our_P2, PRZZ_KAPPA.P2_coeffs)
    match_P3 = compare_polynomials("P3", our_P3, PRZZ_KAPPA.P3_coeffs)
    match_Q = compare_polynomials("Q", our_Q, ref_Q_kappa.tolist())

    kappa_ok = match_P1 and match_P2 and match_P3 and match_Q

    # =========================================================================
    # BENCHMARK B: κ* (R=1.1167)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK B: κ* (R=1.1167)")
    print("=" * 70)

    # Expand our P1 from factorized form
    our_P1_star = expand_factorized_p1(kappa_star_data["polynomials"]["P1"]["tilde_coeffs"])
    our_P2_star = np.array(kappa_star_data["polynomials"]["P2"]["coeffs"])
    our_P3_star = np.array(kappa_star_data["polynomials"]["P3"]["coeffs"])

    # Q for kappa* - handle linear case
    Q_basis = kappa_star_data["polynomials"]["Q"]["coeffs_in_basis"]
    our_Q_star = expand_przz_q(Q_basis)

    print("\n--- Boundary Conditions ---")
    verify_boundary_conditions("P1", our_P1_star.tolist(), 0.0, 1.0)
    verify_boundary_conditions("P2", our_P2_star.tolist(), 0.0)
    verify_boundary_conditions("P3", our_P3_star.tolist(), 0.0)
    verify_boundary_conditions("Q", our_Q_star.tolist(), 1.0)

    print("\n--- Coefficient Comparison ---")
    match_P1_star = compare_polynomials("P1", our_P1_star, PRZZ_KAPPA_STAR.P1_coeffs)
    match_P2_star = compare_polynomials("P2", our_P2_star, PRZZ_KAPPA_STAR.P2_coeffs)
    match_P3_star = compare_polynomials("P3", our_P3_star, PRZZ_KAPPA_STAR.P3_coeffs)
    match_Q_star = compare_polynomials("Q", our_Q_star, PRZZ_KAPPA_STAR.Q_coeffs)

    kappa_star_ok = match_P1_star and match_P2_star and match_P3_star and match_Q_star

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Benchmark κ (R=1.3036):   {'PASS ✓' if kappa_ok else 'FAIL ✗'}")
    print(f"Benchmark κ* (R=1.1167):  {'PASS ✓' if kappa_star_ok else 'FAIL ✗'}")

    if not kappa_ok or not kappa_star_ok:
        print("\n⚠️  TRANSCRIPTION ERRORS DETECTED!")
        print("The stored polynomials do NOT match PRZZ TeX values.")
        print("This is likely the source of the two-benchmark gate failure.")

if __name__ == "__main__":
    main()
