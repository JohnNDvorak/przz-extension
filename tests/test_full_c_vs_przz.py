"""
tests/test_full_c_vs_przz.py
CRITICAL: Compare full c computation (all pairs) with PRZZ targets

The +5 gate uses compute_m1_with_mirror_assembly which computes
a simplified (1,1) structure. The FULL c computation uses
compute_c_paper_with_mirror which sums over ALL (ℓ₁,ℓ₂) pairs.

If the full c matches PRZZ targets, then:
1. Our formula is correct from first principles
2. Optimization would be valid
"""

import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def get_polynomial_dict(benchmark: str):
    """Get polynomial dictionary for evaluation."""
    if benchmark == "kappa":
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    else:
        P1, P2, P3, Q = load_przz_polynomials_kappa_star()

    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


def test_full_c_computation():
    """Test the full c computation against PRZZ targets."""
    print("\n" + "=" * 70)
    print("FULL c COMPUTATION vs PRZZ TARGETS")
    print("=" * 70)

    targets = {
        "kappa": {"R": 1.3036, "c_target": 2.137454406, "kappa_target": 0.417293962},
        "kappa_star": {"R": 1.1167, "c_target": 1.938, "kappa_target": 0.410},
    }

    for benchmark in ["kappa", "kappa_star"]:
        target = targets[benchmark]
        R = target["R"]
        polynomials = get_polynomial_dict(benchmark)

        print(f"\n{benchmark.upper()} (R={R}):")
        print("-" * 50)

        try:
            result = compute_c_paper_with_mirror(
                theta=4.0/7.0,
                R=R,
                n=60,  # quadrature points
                polynomials=polynomials,
                pair_mode="hybrid",
                use_factorial_normalization=True,
                mode="main",
                K=3,
            )

            c_computed = result.total
            c_target = target["c_target"]
            kappa_target = target["kappa_target"]

            # Compute kappa from c
            kappa_computed = 1 - np.log(c_computed) / R

            c_error = (c_computed - c_target) / c_target * 100
            kappa_error = (kappa_computed - kappa_target) * 100  # percentage points

            print(f"  c computed: {c_computed:.10f}")
            print(f"  c target:   {c_target:.10f}")
            print(f"  c error:    {c_error:+.4f}%")
            print()
            print(f"  κ computed: {kappa_computed:.10f}")
            print(f"  κ target:   {kappa_target:.10f}")
            print(f"  κ error:    {kappa_error:+.6f} percentage points")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)


def test_c_ratio():
    """Test if the c ratio between benchmarks is correct."""
    print("\n" + "=" * 70)
    print("C RATIO TEST (full computation)")
    print("=" * 70)

    results = {}
    for benchmark in ["kappa", "kappa_star"]:
        R = 1.3036 if benchmark == "kappa" else 1.1167
        polynomials = get_polynomial_dict(benchmark)

        try:
            result = compute_c_paper_with_mirror(
                theta=4.0/7.0,
                R=R,
                n=60,
                polynomials=polynomials,
                pair_mode="hybrid",
                use_factorial_normalization=True,
                mode="main",
                K=3,
            )
            results[benchmark] = result.total
        except Exception as e:
            print(f"Error for {benchmark}: {e}")
            results[benchmark] = None

    if results["kappa"] and results["kappa_star"]:
        our_ratio = results["kappa"] / results["kappa_star"]
        przz_ratio = 2.137454406 / 1.938

        print(f"\nPRZZ c ratio: c_κ/c_κ* = {przz_ratio:.6f}")
        print(f"Our c ratio:  c_κ/c_κ* = {our_ratio:.6f}")
        print(f"Ratio of ratios: {our_ratio/przz_ratio:.6f}")

        if abs(our_ratio/przz_ratio - 1) < 0.05:
            print("\n✓ VALID: Our c ratio matches PRZZ within 5%")
            print("  This means optimization results would be meaningful!")
        else:
            print(f"\n⚠ Our c ratio differs by {(our_ratio/przz_ratio - 1)*100:.1f}%")


if __name__ == "__main__":
    test_full_c_computation()
    test_c_ratio()
