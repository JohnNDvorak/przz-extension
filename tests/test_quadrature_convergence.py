"""
tests/test_quadrature_convergence.py
Check if increasing quadrature points reduces the c error
"""

import numpy as np
from src.evaluate import compute_c_paper_with_mirror
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def test_quadrature_convergence():
    """Test if c converges to PRZZ target with more quadrature points."""
    print("\n" + "=" * 70)
    print("QUADRATURE CONVERGENCE TEST")
    print("=" * 70)

    targets = {
        "kappa": {"R": 1.3036, "c_target": 2.137454406, "kappa_target": 0.417293962},
        "kappa_star": {"R": 1.1167, "c_target": 1.9379524081, "kappa_target": 0.407511457},
    }

    for benchmark in ["kappa", "kappa_star"]:
        target = targets[benchmark]
        R = target["R"]

        if benchmark == "kappa":
            P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        else:
            P1, P2, P3, Q = load_przz_polynomials_kappa_star()

        polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

        print(f"\n{benchmark.upper()} (R={R}, c_target={target['c_target']:.6f}):")
        print("-" * 60)
        print(f"{'n':<8} {'c':<15} {'error %':<12} {'κ':<15} {'κ error pp':<12}")
        print("-" * 60)

        for n in [40, 60, 80, 100, 120]:
            try:
                result = compute_c_paper_with_mirror(
                    theta=4.0/7.0,
                    R=R,
                    n=n,
                    polynomials=polynomials,
                    pair_mode="hybrid",
                    use_factorial_normalization=True,
                    mode="main",
                    K=3,
                )

                c = result.total
                kappa = 1 - np.log(c) / R

                c_error = (c - target['c_target']) / target['c_target'] * 100
                kappa_error = (kappa - target['kappa_target']) * 100

                print(f"{n:<8} {c:<15.10f} {c_error:+11.4f}% {kappa:<15.10f} {kappa_error:+11.4f}")

            except Exception as e:
                print(f"{n:<8} Error: {e}")

        print("-" * 60)


if __name__ == "__main__":
    test_quadrature_convergence()
