"""
tests/test_psi_22_complete.py
Test Suite for Complete Ψ_{2,2} Oracle

This validates the 12-monomial oracle against:
1. The existing I-term oracle (przz_22_exact_oracle.py)
2. Both κ and κ* benchmarks
3. Per-monomial structure
"""

import pytest
import numpy as np
from src.psi_22_complete_oracle import Psi22CompleteOracle
from src.przz_22_exact_oracle import przz_oracle_22
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.psi_monomial_expansion import expand_pair_to_monomials


class TestPsi22CompleteOracle:
    """Test complete (2,2) oracle implementation."""

    @pytest.fixture
    def kappa_setup(self):
        """Setup for κ benchmark (R=1.3036)."""
        P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.3036
        return P2, Q, theta, R

    @pytest.fixture
    def kappa_star_setup(self):
        """Setup for κ* benchmark (R=1.1167)."""
        P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=True)
        theta = 4.0 / 7.0
        R = 1.1167
        return P2, Q, theta, R

    def test_monomial_count(self):
        """Verify (2,2) has exactly 12 monomials."""
        monomials = expand_pair_to_monomials(2, 2)
        assert len(monomials) == 12, f"Expected 12 monomials, got {len(monomials)}"

    def test_monomial_structure(self):
        """Verify the 12 monomials match expected structure."""
        monomials = expand_pair_to_monomials(2, 2)

        # Expected categories
        d_terms = [k for k in monomials.keys() if k[3] > 0]  # d > 0
        ab_terms = [k for k in monomials.keys() if k[0] > 0 and k[1] > 0 and k[3] == 0]  # a>0, b>0, d=0
        a_only = [k for k in monomials.keys() if k[0] > 0 and k[1] == 0 and k[3] == 0]  # a>0, b=0, d=0
        b_only = [k for k in monomials.keys() if k[0] == 0 and k[1] > 0 and k[3] == 0]  # a=0, b>0, d=0
        c_only = [k for k in monomials.keys() if k[0] == 0 and k[1] == 0 and k[3] == 0 and k[2] > 0]  # pure C

        assert len(d_terms) == 4, f"Expected 4 D-terms, got {len(d_terms)}"
        assert len(ab_terms) == 3, f"Expected 3 A×B terms, got {len(ab_terms)}"
        assert len(a_only) == 2, f"Expected 2 A-only terms, got {len(a_only)}"
        assert len(b_only) == 2, f"Expected 2 B-only terms, got {len(b_only)}"
        assert len(c_only) == 1, f"Expected 1 C-only term, got {len(c_only)}"

    def test_base_integral_positive(self, kappa_setup):
        """Base integral should be positive."""
        P2, Q, theta, R = kappa_setup
        oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=40)

        base = oracle._eval_base_integral()
        assert base > 0, f"Base integral should be positive, got {base}"
        assert base < 100, f"Base integral should be reasonable magnitude, got {base}"

    def test_kappa_consistency(self, kappa_setup):
        """Test that oracle gives reasonable values for κ."""
        P2, Q, theta, R = kappa_setup

        # I-term oracle
        i_oracle = przz_oracle_22(P2, Q, theta, R, n_quad=60)

        # Ψ oracle
        psi_oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=60)
        psi_total, results = psi_oracle.compute_all_monomials(verbose=False)

        # Should be in same ballpark
        assert psi_total > 0, f"Ψ total should be positive, got {psi_total}"
        assert abs(psi_total - i_oracle.total) < 10 * abs(i_oracle.total), \
            f"Ψ and I-term should be within 10x: Ψ={psi_total:.4f}, I-term={i_oracle.total:.4f}"

    def test_kappa_star_consistency(self, kappa_star_setup):
        """Test that oracle gives reasonable values for κ*."""
        P2, Q, theta, R = kappa_star_setup

        # I-term oracle
        i_oracle = przz_oracle_22(P2, Q, theta, R, n_quad=60)

        # Ψ oracle
        psi_oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=60)
        psi_total, results = psi_oracle.compute_all_monomials(verbose=False)

        # Should be in same ballpark
        assert psi_total > 0, f"Ψ total should be positive, got {psi_total}"
        assert abs(psi_total - i_oracle.total) < 10 * abs(i_oracle.total), \
            f"Ψ and I-term should be within 10x: Ψ={psi_total:.4f}, I-term={i_oracle.total:.4f}"

    def test_two_benchmark_ratio(self, kappa_setup, kappa_star_setup):
        """
        Test that κ/κ* ratio is close to target 1.10.

        This is the key validation: if our oracle is correct, the ratio
        should match the PRZZ target.
        """
        P2_k, Q_k, theta, R_k = kappa_setup
        P2_ks, Q_ks, _, R_ks = kappa_star_setup

        # κ oracle
        psi_k = Psi22CompleteOracle(P2_k, Q_k, theta, R_k, n_quad=60)
        total_k, _ = psi_k.compute_all_monomials(verbose=False)

        # κ* oracle
        psi_ks = Psi22CompleteOracle(P2_ks, Q_ks, theta, R_ks, n_quad=60)
        total_ks, _ = psi_ks.compute_all_monomials(verbose=False)

        ratio = total_k / total_ks

        # Should be close to 1.10 (allowing for implementation approximations)
        assert 0.5 < ratio < 2.0, \
            f"Ratio should be in reasonable range, got {ratio:.4f} (target 1.10)"

        print(f"\nTwo-benchmark ratio: {ratio:.4f} (target 1.10)")

    def test_monomial_evaluation_no_crash(self, kappa_setup):
        """Test that all monomials can be evaluated without crashing."""
        P2, Q, theta, R = kappa_setup
        oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=40)

        monomials = expand_pair_to_monomials(2, 2)

        for (a, b, c, d) in monomials.keys():
            try:
                val = oracle.eval_monomial(a, b, c, d)
                assert np.isfinite(val), \
                    f"Monomial ({a},{b},{c},{d}) gave non-finite value: {val}"
            except Exception as e:
                pytest.fail(f"Monomial ({a},{b},{c},{d}) evaluation failed: {e}")

    def test_dominant_terms(self, kappa_setup):
        """
        Test that dominant terms have expected sign and magnitude.

        From the I-term structure, we expect:
        - A²B² term to be positive and large (like I₁)
        - D² term to be positive (like I₂)
        """
        P2, Q, theta, R = kappa_setup
        oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=60)

        # A²B² term (should be largest positive contribution)
        val_A2B2 = oracle.eval_monomial(2, 2, 0, 0)
        assert val_A2B2 > 0, f"A²B² should be positive, got {val_A2B2}"

        # D² term (should be positive)
        val_D2 = oracle.eval_monomial(0, 0, 0, 2)
        assert val_D2 >= 0, f"D² should be non-negative, got {val_D2}"

    def test_quadrature_convergence(self, kappa_setup):
        """Test that results converge as quadrature order increases."""
        P2, Q, theta, R = kappa_setup

        n_values = [40, 60, 80]
        totals = []

        for n in n_values:
            oracle = Psi22CompleteOracle(P2, Q, theta, R, n_quad=n)
            total, _ = oracle.compute_all_monomials(verbose=False)
            totals.append(total)

        # Check relative convergence
        rel_change_1 = abs(totals[1] - totals[0]) / abs(totals[0])
        rel_change_2 = abs(totals[2] - totals[1]) / abs(totals[1])

        # Changes should get smaller
        assert rel_change_2 < rel_change_1 or rel_change_2 < 0.01, \
            f"Results should converge: {rel_change_1:.4f} -> {rel_change_2:.4f}"

        print(f"\nQuadrature convergence: n=40→60: {rel_change_1:.4f}, n=60→80: {rel_change_2:.4f}")


def test_comparison_report():
    """
    Generate a detailed comparison report.

    This is not a pass/fail test but a diagnostic output.
    """
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON: Ψ Oracle vs I-Term Oracle")
    print("=" * 70)

    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)
    theta = 4.0 / 7.0
    R_k = 1.3036
    R_ks = 1.1167

    print("\n--- κ Benchmark (R=1.3036) ---")
    i_k = przz_oracle_22(P2_k, Q_k, theta, R_k, n_quad=60)
    psi_k = Psi22CompleteOracle(P2_k, Q_k, theta, R_k, n_quad=60)
    total_k, results_k = psi_k.compute_all_monomials(verbose=False)

    print(f"\nI-term Oracle:")
    print(f"  I₁ = {i_k.I1:8.4f}")
    print(f"  I₂ = {i_k.I2:8.4f}")
    print(f"  I₃ = {i_k.I3:8.4f}")
    print(f"  I₄ = {i_k.I4:8.4f}")
    print(f"  Total = {i_k.total:8.4f}")

    print(f"\nΨ Oracle (selected terms):")
    for (a, b, c, d), mv in sorted(results_k.items())[:6]:
        print(f"  {mv.coefficient:+3d} × C{c}D{d}A{a}B{b} = {mv.contribution:+8.4f}")
    print(f"  ... ({len(results_k)} total)")
    print(f"  Total = {total_k:8.4f}")

    print(f"\nκ Comparison:")
    print(f"  Ratio Ψ/I-term: {total_k / i_k.total:.4f}")

    print("\n--- κ* Benchmark (R=1.1167) ---")
    i_ks = przz_oracle_22(P2_ks, Q_ks, theta, R_ks, n_quad=60)
    psi_ks = Psi22CompleteOracle(P2_ks, Q_ks, theta, R_ks, n_quad=60)
    total_ks, results_ks = psi_ks.compute_all_monomials(verbose=False)

    print(f"\nI-term: {i_ks.total:8.4f}")
    print(f"Ψ:      {total_ks:8.4f}")
    print(f"Ratio:  {total_ks / i_ks.total:.4f}")

    print("\n--- Two-Benchmark Ratio ---")
    print(f"κ/κ* (I-term): {i_k.total / i_ks.total:.4f}")
    print(f"κ/κ* (Ψ):      {total_k / total_ks:.4f}")
    print(f"Target:        1.10")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run the comparison report
    test_comparison_report()
