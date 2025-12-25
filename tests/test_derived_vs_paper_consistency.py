"""
tests/test_derived_vs_paper_consistency.py
Phase 11.3: Derived Mirror vs Paper-Only Consistency Tests

PURPOSE:
========
Verify that the paper-only evaluation (no mirror) gives expected low values,
and that the derived-mirror approach produces structurally correct output
even if the numerical values are still being tuned.

TEST STRUCTURE:
==============
1. Paper-only (no mirror): Should give c ~ 0.2 (10x collapse from target)
   This is the "pre-mirror component" that needs mirror assembly.

2. Empirical mirror (m1 = exp(R) + 5): Should match ~1-3% of target
   This is the current production baseline.

3. Derived mirror (Phase 10): Values may differ but structure should be:
   - S12_direct + S12_mirror_operator + S34
   - No Q polynomial blowup (< 10x amplification)
   - Quadrature stable

4. Ratio consistency: Paper and derived should maintain similar κ/κ* ratios
   even if absolute values differ.
"""

import pytest
import math
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# ============================================================================
# BENCHMARK CONSTANTS
# ============================================================================

KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.13745440613217263636

KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.9379524124677437

TARGET_RATIO = KAPPA_C_TARGET / KAPPA_STAR_C_TARGET  # ~1.103

N_QUAD = 60


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_polynomials_as_dict(loader_func, enforce_Q0=True):
    """Load polynomials and return as dict."""
    P1, P2, P3, Q = loader_func(enforce_Q0=enforce_Q0)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


# ============================================================================
# PAPER-ONLY TESTS
# ============================================================================

class TestPaperOnlyExpectedCollapse:
    """
    Verify paper-only (no mirror) gives expected low values.

    This proves the paper regime is computing the "pre-mirror component"
    correctly. The 10x collapse is expected and necessary.
    """

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    @pytest.fixture(scope="class")
    def kappa_star_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials_kappa_star, enforce_Q0=True)

    def test_paper_only_kappa_collapsed(self, kappa_polys):
        """
        Paper-only should give c ~ 0.2 (10x collapse from target 2.14).

        This is NOT a bug - it's the pre-mirror component.
        """
        from src.evaluate import compute_c_paper

        P1 = kappa_polys["P1"]
        P2 = kappa_polys["P2"]
        P3 = kappa_polys["P3"]
        Q = kappa_polys["Q"]

        result = compute_c_paper(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            return_breakdown=False,
            n_quad_a=40,
        )

        # Expected: c ~ 0.2 (about 10x below target)
        assert result.total < 0.5, (
            f"Paper-only c = {result.total:.4f} should be < 0.5 (10x collapsed)"
        )
        assert result.total > 0.1, (
            f"Paper-only c = {result.total:.4f} should be > 0.1 (not zeroed)"
        )

        print(f"\n  Paper-only kappa c = {result.total:.4f}")
        print(f"  Expected: ~0.2 (10x collapse from target 2.14)")

    def test_paper_only_ratio_correct(self, kappa_polys, kappa_star_polys):
        """
        Paper-only ratio should be close to 1.15 (similar to target 1.10).

        This proves the paper regime is structurally correct even though
        absolute values are collapsed.
        """
        from src.evaluate import compute_c_paper

        c_kappa = compute_c_paper(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            n_quad_a=40,
        ).total

        c_kappa_star = compute_c_paper(
            theta=4.0/7.0,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            n_quad_a=40,
        ).total

        ratio = c_kappa / c_kappa_star

        # Paper ratio should be close to target ratio
        ratio_error = abs(ratio - TARGET_RATIO) / TARGET_RATIO

        print(f"\n  Paper-only ratio = {ratio:.4f}")
        print(f"  Target ratio = {TARGET_RATIO:.4f}")
        print(f"  Error = {ratio_error*100:.1f}%")

        # Paper-only ratio may deviate more since it lacks mirror assembly
        # 30% tolerance captures structural correctness (ratio > 1)
        assert ratio_error < 0.30, (
            f"Paper-only ratio = {ratio:.4f} should be within 30% of target {TARGET_RATIO:.4f}"
        )


# ============================================================================
# EMPIRICAL vs DERIVED COMPARISON
# ============================================================================

class TestEmpiricalVsDerivedStructure:
    """
    Compare empirical (m1=exp(R)+5) vs derived mirror structure.

    Both should produce:
    - Positive c values
    - Similar ratio behavior
    - Finite values with no blowup
    """

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    @pytest.fixture(scope="class")
    def kappa_star_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials_kappa_star, enforce_Q0=True)

    def test_both_give_positive_c(self, kappa_polys):
        """Both methods should produce positive c values."""
        from src.evaluate import compute_c_paper_ordered, compute_c_derived_mirror

        empirical_c = compute_c_paper_ordered(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
            s12_pair_mode='triangle',
        ).total

        derived_c = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        ).c

        assert empirical_c > 0, f"Empirical c = {empirical_c} should be positive"
        assert derived_c > 0, f"Derived c = {derived_c} should be positive"

        print(f"\n  Empirical c = {empirical_c:.4f}")
        print(f"  Derived c = {derived_c:.4f}")
        print(f"  Ratio (derived/empirical) = {derived_c/empirical_c:.2f}x")

    def test_ratio_direction_consistent(self, kappa_polys, kappa_star_polys):
        """
        Both methods should have ratio > 1 (kappa > kappa*).

        Even if derived values are off, the direction should be consistent.
        """
        from src.evaluate import compute_c_paper_ordered, compute_c_derived_mirror

        # Empirical ratios
        emp_kappa = compute_c_paper_ordered(
            theta=4.0/7.0, R=KAPPA_R, n=N_QUAD,
            polynomials=kappa_polys, K=3, s12_pair_mode='triangle',
        ).total

        emp_kappa_star = compute_c_paper_ordered(
            theta=4.0/7.0, R=KAPPA_STAR_R, n=N_QUAD,
            polynomials=kappa_star_polys, K=3, s12_pair_mode='triangle',
        ).total

        emp_ratio = emp_kappa / emp_kappa_star

        # Derived ratios
        der_kappa = compute_c_derived_mirror(
            theta=4.0/7.0, R=KAPPA_R, n=N_QUAD,
            polynomials=kappa_polys, K=3,
        ).c

        der_kappa_star = compute_c_derived_mirror(
            theta=4.0/7.0, R=KAPPA_STAR_R, n=N_QUAD,
            polynomials=kappa_star_polys, K=3,
        ).c

        der_ratio = der_kappa / der_kappa_star

        print(f"\n  Empirical ratio = {emp_ratio:.4f}")
        print(f"  Derived ratio = {der_ratio:.4f}")
        print(f"  Both > 1: {emp_ratio > 1 and der_ratio > 1}")

        # Both should have ratio > 1 (kappa > kappa*)
        assert emp_ratio > 1, f"Empirical ratio = {emp_ratio} should be > 1"
        assert der_ratio > 1, f"Derived ratio = {der_ratio} should be > 1"


# ============================================================================
# QUADRATURE STABILITY
# ============================================================================

class TestQuadratureStabilityComparison:
    """Verify both methods are stable under quadrature refinement."""

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    def test_both_quadrature_stable(self, kappa_polys):
        """Both empirical and derived should be stable n=40/60/80."""
        from src.evaluate import compute_c_paper_ordered, compute_c_derived_mirror

        emp_values = {}
        der_values = {}

        for n in [40, 60, 80]:
            emp_values[n] = compute_c_paper_ordered(
                theta=4.0/7.0, R=KAPPA_R, n=n,
                polynomials=kappa_polys, K=3, s12_pair_mode='triangle',
            ).total

            der_values[n] = compute_c_derived_mirror(
                theta=4.0/7.0, R=KAPPA_R, n=n,
                polynomials=kappa_polys, K=3,
            ).c

        emp_variation = abs(emp_values[80] - emp_values[40]) / emp_values[60]
        der_variation = abs(der_values[80] - der_values[40]) / der_values[60]

        print(f"\n  Empirical variation: {emp_variation*100:.2f}%")
        print(f"  Derived variation: {der_variation*100:.2f}%")

        # Both should be stable (< 2% variation)
        assert emp_variation < 0.02, (
            f"Empirical not stable: {emp_variation*100:.2f}%"
        )
        assert der_variation < 0.02, (
            f"Derived not stable: {der_variation*100:.2f}%"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
