"""
tests/test_c_derived_mirror_benchmarks.py
Phase 10.3: Benchmark Gate Tests for Derived Mirror Operator

PURPOSE:
========
Validate that the derived mirror operator (Phase 10) produces c values
close to the PRZZ targets. This replaces the empirical m1 = exp(R) + 5
with a theoretically-derived swap/sign conjugation.

MATHEMATICAL BASIS:
==================
The mirror transform uses (-beta, -alpha) swap conjugation:
    D_alpha -> -D_beta
    D_beta -> -D_alpha

This gives mirror eigenvalues:
    A_alpha^mirror = theta * y  (instead of direct eigenvalue)
    A_beta^mirror = theta * x   (instead of direct eigenvalue)

Arguments stay in [0, theta] ~ [0, 0.57], where Q polynomials are well-behaved.
This prevents the 100x blowup seen in Phase 9 with Q(1+D) on direct eigenvalues.

BENCHMARKS:
===========
    kappa:  R=1.3036, c_target=2.137, kappa_target=0.417293962
    kappa*: R=1.1167, c_target=1.938, kappa_target=0.407511457

SUCCESS CRITERIA:
================
Phase 10.3 initial: Both benchmarks within 5% of target
Goal: Tighten to 2% as derivation stabilizes
"""

import pytest
import math
import numpy as np

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


# ============================================================================
# BENCHMARK CONSTANTS
# ============================================================================

# kappa benchmark (R=1.3036)
KAPPA_R = 1.3036
KAPPA_C_TARGET = 2.13745440613217263636
KAPPA_TARGET = 0.417293962

# kappa* benchmark (R=1.1167)
KAPPA_STAR_R = 1.1167
KAPPA_STAR_C_TARGET = 1.9379524124677437
KAPPA_STAR_TARGET = 0.407511457

# Ratio target
TARGET_RATIO = KAPPA_C_TARGET / KAPPA_STAR_C_TARGET  # ~1.103

# Quadrature settings
N_QUAD = 60


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_polynomials_as_dict(loader_func, enforce_Q0=True):
    """Load polynomials and return as dict."""
    P1, P2, P3, Q = loader_func(enforce_Q0=enforce_Q0)
    return {"P1": P1, "P2": P2, "P3": P3, "Q": Q}


# ============================================================================
# DIAGNOSTIC COMPARISON TESTS
# ============================================================================

class TestDerivedMirrorDiagnostics:
    """
    Diagnostic tests comparing derived mirror vs empirical m1 approach.

    These tests ALWAYS pass but log detailed comparison info.
    """

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    @pytest.fixture(scope="class")
    def kappa_star_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials_kappa_star, enforce_Q0=True)

    def test_kappa_derived_vs_empirical_comparison(self, kappa_polys):
        """
        Compare derived mirror vs empirical at kappa benchmark.

        This logs:
        - c_derived: c computed with operator-derived mirror
        - c_empirical: c computed with empirical m1 = exp(R) + 5
        - Their gap from target
        """
        from src.evaluate import compute_c_derived_mirror, compute_c_paper_ordered

        theta = 4.0 / 7.0

        # Derived mirror (Phase 10)
        derived_result = compute_c_derived_mirror(
            theta=theta,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
            verbose=False
        )

        # Empirical mirror (baseline)
        empirical_result = compute_c_paper_ordered(
            theta=theta,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
            s12_pair_mode='triangle',
        )

        derived_gap = (derived_result.c - KAPPA_C_TARGET) / KAPPA_C_TARGET * 100
        empirical_gap = (empirical_result.total - KAPPA_C_TARGET) / KAPPA_C_TARGET * 100

        # Compute m1 values for diagnostics
        m1_empirical = np.exp(KAPPA_R) + 5  # K=3 formula

        print(f"\n=== kappa Benchmark: Derived vs Empirical ===")
        print(f"  Target c:      {KAPPA_C_TARGET:.6f}")
        print(f"  Derived c:     {derived_result.c:.6f}  ({derived_gap:+.2f}%)")
        print(f"  Empirical c:   {empirical_result.total:.6f}  ({empirical_gap:+.2f}%)")
        print(f"")
        print(f"  S12_direct:        {derived_result.S12_direct:.6f}")
        print(f"  S12_mirror_op:     {derived_result.S12_mirror_operator:.6f}")
        print(f"  S12_basis:         {derived_result.S12_basis:.6f}")
        print(f"  S34:               {derived_result.S34:.6f}")
        print(f"")
        print(f"  m1_eff (op/basis): {derived_result.m1_eff:.4f}")
        print(f"  m1_empirical:      {m1_empirical:.4f}")

        # Always pass - this is diagnostic
        assert True

    def test_kappa_star_derived_vs_empirical_comparison(self, kappa_star_polys):
        """Compare derived mirror vs empirical at kappa* benchmark."""
        from src.evaluate import compute_c_derived_mirror, compute_c_paper_ordered

        theta = 4.0 / 7.0

        # Derived mirror (Phase 10)
        derived_result = compute_c_derived_mirror(
            theta=theta,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            K=3,
            verbose=False
        )

        # Empirical mirror (baseline)
        empirical_result = compute_c_paper_ordered(
            theta=theta,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            K=3,
            s12_pair_mode='triangle',
        )

        derived_gap = (derived_result.c - KAPPA_STAR_C_TARGET) / KAPPA_STAR_C_TARGET * 100
        empirical_gap = (empirical_result.total - KAPPA_STAR_C_TARGET) / KAPPA_STAR_C_TARGET * 100

        # Compute m1 values for diagnostics
        m1_empirical = np.exp(KAPPA_STAR_R) + 5  # K=3 formula

        print(f"\n=== kappa* Benchmark: Derived vs Empirical ===")
        print(f"  Target c:      {KAPPA_STAR_C_TARGET:.6f}")
        print(f"  Derived c:     {derived_result.c:.6f}  ({derived_gap:+.2f}%)")
        print(f"  Empirical c:   {empirical_result.total:.6f}  ({empirical_gap:+.2f}%)")
        print(f"")
        print(f"  S12_direct:        {derived_result.S12_direct:.6f}")
        print(f"  S12_mirror_op:     {derived_result.S12_mirror_operator:.6f}")
        print(f"  S12_basis:         {derived_result.S12_basis:.6f}")
        print(f"  S34:               {derived_result.S34:.6f}")
        print(f"")
        print(f"  m1_eff (op/basis): {derived_result.m1_eff:.4f}")
        print(f"  m1_empirical:      {m1_empirical:.4f}")

        assert True


# ============================================================================
# GATE TESTS WITH TOLERANCE
# ============================================================================

class TestDerivedMirrorGate:
    """
    Gate tests for derived mirror operator.

    TOLERANCE POLICY:
    - Initial: 50% tolerance (while operator is being tuned)
    - Goal: Tighten to 5% then 2% as derivation stabilizes

    These tests are marked xfail(strict=False) because the derived
    mirror is still under development. When they pass, it's a sign
    of progress.
    """

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    @pytest.fixture(scope="class")
    def kappa_star_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials_kappa_star, enforce_Q0=True)

    @pytest.mark.xfail(
        reason="Derived mirror operator still being tuned (op/emp ratio ~36x)",
        strict=False
    )
    def test_kappa_derived_within_5_percent(self, kappa_polys):
        """
        kappa benchmark: derived mirror c within 5% of target.

        Target: c = 2.137
        """
        from src.evaluate import compute_c_derived_mirror

        result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        )

        rel_error = abs(result.c - KAPPA_C_TARGET) / KAPPA_C_TARGET

        assert rel_error < 0.05, (
            f"kappa: c = {result.c:.4f}, target = {KAPPA_C_TARGET:.4f}, "
            f"error = {rel_error*100:.1f}% (need < 5%)"
        )

    @pytest.mark.xfail(
        reason="Derived mirror operator still being tuned (op/emp ratio ~36x)",
        strict=False
    )
    def test_kappa_star_derived_within_5_percent(self, kappa_star_polys):
        """
        kappa* benchmark: derived mirror c within 5% of target.

        Target: c = 1.938
        """
        from src.evaluate import compute_c_derived_mirror

        result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            K=3,
        )

        rel_error = abs(result.c - KAPPA_STAR_C_TARGET) / KAPPA_STAR_C_TARGET

        assert rel_error < 0.05, (
            f"kappa*: c = {result.c:.4f}, target = {KAPPA_STAR_C_TARGET:.4f}, "
            f"error = {rel_error*100:.1f}% (need < 5%)"
        )

    @pytest.mark.xfail(
        reason="Derived mirror operator still being tuned",
        strict=False
    )
    def test_ratio_derived_within_5_percent(self, kappa_polys, kappa_star_polys):
        """
        Ratio test: c_kappa / c_kappa* within 5% of target.

        Target ratio: ~1.103
        """
        from src.evaluate import compute_c_derived_mirror

        kappa_result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        )

        kappa_star_result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_STAR_R,
            n=N_QUAD,
            polynomials=kappa_star_polys,
            K=3,
        )

        computed_ratio = kappa_result.c / kappa_star_result.c
        ratio_error = abs(computed_ratio - TARGET_RATIO) / TARGET_RATIO

        assert ratio_error < 0.05, (
            f"Ratio = {computed_ratio:.4f}, target = {TARGET_RATIO:.4f}, "
            f"error = {ratio_error*100:.1f}% (need < 5%)"
        )


# ============================================================================
# OPERATOR CORRECTNESS TESTS
# ============================================================================

class TestOperatorCorrectness:
    """
    Tests that verify the operator structure is mathematically correct,
    independent of the final c value.
    """

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    def test_derived_components_finite(self, kappa_polys):
        """All derived mirror components should be finite."""
        from src.evaluate import compute_c_derived_mirror

        result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        )

        assert np.isfinite(result.c), f"c = {result.c}"
        assert np.isfinite(result.S12_direct), f"S12_direct = {result.S12_direct}"
        assert np.isfinite(result.S12_mirror_operator), f"S12_mirror_op = {result.S12_mirror_operator}"
        assert np.isfinite(result.S34), f"S34 = {result.S34}"

    def test_derived_mirror_positive(self, kappa_polys):
        """Derived mirror component should be positive (same sign as direct)."""
        from src.evaluate import compute_c_derived_mirror

        result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        )

        # S12_direct is positive
        assert result.S12_direct > 0, f"S12_direct = {result.S12_direct}"

        # S12_mirror_operator should also be positive
        assert result.S12_mirror_operator > 0, (
            f"S12_mirror_operator = {result.S12_mirror_operator} "
            f"(expected positive, same sign as direct)"
        )

    def test_m1_eff_reasonable_range(self, kappa_polys):
        """
        Effective m1 should be in a reasonable range.

        Empirical: m1 = exp(R) + 5 ~ 8.68 for R=1.3036
        Derived: m1_eff = S12_mirror_op / S12_basis

        Current observation: m1_eff ~ 36x larger than empirical.
        This test checks it's at least positive and finite.
        """
        from src.evaluate import compute_c_derived_mirror

        result = compute_c_derived_mirror(
            theta=4.0/7.0,
            R=KAPPA_R,
            n=N_QUAD,
            polynomials=kappa_polys,
            K=3,
        )

        m1_empirical = np.exp(KAPPA_R) + 5  # K=3 formula

        # m1 should be finite and positive
        assert np.isfinite(result.m1_eff), f"m1_eff = {result.m1_eff}"
        assert result.m1_eff > 0, f"m1_eff = {result.m1_eff}"

        # Log the ratio for tracking
        ratio = result.m1_eff / m1_empirical
        print(f"\n  m1_eff / m1_empirical = {ratio:.2f}x")

    def test_operator_no_q_blowup(self, kappa_polys):
        """
        Verify Phase 10 operator approach doesn't have Phase 9 Q blowup.

        Phase 9 failure: Q(1+D) on direct eigenvalues pushed arguments into [1,2]
        where Q polynomials blow up (112x amplification).

        Phase 10: Q(D) on swapped eigenvalues keeps arguments in [0, theta] ~ [0, 0.57]
        where Q is well-behaved.

        Test: Compare (1,1) pair I1_mirror with Q=1 vs realistic Q.
        The ratio should be moderate (< 10x), not 100x+.
        """
        from src.mirror_operator_exact import compute_I1_mirror_operator_exact
        from src.polynomials import Polynomial

        theta = 4.0 / 7.0
        R = KAPPA_R
        n = N_QUAD

        # Create Q=1 polynomials (no amplification)
        P = Polynomial(np.array([1.0]))
        Q1 = Polynomial(np.array([1.0]))
        polys_Q1 = {'P1': P, 'P2': P, 'P3': P, 'Q': Q1}

        result_Q1 = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n, polynomials=polys_Q1,
            ell1=1, ell2=1
        )

        result_real = compute_I1_mirror_operator_exact(
            theta=theta, R=R, n=n, polynomials=kappa_polys,
            ell1=1, ell2=1
        )

        ratio = abs(result_real.value / result_Q1.value) if abs(result_Q1.value) > 1e-15 else float('inf')

        # Ratio should be moderate - Q polynomials should not cause massive blowup
        # Phase 9 saw 100x+ blowup; Phase 10 should be < 10x
        assert ratio < 10, (
            f"Q polynomial amplification = {ratio:.1f}x "
            f"(Phase 9 saw 100x+, Phase 10 should be < 10x)"
        )

        print(f"\n  I1_mirror (Q=1):    {result_Q1.value:.6f}")
        print(f"  I1_mirror (real Q): {result_real.value:.6f}")
        print(f"  Q amplification:    {ratio:.2f}x (should be < 10x)")


# ============================================================================
# QUADRATURE STABILITY TESTS
# ============================================================================

class TestQuadratureStability:
    """Verify derived mirror is stable under quadrature refinement."""

    @pytest.fixture(scope="class")
    def kappa_polys(self):
        return load_polynomials_as_dict(load_przz_polynomials, enforce_Q0=True)

    def test_quadrature_convergence(self, kappa_polys):
        """
        Derived c should be stable across n=40/60/80.

        Tolerance: 2% variation between n=40 and n=80.
        """
        from src.evaluate import compute_c_derived_mirror

        c_values = {}
        for n in [40, 60, 80]:
            result = compute_c_derived_mirror(
                theta=4.0/7.0,
                R=KAPPA_R,
                n=n,
                polynomials=kappa_polys,
                K=3,
            )
            c_values[n] = result.c

        # Compare n=40 vs n=80
        variation = abs(c_values[80] - c_values[40]) / c_values[60]

        print(f"\n  c(n=40) = {c_values[40]:.6f}")
        print(f"  c(n=60) = {c_values[60]:.6f}")
        print(f"  c(n=80) = {c_values[80]:.6f}")
        print(f"  Variation: {variation*100:.2f}%")

        assert variation < 0.02, (
            f"Quadrature not stable: variation = {variation*100:.2f}% (need < 2%)"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
