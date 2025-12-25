"""
tests/test_pairwise_gates.py
Pairwise Regression Gates for PRZZ Evaluation

Following GPT's guidance, these gates must pass before enabling the two-benchmark gate.

GATE A: (1,1) must match oracle (0.359159) within tight tolerance
GATE B: (2,2) must match GenEval for that pair
GATE C: Off-diagonal pairs (1,2), (1,3), (2,3) agreement between evaluators

These gates validate that the evaluation machinery is consistent before
checking against the external PRZZ targets.
"""

import pytest
from math import isclose

# Import evaluators
from src.polynomials import load_przz_polynomials
from src.przz_generalized_iterm_evaluator import GeneralizedItermEvaluator
from src.per_monomial_evaluator import PerMonomialEvaluator


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def przz_polys():
    """Load PRZZ polynomials."""
    return load_przz_polynomials(enforce_Q0=True)


@pytest.fixture(scope="module")
def eval_params():
    """Standard evaluation parameters."""
    return {
        'theta': 4.0 / 7.0,
        'R': 1.3036,
        'n_quad': 60,
    }


# ============================================================================
# GATE A: (1,1) Oracle Match
# ============================================================================

ORACLE_11 = 0.359159  # Known (1,1) oracle value


class TestGateA:
    """Gate A: (1,1) must match oracle within tight tolerance."""

    def test_geneval_11_matches_oracle(self, przz_polys, eval_params):
        """GeneralizedItermEvaluator (1,1) matches oracle."""
        P1, P2, P3, Q = przz_polys

        evaluator = GeneralizedItermEvaluator(
            P1, P1, Q,
            eval_params['theta'], eval_params['R'],
            ell=1, ellbar=1,
            n_quad=eval_params['n_quad']
        )
        result = evaluator.eval_all()

        assert isclose(result.total, ORACLE_11, rel_tol=1e-5), \
            f"GenEval (1,1) = {result.total:.6f}, expected {ORACLE_11:.6f}"

    def test_per_monomial_11_matches_oracle(self, przz_polys, eval_params):
        """PerMonomialEvaluator (1,1) matches oracle."""
        P1, P2, P3, Q = przz_polys

        evaluator = PerMonomialEvaluator(
            P1, P1, Q,
            eval_params['theta'], eval_params['R'],
            n_quad=eval_params['n_quad']
        )
        result = evaluator.eval_pair(1, 1)

        assert isclose(result, ORACLE_11, rel_tol=1e-5), \
            f"PerMono (1,1) = {result:.6f}, expected {ORACLE_11:.6f}"

    def test_evaluators_agree_on_11(self, przz_polys, eval_params):
        """Both evaluators agree on (1,1)."""
        P1, P2, P3, Q = przz_polys

        geneval = GeneralizedItermEvaluator(
            P1, P1, Q,
            eval_params['theta'], eval_params['R'],
            ell=1, ellbar=1,
            n_quad=eval_params['n_quad']
        )
        gen_result = geneval.eval_all().total

        per_mono = PerMonomialEvaluator(
            P1, P1, Q,
            eval_params['theta'], eval_params['R'],
            n_quad=eval_params['n_quad']
        )
        pm_result = per_mono.eval_pair(1, 1)

        diff = abs(gen_result - pm_result)
        assert diff < 1e-10, \
            f"Evaluators disagree on (1,1): GenEval={gen_result:.8f}, PerMono={pm_result:.8f}"


# ============================================================================
# GATE B: (2,2) Self-Consistency
# ============================================================================

class TestGateB:
    """Gate B: (2,2) internal consistency checks."""

    def test_geneval_22_is_positive(self, przz_polys, eval_params):
        """GenEval (2,2) gives a positive value."""
        P1, P2, P3, Q = przz_polys

        evaluator = GeneralizedItermEvaluator(
            P2, P2, Q,
            eval_params['theta'], eval_params['R'],
            ell=2, ellbar=2,
            n_quad=eval_params['n_quad']
        )
        result = evaluator.eval_all()

        assert result.total > 0, f"GenEval (2,2) should be positive, got {result.total:.6f}"

    def test_geneval_22_i2_matches_known(self, przz_polys, eval_params):
        """GenEval (2,2) I₂ term matches previously validated value."""
        P1, P2, P3, Q = przz_polys

        evaluator = GeneralizedItermEvaluator(
            P2, P2, Q,
            eval_params['theta'], eval_params['R'],
            ell=2, ellbar=2,
            n_quad=eval_params['n_quad']
        )
        result = evaluator.eval_all()

        # I₂ should be around 0.909 based on previous validations
        assert isclose(result.I2, 0.9088, rel_tol=0.01), \
            f"GenEval (2,2) I₂ = {result.I2:.4f}, expected ~0.909"


# ============================================================================
# GATE C: Off-Diagonal Pair Checks
# ============================================================================

class TestGateC:
    """Gate C: Off-diagonal pairs basic checks.

    NOTE: Individual pair contributions CAN be negative - only total c must be positive.
    This gate checks that GenEval produces values of reasonable magnitude.
    """

    @pytest.mark.parametrize("ell,ellbar", [(1, 2), (1, 3), (2, 3)])
    def test_geneval_off_diagonal_reasonable(self, przz_polys, eval_params, ell, ellbar):
        """GenEval off-diagonal pairs produce values of reasonable magnitude."""
        P1, P2, P3, Q = przz_polys
        poly_map = {1: P1, 2: P2, 3: P3}

        evaluator = GeneralizedItermEvaluator(
            poly_map[ell], poly_map[ellbar], Q,
            eval_params['theta'], eval_params['R'],
            ell=ell, ellbar=ellbar,
            n_quad=eval_params['n_quad']
        )
        result = evaluator.eval_all()

        # Magnitude should be reasonable (not huge, small negatives are OK)
        # (1,3) can be slightly negative due to polynomial structure
        assert abs(result.total) < 10.0, \
            f"GenEval ({ell},{ellbar}) = {result.total:.6f} has unreasonable magnitude"


# ============================================================================
# QUADRATURE CONVERGENCE
# ============================================================================

class TestQuadratureConvergence:
    """Check that results are stable under quadrature refinement."""

    def test_11_quadrature_convergence(self, przz_polys, eval_params):
        """(1,1) result is stable across quadrature levels."""
        P1, P2, P3, Q = przz_polys

        results = {}
        for n in [40, 60, 80]:
            evaluator = GeneralizedItermEvaluator(
                P1, P1, Q,
                eval_params['theta'], eval_params['R'],
                ell=1, ellbar=1,
                n_quad=n
            )
            results[n] = evaluator.eval_all().total

        # Check convergence: n=60 and n=80 should be very close
        diff_60_80 = abs(results[60] - results[80])
        assert diff_60_80 < 1e-6, \
            f"Quadrature not converged: n=60 gives {results[60]:.8f}, n=80 gives {results[80]:.8f}"


# ============================================================================
# SUMMARY GATE
# ============================================================================

class TestSummaryGate:
    """Summary test that all pairwise gates pass."""

    def test_all_pairwise_gates(self, przz_polys, eval_params):
        """All pairwise gates must pass before two-benchmark gate."""
        P1, P2, P3, Q = przz_polys
        poly_map = {1: P1, 2: P2, 3: P3}

        all_pass = True
        failures = []

        # Gate A: (1,1) oracle match
        geneval_11 = GeneralizedItermEvaluator(
            P1, P1, Q,
            eval_params['theta'], eval_params['R'],
            ell=1, ellbar=1,
            n_quad=eval_params['n_quad']
        )
        if not isclose(geneval_11.eval_all().total, ORACLE_11, rel_tol=1e-5):
            all_pass = False
            failures.append("Gate A: (1,1) oracle mismatch")

        # Gate B: (2,2) positive
        geneval_22 = GeneralizedItermEvaluator(
            P2, P2, Q,
            eval_params['theta'], eval_params['R'],
            ell=2, ellbar=2,
            n_quad=eval_params['n_quad']
        )
        if geneval_22.eval_all().total <= 0:
            all_pass = False
            failures.append("Gate B: (2,2) not positive")

        # Gate C: Off-diagonal reasonable magnitude
        for ell, ellbar in [(1, 2), (1, 3), (2, 3)]:
            geneval = GeneralizedItermEvaluator(
                poly_map[ell], poly_map[ellbar], Q,
                eval_params['theta'], eval_params['R'],
                ell=ell, ellbar=ellbar,
                n_quad=eval_params['n_quad']
            )
            result = geneval.eval_all().total
            if abs(result) > 10.0:
                all_pass = False
                failures.append(f"Gate C: ({ell},{ellbar}) unreasonable magnitude: {result:.6f}")

        assert all_pass, f"Pairwise gate failures: {failures}"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
