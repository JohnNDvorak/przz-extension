"""
src/polynomial_scorer.py
Phase 47: Polynomial Scoring Harness

Wraps KappaEngine with:
- Stage A: Contract validation
- Stage B: Microcase ladder
- Stage C: Fast R-sweep
- Stage D: Slow confirmation
- Two-benchmark gate

Uses THETA_CUBED mode exclusively (the validated first-principles formula).

FORMULAS (100% First-Principles):
=================================
g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)
g_I2 = 1 + θ(2-θ) / (2K(2K+1))
base = exp(R) + (2K-1)
m = [f_I1 × g_I1 + (1-f_I1) × g_I2] × base
c = I₁I₂(+R) + m × I₁I₂(-R) + I₃I₄(+R)
κ = 1 - log(c) / R

Created: 2025-12-27 (Phase 47)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import logging

from src.kappa_engine import (
    KappaEngine,
    KappaResult,
    IntegralComponents,
    CorrectionFactors,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BENCHMARK CONSTANTS
# =============================================================================

# κ benchmark (R=1.3036)
KAPPA_TARGET = 0.417293962
C_TARGET_KAPPA = 2.13745440613217263636  # exp(R*(1-κ))

# κ* benchmark (R=1.1167)
KAPPA_STAR_TARGET = 0.407511457
C_TARGET_KAPPA_STAR = 1.93795241121330


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ContractResult:
    """Result of Stage A contract validation."""
    passed: bool
    message: str
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class MicrocaseResult:
    """Result of a single microcase evaluation."""
    name: str
    c: float
    kappa: float
    S12_plus: float
    S12_minus: float
    S34_plus: float
    f_I1: float
    success: bool
    error: Optional[str] = None


@dataclass
class LadderResult:
    """Result of Stage B microcase ladder."""
    cases: Dict[str, MicrocaseResult]
    ladder_valid: bool
    validation_message: str


@dataclass
class SweepPoint:
    """Single point in an R-sweep."""
    R: float
    c: float
    kappa: float
    f_I1: float
    m: float
    n_quad: int


@dataclass
class SweepResult:
    """Result of Stage C/D R-sweep."""
    points: List[SweepPoint]
    R_opt: float
    c_opt: float
    kappa_opt: float


@dataclass
class ConfirmationResult:
    """Result of Stage D slow confirmation."""
    R: float
    levels: Dict[int, SweepPoint]  # n_quad -> result
    converged: bool
    max_drift: float


@dataclass
class TwoBenchmarkResult:
    """Result of two-benchmark gate."""
    kappa: Dict[str, float]
    kappa_star: Dict[str, float]
    ratio: float
    ratio_target: float
    ratio_gap_pct: float
    gate_passed: bool


@dataclass
class FullScoringResult:
    """Complete scoring result from all stages."""
    contract: ContractResult
    ladder: LadderResult
    fast_sweep: SweepResult
    confirmation: ConfirmationResult
    two_benchmark: TwoBenchmarkResult

    # Summary
    R_opt: float
    c_opt: float
    kappa_opt: float
    overall_valid: bool


# =============================================================================
# POLYNOMIAL SCORER
# =============================================================================

class PolynomialScorer:
    """
    Production scoring harness using KappaEngine.

    Wraps KappaEngine with contract validation, microcase ladder,
    R-sweep, and two-benchmark gate.

    All scoring uses the validated THETA_CUBED first-principles formula.
    """

    def __init__(self, K: int = 3, theta: float = 4/7):
        """
        Initialize scorer.

        Args:
            K: Number of mollifier pieces (default 3)
            theta: Mollifier exponent (default 4/7)
        """
        self.K = K
        self.theta = theta

    # =========================================================================
    # STAGE A: CONTRACT VALIDATION
    # =========================================================================

    def validate_contract(self, polynomials: Dict) -> ContractResult:
        """
        Stage A: Contract checks (cheap, deterministic).

        Validates:
        1. Degree checks: P1, P2, P3 within limits
        2. Boundary conditions: P1(0)=0, P1(1)=1, Q(0)≈1
        3. Coefficient sanity: No NaNs, no extreme magnitudes
        4. Evaluator sanity: c is finite at mid-R

        Args:
            polynomials: Dict with keys "P1", "P2", "P3", "Q"

        Returns:
            ContractResult with pass/fail and details
        """
        checks = {}

        P1 = polynomials["P1"]
        P2 = polynomials["P2"]
        P3 = polynomials["P3"]
        Q = polynomials["Q"]

        # Check 1: Degree limits
        max_degree = 10
        for name, poly in [("P1", P1), ("P2", P2), ("P3", P3)]:
            deg = self._get_degree(poly)
            checks[f"{name}_degree"] = deg <= max_degree
            if deg > max_degree:
                return ContractResult(
                    passed=False,
                    message=f"{name} degree {deg} exceeds limit {max_degree}",
                    checks=checks,
                )

        # Check 2: Boundary conditions
        # P1(0) = 0
        p1_at_0 = self._eval_at(P1, 0.0)
        checks["P1(0)=0"] = abs(p1_at_0) < 1e-10
        if not checks["P1(0)=0"]:
            return ContractResult(
                passed=False,
                message=f"P1(0) = {p1_at_0:.6e} ≠ 0",
                checks=checks,
            )

        # P1(1) = 1
        p1_at_1 = self._eval_at(P1, 1.0)
        checks["P1(1)=1"] = abs(p1_at_1 - 1.0) < 1e-10
        if not checks["P1(1)=1"]:
            return ContractResult(
                passed=False,
                message=f"P1(1) = {p1_at_1:.6f} ≠ 1",
                checks=checks,
            )

        # Q(0) ≈ 1 (allow some tolerance for PRZZ paper-literal)
        q_at_0 = self._get_Q_at_zero(Q)
        checks["Q(0)≈1"] = abs(q_at_0 - 1.0) < 1e-5
        if not checks["Q(0)≈1"]:
            return ContractResult(
                passed=False,
                message=f"Q(0) = {q_at_0:.6f} ≠ 1",
                checks=checks,
            )

        # Check 3: Coefficient sanity
        for name, poly in [("P1", P1), ("P2", P2), ("P3", P3), ("Q", Q)]:
            coeffs = self._get_coeffs(poly)
            if np.any(np.isnan(coeffs)):
                checks[f"{name}_no_nan"] = False
                return ContractResult(
                    passed=False,
                    message=f"{name} has NaN coefficients",
                    checks=checks,
                )
            if np.any(np.abs(coeffs) > 1e6):
                checks[f"{name}_bounded"] = False
                return ContractResult(
                    passed=False,
                    message=f"{name} has extreme coefficients",
                    checks=checks,
                )
            checks[f"{name}_valid"] = True

        # Check 4: Evaluator sanity
        R_mid = 1.2
        try:
            result = self._quick_evaluate(polynomials, R_mid, n_quad=20)
            if not np.isfinite(result.c):
                checks["evaluator_sanity"] = False
                return ContractResult(
                    passed=False,
                    message=f"c not finite at R={R_mid}",
                    checks=checks,
                )
            checks["evaluator_sanity"] = True
        except Exception as e:
            checks["evaluator_sanity"] = False
            return ContractResult(
                passed=False,
                message=f"Evaluation failed: {e}",
                checks=checks,
            )

        return ContractResult(
            passed=True,
            message="Contract passed",
            checks=checks,
        )

    # =========================================================================
    # STAGE B: MICROCASE LADDER
    # =========================================================================

    def run_microcase_ladder(
        self,
        polynomials: Dict,
        R: float = 1.3036,
        n_quad: int = 40,
    ) -> LadderResult:
        """
        Stage B: Microcase ladder validation.

        Tests polynomial families through increasing complexity:
        0. P=Q=1 (kernel-only sanity)
        1. P=real, Q=1 (Beta moment without Q interaction)
        2. P=1, Q=real (isolates Q interaction)
        3. P=real, Q=real (production case)

        Args:
            polynomials: Dict with real polynomials
            R: R value for testing
            n_quad: Quadrature points

        Returns:
            LadderResult with all cases and validation
        """
        from src.polynomials import Polynomial, P1Polynomial, PellPolynomial

        P1 = polynomials["P1"]
        P2 = polynomials["P2"]
        P3 = polynomials["P3"]
        Q = polynomials["Q"]

        # Unity polynomials
        P_unity = Polynomial(np.array([1.0]))
        Q_unity = Polynomial(np.array([1.0]))

        # Create P1 unity that satisfies P1(0)=0, P1(1)=1
        # P1(x) = x satisfies this
        P1_unity = P1Polynomial(tilde_coeffs=np.array([0.0]))  # P1(x) = x + x(1-x)*0 = x
        P_ell_unity = PellPolynomial(tilde_coeffs=np.array([1.0]))  # P(x) = x*1 = x

        cases = {}

        # Case 0: P=Q=1 (all unity)
        polys_0 = {"P1": P1_unity, "P2": P_ell_unity, "P3": P_ell_unity, "Q": Q_unity}
        cases["P=Q=1"] = self._evaluate_case("P=Q=1", polys_0, R, n_quad)

        # Case 1: P=real, Q=1
        polys_1 = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_unity}
        cases["P=real, Q=1"] = self._evaluate_case("P=real, Q=1", polys_1, R, n_quad)

        # Case 2: P=1, Q=real
        polys_2 = {"P1": P1_unity, "P2": P_ell_unity, "P3": P_ell_unity, "Q": Q}
        cases["P=1, Q=real"] = self._evaluate_case("P=1, Q=real", polys_2, R, n_quad)

        # Case 3: P=real, Q=real (production)
        polys_3 = polynomials
        cases["P=real, Q=real"] = self._evaluate_case("P=real, Q=real", polys_3, R, n_quad)

        # Validate ladder behavior
        ladder_valid, validation_msg = self._validate_ladder(cases)

        return LadderResult(
            cases=cases,
            ladder_valid=ladder_valid,
            validation_message=validation_msg,
        )

    def _evaluate_case(
        self,
        name: str,
        polynomials: Dict,
        R: float,
        n_quad: int,
    ) -> MicrocaseResult:
        """Evaluate a single microcase."""
        try:
            result = self._quick_evaluate(polynomials, R, n_quad)
            return MicrocaseResult(
                name=name,
                c=result.c,
                kappa=result.kappa,
                S12_plus=result.integrals.S12_plus,
                S12_minus=result.integrals.S12_minus,
                S34_plus=result.integrals.S34_plus,
                f_I1=result.integrals.f_I1,
                success=True,
            )
        except Exception as e:
            return MicrocaseResult(
                name=name,
                c=float('nan'),
                kappa=float('nan'),
                S12_plus=float('nan'),
                S12_minus=float('nan'),
                S34_plus=float('nan'),
                f_I1=float('nan'),
                success=False,
                error=str(e),
            )

    def _validate_ladder(self, cases: Dict[str, MicrocaseResult]) -> Tuple[bool, str]:
        """
        Validate ladder behavior.

        Expected:
        - All cases should succeed
        - c should be positive and finite
        - Progression should be sensible
        """
        # Check all cases succeeded
        for name, case in cases.items():
            if not case.success:
                return False, f"Case '{name}' failed: {case.error}"
            if not np.isfinite(case.c) or case.c <= 0:
                return False, f"Case '{name}' has invalid c={case.c}"

        # Check c values are in reasonable range
        c_values = [case.c for case in cases.values()]
        if max(c_values) > 100 or min(c_values) < 0.01:
            return False, f"c values out of reasonable range: {c_values}"

        return True, "Ladder validation passed"

    # =========================================================================
    # STAGE C: FAST R-SWEEP
    # =========================================================================

    def run_fast_r_sweep(
        self,
        polynomials: Dict,
        R_range: Tuple[float, float] = (0.9, 1.6),
        n_points: int = 15,
        n_quad: int = 30,
    ) -> SweepResult:
        """
        Stage C: Fast R-sweep for candidate screening.

        Uses low n_quad for speed, reports R_opt and c_opt.

        Args:
            polynomials: Polynomial dict
            R_range: (R_min, R_max) range to sweep
            n_points: Number of R points
            n_quad: Quadrature points (low for speed)

        Returns:
            SweepResult with optimal R, c, κ
        """
        R_values = np.linspace(R_range[0], R_range[1], n_points)
        points = []

        for R in R_values:
            try:
                result = self._quick_evaluate(polynomials, R, n_quad)
                points.append(SweepPoint(
                    R=R,
                    c=result.c,
                    kappa=result.kappa,
                    f_I1=result.integrals.f_I1,
                    m=result.corrections.m,
                    n_quad=n_quad,
                ))
            except Exception as e:
                logger.warning(f"R-sweep failed at R={R}: {e}")

        if not points:
            raise ValueError("R-sweep produced no valid points")

        # Find optimal (max κ = min c)
        best = max(points, key=lambda p: p.kappa)

        return SweepResult(
            points=points,
            R_opt=best.R,
            c_opt=best.c,
            kappa_opt=best.kappa,
        )

    # =========================================================================
    # STAGE D: SLOW CONFIRMATION
    # =========================================================================

    def run_slow_confirmation(
        self,
        polynomials: Dict,
        R_opt: float,
        n_quad_levels: List[int] = [60, 80, 120],
    ) -> ConfirmationResult:
        """
        Stage D: Slow confirmation for top candidates.

        Verifies ranking stability across quadrature levels.

        Args:
            polynomials: Polynomial dict
            R_opt: Optimal R from fast sweep
            n_quad_levels: List of quadrature levels to test

        Returns:
            ConfirmationResult with convergence status
        """
        levels = {}

        for n_quad in n_quad_levels:
            try:
                result = self._quick_evaluate(polynomials, R_opt, n_quad)
                levels[n_quad] = SweepPoint(
                    R=R_opt,
                    c=result.c,
                    kappa=result.kappa,
                    f_I1=result.integrals.f_I1,
                    m=result.corrections.m,
                    n_quad=n_quad,
                )
            except Exception as e:
                logger.warning(f"Confirmation failed at n_quad={n_quad}: {e}")

        # Check convergence
        if len(levels) < 2:
            return ConfirmationResult(
                R=R_opt,
                levels=levels,
                converged=False,
                max_drift=float('inf'),
            )

        c_values = [levels[n].c for n in sorted(levels.keys())]
        drifts = []
        for i in range(len(c_values) - 1):
            if c_values[i] != 0:
                drift = abs(c_values[i+1] - c_values[i]) / abs(c_values[i])
                drifts.append(drift)

        max_drift = max(drifts) if drifts else 0.0
        converged = max_drift < 1e-5  # 0.001% tolerance

        return ConfirmationResult(
            R=R_opt,
            levels=levels,
            converged=converged,
            max_drift=max_drift,
        )

    # =========================================================================
    # TWO-BENCHMARK GATE
    # =========================================================================

    def run_two_benchmark_gate(
        self,
        polynomials: Dict,
        polynomials_kappa_star: Optional[Dict] = None,
        n_quad: int = 60,
    ) -> TwoBenchmarkResult:
        """
        Validate against both κ (R=1.3036) and κ* (R=1.1167) benchmarks.

        Any improvement must work on BOTH benchmarks.

        Args:
            polynomials: Polynomials for κ benchmark
            polynomials_kappa_star: Polynomials for κ* (if different), None = same
            n_quad: Quadrature points

        Returns:
            TwoBenchmarkResult with pass/fail
        """
        # Benchmark 1: κ (R=1.3036)
        result_kappa = self._quick_evaluate(polynomials, R=1.3036, n_quad=n_quad)
        kappa_data = {
            "c": result_kappa.c,
            "kappa": result_kappa.kappa,
            "target_c": C_TARGET_KAPPA,
            "target_kappa": KAPPA_TARGET,
            "c_gap_pct": (result_kappa.c / C_TARGET_KAPPA - 1) * 100,
            "kappa_gap_pct": (result_kappa.kappa / KAPPA_TARGET - 1) * 100,
        }

        # Benchmark 2: κ* (R=1.1167)
        if polynomials_kappa_star is None:
            # Use same polynomials (for candidate testing)
            polys_star = polynomials
        else:
            polys_star = polynomials_kappa_star

        result_kappa_star = self._quick_evaluate(polys_star, R=1.1167, n_quad=n_quad)
        kappa_star_data = {
            "c": result_kappa_star.c,
            "kappa": result_kappa_star.kappa,
            "target_c": C_TARGET_KAPPA_STAR,
            "target_kappa": KAPPA_STAR_TARGET,
            "c_gap_pct": (result_kappa_star.c / C_TARGET_KAPPA_STAR - 1) * 100,
            "kappa_gap_pct": (result_kappa_star.kappa / KAPPA_STAR_TARGET - 1) * 100,
        }

        # Ratio check
        ratio = result_kappa.c / result_kappa_star.c
        ratio_target = C_TARGET_KAPPA / C_TARGET_KAPPA_STAR  # ≈ 1.103
        ratio_gap_pct = (ratio / ratio_target - 1) * 100

        # Pass/fail criteria
        gate_passed = (
            abs(kappa_data["c_gap_pct"]) < 5.0 and
            abs(kappa_star_data["c_gap_pct"]) < 5.0 and
            abs(ratio_gap_pct) < 2.0
        )

        return TwoBenchmarkResult(
            kappa=kappa_data,
            kappa_star=kappa_star_data,
            ratio=ratio,
            ratio_target=ratio_target,
            ratio_gap_pct=ratio_gap_pct,
            gate_passed=gate_passed,
        )

    # =========================================================================
    # FULL SCORING PIPELINE
    # =========================================================================

    def score_full(
        self,
        polynomials: Dict,
        R_range: Tuple[float, float] = (0.9, 1.6),
        n_fast: int = 30,
        n_slow_levels: List[int] = [60, 80, 120],
    ) -> FullScoringResult:
        """
        Run the complete 4-stage scoring pipeline.

        Stages:
        A. Contract validation
        B. Microcase ladder
        C. Fast R-sweep
        D. Slow confirmation
        + Two-benchmark gate

        Args:
            polynomials: Dict with P1, P2, P3, Q
            R_range: Range for R-sweep
            n_fast: Quadrature for fast sweep
            n_slow_levels: Quadrature levels for confirmation

        Returns:
            FullScoringResult with all stages
        """
        # Stage A: Contract
        contract = self.validate_contract(polynomials)
        if not contract.passed:
            logger.warning(f"Contract failed: {contract.message}")
            # Return early with failed result
            return FullScoringResult(
                contract=contract,
                ladder=LadderResult({}, False, "Contract failed"),
                fast_sweep=SweepResult([], 0.0, float('nan'), float('nan')),
                confirmation=ConfirmationResult(0.0, {}, False, float('inf')),
                two_benchmark=TwoBenchmarkResult({}, {}, 0.0, 0.0, 0.0, False),
                R_opt=0.0,
                c_opt=float('nan'),
                kappa_opt=float('nan'),
                overall_valid=False,
            )

        # Stage B: Microcase ladder
        ladder = self.run_microcase_ladder(polynomials)
        if not ladder.ladder_valid:
            logger.warning(f"Ladder failed: {ladder.validation_message}")

        # Stage C: Fast R-sweep
        fast_sweep = self.run_fast_r_sweep(
            polynomials, R_range=R_range, n_quad=n_fast
        )

        # Stage D: Slow confirmation
        confirmation = self.run_slow_confirmation(
            polynomials, fast_sweep.R_opt, n_quad_levels=n_slow_levels
        )

        # Two-benchmark gate
        two_benchmark = self.run_two_benchmark_gate(polynomials)

        # Determine final values from highest quadrature level
        max_n = max(confirmation.levels.keys()) if confirmation.levels else n_fast
        if max_n in confirmation.levels:
            final = confirmation.levels[max_n]
            c_opt = final.c
            kappa_opt = final.kappa
        else:
            c_opt = fast_sweep.c_opt
            kappa_opt = fast_sweep.kappa_opt

        overall_valid = (
            contract.passed and
            ladder.ladder_valid and
            confirmation.converged and
            two_benchmark.gate_passed
        )

        return FullScoringResult(
            contract=contract,
            ladder=ladder,
            fast_sweep=fast_sweep,
            confirmation=confirmation,
            two_benchmark=two_benchmark,
            R_opt=fast_sweep.R_opt,
            c_opt=c_opt,
            kappa_opt=kappa_opt,
            overall_valid=overall_valid,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _quick_evaluate(
        self,
        polynomials: Dict,
        R: float,
        n_quad: int,
    ) -> KappaResult:
        """
        Quick evaluation using KappaEngine.

        Converts polynomial dict to coefficient lists and creates engine.
        """
        P1 = polynomials["P1"]
        P2 = polynomials["P2"]
        P3 = polynomials["P3"]
        Q = polynomials["Q"]

        # Extract coefficient lists
        P1_coeffs = self._get_tilde_coeffs(P1)
        P2_coeffs = self._get_tilde_coeffs(P2)
        P3_coeffs = self._get_tilde_coeffs(P3)
        Q_coeffs = self._get_monomial_coeffs(Q)

        engine = KappaEngine(
            P1_coeffs=P1_coeffs,
            P2_coeffs=P2_coeffs,
            P3_coeffs=P3_coeffs,
            Q_coeffs=Q_coeffs,
            theta=self.theta,
            K=self.K,
            R=R,
            n_quad=n_quad,
        )

        return engine.compute_kappa()

    def _get_degree(self, poly) -> int:
        """Get polynomial degree."""
        if hasattr(poly, 'degree'):
            if callable(poly.degree):
                return poly.degree()
            return poly.degree
        if hasattr(poly, 'coeffs'):
            return len(poly.coeffs) - 1
        return 0

    def _eval_at(self, poly, x: float) -> float:
        """Evaluate polynomial at x."""
        if hasattr(poly, 'eval'):
            return float(poly.eval(np.array([x]))[0])
        if hasattr(poly, 'to_monomial'):
            mono = poly.to_monomial()
            return float(mono.eval(np.array([x]))[0])
        return float('nan')

    def _get_Q_at_zero(self, Q) -> float:
        """Get Q(0) value."""
        if hasattr(Q, 'Q_at_zero'):
            return Q.Q_at_zero()
        return self._eval_at(Q, 0.0)

    def _get_coeffs(self, poly) -> np.ndarray:
        """Get coefficient array."""
        if hasattr(poly, 'coeffs'):
            return np.asarray(poly.coeffs)
        if hasattr(poly, 'tilde_coeffs'):
            return np.asarray(poly.tilde_coeffs)
        if hasattr(poly, 'to_monomial'):
            return np.asarray(poly.to_monomial().coeffs)
        return np.array([])

    def _get_tilde_coeffs(self, poly) -> List[float]:
        """Get tilde coefficients as list."""
        if hasattr(poly, 'tilde_coeffs'):
            return poly.tilde_coeffs.tolist()
        # For Polynomial class, assume it's already in the right form
        if hasattr(poly, 'coeffs'):
            return poly.coeffs.tolist()
        return [0.0]

    def _get_monomial_coeffs(self, poly) -> List[float]:
        """Get monomial coefficients as list."""
        if hasattr(poly, 'to_monomial'):
            return poly.to_monomial().coeffs.tolist()
        if hasattr(poly, 'coeffs'):
            return poly.coeffs.tolist()
        return [1.0]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_przz_polynomials(n_quad: int = 80) -> FullScoringResult:
    """
    Score the PRZZ κ polynomials through the full pipeline.

    Args:
        n_quad: Maximum quadrature level

    Returns:
        FullScoringResult
    """
    from src.polynomials import load_przz_polynomials

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q}

    scorer = PolynomialScorer(K=3, theta=4/7)
    return scorer.score_full(polynomials)


def quick_score(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: List[float],
    R: float = 1.3036,
    n_quad: int = 60,
) -> KappaResult:
    """
    Quick scoring of polynomial coefficients at a single R.

    Bypasses the full pipeline for fast candidate screening.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        Q_coeffs: Monomial coefficients for Q
        R: R value
        n_quad: Quadrature points

    Returns:
        KappaResult
    """
    engine = KappaEngine(
        P1_coeffs=P1_coeffs,
        P2_coeffs=P2_coeffs,
        P3_coeffs=P3_coeffs,
        Q_coeffs=Q_coeffs,
        theta=4/7,
        K=3,
        R=R,
        n_quad=n_quad,
    )
    return engine.compute_kappa()


# =============================================================================
# CLI SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Scoring PRZZ polynomials through full pipeline...")
    result = score_przz_polynomials()

    print("\n" + "=" * 70)
    print("SCORING RESULTS")
    print("=" * 70)

    print(f"\nContract: {'PASS' if result.contract.passed else 'FAIL'}")
    print(f"  {result.contract.message}")

    print(f"\nMicrocase Ladder: {'PASS' if result.ladder.ladder_valid else 'FAIL'}")
    for name, case in result.ladder.cases.items():
        print(f"  {name}: c={case.c:.6f}, κ={case.kappa:.6f}")

    print(f"\nFast Sweep: R_opt={result.fast_sweep.R_opt:.4f}")
    print(f"  c_opt={result.fast_sweep.c_opt:.6f}")
    print(f"  κ_opt={result.fast_sweep.kappa_opt:.6f}")

    print(f"\nConfirmation: {'CONVERGED' if result.confirmation.converged else 'NOT CONVERGED'}")
    print(f"  max_drift={result.confirmation.max_drift:.2e}")
    for n, pt in result.confirmation.levels.items():
        print(f"  n={n}: c={pt.c:.10f}")

    print(f"\nTwo-Benchmark Gate: {'PASS' if result.two_benchmark.gate_passed else 'FAIL'}")
    print(f"  κ: c={result.two_benchmark.kappa['c']:.6f} (gap {result.two_benchmark.kappa['c_gap_pct']:+.4f}%)")
    print(f"  κ*: c={result.two_benchmark.kappa_star['c']:.6f} (gap {result.two_benchmark.kappa_star['c_gap_pct']:+.4f}%)")
    print(f"  ratio: {result.two_benchmark.ratio:.4f} (target {result.two_benchmark.ratio_target:.4f}, gap {result.two_benchmark.ratio_gap_pct:+.4f}%)")

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'VALID' if result.overall_valid else 'INVALID'}")
    print(f"  R_opt = {result.R_opt:.4f}")
    print(f"  c_opt = {result.c_opt:.10f}")
    print(f"  κ_opt = {result.kappa_opt:.10f}")
    print("=" * 70)
