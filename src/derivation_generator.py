"""
src/derivation_generator.py
Automatic derivation generator for PRZZ mollifier computations.

PURPOSE:
========
Given any polynomial coefficients (P1, P2, ..., PK, Q), this module:
1. Computes all pair contributions with full breakdown
2. Generates LaTeX derivations showing the structure
3. Produces verification tables for paper appendices
4. Validates results across R values and quadrature settings

The PRZZ combinatorial structure (Cases A/B/C/D/..., pairs, mirror assembly)
is determined by K - this machinery adapts automatically.

SUPPORTS:
=========
- K=3 (baseline PRZZ): 6 pairs, Cases A/B/C
- K=4 (extension): 10 pairs, Cases A/B/C/D
- K=5+ (future): Automatically computed

USAGE:
======
    from src.derivation_generator import DerivationGenerator, BatchProcessor

    # Single configuration
    gen = DerivationGenerator(
        P_coeffs={1: [...], 2: [...], 3: [...]},
        Q_coeffs=[...],
        R=1.3036,
        theta=4/7,
        K=3,
        label="baseline"
    )
    report = gen.generate_report()
    latex = gen.generate_full_latex_appendix()

    # Batch processing
    processor = BatchProcessor()
    processor.add_candidate("baseline", {...})
    processor.add_candidate("optimized", {...})
    processor.run_all_validations()
    processor.generate_comparison_report()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations_with_replacement
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.kappa_engine import KappaEngine, KappaResult
from src.polynomials import P1Polynomial, PellPolynomial, Polynomial


# =============================================================================
# CASE CLASSIFICATION (K-AGNOSTIC)
# =============================================================================

def compute_omega(ell: int) -> int:
    """
    Compute ω(ℓ) for a given piece index.

    For d=1 (single derivative order):
        ω(ℓ) = ℓ - 2

    So:
        ℓ=1 → ω=-1 (Case A: derivative terms)
        ℓ=2 → ω=0  (Case B: no attenuation)
        ℓ=3 → ω=1  (Case C: one auxiliary integral)
        ℓ=4 → ω=2  (Case D: two auxiliary integrals)
        etc.
    """
    return ell - 2


def get_case_letter(omega: int) -> str:
    """Map ω value to case letter."""
    if omega < 0:
        return "A"
    elif omega == 0:
        return "B"
    else:
        # C, D, E, ... for ω = 1, 2, 3, ...
        return chr(ord('C') + omega - 1)


def get_case_type(ell1: int, ell2: int) -> str:
    """Get case type string for a pair."""
    omega1 = compute_omega(ell1)
    omega2 = compute_omega(ell2)
    letter1 = get_case_letter(omega1)
    letter2 = get_case_letter(omega2)
    return f"{letter1}×{letter2}"


def get_all_pairs(K: int) -> List[Tuple[int, int]]:
    """Get all (ℓ₁, ℓ₂) pairs for given K, with ℓ₁ ≤ ℓ₂."""
    return list(combinations_with_replacement(range(1, K + 1), 2))


def count_pairs(K: int) -> int:
    """Number of pairs for K pieces: K(K+1)/2."""
    return K * (K + 1) // 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PairContribution:
    """Contribution from a single (ℓ₁, ℓ₂) pair."""
    ell1: int
    ell2: int
    omega_ell1: int
    omega_ell2: int
    case_type: str
    value: float
    pct_of_total: float


@dataclass
class QuadratureCheck:
    """Result of quadrature convergence check."""
    n_values: List[int]
    kappa_values: List[float]
    c_values: List[float]
    max_variation_pct: float
    passed: bool


@dataclass
class RSweepCheck:
    """Result of R-sweep validation."""
    R_values: List[float]
    kappa_baseline: List[float]
    kappa_optimized: List[float]
    improvements_pct: List[float]
    all_positive: bool
    passed: bool


@dataclass
class DerivationReport:
    """Complete derivation report for a polynomial configuration."""

    # Identification
    label: str
    timestamp: str

    # Parameters
    R: float
    theta: float
    theta_exact: str
    K: int
    n_pairs: int

    # Polynomial coefficients (dict keyed by ℓ)
    P_coeffs: Dict[int, List[float]]
    Q_coeffs: List[float]

    # Results
    kappa: float
    c: float
    S12_total: float

    # Per-pair breakdown
    pairs: Dict[str, PairContribution]

    # Mirror assembly components
    S12_plus: float
    S12_minus: float
    S34_plus: float
    m: float
    g_I1: float
    g_I2: float

    # Validation results
    quadrature_check: Optional[QuadratureCheck] = None
    r_sweep_check: Optional[RSweepCheck] = None

    # Comparison to target
    kappa_target: Optional[float] = None
    kappa_gap_pct: Optional[float] = None


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class DerivationGenerator:
    """
    Generates derivations and documentation for any polynomial configuration.

    Supports K=3, 4, 5, ... with automatic pair enumeration and case classification.
    """

    def __init__(
        self,
        P_coeffs: Dict[int, List[float]],
        Q_coeffs: List[float],
        R: float = 1.3036,
        theta: float = 4/7,
        K: Optional[int] = None,
        n_quad: int = 60,
        label: str = "custom",
    ):
        """
        Initialize derivation generator.

        Args:
            P_coeffs: Dict mapping ℓ → tilde coefficients for P_ℓ
                      e.g., {1: [...], 2: [...], 3: [...]} for K=3
            Q_coeffs: Coefficients for Q polynomial (monomial basis)
            R: Shift parameter
            theta: Mollifier length parameter (typically 4/7)
            K: Number of pieces (inferred from P_coeffs if not given)
            n_quad: Quadrature points
            label: Human-readable label for this configuration
        """
        self.P_coeffs = P_coeffs
        self.Q_coeffs = Q_coeffs
        self.R = R
        self.theta = theta
        self.K = K or max(P_coeffs.keys())
        self.n_quad = n_quad
        self.label = label

        # Validate
        for ell in range(1, self.K + 1):
            if ell not in P_coeffs:
                raise ValueError(f"Missing P_{ell} coefficients for K={self.K}")

        # Build engine (currently only supports K=3)
        if self.K == 3:
            self.engine = KappaEngine(
                P1_coeffs=P_coeffs[1],
                P2_coeffs=P_coeffs[2],
                P3_coeffs=P_coeffs[3],
                Q_coeffs=Q_coeffs,
                theta=theta,
                K=self.K,
                R=R,
                n_quad=n_quad,
            )
        else:
            # Placeholder for K=4+ engine
            self.engine = None

        # Cache
        self._kappa_result: Optional[KappaResult] = None
        self._pair_values: Optional[Dict[str, float]] = None

    def compute(self) -> KappaResult:
        """Run the full κ computation."""
        if self.engine is None:
            raise NotImplementedError(f"K={self.K} engine not yet implemented")
        if self._kappa_result is None:
            self._kappa_result = self.engine.compute_kappa()
        return self._kappa_result

    def compute_pair_breakdown(self) -> Dict[str, float]:
        """Get per-pair S12 contributions."""
        if self._pair_values is not None:
            return self._pair_values

        if self.K == 3:
            # Use unified evaluator for K=3
            from src.unified_s12_evaluator_v3 import compute_S12_unified_v3

            polynomials = {
                'P1': P1Polynomial(tilde_coeffs=np.array(self.P_coeffs[1])),
                'P2': PellPolynomial(tilde_coeffs=np.array(self.P_coeffs[2])),
                'P3': PellPolynomial(tilde_coeffs=np.array(self.P_coeffs[3])),
                'Q': Polynomial(coeffs=np.array(self.Q_coeffs)),
            }
            result = compute_S12_unified_v3(
                R=self.R,
                theta=self.theta,
                polynomials=polynomials,
                n_quad_u=self.n_quad,
                n_quad_t=self.n_quad,
            )
            self._pair_values = result.pair_contributions
        else:
            # Placeholder for K=4+
            self._pair_values = {
                f"{ell1}{ell2}": 0.0
                for ell1, ell2 in get_all_pairs(self.K)
            }

        return self._pair_values

    def run_quadrature_check(self, n_values: List[int] = None) -> QuadratureCheck:
        """Check that results are stable under quadrature refinement."""
        if n_values is None:
            n_values = [40, 60, 80, 100]

        kappa_values = []
        c_values = []

        for n in n_values:
            if self.K == 3:
                engine = KappaEngine(
                    P1_coeffs=self.P_coeffs[1],
                    P2_coeffs=self.P_coeffs[2],
                    P3_coeffs=self.P_coeffs[3],
                    Q_coeffs=self.Q_coeffs,
                    theta=self.theta,
                    K=self.K,
                    R=self.R,
                    n_quad=n,
                )
                result = engine.compute_kappa()
                kappa_values.append(result.kappa)
                c_values.append(result.c)
            else:
                kappa_values.append(0.0)
                c_values.append(0.0)

        # Compute max variation
        if len(kappa_values) > 1:
            max_var = max(abs(k - kappa_values[-1]) / kappa_values[-1] * 100
                          for k in kappa_values)
        else:
            max_var = 0.0

        return QuadratureCheck(
            n_values=n_values,
            kappa_values=kappa_values,
            c_values=c_values,
            max_variation_pct=max_var,
            passed=max_var < 0.01,  # 0.01% threshold
        )

    def run_r_sweep(
        self,
        baseline_P_coeffs: Dict[int, List[float]],
        R_values: List[float] = None,
    ) -> RSweepCheck:
        """Check that improvement persists across R values."""
        if R_values is None:
            R_values = [1.1, 1.2, 1.3036, 1.35, 1.4]

        kappa_baseline = []
        kappa_optimized = []
        improvements = []

        for R in R_values:
            if self.K == 3:
                # Baseline
                engine_base = KappaEngine(
                    P1_coeffs=baseline_P_coeffs[1],
                    P2_coeffs=baseline_P_coeffs[2],
                    P3_coeffs=baseline_P_coeffs[3],
                    Q_coeffs=self.Q_coeffs,
                    theta=self.theta,
                    K=self.K,
                    R=R,
                    n_quad=self.n_quad,
                )
                k_base = engine_base.compute_kappa().kappa

                # Optimized
                engine_opt = KappaEngine(
                    P1_coeffs=self.P_coeffs[1],
                    P2_coeffs=self.P_coeffs[2],
                    P3_coeffs=self.P_coeffs[3],
                    Q_coeffs=self.Q_coeffs,
                    theta=self.theta,
                    K=self.K,
                    R=R,
                    n_quad=self.n_quad,
                )
                k_opt = engine_opt.compute_kappa().kappa

                kappa_baseline.append(k_base)
                kappa_optimized.append(k_opt)
                improvements.append((k_opt / k_base - 1) * 100)
            else:
                kappa_baseline.append(0.0)
                kappa_optimized.append(0.0)
                improvements.append(0.0)

        all_positive = all(imp > 0 for imp in improvements)

        return RSweepCheck(
            R_values=R_values,
            kappa_baseline=kappa_baseline,
            kappa_optimized=kappa_optimized,
            improvements_pct=improvements,
            all_positive=all_positive,
            passed=all_positive,
        )

    def generate_report(
        self,
        include_quadrature: bool = False,
        include_r_sweep: bool = False,
        baseline_P_coeffs: Optional[Dict[int, List[float]]] = None,
    ) -> DerivationReport:
        """Generate complete derivation report."""
        result = self.compute()
        pair_values = self.compute_pair_breakdown()

        # Build pair contributions
        pairs = {}
        total = sum(pair_values.values())

        for pair_key, value in pair_values.items():
            ell1, ell2 = int(pair_key[0]), int(pair_key[1])
            omega1 = compute_omega(ell1)
            omega2 = compute_omega(ell2)
            case_type = get_case_type(ell1, ell2)

            pairs[pair_key] = PairContribution(
                ell1=ell1,
                ell2=ell2,
                omega_ell1=omega1,
                omega_ell2=omega2,
                case_type=case_type,
                value=value,
                pct_of_total=(value / total * 100) if total != 0 else 0,
            )

        # Optional validations
        quad_check = None
        r_sweep = None

        if include_quadrature:
            quad_check = self.run_quadrature_check()

        if include_r_sweep and baseline_P_coeffs:
            r_sweep = self.run_r_sweep(baseline_P_coeffs)

        return DerivationReport(
            label=self.label,
            timestamp=datetime.now().isoformat(),
            R=self.R,
            theta=self.theta,
            theta_exact="4/7" if abs(self.theta - 4/7) < 1e-10 else str(self.theta),
            K=self.K,
            n_pairs=count_pairs(self.K),
            P_coeffs={k: list(v) for k, v in self.P_coeffs.items()},
            Q_coeffs=list(self.Q_coeffs),
            kappa=result.kappa,
            c=result.c,
            S12_total=total,
            pairs=pairs,
            S12_plus=result.integrals.S12_plus,
            S12_minus=result.integrals.S12_minus,
            S34_plus=result.integrals.S34_plus,
            m=result.corrections.m,
            g_I1=result.corrections.g_I1,
            g_I2=result.corrections.g_I2,
            quadrature_check=quad_check,
            r_sweep_check=r_sweep,
        )

    # =========================================================================
    # LATEX GENERATION
    # =========================================================================

    def generate_latex_polynomials(self) -> str:
        """Generate LaTeX for polynomial definitions."""
        lines = []
        lines.append(r"\begin{align*}")

        for ell in range(1, self.K + 1):
            coeffs = self.P_coeffs[ell]
            if ell == 1:
                # P₁ uses (1-x) basis
                terms = ["x"]
                for i, c in enumerate(coeffs):
                    sign = "+" if c >= 0 else "-"
                    terms.append(f"{sign} {abs(c):.6f} x(1-x)^{{{i+1}}}")
                line = f"P_{ell}(x) &= {' '.join(terms)}"
            else:
                # P_ℓ uses monomial basis with x factor
                terms = []
                for i, c in enumerate(coeffs):
                    if i == 0:
                        terms.append(f"{c:.6f} x")
                    else:
                        sign = "+" if c >= 0 else "-"
                        terms.append(f"{sign} {abs(c):.6f} x^{{{i+1}}}")
                line = f"P_{ell}(x) &= {' '.join(terms)}"

            if ell < self.K:
                line += r" \\"
            lines.append(line)

        lines.append(r"\end{align*}")
        return "\n".join(lines)

    def generate_latex_pair_table(self) -> str:
        """Generate LaTeX table for pair contributions."""
        report = self.generate_report()

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Per-pair contributions for " + self.label + f" (K={self.K})" + "}")
        lines.append(r"\begin{tabular}{cccccc}")
        lines.append(r"\hline")
        lines.append(r"Pair $(\ell_1,\ell_2)$ & $\omega(\ell_1)$ & $\omega(\ell_2)$ & Case & Value & \% \\")
        lines.append(r"\hline")

        for pair_key in sorted(report.pairs.keys()):
            p = report.pairs[pair_key]
            lines.append(
                f"({p.ell1},{p.ell2}) & {p.omega_ell1} & {p.omega_ell2} & "
                f"{p.case_type} & {p.value:.6f} & {p.pct_of_total:.1f}\\% \\\\"
            )

        lines.append(r"\hline")
        lines.append(f"Total & & & & {report.S12_total:.6f} & 100\\% \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_latex_quadrature_table(self) -> str:
        """Generate LaTeX table for quadrature convergence."""
        check = self.run_quadrature_check()

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Quadrature convergence for " + self.label + "}")
        lines.append(r"\begin{tabular}{ccc}")
        lines.append(r"\hline")
        lines.append(r"$n_{\text{quad}}$ & $\kappa$ & $c$ \\")
        lines.append(r"\hline")

        for n, k, c in zip(check.n_values, check.kappa_values, check.c_values):
            lines.append(f"{n} & {k:.8f} & {c:.8f} \\\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        status = r"\checkmark PASSED" if check.passed else r"$\times$ FAILED"
        lines.append(f"\\\\Max variation: {check.max_variation_pct:.6f}\\% ({status})")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_latex_r_sweep_table(self, baseline_P_coeffs: Dict[int, List[float]]) -> str:
        """Generate LaTeX table for R-sweep validation."""
        check = self.run_r_sweep(baseline_P_coeffs)

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{R-sweep validation for " + self.label + "}")
        lines.append(r"\begin{tabular}{cccc}")
        lines.append(r"\hline")
        lines.append(r"$R$ & $\kappa$ (baseline) & $\kappa$ (optimized) & Improvement \\")
        lines.append(r"\hline")

        for R, kb, ko, imp in zip(check.R_values, check.kappa_baseline,
                                   check.kappa_optimized, check.improvements_pct):
            lines.append(f"{R:.4f} & {kb:.6f} & {ko:.6f} & {imp:+.2f}\\% \\\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        status = r"\checkmark PASSED" if check.passed else r"$\times$ FAILED"
        lines.append(f"\\\\All improvements positive: {status}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_full_latex_appendix(
        self,
        include_quadrature: bool = True,
        include_r_sweep: bool = False,
        baseline_P_coeffs: Optional[Dict[int, List[float]]] = None,
    ) -> str:
        """Generate complete LaTeX appendix section."""
        lines = []

        lines.append(f"\\subsection{{Configuration: {self.label} (K={self.K})}}")
        lines.append("")
        lines.append(f"Number of pairs: {count_pairs(self.K)}")
        lines.append("")
        lines.append("\\subsubsection{Polynomial Definitions}")
        lines.append(self.generate_latex_polynomials())
        lines.append("")
        lines.append("\\subsubsection{Per-Pair Contributions}")
        lines.append(self.generate_latex_pair_table())

        if include_quadrature:
            lines.append("")
            lines.append("\\subsubsection{Quadrature Convergence}")
            lines.append(self.generate_latex_quadrature_table())

        if include_r_sweep and baseline_P_coeffs:
            lines.append("")
            lines.append("\\subsubsection{R-Sweep Validation}")
            lines.append(self.generate_latex_r_sweep_table(baseline_P_coeffs))

        # Results
        report = self.generate_report()
        lines.append("")
        lines.append("\\subsubsection{Results}")
        lines.append(r"\begin{align*}")
        lines.append(f"R &= {report.R} \\\\")
        lines.append(f"\\theta &= {report.theta_exact} \\\\")
        lines.append(f"c &= {report.c:.10f} \\\\")
        lines.append(f"\\kappa &= {report.kappa:.10f}")
        lines.append(r"\end{align*}")

        return "\n".join(lines)

    def save_report_json(self, path: Path) -> None:
        """Save report as JSON for reproducibility."""
        report = self.generate_report(include_quadrature=True)

        data = {
            "label": report.label,
            "timestamp": report.timestamp,
            "parameters": {
                "R": report.R,
                "theta": report.theta,
                "theta_exact": report.theta_exact,
                "K": report.K,
                "n_pairs": report.n_pairs,
            },
            "polynomials": {
                f"P{k}_tilde": v for k, v in report.P_coeffs.items()
            },
            "Q_mono": report.Q_coeffs,
            "results": {
                "kappa": report.kappa,
                "c": report.c,
                "S12_total": report.S12_total,
            },
            "pairs": {
                k: {
                    "ell1": v.ell1,
                    "ell2": v.ell2,
                    "omega_ell1": v.omega_ell1,
                    "omega_ell2": v.omega_ell2,
                    "case_type": v.case_type,
                    "value": v.value,
                    "pct_of_total": v.pct_of_total,
                }
                for k, v in report.pairs.items()
            },
            "mirror_assembly": {
                "S12_plus": report.S12_plus,
                "S12_minus": report.S12_minus,
                "S34_plus": report.S34_plus,
                "m": report.m,
                "g_I1": report.g_I1,
                "g_I2": report.g_I2,
            },
        }

        if report.quadrature_check:
            data["quadrature_check"] = {
                "n_values": report.quadrature_check.n_values,
                "kappa_values": report.quadrature_check.kappa_values,
                "max_variation_pct": report.quadrature_check.max_variation_pct,
                "passed": report.quadrature_check.passed,
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """Process multiple candidate configurations with full validation."""

    def __init__(self, baseline_P_coeffs: Optional[Dict[int, List[float]]] = None):
        self.candidates: Dict[str, DerivationGenerator] = {}
        self.baseline_P_coeffs = baseline_P_coeffs or {
            1: [0.261076, -1.071007, -0.23684, 0.260233],
            2: [1.048274, 1.319912, -0.940058],
            3: [0.522811, -0.68651, -0.049923],
        }
        self.results: Dict[str, DerivationReport] = {}

    def add_candidate(
        self,
        label: str,
        P_coeffs: Dict[int, List[float]],
        Q_coeffs: Optional[List[float]] = None,
        R: float = 1.3036,
        theta: float = 4/7,
        K: int = 3,
    ):
        """Add a candidate configuration."""
        if Q_coeffs is None:
            Q_coeffs = [0.9999989999999999, -0.6378499999999999, -0.6314839999999999,
                        -1.286264, 2.56088, -1.024352]

        self.candidates[label] = DerivationGenerator(
            P_coeffs=P_coeffs,
            Q_coeffs=Q_coeffs,
            R=R,
            theta=theta,
            K=K,
            label=label,
        )

    def add_from_json(self, label: str, path: Path, R: float = 1.3036):
        """Add candidate from JSON file."""
        with open(path) as f:
            data = json.load(f)

        P_coeffs = {}
        if 'P1_tilde' in data:
            P_coeffs[1] = data['P1_tilde']
        if 'P2_tilde' in data:
            P_coeffs[2] = data['P2_tilde']
        if 'P3_tilde' in data:
            P_coeffs[3] = data['P3_tilde']
        if 'P4_tilde' in data:
            P_coeffs[4] = data['P4_tilde']

        Q_coeffs = data.get('Q_mono')

        self.add_candidate(
            label=label,
            P_coeffs=P_coeffs,
            Q_coeffs=Q_coeffs,
            R=R,
            K=max(P_coeffs.keys()),
        )

    def run_all_validations(self, include_r_sweep: bool = True) -> Dict[str, DerivationReport]:
        """Run all validations on all candidates."""
        for label, gen in self.candidates.items():
            self.results[label] = gen.generate_report(
                include_quadrature=True,
                include_r_sweep=include_r_sweep,
                baseline_P_coeffs=self.baseline_P_coeffs,
            )
        return self.results

    def generate_comparison_table(self) -> str:
        """Generate LaTeX comparison table for all candidates."""
        if not self.results:
            self.run_all_validations()

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Comparison of candidate configurations}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\hline")
        lines.append(r"Configuration & $\kappa$ & $c$ & Quad. & R-sweep \\")
        lines.append(r"\hline")

        for label, report in sorted(self.results.items()):
            quad_status = r"\checkmark" if (report.quadrature_check and report.quadrature_check.passed) else "-"
            r_status = r"\checkmark" if (report.r_sweep_check and report.r_sweep_check.passed) else "-"
            lines.append(f"{label} & {report.kappa:.6f} & {report.c:.6f} & {quad_status} & {r_status} \\\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_full_report(self, output_dir: Path) -> None:
        """Generate full report with all appendices."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run validations
        self.run_all_validations()

        # Save individual reports
        for label, gen in self.candidates.items():
            gen.save_report_json(output_dir / f"{label}.json")

        # Generate combined LaTeX
        lines = []
        lines.append(r"\section{Candidate Configuration Comparison}")
        lines.append(self.generate_comparison_table())
        lines.append("")

        for label, gen in self.candidates.items():
            lines.append(gen.generate_full_latex_appendix(
                include_quadrature=True,
                include_r_sweep=True,
                baseline_P_coeffs=self.baseline_P_coeffs,
            ))
            lines.append("")

        with open(output_dir / "appendix.tex", 'w') as f:
            f.write("\n".join(lines))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def from_przz_baseline(K: int = 3) -> DerivationGenerator:
    """Create generator for PRZZ baseline (κ benchmark)."""
    if K == 3:
        return DerivationGenerator(
            P_coeffs={
                1: [0.261076, -1.071007, -0.23684, 0.260233],
                2: [1.048274, 1.319912, -0.940058],
                3: [0.522811, -0.68651, -0.049923],
            },
            Q_coeffs=[0.9999989999999999, -0.6378499999999999, -0.6314839999999999,
                      -1.286264, 2.56088, -1.024352],
            R=1.3036,
            theta=4/7,
            K=3,
            label="PRZZ baseline (κ)",
        )
    else:
        raise NotImplementedError(f"K={K} baseline not yet implemented")


def from_json(path: Path, label: Optional[str] = None) -> DerivationGenerator:
    """Create generator from a candidate JSON file."""
    with open(path) as f:
        data = json.load(f)

    P_coeffs = {}
    for k in range(1, 10):
        key = f'P{k}_tilde'
        if key in data:
            P_coeffs[k] = data[key]

    # Also check old format
    if 'P1_tilde' in data and 1 not in P_coeffs:
        P_coeffs[1] = data['P1_tilde']
    if 'P2_tilde' in data and 2 not in P_coeffs:
        P_coeffs[2] = data['P2_tilde']
    if 'P3_tilde' in data and 3 not in P_coeffs:
        P_coeffs[3] = data['P3_tilde']

    K = max(P_coeffs.keys())

    return DerivationGenerator(
        P_coeffs=P_coeffs,
        Q_coeffs=data.get('Q_mono', data.get('Q_coeffs')),
        R=data.get('R', 1.3036),
        theta=4/7,
        K=K,
        label=label or path.stem,
    )


def compare_configurations(
    baseline: DerivationGenerator,
    optimized: DerivationGenerator,
) -> str:
    """Generate LaTeX comparison between baseline and optimized configurations."""
    base_report = baseline.generate_report()
    opt_report = optimized.generate_report()

    lines = []
    lines.append(r"\subsection{Comparison: " + baseline.label + " vs " + optimized.label + "}")

    kappa_change = (opt_report.kappa / base_report.kappa - 1) * 100
    c_change = (opt_report.c / base_report.c - 1) * 100

    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Quantity & Baseline & Optimized & Change \\")
    lines.append(r"\hline")
    lines.append(f"$\\kappa$ & {base_report.kappa:.6f} & {opt_report.kappa:.6f} & {kappa_change:+.4f}\\% \\\\")
    lines.append(f"$c$ & {base_report.c:.6f} & {opt_report.c:.6f} & {c_change:+.4f}\\% \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Per-pair delta table
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-pair contribution changes}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Pair & Baseline & Optimized & $\Delta$ & Effect \\")
    lines.append(r"\hline")

    for pair_key in sorted(base_report.pairs.keys()):
        if pair_key in opt_report.pairs:
            base_val = base_report.pairs[pair_key].value
            opt_val = opt_report.pairs[pair_key].value
            delta = opt_val - base_val
            effect = r"$\downarrow$ better" if delta < 0 else r"$\uparrow$ worse"
            lines.append(f"({pair_key}) & {base_val:.6f} & {opt_val:.6f} & {delta:+.6f} & {effect} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
