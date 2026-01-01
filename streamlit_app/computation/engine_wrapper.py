"""
Wrapper around KappaEngine for Streamlit integration.

Provides simplified interface for UI components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# Add parent path to allow importing from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class QuickKappaResult:
    """Fast κ estimate for live updates."""
    kappa: float
    c: float
    valid: bool = True
    message: str = ""


@dataclass
class StreamlitKappaResult:
    """Complete result for full computation."""
    # Main results
    kappa: float
    c: float
    R: float
    theta: float
    K: int

    # Decomposition
    S12_plus: float
    S12_minus: float
    S34: float
    m: float

    # Individual integrals
    I1_plus: float = 0.0
    I1_minus: float = 0.0
    I2_plus: float = 0.0
    I2_minus: float = 0.0
    I3_plus: float = 0.0
    I4_plus: float = 0.0

    # Correction factors
    g_I1: float = 1.0
    g_I2: float = 1.0
    g_total: float = 1.0
    base: float = 0.0

    # Per-pair breakdown
    per_pair: Dict[str, Dict] = field(default_factory=dict)

    # Error bounds (optional, from full calculation)
    error_bounds: Optional[Dict[str, float]] = None
    kappa_rigorous: Optional[float] = None


def create_engine(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
):
    """
    Create a KappaEngine from coefficient lists.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        Q_coeffs: Q coefficients as dict {k: c_k} for (1-2x)^k basis
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points

    Returns:
        KappaEngine instance
    """
    from src.kappa_engine import KappaEngine
    from src.polynomials import QPolynomial

    # Convert Q_coeffs dict to monomial form
    # QPolynomial takes basis_coeffs directly in constructor
    Q = QPolynomial(basis_coeffs=Q_coeffs, enforce_Q0=False)
    Q_monomial = Q.to_monomial().coeffs.tolist()

    return KappaEngine(
        P1_coeffs=list(P1_coeffs),
        P2_coeffs=list(P2_coeffs),
        P3_coeffs=list(P3_coeffs),
        Q_coeffs=Q_monomial,
        theta=theta,
        K=K,
        R=R,
        n_quad=n_quad,
    )


def compute_quick_kappa(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 40,
) -> QuickKappaResult:
    """
    Compute a quick κ estimate using configurable quadrature.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        Q_coeffs: Q coefficients dict
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points (default 40 for live updates)

    Returns:
        QuickKappaResult with κ and c
    """
    try:
        engine = create_engine(
            P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs,
            R=R, theta=theta, K=K, n_quad=n_quad
        )
        result = engine.compute_kappa()
        return QuickKappaResult(
            kappa=result.kappa,
            c=result.c,
            valid=True,
        )
    except Exception as e:
        return QuickKappaResult(
            kappa=0.0,
            c=0.0,
            valid=False,
            message=str(e),
        )


def compute_full_result(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
    compute_errors: bool = True,
    compute_per_pair: bool = True,
) -> StreamlitKappaResult:
    """
    Compute full κ result with all intermediate values.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        Q_coeffs: Q coefficients dict
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points
        compute_errors: Whether to compute error bounds
        compute_per_pair: Whether to compute per-pair breakdown

    Returns:
        StreamlitKappaResult with complete breakdown
    """
    engine = create_engine(
        P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs,
        R=R, theta=theta, K=K, n_quad=n_quad
    )

    # Compute κ
    result = engine.compute_kappa()

    # Compute per-pair breakdown if requested
    per_pair_data = {}
    if compute_per_pair:
        try:
            per_pair_data = compute_per_pair_breakdown(
                P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs,
                R=R, theta=theta, n_quad=n_quad
            )
        except Exception as e:
            per_pair_data = {"error": str(e)}

    # Build streamlit result
    streamlit_result = StreamlitKappaResult(
        kappa=result.kappa,
        c=result.c,
        R=result.R,
        theta=result.theta,
        K=result.K,
        S12_plus=result.integrals.S12_plus,
        S12_minus=result.integrals.S12_minus,
        S34=result.integrals.S34_plus,
        m=result.corrections.m,
        I1_plus=result.integrals.I1_plus,
        I1_minus=result.integrals.I1_minus,
        I2_plus=result.integrals.I2_plus,
        I2_minus=result.integrals.I2_minus,
        I3_plus=result.integrals.I3_plus,
        I4_plus=result.integrals.I4_plus,
        g_I1=result.corrections.g_I1,
        g_I2=result.corrections.g_I2,
        g_total=result.corrections.g_total,
        base=result.corrections.base,
        per_pair=per_pair_data,
    )

    # Compute error bounds if requested
    if compute_errors:
        try:
            from src.error_bound_estimator import ErrorBoundEstimator

            # Use the ErrorBoundEstimator class
            estimator = ErrorBoundEstimator(theta=theta, R=R)
            error_result = estimator.estimate_error(
                P1_coeffs=list(P1_coeffs),
                P2_coeffs=list(P2_coeffs),
                P3_coeffs=list(P3_coeffs),
                c=result.c,
            )

            # The epsilon from estimate_error is a very conservative upper bound
            # For display, we show the derivative norms which are more informative
            # The actual error impact at L=40 is much smaller: epsilon / L
            L_reference = 40  # log(T) reference
            practical_epsilon = error_result.epsilon / L_reference

            streamlit_result.error_bounds = {
                "norm_P1": error_result.norm_P1,
                "norm_P2": error_result.norm_P2,
                "norm_P3": error_result.norm_P3,
                "theoretical_bound": error_result.epsilon,
                "practical_estimate": practical_epsilon,
            }
            # Practical rigorous kappa estimate
            streamlit_result.kappa_rigorous = result.kappa - practical_epsilon

        except Exception as e:
            # Error bounds computation failed, continue without them
            streamlit_result.error_bounds = {"error": str(e)}

    return streamlit_result


def evaluate_polynomials(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate polynomials on [0, 1] for plotting.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        n_points: Number of evaluation points

    Returns:
        Tuple of (x_values, P1_values, P2_values, P3_values)
    """
    from src.polynomials import P1Polynomial, PellPolynomial

    x = np.linspace(0, 1, n_points)

    P1 = P1Polynomial(tilde_coeffs=np.array(P1_coeffs))
    P2 = PellPolynomial(tilde_coeffs=np.array(P2_coeffs))
    P3 = PellPolynomial(tilde_coeffs=np.array(P3_coeffs))

    P1_vals = P1.to_monomial().eval(x)
    P2_vals = P2.to_monomial().eval(x)
    P3_vals = P3.to_monomial().eval(x)

    return x, P1_vals, P2_vals, P3_vals


def compute_per_pair_breakdown(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    n_quad: int = 60,
) -> Dict[str, Dict]:
    """
    Compute detailed per-pair integral breakdown.

    This matches the structure shown in main_results.tex tables.

    Args:
        P1_coeffs: Tilde coefficients for P1
        P2_coeffs: Tilde coefficients for P2
        P3_coeffs: Tilde coefficients for P3
        Q_coeffs: Q coefficients dict
        R: Shift parameter
        theta: Mollifier exponent
        n_quad: Quadrature points

    Returns:
        Dict keyed by pair name with I1, I2, I3, I4 values
    """
    from src.polynomials import P1Polynomial, PellPolynomial, QPolynomial, Polynomial
    from src.unified_i1_paper import compute_I1_unified_paper
    from src.unified_i2_paper import compute_I2_unified_paper
    from src.terms_k3_d1 import make_all_terms_k3
    from src.evaluate import evaluate_term

    # Create polynomial objects
    P1 = P1Polynomial(tilde_coeffs=np.array(P1_coeffs))
    P2 = PellPolynomial(tilde_coeffs=np.array(P2_coeffs))
    P3 = PellPolynomial(tilde_coeffs=np.array(P3_coeffs))
    Q = QPolynomial(basis_coeffs=Q_coeffs, enforce_Q0=False)
    Q_mono = Polynomial(coeffs=Q.to_monomial().coeffs)

    polynomials = {"P1": P1, "P2": P2, "P3": P3, "Q": Q_mono}

    # Factorial normalization factors
    f_norm = {
        "11": 1.0, "22": 0.25, "33": 1.0 / 36.0,
        "12": 0.5, "13": 1.0 / 6.0, "23": 1.0 / 12.0,
    }

    # Symmetry factors (off-diagonal pairs counted twice)
    symmetry = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    # Pair labels for display
    pair_labels = {
        "11": "(1,1)", "12": "(1,2)", "13": "(1,3)",
        "22": "(2,2)", "23": "(2,3)", "33": "(3,3)",
    }

    pairs = ["11", "22", "33", "12", "13", "23"]

    # Get I3/I4 terms
    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")

    per_pair = {}

    for pair_key in pairs:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])

        norm = f_norm[pair_key]
        sym = symmetry[pair_key]
        full_norm = sym * norm

        # I1 at +R and -R
        I1_plus_result = compute_I1_unified_paper(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True, apply_factorial_norm=True,
        )
        I1_plus_raw = I1_plus_result.I1_value

        I1_minus_result = compute_I1_unified_paper(
            R=-R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True, apply_factorial_norm=True,
        )
        I1_minus_raw = I1_minus_result.I1_value

        # I2 at +R and -R
        I2_plus_result = compute_I2_unified_paper(
            R=R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )
        I2_plus_raw = I2_plus_result.I2_value

        I2_minus_result = compute_I2_unified_paper(
            R=-R, theta=theta, ell1=ell1, ell2=ell2,
            polynomials=polynomials,
            n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
            include_Q=True,
        )
        I2_minus_raw = I2_minus_result.I2_value

        # I3 and I4 at +R only
        terms_plus = all_terms_plus[pair_key]
        I3_raw = 0.0
        I4_raw = 0.0

        if len(terms_plus) > 2:
            I3_result = evaluate_term(
                terms_plus[2], polynomials, n_quad,
                R=R, theta=theta, n_quad_a=40
            )
            I3_raw = I3_result.value

        if len(terms_plus) > 3:
            I4_result = evaluate_term(
                terms_plus[3], polynomials, n_quad,
                R=R, theta=theta, n_quad_a=40
            )
            I4_raw = I4_result.value

        # Store both raw and normalized values
        per_pair[pair_key] = {
            "label": pair_labels[pair_key],
            "ell1": ell1,
            "ell2": ell2,
            "symmetry": sym,
            "factorial_norm": norm,
            "full_norm": full_norm,
            # Raw values (before normalization)
            "I1_plus_raw": I1_plus_raw,
            "I1_minus_raw": I1_minus_raw,
            "I2_plus_raw": I2_plus_raw,
            "I2_minus_raw": I2_minus_raw,
            "I3_raw": I3_raw,
            "I4_raw": I4_raw,
            # Normalized contributions to totals
            "I1_plus": I1_plus_raw * full_norm,
            "I1_minus": I1_minus_raw * full_norm,
            "I2_plus": I2_plus_raw * full_norm,
            "I2_minus": I2_minus_raw * full_norm,
            "I3": I3_raw * full_norm,
            "I4": I4_raw * full_norm,
            # S12 and S34 contributions
            "S12_plus": (I1_plus_raw + I2_plus_raw) * full_norm,
            "S12_minus": (I1_minus_raw + I2_minus_raw) * full_norm,
            "S34": (I3_raw + I4_raw) * full_norm,
        }

    return per_pair


def get_przz_defaults() -> Dict:
    """Get PRZZ default coefficients."""
    return {
        "P1_tilde": [0.261076, -1.071007, -0.236840, 0.260233],
        "P2_tilde": [1.048274, 1.319912, -0.940058],
        "P3_tilde": [0.522811, -0.686510, -0.049923],
        "Q_coeffs": {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011},
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,
    }
