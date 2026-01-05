"""
Constants and default values for the PRZZ Mollifier Explorer.

These values are from PRZZ (arXiv:1802.10521) Section 8 for K=3, d=1 mollifier.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from fractions import Fraction
import functools

# Mathematical constants
THETA = 4/7  # Exactly 4/7
THETA_FRACTION = Fraction(4, 7)
K = 3  # Number of mollifier pieces

# PRZZ baseline R values
R_PRZZ_KAPPA = 1.3036      # For κ (full zeros)
R_PRZZ_KAPPA_STAR = 1.1167  # For κ* (simple critical-line zeros)
R_PRZZ = R_PRZZ_KAPPA       # Default

# Optimized R values (saturation points where c=1)
R_OPTIMIZED_KAPPA = 1.149760231531068  # Saturation point for κ (v16)
R_OPTIMIZED_KAPPA_STAR = 1.07965575130865  # Saturation point for κ* (v16)
R_OPTIMIZED = R_OPTIMIZED_KAPPA   # Default
R_MIN = 0.5
R_MAX = 2.0
R_STEP = 0.001

# ============================================================
# PRZZ κ BASELINE (full zeros on critical line)
# ============================================================
PRZZ_P1_TILDE = [0.261076, -1.071007, -0.236840, 0.260233]
PRZZ_P2_TILDE = [1.048274, 1.319912, -0.940058]
PRZZ_P3_TILDE = [0.522811, -0.686510, -0.049923]
PRZZ_Q_COEFFS = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}

# Target values for κ
KAPPA_TARGET = 0.417293962
C_TARGET = 2.13745440613217263636

# ============================================================
# PRZZ κ* BASELINE (simple critical-line zeros)
# ============================================================
PRZZ_KAPPA_STAR_P1_TILDE = [0.052703, -0.657999, -0.003193, -0.101832]
PRZZ_KAPPA_STAR_P2_TILDE = [1.049837, -0.097446]
PRZZ_KAPPA_STAR_P3_TILDE = [0.035113, -0.156465]
PRZZ_KAPPA_STAR_Q_COEFFS = {0: 0.483777, 1: 0.516223}  # Linear Q

# Target values for κ*
KAPPA_STAR_TARGET = 0.407511457
C_STAR_TARGET = 1.9379524124677437

# ============================================================
# OPTIMIZED CONFIGURATIONS (Our breakthroughs!)
# ============================================================
OPTIMIZED_P1_TILDE = [-2.0, 0.9375, 1.0, -0.6]  # Universal P1!
OPTIMIZED_P2_TILDE = [0.5241, 1.3199, -0.9401]
OPTIMIZED_P3_TILDE = [0.1367, -0.6865, -0.0499]

# Optimized results (explicit finite-height bounds)
OPTIMIZED_KAPPA_MAIN = 1.0000
OPTIMIZED_KAPPA_RIGOROUS = 0.8650
OPTIMIZED_KAPPA_STAR_MAIN = 1.0000
OPTIMIZED_KAPPA_STAR_RIGOROUS = 0.84

# Constraint bounds
CONSTRAINT_BOUNDS = {
    "cap1": (-1.0, 1.0),
    "cap2": (-2.0, 2.0),
    "unbounded": (-10.0, 10.0),
}

# Slider configuration
SLIDER_STEP = 0.001

# Number of coefficients per polynomial (in tilde basis)
N_COEFFS_P1 = 4  # tilde coeffs for P1
N_COEFFS_P2 = 3  # tilde coeffs for P2, P3
N_COEFFS_P3 = 3

# Quadrature settings
QUICK_QUADRATURE = 30  # For live updates
FULL_QUADRATURE = 60   # For full calculation
N_QUAD_MIN = 20        # Minimum quadrature points
N_QUAD_MAX = 100       # Maximum quadrature points
N_QUAD_LIVE_DEFAULT = 40   # Default for live display
N_QUAD_FULL_DEFAULT = 60   # Default for full computation

# Color scheme
COLORS = {
    "P1": "#1f77b4",  # Blue
    "P2": "#2ca02c",  # Green
    "P3": "#d62728",  # Red
    "kappa": "#9467bd",  # Purple
    "c": "#ff7f0e",   # Orange
    "within_cap": "#2ca02c",  # Green
    "exceeds_cap": "#d62728",  # Red
}

# Error source names
ERROR_SOURCES = ["C_contour", "C_Taylor", "C_I5", "C_EM"]
ERROR_ORDERS = {
    "C_contour": "O(T/L)",
    "C_Taylor": "O(T/L)",
    "C_I5": "O(T/L^2)",
    "C_EM": "O(T/L)",
}

# Formulas for display
FORMULAS = {
    "kappa": r"\kappa \ge 1 - \frac{\max(\log c, 0)}{R}",
    "c_assembly": r"c = S_{12}(+R) + m \cdot S_{12}(-R) + S_{34}(+R)",
    "m_formula": r"m = e^R + (2K - 1)",
}


def get_przz_defaults(mode: str = "kappa") -> Dict:
    """Return PRZZ default polynomial coefficients.

    Args:
        mode: "kappa" for full zeros, "kappa_star" for simple critical-line zeros
    """
    if mode == "kappa_star":
        return {
            "P1_tilde": PRZZ_KAPPA_STAR_P1_TILDE.copy(),
            "P2_tilde": PRZZ_KAPPA_STAR_P2_TILDE.copy(),
            "P3_tilde": PRZZ_KAPPA_STAR_P3_TILDE.copy(),
            "Q_coeffs": PRZZ_KAPPA_STAR_Q_COEFFS.copy(),
            "R": R_PRZZ_KAPPA_STAR,
            "theta": THETA,
            "K": K,
        }
    else:  # kappa
        return {
            "P1_tilde": PRZZ_P1_TILDE.copy(),
            "P2_tilde": PRZZ_P2_TILDE.copy(),
            "P3_tilde": PRZZ_P3_TILDE.copy(),
            "Q_coeffs": PRZZ_Q_COEFFS.copy(),
            "R": R_PRZZ_KAPPA,
            "theta": THETA,
            "K": K,
        }


def get_optimized_defaults(mode: str = "kappa") -> Dict:
    """Return our optimized polynomial coefficients.

    Args:
        mode: "kappa" for full zeros, "kappa_star" for simple critical-line zeros
    """
    if mode == "kappa_star":
        return {
            "P1_tilde": OPTIMIZED_P1_TILDE.copy(),
            "P2_tilde": PRZZ_KAPPA_STAR_P2_TILDE.copy(),  # Use PRZZ P2/P3 for κ*
            "P3_tilde": PRZZ_KAPPA_STAR_P3_TILDE.copy(),
            "Q_coeffs": PRZZ_KAPPA_STAR_Q_COEFFS.copy(),
            "R": R_OPTIMIZED_KAPPA_STAR,  # Saturation point for κ* (v16)
            "theta": THETA,
            "K": K,
        }
    else:  # kappa
        return {
            "P1_tilde": OPTIMIZED_P1_TILDE.copy(),
            "P2_tilde": OPTIMIZED_P2_TILDE.copy(),
            "P3_tilde": OPTIMIZED_P3_TILDE.copy(),
            "Q_coeffs": PRZZ_Q_COEFFS.copy(),
            "R": R_OPTIMIZED_KAPPA,  # Saturation point for κ (v16)
            "theta": THETA,
            "K": K,
        }


def coefficient_names(poly_name: str) -> List[str]:
    """Return coefficient names for a polynomial."""
    if poly_name == "P1":
        return [f"a{i}" for i in range(N_COEFFS_P1)]
    elif poly_name in ("P2", "P3"):
        return [f"b{i}" if poly_name == "P2" else f"c{i}" for i in range(N_COEFFS_P2)]
    else:
        raise ValueError(f"Unknown polynomial: {poly_name}")


# ============================================================
# PRECOMPUTED DATA LOADER
# ============================================================

@functools.lru_cache(maxsize=1)
def load_precomputed_data() -> Dict:
    """Load pre-computed results for instant display.

    Returns cached data for PRZZ baseline, optimized configs, R sweep, etc.
    """
    precomputed_path = Path(__file__).parent.parent / "data" / "precomputed.json"
    if precomputed_path.exists():
        with open(precomputed_path, "r") as f:
            return json.load(f)
    return {}


def get_precomputed_result(config_name: str) -> Optional[Dict]:
    """Get pre-computed result for a specific configuration.

    Args:
        config_name: One of "przz_baseline_kappa", "optimized_kappa",
                     "theoretical_limit_kappa", "przz_baseline_kappa_star",
                     "optimized_kappa_star", "theoretical_limit_kappa_star"

    Returns:
        Dict with results or None if not found
    """
    data = load_precomputed_data()
    return data.get(config_name)


def get_r_sweep_data() -> List[Dict]:
    """Get pre-computed R sweep data for instant display."""
    data = load_precomputed_data()
    return data.get("r_sweep_kappa", {}).get("data", [])


def get_asymptotic_data() -> List[Dict]:
    """Get pre-computed asymptotic (L -> infinity) data."""
    data = load_precomputed_data()
    return data.get("asymptotic_data", {}).get("data", [])


def get_historical_milestones() -> List[Dict]:
    """Get historical milestones for timeline display."""
    data = load_precomputed_data()
    return data.get("historical_milestones", [])


def get_pair_contributions() -> Dict:
    """Get per-pair breakdown data."""
    data = load_precomputed_data()
    return data.get("pair_contributions", {})
