"""
Full calculation orchestration with progress reporting.

Provides step-by-step computation with status updates.
"""

import streamlit as st
from typing import Dict, Optional
import json

from .caching import cached_full_kappa
from .engine_wrapper import StreamlitKappaResult


def run_full_calculation(
    P1_coeffs: list,
    P2_coeffs: list,
    P3_coeffs: list,
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    K: int = 3,
    n_quad: int = 60,
) -> Optional[Dict]:
    """
    Run full κ calculation with progress updates.

    Displays a progress bar and status messages during computation.

    Args:
        P1_coeffs: P1 tilde coefficients
        P2_coeffs: P2 tilde coefficients
        P3_coeffs: P3 tilde coefficients
        Q_coeffs: Q coefficients dict
        R: Shift parameter
        theta: Mollifier exponent
        K: Number of pieces
        n_quad: Quadrature points

    Returns:
        Dict with complete result or None if cancelled
    """
    # Convert to tuples for caching
    P1_tuple = tuple(P1_coeffs)
    P2_tuple = tuple(P2_coeffs)
    P3_tuple = tuple(P3_coeffs)
    Q_json = json.dumps({str(k): v for k, v in Q_coeffs.items()})

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Initializing computation...")
    progress_bar.progress(10)

    status_text.text("Computing I1, I2 integrals at +R and -R...")
    progress_bar.progress(25)

    status_text.text("Computing I3, I4 integrals...")
    progress_bar.progress(40)

    status_text.text("Assembling c and computing kappa...")
    progress_bar.progress(60)

    # Actual computation (cached)
    try:
        result = cached_full_kappa(
            P1_tuple=P1_tuple,
            P2_tuple=P2_tuple,
            P3_tuple=P3_tuple,
            Q_json=Q_json,
            R=R,
            theta=theta,
            K=K,
            n_quad=n_quad,
        )
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Computation failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

    status_text.text("Finalizing results...")
    progress_bar.progress(100)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return result


def display_quick_result(kappa: float, c: float, R: float):
    """
    Display quick result metrics in the UI.

    Args:
        kappa: Computed κ value
        c: Main-term constant
        R: Shift parameter
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="κ",
            value=f"{kappa:.6f}",
            help="Proportion of zeta zeros on critical line"
        )

    with col2:
        st.metric(
            label="c",
            value=f"{c:.6f}",
            help="Main-term constant"
        )

    with col3:
        st.metric(
            label="R",
            value=f"{R:.4f}",
            help="Shift parameter"
        )


def display_full_result(result: Dict):
    """
    Display full computation result with decomposition.

    Args:
        result: Dict from cached_full_kappa
    """
    # Safe access to all values
    kappa = result.get('kappa', 0)
    kappa_rigorous = result.get('kappa_rigorous')
    c = result.get('c', 0)
    m = result.get('m', 0)
    S12_plus = result.get('S12_plus', 0)
    S12_minus = result.get('S12_minus', 0)
    S34 = result.get('S34', 0)

    # Main results
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="κ (main)",
            value=f"{kappa:.6f}",
        )

    with col2:
        if kappa_rigorous is not None:
            st.metric(
                label="κ (rigorous)",
                value=f"{kappa_rigorous:.6f}",
            )
        else:
            st.metric(label="κ (rigorous)", value="N/A")

    with col3:
        st.metric(
            label="c",
            value=f"{c:.6f}",
        )

    with col4:
        st.metric(
            label="m",
            value=f"{m:.4f}",
            help="Mirror multiplier"
        )

    # Decomposition
    st.markdown("**Decomposition:**")
    st.markdown(f"$$c = S_{{12}}(+R) + m \\times S_{{12}}(-R) + S_{{34}}(+R)$$")
    st.markdown(
        f"$$c = {S12_plus:.6f} + {m:.4f} \\times "
        f"{S12_minus:.6f} + {S34:.6f} = {c:.6f}$$"
    )
