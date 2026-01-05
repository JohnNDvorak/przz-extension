"""
Error breakdown table visualization.

Shows error bounds and polynomial derivative norms.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from ..computation.caching import cached_error_bounds


def create_error_table(error_bounds: Dict) -> pd.DataFrame:
    """
    Create a DataFrame for error breakdown display.

    Args:
        error_bounds: Dict with error values (norms, bounds)

    Returns:
        DataFrame with error information
    """
    if "error" in error_bounds:
        return pd.DataFrame({"Error": [error_bounds["error"]]})

    data = [
        {
            "Metric": "||P1'||_inf (derivative norm)",
            "Value": f"{error_bounds.get('norm_P1', 0):.4f}",
        },
        {
            "Metric": "||P2'||_inf (derivative norm)",
            "Value": f"{error_bounds.get('norm_P2', 0):.4f}",
        },
        {
            "Metric": "||P3'||_inf (derivative norm)",
            "Value": f"{error_bounds.get('norm_P3', 0):.4f}",
        },
        {
            "Metric": "Practical error (at L=40)",
            "Value": f"{error_bounds.get('practical_estimate', 0):.6f}",
        },
        {
            "Metric": "Theoretical upper bound",
            "Value": f"{error_bounds.get('theoretical_bound', 0):.4f}",
        },
    ]

    return pd.DataFrame(data)


def render_error_breakdown(
    result: Optional[Dict],
    coeffs: Optional[Dict] = None,
    R: Optional[float] = None,
    theta: Optional[float] = None,
):
    """
    Render error breakdown table in Streamlit.

    Args:
        result: Dict from full computation with error_bounds key
        coeffs: Current coefficient dict (P1/P2/P3)
        R: Shift parameter
        theta: Mollifier exponent
    """
    if result is None:
        st.info("Click 'Compute Full Result' to see error analysis")
        return

    error_bounds = result.get("error_bounds")
    if error_bounds is None:
        if coeffs is None:
            st.warning("Error bounds not available")
            return
        if R is None:
            R = result.get("R")
        if R is None:
            st.warning("Error bounds require a valid R value")
            return
        if theta is None:
            theta = st.session_state.get("theta", 4 / 7)
        c = result.get("c")
        if c is None:
            st.warning("Error bounds require a valid c value")
            return
        with st.spinner("Computing error bounds..."):
            error_bounds = cached_error_bounds(
                P1_tuple=tuple(coeffs["P1_tilde"]),
                P2_tuple=tuple(coeffs["P2_tilde"]),
                P3_tuple=tuple(coeffs["P3_tilde"]),
                R=R,
                theta=theta,
                c=c,
            )
        if error_bounds is None:
            st.warning("Error bounds not available")
            return

    if "error" in error_bounds:
        st.error(f"Error computing bounds: {error_bounds['error']}")
        return

    # Display table
    st.markdown("**Error Analysis:**")
    df = create_error_table(error_bounds)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Kappa comparison
    st.markdown("**Impact on kappa:**")
    col1, col2, col3 = st.columns(3)

    kappa_main = result.get("kappa", 0)
    kappa_rigorous = result.get("kappa_rigorous")
    if kappa_rigorous is None and "practical_estimate" in error_bounds:
        kappa_rigorous = kappa_main - error_bounds.get("practical_estimate", 0)

    with col1:
        st.metric("kappa (main)", f"{kappa_main:.6f}")

    with col2:
        if kappa_rigorous is not None:
            st.metric("kappa (explicit)", f"{kappa_rigorous:.6f}")
        else:
            st.metric("kappa (explicit)", "N/A")

    with col3:
        if kappa_rigorous is not None:
            gap = kappa_main - kappa_rigorous
            gap_pct = (gap / kappa_main) * 100 if kappa_main != 0 else 0
            st.metric("Gap", f"{gap:.6f} ({gap_pct:.2f}%)")
        else:
            st.metric("Gap", "N/A")

    # Explanation
    st.markdown("---")
    st.markdown("""
    **Error Analysis Notes:**

    - **Derivative norms ||P'||_inf**: Maximum of |P'(x)| on [0,1]. Larger values indicate
      more aggressive polynomial shapes that may impact error bounds.
    - **Practical error**: Conservative estimate at L = log(T) = 40
    - **Theoretical bound**: Very conservative upper bound (assumes worst-case correlations)
    - **Q(0) normalization**: We enforce $q_0 = 1 - \sum_{k \\ge 1} q_k$. Using PRZZ's
      truncated coefficients reproduces their published digits.

    The actual error in validated computations is typically much smaller than these bounds.
    PRZZ reproduction is within 0.0005% (κ) and 0.0004% (κ*).
    """)

    st.markdown("""
    **Scaling note:**
    The paper shows the error contribution entering as
    $(C_{\text{per\_L}}/L + C_{\text{per\_L}^2}/L^2)/(R \\cdot c)$, so smaller $R$ increases the
    explicit gap even when $\\kappa_{\\text{main}}$ improves.
    """)

    st.markdown("""
    **Explicit vs certified:**
    These bounds are explicit numerical evaluations of PRZZ's asymptotic error constants.
    The paper denotes them $\\kappa_{\\text{explicit}}$; the app stores them as kappa_rigorous.
    We reserve **certified** for bounds verified by interval arithmetic.
    """)

    st.markdown("""
    **LMFDB sanity check (paper):**
    - At $T_{\\max} \\approx 5\\times 10^6$, the formula gives $\\kappa \\geq 0.656$
    - This lies within the Platt--Trudgian verified RH height ($3\\times 10^{12}$), where $N_0(T)=N(T)$
    - The ~34% gap indicates conservative error analysis; $\\max |\\Delta_n| = 1.448$ matches known $S(T)$ bounds
    - This validates the formula implementation but is not independent evidence for the main theorems
    """)

    st.markdown("#### Error Source Breakdown at Optimal R")
    source_rows = [
        {"Source": "C_contour", "Constant": "1.723", "Order": "O(T/L)", "Contribution": "43.1%"},
        {"Source": "C_Taylor", "Constant": "3.919", "Order": "O(T/L)", "Contribution": "49.2%"},
        {"Source": "C_I5", "Constant": "1.697", "Order": "O(T/L^2)", "Contribution": "2.1%"},
        {"Source": "C_EM", "Constant": "0.529", "Order": "O(T/L)", "Contribution": "5.6%"},
    ]
    st.table(source_rows)
    st.caption("Total error at L = 40 is ~13.5% of kappa_main (paper table).")

    st.markdown("#### Stability Note")
    st.caption("Stability checks are reported in the paper; this tab focuses on explicit error constants and their impact on the bound.")
