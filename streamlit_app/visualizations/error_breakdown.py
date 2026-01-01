"""
Error breakdown table visualization.

Shows error bounds and polynomial derivative norms.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional


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


def render_error_breakdown(result: Optional[Dict]):
    """
    Render error breakdown table in Streamlit.

    Args:
        result: Dict from full computation with error_bounds key
    """
    if result is None:
        st.info("Click 'Compute Full Result' to see error analysis")
        return

    error_bounds = result.get("error_bounds")
    if error_bounds is None:
        st.warning("Error bounds not available")
        return

    if "error" in error_bounds:
        st.error(f"Error computing bounds: {error_bounds['error']}")
        return

    # Display table
    st.markdown("**Error Analysis:**")
    df = create_error_table(error_bounds)
    st.dataframe(df, width='stretch', hide_index=True)

    # Kappa comparison
    st.markdown("**Impact on kappa:**")
    col1, col2, col3 = st.columns(3)

    kappa_main = result.get("kappa", 0)
    kappa_rigorous = result.get("kappa_rigorous")

    with col1:
        st.metric("kappa (main)", f"{kappa_main:.6f}")

    with col2:
        if kappa_rigorous is not None:
            st.metric("kappa (rigorous)", f"{kappa_rigorous:.6f}")
        else:
            st.metric("kappa (rigorous)", "N/A")

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

    The actual error in validated computations is typically much smaller than these bounds.
    PRZZ baseline achieves ~0.003% precision.
    """)
