"""
PRZZ Mollifier Explorer - Interactive Streamlit Application

Main entry point for the interactive mollifier polynomial explorer.

Usage:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Zeta Mollifier Explorer",
    page_icon="ζ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import other modules
from streamlit_app.utils.state_management import (
    initialize_state, get_coefficients, get_R, reset_to_przz
)
from streamlit_app.utils.constants import (
    PRZZ_Q_COEFFS, PRZZ_KAPPA_STAR_Q_COEFFS,
    get_przz_defaults, get_optimized_defaults,
    N_QUAD_MIN, N_QUAD_MAX, N_QUAD_LIVE_DEFAULT, N_QUAD_FULL_DEFAULT
)
from streamlit_app.components.coefficient_input import render_all_coefficients
from streamlit_app.components.constraint_mode import render_constraint_mode
from streamlit_app.components.r_parameter import render_r_parameter
from streamlit_app.components.computation_button import render_compute_button, render_reset_button
from streamlit_app.computation.full_calculation import display_quick_result, display_full_result
from streamlit_app.computation.caching import cached_quick_kappa
from streamlit_app.visualizations.polynomial_plot import render_polynomial_plot
from streamlit_app.visualizations.decomposition_waterfall import render_decomposition
from streamlit_app.visualizations.error_breakdown import render_error_breakdown
from streamlit_app.visualizations.coefficient_amplitude import render_coefficient_amplitude
from streamlit_app.visualizations.kappa_heatmap import render_sensitivity_heatmap
from streamlit_app.visualizations.per_pair_breakdown import render_per_pair_breakdown
from streamlit_app.visualizations.integrals_table import render_integrals_table
from streamlit_app.visualizations.derivations import render_derivations_tab
from streamlit_app.visualizations.theorems import render_theorems_tab, render_quick_reference
from streamlit_app.visualizations.derivation_chain import render_derivation_chain
from streamlit_app.visualizations.r_sweep import render_r_sweep_tab
from streamlit_app.visualizations.asymptotic import render_asymptotic_tab
from streamlit_app.visualizations.leaderboard_display import (
    render_leaderboard_sidebar, render_leaderboard_full, render_save_button
)
from streamlit_app.export.latex_export import generate_full_report
from streamlit_app.export.json_export import export_to_json_string

import json


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_state()

    # Header
    st.title("Zeta Mollifier Explorer")
    st.markdown(
        "Based on **Pratt, Robles, Zaharescu & Zeindler**, "
        "*[More Than Five-Twelfths of the Zeros of ζ Are on the Critical Line](https://arxiv.org/abs/1802.10521)* "
        "(2019) — Interactive exploration of mollifier polynomials for computing "
        "$\\kappa$, the proportion of Riemann zeta zeros on the critical line."
    )

    # Sidebar - Input Controls
    with st.sidebar:
        st.header("Configuration")

        # Mode selector (κ vs κ*)
        st.subheader("Computation Mode")
        mode = st.radio(
            "Select metric",
            ["kappa", "kappa_star"],
            format_func=lambda x: "κ (zeros on critical line)" if x == "kappa" else "κ* (simple zeros)",
            key="computation_mode",
            horizontal=False,
        )
        # Clear stale result when mode changes
        if st.session_state.get("mode") != mode:
            st.session_state.last_result = None
        st.session_state.mode = mode

        # Quick preset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load PRZZ", width='stretch'):
                defaults = get_przz_defaults(mode)
                st.session_state.P1_tilde = defaults["P1_tilde"]
                st.session_state.P2_tilde = defaults["P2_tilde"]
                st.session_state.P3_tilde = defaults["P3_tilde"]
                st.session_state.Q_coeffs = defaults["Q_coeffs"]
                st.session_state.R_value = defaults["R"]
                st.session_state.last_result = None  # Clear stale result
                st.rerun()
        with col2:
            if st.button("Load Best", type="primary", width='stretch'):
                defaults = get_optimized_defaults(mode)
                st.session_state.P1_tilde = defaults["P1_tilde"]
                st.session_state.P2_tilde = defaults["P2_tilde"]
                st.session_state.P3_tilde = defaults["P3_tilde"]
                st.session_state.Q_coeffs = defaults["Q_coeffs"]
                st.session_state.R_value = defaults["R"]
                st.session_state.last_result = None  # Clear stale result
                st.rerun()

        st.divider()

        # Constraint mode
        render_constraint_mode()
        st.divider()

        # R parameter
        render_r_parameter()
        st.divider()

        # Advanced settings (n_quad)
        with st.expander("Advanced Settings", expanded=False):
            st.caption("Quadrature precision (higher = slower but more accurate)")
            if "n_quad_live" not in st.session_state:
                st.session_state.n_quad_live = N_QUAD_LIVE_DEFAULT
            st.session_state.n_quad_live = st.slider(
                "Live update precision",
                min_value=N_QUAD_MIN,
                max_value=N_QUAD_MAX,
                value=st.session_state.n_quad_live,
                step=10,
                key="n_quad_live_slider",
                help="Lower values update faster but less accurately"
            )
        st.divider()

        # Polynomial coefficients
        render_all_coefficients()
        st.divider()

        # Buttons
        render_reset_button()

        # Leaderboard in sidebar
        st.divider()
        render_leaderboard_sidebar()

    # Main content area
    coeffs = get_coefficients()
    R = get_R()
    Q_json = json.dumps({str(k): v for k, v in coeffs["Q_coeffs"].items()})

    # Quick kappa computation for live display
    quick_result = cached_quick_kappa(
        P1_tuple=tuple(coeffs["P1_tilde"]),
        P2_tuple=tuple(coeffs["P2_tilde"]),
        P3_tuple=tuple(coeffs["P3_tilde"]),
        Q_json=Q_json,
        R=R,
        theta=st.session_state.theta,
        K=st.session_state.K,
        n_quad=st.session_state.n_quad_live,
    )

    # Live results bar
    st.markdown("### Live Results")
    if quick_result["valid"]:
        display_quick_result(quick_result["kappa"], quick_result["c"], R)
        # Save to leaderboard button
        col_save, col_spacer = st.columns([1, 3])
        with col_save:
            render_save_button(
                kappa=quick_result["kappa"],
                c=quick_result["c"],
                R=R,
                P1_tilde=coeffs["P1_tilde"],
                P2_tilde=coeffs["P2_tilde"],
                P3_tilde=coeffs["P3_tilde"],
                Q_coeffs=coeffs["Q_coeffs"],
                source="manual",
            )
    else:
        st.error(f"Computation error: {quick_result['message']}")

    # Full computation button
    st.divider()
    try:
        result = render_compute_button()
    except Exception as e:
        st.error(f"Full computation error: {e}")
        import traceback
        st.code(traceback.format_exc())
        result = None

    # Visualization tabs
    st.divider()
    tabs = st.tabs([
        "Overview",
        "Theorems",
        "Polynomials",
        "R Sweep",
        "Decomposition",
        "Integrals",
        "Per-Pair",
        "Error Bounds",
        "Sensitivity",
        "Asymptotic",
        "Derivations",
        "Leaderboard"
    ])

    # Tab 0: Overview
    with tabs[0]:
        render_quick_reference()

        st.divider()
        st.markdown("### Current Polynomial Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**P1 (tilde):**")
            st.code(str(coeffs["P1_tilde"]))
        with col2:
            st.markdown("**P2 (tilde):**")
            st.code(str(coeffs["P2_tilde"]))
        with col3:
            st.markdown("**P3 (tilde):**")
            st.code(str(coeffs["P3_tilde"]))

    # Tab 1: Theorems
    with tabs[1]:
        render_theorems_tab()

    # Tab 2: Polynomials
    with tabs[2]:
        try:
            st.markdown("### Polynomial Shapes")
            render_polynomial_plot(
                coeffs["P1_tilde"],
                coeffs["P2_tilde"],
                coeffs["P3_tilde"],
                mode=st.session_state.get("mode", "kappa")
            )
        except Exception as e:
            st.error(f"Polynomials tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 3: R Sweep
    with tabs[3]:
        try:
            render_r_sweep_tab()
        except Exception as e:
            st.error(f"R Sweep tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 4: Decomposition
    with tabs[4]:
        try:
            st.markdown("### c Decomposition")
            render_decomposition(result)
        except Exception as e:
            st.error(f"Decomposition tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 5: Integrals
    with tabs[5]:
        try:
            render_integrals_table(result)
        except Exception as e:
            st.error(f"Integrals tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 6: Per-Pair
    with tabs[6]:
        try:
            render_per_pair_breakdown(result)
        except Exception as e:
            st.error(f"Per-Pair tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 7: Error Bounds
    with tabs[7]:
        try:
            st.markdown("### Error Analysis")
            render_error_breakdown(result)
        except Exception as e:
            st.error(f"Error Bounds tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 8: Sensitivity
    with tabs[8]:
        try:
            st.markdown("### Sensitivity Analysis")
            render_sensitivity_heatmap(
                coeffs["P1_tilde"],
                coeffs["P2_tilde"],
                coeffs["P3_tilde"],
                coeffs["Q_coeffs"],
                R,
                st.session_state.theta,
                st.session_state.K
            )
        except Exception as e:
            st.error(f"Sensitivity tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 9: Asymptotic
    with tabs[9]:
        try:
            render_asymptotic_tab()
        except Exception as e:
            st.error(f"Asymptotic tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 10: Derivations
    with tabs[10]:
        try:
            render_derivations_tab(result, coeffs, R)
        except Exception as e:
            st.error(f"Derivations tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 11: Leaderboard
    with tabs[11]:
        try:
            render_leaderboard_full()
        except Exception as e:
            st.error(f"Leaderboard tab error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Export section
    if result is not None:
        st.divider()
        st.markdown("### Export Results")

        try:
            col1, col2 = st.columns(2)

            with col1:
                latex_report = generate_full_report(
                    coeffs["P1_tilde"],
                    coeffs["P2_tilde"],
                    coeffs["P3_tilde"],
                    coeffs["Q_coeffs"],
                    result,
                    st.session_state.constraint_mode
                )
                st.download_button(
                    label="Download LaTeX Report",
                    data=latex_report,
                    file_name="mollifier_report.tex",
                    mime="text/x-latex",
                    width='stretch',
                )

            with col2:
                json_report = export_to_json_string(
                    coeffs["P1_tilde"],
                    coeffs["P2_tilde"],
                    coeffs["P3_tilde"],
                    coeffs["Q_coeffs"],
                    result,
                    st.session_state.constraint_mode
                )
                st.download_button(
                    label="Download JSON Report",
                    data=json_report,
                    file_name="mollifier_report.json",
                    mime="application/json",
                    width='stretch',
                )
        except Exception as e:
            st.error(f"Export error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Footer
    st.divider()
    st.caption(
        "PRZZ Mollifier Explorer | Based on Pratt-Robles-Zaharescu-Zeindler (2019) | "
        f"kappa = 1 - log(c)/R"
    )


if __name__ == "__main__":
    main()
