"""
Kappa sensitivity heatmap visualization.

Shows how kappa varies with two coefficient variations.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def _compute_single_point(args):
    """Compute kappa for a single grid point. Designed for parallel execution."""
    (i, j, x, y, P1_base, P2_base, P3_base, Q_coeffs, R, theta, K,
     vary_idx_1, vary_idx_2) = args

    from ..computation.engine_wrapper import compute_quick_kappa

    P1 = list(P1_base)
    P2 = list(P2_base)
    P3 = list(P3_base)

    if vary_idx_1[0] == "P1":
        P1[vary_idx_1[1]] = x
    elif vary_idx_1[0] == "P2":
        P2[vary_idx_1[1]] = x
    else:
        P3[vary_idx_1[1]] = x

    if vary_idx_2[0] == "P1":
        P1[vary_idx_2[1]] = y
    elif vary_idx_2[0] == "P2":
        P2[vary_idx_2[1]] = y
    else:
        P3[vary_idx_2[1]] = y

    result = compute_quick_kappa(P1, P2, P3, Q_coeffs, R, theta, K, n_quad=30)
    kappa = result.kappa if result.valid else np.nan

    return (i, j, kappa)


def compute_kappa_grid(
    P1_base: List[float],
    P2_base: List[float],
    P3_base: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float,
    K: int,
    vary_idx_1: Tuple[str, int],
    vary_idx_2: Tuple[str, int],
    range_1: Tuple[float, float],
    range_2: Tuple[float, float],
    resolution: int = 15,
    progress_callback=None,
    n_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute kappa on a 2D grid varying two coefficients.

    Uses parallel execution for ~3-4x speedup on multi-core systems.
    """
    x_vals = np.linspace(range_1[0], range_1[1], resolution)
    y_vals = np.linspace(range_2[0], range_2[1], resolution)
    kappa_grid = np.zeros((resolution, resolution))

    total = resolution * resolution

    # Build list of all computation tasks
    tasks = []
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            tasks.append((
                i, j, x, y,
                P1_base, P2_base, P3_base, Q_coeffs,
                R, theta, K, vary_idx_1, vary_idx_2
            ))

    # Execute in parallel with ThreadPoolExecutor
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_single_point, task): task for task in tasks}

        for future in as_completed(futures):
            i, j, kappa = future.result()
            kappa_grid[j, i] = kappa
            completed += 1
            if progress_callback:
                progress_callback(completed / total)

    return x_vals, y_vals, kappa_grid


def create_heatmap(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    kappa_grid: np.ndarray,
    x_label: str,
    y_label: str,
    current_x: float,
    current_y: float,
) -> go.Figure:
    """Create a heatmap figure from computed grid."""
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=kappa_grid,
        colorscale="Viridis",
        colorbar=dict(title="kappa"),
        hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<br>kappa: %{{z:.6f}}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[current_x],
        y=[current_y],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='white')),
        name='Current position',
        hovertemplate=f"Current: ({current_x:.4f}, {current_y:.4f})<extra></extra>",
    ))

    fig.update_layout(
        title="Kappa Sensitivity Heatmap",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=500,
    )

    return fig


def render_sensitivity_heatmap(
    P1_coeffs: List[float],
    P2_coeffs: List[float],
    P3_coeffs: List[float],
    Q_coeffs: Dict[int, float],
    R: float,
    theta: float = 4/7,
    K: int = 3,
):
    """Render sensitivity heatmap with controls."""

    # Initialize session state for heatmap
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None

    st.markdown("**Select coefficients to vary:**")

    coeff_options = []
    coeff_values = {}

    for i, v in enumerate(P1_coeffs):
        label = f"P1.a{i}"
        coeff_options.append(label)
        coeff_values[label] = ("P1", i, v)

    for i, v in enumerate(P2_coeffs):
        label = f"P2.b{i}"
        coeff_options.append(label)
        coeff_values[label] = ("P2", i, v)

    for i, v in enumerate(P3_coeffs):
        label = f"P3.c{i}"
        coeff_options.append(label)
        coeff_values[label] = ("P3", i, v)

    col1, col2 = st.columns(2)

    with col1:
        x_coeff = st.selectbox("X-axis coefficient", coeff_options, index=0, key="heatmap_x_coeff")

    with col2:
        y_coeff = st.selectbox("Y-axis coefficient", coeff_options, index=min(1, len(coeff_options)-1), key="heatmap_y_coeff")

    x_info = coeff_values[x_coeff]
    y_info = coeff_values[y_coeff]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_min = st.number_input("X min", value=float(x_info[2] - 1.0), step=0.1, key="heatmap_x_min")
    with col2:
        x_max = st.number_input("X max", value=float(x_info[2] + 1.0), step=0.1, key="heatmap_x_max")
    with col3:
        y_min = st.number_input("Y min", value=float(y_info[2] - 1.0), step=0.1, key="heatmap_y_min")
    with col4:
        y_max = st.number_input("Y max", value=float(y_info[2] + 1.0), step=0.1, key="heatmap_y_max")

    col_res, col_workers = st.columns(2)
    with col_res:
        resolution = st.slider("Resolution", min_value=5, max_value=25, value=5, key="heatmap_resolution")
    with col_workers:
        n_workers = st.slider("Parallel workers", min_value=1, max_value=8, value=4, key="heatmap_workers")

    # Show time estimate (with parallel speedup)
    n_points = resolution * resolution
    # ~0.5 sec per point with n_quad=30, parallelized
    est_seconds = (n_points * 0.5) / n_workers
    if est_seconds < 60:
        st.caption(f"Grid: {resolution}×{resolution} = {n_points} points × {n_workers} workers. Estimated: **{est_seconds:.0f} seconds**")
    else:
        st.caption(f"Grid: {resolution}×{resolution} = {n_points} points × {n_workers} workers. Estimated: **{est_seconds/60:.1f} minutes**")

    if st.button("Generate Heatmap", width='stretch', key="btn_generate_heatmap"):
        progress_bar = st.progress(0, text="Computing sensitivity grid...")
        status_text = st.empty()

        import time
        start_time = time.time()

        def update_progress(pct):
            elapsed = time.time() - start_time
            if pct > 0:
                remaining = (elapsed / pct) * (1 - pct)
                status_text.text(f"Progress: {pct*100:.0f}% | Elapsed: {elapsed:.0f}s | Remaining: ~{remaining:.0f}s")
            progress_bar.progress(pct)

        x_vals, y_vals, kappa_grid = compute_kappa_grid(
            P1_coeffs, P2_coeffs, P3_coeffs, Q_coeffs, R, theta, K,
            vary_idx_1=(x_info[0], x_info[1]),
            vary_idx_2=(y_info[0], y_info[1]),
            range_1=(x_min, x_max),
            range_2=(y_min, y_max),
            resolution=resolution,
            progress_callback=update_progress,
            n_workers=n_workers,
        )

        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text=f"Complete! ({elapsed:.1f}s)")
        status_text.empty()

        # Store in session state
        st.session_state.heatmap_data = {
            "x_vals": x_vals,
            "y_vals": y_vals,
            "kappa_grid": kappa_grid,
            "x_coeff": x_coeff,
            "y_coeff": y_coeff,
            "current_x": x_info[2],
            "current_y": y_info[2],
        }

    # Display stored heatmap data
    if st.session_state.heatmap_data is not None:
        data = st.session_state.heatmap_data
        fig = create_heatmap(
            data["x_vals"], data["y_vals"], data["kappa_grid"],
            data["x_coeff"], data["y_coeff"],
            data["current_x"], data["current_y"]
        )
        st.plotly_chart(fig, use_container_width=True)

        valid_kappas = data["kappa_grid"][~np.isnan(data["kappa_grid"])]
        if len(valid_kappas) > 0:
            # Find max location
            max_val = np.nanmax(data["kappa_grid"])
            max_idx = np.unravel_index(np.nanargmax(data["kappa_grid"]), data["kappa_grid"].shape)
            max_x = data["x_vals"][max_idx[1]]
            max_y = data["y_vals"][max_idx[0]]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max kappa", f"{max_val:.6f}")
            with col2:
                st.metric("Min kappa", f"{np.min(valid_kappas):.6f}")
            with col3:
                st.metric("Range", f"{np.max(valid_kappas) - np.min(valid_kappas):.6f}")

            # Show optimal point details
            przz_kappa = 0.417293962
            improvement = (max_val - przz_kappa) / przz_kappa * 100
            if improvement > 0:
                st.success(f"**Optimal point found!** {data['x_coeff']} = {max_x:.6f}, {data['y_coeff']} = {max_y:.6f} → κ = {max_val:.6f} ({improvement:+.1f}% vs PRZZ main-term baseline)")
            else:
                st.info(f"**Optimal point found!** {data['x_coeff']} = {max_x:.6f}, {data['y_coeff']} = {max_y:.6f} → κ = {max_val:.6f}")

            # Auto-log to leaderboard if it beats PRZZ by more than 5%
            if improvement > 5:
                from .leaderboard_display import add_current_to_leaderboard
                # Build the optimal configuration
                opt_P1 = list(P1_coeffs)
                opt_P2 = list(P2_coeffs)
                opt_P3 = list(P3_coeffs)

                x_coeff = data["x_coeff"]
                y_coeff = data["y_coeff"]

                if x_coeff.startswith("P1"):
                    idx = int(x_coeff.split(".a")[1])
                    opt_P1[idx] = max_x
                elif x_coeff.startswith("P2"):
                    idx = int(x_coeff.split(".b")[1])
                    opt_P2[idx] = max_x
                elif x_coeff.startswith("P3"):
                    idx = int(x_coeff.split(".c")[1])
                    opt_P3[idx] = max_x

                if y_coeff.startswith("P1"):
                    idx = int(y_coeff.split(".a")[1])
                    opt_P1[idx] = max_y
                elif y_coeff.startswith("P2"):
                    idx = int(y_coeff.split(".b")[1])
                    opt_P2[idx] = max_y
                elif y_coeff.startswith("P3"):
                    idx = int(y_coeff.split(".c")[1])
                    opt_P3[idx] = max_y

                # Compute c for this configuration
                from ..computation.engine_wrapper import compute_quick_kappa
                result = compute_quick_kappa(opt_P1, opt_P2, opt_P3, Q_coeffs, R, theta, K)

                if result.valid:
                    added = add_current_to_leaderboard(
                        kappa=max_val,
                        c=result.c,
                        R=R,
                        P1_tilde=opt_P1,
                        P2_tilde=opt_P2,
                        P3_tilde=opt_P3,
                        Q_coeffs=Q_coeffs,
                        source="heatmap",
                        notes=f"Found via {x_coeff}/{y_coeff} heatmap scan",
                    )
                    if added:
                        st.toast(f"Auto-saved to leaderboard! κ = {max_val:.6f}", icon="🏆")

            # Button to apply optimal values
            if st.button("Apply Optimal Values to Coefficients", key="btn_apply_optimal"):
                # Parse coefficient names to update session state
                x_coeff = data["x_coeff"]
                y_coeff = data["y_coeff"]

                # Update the appropriate coefficient in session state
                if x_coeff.startswith("P1"):
                    idx = int(x_coeff.split(".a")[1])
                    st.session_state.P1_tilde[idx] = max_x
                elif x_coeff.startswith("P2"):
                    idx = int(x_coeff.split(".b")[1])
                    st.session_state.P2_tilde[idx] = max_x
                elif x_coeff.startswith("P3"):
                    idx = int(x_coeff.split(".c")[1])
                    st.session_state.P3_tilde[idx] = max_x

                if y_coeff.startswith("P1"):
                    idx = int(y_coeff.split(".a")[1])
                    st.session_state.P1_tilde[idx] = max_y
                elif y_coeff.startswith("P2"):
                    idx = int(y_coeff.split(".b")[1])
                    st.session_state.P2_tilde[idx] = max_y
                elif y_coeff.startswith("P3"):
                    idx = int(y_coeff.split(".c")[1])
                    st.session_state.P3_tilde[idx] = max_y

                st.rerun()
