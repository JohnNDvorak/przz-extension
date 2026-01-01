"""Visualization components for polynomials, decomposition, and error analysis."""

from .polynomial_plot import render_polynomial_plot
from .decomposition_waterfall import render_decomposition
from .error_breakdown import render_error_breakdown
from .coefficient_amplitude import render_coefficient_amplitude
from .kappa_heatmap import render_sensitivity_heatmap
from .per_pair_breakdown import render_per_pair_breakdown
from .integrals_table import render_integrals_table
from .derivations import render_derivations_tab

__all__ = [
    "render_polynomial_plot",
    "render_decomposition",
    "render_error_breakdown",
    "render_coefficient_amplitude",
    "render_sensitivity_heatmap",
    "render_per_pair_breakdown",
    "render_integrals_table",
    "render_derivations_tab",
]
