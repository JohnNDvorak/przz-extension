"""
NOLH (Nearly Orthogonal Latin Hypercube) Optimization Module

This module provides systematic multi-parameter optimization for the K=3
polynomial space using Latin Hypercube sampling with statistical analysis.

Modules:
    design.py - NOLH design generation and parameter bounds
    runner.py - Batch execution of design points
    analysis.py - Statistical analysis of results

Created: 2025-12-28 (Phase 49)
"""

from .design import generate_nolh_design, NOLHDesign, get_parameter_bounds
from .runner import evaluate_design_point, run_nolh_batch, NOLHResult
from .analysis import compute_main_effects, fit_response_surface

__all__ = [
    'generate_nolh_design',
    'NOLHDesign',
    'get_parameter_bounds',
    'evaluate_design_point',
    'run_nolh_batch',
    'NOLHResult',
    'compute_main_effects',
    'fit_response_surface',
]
