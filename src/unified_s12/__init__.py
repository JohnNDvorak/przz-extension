"""
src/unified_s12/
Phase 45: Exact Mirror Operator Implementation

This package provides first-principles computation of S12 with exact mirror semantics,
eliminating the need for empirical scalar m approximations.

Key modules:
- mirror_transform_exact.py: Exact mirror operator with proper eigenvalue mapping
- mirror_oracle_bruteforce.py: Validation oracle for small Q degrees

Created: 2025-12-27 (Phase 45)
"""

from src.unified_s12.mirror_transform_exact import (
    compute_s12_direct,
    compute_s12_mirror_exact,
    compute_s12_total_exact,
    compute_m_eff,
    compute_c_exact,
)

__all__ = [
    "compute_s12_direct",
    "compute_s12_mirror_exact",
    "compute_s12_total_exact",
    "compute_m_eff",
    "compute_c_exact",
]
