"""
src/mirror_transform/
Phase 31C: Operator-Level Mirror Transform

This package implements the PRZZ mirror transform at the operator level,
including proper swap and chain rule handling.

The key identity is:
    I(α,β) + T^{-(α+β)} · I(-β,-α)

where the mirror term involves:
1. Swap: (α,β) → (-β,-α)
2. Prefactor: T^{-(α+β)} = exp(2R) at α=β=-R
3. Chain rule: D_α and D_β transform under the swap

Created: 2025-12-26 (Phase 31C)
"""

from .spec import MirrorTransformPieces
from .microcase_11_pq1 import (
    compute_microcase_direct,
    compute_microcase_mirror,
    compute_microcase_total,
)
