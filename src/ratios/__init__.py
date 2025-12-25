"""
src/ratios/ - CFZ 4-Shift Ratios Object

This package owns the upstream ratios conjecture machinery:
- FourShifts(α,β,γ,δ) parameter object
- CFZ integrand terms (direct + dual)
- Diagonal specialization (γ=α, δ=β)
- Arithmetic factor A and its derivatives
- J₁ five-piece decomposition for K=3

The paper's mirror logic is born here, then differentiation + diagonal
specialization produces the multi-term structure. The "+5" in m₁ = exp(R) + 5
is likely combinatorial from the five-term J₁ decomposition.

Phase 14 Implementation (2025-12-23)
"""

from src.ratios.cfz_conjecture import FourShifts, CfzTerms, cfz_integrand_terms
from src.ratios.diagonalize import (
    EULER_MASCHERONI,
    zeta_1_plus_eps,
    inv_zeta_1_plus_eps,
    apply_neat_identity,
    diagonalize_gamma_eq_alpha_delta_eq_beta,
)
from src.ratios.arithmetic_factor import (
    primes_up_to,
    A11_prime_sum,
    prime_sum_converges,
)
from src.ratios.j1_k3_decomposition import (
    J1Pieces,
    build_J1_pieces_K3,
    sum_J1,
    count_active_pieces,
)
from src.ratios.microcase_plus5 import (
    microcase_plus5_signature,
    analyze_constant_vs_expR,
    estimate_constant_offset,
)
from src.ratios.bridge_to_S12 import (
    compute_S12_from_J1_pieces_micro,
    decompose_m1_from_pieces,
    analyze_piece_exp_vs_constant,
    get_operator_mirror_piece,
)

__all__ = [
    # CFZ conjecture
    'FourShifts',
    'CfzTerms',
    'cfz_integrand_terms',
    # Diagonalization
    'EULER_MASCHERONI',
    'zeta_1_plus_eps',
    'inv_zeta_1_plus_eps',
    'apply_neat_identity',
    'diagonalize_gamma_eq_alpha_delta_eq_beta',
    # Arithmetic factor
    'primes_up_to',
    'A11_prime_sum',
    'prime_sum_converges',
    # J1 decomposition
    'J1Pieces',
    'build_J1_pieces_K3',
    'sum_J1',
    'count_active_pieces',
    # Microcase +5
    'microcase_plus5_signature',
    'analyze_constant_vs_expR',
    'estimate_constant_offset',
    # Bridge to S12
    'compute_S12_from_J1_pieces_micro',
    'decompose_m1_from_pieces',
    'analyze_piece_exp_vs_constant',
    'get_operator_mirror_piece',
]
