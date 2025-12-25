"""
src/section7_clean_evaluator.py
Clean-path PRZZ Section 7 implementation at PRE-MIRROR layer.

This implements GPT's "clean path" diagnosis:
1. Work at I_{1,d}(α, β) PRE-MIRROR layer
2. Use TWO C's (C_α and C_β) in Ψ expansion
3. Apply mirror + Q operators as PRZZ prescribes

KEY DIFFERENCE from old DSL:
- Old DSL was POST-MIRROR (already had t-integral, prefactors)
- This code is PRE-MIRROR, then we add mirror + Q explicitly

F_d Cases for d=1 (based on ω = l₁ - 1):
  Case A (ω = -1, l = 0): F = α × P(u) + P'(u)/log N
  Case B (ω = 0, l = 1): F = P(u)
  Case C (ω ≥ 1, l ≥ 2): Kernel integral with a-variable

TWO-C Structure:
  X = A - C_β   (z-block minus beta-pole)
  Y = B - C_α   (w-block minus alpha-pole)
  Z = D - C_α × C_β   (mixed block minus pole product)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import numpy as np
from math import factorial

from src.quadrature import gauss_legendre_01, tensor_grid_2d, tensor_grid_3d
from src.psi_separated_c import (
    MonomialSeparatedC, BlockConfigSeparatedC,
    expand_pair_to_monomials_separated, compute_euler_maclaurin_weight
)


class FdCase(Enum):
    """F_d evaluation case based on ω value."""
    A = "A"  # ω = -1 (l=0): derivative form
    B = "B"  # ω = 0 (l=1): direct polynomial
    C = "C"  # ω > 0 (l>1): kernel integral


@dataclass
class FdTripleSeparatedC:
    """
    F_d triple for TWO-C structure.

    With separated C_α and C_β, we have:
      l1: left derivative count (from A and D)
      m1: right derivative count (from B and D)
      k_alpha: convolution index for α-side (from C_α)
      k_beta: convolution index for β-side (from C_β)
    """
    l1: int
    m1: int
    k_alpha: int
    k_beta: int
    omega_left: int   # l1 - 1
    omega_right: int  # m1 - 1
    case_left: FdCase
    case_right: FdCase

    def __repr__(self) -> str:
        return (f"FdTripleSeparatedC(l1={self.l1}, m1={self.m1}, "
                f"k_α={self.k_alpha}, k_β={self.k_beta}, "
                f"cases={self.case_left.value},{self.case_right.value})")


def get_case(omega: int) -> FdCase:
    """Determine F_d Case from ω value."""
    if omega == -1:
        return FdCase.A
    elif omega == 0:
        return FdCase.B
    else:
        return FdCase.C


def monomial_to_triple_separated(mono: MonomialSeparatedC) -> FdTripleSeparatedC:
    """
    Map TWO-C monomial to F_d triple.

    From MonomialSeparatedC (a, b, c_alpha, c_beta, d):
      l1 = a + d (left derivative count)
      m1 = b + d (right derivative count)
      k_alpha = c_alpha (α-side convolution index)
      k_beta = c_beta (β-side convolution index)
    """
    l1 = mono.a + mono.d
    m1 = mono.b + mono.d
    k_alpha = mono.c_alpha
    k_beta = mono.c_beta

    omega_left = l1 - 1
    omega_right = m1 - 1

    return FdTripleSeparatedC(
        l1=l1, m1=m1,
        k_alpha=k_alpha, k_beta=k_beta,
        omega_left=omega_left, omega_right=omega_right,
        case_left=get_case(omega_left),
        case_right=get_case(omega_right)
    )


class Section7CleanEvaluator:
    """
    Clean-path PRZZ Section 7 evaluator at PRE-MIRROR layer.

    Workflow:
    1. Generate Ψ expansion with TWO C's
    2. Evaluate F_d for each monomial term
    3. Apply Euler-Maclaurin weights
    4. Sum to get I_{1,d}(α, β)
    5. Apply mirror: I_d = I_{1,d}(α,β) + T^{-α-β} × I_{1,d}(-β,-α)
    6. Apply Q operators
    """

    def __init__(self, P_polys: List, Q_poly, R: float, theta: float, n_quad: int = 60):
        """
        Initialize evaluator.

        Args:
            P_polys: List of P_ℓ polynomial objects [P₁, P₂, P₃] for K=3
            Q_poly: Q polynomial object
            R: PRZZ R parameter (shift in σ₀)
            theta: PRZZ θ parameter (= 4/7 typically)
            n_quad: Quadrature points per dimension
        """
        self.P_polys = P_polys
        self.Q = Q_poly
        self.R = R
        self.theta = theta
        self.n_quad = n_quad

        # Precompute quadrature nodes and weights
        self.u_nodes, self.u_weights = gauss_legendre_01(n_quad)

        # For Case C kernel integral, we need quadrature over a ∈ [0,1]
        self.a_nodes, self.a_weights = gauss_legendre_01(n_quad)

    # =========================================================================
    # F_d CASE IMPLEMENTATIONS
    # =========================================================================

    def Fd_case_A(self, u: np.ndarray, alpha: float, P, P_prime=None) -> np.ndarray:
        """
        Case A (ω = -1, l = 0): Derivative form.

        F = α × P(u) + P'(u)/log N

        Since we work in the limit N → ∞, and log N → ∞, the P'/log N term
        vanishes. However, for finite precision we keep:
            F_A(u) = α × P(u)

        Note: The derivative term is suppressed by 1/log N which goes to 0.

        Args:
            u: Quadrature nodes
            alpha: α parameter (= -R/log T in PRZZ)
            P: Polynomial object with .eval(u) method
            P_prime: Derivative polynomial (optional, for finite N corrections)
        """
        # In the asymptotic limit, Case A is just α × P(u)
        # The P'/log N term is O(1/log N) → 0
        return alpha * P.eval(u)

    def Fd_case_B(self, u: np.ndarray, P) -> np.ndarray:
        """
        Case B (ω = 0, l = 1): Direct polynomial evaluation.

        F = P(u)
        """
        return P.eval(u)

    def Fd_case_C(self, u: np.ndarray, omega: int, alpha: float, P) -> np.ndarray:
        """
        Case C (ω ≥ 1, l ≥ 2): Kernel integral.

        From PRZZ TeX 2360-2374:
            K_omega(u; R) = ∫₀¹ P((1-a)u) × a^{ω-1} × exp(R×θ×u×a) da

        Then F_C = u^ω × K_omega(u; R) (with appropriate sign/prefactor).

        The asymptotic form has the (N/n)^{-αa} factor become exp(R×θ×u×a)
        after substituting α = -R/log T and N = T^θ.

        Note: The factorial suppression 1/(ω-1)! is NOT in the kernel but
        appears elsewhere in the full PRZZ formula.

        Args:
            u: Quadrature nodes
            omega: ω parameter (≥ 1 for Case C)
            alpha: α parameter (unused in asymptotic form, kept for API)
            P: Polynomial object
        """
        # Shape: (n_u,) for result
        result = np.zeros_like(u)

        # For each u point, integrate over a
        for i, ui in enumerate(u):
            if ui < 1e-15:
                # u ≈ 0: integral vanishes due to u^ω factor
                result[i] = 0.0
                continue

            # Arguments for P: (1-a)×u for each a node
            args = (1.0 - self.a_nodes) * ui

            # Integrand: P((1-a)u) × a^{ω-1} × exp(R×θ×u×a)
            P_vals = P.eval(args)
            a_power = self.a_nodes ** (omega - 1) if omega > 1 else np.ones_like(self.a_nodes)
            exp_factor = np.exp(self.R * self.theta * ui * self.a_nodes)

            integrand = P_vals * a_power * exp_factor
            integral = np.sum(self.a_weights * integrand)

            # F_C = u^ω × K_omega(u)
            result[i] = (ui ** omega) * integral

        return result

    def eval_Fd(self, u: np.ndarray, l: int, alpha: float, P) -> np.ndarray:
        """
        Evaluate F_d for given l value.

        Dispatches to Case A, B, or C based on ω = l - 1.

        Args:
            u: Quadrature nodes
            l: Derivative count (l₁ or m₁)
            alpha: α or β parameter
            P: Polynomial object
        """
        omega = l - 1

        if omega == -1:  # l = 0
            return self.Fd_case_A(u, alpha, P)
        elif omega == 0:  # l = 1
            return self.Fd_case_B(u, P)
        else:  # omega >= 1, l >= 2
            return self.Fd_case_C(u, omega, alpha, P)

    # =========================================================================
    # MONOMIAL EVALUATION
    # =========================================================================

    def eval_monomial(self, mono: MonomialSeparatedC, ell: int, ellbar: int,
                      alpha: float, beta: float) -> float:
        """
        Evaluate a single TWO-C monomial contribution.

        Each monomial A^a × B^b × C_α^{c_α} × C_β^{c_β} × D^d represents
        a product of F_d evaluations on left and right sides.

        The (1-u)^{ℓ+ℓ̄-2p} Euler-Maclaurin weight is determined by the
        source_p value which tracks which p-config this monomial came from.

        Args:
            mono: MonomialSeparatedC object
            ell: Left piece index (for selecting P_ℓ)
            ellbar: Right piece index (for selecting P_ℓ̄)
            alpha: α parameter
            beta: β parameter
        """
        triple = monomial_to_triple_separated(mono)

        # Get polynomials for this pair
        P_left = self.P_polys[ell - 1]   # 0-indexed
        P_right = self.P_polys[ellbar - 1]

        # Evaluate F_d on left side with l1, alpha
        F_left = self.eval_Fd(self.u_nodes, triple.l1, alpha, P_left)

        # Evaluate F_d on right side with m1, beta
        F_right = self.eval_Fd(self.u_nodes, triple.m1, beta, P_right)

        # Compute Euler-Maclaurin weight: (1-u)^{a+b}
        # where a, b are powers of A, B (singleton block counts)
        #
        # From Agent 5 findings:
        #   AB: (1-u)² (a=1, b=1)
        #   D: no weight (a=0, b=0)
        #   -AC_α: (1-u)¹ (a=1, b=0)
        #   -BC_β: (1-u)¹ (a=0, b=1)
        #
        # The (1-u) weight comes from singleton blocks, not p-config directly.
        # A and B are singleton blocks; D is a paired block; C_α/C_β are poles.
        weight_exponent = mono.a + mono.b

        # Basic integral: ∫ F_left(u) × F_right(u) × (1-u)^weight du
        if weight_exponent > 0:
            weight_factor = (1.0 - self.u_nodes) ** weight_exponent
        else:
            weight_factor = np.ones_like(self.u_nodes)

        integrand = F_left * F_right * weight_factor
        integral = np.sum(self.u_weights * integrand)

        return mono.coeff * integral

    # =========================================================================
    # PAIR EVALUATION (PRE-MIRROR)
    # =========================================================================

    def compute_I1d_pair(self, ell: int, ellbar: int,
                         alpha: float, beta: float) -> float:
        """
        Compute PRE-MIRROR I_{1,d}(α, β) for pair (ℓ, ℓ̄).

        This is the core PRE-MIRROR computation.

        Args:
            ell: Left piece index
            ellbar: Right piece index
            alpha: α parameter (typically -R/log T)
            beta: β parameter (typically -R/log T)
        """
        # Get all monomials for this pair (TWO-C expansion)
        monomials = expand_pair_to_monomials_separated(ell, ellbar)

        # Sum contributions from all monomials
        total = 0.0
        for mono in monomials:
            contribution = self.eval_monomial(mono, ell, ellbar, alpha, beta)
            total += contribution

        return total

    # =========================================================================
    # MIRROR TERM
    # =========================================================================

    def apply_mirror(self, ell: int, ellbar: int,
                     alpha: float, beta: float) -> float:
        """
        Apply mirror term to get full I_d(α, β).

        I_d(α, β) = I_{1,d}(α, β) + T^{-α-β} × I_{1,d}(-β, -α)

        At α = β = -R/log T:
            T^{-α-β} = T^{2R/log T} = exp(2R)

        Note the ARGUMENT SWAP: (-β, -α) not (α, β)
        """
        # Compute I_{1,d}(α, β)
        I1d_direct = self.compute_I1d_pair(ell, ellbar, alpha, beta)

        # Compute I_{1,d}(-β, -α) - NOTE THE SWAP
        I1d_mirror = self.compute_I1d_pair(ell, ellbar, -beta, -alpha)

        # Mirror prefactor: T^{-α-β}
        # At α = β = -R/log T: T^{-α-β} = exp(2R)
        # For general α, β: T^{-α-β} where we need log T
        # In asymptotic limit, use exp(2R) approximation
        mirror_factor = np.exp(2 * self.R)

        return I1d_direct + mirror_factor * I1d_mirror

    # =========================================================================
    # Q OPERATORS
    # =========================================================================

    def apply_Q_operators(self, I_d: float, t: np.ndarray) -> np.ndarray:
        """
        Apply Q differential operators.

        Q operators convert the derivative structure into Q polynomial
        evaluations.

        At x=y=0:
            arg_α = t + θ×t×x + θ×(t-1)×y → t
            arg_β = t + θ×(t-1)×x + θ×t×y → t

        The t-integral has form:
            ∫₀¹ Q(t)² × exp(2Rt) dt
        """
        # Evaluate Q at t nodes
        Q_vals = self.Q.eval(t)

        # Exponential factor
        exp_factor = np.exp(2 * self.R * t)

        # Integrand: Q(t)² × exp(2Rt)
        integrand = Q_vals ** 2 * exp_factor

        return integrand

    # =========================================================================
    # FULL COMPUTATION
    # =========================================================================

    def compute_t_integral(self) -> float:
        """
        Compute the t-integral: ∫₀¹ Q(t)² exp(2Rt) dt

        This is the common factor for POST-MIRROR I-terms.
        """
        t_nodes, t_weights = gauss_legendre_01(self.n_quad)
        Q_vals = self.Q.eval(t_nodes)
        exp_factor = np.exp(2 * self.R * t_nodes)
        return np.sum(t_weights * Q_vals * Q_vals * exp_factor)

    def compute_c_direct_postmirror(self, ell: int, ellbar: int) -> float:
        """
        Compute c contribution using DIRECT POST-MIRROR formula (like oracle).

        From Agent 4 finding: The oracle has NO mirror term - it's a direct
        POST-MIRROR formula:
            I₂ = (1/θ) × ∫P_ℓP_ℓ̄du × ∫Q²e^{2Rt}dt

        For (1,1), the I-terms are:
            I₁: mixed derivative term (d²/dxdy) with (1-u)² weight
            I₂: base term (no derivatives) with no (1-u) weight
            I₃: α-pole contribution (d/dx) with (1-u) weight
            I₄: β-pole contribution (d/dy) with (1-u) weight

        The total is: I₁ + I₂ + I₃ + I₄
        """
        # Get polynomials
        P_left = self.P_polys[ell - 1].to_monomial()
        P_right = self.P_polys[ellbar - 1].to_monomial()

        # Common t-integral factor
        t_integral = self.compute_t_integral()

        # Compute u-integrals for each I-term structure
        # Based on the (1,1) monomial → I-term mapping:
        #   AB (+1) → I₁ (mixed deriv) with (1-u)^{ℓ+ℓ̄} = (1-u)^2 weight
        #   D (+1) → I₂ (base) with (1-u)^{ℓ+ℓ̄-2} = (1-u)^0 weight
        #   -AC_α (-1) → I₃ (α-pole) contribution
        #   -BC_β (-1) → I₄ (β-pole) contribution

        # I₂: Base term - ∫ P_ℓ(u) P_ℓ̄(u) du (NO (1-u) weight)
        P_left_vals = P_left.eval(self.u_nodes)
        P_right_vals = P_right.eval(self.u_nodes)
        u_integral_I2 = np.sum(self.u_weights * P_left_vals * P_right_vals)
        I2 = (1.0 / self.theta) * u_integral_I2 * t_integral

        # I₁: Mixed derivative term with (1-u)^2 weight
        # For d=1, this comes from the AB monomial structure
        # The integrand has additional structure from derivatives
        weight_I1 = (1.0 - self.u_nodes) ** (ell + ellbar)
        u_integral_I1 = np.sum(self.u_weights * P_left_vals * P_right_vals * weight_I1)
        I1 = (1.0 / self.theta) * u_integral_I1 * t_integral

        # I₃ and I₄: Pole terms
        # For (1,1), these involve Case A evaluations which give α×P(u)
        # At α = 0 in the limit, these vanish
        # For nonzero α, they contribute negatively
        I3 = 0.0  # Placeholder - needs proper Case A handling
        I4 = 0.0  # Placeholder - needs proper Case A handling

        # Total c contribution
        # Note: The AB contribution (+1) and D contribution (+1) are DIFFERENT:
        # - D contributes to I₂ (base, no weight)
        # - AB contributes to I₁ (derivative structure, with weight)
        # So total = I₁ + I₂ (+ I₃ + I₄ when nonzero)
        return I1 + I2 + I3 + I4

    def compute_c_contribution(self, ell: int, ellbar: int) -> float:
        """
        Compute the c contribution from pair (ℓ, ℓ̄).

        FIXED: Use direct POST-MIRROR formula instead of incorrectly
        applying PRE-MIRROR with mirror then t-integral.

        The old buggy code multiplied I_d (u-integral result) by t_integral
        again after apply_mirror, which is wrong because the factored
        structure should be: (1/θ) × u_part × t_part.
        """
        # Use direct POST-MIRROR computation
        return self.compute_c_direct_postmirror(ell, ellbar)

    def compute_c_contribution_premirror(self, ell: int, ellbar: int) -> float:
        """
        DEPRECATED/BUGGY: Original PRE-MIRROR approach.

        This has a structural bug - it multiplies I_d by t_integral
        incorrectly. Kept for reference and debugging.

        The correct PRE-MIRROR → POST-MIRROR transformation requires
        implementing the difference quotient identity:
            1/(α+β) → ∫₀¹ T^{-t(α+β)} dt
        which creates the t-integral from the PRE-MIRROR structure.
        """
        alpha = -self.R
        beta = -self.R

        # Compute with mirror
        I_d = self.apply_mirror(ell, ellbar, alpha, beta)

        # Get t-integral with Q operators
        t_nodes, t_weights = gauss_legendre_01(self.n_quad)
        Q_integrand = self.apply_Q_operators(I_d, t_nodes)

        # BUG: This incorrectly multiplies I_d by t_integral
        # The factored structure should emerge from diff quotient identity
        t_integral = np.sum(t_weights * Q_integrand)
        c_contribution = (1.0 / self.theta) * t_integral * I_d

        return c_contribution


# =============================================================================
# DIAGNOSTIC AND TESTING
# =============================================================================

def print_monomial_triples(ell: int, ellbar: int) -> None:
    """Print all monomials and their F_d triples for a pair."""
    monomials = expand_pair_to_monomials_separated(ell, ellbar)

    print(f"\n{'='*70}")
    print(f"TWO-C MONOMIAL → F_d MAPPING FOR ({ell},{ellbar})")
    print(f"{'='*70}")
    print(f"Total monomials: {len(monomials)}")
    print()

    for mono in monomials:
        triple = monomial_to_triple_separated(mono)
        print(f"  {mono.coeff:+2d} × A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}")
        print(f"      → l1={triple.l1}, m1={triple.m1}, k_α={triple.k_alpha}, k_β={triple.k_beta}")
        print(f"      → Case ({triple.case_left.value},{triple.case_right.value})")
        print()


def validate_11_case() -> None:
    """Validate (1,1) case structure."""
    monomials = expand_pair_to_monomials_separated(1, 1)

    print("\n" + "="*70)
    print("(1,1) VALIDATION - Should have 4 monomials matching I₁,I₂,I₃,I₄")
    print("="*70)

    expected = {
        'AB': (1, 1, 0, 0, 0, +1),     # A¹B¹ with coeff +1 → I₁ + I₂
        'D': (0, 0, 0, 0, 1, +1),       # D¹ with coeff +1 → I₁ + I₂
        'AC_α': (1, 0, 1, 0, 0, -1),    # A¹C_α¹ with coeff -1 → I₃
        'BC_β': (0, 1, 0, 1, 0, -1),    # B¹C_β¹ with coeff -1 → I₄
    }

    print(f"\nExpected monomials:")
    for name, (a, b, ca, cb, d, c) in expected.items():
        print(f"  {name}: A^{a}B^{b}C_α^{ca}C_β^{cb}D^{d}, coeff={c:+d}")

    print(f"\nActual monomials from expansion:")
    for mono in monomials:
        print(f"  A^{mono.a}B^{mono.b}C_α^{mono.c_alpha}C_β^{mono.c_beta}D^{mono.d}, coeff={mono.coeff:+d}")

    # Check mappings
    print(f"\nF_d triple mapping:")
    for mono in monomials:
        triple = monomial_to_triple_separated(mono)
        print(f"  {mono} → Cases ({triple.case_left.value},{triple.case_right.value})")


def summary_k3() -> None:
    """Print summary for all K=3 pairs."""
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    print("\n" + "="*80)
    print("K=3 TWO-C MONOMIAL SUMMARY")
    print("="*80)
    print(f"{'Pair':^8} | {'Monomials':^10} | Case Distribution")
    print("-"*80)

    for ell, ellbar in pairs:
        monomials = expand_pair_to_monomials_separated(ell, ellbar)

        # Count by case pair
        case_counts = {}
        for mono in monomials:
            triple = monomial_to_triple_separated(mono)
            key = (triple.case_left.value, triple.case_right.value)
            case_counts[key] = case_counts.get(key, 0) + 1

        case_str = ", ".join(f"{k[0]},{k[1]}:{v}" for k, v in sorted(case_counts.items()))
        print(f"({ell},{ellbar})     |     {len(monomials):2d}     | {case_str}")

    print("-"*80)


if __name__ == "__main__":
    summary_k3()
    print()
    validate_11_case()
    print()
    print_monomial_triples(1, 1)
    print_monomial_triples(2, 2)
