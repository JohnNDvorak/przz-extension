"""
src/mirror_transform/spec.py
Phase 31C.1: Mirror Transform Specification

This module defines the data structures for operator-level mirror transforms.

PRZZ MIRROR IDENTITY (TeX 1502-1511):
====================================
The key identity has the form:
    I(α,β) + T^{-(α+β)} · I(-β,-α)

At the PRZZ evaluation point α = β = -R:
- T^{-(α+β)} = T^{2R} = exp(2R/θ) or exp(2R) depending on convention
- The mirror term becomes: exp(2R) · I(R, R)

SWAP TRANSFORM:
==============
The swap (α,β) → (-β,-α) affects:
1. The exponential factor: exp(αx + βy) → exp(-βx - αy)
2. The differential operators: D_α → -D_β, D_β → -D_α (chain rule)
3. Any polynomial arguments that depend on α, β

CHAIN RULE:
==========
If F(α,β) is computed using D_α and D_β, then under swap:
    D_α F(α,β)|_{α→-β} = -D_β F(-β,-α)
    D_β F(α,β)|_{β→-α} = -D_α F(-β,-α)

This sign flip is crucial and is where many earlier attempts failed.

Created: 2025-12-26 (Phase 31C)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MirrorTransformPieces:
    """
    Pieces of an operator-level mirror transform.

    The total integrand has the form:
        total = direct + mirror

    where:
        direct = I(α,β) evaluated at α=β=-R
        mirror = T^{-(α+β)} · I(-β,-α) with proper swap/chain rule

    Attributes:
        direct: The direct term I(α,β) at evaluation point
        mirror: The mirror term with swap and prefactor
        total: direct + mirror
        diagnostics: Additional diagnostic information
    """

    direct: float
    mirror: float
    total: float

    # Detailed breakdown
    diagnostics: Dict

    def __post_init__(self):
        """Verify total matches components."""
        if abs(self.total - (self.direct + self.mirror)) > 1e-10:
            raise ValueError(
                f"total ({self.total}) != direct ({self.direct}) + mirror ({self.mirror})"
            )


@dataclass
class SwapTransformSpec:
    """
    Specification for the (α,β) → (-β,-α) swap transform.

    This encodes how various quantities transform under the swap.
    """

    # Original point
    alpha: float
    beta: float

    # Swapped point
    alpha_swapped: float  # = -beta
    beta_swapped: float   # = -alpha

    # Prefactor from T^{-(α+β)}
    prefactor: float

    # Derivative sign changes from chain rule
    D_alpha_sign: int  # -1 (D_α → -D_β)
    D_beta_sign: int   # -1 (D_β → -D_α)

    @classmethod
    def at_przz_point(cls, R: float, theta: float = 4/7) -> "SwapTransformSpec":
        """Create spec at PRZZ evaluation point α=β=-R."""
        import math

        alpha = -R
        beta = -R

        # Swap: (α,β) → (-β,-α) = (R, R)
        alpha_swapped = -beta  # = R
        beta_swapped = -alpha  # = R

        # T^{-(α+β)} = T^{2R} = exp(2R)
        # Note: some conventions use exp(2R/θ), but exp(2R) is more common
        prefactor = math.exp(2 * R)

        return cls(
            alpha=alpha,
            beta=beta,
            alpha_swapped=alpha_swapped,
            beta_swapped=beta_swapped,
            prefactor=prefactor,
            D_alpha_sign=-1,  # Chain rule: ∂/∂α → -∂/∂β
            D_beta_sign=-1,   # Chain rule: ∂/∂β → -∂/∂α
        )


@dataclass
class ChainRuleResult:
    """
    Result of applying chain rule to derivative operators under swap.

    For the swap (α,β) → (-β,-α):
        ∂/∂α becomes -∂/∂β (evaluated at swapped point)
        ∂/∂β becomes -∂/∂α (evaluated at swapped point)

    For mixed derivatives:
        ∂²/∂α∂β → (-1)·(-1)·∂²/∂β∂α = ∂²/∂α∂β
        (The two minus signs cancel!)

    This is why the mirror term has the SAME sign as direct for I₁.
    """

    # Original derivative order
    d_alpha: int
    d_beta: int

    # Overall sign from chain rule
    sign: int  # (-1)^(d_alpha + d_beta) for the swap

    # What the derivative becomes after swap
    d_alpha_after: int  # = d_beta
    d_beta_after: int   # = d_alpha

    @classmethod
    def for_I1(cls) -> "ChainRuleResult":
        """
        Chain rule for I₁ which has ∂²/∂α∂β.

        Under swap:
            ∂²/∂α∂β → (-1)·(-1)·∂²/∂β∂α = ∂²/∂α∂β

        Sign = (-1)^2 = +1
        """
        return cls(
            d_alpha=1,
            d_beta=1,
            sign=1,  # (-1)^(1+1) = +1
            d_alpha_after=1,  # swapped: was d_beta
            d_beta_after=1,   # swapped: was d_alpha
        )

    @classmethod
    def for_I3(cls) -> "ChainRuleResult":
        """
        Chain rule for I₃ which has ∂/∂α only.

        Under swap:
            ∂/∂α → -∂/∂β

        Sign = (-1)^1 = -1
        """
        return cls(
            d_alpha=1,
            d_beta=0,
            sign=-1,  # (-1)^1 = -1
            d_alpha_after=0,  # swapped: was d_beta = 0
            d_beta_after=1,   # swapped: was d_alpha = 1
        )

    @classmethod
    def for_I4(cls) -> "ChainRuleResult":
        """
        Chain rule for I₄ which has ∂/∂β only.

        Under swap:
            ∂/∂β → -∂/∂α

        Sign = (-1)^1 = -1
        """
        return cls(
            d_alpha=0,
            d_beta=1,
            sign=-1,  # (-1)^1 = -1
            d_alpha_after=1,  # swapped: was d_beta = 1
            d_beta_after=0,   # swapped: was d_alpha = 0
        )
