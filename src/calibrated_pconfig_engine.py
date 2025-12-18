"""
src/calibrated_pconfig_engine.py
Calibrated p-Config Engine Using I-Term Integral Values

KEY INSIGHT from GPT: Blocks (A, B, C, D) should be CONSTANTS calibrated
to the I-term INTEGRAL VALUES, not pointwise integrands.

For (1,1):
  I₁ = ∫∫ (I₁ integrand) = 0.426  (AB term)
  I₂ = ∫∫ (I₂ integrand) = 0.385  (D term)
  |I₃| = ∫∫ (I₃ integrand) = 0.226  (AC term)
  |I₄| = ∫∫ (I₄ integrand) = 0.226  (BC term)

We set CONSTANT blocks such that:
  A × B = I₁
  D = I₂
  A × C = |I₃|
  B × C = |I₄|

Then Ψ = XY + Z integrates to:
  (A-C)(B-C) + (D-C²) = AB + D - AC - BC = I₁ + I₂ - |I₃| - |I₄| = 0.359 ✓

For higher pairs, we use the I-term oracle to get (I₁, I₂, I₃, I₄) for that
polynomial pair, then calibrate blocks accordingly.

PRZZ Reference: arXiv:1802.10521, Section 7
"""

from __future__ import annotations
import numpy as np
from math import factorial, comb, log, exp, sqrt
from typing import Tuple, Dict, NamedTuple
from dataclasses import dataclass


@dataclass
class CalibratedBlocks:
    """Calibrated block values (constants, not functions of u,t)."""
    A: float
    B: float
    C: float
    D: float

    @property
    def X(self) -> float:
        return self.A - self.C

    @property
    def Y(self) -> float:
        return self.B - self.C

    @property
    def Z(self) -> float:
        return self.D - self.C * self.C

    def psi_11(self) -> float:
        """Ψ for (1,1): XY + Z"""
        return self.X * self.Y + self.Z

    def psi_22(self) -> float:
        """Ψ for (2,2): X²Y² + 4XYZ + 2Z²"""
        X, Y, Z = self.X, self.Y, self.Z
        return X*X*Y*Y + 4*X*Y*Z + 2*Z*Z

    def psi_33(self) -> float:
        """Ψ for (3,3): X³Y³ + 9X²Y²Z + 18XYZ² + 6Z³"""
        X, Y, Z = self.X, self.Y, self.Z
        return X**3*Y**3 + 9*X*X*Y*Y*Z + 18*X*Y*Z*Z + 6*Z**3

    def psi_12(self) -> float:
        """Ψ for (1,2): XY² + 2YZ"""
        X, Y, Z = self.X, self.Y, self.Z
        return X*Y*Y + 2*Y*Z

    def psi_13(self) -> float:
        """Ψ for (1,3): XY³ + 3Y²Z"""
        X, Y, Z = self.X, self.Y, self.Z
        return X*Y**3 + 3*Y*Y*Z

    def psi_23(self) -> float:
        """Ψ for (2,3): X²Y³ + 6XY²Z + 6YZ²"""
        X, Y, Z = self.X, self.Y, self.Z
        return X*X*Y**3 + 6*X*Y*Y*Z + 6*Y*Z*Z

    def psi(self, ell: int, ellbar: int) -> float:
        """Evaluate Ψ for any pair."""
        key = (ell, ellbar)
        if key == (1, 1):
            return self.psi_11()
        elif key == (2, 2):
            return self.psi_22()
        elif key == (3, 3):
            return self.psi_33()
        elif key == (1, 2):
            return self.psi_12()
        elif key == (1, 3):
            return self.psi_13()
        elif key == (2, 3):
            return self.psi_23()
        else:
            raise ValueError(f"Unsupported pair {key}")


def calibrate_blocks_from_iterms(I1: float, I2: float, I3: float, I4: float,
                                  symmetric: bool = True) -> CalibratedBlocks:
    """
    Calibrate blocks from I-term integral values.

    For symmetric pairs (A = B), we solve:
      A² = I₁  →  A = sqrt(|I₁|) × sign(I₁)
      A × C = |I₃|  →  C = |I₃| / A
      D = I₂

    For asymmetric pairs, we use:
      A × B = I₁
      A × C = |I₃|
      B × C = |I₄|
      D = I₂

    Returns:
      CalibratedBlocks with A, B, C, D values
    """
    if symmetric:
        # A = B
        if abs(I1) > 1e-15:
            A = sqrt(abs(I1)) * (1 if I1 >= 0 else -1)
        else:
            A = 0.0

        B = A

        if abs(A) > 1e-15:
            C = abs(I3) / A
        else:
            C = 0.0

        D = I2

    else:
        # General case: A ≠ B
        # From AB = I₁, AC = |I₃|, BC = |I₄|:
        # (AC)(BC)/(AB) = C² × (|I₃||I₄|/I₁)
        # C = sqrt(|I₃||I₄|/I₁) if I₁ ≠ 0

        if abs(I1) > 1e-15:
            C_sq = abs(I3) * abs(I4) / abs(I1)
            C = sqrt(C_sq) if C_sq >= 0 else 0.0
        else:
            C = 0.0

        # A = |I₃| / C, B = |I₄| / C
        if abs(C) > 1e-15:
            A = abs(I3) / C
            B = abs(I4) / C
        else:
            # Fallback: use symmetric formula
            A = sqrt(abs(I1)) if I1 >= 0 else 0.0
            B = A

        D = I2

    return CalibratedBlocks(A=A, B=B, C=C, D=D)


class CalibratedPConfigEngine:
    """
    Evaluates pairs using calibrated constant blocks.

    For each pair (ℓ, ℓ̄):
    1. Get I-term values (I₁, I₂, I₃, I₄) from oracle
    2. Calibrate blocks A, B, C, D
    3. Compute Ψ value directly (since blocks are constants)
    4. Return Ψ as the pair contribution
    """

    def __init__(self, P_left, P_right, Q, theta: float, R: float,
                 ell: int, ellbar: int, n_quad: int = 60):
        self.P_left = P_left
        self.P_right = P_right
        self.Q = Q
        self.theta = theta
        self.R = R
        self.ell = ell
        self.ellbar = ellbar
        self.n_quad = n_quad

        # Get I-term values from oracle
        self.I1, self.I2, self.I3, self.I4 = self._compute_iterms()

        # Calibrate blocks
        symmetric = (P_left is P_right)
        self.blocks = calibrate_blocks_from_iterms(
            self.I1, self.I2, self.I3, self.I4, symmetric
        )

    def _compute_iterms(self) -> Tuple[float, float, float, float]:
        """
        Compute I-term values using the validated oracle.

        The przz_22_exact_oracle gives correct I-term values for any polynomial pair.
        This is the validated formula that matches PRZZ for diagonal pairs.
        """
        from src.przz_22_exact_oracle import przz_oracle_22

        # Use the validated oracle
        oracle = przz_oracle_22(self.P_left, self.Q, self.theta, self.R, self.n_quad)

        # For asymmetric pairs, we need to handle differently
        if self.P_left is not self.P_right:
            # For cross-pairs, compute separate I-terms for each side
            # This is an approximation - the true cross-pair formula may differ
            oracle_left = przz_oracle_22(self.P_left, self.Q, self.theta, self.R, self.n_quad)
            oracle_right = przz_oracle_22(self.P_right, self.Q, self.theta, self.R, self.n_quad)

            # Geometric mean approximation for cross terms
            I1 = np.sqrt(oracle_left.I1 * oracle_right.I1) if oracle_left.I1 * oracle_right.I1 >= 0 else 0
            I2 = np.sqrt(oracle_left.I2 * oracle_right.I2)
            I3 = -np.sqrt(abs(oracle_left.I3 * oracle_right.I3))
            I4 = -np.sqrt(abs(oracle_left.I4 * oracle_right.I4))
        else:
            # Symmetric pair - use oracle directly
            I1 = oracle.I1
            I2 = oracle.I2
            I3 = oracle.I3
            I4 = oracle.I4

        return I1, I2, I3, I4

    def eval_pair(self, verbose: bool = False) -> float:
        """
        Evaluate the pair contribution.

        Since blocks are constants, Ψ is a constant, and the integral is just Ψ.
        """
        psi_val = self.blocks.psi(self.ell, self.ellbar)

        if verbose:
            print(f"\nPair ({self.ell},{self.ellbar}):")
            print(f"  I-terms: I₁={self.I1:.6f}, I₂={self.I2:.6f}, "
                  f"I₃={self.I3:.6f}, I₄={self.I4:.6f}")
            print(f"  Blocks: A={self.blocks.A:.6f}, B={self.blocks.B:.6f}, "
                  f"C={self.blocks.C:.6f}, D={self.blocks.D:.6f}")
            print(f"  X={self.blocks.X:.6f}, Y={self.blocks.Y:.6f}, Z={self.blocks.Z:.6f}")
            print(f"  Ψ = {psi_val:.6f}")

        return psi_val


def validate_11():
    """Validate calibrated blocks on (1,1)."""
    from src.polynomials import load_przz_polynomials
    from src.przz_22_exact_oracle import przz_oracle_22

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036
    n_quad = 60

    print("="*70)
    print("CALIBRATED P-CONFIG ENGINE VALIDATION: (1,1)")
    print("="*70)

    # Oracle reference
    oracle = przz_oracle_22(P1, Q, theta, R, n_quad)
    print(f"\nI-term Oracle:")
    print(f"  I₁ = {oracle.I1:.6f}")
    print(f"  I₂ = {oracle.I2:.6f}")
    print(f"  I₃ = {oracle.I3:.6f}")
    print(f"  I₄ = {oracle.I4:.6f}")
    print(f"  Total = {oracle.total:.6f}")

    # Calibrated engine
    engine = CalibratedPConfigEngine(P1, P1, Q, theta, R, 1, 1, n_quad)
    result = engine.eval_pair(verbose=True)

    print(f"\nCalibrated Ψ = {result:.6f}")
    print(f"Oracle total = {oracle.total:.6f}")
    print(f"Ratio = {result/oracle.total:.6f}")

    # Verify algebra
    print("\nAlgebraic verification:")
    A, B, C, D = engine.blocks.A, engine.blocks.B, engine.blocks.C, engine.blocks.D
    print(f"  AB = {A*B:.6f} (should be I₁ = {oracle.I1:.6f})")
    print(f"  D  = {D:.6f} (should be I₂ = {oracle.I2:.6f})")
    print(f"  AC = {A*C:.6f} (should be |I₃| = {abs(oracle.I3):.6f})")
    print(f"  BC = {B*C:.6f} (should be |I₄| = {abs(oracle.I4):.6f})")
    print(f"  AB + D - AC - BC = {A*B + D - A*C - B*C:.6f}")

    if abs(result - oracle.total) < 0.01 * abs(oracle.total):
        print("\n✓ PASSED (within 1%)")
        return True
    else:
        print("\n✗ FAILED")
        return False


def eval_full_k3_calibrated(polys: Dict, theta: float = 4.0/7.0, R: float = 1.3036,
                            n_quad: int = 60, verbose: bool = False) -> Dict:
    """Evaluate all K=3 pairs using calibrated p-config engine."""
    P1, P2, P3, Q = polys['P1'], polys['P2'], polys['P3'], polys['Q']
    poly_map = {1: P1, 2: P2, 3: P3}

    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

    per_pair = {}
    total_c = 0.0

    for (ell, ellbar) in pairs:
        engine = CalibratedPConfigEngine(
            poly_map[ell], poly_map[ellbar], Q,
            theta, R, ell, ellbar, n_quad
        )
        contrib = engine.eval_pair(verbose=verbose)

        # Symmetry factor
        sym_factor = 1 if ell == ellbar else 2
        pair_total = sym_factor * contrib

        per_pair[(ell, ellbar)] = pair_total
        total_c += pair_total

        if not verbose:
            print(f"({ell},{ellbar}): Ψ={contrib:.6f}, sym×{sym_factor} = {pair_total:.6f}")

    kappa = 1 - log(total_c) / R if total_c > 0 else float('nan')

    return {
        'total_c': total_c,
        'kappa': kappa,
        'per_pair': per_pair,
        'R': R,
        'theta': theta
    }


def two_benchmark_test():
    """Test on both κ and κ* benchmarks."""
    from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

    print("\n" + "="*70)
    print("TWO-BENCHMARK TEST")
    print("="*70)

    # Benchmark 1: κ (R=1.3036)
    print("\n--- Benchmark 1: κ (R=1.3036) ---")
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
    result_kappa = eval_full_k3_calibrated(polys_kappa, R=1.3036)
    print(f"\nTotal c = {result_kappa['total_c']:.6f}")
    print(f"Target c = 2.137")
    print(f"Factor = {2.137 / result_kappa['total_c']:.4f}")

    # Benchmark 2: κ* (R=1.1167)
    print("\n--- Benchmark 2: κ* (R=1.1167) ---")
    try:
        P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star(enforce_Q0=True)
        polys_kappa_star = {'P1': P1s, 'P2': P2s, 'P3': P3s, 'Q': Qs}
        result_kappa_star = eval_full_k3_calibrated(polys_kappa_star, R=1.1167)
        print(f"\nTotal c = {result_kappa_star['total_c']:.6f}")
        print(f"Target c = 1.939")
        print(f"Factor = {1.939 / result_kappa_star['total_c']:.4f}")

        # Ratio comparison
        print("\n--- Factor Ratio Comparison ---")
        factor_kappa = 2.137 / result_kappa['total_c']
        factor_kappa_star = 1.939 / result_kappa_star['total_c']
        print(f"κ factor = {factor_kappa:.4f}")
        print(f"κ* factor = {factor_kappa_star:.4f}")
        print(f"Ratio = {factor_kappa / factor_kappa_star:.4f} (target ≈ 1.0)")
    except Exception as e:
        print(f"κ* polynomials not available: {e}")


def main():
    from src.polynomials import load_przz_polynomials

    # Validate on (1,1)
    validate_11()

    # Full K=3 evaluation
    print("\n" + "="*70)
    print("FULL K=3 EVALUATION")
    print("="*70)

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}

    result = eval_full_k3_calibrated(polys, verbose=True)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total c = {result['total_c']:.6f}")
    print(f"Target c = 2.137")
    print(f"Ratio = {result['total_c']/2.137:.4f}")
    print(f"κ = {result['kappa']:.6f}")
    print(f"Target κ = 0.417")

    # Two-benchmark test
    two_benchmark_test()


if __name__ == "__main__":
    main()
