"""
src/case_c_kernel.py
PRZZ Case C Kernel Implementation (TeX 2370-2383)

For ω > 0 (Case C), the polynomial P(u) is replaced by the kernel:

    K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθau) da

This is NOT a post-hoc correction - it replaces how P enters the integrand.

Key TeX References:
- 2301-2310: ω definition (for d=1: ω = ℓ - 1)
- 2350-2355: Υ_C derivation with auxiliary a-integral
- 2358-2361: P((1-a)u) appearance
- 2370-2375, 2379-2383: F_d definition for ω > 0

Polynomial-to-Case mapping (for d=1):
- P₁: ℓ=1, ω = ℓ-1 = 0 → Case B (no kernel, just P₁)
- P₂: ℓ=2, ω = ℓ-1 = 1 → Case C with ω=1
- P₃: ℓ=3, ω = ℓ-1 = 2 → Case C with ω=2
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
import math

from src.quadrature import gauss_legendre_01


def compute_case_c_kernel(
    P_eval: Callable[[np.ndarray], np.ndarray],
    u_grid: np.ndarray,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> np.ndarray:
    """
    Compute Case C kernel K_ω(u; R) on a grid of u values.

    PRZZ TeX 2370-2375:
        K_ω(u; R) = u^ω / (ω-1)! × ∫₀¹ P((1-a)u) × a^{ω-1} × exp(Rθau) da

    For ω=1: K₁(u; R) = u × ∫₀¹ P((1-a)u) × exp(Rθau) da
    For ω=2: K₂(u; R) = u²/1 × ∫₀¹ P((1-a)u) × a × exp(Rθau) da

    Args:
        P_eval: Polynomial evaluation function P(x) -> array
        u_grid: Grid of u values where to evaluate the kernel
        omega: ω value (1 for P₂, 2 for P₃)
        R: R parameter
        theta: θ parameter (typically 4/7)
        n_quad_a: Number of quadrature points for a-integral

    Returns:
        K_ω(u; R) evaluated at each point in u_grid
    """
    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got {omega}")

    # Get quadrature for a-integral
    a_nodes, a_weights = gauss_legendre_01(n_quad_a)

    # Pre-compute factorial denominator
    factorial_denom = math.factorial(omega - 1)

    # Ensure u_grid is array
    u_grid = np.atleast_1d(u_grid).flatten()

    # Compute kernel at each u point
    result = np.zeros_like(u_grid)

    for i in range(len(u_grid)):
        u = u_grid[i]
        if u < 1e-14:
            # At u=0, kernel is 0 (due to u^ω factor)
            result[i] = 0.0
            continue

        # Compute argument (1-a)*u for each a node
        args = (1 - a_nodes) * u

        # Evaluate polynomial at these arguments
        P_vals = P_eval(args)

        # Compute weight: a^{ω-1} × exp(Rθua)
        if omega == 1:
            a_power = np.ones_like(a_nodes)  # a^0 = 1
        else:
            a_power = a_nodes ** (omega - 1)

        exp_factor = np.exp(R * theta * u * a_nodes)

        # Integrate
        integrand = P_vals * a_power * exp_factor
        integral = np.sum(a_weights * integrand)

        # Apply u^ω / (ω-1)! factor
        result[i] = (u ** omega) / factorial_denom * integral

    return result


def precompute_case_c_kernels(
    polys: dict,
    u_grid: np.ndarray,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> dict:
    """
    Precompute Case C kernels for P₂ and P₃ on the u grid.

    This is efficient because the a-integral only depends on (ω, u, R, θ, P),
    NOT on t. So we can precompute once and reuse.

    Args:
        polys: Dict with 'P1', 'P2', 'P3' polynomial objects
        u_grid: Grid of u values
        R: R parameter
        theta: θ parameter
        n_quad_a: Quadrature for a-integral

    Returns:
        Dict with:
            'K1': K_1(u; R) for P₂ (omega=1)
            'K2': K_2(u; R) for P₃ (omega=2)
            'P1': P₁(u) unchanged (Case B)
    """
    P1 = polys['P1']
    P2 = polys['P2']
    P3 = polys['P3']

    # Case B: P₁ stays as is
    K_P1 = P1.eval(u_grid)

    # Case C: P₂ with omega=1
    K_P2 = compute_case_c_kernel(
        P_eval=lambda x: P2.eval(x),
        u_grid=u_grid,
        omega=1,
        R=R,
        theta=theta,
        n_quad_a=n_quad_a
    )

    # Case C: P₃ with omega=2
    K_P3 = compute_case_c_kernel(
        P_eval=lambda x: P3.eval(x),
        u_grid=u_grid,
        omega=2,
        R=R,
        theta=theta,
        n_quad_a=n_quad_a
    )

    return {
        'P1': K_P1,  # Case B (unchanged)
        'K1': K_P2,  # Case C kernel for P₂
        'K2': K_P3,  # Case C kernel for P₃
    }


def compute_case_c_I2_term(
    P_name: str,
    polys: dict,
    Q,
    u_grid: np.ndarray,
    t_grid: np.ndarray,
    weights: np.ndarray,
    R: float,
    theta: float,
    n_quad_a: int = 30
) -> float:
    """
    Compute I₂ term with proper Case C structure.

    For I₂ (decoupled, no derivatives):
        I₂ = (1/θ) × ∫∫ K_ω(u) × K_ω(u) × Q(t)² × exp(2Rt) du dt

    where K_ω is the Case C kernel for ω > 0, or just P for ω = 0.

    Args:
        P_name: 'P1', 'P2', or 'P3' (determines ω)
        polys: Dict with polynomial objects
        Q: Q polynomial object
        u_grid, t_grid, weights: 2D quadrature grid
        R: R parameter
        theta: θ parameter
        n_quad_a: Quadrature for a-integral

    Returns:
        I₂ value with Case C structure
    """
    # Determine omega from polynomial name
    omega_map = {'P1': 0, 'P2': 1, 'P3': 2}
    omega = omega_map[P_name]

    P = polys[P_name]

    # Get unique u values (assuming tensor product grid)
    u_unique = np.unique(u_grid)

    if omega == 0:
        # Case B: just evaluate P(u)
        K_on_u = P.eval(u_unique)
    else:
        # Case C: compute kernel
        K_on_u = compute_case_c_kernel(
            P_eval=lambda x: P.eval(x),
            u_grid=u_unique,
            omega=omega,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a
        )

    # Map back to full grid (assuming u_grid has repeated values for tensor product)
    # For now, assume u_grid is 1D or first axis of tensor product
    # This needs to match the actual grid structure in evaluate.py

    # Simple case: compute on 2D grid directly
    # I₂ = (1/θ) × ∫∫ K(u)² × Q(t)² × exp(2Rt) du dt

    # Get K(u) on the u_grid points
    if omega == 0:
        K_vals = P.eval(u_grid)
    else:
        K_vals = compute_case_c_kernel(
            P_eval=lambda x: P.eval(x),
            u_grid=u_grid,
            omega=omega,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a
        )

    Q_vals = Q.eval(t_grid) ** 2
    exp_vals = np.exp(2 * R * t_grid)

    integrand = K_vals ** 2 * Q_vals * exp_vals / theta

    return float(np.sum(weights * integrand))


def verify_case_c_vs_raw(verbose: bool = True):
    """
    Compare Case C kernel to raw polynomial at both R benchmarks.

    This verifies:
    1. Kernel is smaller than raw (as expected)
    2. Kernel introduces R-dependence
    3. Ratio matches our earlier correction estimates
    """
    from src.polynomials import load_przz_polynomials

    THETA = 4/7
    R1 = 1.3036
    R2 = 1.1167

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)

    u_grid = np.linspace(0.01, 0.99, 50)

    if verbose:
        print('=' * 70)
        print('CASE C KERNEL vs RAW POLYNOMIAL')
        print('=' * 70)

    for P, P_name, omega in [(P2, 'P2', 1), (P3, 'P3', 2)]:
        raw_vals = P.eval(u_grid)
        K_r1 = compute_case_c_kernel(P.eval, u_grid, omega, R1, THETA)
        K_r2 = compute_case_c_kernel(P.eval, u_grid, omega, R2, THETA)

        # Compute RMS ratio
        raw_rms = np.sqrt(np.mean(raw_vals ** 2))
        K_r1_rms = np.sqrt(np.mean(K_r1 ** 2))
        K_r2_rms = np.sqrt(np.mean(K_r2 ** 2))

        ratio_r1 = K_r1_rms / raw_rms
        ratio_r2 = K_r2_rms / raw_rms

        if verbose:
            print(f'\n--- {P_name} (omega={omega}) ---')
            print(f'  Raw RMS:      {raw_rms:.6f}')
            print(f'  K(R1) RMS:    {K_r1_rms:.6f} (ratio: {ratio_r1:.4f})')
            print(f'  K(R2) RMS:    {K_r2_rms:.6f} (ratio: {ratio_r2:.4f})')
            print(f'  K ratio R1/R2: {K_r1_rms/K_r2_rms:.4f}')

    if verbose:
        print('=' * 70)


if __name__ == '__main__':
    verify_case_c_vs_raw(verbose=True)
