"""
canary_pair_analysis.py
Identify canary pairs: monomials with same (a,b) but different (c_alpha, c_beta, d)

GPT's key question:
"For (2,2), identify two monomials with the same (a,b) but different (cα,cβ,d).
From the TeX, do they map to the same kernel or different kernels?"
"""

from src.psi_expansion import expand_psi, MonomialTwoC
from collections import defaultdict


def find_canary_pairs(ell: int, ellbar: int):
    """Find pairs of monomials with same (a,b) but different full signature."""
    monomials = expand_psi(ell, ellbar)

    # Group by (a, b)
    ab_groups = defaultdict(list)
    for m in monomials:
        ab_groups[(m.a, m.b)].append(m)

    print(f"\n{'='*70}")
    print(f"CANARY PAIR ANALYSIS: ({ell},{ellbar})")
    print(f"{'='*70}")

    canary_pairs = []
    for (a, b), group in sorted(ab_groups.items()):
        if len(group) > 1:
            print(f"\n(a={a}, b={b}): {len(group)} monomials SHARE same (a,b)")
            print("-" * 50)
            for m in group:
                print(f"  Coeff={m.coeff:+3d}, c_α={m.c_alpha}, c_β={m.c_beta}, d={m.d}")
                print(f"    Full key: {m.key()}")
                print(f"    Current lookup: ({a}, {b}, {m.weight_exponent})")

            # Store as canary pair
            canary_pairs.append({
                'ab': (a, b),
                'monomials': [(m.key(), m.coeff) for m in group]
            })

    return canary_pairs


def analyze_canary_kernel_question():
    """
    Key question: Do monomials with same (a,b) but different (c_alpha, c_beta, d)
    map to the SAME kernel or DIFFERENT kernels in PRZZ?
    """
    print("\n" + "="*70)
    print("KERNEL MAPPING QUESTION")
    print("="*70)

    print("""
For (2,2), the clearest canary pairs are:

CANARY 1: (a=0, b=0) - Two monomials, BOTH use integral_grid[(0,0,0)]
  - D²:         (0,0,0,0,2) coeff=+2  <- D is (ζ'/ζ)' paired block
  - C_α²×C_β²:  (0,0,2,2,0) coeff=-1  <- Pure pole factors

CANARY 2: (a=0, b=1) - Two monomials, BOTH use integral_grid[(0,1,1)]
  - B×C_β×D:     (0,1,0,1,1) coeff=-4  <- B singleton + pole + paired
  - B×C_α×C_β²:  (0,1,1,2,0) coeff=+2  <- B singleton + pure poles

CANARY 3: (a=1, b=0) - Two monomials, BOTH use integral_grid[(1,0,1)]
  - A×C_α×D:     (1,0,1,0,1) coeff=-4  <- A singleton + pole + paired
  - A×C_α²×C_β:  (1,0,2,1,0) coeff=+2  <- A singleton + pure poles

QUESTION: In PRZZ's formula, do these pairs use:
  (A) The SAME kernel (just different coefficients)?
  (B) DIFFERENT kernels (requiring different integrand evaluation)?
""")

    print("""
INTERPRETATION from PRZZ Section 7:

From the Ψ expansion structure (psi_expansion.py):
  - A = ζ'/ζ(1+α+s) with z-derivative (singleton x-block)
  - B = ζ'/ζ(1+β+u) with w-derivative (singleton y-block)
  - C_α = pole contribution from 1/ζ(1+α+s)
  - C_β = pole contribution from 1/ζ(1+β+u)
  - D = (ζ'/ζ)'(1+s+u) = mixed derivative paired block

Key insight from PRZZ:
  - A and B are SINGLETON contributions → map to P derivatives
  - D is a PAIRED contribution → maps to paired integral (I₂-type base)
  - C_α and C_β are POLE FACTORS → may be coefficients OR separate kernels

The question is whether C_α/C_β/D all evaluate against the SAME base kernel F(x,y),
or whether they require different kernel constructions.
""")

    print("""
HYPOTHESIS A: C_α/C_β are just coefficient factors
  - The kernel F(x,y) = P_L(u-x)×P_R(u-y)×Q(α)×Q(β)×exp(R(α+β)) is universal
  - C_α/C_β multiply the result, don't change the integral
  - D contributes to the base term (a=b=0, no derivatives)
  - Current implementation would be CORRECT

HYPOTHESIS B: C_α/C_β require pole residue integrals
  - Pole contributions have different integrand structure
  - May involve S(z) zeta ratio factors from PRZZ Section 7
  - Would need separate kernel evaluation per monomial family

EVIDENCE NEEDED:
  - Check PRZZ TeX around lines 2301-2400 for how poles enter
  - Check if S(z) = ζ(1+z)/ζ'(1+z) appears differently for C vs A/B/D
  - Check if Case C auxiliary integral applies to C_α/C_β structures
""")


def show_current_collapsing():
    """Show exactly what values are collapsed in current implementation."""
    from src.polynomials import load_przz_polynomials
    from src.psi_series_evaluator import PsiSeriesEvaluator

    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
    theta = 4.0 / 7.0
    R = 1.3036

    evaluator = PsiSeriesEvaluator(P2, P2, Q, R, theta, max_order=2, n_quad=60)
    integral_grid = evaluator.compute_integral_grid()

    print("\n" + "="*70)
    print("CURRENT COLLAPSING IN (2,2)")
    print("="*70)

    monomials = expand_psi(2, 2)

    # Group by current lookup key
    lookup_groups = defaultdict(list)
    for m in monomials:
        key = (m.a, m.b, m.weight_exponent)
        lookup_groups[key].append(m)

    print("\nMonomials grouped by CURRENT lookup key (a, b, weight):")
    print("-" * 60)

    for key, group in sorted(lookup_groups.items()):
        integral_val = integral_grid.get(key, 0.0)
        print(f"\nLookup key {key} -> integral = {integral_val:.6f}")
        if len(group) > 1:
            print(f"  *** COLLISION: {len(group)} monomials share this integral ***")
        for m in group:
            contrib = m.coeff * integral_val
            print(f"    {m} -> coeff={m.coeff:+3d} -> contrib={contrib:+.6f}")


if __name__ == "__main__":
    canary_pairs = find_canary_pairs(2, 2)
    analyze_canary_kernel_question()
    show_current_collapsing()
