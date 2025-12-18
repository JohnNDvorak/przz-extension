"""
src/test_polynomial_coefficient_magnitudes.py
Compare polynomial coefficient magnitudes between κ and κ* benchmarks.

If the κ* polynomials are systematically smaller, this could indicate
a transcription error or a normalization issue.
"""

import json
import numpy as np
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star


def compare_polynomial_magnitudes():
    """Compare polynomial coefficient magnitudes."""

    # Load both sets of polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star(enforce_Q0=True)

    print("\n" + "=" * 70)
    print("POLYNOMIAL COEFFICIENT MAGNITUDES")
    print("=" * 70)

    # P1 comparison
    # Access monomial coefficients
    P1_k_coeffs = P1_k.to_monomial().coeffs
    P1_ks_coeffs = P1_ks.to_monomial().coeffs

    print("\n--- P₁ Monomial Coefficients ---")
    print(f"κ  P₁: {list(P1_k_coeffs)}")
    print(f"κ* P₁: {list(P1_ks_coeffs)}")
    print(f"Max coeff κ:  {max(abs(c) for c in P1_k_coeffs):.6f}")
    print(f"Max coeff κ*: {max(abs(c) for c in P1_ks_coeffs):.6f}")
    print(f"Sum abs κ:    {sum(abs(c) for c in P1_k_coeffs):.6f}")
    print(f"Sum abs κ*:   {sum(abs(c) for c in P1_ks_coeffs):.6f}")

    # P2 comparison
    P2_k_coeffs = P2_k.to_monomial().coeffs
    P2_ks_coeffs = P2_ks.to_monomial().coeffs
    print("\n--- P₂ Monomial Coefficients ---")
    print(f"κ  P₂ deg: {len(P2_k_coeffs)-1}, coeffs: {list(P2_k_coeffs)}")
    print(f"κ* P₂ deg: {len(P2_ks_coeffs)-1}, coeffs: {list(P2_ks_coeffs)}")
    print(f"Max coeff κ:  {max(abs(c) for c in P2_k_coeffs):.6f}")
    print(f"Max coeff κ*: {max(abs(c) for c in P2_ks_coeffs):.6f}")
    print(f"Sum abs κ:    {sum(abs(c) for c in P2_k_coeffs):.6f}")
    print(f"Sum abs κ*:   {sum(abs(c) for c in P2_ks_coeffs):.6f}")
    print(f"Coefficient ratio (sum abs): {sum(abs(c) for c in P2_k_coeffs) / sum(abs(c) for c in P2_ks_coeffs):.4f}")

    # P3 comparison
    P3_k_coeffs = P3_k.to_monomial().coeffs
    P3_ks_coeffs = P3_ks.to_monomial().coeffs
    print("\n--- P₃ Monomial Coefficients ---")
    print(f"κ  P₃ deg: {len(P3_k_coeffs)-1}, coeffs: {list(P3_k_coeffs)}")
    print(f"κ* P₃ deg: {len(P3_ks_coeffs)-1}, coeffs: {list(P3_ks_coeffs)}")
    print(f"Max coeff κ:  {max(abs(c) for c in P3_k_coeffs):.6f}")
    print(f"Max coeff κ*: {max(abs(c) for c in P3_ks_coeffs):.6f}")
    print(f"Sum abs κ:    {sum(abs(c) for c in P3_k_coeffs):.6f}")
    print(f"Sum abs κ*:   {sum(abs(c) for c in P3_ks_coeffs):.6f}")
    print(f"Coefficient ratio (sum abs): {sum(abs(c) for c in P3_k_coeffs) / sum(abs(c) for c in P3_ks_coeffs):.4f}")

    # Q comparison
    Q_k_coeffs = Q_k.to_monomial().coeffs
    Q_ks_coeffs = Q_ks.to_monomial().coeffs
    print("\n--- Q Monomial Coefficients ---")
    print(f"κ  Q deg: {len(Q_k_coeffs)-1}")
    print(f"κ* Q deg: {len(Q_ks_coeffs)-1}")
    print(f"Max coeff κ:  {max(abs(c) for c in Q_k_coeffs):.6f}")
    print(f"Max coeff κ*: {max(abs(c) for c in Q_ks_coeffs):.6f}")

    # Load raw JSON to see the original coefficients
    print("\n" + "=" * 70)
    print("RAW JSON COEFFICIENTS")
    print("=" * 70)

    with open("data/przz_parameters.json") as f:
        params_k = json.load(f)

    with open("data/przz_parameters_kappa_star.json") as f:
        params_ks = json.load(f)

    print("\n--- κ Benchmark (R=1.3036) Raw Coefficients ---")
    for key in ["P1", "P2", "P3"]:
        if key in params_k["polynomials"]:
            tilde = params_k["polynomials"][key].get("tilde_coeffs", [])
            print(f"{key} tilde: {tilde}")

    print("\n--- κ* Benchmark (R=1.1167) Raw Coefficients ---")
    for key in ["P1", "P2", "P3"]:
        if key in params_ks["polynomials"]:
            tilde = params_ks["polynomials"][key].get("tilde_coeffs", [])
            print(f"{key} tilde: {tilde}")

    # Check P(0) values
    print("\n--- P(0) Values ---")
    print(f"κ  P₁(0) = {P1_k.eval(0):.6f}, P₂(0) = {P2_k.eval(0):.6f}, P₃(0) = {P3_k.eval(0):.6f}")
    print(f"κ* P₁(0) = {P1_ks.eval(0):.6f}, P₂(0) = {P2_ks.eval(0):.6f}, P₃(0) = {P3_ks.eval(0):.6f}")

    # Check P(1) values
    print("\n--- P(1) Values ---")
    print(f"κ  P₁(1) = {P1_k.eval(1):.6f}, P₂(1) = {P2_k.eval(1):.6f}, P₃(1) = {P3_k.eval(1):.6f}")
    print(f"κ* P₁(1) = {P1_ks.eval(1):.6f}, P₂(1) = {P2_ks.eval(1):.6f}, P₃(1) = {P3_ks.eval(1):.6f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
If the κ* polynomials are systematically smaller than κ polynomials,
this would explain the ||P||² ratio differences.

Possible explanations:
1. TRANSCRIPTION ERROR: κ* coefficients were divided by some factor
2. PRZZ NORMALIZATION: Polynomials may be defined with different normalization
3. DIFFERENT OPTIMIZATION CONSTRAINTS: κ* optimization found smaller polynomials

Check: If we SCALE κ* polynomials by √(||P_κ||²/||P_κ*||²) for each polynomial,
do the I₂ ratios become consistent?
""")


if __name__ == "__main__":
    compare_polynomial_magnitudes()
