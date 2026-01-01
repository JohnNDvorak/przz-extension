#!/usr/bin/env python3
"""
scripts/reproduce_5213_result.py
Phase 48.1: Single-Command Reproducibility Script

This script reproduces the κ = 0.5213 result from scratch.
It serves as the "paper appendix script" - a single command that:
1. Loads optimal_polynomials.json
2. Runs the evaluator in exactly one declared mode
3. Prints c, κ, component decomposition (S12±, S34, m)
4. Prints checksums of all key intermediate scalars
5. Verifies against stored values

Usage:
    python scripts/reproduce_5213_result.py

Created: 2025-12-28 (Phase 48 - Adversarial Verification)
"""

import json
import hashlib
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kappa_engine import KappaEngine


def compute_checksum(values_dict):
    """Compute SHA256 checksum of a dictionary of scalar values."""
    # Create deterministic string representation
    sorted_items = sorted(values_dict.items())
    string_repr = "|".join(f"{k}={v:.15e}" for k, v in sorted_items)
    return hashlib.sha256(string_repr.encode()).hexdigest()[:16]


def main():
    print("=" * 70)
    print("PHASE 48.1: SINGLE-COMMAND REPRODUCIBILITY")
    print("Reproducing κ = 0.5213 Result from Scratch")
    print("=" * 70)

    # ========================================
    # Step 1: Load optimal polynomials
    # ========================================
    print("\n[Step 1] Loading optimal_polynomials.json...")

    data_path = Path(__file__).parent.parent / "data" / "optimal_polynomials.json"
    with open(data_path) as f:
        data = json.load(f)

    print(f"  File: {data_path}")
    print(f"  Loaded successfully")

    # ========================================
    # Step 2: Extract parameters
    # ========================================
    print("\n[Step 2] Extracting parameters...")

    P1_coeffs = data['P1_tilde']
    P2_coeffs = data['P2_tilde']
    P3_coeffs = data['P3_tilde']
    Q_coeffs = data['Q_mono']

    R = 1.3036
    theta = 4/7
    K = 3
    n_quad = 60  # Production setting

    print(f"  R = {R}")
    print(f"  θ = {theta:.10f} (4/7)")
    print(f"  K = {K}")
    print(f"  n_quad = {n_quad}")
    print(f"  P1 coeffs: {P1_coeffs}")
    print(f"  P2 coeffs: {P2_coeffs}")
    print(f"  P3 coeffs: {P3_coeffs}")
    print(f"  Q coeffs: {Q_coeffs}")

    # ========================================
    # Step 3: Fresh computation from scratch
    # ========================================
    print("\n[Step 3] Computing c and κ from scratch...")

    engine = KappaEngine(
        P1_coeffs=P1_coeffs,
        P2_coeffs=P2_coeffs,
        P3_coeffs=P3_coeffs,
        Q_coeffs=Q_coeffs,
        theta=theta,
        K=K,
        R=R,
        n_quad=n_quad,
    )
    result = engine.compute_kappa()

    # ========================================
    # Step 4: Extract all components
    # ========================================
    print("\n[Step 4] Component Decomposition:")

    integrals = result.integrals
    corrections = result.corrections

    # Individual integrals
    print(f"\n  Individual Integrals:")
    print(f"    I1(+R) = {integrals.I1_plus:.10f}")
    print(f"    I2(+R) = {integrals.I2_plus:.10f}")
    print(f"    I1(-R) = {integrals.I1_minus:.10f}")
    print(f"    I2(-R) = {integrals.I2_minus:.10f}")
    print(f"    I3(+R) = {integrals.I3_plus:.10f}")
    print(f"    I4(+R) = {integrals.I4_plus:.10f}")

    # Composite components
    print(f"\n  Composite Components:")
    print(f"    S12(+R) = {integrals.S12_plus:.10f}")
    print(f"    S12(-R) = {integrals.S12_minus:.10f}")
    print(f"    S34(+R) = {integrals.S34_plus:.10f}")

    # Corrections
    print(f"\n  Corrections:")
    print(f"    m (mirror multiplier) = {corrections.m:.10f}")
    print(f"    f_I1 = {corrections.f_I1:.10f}")
    print(f"    g_I1 = {corrections.g_I1:.10f}")
    print(f"    g_I2 = {corrections.g_I2:.10f}")

    # Assembly verification
    c_assembled = integrals.S12_plus + corrections.m * integrals.S12_minus + integrals.S34_plus

    print(f"\n  Assembly Formula:")
    print(f"    c = S12(+R) + m × S12(-R) + S34(+R)")
    print(f"    c = {integrals.S12_plus:.6f} + {corrections.m:.6f} × {integrals.S12_minus:.6f} + {integrals.S34_plus:.6f}")
    print(f"    c = {c_assembled:.10f}")

    # ========================================
    # Step 5: Final results
    # ========================================
    print("\n[Step 5] Final Results:")
    print(f"    c = {result.c:.10f}")
    print(f"    κ = {result.kappa:.10f}")

    # Verify κ > 0.5
    kappa_threshold = 0.5
    c_threshold = np.exp(R / 2)  # c < exp(R/2) ⟺ κ > 0.5

    print(f"\n  Threshold Analysis:")
    print(f"    κ > 0.5 threshold: c < exp(R/2) = {c_threshold:.6f}")
    print(f"    Actual c = {result.c:.6f}")
    print(f"    Margin: {(c_threshold - result.c) / c_threshold * 100:.2f}% below threshold")
    print(f"    κ = {result.kappa:.6f} > 0.5? {'YES ✓' if result.kappa > 0.5 else 'NO ✗'}")

    # ========================================
    # Step 6: Compare against stored values
    # ========================================
    print("\n[Step 6] Verification Against Stored Values:")

    stored = data['kappa_benchmark']
    stored_decomp = data['decomposition']

    checks = [
        ("c", result.c, stored['c']),
        ("κ", result.kappa, stored['kappa']),
        ("S12(+R)", integrals.S12_plus, stored_decomp['S12_plus']),
        ("S12(-R)", integrals.S12_minus, stored_decomp['S12_minus']),
        ("S34(+R)", integrals.S34_plus, stored_decomp['S34_plus']),
        ("m", corrections.m, stored_decomp['m']),
    ]

    all_match = True
    print(f"\n  {'Component':<12} | {'Computed':>14} | {'Stored':>14} | {'Rel Diff':>12} | Status")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}-+-------")

    for name, computed, stored_val in checks:
        rel_diff = abs(computed - stored_val) / (abs(stored_val) + 1e-15)
        match = rel_diff < 0.001  # 0.1% tolerance
        all_match &= match
        status = "MATCH" if match else "DIFF"
        print(f"  {name:<12} | {computed:>14.8f} | {stored_val:>14.8f} | {rel_diff:>12.2e} | {status}")

    # ========================================
    # Step 7: Checksums
    # ========================================
    print("\n[Step 7] Intermediate Value Checksums:")

    key_values = {
        "c": result.c,
        "kappa": result.kappa,
        "I1_plus": integrals.I1_plus,
        "I2_plus": integrals.I2_plus,
        "I1_minus": integrals.I1_minus,
        "I2_minus": integrals.I2_minus,
        "I3_plus": integrals.I3_plus,
        "I4_plus": integrals.I4_plus,
        "S12_plus": integrals.S12_plus,
        "S12_minus": integrals.S12_minus,
        "S34_plus": integrals.S34_plus,
        "m": corrections.m,
        "f_I1": corrections.f_I1,
        "g_I1": corrections.g_I1,
        "g_I2": corrections.g_I2,
    }

    checksum = compute_checksum(key_values)
    print(f"  Checksum (SHA256-16): {checksum}")

    # ========================================
    # Step 8: Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  RESULT: κ = {result.kappa:.6f} (c = {result.c:.6f})")
    print(f"  STATUS: {'ALL CHECKS PASSED ✓' if all_match else 'SOME CHECKS FAILED ✗'}")
    print(f"  κ > 0.5: {'CONFIRMED ✓' if result.kappa > 0.5 else 'NOT CONFIRMED ✗'}")
    print(f"  Checksum: {checksum}")

    print("\n" + "=" * 70)

    return result.kappa > 0.5 and all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
