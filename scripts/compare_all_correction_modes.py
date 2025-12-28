#!/usr/bin/env python3
"""
scripts/compare_all_correction_modes.py
Compare all correction modes on both PRZZ benchmarks.

Created: 2025-12-27
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
from src.evaluator.correction_policy import (
    CorrectionMode,
    get_g_correction,
    is_derived_mode,
)


def main():
    print("=" * 80)
    print("COMPARING ALL CORRECTION MODES ON PRZZ BENCHMARKS")
    print("=" * 80)

    theta = 4/7
    K = 3

    # Benchmark targets
    c_kappa = 2.13745440613217263636
    c_kappa_star = 1.93795241121330

    # f_I1 values (computed from integrals)
    f_I1_kappa = 0.233
    f_I1_kappa_star = 0.326

    benchmarks = [
        ("κ", 1.3036, c_kappa, f_I1_kappa),
        ("κ*", 1.1167, c_kappa_star, f_I1_kappa_star),
    ]

    modes = [
        (CorrectionMode.DERIVED_BASELINE_ONLY, None, False),  # No f_I1 needed
        (CorrectionMode.FIRST_PRINCIPLES_I1_I2, "use_benchmark", False),
        (CorrectionMode.THETA_2_MINUS_THETA, "use_benchmark", False),
        (CorrectionMode.FULL_SECOND_ORDER, "use_benchmark", False),
        (CorrectionMode.THETA_CUBED, "use_benchmark", False),
        (CorrectionMode.ANCHORED_TWO_BENCHMARKS, "use_benchmark", True),
    ]

    print(f"\nParameters: θ = {theta:.6f}, K = {K}")
    print(f"\nTarget c values:")
    print(f"  κ (R=1.3036):  c = {c_kappa:.10f}")
    print(f"  κ* (R=1.1167): c = {c_kappa_star:.10f}")

    print("\n" + "-" * 80)
    print("CORRECTION MODE COMPARISON")
    print("-" * 80)

    # Results storage for summary table
    results = []

    for mode, f_I1_type, allow_anchoring in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.name}")
        print(f"  Derived: {is_derived_mode(mode)}")
        print(f"  Anchored: {allow_anchoring}")
        print("=" * 60)

        mode_results = {"mode": mode.name}

        for benchmark_name, R, c_target, f_I1 in benchmarks:
            # Determine f_I1 to use
            if f_I1_type is None:
                f_I1_param = None
            else:
                f_I1_param = f_I1

            try:
                result = get_g_correction(
                    R=R,
                    theta=theta,
                    K=K,
                    f_I1=f_I1_param,
                    mode=mode,
                    allow_target_anchoring=allow_anchoring,
                )

                # Compute c using the mirror formula
                # c = I1I2(+R) + m × I1I2(-R) + I3I4(+R)
                # For simplicity, we compare just the g values vs calibrated

                print(f"\n  {benchmark_name} benchmark (R={R}):")
                print(f"    g = {result.g:.10f}")
                print(f"    g_baseline = {result.g_baseline:.10f}")
                if result.g_I1 is not None:
                    print(f"    g_I1 = {result.g_I1:.10f}")
                    print(f"    g_I2 = {result.g_I2:.10f}")
                    print(f"    f_I1 = {result.f_I1:.6f}")

                # Compare to calibrated g_total
                g_calibrated = f_I1 * 1.00091428 + (1 - f_I1) * 1.01945154
                g_gap_pct = (result.g / g_calibrated - 1) * 100
                print(f"    g_calibrated = {g_calibrated:.10f}")
                print(f"    g gap: {g_gap_pct:+.6f}%")

                mode_results[f"{benchmark_name}_g"] = result.g
                mode_results[f"{benchmark_name}_gap"] = g_gap_pct

            except ValueError as e:
                print(f"\n  {benchmark_name}: ERROR - {e}")
                mode_results[f"{benchmark_name}_g"] = None
                mode_results[f"{benchmark_name}_gap"] = None

        results.append(mode_results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: g-value gaps from calibrated")
    print("=" * 80)
    print(f"\n{'Mode':<35} {'κ gap':<15} {'κ* gap':<15} {'Mean |gap|':<15}")
    print("-" * 80)

    for r in results:
        kappa_gap = r.get("κ_gap")
        kappa_star_gap = r.get("κ*_gap")

        if kappa_gap is not None and kappa_star_gap is not None:
            mean_abs_gap = (abs(kappa_gap) + abs(kappa_star_gap)) / 2
            print(f"{r['mode']:<35} {kappa_gap:+.6f}%      {kappa_star_gap:+.6f}%      {mean_abs_gap:.6f}%")
        else:
            print(f"{r['mode']:<35} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    print("\n" + "=" * 80)
    print("g_I1 and g_I2 COMPARISON")
    print("=" * 80)

    g_I1_calibrated = 1.00091428
    g_I2_calibrated = 1.01945154

    print(f"\nCalibrated values:")
    print(f"  g_I1_calibrated = {g_I1_calibrated:.10f}")
    print(f"  g_I2_calibrated = {g_I2_calibrated:.10f}")

    print(f"\nDerived formulas:")

    # FIRST_PRINCIPLES_I1_I2
    g_I1_fp = 1.0
    g_I2_fp = 1 + theta / (2 * K * (2 * K + 1))
    print(f"\n  FIRST_PRINCIPLES_I1_I2:")
    print(f"    g_I1 = 1.0")
    print(f"    g_I2 = 1 + θ/(2K(2K+1)) = {g_I2_fp:.10f}")
    print(f"    g_I1 gap: {(g_I1_fp/g_I1_calibrated-1)*100:+.6f}%")
    print(f"    g_I2 gap: {(g_I2_fp/g_I2_calibrated-1)*100:+.6f}%")

    # THETA_2_MINUS_THETA
    g_I1_t2t = 1.0
    g_I2_t2t = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))
    print(f"\n  THETA_2_MINUS_THETA:")
    print(f"    g_I1 = 1.0")
    print(f"    g_I2 = 1 + θ(2-θ)/(2K(2K+1)) = {g_I2_t2t:.10f}")
    print(f"    g_I1 gap: {(g_I1_t2t/g_I1_calibrated-1)*100:+.6f}%")
    print(f"    g_I2 gap: {(g_I2_t2t/g_I2_calibrated-1)*100:+.6f}%")

    # FULL_SECOND_ORDER
    g_I1_fso = 1 + theta * (1 - theta) / (2 * K * (2 * K + 1)**2)
    g_I2_fso = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))
    print(f"\n  FULL_SECOND_ORDER:")
    print(f"    g_I1 = 1 + θ(1-θ)/(2K(2K+1)²) = {g_I1_fso:.10f}")
    print(f"    g_I2 = 1 + θ(2-θ)/(2K(2K+1)) = {g_I2_fso:.10f}")
    print(f"    g_I1 gap: {(g_I1_fso/g_I1_calibrated-1)*100:+.6f}%")
    print(f"    g_I2 gap: {(g_I2_fso/g_I2_calibrated-1)*100:+.6f}%")

    # THETA_CUBED
    g_I1_tc = 1 + (3/28) * theta**3 / (K * (2 * K + 1))
    g_I2_tc = 1 + theta * (2 - theta) / (2 * K * (2 * K + 1))
    print(f"\n  THETA_CUBED (BEST!):")
    print(f"    g_I1 = 1 + (3/28)×θ³/(K(2K+1)) = {g_I1_tc:.10f}")
    print(f"    g_I2 = 1 + θ(2-θ)/(2K(2K+1)) = {g_I2_tc:.10f}")
    print(f"    g_I1 gap: {(g_I1_tc/g_I1_calibrated-1)*100:+.6f}%")
    print(f"    g_I2 gap: {(g_I2_tc/g_I2_calibrated-1)*100:+.6f}%")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The FULL_SECOND_ORDER mode includes:
  g_I1 = 1 + θ(1-θ)/(2K(2K+1)²)
  g_I2 = 1 + θ(2-θ)/(2K(2K+1))

This captures the elegant relationship:
  epsilon_I1 = epsilon_I2 / (2K+1)

where epsilon_I2 = θ(1-θ)/(2K(2K+1)) is the difference between
the θ(2-θ) and θ terms in the g_I2 formula.
""")


if __name__ == "__main__":
    main()
