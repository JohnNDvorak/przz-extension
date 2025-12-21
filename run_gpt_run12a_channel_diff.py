#!/usr/bin/env python3
"""
GPT Run 12A: Channel-by-Channel Diff (OLD vs V2) under tex_mirror

This script diagnoses WHY V2 terms catastrophically fail under tex_mirror
by printing ALL intermediate values for both term versions at both benchmarks.

Output: docs/RUN12A_CHANNEL_DIFF.md

The key diagnostic question:
- V2 is proven correct for individual I-terms (ratio=1.0)
- BUT V2 breaks tex_mirror (c≈0.775 vs target 2.137)
- WHERE does the collapse happen?

Expected diagnostic outcomes:
- If plus channels differ hugely: V2 term builder feeds different object
- If minus_base differs hugely: V2's (1-u) power affects -R materially
- If minus_op differs but minus_base doesn't: Operator-shape incompatible with V2
- If m_implied explodes: Near-zero denominator issue
- If everything sane except c: Assembly bug in V2 path

Usage:
    python run_gpt_run12a_channel_diff.py
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star
from src.evaluate import compute_c_paper_tex_mirror


THETA = 4.0 / 7.0

# PRZZ Targets
TARGETS = {
    "kappa": {
        "name": "κ",
        "R": 1.3036,
        "c_target": 2.13745440613217263636,
        "kappa_target": 0.417293962,
    },
    "kappa_star": {
        "name": "κ*",
        "R": 1.1167,
        "c_target": 1.93801,
        "kappa_target": 0.405,
    }
}


def safe_ratio(a: float, b: float, threshold: float = 1e-10) -> str:
    """Return ratio a/b with explosion guard, formatted as string."""
    if abs(b) < threshold:
        if a > threshold:
            return "+∞"
        elif a < -threshold:
            return "-∞"
        else:
            return "NaN"
    ratio = a / b
    return f"{ratio:.4f}"


def compute_kappa(c: float, R: float) -> float:
    """Compute κ from c using κ = 1 - log(c)/R."""
    if c <= 0:
        return float('nan')
    return 1.0 - np.log(c) / R


def extract_channels(result) -> Dict[str, float]:
    """Extract all channel values from TexMirrorResult."""
    return {
        "I1_plus": result.I1_plus,
        "I2_plus": result.I2_plus,
        "S34_plus": result.S34_plus,
        "I1_minus_base": result.I1_minus_base,
        "I2_minus_base": result.I2_minus_base,
        "I1_minus_op": result.I1_minus_shape,  # alias
        "I2_minus_op": result.I2_minus_shape,  # alias
        "m1_implied": result.m1_implied,
        "m2_implied": result.m2_implied,
        "A1": result.A1,
        "A2": result.A2,
        "m1": result.m1,
        "m2": result.m2,
        "c": result.c,
    }


def run_channel_extraction() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run tex_mirror for all 4 combinations and extract channels."""

    # Load polynomials
    P1_k, P2_k, P3_k, Q_k = load_przz_polynomials(enforce_Q0=True)
    polys_kappa = {"P1": P1_k, "P2": P2_k, "P3": P3_k, "Q": Q_k}

    P1_ks, P2_ks, P3_ks, Q_ks = load_przz_polynomials_kappa_star()
    polys_kappa_star = {"P1": P1_ks, "P2": P2_ks, "P3": P3_ks, "Q": Q_ks}

    benchmarks = {
        "kappa": (polys_kappa, TARGETS["kappa"]),
        "kappa_star": (polys_kappa_star, TARGETS["kappa_star"]),
    }

    results = {}

    for bench_key, (polys, target) in benchmarks.items():
        R = target["R"]
        results[bench_key] = {}

        for terms_ver in ["old", "v2"]:
            try:
                result = compute_c_paper_tex_mirror(
                    theta=THETA,
                    R=R,
                    n=60,
                    polynomials=polys,
                    terms_version=terms_ver,
                    i2_source="dsl",  # Use DSL for consistency
                    tex_exp_component="exp_R_ref",
                )
                results[bench_key][terms_ver] = extract_channels(result)
            except Exception as e:
                print(f"ERROR: {bench_key}/{terms_ver}: {e}")
                results[bench_key][terms_ver] = {"error": str(e)}

    return results


def analyze_collapse(results: Dict) -> List[str]:
    """Analyze results to identify where V2 collapses."""
    analysis = []

    # Check each benchmark
    for bench_key in ["kappa", "kappa_star"]:
        bench_name = TARGETS[bench_key]["name"]
        old = results[bench_key].get("old", {})
        v2 = results[bench_key].get("v2", {})

        if "error" in old or "error" in v2:
            analysis.append(f"- **{bench_name}**: Error in extraction (see table)")
            continue

        # Compare key variables
        vars_to_check = [
            ("I1_plus", "I1 (+R) channel"),
            ("I2_plus", "I2 (+R) channel"),
            ("S34_plus", "S34 (+R) channel"),
            ("I1_minus_base", "I1 (-R) base"),
            ("I2_minus_base", "I2 (-R) base"),
            ("I1_minus_op", "I1 (-R) operator"),
            ("I2_minus_op", "I2 (-R) operator"),
            ("m1_implied", "m1 implied (shape)"),
            ("m2_implied", "m2 implied (shape)"),
            ("m1", "m1 full weight"),
            ("m2", "m2 full weight"),
            ("c", "final c"),
        ]

        large_diffs = []
        for var_key, var_name in vars_to_check:
            old_val = old.get(var_key, 0)
            v2_val = v2.get(var_key, 0)

            if abs(old_val) < 1e-10:
                if abs(v2_val) > 1e-6:
                    large_diffs.append((var_name, "OLD≈0, V2 differs", v2_val))
            else:
                ratio = v2_val / old_val
                pct_diff = 100 * (v2_val - old_val) / abs(old_val)
                if abs(pct_diff) > 20:  # More than 20% difference
                    large_diffs.append((var_name, f"{pct_diff:+.1f}%", ratio))

        if large_diffs:
            analysis.append(f"\n### {bench_name} Analysis")
            analysis.append("")
            analysis.append("Variables with >20% difference (V2 vs OLD):")
            analysis.append("")
            for var_name, diff_desc, extra in large_diffs:
                analysis.append(f"- **{var_name}**: {diff_desc}")

            # Identify likely collapse point
            collapse_vars = [v[0] for v in large_diffs]

            if "I1 (+R) channel" in collapse_vars or "I2 (+R) channel" in collapse_vars:
                analysis.append("")
                analysis.append("**DIAGNOSIS**: Plus channels differ → V2 term builder issue")
            elif "I1 (-R) base" in collapse_vars or "I2 (-R) base" in collapse_vars:
                analysis.append("")
                analysis.append("**DIAGNOSIS**: Minus base differs → V2 (1-u) power affects -R")
            elif ("I1 (-R) operator" in collapse_vars or "I2 (-R) operator" in collapse_vars) and \
                 "I1 (-R) base" not in collapse_vars and "I2 (-R) base" not in collapse_vars:
                analysis.append("")
                analysis.append("**DIAGNOSIS**: Operator differs but base doesn't → Shape model incompatible with V2")
            elif "m1 implied (shape)" in collapse_vars or "m2 implied (shape)" in collapse_vars:
                analysis.append("")
                analysis.append("**DIAGNOSIS**: Implied weights explode → Near-zero denominator in V2")
            else:
                analysis.append("")
                analysis.append("**DIAGNOSIS**: Differences propagate through assembly")
        else:
            analysis.append(f"\n### {bench_name}: No major differences found (all <20%)")

    return analysis


def generate_markdown(results: Dict) -> str:
    """Generate markdown report from results."""

    lines = []
    lines.append("# Run 12A: Channel-by-Channel Diff (OLD vs V2)")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Diagnose WHY V2 terms fail under tex_mirror assembly:")
    lines.append("- V2 is proven correct for individual I-terms (ratio=1.0)")
    lines.append("- BUT V2 breaks tex_mirror (c≈0.775 vs target 2.137)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Channel Comparison Table")
    lines.append("")

    # Get variable names (excluding error)
    var_names = [
        "I1_plus", "I2_plus", "S34_plus",
        "I1_minus_base", "I2_minus_base",
        "I1_minus_op", "I2_minus_op",
        "m1_implied", "m2_implied",
        "A1", "A2",
        "m1", "m2",
        "c"
    ]

    # Build table
    lines.append("| Variable | κ OLD | κ V2 | κ V2/OLD | κ* OLD | κ* V2 | κ* V2/OLD |")
    lines.append("|----------|-------|------|----------|--------|-------|-----------|")

    for var in var_names:
        k_old = results["kappa"].get("old", {}).get(var, float('nan'))
        k_v2 = results["kappa"].get("v2", {}).get(var, float('nan'))
        ks_old = results["kappa_star"].get("old", {}).get(var, float('nan'))
        ks_v2 = results["kappa_star"].get("v2", {}).get(var, float('nan'))

        k_ratio = safe_ratio(k_v2, k_old)
        ks_ratio = safe_ratio(ks_v2, ks_old)

        # Format values
        def fmt(v):
            if isinstance(v, float):
                if abs(v) < 1e-6:
                    return f"{v:.2e}"
                return f"{v:.6f}"
            return str(v)

        lines.append(f"| {var} | {fmt(k_old)} | {fmt(k_v2)} | {k_ratio} | {fmt(ks_old)} | {fmt(ks_v2)} | {ks_ratio} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Targets")
    lines.append("")
    lines.append("| Benchmark | c target | κ target |")
    lines.append("|-----------|----------|----------|")
    lines.append(f"| κ (R=1.3036) | 2.137454 | 0.417294 |")
    lines.append(f"| κ* (R=1.1167) | 1.938010 | 0.405000 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")

    # Add analysis
    analysis = analyze_collapse(results)
    lines.extend(analysis)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("| If this differs | Then... |")
    lines.append("|-----------------|---------|")
    lines.append("| Plus channels (I1_plus, I2_plus) | V2 term builder feeds different object |")
    lines.append("| Minus base (I1_minus_base, I2_minus_base) | V2's (1-u) power materially affects -R |")
    lines.append("| Minus op but not base | Operator-shape incompatible with V2 structure |")
    lines.append("| m_implied explodes | Near-zero denominator in V2 |")
    lines.append("| Only c differs | Assembly bug in V2 path |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append("### κ Benchmark (R=1.3036)")
    lines.append("")
    lines.append("#### OLD terms")
    lines.append("```")
    for k, v in sorted(results["kappa"].get("old", {}).items()):
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("#### V2 terms")
    lines.append("```")
    for k, v in sorted(results["kappa"].get("v2", {}).items()):
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("### κ* Benchmark (R=1.1167)")
    lines.append("")
    lines.append("#### OLD terms")
    lines.append("```")
    for k, v in sorted(results["kappa_star"].get("old", {}).items()):
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("#### V2 terms")
    lines.append("```")
    for k, v in sorted(results["kappa_star"].get("v2", {}).items()):
        lines.append(f"{k}: {v}")
    lines.append("```")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("GPT Run 12A: Channel-by-Channel Diff (OLD vs V2)")
    print("=" * 70)
    print()

    # Run extraction
    print("Extracting channel values for all 4 combinations...")
    results = run_channel_extraction()
    print("Done.")
    print()

    # Print summary to terminal
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()

    var_names = ["I1_plus", "I2_plus", "S34_plus", "I1_minus_base", "I2_minus_base",
                 "m1_implied", "m2_implied", "A1", "A2", "m1", "m2", "c"]

    print(f"{'Variable':<18} {'κ OLD':>12} {'κ V2':>12} {'V2/OLD':>10} {'κ* OLD':>12} {'κ* V2':>12} {'V2/OLD':>10}")
    print("-" * 90)

    for var in var_names:
        k_old = results["kappa"].get("old", {}).get(var, float('nan'))
        k_v2 = results["kappa"].get("v2", {}).get(var, float('nan'))
        ks_old = results["kappa_star"].get("old", {}).get(var, float('nan'))
        ks_v2 = results["kappa_star"].get("v2", {}).get(var, float('nan'))

        k_ratio = safe_ratio(k_v2, k_old)
        ks_ratio = safe_ratio(ks_v2, ks_old)

        print(f"{var:<18} {k_old:>12.6f} {k_v2:>12.6f} {k_ratio:>10} {ks_old:>12.6f} {ks_v2:>12.6f} {ks_ratio:>10}")

    print()

    # Generate markdown
    print("Generating markdown report...")
    md_content = generate_markdown(results)

    # Write to file
    output_path = "/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension/docs/RUN12A_CHANNEL_DIFF.md"
    with open(output_path, "w") as f:
        f.write(md_content)

    print(f"Report written to: {output_path}")
    print()

    # Print analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    analysis = analyze_collapse(results)
    for line in analysis:
        print(line)


if __name__ == "__main__":
    main()
