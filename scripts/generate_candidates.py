#!/usr/bin/env python3
"""
scripts/generate_candidates.py
Phase 47b: Generate Candidate Polynomial Files

Expands recipe-driven candidates.json into concrete coefficient files
that can be scored by score_polynomials.py.

Usage:
    python scripts/generate_candidates.py
    python scripts/generate_candidates.py --input my_candidates.json
    python scripts/generate_candidates.py --output-dir my_candidates/

Created: 2025-12-28 (Phase 47b)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.polynomials import load_przz_polynomials


def renormalize_Q0(Q_mono: np.ndarray) -> np.ndarray:
    """Renormalize Q so that Q(0) = 1."""
    Q0 = Q_mono[0]
    if abs(Q0) < 1e-15:
        raise ValueError("Q(0) is zero, cannot renormalize")
    return Q_mono / Q0


def expand_recipe(recipe: dict, base: dict, Q_base: np.ndarray) -> dict:
    """
    Expand a recipe into concrete coefficient arrays.

    Args:
        recipe: Recipe dict with "type" and parameters
        base: Base coefficient dict with P1_tilde, P2_tilde, P3_tilde, Q_mono
        Q_base: Q_mono as numpy array

    Returns:
        dict with P1_tilde, P2_tilde, P3_tilde, Q_mono as lists
    """
    data = {k: list(v) if isinstance(v, np.ndarray) else list(v) for k, v in base.items()}
    recipe_type = recipe["type"]

    if recipe_type == "baseline":
        # Use PRZZ as-is
        pass

    elif recipe_type == "Q_constant":
        # Replace Q with constant value
        data["Q_mono"] = [float(recipe["value"])]

    elif recipe_type == "Q_blend_with_constant_1":
        # Q = (1-lambda)*1 + lambda*Q_przz, then renormalize
        lam = float(recipe["lambda"])
        Q1 = np.zeros_like(Q_base)
        Q1[0] = 1.0  # Q(x) = 1 has only constant term
        Q_new = (1 - lam) * Q1 + lam * Q_base
        Q_new = renormalize_Q0(Q_new)
        data["Q_mono"] = list(Q_new)

    elif recipe_type == "scale_P_tilde":
        # Scale one P_tilde by a factor
        which = recipe["which"]  # "P2" or "P3"
        scale = float(recipe["scale"])
        key = f"{which}_tilde"
        arr = np.asarray(data[key], dtype=float)
        data[key] = list(scale * arr)

    elif recipe_type == "jitter_Q_mono":
        # Add multiplicative noise to Q coefficients
        sigma = float(recipe["sigma"])
        seed = int(recipe["seed"])
        rng = np.random.default_rng(seed)
        Q_new = Q_base * (1 + sigma * rng.normal(size=Q_base.shape))
        if recipe.get("renormalize_Q0", True):
            Q_new = renormalize_Q0(Q_new)
        data["Q_mono"] = list(Q_new)

    elif recipe_type == "jitter_P_tilde":
        # Add multiplicative noise to P_tilde coefficients
        sigma = float(recipe["sigma"])
        seed = int(recipe["seed"])
        rng = np.random.default_rng(seed)
        for which in recipe["which"]:
            key = f"{which}_tilde"
            arr = np.asarray(data[key], dtype=float)
            data[key] = list(arr * (1 + sigma * rng.normal(size=arr.shape)))

    else:
        raise ValueError(f"Unknown recipe type: {recipe_type}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate polynomial files from recipes"
    )
    parser.add_argument(
        "--input",
        default="candidates.json",
        help="Input batch JSON file (default: candidates.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="candidate_files",
        help="Output directory for candidate files (default: candidate_files)",
    )
    args = parser.parse_args()

    # Load batch file
    batch_path = Path(args.input)
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        return 1

    batch = json.load(open(batch_path))
    print(f"Loaded batch from {batch_path}")
    print(f"  {len(batch['candidates'])} candidates to generate")

    # Create output directory
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)
    print(f"Output directory: {outdir}")

    # Load PRZZ baseline
    enforce_Q0 = batch.get("baseline", {}).get("enforce_Q0", False)
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=enforce_Q0)

    base = {
        "P1_tilde": np.asarray(P1.tilde_coeffs, dtype=float),
        "P2_tilde": np.asarray(P2.tilde_coeffs, dtype=float),
        "P3_tilde": np.asarray(P3.tilde_coeffs, dtype=float),
        "Q_mono": np.asarray(Q.to_monomial().coeffs, dtype=float),
    }
    Q_base = base["Q_mono"].copy()

    print(f"\nBaseline polynomials loaded:")
    print(f"  P1_tilde: {len(base['P1_tilde'])} coeffs")
    print(f"  P2_tilde: {len(base['P2_tilde'])} coeffs")
    print(f"  P3_tilde: {len(base['P3_tilde'])} coeffs")
    print(f"  Q_mono: {len(base['Q_mono'])} coeffs, Q(0)={Q_base[0]:.6f}")

    # Generate each candidate
    print(f"\nGenerating candidates...")
    for cand in batch["candidates"]:
        cand_id = cand["id"]
        recipe = cand["recipe"]

        try:
            data = expand_recipe(recipe, base, Q_base)

            # Write to file
            outpath = outdir / f"{cand_id}.json"
            with open(outpath, "w") as f:
                json.dump(data, f, indent=2)

            print(f"  {cand_id}: OK -> {outpath}")

        except Exception as e:
            print(f"  {cand_id}: FAILED - {e}")

    print(f"\nDone. {len(batch['candidates'])} candidate files in {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
