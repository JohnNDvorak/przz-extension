#!/usr/bin/env python3
"""
Gate PSD/Cauchy-Schwarz: Verify pair matrix is positive semi-definite.

This is the "physics-level" invariant that doesn't depend on implementation
details - only on whether the algebra corresponds to a bona fide quadratic form.

Per GPT guidance:
1. Convert reported pair table to RAW symmetric matrix G by dividing off-diagonals by 2
   (since the table already includes the (2-δ) factor for off-diagonal pairs)
2. Check Cauchy-Schwarz: |G_ij| ≤ √(G_ii * G_jj) + ε
3. Check PSD: λ_min(G) ≥ -ε

If the optimizer is real, we expect λ_min(G) to be small (near-degenerate),
consistent with finding maximum cancellation.

Created: 2025-12-28
"""

import json
import numpy as np
import pytest
from pathlib import Path


def load_pair_data(json_path: Path) -> dict:
    """Load pair data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def build_raw_gram_matrix(pairs: dict) -> np.ndarray:
    """
    Build raw 3x3 Gram matrix from pair contributions.

    The JSON pair values already include the (2-δ) symmetry factor:
    - Diagonal pairs (11, 22, 33): factor = 1
    - Off-diagonal pairs (12, 13, 23): factor = 2

    We divide off-diagonals by 2 to get the raw bilinear form entries.
    """
    G = np.zeros((3, 3))

    # Diagonal entries (no adjustment needed)
    G[0, 0] = pairs["11"]["value"]
    G[1, 1] = pairs["22"]["value"]
    G[2, 2] = pairs["33"]["value"]

    # Off-diagonal entries (divide by 2 to remove symmetry factor)
    G[0, 1] = G[1, 0] = pairs["12"]["value"] / 2
    G[0, 2] = G[2, 0] = pairs["13"]["value"] / 2
    G[1, 2] = G[2, 1] = pairs["23"]["value"] / 2

    return G


def compute_correlation_matrix(G: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix ρ_ij = G_ij / √(G_ii * G_jj).

    Correlation should satisfy |ρ_ij| ≤ 1 (Cauchy-Schwarz).
    """
    n = G.shape[0]
    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(abs(G[i, i]) * abs(G[j, j]))
            if denom > 1e-15:
                rho[i, j] = G[i, j] / denom
            else:
                rho[i, j] = 0.0
    return rho


def check_cauchy_schwarz(G: np.ndarray, tol: float = 1e-10) -> tuple:
    """
    Check Cauchy-Schwarz inequality: |G_ij| ≤ √(G_ii * G_jj) + tol.

    Returns (passed: bool, violations: list of (i, j, ratio)).
    """
    n = G.shape[0]
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            bound = np.sqrt(abs(G[i, i]) * abs(G[j, j]))
            if abs(G[i, j]) > bound + tol:
                ratio = abs(G[i, j]) / bound if bound > 1e-15 else float('inf')
                violations.append((i + 1, j + 1, ratio))  # 1-indexed for display
    return len(violations) == 0, violations


def check_psd(G: np.ndarray, tol: float = 1e-10) -> tuple:
    """
    Check positive semi-definiteness via eigenvalues.

    Returns (passed: bool, eigenvalues: ndarray, lambda_min: float).
    """
    eigenvalues = np.linalg.eigvalsh(G)
    lambda_min = eigenvalues.min()
    passed = lambda_min >= -tol
    return passed, eigenvalues, lambda_min


class TestPSDGateBaseline:
    """Test PSD gate on PRZZ baseline polynomials."""

    @pytest.fixture
    def baseline_data(self):
        path = Path(__file__).parent.parent / "data" / "derivation_report" / "kappa_baseline.json"
        return load_pair_data(path)

    def test_baseline_cauchy_schwarz(self, baseline_data):
        """Baseline should satisfy Cauchy-Schwarz."""
        G = build_raw_gram_matrix(baseline_data["pairs"])
        passed, violations = check_cauchy_schwarz(G)

        if not passed:
            for i, j, ratio in violations:
                print(f"  C-S VIOLATION: pair ({i},{j}) has |G_ij|/√(G_ii·G_jj) = {ratio:.4f}")

        assert passed, f"Baseline violates Cauchy-Schwarz: {violations}"

    def test_baseline_psd(self, baseline_data):
        """Baseline should be positive semi-definite."""
        G = build_raw_gram_matrix(baseline_data["pairs"])
        passed, eigenvalues, lambda_min = check_psd(G)

        print(f"  Baseline eigenvalues: {eigenvalues}")
        print(f"  λ_min = {lambda_min:.6e}")

        assert passed, f"Baseline not PSD: λ_min = {lambda_min:.6e}"

    def test_baseline_correlation_bounded(self, baseline_data):
        """Baseline correlations should be in [-1, 1]."""
        G = build_raw_gram_matrix(baseline_data["pairs"])
        rho = compute_correlation_matrix(G)

        print("\n  Baseline correlation matrix:")
        print(f"  ρ(1,2) = {rho[0, 1]:+.4f}")
        print(f"  ρ(1,3) = {rho[0, 2]:+.4f}")
        print(f"  ρ(2,3) = {rho[1, 2]:+.4f}")

        assert np.all(np.abs(rho) <= 1.0 + 1e-10), "Correlations out of bounds"


class TestPSDGateOptimized:
    """Test PSD gate on optimized polynomials (α=70, β=-30)."""

    @pytest.fixture
    def optimized_data(self):
        path = Path(__file__).parent.parent / "data" / "derivation_report" / "kappa_optimal.json"
        return load_pair_data(path)

    def test_optimized_cauchy_schwarz(self, optimized_data):
        """Optimized should satisfy Cauchy-Schwarz."""
        G = build_raw_gram_matrix(optimized_data["pairs"])
        passed, violations = check_cauchy_schwarz(G)

        if not passed:
            for i, j, ratio in violations:
                print(f"  C-S VIOLATION: pair ({i},{j}) has |G_ij|/√(G_ii·G_jj) = {ratio:.4f}")

        assert passed, f"Optimized violates Cauchy-Schwarz: {violations}"

    def test_optimized_psd(self, optimized_data):
        """Optimized should be positive semi-definite."""
        G = build_raw_gram_matrix(optimized_data["pairs"])
        passed, eigenvalues, lambda_min = check_psd(G)

        print(f"\n  Optimized eigenvalues: {eigenvalues}")
        print(f"  λ_min = {lambda_min:.6e}")

        # If optimizer is "real", λ_min should be small (near-degenerate)
        if lambda_min < 0.01:
            print("  NOTE: Small λ_min suggests near-maximal cancellation - expected for optimization")

        assert passed, f"Optimized not PSD: λ_min = {lambda_min:.6e}"

    def test_optimized_correlation_bounded(self, optimized_data):
        """Optimized correlations should be in [-1, 1]."""
        G = build_raw_gram_matrix(optimized_data["pairs"])
        rho = compute_correlation_matrix(G)

        print("\n  Optimized correlation matrix:")
        print(f"  ρ(1,2) = {rho[0, 1]:+.4f}")
        print(f"  ρ(1,3) = {rho[0, 2]:+.4f}  <-- expect large negative (destructive interference)")
        print(f"  ρ(2,3) = {rho[1, 2]:+.4f}  <-- expect large negative (destructive interference)")

        assert np.all(np.abs(rho) <= 1.0 + 1e-10), "Correlations out of bounds"


class TestPSDGateKappaStar:
    """Test PSD gate on κ* benchmark (R=1.1167)."""

    @pytest.fixture
    def kappa_star_baseline(self):
        path = Path(__file__).parent.parent / "data" / "derivation_report" / "kappa_star_baseline.json"
        return load_pair_data(path)

    @pytest.fixture
    def kappa_star_optimal(self):
        path = Path(__file__).parent.parent / "data" / "derivation_report" / "kappa_star_optimal.json"
        return load_pair_data(path)

    def test_kappa_star_baseline_psd(self, kappa_star_baseline):
        """κ* baseline should be PSD."""
        G = build_raw_gram_matrix(kappa_star_baseline["pairs"])
        passed, eigenvalues, lambda_min = check_psd(G)

        print(f"\n  κ* baseline eigenvalues: {eigenvalues}")
        print(f"  λ_min = {lambda_min:.6e}")

        assert passed, f"κ* baseline not PSD: λ_min = {lambda_min:.6e}"

    def test_kappa_star_optimal_psd(self, kappa_star_optimal):
        """κ* optimized should be PSD."""
        G = build_raw_gram_matrix(kappa_star_optimal["pairs"])
        passed, eigenvalues, lambda_min = check_psd(G)

        print(f"\n  κ* optimized eigenvalues: {eigenvalues}")
        print(f"  λ_min = {lambda_min:.6e}")

        assert passed, f"κ* optimized not PSD: λ_min = {lambda_min:.6e}"


class TestPSDComprehensiveSummary:
    """Comprehensive summary of all PSD/CS checks."""

    def test_full_summary(self):
        """Run full summary across all configurations."""
        base_path = Path(__file__).parent.parent / "data" / "derivation_report"

        configs = [
            ("κ baseline", "kappa_baseline.json"),
            ("κ optimized", "kappa_optimal.json"),
            ("κ* baseline", "kappa_star_baseline.json"),
            ("κ* optimized", "kappa_star_optimal.json"),
        ]

        print("\n" + "=" * 70)
        print("PSD/CAUCHY-SCHWARZ GATE SUMMARY")
        print("=" * 70)

        all_passed = True
        for name, filename in configs:
            try:
                data = load_pair_data(base_path / filename)
            except FileNotFoundError:
                print(f"\n{name}: FILE NOT FOUND")
                continue

            G = build_raw_gram_matrix(data["pairs"])
            rho = compute_correlation_matrix(G)
            cs_passed, violations = check_cauchy_schwarz(G)
            psd_passed, eigenvalues, lambda_min = check_psd(G)

            print(f"\n{name}:")
            print(f"  Raw Gram matrix G:")
            print(f"    G_11 = {G[0, 0]:+.6f}  G_12 = {G[0, 1]:+.6f}  G_13 = {G[0, 2]:+.6f}")
            print(f"    G_21 = {G[1, 0]:+.6f}  G_22 = {G[1, 1]:+.6f}  G_23 = {G[1, 2]:+.6f}")
            print(f"    G_31 = {G[2, 0]:+.6f}  G_32 = {G[2, 1]:+.6f}  G_33 = {G[2, 2]:+.6f}")
            print(f"  Correlations: ρ(1,2)={rho[0, 1]:+.4f}, ρ(1,3)={rho[0, 2]:+.4f}, ρ(2,3)={rho[1, 2]:+.4f}")
            print(f"  Eigenvalues: {eigenvalues}")
            print(f"  λ_min = {lambda_min:.6e}")
            print(f"  Cauchy-Schwarz: {'PASS' if cs_passed else 'FAIL'}")
            print(f"  PSD: {'PASS' if psd_passed else 'FAIL'}")

            if not cs_passed or not psd_passed:
                all_passed = False

        print("\n" + "=" * 70)
        print(f"OVERALL: {'ALL GATES PASS' if all_passed else 'SOME GATES FAILED'}")
        print("=" * 70)

        assert all_passed, "Some configurations failed PSD/CS gate"


if __name__ == "__main__":
    # Run quick check
    base_path = Path(__file__).parent.parent / "data" / "derivation_report"

    print("\n" + "=" * 70)
    print("PSD/CAUCHY-SCHWARZ GATE - Quick Check")
    print("=" * 70)

    for name, filename in [
        ("κ baseline", "kappa_baseline.json"),
        ("κ optimized", "kappa_optimal.json"),
    ]:
        data = load_pair_data(base_path / filename)
        G = build_raw_gram_matrix(data["pairs"])
        rho = compute_correlation_matrix(G)
        cs_passed, _ = check_cauchy_schwarz(G)
        psd_passed, eigenvalues, lambda_min = check_psd(G)

        print(f"\n{name}:")
        print(f"  ρ(1,3) = {rho[0, 2]:+.4f} (expect negative for optimized)")
        print(f"  ρ(2,3) = {rho[1, 2]:+.4f} (expect negative for optimized)")
        print(f"  λ_min = {lambda_min:.6e}")
        print(f"  Cauchy-Schwarz: {'PASS' if cs_passed else 'FAIL'}")
        print(f"  PSD: {'PASS' if psd_passed else 'FAIL'}")
