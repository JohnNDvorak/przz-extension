"""
scripts/nolh_optimization/design.py
NOLH Design Generation for K=3 Polynomial Optimization

Generates Nearly Orthogonal Latin Hypercube designs for exploring the
13-parameter K=3 polynomial space (P1: 4, P2: 3, P3: 3, Q: 3).

Created: 2025-12-28 (Phase 49)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import json
from pathlib import Path


# =============================================================================
# BASELINE VALUES
# =============================================================================

# PRZZ baseline (R=1.3036)
P1_PRZZ = [0.261076, -1.071007, -0.236840, 0.260233]
P2_PRZZ = [1.048274, 1.319912, -0.940058]
P3_PRZZ = [0.522811, -0.686510, -0.049923]

# Q basis coefficients (k=0,1,3,5) - note: k=2,4 are zero
Q_BASIS_PRZZ = {
    0: 0.490464,  # Will be recomputed to enforce Q(0)=1
    1: 0.636851,
    3: -0.159327,
    5: 0.032011,
}

# Optimized values from Phase 47c (κ=0.4620)
P2_OPT = [1.1825235016106925, -0.2030695524988122, -0.21781895965589626]
P3_OPT = [-1.0852407847424503, -2.260094537754199, -0.12557532824531475]

# =============================================================================
# κ* BASELINE VALUES (R=1.1167, simple zeros)
# =============================================================================

# κ* has simpler polynomials: P2/P3 have 2 coeffs each (vs 3 for κ)
P1_KAPPA_STAR = [0.052703, -0.657999, -0.003193, -0.101832]  # 4 coeffs (same as κ)
P2_KAPPA_STAR = [1.049837, -0.097446]                         # 2 coeffs (vs 3 for κ)
P3_KAPPA_STAR = [0.035113, -0.156465]                         # 2 coeffs (vs 3 for κ)

# Q basis for κ* is LINEAR (only k=0,1 terms)
Q_BASIS_KAPPA_STAR = {
    0: 0.483777,  # Will be recomputed to enforce Q(0)=1
    1: 0.516223,
}

# R values
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167

# Parameter names in order
PARAM_NAMES = [
    "p1_0", "p1_1", "p1_2", "p1_3",  # P1 (4 params)
    "p2_0", "p2_1", "p2_2",           # P2 (3 params)
    "p3_0", "p3_1", "p3_2",           # P3 (3 params)
    "q_1", "q_3", "q_5",              # Q (3 params, skip q_0)
]

# Parameter names when Q is fixed
PARAM_NAMES_FIXED_Q = [
    "p1_0", "p1_1", "p1_2", "p1_3",  # P1 (4 params)
    "p2_0", "p2_1", "p2_2",           # P2 (3 params)
    "p3_0", "p3_1", "p3_2",           # P3 (3 params)
]

# Parameter names for κ* (simpler polynomials: 8 params)
PARAM_NAMES_KAPPA_STAR = [
    "p1_0", "p1_1", "p1_2", "p1_3",  # P1 (4 params)
    "p2_0", "p2_1",                   # P2 (2 params)
    "p3_0", "p3_1",                   # P3 (2 params)
]

# Parameter names for κ* with Q (not typically used)
PARAM_NAMES_KAPPA_STAR_WITH_Q = [
    "p1_0", "p1_1", "p1_2", "p1_3",  # P1 (4 params)
    "p2_0", "p2_1",                   # P2 (2 params)
    "p3_0", "p3_1",                   # P3 (2 params)
    "q_1",                            # Q (1 param - linear Q)
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NOLHDesign:
    """Container for NOLH design matrix and metadata."""
    n_samples: int
    n_params: int
    param_names: List[str]
    bounds: List[Tuple[float, float]]
    center: np.ndarray  # Center point (current best)
    samples: np.ndarray  # Shape (n_samples, n_params)
    seed: int
    max_coeff_magnitude: Optional[float] = None  # Coefficient constraint if applied
    polynomial_set: str = "kappa"  # "kappa" (10 params) or "kappa_star" (8 params)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "n_samples": self.n_samples,
            "n_params": self.n_params,
            "param_names": self.param_names,
            "bounds": [list(b) for b in self.bounds],
            "center": self.center.tolist(),
            "samples": self.samples.tolist(),
            "seed": self.seed,
            "max_coeff_magnitude": self.max_coeff_magnitude,
            "polynomial_set": self.polynomial_set,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NOLHDesign":
        """Create from dict."""
        return cls(
            n_samples=d["n_samples"],
            n_params=d["n_params"],
            param_names=d["param_names"],
            bounds=[tuple(b) for b in d["bounds"]],
            center=np.array(d["center"]),
            samples=np.array(d["samples"]),
            seed=d["seed"],
            max_coeff_magnitude=d.get("max_coeff_magnitude"),
            polynomial_set=d.get("polynomial_set", "kappa"),
        )

    def save(self, filepath: str):
        """Save design to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "NOLHDesign":
        """Load design from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# BOUND COMPUTATION
# =============================================================================

def compute_bound(value: float, pct: float) -> Tuple[float, float]:
    """
    Compute symmetric percentage bounds around a value.

    Handles sign correctly: for negative values, "larger" means more negative.

    Args:
        value: Center value
        pct: Percentage variation (e.g., 0.5 for ±50%)

    Returns:
        (lo, hi) tuple where lo < hi
    """
    if value > 0:
        lo = value * (1 - pct)
        hi = value * (1 + pct)
    elif value < 0:
        lo = value * (1 + pct)  # More negative
        hi = value * (1 - pct)  # Less negative
    else:
        # value == 0: use absolute range
        lo = -pct
        hi = pct
    return (lo, hi)


def get_parameter_bounds(
    p1_pct: float = 0.5,
    p2_pct: float = 0.3,
    p3_pct: float = 0.3,
    q_pct: float = 0.25,
    use_optimized_p2p3: bool = True,
    fix_Q: bool = False,
    max_coeff_magnitude: Optional[float] = None,
    polynomial_set: str = "kappa",
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Compute parameter bounds for NOLH design.

    Args:
        p1_pct: Percentage variation for P1 (default ±50%)
        p2_pct: Percentage variation for P2 (default ±30%)
        p3_pct: Percentage variation for P3 (default ±30%)
        q_pct: Percentage variation for Q (default ±25%)
        use_optimized_p2p3: If True, center P2/P3 on optimized values (only for κ)
        fix_Q: If True, do not include Q parameters (keep at baseline values)
        max_coeff_magnitude: If set, clip all P1/P2/P3 coefficient bounds to
            [-max, +max]. This constrains the search to match Conrey/PRZZ scale.
            Common values: 1.0 (Conrey/PRZZ scale), 2.0 (moderate), 4.0 (relaxed)
        polynomial_set: "kappa" (10 params) or "kappa_star" (8 params)

    Returns:
        (bounds, center) where:
        - bounds: List of (lo, hi) tuples
        - center: np.ndarray of center values
    """
    bounds = []
    center = []

    if polynomial_set == "kappa_star":
        # κ* baselines (8 params: P1:4 + P2:2 + P3:2)
        p1_baseline = P1_KAPPA_STAR
        p2_baseline = P2_KAPPA_STAR  # 2 coeffs
        p3_baseline = P3_KAPPA_STAR  # 2 coeffs
        q_basis = Q_BASIS_KAPPA_STAR
        q_keys = [1]  # Linear Q only has k=1 (k=0 is computed)
    else:
        # κ baselines (10 params: P1:4 + P2:3 + P3:3)
        p1_baseline = P1_PRZZ
        p2_baseline = P2_OPT if use_optimized_p2p3 else P2_PRZZ  # 3 coeffs
        p3_baseline = P3_OPT if use_optimized_p2p3 else P3_PRZZ  # 3 coeffs
        q_basis = Q_BASIS_PRZZ
        q_keys = [1, 3, 5]  # Full Q has k=1,3,5

    # P1: 4 coefficients
    for v in p1_baseline:
        bounds.append(compute_bound(v, p1_pct))
        center.append(v)

    # P2: 3 coefficients (κ) or 2 coefficients (κ*)
    for v in p2_baseline:
        bounds.append(compute_bound(v, p2_pct))
        center.append(v)

    # P3: 3 coefficients (κ) or 2 coefficients (κ*)
    for v in p3_baseline:
        bounds.append(compute_bound(v, p3_pct))
        center.append(v)

    # Q: Only include if not fixed
    if not fix_Q:
        for k in q_keys:
            v = q_basis[k]
            bounds.append(compute_bound(v, q_pct))
            center.append(v)

    # Apply coefficient magnitude constraint to polynomial params
    # κ: first 10 params, κ*: first 8 params
    if max_coeff_magnitude is not None:
        n_poly_params = 8 if polynomial_set == "kappa_star" else 10
        for i in range(min(n_poly_params, len(bounds))):
            lo, hi = bounds[i]
            # Clip to [-max, +max]
            lo = max(lo, -max_coeff_magnitude)
            hi = min(hi, max_coeff_magnitude)
            # Also clip center if outside bounds
            center[i] = max(min(center[i], hi), lo)
            bounds[i] = (lo, hi)

    return bounds, np.array(center)


# =============================================================================
# DESIGN GENERATION
# =============================================================================

def generate_nolh_design(
    n_samples: int = 49,
    seed: int = 42,
    p1_pct: float = 0.5,
    p2_pct: float = 0.3,
    p3_pct: float = 0.3,
    q_pct: float = 0.25,
    use_optimized_p2p3: bool = True,
    fix_Q: bool = False,
    max_coeff_magnitude: Optional[float] = None,
    polynomial_set: str = "kappa",
) -> NOLHDesign:
    """
    Generate NOLH design for K=3 polynomial optimization.

    Args:
        n_samples: Number of design points (default 49)
        seed: Random seed for reproducibility
        p1_pct: P1 variation percentage
        p2_pct: P2 variation percentage
        p3_pct: P3 variation percentage
        q_pct: Q variation percentage
        use_optimized_p2p3: Center P2/P3 on optimized values (only for κ)
        fix_Q: If True, keep Q at baseline values (only optimize P1/P2/P3)
        max_coeff_magnitude: If set, constrain all P1/P2/P3 coefficients to
            [-max, +max]. Use 1.0 for Conrey/PRZZ scale, 2.0 for moderate, etc.
        polynomial_set: "kappa" (10 params) or "kappa_star" (8 params)

    Returns:
        NOLHDesign with samples scaled to parameter bounds
    """
    from scipy.stats.qmc import LatinHypercube

    # Select parameter names based on polynomial_set and fix_Q
    if polynomial_set == "kappa_star":
        param_names = PARAM_NAMES_KAPPA_STAR if fix_Q else PARAM_NAMES_KAPPA_STAR_WITH_Q
    else:
        param_names = PARAM_NAMES_FIXED_Q if fix_Q else PARAM_NAMES
    n_params = len(param_names)

    # Get bounds and center
    bounds, center = get_parameter_bounds(
        p1_pct=p1_pct,
        p2_pct=p2_pct,
        p3_pct=p3_pct,
        q_pct=q_pct,
        use_optimized_p2p3=use_optimized_p2p3,
        fix_Q=fix_Q,
        max_coeff_magnitude=max_coeff_magnitude,
        polynomial_set=polynomial_set,
    )

    # Generate Latin Hypercube samples in [0,1]^d
    sampler = LatinHypercube(d=n_params, optimization="random-cd", seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale to bounds
    samples = np.zeros_like(unit_samples)
    for i, (lo, hi) in enumerate(bounds):
        samples[:, i] = lo + (hi - lo) * unit_samples[:, i]

    return NOLHDesign(
        n_samples=n_samples,
        n_params=n_params,
        param_names=param_names,
        bounds=bounds,
        center=center,
        samples=samples,
        seed=seed,
        max_coeff_magnitude=max_coeff_magnitude,
        polynomial_set=polynomial_set,
    )


def sample_to_polynomials(
    sample: np.ndarray,
    fix_Q: bool = False,
    polynomial_set: str = "kappa",
) -> Dict[str, List[float]]:
    """
    Convert a single NOLH sample to polynomial coefficient dictionaries.

    Args:
        sample: Array of parameter values
            - κ: 10 if fix_Q, 13 otherwise
            - κ*: 8 if fix_Q, 9 otherwise
        fix_Q: If True, sample only has P1/P2/P3 params; use baseline Q values
        polynomial_set: "kappa" (P2/P3 have 3 coeffs) or "kappa_star" (P2/P3 have 2 coeffs)

    Returns:
        Dict with keys 'P1_tilde', 'P2_tilde', 'P3_tilde', 'Q_basis'
    """
    if polynomial_set == "kappa_star":
        # κ*: P1(4) + P2(2) + P3(2) = 8 params
        result = {
            "P1_tilde": sample[0:4].tolist(),
            "P2_tilde": sample[4:6].tolist(),  # 2 coeffs
            "P3_tilde": sample[6:8].tolist(),  # 2 coeffs
        }
        if fix_Q:
            # Use κ* Q values (linear Q)
            result["Q_basis"] = {
                1: Q_BASIS_KAPPA_STAR[1],
            }
        else:
            # Q param is in sample[8]
            result["Q_basis"] = {
                1: sample[8],
            }
    else:
        # κ: P1(4) + P2(3) + P3(3) = 10 params
        result = {
            "P1_tilde": sample[0:4].tolist(),
            "P2_tilde": sample[4:7].tolist(),  # 3 coeffs
            "P3_tilde": sample[7:10].tolist(),  # 3 coeffs
        }
        if fix_Q:
            # Use PRZZ Q values
            result["Q_basis"] = {
                1: Q_BASIS_PRZZ[1],
                3: Q_BASIS_PRZZ[3],
                5: Q_BASIS_PRZZ[5],
            }
        else:
            # Q params are in sample[10:13]
            result["Q_basis"] = {
                1: sample[10],
                3: sample[11],
                5: sample[12],
            }

    return result


def polynomials_to_engine_format(polys: Dict, polynomial_set: str = "kappa") -> Dict[str, List[float]]:
    """
    Convert polynomial dict to KappaEngine format.

    The Q polynomial needs special handling:
    - Input: Q_basis with keys depending on polynomial_set
      - κ: keys 1, 3, 5
      - κ*: key 1 only (linear Q)
    - Output: Q_mono (monomial coefficients)

    We enforce Q(0)=1 by computing q_0 = 1 - sum(other basis coeffs)
    """
    # P1, P2, P3 pass through directly
    result = {
        "P1_tilde": polys["P1_tilde"],
        "P2_tilde": polys["P2_tilde"],
        "P3_tilde": polys["P3_tilde"],
    }

    # Q: Convert basis to monomial with Q(0)=1 constraint
    q_basis = polys["Q_basis"]

    if polynomial_set == "kappa_star":
        # κ*: Linear Q with only k=0,1 terms
        q_1 = q_basis[1]
        # Enforce Q(0) = 1: q_0 = 1 - q_1
        q_0 = 1.0 - q_1
        basis_coeffs = {0: q_0, 1: q_1}
    else:
        # κ: Full Q with k=0,1,3,5 terms
        q_1 = q_basis[1]
        q_3 = q_basis[3]
        q_5 = q_basis[5]
        # Enforce Q(0) = 1: q_0 = 1 - sum of other basis coeffs
        q_0 = 1.0 - (q_1 + q_3 + q_5)
        basis_coeffs = {0: q_0, 1: q_1, 3: q_3, 5: q_5}

    # Expand Q from (1-2x)^k basis to monomial basis
    from src.polynomials import QPolynomial

    Q = QPolynomial(basis_coeffs=basis_coeffs, enforce_Q0=False)
    Q_mono = Q.to_monomial()

    result["Q_mono"] = Q_mono.coeffs.tolist()

    return result


# =============================================================================
# VALIDATION
# =============================================================================

def validate_design(design: NOLHDesign) -> Tuple[bool, List[str]]:
    """
    Validate that design is properly constructed.

    Returns:
        (valid, issues) where issues is list of problems found
    """
    issues = []

    # Check dimensions
    if design.samples.shape != (design.n_samples, design.n_params):
        issues.append(f"Shape mismatch: {design.samples.shape} vs ({design.n_samples}, {design.n_params})")

    # Check bounds respected
    for i, (lo, hi) in enumerate(design.bounds):
        col = design.samples[:, i]
        if col.min() < lo - 1e-10:
            issues.append(f"Param {design.param_names[i]}: min {col.min():.6f} < bound {lo:.6f}")
        if col.max() > hi + 1e-10:
            issues.append(f"Param {design.param_names[i]}: max {col.max():.6f} > bound {hi:.6f}")

    # Check Latin property (each row/column should have unique stratification)
    # Simplified check: no duplicates in discretized values
    n = design.n_samples
    for i in range(design.n_params):
        col = design.samples[:, i]
        lo, hi = design.bounds[i]
        # Discretize to n bins
        bins = ((col - lo) / (hi - lo) * n).astype(int)
        bins = np.clip(bins, 0, n-1)
        if len(np.unique(bins)) < n * 0.8:  # Allow some tolerance
            issues.append(f"Param {design.param_names[i]}: poor Latin coverage ({len(np.unique(bins))}/{n})")

    return len(issues) == 0, issues


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("NOLH Design Generation Test")
    print("=" * 60)

    # Generate design
    design = generate_nolh_design(n_samples=49, seed=42)

    print(f"\nDesign created:")
    print(f"  Samples: {design.n_samples}")
    print(f"  Parameters: {design.n_params}")
    print(f"  Seed: {design.seed}")

    print(f"\nParameter bounds:")
    for i, (name, (lo, hi)) in enumerate(zip(design.param_names, design.bounds)):
        center = design.center[i]
        print(f"  {name}: [{lo:.6f}, {hi:.6f}] (center={center:.6f})")

    # Validate
    valid, issues = validate_design(design)
    print(f"\nValidation: {'PASS' if valid else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # Show sample statistics
    print(f"\nSample statistics:")
    for i, name in enumerate(design.param_names):
        col = design.samples[:, i]
        print(f"  {name}: min={col.min():.6f}, max={col.max():.6f}, mean={col.mean():.6f}")

    # Test conversion
    print(f"\nSample 0 as polynomials:")
    polys = sample_to_polynomials(design.samples[0])
    print(f"  P1_tilde: {polys['P1_tilde']}")
    print(f"  P2_tilde: {polys['P2_tilde']}")
    print(f"  P3_tilde: {polys['P3_tilde']}")
    print(f"  Q_basis: {polys['Q_basis']}")

    # Save test
    test_path = "/tmp/nolh_design_test.json"
    design.save(test_path)
    print(f"\nSaved to: {test_path}")

    # Load test
    loaded = NOLHDesign.load(test_path)
    print(f"Loaded: {loaded.n_samples} samples, {loaded.n_params} params")
