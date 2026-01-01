"""
scripts/nolh_optimization/analysis.py
Statistical Analysis for NOLH Optimization Results

Provides:
- Main effects computation
- Response surface fitting
- Parameter importance ranking
- Interaction detection

Created: 2025-12-28 (Phase 49)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from .runner import NOLHBatchResults, NOLHResult


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MainEffects:
    """Main effects of each parameter on c."""
    param_names: List[str]
    effects: Dict[str, float]  # Parameter name -> effect on c
    std_errors: Dict[str, float]  # Standard errors

    def ranked(self) -> List[Tuple[str, float]]:
        """Return parameters ranked by absolute effect."""
        return sorted(
            self.effects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

    def top_n(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N parameters by absolute effect."""
        return self.ranked()[:n]


@dataclass
class ResponseSurface:
    """Quadratic response surface model for c."""
    param_names: List[str]
    intercept: float
    linear_coeffs: Dict[str, float]
    quadratic_coeffs: Dict[str, float]
    interaction_coeffs: Dict[Tuple[str, str], float]
    r_squared: float

    def predict(self, params: Dict[str, float]) -> float:
        """Predict c for given parameter values."""
        result = self.intercept

        for name, coeff in self.linear_coeffs.items():
            result += coeff * params[name]

        for name, coeff in self.quadratic_coeffs.items():
            result += coeff * params[name] ** 2

        for (n1, n2), coeff in self.interaction_coeffs.items():
            result += coeff * params[n1] * params[n2]

        return result


# =============================================================================
# MAIN EFFECTS
# =============================================================================

def compute_main_effects(results: NOLHBatchResults) -> MainEffects:
    """
    Compute main effect of each parameter on c.

    Method: For each parameter, partition samples into low/high halves
    and compute the difference in mean response.

    Effect = mean(c | param > median) - mean(c | param < median)

    A negative effect means increasing the parameter decreases c (good).
    """
    valid_results = [r for r in results.results if r.valid]
    if len(valid_results) < 4:
        raise ValueError("Need at least 4 valid results for main effects")

    param_names = results.design.param_names
    effects = {}
    std_errors = {}

    # Extract parameter values and responses
    n_valid = len(valid_results)
    X = np.zeros((n_valid, len(param_names)))
    y = np.zeros(n_valid)

    for i, r in enumerate(valid_results):
        for j, name in enumerate(param_names):
            X[i, j] = r.params[name]
        y[i] = r.c

    # Compute effect for each parameter
    for j, name in enumerate(param_names):
        col = X[:, j]
        median = np.median(col)

        low_mask = col <= median
        high_mask = col > median

        if low_mask.sum() > 0 and high_mask.sum() > 0:
            mean_low = y[low_mask].mean()
            mean_high = y[high_mask].mean()
            effect = mean_high - mean_low

            # Standard error (rough estimate)
            std_low = y[low_mask].std() / np.sqrt(low_mask.sum())
            std_high = y[high_mask].std() / np.sqrt(high_mask.sum())
            se = np.sqrt(std_low**2 + std_high**2)
        else:
            effect = 0.0
            se = float('inf')

        effects[name] = effect
        std_errors[name] = se

    return MainEffects(
        param_names=param_names,
        effects=effects,
        std_errors=std_errors,
    )


# =============================================================================
# RESPONSE SURFACE
# =============================================================================

def fit_response_surface(
    results: NOLHBatchResults,
    include_interactions: bool = True,
    include_quadratic: bool = True,
) -> ResponseSurface:
    """
    Fit quadratic response surface model.

    Model: c = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + Σβᵢⱼxᵢxⱼ

    Uses sklearn LinearRegression for fitting.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    valid_results = [r for r in results.results if r.valid]
    if len(valid_results) < 10:
        raise ValueError("Need at least 10 valid results for response surface")

    param_names = results.design.param_names
    n_params = len(param_names)

    # Extract data
    n_valid = len(valid_results)
    X = np.zeros((n_valid, n_params))
    y = np.zeros(n_valid)

    for i, r in enumerate(valid_results):
        for j, name in enumerate(param_names):
            X[i, j] = r.params[name]
        y[i] = r.c

    # Normalize X for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # Determine polynomial degree
    degree = 2 if (include_interactions or include_quadratic) else 1

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)

    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Extract coefficients (in normalized space)
    # For simplicity, we'll just report linear coefficients denormalized
    linear_coeffs = {}
    for j, name in enumerate(param_names):
        # Linear coefficient in original space
        linear_coeffs[name] = model.coef_[j] / X_std[j]

    # Quadratic and interaction terms are more complex to extract
    # For now, leave them in normalized form
    quadratic_coeffs = {}
    interaction_coeffs = {}

    if include_quadratic:
        # Indices for x_i^2 terms
        for j, name in enumerate(param_names):
            idx = n_params + j  # Rough index, depends on PolynomialFeatures order
            if idx < len(model.coef_):
                quadratic_coeffs[name] = model.coef_[idx] / (X_std[j] ** 2)

    # R-squared
    y_pred = model.predict(X_poly)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ResponseSurface(
        param_names=param_names,
        intercept=model.intercept_,
        linear_coeffs=linear_coeffs,
        quadratic_coeffs=quadratic_coeffs,
        interaction_coeffs=interaction_coeffs,
        r_squared=r_squared,
    )


# =============================================================================
# ANALYSIS SUMMARY
# =============================================================================

def print_analysis_summary(results: NOLHBatchResults):
    """Print summary of NOLH analysis."""
    print("\n" + "=" * 60)
    print("NOLH ANALYSIS SUMMARY")
    print("=" * 60)

    # Basic stats
    valid = [r for r in results.results if r.valid]
    print(f"\nResults: {len(valid)}/{len(results.results)} valid")

    if not valid:
        print("No valid results to analyze.")
        return

    c_values = [r.c for r in valid]
    kappa_values = [r.kappa for r in valid]

    print(f"\nc statistics:")
    print(f"  min:  {min(c_values):.6f}")
    print(f"  max:  {max(c_values):.6f}")
    print(f"  mean: {np.mean(c_values):.6f}")
    print(f"  std:  {np.std(c_values):.6f}")

    print(f"\nκ statistics:")
    print(f"  min:  {min(kappa_values):.4f}")
    print(f"  max:  {max(kappa_values):.4f}")
    print(f"  mean: {np.mean(kappa_values):.4f}")

    # Main effects
    if len(valid) >= 4:
        print("\nMain Effects (effect on c):")
        effects = compute_main_effects(results)
        for name, effect in effects.top_n(10):
            direction = "↓" if effect < 0 else "↑"
            print(f"  {name}: {effect:+.6f} {direction}")

    # Best point
    best = results.best
    if best:
        print(f"\nBest point (lowest c):")
        print(f"  Point ID: {best.point_id}")
        print(f"  c:     {best.c:.6f}")
        print(f"  κ:     {best.kappa:.4f}")
        print(f"  Key parameters:")
        for name, val in sorted(best.params.items()):
            print(f"    {name}: {val:.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Analysis module - run via run_nolh_exploration.py")
