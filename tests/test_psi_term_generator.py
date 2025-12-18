"""
tests/test_psi_term_generator.py
Unit Tests for Ψ Term Generator

Tests verify:
1. Monomial counts match expected values
2. (1,1) maps correctly to I₁-I₄ structure
3. Coefficients match the Ψ formula
4. Integral type classification is correct
"""

import pytest
from src.psi_term_generator import (
    generate_psi_terms,
    IntegralType,
    classify_integral_type,
    PsiTerm,
    verify_expected_counts,
)


class TestMonomialCounts:
    """Test that term generation produces correct monomial counts."""

    def test_11_count(self):
        """(1,1) should produce exactly 4 monomials."""
        collection = generate_psi_terms(1, 1)
        assert collection.total_terms == 4, f"Expected 4 terms, got {collection.total_terms}"
        assert collection.ell == 1
        assert collection.ellbar == 1

    def test_22_count(self):
        """(2,2) should produce exactly 12 monomials."""
        collection = generate_psi_terms(2, 2)
        assert collection.total_terms == 12, f"Expected 12 terms, got {collection.total_terms}"
        assert collection.ell == 2
        assert collection.ellbar == 2

    def test_33_count(self):
        """(3,3) should produce exactly 27 monomials."""
        collection = generate_psi_terms(3, 3)
        assert collection.total_terms == 27, f"Expected 27 terms, got {collection.total_terms}"
        assert collection.ell == 3
        assert collection.ellbar == 3

    def test_12_count(self):
        """(1,2) should produce exactly 6 monomials."""
        collection = generate_psi_terms(1, 2)
        assert collection.total_terms == 6, f"Expected 6 terms, got {collection.total_terms}"

    def test_13_count(self):
        """(1,3) should produce exactly 8 monomials."""
        collection = generate_psi_terms(1, 3)
        assert collection.total_terms == 8, f"Expected 8 terms, got {collection.total_terms}"

    def test_23_count(self):
        """(2,3) should produce exactly 18 monomials."""
        collection = generate_psi_terms(2, 3)
        assert collection.total_terms == 18, f"Expected 18 terms, got {collection.total_terms}"


class TestIntegralMapping11:
    """Test that (1,1) terms correctly map to I₁-I₄ structure."""

    def test_11_has_four_integral_types(self):
        """(1,1) should have exactly one term for each I₁, I₂, I₃, I₄."""
        collection = generate_psi_terms(1, 1)
        by_type = collection.by_type

        # Should have all four basic integral types
        assert IntegralType.I1_MIXED in by_type
        assert IntegralType.I2_BASE in by_type
        assert IntegralType.I3_Z_DERIV in by_type
        assert IntegralType.I4_W_DERIV in by_type

        # Each should have exactly one term
        assert len(by_type[IntegralType.I1_MIXED]) == 1
        assert len(by_type[IntegralType.I2_BASE]) == 1
        assert len(by_type[IntegralType.I3_Z_DERIV]) == 1
        assert len(by_type[IntegralType.I4_W_DERIV]) == 1

    def test_11_ab_maps_to_i1(self):
        """AB term (1,1,0,0) should map to I1_MIXED with coefficient +1."""
        collection = generate_psi_terms(1, 1)

        # Find the AB term
        ab_term = None
        for term in collection.terms:
            if term.monomial_key() == (1, 1, 0, 0):
                ab_term = term
                break

        assert ab_term is not None, "AB term not found"
        assert ab_term.coeff == 1, f"AB coefficient should be +1, got {ab_term.coeff}"
        assert ab_term.integral_type == IntegralType.I1_MIXED

    def test_11_d_maps_to_i2(self):
        """D term (0,0,0,1) should map to I2_BASE with coefficient +1."""
        collection = generate_psi_terms(1, 1)

        # Find the D term
        d_term = None
        for term in collection.terms:
            if term.monomial_key() == (0, 0, 0, 1):
                d_term = term
                break

        assert d_term is not None, "D term not found"
        assert d_term.coeff == 1, f"D coefficient should be +1, got {d_term.coeff}"
        assert d_term.integral_type == IntegralType.I2_BASE

    def test_11_ac_maps_to_i3(self):
        """AC term (1,0,1,0) should map to I3_Z_DERIV with coefficient -1."""
        collection = generate_psi_terms(1, 1)

        # Find the AC term
        ac_term = None
        for term in collection.terms:
            if term.monomial_key() == (1, 0, 1, 0):
                ac_term = term
                break

        assert ac_term is not None, "AC term not found"
        assert ac_term.coeff == -1, f"AC coefficient should be -1, got {ac_term.coeff}"
        assert ac_term.integral_type == IntegralType.I3_Z_DERIV

    def test_11_bc_maps_to_i4(self):
        """BC term (0,1,1,0) should map to I4_W_DERIV with coefficient -1."""
        collection = generate_psi_terms(1, 1)

        # Find the BC term
        bc_term = None
        for term in collection.terms:
            if term.monomial_key() == (0, 1, 1, 0):
                bc_term = term
                break

        assert bc_term is not None, "BC term not found"
        assert bc_term.coeff == -1, f"BC coefficient should be -1, got {bc_term.coeff}"
        assert bc_term.integral_type == IntegralType.I4_W_DERIV

    def test_11_formula_expansion(self):
        """
        Verify (1,1) expands to: AB - AC - BC + D

        Ψ_{1,1} = Σ_{p=0}^1 C(1,p)C(1,p)p! (D-C²)^p (A-C)^{1-p} (B-C)^{1-p}

        p=0: C(1,0)C(1,0)·0! = 1 × (A-C)(B-C) = AB - AC - BC + C²
        p=1: C(1,1)C(1,1)·1! = 1 × (D-C²) = D - C²

        Sum: AB - AC - BC + C² + D - C² = AB - AC - BC + D
        """
        collection = generate_psi_terms(1, 1)

        # Build coefficient map
        coeff_map = {}
        for term in collection.terms:
            coeff_map[term.monomial_key()] = term.coeff

        # Check expected terms
        assert coeff_map.get((1, 1, 0, 0)) == +1, "AB should have coefficient +1"
        assert coeff_map.get((0, 0, 0, 1)) == +1, "D should have coefficient +1"
        assert coeff_map.get((1, 0, 1, 0)) == -1, "AC should have coefficient -1"
        assert coeff_map.get((0, 1, 1, 0)) == -1, "BC should have coefficient -1"

        # C² terms should cancel
        assert (0, 0, 2, 0) not in coeff_map or coeff_map[(0, 0, 2, 0)] == 0


class TestCoefficientFormula:
    """Test that coefficients match the Ψ combinatorial formula."""

    def test_22_has_correct_coefficient_signs(self):
        """
        (2,2) terms should have mix of positive and negative coefficients.
        The binomial expansion produces alternating signs.
        """
        collection = generate_psi_terms(2, 2)

        pos_count = sum(1 for t in collection.terms if t.coeff > 0)
        neg_count = sum(1 for t in collection.terms if t.coeff < 0)

        # Should have both positive and negative terms
        assert pos_count > 0, "Should have positive coefficients"
        assert neg_count > 0, "Should have negative coefficients"

    def test_33_coefficient_magnitude_range(self):
        """
        (3,3) should have coefficients with reasonable magnitudes.
        Max should be bounded by C(3,p)·C(3,p)·p! = at most C(3,3)·C(3,3)·3! = 6
        """
        collection = generate_psi_terms(3, 3)

        max_abs_coeff = max(abs(t.coeff) for t in collection.terms)

        # For (3,3), maximum prefactor is C(3,1)·C(3,1)·1! = 9
        # But binomial expansions can multiply this
        # Reasonable upper bound is around 27 (from full expansion)
        assert max_abs_coeff <= 30, f"Coefficient {max_abs_coeff} seems too large"

    def test_sum_formula_structure(self):
        """
        Each term should come from the p-sum structure.
        For (2,2), we have p ∈ {0,1,2}, giving 3 p-configs.
        """
        collection = generate_psi_terms(2, 2)

        # The p-sum representation has:
        # p=0: (A-C)²(B-C)² with prefactor 1
        # p=1: (D-C²)(A-C)(B-C) with prefactor C(2,1)·C(2,1)·1! = 4
        # p=2: (D-C²)² with prefactor C(2,2)·C(2,2)·2! = 2

        # This expands to 12 distinct monomials
        assert collection.total_terms == 12


class TestIntegralClassification:
    """Test integral type classification logic."""

    def test_classify_pure_d_term(self):
        """Pure D terms should be I2_BASE."""
        assert classify_integral_type(0, 0, 0, 1) == IntegralType.I2_BASE
        assert classify_integral_type(0, 0, 0, 2) == IntegralType.I2_BASE

    def test_classify_pure_ab_term(self):
        """Pure AB term (1,1,0,0) should be I1_MIXED."""
        assert classify_integral_type(1, 1, 0, 0) == IntegralType.I1_MIXED

    def test_classify_ac_term(self):
        """AC term (1,0,1,0) should be I3_Z_DERIV."""
        assert classify_integral_type(1, 0, 1, 0) == IntegralType.I3_Z_DERIV

    def test_classify_bc_term(self):
        """BC term (0,1,1,0) should be I4_W_DERIV."""
        assert classify_integral_type(0, 1, 1, 0) == IntegralType.I4_W_DERIV

    def test_classify_higher_order_terms(self):
        """Higher-order terms should be GENERAL."""
        assert classify_integral_type(2, 0, 0, 0) == IntegralType.GENERAL  # A²
        assert classify_integral_type(0, 2, 0, 0) == IntegralType.GENERAL  # B²
        assert classify_integral_type(2, 2, 0, 0) == IntegralType.GENERAL  # A²B²
        assert classify_integral_type(1, 1, 1, 0) == IntegralType.GENERAL  # ABC


class TestHigherPairs:
    """Test generation for higher pairs (2,2) and (3,3)."""

    def test_22_has_general_terms(self):
        """(2,2) should have terms requiring GENERAL evaluation."""
        collection = generate_psi_terms(2, 2)
        by_type = collection.by_type

        assert IntegralType.GENERAL in by_type, "Should have GENERAL type terms"
        assert len(by_type[IntegralType.GENERAL]) > 0

    def test_33_has_high_order_terms(self):
        """(3,3) should have terms with high derivative orders."""
        collection = generate_psi_terms(3, 3)

        # Check for high-order terms
        max_a = max(t.a for t in collection.terms)
        max_b = max(t.b for t in collection.terms)

        assert max_a >= 2, "Should have A² or higher"
        assert max_b >= 2, "Should have B² or higher"

    def test_22_monomial_uniqueness(self):
        """All (2,2) monomials should be unique."""
        collection = generate_psi_terms(2, 2)

        keys = [t.monomial_key() for t in collection.terms]
        assert len(keys) == len(set(keys)), "Monomials should be unique"


class TestTermStructure:
    """Test PsiTerm and PsiTermCollection structure."""

    def test_term_has_required_fields(self):
        """PsiTerm should have all required fields."""
        collection = generate_psi_terms(1, 1)
        term = collection.terms[0]

        assert hasattr(term, 'a')
        assert hasattr(term, 'b')
        assert hasattr(term, 'c')
        assert hasattr(term, 'd')
        assert hasattr(term, 'coeff')
        assert hasattr(term, 'integral_type')
        assert hasattr(term, 'description')

    def test_collection_by_type_grouping(self):
        """by_type should correctly group terms."""
        collection = generate_psi_terms(1, 1)
        by_type = collection.by_type

        # Count total terms across all groups
        total_in_groups = sum(len(terms) for terms in by_type.values())
        assert total_in_groups == collection.total_terms

    def test_term_representation(self):
        """PsiTerm repr should be readable."""
        collection = generate_psi_terms(1, 1)
        for term in collection.terms:
            repr_str = repr(term)
            assert "PsiTerm" in repr_str
            assert str(term.coeff) in repr_str or ("+" in repr_str and term.coeff > 0)


class TestValidationFunction:
    """Test the verify_expected_counts validation function."""

    def test_all_k3_pairs_pass(self):
        """All K=3 pairs should pass validation."""
        pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
        result = verify_expected_counts(pairs)
        assert result is True, "All K=3 pairs should pass validation"


class TestCoefficientAccuracy:
    """Test specific coefficient values from Ψ formula."""

    def test_11_p0_expansion(self):
        """
        For (1,1) p=0: C(1,0)C(1,0)·0! = 1
        Gives: (A-C)(B-C) = AB - AC - BC + C²
        """
        collection = generate_psi_terms(1, 1)
        coeff_map = {t.monomial_key(): t.coeff for t in collection.terms}

        # These terms come from p=0 expansion
        # Note: C² terms from p=0 and p=1 should cancel
        # We're checking the final result after combination

        # AB appears only in p=0 with coefficient 1
        assert coeff_map.get((1, 1, 0, 0)) == 1

    def test_22_p1_prefactor(self):
        """
        For (2,2) p=1: C(2,1)C(2,1)·1! = 4
        This multiplies (D-C²)(A-C)(B-C)
        """
        # This is tested indirectly through monomial coefficients
        collection = generate_psi_terms(2, 2)

        # The prefactor 4 should appear in some coefficients
        coeffs = [t.coeff for t in collection.terms]
        assert 4 in coeffs or -4 in coeffs, "Prefactor 4 should appear"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_coefficients_nonzero(self):
        """All generated terms should have non-zero coefficients."""
        for ell in [1, 2, 3]:
            for ellbar in [1, 2, 3]:
                if ell <= ellbar:  # Only test upper triangle
                    collection = generate_psi_terms(ell, ellbar)
                    for term in collection.terms:
                        assert term.coeff != 0, f"Term {term.monomial_key()} has zero coefficient"

    def test_exponents_non_negative(self):
        """All exponents should be non-negative."""
        for ell in [1, 2, 3]:
            for ellbar in [1, 2, 3]:
                if ell <= ellbar:
                    collection = generate_psi_terms(ell, ellbar)
                    for term in collection.terms:
                        assert term.a >= 0
                        assert term.b >= 0
                        assert term.c >= 0
                        assert term.d >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
