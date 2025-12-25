#!/usr/bin/env python3
"""
Manual test script for psi_term_generator.
Run this to verify the implementation works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now try imports
try:
    from psi_term_generator import generate_psi_terms, verify_expected_counts
    print("✓ Successfully imported psi_term_generator")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test (1,1)
print("\n" + "="*70)
print("Testing (1,1) pair")
print("="*70)

try:
    collection = generate_psi_terms(1, 1)
    print(f"✓ Generated {collection.total_terms} terms for (1,1)")

    if collection.total_terms == 4:
        print("✓ Count is correct (expected 4)")
    else:
        print(f"✗ Count is wrong (expected 4, got {collection.total_terms})")

    # Show the terms
    print("\nTerms:")
    for term in collection.terms:
        print(f"  {term}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test (2,2)
print("\n" + "="*70)
print("Testing (2,2) pair")
print("="*70)

try:
    collection = generate_psi_terms(2, 2)
    print(f"✓ Generated {collection.total_terms} terms for (2,2)")

    if collection.total_terms == 12:
        print("✓ Count is correct (expected 12)")
    else:
        print(f"✗ Count is wrong (expected 12, got {collection.total_terms})")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test (3,3)
print("\n" + "="*70)
print("Testing (3,3) pair")
print("="*70)

try:
    collection = generate_psi_terms(3, 3)
    print(f"✓ Generated {collection.total_terms} terms for (3,3)")

    if collection.total_terms == 27:
        print("✓ Count is correct (expected 27)")
    else:
        print(f"✗ Count is wrong (expected 27, got {collection.total_terms})")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run full validation
print("\n" + "="*70)
print("Running full validation")
print("="*70)

pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
result = verify_expected_counts(pairs)

if result:
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
else:
    print("\n✗ Some tests failed")
    sys.exit(1)
