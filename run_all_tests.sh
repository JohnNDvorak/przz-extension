#!/bin/bash
#
# PRZZ Ψ Oracle Test Runner
# Run this from the przz-extension directory
#

set -e  # Exit on first error

cd /Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/przz-extension
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "========================================"
echo "PRZZ Ψ ORACLE TEST SUITE"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

run_test() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "----------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ PASSED${NC}: $name"
        ((passed++))
    else
        echo -e "${RED}✗ FAILED${NC}: $name"
        ((failed++))
    fi
}

echo ""
echo "========================================"
echo "PHASE 1: New Ψ Oracle Tests"
echo "========================================"

run_test "Ψ Term Generator" "python3 -m pytest tests/test_psi_term_generator.py -v --tb=short 2>&1 | tail -20"

run_test "(1,2) Oracle" "python3 -m pytest tests/test_psi_12_oracle.py -v --tb=short 2>&1 | tail -20"

run_test "(1,3) Oracle" "python3 -m pytest tests/test_psi_13_oracle.py -v --tb=short 2>&1 | tail -20"

run_test "(2,2) Oracle" "python3 -m pytest tests/test_psi_22_complete.py -v --tb=short 2>&1 | tail -20"

run_test "(2,3) Oracle" "python3 -m pytest tests/test_psi_23_oracle.py -v --tb=short 2>&1 | tail -20"

run_test "(3,3) Oracle" "python3 -m pytest tests/test_psi_33_oracle.py -v --tb=short 2>&1 | tail -20"

run_test "Unified Evaluator" "python3 -m pytest tests/test_psi_unified.py -v --tb=short 2>&1 | tail -20"

echo ""
echo "========================================"
echo "PHASE 2: Demo Runs"
echo "========================================"

echo ""
echo "----------------------------------------"
echo "Running: Unified Evaluator Demo"
echo "----------------------------------------"
python3 -c "
from src.psi_unified_evaluator import evaluate_c_psi, print_evaluation_report
from src.polynomials import load_przz_polynomials, load_przz_polynomials_kappa_star

print('=== κ Benchmark (R=1.3036) ===')
P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=True)
polys = {'P1': P1, 'P2': P2, 'P3': P3, 'Q': Q}
result_k = evaluate_c_psi(4/7, 1.3036, 60, polys)
print(f'c = {result_k.c_total:.6f} (target: 2.137)')
print(f'κ = {result_k.kappa:.6f} (target: 0.417)')

print()
print('=== κ* Benchmark (R=1.1167) ===')
P1s, P2s, P3s, Qs = load_przz_polynomials_kappa_star(enforce_Q0=True)
polys_s = {'P1': P1s, 'P2': P2s, 'P3': P3s, 'Q': Qs}
result_ks = evaluate_c_psi(4/7, 1.1167, 60, polys_s)
print(f'c = {result_ks.c_total:.6f} (target: 1.938)')
print(f'κ = {result_ks.kappa:.6f} (target: 0.408)')

print()
ratio = result_k.c_total / result_ks.c_total
print(f'=== Ratio c_κ/c_κ* = {ratio:.4f} (target: 1.10) ===')
" 2>&1 || echo -e "${YELLOW}Demo had issues but continuing...${NC}"

echo ""
echo "----------------------------------------"
echo "Running: Two-Benchmark Comparison"
echo "----------------------------------------"
python3 src/two_benchmark_psi_test.py 2>&1 | head -60 || echo -e "${YELLOW}Two-benchmark test had issues${NC}"

echo ""
echo "========================================"
echo "PHASE 3: Full Regression Suite"
echo "========================================"

echo ""
echo "Running full test suite (this may take a minute)..."
python3 -m pytest tests/ -q --tb=no 2>&1 | tail -10

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "Phase 1 Results: $passed passed, $failed failed"
echo ""
echo "Key files to check if issues:"
echo "  - src/psi_unified_evaluator.py"
echo "  - src/psi_*_oracle.py"
echo "  - tests/test_psi_*.py"
echo ""
echo "========================================"
echo "DONE"
echo "========================================"
