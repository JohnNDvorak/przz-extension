# ARCHIVE INDEX

**Created:** 2025-12-18
**Purpose:** Document all archived files from the aggressive cleanup of przz-extension

## Summary

| Category | Files | Description |
|----------|-------|-------------|
| superseded_evaluators | 13 | Old evaluator implementations superseded by section7_evaluator.py |
| superseded_oracles | 4 | Duplicate oracle variants superseded by validated oracles |
| disproven_hypotheses | 7 | Hypothesis test scripts for approaches that were disproven |
| session_diagnostics | 62 | One-off debugging/analysis scripts from Sessions 10-12 |
| session_handoffs | 15 | Session progress docs and handoff notes |
| superseded_docs | 34 | Documentation subsumed by current reference docs |
| output_files | 6 | Text output artifacts from diagnostic runs |
| **TOTAL** | **141** | |

**Note:** Some modules initially archived were restored as they are still dependencies of active code.

---

## Superseded Evaluators (19 files)

These evaluators were superseded by `section7_evaluator.py` which is the current best implementation.

| File | Purpose | Superseded By |
|------|---------|---------------|
| evaluate.py | Old V1 DSL evaluation pipeline | section7_evaluator.py |
| term_dsl.py | Old V1 term DSL | psi_expansion.py |
| terms_k3_d1.py | Old V1 manual term tables (2-variable bug) | section7_evaluator.py |
| psi_series_evaluator.py | Series-coefficient extraction (Session 11) | section7_evaluator.py |
| section7_clean_evaluator.py | Earlier PRE-MIRROR variant | section7_evaluator.py |
| przz_generalized_iterm_evaluator.py | GenEval-style (incomplete weights) | section7_evaluator.py |
| przz_iterm_evaluator_fixed.py | Partial 0-based indexing fix | section7_evaluator.py |
| generalized_monomial_evaluator.py | Earlier per-monomial version | per_monomial_evaluator.py |
| przz_monomial_evaluator.py | Limited pair support | section7_evaluator.py |
| przz_iterm_monomial_evaluator.py | I-term + monomial hybrid | section7_evaluator.py |
| przz_section7_evaluator.py | Earlier Section 7 evaluator | section7_evaluator.py |
| psi_block_evaluator.py | Block-based Ψ evaluator | section7_evaluator.py |
| psi_general_evaluator.py | General Ψ evaluator (585 lines) | section7_evaluator.py |
| psi_monomial_evaluator.py | Minimal monomial evaluator | section7_evaluator.py |
| psi_unified_evaluator.py | Unified Ψ evaluator (Session 12) | section7_evaluator.py |
| section7_config_evaluator.py | Config-based Section 7 | section7_evaluator.py |
| series_evaluator.py | Series-based evaluator | hybrid_evaluator.py |
| calibrated_pconfig_engine.py | Calibrated polynomial config engine | section7_evaluator.py |
| section7_pconfig_engine.py | Section 7 poly config engine | section7_evaluator.py |

---

## Superseded Oracles (5 files)

Duplicate oracle variants superseded by validated pair-specific oracles.

| File | Purpose | Superseded By |
|------|---------|---------------|
| przz_22_oracle.py | (2,2) oracle (simplified) | przz_22_exact_oracle.py |
| psi_22_complete_oracle.py | (2,2) comprehensive oracle | psi_22_monomial_oracle.py |
| psi_22_full_oracle.py | Full PRZZ formula (2,2) | psi_22_monomial_oracle.py |
| psi_22_oracle.py | Simpler (2,2) oracle | psi_22_monomial_oracle.py |
| q_operator_oracle.py | Detailed Q-operator oracle | przz_section7_oracle.py |

---

## Disproven Hypotheses (7 files)

Hypothesis test scripts for approaches that were investigated and ruled out.

| File | Hypothesis | Finding |
|------|------------|---------|
| test_normalization_hypothesis.py | Factorial normalization fixes gap | DISPROVEN - No improvement |
| test_combined_normalization.py | Combined normalization factor | DISPROVEN - Wrong approach |
| test_factorial_normalization.py | Factorial normalization detailed | DISPROVEN |
| test_polynomial_norm_fix.py | Polynomial normalization fix | DISPROVEN |
| test_scale_factor.py | Global scale factor | DISPROVEN - Failed Benchmark 2 |
| test_theta_6_hypothesis.py | θ/6 factor correction | DISPROVEN |
| polynomial_normalization_hypothesis.py | Polynomial normalization | DISPROVEN |

---

## Session Diagnostics (67 files)

One-off debugging and analysis scripts from Sessions 10-12 investigating the two-benchmark failure.

### Analysis Scripts (from src/)
- analyze_all_pairs_derivatives.py - All-pairs derivative analysis
- analyze_derivative_ratio.py - Derivative ratio behavior
- analyze_psi_signs.py - Sign pattern analysis
- case_c_gap_analysis.py - Case C gap investigation
- case_c_integral.py - Case C integral handling
- case_c_investigation.py - Case C deep investigation
- case_distribution_analyzer.py - Case A/B/C distribution
- check_polynomial_integrals.py - Polynomial integral validation
- compare_term_structure.py - Term structure comparison
- composition.py - Polynomial composition utilities
- compute_c_exact_case_c.py - Exact c for Case C
- compute_corrected_c_v2.py - Corrected c (v2)
- compute_corrected_c.py - Corrected c (v1)
- compute_sign_statistics.py - Sign statistics
- debug_mirror_term.py - Mirror term debugging
- debug_structure.py - Structure debugging
- derivative_ratio_calculator.py - Derivative ratio computation
- gap_attribution.py - Gap attribution analysis
- gap_fingerprint.py - Gap fingerprinting
- i5_diagonal.py - I₅ arithmetic correction analysis
- investigate_12_anomaly.py - (1,2) anomaly investigation
- mirror_check.py - Mirror transformation check
- mirror_integral_analysis.py - Mirror integral analysis
- mirror_term_test.py - Mirror transformation test
- mirror_trace.py - Mirror transformation trace
- mollifier_profiles.py - Mollifier profile analysis
- pair_11_diagnostic.py - (1,1) specific diagnostic
- polynomial_scaling_analysis.py - Polynomial scaling analysis
- psi_block_analysis.py - Block structure analysis
- psi_block_configs.py - Block config definitions
- psi_combinatorial.py - Combinatorial structure
- psi_fd_mapping.py - FD mapping for Ψ
- psi_monomial_expansion.py - Monomial expansion details
- psi_numerical_validation.py - Numerical validation
- psi_term_generator.py - Term generation utility
- ratio_reversal_diagnostic.py - Ratio reversal analysis
- raw_vs_przz_diagnostic.py - Raw vs PRZZ comparison
- second_benchmark.py - Second benchmark test
- separate_vs_summed.py - Separate vs summed structure
- test_case_c_correction.py - Case C correction validation
- test_case_c_derivatives.py - Case C derivative tests
- test_clean_evaluator.py - Clean evaluator test
- test_i2_vs_derivative_terms.py - I₂ vs derivative comparison
- test_polynomial_coefficient_magnitudes.py - Coefficient magnitude analysis
- test_polynomial_integral_ratios.py - Polynomial integral ratios
- test_r_dependent_factor.py - R-dependent scaling test
- test_two_benchmark_gate.py - Two-benchmark validation
- track3_i2_baseline.py - I₂-only baseline diagnostic
- two_benchmark_psi_test.py - Ψ two-benchmark test
- variable_scaling_check.py - Variable scaling validation
- verify_przz_polynomials.py - Polynomial verification

### Root-Level Diagnostics
- analyze_derivative_reduction.py
- analyze_kappa_ratio.py
- analyze_polynomial_degree_scaling.py
- analyze_weight_detailed.py
- analyze_weight_suppression.py
- compute_diagnosis.py
- detailed_ratio_analysis.py
- full_c_breakdown.py
- ratio_reversal_diagnosis.py
- simple_ratio_breakdown.py
- canary_pair_analysis.py
- trace_22_monomials.py

### Key Findings from Diagnostics
1. **Polynomial degree matters**: κ* P₂/P₃ are degree 2, κ P₂/P₃ are degree 3
2. **Ratio reversal identified**: I₂ ratio 1.66, derivative ratio 3.54, combined 2.09 (target 1.10)
3. **Case C makes gap worse**: 71% decrease, not improvement
4. **(1-a)^k attenuation**: Suppresses higher-degree terms by ~50%
5. **R-dependent scaling**: Both oracle AND DSL fail two-benchmark gate

---

## Session Handoffs (15 files)

Session progress documentation and handoff notes for knowledge transfer.

| File | Date | Purpose |
|------|------|---------|
| HANDOFF_2025_12_17_SESSION2.md | 2025-12-17 | Ratio reversal mystery analysis |
| HANDOFF_2025_12_17_SESSION3.md | 2025-12-17 | Derivative analysis |
| HANDOFF_2025_12_17_SESSION4.md | 2025-12-17 | Continuation |
| HANDOFF_2025_12_17_SESSION5.md | 2025-12-17 | Continuation |
| HANDOFF_2025_12_18_SESSION6.md | 2025-12-18 | Track 1 investigation |
| HANDOFF_2025_12_18_SESSION7.md | 2025-12-18 | Track 2 investigation |
| HANDOFF_2025_12_18_SESSION8.md | 2025-12-18 | Per-monomial evaluator discovery |
| SESSION_10_FINDINGS.md | 2025-12-17 | Session 10 findings |
| SESSION_11_PROGRESS.md | 2025-12-17 | Session 11 progress |
| SESSION_12_PROGRESS.md | 2025-12-18 | Session 12 progress |
| SESSION_12_KERNEL_RESEARCH.md | 2025-12-18 | Kernel investigation notes |
| SESSION_SUMMARY_2025_12_17.md | 2025-12-17 | Dec 17 consolidated findings |
| SESSION_TRANSITION_2025_12_15.md | 2025-12-15 | Session transition |
| SESSION_TRANSITION_BARB.md | - | Session transition |
| SESSION_TRANSITION.md | - | Session transition |

---

## Superseded Docs (34 files)

Documentation that has been subsumed by comprehensive current docs or contains outdated analysis.

### From docs/
| File | Subsumed By |
|------|-------------|
| CASE_C_FINDINGS.md | CASE_C_ANALYSIS.md |
| GAP_ANALYSIS_SUMMARY.md | HANDOFF_SUMMARY.md |
| RATIO_REVERSAL_INVESTIGATION.md | FD_KERNEL_ATTENUATION_REPORT.md |
| ORACLE_INVESTIGATION_2025_12_16.md | Methodology locked |
| DEEP_INVESTIGATION_2025_12_16.md | HANDOFF_SUMMARY.md |
| RATIO_REVERSAL_ANALYSIS.md | FD_KERNEL_ATTENUATION_REPORT.md |
| AGENT_FINDINGS_CONSOLIDATED.md | Session output |
| GPT_FEEDBACK_INTEGRATION.md | Session output |
| PER_MONOMIAL_ANALYSIS.md | HANDOFF_SUMMARY.md |
| INVESTIGATION_POLYNOMIAL_NORMALIZATION.md | Session output |
| R_SCALING_INVESTIGATION.md | HANDOFF_SUMMARY.md |
| PSI_33_ORACLE_IMPLEMENTATION.md | Implementation detail |

### From root
- COMPUTATION_CHECKLIST.md
- CROSS_PAIR_ORACLES_SUMMARY.md
- DEGREE_SCALING_SUMMARY.md
- DERIVATIVE_ANALYSIS_REPORT.md
- DIAGNOSIS_COMPLETE.md
- DIAGNOSIS_INDEX.md
- DIAGNOSIS_REPORT.md
- MYSTERY_SOLVED.md
- NUMERICAL_PREDICTIONS.md
- POLYNOMIAL_DEGREE_ANALYSIS_REPORT.md
- PSI_22_ORACLE_DESIGN.md
- PSI_22_ORACLE_IMPLEMENTATION_SUMMARY.md
- PSI_TERM_GENERATOR_README.md
- PSI_UNIFIED_EVALUATOR_README.md
- QUICK_START_PSI22_ORACLE.md
- RATIO_ANALYSIS_SUMMARY.md
- RATIO_BREAKDOWN_ANALYSIS.md
- RATIO_REVERSAL_SUMMARY.md
- SMOKING_GUN.md
- TERM_STRUCTURE_ANALYSIS.md
- TWO_BENCHMARK_PSI_ORACLE_REPORT.md
- WEIGHT_INVESTIGATION_REPORT.md

---

## Output Files (6 files)

Text output artifacts from diagnostic runs.

| File | Source |
|------|--------|
| detailed_analysis_output.txt | analyze_weight_detailed.py |
| full_breakdown_output.txt | full_c_breakdown.py |
| ratio_analysis_output.txt | analyze_kappa_ratio.py |
| RATIO_TABLE.txt | Ratio analysis |
| QUICK_REFERENCE.txt | Reference card |
| TERM_RATIO_SUMMARY.txt | Term ratio analysis |

---

## Current Active Files (Post-Cleanup)

### src/ (26 files)
**Core (4):** polynomials.py, quadrature.py, series.py, psi_expansion.py
**Evaluators (4):** section7_evaluator.py, per_monomial_evaluator.py, hybrid_evaluator.py, series_backed_evaluator.py
**Oracles (6):** przz_22_exact_oracle.py, psi_22_monomial_oracle.py, psi_12_oracle.py, psi_13_oracle.py, psi_23_oracle.py, psi_33_oracle.py
**FD Validation (4):** fd_oracle.py, fd_oracle_22.py, fd_evaluation.py, section7_fd_evaluator.py
**Supporting (7):** paper_integrator.py, psi_separated_c.py, reference_bivariate.py, przz_section7_oracle.py, case_c_kernel.py, case_c_exact.py, arithmetic_constants.py
**Init (1):** __init__.py

### docs/ (15 files)
- HANDOFF_SUMMARY.md - Complete project state
- TRUTH_SPEC.md - Mathematical specification
- CASE_C_ANALYSIS.md - Case C reference
- CASE_C_KEY_QUESTIONS.md - Quick reference
- CASE_DISTRIBUTION_ANALYSIS.md - Per-pair structure
- FD_KERNEL_ATTENUATION_REPORT.md - Polynomial degree analysis
- CANARY_PAIR_VERDICT.md - Kernel verdict
- PHASE_1_READINESS.md - Milestone tracking
- PSI_IMPLEMENTATION_PLAN.md - Ψ expansion roadmap
- PSI_INVESTIGATION_FINDINGS.md - Complete monomial enumeration
- PSI_12_ORACLE_README.md - (1,2) oracle specification
- SIGN_PATTERN_ANALYSIS.md - Sign handling reference
- STRUCTURE_COMPARISON.md - DSL vs PRZZ comparison
- OMEGA_CASE_MAPPING.md - Indexing convention
- target_assumptions.md - PRZZ variant specification

### Root Level (10 files)
- CLAUDE.md - Project guidelines
- README.md - Project readme
- ARCHITECTURE.md - Implementation design
- TECHNICAL_ANALYSIS.md - Mathematical derivations
- TERM_DSL.md - Term data model specification
- VALIDATION.md - Test plan and checkpoints
- OPTIMIZATION.md - Strategy for improving κ
- run_two_benchmark_gate.py - Validation gate
- run_psi_12_oracle.py - (1,2) validation
- run_all_tests.sh - Test runner

### tests/ (26 files)
All test files retained.

---

## Impact Summary

| Location | Before | After | Reduction |
|----------|--------|-------|-----------|
| src/ | 108 | 38 | 65% |
| docs/ | 36 | 15 | 58% |
| root | 59 | 10 | 83% |
| **Total** | **203** | **63 + 26 tests** | **69%** |

**Files Restored (Dependencies):**
- evaluate.py, term_dsl.py, terms_k3_d1.py (still used by oracles/tests)
- psi_series_evaluator.py (used by hybrid_evaluator)
- psi_unified_evaluator.py, psi_22_complete_oracle.py (used by tests)
- composition.py, i5_diagonal.py, przz_generalized_iterm_evaluator.py (used by tests)
- psi_monomial_expansion.py, psi_term_generator.py, psi_block_configs.py (dependency chain)
