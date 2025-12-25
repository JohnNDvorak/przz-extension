#!/usr/bin/env python3
"""
Extract polynomial coefficients from RMS_PRZZ.tex.

Parses PRZZ TeX lines 2567-2586 (kappa) and 2587-2598 (kappa*) to extract
polynomial coefficients directly from the source, eliminating transcription errors.

Usage:
    python tools/extract_polys_from_przz_txt.py
    python tools/extract_polys_from_przz_txt.py --validate
    python tools/extract_polys_from_przz_txt.py --output src/data/przz_polys.json
"""

from __future__ import annotations
import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class ExtractedPolynomial:
    """Polynomial coefficients extracted from TeX."""
    name: str                   # "P1", "P2", "P3", "Q"
    form: str                   # "constrained", "monomial", "przz_basis"
    raw_tex: str                # Original TeX string
    coefficients: List[float]   # Parsed coefficients
    line_number: int            # TeX line number for traceability


@dataclass
class ExtractedBenchmark:
    """All polynomials for one benchmark."""
    name: str                   # "kappa" or "kappa_star"
    R: float
    theta: str                  # "4/7"
    kappa_target: float
    P1: ExtractedPolynomial
    P2: ExtractedPolynomial
    P3: ExtractedPolynomial
    Q: ExtractedPolynomial
    tex_line_start: int
    tex_line_end: int


def parse_number(s: str) -> float:
    """Parse a number from TeX, handling negative signs and spaces."""
    s = s.strip().replace(" ", "")
    return float(s)


def parse_p1_tilde_coeffs(tex_lines: List[str], start_line: int) -> Tuple[List[float], str, int]:
    """
    Parse P1 tilde coefficients from TeX.

    P1 form: x + c0*x(1-x) + c1*x(1-x)^2 + c2*x(1-x)^3 + c3*x(1-x)^4

    Returns:
        (tilde_coeffs, raw_tex, line_number)
    """
    # Find P1 definition
    for i, line in enumerate(tex_lines[start_line:], start=start_line):
        if "P_{1}(x)" in line and "&=" in line:
            # Extract full definition (may span multiple lines)
            full_line = line
            j = i + 1
            while j < len(tex_lines) and "\\\\" not in tex_lines[j-1]:
                full_line += " " + tex_lines[j]
                j += 1

            # Parse coefficients: look for patterns like "0.261076 x (1 - x)"
            # Pattern: sign number x (1 - x)^k or sign number x (1-x)^k
            raw_tex = full_line.strip()

            # Extract coefficients from patterns like "+ 0.261076 x (1 - x)"
            # Regex to match coefficient patterns
            pattern = r'([+-]?\s*[\d.]+)\s*x\s*\(1\s*-\s*x\)(?:\^(\d+))?'
            matches = re.findall(pattern, full_line)

            coeffs = []
            for coeff_str, power_str in matches:
                coeff = parse_number(coeff_str)
                power = int(power_str) if power_str else 1
                # Extend list if needed
                while len(coeffs) < power:
                    coeffs.append(0.0)
                coeffs[power - 1] = coeff

            return coeffs, raw_tex, i + 1

    raise ValueError(f"Could not find P1 definition starting from line {start_line}")


def parse_p_monomial_coeffs(tex_lines: List[str], start_line: int, poly_name: str) -> Tuple[List[float], str, int]:
    """
    Parse P2/P3 monomial coefficients from TeX.

    P2/P3 form: c1*x + c2*x^2 + c3*x^3 + ...

    Returns:
        (tilde_coeffs, raw_tex, line_number) where tilde_coeffs = [c1, c2, c3, ...]
    """
    search_str = f"P_{{{poly_name[-1]}}}(x)"

    for i, line in enumerate(tex_lines[start_line:], start=start_line):
        if search_str in line and "&=" in line:
            # Collect lines until we hit the next P or end of align block
            full_line = line
            j = i + 1
            while j < len(tex_lines):
                next_line = tex_lines[j]
                # Stop if we hit the next polynomial definition or align end
                if "P_{" in next_line or "Q(x)" in next_line or "\\end{align" in next_line:
                    break
                full_line += " " + next_line
                j += 1
                # Also stop after the first "\\" that ends this definition
                if "\\\\" in next_line:
                    break

            raw_tex = full_line.strip()

            # Extract coefficients: patterns like "1.048274 x" or "- 0.940058 x^3"
            # Match: optional sign, number, x, optional ^power
            pattern = r'([+-]?\s*[\d.]+)\s*x(?:\^(\d+))?'
            matches = re.findall(pattern, full_line)

            coeffs = []
            for coeff_str, power_str in matches:
                coeff = parse_number(coeff_str)
                power = int(power_str) if power_str else 1
                # Extend list if needed
                while len(coeffs) < power:
                    coeffs.append(0.0)
                coeffs[power - 1] = coeff

            return coeffs, raw_tex, i + 1

    raise ValueError(f"Could not find {poly_name} definition starting from line {start_line}")


def parse_q_basis_coeffs(tex_lines: List[str], start_line: int) -> Tuple[Dict[int, float], str, int]:
    """
    Parse Q coefficients in (1-2x)^k basis from TeX.

    Q form: c0 + c1*(1-2x) + c3*(1-2x)^3 + c5*(1-2x)^5

    Returns:
        ({k: ck}, raw_tex, line_number)
    """
    for i, line in enumerate(tex_lines[start_line:], start=start_line):
        if "Q(x)" in line and "=" in line:
            # Collect full Q definition (may span multiple lines until \end{align})
            full_line = line
            j = i + 1
            while j < len(tex_lines):
                next_line = tex_lines[j]
                full_line += " " + next_line
                j += 1
                # Stop at end of align block
                if "\\end{align" in next_line:
                    break

            raw_tex = full_line.strip()
            coeffs = {}

            # First, check for special form: "0.483777 + (1 - 0.483777) (1 - 2 x)"
            special_match = re.search(r'([\d.]+)\s*\+\s*\(1\s*-\s*([\d.]+)\)\s*\(1\s*-\s*2\s*x\)', full_line)
            if special_match:
                c0 = parse_number(special_match.group(1))
                c0_check = parse_number(special_match.group(2))
                if abs(c0 - c0_check) < 1e-10:
                    c1 = 1.0 - c0
                    coeffs = {0: c0, 1: c1}
                    return coeffs, raw_tex, i + 1

            # Standard form: extract constant and (1-2x)^k terms
            # Match constant term (at beginning or after =)
            const_match = re.search(r'=\s*([\d.]+)(?:\s*[+-])', full_line)
            if const_match:
                coeffs[0] = parse_number(const_match.group(1))

            # Match (1-2x)^k terms: "sign coeff (1 - 2 x)^k" or "sign coeff (1-2x)^k"
            # Also handle forms like "0.636851 (1 - 2 x)" without explicit sign
            pattern = r'([+-]?\s*[\d.]+)\s*\(1\s*-\s*2\s*x\)(?:\^(\d+))?'
            matches = re.findall(pattern, full_line)

            for coeff_str, power_str in matches:
                coeff = parse_number(coeff_str)
                power = int(power_str) if power_str else 1
                coeffs[power] = coeff

            if coeffs:
                return coeffs, raw_tex, i + 1

    raise ValueError(f"Could not find Q definition starting from line {start_line}")


def extract_kappa_benchmark(tex_content: str) -> ExtractedBenchmark:
    """
    Extract kappa benchmark data from TeX (lines 2567-2586).
    """
    lines = tex_content.split('\n')

    # Find the start: "R=1.3036"
    start_line = None
    for i, line in enumerate(lines):
        if "R=1.3036" in line or "R = 1.3036" in line:
            start_line = i
            break

    if start_line is None:
        raise ValueError("Could not find kappa benchmark start (R=1.3036)")

    # Parse polynomials
    p1_coeffs, p1_tex, p1_line = parse_p1_tilde_coeffs(lines, start_line)
    p2_coeffs, p2_tex, p2_line = parse_p_monomial_coeffs(lines, start_line, "P2")
    p3_coeffs, p3_tex, p3_line = parse_p_monomial_coeffs(lines, start_line, "P3")
    q_coeffs, q_tex, q_line = parse_q_basis_coeffs(lines, start_line)

    # Find end: "kappa >= 0.417293962"
    end_line = start_line + 30  # reasonable default
    for i, line in enumerate(lines[start_line:], start=start_line):
        if "0.417293962" in line:
            end_line = i
            break

    return ExtractedBenchmark(
        name="kappa",
        R=1.3036,
        theta="4/7",
        kappa_target=0.417293962,
        P1=ExtractedPolynomial("P1", "constrained", p1_tex, p1_coeffs, p1_line),
        P2=ExtractedPolynomial("P2", "monomial", p2_tex, p2_coeffs, p2_line),
        P3=ExtractedPolynomial("P3", "monomial", p3_tex, p3_coeffs, p3_line),
        Q=ExtractedPolynomial("Q", "przz_basis", q_tex, list(q_coeffs.items()), q_line),
        tex_line_start=start_line + 1,
        tex_line_end=end_line + 1,
    )


def extract_kappa_star_benchmark(tex_content: str) -> ExtractedBenchmark:
    """
    Extract kappa* benchmark data from TeX (lines 2587-2598).
    """
    lines = tex_content.split('\n')

    # Find the start: after "simple zeros" or "R=1.1167"
    start_line = None
    for i, line in enumerate(lines):
        if "R=1.1167" in line or "R = 1.1167" in line:
            start_line = max(0, i - 15)  # Look back for P1
            break

    if start_line is None:
        # Alternative: search for the second P1 definition
        p1_count = 0
        for i, line in enumerate(lines):
            if "P_{1}(x)" in line and "&=" in line:
                p1_count += 1
                if p1_count == 2:
                    start_line = i
                    break

    if start_line is None:
        raise ValueError("Could not find kappa* benchmark start")

    # Parse polynomials
    p1_coeffs, p1_tex, p1_line = parse_p1_tilde_coeffs(lines, start_line)
    p2_coeffs, p2_tex, p2_line = parse_p_monomial_coeffs(lines, start_line, "P2")
    p3_coeffs, p3_tex, p3_line = parse_p_monomial_coeffs(lines, start_line, "P3")
    q_coeffs, q_tex, q_line = parse_q_basis_coeffs(lines, start_line)

    # Find end: "0.407511457"
    end_line = start_line + 30
    for i, line in enumerate(lines[start_line:], start=start_line):
        if "0.407511457" in line:
            end_line = i
            break

    # Convert Q coeffs dict to list for storage
    q_coeffs_list = [(k, v) for k, v in sorted(q_coeffs.items())]

    return ExtractedBenchmark(
        name="kappa_star",
        R=1.1167,
        theta="4/7",
        kappa_target=0.407511457,
        P1=ExtractedPolynomial("P1", "constrained", p1_tex, p1_coeffs, p1_line),
        P2=ExtractedPolynomial("P2", "monomial", p2_tex, p2_coeffs, p2_line),
        P3=ExtractedPolynomial("P3", "monomial", p3_tex, p3_coeffs, p3_line),
        Q=ExtractedPolynomial("Q", "przz_basis", q_tex, q_coeffs_list, q_line),
        tex_line_start=start_line + 1,
        tex_line_end=end_line + 1,
    )


def validate_against_json(
    extracted: ExtractedBenchmark,
    json_path: Path,
) -> Dict[str, Tuple[bool, str]]:
    """
    Validate extracted coefficients against existing JSON.

    Returns:
        Dict with per-polynomial (match_status, message)
    """
    with open(json_path) as f:
        data = json.load(f)

    results = {}

    # Validate P1
    json_p1 = data["polynomials"]["P1"]["tilde_coeffs"]
    extracted_p1 = extracted.P1.coefficients
    if len(json_p1) == len(extracted_p1):
        match = all(abs(a - b) < 1e-10 for a, b in zip(json_p1, extracted_p1))
        results["P1"] = (match, f"JSON: {json_p1}, Extracted: {extracted_p1}")
    else:
        results["P1"] = (False, f"Length mismatch: JSON has {len(json_p1)}, extracted has {len(extracted_p1)}")

    # Validate P2
    json_p2 = data["polynomials"]["P2"]["tilde_coeffs"]
    extracted_p2 = extracted.P2.coefficients
    if len(json_p2) == len(extracted_p2):
        match = all(abs(a - b) < 1e-10 for a, b in zip(json_p2, extracted_p2))
        results["P2"] = (match, f"JSON: {json_p2}, Extracted: {extracted_p2}")
    else:
        results["P2"] = (False, f"Length mismatch: JSON has {len(json_p2)}, extracted has {len(extracted_p2)}")

    # Validate P3
    json_p3 = data["polynomials"]["P3"]["tilde_coeffs"]
    extracted_p3 = extracted.P3.coefficients
    if len(json_p3) == len(extracted_p3):
        match = all(abs(a - b) < 1e-10 for a, b in zip(json_p3, extracted_p3))
        results["P3"] = (match, f"JSON: {json_p3}, Extracted: {extracted_p3}")
    else:
        results["P3"] = (False, f"Length mismatch: JSON has {len(json_p3)}, extracted has {len(extracted_p3)}")

    # Validate Q
    json_q_terms = data["polynomials"]["Q"]["coeffs_in_basis_terms"]
    json_q = {item["k"]: item["c"] for item in json_q_terms}
    extracted_q = dict(extracted.Q.coefficients)

    if set(json_q.keys()) == set(extracted_q.keys()):
        match = all(abs(json_q[k] - extracted_q[k]) < 1e-10 for k in json_q)
        results["Q"] = (match, f"JSON: {json_q}, Extracted: {extracted_q}")
    else:
        results["Q"] = (False, f"Key mismatch: JSON has {set(json_q.keys())}, extracted has {set(extracted_q.keys())}")

    return results


def write_extracted_json(
    kappa: ExtractedBenchmark,
    kappa_star: ExtractedBenchmark,
    output_path: Path,
) -> None:
    """
    Write extracted polynomials to JSON.
    """
    def benchmark_to_dict(b: ExtractedBenchmark) -> dict:
        return {
            "name": b.name,
            "R": b.R,
            "theta": b.theta,
            "kappa_target": b.kappa_target,
            "tex_lines": f"{b.tex_line_start}-{b.tex_line_end}",
            "polynomials": {
                "P1": {
                    "form": b.P1.form,
                    "tilde_coeffs": b.P1.coefficients,
                    "tex_line": b.P1.line_number,
                },
                "P2": {
                    "form": b.P2.form,
                    "tilde_coeffs": b.P2.coefficients,
                    "tex_line": b.P2.line_number,
                },
                "P3": {
                    "form": b.P3.form,
                    "tilde_coeffs": b.P3.coefficients,
                    "tex_line": b.P3.line_number,
                },
                "Q": {
                    "form": b.Q.form,
                    "coeffs_in_basis_terms": [{"k": k, "c": c} for k, c in b.Q.coefficients],
                    "tex_line": b.Q.line_number,
                },
            },
        }

    data = {
        "extracted_from": "RMS_PRZZ.tex",
        "extraction_date": "2025-12-24",
        "kappa": benchmark_to_dict(kappa),
        "kappa_star": benchmark_to_dict(kappa_star),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract PRZZ polynomials from TeX")
    parser.add_argument("tex_path", nargs="?",
                       default=None,
                       help="Path to RMS_PRZZ.tex")
    parser.add_argument("--output", "-o",
                       default=None,
                       help="Output JSON path")
    parser.add_argument("--validate", "-v",
                       action="store_true",
                       help="Validate against existing JSON files")
    args = parser.parse_args()

    # Find TeX file
    if args.tex_path:
        tex_path = Path(args.tex_path)
    else:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent / "RMS_PRZZ.tex",
            Path("/Users/john.n.dvorak/Documents/Git/Zeta_Mollifier_Optimization/RMS_PRZZ.tex"),
        ]
        tex_path = None
        for candidate in candidates:
            if candidate.exists():
                tex_path = candidate
                break

        if tex_path is None:
            print("ERROR: Could not find RMS_PRZZ.tex. Please provide path as argument.")
            return 1

    print(f"Reading TeX from: {tex_path}")
    tex_content = tex_path.read_text()

    # Extract benchmarks
    print("\nExtracting kappa benchmark...")
    kappa = extract_kappa_benchmark(tex_content)
    print(f"  P1 tilde_coeffs: {kappa.P1.coefficients}")
    print(f"  P2 tilde_coeffs: {kappa.P2.coefficients}")
    print(f"  P3 tilde_coeffs: {kappa.P3.coefficients}")
    print(f"  Q coeffs: {dict(kappa.Q.coefficients)}")

    print("\nExtracting kappa* benchmark...")
    kappa_star = extract_kappa_star_benchmark(tex_content)
    print(f"  P1 tilde_coeffs: {kappa_star.P1.coefficients}")
    print(f"  P2 tilde_coeffs: {kappa_star.P2.coefficients}")
    print(f"  P3 tilde_coeffs: {kappa_star.P3.coefficients}")
    print(f"  Q coeffs: {dict(kappa_star.Q.coefficients)}")

    # Validate if requested
    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION AGAINST EXISTING JSON FILES")
        print("=" * 60)

        data_dir = Path(__file__).parent.parent / "data"

        print("\nKappa benchmark:")
        kappa_json = data_dir / "przz_parameters.json"
        if kappa_json.exists():
            results = validate_against_json(kappa, kappa_json)
            all_match = True
            for poly, (match, msg) in results.items():
                status = "MATCH" if match else "MISMATCH"
                print(f"  {poly}: {status}")
                if not match:
                    print(f"    {msg}")
                    all_match = False
            if all_match:
                print("  All polynomials MATCH!")
        else:
            print(f"  WARNING: {kappa_json} not found")

        print("\nKappa* benchmark:")
        kappa_star_json = data_dir / "przz_parameters_kappa_star.json"
        if kappa_star_json.exists():
            results = validate_against_json(kappa_star, kappa_star_json)
            all_match = True
            for poly, (match, msg) in results.items():
                status = "MATCH" if match else "MISMATCH"
                print(f"  {poly}: {status}")
                if not match:
                    print(f"    {msg}")
                    all_match = False
            if all_match:
                print("  All polynomials MATCH!")
        else:
            print(f"  WARNING: {kappa_star_json} not found")

    # Write output if requested
    if args.output:
        output_path = Path(args.output)
        print(f"\nWriting extracted data to: {output_path}")
        write_extracted_json(kappa, kappa_star, output_path)
        print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
