# Run 12A: Channel-by-Channel Diff (OLD vs V2)

**Generated**: 2025-12-21 09:27:11

## Purpose

Diagnose WHY V2 terms fail under tex_mirror assembly:
- V2 is proven correct for individual I-terms (ratio=1.0)
- BUT V2 breaks tex_mirror (c≈0.775 vs target 2.137)

---

## Channel Comparison Table

| Variable | κ OLD | κ V2 | κ V2/OLD | κ* OLD | κ* V2 | κ* V2/OLD |
|----------|-------|------|----------|--------|-------|-----------|
| I1_plus | 0.084872 | -0.111107 | -1.3091 | 0.120230 | 0.042175 | 0.3508 |
| I2_plus | 0.712608 | 0.712608 | 1.0000 | 0.494352 | 0.494352 | 1.0000 |
| S34_plus | -0.337937 | -1.211082 | 3.5838 | -0.288519 | -0.826543 | 2.8648 |
| I1_minus_base | 0.051266 | 0.008984 | 0.1752 | 0.070630 | 0.055505 | 0.7859 |
| I2_minus_base | 0.168855 | 0.168855 | 1.0000 | 0.145814 | 0.145814 | 1.0000 |
| I1_minus_op | 0.053561 | 0.006944 | 0.1297 | 0.072761 | 0.057048 | 0.7840 |
| I2_minus_op | 0.168855 | 0.168855 | 1.0000 | 0.145814 | 0.145814 | 1.0000 |
| m1_implied | 1.044761 | 0.773001 | 0.7399 | 1.030177 | 1.027810 | 0.9977 |
| m2_implied | 1.000000 | 1.000000 | 1.0000 | 1.000000 | 1.000000 | 1.0000 |
| A1 | 5.955967 | 5.955967 | 1.0000 | 5.955967 | 5.955967 | 1.0000 |
| A2 | 7.955967 | 7.955967 | 1.0000 | 7.955967 | 7.955967 | 1.0000 |
| m1 | 6.222563 | 4.603966 | 0.7399 | 6.135698 | 6.121601 | 0.9977 |
| m2 | 7.955967 | 7.955967 | 1.0000 | 7.955967 | 7.955967 | 1.0000 |
| c | 2.121955 | 0.775182 | 0.3653 | 1.919518 | 1.209855 | 0.6303 |

---

## Targets

| Benchmark | c target | κ target |
|-----------|----------|----------|
| κ (R=1.3036) | 2.137454 | 0.417294 |
| κ* (R=1.1167) | 1.938010 | 0.405000 |

---

## Analysis


### κ Analysis

Variables with >20% difference (V2 vs OLD):

- **I1 (+R) channel**: -230.9%
- **S34 (+R) channel**: -258.4%
- **I1 (-R) base**: -82.5%
- **I1 (-R) operator**: -87.0%
- **m1 implied (shape)**: -26.0%
- **m1 full weight**: -26.0%
- **final c**: -63.5%

**DIAGNOSIS**: Plus channels differ → V2 term builder issue

### κ* Analysis

Variables with >20% difference (V2 vs OLD):

- **I1 (+R) channel**: -64.9%
- **S34 (+R) channel**: -186.5%
- **I1 (-R) base**: -21.4%
- **I1 (-R) operator**: -21.6%
- **final c**: -37.0%

**DIAGNOSIS**: Plus channels differ → V2 term builder issue

---

## Interpretation Guide

| If this differs | Then... |
|-----------------|---------|
| Plus channels (I1_plus, I2_plus) | V2 term builder feeds different object |
| Minus base (I1_minus_base, I2_minus_base) | V2's (1-u) power materially affects -R |
| Minus op but not base | Operator-shape incompatible with V2 structure |
| m_implied explodes | Near-zero denominator in V2 |
| Only c differs | Assembly bug in V2 path |

---

## Raw Data

### κ Benchmark (R=1.3036)

#### OLD terms
```
A1: 5.955967441223228
A2: 7.955967441223228
I1_minus_base: 0.05126645779097773
I1_minus_op: 0.05356119688345537
I1_plus: 0.0848719585571232
I2_minus_base: 0.1688547387238544
I2_minus_op: 0.1688547387238544
I2_plus: 0.7126082993883718
S34_plus: -0.33793654576343785
c: 2.1219552605161045
m1: 6.222562636401785
m1_implied: 1.0447610229252369
m2: 7.955967441223228
m2_implied: 1.0
```

#### V2 terms
```
A1: 5.955967441223228
A2: 7.955967441223228
I1_minus_base: 0.008983507360447337
I1_minus_op: 0.0069442559338784085
I1_plus: -0.1111069075358013
I2_minus_base: 0.1688547387238544
I2_minus_op: 0.1688547387238544
I2_plus: 0.7126082993883718
S34_plus: -1.211082445198269
c: 0.7751815124832431
m1: 4.603965977453318
m1_implied: 0.7730005281069439
m2: 7.955967441223228
m2_implied: 1.0
```

### κ* Benchmark (R=1.1167)

#### OLD terms
```
A1: 5.955967441223228
A2: 7.955967441223228
I1_minus_base: 0.07062961801808608
I1_minus_op: 0.0727609748682767
I1_plus: 0.12022998570182211
I2_minus_base: 0.14581420610550752
I2_minus_op: 0.14581420610550752
I2_plus: 0.4943519802155149
S34_plus: -0.2885190783316973
c: 1.9195179611359887
m1: 6.135697876720031
m1_implied: 1.0301765308945159
m2: 7.955967441223228
m2_implied: 1.0
```

#### V2 terms
```
A1: 5.955967441223228
A2: 7.955967441223228
I1_minus_base: 0.055504660862878086
I1_minus_op: 0.057048225602684355
I1_plus: 0.042174671536909195
I2_minus_base: 0.14581420610550752
I2_minus_op: 0.14581420610550752
I2_plus: 0.4943519802155149
S34_plus: -0.8265425135938713
c: 1.2098545886709293
m1: 6.121600762655789
m1_implied: 1.027809641853674
m2: 7.955967441223228
m2_implied: 1.0
```