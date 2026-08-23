# GlassBox enrichment report

_Generated 2026-08-23T16:36:35.375886+00:00_

## Run

| Metric | Value |
|---|---|
| rows | 1000 |
| classified | 793 |
| classified_pct | 79.3 |
| brand_resolved | 757 |
| brand_resolved_pct | 75.7 |
| attribute_slot_fill_pct | 23.14 |
| invoice_over_40_chars | 0 |
| mobile_outside_60_80 | 147 |
| needs_review | 382 |
| needs_review_pct | 38.2 |
| model_proposed_values | 0 |
| elapsed_s | 4.671 |
| rows_per_s | 214.1 |

## Compliance

| Metric | Value |
|---|---|
| rows | 1000 |
| invoice_within_40_chars_pct | 100.0 |
| invoice_all_caps_pct | 100.0 |
| invoice_mean_chars | 22.7 |
| mobile_within_60_80_pct | 85.3 |
| mobile_mean_chars | 65.3 |
| uom_spacing_violations | 0 |
| lov_values_checked | 1068 |
| lov_compliance_pct | 100.0 |
| schema_conformance_pct | 100.0 |
| cells_populated | 42049 |
| provenance_coverage_pct | 99.99 |

## Cells by source

| Source | Cells |
|---|---|
| derived | 26864 |
| input_copy | 7442 |
| regex_extract | 3123 |
| constraint_solver | 2000 |
| lexicon_exact | 1629 |
| crosswalk | 716 |
| lexicon_fuzzy | 275 |

## Blanks by reason

| Reason | Cells |
|---|---|
| unresolved | 12335 |
| not_derivable | 11307 |
| contract_template | 9530 |

## Triage

| Bucket | Rows |
|---|---|
| auto_publish | 618 |
| needs_review | 297 |
| blocked | 85 |

## Top review reasons

| Reason | Rows |
|---|---|
| BRAND_NAME | 243 |
| MANUFACTURER_NAME | 243 |
| brand unresolved | 243 |
| Classpath | 207 |
| no taxonomy match, so the attribute contract and all five descriptions | 207 |
| MOBILE_DESC | 147 |
| classification was near-tied with Layout & Measuring (margin 0.04) | 5 |
| classification was near-tied with Asphalt Shingles (margin 0.02) | 3 |
| classification was near-tied with Layout & Measuring (margin 0.03) | 2 |
| classification was near-tied with Balusters (margin 0.02) | 2 |
| classification was near-tied with Railing Systems (margin 0.08) | 2 |
| classification was near-tied with Batteries & Chargers (margin 0.09) | 2 |
| classification was near-tied with Railing Systems (margin 0.02) | 1 |
| classification was near-tied with LED Lamps (margin 0.04) | 1 |
| classification was near-tied with Ranges (margin 0.01) | 1 |
| classification was near-tied with Circular Saws (margin 0.06) | 1 |
| classification was near-tied with Saw Blades (margin 0.00) | 1 |
| classification was near-tied with Saw Blades (margin 0.01) | 1 |

## Vocabulary induced

| Item | Count |
|---|---|
| groups | 76 |
| brands | 51 |
| brand_candidates_for_review | 181 |
| aliases | 59 |
| product_types | 137 |
| series | 42 |
| unknown_unit_spellings | 39 |
| attested_brands | 51 |

## Notes

- The 200-row Input-vs-Delivery-Format workbook referenced by the Solution Guide is not published on the portal, so field-level accuracy against 200 labelled rows is not computable by anyone. It is not estimated or simulated here.
- Gold accuracy below is exact-match against the two fully-worked rows that the Expected Output sheet does publish.
- Compliance metrics are label-free and therefore run over every row of the input, not a sample.
- UNSPSC and Country Of Origin are emitted blank on purpose: both gold rows leave them blank, and inventing a classification code would be a fabricated value.
- score_against_gold() is implemented and ready; supply a labelled delivery-format CSV and it reports per-field exact accuracy with no pipeline changes.
