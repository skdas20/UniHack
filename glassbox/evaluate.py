"""Evaluation and compliance measurement.

The Solution Guide asks for exactly this, and says judges will look for it:

> *"Field-level accuracy against the 200 known-good rows, character-limit
> compliance, and percentage of values found in the LOV are all simple,
> credible metrics. Show your evaluation."*

One honest complication: **the 200-row ground-truth workbook is not published
on the portal.** The Resources tab ships the Solution Guide, the 1,000-row
input, and a header sheet containing two worked rows. So field-level accuracy
against 200 labelled rows cannot be computed by anyone, and we do not pretend
to compute it.

What is measurable is measured, and split into two honest categories:

* **Gold accuracy** -- exact-match against the two published rows, which is
  real supervised accuracy on a tiny sample. Our renderer reproduces all five
  channels of both rows exactly (see tests/test_gold_channels.py).
* **Compliance** -- properties that are checkable on *every* row without any
  labels: character limits, LOV membership, UOM spacing, schema conformance,
  provenance coverage. These run over all 1,000 rows.

:func:`evaluate` also loads a real gold file if one is supplied, so the moment
the organisers hand over the 200-row workbook the same harness scores against
it with no code changes -- see :func:`score_against_gold`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import units as U
from .attributes import CONTRACTS, all_lov_values
from .provenance import EnrichedRow, Source
from .render import INVOICE_MAX, MOBILE_MAX, MOBILE_MIN
from .schema import OutputSchema

#: Channels whose exact text we can check against the published rows.
GOLD_CHANNELS = (
    "SHORT_DESC", "INVOICE_DESC", "MOBILE_DESC", "RETAIL_DESC", "LONG_DESC1",
)

#: Fields worth scoring individually when a labelled gold file is available.
SCORED_FIELDS = (
    "MANUFACTURER_NAME", "BRAND_NAME", "MANUFACTURER_PART_NUMBER", "Classpath",
    "Dept", "Class", "Fine", "Product Name",
    "MOBILE_DESC", "INVOICE_DESC", "SHORT_DESC", "LONG_DESC1", "RETAIL_DESC",
    "Product Image", "Specification Sheet",
)

_GLUED_UOM = re.compile(r"\d(?:in|ft|mm|cm|lb|oz|gal|qt|hr|min|sec)\b", re.IGNORECASE)


def compliance(rows: Sequence[EnrichedRow], schema: OutputSchema) -> dict[str, object]:
    """Label-free checks that run over every row."""
    n = max(len(rows), 1)

    invoice_lengths = [len(r.value("INVOICE_DESC")) for r in rows]
    invoice_present = [x for x in invoice_lengths if x]
    invoice_ok = sum(1 for x in invoice_present if x <= INVOICE_MAX)

    mobile_lengths = [len(r.value("MOBILE_DESC")) for r in rows]
    mobile_present = [x for x in mobile_lengths if x]
    mobile_ok = sum(1 for x in mobile_present if MOBILE_MIN <= x <= MOBILE_MAX)

    caps_ok = sum(
        1 for r in rows
        if not r.value("INVOICE_DESC") or r.value("INVOICE_DESC").isupper()
    )

    # UOM spacing: no digit glued to a unit anywhere in generated prose.
    spacing_violations = 0
    prose_fields = ("SHORT_DESC", "LONG_DESC1", "RETAIL_DESC", "MOBILE_DESC")
    for r in rows:
        for header in prose_fields:
            if _GLUED_UOM.search(r.value(header)):
                spacing_violations += 1
                break

    # LOV membership over every emitted attribute value.
    lov = all_lov_values()
    checked = violations = 0
    for r in rows:
        for slot_index in schema.slots("attribute_value"):
            label_h, value_h, _uom_h = schema.attribute_headers(slot_index)
            label, value = r.value(label_h), r.value(value_h)
            if not label or not value:
                continue
            permitted = lov.get(label)
            if not permitted:
                continue  # a free-text or measurement slot has no vocabulary
            checked += 1
            if value not in permitted:
                violations += 1

    # Schema conformance: every record must match the header list exactly.
    schema_problems = 0
    for r in rows:
        if schema.validate(r.as_record(schema)):
            schema_problems += 1

    # Provenance coverage: share of populated cells carrying real evidence.
    populated = evidenced = 0
    for r in rows:
        for cell in r.cells.values():
            if not cell:
                continue
            populated += 1
            if cell.prov.evidence or cell.prov.source in {
                Source.INPUT_COPY, Source.DERIVED, Source.CROSSWALK,
                Source.CONSTRAINT_SOLVER, Source.LEXICON_EXACT,
            }:
                evidenced += 1

    return {
        "rows": len(rows),
        "invoice_within_40_chars_pct": round(100 * invoice_ok / max(len(invoice_present), 1), 2),
        "invoice_all_caps_pct": round(100 * caps_ok / n, 2),
        "invoice_mean_chars": round(sum(invoice_present) / max(len(invoice_present), 1), 1),
        "mobile_within_60_80_pct": round(100 * mobile_ok / max(len(mobile_present), 1), 2),
        "mobile_mean_chars": round(sum(mobile_present) / max(len(mobile_present), 1), 1),
        "uom_spacing_violations": spacing_violations,
        "lov_values_checked": checked,
        "lov_compliance_pct": round(100 * (checked - violations) / max(checked, 1), 2),
        "schema_conformance_pct": round(100 * (len(rows) - schema_problems) / n, 2),
        "cells_populated": populated,
        "provenance_coverage_pct": round(100 * evidenced / max(populated, 1), 2),
    }


def gold_channel_check() -> dict[str, object]:
    """Re-run the exact-match check against the two published rows.

    Imported lazily from the test module so the number in the report and the
    number in CI are produced by the same code, not two drifting copies.
    """
    try:
        import sys
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tests.test_gold_channels import (  # type: ignore
            BUILDERS,
            GOLD1,
            GOLD1_EXPECTED,
            GOLD1_LONG,
            GOLD2,
            GOLD2_EXPECTED,
            long_description,
        )
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": repr(exc)}

    checks: list[tuple[str, bool]] = []
    for name, inp, expected in (
        ("gold1", GOLD1, GOLD1_EXPECTED), ("gold2", GOLD2, GOLD2_EXPECTED)
    ):
        for channel, want in expected.items():
            got, _ = BUILDERS[channel](inp)
            checks.append((f"{name}:{channel}", got == want))
    long_text, _ = long_description(GOLD1)
    checks.append(("gold1:LONG_DESC1", long_text == GOLD1_LONG))

    passed = sum(1 for _n, ok in checks if ok)
    return {
        "available": True,
        "channels_checked": len(checks),
        "exact_matches": passed,
        "exact_match_pct": round(100 * passed / max(len(checks), 1), 2),
        "failures": [name for name, ok in checks if not ok],
    }


# --- scoring against a real labelled file -----------------------------------


@dataclass
class FieldScore:
    field: str
    compared: int = 0
    exact: int = 0
    normalised: int = 0  # matches after case/whitespace normalisation

    @property
    def exact_pct(self) -> float:
        return round(100 * self.exact / max(self.compared, 1), 2)

    @property
    def normalised_pct(self) -> float:
        return round(100 * self.normalised / max(self.compared, 1), 2)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def score_against_gold(
    rows: Sequence[EnrichedRow],
    gold_path: str | Path,
    *,
    key: str = "Mfg_Part_Num",
    fields: Sequence[str] = SCORED_FIELDS,
) -> dict[str, object]:
    """Field-level accuracy against a labelled delivery-format file.

    Written now, unused until the 200-row workbook exists. Join is on
    ``Mfg_Part_Num``, so it works whether the organisers hand over 200 rows or
    20,000, and it needs no change to the pipeline.
    """
    gold_path = Path(gold_path)
    if not gold_path.exists():
        return {"available": False, "reason": f"{gold_path} not found"}

    with gold_path.open(newline="", encoding="utf-8-sig") as fh:
        gold = {
            (rec.get(key) or "").strip(): rec
            for rec in csv.DictReader(fh)
            if (rec.get(key) or "").strip()
        }
    if not gold:
        return {"available": False, "reason": f"no rows keyed by {key!r} in {gold_path}"}

    scores = {f: FieldScore(f) for f in fields}
    matched = 0
    for row in rows:
        want = gold.get(row.value(key))
        if want is None:
            continue
        matched += 1
        for f in fields:
            if f not in want:
                continue
            expected, got = (want.get(f) or "").strip(), row.value(f)
            if not expected:
                continue
            s = scores[f]
            s.compared += 1
            if got == expected:
                s.exact += 1
                s.normalised += 1
            elif _norm(got) == _norm(expected):
                s.normalised += 1

    comparable = [s for s in scores.values() if s.compared]
    overall_exact = (
        round(sum(s.exact for s in comparable) / max(sum(s.compared for s in comparable), 1) * 100, 2)
        if comparable else 0.0
    )
    return {
        "available": True,
        "gold_file": str(gold_path),
        "gold_rows": len(gold),
        "rows_matched": matched,
        "overall_exact_pct": overall_exact,
        "fields": {
            s.field: {"compared": s.compared, "exact_pct": s.exact_pct,
                      "normalised_pct": s.normalised_pct}
            for s in comparable
        },
    }


def evaluate(
    rows: Sequence[EnrichedRow],
    schema: OutputSchema,
    *,
    gold_path: str | Path | None = None,
) -> dict[str, object]:
    """The full evaluation payload written into the run report."""
    result: dict[str, object] = {
        "compliance": compliance(rows, schema),
        "gold_channel_check": gold_channel_check(),
        "notes": [
            "The 200-row Input-vs-Delivery-Format workbook referenced by the "
            "Solution Guide is not published on the portal, so field-level "
            "accuracy against 200 labelled rows is not computable by anyone. "
            "It is not estimated or simulated here.",
            "Gold accuracy below is exact-match against the two fully-worked "
            "rows that the Expected Output sheet does publish.",
            "Compliance metrics are label-free and therefore run over every "
            "row of the input, not a sample.",
            "UNSPSC and Country Of Origin are emitted blank on purpose: both "
            "gold rows leave them blank, and inventing a classification code "
            "would be a fabricated value.",
            "score_against_gold() is implemented and ready; supply a labelled "
            "delivery-format CSV and it reports per-field exact accuracy with "
            "no pipeline changes.",
        ],
    }
    if gold_path:
        result["gold_file_scoring"] = score_against_gold(rows, gold_path)
    return result
