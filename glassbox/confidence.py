"""Confidence scoring and the human-review queue.

The score is computed **from the provenance**, not from a model's self-reported
certainty. A row's confidence is a function of *how* its cells were obtained:
a value read out of the input with a character span behind it is worth more
than one a language model recalled about the part number, and the score says so.

That has a useful property. It cannot be gamed by a fluent generator. A row
whose 252 cells are all beautifully written but all model-proposed scores low,
which is exactly the outcome the Solution Guide asks for:

> *"A fluent description made of invented values scores zero."*

Review reasons are specific and actionable -- ``"INVOICE_DESC is 43 chars, over
the 40 limit"``, not ``"low confidence"`` -- because the point of the queue is
that a person can clear an item in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import EnrichedRow, Source

#: How much a cell contributes to row confidence, by how it was obtained.
#: Rules, lexicons and solvers are trusted; generative sources are not.
SOURCE_WEIGHT: dict[Source, float] = {
    Source.INPUT_COPY: 1.00,
    Source.LEXICON_EXACT: 0.98,
    Source.REGEX_EXTRACT: 0.95,
    Source.CONSTRAINT_SOLVER: 0.95,
    Source.CROSSWALK: 0.92,
    Source.DERIVED: 0.90,
    Source.LEXICON_FUZZY: 0.85,
    Source.LOCAL_MODEL: 0.60,
    Source.LLM: 0.50,
    # Intentional blanks are neither credit nor penalty; they are excluded
    # from the mean rather than scored, so a category with a long contract is
    # not punished for honestly leaving slots empty.
    Source.CONTRACT_TEMPLATE: 0.0,
    Source.NOT_DERIVABLE: 0.0,
    Source.PLACEHOLDER_DROPPED: 0.0,
    Source.UNRESOLVED: 0.0,
}

#: Cells that must be right for the row to be usable at all. A problem in any
#: of these caps the row's confidence, however good everything else is.
CRITICAL = (
    "BRAND_NAME",
    "MANUFACTURER_NAME",
    "Classpath",
    "SHORT_DESC",
    "INVOICE_DESC",
    "MOBILE_DESC",
)

#: Below this, a row goes to the review queue regardless of flags.
REVIEW_THRESHOLD = 0.70


@dataclass
class Verdict:
    confidence: float = 0.0
    needs_review: bool = False
    reasons: list[str] = field(default_factory=list)
    #: Sub-scores, surfaced in the UI so the number is never a black box.
    components: dict[str, float] = field(default_factory=dict)


def score(row: EnrichedRow, *, checks: dict[str, object] | None = None) -> Verdict:
    """Score one enriched row and decide whether a human should look at it."""
    verdict = Verdict()
    checks = checks or {}

    populated = [c for c in row.cells.values() if c]
    if not populated:
        verdict.reasons.append("no cell was populated at all")
        verdict.needs_review = True
        return verdict

    # --- 1. provenance quality: mean weighted confidence of populated cells ---
    total = 0.0
    for cell in populated:
        weight = SOURCE_WEIGHT.get(cell.prov.source, 0.5)
        total += weight * max(0.0, min(1.0, cell.prov.confidence))
    provenance_quality = total / len(populated)
    verdict.components["provenance_quality"] = round(provenance_quality, 4)

    # --- 2. critical-field completeness ---
    missing_critical = [h for h in CRITICAL if not row.cells.get(h)]
    critical_score = 1.0 - (len(missing_critical) / len(CRITICAL))
    verdict.components["critical_fields"] = round(critical_score, 4)
    for header in missing_critical:
        verdict.reasons.append(f"{header} is empty")

    # --- 3. generative exposure: what share of populated cells was guessed ---
    generative = row.generative_cells()
    generative_share = len(generative) / len(populated)
    verdict.components["extracted_share"] = round(1.0 - generative_share, 4)
    if generative:
        verdict.reasons.append(
            f"{len(generative)} value(s) proposed by a model rather than read "
            f"from the input: {', '.join(sorted(generative)[:4])}"
            + ("..." if len(generative) > 4 else "")
        )

    # --- 4. hard compliance checks, supplied by the pipeline ---
    compliance_penalty = 0.0
    invoice_len = checks.get("invoice_length")
    if isinstance(invoice_len, int) and invoice_len > 40:
        verdict.reasons.append(f"INVOICE_DESC is {invoice_len} chars, over the 40 limit")
        compliance_penalty += 0.25
    mobile_len = checks.get("mobile_length")
    if isinstance(mobile_len, int) and mobile_len and not 60 <= mobile_len <= 80:
        verdict.reasons.append(
            f"MOBILE_DESC is {mobile_len} chars, outside the 60-80 window"
        )
        compliance_penalty += 0.10
    lov_violations = checks.get("lov_violations") or []
    if isinstance(lov_violations, (list, tuple)) and lov_violations:
        verdict.reasons.append(
            f"{len(lov_violations)} attribute value(s) outside the controlled "
            f"vocabulary: {', '.join(str(v) for v in lov_violations[:3])}"
        )
        compliance_penalty += 0.25
    if checks.get("unclassified"):
        verdict.reasons.append(
            "no taxonomy match, so the attribute contract and all five "
            "descriptions fall back to the generic template"
        )
        compliance_penalty += 0.20
    if checks.get("brand_unresolved"):
        verdict.reasons.append(
            "brand unresolved: every brand column held a placeholder and no "
            "alias was found in the description"
        )
        compliance_penalty += 0.20
    margin = checks.get("classification_margin")
    if isinstance(margin, float) and 0.0 < margin < 0.10:
        runner_up = checks.get("runner_up") or "the runner-up"
        verdict.reasons.append(
            f"classification was near-tied with {runner_up} (margin {margin:.2f})"
        )
        compliance_penalty += 0.08
    verdict.components["compliance_penalty"] = round(compliance_penalty, 4)

    # --- 5. combine ---
    base = (
        0.45 * provenance_quality
        + 0.30 * critical_score
        + 0.25 * (1.0 - generative_share)
    )
    verdict.confidence = max(0.0, min(1.0, base - compliance_penalty))
    verdict.needs_review = bool(verdict.reasons) or verdict.confidence < REVIEW_THRESHOLD

    row.confidence = verdict.confidence
    row.needs_review = verdict.needs_review
    row.review_reasons = list(verdict.reasons)
    return verdict


def triage(rows: list[EnrichedRow]) -> dict[str, list[EnrichedRow]]:
    """Split a batch into the three buckets a content team actually works in."""
    auto, review, blocked = [], [], []
    for row in rows:
        if row.confidence >= 0.85 and not row.needs_review:
            auto.append(row)
        elif row.confidence >= 0.45:
            review.append(row)
        else:
            blocked.append(row)
    return {"auto_publish": auto, "needs_review": review, "blocked": blocked}


def reason_histogram(rows: list[EnrichedRow]) -> dict[str, int]:
    """Which review reasons dominate, so the pipeline can be improved next."""
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.review_reasons:
            # bucket by the reason's stable prefix, not its interpolated detail
            key = reason.split(":")[0].split(" is ")[0].split(" value(s)")[0]
            key = key.strip()[:70]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
