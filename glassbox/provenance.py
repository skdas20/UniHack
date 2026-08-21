"""Provenance: every cell knows where it came from.

The challenge statement asks for *"traceable outputs"* and *"explainability"*.
This module is how we mean it literally: the pipeline does not produce a
252-column row of strings, it produces a 252-column row of :class:`Cell`
objects, each carrying the decision that created it -- the mechanism, the
confidence, the rule that fired, and where relevant the exact character span
of the raw description that the value was read out of.

That makes three things possible that a plain string pipeline cannot do:

* the UI can highlight, inside the raw input, the substring that produced any
  value a judge hovers over;
* the confidence engine can score a row by *how* its cells were obtained
  rather than by a vibe;
* a wrong value can be traced to the exact rule that produced it, which is the
  difference between a demo and something a content team could actually adopt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


class Source(str, Enum):
    """How a cell's value was obtained. Ordered roughly by trustworthiness."""

    #: Copied verbatim from the input row.
    INPUT_COPY = "input_copy"
    #: Read out of the raw text by a deterministic pattern.
    REGEX_EXTRACT = "regex_extract"
    #: Exact hit against an induced or curated lexicon.
    LEXICON_EXACT = "lexicon_exact"
    #: Approximate hit against a lexicon (carries the similarity score).
    LEXICON_FUZZY = "lexicon_fuzzy"
    #: Emitted because the category's attribute contract requires the slot.
    CONTRACT_TEMPLATE = "contract_template"
    #: Computed from other resolved cells (units, fractions, crosswalks).
    DERIVED = "derived"
    #: Produced by the two-sided length packer.
    CONSTRAINT_SOLVER = "constraint_solver"
    #: Mapped through a taxonomy crosswalk.
    CROSSWALK = "crosswalk"
    #: A locally-hosted fine-tuned model predicted it.
    LOCAL_MODEL = "local_model"
    #: A hosted LLM proposed it (always validated before it is accepted).
    LLM = "llm"
    #: Deliberately left blank: not a function of any available input.
    NOT_DERIVABLE = "not_derivable"
    #: Deliberately left blank: the source field held a placeholder.
    PLACEHOLDER_DROPPED = "placeholder_dropped"
    #: Nothing was found and nothing was invented.
    UNRESOLVED = "unresolved"


#: Sources that represent an honest, intentional blank rather than a failure.
INTENTIONAL_BLANKS = frozenset(
    {Source.NOT_DERIVABLE, Source.PLACEHOLDER_DROPPED, Source.CONTRACT_TEMPLATE}
)

#: Sources whose output was never checked against a controlled vocabulary and
#: therefore must not be trusted without validation.
GENERATIVE = frozenset({Source.LLM, Source.LOCAL_MODEL})


@dataclass(frozen=True, slots=True)
class Evidence:
    """A character span of some source text that justifies a value."""

    text: str
    start: int
    end: int
    field: str = "Part_Desc"

    @property
    def snippet(self) -> str:
        return self.text[self.start : self.end]

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "start": self.start,
            "end": self.end,
            "snippet": self.snippet,
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    """Why a cell holds the value it holds."""

    source: Source
    rule: str = ""
    confidence: float = 1.0
    evidence: tuple[Evidence, ...] = ()
    #: Free-form detail shown in the audit view (fuzzy score, dropped
    #: candidates, the system of record for a non-derivable field, ...).
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "rule": self.rule,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "evidence": [e.as_dict() for e in self.evidence],
        }


@dataclass(slots=True)
class Cell:
    """A single output cell: its value and the decision behind it."""

    value: str = ""
    prov: Provenance = field(
        default_factory=lambda: Provenance(Source.UNRESOLVED, rule="not attempted")
    )

    def __str__(self) -> str:  # so a Cell can be written straight to a sheet
        return self.value

    def __bool__(self) -> bool:
        return bool(self.value.strip())

    @property
    def is_intentional_blank(self) -> bool:
        return not self and self.prov.source in INTENTIONAL_BLANKS

    def with_value(self, value: str) -> "Cell":
        return Cell(value=value, prov=self.prov)


# --- constructors, so call sites stay readable -------------------------------


def copied(value: str, field_name: str) -> Cell:
    return Cell(
        value,
        Provenance(
            Source.INPUT_COPY,
            rule=f"passthrough:{field_name}",
            confidence=1.0,
        ),
    )


def extracted(
    value: str,
    rule: str,
    *,
    text: str,
    start: int,
    end: int,
    field_name: str = "Part_Desc",
    confidence: float = 0.9,
) -> Cell:
    return Cell(
        value,
        Provenance(
            Source.REGEX_EXTRACT,
            rule=rule,
            confidence=confidence,
            evidence=(Evidence(text, start, end, field_name),),
        ),
    )


def matched(
    value: str,
    rule: str,
    *,
    score: float,
    exact: bool,
    evidence: Iterable[Evidence] = (),
    detail: str = "",
) -> Cell:
    return Cell(
        value,
        Provenance(
            Source.LEXICON_EXACT if exact else Source.LEXICON_FUZZY,
            rule=rule,
            confidence=score,
            evidence=tuple(evidence),
            detail=detail,
        ),
    )


def derived(value: str, rule: str, *, confidence: float = 0.95, detail: str = "") -> Cell:
    return Cell(
        value,
        Provenance(Source.DERIVED, rule=rule, confidence=confidence, detail=detail),
    )


def solved(value: str, rule: str, *, confidence: float, detail: str = "") -> Cell:
    return Cell(
        value,
        Provenance(
            Source.CONSTRAINT_SOLVER, rule=rule, confidence=confidence, detail=detail
        ),
    )


def generated(
    value: str,
    rule: str,
    *,
    source: Source = Source.LLM,
    confidence: float = 0.6,
    detail: str = "",
) -> Cell:
    return Cell(value, Provenance(source, rule=rule, confidence=confidence, detail=detail))


def not_derivable(header: str, system_of_record: str) -> Cell:
    """An honest blank. Named after the reason, not the absence."""
    return Cell(
        "",
        Provenance(
            Source.NOT_DERIVABLE,
            rule=f"not-a-function-of-input:{header}",
            confidence=1.0,
            detail=f"requires {system_of_record}; refusing to fabricate",
        ),
    )


def placeholder_dropped(raw: str, field_name: str) -> Cell:
    return Cell(
        "",
        Provenance(
            Source.PLACEHOLDER_DROPPED,
            rule=f"placeholder:{field_name}",
            confidence=1.0,
            detail=f"{raw!r} is a placeholder meaning empty, not a value",
        ),
    )


def contract_blank(label: str) -> Cell:
    """A slot the category contract requires but for which no value was found."""
    return Cell(
        "",
        Provenance(
            Source.CONTRACT_TEMPLATE,
            rule=f"contract-slot:{label}",
            confidence=1.0,
            detail="slot required by category contract; no value found in source",
        ),
    )


def unresolved(rule: str, detail: str = "") -> Cell:
    return Cell("", Provenance(Source.UNRESOLVED, rule=rule, confidence=0.0, detail=detail))


@dataclass
class EnrichedRow:
    """A full delivery-format row, as cells, plus the audit verdict."""

    index: int
    cells: dict[str, Cell] = field(default_factory=dict)
    #: Populated by confidence.py
    confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    #: Per-stage timings, useful for the scalability slide.
    timings_ms: dict[str, float] = field(default_factory=dict)

    def set(self, header: str, cell: Cell) -> None:
        self.cells[header] = cell

    def value(self, header: str, default: str = "") -> str:
        cell = self.cells.get(header)
        return cell.value if cell is not None else default

    def prov(self, header: str) -> Provenance | None:
        cell = self.cells.get(header)
        return cell.prov if cell is not None else None

    def as_record(self, schema) -> dict[str, str]:
        """Flatten to plain strings in schema order, for the delivery sheet."""
        rec = schema.blank_record()
        for header, cell in self.cells.items():
            if header in rec:
                rec[header] = cell.value
        return rec

    def audit_record(self) -> dict[str, Any]:
        """The provenance sidecar, emitted alongside the delivery sheet."""
        return {
            "row_index": self.index,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "cells": {
                header: {"value": cell.value, **cell.prov.as_dict()}
                for header, cell in self.cells.items()
                if cell.value or cell.prov.source is not Source.UNRESOLVED
            },
        }

    # --- aggregate views used by the confidence engine and the UI ---

    def filled(self) -> dict[str, Cell]:
        return {h: c for h, c in self.cells.items() if c}

    def by_source(self) -> dict[Source, int]:
        counts: dict[Source, int] = {}
        for cell in self.cells.values():
            counts[cell.prov.source] = counts.get(cell.prov.source, 0) + 1
        return counts

    def generative_cells(self) -> dict[str, Cell]:
        """Cells whose value came from a generative model, for the review view."""
        return {
            h: c for h, c in self.cells.items() if c and c.prov.source in GENERATIVE
        }


def rescore(cell: Cell, confidence: float, *, detail: str = "") -> Cell:
    """Return a copy of ``cell`` with an adjusted confidence."""
    prov = replace(
        cell.prov,
        confidence=confidence,
        detail=detail or cell.prov.detail,
    )
    return Cell(cell.value, prov)
