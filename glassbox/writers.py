"""Emitting the delivery sheet, the provenance sidecar, and the audit report.

The submission requirement is explicit:

> *"Your solution should generate a downloadable XLSX or CSV file from the given
> input data, with all the required static headers populated. Do not modify,
> remove, or rename any of the headers."*

So the delivery writer takes the header list from the schema object -- which was
itself read from the organisers' sheet -- and writes exactly those headers, in
exactly that order, and validates every record against them before writing. It
is not possible for this writer to emit a renamed or reordered column.

Three artefacts come out of a run:

* ``enriched.xlsx`` / ``.csv`` -- the delivery sheet, 252 columns.
* ``provenance.jsonl`` -- one JSON object per row: every populated cell with
  its source, rule, confidence and evidence span. This is the file that makes
  the output auditable rather than merely plausible.
* ``report.json`` / ``report.md`` -- the evaluation and compliance summary.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .provenance import EnrichedRow, Source
from .schema import OutputSchema


def write_delivery_csv(
    path: str | Path,
    schema: OutputSchema,
    rows: Sequence[EnrichedRow],
    *,
    strict: bool = True,
) -> Path:
    """Write the 252-column delivery sheet as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(schema.headers), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            record = row.as_record(schema)
            problems = schema.validate(record)
            if problems and strict:
                raise ValueError(f"row {row.index}: {problems}")
            writer.writerow(record)
    return path


def write_delivery_xlsx(
    path: str | Path,
    schema: OutputSchema,
    rows: Sequence[EnrichedRow],
    *,
    highlight_review: bool = True,
) -> Path:
    """Write the delivery sheet as XLSX, with review rows tinted.

    The tint is cosmetic but it is the first thing a content lead asks for:
    show me which rows a person still has to look at.
    """
    import xlsxwriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    sheet = book.add_worksheet("Delivery Format")

    header_fmt = book.add_format(
        {"bold": True, "bg_color": "#1F2937", "font_color": "#FFFFFF",
         "text_wrap": False, "valign": "vcenter", "border": 1, "border_color": "#374151"}
    )
    review_fmt = book.add_format({"bg_color": "#FEF3C7"})
    blocked_fmt = book.add_format({"bg_color": "#FEE2E2"})

    for col, header in enumerate(schema.headers):
        sheet.write(0, col, header, header_fmt)
    sheet.freeze_panes(1, 1)
    sheet.set_row(0, 28)

    for r, row in enumerate(rows, start=1):
        record = row.as_record(schema)
        fmt = None
        if highlight_review:
            if row.confidence < 0.45:
                fmt = blocked_fmt
            elif row.needs_review:
                fmt = review_fmt
        for c, header in enumerate(schema.headers):
            value = record.get(header, "")
            if fmt is not None:
                sheet.write(r, c, value, fmt)
            else:
                sheet.write(r, c, value)

    book.close()
    return path


def write_provenance(path: str | Path, rows: Sequence[EnrichedRow]) -> Path:
    """One JSON object per row: the full audit trail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.audit_record(), ensure_ascii=False) + "\n")
    return path


def write_review_queue(
    path: str | Path, schema: OutputSchema, rows: Sequence[EnrichedRow]
) -> Path:
    """A narrow CSV a human can actually work through, worst rows first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_index", "confidence", "Mfg_Part_Num", "Part_Desc",
        "BRAND_NAME", "MANUFACTURER_NAME", "Classpath",
        "SHORT_DESC", "INVOICE_DESC", "invoice_chars",
        "MOBILE_DESC", "mobile_chars", "review_reasons",
    ]
    flagged = sorted(
        (r for r in rows if r.needs_review), key=lambda r: r.confidence
    )
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in flagged:
            writer.writerow(
                {
                    "row_index": row.index,
                    "confidence": round(row.confidence, 3),
                    "Mfg_Part_Num": row.value("Mfg_Part_Num"),
                    "Part_Desc": row.value("Part_Desc"),
                    "BRAND_NAME": row.value("BRAND_NAME"),
                    "MANUFACTURER_NAME": row.value("MANUFACTURER_NAME"),
                    "Classpath": row.value("Classpath"),
                    "SHORT_DESC": row.value("SHORT_DESC"),
                    "INVOICE_DESC": row.value("INVOICE_DESC"),
                    "invoice_chars": len(row.value("INVOICE_DESC")),
                    "MOBILE_DESC": row.value("MOBILE_DESC"),
                    "mobile_chars": len(row.value("MOBILE_DESC")),
                    "review_reasons": " | ".join(row.review_reasons),
                }
            )
    return path


def source_histogram(rows: Sequence[EnrichedRow]) -> dict[str, int]:
    """How many populated cells came from each mechanism, across the batch."""
    counts: dict[str, int] = {}
    for row in rows:
        for cell in row.cells.values():
            if not cell:
                continue
            key = cell.prov.source.value
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def blank_reason_histogram(rows: Sequence[EnrichedRow]) -> dict[str, int]:
    """Why cells are blank -- an intentional blank is not a failure."""
    counts: dict[str, int] = {}
    for row in rows:
        for cell in row.cells.values():
            if cell:
                continue
            key = cell.prov.source.value
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_report(
    path: str | Path,
    payload: dict,
    *,
    also_markdown: bool = True,
) -> Path:
    """Write the run report as JSON, and optionally as readable Markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if also_markdown:
        md_path = path.with_suffix(".md")
        md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return path


def _render_markdown(payload: dict) -> str:
    lines = ["# GlassBox enrichment report", ""]
    lines.append(f"_Generated {payload.get('generated_utc', '')}_")
    lines.append("")

    def table(title: str, data: dict, key_header: str = "Metric", val_header: str = "Value"):
        if not data:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"| {key_header} | {val_header} |")
        lines.append("|---|---|")
        for key, value in data.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    table("Run", payload.get("run", {}))
    table("Compliance", payload.get("compliance", {}))
    table("Cells by source", payload.get("cells_by_source", {}), "Source", "Cells")
    table("Blanks by reason", payload.get("blanks_by_reason", {}), "Reason", "Cells")
    table("Triage", payload.get("triage", {}), "Bucket", "Rows")
    table("Top review reasons", payload.get("review_reasons", {}), "Reason", "Rows")
    table("Vocabulary induced", payload.get("vocabulary", {}), "Item", "Count")

    notes = payload.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)
