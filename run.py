"""GlassBox CLI -- enrich a catalogue end to end.

    python run.py                                  # the shipped 1000-row sample
    python run.py --input mydata.csv --limit 50     # any CSV or XLSX
    python run.py --no-xlsx --out outputs/run2     # CSV only

Runs with no configuration, no network and no API key. Outputs land in
``outputs/``:

    enriched.xlsx / enriched.csv   the 252-column delivery sheet
    provenance.jsonl              every populated cell, with its evidence
    review_queue.csv              the rows a human should look at, worst first
    report.json / report.md       compliance and evaluation summary
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glassbox.confidence import reason_histogram, triage  # noqa: E402
from glassbox.evaluate import evaluate  # noqa: E402
from glassbox.pipeline import Pipeline  # noqa: E402
from glassbox.schema import OutputSchema, load_input  # noqa: E402
from glassbox.writers import (  # noqa: E402
    blank_reason_histogram,
    source_histogram,
    write_delivery_csv,
    write_delivery_xlsx,
    write_provenance,
    write_report,
    write_review_queue,
)

DEFAULT_INPUT = "data/raw/input_1000.csv"
DEFAULT_SCHEMA = "data/raw/expected_output_schema.csv"
DEFAULT_VOCAB = "data/vocab/induced.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="GlassBox catalogue enrichment")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB,
                    help="reuse a previously induced vocabulary; induced fresh if absent")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows")
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--propose", action="store_true",
                    help="enable the optional hosted-model proposal layer "
                         "(requires NVIDIA_API_KEY; the core run never needs it)")
    args = ap.parse_args()

    schema = OutputSchema.from_csv(args.schema)
    rows = load_input(args.input)
    if args.limit:
        rows = rows[: args.limit]

    print(f"input   : {args.input}  ({len(rows)} rows)")
    print(f"schema  : {args.schema}  ({len(schema)} columns, "
          f"{schema.n_attribute_slots} attribute slots)")

    proposer = None
    if args.propose:
        from glassbox.enrich import build_proposer

        proposer = build_proposer()
        print(f"proposer: {'enabled' if proposer else 'unavailable (no API key) -- core run unaffected'}")

    vocab = None
    if args.vocab and Path(args.vocab).exists():
        from glassbox.induce import InducedVocabulary

        vocab = InducedVocabulary.from_json(args.vocab)
        print(f"vocab   : {args.vocab} ({len(vocab.brands)} brands reused)")

    pipeline = Pipeline(schema, vocab=vocab, proposer=proposer)

    def progress(done: int, total: int) -> None:
        pct = 100 * done / max(total, 1)
        print(f"\r  enriching {done}/{total} ({pct:.0f}%)", end="", flush=True)

    started = time.perf_counter()
    enriched = pipeline.run(rows, progress=progress)
    print()

    # --- write artefacts ---
    out_dir = Path(args.out)
    csv_path = write_delivery_csv(out_dir / "enriched.csv", schema, enriched)
    xlsx_path = None
    if not args.no_xlsx:
        xlsx_path = write_delivery_xlsx(out_dir / "enriched.xlsx", schema, enriched)
    prov_path = write_provenance(out_dir / "provenance.jsonl", enriched)
    queue_path = write_review_queue(out_dir / "review_queue.csv", schema, enriched)

    buckets = triage(enriched)
    evaluation = evaluate(enriched, schema)
    report = {
        "run": pipeline.stats.as_dict(),
        "compliance": evaluation["compliance"],
        "gold_channel_check": evaluation["gold_channel_check"],
        "cells_by_source": source_histogram(enriched),
        "blanks_by_reason": blank_reason_histogram(enriched),
        "triage": {k: len(v) for k, v in buckets.items()},
        "review_reasons": reason_histogram(enriched),
        "vocabulary": {k: int(v) for k, v in pipeline.vocab.stats.items()},
        "notes": evaluation["notes"],
    }
    report_path = write_report(out_dir / "report.json", report)

    # --- console summary ---
    stats = pipeline.stats.as_dict()
    print("\n" + "=" * 68)
    print("RUN")
    for key in ("rows", "classified_pct", "brand_resolved_pct",
                "attribute_slot_fill_pct", "needs_review_pct", "rows_per_s"):
        print(f"  {key:<28} {stats[key]}")
    print("\nCOMPLIANCE")
    for key, value in evaluation["compliance"].items():
        print(f"  {key:<28} {value}")
    print("\nTRIAGE")
    for key, value in buckets.items():
        print(f"  {key:<28} {len(value)}")
    print("\nCELLS BY SOURCE")
    for key, value in list(source_histogram(enriched).items())[:8]:
        print(f"  {key:<28} {value}")
    print("\nWROTE")
    for path in (csv_path, xlsx_path, prov_path, queue_path, report_path,
                 report_path.with_suffix(".md")):
        if path:
            size = Path(path).stat().st_size
            print(f"  {str(path):<40} {size / 1024:>9.1f} KB")
    print(f"\ntotal {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
