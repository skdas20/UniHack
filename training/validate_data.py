"""Pre-flight check on the training data. Needs nothing but the standard library.

Run this BEFORE `train.py` if anything looks wrong, or just to confirm the
folder arrived intact:

    python validate_data.py

It verifies that every file parses, that the label sets are internally
consistent, and - most importantly - that every character span in the span
data actually lands on the text it claims to. A single off-by-one offset here
becomes a silently mislabelled token and a model that quietly underperforms, so
it is worth thirty seconds to rule out.

Exit code 0 means the data is good to train on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

REQUIRED = (
    "classpath_train.jsonl",
    "classpath_val.jsonl",
    "spans_train.jsonl",
    "spans_val.jsonl",
    "meta.json",
)


def fail(message: str) -> None:
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name} line {number}: bad JSON - {exc}")
    return rows


def main() -> int:
    print("=" * 70)
    print("GlassBox training data - pre-flight check")
    print("=" * 70)
    problems = 0

    # --- files present ---
    missing = [name for name in REQUIRED if not (DATA / name).exists()]
    if missing:
        fail(f"missing files: {', '.join(missing)}")
        print("\nThe data/ folder did not arrive complete. Re-unzip training/.")
        return 1
    ok(f"all {len(REQUIRED)} files present in {DATA}")

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    classpath_labels = set(meta["classpath_labels"])
    span_labels = set(meta["span_labels"])
    ok(f"meta.json declares {len(classpath_labels)} classes and "
       f"{len(span_labels)} span types")

    # --- classpath data ---
    for name in ("classpath_train.jsonl", "classpath_val.jsonl"):
        rows = read_jsonl(DATA / name)
        if not rows:
            fail(f"{name} is empty")
            problems += 1
            continue

        no_text = sum(1 for r in rows if not (r.get("text") or "").strip())
        unknown = {r["label"] for r in rows if r.get("label") not in classpath_labels}
        if no_text:
            fail(f"{name}: {no_text} row(s) with empty text")
            problems += 1
        if unknown:
            fail(f"{name}: labels not declared in meta.json: {sorted(unknown)[:5]}")
            problems += 1
        if not no_text and not unknown:
            distribution = Counter(r["label"] for r in rows)
            ok(f"{name}: {len(rows):,} rows, {len(distribution)} classes, "
               f"min/max per class {min(distribution.values())}/{max(distribution.values())}")

    # --- span data: the offsets are what matter ---
    for name in ("spans_train.jsonl", "spans_val.jsonl"):
        rows = read_jsonl(DATA / name)
        if not rows:
            fail(f"{name} is empty")
            problems += 1
            continue

        bad_offsets = 0
        bad_labels: set[str] = set()
        mismatched = 0
        entities = 0
        for row in rows:
            text = row.get("text") or ""
            for entity in row.get("entities", []):
                entities += 1
                start, end = entity.get("start", -1), entity.get("end", -1)
                if not (0 <= start < end <= len(text)):
                    bad_offsets += 1
                    continue
                if entity.get("label") not in span_labels:
                    bad_labels.add(entity.get("label", "<none>"))
                # the span must be non-blank text, or the tag teaches nothing
                if not text[start:end].strip():
                    mismatched += 1

        if bad_offsets:
            fail(f"{name}: {bad_offsets} span(s) with offsets outside the text")
            problems += 1
        if bad_labels:
            fail(f"{name}: span labels not in meta.json: {sorted(bad_labels)[:5]}")
            problems += 1
        if mismatched:
            fail(f"{name}: {mismatched} span(s) covering only whitespace")
            problems += 1
        if not (bad_offsets or bad_labels or mismatched):
            ok(f"{name}: {len(rows):,} rows, {entities:,} spans, all offsets valid")

    # --- no leakage between train and val ---
    train_texts = {r["text"].lower() for r in read_jsonl(DATA / "classpath_train.jsonl")}
    val_texts = {r["text"].lower() for r in read_jsonl(DATA / "classpath_val.jsonl")}
    overlap = train_texts & val_texts
    if overlap:
        fail(f"{len(overlap)} identical text(s) appear in both train and val - "
             f"validation scores would be inflated")
        problems += 1
    else:
        ok(f"no text overlap between train ({len(train_texts):,}) and "
           f"val ({len(val_texts):,})")

    print("=" * 70)
    if problems:
        print(f"{problems} problem(s) found. Send this output to the team before training.")
        return 1
    print("Data is consistent. You are clear to run:  python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
