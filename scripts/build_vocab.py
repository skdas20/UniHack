"""Run the vocabulary induction pass and report what was learned.

    python scripts/build_vocab.py [input.csv] [--out data/vocab/induced.json]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox.induce import induce_from_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data/raw/input_1000.csv")
    ap.add_argument("--out", default="data/vocab/induced.json")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    vocab = induce_from_file(args.input)
    vocab.to_json(args.out)

    print(f"rows                 : {vocab.n_rows}")
    for key, value in vocab.stats.items():
        print(f"{key:21}: {value:.0f}")
    print(f"\nwritten -> {args.out}")

    print(f"\n=== brands (top {args.top} by support) ===")
    ranked = sorted(vocab.brands.values(), key=lambda b: -b.support)
    for entry in ranked[: args.top]:
        mark = "attested" if entry.attested else "OBSERVED"
        aliases = ", ".join(sorted(entry.aliases)[:6])
        print(
            f"  {entry.canonical:<28} {entry.support:>4}  {mark:<8} "
            f"conf={entry.confidence:.2f}  via {entry.linkage:<26} [{aliases}]"
        )

    print(f"\n=== brand candidates held for human review (top {args.top}) ===")
    held = sorted(vocab.brand_candidates.values(), key=lambda b: -b.support)
    for entry in held[: args.top]:
        aliases = ", ".join(sorted(entry.aliases)[:4])
        print(
            f"  {entry.canonical:<24} {entry.support:>4}  pos={entry.mean_position:.2f}  "
            f"[{aliases}]"
        )
    if not held:
        print("  none")

    print(f"\n=== product types (top {args.top}) ===")
    for entry in sorted(vocab.product_types.values(), key=lambda p: -p.support)[: args.top]:
        print(f"  {entry.canonical:<40} {entry.support:>4}  groups={len(entry.groups)}")

    print(f"\n=== series (top {args.top}) ===")
    for name, count in sorted(vocab.series.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {name:<40} {count:>4}")

    print("\n=== unknown unit spellings (for human approval) ===")
    items = list(vocab.unknown_units.items())[:30]
    print("  " + ", ".join(f"{k}({v})" for k, v in items) if items else "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
