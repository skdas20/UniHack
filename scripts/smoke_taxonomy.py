"""Classify the whole working dataset and report coverage.

    python scripts/smoke_taxonomy.py [input.csv] [--show-misses 40]

Two passes: classify on lexical evidence alone, fit the group prior from the
confident results, then reclassify so ambiguous rows can inherit their
distributor's prior.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox import textnorm as T  # noqa: E402
from glassbox.schema import load_input  # noqa: E402
from glassbox.taxonomy import (  # noqa: E402
    GroupPrior,
    classify,
    prepare_text,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data/raw/input_1000.csv")
    ap.add_argument("--show-misses", type=int, default=40)
    ap.add_argument("--show-hits", type=int, default=0)
    args = ap.parse_args()

    rows = load_input(args.input)
    prepared = [
        (row, prepare_text(row.desc, row.mpn), T.clean(row.get("Part_Manuf")))
        for row in rows
    ]

    # pass 1 -- lexical only
    prior = GroupPrior()
    first = []
    for row, text, group in prepared:
        result = classify(text, group=group)
        first.append(result)
        if result.ok and result.confidence >= 0.7:
            prior.observe(group, result.category.classpath, result.confidence)

    # pass 2 -- with the fitted prior
    second = [classify(text, group=group, prior=prior) for _row, text, group in prepared]

    def report(label: str, results) -> Counter:
        hit = sum(1 for r in results if r.ok)
        print(f"\n{label}: {hit}/{len(results)} classified ({hit / len(results):.1%})")
        leaves: Counter = Counter()
        for r in results:
            if r.ok:
                leaves[r.category.leaf] += 1
        return leaves

    report("pass 1 (lexical only)", first)
    leaves = report("pass 2 (with group prior)", second)

    print("\n=== leaf distribution ===")
    for leaf, count in leaves.most_common(40):
        print(f"  {count:>4}  {leaf}")

    misses = [
        (row, text, r)
        for (row, text, _g), r in zip(prepared, second)
        if not r.ok
    ]
    print(f"\n=== unclassified: {len(misses)} rows ===")
    for row, text, r in misses[: args.show_misses]:
        reason = r.provenance.detail[:70]
        print(f"  {row.desc[:74]:<76} | {reason}")

    if args.show_hits:
        print("\n=== sample classifications ===")
        shown = 0
        for (row, _t, _g), r in zip(prepared, second):
            if not r.ok:
                continue
            print(f"  {row.desc[:60]:<62} -> {r.category.leaf:<28} c={r.confidence:.2f}")
            shown += 1
            if shown >= args.show_hits:
                break

    low = [r for r in second if r.ok and r.confidence < 0.6]
    print(f"\nlow-confidence classifications (<0.60): {len(low)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
