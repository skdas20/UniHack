"""Extract attributes for a sample of rows and show the filled contracts."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox import textnorm as T  # noqa: E402
from glassbox.attributes import contract_for  # noqa: E402
from glassbox.extract import ExtractionContext, coverage, extract_contract  # noqa: E402
from glassbox.induce import InducedVocabulary  # noqa: E402
from glassbox.schema import load_input  # noqa: E402
from glassbox.taxonomy import GroupPrior, classify, prepare_text  # noqa: E402

SAMPLES = [
    'DCB518ASTS06G Diablo 1/2"x18" - Sanding Belt 6pc',
    '49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc',
    "564922 60W Led BA11 50k 3pk",
    "1nx6-20' Pebble Beach Grooved - Trex Enhance Basics Decking",
    "10-4 SO Cord (Linear Foot)",
    "PDSH4816AF Dishwasher SS - Display Only",
    "DPH31B 1\" #3 Phillips Drive - Bit",
    "JT1-549 JWBS18SFX 18\" Bandsaw - 1.75HP 1PH 115V",
    "4'x10' HardiePanel Smooth - Primed",
    "42275BK Kichler Ceiling Lt",
    "55418901 6/6/6 UD Triplex Aluminum Wire",
    "4x4 2G GFI Box Cover",
    "3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/vocab/induced.json")
    ap.add_argument("--input", default="data/raw/input_1000.csv")
    ap.add_argument("--all", action="store_true", help="report coverage over the whole file")
    args = ap.parse_args()

    vocab = InducedVocabulary.from_json(args.vocab)
    series_lex = tuple(vocab.series)

    rows = load_input(args.input)
    by_desc = {r.desc: r for r in rows}

    # fit the prior once
    prior = GroupPrior()
    for row in rows:
        result = classify(prepare_text(row.desc, row.mpn), group=T.clean(row.get("Part_Manuf")))
        if result.ok and result.confidence >= 0.7:
            prior.observe(T.clean(row.get("Part_Manuf")), result.category.classpath, result.confidence)

    for desc in SAMPLES:
        row = by_desc.get(desc)
        mpn = row.mpn if row else ""
        group = T.clean(row.get("Part_Manuf")) if row else ""
        text = prepare_text(desc, mpn)
        result = classify(text, group=group, prior=prior)
        leaf = result.category.leaf if result.ok else "<unclassified>"
        contract = contract_for(result.category.contract if result.ok else "generic")

        body, _ = T.strip_leading_mpn(T.clean(desc), mpn)
        body, _ = T.strip_noise(body)
        ctx = ExtractionContext(
            text=body, mpn=mpn, contract_name=contract.name, series_lexicon=series_lex
        )
        filled = extract_contract(ctx, contract)

        print("=" * 100)
        print(f"IN   {desc}")
        print(f"CAT  {leaf}   [contract={contract.name}]  coverage={coverage(filled):.0%}")
        for label in contract.labels():
            f = filled[label]
            if not f:
                continue
            ev = f.cell.prov.evidence
            snip = ev[0].snippet if ev else ""
            print(
                f"       {label:<26} = {f.with_uom:<28} "
                f"[{f.cell.prov.rule}] {'<- ' + repr(snip) if snip else ''}"
            )

    if args.all:
        total, filled_slots, slots = 0, 0, 0
        for row in rows:
            text = prepare_text(row.desc, row.mpn)
            result = classify(text, group=T.clean(row.get("Part_Manuf")), prior=prior)
            contract = contract_for(result.category.contract if result.ok else "generic")
            body, _ = T.strip_leading_mpn(T.clean(row.desc), row.mpn)
            body, _ = T.strip_noise(body)
            ctx = ExtractionContext(
                text=body, mpn=row.mpn, contract_name=contract.name, series_lexicon=series_lex
            )
            f = extract_contract(ctx, contract)
            total += 1
            slots += len(f)
            filled_slots += sum(1 for x in f.values() if x)
        print("\n" + "=" * 100)
        print(f"rows={total}  slots={slots}  filled={filled_slots}  "
              f"mean slot coverage={filled_slots / max(slots, 1):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
