"""Build the training corpus for the distilled local models.

Produces a self-contained ``training/`` package that needs no part of this
repository to run: unzip it on a GPU box, make a venv, ``pip install -r
requirements.txt``, ``python train.py``.

## Where the labels come from

There is no labelled dataset in the pack, so we distil the rule engine. For
every row the deterministic pipeline classifies at or above a confidence
threshold, we emit a **silver label**: the classpath it chose, plus the
character spans its extractors read attribute values out of. That gives two
supervised tasks from unlabelled data:

* **classpath** -- sequence classification over the taxonomy leaves.
* **spans** -- BIO token classification over attribute labels.

## Why augment

The 1,000-row input yields roughly 800 confidently-labelled rows spread over
~60 populated leaves. That is far too thin to fine-tune on directly, and the
class distribution is severely skewed (114 LED lamp rows, 3 skylights).

So each row is augmented with transformations that preserve the label while
varying exactly the things that vary in real distributor data: shorthand
expanded or contracted, the MPN prefix present or absent, token order in the
size run, casing, and separator style. Every augmentation is a transformation a
different distributor's export could plausibly have produced, which is the only
kind worth training on -- augmenting with noise that never occurs teaches the
model nothing and costs accuracy.

Rare classes are oversampled toward a floor so the head classes cannot swamp
them.

    python scripts/make_training_data.py [--min-confidence 0.7] [--target 60]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox import textnorm as T  # noqa: E402
from glassbox.attributes import contract_for  # noqa: E402
from glassbox.extract import (  # noqa: E402
    ExtractionContext,
    contract_lov_words,
    extract_contract,
)
from glassbox.induce import InducedVocabulary, induce  # noqa: E402
from glassbox.schema import load_input  # noqa: E402
from glassbox.taxonomy import (  # noqa: E402
    CATEGORIES,
    GroupPrior,
    classify,
    prepare_text,
)

RNG = random.Random(20260823)

# --- augmentation -----------------------------------------------------------

#: Reverse of the expansion lexicon: full form -> shorthand. Applied to make a
#: clean description look like a terser distributor's export.
_CONTRACTIONS = {
    "light": "Lt", "lights": "Lts", "electric": "Elect", "white": "Wh",
    "black": "Bk", "stainless steel": "SS", "square": "Sq", "exterior": "Ext",
    "interior": "Int", "refrigerator": "Refrig", "dishwasher": "DW",
    "grooved": "Grvd", "primed": "Prmd", "professional": "Prof",
    "aluminum": "Alum", "galvanized": "Galv", "assembly": "Assy",
    "adjustable": "Adj", "receptacle": "Recep", "fluorescent": "Fluor",
    "reciprocating": "Recip", "circular": "Circ", "oscillating": "Osc",
    "cordless": "Crdls", "battery": "Batt", "charger": "Chgr",
    "diameter": "Dia", "thickness": "Thk", "mounting": "Mtg",
}


def aug_contract_words(text: str) -> str:
    """Turn a full word back into trade shorthand: ``Light`` -> ``Lt``."""
    out = text
    for full, short in _CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(full)}\b", short, out, count=1, flags=re.IGNORECASE)
    return out


def aug_expand_words(text: str) -> str:
    """The other direction: expand shorthand the way a cleaner feed would."""
    return T.expand(text)


def aug_drop_mpn(text: str, mpn: str) -> str:
    stripped, _ = T.strip_leading_mpn(text, mpn)
    return stripped


def aug_add_mpn(text: str, mpn: str) -> str:
    if mpn and not text.upper().startswith(mpn.upper()):
        return f"{mpn} {text}"
    return text


def aug_separator(text: str) -> str:
    """Distributors disagree about separators: ``-``, ``,``, ``|``, or none."""
    choice = RNG.choice([" - ", ", ", " | ", " "])
    return re.sub(r"\s+-\s+", choice, text, count=1)


def aug_case(text: str) -> str:
    mode = RNG.random()
    if mode < 0.25:
        return text.upper()
    if mode < 0.45:
        return text.lower()
    if mode < 0.60:
        return T.title_case(text)
    return text


def aug_quote_style(text: str) -> str:
    """``14"`` written as ``14 in``, ``14in``, ``14 IN.`` or ``14″``."""
    def repl(m: re.Match[str]) -> str:
        magnitude = m.group(1)
        return RNG.choice(
            [f"{magnitude} in", f"{magnitude}in", f"{magnitude} IN.",
             f'{magnitude}"', f"{magnitude} inch"]
        )

    return re.sub(r'(\d+(?:[-./]\d+)*)"', repl, text)


def aug_trailing_noise(text: str) -> str:
    return text + RNG.choice(
        ["", "", " - Display Only", " (Linear Foot)", " - Special Order",
         " NEW", " - Clearance"]
    )


AUGMENTATIONS = (
    ("identity", lambda t, m: t),
    ("contract_words", lambda t, m: aug_contract_words(t)),
    ("expand_words", lambda t, m: aug_expand_words(t)),
    ("drop_mpn", lambda t, m: aug_drop_mpn(t, m)),
    ("add_mpn", lambda t, m: aug_add_mpn(t, m)),
    ("separator", lambda t, m: aug_separator(t)),
    ("case", lambda t, m: aug_case(t)),
    ("quote_style", lambda t, m: aug_quote_style(t)),
    ("trailing_noise", lambda t, m: aug_trailing_noise(t)),
    ("expand_then_case", lambda t, m: aug_case(aug_expand_words(t))),
    ("contract_then_quotes", lambda t, m: aug_quote_style(aug_contract_words(t))),
    ("drop_mpn_then_expand", lambda t, m: aug_expand_words(aug_drop_mpn(t, m))),
)


# --- span labelling ---------------------------------------------------------


def span_examples(
    text: str, filled: dict, contract_name: str
) -> list[dict]:
    """BIO entities from the extractors' evidence spans."""
    entities = []
    for label, f in filled.items():
        if not f:
            continue
        for ev in f.cell.prov.evidence:
            if ev.start >= ev.end:
                continue
            snippet = ev.snippet
            if not snippet or snippet not in text:
                continue
            start = text.index(snippet)
            entities.append(
                {
                    "start": start,
                    "end": start + len(snippet),
                    "label": re.sub(r"[^A-Za-z]+", "_", label).strip("_").upper(),
                    "value": f.value,
                }
            )
    # drop overlaps, keeping the longest
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    kept: list[dict] = []
    for entity in entities:
        if kept and entity["start"] < kept[-1]["end"]:
            continue
        kept.append(entity)
    return kept


# --- main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/input_1000.csv")
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument(
        "--target", type=int, default=60,
        help="oversample each class up to at least this many examples",
    )
    ap.add_argument("--val-split", type=float, default=0.12)
    args = ap.parse_args()

    rows = load_input(args.input)
    print(f"loaded {len(rows)} rows from {args.input}")

    vocab_path = Path("data/vocab/induced.json")
    vocab = (
        InducedVocabulary.from_json(vocab_path)
        if vocab_path.exists()
        else induce(rows)
    )
    series_lex = tuple(vocab.series)

    # fit the prior exactly as the pipeline does
    prior = GroupPrior()
    for row in rows:
        group = T.clean(row.get("Part_Manuf"))
        result = classify(prepare_text(row.desc, row.mpn), group=group)
        if result.ok and result.confidence >= args.min_confidence:
            prior.observe(group, result.category.classpath, result.confidence)

    # --- collect silver labels ---
    seeds: list[dict] = []
    lov_cache: dict[str, frozenset] = {}
    skipped_low_confidence = 0

    for row in rows:
        group = T.clean(row.get("Part_Manuf"))
        result = classify(prepare_text(row.desc, row.mpn), group=group, prior=prior)
        if not result.ok or result.confidence < args.min_confidence:
            skipped_low_confidence += 1
            continue

        category = result.category
        contract = contract_for(category.contract)
        if contract.name not in lov_cache:
            lov_cache[contract.name] = contract_lov_words(contract)

        body, _ = T.strip_leading_mpn(T.clean(row.desc), row.mpn)
        body, _ = T.strip_noise(body)
        ctx = ExtractionContext(
            text=body, mpn=row.mpn, contract_name=contract.name,
            series_lexicon=series_lex, lov_words=lov_cache[contract.name],
        )
        filled = extract_contract(ctx, contract)

        seeds.append(
            {
                "text": T.clean(row.desc),
                "body": body,
                "mpn": row.mpn,
                "classpath": category.classpath,
                "leaf": category.leaf,
                "contract": contract.name,
                "teacher_confidence": round(result.confidence, 4),
                "entities": span_examples(body, filled, contract.name),
            }
        )

    print(f"silver-labelled: {len(seeds)}  (skipped {skipped_low_confidence} "
          f"below confidence {args.min_confidence})")

    by_class: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        by_class[seed["classpath"]].append(seed)
    print(f"populated classes: {len(by_class)} of {len(CATEGORIES)} taxonomy leaves")

    # --- augment, with a per-class floor ---
    classpath_examples: list[dict] = []
    span_records: list[dict] = []

    for classpath, members in by_class.items():
        needed = max(args.target, len(members))
        produced = 0
        attempt = 0
        # Randomised chains rather than a fixed cycle. Cycling through the
        # transformation list deterministically collapses under de-duplication:
        # `case` on an already-uppercase description is a no-op, `expand_words`
        # on text with no shorthand is a no-op, and the same seed row keeps
        # producing the same handful of strings. Composing 1-3 randomly chosen
        # operations per example gives the variety the fine-tune actually needs.
        while produced < needed and attempt < needed * 12:
            seed = members[RNG.randrange(len(members))]
            chain = RNG.sample(AUGMENTATIONS, k=RNG.randint(1, 3))
            name = "+".join(n for n, _ in chain)
            attempt += 1
            try:
                text = seed["text"]
                for _n, fn in chain:
                    text = fn(text, seed["mpn"])
                text = T.clean(text)
            except Exception:
                continue
            if not text or len(text) < 4:
                continue
            classpath_examples.append(
                {
                    "text": text,
                    "label": classpath,
                    "leaf": seed["leaf"],
                    "augmentation": name,
                    "teacher_confidence": seed["teacher_confidence"],
                }
            )
            produced += 1

        # span examples stay unaugmented: shifting the text invalidates offsets,
        # and a mislabelled span is worse than a missing one.
        for seed in members:
            if seed["entities"]:
                span_records.append(
                    {
                        "text": seed["body"],
                        "entities": seed["entities"],
                        "classpath": classpath,
                    }
                )

    # de-duplicate identical (text, label) pairs produced by different routes
    seen: set[tuple[str, str]] = set()
    deduped = []
    for example in classpath_examples:
        key = (example["text"].lower(), example["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    classpath_examples = deduped
    RNG.shuffle(classpath_examples)
    RNG.shuffle(span_records)

    # --- split ---
    def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
        cut = max(1, int(len(items) * args.val_split))
        return items[cut:], items[:cut]

    cls_train, cls_val = split(classpath_examples)
    span_train, span_val = split(span_records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items: list[dict]) -> None:
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name:<26} {len(items):>6} examples  {path.stat().st_size/1024:>8.1f} KB")

    dump("classpath_train.jsonl", cls_train)
    dump("classpath_val.jsonl", cls_val)
    dump("spans_train.jsonl", span_train)
    dump("spans_val.jsonl", span_val)

    labels = sorted({e["label"] for e in classpath_examples})
    span_labels = sorted({e["label"] for r in span_records for e in r["entities"]})
    meta = {
        "generated_from": args.input,
        "source_rows": len(rows),
        "silver_labelled_rows": len(seeds),
        "min_teacher_confidence": args.min_confidence,
        "classpath_labels": labels,
        "span_labels": span_labels,
        "class_floor": args.target,
        "augmentations": [name for name, _ in AUGMENTATIONS],
        "class_distribution": dict(Counter(e["label"] for e in classpath_examples).most_common()),
        "note": (
            "Labels are silver: produced by the deterministic rule engine, not "
            "by human annotation. This is a distillation set. A model trained "
            "on it learns to generalise the rules to phrasings the rules miss, "
            "which is the point -- it cannot exceed the teacher on the cases "
            "the teacher already handles."
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  meta.json                  {len(labels)} classpath labels, "
          f"{len(span_labels)} span labels")

    print("\nclass distribution (top 12):")
    for label, count in Counter(e["label"] for e in classpath_examples).most_common(12):
        print(f"  {count:>5}  {label.rsplit('>', 1)[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
