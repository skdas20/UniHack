"""Tests for the label alignment and metrics. No GPU, no torch, no downloads.

    python test_tagging.py

Worth running before train.py if you have changed anything, and it is how we
verified the training script's logic without a GPU on hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tagging import (  # noqa: E402
    IGNORE,
    align_bio,
    build_tag_list,
    classification_metrics,
    decode_spans,
    token_metrics,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------------------
# BIO alignment
# --------------------------------------------------------------------------

def test_align_basic() -> None:
    text = 'Milw 14"x1/8"x1" Masonry Cut Off Disc'
    tags = build_tag_list(["DIAMETER", "THICKNESS"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}

    # a plausible word-piece split, with specials at both ends
    offsets = [
        (0, 0),          # [CLS]
        (0, 4),          # Milw
        (5, 7),          # 14
        (7, 8),          # "
        (8, 9),          # x
        (9, 12),         # 1/8
        (17, 24),        # Masonry
        (0, 0),          # [SEP]
        (0, 0),          # padding
    ]
    entities = [
        {"start": 5, "end": 8, "label": "DIAMETER"},    # 14"
        {"start": 9, "end": 12, "label": "THICKNESS"},  # 1/8
    ]
    labels = align_bio(offsets, entities, tag_to_id)

    check("specials and padding are ignored",
          labels[0] == IGNORE and labels[-1] == IGNORE and labels[-2] == IGNORE,
          str(labels))
    check("token outside every entity is O",
          labels[1] == tag_to_id["O"], f"got {labels[1]}")
    check("token at the entity start is B-",
          labels[2] == tag_to_id["B-DIAMETER"], f"got {labels[2]}")
    check("token inside the entity is I-",
          labels[3] == tag_to_id["I-DIAMETER"], f"got {labels[3]}")
    check("separator between entities is O",
          labels[4] == tag_to_id["O"], f"got {labels[4]}")
    check("second entity starts a fresh B-",
          labels[5] == tag_to_id["B-THICKNESS"], f"got {labels[5]}")
    check("trailing word is O",
          labels[6] == tag_to_id["O"], f"got {labels[6]}")


def test_align_rejects_unknown_label() -> None:
    tags = build_tag_list(["GRIT"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    labels = align_bio(
        [(0, 4)], [{"start": 0, "end": 4, "label": "NOT_A_DECLARED_LABEL"}], tag_to_id
    )
    check("an undeclared entity label falls back to O rather than crashing",
          labels == [tag_to_id["O"]], str(labels))


def test_align_partial_token_not_tagged() -> None:
    """A token straddling the entity boundary must not inherit the tag."""
    tags = build_tag_list(["WIDTH"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    # entity covers 5..8 but the token spans 5..12, so it is not wholly inside
    labels = align_bio([(5, 12)], [{"start": 5, "end": 8, "label": "WIDTH"}], tag_to_id)
    check("token wider than the entity is not tagged",
          labels == [tag_to_id["O"]], str(labels))


# --------------------------------------------------------------------------
# span decoding
# --------------------------------------------------------------------------

def test_decode_spans() -> None:
    tags = build_tag_list(["GRIT", "WIDTH"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    sequence = [
        tag_to_id["O"],
        tag_to_id["B-GRIT"],
        tag_to_id["I-GRIT"],
        tag_to_id["O"],
        tag_to_id["B-WIDTH"],
    ]
    valid = [True] * len(sequence)
    spans = decode_spans(sequence, valid, id_to_tag)
    check("two spans decoded with correct boundaries",
          spans == {(1, 3, "GRIT"), (4, 5, "WIDTH")}, str(spans))

    # an I- tag with no preceding B- of the same type must not open a span
    orphan = [tag_to_id["O"], tag_to_id["I-GRIT"], tag_to_id["O"]]
    check("orphan I- tag does not open a span",
          decode_spans(orphan, [True] * 3, id_to_tag) == set(),
          str(decode_spans(orphan, [True] * 3, id_to_tag)))

    # adjacent entities of the same type stay separate
    adjacent = [tag_to_id["B-GRIT"], tag_to_id["B-GRIT"]]
    check("adjacent B- tags produce two spans, not one",
          decode_spans(adjacent, [True] * 2, id_to_tag) == {(0, 1, "GRIT"), (1, 2, "GRIT")},
          str(decode_spans(adjacent, [True] * 2, id_to_tag)))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_classification_metrics() -> None:
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
    labels = np.array([1, 0, 1, 0])
    result = classification_metrics(logits, labels)
    check("perfect predictions score 1.0",
          abs(result["accuracy"] - 1.0) < 1e-9 and abs(result["macro_f1"] - 1.0) < 1e-9,
          str(result))

    labels_wrong = np.array([0, 0, 1, 0])
    result = classification_metrics(logits, labels_wrong)
    check("one error out of four gives 0.75 accuracy",
          abs(result["accuracy"] - 0.75) < 1e-9, str(result))
    check("macro F1 penalises the minority class harder than accuracy does",
          result["macro_f1"] < result["accuracy"], str(result))


def test_token_metrics_are_not_flattered_by_O() -> None:
    """The reason we report entity F1: an all-O model must score 0 on it."""
    tags = build_tag_list(["GRIT"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}
    n_tags = len(tags)

    # truth: one GRIT entity in a sequence of 10 tokens
    labels = np.full((1, 10), tag_to_id["O"])
    labels[0, 4] = tag_to_id["B-GRIT"]
    labels[0, 5] = tag_to_id["I-GRIT"]

    # a model that always predicts O
    logits = np.zeros((1, 10, n_tags))
    logits[..., tag_to_id["O"]] = 1.0

    result = token_metrics(logits, labels, id_to_tag)
    check("all-O model still scores high token accuracy (which is why we don't rely on it)",
          result["token_accuracy"] >= 0.79, str(result))
    check("all-O model scores zero entity F1",
          result["entity_f1"] == 0.0, str(result))

    # now a perfect model
    perfect = np.zeros((1, 10, n_tags))
    for position in range(10):
        perfect[0, position, labels[0, position]] = 1.0
    result = token_metrics(perfect, labels, id_to_tag)
    check("perfect model scores entity F1 of 1.0",
          abs(result["entity_f1"] - 1.0) < 1e-9, str(result))


def test_ignored_positions_excluded() -> None:
    tags = build_tag_list(["GRIT"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    labels = np.array([[IGNORE, tag_to_id["B-GRIT"], IGNORE]])
    logits = np.zeros((1, 3, len(tags)))
    logits[0, 0, tag_to_id["B-GRIT"]] = 1.0   # wrong, but ignored
    logits[0, 1, tag_to_id["B-GRIT"]] = 1.0   # right
    logits[0, 2, tag_to_id["I-GRIT"]] = 1.0   # wrong, but ignored

    result = token_metrics(logits, labels, id_to_tag)
    check("padding positions do not affect accuracy",
          abs(result["token_accuracy"] - 1.0) < 1e-9, str(result))


# --------------------------------------------------------------------------
# the real data, end to end through the alignment
# --------------------------------------------------------------------------

def test_real_data_aligns() -> None:
    data = Path(__file__).resolve().parent / "data"
    meta_path, spans_path = data / "meta.json", data / "spans_train.jsonl"
    if not (meta_path.exists() and spans_path.exists()):
        print("  skip  real-data alignment (data/ not present)")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tags = build_tag_list(meta["span_labels"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}

    rows = [
        json.loads(line)
        for line in spans_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tagged_rows = 0
    for row in rows:
        text = row["text"]
        # character-level offsets stand in for a tokenizer here: if alignment
        # works per character it works per word piece, and this needs no download
        offsets = [(i, i + 1) for i in range(len(text))]
        labels = align_bio(offsets, row["entities"], tag_to_id)
        if any(label != tag_to_id["O"] and label != IGNORE for label in labels):
            tagged_rows += 1

    check(f"every span row produces at least one non-O tag ({tagged_rows}/{len(rows)})",
          tagged_rows == len(rows),
          f"{len(rows) - tagged_rows} row(s) would train with no positive labels")


def main() -> int:
    print("=" * 70)
    print("GlassBox training logic - tests (no GPU, no torch, no downloads)")
    print("=" * 70)
    for test in (
        test_align_basic,
        test_align_rejects_unknown_label,
        test_align_partial_token_not_tagged,
        test_decode_spans,
        test_classification_metrics,
        test_token_metrics_are_not_flattered_by_O,
        test_ignored_positions_excluded,
        test_real_data_aligns,
    ):
        print(f"\n{test.__name__}")
        test()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
