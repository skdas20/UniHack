"""Label alignment and metrics. Deliberately free of torch and transformers.

Everything here is pure Python plus numpy, which means it is testable without a
GPU, without a 2 GB install, and without downloading a model. That matters
because these are the two places this kind of script usually goes quietly wrong:

* **BIO alignment.** An off-by-one in the offset-to-tag mapping produces a model
  that trains happily and predicts badly. It does not raise; it just costs you
  five points of F1 and you never find out why.
* **Entity F1.** Token accuracy looks excellent on span data because most tokens
  are ``O``. Reporting it alone would flatter the model. Entity-level precision
  and recall over decoded spans is the number that means something.

`test_tagging.py` exercises both against hand-worked examples.
"""

from __future__ import annotations

import numpy as np

#: Ignored index for loss and metrics -- the HuggingFace convention.
IGNORE = -100


def build_tag_list(span_labels: list[str]) -> list[str]:
    """``["GRIT", "WIDTH"]`` -> ``["O", "B-GRIT", "I-GRIT", "B-WIDTH", "I-WIDTH"]``."""
    tags = ["O"]
    for label in span_labels:
        tags.append(f"B-{label}")
        tags.append(f"I-{label}")
    return tags


def align_bio(
    offsets: list[tuple[int, int]],
    entities: list[dict],
    tag_to_id: dict[str, int],
) -> list[int]:
    """Map character-span entities onto word-piece tokens as BIO tag ids.

    ``offsets`` is a fast tokenizer's ``offset_mapping``: one ``(start, end)``
    character range per token, with ``(0, 0)`` marking special tokens and
    padding, which become :data:`IGNORE`.

    A token is ``B-`` when it begins exactly at the entity's start and ``I-``
    when it merely falls inside. Tokens overlapping no entity are ``O``.
    """
    outside = tag_to_id["O"]
    labels: list[int] = []
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE)
            continue
        tag_id = outside
        for entity in entities:
            e_start, e_end = entity["start"], entity["end"]
            # the token must sit wholly inside the entity to inherit its tag
            if start >= e_start and end <= e_end:
                prefix = "B" if start == e_start else "I"
                candidate = f"{prefix}-{entity['label']}"
                tag_id = tag_to_id.get(candidate, outside)
                break
        labels.append(tag_id)
    return labels


def decode_spans(
    tag_ids: "np.ndarray | list[int]",
    valid: "np.ndarray | list[bool]",
    id_to_tag: dict[int, str],
) -> set[tuple[int, int, str]]:
    """Decode a BIO tag sequence into ``(start_token, end_token, label)`` spans."""
    out: set[tuple[int, int, str]] = set()
    start: int | None = None
    current: str | None = None

    for position, (tag_id, is_valid) in enumerate(zip(tag_ids, valid)):
        if not is_valid:
            continue
        tag = id_to_tag.get(int(tag_id), "O")
        if tag.startswith("B-"):
            if start is not None and current:
                out.add((start, position, current))
            start, current = position, tag[2:]
        elif tag.startswith("I-") and current == tag[2:]:
            continue
        else:
            if start is not None and current:
                out.add((start, position, current))
            start, current = None, None

    if start is not None and current:
        out.add((start, len(list(tag_ids)), current))
    return out


def classification_metrics(logits, labels) -> dict[str, float]:
    """Accuracy and macro F1, without pulling in scikit-learn."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean()) if labels.size else 0.0

    f1s: list[float] = []
    for label in np.unique(labels):
        tp = float(((preds == label) & (labels == label)).sum())
        fp = float(((preds == label) & (labels != label)).sum())
        fn = float(((preds != label) & (labels == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def token_metrics(logits, labels, id_to_tag: dict[int, str]) -> dict[str, float]:
    """Token accuracy plus entity-level precision, recall and F1.

    Entity F1 is the honest number. Token accuracy on span data is dominated by
    the ``O`` class -- a model that predicts ``O`` everywhere scores well above
    0.9 on it and has learned nothing.
    """
    logits, labels = np.asarray(logits), np.asarray(labels)
    preds = np.argmax(logits, axis=-1)
    mask = labels != IGNORE

    flat_true = labels[mask]
    flat_pred = preds[mask]
    accuracy = float((flat_pred == flat_true).mean()) if flat_true.size else 0.0

    tp = fp = fn = 0
    for row_pred, row_true, row_mask in zip(preds, labels, mask):
        predicted = decode_spans(row_pred, row_mask, id_to_tag)
        actual = decode_spans(row_true, row_mask, id_to_tag)
        tp += len(predicted & actual)
        fp += len(predicted - actual)
        fn += len(actual - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_accuracy": accuracy,
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
    }
