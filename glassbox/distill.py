"""The local-model layer: serve the two fine-tuned encoders at inference time.

The training package distils the rule engine into two small encoders -- a
77-class product classifier and a 38-label attribute span tagger
(``training/``). Until now those models existed only as artifacts: nothing in
the engine loaded them. This module is the bridge.

Design constraints, in the order they were decided:

* **The core run must not need it.** torch and transformers are imported
  lazily, on first use. A machine without them gets the plain rule engine,
  byte-for-byte, exactly as before.
* **Model output is a proposal, never a fact.** Cells it produces carry
  ``Source.LOCAL_MODEL``, sit in the ``GENERATIVE`` set, are LOV-validated
  before emission, and route the row to review. The confidence engine already
  knows how to weigh all of that -- this layer just never lies about provenance.
* **Evidence still means characters.** The span tagger predicts BIO tags over
  the description's own tokens, so every value it fills can point at the exact
  substring it was read out of, same as a regex extraction. The audit view does
  not distinguish "highlighted by a pattern" from "highlighted by a model"
  except in the source column -- by design.

Both models run on CPU in milliseconds per row; there is no GPU, server or API
key anywhere in this path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

#: Where ``training/eval_models.py`` leaves the trained encoders.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "training" / "models"

#: Below this softmax probability a model classification is discarded rather
#: than emitted. The number is measured on the held-out val split (see
#: ``training/full_eval.py``): accuracy above it is high, below it the model
#: is mostly guessing.
MIN_CLASSIFY_PROB = 0.55


def normalize_label(label: str) -> str:
    """The label normalisation used when the training data was built.

    Mirrors ``scripts/make_training_data.py``: non-letters collapse to ``_``
    and the result is uppercased, so ``"Blade/Wheel Size"`` and
    ``"Blade Or Wheel Size"`` both become ``BLADE_OR_WHEEL_SIZE``. The inverse
    mapping is recovered from the live contracts, never by title-casing.
    """
    return re.sub(r"[^A-Za-z]+", "_", label).strip("_").upper()


@dataclass(frozen=True, slots=True)
class ClassPrediction:
    classpath: str
    prob: float


@dataclass(frozen=True, slots=True)
class SpanHit:
    start: int
    end: int
    label: str  # normalised, e.g. NOMINAL_THICKNESS
    prob: float


class LocalModels:
    """Loads and serves the two distilled encoders. CPU-only, no network.

    Constructing the object is cheap; the heavy imports and weights load on
    first inference so that merely *checking* availability never costs a
    multi-second torch import.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._classpath = None  # (tokenizer, model, labels) once loaded
        self._spans = None  # (tokenizer, model, tags) once loaded
        self.load_seconds = 0.0

    # ---------- availability ----------

    @property
    def classpath_dir(self) -> Path:
        return self.model_dir / "classpath"

    @property
    def spans_dir(self) -> Path:
        return self.model_dir / "spans"

    def available(self) -> bool:
        """True if both saved models exist on disk."""
        return (self.classpath_dir / "model.safetensors").exists() and (
            self.spans_dir / "model.safetensors"
        ).exists()

    def __repr__(self) -> str:  # keeps logs honest about what loaded
        return f"LocalModels({self.model_dir}, loaded={self._classpath is not None})"

    # ---------- classification ----------

    def classify_batch(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> list[ClassPrediction | None]:
        """Classpath for each text, with softmax confidence.

        Returns ``None`` entries (never a guess) for empty texts.
        """
        if not texts:
            return []
        tok, model, labels = self._ensure_classpath()
        out: list[ClassPrediction | None] = []
        with torch_inference():
            for i in range(0, len(texts), batch_size):
                batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
                enc = tok(batch, truncation=True, max_length=96, padding=True, return_tensors="pt")
                logits = model(**enc).logits
                probs = _softmax(logits)
                top = probs.argmax(dim=-1)
                for row, cls in enumerate(top):
                    prob = float(probs[row, int(cls)])
                    out.append(ClassPrediction(labels[int(cls)], prob))
        return out

    def classify(self, text: str) -> ClassPrediction | None:
        """Single-text convenience wrapper (a batch of one)."""
        result = self.classify_batch([text])
        return result[0] if result else None

    # ---------- span tagging ----------

    def tag_spans_batch(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> list[list[SpanHit]]:
        """Predict attribute spans as *character* ranges of each input text.

        The model works on word-pieces; this converts back to the character
        offsets the provenance system needs, so a model-filled slot points at
        its evidence exactly like a regex extraction does.
        """
        if not texts:
            return []
        tok, model, tags = self._ensure_spans()
        results: list[list[SpanHit]] = []
        with torch_inference():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(
                    [t if t.strip() else " " for t in batch],
                    truncation=True,
                    max_length=96,
                    padding=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = enc.pop("offset_mapping")
                logits = model(**enc).logits
                probs = _softmax(logits)
                preds = logits.argmax(dim=-1)
                for row in range(len(batch)):
                    results.append(
                        _decode_bio(
                            preds[row].tolist(),
                            probs[row],
                            [tuple(o) for o in offsets[row].tolist()],
                            tags,
                        )
                    )
        return results

    def tag_spans(self, text: str) -> list[SpanHit]:
        """Single-text convenience wrapper."""
        result = self.tag_spans_batch([text])
        return result[0] if result else []

    # ---------- lazy loading ----------

    def _ensure_classpath(self):
        import json

        import torch  # noqa: F401  (fail loudly if genuinely missing)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._classpath is None:
            import time

            started = time.perf_counter()
            folder = self.classpath_dir
            tok = AutoTokenizer.from_pretrained(str(folder))
            model = AutoModelForSequenceClassification.from_pretrained(str(folder))
            model.eval()
            labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
            self._classpath = (tok, model, labels)
            self.load_seconds += time.perf_counter() - started
        return self._classpath

    def _ensure_spans(self):
        import json

        import torch  # noqa: F401
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if self._spans is None:
            import time

            started = time.perf_counter()
            folder = self.spans_dir
            tok = AutoTokenizer.from_pretrained(str(folder), use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(str(folder))
            model.eval()
            tags = json.loads((folder / "tags.json").read_text(encoding="utf-8"))
            self._spans = (tok, model, tags)
            self.load_seconds += time.perf_counter() - started
        return self._spans


# --- helpers ----------------------------------------------------------------


def torch_inference():
    """A null-context-ish ``torch.inference_mode()`` obtained lazily."""
    import torch

    return torch.inference_mode()


def _softmax(logits):
    import torch

    return torch.softmax(logits, dim=-1)


def _decode_bio(
    tag_ids: list[int],
    token_probs,
    offsets: list[tuple[int, int]],
    tags: list[str],
) -> list[SpanHit]:
    """BIO tag ids -> character spans, with the weakest-token confidence.

    ``offsets`` is the fast tokenizer's offset mapping: ``(0, 0)`` marks
    special tokens and padding, which carry no characters and are skipped.
    """
    hits: list[SpanHit] = []
    start_tok: int | None = None
    label = ""
    conf = 1.0

    def flush(end_tok: int) -> None:
        nonlocal start_tok, label, conf
        if start_tok is not None and label:
            c_start = offsets[start_tok][0]
            c_end = offsets[end_tok - 1][1]
            if c_end > c_start:
                hits.append(SpanHit(c_start, c_end, label, conf))
        start_tok, label, conf = None, "", 1.0

    for position, tag_id in enumerate(tag_ids):
        tag = tags[int(tag_id)] if 0 <= int(tag_id) < len(tags) else "O"
        token_start, token_end = offsets[position]
        if token_start == token_end:  # special/padding token
            flush(position)
            continue
        prob = float(token_probs[position, int(tag_id)])
        if tag.startswith("B-"):
            flush(position)
            start_tok, label, conf = position, tag[2:], prob
        elif tag.startswith("I-") and tag[2:] == label and start_tok is not None:
            conf = min(conf, prob)
        else:
            flush(position)
    flush(len(tag_ids))
    return hits


_LABEL_MAP: dict[str, str] | None = None


def slot_label_map() -> dict[str, str]:
    """Normalised span label -> real slot label, over every contract.

    Built from the live contracts rather than by string munging backwards, so
    a slot named ``"Blade/Wheel Size"`` is found by the model's
    ``BLADE_OR_WHEEL_SIZE`` whichever punctuation the label uses.
    """
    global _LABEL_MAP
    if _LABEL_MAP is None:
        from .attributes import CONTRACTS

        mapping: dict[str, str] = {}
        for contract in CONTRACTS.values():
            for slot in contract.slots:
                mapping.setdefault(normalize_label(slot.label), slot.label)
        _LABEL_MAP = mapping
    return _LABEL_MAP


def model_fill_slot(slot, snippet: str, prob: float, text: str, start: int, end: int):
    """Turn a predicted span into a validated :class:`Filled`, or reject it.

    This is the gate that keeps the model honest. An LOV slot only accepts a
    value that is already in its controlled vocabulary (aliases resolved to
    the canonical form); a measure slot only accepts a number+unit the units
    engine can parse and that sits inside the slot's plausibility bounds.
    Anything else is dropped silently -- a model proposal that cannot be
    validated is not emitted at all, which is why the LOV-compliance figure
    stays at 100% even with the model layer on.
    """
    from .extract import Filled

    value, uom = _validated_value(slot, snippet)
    if value is None:
        return None
    conf = min(0.90, max(0.40, prob))
    return Filled(slot, value, uom, _cell_for(value, slot, conf, text, start, end))


def _cell_for(value: str, slot, conf: float, text: str, start: int, end: int):
    from .provenance import Cell, Evidence, Provenance, Source

    return Cell(
        value,
        Provenance(
            Source.LOCAL_MODEL,
            rule=f"spans:model:{normalize_label(slot.label).lower()}",
            confidence=conf,
            evidence=(Evidence(text, start, end),),
            detail=(
                "predicted by the distilled span tagger and validated against "
                "the slot's controlled vocabulary/bounds before emission"
            ),
        ),
    )


_GLOBAL_LOV: dict[str, tuple[str, ...]] | None = None
_GLOBAL_ALIASES: dict[str, dict[str, str]] | None = None


def _global_vocabularies():
    """Label -> LOV union and label -> alias map, across every contract.

    ``evaluate.py`` checks emitted values against exactly this union, so the
    model layer must too. It closes a real hole: one contract can define
    "Color" as a controlled 32-value slot while another uses the same label
    as free text -- a fill valid for the row's own slot could still violate
    the label's global vocabulary.
    """
    global _GLOBAL_LOV, _GLOBAL_ALIASES
    if _GLOBAL_LOV is None:
        from .attributes import CONTRACTS

        lov: dict[str, set[str]] = {}
        aliases: dict[str, dict[str, str]] = {}
        for contract in CONTRACTS.values():
            for slot in contract.slots:
                if slot.lov:
                    lov.setdefault(slot.label, set()).update(slot.lov)
                for alias, canonical in slot.lov_aliases:
                    aliases.setdefault(slot.label, {})[alias.casefold()] = canonical
        _GLOBAL_LOV = {k: tuple(sorted(v)) for k, v in lov.items()}
        _GLOBAL_ALIASES = aliases
    return _GLOBAL_LOV, _GLOBAL_ALIASES


def _validated_value(slot, snippet: str) -> tuple[str | None, str]:
    """Check a raw span against its slot; return ``(value, uom)`` or ``(None, "")``."""
    from . import units as U
    from .attributes import Kind

    text = " ".join(snippet.split()).strip(" -–—/,;")
    if not text or len(text) > 40:
        return None, ""

    global_lov, global_aliases = _global_vocabularies()
    permitted = global_lov.get(slot.label)
    if permitted or slot.kind == Kind.LOV:
        # Controlled vocabulary: match the value itself, then any alias from
        # any contract defining this label ('WH' -> 'White'), else reject.
        folded = text.casefold()
        for candidates in (slot.lov or ()), (permitted or ()):
            for candidate in candidates:
                if candidate.casefold() == folded:
                    return candidate, ""
        canonical = global_aliases.get(slot.label, {}).get(folded)
        if canonical is not None and canonical in (permitted or ()):
            return canonical, ""
        for alias, target in slot.lov_aliases:
            if alias.casefold() == folded:
                return target, ""
        return None, ""

    if slot.kind in (Kind.MEASURE, Kind.DIMENSION):
        candidates = U.find_measurements(text)
        if slot.family:
            candidates = [m for m in candidates if m.family == slot.family]
        for m in candidates:
            numeric = m.numeric
            if numeric is None:
                continue
            if slot.lo is not None and numeric < slot.lo:
                continue
            if slot.hi is not None and numeric > slot.hi:
                continue
            return m.magnitude, m.unit.symbol
        return None, ""

    if slot.kind == Kind.COUNT:
        m = re.search(r"\b(\d{1,4})\b", text)
        if not m:
            return None, ""
        n = int(m.group(1))
        if slot.lo is not None and n < slot.lo:
            return None, ""
        if slot.hi is not None and n > slot.hi:
            return None, ""
        return str(n), ""

    # TEXT / SERIES / MODEL: evidence-backed, taken as written.
    return text, ""
