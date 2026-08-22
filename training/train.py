"""Train the GlassBox local models. One command, no configuration.

    python train.py                  # trains both models
    python train.py --task classpath # just the classifier
    python train.py --task spans     # just the span tagger

Everything is pre-computed. This script reads the JSONL files in ``data/``,
fine-tunes two small encoder models, prints their scores, and writes them to
``models/``. It auto-detects your GPU and picks a batch size and precision that
fit; on a 6 GB card the defaults are sized to run comfortably.

Two models, because they do different jobs:

* **classpath** -- sequence classification over the taxonomy leaves. Answers
  "what kind of product is this?" for the descriptions the rule engine's
  keyword contracts do not cover.
* **spans**     -- token classification (BIO) over attribute labels. Answers
  "which characters of this string are the diameter, the voltage, the grit?"

Both are *distilled from the rule engine*: their labels were produced by the
deterministic pipeline, not by hand. The point is generalisation. The rules are
precise but brittle -- they only fire on phrasings someone anticipated. A model
trained on their output learns the pattern behind them and fires on phrasings
nobody wrote a rule for. It cannot beat the teacher where the teacher already
works; it earns its place on the 21% of rows the teacher leaves unclassified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Label alignment and metrics live in tagging.py, which imports neither torch
# nor transformers so it can be unit-tested without a GPU or a 2 GB install.
# test_tagging.py covers both, including a pass over the real span data.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tagging import (  # noqa: E402
    align_bio,
    build_tag_list,
    classification_metrics as _classification_metrics,
    token_metrics as _token_metrics,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"
SEED = 20260823

#: Default backbone. 66M parameters, trains in minutes on a 6 GB card, and its
#: tokenizer needs no extra packages -- which matters more than a point of
#: accuracy when someone else has to run this once, tonight, on a deadline.
DEFAULT_MODEL = "distilbert-base-uncased"

#: A stronger option if you have the VRAM and the patience. Needs sentencepiece.
STRONGER_MODEL = "microsoft/deberta-v3-small"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


@dataclass
class Env:
    device: str = "cpu"
    gpu_name: str = ""
    vram_gb: float = 0.0
    fp16: bool = False
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 96

    def describe(self) -> str:
        if self.device == "cuda":
            return (
                f"{self.gpu_name} - {self.vram_gb:.1f} GB VRAM - "
                f"batch {self.batch_size}x{self.grad_accum} - "
                f"{'fp16' if self.fp16 else 'fp32'}"
            )
        return f"CPU only - batch {self.batch_size}x{self.grad_accum} (slow but works)"


def detect_env() -> Env:
    """Size the run to the hardware rather than asking the operator to guess."""
    if not torch.cuda.is_available():
        return Env(device="cpu", batch_size=8, grad_accum=2, fp16=False)

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / (1024**3)
    env = Env(device="cuda", gpu_name=props.name, vram_gb=vram, fp16=True)

    # Effective batch stays ~32 in every tier; only the split changes, so the
    # learning-rate schedule does not need retuning per card.
    if vram < 5:
        env.batch_size, env.grad_accum, env.max_length = 8, 4, 80
    elif vram < 9:  # the 6 GB target
        env.batch_size, env.grad_accum, env.max_length = 16, 2, 96
    elif vram < 17:
        env.batch_size, env.grad_accum, env.max_length = 32, 1, 128
    else:
        env.batch_size, env.grad_accum, env.max_length = 64, 1, 128
    return env


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing data file: {path}\nDid you unzip the whole folder?")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ClasspathDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, label_to_id: dict[str, int], max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tok(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        encoded = {k: torch.tensor(v) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.label_to_id[row["label"]])
        return encoded


class SpanDataset(Dataset):
    """Character spans -> BIO tags aligned to word-piece offsets."""

    def __init__(self, rows: list[dict], tokenizer, tag_to_id: dict[str, int], max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.tag_to_id = tag_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        text = row["text"]
        encoded = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        labels = align_bio(offsets, row["entities"], self.tag_to_id)

        out = {k: torch.tensor(v) for k, v in encoded.items()}
        out["labels"] = torch.tensor(labels)
        return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def classification_metrics(eval_pred) -> dict[str, float]:
    """Trainer adapter around the tested implementation."""
    logits, labels = eval_pred
    return _classification_metrics(logits, labels)


def make_token_metrics(id_to_tag: dict[int, str]):
    """Trainer adapter around the tested implementation."""

    def compute(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        return _token_metrics(logits, labels, id_to_tag)

    return compute


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def build_args(env: Env, out_dir: Path, epochs: float, lr: float) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(out_dir / "_checkpoints"),
        per_device_train_batch_size=env.batch_size,
        per_device_eval_batch_size=max(env.batch_size, 16),
        gradient_accumulation_steps=env.grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=1,
        seed=SEED,
        fp16=env.fp16,
        report_to=[],
        dataloader_num_workers=0,  # Windows-safe
    )
    # transformers renamed this argument; support both so the script does not
    # break on whichever version pip resolves.
    try:
        return TrainingArguments(eval_strategy="epoch", save_strategy="no", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", save_strategy="no", **kwargs)


def train_classpath(env: Env, model_name: str, epochs: float) -> dict:
    print("\n" + "=" * 72)
    print("TASK 1/2 - classpath classifier")
    print("=" * 72)

    train_rows = read_jsonl(DATA / "classpath_train.jsonl")
    val_rows = read_jsonl(DATA / "classpath_val.jsonl")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    labels = meta["classpath_labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}

    print(f"  train {len(train_rows):,} - val {len(val_rows):,} - {len(labels)} classes")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: label for label, i in label_to_id.items()},
        label2id=label_to_id,
    )

    out_dir = MODELS / "classpath"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=build_args(env, out_dir, epochs, lr=3e-5),
        train_dataset=ClasspathDataset(train_rows, tokenizer, label_to_id, env.max_length),
        eval_dataset=ClasspathDataset(val_rows, tokenizer, label_to_id, env.max_length),
        compute_metrics=classification_metrics,
    )
    started = time.perf_counter()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  accuracy  {metrics.get('eval_accuracy', 0):.4f}")
    print(f"  macro F1  {metrics.get('eval_macro_f1', 0):.4f}")
    print(f"  trained in {elapsed/60:.1f} min - saved to {out_dir}")
    return {
        "task": "classpath",
        "model": model_name,
        "classes": len(labels),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "accuracy": round(float(metrics.get("eval_accuracy", 0)), 4),
        "macro_f1": round(float(metrics.get("eval_macro_f1", 0)), 4),
        "minutes": round(elapsed / 60, 2),
    }


def train_spans(env: Env, model_name: str, epochs: float) -> dict:
    print("\n" + "=" * 72)
    print("TASK 2/2 - attribute span tagger")
    print("=" * 72)

    train_rows = read_jsonl(DATA / "spans_train.jsonl")
    val_rows = read_jsonl(DATA / "spans_val.jsonl")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))

    tags = build_tag_list(meta["span_labels"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    print(f"  train {len(train_rows):,} - val {len(val_rows):,} - {len(tags)} BIO tags")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(
            "The span tagger needs a fast tokenizer for character offsets.\n"
            f"{model_name} did not provide one -- use --model distilbert-base-uncased."
        )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(tags), id2label=id_to_tag, label2id=tag_to_id
    )

    out_dir = MODELS / "spans"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=build_args(env, out_dir, epochs, lr=5e-5),
        train_dataset=SpanDataset(train_rows, tokenizer, tag_to_id, env.max_length),
        eval_dataset=SpanDataset(val_rows, tokenizer, tag_to_id, env.max_length),
        compute_metrics=make_token_metrics(id_to_tag),
    )
    started = time.perf_counter()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "tags.json").write_text(
        json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  token accuracy    {metrics.get('eval_token_accuracy', 0):.4f}")
    print(f"  entity precision  {metrics.get('eval_entity_precision', 0):.4f}")
    print(f"  entity recall     {metrics.get('eval_entity_recall', 0):.4f}")
    print(f"  entity F1         {metrics.get('eval_entity_f1', 0):.4f}")
    print(f"  trained in {elapsed/60:.1f} min - saved to {out_dir}")
    return {
        "task": "spans",
        "model": model_name,
        "tags": len(tags),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "token_accuracy": round(float(metrics.get("eval_token_accuracy", 0)), 4),
        "entity_precision": round(float(metrics.get("eval_entity_precision", 0)), 4),
        "entity_recall": round(float(metrics.get("eval_entity_recall", 0)), 4),
        "entity_f1": round(float(metrics.get("eval_entity_f1", 0)), 4),
        "minutes": round(elapsed / 60, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the GlassBox local models")
    ap.add_argument("--task", choices=["both", "classpath", "spans"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF backbone (default {DEFAULT_MODEL}; "
                         f"stronger: {STRONGER_MODEL})")
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=0, help="override auto-detection")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    env = Env(device="cpu", batch_size=8, grad_accum=2) if args.cpu else detect_env()
    if args.batch_size:
        env.batch_size = args.batch_size

    print("=" * 72)
    print("GlassBox - local model training")
    print("=" * 72)
    print(f"  python      {platform.python_version()} on {platform.system()}")
    print(f"  torch       {torch.__version__}")
    print(f"  hardware    {env.describe()}")
    print(f"  backbone    {args.model}")
    print(f"  epochs      {args.epochs}")
    if env.device == "cpu":
        print("\n  NOTE: no CUDA GPU detected. Training will still complete but "
              "will take roughly 20-40 minutes instead of 3-8.")

    MODELS.mkdir(parents=True, exist_ok=True)
    results = []
    if args.task in {"both", "classpath"}:
        results.append(train_classpath(env, args.model, args.epochs))
    if args.task in {"both", "spans"}:
        results.append(train_spans(env, args.model, args.epochs))

    report = {
        "hardware": env.describe(),
        "backbone": args.model,
        "epochs": args.epochs,
        "torch": torch.__version__,
        "results": results,
    }
    (MODELS / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    for result in results:
        headline = result.get("macro_f1", result.get("entity_f1", 0))
        print(f"  {result['task']:<12} F1 {headline:.4f}  ({result['minutes']:.1f} min)")
    print(f"\n  models/         the trained weights -- send this folder back")
    print(f"  models/metrics.json  the scores -- these go on the slide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
