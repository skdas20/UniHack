"""One-file model eval for GlassBox.

Run this and nothing else. It scores the two local models on the held-out
val sets and prints the numbers you quote (classpath macro F1, span entity F1).

    python eval_models.py              # eval if models exist, otherwise train then eval
    python eval_models.py --eval-only  # eval saved models; do not train
    python eval_models.py --retrain    # train from scratch, then eval

You already have a venv at the *project root*, not inside this folder:

    cd ..
    .\\.venv\\Scripts\\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers tokenizers numpy
    cd training
    python eval_models.py
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"
SEED = 20260823
DEFAULT_MODEL = "distilbert-base-uncased"
IGNORE = -100


def _die_missing_packages(names: list[str]) -> None:
    root = HERE.parent
    activate = root / ".venv" / "Scripts" / "Activate.ps1"
    print("Missing packages:", ", ".join(names))
    print()
    print("training\\.venv does not exist. Use the project-root venv:")
    print()
    print(f'  cd "{root}"')
    if activate.exists():
        print(r"  .\.venv\Scripts\Activate.ps1")
    else:
        print(r"  python -m venv .venv")
        print(r"  .\.venv\Scripts\Activate.ps1")
    print("  python -m pip install --upgrade pip")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu130")
    print("  pip install transformers tokenizers numpy accelerate")
    print(f'  cd "{HERE}"')
    print("  python eval_models.py")
    print()
    print("If nvidia-smi shows CUDA 12.x, try cu128 instead of cu130.")
    raise SystemExit(1)


try:
    import numpy as np
except ImportError:
    _die_missing_packages(["numpy"])

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    _die_missing_packages(["torch"])

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError:
    _die_missing_packages(["transformers"])

try:
    import accelerate  # noqa: F401
except ImportError:
    _die_missing_packages(["accelerate"])


# --------------------------------------------------------------------------
# metrics (inlined so this file does not depend on tagging.py)
# --------------------------------------------------------------------------


def build_tag_list(span_labels: list[str]) -> list[str]:
    tags = ["O"]
    for label in span_labels:
        tags.append(f"B-{label}")
        tags.append(f"I-{label}")
    return tags


def align_bio(offsets, entities, tag_to_id: dict[str, int]) -> list[int]:
    outside = tag_to_id["O"]
    labels: list[int] = []
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE)
            continue
        tag_id = outside
        for entity in entities:
            e_start, e_end = entity["start"], entity["end"]
            if start >= e_start and end <= e_end:
                prefix = "B" if start == e_start else "I"
                tag_id = tag_to_id.get(f"{prefix}-{entity['label']}", outside)
                break
        labels.append(tag_id)
    return labels


def decode_spans(tag_ids, valid, id_to_tag: dict[int, str]) -> set[tuple[int, int, str]]:
    out: set[tuple[int, int, str]] = set()
    start: int | None = None
    current: str | None = None
    n = len(list(tag_ids))
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
        out.add((start, n, current))
    return out


def classification_metrics_np(logits, labels) -> dict[str, float]:
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
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def token_metrics_np(logits, labels, id_to_tag: dict[int, str]) -> dict[str, float]:
    logits, labels = np.asarray(logits), np.asarray(labels)
    preds = np.argmax(logits, axis=-1)
    mask = labels != IGNORE
    flat_true, flat_pred = labels[mask], preds[mask]
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


def hf_classification_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    return classification_metrics_np(logits, labels)


def make_hf_token_metrics(id_to_tag: dict[int, str]):
    def compute(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        return token_metrics_np(logits, labels, id_to_tag)

    return compute


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing data file: {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ClasspathDataset(Dataset):
    def __init__(self, rows, tokenizer, label_to_id, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
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
    def __init__(self, rows, tokenizer, tag_to_id, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.tag_to_id = tag_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tok(
            row["text"],
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
# hardware
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
        return f"CPU only - batch {self.batch_size}x{self.grad_accum}"


def detect_env() -> Env:
    if not torch.cuda.is_available():
        return Env(device="cpu", batch_size=8, grad_accum=2, fp16=False)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / (1024**3)
    env = Env(device="cuda", gpu_name=props.name, vram_gb=vram, fp16=True)
    if vram < 5:
        env.batch_size, env.grad_accum, env.max_length = 8, 4, 80
    elif vram < 9:
        env.batch_size, env.grad_accum, env.max_length = 16, 2, 96
    elif vram < 17:
        env.batch_size, env.grad_accum, env.max_length = 32, 1, 128
    else:
        env.batch_size, env.grad_accum, env.max_length = 64, 1, 128
    return env


def build_args(env: Env, out_dir: Path, epochs: float, lr: float, optimizer_steps: int = 0) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(out_dir / "_checkpoints"),
        per_device_train_batch_size=env.batch_size,
        per_device_eval_batch_size=max(env.batch_size, 16),
        gradient_accumulation_steps=env.grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=1,
        seed=SEED,
        fp16=env.fp16 and env.device == "cuda",
        report_to=[],
        dataloader_num_workers=0,
    )
    if env.device == "cpu":
        kwargs["use_cpu"] = True
    # transformers <5 takes warmup_ratio (a fraction of total steps); v5
    # removed it, so compute the same 10% warmup in steps ourselves.
    try:
        return TrainingArguments(
            eval_strategy="epoch", save_strategy="no", warmup_ratio=0.1, **kwargs
        )
    except TypeError:
        pass
    warmup_steps = max(1, int(0.1 * max(optimizer_steps, 1)))
    try:
        return TrainingArguments(
            eval_strategy="epoch", save_strategy="no", warmup_steps=warmup_steps, **kwargs
        )
    except TypeError:
        kwargs.pop("use_cpu", None)
        return TrainingArguments(
            evaluation_strategy="epoch", save_strategy="no", warmup_steps=warmup_steps, **kwargs
        )


def model_ready(task: str) -> bool:
    folder = MODELS / task
    return (folder / "config.json").exists()


# --------------------------------------------------------------------------
# classpath
# --------------------------------------------------------------------------


def run_classpath(env: Env, model_name: str, epochs: float, train: bool) -> dict:
    print("\n" + "=" * 72)
    print("CLASSPATH classifier — val eval")
    print("=" * 72)

    train_rows = read_jsonl(DATA / "classpath_train.jsonl")
    val_rows = read_jsonl(DATA / "classpath_val.jsonl")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    labels = meta["classpath_labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    out_dir = MODELS / "classpath"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  train {len(train_rows):,}  val {len(val_rows):,}  classes {len(labels)}")

    if train:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label={i: label for label, i in label_to_id.items()},
            label2id=label_to_id,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(out_dir))

    steps_per_epoch = -(-len(train_rows) // (env.batch_size * env.grad_accum))
    trainer = Trainer(
        model=model,
        args=build_args(env, out_dir, epochs, lr=3e-5, optimizer_steps=int(steps_per_epoch * epochs)),
        train_dataset=ClasspathDataset(train_rows, tokenizer, label_to_id, env.max_length),
        eval_dataset=ClasspathDataset(val_rows, tokenizer, label_to_id, env.max_length),
        compute_metrics=hf_classification_metrics,
    )

    started = time.perf_counter()
    if train:
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        (out_dir / "labels.json").write_text(
            json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    metrics = trainer.evaluate()
    elapsed = time.perf_counter() - started

    result = {
        "task": "classpath",
        "model": model_name,
        "trained_this_run": train,
        "classes": len(labels),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "accuracy": round(float(metrics.get("eval_accuracy", 0)), 4),
        "macro_f1": round(float(metrics.get("eval_macro_f1", 0)), 4),
        "minutes": round(elapsed / 60, 2),
    }
    print(f"\n  accuracy   {result['accuracy']:.4f}")
    print(f"  macro F1   {result['macro_f1']:.4f}   ← quote this")
    return result


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------


def run_spans(env: Env, model_name: str, epochs: float, train: bool) -> dict:
    print("\n" + "=" * 72)
    print("SPANS attribute tagger — val eval")
    print("=" * 72)

    train_rows = read_jsonl(DATA / "spans_train.jsonl")
    val_rows = read_jsonl(DATA / "spans_val.jsonl")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    tags = build_tag_list(meta["span_labels"])
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}
    out_dir = MODELS / "spans"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  train {len(train_rows):,}  val {len(val_rows):,}  BIO tags {len(tags)}")

    if train:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not tokenizer.is_fast:
            raise SystemExit("Span tagger needs a fast tokenizer. Use distilbert-base-uncased.")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(tags), id2label=id_to_tag, label2id=tag_to_id
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(str(out_dir))

    steps_per_epoch = -(-len(train_rows) // (env.batch_size * env.grad_accum))
    trainer = Trainer(
        model=model,
        args=build_args(env, out_dir, epochs, lr=5e-5, optimizer_steps=int(steps_per_epoch * epochs)),
        train_dataset=SpanDataset(train_rows, tokenizer, tag_to_id, env.max_length),
        eval_dataset=SpanDataset(val_rows, tokenizer, tag_to_id, env.max_length),
        compute_metrics=make_hf_token_metrics(id_to_tag),
    )

    started = time.perf_counter()
    if train:
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        (out_dir / "tags.json").write_text(
            json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    metrics = trainer.evaluate()
    elapsed = time.perf_counter() - started

    result = {
        "task": "spans",
        "model": model_name,
        "trained_this_run": train,
        "tags": len(tags),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "token_accuracy": round(float(metrics.get("eval_token_accuracy", 0)), 4),
        "entity_precision": round(float(metrics.get("eval_entity_precision", 0)), 4),
        "entity_recall": round(float(metrics.get("eval_entity_recall", 0)), 4),
        "entity_f1": round(float(metrics.get("eval_entity_f1", 0)), 4),
        "minutes": round(elapsed / 60, 2),
    }
    print(f"\n  token accuracy     {result['token_accuracy']:.4f}   (easy to inflate — ignore)")
    print(f"  entity precision   {result['entity_precision']:.4f}")
    print(f"  entity recall      {result['entity_recall']:.4f}")
    print(f"  entity F1          {result['entity_f1']:.4f}   ← quote this")
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def decide_train(task: str, eval_only: bool, retrain: bool) -> bool:
    ready = model_ready(task)
    if eval_only:
        if not ready:
            raise SystemExit(
                f"No saved {task} model at {MODELS / task}\n"
                f"Run without --eval-only so this script can train it first."
            )
        return False
    if retrain:
        return True
    if ready:
        print(f"  {task}: found {MODELS / task} — eval only")
        return False
    print(f"  {task}: no saved model — will train, then eval")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Train (if needed) and eval GlassBox local models")
    ap.add_argument("--task", choices=["both", "classpath", "spans"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="never train; fail if models are missing")
    ap.add_argument("--retrain", action="store_true", help="ignore saved models and train from scratch")
    args = ap.parse_args()

    if not (DATA / "meta.json").exists():
        raise SystemExit(f"Cannot find {DATA / 'meta.json'}. Run this from the training folder.")

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    env = Env(device="cpu", batch_size=8, grad_accum=2) if args.cpu else detect_env()
    if args.batch_size:
        env.batch_size = args.batch_size
    if not args.cpu and env.device != "cuda":
        print("ERROR: torch cannot see a CUDA GPU (this install is CPU-only).")
        print("Your machine has an NVIDIA GPU. Replace torch with a CUDA build:")
        print()
        print("  python -m pip uninstall -y torch")
        print("  python -m pip install torch --index-url https://download.pytorch.org/whl/cu130")
        print()
        print("Then re-run: python eval_models.py --retrain")
        print("Pass --cpu only if you really want a CPU run.")
        raise SystemExit(1)

    print("=" * 72)
    print("GlassBox — proper model eval")
    print("=" * 72)
    print(f"  python      {platform.python_version()} on {platform.system()}")
    print(f"  torch       {torch.__version__}")
    print(f"  hardware    {env.describe()}")
    print(f"  backbone    {args.model}")
    print(f"  val data    {DATA}")
    if env.device == "cpu":
        print("\n  NOTE: no CUDA GPU. Training (if needed) is 20-40 min; eval-only is a few minutes.")

    MODELS.mkdir(parents=True, exist_ok=True)
    results = []

    if args.task in {"both", "classpath"}:
        train = decide_train("classpath", args.eval_only, args.retrain)
        results.append(run_classpath(env, args.model, args.epochs, train))
    if args.task in {"both", "spans"}:
        train = decide_train("spans", args.eval_only, args.retrain)
        results.append(run_spans(env, args.model, args.epochs, train))

    report = {
        "hardware": env.describe(),
        "backbone": args.model,
        "epochs": args.epochs,
        "torch": torch.__version__,
        "max_length": env.max_length,
        "results": results,
    }
    out = MODELS / "metrics.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("QUOTE THESE")
    print("=" * 72)
    for result in results:
        if result["task"] == "classpath":
            print(f"  classpath   accuracy {result['accuracy']:.4f}   macro F1 {result['macro_f1']:.4f}")
        else:
            print(
                f"  spans       entity F1 {result['entity_f1']:.4f}   "
                f"(P {result['entity_precision']:.4f}  R {result['entity_recall']:.4f})"
            )
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
