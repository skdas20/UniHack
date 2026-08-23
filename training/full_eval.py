"""The full model evaluation: everything that goes on the results slide.

Complements ``eval_models.py`` (which produces the two headline numbers) with
the analysis a judge or a teammate will actually ask for:

* **near-duplicate audit** -- the training data is augmented, so a random
  val split contains paraphrase-twins of training rows. We measure each val
  example's similarity to its nearest training example and report the score
  on the subset that is *genuinely unseen*, which is the honest generalisation
  number;
* per-class F1 with support, worst classes, top confusion pairs;
* softmax-confidence calibration buckets (does 0.9 mean 90%?);
* per-label span F1, so "which attributes does the tagger know?" has an answer;
* CPU latency and throughput, the number the commercial story stands on;
* **rule-gap coverage**: of the rows the deterministic engine cannot classify,
  how many does the distilled classifier recover, at what confidence -- the
  end-to-end reason these models exist.

Run from ``training/``:

    python full_eval.py            # everything; writes models/full_eval.json
    python full_eval.py --quick    # skip the latency sweep

Reuses ``glassbox.distill.LocalModels`` for inference, so the eval exercises
the exact serving path the pipeline uses.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so glassbox.* imports work

from glassbox.distill import LocalModels  # noqa: E402

DATA = HERE / "data"
MODELS = HERE / "models"
SIM_BATCH = 32


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- similarity (the near-duplicate audit) ----------------------------------


def bow(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def nearest_similarity(val_bows: list[frozenset], train_bows: list[frozenset]) -> np.ndarray:
    best = np.zeros(len(val_bows))
    for i, v in enumerate(val_bows):
        best[i] = max((jaccard(v, t) for t in train_bows), default=0.0)
    return best


# --- metric helpers ----------------------------------------------------------


def per_class_counts(preds: list[str], golds: list[str]) -> dict[str, dict]:
    labels = sorted(set(golds))
    out: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, golds) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {
            "support": tp + fn,
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    return out


def confusion_pairs(preds: list[str], golds: list[str], top: int = 10) -> list[list]:
    counts = Counter((g, p) for p, g in zip(preds, golds) if p != g)
    return [[g, p, n] for (g, p), n in counts.most_common(top)]


def macro_f1(preds: list[str], golds: list[str]) -> float:
    per = per_class_counts(preds, golds)
    return float(np.mean([v["f1"] for v in per.values()])) if per else 0.0


def entity_f1_by_label(rows, hits_per_row) -> tuple[dict, dict]:
    """Per-label and overall entity P/R/F1 from character spans."""
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for row, hits in zip(rows, hits_per_row):
        gold = {(e["start"], e["end"], e["label"]) for e in row["entities"]}
        pred = {(h.start, h.end, h.label) for h in hits}
        for span in pred & gold:
            tp[span[2]] += 1
        for span in pred - gold:
            fp[span[2]] += 1
        for span in gold - pred:
            fn[span[2]] += 1

    def metrics(counter_tp, counter_fp, counter_fn, keys):
        out = {}
        for label in keys:
            t, f, n = counter_tp[label], counter_fp[label], counter_fn[label]
            p = t / (t + f) if t + f else 0.0
            r = t / (t + n) if t + n else 0.0
            out[label] = {
                "support": t + n,
                "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            }
        return out

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = metrics(tp, fp, fn, labels)
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P = T / (T + F) if T + F else 0.0
    R = T / (T + N) if T + N else 0.0
    overall = {
        "precision": round(P, 4),
        "recall": round(R, 4),
        "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
        "tp": T, "fp": F, "fn": N,
    }
    return per_label, overall


# --- sections ----------------------------------------------------------------


def eval_classpath(models: LocalModels, rows: list[dict]) -> dict:
    texts = [r["text"] for r in rows]
    golds = [r["label"] for r in rows]
    preds = models.classify_batch(texts, batch_size=SIM_BATCH)
    pred_labels = [p.classpath if p else "" for p in preds]
    probs = [p.prob if p else 0.0 for p in preds]
    correct = [int(pl == g) for pl, g in zip(pred_labels, golds)]

    per_class = per_class_counts(pred_labels, golds)
    by_support = sorted(per_class.items(), key=lambda kv: -kv[1]["support"])

    # near-duplicate audit
    train_rows = read_jsonl(DATA / "classpath_train.jsonl")
    sims = nearest_similarity([bow(t) for t in texts], [bow(r["text"]) for r in train_rows])
    strict_idx = [i for i, s in enumerate(sims) if s < 0.6]
    twin_idx = [i for i, s in enumerate(sims) if s >= 0.6]

    def subset_macro(idxs: list[int]) -> float:
        return macro_f1([pred_labels[i] for i in idxs], [golds[i] for i in idxs])

    # confidence calibration
    buckets = {}
    for lo, hi in ((0.0, 0.55), (0.55, 0.7), (0.7, 0.9), (0.9, 1.01)):
        idxs = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not idxs:
            continue
        buckets[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {
            "n": len(idxs),
            "accuracy": round(float(np.mean([correct[i] for i in idxs])), 4),
        }
    acc_above_70 = [correct[i] for i, p in enumerate(probs) if p >= 0.70]

    return {
        "val_examples": len(rows),
        "classes": len(per_class),
        "accuracy": round(float(np.mean(correct)), 4),
        "macro_f1": round(macro_f1(pred_labels, golds), 4),
        "near_duplicate_audit": {
            "method": "Jaccard over bag-of-words vs every training text",
            "val_with_twin_in_train": len(twin_idx),
            "val_strictly_novel": len(strict_idx),
            "macro_f1_on_near_duplicates": round(subset_macro(twin_idx), 4) if twin_idx else None,
            "macro_f1_on_strictly_novel": round(subset_macro(strict_idx), 4) if strict_idx else None,
            "mean_nearest_similarity": round(float(sims.mean()), 4),
        },
        "confidence_calibration": {
            "mean_prob_when_correct": round(float(np.mean([p for p, c in zip(probs, correct) if c])), 4),
            "mean_prob_when_wrong": round(float(np.mean([p for p, c in zip(probs, correct) if not c])), 4) if not all(correct) else None,
            "buckets": buckets,
            "accuracy_when_prob_ge_0.70": round(float(np.mean(acc_above_70)), 4) if acc_above_70 else None,
            "share_of_predictions_ge_0.70": round(len(acc_above_70) / len(rows), 4),
        },
        "worst_classes": [
            {"class": leaf(k), **v} for k, v in sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:10]
        ],
        "best_classes": [
            {"class": leaf(k), **v} for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]["f1"])[:5]
        ],
        "top_confusions": [
            {"gold": leaf(g), "predicted": leaf(p), "n": n}
            for g, p, n in confusion_pairs(pred_labels, golds)
        ],
        "largest_classes": [{"class": leaf(k), **v} for k, v in by_support[:8]],
    }


def leaf(classpath: str) -> str:
    return classpath.rsplit(">", 1)[-1]


def eval_spans(models: LocalModels, rows: list[dict]) -> dict:
    texts = [r["text"] for r in rows]
    hits = models.tag_spans_batch(texts, batch_size=SIM_BATCH)
    per_label, overall = entity_f1_by_label(rows, hits)
    by_support = sorted(per_label.items(), key=lambda kv: (-kv[1]["support"], kv[0]))
    span_lengths = [
        h.end - h.start for row_hits in hits for h in row_hits
    ]
    n_pred = sum(len(h) for h in hits)
    return {
        "val_examples": len(rows),
        "labels_seen": len(per_label),
        "overall": overall,
        "predicted_spans": n_pred,
        "mean_span_chars": round(float(np.mean(span_lengths)), 1) if span_lengths else 0.0,
        "best_labels": [
            {"label": k, **v} for k, v in sorted(per_label.items(), key=lambda kv: -kv[1]["f1"])[:8]
        ],
        "worst_labels": [
            {"label": k, **v}
            for k, v in sorted(per_label.items(), key=lambda kv: (kv[1]["f1"], -kv[1]["support"]))[:10]
        ],
        "largest_labels": [{"label": k, **v} for k, v in by_support[:10]],
    }


def eval_latency(models: LocalModels, cp_rows, sp_rows) -> dict:
    import torch

    cp_texts = [r["text"] for r in cp_rows]
    sp_texts = [r["text"] for r in sp_rows]

    def bench(fn, texts, batch_size, repeat=3) -> float:
        best = float("inf")
        for _ in range(repeat):
            started = time.perf_counter()
            fn(texts, batch_size)
            best = min(best, time.perf_counter() - started)
        return best

    out: dict = {"device": "cpu", "torch": torch.__version__}
    for bs in (1, 8, 32):
        secs = bench(lambda t, b: models.classify_batch(t, batch_size=b), cp_texts, bs)
        out[f"classify_batch{bs}_ms_per_row"] = round(1000 * secs / len(cp_texts), 2)
        out[f"classify_batch{bs}_rows_per_min"] = int(60 * len(cp_texts) / secs)
    for bs in (8, 32):
        secs = bench(lambda t, b: models.tag_spans_batch(t, batch_size=b), sp_texts, bs)
        out[f"spans_batch{bs}_ms_per_row"] = round(1000 * secs / len(sp_texts), 2)
        out[f"spans_batch{bs}_rows_per_min"] = int(60 * len(sp_texts) / secs)
    # the combined per-row cost of the full model layer as the pipeline uses it
    cls = out["classify_batch32_ms_per_row"]
    spn = out["spans_batch32_ms_per_row"]
    out["full_layer_ms_per_row"] = round(cls + spn, 2)
    out["full_layer_rows_per_min"] = int(60000 / (cls + spn))
    return out


def eval_rule_gap(models: LocalModels) -> dict:
    """Where the models earn their keep: the rows the rules cannot place."""
    from glassbox import textnorm as T
    from glassbox.schema import load_input
    from glassbox.taxonomy import GroupPrior, classify, prepare_text

    rows = load_input(HERE.parent / "data" / "raw" / "input_1000.csv")

    prior = GroupPrior()
    for row in rows:
        group = T.clean(row.get("Part_Manuf"))
        result = classify(prepare_text(row.desc, row.mpn), group=group)
        if result.ok and result.confidence >= 0.70:
            prior.observe(group, result.category.classpath, result.confidence)

    gaps: list[tuple[object, str]] = []
    confident: list[tuple[object, object]] = []
    for row in rows:
        group = T.clean(row.get("Part_Manuf"))
        result = classify(prepare_text(row.desc, row.mpn), group=group, prior=prior)
        text = prepare_text(row.desc, row.mpn)
        if not result.ok:
            gaps.append((row, text))
        elif result.confidence >= 0.70:
            confident.append((result, row))

    preds = models.classify_batch([t for _r, t in gaps], batch_size=SIM_BATCH) if gaps else []

    thresholds = (0.55, 0.70, 0.90)
    rescued = {
        f">={t}": sum(1 for p in preds if p is not None and p.prob >= t) for t in thresholds
    }

    # distillation sanity: on rows the rules *do* classify confidently, how
    # often does the model agree with its teacher?
    cap = 400  # enough for a tight confidence interval, cheap on CPU
    sample = confident[:cap]
    if sample:
        model_preds = models.classify_batch(
            [prepare_text(r.desc, r.mpn) for _res, r in sample], batch_size=SIM_BATCH
        )
        agreed = sum(
            1
            for (res, _r), mp in zip(sample, model_preds)
            if mp is not None and mp.classpath == res.category.classpath
        )
        disagreed = len(sample) - agreed

    return {
        "corpus_rows": len(rows),
        "rule_unclassified": len(gaps),
        "rule_classified_pct": round(100 * (len(rows) - len(gaps)) / len(rows), 1),
        "model_rescues_at_threshold": rescued,
        "model_rescued_pct_of_gap": {
            k: round(100 * v / max(len(gaps), 1), 1) for k, v in rescued.items()
        },
        "combined_classified_pct_at_0.55": round(
            100 * (len(rows) - len(gaps) + rescued[">=0.55"]) / len(rows), 1
        ),
        "teacher_agreement": {
            "sampled_confident_rows": len(sample),
            "agrees_with_rules": agreed,
            "disagrees": disagreed,
            "agreement_pct": round(100 * agreed / max(len(sample), 1), 1),
        },
        "example_rescues": [
            {
                "desc": row.desc[:70],
                "predicted": leaf(p.classpath),
                "prob": round(p.prob, 3),
            }
            for (row, _t), p in list(zip(gaps, preds))[:8]
            if p is not None and p.prob >= 0.55
        ],
    }


# --- main ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Full GlassBox model evaluation")
    ap.add_argument("--quick", action="store_true", help="skip the latency sweep")
    args = ap.parse_args()

    print("=" * 72)
    print("GlassBox — full model evaluation")
    print(f"  python {platform.python_version()} on {platform.system()} | CPU inference")
    print("=" * 72)

    models = LocalModels()
    if not models.available():
        raise SystemExit(f"trained models not found under {models.model_dir}")

    cp_val = read_jsonl(DATA / "classpath_val.jsonl")
    sp_val = read_jsonl(DATA / "spans_val.jsonl")

    print(f"\n[1/4] classpath classifier — {len(cp_val)} val examples …")
    started = time.perf_counter()
    cp = eval_classpath(models, cp_val)
    print(f"  accuracy {cp['accuracy']:.4f}  macro F1 {cp['macro_f1']:.4f}  ({time.perf_counter() - started:.1f}s)")
    audit = cp["near_duplicate_audit"]
    print(f"  near-duplicates of train rows: {audit['val_with_twin_in_train']} / {cp['val_examples']}")
    if audit["macro_f1_on_strictly_novel"] is not None:
        print(f"  macro F1 on strictly novel val rows: {audit['macro_f1_on_strictly_novel']:.4f}")

    print(f"\n[2/4] span tagger — {len(sp_val)} val examples …")
    started = time.perf_counter()
    sp = eval_spans(models, sp_val)
    print(f"  entity F1 {sp['overall']['f1']:.4f}  (P {sp['overall']['precision']:.4f}  R {sp['overall']['recall']:.4f})  ({time.perf_counter() - started:.1f}s)")

    print("\n[3/4] rule-gap coverage on the 1000-row corpus …")
    started = time.perf_counter()
    gap = eval_rule_gap(models)
    print(f"  rules alone classify {gap['rule_classified_pct']}% of the corpus")
    print(f"  + model layer (prob>=0.55): {gap['combined_classified_pct_at_0.55']}%")
    print(f"  teacher agreement on confident rows: {gap['teacher_agreement']['agreement_pct']}%  ({time.perf_counter() - started:.1f}s)")

    latency = None
    if not args.quick:
        print("\n[4/4] CPU latency sweep …")
        started = time.perf_counter()
        latency = eval_latency(models, cp_val, sp_val)
        print(f"  classify {latency['classify_batch32_ms_per_row']} ms/row | spans {latency['spans_batch32_ms_per_row']} ms/row (batch 32)")
        print(f"  full model layer: {latency['full_layer_ms_per_row']} ms/row -> {latency['full_layer_rows_per_min']:,} rows/min  ({time.perf_counter() - started:.1f}s)")

    report = {
        "classpath": cp,
        "spans": sp,
        "rule_gap": gap,
        "latency_cpu": latency,
    }
    out = MODELS / "full_eval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
