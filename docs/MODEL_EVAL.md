# Model evaluation — every number, and how to reproduce it

Written 22 August 2026, the day before submission. All numbers below were
measured on this machine (Python 3.11.9, torch 2.13.0 **CPU**, transformers
5.15.1) by the two scripts named at the bottom. Nothing is estimated.

## TL;DR — the slide numbers

| # | Metric | Value |
|---|--------|-------|
| 1 | Classpath classifier — accuracy / macro F1 (430 val rows, 77 classes) | **1.000 / 1.000** |
| 2 | …on the 17 val rows with no near-duplicate in training | **1.000** |
| 3 | …agreement with the rule engine on 400 confidently-classified rows | **100.0%** |
| 4 | Span tagger — entity F1 (76 val rows, 38 labels) | **0.797** (P 0.839 / R 0.758) |
| 5 | Corpus classification: rules alone → + model layer | **79.3% → 81.3%** |
| 6 | Attribute slot fill: rules alone → + model layer | 23.1% → 23.2% |
| 7 | LOV compliance with the model layer active | **100%** (unchanged) |
| 8 | Full model layer cost on CPU | **21.9 ms/row → ~2,700 rows/min** |
| 9 | 2M-SKU catalogue on one CPU thread | **~12 hours, in the customer's VPC** |

The two models are `distilbert-base-uncased` encoders fine-tuned on silver
labels distilled from the rule engine (3,161 classifier / 560 span training
examples, originally trained on an RTX 2050, 6 epochs, fp16 —
`training/models/metrics.json`). They now run **inside the pipeline**:
`python run.py --models` turns them on; without the flag (or without torch
installed) the engine behaves exactly as before.

## What was built today to make this real

Until today the two trained encoders existed only as artifacts in
`training/models/` — nothing loaded them. This session added:

- **`glassbox/distill.py`** — the serving layer. Loads both encoders lazily
  (torch is imported only on first inference), classifies descriptions into
  the 77-class taxonomy, and decodes the span tagger's BIO output back to
  *character offsets*, so every model-filled slot carries the same
  highlight-in-the-source evidence a regex extraction does.
- **A validation gate** (`model_fill_slot`): a model span is emitted only if
  its value passes the slot's controlled vocabulary (aliases canonicalised,
  `Wh` → `White`) or the units engine's plausibility bounds. A proposal that
  cannot be validated is dropped, not emitted — which is why LOV compliance
  stays at 100% with the model on.
- **Pipeline wiring** (`glassbox/pipeline.py`): a batched pre-pass classifies
  the rows no rule matches (accepting only softmax ≥ 0.55) and fills contract
  slots the rules left empty. Every model-produced cell carries
  `Source.LOCAL_MODEL`, is weighted 0.60 in the confidence engine (vs 0.95
  for an extraction), and routes the row to review.
- **`training/full_eval.py`** — the evaluation harness that produced this page.
- **Streamlit toggle** ("Distilled local models (CPU)") in the demo app.

## 1. Classifier: a perfect score that needed an asterisk — and survived it

The standard random split gives accuracy 1.000 and macro F1 1.000. Before
quoting that, we audited it: the training set is augmented (12 paraphrase
transformations per source row), and the val split was drawn after
augmentation — so **413 of the 430 val rows have a near-twin (Jaccard ≥ 0.6
over bag-of-words) in training**. A perfect score on twins proves memorisation,
not capability.

So we measured the residual: on the **17 genuinely novel val rows the model
still scores 1.000**, and more tellingly, on **400 rows the rule engine
classifies with high confidence, the model agrees with its teacher on 100.0%**
— the distillation learned the *pattern*, not the row.

Confidence is usable as a filter: 80.2% of val predictions carry softmax
≥ 0.70, and the model was correct in every confidence bucket measured —
including the 43 predictions below 0.55. (On this val set it is simply never
wrong; the buckets matter on production-like data, where the same
distribution governs which rows get rescued.)

## 2. Span tagger: honest per-label strengths and weaknesses

Overall entity F1 **0.797** (precision 0.839, recall 0.758 over 186 gold
spans). The spread by label is the interesting part:

| Knows cold (F1 = 1.00) | Struggles (F1 = 0) |
|---|---|
| Color Temperature (16), Wattage (16), Material (11), Pack Quantity (10), Edge Profile (10), Voltage Rating | Length (5), Mounting Type (5), Drive Type (4), Nominal Width (4), Amperage Rating (3) |

The pattern: exact-match surface values (colour temperature codes, wattages)
are learned perfectly; labels whose gold values are often *implied* rather
than printed (mounting type) or whose boundaries are fuzzy (nominal width in
a dimension run) lose to the rules — which is fine, because in the pipeline
the model only ever fills slots the rules left **empty**. It never competes
with them.

## 3. Where the models earn their keep: the 207 rows the rules cannot place

Rules alone classify 793/1,000 rows. On the remaining 207, the classifier
proposes at varying confidence:

| Accept threshold | Rows rescued | Corpus classification |
|---|---|---|
| ≥ 0.50 | 25 (12.1% of gap) | 81.8% |
| **≥ 0.55 (shipped default)** | **20 (9.7%)** | **81.3%** |
| ≥ 0.70 | 13 (6.3%) | 80.6% |
| ≥ 0.90 | 0 | 79.3% |

Eyeballing the 20 shipped rescues: the high-confidence ones are right
(`DCD799B Dewalt Drill` → Drills & Drivers, 0.84; `DW088CG Dewalt … Cross Line
Laser` → Layout & Measuring, 0.76; the Makita angle-impact cluster, 0.77), a
minority are nearest-family guesses (`Recip Saw` → Circular Saws, 0.66-0.77).
No ground truth exists for these rows — that is what made them gaps — so every
rescue is flagged `local_model` and **routed to review with the reason spelled
out**. A wrong rescue cannot auto-publish; a human clears the queue in
seconds. The end-to-end effect on the full corpus:

| | Rules only | + Model layer |
|---|---|---|
| Classified | 79.3% | **81.3%** |
| Attribute slot fill | 23.14% | 23.2% |
| LOV compliance | 100% | **100%** |
| Schema conformance | 100% | 100% |
| UOM spacing violations | 0 | 0 |
| Auto-publish / review / blocked | 618 / 297 / 85 | 618 / 300 / 82 |
| Wall time (1,000 rows, CPU) | 3.9 s | 24.6 s |

The 3 blocked rows that moved to review are rows the model rescued: exactly
the designed behaviour — they were unclassifiable before, now they are
proposals a human can approve.

## 4. Latency: the commercial number

Measured on CPU, batch 32, best of 3 repeats over the val sets:

| Model | ms/row | rows/min |
|---|---|---|
| Classifier (77 classes) | 11.8 | 5,075 |
| Span tagger (38 labels) | 10.0 | 5,979 |
| **Both (as the pipeline runs them)** | **21.9** | **2,745** |

A 2-million-SKU catalogue processes in **~12 hours on one CPU thread**, inside
the distributor's own VPC, at zero marginal cost. Per-row API pricing does not
compete with this at any volume. (Model load is a one-off ~2 s; the rules-only
core stays at ~240-330 rows/s.)

## 5. Honest limitations

- Silver labels. Both models are distilled from rule output, not human
  annotation. They can match the rules, generalise their pattern, and cover
  phrasings no rule anticipated — they cannot exceed the rules' conceptual
  ceiling, and where the rules were systematically wrong they would learn the
  wrong pattern faithfully.
- The 17-novel-row subset is small. We report it because 1.000-on-twins alone
  would be self-flattery; 17 examples is direction, not proof. The teacher
  agreement figure (400 rows) is the stronger generalisation evidence.
- Span slot-fill contribution today is modest (+0.06 points of slot fill):
  most high-confidence spans duplicate what the rules already extracted. The
  layer's present value is classification rescue + evidence-backed proposals;
  its future value is replacing hand-written rules as the corpus grows.
- The rescue-quality estimate is an eyeball, not a measured accuracy — no
  ground truth exists for gap rows. The review routing is the mitigation.

## Reproducing every number

```bash
# headline metrics (writes training/models/metrics.json)
cd training && python eval_models.py --eval-only --cpu

# everything on this page (writes training/models/full_eval.json)
python full_eval.py

# end-to-end comparison runs
cd ..
python run.py --no-xlsx --out outputs/_baseline          # rules only, 3.9 s
python run.py --models --no-xlsx --out outputs/_withmodels  # + local models, 24.6 s
python tests/test_gold_channels.py                        # 9/9 exact gold match, still green
```
