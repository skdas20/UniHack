# GlassBox

**Six columns in, 252 out — and every one of them can show you its evidence.**

Built for **UniHack 2026** (Unilog × Hack2skill) · challenge: *AI-Powered
Product Intelligence for Industrial Commerce* · team **SKavengers**.

```bash
git clone https://github.com/skdas20/UniHack.git && cd UniHack
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python run.py                                      # 1000 rows, ~7 seconds
python -m streamlit run app/app.py                 # the demo
```

No API key. No network. No model download. That is a design requirement, not a
convenience: the submission brief says the prototype must be *"dynamic and
capable of processing the evaluation test dataset during assessment"*, so the
core path cannot depend on a secret being present on a machine we never see.

---

## The problem

Unilog builds product content for industrial distributors — the titles,
descriptions, attributes and images that let a contractor find the right part
online. The data distributors hand over is barely usable. Here is one row of the
supplied dataset, complete:

```
Mfg_Part_Num : 49-94-1940
Part_Desc    : 49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc
E1_Brand     : -- Unbranded --            <- placeholder, means empty
Unilog_Brand : -- No Unilog Brand --      <- placeholder, means empty
DIB_Brand    : -- No DIB Brand --         <- placeholder, means empty
Part_Manuf   : Milwaukee Accessory (4031) <- a distributor account, not a brand
```

From that, the delivery format requires **252 columns**: brand with the correct
registered mark, manufacturer legal entity, two parallel category hierarchies,
fifty attribute triples, twenty feature bullets, asset filenames — and the same
product rewritten **five times at five different lengths and casings** for the
till receipt, the mobile app, the search page, the product page and the
marketing block.

Measured over the supplied 1,000 rows: 799 carry `-- Unbranded --`, **all 1,000**
carry `-- No Unilog Brand --`, and 755 carry `-- No DIB Brand --`.

---

## What we found that changed the design

Three things, all documented with evidence in
[`docs/DERIVED_RULES.md`](docs/DERIVED_RULES.md).

### 1. The reference files the challenge is built around are not published

The Solution Guide describes seven reference workbooks and calls one of them
*"the most important file in the pack"* — a 27,000-row approved brand list, a
161,000-row cross-category List of Values, a 500-entry UOM master, and a 200-row
labelled ground-truth workbook. **None of them are downloadable.** The Resources
tab ships the Solution Guide, the 1,000-row input, and a header sheet containing
two worked rows. The portal says so quietly:

> *"The relevant information from these references is already represented within
> the columns of the provided datasets."*

We took that literally and made the engine **induce its own vocabularies from
the corpus**. That turns a missing-file problem into the strongest property of
the submission: a pipeline needing a hand-curated 27,000-row dictionary can only
process catalogues someone already curated. One that derives its lexicon from
the data in front of it can be pointed at a new distributor's export on day one.

### 2. The gold rows' key values are not in their inputs

Gold row 1's entire input is `PDSH4816AF Dishwasher SS - Display Only`. Its
output asserts `120 V`, `15 A`, `47 dBA`, `50-1/4 in Depth With Door Open` and
ENERGY STAR certification. **None of that is in the input** — it was enriched
from the manufacturer's own site, which is why the row also carries an
`MFR URL`.

No amount of AI recovers those numbers from six columns of distributor data. So
values are separated into two tiers that never mix: **extracted** (present in
the input, with a character span proving it) and **proposed** (a model's
knowledge of the part, labelled as such, scored lower, and always routed to
review). The Solution Guide is blunt about why this matters:

> *"A fluent description made of invented values scores zero."*

### 3. The attribute block is a contract, and the channels need a solver

Both gold rows emit the **same 15 attribute labels in the same order, including
labels whose value is blank**. Attribute output is a per-category contract, not
free extraction — a pipeline emitting "whatever it found" produces a
structurally wrong row even when every value is correct.

And `MOBILE_DESC` has a two-sided 60–80 character window. Gold row 1 lands at 75
with its manufacturer included; gold row 2 *drops* its manufacturer and *appends*
a mounting type to climb back over 60. That is a search with drop-and-append
moves over an ordered candidate list — so we implemented it as one, and it can
show you which candidates it dropped and why.

---

## Results

Measured, reproducible, and printed by `python run.py` on the supplied 1,000
rows. Nothing here is estimated.

### Accuracy against the published ground truth

| | |
|---|---|
| Gold description channels reproduced **character-for-character** | **9 / 9** |
| Longest exact match | **390 characters** (`LONG_DESC1`) |

The two fully-worked rows are the only labelled ground truth that exists.
[`tests/test_gold_channels.py`](tests/test_gold_channels.py) feeds each row's own
attribute values through the renderer and asserts all five channels come back
exactly as published — including the deliberate inconsistency that the same
value is written `120 V` in long copy and `120V` in the invoice line.

### Compliance, over every row

| Metric | Result |
|---|---|
| `INVOICE_DESC` within 40 characters | **100%** |
| `INVOICE_DESC` all-caps | **100%** |
| `MOBILE_DESC` inside the 60–80 window | **85.3%** |
| Attribute values inside the controlled vocabulary | **100%** |
| UOM spacing violations (`24in` instead of `24 in`) | **0** |
| Delivery-schema conformance (252 headers, exact order) | **100%** |
| Populated cells carrying provenance | **~100%** |

### Coverage and throughput

| Metric | Result |
|---|---|
| Rows classified into the taxonomy | **79.3%** |
| Brand resolved | **75.7%** |
| Attribute contract slots filled from text alone | **23.1%** |
| Auto-publishable / needs review / blocked | **618 / 297 / 85** |
| Throughput | **~330 rows/sec**, single-threaded CPU |
| Full 1,000-row run | **~4 seconds** |

We report the unflattering numbers too. 23.1% slot fill is low, and it is
honest: most inputs are ten words long and simply do not contain a voltage. The
optional model layer raises it; the audit trail keeps the two apart.

### The distilled local models

Two fine-tuned encoders (77-class classifier, 38-label span tagger) run on
CPU and attach to the pipeline with `python run.py --models`. Full evaluation,
per-class and per-label breakdowns, the near-duplicate audit behind the
classifier's 1.000 macro F1, and latency measurements live in
[`docs/MODEL_EVAL.md`](docs/MODEL_EVAL.md). The four quotable lines:

| Metric | Result |
|---|---|
| Classifier macro F1 (val; on the 17 novel val rows; teacher agreement) | **1.000 / 1.000 / 100.0%** |
| Span tagger entity F1 | **0.797** (P 0.839 / R 0.758) |
| Corpus classification, rules → + model layer | **79.3% → 81.3%**, LOV compliance still 100% |
| Full model layer on CPU | **21.9 ms/row → 2M SKUs overnight on one thread** |

---

## How it works

```
                     ┌──────────────────────────────────────┐
   1000 raw rows ───►│  PASS 1 · learn from the corpus      │
   (6 columns)       │  induce.py   brands, series, units   │
                     │  taxonomy.py distributor priors      │
                     └───────────────────┬──────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │  PASS 2 · enrich each row            │
                     │                                      │
                     │  textnorm  placeholders, shorthand   │
                     │  entity    6-rung brand ladder       │
                     │  taxonomy  Classpath + Dept/Class/Fine│
                     │  extract   fill the category contract│
                     │  units     fractions, UOM, 2 modes   │
                     │  render    5 channels + packer       │
                     │  confidence score from provenance    │
                     └───────────────────┬──────────────────┘
                                         ▼
        enriched.xlsx ·  provenance.jsonl ·  review_queue.csv ·  report.md
        252 columns      every cell's        triaged, with       compliance
                         evidence            reasons             metrics
```

### The modules

| File | Job |
|---|---|
| [`schema.py`](glassbox/schema.py) | Reads the 252 headers from the organisers' sheet **at runtime**; discovers repeating column families by pattern. A renamed or reordered column is structurally impossible. |
| [`provenance.py`](glassbox/provenance.py) | Every cell is a `Cell(value, Provenance)` carrying mechanism, rule, confidence and the source character span. |
| [`induce.py`](glassbox/induce.py) | Learns brands, series, product types and unknown unit spellings from the corpus using group-conditional TF-IDF. |
| [`units.py`](glassbox/units.py) | ~90 measurement families, exact 64th fractions, and the two rendering modes. |
| [`textnorm.py`](glassbox/textnorm.py) | Placeholders, noise markers, and abbreviation lexicons in both directions. |
| [`entity.py`](glassbox/entity.py) | Brand and manufacturer resolution through a six-rung evidence ladder that reports which rung it stopped on. |
| [`taxonomy.py`](glassbox/taxonomy.py) | 87 leaves; one classification drives both hierarchies so they cannot disagree. |
| [`attributes.py`](glassbox/attributes.py) | 17 ordered per-category contracts, 313 controlled values, trade dimension conventions. |
| [`extract.py`](glassbox/extract.py) | Fills contracts from text only, claiming spans so no two slots read the same characters. |
| [`render.py`](glassbox/render.py) | The five channels and the two-sided constraint solver. |
| [`confidence.py`](glassbox/confidence.py) | Scores from provenance, not model self-report. Triage and specific review reasons. |
| [`evaluate.py`](glassbox/evaluate.py) | Measures what is measurable; states plainly what is not. |
| [`enrich.py`](glassbox/enrich.py) | The **optional** hosted-model layer. Proposals only, validated, labelled, never required. |
| [`distill.py`](glassbox/distill.py) | The **optional** local-model layer. Serves the two distilled encoders on CPU; every span is validated and evidence-backed. |

### Details that took the most work to get right

- **`50k` means 5000 K.** Lighting writes colour temperature in hundreds.
- **`P150` is a grit**, not a model number.
- **`10-4 SO`** is 10 AWG, 4 conductors, SO jacket — three attributes in five
  unlabelled characters.
- **`14"x1/8"x1"` on a wheel** is diameter × thickness × arbor, in that order.
  Position carries meaning; swapping them silently corrupts the record.
- **`1nx6-20'`** — the `n` marks nominal sizing; the first two numbers are
  inches, the trailing one is feet.
- **`™` must survive normalisation.** Unicode NFKC decomposes it into the
  letters `TM`, turning `CleanBoost™` into `CleanBoostTM`. Caught by the gold
  comparison.
- **`Standard/Approvals` is ASCII-sorted**, which is why `cUL Listed` precedes
  `ENERGY STAR`. A case-insensitive "smarter" sort disagrees with the gold row.

### What we deliberately leave blank

`PART_NUMBER`, `SKU`, `List Price`, `UPC`/`EAN`/`GTIN`, `Country Of Origin` and
`UNSPSC` are emitted **empty**, each with a recorded reason naming the system of
record that would supply it. They are not functions of any input column — and
both gold rows leave `UNSPSC` and country of origin blank as well. Inventing an
8-digit part number is the single worst thing this pipeline could do.

---

## The optional layers

Both are genuinely optional. The core is complete without either.

**Hosted model proposals** ([`enrich.py`](glassbox/enrich.py)) — NVIDIA NIM,
OpenAI-compatible. Fills blank contract slots from a model's knowledge of the
specific part number. Every proposal is validated against the slot's controlled
vocabulary and plausibility bounds, tagged `Source.LLM`, weighted 0.50 against
0.95 for a text extraction, and routed to review. Enable with `--propose`; see
[`.env.example`](.env.example) for where the key goes.

**Distilled local models** ([`training/`](training/)) — we take every row the
rule engine classified confidently, treat its output as a silver label, and
fine-tune two small encoders: a 77-class product classifier and a 38-type
attribute span tagger. This is distillation: the model learns the pattern behind
the rules and generalises to phrasings no rule covers, which is how the
remaining 21% gets addressed. The models are served by
[`glassbox/distill.py`](glassbox/distill.py): `python run.py --models` turns
them on (CPU-only, no network), every value they produce is validated against
its slot's controlled vocabulary, tagged `local_model`, and routed to review.
It runs on CPU in milliseconds at zero marginal cost — a distributor with 2M
SKUs can process the catalogue inside their own VPC. Training and full
evaluation instructions: [`training/README.md`](training/README.md) and
[`docs/MODEL_EVAL.md`](docs/MODEL_EVAL.md).

---

## Repository layout

```
glassbox/           the engine (13 modules, no framework)
app/app.py          the Streamlit demo — the working prototype link
training/           self-contained GPU package: data + train.py + README
scripts/            build_vocab, make_training_data, smoke tests
tests/              gold-row exact-match verification
docs/
  DERIVED_RULES.md  every rule, with the evidence that produced it
  DEMO_VIDEO.md     3-minute shot list and narration
  HANDOVER.md       team onboarding — read this first if you're new
data/raw/           the supplied input and header sheet
outputs/            generated: enriched sheet, provenance, review queue, report
run.py              the CLI
```

## Reproducing every number in this README

```bash
python run.py                          # all run + compliance metrics
python run.py --models                 # same run + the distilled local models
python tests/test_gold_channels.py     # the 9/9 exact-match check
python scripts/build_vocab.py          # the induction report
python -m pytest tests -q              # the test suite
cd training && python eval_models.py --eval-only --cpu   # headline model metrics
cd training && python full_eval.py     # the full model evaluation
```

`outputs/report.md` is written on every run and contains the full metric set,
the source histogram, the triage split and the top review reasons.

## Team

**SKavengers** — UniHack 2026, Unilog × Hack2skill.
Challenge: *AI-Powered Product Intelligence for Industrial Commerce*.

## Licence

MIT. Note that the UniHack terms transfer IP in winning solutions to the
organisers on confirmation of an award.
