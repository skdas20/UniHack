# Handover — everything you need to know about this project

For anyone joining: teammates, the person running the GPU training, or whoever
records the demo video. Read this once, top to bottom. Twenty minutes.

---

## 1. The competition, in one screen

| | |
|---|---|
| **Event** | UniHack 2026 — hosted by **Unilog Corp** on Hack2skill |
| **Challenge** | AI-Powered Product Intelligence for Industrial Commerce (single challenge, every team solves the same one) |
| **Team** | SKavengers |
| **Prototype deadline** | **Sun 23 Aug 2026, 11:59 PM IST** |
| **Evaluation** | 24 Aug – 1 Sep · **Finale 4 Sep 2026** |
| **Prize pool** | ₹5,00,000 — Winner ₹2L · 1st runner-up ₹1.5L · 2nd ₹1L · 2 × Special ₹25k |
| **Also on offer** | Internships, PPOs and full-time roles at Unilog |

**The evaluation is a hard funnel:** `PPT → Demo Video → Prototype → GitHub`.
The judges review the deck first. Only if it passes do they watch the video.
Only if that passes do they open the prototype. Only then the repository. Every
gate is a place to be eliminated, so effort belongs at the front.

**Six things must be submitted:**

1. Prototype deck — PDF, ≤5 MB, **mandatory template** (in `docs/`)
2. A live, publicly reachable prototype URL
3. Public GitHub repository
4. 3-minute demo video
5. Solution overview text (≤2056 characters)
6. The enriched output file

One term worth knowing: **IP in winning solutions transfers to the organisers**
on confirmation of an award. Don't put anything in the repo you want to keep.

---

## 2. What Unilog actually needs

Unilog builds product content for industrial distributors — the titles,
descriptions, specs and images that let a contractor search "14 inch masonry cut
off wheel" and find the right part. Their bottleneck is that raw distributor data
is unusable and cleaning it is done by humans at enormous cost.

We are building the enrichment pipeline that sits in between. Given a messy row,
produce a complete, standardised, search-ready product record.

**In:** 1,000 rows, 6 columns. **Out:** 252 columns.

The one row that explains the whole job:

```
Mfg_Part_Num : 49-94-1940
Part_Desc    : 49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc
E1_Brand     : -- Unbranded --
Unilog_Brand : -- No Unilog Brand --
DIB_Brand    : -- No DIB Brand --
Part_Manuf   : Milwaukee Accessory (4031)
```

`Milw` is Milwaukee. `14"x1/8"x1"` is diameter, thickness, arbor — in that
order, and swapping them corrupts the record. Three of the six columns are
placeholder text meaning "empty". And the output needs the brand with its
registered mark, a three-level category path, fifty attribute triples, and the
same product written five ways at five lengths.

**The hard constraints that decide scoring:**

- `24 in`, never `24in`. One space, approved abbreviation only.
- `50.25"` becomes `50-1/4 in`. Trade buyers search fractions.
- Attribute values must come from a controlled vocabulary. Invented values score
  zero even if the prose reads beautifully.
- Placeholders are not data.
- `INVOICE_DESC` ≤ 40 characters, ALL CAPS. `MOBILE_DESC` 60–80 characters.

---

## 3. What we built, and the four things that make it different

**GlassBox.** Six columns in, 252 out, and every output cell can show you its
evidence.

Most submissions will be: pandas → fuzzy match → one large LLM prompt →
Streamlit before/after view. Four things separate ours.

### ① It learns its own vocabulary

The challenge describes seven reference workbooks — a 27,000-row brand list, a
161,000-row List of Values, a 500-entry UOM master, a 200-row labelled ground
truth. **None of them are published on the portal.** Teams following the guide
literally will discover this late and hardcode a few brands to make a demo work
— and the submission page has a trap waiting: *"should not be mocked,
hardcoded... capable of processing the evaluation test dataset."* The judges run
your code on data you have never seen.

We induce the brand lexicon, series names, product vocabulary and unit spellings
from the corpus itself. 51 brands learned, every one linked to attested evidence;
181 brand-shaped tokens held back for human review rather than guessed at. Pitch
line: *works on a new distributor's catalogue on day one, zero dictionary
maintenance.*

### ② Every cell carries its provenance

The challenge statement asks for *"traceable outputs"* and *"explainability"*.
Almost nobody will build it properly. We emit, for every cell: the mechanism
that produced it, the rule that fired, a confidence, and **the exact character
span of the source description it was read out of**. The UI highlights it. This
is the differentiator — hold on it in the video.

### ③ It refuses to invent

`PART_NUMBER`, `List Price`, `UPC`, `Country Of Origin`, `UNSPSC` are emitted
blank with a recorded reason naming the system of record that would supply them.
This sounds like a weakness and scores as rigour — the Solution Guide explicitly
says *"Noticing and reporting such gaps is a strength, not a failure."*

### ④ It runs anywhere, and distils to a local model

Zero-config, no API key, no network, ~240 rows/sec on a laptop CPU. Optional
layers attach on top but are never required — so it cannot fail on the judges'
machine. And we distil the rule engine into two small local models that run on
CPU at zero marginal cost, which is the enterprise-scale answer.

---

## 4. Where things stand

**Done and verified:**

- The engine — 13 modules, end to end, 1,000 rows in ~7 seconds
- **9/9 gold description channels reproduced character-for-character**
- 100% invoice-length, all-caps, LOV and schema compliance; 0 UOM errors
- 79.3% classified, 75.7% brand resolved, review triage working
- The Streamlit demo app, five views, dark themed
- Training data generated: 3,591 classifier examples, 636 span examples
- `training/` package ready to hand to the GPU machine
- Docs: `DERIVED_RULES.md`, `DEMO_VIDEO.md`, this file

**Not done, and deliberately not claimed:**

- The GPU training run never happened. The package is built, its data validated
  and its logic tested, but no model was trained — so no trained-model accuracy
  appears in the deck, the README, or the submission text. See
  `docs/SUBMISSION.md` for the answer to give if a judge asks.

**Still to do:**
- Record the 3-minute video → `docs/DEMO_VIDEO.md`
- Fill the mandatory deck → `docs/UniHack_Prototype_Template.pptx`
- Deploy the app to Hugging Face Spaces for the live URL
- Paste the six fields into the Hack2skill submission form

---

## 5. Running it

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

python run.py                          # enrich 1000 rows → outputs/
python -m streamlit run app/app.py     # the demo
python tests/test_gold_channels.py     # the 9/9 verification
python scripts/build_vocab.py          # what induction learned
```

**Please always use the virtual environment.** Nothing needs a GPU, a key or an
internet connection.

### Where the API key goes

There is **one** optional key: `NVIDIA_API_KEY`, for the model-proposal layer.

- Get it at **https://build.nvidia.com** → sign in → any model page → *Get API
  Key*. Free credits, no card. Looks like `nvapi-...`.
- **Locally:** `cp .env.example .env` and paste it in. `.env` is gitignored —
  never commit it.
- **On Hugging Face Spaces:** *Settings → Variables and secrets → New secret*,
  named `NVIDIA_API_KEY`. Never paste it into `app.py`.
- It stays off unless explicitly enabled: `python run.py --propose`, or the
  sidebar toggle (disabled until a key exists).

Training needs **no** key at all.

---

## 6. The deck (`docs/UniHack_Prototype_Template.pptx`)

Template is mandatory and has 15 slides. What goes on the ones that matter:

| Slide | Content |
|---|---|
| **Brief about solution** | "Six columns in, 252 out — and every one can show its evidence." One sentence on the induction engine, one on provenance. |
| **How does it enrich minimal info?** | The `49-94-1940` row → the enriched record. Show the actual before/after. |
| **How do you ensure accuracy and trust?** | The strongest slide we have: 9/9 gold channels exact · 100% LOV · confidence scored *from provenance* · triage + review queue · and we refuse to invent, with the blank-reason list. |
| **What makes it scalable?** | 240 rows/sec single-threaded · zero-config so it runs anywhere · vocabulary induction means a new manufacturer needs no dictionary work · distilled local model at zero marginal cost inside the customer's VPC. |
| **Opportunities / USP** | The four differentiators from section 3, in that order. Lead with provenance. |
| **Architecture** | The two-pass diagram in the README. |
| **Snapshots of the MVP** | Screenshots of the Enrich tab (with the *Read from* column visible), Compliance tab, Review queue. |
| **Links** | GitHub · HF Spaces URL · video URL. |

Two rules for the deck: **every number must be reproducible by `python run.py`**,
and put the provenance screenshot on the accuracy slide. That single image is
the whole argument.

---

## 7. What to say, and what not to

**Say:**
- "Every one of the 252 cells can show you the characters it came from."
- "The reference dictionaries aren't published, so the engine induces its own —
  which means it works on an uncurated catalogue."
- "It runs with no API key and no network, so it cannot fail on your machine."
- "We leave fields blank on purpose, and record why."
- "All five description channels of both published gold rows reproduce exactly."

**Don't say:**
- Any accuracy figure against "200 ground-truth rows" — that file does not
  exist and a judge from Unilog knows it.
- "AI-powered" as the headline. Most of this is deterministic, and that's the
  strength. Say *auditable*.
- A number that `python run.py` doesn't print.

---

## 8. Honest weaknesses, and the answers

Judges will find these. Better to have the answer ready.

| Weakness | The answer |
|---|---|
| Attribute slot fill is only 23% | The inputs are ten words long and genuinely don't contain a voltage. We fill what's there, mark the rest as contract slots, and the proposal layer closes the gap under review. Compare: inventing values to look complete. |
| 21% of rows unclassified | Every one is routed to review with a reason, never guessed. That's what the distilled model is for. |
| No true accuracy number | Because no labelled set was published. We measure exact-match on the two rows that were, plus label-free compliance on all 1,000. `score_against_gold()` is written and ready the moment a labelled file appears. |
| The taxonomy is hand-seeded | 87 leaves, chosen from what's actually in the data. The *vocabularies* are induced; the category skeleton is domain design, and it's declarative and reviewable in one file. |
| Faucets and fittings aren't covered | There is not one faucet or fitting in the 1,000 rows. The guide's advice to go deep there would produce a demo that can't process a single supplied row. |

---

## 9. Who's doing what

| Task | Notes |
|---|---|
| GPU training | `training/README.md` — self-contained, ~10 min of attention |
| Demo video | `docs/DEMO_VIDEO.md` — full shot list, don't improvise |
| Deck | Section 6 above + the mandatory template |
| HF Spaces deploy | `app/README.md` has the Space card; needs an HF account |
| Submission form | Six fields; **check the video link in a private window first** |

Deadline is **Sun 23 Aug, 11:59 PM IST**. The video and deck are the two gates
that decide whether anyone looks at the engine at all — weight the time
accordingly.
