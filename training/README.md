# Training the GlassBox models — read this first

Hi, and thanks for running this. You have the GPU; nobody else on the team does.

**What you need to do:** make a virtual environment, install two things, run one
script, and send back one folder. Everything else — the dataset, the labels, the
preprocessing, the tokenisation, the hyperparameters — is already done and sits
in this folder. You should not need to write a line of code or understand the
project to make this work.

**Time:** about 10 minutes of your attention, plus 5–20 minutes of the GPU
grinding on its own.

**Deadline:** the submission closes **Sunday 23 August 2026, 11:59 PM IST**, so
we need `models/` back before then. Earlier is much better — the results go on a
slide and into the demo video.

---

## 1. Make a virtual environment

Please use a venv. It keeps this off your system Python and you can delete the
whole folder afterwards.

**Windows (PowerShell):**
```powershell
cd training
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS / WSL:**
```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Python 3.10, 3.11 or 3.12 all work. 3.13 may fight you on torch wheels — if it
does, use 3.11.

## 2. Install PyTorch, then the rest

**Install torch first, on its own.** If you let the requirements file pull it in,
pip usually resolves a CPU-only build and your GPU sits idle.

Pick the line matching your CUDA. `nvidia-smi` prints your driver's CUDA version
in the top-right corner; choose the closest one at or below it.

```bash
# CUDA 12.1 (most common on current drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then:
```bash
pip install -r requirements.txt
```

**Verify the GPU is visible before training.** This one line saves the most time:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

You want `True` and your card's name. If it says `False`, the CPU-only torch got
installed — `pip uninstall torch` and redo step 2 with the index URL.

## 3. Two 10-second checks first

Both need nothing but the standard library and numpy, and between them they rule
out the two ways this usually wastes an evening:

```bash
python validate_data.py    # did the data arrive intact and consistent?
python test_tagging.py     # is the label-alignment logic correct?
```

You want `Data is consistent.` and `all passed`. If either complains, stop and
send me the output — don't try to fix it.

(These exist because I don't have a GPU and couldn't run `train.py` end to end
before handing it over. The label alignment and the metrics are the parts most
likely to be silently wrong, so they're covered by tests that run anywhere,
including a pass over all 560 real span examples.)

## 4. Run it

```bash
python train.py
```

That's the whole thing. The script:

- detects your GPU and sizes the batch and precision to your VRAM (a 6 GB card
  is the target, and is comfortably enough);
- trains **two** models, one after the other;
- prints the scores;
- writes everything to `models/`.

One thing to expect in the output: the class distribution is uneven. The
commonest category has 117 training examples and the rarest has 5, because that
is how the source catalogue is shaped -- 114 rows of LED lamps, 3 of skylights.
Macro F1 will therefore sit meaningfully below accuracy, and that is the honest
number to quote. Don't be alarmed by it.

Expected wall time on a 6 GB card: roughly **3-8 minutes for the classifier**
and **2-5 minutes for the span tagger**. On CPU it still completes, it just
takes 20-40 minutes.

You'll see something like:

```
========================================================================
GlassBox - local model training
========================================================================
  python      3.11.9 on Windows
  torch       2.5.1+cu121
  hardware    NVIDIA GeForce RTX 3060 Laptop GPU - 6.0 GB VRAM - batch 16x2 - fp16
  backbone    distilbert-base-uncased
  epochs      6.0

TASK 1/2 - classpath classifier
  train 3,161 - val 430 - 77 classes
  ...
  accuracy  0.9x
  macro F1  0.9x
```

## 5. Send back one folder

When it finishes you'll have:

```
training/models/
├── classpath/        the product classifier
├── spans/            the attribute span tagger
└── metrics.json      the scores  <-- we need this one for the slide
```

**Zip `training/models/` and send it back.** It'll be roughly 250–500 MB. If
that's awkward over chat, Google Drive or WeTransfer is fine — or if you're
comfortable with git, push to a branch:

```bash
git checkout -b trained-models
git add -f training/models
git commit -m "Add trained classpath and span models"
git push origin trained-models
```

(The `-f` is needed because `models/` is gitignored by default so nobody
accidentally commits half a gigabyte.)

**If you can only send one file, send `models/metrics.json`.** It's a few
kilobytes and it's what goes on the results slide.

---

## 6. After training: evaluate, then serve

Two eval scripts live next to `train.py`:

```bash
python eval_models.py --eval-only --cpu   # the two headline numbers (macro F1, entity F1)
python full_eval.py                       # everything on docs/MODEL_EVAL.md's page:
                                         #   per-class F1, confusion pairs, confidence
                                         #   calibration, per-label span F1, CPU latency,
                                         #   and rule-gap coverage on the 1000-row corpus
```

`full_eval.py` is the one to run before finishing the deck: it writes
`models/full_eval.json`, which is every number in
[`../docs/MODEL_EVAL.md`](../docs/MODEL_EVAL.md).

The trained models are served in the main pipeline by
[`../glassbox/distill.py`](../glassbox/distill.py):

```bash
cd ..
python run.py --models     # rules + both encoders, CPU-only, no network
```

Everything the models emit is validated against the slot vocabularies, tagged
`local_model` in the provenance sidecar, and routed to review — the
confidence engine was built for them before they existed.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU-only torch. `pip uninstall torch`, redo step 2 with the `--index-url`. |
| `CUDA out of memory` | `python train.py --batch-size 8`. Still failing? `--batch-size 4`. |
| `missing data file: .../classpath_train.jsonl` | The `data/` folder didn't come across. Re-unzip the whole `training/` directory. |
| Fails on `microsoft/deberta-v3-small` | Stick to the default. `python train.py --model distilbert-base-uncased`. |
| Download of the base model stalls | It's pulling ~250 MB from Hugging Face on first run. Needs internet once; after that it's cached. |
| `TrainingArguments` complains about `eval_strategy` | Version skew — the script already handles both spellings, so update transformers: `pip install -U transformers`. |
| Something else | Send me the last 30 lines of output. Don't debug it yourself, it's not your problem. |

**You do not need an API key for this.** Training is entirely local. (There is
an NVIDIA API key elsewhere in the project, for a different, optional feature —
it has nothing to do with what you're running.)

---

## What you're actually training (the 3-minute version)

Only read this if you're curious, or if you're the one making the video.

### The problem

Unilog builds product content for industrial distributors — the titles,
descriptions and specs that let a contractor find the right part online. The raw
data distributors hand over is close to unusable. A real row from the dataset,
in full:

```
Mfg_Part_Num : 49-94-1940
Part_Desc    : 49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc
E1_Brand     : -- Unbranded --
Unilog_Brand : -- No Unilog Brand --
DIB_Brand    : -- No DIB Brand --
Part_Manuf   : Milwaukee Accessory (4031)
```

Six columns, three of which are placeholders meaning "empty". From that, the
hackathon requires **252 columns** of clean, structured, search-ready product
data: brand with the right trademark symbol, a three-level category path,
fifty attribute triples, and the same product description rewritten five
different ways at five different lengths for the till receipt, the mobile app,
the search page, the product page and the marketing block.

### Our approach, and why these two models exist

Most of the pipeline is deterministic — rules, lexicons and a constraint solver,
not a language model. That's a deliberate choice: the judges will run our code
on a dataset we've never seen, and rules that fail do so visibly and
explainably, while a generative model that fails produces confident nonsense.

The rules do well. They classify 79% of the catalogue and hit 100% compliance on
every checkable house-style constraint. But they're brittle by nature: a keyword
contract only fires on a phrasing somebody anticipated. The remaining 21% are
rows like `9A-570-240 Abranet 2.75x30` — an abrasive disc that never uses the
word "abrasive", "disc", or anything else a rule was written for.

**That's the gap these two models fill.** We took every row the rule engine
classified confidently, treated its output as a label, and built a training set
from it — 3,591 examples for the classifier, 636 for the span tagger. You are
fine-tuning small encoders on that. This is **distillation**: the model learns
the *pattern behind* the rules and generalises to phrasings no rule covers.

- **`classpath`** — sequence classification over 77 product categories. "What
  kind of product is this?"
- **`spans`** — token classification over 38 attribute types. "Which exact
  characters of this string are the diameter, the voltage, the grit?"

The labels are *silver*, not human-annotated. So the model can't beat the rule
engine where the rules already work — that's not the point. It earns its place
on the fifth of the catalogue the rules leave behind.

### Why it matters commercially

This is the line for the slide and the video: **the trained models run locally,
on CPU, in milliseconds, at zero marginal cost.** No per-row API charge, no rate
limit, no data leaving the customer's network. A distributor with 2 million SKUs
can run the whole catalogue overnight inside their own VPC. That's a genuinely
different answer from "we call an LLM API for every row", which is what most
submissions will be — and it's why your 6 GB card is doing something worth
doing.

---

## If you're also making the demo video

The submission needs a **3-minute video**, and it is the second gate: the judges
watch it only if the deck passes, and they look at the code only if the video
passes. So it matters.

Full shot list, timings and narration are in **`../docs/DEMO_VIDEO.md`** —
please read that rather than improvising. The short version:

- **Record the app, not slides.** `python -m streamlit run app/app.py` from the
  project root, then screen-record the browser at 1920×1080.
- **Lead with the raw row.** Show the six columns with three placeholders struck
  through. The problem has to land in the first 20 seconds.
- **The money shot is the "Read from" column** in the attribute table — every
  extracted value next to the exact characters it was read out of. That's our
  differentiator; hold on it.
- **Show a blank we refuse to fill.** Click a `not derivable` cell and read the
  reason aloud. Sounds like a weakness, judges score it as rigour — and the
  Solution Guide explicitly asks for it.
- **End on the numbers**, then the training results from your `metrics.json`.
- **Narrate it.** A silent screen recording with background music scores worse
  than a plain voice explaining what's on screen. Phone mic is fine.

Nothing needs to be scripted word-for-word, but do hit those five beats in that
order.

---

## Questions

Ping the team chat. Don't sink time into debugging this — if `train.py` doesn't
run in ten minutes, something is wrong on our side, not yours, and we'd rather
fix it than have you lose an evening.
