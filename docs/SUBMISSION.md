# Submission pack — everything to paste, in order

Deadline: **Sun 23 Aug 2026, 11:59 PM IST.**
Form: <https://hack2skill.com/event/unilog2026/dashboard/submissions>

The six required fields, with the exact content for each.

---

## 1. Challenge

Select: **AI-Powered Product Intelligence for Industrial Commerce**
(the only option)

---

## 2. Prototype deck — PDF upload, max 5 MB

Upload **`outputs/GlassBox_UniHack_Deck.pdf`** — 14 slides, 0.68 MB, built on
the organisers' mandatory template with their branding intact.

The editable source is `outputs/GlassBox_UniHack_Deck.pptx` if you want to
change anything. If you edit it, re-export to PDF: open in PowerPoint →
*File → Save As → PDF*.

**One optional edit before uploading:** slide 1 lists the team leader only. To
add the other two members, open `scripts/build_deck.py`, put their names in
`TEAM_MEMBERS` at the top, and run `python scripts/build_deck.py`, then
re-export the PDF. Skip it if you're short on time — it isn't required.

---

## 3. Solution overview — paste the block below

```
GlassBox turns six columns of distributor data into a complete 252-column
product record, and every cell can show the evidence behind it.

THE PROBLEM. A supplied row reads "49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off
Disc", with three of six columns holding placeholder text meaning empty. 799 of
1000 rows are unbranded; all 1000 carry "-- No Unilog Brand --".

WHAT WE BUILT. A two-pass engine. Pass one learns the catalogue: brands,
series, product types and unit spellings are induced from the corpus using
group-conditional TF-IDF over distributor accounts, resolving "Milw" to
Milwaukee. This matters because the seven reference workbooks the challenge
describes - a 27,000-row brand list, a 161,000-row list of values - are not
published on the portal, so the engine derives its own vocabularies and works
on an uncurated catalogue on day one. Pass two enriches each row: a six-rung
brand evidence ladder, dual-hierarchy classification over 87 leaves,
per-category attribute contracts constrained to 313 controlled values, and five
description channels each built to its own house contract - including a
constraint solver for the two-sided 60-80 character mobile window and the hard
40-character all-caps invoice line.

TRACEABILITY. Every cell carries the mechanism, rule, confidence and exact
source character span behind it. Confidence is computed from that provenance
rather than self-reported, so fluent invented values cannot score well.
Non-derivable fields - internal part numbers, prices, UPC, country of origin -
are left blank with a recorded reason naming the system of record.

RESULTS, all reproducible via "python run.py": all five description channels of
both published gold rows reproduced character-for-character, including a
390-character long description (9/9 exact); 100% invoice character-limit and
all-caps compliance; 100% controlled-vocabulary compliance; zero unit-format
violations; 100% schema conformance; 79% classified; 1000 rows in ~5s at 204
rows/sec on one CPU core.

It needs no API key, no network and no model download, so it cannot fail on the
evaluation dataset. Optional NVIDIA NIM proposals and a rule-distillation
training package attach on top and are never required.
```

---

## 4. GitHub repository

```
https://github.com/skdas20/UniHack
```

---

## 5. Working prototype link — **you must do this one**

This is the only field I cannot produce for you; it needs your account. Two
options, both free. **Streamlit is faster — use it.**

### Streamlit Community Cloud (about 4 minutes)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. *Create app* → *Deploy a public app from GitHub*.
3. Repository `skdas20/UniHack`, branch `main`, main file path
   **`app/app.py`**.
4. *Deploy*. First build takes 2–3 minutes while it installs dependencies.
5. Copy the `*.streamlit.app` URL.

Nothing else to configure. No secrets needed — the app runs without an API key.

### Hugging Face Spaces (fallback, about 6 minutes)

1. <https://huggingface.co/new-space> → SDK **Streamlit**, visibility Public.
2. In the Space's *Files* tab, upload the repository contents, **or** point it
   at the GitHub repo.
3. The Space card is already written at `app/README.md`.

**Before pasting the URL into the form, open it in a private browsing window.**
A prototype link only you can reach is the most common way teams lose this gate.

---

## 6. Demo video link — 3 minutes

See `docs/DEMO_VIDEO.md` for the full shot list and narration.

**On whether this is heavy to run: it isn't.** The whole app is a single
Streamlit process; 1,000 rows enrich in about five seconds on one CPU core,
with no GPU, no model download and no network calls. Any laptop that can run a
browser can record this. If you'd rather hand it to your other friend, the two
commands are:

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

...and `docs/DEMO_VIDEO.md` is written so someone with no context can follow it.

Upload as **YouTube unlisted** (not private) or Google Drive with
*anyone with the link* sharing. Then open the link in a private window to check.

---

## Order of operations, given the time left

1. **Deploy the app** (4 min) — do this first, it's the only external dependency.
2. **Upload the deck and paste fields 1–4** (3 min).
3. **Record the video** (20–30 min including retakes) and paste field 6.
4. **Submit.** You can submit and then edit if the form allows re-submission —
   check once you're in.

If the clock beats you, submit with the deck, repo and prototype link in place
and the video last. The deck is the first gate; get that in.

---

## What is deliberately not claimed anywhere

The rule-distillation models were never trained — the GPU run didn't happen. So
**no trained-model accuracy appears in the deck, the README or the overview
text.** The training package is presented as built and validated, with the run
pending, which is exactly what it is.

Do not add a model accuracy number to any field. If a judge asks in the finale,
the honest and strong answer is: the deterministic engine already hits the
numbers above without any model, and the distillation package is ready to run
against the 21% of rows the rules currently route to review.
