# The 3-minute demo video — shot list and narration

## Why this document is strict about it

The submission page states the evaluation order plainly:

> **PPT → Demo → Prototype → GitHub**
> Your Prototype Deck will be the first thing to be reviewed. Only if your PPT
> passes, your Video Demo will be reviewed. Only if your Demo Video passes, your
> Prototype will be reviewed. Only if all the above passes, your GitHub Repo
> will be reviewed.

So the video is the second gate. A judge who is unconvinced at 0:45 never opens
the repository. Everything below is ordered to survive that.

**Hard requirement: 3 minutes.** Going over reads as not knowing what matters.

---

## Before you record

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
python run.py                    # ~7s, generates outputs/ so the export tab is real
python -m streamlit run app/app.py
```

Then in the browser:

- Set **Rows to process** to **250**. It runs in about two seconds and every
  metric on screen is real. (1000 also works and takes ~7s; 250 keeps the
  demo snappy and the numbers are honestly labelled either way.)
- Full-screen the browser. Hide bookmarks, close other tabs, use a clean
  profile. `Ctrl+Shift+P` in Firefox or a fresh Chrome profile is easiest.
- Record at **1920×1080**. OBS Studio is free and fine; Windows `Win+G` works
  too.
- **Record audio.** A voice explaining the screen scores materially better than
  music over silence. A phone headset is completely adequate. Speak slowly —
  three minutes is longer than it feels.

Pick **one** row and stay on it for the whole demo so the viewer can follow a
single product all the way through. Recommended:

```
49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc
```

It is ideal: three placeholder columns, brand shorthand (`Milw`), a three-part
trade dimension run, and an implied category. Alternative if you prefer
appliances: `PDSH4816AF Dishwasher SS - Display Only`.

---

## Shot list

### 0:00 – 0:25 · The problem, on screen, immediately

**Show:** the **Enrich** tab, left panel — the raw catalogue row.

**Say:**
> "This is one row of a real industrial distributor's catalogue. Six columns.
> The description is `Milw 14 by 1/8 by 1 inch Masonry Cut Off Disc`. Three of
> the brand columns say `Unbranded` — those aren't missing values, they're
> placeholder text that means empty. Seven hundred and ninety-nine of the
> thousand rows look like this."

**Do:** point the cursor at the struck-through placeholders. Don't click
anything yet.

> "Unilog needs 252 columns of clean, search-ready product data out of this."

---

### 0:25 – 0:55 · What comes out

**Show:** the right panel — the enriched record.

**Say:**
> "Here's what our engine produces. Brand resolved to `Diablo` with the
> registered mark. Manufacturer `Freud America` — a different company from the
> brand, which the data doesn't tell you. A three-level category path. And the
> same product written five different ways."

**Do:** scroll slowly down the five description fields.

> "Forty characters in all caps for the till receipt. Sixty to eighty for the
> mobile app. A full title for the search page. Note the units — `120 V` with a
> space in the long copy, `120V` glued in the invoice line. Those are two
> different house rules for the same value, and both gold examples the
> organisers published prove it."

---

### 0:55 – 1:35 · The differentiator — every value shows its evidence

This is the most important 40 seconds of the video. Slow down.

**Show:** scroll to the **Attribute block** table.

**Say:**
> "Now the part I'd like you to look at. Every attribute has a `How` column and
> a `Read from` column."

**Do:** cursor along the `Read from` column.

> "Diameter, fourteen inches — read from these exact characters. Thickness,
> one eighth — same source span. This isn't a model that produced plausible
> numbers. Every cell knows the characters it came from, the rule that fired,
> and its confidence."

**Do:** scroll to **Where each value came from**, pick `SHORT_DESC` from the
dropdown, and let the reasoning panel and highlighted source render.

> "Pick any cell and you get the mechanism, the rule, and the source
> highlighted in the original text. Two hundred and fifty-two columns, all
> auditable. That's what makes this adoptable by a content team rather than
> just a demo."

---

### 1:35 – 2:00 · What we refuse to invent

**Show:** still in the cell inspector, pick a `not derivable` field — e.g.
`PART_NUMBER` — or scroll the enriched panel to a blank showing its reason.

**Say:**
> "And here's a blank. `PART_NUMBER` — the distributor's internal ERP number.
> It isn't a function of any input column. No amount of AI recovers it from six
> columns of distributor data, so we leave it empty and record why."

**Do:** hop to the **Compliance** tab, scroll to *What we do not claim*.

> "Same for UNSPSC and country of origin — both published gold rows leave those
> blank too. The Solution Guide says a fluent description built from invented
> values scores zero. We took that seriously: we'd rather show a blank with a
> reason than a confident guess."

---

### 2:00 – 2:35 · It measures itself

**Show:** the **Compliance** tab, top section.

**Say:**
> "Everything is measured. Against the two fully-worked rows the organisers
> published, all five description channels reproduce character-for-character —
> including a three-hundred-and-ninety-character long description."

**Do:** point at the gold-row metrics, then scroll to the compliance grid.

> "Across the whole catalogue: a hundred percent of invoice lines inside forty
> characters, a hundred percent all-caps, a hundred percent of attribute values
> inside the controlled vocabulary, zero unit-spacing errors, a hundred percent
> schema conformance."

**Do:** switch to the **Review queue** tab.

> "And it knows what it doesn't know. Rows are triaged — auto-publishable,
> needs review, or blocked — and every flagged row carries a specific reason a
> human can clear in seconds. That's the human-in-the-loop story: the engine
> does the volume and asks for help precisely where it should."

---

### 2:35 – 3:00 · Scale, and close

**Show:** the **Induced vocabulary** tab.

**Say:**
> "One last thing. The reference files this challenge describes — a
> twenty-seven-thousand-row brand list, a hundred-and-sixty-one-thousand-row
> list of values — aren't published. So the engine learns its own. Fifty-one
> brands induced from the corpus, every one linked to attested evidence, and
> the brand-shaped tokens it *couldn't* prove are held back for a human instead
> of guessed at."

**Do:** briefly show the "Held back for a human" panel, then close on the
metrics row at the top of the page.

> "That means it works on a distributor's catalogue nobody has curated yet —
> day one, no dictionary maintenance. A thousand rows in seven seconds on a
> laptop, with no API key and no network. Plus a distilled local model that runs
> the same job on CPU at zero marginal cost. That's GlassBox — six columns in,
> two hundred and fifty-two out, and every one of them can show you its
> evidence."

---

## If the trained models are ready in time

Add 10 seconds before the close, and trim the vocabulary section:

> "We also distilled the rule engine into two small local models — a category
> classifier and an attribute span tagger, trained on labels the rule engine
> generated. [X] percent macro F1 on held-out data, running on CPU in
> milliseconds, so an enterprise catalogue costs nothing per row to process."

Take `[X]` from `training/models/metrics.json`. **Do not state a number that
isn't in that file** — a judge who reruns the training and gets something else
is the worst possible outcome.

---

## Common ways this video goes wrong

| Mistake | Why it costs you |
|---|---|
| Slides instead of the running app | Judges are explicitly checking a *working* prototype at this stage |
| Silent recording with music | The reasoning is the product; nobody infers it from a cursor moving |
| Overrunning 3 minutes | Reads as not knowing what matters |
| Showing code in an editor | Nobody assesses architecture from a video; that's the repo's job |
| Jumping between rows | The viewer loses the thread; one product, start to finish |
| Claiming a number not on screen | Instantly checkable, and fatal to credibility |
| Skipping the evidence/provenance shot | It's the only part nobody else will have |
| Apologising for what's incomplete | State scope confidently; the blanks are a design decision, so say so |

## Upload

- YouTube **unlisted** (not private — judges must be able to open it without an
  invite), or Google Drive with link sharing set to *anyone with the link*.
- Title: `UniHack 2026 — GlassBox — Team SKavengers`
- **Open the link in a private browsing window before submitting.** A video only
  you can see is the single most common way teams lose the demo gate.
