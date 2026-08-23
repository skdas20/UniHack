# Demo video — recording script

**For whoever is recording. You need no knowledge of this project.** Everything
is a live website; you don't install anything or run any code.

**Target: 3 minutes.** Do not go over — the judges' brief caps it.

---

## Setup (2 minutes)

1. Open **<https://glassbox-unihack.streamlit.app/>** in Chrome.
   - First load can take ~30 seconds if the app has gone to sleep. Wait until
     you see six numbers across the top (Rows enriched, Classified, Brand
     resolved, …). If it says "Yes, get this app back up!", click it and wait.
2. Press **F11** for full screen. Close other tabs first.
3. In the **left sidebar**, check "Rows to process" is on **250**. Leave it.
4. Start your recorder at **1920×1080**. OBS Studio, or **Win + G** on Windows.
5. **Record your voice.** A phone headset is fine. A silent screen recording
   with music scores worse than a plain voice explaining what's on screen.

**Set the demo row before you start recording:**

- Click the **"Catalogue row"** dropdown (under the tabs).
- Type `49-94-1940` — the list filters as you type.
- Select **`49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc`**.

Now start recording. Stay on this one row the whole video.

---

## Beat 1 — the problem (0:00 – 0:25)

**Screen:** the left-hand panel, headed *RAW CATALOGUE ROW — 6 COLUMNS*.
Move the mouse slowly over the three struck-through lines.

**Say:**

> "This is one row of a real industrial distributor's catalogue. Six columns
> total. The description reads 'Milw fourteen by one-eighth by one inch Masonry
> Cut Off Disc'. Three of the six columns say 'Unbranded' or 'No Brand' — and
> those aren't missing values, that's placeholder text that means empty. Seven
> hundred and ninety-nine of the thousand rows look like this.
>
> From that, Unilog needs 252 columns of clean, search-ready product data."

---

## Beat 2 — what comes out (0:25 – 0:55)

**Screen:** the right-hand panel, *ENRICHED, SEARCH-READY RECORD*. Scroll it
slowly so the five description fields pass by.

**Say:**

> "Here's what the engine produces. The brand resolved to Milwaukee — from the
> shorthand 'Milw', with nothing in the brand columns to help. The manufacturer
> is Milwaukee Electric Tool Corporation, which is a separate field and often a
> different company. A three-level category path. And then the same product
> written five different ways.
>
> Forty characters in capitals for the till receipt. Sixty to eighty for the
> mobile app. A full title for the search page. Look at the units — spaced in
> the long description, but jammed together in the invoice line. Those are two
> different house rules for the same value, and both of the worked examples the
> organisers published prove it."

---

## Beat 3 — the differentiator (0:55 – 1:40) **← the most important shot**

**Screen:** scroll down to the table headed **Attribute block**. Move the mouse
slowly down the last column, **"Read from"**.

**Slow down here. This is the part nobody else will have.**

**Say:**

> "Now the part I'd like you to look at closely. Every attribute in this table
> has a 'How' column and a 'Read from' column.
>
> Diameter, fourteen inches — read from *these* exact characters of the original
> description. Thickness, one eighth of an inch — same source. Arbor size, one
> inch. This isn't a model that produced plausible-looking numbers. Every single
> cell knows the characters it came from, the rule that fired, and its own
> confidence score."

**Then:** scroll a little further to **"Where each value came from"**. Click the
**"Output cell"** dropdown and pick **`SHORT_DESC`**. Two panels appear.

**Say:**

> "Pick any cell and you get the mechanism, the rule, the confidence, and the
> source text highlighted underneath. Two hundred and fifty-two columns, all of
> them auditable. That's the difference between a demo and something a content
> team could actually adopt."

---

## Beat 4 — what it refuses to invent (1:40 – 2:05)

**Screen:** same "Output cell" dropdown. Scroll the enriched panel until you can
see a field showing grey italic text saying **blank — …**. (Or pick a cell and
read the "Reasoning" line.)

**Say:**

> "And here's a blank, on purpose. The distributor's internal part number isn't
> a function of any input column — no amount of AI recovers it from six columns
> of distributor data. So we leave it empty and record the reason, naming the
> system that would actually supply it.
>
> The challenge brief says a fluent description built from invented values
> scores zero. We took that seriously: we'd rather show a blank with a reason
> than a confident guess."

---

## Beat 5 — it measures itself (2:05 – 2:40)

**Screen:** click the **Compliance** tab at the top.

**Say:**

> "Everything is measured. Against the two fully-worked rows the organisers
> published, all five description channels reproduce character for character —
> nine out of nine, including a three-hundred-and-ninety-character long
> description."

**Then:** scroll down to the grid of percentages.

> "Across the whole catalogue: a hundred percent of invoice lines inside forty
> characters, a hundred percent in capitals, a hundred percent of attribute
> values inside the approved vocabulary, zero unit-formatting errors, a hundred
> percent schema conformance."

**Then:** click the **Review queue** tab.

> "And it knows what it doesn't know. Every row is triaged — auto-publishable,
> needs review, or blocked — and each flagged row carries a specific reason a
> person can clear in seconds. That's the human-in-the-loop story: the engine
> does the volume and asks for help exactly where it should."

---

## Beat 6 — close (2:40 – 3:00)

**Screen:** click the **Induced vocabulary** tab.

**Say:**

> "One last thing. The reference files this challenge describes — a
> twenty-seven-thousand-row brand list, a hundred-and-sixty-one-thousand-row
> list of approved values — aren't published anywhere. So the engine learns its
> own. Fifty-one brands induced from the catalogue itself, every one backed by
> evidence, and the brand-shaped words it *couldn't* prove are held back for a
> human instead of guessed at.
>
> That means it works on a distributor's catalogue nobody has curated yet. No
> API key, no network, no model download. That's GlassBox — six columns in, two
> hundred and fifty-two out, and every one of them can show you its evidence."

**Stop recording.**

---

## Optional 10-second extra, only if you're under 2:50

Go back to the sidebar, drag **Rows to process** to **1000**, and let it run
while you say:

> "And here's the full thousand-row file, processed live."

Then stop. Don't do this if it puts you over three minutes.

---

## Upload

1. YouTube → **Unlisted**. *Not Private* — private means the judges can't open it.
2. Title: `UniHack 2026 — GlassBox — Team SKavengers`
3. Copy the link.
4. **Open the link in a private/incognito window to confirm it plays.** A video
   only you can see is the single most common way teams lose this gate.
5. Send the link back.

---

## If you are short on time

Record **beats 1, 3 and 5 only** — the problem, the provenance, the results.
A tight 90-second video that lands the "Read from" column beats a rushed
three-minute one that skims it.

## Things to avoid

- Don't show code or an editor. Nobody assesses architecture from a video.
- Don't jump between catalogue rows — the viewer loses the thread.
- Don't read numbers that aren't on screen.
- Don't apologise for anything. The blanks are a deliberate design decision;
  say so confidently.
