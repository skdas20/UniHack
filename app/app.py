"""GlassBox -- the demo application.

Deployed at Hugging Face Spaces as the working prototype link. Five views:

1. **Enrich**      raw row on the left, enriched record on the right, and the
                   provenance behind any cell you click -- including the exact
                   characters of the source description it was read from.
2. **Compliance**  the label-free metrics, measured over every row.
3. **Review**      the triaged queue, worst rows first, with specific reasons.
4. **Vocabulary**  what the induction pass learned, and what it held back.
5. **Export**      the delivery sheet, provenance sidecar and report.

Runs with no API key. The optional proposal layer is switched on from the
sidebar and only if a key is present in the environment.
"""

from __future__ import annotations

import html
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox.confidence import reason_histogram, triage  # noqa: E402
from glassbox.evaluate import evaluate  # noqa: E402
from glassbox.induce import InducedVocabulary  # noqa: E402
from glassbox.pipeline import Pipeline  # noqa: E402
from glassbox.provenance import EnrichedRow, Source  # noqa: E402
from glassbox.schema import OutputSchema, RawRow, load_input  # noqa: E402
from glassbox.writers import (  # noqa: E402
    blank_reason_histogram,
    source_histogram,
)

DEFAULT_INPUT = ROOT / "data" / "raw" / "input_1000.csv"
SCHEMA_PATH = ROOT / "data" / "raw" / "expected_output_schema.csv"
VOCAB_PATH = ROOT / "data" / "vocab" / "induced.json"

st.set_page_config(
    page_title="GlassBox · Product Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# styling
# --------------------------------------------------------------------------

st.markdown(
    """
<style>
  :root {
    --gb-ink: #e8eaed;
    --gb-dim: #9aa3ad;
    --gb-line: #2a3037;
    --gb-panel: #14181d;
    --gb-accent: #58c4a0;
    --gb-warn: #e8b555;
    --gb-bad: #e3776a;
  }
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  h1, h2, h3 { letter-spacing: -0.02em; }

  .gb-hero {
    border: 1px solid var(--gb-line); border-radius: 14px;
    padding: 1.15rem 1.35rem; margin-bottom: 1.1rem;
    background: linear-gradient(135deg, rgba(88,196,160,.10), rgba(88,196,160,0) 60%);
  }
  .gb-hero h1 { margin: 0 0 .3rem 0; font-size: 1.6rem; }
  .gb-hero p  { margin: 0; color: var(--gb-dim); font-size: .93rem; line-height:1.5; }

  .gb-panel {
    border: 1px solid var(--gb-line); border-radius: 12px;
    padding: 1rem 1.15rem; background: var(--gb-panel); height: 100%;
  }
  .gb-panel h4 {
    margin: 0 0 .7rem 0; font-size: .72rem; letter-spacing: .12em;
    text-transform: uppercase; color: var(--gb-dim); font-weight: 600;
  }
  .gb-raw {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: .82rem; line-height: 1.75; color: var(--gb-ink);
    white-space: pre-wrap; word-break: break-word;
  }
  .gb-raw .k { color: var(--gb-dim); }
  .gb-ph {
    color: #6b7280; text-decoration: line-through;
    text-decoration-color: var(--gb-bad); text-decoration-thickness: 1px;
  }
  mark.gb-hit {
    background: rgba(88,196,160,.26); color: var(--gb-ink);
    border-bottom: 2px solid var(--gb-accent);
    padding: 0 2px; border-radius: 3px;
  }

  .gb-field { margin-bottom: .78rem; }
  .gb-field .lbl {
    font-size: .66rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--gb-dim); font-weight: 600;
  }
  .gb-field .val {
    font-size: .92rem; color: var(--gb-ink); line-height: 1.45;
    word-break: break-word;
  }
  .gb-field .val.mono {
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .84rem;
  }
  .gb-field .blank { color: #5f6b76; font-style: italic; font-size: .84rem; }

  .gb-chip {
    display: inline-block; font-size: .63rem; padding: .1rem .42rem;
    border-radius: 5px; border: 1px solid var(--gb-line);
    color: var(--gb-dim); margin-left: .38rem; vertical-align: 1px;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .gb-chip.ok   { color: var(--gb-accent); border-color: rgba(88,196,160,.4); }
  .gb-chip.warn { color: var(--gb-warn);   border-color: rgba(232,181,85,.4); }
  .gb-chip.bad  { color: var(--gb-bad);    border-color: rgba(227,119,106,.4); }

  .gb-count { font-family: ui-monospace, Menlo, Consolas, monospace; font-size:.72rem; }
  .gb-note {
    font-size: .78rem; color: var(--gb-dim); border-left: 2px solid var(--gb-line);
    padding-left: .7rem; margin: .45rem 0 0 0; line-height: 1.55;
  }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
  div[data-testid="stMetricLabel"] { font-size: .74rem; color: var(--gb-dim); }
  .stTabs [data-baseweb="tab"] { font-size: .88rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# data + pipeline
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_schema() -> OutputSchema:
    return OutputSchema.from_csv(SCHEMA_PATH)


@st.cache_resource(show_spinner=False)
def get_vocab() -> InducedVocabulary | None:
    if VOCAB_PATH.exists():
        return InducedVocabulary.from_json(VOCAB_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_rows(source: str, payload: bytes | None, limit: int) -> list[dict]:
    if payload is not None:
        buffer = io.BytesIO(payload)
        frame = pd.read_csv(buffer, dtype=str, keep_default_na=False)
        records = frame.to_dict("records")
        rows = [{"index": i, "data": r} for i, r in enumerate(records)]
    else:
        rows = [{"index": r.index, "data": r.data} for r in load_input(DEFAULT_INPUT)]
    return rows[:limit] if limit else rows


@st.cache_data(show_spinner=False, max_entries=4)
def run_pipeline(
    row_payload: list[dict], use_proposer: bool
) -> tuple[list[dict], dict, dict, dict]:
    """Enrich, and return JSON-safe structures so Streamlit can cache them."""
    schema = get_schema()
    rows = [RawRow(index=r["index"], data=r["data"]) for r in row_payload]

    proposer = None
    proposer_stats: dict = {}
    if use_proposer:
        from glassbox.enrich import build_proposer

        proposer = build_proposer()

    pipeline = Pipeline(schema, vocab=get_vocab(), proposer=proposer)
    enriched = pipeline.run(rows)

    if proposer is not None:
        proposer_stats = proposer.stats.as_dict()

    evaluation = evaluate(enriched, schema)
    buckets = triage(enriched)
    summary = {
        "run": pipeline.stats.as_dict(),
        "compliance": evaluation["compliance"],
        "gold": evaluation["gold_channel_check"],
        "notes": evaluation["notes"],
        "cells_by_source": source_histogram(enriched),
        "blanks_by_reason": blank_reason_histogram(enriched),
        "triage": {k: len(v) for k, v in buckets.items()},
        "review_reasons": reason_histogram(enriched),
        "vocabulary": {k: int(v) for k, v in pipeline.vocab.stats.items()},
        "proposer": proposer_stats,
    }

    payload = [
        {
            "index": r.index,
            "confidence": r.confidence,
            "needs_review": r.needs_review,
            "reasons": r.review_reasons,
            "record": r.as_record(schema),
            "audit": r.audit_record(),
        }
        for r in enriched
    ]
    return payload, summary, {}, {}


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------

PLACEHOLDERS = {"-- Unbranded --", "-- No Unilog Brand --", "-- No DIB Brand --",
                "COMMODITY - UNBRANDED", "-"}

SOURCE_TONE = {
    "input_copy": "ok", "lexicon_exact": "ok", "regex_extract": "ok",
    "constraint_solver": "ok", "derived": "ok", "crosswalk": "ok",
    "lexicon_fuzzy": "warn", "local_model": "warn", "llm": "warn",
    "not_derivable": "", "placeholder_dropped": "", "contract_template": "",
    "unresolved": "bad",
}

SOURCE_LABEL = {
    "input_copy": "copied", "lexicon_exact": "lexicon", "regex_extract": "extracted",
    "constraint_solver": "solver", "derived": "derived", "crosswalk": "crosswalk",
    "lexicon_fuzzy": "fuzzy", "local_model": "local model", "llm": "model proposal",
    "not_derivable": "not derivable", "placeholder_dropped": "placeholder",
    "contract_template": "contract slot", "unresolved": "unresolved",
}


def raw_panel(record: dict) -> str:
    lines = []
    for key in ("Mfg_Part_Num", "Part_Desc", "E1_Brand", "Unilog_Brand",
                "DIB_Brand", "Part_Manuf"):
        value = record.get(key, "") or ""
        shown = html.escape(value)
        if value.strip() in PLACEHOLDERS:
            shown = f'<span class="gb-ph">{shown}</span>'
        lines.append(f'<span class="k">{key:<14}</span>{shown or "&mdash;"}')
    return (
        '<div class="gb-panel"><h4>Raw catalogue row &mdash; 6 columns</h4>'
        f'<div class="gb-raw">{"<br>".join(lines)}</div>'
        '<p class="gb-note">Struck-through values are placeholders: they look '
        "like data and mean empty. 799 of the 1000 rows carry "
        "<code>-- Unbranded --</code>, and every single row carries "
        "<code>-- No Unilog Brand --</code>.</p></div>"
    )


def chip(audit_cell: dict | None) -> str:
    if not audit_cell:
        return ""
    source = audit_cell.get("source", "")
    tone = SOURCE_TONE.get(source, "")
    label = SOURCE_LABEL.get(source, source)
    conf = audit_cell.get("confidence")
    text = f"{label} {conf:.2f}" if isinstance(conf, float) and conf else label
    return f'<span class="gb-chip {tone}">{html.escape(text)}</span>'


def field_block(
    label: str, value: str, audit: dict, *, mono: bool = False, suffix: str = ""
) -> str:
    cell = (audit.get("cells") or {}).get(label)
    if value:
        klass = "val mono" if mono else "val"
        body = f'<div class="{klass}">{html.escape(value)}{suffix}</div>'
    else:
        reason = ""
        if cell:
            reason = cell.get("detail") or SOURCE_LABEL.get(cell.get("source", ""), "")
        body = f'<div class="blank">blank &mdash; {html.escape(reason or "not attempted")}</div>'
    return (
        f'<div class="gb-field"><div class="lbl">{html.escape(label)}{chip(cell)}</div>'
        f"{body}</div>"
    )


def highlight(text: str, spans: list[tuple[int, int]]) -> str:
    """Render ``text`` with ``spans`` wrapped in a highlight mark."""
    if not text:
        return ""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, cursor = [], 0
    for start, end in merged:
        start, end = max(0, start), min(len(text), end)
        if start >= end:
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(f'<mark class="gb-hit">{html.escape(text[start:end])}</mark>')
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def length_chip(value: str, lo: int | None, hi: int) -> str:
    n = len(value)
    if not n:
        return ""
    ok = (lo is None or n >= lo) and n <= hi
    tone = "ok" if ok else "bad"
    window = f"{lo}-{hi}" if lo else f"max {hi}"
    return f'<span class="gb-chip {tone}">{n} chars / {window}</span>'


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### GlassBox")
    st.caption("Traceable product intelligence for industrial commerce")

    uploaded = st.file_uploader(
        "Catalogue CSV", type=["csv"],
        help="Six columns: Mfg_Part_Num, Part_Desc, E1_Brand, Unilog_Brand, "
             "DIB_Brand, Part_Manuf. Leave empty to use the supplied 1000-row sample.",
    )
    limit = st.select_slider(
        "Rows to process", options=[25, 50, 100, 250, 500, 1000], value=250
    )

    st.divider()
    st.markdown("**Optional enrichment layer**")
    proposer_available = False
    try:
        from glassbox.enrich import ProposerConfig

        proposer_available = bool(ProposerConfig.from_env().api_key)
    except Exception:
        proposer_available = False

    use_proposer = st.toggle(
        "Model proposals (NVIDIA NIM)",
        value=False,
        disabled=not proposer_available,
        help="Fills blank contract slots from a hosted model's knowledge of the "
             "part number. Proposals are validated against each slot's "
             "controlled vocabulary, labelled as proposals, and always routed "
             "to review. The pipeline is complete without this.",
    )
    if not proposer_available:
        st.caption("No `NVIDIA_API_KEY` in the environment — the deterministic "
                   "core runs unaffected.")

    st.divider()
    st.caption(
        "The engine needs no API key, no network and no model download. "
        "That is deliberate: it has to run first time on the evaluation "
        "dataset, on a machine we will never see."
    )

payload_bytes = uploaded.getvalue() if uploaded is not None else None
row_payload = load_rows("upload" if uploaded else "sample", payload_bytes, limit)

st.markdown(
    """
<div class="gb-hero">
  <h1>GlassBox</h1>
  <p><b>Six columns in, 252 out — and every one of them can show you its evidence.</b><br>
  A self-bootstrapping enrichment engine for industrial catalogues: it induces its own
  brand and attribute vocabularies from the corpus, cites the exact characters behind
  every value it emits, and refuses to invent the ones it cannot derive.</p>
</div>
""",
    unsafe_allow_html=True,
)

started = time.perf_counter()
with st.spinner(f"Learning the catalogue's vocabulary, then enriching {len(row_payload)} rows…"):
    rows, summary, _a, _b = run_pipeline(row_payload, use_proposer and proposer_available)
elapsed = time.perf_counter() - started

run = summary["run"]
comp = summary["compliance"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows enriched", f"{run['rows']:,}", f"{run['rows_per_s']:.0f}/sec")
c2.metric("Classified", f"{run['classified_pct']:.1f}%")
c3.metric("Brand resolved", f"{run['brand_resolved_pct']:.1f}%")
c4.metric("Invoice ≤ 40 chars", f"{comp['invoice_within_40_chars_pct']:.1f}%")
c5.metric("LOV compliance", f"{comp['lov_compliance_pct']:.1f}%")
c6.metric("Auto-publishable", f"{summary['triage']['auto_publish']:,}",
          f"{summary['triage']['needs_review']:,} to review")

tab_enrich, tab_compliance, tab_review, tab_vocab, tab_export = st.tabs(
    ["Enrich", "Compliance", "Review queue", "Induced vocabulary", "Export"]
)

# --------------------------------------------------------------------------
# 1. enrich
# --------------------------------------------------------------------------

with tab_enrich:
    options = list(range(len(rows)))

    def _fmt(i: int) -> str:
        rec = rows[i]["record"]
        flag = "⚠ " if rows[i]["needs_review"] else "✓ "
        return f"{flag}{rec.get('Mfg_Part_Num','')} — {rec.get('Part_Desc','')[:64]}"

    picked = st.selectbox("Catalogue row", options, format_func=_fmt, index=0)
    row = rows[picked]
    record, audit = row["record"], row["audit"]
    cells = audit.get("cells") or {}

    left, right = st.columns([1, 1.35], gap="medium")

    with left:
        st.markdown(raw_panel(record), unsafe_allow_html=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        conf = row["confidence"]
        tone = "ok" if conf >= 0.85 else ("warn" if conf >= 0.45 else "bad")
        reasons = row["reasons"]
        reason_html = (
            "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
            or "<li>none — this row is auto-publishable</li>"
        )
        st.markdown(
            f"""<div class="gb-panel"><h4>Audit verdict</h4>
            <div class="gb-field"><div class="lbl">Confidence
            <span class="gb-chip {tone}">{conf:.2f}</span></div>
            <div class="val">{'Needs human review' if row['needs_review'] else 'Auto-publishable'}</div></div>
            <div class="lbl" style="margin-top:.6rem">Reasons</div>
            <ul class="gb-note" style="margin-top:.3rem">{reason_html}</ul>
            </div>""",
            unsafe_allow_html=True,
        )

    with right:
        blocks = [
            field_block("BRAND_NAME", record.get("BRAND_NAME", ""), audit),
            field_block("MANUFACTURER_NAME", record.get("MANUFACTURER_NAME", ""), audit),
            field_block("Classpath", record.get("Classpath", ""), audit, mono=True),
            field_block("Product Name", record.get("Product Name", ""), audit),
            field_block("SHORT_DESC", record.get("SHORT_DESC", ""), audit),
            field_block(
                "INVOICE_DESC", record.get("INVOICE_DESC", ""), audit, mono=True,
                suffix=length_chip(record.get("INVOICE_DESC", ""), None, 40),
            ),
            field_block(
                "MOBILE_DESC", record.get("MOBILE_DESC", ""), audit,
                suffix=length_chip(record.get("MOBILE_DESC", ""), 60, 80),
            ),
            field_block("RETAIL_DESC", record.get("RETAIL_DESC", ""), audit),
            field_block("LONG_DESC1", record.get("LONG_DESC1", ""), audit),
        ]
        st.markdown(
            '<div class="gb-panel"><h4>Enriched, search-ready record</h4>'
            + "".join(blocks)
            + "</div>",
            unsafe_allow_html=True,
        )

    # -- attribute block --
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    attr_rows = []
    for slot in range(1, 51):
        label = record.get(f"ATTRIBUTE_LABEL {slot}", "")
        if not label:
            continue
        value = record.get(f"ATTRIBUTE_VALUE {slot}", "")
        uom = record.get(f"ATTRIBUTE_UOM {slot}", "")
        cell = cells.get(f"ATTRIBUTE_VALUE {slot}") or {}
        evidence = (cell.get("evidence") or [{}])[0].get("snippet", "")
        attr_rows.append(
            {
                "#": slot,
                "Attribute": label,
                "Value": value,
                "UOM": uom,
                "How": SOURCE_LABEL.get(cell.get("source", ""), ""),
                "Confidence": round(cell.get("confidence", 0) or 0, 2) if value else None,
                "Read from": evidence,
            }
        )
    if attr_rows:
        st.markdown("##### Attribute block")
        st.caption(
            "Every slot the category's contract requires is emitted, including "
            "the blank ones — both gold rows do exactly that, and the label "
            "sequence is the category's signature."
        )
        st.dataframe(pd.DataFrame(attr_rows), width="stretch", hide_index=True,
                     height=min(60 + 35 * len(attr_rows), 560))

    # -- evidence highlighting --
    st.markdown("##### Where each value came from")
    st.caption(
        "Pick any populated cell to see the mechanism that produced it and, "
        "where one exists, the exact characters of the source it was read from."
    )
    populated = sorted(
        h for h, c in cells.items() if c.get("value") and not h.startswith("ATTRIBUTE_LABEL")
    )
    if populated:
        chosen = st.selectbox("Output cell", populated,
                              index=min(populated.index("SHORT_DESC") if "SHORT_DESC" in populated else 0,
                                        len(populated) - 1))
        cell = cells[chosen]
        ev = cell.get("evidence") or []
        st.markdown(
            f"""<div class="gb-panel">
            <div class="gb-field"><div class="lbl">Value</div>
              <div class="val mono">{html.escape(str(cell.get('value','')))}</div></div>
            <div class="gb-field"><div class="lbl">Mechanism</div>
              <div class="val">{html.escape(SOURCE_LABEL.get(cell.get('source',''), cell.get('source','')))}
              <span class="gb-chip">rule: {html.escape(str(cell.get('rule','')))}</span>
              <span class="gb-chip {'ok' if (cell.get('confidence') or 0) >= .85 else 'warn'}">
              confidence {cell.get('confidence', 0):.2f}</span></div></div>
            <div class="gb-field"><div class="lbl">Reasoning</div>
              <div class="val" style="font-size:.85rem">{html.escape(str(cell.get('detail','') or '—'))}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )
        if ev:
            source_text = ev[0].get("snippet", "")
            # reconstruct highlight against the description we still have
            desc = record.get("Part_Desc", "")
            spans = []
            if source_text and source_text in desc:
                pos = desc.index(source_text)
                spans.append((pos, pos + len(source_text)))
            body = highlight(desc, spans) if spans else html.escape(desc)
            st.markdown(
                f'<div class="gb-panel"><h4>Evidence in the source description</h4>'
                f'<div class="gb-raw">{body}</div></div>',
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# 2. compliance
# --------------------------------------------------------------------------

with tab_compliance:
    gold = summary["gold"]
    if gold.get("available"):
        st.markdown("##### Exact-match accuracy against the published gold rows")
        g1, g2, g3 = st.columns(3)
        g1.metric("Channels checked", gold["channels_checked"])
        g2.metric("Exact matches", gold["exact_matches"])
        g3.metric("Exact-match rate", f"{gold['exact_match_pct']:.0f}%")
        if gold.get("failures"):
            st.error("Failing channels: " + ", ".join(gold["failures"]))
        else:
            st.success(
                "All five description channels of both published rows are "
                "reproduced character-for-character, including the "
                "390-character long description."
            )

    st.markdown("##### Label-free compliance, measured over every row")
    metrics = [
        ("Invoice ≤ 40 chars", f"{comp['invoice_within_40_chars_pct']:.1f}%"),
        ("Invoice ALL CAPS", f"{comp['invoice_all_caps_pct']:.1f}%"),
        ("Mobile in 60–80", f"{comp['mobile_within_60_80_pct']:.1f}%"),
        ("LOV compliance", f"{comp['lov_compliance_pct']:.1f}%"),
        ("UOM spacing errors", f"{comp['uom_spacing_violations']}"),
        ("Schema conformance", f"{comp['schema_conformance_pct']:.1f}%"),
        ("Provenance coverage", f"{comp['provenance_coverage_pct']:.1f}%"),
        ("Cells populated", f"{comp['cells_populated']:,}"),
    ]
    cols = st.columns(4)
    for i, (label, value) in enumerate(metrics):
        cols[i % 4].metric(label, value)

    left, right = st.columns(2)
    with left:
        st.markdown("##### How every populated cell was produced")
        src = pd.DataFrame(
            [{"Mechanism": SOURCE_LABEL.get(k, k), "Cells": v}
             for k, v in summary["cells_by_source"].items()]
        )
        st.dataframe(src, width="stretch", hide_index=True)
        st.caption(
            "Confidence is computed from this distribution, not from a model's "
            "self-report. A row of fluent invented values cannot score well."
        )
    with right:
        st.markdown("##### Why cells are blank")
        blanks = pd.DataFrame(
            [{"Reason": SOURCE_LABEL.get(k, k), "Cells": v}
             for k, v in summary["blanks_by_reason"].items()]
        )
        st.dataframe(blanks, width="stretch", hide_index=True)
        st.caption(
            "An intentional blank is not a failure. `not derivable` means the "
            "field is not a function of any input column — a distributor's "
            "internal part number cannot be recovered from six columns of "
            "distributor data, so it is left empty and the reason recorded."
        )

    st.markdown("##### What we do not claim")
    for note in summary["notes"]:
        st.markdown(f'<p class="gb-note">{html.escape(note)}</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 3. review queue
# --------------------------------------------------------------------------

with tab_review:
    t = summary["triage"]
    a, b, c = st.columns(3)
    a.metric("Auto-publish", f"{t['auto_publish']:,}")
    b.metric("Needs review", f"{t['needs_review']:,}")
    c.metric("Blocked", f"{t['blocked']:,}")
    st.caption(
        "Triage is the deliverable a content team actually works from: a row is "
        "auto-publishable, reviewable, or blocked, and every reviewable row "
        "carries a specific reason a person can clear in seconds."
    )

    flagged = sorted((r for r in rows if r["needs_review"]), key=lambda r: r["confidence"])
    if flagged:
        queue = pd.DataFrame(
            [
                {
                    "Confidence": round(r["confidence"], 3),
                    "MPN": r["record"].get("Mfg_Part_Num", ""),
                    "Description": r["record"].get("Part_Desc", ""),
                    "Brand": r["record"].get("BRAND_NAME", ""),
                    "Classpath": r["record"].get("Classpath", ""),
                    "Invoice": r["record"].get("INVOICE_DESC", ""),
                    "Mobile chars": len(r["record"].get("MOBILE_DESC", "")),
                    "Reasons": " | ".join(r["reasons"]),
                }
                for r in flagged
            ]
        )
        st.dataframe(queue, width="stretch", hide_index=True, height=460)
    else:
        st.success("Nothing flagged in this batch.")

    st.markdown("##### Which problems dominate")
    st.caption("This is the roadmap: fix the top reason and the queue shrinks fastest.")
    hist = pd.DataFrame(
        [{"Reason": k, "Rows": v} for k, v in summary["review_reasons"].items()]
    )
    if not hist.empty:
        st.dataframe(hist, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# 4. vocabulary
# --------------------------------------------------------------------------

with tab_vocab:
    st.markdown("##### What the induction pass learned from this catalogue")
    st.caption(
        "The Solution Guide builds its architecture on seven reference "
        "workbooks — a 27,000-row brand list, a 161,000-row list of values, a "
        "500-entry UOM master. None of them are published on the portal. So the "
        "engine derives its vocabularies from the corpus in front of it, which "
        "also means it works on a distributor's catalogue nobody has curated yet."
    )
    v = summary["vocabulary"]
    cols = st.columns(5)
    for i, (label, key) in enumerate(
        [
            ("Distributor groups", "groups"),
            ("Brands induced", "brands"),
            ("Held for review", "brand_candidates_for_review"),
            ("Product types", "product_types"),
            ("Series names", "series"),
        ]
    ):
        cols[i].metric(label, f"{v.get(key, 0):,}")

    vocab = get_vocab()
    if vocab:
        left, right = st.columns(2)
        with left:
            st.markdown("**Accepted brands** — every one linked to attested evidence")
            frame = pd.DataFrame(
                [
                    {
                        "Brand": e.canonical,
                        "Aliases": ", ".join(sorted(e.aliases)[:4]),
                        "Rows": e.support,
                        "Linked via": e.linkage,
                        "Conf": round(e.confidence, 2),
                    }
                    for e in sorted(vocab.brands.values(), key=lambda x: -x.support)
                ]
            )
            st.dataframe(frame, width="stretch", hide_index=True, height=380)
        with right:
            st.markdown("**Held back for a human** — brand-shaped, but unproven")
            held = pd.DataFrame(
                [
                    {
                        "Token": e.canonical,
                        "Rows": e.support,
                        "Head-weighted": round(e.mean_position, 2),
                        "Why held": e.rejection,
                    }
                    for e in sorted(
                        vocab.brand_candidates.values(), key=lambda x: -x.support
                    )
                ]
            )
            st.dataframe(held, width="stretch", hide_index=True, height=380)
            st.caption(
                "`Decking` and `Grooved` are genuinely distinctive inside their "
                "distributor's rows, which is exactly why distinctiveness alone "
                "cannot promote a token to a brand. These are surfaced, not guessed."
            )

        if vocab.unknown_units:
            st.markdown("**Unit spellings the lexicon did not know**")
            st.caption(
                "Reported for approval rather than silently dropped — this is how "
                "the UOM lexicon grows when a new distributor arrives."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"Spelling": k, "Occurrences": v}
                     for k, v in list(vocab.unknown_units.items())[:40]]
                ),
                width="stretch", hide_index=True, height=240,
            )

# --------------------------------------------------------------------------
# 5. export
# --------------------------------------------------------------------------

with tab_export:
    st.markdown("##### Download the run")
    st.caption(
        "The delivery sheet carries the organisers' 252 headers, in their order, "
        "taken from their own Expected Output sheet at runtime — a renamed or "
        "reordered column is structurally impossible."
    )
    schema = get_schema()
    frame = pd.DataFrame([r["record"] for r in rows], columns=list(schema.headers))

    a, b, c = st.columns(3)
    a.download_button(
        "Delivery sheet (CSV)",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="glassbox_enriched.csv",
        mime="text/csv",
        width="stretch",
    )
    provenance = "\n".join(json.dumps(r["audit"], ensure_ascii=False) for r in rows)
    b.download_button(
        "Provenance sidecar (JSONL)",
        provenance.encode("utf-8"),
        file_name="glassbox_provenance.jsonl",
        mime="application/x-ndjson",
        width="stretch",
    )
    c.download_button(
        "Run report (JSON)",
        json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="glassbox_report.json",
        mime="application/json",
        width="stretch",
    )

    st.markdown("##### Delivery sheet preview")
    preview_cols = [
        "Mfg_Part_Num", "Part_Desc", "MANUFACTURER_NAME", "BRAND_NAME",
        "Classpath", "MOBILE_DESC", "INVOICE_DESC", "SHORT_DESC",
    ]
    st.dataframe(
        frame[[c for c in preview_cols if c in frame.columns]],
        width="stretch", hide_index=True, height=420,
    )
    st.caption(f"{len(frame):,} rows × {len(frame.columns)} columns · "
               f"enriched in {elapsed:.2f}s")
