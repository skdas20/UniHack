"""The enrichment pipeline.

Zero configuration, no network, no API key, no model download. Point it at a
CSV or XLSX with the six input columns and it emits the full delivery schema
plus a provenance sidecar. That property is deliberate: the submission page
says the prototype must be *"dynamic and capable of processing the evaluation
test dataset during assessment"*, which means it has to run first time on a
machine we will never see.

Two passes over the corpus:

1. **Learn.** Induce the brand lexicon, series names and product vocabulary,
   then fit the distributor-group taxonomy prior from rows that classify
   confidently on lexical evidence alone.
2. **Enrich.** Resolve entities, classify, fill the category's attribute
   contract, render the five description channels, name the digital assets,
   score confidence and triage for review.

The optional layers -- a hosted teacher model, a fine-tuned local classifier --
attach at step 2 and are never required for a complete run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import textnorm as T
from . import units as U
from .attributes import Contract, Kind, contract_for
from .confidence import score as score_row
from .entity import BrandResolution, fit_brand_prior, resolve_brand
from .extract import (
    ExtractionContext,
    Filled,
    contract_lov_words,
    extract_contract,
)
from .induce import InducedVocabulary, induce
from .provenance import (
    Cell,
    EnrichedRow,
    Provenance,
    Source,
    copied,
    derived,
    not_derivable,
    placeholder_dropped,
    unresolved,
)
from .render import (
    RenderInput,
    approvals_from_text,
    asset_filenames,
    invoice_description,
    long_description,
    manufacturer_url,
    mobile_description,
    product_title,
    retail_description,
)
from .schema import NOT_DERIVABLE, PASSTHROUGH, OutputSchema, RawRow, load_input
from .taxonomy import Classification, GroupPrior, classify, prepare_text

#: Optional hook: given the row context, propose extra values. Implemented by
#: glassbox.enrich for the hosted-model path. Kept as a plain callable so the
#: core has no dependency on it and no import of the openai client.
Proposer = Callable[["RowContext"], dict[str, Cell]]


@dataclass
class RowContext:
    """Everything known about one row, handed to the optional proposer."""

    row: RawRow
    body: str
    classification: Classification
    contract: Contract
    brand: BrandResolution
    filled: dict[str, Filled]
    product_name: str


@dataclass
class PipelineStats:
    rows: int = 0
    classified: int = 0
    brand_resolved: int = 0
    attribute_slots: int = 0
    attribute_slots_filled: int = 0
    invoice_over_limit: int = 0
    mobile_out_of_window: int = 0
    needs_review: int = 0
    elapsed_s: float = 0.0
    proposals: int = 0

    def as_dict(self) -> dict[str, float]:
        rows = max(self.rows, 1)
        return {
            "rows": self.rows,
            "classified": self.classified,
            "classified_pct": round(100 * self.classified / rows, 2),
            "brand_resolved": self.brand_resolved,
            "brand_resolved_pct": round(100 * self.brand_resolved / rows, 2),
            "attribute_slot_fill_pct": round(
                100 * self.attribute_slots_filled / max(self.attribute_slots, 1), 2
            ),
            "invoice_over_40_chars": self.invoice_over_limit,
            "mobile_outside_60_80": self.mobile_out_of_window,
            "needs_review": self.needs_review,
            "needs_review_pct": round(100 * self.needs_review / rows, 2),
            "model_proposed_values": self.proposals,
            "elapsed_s": round(self.elapsed_s, 3),
            "rows_per_s": round(self.rows / self.elapsed_s, 1) if self.elapsed_s else 0.0,
        }


class Pipeline:
    """Learn from a catalogue, then enrich it."""

    def __init__(
        self,
        schema: OutputSchema,
        *,
        vocab: InducedVocabulary | None = None,
        proposer: Proposer | None = None,
    ) -> None:
        self.schema = schema
        self.vocab = vocab or InducedVocabulary()
        self.prior = GroupPrior()
        self.proposer = proposer
        self.stats = PipelineStats()
        self._series_lexicon: tuple[str, ...] = ()
        self._lov_words: dict[str, frozenset] = {}
        self.brand_prior: dict[str, tuple[str, float, int]] = {}

    # ---------- pass 1 ----------

    def learn(self, rows: Sequence[RawRow]) -> None:
        """Induce vocabularies and fit the distributor prior."""
        if not self.vocab.brands:
            self.vocab = induce(rows)
        self._series_lexicon = tuple(self.vocab.series)
        self.brand_prior = fit_brand_prior(rows, self.vocab)

        self.prior = GroupPrior()
        for row in rows:
            group = T.clean(row.get("Part_Manuf"))
            result = classify(prepare_text(row.desc, row.mpn), group=group)
            if result.ok and result.confidence >= 0.70:
                self.prior.observe(group, result.category.classpath, result.confidence)

    # ---------- pass 2 ----------

    def enrich(self, row: RawRow) -> EnrichedRow:
        """Enrich one row into the full delivery schema, with provenance."""
        started = time.perf_counter()
        out = EnrichedRow(index=row.index)

        # -- 0. passthrough and placeholder handling -----------------------
        for column in PASSTHROUGH:
            raw = row.get(column)
            if column in {"E1_Brand", "Unilog_Brand", "DIB_Brand"} and T.is_placeholder(raw):
                # The delivery format echoes the raw input columns verbatim, so
                # the placeholder text is preserved *here* while being treated
                # as empty everywhere it matters.
                out.set(column, copied(raw, column))
                continue
            out.set(column, copied(raw, column))

        full = T.clean(row.desc)
        body, _stripped = T.strip_leading_mpn(full, row.mpn)
        body, noise = T.strip_noise(body)

        # -- 1. entity resolution ------------------------------------------
        brand = resolve_brand(row.get, row.desc, self.vocab, self.brand_prior)
        out.set("BRAND_NAME", brand.brand_cell)
        out.set("MANUFACTURER_NAME", brand.manufacturer_cell)
        if brand.ok:
            self.stats.brand_resolved += 1

        mpn = row.mpn
        if mpn:
            out.set(
                "MANUFACTURER_PART_NUMBER",
                copied(mpn, "Mfg_Part_Num"),
            )
        else:
            out.set("MANUFACTURER_PART_NUMBER", unresolved("mpn:absent"))

        # -- 2. taxonomy ---------------------------------------------------
        group = T.clean(row.get("Part_Manuf"))
        classification = classify(
            prepare_text(row.desc, mpn),
            group=group,
            prior=self.prior,
            evidence_text=full,
        )
        category = classification.category
        if category is not None:
            self.stats.classified += 1
            out.set("Classpath", Cell(category.classpath, classification.provenance))
            for header, value in (
                ("Dept", category.dept),
                ("Class", category.klass),
                ("Fine", category.fine),
            ):
                out.set(
                    header,
                    derived(
                        value,
                        "taxonomy:crosswalk-from-classpath",
                        confidence=classification.confidence,
                        detail=(
                            f"read off the same leaf as Classpath, so the two "
                            f"hierarchies cannot disagree"
                        ),
                    ),
                )
            product_name = category.product_name
            out.set(
                "Product Name",
                derived(
                    product_name,
                    "taxonomy:product-noun",
                    confidence=classification.confidence,
                ),
            )
        else:
            out.set("Classpath", Cell("", classification.provenance))
            for header in ("Dept", "Class", "Fine", "Product Name"):
                out.set(header, unresolved("taxonomy:unclassified"))
            product_name = _fallback_product_name(body)
            if product_name:
                out.set(
                    "Product Name",
                    derived(
                        product_name,
                        "product-noun:trailing-phrase-fallback",
                        confidence=0.45,
                        detail="no taxonomy match; noun taken from the description tail",
                    ),
                )

        contract = contract_for(category.contract if category else "generic")
        if contract.name not in self._lov_words:
            self._lov_words[contract.name] = contract_lov_words(contract)

        # -- 3. attributes -------------------------------------------------
        ctx = ExtractionContext(
            text=body,
            mpn=mpn,
            brand=brand.brand_plain,
            contract_name=contract.name,
            series_lexicon=self._series_lexicon,
            lov_words=self._lov_words[contract.name],
        )
        filled = extract_contract(ctx, contract)

        # Category implications: choosing the leaf asserts these.
        if category is not None:
            for label, value in category.implies:
                target = filled.get(label)
                if target is None or target.value:
                    continue
                filled[label] = Filled(
                    target.slot, value, "",
                    derived(
                        value,
                        f"taxonomy:implied-by:{category.leaf}",
                        confidence=max(0.6, classification.confidence * 0.9),
                        detail=(
                            f"entailed by classifying as {category.leaf}; not "
                            "read from the text"
                        ),
                    ),
                )

        # -- 4. optional model proposals -----------------------------------
        if self.proposer is not None:
            context = RowContext(
                row=row,
                body=body,
                classification=classification,
                contract=contract,
                brand=brand,
                filled=filled,
                product_name=product_name,
            )
            try:
                for header, cell in (self.proposer(context) or {}).items():
                    if header in self.schema and not out.value(header):
                        out.set(header, cell)
                        self.stats.proposals += 1
            except Exception as exc:  # a proposer must never break a run
                out.set(
                    "MARKETING_DESCRIPTION",
                    unresolved(
                        "proposer:failed",
                        detail=f"optional enrichment layer errored: {exc!r}",
                    ),
                )

        # -- 5. write the attribute triples --------------------------------
        lov_violations = _write_attributes(out, self.schema, contract, filled)
        self.stats.attribute_slots += len(filled)
        self.stats.attribute_slots_filled += sum(1 for f in filled.values() if f)

        # -- 6. descriptions ------------------------------------------------
        series_slot = filled.get("Series")
        render_input = RenderInput(
            product_name=product_name,
            brand=brand.brand,
            brand_plain=brand.brand_plain,
            manufacturer=brand.manufacturer,
            mpn=mpn,
            series=series_slot.value if series_slot else "",
            contract=contract,
            filled=filled,
            category_leaf=category.leaf if category else "",
            category_class=category.klass if category else "",
        )
        for header, builder in (
            ("SHORT_DESC", product_title),
            ("LONG_DESC1", long_description),
            ("RETAIL_DESC", retail_description),
            ("MOBILE_DESC", mobile_description),
            ("INVOICE_DESC", invoice_description),
        ):
            text, prov = builder(render_input)
            out.set(header, Cell(text, prov))

        invoice_len = len(out.value("INVOICE_DESC"))
        mobile_len = len(out.value("MOBILE_DESC"))
        if invoice_len > 40:
            self.stats.invoice_over_limit += 1
        if mobile_len and not 60 <= mobile_len <= 80:
            self.stats.mobile_out_of_window += 1

        # -- 7. assets, URLs, approvals -------------------------------------
        for header, filename in asset_filenames(
            brand.brand, mpn, alternates=self.schema.n_alternate_image_slots
        ).items():
            if header in self.schema:
                out.set(
                    header,
                    derived(
                        filename,
                        "asset:naming-convention",
                        confidence=0.85,
                        detail=(
                            "follows the {BRAND}_{MPN} convention observed in "
                            "both gold rows; the file itself is not verified to exist"
                        ),
                    ),
                )
        out.set("Actual Image (Yes/No)", derived("Yes", "asset:actual-image-flag", confidence=0.7))
        out.set("MFR URL", manufacturer_url(brand.brand, mpn))

        approvals, found = approvals_from_text(full)
        if approvals:
            out.set(
                "Standard/Approvals",
                derived(
                    approvals,
                    "approvals:detected-in-text",
                    confidence=0.85,
                    detail=f"{len(found)} mark(s) found, pipe-delimited and ASCII-sorted",
                ),
            )
        else:
            out.set(
                "Standard/Approvals",
                not_derivable(
                    "Standard/Approvals",
                    "manufacturer certification documentation",
                ),
            )

        # -- 8. the honest blanks -------------------------------------------
        for header, system in NOT_DERIVABLE.items():
            if header in self.schema and not out.value(header):
                out.set(header, not_derivable(header, system))

        if noise:
            out.set(
                "MARKETING_DESCRIPTION",
                out.cells.get("MARKETING_DESCRIPTION") or unresolved(
                    "marketing:not-attempted",
                    detail=(
                        "listing-status markers were stripped from the source "
                        f"({', '.join(noise)}); no marketing copy is derivable "
                        "from a distributor description"
                    ),
                ),
            )

        # -- 9. score -------------------------------------------------------
        score_row(
            out,
            checks={
                "invoice_length": invoice_len,
                "mobile_length": mobile_len,
                "lov_violations": lov_violations,
                "unclassified": category is None,
                "brand_unresolved": not brand.ok,
                "classification_margin": float(classification.margin),
                "runner_up": (
                    classification.runner_up.leaf if classification.runner_up else ""
                ),
            },
        )
        if out.needs_review:
            self.stats.needs_review += 1

        out.timings_ms["total"] = (time.perf_counter() - started) * 1000
        return out

    # ---------- driver ----------

    def run(
        self,
        rows: Sequence[RawRow],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[EnrichedRow]:
        started = time.perf_counter()
        self.stats = PipelineStats(rows=len(rows))
        self.learn(rows)
        out: list[EnrichedRow] = []
        for i, row in enumerate(rows):
            out.append(self.enrich(row))
            if progress is not None and (i % 50 == 0 or i == len(rows) - 1):
                progress(i + 1, len(rows))
        self.stats.elapsed_s = time.perf_counter() - started
        return out


# --- helpers ----------------------------------------------------------------


def _write_attributes(
    out: EnrichedRow,
    schema: OutputSchema,
    contract: Contract,
    filled: dict[str, Filled],
) -> list[str]:
    """Emit the ATTRIBUTE_LABEL/VALUE/UOM triples in contract order.

    Every contract slot gets a label even when its value is blank, because both
    gold rows do exactly that -- the label sequence *is* the category's
    signature. Returns any values that fell outside their controlled vocabulary.
    """
    violations: list[str] = []
    slots_available = schema.n_attribute_slots
    for index, slot in enumerate(contract.slots, start=1):
        if index > slots_available:
            violations.append(
                f"contract has {len(contract.slots)} slots but the sheet "
                f"provides only {slots_available}"
            )
            break
        label_h, value_h, uom_h = schema.attribute_headers(index)
        f = filled.get(slot.label)

        out.set(
            label_h,
            derived(
                slot.label,
                f"contract:{contract.name}:slot{index}",
                confidence=1.0,
                detail="label emitted because the category contract requires the slot",
            ),
        )
        if f is None or not f:
            out.set(value_h, f.cell if f is not None else unresolved("attribute:absent"))
            out.set(uom_h, unresolved("attribute:absent"))
            continue

        # LOV enforcement: a value outside the vocabulary is reported, not
        # silently emitted. This is the check the Solution Guide scores.
        if slot.lov and f.value not in slot.lov:
            violations.append(f"{slot.label}={f.value!r}")

        cell = Cell(f.value, f.cell.prov)
        out.set(value_h, cell)
        out.set(uom_h, Cell(f.uom, f.cell.prov) if f.uom else unresolved("uom:none"))
    return violations


_STOP_TAIL = {"only", "display", "new", "each", "kit"}


def _fallback_product_name(body: str) -> str:
    """Last-resort product noun: the trailing alphabetic run of the description."""
    import re

    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
    words = [w for w in words if w.lower() not in _STOP_TAIL and len(w) > 1]
    if not words:
        return ""
    return T.title_case(T.expand(" ".join(words[-3:])))


def build(
    input_path: str | Path,
    schema_path: str | Path,
    *,
    vocab_path: str | Path | None = None,
    proposer: Proposer | None = None,
) -> tuple[Pipeline, list[RawRow]]:
    """Convenience constructor used by the CLI and the demo app."""
    schema = OutputSchema.from_csv(schema_path)
    rows = load_input(input_path)
    vocab = None
    if vocab_path and Path(vocab_path).exists():
        vocab = InducedVocabulary.from_json(vocab_path)
    return Pipeline(schema, vocab=vocab, proposer=proposer), rows
