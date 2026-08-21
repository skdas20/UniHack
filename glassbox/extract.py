"""Filling the attribute contracts from the raw text.

Everything produced here is **EXTRACTED**: it exists in the input string and
carries the character span that proves it. Nothing in this module invents a
value. Model-proposed values arrive later, through :mod:`glassbox.enrich`, and
are kept separately so a reviewer can always tell the two apart.

Several extractors encode trade conventions that a generic number-grabber gets
wrong, and each one was written against a real row of the working dataset:

* ``564922 60W Led BA11 50k 3pk`` -- ``50k`` is 5000 K, not 50 K. Lighting
  writes colour temperature in hundreds.
* ``3M 775L Stikit Film P150`` -- ``P150`` is a 150-grit designation, not a
  model number.
* ``10-4 SO Cord (Linear Foot)`` -- 10 AWG, 4 conductors, SO jacket. Three
  attributes hidden in five characters, none of them labelled.
* ``1nx6-20' ... Decking`` -- the ``n`` marks nominal sizing; thickness and
  width are inches, the trailing number is feet.
* ``49-94-1940 Milw 14"x1/8"x1"`` -- diameter, thickness, arbor, in that order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import textnorm as T
from . import units as U
from .attributes import Contract, Kind, Slot, Style, dim_order
from .provenance import Cell, Evidence, Provenance, Source, contract_blank


@dataclass(slots=True)
class Filled:
    """One resolved attribute slot."""

    slot: Slot
    value: str = ""
    uom: str = ""
    cell: Cell = field(default_factory=Cell)

    def __bool__(self) -> bool:
        return bool(self.value.strip())

    @property
    def label(self) -> str:
        return self.slot.label

    @property
    def with_uom(self) -> str:
        """``"50-1/4"`` + ``"in"`` -> ``"50-1/4 in"``."""
        if self.uom and self.value:
            return f"{self.value} {self.uom}"
        return self.value


@dataclass
class ExtractionContext:
    """Everything the extractors need about one row."""

    text: str  # the cleaned description body used for evidence spans
    mpn: str = ""
    brand: str = ""
    contract_name: str = "generic"
    #: Induced series names, best-first, used by the SERIES extractor.
    series_lexicon: tuple[str, ...] = ()
    #: Lowercased words belonging to any controlled vocabulary in this
    #: contract, used to keep attribute words out of series names.
    lov_words: frozenset = frozenset()
    #: Spans already claimed, so two slots never read the same characters.
    claimed: list[tuple[int, int]] = field(default_factory=list)

    def claim(self, start: int, end: int) -> None:
        self.claimed.append((start, end))

    def is_claimed(self, start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in self.claimed)


# --- helpers ----------------------------------------------------------------


def _evidence(ctx: ExtractionContext, start: int, end: int) -> tuple[Evidence, ...]:
    if 0 <= start <= end <= len(ctx.text):
        return (Evidence(ctx.text, start, end),)
    return ()


def _extracted_cell(
    ctx: ExtractionContext,
    rule: str,
    start: int,
    end: int,
    *,
    confidence: float = 0.92,
    detail: str = "",
) -> Cell:
    return Cell(
        "",  # value is carried on Filled; the cell mirrors it later
        Provenance(
            Source.REGEX_EXTRACT,
            rule=rule,
            confidence=confidence,
            evidence=_evidence(ctx, start, end),
            detail=detail,
        ),
    )


def _in_bounds(slot: Slot, value: float | None) -> bool:
    if value is None:
        return True
    if slot.lo is not None and value < slot.lo:
        return False
    if slot.hi is not None and value > slot.hi:
        return False
    return True


def contract_lov_words(contract: Contract) -> frozenset:
    """Every word appearing in any of a contract's controlled vocabularies."""
    out = set()
    word_re = re.compile(r"[A-Za-z][A-Za-z'-]*")
    for slot in contract.slots:
        for value in slot.lov:
            out.update(w.lower() for w in word_re.findall(value))
        for alias, _canonical in slot.lov_aliases:
            out.update(w.lower() for w in word_re.findall(alias))
        for cue in slot.cues:
            out.update(w.lower() for w in word_re.findall(cue))
    return frozenset(out)


def _trim_lov_words(name: str, lov_words: frozenset) -> str:
    """Strip leading and trailing words that belong to a controlled vocabulary."""
    words = name.split()
    while words and words[0].lower() in lov_words:
        words.pop(0)
    while words and words[-1].lower() in lov_words:
        words.pop()
    return " ".join(words)


def _phrase_re(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", phrase.strip()) if p]
    body = r"[\s\-]*".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?:e?s)?(?![A-Za-z0-9])", re.IGNORECASE)


# --- specialised extractors -------------------------------------------------

#: Lighting colour temperature shorthand: 27k = 2700 K, 50k = 5000 K.
_CCT_SHORT = re.compile(r"(?<![A-Za-z0-9.])(\d{2})\s*[kK](?![A-Za-z0-9])")
#: Explicit form: 2700K, 5000 K.
_CCT_FULL = re.compile(r"(?<![A-Za-z0-9.])([1-6]\d{3})\s*[kK](?![A-Za-z0-9])")
#: Multi-CCT / selectable products state a set rather than a value.
_CCT_SELECTABLE = re.compile(r"\b(multi\s*cct|selectable\s*cct|cct\s*select)\b", re.IGNORECASE)


def extract_color_temperature(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """``50k`` -> ``5000 K``; ``27k`` -> ``2700 K``; ``3000K`` -> ``3000 K``."""
    if _CCT_SELECTABLE.search(ctx.text):
        m = _CCT_SELECTABLE.search(ctx.text)
        return Filled(
            slot,
            "Selectable",
            "",
            _extracted_cell(
                ctx, "cct:selectable", m.start(), m.end(), confidence=0.9,
                detail="product offers multiple colour temperatures",
            ),
        )
    m = _CCT_FULL.search(ctx.text)
    if m and not ctx.is_claimed(m.start(), m.end()):
        ctx.claim(m.start(), m.end())
        return Filled(
            slot, m.group(1), "K",
            _extracted_cell(ctx, "cct:explicit", m.start(), m.end()),
        )
    for m in _CCT_SHORT.finditer(ctx.text):
        if ctx.is_claimed(m.start(), m.end()):
            continue
        hundreds = int(m.group(1))
        kelvin = hundreds * 100
        if not (1500 <= kelvin <= 7000):
            continue
        ctx.claim(m.start(), m.end())
        return Filled(
            slot, str(kelvin), "K",
            _extracted_cell(
                ctx, "cct:hundreds-shorthand", m.start(), m.end(),
                detail=f"lighting shorthand {m.group(0)!r} means {kelvin} K",
            ),
        )
    return None


#: ``P150``, ``150 Grit``, ``220Grit``, ``80 grit``.
_GRIT_P = re.compile(r"(?<![A-Za-z0-9])[Pp](\d{2,4})(?![A-Za-z0-9])")
_GRIT_WORD = re.compile(r"(?<![A-Za-z0-9])(\d{2,4})\s*grit\b", re.IGNORECASE)


def extract_grit(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    for pattern, rule in ((_GRIT_WORD, "grit:explicit"), (_GRIT_P, "grit:p-scale")):
        m = pattern.search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        value = m.group(1)
        if not _in_bounds(slot, float(value)):
            continue
        ctx.claim(m.start(), m.end())
        detail = (
            "FEPA P-scale designation" if rule.endswith("p-scale") else ""
        )
        return Filled(
            slot, value, "grit",
            _extracted_cell(ctx, rule, m.start(), m.end(), detail=detail),
        )
    return None


#: ``10-4 SO``, ``12-2 NM-B``, ``6/6/6 UD``, ``14/3``.
_WIRE_DASH = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,4}(?:/\d)?)\s*[-/]\s*(\d{1,2})(?![\d/])(?![A-Za-z0-9])"
)
_WIRE_TRIPLE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})/(\d{1,2})/(\d{1,2})(?![\d/])")


def extract_wire_spec(ctx: ExtractionContext) -> dict[str, tuple[str, str, int, int, str]]:
    """Decode the ``gauge-conductors`` shorthand used across wire and cord.

    Returns ``{slot_label: (value, uom, start, end, rule)}``.

    ``6/6/6`` is three conductors all of 6 AWG -- the repetition *is* the
    conductor count, which a naive parser reads as the number 6/6/6.
    """
    out: dict[str, tuple[str, str, int, int, str]] = {}
    m = _WIRE_TRIPLE.search(ctx.text)
    if m:
        gauges = {m.group(1), m.group(2), m.group(3)}
        if len(gauges) == 1:
            out["Conductor Size"] = (m.group(1), "AWG", m.start(), m.end(), "wire:triple-gauge")
            out["Number of Conductors"] = ("3", "", m.start(), m.end(), "wire:triple-count")
            return out
    m = _WIRE_DASH.search(ctx.text)
    if m:
        gauge, conductors = m.group(1), m.group(2)
        # A plausible building-wire gauge, and a plausible conductor count.
        try:
            g = float(gauge.split("/")[0])
            c = int(conductors)
        except ValueError:
            return out
        if 1 <= g <= 4000 and 1 <= c <= 60:
            out["Conductor Size"] = (gauge, "AWG", m.start(), m.end(), "wire:gauge-dash")
            out["Number of Conductors"] = (conductors, "", m.start(), m.end(), "wire:conductor-dash")
    return out


#: Nominal trade sizing without units: ``1x6-12'``, ``1nx6-20'``, ``4x4``,
#: ``2.75x30``, ``4x8``. The optional ``n`` marks "nominal".
# A trade magnitude: 4, 2.75, 1/2, or the mixed form 4-1/2. Crucially a bare
# "-<digits>" is NOT part of it: in "1nx6-20'" the "-20" is the board length,
# and letting the width pattern eat it produced a phantom "6-20 in".
# Fraction branch first: with the integer branch leading, "1/8" in
# "4-1/2x1/8x7/8" matches as the bare integer "1" and the size is silently
# wrong. Alternation order is load-bearing here.
_MAG = r"\d+/\d+|\d+(?:\.\d+)?(?:-\d+/\d+)?"

_NOMINAL = re.compile(
    r"(?<![A-Za-z0-9])"
    rf"(?P<a>{_MAG})n?"
    r"\s*[xX]\s*"
    rf"(?P<b>{_MAG})n?"
    # either a third x-separated part (4-1/2x1/8x7/8) ...
    rf"(?:\s*[xX]\s*(?P<d>{_MAG})n?)?"
    # ... or the "-NN'" board-length tail (1nx6-20')
    rf"(?:\s*-\s*(?P<c>{_MAG})\s*'?)?"
    r"(?![A-Za-z0-9])"
)


@dataclass(frozen=True, slots=True)
class NominalSize:
    parts: tuple[str, ...]
    units: tuple[str, ...]
    start: int
    end: int
    raw: str


def find_nominal_size(text: str) -> NominalSize | None:
    """``"1nx6-20'"`` -> parts ``("1","6","20")`` units ``("in","in","ft")``.

    The trailing ``-NN'`` group is the board length in feet; the leading pair is
    nominal thickness and width in inches. This is the single most common size
    idiom in the decking and lumber rows and it carries no unit markers at all.
    """
    m = _NOMINAL.search(text or "")
    if not m:
        return None
    a, b, c, d = m.group("a"), m.group("b"), m.group("c"), m.group("d")
    if d:
        # three x-separated parts are all the same unit
        return NominalSize((a, b, d), ("in", "in", "in"), m.start(), m.end(), m.group(0))
    if c:
        # the trailing "-NN'" group is the board length, in feet
        return NominalSize((a, b, c), ("in", "in", "ft"), m.start(), m.end(), m.group(0))
    return NominalSize((a, b), ("in", "in"), m.start(), m.end(), m.group(0))


#: Pack quantities: ``3pk``, ``6pc``, ``4pc``, ``50 Disc/Box``, ``(BDL)``.
_PACK_SUFFIX = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,4})\s*(pk|pc|pcs|pack|ct|count|ea)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_PACK_PER = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,4})\s*[A-Za-z]+\s*/\s*(?:box|bx|case|cs|pk|pack|bag|roll|rl)\b",
    re.IGNORECASE,
)


def extract_pack_quantity(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    for pattern, rule, uom in (
        (_PACK_PER, "pack:per-container", "pc"),
        (_PACK_SUFFIX, "pack:suffix", ""),
    ):
        m = pattern.search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        value = m.group(1)
        if not _in_bounds(slot, float(value)):
            continue
        unit = uom or U.canonicalise(m.group(2))
        ctx.claim(m.start(), m.end())
        return Filled(
            slot, value, unit,
            _extracted_cell(ctx, rule, m.start(), m.end()),
        )
    return None


#: Driver bit drive sizes: ``#3``, ``T25``, ``PH2``, ``SQ2``.
_DRIVE_SIZE = re.compile(
    r"(?<![A-Za-z0-9])(#\s?\d|T\d{2}|PH\s?\d|SQ\s?\d|R\d)(?![A-Za-z0-9])"
)


def extract_drive_size(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    m = _DRIVE_SIZE.search(ctx.text)
    if not m or ctx.is_claimed(m.start(), m.end()):
        return None
    ctx.claim(m.start(), m.end())
    value = re.sub(r"\s+", "", m.group(1)).upper()
    return Filled(slot, value, "", _extracted_cell(ctx, "bit:drive-size", m.start(), m.end()))


#: Gang counts on wiring devices: ``2G``, ``1 Gang``.
_GANG = re.compile(r"(?<![A-Za-z0-9])(\d)\s*(?:G|gang)(?![A-Za-z0-9])", re.IGNORECASE)


def extract_gangs(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    m = _GANG.search(ctx.text)
    if not m or ctx.is_claimed(m.start(), m.end()):
        return None
    if not _in_bounds(slot, float(m.group(1))):
        return None
    ctx.claim(m.start(), m.end())
    return Filled(slot, m.group(1), "", _extracted_cell(ctx, "device:gang-count", m.start(), m.end()))


# --- generic extractors -----------------------------------------------------


def extract_measure(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """First unclaimed measurement of the slot's family, within bounds.

    Cue phrases take priority: a slot named ``Depth With Door Open`` prefers a
    measurement adjacent to the words "door open" over the first length in the
    string.
    """
    measures = [m for m in U.find_measurements(ctx.text) if m.family == slot.family]
    if not measures:
        return None

    if slot.cues:
        for cue in slot.cues:
            cue_match = _phrase_re(cue).search(ctx.text)
            if not cue_match:
                continue
            near = sorted(
                measures,
                key=lambda m: min(
                    abs(m.start - cue_match.end()), abs(cue_match.start() - m.end)
                ),
            )
            for m in near:
                if ctx.is_claimed(m.start, m.end) or not _in_bounds(slot, m.numeric):
                    continue
                ctx.claim(m.start, m.end)
                return Filled(
                    slot, m.magnitude, m.unit.canonical,
                    _extracted_cell(
                        ctx, f"measure:{slot.family}:cue({cue})", m.start, m.end
                    ),
                )

    for m in measures:
        if ctx.is_claimed(m.start, m.end) or not _in_bounds(slot, m.numeric):
            continue
        ctx.claim(m.start, m.end)
        return Filled(
            slot, m.magnitude, m.unit.canonical,
            _extracted_cell(ctx, f"measure:{slot.family}", m.start, m.end,
                            confidence=0.85),
        )
    return None


def extract_lov(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """Match one of the slot's permitted values, or a declared alias of one.

    Longest candidate first, so ``Black Stainless Steel`` wins over ``Black``
    and ``Square Edge`` wins over ``Square``.
    """
    candidates: list[tuple[str, str]] = [(v, v) for v in slot.lov]
    candidates += [(alias, canonical) for alias, canonical in slot.lov_aliases
                   if canonical in slot.lov]
    candidates.sort(key=lambda pair: -len(pair[0]))

    for surface, canonical in candidates:
        m = _phrase_re(surface).search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        ctx.claim(m.start(), m.end())
        exact = surface.lower() == canonical.lower()
        return Filled(
            slot, canonical, "",
            Cell(
                "",
                Provenance(
                    Source.LEXICON_EXACT if exact else Source.LEXICON_FUZZY,
                    rule=f"lov:{slot.label}",
                    confidence=0.95 if exact else 0.88,
                    evidence=_evidence(ctx, m.start(), m.end()),
                    detail=(
                        "" if exact
                        else f"alias {surface!r} normalised to LOV value {canonical!r}"
                    ),
                ),
            ),
        )
    return None


def extract_count(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """A bare integer bound to one of the slot's cue words: ``5 Wash Cycles``."""
    for cue in slot.cues:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])(\d{{1,4}})\s*[-]?\s*(?:{re.escape(cue)})(?:e?s)?"
            rf"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        m = pattern.search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        if not _in_bounds(slot, float(m.group(1))):
            continue
        ctx.claim(m.start(), m.end())
        return Filled(
            slot, m.group(1), "",
            _extracted_cell(ctx, f"count:{cue}", m.start(), m.end()),
        )
    return None


def extract_series(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """Match an induced series name, longest first.

    Induced series names bleed attribute words. The corpus yields
    ``Pebble Beach Grooved`` and ``Phillips Drive``, where ``Grooved`` is an
    edge profile and ``Phillips`` a drive type -- both already covered by their
    own LOV slots. Any word belonging to a controlled vocabulary in *this*
    contract is therefore trimmed off the ends before the name is accepted,
    which leaves ``Pebble Beach`` and rejects ``Phillips Drive`` entirely once
    both of its words are stripped.
    """
    for name in sorted(ctx.series_lexicon, key=len, reverse=True):
        if len(name) < 4:
            continue
        m = _phrase_re(name).search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        trimmed = _trim_lov_words(name, ctx.lov_words)
        if not trimmed or len(trimmed.split()) < 2:
            continue  # nothing distinctive survived the trim
        ctx.claim(m.start(), m.end())
        name = trimmed
        return Filled(
            slot, name, "",
            Cell(
                "",
                Provenance(
                    Source.LEXICON_EXACT,
                    rule="series:induced-lexicon",
                    confidence=0.82,
                    evidence=_evidence(ctx, m.start(), m.end()),
                    detail="series name induced from the corpus, not hand-listed",
                ),
            ),
        )
    return None


def extract_text_cue(ctx: ExtractionContext, slot: Slot) -> Filled | None:
    """Whichever cue phrase is present becomes the value. Used for free slots."""
    for cue in slot.cues:
        m = _phrase_re(cue).search(ctx.text)
        if not m or ctx.is_claimed(m.start(), m.end()):
            continue
        ctx.claim(m.start(), m.end())
        return Filled(
            slot, T.title_case(cue), "",
            _extracted_cell(ctx, f"cue:{cue}", m.start(), m.end(), confidence=0.8),
        )
    return None


# --- dimensions -------------------------------------------------------------


def assign_dimensions(ctx: ExtractionContext, contract: Contract) -> dict[str, Filled]:
    """Map a dimension run onto the category's ordered dimension slots.

    Handles both idioms: a unit-bearing run (``14"x1/8"x1"``) and a bare
    nominal run (``1nx6-20'``). Also fills the ``Size`` slot with the whole
    run rendered as a phrase, which is what the gold rows put there.
    """
    out: dict[str, Filled] = {}

    group = U.find_dimension_group(ctx.text)
    parts: list[tuple[str, str]] = []
    span = (0, 0)
    rule = ""

    if group is not None:
        measures, start, end = group
        if not ctx.is_claimed(start, end):
            parts = [(m.magnitude, m.unit.canonical) for m in measures]
            span = (start, end)
            rule = "dimension:unit-bearing-run"

    if not parts:
        nominal = find_nominal_size(ctx.text)
        if nominal is not None and not ctx.is_claimed(nominal.start, nominal.end):
            parts = list(zip(nominal.parts, nominal.units))
            span = (nominal.start, nominal.end)
            rule = "dimension:nominal-trade-size"

    if not parts:
        return out

    order = dim_order(contract.name, len(parts))
    if not order:
        return out

    ctx.claim(*span)
    for (magnitude, unit), label in zip(parts, order):
        slot = contract.slot(label)
        if slot is None:
            continue
        value = U.to_fraction(magnitude) if unit in {"in", "ft"} else magnitude
        out[label] = Filled(
            slot, value, unit,
            _extracted_cell(
                ctx, f"{rule}:{label}", *span,
                confidence=0.88,
                detail=(
                    f"position {order.index(label) + 1} of {len(parts)} in a "
                    f"{contract.name} dimension run -> {label}"
                ),
            ),
        )

    size_slot = contract.slot("Size")
    if size_slot is not None:
        rendered = " x ".join(
            f"{m} {u}".strip() if u else m
            for m, u in ((U.to_fraction(mm) if uu in {"in", "ft"} else mm, uu)
                         for mm, uu in parts)
        )
        out["Size"] = Filled(
            size_slot, rendered, "",
            _extracted_cell(ctx, f"{rule}:Size", *span, confidence=0.9),
        )
    return out


# --- the driver -------------------------------------------------------------

#: Slots whose extraction needs a bespoke routine rather than the generic one.
_SPECIALISED = {
    "Color Temperature": extract_color_temperature,
    "Grit": extract_grit,
    "Pack Quantity": extract_pack_quantity,
    "Drive Size": extract_drive_size,
    "Number of Gangs": extract_gangs,
}


def extract_contract(
    ctx: ExtractionContext, contract: Contract
) -> dict[str, Filled]:
    """Fill as many of the contract's slots as the text supports.

    Order matters. Dimensions are assigned first because they consume the
    largest, least ambiguous spans; wire shorthand next because it is dense and
    unlabelled; then specialised extractors; then the generic ones. Every
    extractor claims the characters it used, so no two slots can be filled from
    the same evidence.
    """
    filled: dict[str, Filled] = {}

    # 1. dimension runs
    filled.update(assign_dimensions(ctx, contract))

    # 2. wire shorthand, if this category has the slots for it
    if contract.slot("Conductor Size") is not None:
        for label, (value, uom, start, end, rule) in extract_wire_spec(ctx).items():
            slot = contract.slot(label)
            if slot is None or label in filled or ctx.is_claimed(start, end):
                continue
            filled[label] = Filled(
                slot, value, uom,
                _extracted_cell(ctx, rule, start, end, confidence=0.87),
            )
        for label in ("Conductor Size", "Number of Conductors"):
            if label in filled:
                f = filled[label]
                for ev in f.cell.prov.evidence:
                    ctx.claim(ev.start, ev.end)

    # 3. specialised, then generic. Series is deferred to a final pass so every
    #    LOV slot claims its characters first -- otherwise a greedy induced
    #    series name swallows the attribute words sitting next to it.
    for slot in contract.slots:
        if slot.label in filled or slot.kind is Kind.SERIES:
            continue
        special = _SPECIALISED.get(slot.label)
        result: Filled | None = None
        if special is not None:
            result = special(ctx, slot)
        elif slot.kind is Kind.MEASURE:
            result = extract_measure(ctx, slot)
        elif slot.kind is Kind.LOV:
            result = extract_lov(ctx, slot)
        elif slot.kind is Kind.COUNT:
            result = extract_count(ctx, slot)
        elif slot.kind is Kind.SERIES:
            result = extract_series(ctx, slot)
        elif slot.kind is Kind.TEXT and slot.cues:
            result = extract_text_cue(ctx, slot)
        if result is not None and result.value:
            filled[slot.label] = result

    # 4. series, now that the attribute slots have taken their spans
    for slot in contract.slots:
        if slot.kind is Kind.SERIES and slot.label not in filled:
            series = extract_series(ctx, slot)
            if series is not None and series.value:
                filled[slot.label] = series

    # 5. every remaining slot is an explicit contract blank, not an omission
    for slot in contract.slots:
        if slot.label not in filled:
            filled[slot.label] = Filled(slot, "", "", contract_blank(slot.label))

    return filled


def coverage(filled: dict[str, Filled]) -> float:
    """Share of contract slots that carry a value."""
    if not filled:
        return 0.0
    return sum(1 for f in filled.values() if f) / len(filled)
