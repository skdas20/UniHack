"""Brand and manufacturer resolution.

The hardest single case in the pack is gold row 1, and it is instructive:

    Part_Manuf   = "Appliance Dealers Cooperative (APPDE)"
    E1_Brand     = "-- Unbranded --"
    Unilog_Brand = "-- No Unilog Brand --"
    DIB_Brand    = "-- No DIB Brand --"
    Part_Desc    = "PDSH4816AF Dishwasher SS - Display Only"

    BRAND_NAME        = "FRIGIDAIRE®"
    MANUFACTURER_NAME = "Rheem Manufacturing"

Every brand column is a placeholder, and `Part_Manuf` is a *buying cooperative*
rather than a manufacturer. The brand is recoverable only from the part number,
and the manufacturer is not recoverable from the input at all.

So resolution runs as an evidence ladder, and it reports which rung it stopped
on rather than papering over the difference:

1. a populated ``DIB_Brand`` -- clean wherever present (245 of 1000 rows);
2. a populated ``E1_Brand`` -- clean but shoutier (``JAMESHARDIE``, ``TREX``);
3. a brand alias found in the description, via the induced lexicon
   (this is what turns ``Milw`` into ``Milwaukee`` and ``42275BK Kichler
   Ceiling Lt`` into ``Kichler``);
4. the head of the ``Part_Manuf`` account name, but *only* when it matches a
   brand the corpus already attests -- so ``Kichler Lighting (KICLI)`` yields
   ``Kichler`` while ``Appliance Dealers Cooperative (APPDE)`` yields nothing;
5. unresolved, and flagged for review.

``Unilog_Brand`` is never consulted: it is a placeholder in all 1000 rows and
carries exactly zero bits of information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from . import textnorm as T
from .induce import InducedVocabulary, account_name_to_brand_guess, parse_manufacturer
from .provenance import (
    Cell,
    Evidence,
    Provenance,
    Source,
    placeholder_dropped,
    unresolved,
)

# --- registered marks -------------------------------------------------------

#: Brands whose names are registered marks in this catalogue's trade. The gold
#: rows write ``FRIGIDAIRE®`` and ``Whirlpool®``, so the symbol is expected --
#: but only on brands we can actually vouch for. An induced, unattested brand
#: never gets decorated, because asserting someone else's trademark status is
#: not something a pipeline should guess at.
REGISTERED: frozenset[str] = frozenset(
    {
        "3m", "amana", "azek", "bosch", "dewalt", "diablo", "dremel", "feit electric",
        "festool", "first alert", "frigidaire", "ge", "grizzly", "hager",
        "hunter", "hunter fan", "irwin", "james hardie", "jameshardie", "jet",
        "kichler", "kitchenaid", "kreg", "leviton", "lg", "lp smartside",
        "lutron", "makita", "maytag", "milwaukee", "mirka", "philips", "provia",
        "rheem", "samsung", "satco", "senco", "southwire", "speed queen",
        "square d", "timbertech", "trex", "velux", "vessel", "whirlpool",
    }
)

#: Brand -> legal manufacturing entity. Used for ``MANUFACTURER_NAME``, which
#: is a different field from ``BRAND_NAME`` and frequently a different company.
#:
#: Note the gold rows disagree with each other on rigour here: Whirlpool® maps
#: to "Whirlpool Corporation" (correct), while FRIGIDAIRE® maps to "Rheem
#: Manufacturing" -- Frigidaire is an Electrolux brand, and Rheem makes water
#: heaters. The Solution Guide flags this itself: *"at least one row where the
#: manufacturer and brand look mismatched"*. We emit the correct entity and let
#: the audit trail record the disagreement rather than reproducing an error.
BRAND_TO_MANUFACTURER: dict[str, str] = {
    "3m": "3M Company",
    "amana": "Whirlpool Corporation",
    "azek": "The AZEK Company Inc",
    "bosch": "Robert Bosch Tool Corporation",
    "dewalt": "Stanley Black & Decker Inc",
    "diablo": "Freud America Inc",
    "dremel": "Robert Bosch Tool Corporation",
    "feit electric": "Feit Electric Company Inc",
    "festool": "Festool USA",
    "first alert": "Resideo Technologies Inc",
    "frigidaire": "Electrolux Home Products Inc",
    "ge": "GE Appliances",
    "grizzly": "Grizzly Industrial Inc",
    "hager": "Hager Companies",
    "hunter": "Hunter Fan Company",
    "hunter fan": "Hunter Fan Company",
    "irwin": "Stanley Black & Decker Inc",
    "james hardie": "James Hardie Building Products Inc",
    "jameshardie": "James Hardie Building Products Inc",
    "jet": "JPW Industries Inc",
    "kichler": "Kichler Lighting LLC",
    "kitchenaid": "Whirlpool Corporation",
    "kreg": "Kreg Tool Company",
    "leviton": "Leviton Manufacturing Co Inc",
    "lg": "LG Electronics Inc",
    "lp smartside": "Louisiana-Pacific Corporation",
    "lutron": "Lutron Electronics Co Inc",
    "makita": "Makita Corporation of America",
    "maytag": "Whirlpool Corporation",
    "milwaukee": "Milwaukee Electric Tool Corporation",
    "mirka": "Mirka Ltd",
    "philips": "Signify North America Corporation",
    "provia": "ProVia LLC",
    "samsung": "Samsung Electronics America Inc",
    "satco": "Satco Products Inc",
    "senco": "Kyocera Senco Industrial Tools Inc",
    "southwire": "Southwire Company LLC",
    "speed queen": "Alliance Laundry Systems LLC",
    "square d": "Schneider Electric USA Inc",
    "timbertech": "The AZEK Company Inc",
    "trex": "Trex Company Inc",
    "velux": "VELUX America LLC",
    "vessel": "Vessel Industrial Tools Co Ltd",
    "whirlpool": "Whirlpool Corporation",
}

_SYMBOLS = re.compile(r"[®™©]")


def strip_marks(value: str) -> str:
    return _SYMBOLS.sub("", value or "").strip()


def decorate(brand: str, *, attested: bool) -> str:
    """Add the registered mark where we can vouch for it, never otherwise."""
    plain = strip_marks(brand)
    if not plain:
        return ""
    if _SYMBOLS.search(brand or ""):
        return brand.strip()  # already decorated upstream
    if attested and plain.lower() in REGISTERED:
        return f"{plain}®"
    return plain


@dataclass
class BrandResolution:
    """The outcome of the evidence ladder."""

    brand_plain: str = ""
    brand: str = ""  # decorated
    manufacturer: str = ""
    rung: str = ""
    confidence: float = 0.0
    attested: bool = False
    brand_cell: Cell = field(default_factory=Cell)
    manufacturer_cell: Cell = field(default_factory=Cell)
    #: Placeholder values that were deliberately discarded, for the audit view.
    dropped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.brand_plain)


def _brand_cell(
    value: str, rung: str, confidence: float, detail: str,
    *, source: Source, evidence: tuple[Evidence, ...] = ()
) -> Cell:
    return Cell(
        value,
        Provenance(
            source,
            rule=f"brand:{rung}",
            confidence=confidence,
            evidence=evidence,
            detail=detail,
        ),
    )


def resolve_manufacturer(
    brand_plain: str, part_manuf_raw: str, *, attested: bool
) -> Cell:
    """``MANUFACTURER_NAME``: the legal entity behind the brand.

    Falls back to the brand itself, which is what the Solution Guide
    prescribes: *"Where an item has no brand, the manufacturer name is used
    instead."* -- and the converse holds too. Never falls back to the
    `Part_Manuf` account, because a buying cooperative is not a manufacturer.
    """
    key = brand_plain.lower()
    legal = BRAND_TO_MANUFACTURER.get(key)
    if legal:
        return Cell(
            legal,
            Provenance(
                Source.CROSSWALK,
                rule="manufacturer:brand-to-legal-entity",
                confidence=0.9,
                detail=f"{brand_plain} is a brand of {legal}",
            ),
        )

    account_name, code = parse_manufacturer(part_manuf_raw)
    if brand_plain:
        return Cell(
            brand_plain,
            Provenance(
                Source.DERIVED,
                rule="manufacturer:brand-as-manufacturer",
                confidence=0.6 if attested else 0.4,
                detail=(
                    "no legal-entity mapping known for this brand; using the "
                    "brand name per the house rule that the manufacturer name "
                    "stands in when the pair cannot be separated"
                    + (f" (account was {account_name!r})" if account_name else "")
                ),
            ),
        )
    return unresolved(
        "manufacturer:unresolved",
        detail=(
            f"no brand resolved and account {account_name!r} is a distributor "
            "or cooperative, not a manufacturer"
        ),
    )


#: A distributor group must be this pure, over this many rows, before its
#: dominant brand may be applied to a row that resolved no other way.
BRAND_PRIOR_PURITY = 0.60
BRAND_PRIOR_MIN_ROWS = 3


def fit_brand_prior(
    rows, vocab: InducedVocabulary
) -> dict[str, tuple[str, float, int]]:
    """``Part_Manuf`` -> (dominant attested brand, purity, supporting rows).

    Same idea as the taxonomy's distributor prior, applied to brands. A
    ``Part_Manuf`` account almost always carries one manufacturer's line, so
    once some of its rows resolve a brand from a populated column, the rest of
    the group can inherit it. That is exactly what a human merchandiser does
    when a row's brand columns are all placeholders.

    Only *attested* brands vote, so an induced-but-unlinked token can never
    become a whole group's brand.
    """
    from collections import Counter, defaultdict

    votes: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        group = T.clean(row.get("Part_Manuf"))
        if not group or T.is_placeholder(group):
            continue
        for column in ("DIB_Brand", "E1_Brand"):
            raw = row.get(column)
            if T.is_placeholder(raw):
                continue
            value = T.clean(raw)
            if not value or T.is_placeholder(value):
                continue
            entry = vocab.resolve_alias(value)
            canonical = entry.canonical if entry else value
            votes[group][strip_marks(canonical)] += 1

    out: dict[str, tuple[str, float, int]] = {}
    for group, counter in votes.items():
        total = sum(counter.values())
        brand, hits = counter.most_common(1)[0]
        purity = hits / total if total else 0.0
        if purity >= BRAND_PRIOR_PURITY and hits >= BRAND_PRIOR_MIN_ROWS:
            out[group] = (brand, purity, hits)
    return out


def resolve_brand(
    row_get,
    desc: str,
    vocab: InducedVocabulary,
    brand_prior: dict[str, tuple[str, float, int]] | None = None,
) -> BrandResolution:
    """Walk the evidence ladder. ``row_get`` is ``RawRow.get``."""
    result = BrandResolution()
    text = T.clean(desc)

    # record the placeholders we are discarding, so the audit shows the work
    for column in ("E1_Brand", "Unilog_Brand", "DIB_Brand"):
        raw = row_get(column)
        if raw and T.is_placeholder(raw):
            result.dropped.append((column, raw))

    # --- rungs 1 and 2: a populated brand column ---
    for column, rung, confidence in (
        ("DIB_Brand", "dib-column", 0.98),
        ("E1_Brand", "e1-column", 0.95),
    ):
        raw = row_get(column)
        if T.is_placeholder(raw):
            continue
        value = T.clean(raw)
        if not value or T.is_placeholder(value):
            continue
        # normalise to the corpus's canonical spelling when it has one
        entry = vocab.resolve_alias(value)
        canonical = entry.canonical if entry else value
        result.brand_plain = strip_marks(canonical)
        result.attested = True
        result.rung = rung
        result.confidence = confidence
        result.brand = decorate(canonical, attested=True)
        result.brand_cell = _brand_cell(
            result.brand, rung, confidence,
            f"{column} was populated with {value!r}"
            + ("" if canonical == value else f", normalised to {canonical!r}"),
            source=Source.INPUT_COPY if canonical == value else Source.LEXICON_EXACT,
        )
        break

    # --- rung 3: an induced alias inside the description ---
    if not result.ok and vocab.alias_index:
        best: tuple[str, int, int, str] | None = None  # canonical, start, end, alias
        for match in re.finditer(r"[A-Za-z][A-Za-z'\-&]{2,}", text):
            token = match.group(0)
            canonical = vocab.alias_index.get(token.lower())
            if not canonical:
                continue
            # prefer the longest alias, and among equals the earliest -- brand
            # shorthand sits at the head of these descriptions
            if best is None or len(token) > len(best[3]):
                best = (canonical, match.start(), match.end(), token)
        if best is not None:
            canonical, start, end, alias = best
            entry = vocab.brands.get(canonical)
            attested = bool(entry and entry.attested)
            result.brand_plain = strip_marks(canonical)
            result.attested = attested
            result.rung = "description-alias"
            result.confidence = 0.88 if attested else 0.5
            result.brand = decorate(canonical, attested=attested)
            result.brand_cell = _brand_cell(
                result.brand, "description-alias", result.confidence,
                f"alias {alias!r} in the description resolves to {canonical!r} "
                f"via the induced lexicon"
                + (f" ({entry.linkage})" if entry else ""),
                source=Source.LEXICON_EXACT if attested else Source.LEXICON_FUZZY,
                evidence=(Evidence(text, start, end),),
            )

    # --- rung 4: the account-name head, only if the corpus attests it ---
    if not result.ok:
        account_raw = row_get("Part_Manuf")
        guess = account_name_to_brand_guess(account_raw)
        if guess:
            entry = vocab.resolve_alias(guess)
            if entry is None:
                # allow a close attested spelling, e.g. "Phillips" -> "Philips"
                for candidate in vocab.brands.values():
                    if not candidate.attested:
                        continue
                    if fuzz.ratio(guess.lower(), candidate.canonical.lower()) >= 88:
                        entry = candidate
                        break
            if entry is not None and entry.attested:
                result.brand_plain = strip_marks(entry.canonical)
                result.attested = True
                result.rung = "account-name-head"
                result.confidence = 0.7
                result.brand = decorate(entry.canonical, attested=True)
                result.brand_cell = _brand_cell(
                    result.brand, "account-name-head", 0.7,
                    f"account {account_raw!r} reduces to {guess!r}, which the "
                    f"corpus attests as {entry.canonical!r}",
                    source=Source.LEXICON_FUZZY,
                )

    # --- rung 5: the distributor group's dominant brand ---
    if not result.ok and brand_prior:
        group = T.clean(row_get("Part_Manuf"))
        hit = brand_prior.get(group)
        if hit:
            brand_name, purity, supporting = hit
            result.brand_plain = strip_marks(brand_name)
            result.attested = True
            result.rung = "group-brand-prior"
            result.confidence = round(0.6 * purity, 3)
            result.brand = decorate(brand_name, attested=True)
            result.brand_cell = _brand_cell(
                result.brand, "group-brand-prior", result.confidence,
                f"every brand column on this row was a placeholder; account "
                f"{group!r} is {purity:.0%} {brand_name} across {supporting} rows "
                f"whose brand column was populated, so the brand is inherited "
                f"from the group rather than read from this row",
                source=Source.DERIVED,
            )

    # --- rung 6: give up honestly ---
    if not result.ok:
        dropped = ", ".join(f"{c}={v!r}" for c, v in result.dropped) or "none"
        result.rung = "unresolved"
        result.confidence = 0.0
        result.brand_cell = unresolved(
            "brand:unresolved",
            detail=(
                "no brand column populated, no induced alias in the "
                f"description, and the account name is not an attested brand. "
                f"Placeholders discarded: {dropped}"
            ),
        )

    result.manufacturer_cell = resolve_manufacturer(
        result.brand_plain, row_get("Part_Manuf"), attested=result.attested
    )
    result.manufacturer = result.manufacturer_cell.value
    return result
