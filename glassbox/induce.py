"""Vocabulary induction: learn the controlled vocabularies from the corpus.

## Why this module exists

The Solution Guide builds its whole architecture on seven reference workbooks --
a 27,000-row approved manufacturer/brand list, a 161,000-row cross-category List
of Values, a 500-entry UOM master, and so on. **None of them are published on
the portal.** The Resources tab ships the Solution Guide, the 1,000-row input,
and a header sheet with two worked rows, and then says quietly:

> *"The relevant information from these references is already represented
> within the columns of the provided datasets."*

So we take that literally and *induce* the vocabularies from the data.

That turns a missing-file problem into the strongest property of the
submission. A pipeline that needs a hand-maintained 27,000-row dictionary can
only ever process catalogues someone has already curated. A pipeline that
derives its lexicon from the corpus in front of it can be pointed at a new
distributor's export on day one -- which is the actual enterprise problem
Unilog has.

## How it works

The signal is that `Part_Manuf` **groups** the catalogue, and within a group the
brand shorthand is nearly constant while the rest of the description varies:

    Milwaukee Accessory (4031)  ->  "49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc"
                                    "49-94-0445 Milw 10"x3/32"x5/8" Performance+ ..."
                                    "49-94-0501 Milw 4"x1/4"x5/8" Metal Grinding Wheel"

`Milw` is frequent inside the group and absent everywhere else. That is a
textbook distinctiveness signal, so we score every token with

    salience(t, g) = P(t | g) * log(G / groups_containing(t))

which is group-conditional TF-IDF, and keep the tokens that clear a threshold.
The surviving tokens are candidate brand aliases; they are then linked to a
canonical brand using, in order of preference, the group's own clean brand
columns, a fuzzy match against brands seen elsewhere in the corpus, and finally
the alias itself.

The same corpus pass also induces product types (trailing head-noun phrases),
series names, per-category attribute value vocabularies, and -- importantly --
a list of *unknown unit spellings*, which are reported rather than silently
dropped so a human can approve them into the lexicon.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

from rapidfuzz import fuzz, process

from . import textnorm as T
from . import units as U
from .schema import RawRow

# --- tuning -----------------------------------------------------------------

#: A token must cover at least this share of its group's rows to be a
#: candidate brand alias. 0.25 keeps "Milw" (near-universal in its group) and
#: "Trex" while rejecting incidental product words.
MIN_GROUP_COVERAGE = 0.25

#: ...and must appear in at most this many distinct groups to count as
#: distinctive. Brands do leak across distributors (Philips is sold by more
#: than one), so this is 3 rather than 1.
MAX_GROUPS_FOR_ALIAS = 3

#: Minimum salience score to accept.
MIN_SALIENCE = 0.08

#: Fuzzy threshold for linking an alias to a canonical brand name.
BRAND_LINK_THRESHOLD = 82

#: Words that are never brands, however distinctive they look. Kept small and
#: purely structural -- product nouns are excluded by the product-type pass, not
#: by a hand-written blocklist.
_NEVER_BRAND = frozenset(
    {
        "the", "and", "for", "with", "without", "new", "old", "each", "per",
        "kit", "set", "pack", "box", "case", "roll", "pair", "assembly",
        "only", "display", "model", "series", "type", "size", "color", "colour",
        "left", "right", "front", "back", "upper", "lower", "inner", "outer",
        "small", "medium", "large", "long", "short", "wide", "narrow", "deep",
        "black", "white", "brown", "grey", "gray", "silver", "gold", "clear",
    }
)

_MANUF_CODE_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<code>[A-Z0-9]{3,8})\)\s*$")


# --- results ----------------------------------------------------------------


@dataclass
class BrandEntry:
    """One induced brand, with the evidence that produced it."""

    canonical: str
    aliases: list[str] = field(default_factory=list)
    #: `Part_Manuf` groups this brand was observed in.
    groups: list[str] = field(default_factory=list)
    #: Rows supporting the entry.
    support: int = 0
    #: How the canonical form was decided.
    linkage: str = ""
    confidence: float = 0.0
    #: True when the canonical spelling came from a clean brand column rather
    #: than from a guess, and can therefore carry a registered-mark symbol.
    attested: bool = False
    #: Mean normalised position of the alias within its description body.
    #: Brand shorthand sits at the head (``Milw 14"x1/8"...``); product nouns
    #: sit at the tail (``... Grooved Decking``). A cheap, strong separator.
    mean_position: float = 0.0
    #: Set on review candidates: why the entry was not auto-accepted.
    rejection: str = ""


@dataclass
class ProductTypeEntry:
    canonical: str
    support: int = 0
    #: Manufacturer groups it occurs in, for the classifier's priors.
    groups: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)


@dataclass
class InducedVocabulary:
    """Everything learned from one pass over a catalogue."""

    n_rows: int = 0
    brands: dict[str, BrandEntry] = field(default_factory=dict)
    #: Induced aliases that look brand-shaped but could not be linked to any
    #: attested spelling. These are *not* used for enrichment; they are routed
    #: to the human-review queue. Keeping them separate is what stops product
    #: nouns like "Decking" and "Grooved" -- which are genuinely distinctive
    #: within their distributor group -- from being promoted to brands.
    brand_candidates: dict[str, BrandEntry] = field(default_factory=dict)
    #: lowercased alias -> canonical brand
    alias_index: dict[str, str] = field(default_factory=dict)
    product_types: dict[str, ProductTypeEntry] = field(default_factory=dict)
    series: dict[str, int] = field(default_factory=dict)
    #: manufacturer group -> parsed (name, code)
    groups: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Unit spellings seen in the corpus that the canonical table did not know.
    #: Surfaced for human approval instead of being dropped.
    unknown_units: dict[str, int] = field(default_factory=dict)
    #: Measurement families observed per product type -> drives the attribute
    #: contract when no curated contract exists for a category.
    observed_families: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Free-text tokens that look like colour/material/finish values.
    value_vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    # --- lookup API used by the rest of the pipeline ---

    def resolve_alias(self, token: str) -> BrandEntry | None:
        canonical = self.alias_index.get(token.strip().lower())
        return self.brands.get(canonical) if canonical else None

    def canonical_brands(self) -> list[str]:
        return sorted(self.brands)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_rows": self.n_rows,
            "brands": {k: asdict(v) for k, v in self.brands.items()},
            "brand_candidates": {k: asdict(v) for k, v in self.brand_candidates.items()},
            "alias_index": self.alias_index,
            "product_types": {k: asdict(v) for k, v in self.product_types.items()},
            "series": self.series,
            "groups": self.groups,
            "unknown_units": self.unknown_units,
            "observed_families": self.observed_families,
            "value_vocab": self.value_vocab,
            "stats": self.stats,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "InducedVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls(
            n_rows=payload.get("n_rows", 0),
            alias_index=payload.get("alias_index", {}),
            series=payload.get("series", {}),
            groups=payload.get("groups", {}),
            unknown_units=payload.get("unknown_units", {}),
            observed_families=payload.get("observed_families", {}),
            value_vocab=payload.get("value_vocab", {}),
            stats=payload.get("stats", {}),
        )
        vocab.brands = {k: BrandEntry(**v) for k, v in payload.get("brands", {}).items()}
        vocab.brand_candidates = {
            k: BrandEntry(**v) for k, v in payload.get("brand_candidates", {}).items()
        }
        vocab.product_types = {
            k: ProductTypeEntry(**v) for k, v in payload.get("product_types", {}).items()
        }
        return vocab


# --- helpers ----------------------------------------------------------------


def parse_manufacturer(raw: str) -> tuple[str, str]:
    """``"Milwaukee Accessory (4031)"`` -> ``("Milwaukee Accessory", "4031")``."""
    text = T.clean(raw)
    if T.is_placeholder(text):
        return "", ""
    m = _MANUF_CODE_RE.match(text)
    if m:
        return T.clean(m.group("name")), m.group("code")
    return text, ""


def _candidate_tokens(desc: str, mpn: str) -> list[str]:
    """Alphabetic tokens from a description that could name a brand."""
    body, _ = T.strip_leading_mpn(desc, mpn)
    body, _ = T.strip_noise(body)
    out: list[str] = []
    mpn_low = mpn.lower()
    for token, _s, _e in T.tokens(body):
        low = token.lower()
        if len(token) < 3 or not token[0].isalpha():
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", token):
            continue
        if low in _NEVER_BRAND or low in T.EXPANSIONS:
            continue
        if low == mpn_low or low in mpn_low:
            continue
        if U.lookup(low) is not None:
            continue
        out.append(token)
    return out


def _clean_brand_values(rows: Sequence[RawRow], column: str) -> Counter:
    """Attested brand spellings from one column, placeholders removed."""
    counts: Counter = Counter()
    for row in rows:
        raw = row.get(column)
        if T.is_placeholder(raw):
            continue
        value = T.clean(raw)
        if value and not T.is_placeholder(value):
            counts[value] += 1
    return counts


#: Trailing words in a `Part_Manuf` account name that describe the *account*,
#: not the brand. "Kichler Lighting (KICLI)" sells Kichler; "Satco Prod Inc"
#: sells Satco. Stripping these is what lets a fuzzy match land on the brand.
_ACCOUNT_SUFFIXES = (
    "cooperative", "coop", "co-op", "accessory", "accessories",
    "incorporated", "inc", "llc", "llp", "ltd", "limited", "plc",
    "corporation", "corp", "company", "co", "holdings", "group",
    "manufacturing", "mfg", "products", "product", "prod",
    "industries", "industrial", "intl", "international",
    "supply", "supplies", "distribution", "distributors", "dist",
    "lighting", "electric", "electrical", "tools", "tool",
    "usa", "us", "america", "american", "north",
    "division", "div", "dv", "sales", "systems", "solutions",
    "building", "materials", "lumber", "metals", "gear",
)


def account_name_to_brand_guess(name: str) -> str:
    """``"Satco Prod Inc"`` -> ``"Satco"``; ``"Southwire/g Turner"`` -> ``"Southwire"``.

    A distributor account name is not a brand, but it very often *contains*
    one as its head. Trimming the account furniture gives the fuzzy linker a
    target it can actually hit.
    """
    text = T.clean(name)
    if not text:
        return ""
    # "Black & Decker/dewlt" and "Southwire/g Turner": the head is the brand.
    text = text.split("/")[0].strip()
    words = [w for w in re.split(r"\s+", text) if w]
    while len(words) > 1 and words[-1].strip(".,").lower() in _ACCOUNT_SUFFIXES:
        words.pop()
    return " ".join(words).strip(" .,&-")


@dataclass
class _LinkTiers:
    """Canonical-brand candidates, strongest evidence first."""

    tiers: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def add(self, label: str, values: Iterable[str]) -> None:
        mapping: dict[str, str] = {}
        for value in values:
            if value:
                mapping.setdefault(value.lower(), value)
        if mapping:
            self.tiers.append((label, mapping))

    def exact(self, token: str) -> tuple[str, str] | None:
        low = token.lower()
        for label, mapping in self.tiers:
            if low in mapping:
                return mapping[low], label
        return None

    def fuzzy(self, token: str, threshold: float) -> tuple[str, str, float] | None:
        """First tier with a hit wins, so a clean brand column outranks an
        account name even when the account name scores higher."""
        for label, mapping in self.tiers:
            best_name, best_score = "", 0.0
            for name in dict.fromkeys(mapping.values()):
                score = brand_similarity(token, name)
                if score > best_score:
                    best_name, best_score = name, score
            if best_score >= threshold:
                return best_name, label, best_score
        return None


#: A brand abbreviation must be at least this long before a prefix match is
#: allowed. ``Milw`` -> ``Milwaukee`` is credible at 4; ``Fir`` -> ``First
#: Alert`` is not credible at 3, and "Fir" is lumber.
MIN_PREFIX_ALIAS = 4


def brand_similarity(token: str, target: str) -> float:
    """Similarity between an induced alias and a candidate brand name.

    Off-the-shelf ``fuzz.WRatio`` is wrong for this job because it rewards
    matching *any* word of a multi-word name. Measured against the corpus it
    produced ``Fan`` -> ``Hunter Fan`` (90), ``Door`` -> ``United Window &
    Door`` (90) and ``Alert`` -> ``First Alert`` (90) -- three product nouns
    promoted to brands purely because they appear somewhere in a brand name.

    A trade abbreviation shortens the *head* of a brand name, so that is what
    we score against, and only that.
    """
    tok = token.strip().lower()
    tgt = target.strip().lower()
    if not tok or not tgt:
        return 0.0
    if tok == tgt:
        return 100.0

    head = re.split(r"[\s/&,\-]+", tgt, maxsplit=1)[0]
    if not head:
        return 0.0
    if tok == head:
        return 100.0

    # Every genuine trade abbreviation keeps the brand's initial: Milw(aukee),
    # Dew(a)lt, South(wire). Requiring it costs nothing and eliminates a whole
    # family of coincidences the corpus actually produced -- Wire/Southwire,
    # Reel/Dremel, Mounts/StealthMounts, T-Square/Square D.
    if tok[0] != head[0]:
        return 0.0

    # A prefix of the head: Milw|aukee.
    if len(tok) >= MIN_PREFIX_ALIAS and head.startswith(tok):
        return 96.0

    # ...or a compression with the letters in order: dewlt <- dewalt. Bounded
    # by length ratio, because an unbounded subsequence test accepts
    # Timer <- Timbertech (5 of 10 letters) and Cntr <- Century.
    if (
        len(tok) >= MIN_PREFIX_ALIAS
        and len(tok) / len(head) >= 0.6
        and head[: min(3, len(tok))] == tok[: min(3, len(tok))]
        and _is_subsequence(tok, head)
    ):
        return max(88.0, float(fuzz.ratio(tok, head)))

    return float(fuzz.ratio(tok, head))


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


# --- the induction pass -----------------------------------------------------


def induce(rows: Sequence[RawRow], *, verbose: bool = False) -> InducedVocabulary:
    """One pass over the catalogue; returns every vocabulary it can support."""
    vocab = InducedVocabulary(n_rows=len(rows))

    # ---- group the corpus by distributor/manufacturer ----
    by_group: dict[str, list[RawRow]] = defaultdict(list)
    for row in rows:
        raw_group = T.clean(row.get("Part_Manuf"))
        key = raw_group if not T.is_placeholder(raw_group) else "<ungrouped>"
        by_group[key].append(row)
        if key not in vocab.groups:
            name, code = parse_manufacturer(raw_group)
            vocab.groups[key] = {"name": name, "code": code, "rows": "0"}
    for key, members in by_group.items():
        vocab.groups[key]["rows"] = str(len(members))

    n_groups = max(len(by_group), 1)

    # ---- pass 1: product language, before any brand knowledge exists ----
    # Run first so that brand induction can consult it. A token that shows up
    # inside induced product-type phrases is product vocabulary, and product
    # vocabulary must never be *fuzzy*-linked to a brand. This is what stops
    # "Metal" (from "Metal Cut Off Disc") becoming the brand "Metalmark" and
    # "Decking" becoming a brand at all -- and it is derived from the corpus
    # rather than typed into a blocklist, so it adapts to a new catalogue.
    _induce_product_language(rows, vocab)
    product_words: frozenset[str] = frozenset(
        word.lower()
        for name in vocab.product_types
        for word in re.findall(r"[A-Za-z][A-Za-z'\-]*", name)
    )

    # ---- token statistics, with positions ----
    token_groups: dict[str, set[str]] = defaultdict(set)
    group_token_counts: dict[str, Counter] = {}
    position_sum: dict[str, float] = defaultdict(float)
    position_n: dict[str, int] = defaultdict(int)
    for group, members in by_group.items():
        counts: Counter = Counter()
        for row in members:
            candidates = _candidate_tokens(row.desc, row.mpn)
            total = max(len(candidates), 1)
            seen: set[str] = set()
            for position, token in enumerate(candidates):
                low = token.lower()
                # normalised position of the token's first occurrence
                if low not in seen:
                    position_sum[low] += position / total
                    position_n[low] += 1
                    seen.add(low)
            for token in {t for t in candidates}:
                counts[token] += 1
                token_groups[token.lower()].add(group)
        group_token_counts[group] = counts

    def mean_position(token: str) -> float:
        low = token.lower()
        n = position_n.get(low, 0)
        return position_sum[low] / n if n else 0.5

    # ---- link targets, strongest evidence first ----
    dib = _clean_brand_values(rows, "DIB_Brand")
    e1 = _clean_brand_values(rows, "E1_Brand")
    unilog = _clean_brand_values(rows, "Unilog_Brand")
    attested: Counter = Counter()
    for counter in (dib, e1, unilog):
        attested.update(counter)

    tiers = _LinkTiers()
    tiers.add("dib-brand-column", dib)
    tiers.add("e1-brand-column", e1)
    tiers.add("unilog-brand-column", unilog)
    tiers.add("known-shorthand", set(T.BRAND_SHORTHAND.values()))
    tiers.add(
        "account-name-head",
        {
            account_name_to_brand_guess(meta["name"])
            for meta in vocab.groups.values()
            if meta["name"]
        },
    )

    # ---- brand alias induction ----
    for group, counts in group_token_counts.items():
        size = len(by_group[group])
        if not size:
            continue
        for token, hits in counts.items():
            low = token.lower()
            coverage = hits / size
            n_token_groups = len(token_groups[low])
            if coverage < MIN_GROUP_COVERAGE:
                continue
            if n_token_groups > MAX_GROUPS_FOR_ALIAS:
                continue
            idf = math.log(n_groups / n_token_groups)
            salience = coverage * idf
            if salience < MIN_SALIENCE:
                continue

            canonical, linkage, confidence, is_attested = _link(
                token, by_group[group], tiers, product_words
            )
            position = mean_position(token)

            # Unattested candidates are held back for review rather than used.
            # Product nouns are genuinely distinctive inside a distributor
            # group -- "Decking" covers most of Boise Cascade's rows -- so
            # distinctiveness alone cannot promote a token to a brand.
            target = vocab.brands if is_attested else vocab.brand_candidates
            rejection = ""
            if not is_attested:
                why = (
                    "token is product vocabulary, not a brand"
                    if linkage.startswith("blocked:")
                    else f"no attested spelling within {BRAND_LINK_THRESHOLD} similarity"
                )
                rejection = (
                    f"{why}; mean position {position:.2f} "
                    f"({'head' if position < 0.4 else 'tail'}-weighted)"
                )

            entry = target.get(canonical)
            if entry is None:
                entry = BrandEntry(
                    canonical=canonical,
                    linkage=linkage,
                    confidence=confidence,
                    attested=is_attested,
                    rejection=rejection,
                )
                target[canonical] = entry
            if low not in {a.lower() for a in entry.aliases}:
                entry.aliases.append(token)
            if group not in entry.groups:
                entry.groups.append(group)
            entry.support += hits
            entry.confidence = max(entry.confidence, confidence)
            entry.mean_position = (
                position if entry.mean_position == 0.0 else (entry.mean_position + position) / 2
            )

    # Attested column values are brands even when no shorthand was induced.
    for value, support in attested.items():
        entry = vocab.brands.get(value)
        if entry is None:
            vocab.brands[value] = BrandEntry(
                canonical=value,
                aliases=[value],
                support=support,
                linkage="attested-column",
                confidence=0.99,
                attested=True,
            )
        else:
            entry.support += support
            entry.attested = True
        # A token promoted to a real brand must not linger in the review queue.
        vocab.brand_candidates.pop(value, None)

    # ---- alias index (longest alias wins, so "timbertech" beats "tech") ----
    for canonical, entry in vocab.brands.items():
        for alias in {*entry.aliases, canonical}:
            low = alias.lower()
            existing = vocab.alias_index.get(low)
            if existing is None or len(canonical) > len(existing):
                vocab.alias_index[low] = canonical

    # ---- pass 2: re-induce product language, now that brands are known ----
    # The second pass produces cleaner phrases because brand aliases can be
    # stripped out of them ("Trex Enhance Basics Decking" -> "Enhance Basics
    # Decking"), which also sharpens the series lexicon.
    _induce_product_language(rows, vocab)

    vocab.stats = {
        "groups": float(n_groups),
        "brands": float(len(vocab.brands)),
        "brand_candidates_for_review": float(len(vocab.brand_candidates)),
        "aliases": float(len(vocab.alias_index)),
        "product_types": float(len(vocab.product_types)),
        "series": float(len(vocab.series)),
        "unknown_unit_spellings": float(len(vocab.unknown_units)),
        "attested_brands": float(sum(1 for b in vocab.brands.values() if b.attested)),
    }
    if verbose:
        print(json.dumps(vocab.stats, indent=2))
    return vocab


def _link(
    token: str,
    members: Sequence[RawRow],
    tiers: _LinkTiers,
    product_words: frozenset[str] = frozenset(),
) -> tuple[str, str, float, bool]:
    """Decide the canonical brand for an induced alias.

    Preference order, strongest evidence first:

    1. the group's own clean brand column -- ``DIB_Brand`` is clean wherever it
       is populated, and it is populated in 245 of 1000 rows;
    2. an exact hit against an attested spelling anywhere in the corpus;
    3. a *tiered* fuzzy hit. Tiering is the important part: matching ``Milw``
       against every candidate at once picks the distributor account name
       ``Milwaukee Accessory`` (WRatio 90) over the actual brand ``Milwaukee``.
       Trying the clean brand columns first and only falling through to account
       names produces ``Milwaukee``;
    4. nothing -- the alias is returned unlinked and flagged, so it lands in
       the review queue instead of the lexicon.
    """
    low = token.lower()

    # 1. clean brand column inside this group
    column_votes: Counter = Counter()
    for row in members:
        for column in ("DIB_Brand", "E1_Brand"):
            raw = row.get(column)
            if T.is_placeholder(raw):
                continue
            value = T.clean(raw)
            if value and not T.is_placeholder(value):
                column_votes[value] += 1
    if column_votes:
        best, votes = column_votes.most_common(1)[0]
        # Only adopt it as *this token's* canonical form if the two agree; the
        # group's dominant brand is not automatically this alias's brand.
        if max(
            fuzz.partial_ratio(low, best.lower()),
            fuzz.WRatio(token, best),
        ) >= BRAND_LINK_THRESHOLD:
            return best, f"group-brand-column({votes} rows)", 0.97, True

    # 2. exact attested spelling. Exact hits are allowed even for tokens that
    #    are also product vocabulary: "Element" really is an appliance brand
    #    and "Edge" really is an eyewear brand, and an exact match against a
    #    populated brand column is strong enough evidence to say so.
    hit = tiers.exact(token)
    if hit:
        canonical, label = hit
        return canonical, f"exact:{label}", 0.95, True

    # 3. tiered fuzzy link -- but never for product vocabulary, where a fuzzy
    #    hit is far more likely to be a coincidence than a brand.
    if token.lower() in product_words:
        return T.title_case(token), "blocked:product-vocabulary", 0.0, False
    fuzzy = tiers.fuzzy(token, BRAND_LINK_THRESHOLD)
    if fuzzy:
        canonical, label, score = fuzzy
        return canonical, f"fuzzy:{label}({score:.0f})", score / 100.0, True

    # 4. unlinked -- held for review, never used for enrichment
    return T.title_case(token), "unlinked", 0.0, False


_SEGMENT_SPLIT = re.compile(r"\s+-\s+|\s*[|;]\s*")


def _induce_product_language(rows: Sequence[RawRow], vocab: InducedVocabulary) -> None:
    """Induce product types, series names, families and unknown unit spellings."""
    type_counts: Counter = Counter()
    type_groups: dict[str, set[str]] = defaultdict(set)
    type_variants: dict[str, set[str]] = defaultdict(set)
    series_counts: Counter = Counter()
    families: dict[str, Counter] = defaultdict(Counter)
    unknown: Counter = Counter()
    value_vocab: dict[str, Counter] = defaultdict(Counter)

    # any run of digits followed by 1-5 letters that the unit table rejected
    unknown_unit_re = re.compile(r"\b\d+(?:[./-]\d+)?\s*([A-Za-z]{1,5})\b")

    for row in rows:
        group = T.clean(row.get("Part_Manuf")) or "<ungrouped>"
        desc = T.clean(row.desc)
        body, _ = T.strip_leading_mpn(desc, row.mpn)
        body, _noise = T.strip_noise(body)

        # -- product type: the head noun phrase, usually the trailing segment
        segments = [s for s in _SEGMENT_SPLIT.split(body) if s.strip()]
        phrase = _head_noun_phrase(segments, vocab)
        if phrase:
            canonical = T.title_case(T.expand(phrase))
            type_counts[canonical] += 1
            type_groups[canonical].add(group)
            if canonical.lower() != phrase.lower():
                type_variants[canonical].add(phrase)

            for measure in U.find_measurements(body):
                families[canonical][measure.family] += 1

        # -- series: title-cased multiword run that is not the type or a brand
        for candidate in _series_candidates(body, vocab, phrase or ""):
            series_counts[candidate] += 1

        # -- unknown unit spellings, for human approval
        for m in unknown_unit_re.finditer(body):
            suffix = m.group(1)
            if U.lookup(suffix) is None and suffix.lower() not in T.EXPANSIONS:
                prev = body[m.start() - 1] if m.start() else ""
                if not prev.isalpha():
                    unknown[suffix] += 1

        # -- colour / material / finish value vocabulary
        for token, _s, _e in T.tokens(body):
            full = T.expand_token(token)
            if not full:
                continue
            if full in {"Black", "White", "Stainless Steel", "Black Stainless Steel",
                        "Almond", "Bronze", "Brass", "Chrome", "Clear", "Aluminum",
                        "Steel", "Copper", "Nickel", "Satin Nickel", "Brushed Nickel",
                        "Polished Nickel", "Polished Brass", "Oil Rubbed Bronze"}:
                value_vocab["Color/Material"][full] += 1

    vocab.product_types = {
        name: ProductTypeEntry(
            canonical=name,
            support=count,
            groups=sorted(type_groups[name]),
            variants=sorted(type_variants[name]),
        )
        for name, count in type_counts.items()
        if count >= 2  # a phrase seen once is not yet vocabulary
    }
    vocab.series = {k: v for k, v in series_counts.items() if v >= 2}
    vocab.observed_families = {
        k: dict(v.most_common()) for k, v in families.items() if k in vocab.product_types
    }
    vocab.unknown_units = dict(unknown.most_common())
    vocab.value_vocab = {k: dict(v.most_common()) for k, v in value_vocab.items()}


def _head_noun_phrase(segments: list[str], vocab: InducedVocabulary) -> str:
    """Pick the phrase naming the product from the description's segments.

    Distributor descriptions put the noun last, either after a ``-`` separator
    (``... - Sanding Belt 6pc``) or at the end of the only segment
    (``Milw 14"x1/8"x1" Masonry Cut Off Disc``). We take the last segment, drop
    measurements, brand aliases and pack counts, and keep the trailing run of
    alphabetic words.
    """
    if not segments:
        return ""
    for segment in reversed(segments):
        cleaned = segment
        # remove measurements so "6pc" and '14"x1/8"' do not enter the phrase
        for measure in sorted(U.find_measurements(cleaned), key=lambda m: -m.start):
            cleaned = cleaned[: measure.start] + " " + cleaned[measure.end :]
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", cleaned)]
        # drop brand aliases and leading noise
        words = [w for w in words if w.lower() not in vocab.alias_index]
        words = [w for w in words if len(w) > 1 or w.lower() in {"t"}]
        if not words:
            continue
        # keep at most the last four words -- longer runs are marketing prose
        phrase = " ".join(words[-4:])
        if phrase.strip():
            return phrase.strip()
    return ""


_TITLE_RUN = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[IVX]+|\d+|[A-Z]\+?))+)")


def _series_candidates(body: str, vocab: InducedVocabulary, product_phrase: str) -> list[str]:
    """Title-cased multi-word runs that name a product line, not the item type."""
    out: list[str] = []
    product_words = {w.lower() for w in product_phrase.split()}
    for m in _TITLE_RUN.finditer(body):
        run = m.group(1).strip()
        words = run.split()
        if len(words) < 2:
            continue
        lowered = {w.lower() for w in words}
        if lowered & product_words:
            continue
        if any(w.lower() in vocab.alias_index for w in words):
            # "Trex Enhance Basics" -> keep "Enhance Basics"
            words = [w for w in words if w.lower() not in vocab.alias_index]
            if len(words) < 2:
                continue
            run = " ".join(words)
        if any(w.lower() in T.EXPANSIONS or U.lookup(w) for w in words):
            continue
        out.append(run)
    return out


# --- convenience ------------------------------------------------------------


def induce_from_file(path: str | Path, *, verbose: bool = False) -> InducedVocabulary:
    from .schema import load_input

    return induce(load_input(path), verbose=verbose)
