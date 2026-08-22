"""Text hygiene, and the two abbreviation lexicons.

Distributor descriptions are written for a warehouse terminal with a 40-column
screen, so they are dense with trade shorthand: ``Milw 14"x1/8"x1" Masonry Cut
Off Disc``, ``DC5004WE SQ Elect Dryer Wh``, ``S11646 6" Downlight Wh``.

Enrichment needs the abbreviation map in **both** directions, and they are not
inverses of each other:

* **expansion** (read side) -- ``Elect`` -> ``Electric``, ``Wh`` -> ``White``,
  ``SST`` -> ``Stainless Steel``. Used to understand the input and to write the
  long-form channels.
* **contraction** (write side) -- ``Stainless Steel`` -> ``SST``, ``Built-in``
  -> ``BLTLN``. Used only by ``INVOICE_DESC``, which has 40 characters and must
  still be readable to a counter clerk.

Both lexicons are seeded here from the trade shorthand actually present in the
working dataset and are then *extended by corpus induction* (see
:mod:`glassbox.induce`), which is how the pipeline copes with a distributor
whose shorthand nobody has seen before.
"""

from __future__ import annotations

import re
import unicodedata

# --- placeholders -----------------------------------------------------------

#: Values that look like data and mean "empty". Section 9 of DERIVED_RULES.md
#: measures these: 799/1000 rows carry ``-- Unbranded --`` and *every* row
#: carries ``-- No Unilog Brand --``.
PLACEHOLDER_LITERALS = frozenset(
    {
        "-- unbranded --",
        "-- no unilog brand --",
        "-- no dib brand --",
        "commodity - unbranded",
        "unbranded",
        "no brand",
        # The dash-stripped forms matter too: clean() removes the surrounding
        # "--" before anything else sees the value, so "-- No DIB Brand --"
        # arrives here as "No DIB Brand". Without these three literals it
        # sails through as a legitimate brand and becomes the single
        # highest-support "brand" in the whole corpus.
        "no unilog brand",
        "no dib brand",
        "no e1 brand",
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "--",
        "---",
        ".",
        "?",
        "tbd",
        "unknown",
        "misc",
        "miscellaneous",
        "generic",
    }
)

#: Anything wrapped in the ``-- ... --`` convention is a placeholder by
#: construction, even a spelling we have never seen.
_PLACEHOLDER_SHAPE = re.compile(r"^\s*--+\s*[^-].*?\s*--+\s*$")

#: ``No <anything> Brand`` is the house negation idiom, so it generalises to a
#: column this dataset does not happen to contain.
_NEGATED_BRAND = re.compile(r"^\s*no\s+[\w\s]{0,24}\bbrands?\s*$", re.IGNORECASE)


def is_placeholder(value: str | None) -> bool:
    """True when a field is present but carries no information."""
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    if _PLACEHOLDER_SHAPE.match(text):
        return True
    # Compare both as-is and with the "--" wrapper removed, because clean()
    # strips those dashes and callers are not always careful about ordering.
    bare = text.strip("-").strip()
    if text.lower() in PLACEHOLDER_LITERALS or bare.lower() in PLACEHOLDER_LITERALS:
        return True
    if _NEGATED_BRAND.match(bare):
        return True
    # A field of pure punctuation is not a value.
    return not any(ch.isalnum() for ch in text)


# --- cleaning ---------------------------------------------------------------

_QUOTE_MAP = {
    "“": '"',  # left double
    "”": '"',  # right double
    "″": '"',  # double prime
    "′": "'",  # prime
    "‘": "'",
    "’": "'",
    "´": "'",
    "—": "-",  # em dash
    "–": "-",  # en dash
    " ": " ",  # nbsp
}
_QUOTE_RE = re.compile("|".join(map(re.escape, _QUOTE_MAP)))

#: Operational markers that describe the *listing*, not the product. They must
#: not survive into customer-facing copy, but they are worth keeping as a flag.
NOISE_MARKERS: tuple[str, ...] = (
    "display only",
    "display model",
    "floor model",
    "clearance",
    "discontinued",
    "special order",
    "do not order",
    "obsolete",
    "linear foot",
    "per foot",
    "cut to length",
)
_NOISE_RE = re.compile(
    r"[\s\-,(]*\b(?:" + "|".join(re.escape(n) for n in NOISE_MARKERS) + r")\b[\s)]*",
    re.IGNORECASE,
)


#: Characters that must survive normalisation untouched. NFKC is wanted here --
#: it folds ``″`` to ``"`` and ``⁄`` to ``/`` -- but it also decomposes ``™``
#: into the two letters ``TM`` and ``℠`` into ``SM``, which silently destroys
#: the registered marks the delivery format mandates on every brand name. The
#: gold row is ``With CleanBoost™``, not ``With CleanBoostTM``. So these are
#: parked in the Unicode private-use area across the NFKC call and restored.
#: The prime marks are here for the same reason as the registered marks, one
#: step further on: NFKC decomposes U+2033 DOUBLE PRIME into *two* U+2032
#: PRIME characters, so ``24″`` normalises to ``24''`` before the quote map
#: ever sees it. That still parses as inches -- ``''`` is a declared alias --
#: but it leaves the wrong glyphs in customer-facing copy. Protecting them
#: means the quote map converts ``″`` straight to ``"``.
_PROTECTED = "®™©℠″′"
_PROTECT_MAP = {ch: chr(0xE000 + i) for i, ch in enumerate(_PROTECTED)}
_RESTORE_MAP = {v: k for k, v in _PROTECT_MAP.items()}
_PROTECT_RE = re.compile("|".join(map(re.escape, _PROTECT_MAP)))
_RESTORE_RE = re.compile("|".join(map(re.escape, _RESTORE_MAP)))


def clean(text: str | None) -> str:
    """Normalise quotes and whitespace without changing meaning."""
    if not text:
        return ""
    out = _PROTECT_RE.sub(lambda m: _PROTECT_MAP[m.group(0)], text)
    out = unicodedata.normalize("NFKC", out)
    out = _RESTORE_RE.sub(lambda m: _RESTORE_MAP[m.group(0)], out)
    out = _QUOTE_RE.sub(lambda m: _QUOTE_MAP[m.group(0)], out)
    out = re.sub(r"\s+", " ", out).strip()
    # Distributor exports often leave a dangling separator.
    out = re.sub(r"[\s\-,;:]+$", "", out)
    out = re.sub(r"^[\s\-,;:]+", "", out)
    return out


def strip_noise(text: str) -> tuple[str, list[str]]:
    """Remove listing-status markers. Returns (clean text, markers found)."""
    found = [
        m.group(0).strip(" -,()")
        for m in _NOISE_RE.finditer(text)
        if m.group(0).strip(" -,()")
    ]
    if not found:
        return text, []
    out = _NOISE_RE.sub(" ", text)
    return clean(out), found


def strip_leading_mpn(text: str, mpn: str) -> tuple[str, bool]:
    """Drop a part number repeated at the head of its own description.

    ``"PDSH4816AF Dishwasher SS"`` -> ``"Dishwasher SS"``. Extremely common
    (the MPN is prepended by the export), and it pollutes every downstream
    token statistic if left in place.
    """
    if not mpn:
        return text, False
    stripped = text.strip()
    if stripped.upper().startswith(mpn.upper()):
        remainder = stripped[len(mpn) :].lstrip(" -,:")
        if remainder:
            return remainder, True
    return text, False


# --- expansion lexicon (read side) ------------------------------------------

#: Trade shorthand -> full form. Keys are matched case-insensitively as whole
#: tokens. Ordered by family for maintenance, not by precedence.
EXPANSIONS: dict[str, str] = {
    # materials & finishes
    "ss": "Stainless Steel",
    "sst": "Stainless Steel",
    "sts": "Stainless Steel",
    "bss": "Black Stainless Steel",
    "stl": "Steel",
    "alum": "Aluminum",
    "al": "Aluminum",
    "brs": "Brass",
    "cu": "Copper",
    "pvc": "PVC",
    "cpvc": "CPVC",
    "abs": "ABS",
    "galv": "Galvanized",
    "gi": "Galvanized Iron",
    "ci": "Cast Iron",
    "poly": "Polyethylene",
    "nyl": "Nylon",
    "rbr": "Rubber",
    "cer": "Ceramic",
    "prc": "Porcelain",
    "chr": "Chrome",
    "blk": "Black",
    "bk": "Black",
    "wh": "White",
    "wht": "White",
    "biscuit": "Biscuit",
    "alm": "Almond",
    "bz": "Bronze",
    "orb": "Oil Rubbed Bronze",
    "sn": "Satin Nickel",
    "bn": "Brushed Nickel",
    "pn": "Polished Nickel",
    "pb": "Polished Brass",
    "clr": "Clear",
    "frst": "Frosted",
    # product families
    "lt": "Light",
    "lts": "Lights",
    "lgt": "Light",
    "fixt": "Fixture",
    "flor": "Fluorescent",
    "fluor": "Fluorescent",
    "incan": "Incandescent",
    "hal": "Halogen",
    "led": "LED",
    "cfl": "CFL",
    "hid": "HID",
    "bulb": "Bulb",
    "lamp": "Lamp",
    "recep": "Receptacle",
    "rcpt": "Receptacle",
    "sw": "Switch",
    "swtch": "Switch",
    "brkr": "Breaker",
    "cb": "Circuit Breaker",
    "pnl": "Panel",
    "encl": "Enclosure",
    "cond": "Conduit",
    "conn": "Connector",
    "term": "Terminal",
    "wr": "Wire",
    "cbl": "Cable",
    "crd": "Cord",
    # appliances
    "elect": "Electric",
    "elec": "Electric",
    "el": "Electric",
    "gas": "Gas",
    "dw": "Dishwasher",
    "ref": "Refrigerator",
    "fridge": "Refrigerator",
    "frzr": "Freezer",
    "wshr": "Washer",
    "dryr": "Dryer",
    "mw": "Microwave",
    "mwo": "Microwave Oven",
    "rng": "Range",
    "ckt": "Cooktop",
    "wall ov": "Wall Oven",
    "ov": "Oven",
    "hd": "Hood",
    "dsps": "Disposer",
    # tools & abrasives
    "cutoff": "Cut Off",
    "grnd": "Grinding",
    "grndr": "Grinder",
    "abr": "Abrasive",
    "snd": "Sanding",
    "sndr": "Sander",
    "blde": "Blade",
    "bld": "Blade",
    "bit": "Bit",
    "drl": "Drill",
    "hmr": "Hammer",
    "impct": "Impact",
    "drvr": "Driver",
    "wrnch": "Wrench",
    "scrw": "Screw",
    "chgr": "Charger",
    "batt": "Battery",
    "cordless": "Cordless",
    "recip": "Reciprocating",
    "circ": "Circular",
    "osc": "Oscillating",
    # building materials
    "pnl'd": "Panel",
    "lvl": "Laminated Veneer Lumber",
    "osb": "Oriented Strand Board",
    "ply": "Plywood",
    "trtd": "Treated",
    "pt": "Pressure Treated",
    "kd": "Kiln Dried",
    "sq": "Square",
    "sq edge": "Square Edge",
    "grvd": "Grooved",
    "prmd": "Primed",
    "unfin": "Unfinished",
    "mldg": "Molding",
    "trm": "Trim",
    "dck": "Decking",
    # geometry & fit
    "od": "Outside Diameter",
    "id": "Inside Diameter",
    "dia": "Diameter",
    "thk": "Thickness",
    "lg": "Length",
    "wd": "Width",
    "ht": "Height",
    "dp": "Depth",
    "mnt": "Mounting",
    "mtg": "Mounting",
    "bltln": "Built-in",
    "bltin": "Built-in",
    "fs": "Free Standing",
    "slidein": "Slide-in",
    "undrmnt": "Undermount",
    "surf": "Surface",
    "flsh": "Flush",
    "semiflsh": "Semi-Flush",
    "pend": "Pendant",
    "chand": "Chandelier",
    "sconce": "Sconce",
    "recsd": "Recessed",
    "ext": "Exterior",
    "int": "Interior",
    "indr": "Indoor",
    "outdr": "Outdoor",
    # general
    "asy": "Assembly",
    "assy": "Assembly",
    "kit": "Kit",
    "repl": "Replacement",
    "accy": "Accessory",
    "adj": "Adjustable",
    "hd duty": "Heavy Duty",
    "hvy": "Heavy",
    "lt duty": "Light Duty",
    "std": "Standard",
    "univ": "Universal",
    "compl": "Complete",
    "prof": "Professional",
    "comm": "Commercial",
    "resid": "Residential",
    "indus": "Industrial",
    "ser": "Series",
    "srs": "Series",
    "mdl": "Model",
    "no": "Number",
    "qty": "Quantity",
    "ea": "Each",
}

#: Tokens that are shorthand for a *brand*, resolved by entity.py rather than
#: expanded as vocabulary. Kept separate so brand shorthand never leaks into a
#: product-type noun phrase.
BRAND_SHORTHAND: dict[str, str] = {
    "milw": "Milwaukee",
    "milwaukee": "Milwaukee",
    "dewlt": "DEWALT",
    "dewalt": "DEWALT",
    "dw": "DEWALT",  # only in tool context; entity.py gates this
    "mkta": "Makita",
    "makita": "Makita",
    "bosch": "Bosch",
    "diablo": "Diablo",
    "freud": "Freud",
    "festool": "Festool",
    "kreg": "Kreg",
    "grizzly": "Grizzly",
    "jet": "JET",
    "satco": "Satco",
    "kichler": "Kichler",
    "philips": "Philips",
    "phillips": "Philips",
    "trex": "Trex",
    "timbertech": "TimberTech",
    "azek": "AZEK",
    "hardie": "James Hardie",
    "hardiepanel": "James Hardie",
    "hardieplank": "James Hardie",
    "leviton": "Leviton",
    "southwire": "Southwire",
    "lutron": "Lutron",
    "frigidaire": "Frigidaire",
    "whirlpool": "Whirlpool",
    "maytag": "Maytag",
    "kitchenaid": "KitchenAid",
    "lg": "LG",
    "samsung": "Samsung",
    "bosch appl": "Bosch",
    "ge": "GE",
    "speed queen": "Speed Queen",
    "sq": "Speed Queen",  # context-gated: only for laundry item types
    "element": "Element",
    "mirka": "Mirka",
    "edge": "Edge Eyewear",
    "vessel": "Vessel",
    "hunter": "Hunter Fan",
    "square d": "Square D",
    "3m": "3M",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\d./\-]*|[\"'#]")


def tokens(text: str) -> list[tuple[str, int, int]]:
    """Tokenise, keeping source spans so evidence can point back at the input."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


def expand_token(token: str) -> str | None:
    """Expand one shorthand token, or None if it is not shorthand."""
    return EXPANSIONS.get(token.strip().strip(".").lower())


def expand(text: str, *, skip: frozenset[str] = frozenset()) -> str:
    """Expand every known shorthand token in ``text``.

    ``skip`` holds lowercased tokens that must be left alone -- in practice the
    MPN and any already-resolved brand, so that ``SQ`` in ``SQ Elect Dryer``
    does not become ``Square`` when entity resolution has already claimed it
    for Speed Queen.
    """
    out: list[str] = []
    cursor = 0
    for token, start, end in tokens(text):
        low = token.lower()
        if low in skip:
            continue
        full = EXPANSIONS.get(low)
        if not full or full.lower() == low:
            continue
        out.append(text[cursor:start])
        out.append(full)
        cursor = end
    out.append(text[cursor:])
    return clean("".join(out))


# --- contraction lexicon (write side, INVOICE_DESC only) --------------------

#: Full form -> counter-clerk abbreviation. Verified against the gold rows:
#: ``Stainless Steel`` -> ``SST`` and ``Built-in`` -> ``BLTLN`` both appear in
#: the two published INVOICE_DESC values.
INVOICE_ABBREV: dict[str, str] = {
    "stainless steel": "SST",
    "black stainless steel": "BSST",
    "built-in": "BLTLN",
    "built in": "BLTLN",
    "free standing": "FRSTD",
    "freestanding": "FRSTD",
    "slide-in": "SLDIN",
    "under counter": "UNDCTR",
    "undermount": "UNDMT",
    "surface mount": "SURFMT",
    "flush mount": "FLSHMT",
    "semi-flush": "SEMIFL",
    "wall mount": "WALLMT",
    "ceiling mount": "CLGMT",
    "professional series": "PRO SER",
    "professional": "PRO",
    "commercial": "COMM",
    "residential": "RES",
    "industrial": "IND",
    "heavy duty": "HD",
    "light duty": "LD",
    "adjustable": "ADJ",
    "aluminum": "ALUM",
    "galvanized": "GALV",
    "polished chrome": "PCHR",
    "brushed nickel": "BRSNKL",
    "satin nickel": "SATNKL",
    "oil rubbed bronze": "ORB",
    "matte black": "MTBLK",
    "black": "BLK",
    "white": "WHT",
    "almond": "ALM",
    "biscuit": "BISC",
    "bisque": "BISQ",
    "chrome": "CHR",
    "bronze": "BRZ",
    "brass": "BRS",
    "nickel": "NKL",
    "copper": "CU",
    "cast iron": "CI",
    "steel": "STL",
    "plastic": "PLAS",
    "rubber": "RBR",
    "ceramic": "CER",
    "porcelain": "PRC",
    "refrigerator": "REFRIG",
    "dishwasher": "DISHWASHER",
    "microwave": "MICROWAVE",
    "washer": "WASHER",
    "dryer": "DRYER",
    "electric": "ELEC",
    "cordless": "CRDLS",
    "reciprocating": "RECIP",
    "circular": "CIRC",
    "oscillating": "OSC",
    "fluorescent": "FLUOR",
    "incandescent": "INCAN",
    "receptacle": "RECEP",
    "circuit breaker": "CB",
    "load center": "LDCTR",
    "cut off": "CUTOFF",
    "grinding": "GRND",
    "sanding": "SND",
    "abrasive": "ABR",
    "assembly": "ASY",
    "replacement": "REPL",
    "accessory": "ACCY",
    "universal": "UNIV",
    "standard": "STD",
    "number of": "",
    "with": "W/",
    "without": "W/O",
    "and": "&",
    "square": "SQ",
    "diameter": "DIA",
    "thickness": "THK",
    "length": "LG",
    "width": "WD",
    "height": "HT",
    "depth": "DP",
    "mounting": "MNT",
    "exterior": "EXT",
    "interior": "INT",
    "outdoor": "OUTDR",
    "indoor": "INDR",
}


def contract_for_invoice(text: str) -> str:
    """Abbreviate aggressively for the 40-character invoice channel."""
    if not text:
        return ""
    out = text
    # Longest phrase first, so "black stainless steel" wins over "black".
    for phrase in sorted(INVOICE_ABBREV, key=len, reverse=True):
        replacement = INVOICE_ABBREV[phrase]
        out = re.sub(rf"\b{re.escape(phrase)}\b", replacement, out, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out).strip().upper()


# --- casing -----------------------------------------------------------------

#: Words that stay lowercase inside a title, and acronyms that stay uppercase.
_LOWER_WORDS = frozenset({"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"})
_KEEP_UPPER = frozenset(
    {
        "LED", "CFL", "HID", "PVC", "CPVC", "ABS", "SST", "GFCI", "AFCI", "USB",
        "AC", "DC", "UL", "CSA", "NSF", "ASSE", "ADA", "MDF", "OSB", "LVL",
        "HP", "RPM", "TPI", "AWG", "BTU", "CFM", "NEMA", "IP", "3M", "LG", "GE",
        "II", "III", "IV", "XL", "XXL",
    }
)


def title_case(text: str) -> str:
    """Trade title casing: preserve acronyms, lowercase joining words."""
    if not text:
        return ""
    words = text.split()
    out: list[str] = []
    for i, word in enumerate(words):
        bare = word.strip(".,;:()")
        if bare.upper() in _KEEP_UPPER:
            out.append(word.replace(bare, bare.upper()))
        elif i > 0 and bare.lower() in _LOWER_WORDS:
            out.append(word.lower())
        elif bare.isupper() and len(bare) > 1 and any(c.isdigit() for c in bare):
            out.append(word)  # model-number-ish, leave alone
        elif "-" in bare:
            out.append("-".join(p[:1].upper() + p[1:] for p in word.split("-")))
        else:
            out.append(word[:1].upper() + word[1:] if word else word)
    return " ".join(out)
