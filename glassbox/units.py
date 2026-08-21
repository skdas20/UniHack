"""Units of measure: canonical abbreviations, fractions, and rendering modes.

Three jobs, all of them scored:

1. **Canonicalisation.** ``"``, ``in.``, ``inch``, ``INCHES`` and ``IN`` are all
   the same unit and exactly one spelling is permitted in output.
2. **Decimal to fraction.** Manufacturers publish ``50.25``; trade buyers search
   ``50-1/4``. The conversion is exact over 64ths and must never introduce
   floating-point noise.
3. **Rendering mode.** The gold rows prove there are two, and using the wrong
   one is a scored error:

   * ``SPACED``  -> ``120 V``, ``24 in``, ``47 dBA``  (LONG_DESC1, attributes)
   * ``GLUED``   -> ``120V``, ``50-1/4IN``, ``41DBA`` (INVOICE_DESC only)

The house rule for spaced mode is a single space between magnitude and unit,
always -- ``24 in``, never ``24in`` and never ``24  in``.

The organisers' master UOM workbook is not published on the portal, so the
canonical table below is a curated core covering every measurement family that
actually occurs in the working dataset, and it is *extended at runtime* by
:mod:`glassbox.induce`, which mines the corpus for unit spellings the core
does not yet know and reports them rather than silently dropping them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import lru_cache


class RenderMode(str, Enum):
    SPACED = "spaced"  # "120 V"  -- descriptions, attribute blocks
    GLUED = "glued"  # "120V"   -- INVOICE_DESC only


@dataclass(frozen=True, slots=True)
class Unit:
    """One canonical unit of measure."""

    canonical: str  # the only permitted output spelling
    family: str  # measurement family, e.g. "length"
    aliases: tuple[str, ...] = ()
    #: Prefer fractional rendering for magnitudes in this unit (imperial trade
    #: sizes) rather than decimal.
    fractional: bool = False
    #: Uppercase form used by INVOICE_DESC, when it differs from ``.upper()``.
    invoice: str = ""
    #: Resolvable when a UOM is stated explicitly, but never *scanned* out of
    #: free text. Single ambiguous letters cause more harm than good in a US
    #: industrial catalogue: ``3M`` is a brand not 3 metres, ``775L`` is a
    #: series not 775 litres, and metric/Celsius simply do not occur here.
    scan_excluded: bool = False

    def render(self, magnitude: str, mode: RenderMode) -> str:
        if mode is RenderMode.GLUED:
            return f"{magnitude}{self.invoice or self.canonical.upper()}"
        return f"{magnitude} {self.canonical}"


# --- canonical table --------------------------------------------------------
# Every family present in the 1000-row working dataset. Aliases are lowercase;
# matching is case-insensitive and punctuation-insensitive.

_UNITS: tuple[Unit, ...] = (
    # length -- imperial trade sizes render as fractions
    Unit("in", "length", ('"', "in.", "inch", "inches", "”", "″", "''"), fractional=True),
    Unit("ft", "length", ("'", "ft.", "foot", "feet", "’"), fractional=True),
    Unit("yd", "length", ("yd.", "yard", "yards")),
    Unit("mm", "length", ("mm.", "millimeter", "millimetre", "millimeters")),
    Unit("cm", "length", ("cm.", "centimeter", "centimeters")),
    Unit("m", "length", ("m.", "meter", "metre", "meters"), scan_excluded=True),
    # electrical
    Unit("V", "voltage", ("v", "v.", "volt", "volts", "vac", "vdc")),
    Unit("A", "current", ("a", "a.", "amp", "amps", "ampere", "amperes")),
    Unit("mA", "current", ("ma", "milliamp", "milliamps")),
    Unit("W", "power", ("w", "w.", "watt", "watts")),
    Unit("kW", "power", ("kw", "kilowatt", "kilowatts")),
    Unit("kW-hr", "energy", ("kwh", "kw-hr", "kilowatt hour", "kilowatt-hour")),
    Unit("HP", "power", ("hp", "hp.", "horsepower", "h.p.")),
    Unit("Hz", "frequency", ("hz", "hertz", "cycle", "cycles")),
    Unit("ohm", "resistance", ("ohms", "Ω")),
    Unit("AWG", "wire gauge", ("awg", "ga", "gauge", "ga.")),
    Unit("kcmil", "wire gauge", ("kcmil", "mcm")),
    Unit("K", "color temperature", ("k", "kelvin")),
    Unit("lm", "luminous flux", ("lm", "lumen", "lumens")),
    Unit("cd", "luminous intensity", ("cd", "candela"), scan_excluded=True),
    Unit("fc", "illuminance", ("fc", "foot-candle", "footcandle")),
    Unit("CRI", "color rendering", ("cri",)),
    # rotation / speed
    Unit("RPM", "rotational speed", ("rpm", "r.p.m.", "rev/min")),
    Unit("SPM", "stroke rate", ("spm", "strokes/min")),
    Unit("SFPM", "surface speed", ("sfpm", "sfm", "ft/min")),
    # mass / force
    Unit("lb", "weight", ("lb.", "lbs", "lbs.", "pound", "pounds", "#")),
    Unit("oz", "weight", ("oz.", "ounce", "ounces")),
    Unit("kg", "weight", ("kg.", "kilogram", "kilograms")),
    Unit("g", "weight", ("gram", "grams"), scan_excluded=True),
    Unit("lb-ft", "torque", ("lb-ft", "ft-lb", "ft.lb", "foot-pound")),
    Unit("in-lb", "torque", ("in-lb", "inch-pound", "in.lb")),
    Unit("Nm", "torque", ("nm", "newton-meter", "newton meter")),
    # volume / flow / pressure
    Unit("gal", "volume", ("gal.", "gallon", "gallons")),
    Unit("qt", "volume", ("qt.", "quart", "quarts")),
    Unit("pt", "volume", ("pt.", "pint", "pints"), scan_excluded=True),
    Unit("fl oz", "volume", ("fl. oz.", "fl oz.", "fluid ounce", "fluid ounces")),
    Unit("L", "volume", ("l", "liter", "litre", "liters"), scan_excluded=True),
    Unit("mL", "volume", ("ml", "milliliter", "millilitre")),
    Unit("cu ft", "volume", ("cu. ft.", "cuft", "ft3", "cubic foot", "cubic feet")),
    Unit("cu in", "volume", ("cu. in.", "cuin", "in3", "cubic inch", "cubic inches")),
    Unit("gpm", "flow rate", ("g.p.m.", "gal/min", "gallons per minute")),
    Unit("CFM", "air flow", ("cfm", "c.f.m.", "ft3/min")),
    Unit("psi", "pressure", ("p.s.i.", "lb/in2", "pounds per square inch")),
    Unit("bar", "pressure", ("bars",)),
    # area
    Unit("sq ft", "area", ("sq. ft.", "sqft", "ft2", "square foot", "square feet")),
    Unit("sq in", "area", ("sq. in.", "sqin", "in2", "square inch", "square inches")),
    # thermal / acoustic
    Unit("BTU", "heat", ("btu", "b.t.u.", "btus")),
    Unit("dBA", "sound level", ("dba", "db(a)", "decibel", "decibels", "db")),
    Unit("degF", "temperature", ("f", "°f", "deg f", "fahrenheit"), scan_excluded=True),
    Unit("degC", "temperature", ("c", "°c", "deg c", "celsius", "centigrade"), scan_excluded=True),
    # time
    Unit("hr", "time", ("hr.", "hrs", "hour", "hours")),
    Unit("min", "time", ("min.", "mins", "minute", "minutes")),
    Unit("sec", "time", ("sec.", "secs", "second", "seconds")),
    Unit("yr", "time", ("yr.", "yrs", "year", "years")),
    # count / packaging -- very common in this dataset ("3pk", "50 Disc/Box")
    Unit("pc", "count", ("pc.", "pcs", "piece", "pieces", "pct")),
    Unit("pk", "count", ("pk.", "pack", "packs", "pkg")),
    Unit("ea", "count", ("each",)),
    Unit("bx", "count", ("box", "boxes")),
    Unit("cs", "count", ("case", "cases")),
    Unit("rl", "count", ("roll", "rolls")),
    Unit("bdl", "count", ("bundle", "bundles")),
    Unit("pr", "count", ("pair", "pairs")),
    Unit("st", "count", ("set", "sets"), scan_excluded=True),
    # abrasives / tooling
    Unit("grit", "abrasive grade", ("grit", "grits")),
    Unit("TPI", "tooth pitch", ("tpi", "teeth per inch")),
    Unit("T", "tooth count", ("tooth", "teeth"), scan_excluded=True),
    Unit("PH", "electrical phase", ("ph", "phase")),
)


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, Unit]:
    """Lowercased alias (and canonical) -> Unit. Longest alias wins on tie."""
    out: dict[str, Unit] = {}
    for unit in _UNITS:
        for key in (unit.canonical, *unit.aliases):
            k = key.strip().lower()
            if not k:
                continue
            # First declaration wins, so the canonical form is never shadowed
            # by another unit's alias (e.g. "m" length vs "min").
            out.setdefault(k, unit)
    return out


@lru_cache(maxsize=1)
def canonical_units() -> tuple[str, ...]:
    return tuple(u.canonical for u in _UNITS)


def lookup(token: str) -> Unit | None:
    """Resolve a raw unit spelling to its canonical Unit, or None."""
    if not token:
        return None
    key = token.strip().lower()
    amap = _alias_map()
    if key in amap:
        return amap[key]
    # tolerate trailing punctuation and internal spacing noise
    stripped = re.sub(r"[.\s]+$", "", key)
    if stripped in amap:
        return amap[stripped]
    collapsed = re.sub(r"\s+", " ", stripped)
    return amap.get(collapsed)


def canonicalise(token: str) -> str:
    """Return the approved spelling for a unit token, or the token unchanged."""
    unit = lookup(token)
    return unit.canonical if unit else token.strip()


# --- fractions --------------------------------------------------------------

#: Every 64th, reduced -- the published Decimal_Fraction table, generated
#: rather than transcribed so it cannot contain a typo.
DECIMAL_TO_FRACTION: dict[str, str] = {}
FRACTION_TO_DECIMAL: dict[str, float] = {}
for _n in range(1, 64):
    _fr = Fraction(_n, 64)
    _label = f"{_fr.numerator}/{_fr.denominator}"
    _dec = float(_fr)
    FRACTION_TO_DECIMAL.setdefault(_label, _dec)
    # index by the exact decimal and by common rounded spellings
    DECIMAL_TO_FRACTION.setdefault(f"{_dec:.6f}".rstrip("0"), _label)
    for _places in (2, 3, 4, 5):
        DECIMAL_TO_FRACTION.setdefault(f"{round(_dec, _places):g}", _label)
del _n, _fr, _label, _dec, _places

_FRACTION_RE = re.compile(r"^\s*(?:(\d+)[-\s])?(\d+)\s*/\s*(\d+)\s*$")
_DECIMAL_RE = re.compile(r"^\s*(\d+)\.(\d+)\s*$")


def to_fraction(magnitude: str, *, max_denominator: int = 64) -> str:
    """``"50.25"`` -> ``"50-1/4"``. Non-trade-sizes pass through unchanged.

    A trade fraction is an exact binary fraction -- halves through 64ths -- and
    nothing else. Two things this deliberately refuses to do:

    * ``"3.7"`` -> ``"3-7/10"``. Exact, but tenths are not a trade size and no
      buyer searches for them, so ``3.7`` is returned untouched.
    * ``"3.14"`` -> ``"3-9/64"``. Approximate. Inventing precision that was not
      in the source is worse than leaving the decimal alone.
    """
    text = magnitude.strip()
    m = _DECIMAL_RE.match(text)
    if not m:
        return text
    whole, frac_digits = m.group(1), m.group(2)
    frac_value = Fraction(f"0.{frac_digits}")
    if frac_value == 0:
        return whole
    limited = frac_value.limit_denominator(max_denominator)
    if limited != frac_value:
        return text  # not an exact trade fraction; do not fake precision
    den = limited.denominator
    if den != 1 and (den & (den - 1)) != 0:
        return text  # exact but not a binary fraction (tenths, thirds, ...)
    if den == 1:
        return str(int(whole) + limited.numerator)
    frac = f"{limited.numerator}/{den}"
    return frac if whole == "0" else f"{whole}-{frac}"


def to_decimal(magnitude: str) -> float | None:
    """``"50-1/4"`` -> ``50.25``. Returns None if not numeric."""
    text = magnitude.strip()
    m = _FRACTION_RE.match(text)
    if m:
        whole, num, den = m.groups()
        if int(den) == 0:
            return None
        value = int(num) / int(den)
        return (int(whole) if whole else 0) + value
    try:
        return float(text)
    except ValueError:
        return None


def normalise_magnitude(magnitude: str, unit: Unit | None) -> str:
    """Apply fraction preference and strip redundant zeros."""
    text = magnitude.strip().lstrip("+")
    if unit is not None and unit.fractional:
        text = to_fraction(text)
    # 5.0 -> 5, 5.50 -> 5.5, but never touch an already-fractional form
    if "/" not in text and _DECIMAL_RE.match(text):
        trimmed = text.rstrip("0").rstrip(".")
        text = trimmed or "0"
    return text


# --- measurement extraction -------------------------------------------------

#: A magnitude: integer, decimal, bare fraction, or mixed number.
MAGNITUDE = r"\d+(?:\.\d+)?(?:[-\s]\d+/\d+)?|\d+/\d+"

#: All known unit spellings, longest first so ``in.`` beats ``in``.
def _unit_alternation() -> str:
    keys = sorted(_alias_map(), key=len, reverse=True)
    return "|".join(re.escape(k) for k in keys)


@lru_cache(maxsize=1)
def _measure_re() -> re.Pattern[str]:
    # No trailing lookahead here on purpose: whether a following letter is
    # legal depends on the unit that matched and on whether the letter is the
    # dimension separator ``x``. That decision lives in _plausible(), which can
    # express it; a fixed regex lookahead cannot, and gets `14"x1/8"` wrong.
    return re.compile(
        rf"(?P<mag>{MAGNITUDE})\s*(?P<unit>{_unit_alternation()})",
        re.IGNORECASE,
    )


def _plausible(text: str, start: int, end: int, unit: Unit, unit_token: str) -> bool:
    """Is this regex hit a real measurement, or noise inside an identifier?

    Three tests, each earning its place against the working dataset:

    * ``scan_excluded`` units never match in free text (``3M``, ``775L``).
    * A magnitude preceded by a letter is inside a token, not a measurement:
      ``DCB518ASTS06G`` must not yield ``06 g``. A preceding ``x`` is the one
      exception, because it is the dimension separator in ``14"x1/8"``.
    * An *alphabetic* unit followed by a letter is part of a longer word:
      ``TV2000WN`` is not ``2000 W``. A following ``x`` before a digit is again
      the dimension-separator exception. A *symbolic* unit (``"``, ``'``, ``#``)
      needs no such guard, which is exactly why ``14"x`` must stay legal.
    """
    if unit.scan_excluded:
        return False

    prev = text[start - 1] if start > 0 else ""
    if prev.isalpha() and prev not in {"x", "X"}:
        return False

    nxt = text[end] if end < len(text) else ""
    if unit_token[-1:].isalpha() and nxt.isalpha():
        after = text[end + 1] if end + 1 < len(text) else ""
        if not (nxt in {"x", "X"} and (after.isdigit() or after == ".")):
            return False
    return True


@dataclass(frozen=True, slots=True)
class Measurement:
    magnitude: str  # normalised, e.g. "50-1/4"
    unit: Unit
    start: int  # span in the source string
    end: int
    raw: str

    def render(self, mode: RenderMode = RenderMode.SPACED) -> str:
        return self.unit.render(self.magnitude, mode)

    @property
    def family(self) -> str:
        return self.unit.family

    @property
    def numeric(self) -> float | None:
        return to_decimal(self.magnitude)


def find_measurements(text: str) -> list[Measurement]:
    """Extract every ``magnitude + unit`` pair, with source spans.

    Spans matter: they are the evidence a judge sees highlighted in the raw
    description when they hover an output value.
    """
    out: list[Measurement] = []
    for m in _measure_re().finditer(text or ""):
        unit = lookup(m.group("unit"))
        if unit is None:
            continue
        if not _plausible(text, m.start(), m.end(), unit, m.group("unit")):
            continue
        magnitude = normalise_magnitude(m.group("mag"), unit)
        out.append(
            Measurement(
                magnitude=magnitude,
                unit=unit,
                start=m.start(),
                end=m.end(),
                raw=m.group(0),
            )
        )
    return out


#: Dimension runs like ``14"x1/8"x1"``, ``4'x10'``, ``1x6-12'``, ``24 in W x 24-1/4 in D``
_DIM_SPLIT = re.compile(r"\s*[x×]\s*", re.IGNORECASE)


def find_dimension_group(text: str) -> tuple[list[Measurement], int, int] | None:
    """Find the longest ``A x B [x C]`` run of measurements in ``text``.

    Returns the parts plus the overall span, or None. Used for the ``Size``
    attribute and for ordering abrasive/decking dimensions.
    """
    measures = find_measurements(text)
    if len(measures) < 2:
        return None
    best: tuple[list[Measurement], int, int] | None = None
    run: list[Measurement] = []
    for i, m in enumerate(measures):
        if not run:
            run = [m]
            continue
        gap = text[run[-1].end : m.start]
        if _DIM_SPLIT.fullmatch(gap) or gap.strip().lower() in {"x", "×"}:
            run.append(m)
        else:
            if len(run) >= 2 and (best is None or len(run) > len(best[0])):
                best = (list(run), run[0].start, run[-1].end)
            run = [m]
    if len(run) >= 2 and (best is None or len(run) > len(best[0])):
        best = (list(run), run[0].start, run[-1].end)
    return best


def render_dimension_group(
    parts: list[Measurement],
    mode: RenderMode = RenderMode.SPACED,
    labels: tuple[str, ...] = (),
) -> str:
    """``[14 in, 1/8 in, 1 in]`` -> ``"14 in x 1/8 in x 1 in"``.

    With ``labels=("H","W","D")`` -> ``"33-7/16 in H x 23-7/8 in W x ..."``,
    which is the form the gold rows use for the ``Size`` attribute.
    """
    chunks = []
    for i, part in enumerate(parts):
        text = part.render(mode)
        if i < len(labels) and labels[i]:
            text = f"{text} {labels[i]}"
        chunks.append(text)
    return " x ".join(chunks)


@lru_cache(maxsize=1)
def _glued_re() -> re.Pattern[str]:
    return re.compile(
        rf"(?P<mag>\d(?:[\d./-]*\d)?)(?P<unit>{_unit_alternation()})",
        re.IGNORECASE,
    )


def enforce_spacing(text: str, *, protect: tuple[str, ...] = ()) -> str:
    """Repair ``24in`` -> ``24 in`` anywhere in free text.

    A final safety net applied to every generated description, so a house-style
    violation cannot escape even if a generative layer introduced it.

    Two guards keep it from vandalising identifiers, which is the obvious way
    for a naive implementation of this to do more harm than good:

    * a magnitude preceded by a letter is part of a token, not a measurement --
      ``DCB518ASTS06G`` must not become ``DCB518 ASTS06G``. The one exception
      is a preceding ``x``/``×``, which is a dimension separator
      (``14"x1/8"`` -> ``14 in x 1/8 in``).
    * anything in ``protect`` (in practice the MPN) is skipped outright.
    """
    if not text:
        return text

    protected_spans: list[tuple[int, int]] = []
    for token in protect:
        if not token:
            continue
        start = text.find(token)
        while start != -1:
            protected_spans.append((start, start + len(token)))
            start = text.find(token, start + 1)

    def _is_protected(a: int, b: int) -> bool:
        return any(a < end and b > begin for begin, end in protected_spans)

    out: list[str] = []
    cursor = 0
    for m in _glued_re().finditer(text):
        unit = lookup(m.group("unit"))
        if unit is None:
            continue
        start = m.start()
        if not _plausible(text, start, m.end(), unit, m.group("unit")):
            continue
        if _is_protected(start, m.end()):
            continue
        out.append(text[cursor:start])
        out.append(f"{m.group('mag')} {unit.canonical}")
        cursor = m.end()
        # A dimension separator that was glued to the unit needs breathing room
        # too: "1/2 inx18 in" is not an improvement on "1/2"x18"".
        nxt = text[cursor : cursor + 1]
        after = text[cursor + 1 : cursor + 2]
        if nxt in {"x", "X"} and (after.isdigit() or after == "."):
            out.append(" x ")
            cursor += 1
    out.append(text[cursor:])
    return "".join(out)
