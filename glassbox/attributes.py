"""Attribute contracts: the ordered slot list each category must emit.

## The structural finding

Both gold rows are `Built-In Dishwashers` and both emit **exactly the same 15
attribute labels in exactly the same order**, including labels whose value is
blank. Row 1 leaves `Model`, `Plug Type` and `Color` empty; row 2 leaves
`Model`, `Number of Wash Cycles`, `Plug Type` and `Maximum Height` empty. Both
still emit all fifteen labels.

Attribute output is therefore a **contract**, not free extraction. A pipeline
that emits "whatever it found" produces a structurally wrong row even when
every value it found is correct, because the slots no longer line up with the
category. So each category carries an ordered :class:`Contract`, and the
renderer always walks the full slot list.

## The honesty finding

This one matters more. Gold row 1's input is the entire string:

    PDSH4816AF Dishwasher SS - Display Only

and its output asserts `120 V`, `15 A`, `47 dBA`, `24 in W x 24-1/4 in D`,
`50-1/4 in Depth With Door Open`, ENERGY STAR certification and a 1-year
warranty. **None of that is in the input.** It was enriched from the
manufacturer's own site -- which is why the row also carries
`MFR URL = frigidaire.com/.../PDSH4816AF`.

No amount of AI recovers those numbers from six columns of distributor data.
Pretending otherwise is precisely the failure mode the Solution Guide warns
about: *"A fluent description made of invented values scores zero."*

So values are separated into two tiers that are never mixed:

* **EXTRACTED** -- read out of the input text, with a character span proving it.
  High confidence, no review needed.
* **PROPOSED** -- supplied by a model's parametric knowledge of the specific
  part number, or by a manufacturer-site fetch. Tagged with its source, held at
  low confidence, and always routed to human review before it can be trusted.

The delivery sheet is filled from both, but the audit sidecar keeps them
distinguishable, and the confidence score treats them very differently. A
reviewer can therefore see at a glance which cells are facts about the input
and which are a machine's best guess about the world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from . import textnorm as T
from . import units as U
from .provenance import Cell, Evidence, Provenance, Source, contract_blank


class Kind(str, Enum):
    """What sort of thing fills a slot, which decides how it is extracted."""

    MEASURE = "measure"  # magnitude + unit of a given family
    DIMENSION = "dimension"  # an A x B [x C] run
    LOV = "lov"  # one of an enumerated set of permitted values
    SERIES = "series"  # a product-line name
    MODEL = "model"  # a model designator distinct from the MPN
    COUNT = "count"  # a bare integer (wash cycles, tooth count)
    TEXT = "text"  # free text, last resort
    ADDITIONAL = "additional"  # the trailing catch-all list


class Style(str, Enum):
    """How a filled slot renders into prose. Derived from the gold rows."""

    #: "Stainless Steel" -- bare value, no label.
    VALUE = "value"
    #: "Leg Mounting" -- value then the head noun of the label.
    VALUE_LABEL_HEAD = "value_label_head"
    #: "47 dBA Sound Level" -- value, unit, then the full label.
    VALUE_UOM_LABEL = "value_uom_label"
    #: "50-1/4 in Depth With Door Open" -- same shape, kept separate because
    #: dimensional slots read naturally and acoustic ones do not.
    VALUE_UOM_FULL_LABEL = "value_uom_full_label"
    #: "24 in W x 24-1/4 in D" -- the value already reads as a phrase.
    VALUE_ONLY_UOM = "value_only_uom"
    #: "5-Wash Cycle" in the title, "5 Wash Cycles" in long copy.
    COUNT_HYPHEN = "count_hyphen"
    #: "Professional Series" -- value already contains its own noun.
    SERIES = "series"
    #: "Additional Information: a, b, c" -- always last.
    SUFFIX_LIST = "suffix_list"
    #: Never rendered into prose; structured output only.
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class Slot:
    """One position in a category's attribute contract."""

    label: str
    kind: Kind = Kind.TEXT
    style: Style = Style.VALUE
    #: For MEASURE slots: the measurement family to look for.
    family: str = ""
    #: For MEASURE slots: plausible magnitude bounds, which is how "15 A" is
    #: distinguished from "15 in" when both are amperage-shaped numbers.
    lo: float | None = None
    hi: float | None = None
    #: For LOV slots: the permitted values. This *is* the controlled
    #: vocabulary -- nothing outside it may ever be emitted.
    lov: tuple[str, ...] = ()
    #: Extra surface forms mapping onto a LOV value: ("SST", "Stainless Steel").
    lov_aliases: tuple[tuple[str, str], ...] = ()
    #: Phrases that, if present, select this slot's value directly.
    cues: tuple[str, ...] = ()
    #: Channel participation.
    in_title: bool = False
    in_invoice: bool = False
    in_mobile: bool = False
    in_long: bool = True
    #: Ordering weight for LONG_DESC1 and the attribute block, which follow
    #: contract order. Lower comes first.
    rank: int = 50
    #: Ordering weight for SHORT_DESC and INVOICE_DESC, which do **not** follow
    #: contract order. Both gold rows put mounting first, then counts, then
    #: material and colour, and only then electrical and dimensional values:
    #:
    #:   DISHWASHER  LEG   5    SST        120V 15A  50-1/4IN
    #:               ^mnt  ^cyc ^material  ^electrical  ^dimension
    #:
    #: Slots leaving this at 0 sort after every ranked slot, in `rank` order.
    title_rank: int = 0

    @property
    def channel_rank(self) -> int:
        """Sort key for the title and invoice channels."""
        return self.title_rank if self.title_rank else 1000 + self.rank

    @property
    def label_head(self) -> str:
        """``"Mounting Type"`` -> ``"Mounting"``; used by VALUE_LABEL_HEAD."""
        words = self.label.split()
        if len(words) > 1 and words[-1].lower() in {
            "type", "rating", "level", "style", "option", "class", "grade",
        }:
            return " ".join(words[:-1])
        return self.label


@dataclass(frozen=True, slots=True)
class Contract:
    """An ordered attribute contract for one category."""

    name: str
    slots: tuple[Slot, ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.slots)

    def slot(self, label: str) -> Slot | None:
        for s in self.slots:
            if s.label == label:
                return s
        return None


# --- shared vocabularies ----------------------------------------------------
# These are the controlled vocabularies. They stand in for the LOV workbook
# that the portal does not publish, and they are extended at runtime by the
# values the induction pass observes in the corpus.

MATERIALS = (
    "Stainless Steel", "Black Stainless Steel", "Steel", "Aluminum", "Brass",
    "Bronze", "Copper", "Cast Iron", "Chrome", "Nickel", "Zinc", "Plastic",
    "PVC", "CPVC", "ABS", "Polyethylene", "Nylon", "Rubber", "Ceramic",
    "Porcelain", "Glass", "Wood", "Composite", "Fiber Cement", "Vinyl",
    "Fiberglass", "Gypsum", "Concrete", "Leather", "Canvas", "Aluminum Oxide",
    "Silicon Carbide", "Zirconia Alumina", "Ceramic Aluminum Oxide",
    "Carbide", "Diamond", "Bi-Metal", "High Speed Steel",
)

COLORS = (
    "White", "Black", "Almond", "Biscuit", "Bisque", "Stainless Steel",
    "Black Stainless Steel", "Gray", "Charcoal", "Brown", "Beige", "Clay",
    "Silver", "Gold", "Bronze", "Chrome", "Nickel", "Brushed Nickel",
    "Satin Nickel", "Polished Nickel", "Polished Brass", "Oil Rubbed Bronze",
    "Matte Black", "Clear", "Frosted", "Red", "Blue", "Green", "Yellow",
    "Orange", "Ivory", "Natural",
)

_COLOR_ALIASES = (
    ("SS", "Stainless Steel"), ("SST", "Stainless Steel"),
    ("STS", "Stainless Steel"), ("BSS", "Black Stainless Steel"),
    ("BK", "Black"), ("BLK", "Black"), ("WH", "White"), ("WHT", "White"),
    ("ALM", "Almond"), ("BZ", "Bronze"), ("ORB", "Oil Rubbed Bronze"),
    ("SN", "Satin Nickel"), ("BN", "Brushed Nickel"), ("PN", "Polished Nickel"),
    ("PB", "Polished Brass"), ("CLR", "Clear"), ("GRY", "Gray"),
    ("BSL", "Brushed Stainless"), ("BO", "Black"), ("DBZ", "Dark Bronze"),
)

MOUNTING = (
    "Built-in", "Free Standing", "Slide-in", "Drop-in", "Under Counter",
    "Countertop", "Wall Mount", "Ceiling Mount", "Flush Mount", "Semi-Flush",
    "Surface Mount", "Recessed", "Pendant", "Post Mount", "Leg", "Pedestal",
    "Undermount", "Over the Range", "Chain Hung", "Track",
)

FUEL = ("Electric", "Gas", "Natural Gas", "Propane", "Dual Fuel", "Induction")

LAMP_BASE = (
    "Medium", "Candelabra", "Intermediate", "Mogul", "GU10", "GU24", "G9",
    "G4", "Bi-Pin", "Wedge", "E26", "E12", "E39", "Pin",
)

LAMP_SHAPE = (
    "A19", "A15", "A21", "BA11", "B11", "BR30", "BR40", "PAR16", "PAR20",
    "PAR30", "PAR38", "MR16", "ST19", "ST64", "G16", "G25", "G40", "T8",
    "T5", "T12", "CA10", "F15", "S14",
)

EDGE_PROFILE = ("Grooved", "Square Edge", "Scalloped", "Bullnose", "Tongue and Groove")

ABRASIVE_BACKING = ("Cloth", "Paper", "Film", "Fiber", "Mesh", "Net", "Foam")


def _lov_material() -> Slot:
    return Slot(
        "Material", Kind.LOV, Style.VALUE, lov=MATERIALS,
        lov_aliases=_COLOR_ALIASES, in_title=True, in_invoice=True, rank=80,
        title_rank=30,
    )


def _lov_color() -> Slot:
    return Slot(
        "Color", Kind.LOV, Style.VALUE, lov=COLORS,
        lov_aliases=_COLOR_ALIASES, in_title=True, in_invoice=True, rank=85,
        title_rank=35,
    )


def _series() -> Slot:
    return Slot(
        "Series", Kind.SERIES, Style.SERIES,
        in_title=True, in_mobile=True, rank=5,
    )


def _model() -> Slot:
    return Slot("Model", Kind.MODEL, Style.HIDDEN, in_long=False, rank=6)


def _additional() -> Slot:
    return Slot(
        "Additional Information", Kind.ADDITIONAL, Style.SUFFIX_LIST, rank=999
    )


def _voltage() -> Slot:
    return Slot(
        "Voltage Rating", Kind.MEASURE, Style.VALUE_ONLY_UOM, family="voltage",
        lo=1, hi=1000, in_invoice=True, rank=30,
    )


def _amperage() -> Slot:
    return Slot(
        "Amperage Rating", Kind.MEASURE, Style.VALUE_ONLY_UOM, family="current",
        lo=0.1, hi=400, in_invoice=True, rank=32,
    )


def _sound() -> Slot:
    return Slot(
        "Sound Level", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="sound level",
        lo=20, hi=120, in_invoice=True, rank=70,
    )


def _size() -> Slot:
    return Slot(
        "Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, in_title=False,
        in_invoice=False, rank=60,
    )


def _mounting(values: tuple[str, ...] = MOUNTING) -> Slot:
    return Slot(
        "Mounting Type", Kind.LOV, Style.VALUE_LABEL_HEAD, lov=values,
        lov_aliases=(("BLTLN", "Built-in"), ("BLTIN", "Built-in"),
                     ("FS", "Free Standing"), ("SLIDEIN", "Slide-in")),
        in_title=True, in_invoice=True, in_mobile=True, rank=40,
        title_rank=10,
    )


# --- the contracts ----------------------------------------------------------

CONTRACTS: dict[str, Contract] = {}


def _register(contract: Contract) -> Contract:
    CONTRACTS[contract.name] = contract
    return contract


#: Verified against both published gold rows, label-for-label and in order.
#: This is the one contract in the file that is ground truth rather than
#: reasoned design, so it is kept exactly as observed.
DISHWASHER = _register(
    Contract(
        "dishwasher",
        (
            _series(),
            _model(),
            Slot(
                "Number of Wash Cycles", Kind.COUNT, Style.COUNT_HYPHEN,
                cues=("wash cycle", "cycle"), lo=1, hi=20,
                in_title=True, in_invoice=True, rank=20, title_rank=20,
            ),
            _voltage(),
            _amperage(),
            _mounting(),
            Slot(
                "Plug Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                lov=("NEMA 5-15P", "NEMA 5-20P", "NEMA 14-30P", "NEMA 14-50P",
                     "Cord and Plug", "Hardwired"),
                rank=45,
            ),
            _size(),
            Slot(
                "Depth With Door Open", Kind.MEASURE, Style.VALUE_UOM_FULL_LABEL,
                family="length", lo=30, hi=80, cues=("door open",), rank=62,
                in_invoice=True,
            ),
            Slot(
                "Minimum Height", Kind.MEASURE, Style.VALUE_UOM_FULL_LABEL,
                family="length", lo=10, hi=100, cues=("minimum height", "min height"),
                rank=64,
            ),
            Slot(
                "Maximum Height", Kind.MEASURE, Style.VALUE_UOM_FULL_LABEL,
                family="length", lo=10, hi=100, cues=("maximum height", "max height"),
                rank=66,
            ),
            _sound(),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "laundry",
        (
            _series(),
            _model(),
            Slot("Fuel Type", Kind.LOV, Style.VALUE_LABEL_HEAD, lov=FUEL,
                 in_title=True, in_invoice=True, in_mobile=True, rank=10),
            Slot("Capacity", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="volume",
                 lo=1, hi=10, in_title=True, rank=18),
            Slot("Number of Cycles", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("cycle",), lo=1, hi=30, rank=20),
            _voltage(),
            _amperage(),
            _mounting(("Free Standing", "Stackable", "Under Counter", "Built-in")),
            _size(),
            Slot("Load Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Top Load", "Front Load"), in_title=True, rank=22),
            _sound(),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "refrigerator",
        (
            _series(),
            _model(),
            Slot("Configuration", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("French Door", "Side by Side", "Top Freezer",
                      "Bottom Freezer", "Single Door", "Counter Depth",
                      "Upright", "Chest"),
                 in_title=True, in_mobile=True, rank=12),
            Slot("Total Capacity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="volume", lo=1, hi=40, in_title=True, rank=18),
            _voltage(),
            _amperage(),
            _mounting(("Free Standing", "Built-in", "Counter Depth", "Under Counter")),
            _size(),
            Slot("Number of Doors", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("door",), lo=1, hi=4, rank=24),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "range",
        (
            _series(),
            _model(),
            Slot("Fuel Type", Kind.LOV, Style.VALUE_LABEL_HEAD, lov=FUEL,
                 in_title=True, in_invoice=True, in_mobile=True, rank=10),
            Slot("Width", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=12, hi=60, in_title=True, in_invoice=True, rank=15),
            Slot("Number of Burners", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("burner", "element"), lo=1, hi=8, rank=20),
            _voltage(),
            _amperage(),
            _mounting(("Free Standing", "Slide-in", "Drop-in", "Built-in", "Countertop")),
            _size(),
            Slot("Oven Capacity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="volume", lo=1, hi=12, rank=50),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "lamp",
        (
            _series(),
            _model(),
            Slot("Wattage", Kind.MEASURE, Style.VALUE_ONLY_UOM, family="power",
                 lo=0.5, hi=1000, in_title=True, in_invoice=True, in_mobile=True,
                 rank=10),
            Slot("Lamp Shape", Kind.LOV, Style.VALUE_LABEL_HEAD, lov=LAMP_SHAPE,
                 in_title=True, in_invoice=True, rank=14),
            Slot("Base Type", Kind.LOV, Style.VALUE_LABEL_HEAD, lov=LAMP_BASE,
                 lov_aliases=(("MED", "Medium"), ("CAND", "Candelabra"),
                              ("INT", "Intermediate")),
                 in_title=True, in_invoice=True, rank=18),
            Slot("Color Temperature", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="color temperature", lo=1500, hi=7000,
                 in_title=True, in_invoice=True, rank=22),
            Slot("Lumens", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="luminous flux", lo=20, hi=30000, rank=26),
            _voltage(),
            Slot("Bulb Technology", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("LED", "Incandescent", "Fluorescent", "CFL", "Halogen", "HID"),
                 in_title=True, rank=8),
            Slot("Dimmable", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Dimmable", "Non-Dimmable"), rank=30),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=100, in_invoice=True, rank=90),
            Slot("Average Life", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="time", lo=100, hi=100000, rank=40),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "fixture",
        (
            _series(),
            _model(),
            Slot("Number of Lights", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("light", "lamp"), lo=1, hi=24, in_title=True, rank=12),
            Slot("Width", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=1, hi=120, in_title=True, in_invoice=True, rank=16),
            _mounting(("Flush Mount", "Semi-Flush", "Pendant", "Wall Mount",
                       "Ceiling Mount", "Recessed", "Surface Mount",
                       "Chain Hung", "Post Mount", "Track")),
            Slot("Wattage", Kind.MEASURE, Style.VALUE_ONLY_UOM, family="power",
                 lo=0.5, hi=1000, in_invoice=True, rank=24),
            _voltage(),
            Slot("Color Temperature", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="color temperature", lo=1500, hi=7000, rank=28),
            Slot("Lumens", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="luminous flux", lo=20, hi=60000, rank=30),
            Slot("Location Rating", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Indoor", "Outdoor", "Damp Location", "Wet Location"),
                 rank=44),
            _size(),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "abrasive_wheel",
        (
            _series(),
            Slot("Diameter", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.5, hi=30, in_title=True, in_invoice=True, rank=10),
            Slot("Thickness", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.01, hi=2, in_title=True, in_invoice=True, rank=12),
            Slot("Arbor Size", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=3, in_title=True, in_invoice=True, rank=14),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=16),
            Slot("Application Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Metal", "Masonry", "Stainless Steel", "Concrete", "Stone",
                      "Tile", "Cast Iron", "Aluminum", "Wood"),
                 in_title=True, in_invoice=True, rank=20),
            Slot("Abrasive Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Aluminum Oxide", "Silicon Carbide", "Zirconia Alumina",
                      "Ceramic Aluminum Oxide", "Diamond"),
                 rank=24),
            Slot("Wheel Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Type 1", "Type 27", "Type 29", "Depressed Center", "Flat"),
                 rank=28),
            Slot("Maximum Speed", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="rotational speed", lo=100, hi=100000, rank=40),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=500, in_invoice=True, rank=90),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "abrasive_sheet",
        (
            _series(),
            Slot("Grit", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="abrasive grade", lo=8, hi=5000,
                 in_title=True, in_invoice=True, in_mobile=True, rank=10),
            Slot("Width", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=60, in_title=True, in_invoice=True, rank=14),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=600, in_title=True, in_invoice=True, rank=16),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=18),
            Slot("Abrasive Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Aluminum Oxide", "Silicon Carbide", "Zirconia Alumina",
                      "Ceramic Aluminum Oxide", "Garnet"),
                 rank=24),
            Slot("Backing Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=ABRASIVE_BACKING, rank=28),
            Slot("Attachment Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Hook and Loop", "Pressure Sensitive Adhesive", "Stikit",
                      "Plain Back", "Arbor Hole"),
                 rank=32),
            Slot("Number of Holes", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("hole",), lo=0, hi=200, rank=36),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=500, in_invoice=True, rank=90),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "saw_blade",
        (
            _series(),
            Slot("Diameter", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=1, hi=30, in_title=True, in_invoice=True, rank=10),
            Slot("Tooth Count", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("tooth", "teeth"), lo=2, hi=200,
                 in_title=True, in_invoice=True, rank=14),
            Slot("Teeth Per Inch", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="tooth pitch", lo=1, hi=40, rank=16),
            Slot("Arbor Size", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=3, in_invoice=True, rank=18),
            Slot("Kerf", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.01, hi=0.5, rank=22),
            Slot("Application Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Wood", "Metal", "Masonry", "Plastic", "Tile", "Concrete",
                      "Multi-Material", "Aluminum", "Cast Iron"),
                 in_title=True, rank=26),
            Slot("Blade Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Carbide", "Bi-Metal", "High Speed Steel", "Diamond",
                      "Carbon Steel"),
                 rank=30),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=200, in_invoice=True, rank=90),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "bit",
        (
            _series(),
            Slot("Drive Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Phillips", "Square", "Torx", "Hex", "Slotted", "Combination",
                      "Pozidriv", "Star", "Spline"),
                 in_title=True, in_invoice=True, rank=10),
            Slot("Drive Size", Kind.TEXT, Style.VALUE_LABEL_HEAD,
                 cues=("#1", "#2", "#3", "#0", "t10", "t15", "t20", "t25", "t27", "t30"),
                 in_title=True, in_invoice=True, rank=12),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.2, hi=30, in_title=True, in_invoice=True, rank=16),
            Slot("Diameter", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.01, hi=6, in_title=True, rank=18),
            Slot("Shank Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("1/4 in Hex", "Round", "SDS-Plus", "SDS-Max", "Straight",
                      "Quick Change"),
                 rank=24),
            Slot("Bit Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("High Speed Steel", "Carbide", "Cobalt", "Titanium",
                      "Black Oxide", "Diamond"),
                 rank=28),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=200, in_invoice=True, rank=90),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "power_tool",
        (
            _series(),
            _model(),
            Slot("Power Source", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Cordless", "Corded Electric", "Pneumatic", "Gas"),
                 in_title=True, in_invoice=True, in_mobile=True, rank=10),
            Slot("Voltage Rating", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="voltage", lo=3, hi=600,
                 in_title=True, in_invoice=True, rank=14),
            Slot("Motor Power", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="power",
                 lo=0.1, hi=20, in_title=True, in_invoice=True, rank=18),
            _amperage(),
            Slot("No Load Speed", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="rotational speed", lo=100, hi=60000, rank=24),
            Slot("Blade or Wheel Size", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=1, hi=30, in_title=True, rank=28),
            Slot("Electrical Phase", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="electrical phase", lo=1, hi=3, rank=32),
            _size(),
            Slot("Included Items", Kind.TEXT, Style.VALUE_LABEL_HEAD,
                 cues=("kit", "bare tool", "tool only", "with battery"), rank=60),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "decking",
        (
            _series(),
            Slot("Color", Kind.SERIES, Style.VALUE, in_title=True,
                 in_invoice=True, in_mobile=True, rank=8),
            Slot("Nominal Thickness", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=0.4, hi=3, in_title=True, rank=12),
            Slot("Nominal Width", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=1, hi=24, in_title=True, in_invoice=True,
                 rank=14),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=4, hi=40, in_title=True, in_invoice=True, rank=16),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=18),
            Slot("Edge Profile", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=EDGE_PROFILE,
                 lov_aliases=(("GRVD", "Grooved"), ("SQ EDGE", "Square Edge")),
                 in_title=True, in_invoice=True, rank=22),
            Slot("Material", Kind.LOV, Style.VALUE,
                 lov=("Composite", "PVC", "Cellular PVC", "Capped Composite",
                      "Wood", "Aluminum"),
                 in_title=True, rank=26),
            Slot("Surface Texture", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Wood Grain", "Smooth", "Brushed", "Embossed", "Variegated"),
                 rank=30),
            Slot("Warranty Term", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="time", lo=1, hi=100, rank=50),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "sheet_good",
        (
            _series(),
            Slot("Nominal Thickness", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=0.05, hi=6, in_title=True, in_invoice=True,
                 rank=10),
            Slot("Nominal Width", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=0.5, hi=20, in_title=True, in_invoice=True,
                 rank=12),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=1, hi=100, in_title=True, in_invoice=True, rank=14),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=16),
            Slot("Material", Kind.LOV, Style.VALUE,
                 lov=("Fiber Cement", "Engineered Wood", "OSB", "Plywood",
                      "Gypsum", "Vinyl", "Steel", "Aluminum", "Composite",
                      "Asphalt", "Mineral Fiber"),
                 in_title=True, rank=20),
            Slot("Surface Finish", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Smooth", "Primed", "Unfinished", "Cedarmill", "Textured",
                      "Painted", "Stucco"),
                 in_title=True, in_invoice=True, rank=24),
            Slot("Edge Profile", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=EDGE_PROFILE, rank=28),
            Slot("Coverage", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="area",
                 lo=1, hi=2000, rank=40),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "lumber",
        (
            Slot("Species", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Douglas Fir", "Southern Yellow Pine", "White Pine",
                      "Spruce", "Hemlock", "Cedar", "Redwood", "Poplar", "Oak"),
                 in_title=True, in_invoice=True, rank=8),
            Slot("Nominal Thickness", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=0.4, hi=12, in_title=True, rank=10),
            Slot("Nominal Width", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="length", lo=1, hi=24, in_title=True, rank=12),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=2, hi=40, in_title=True, in_invoice=True, rank=14),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=16),
            Slot("Grade", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Select", "STK", "Clear", "Number 1", "Number 2",
                      "Construction", "Standard", "Utility"),
                 in_title=True, rank=20),
            Slot("Treatment", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Pressure Treated", "Kiln Dried", "Untreated",
                      "Fire Retardant", "Green"),
                 in_title=True, rank=24),
            Slot("Surface Finish", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Smooth", "Rough Sawn", "S4S", "S1S2E"), rank=28),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "wire",
        (
            _series(),
            Slot("Conductor Size", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="wire gauge", lo=1, hi=2000,
                 in_title=True, in_invoice=True, in_mobile=True, rank=8),
            Slot("Number of Conductors", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("conductor",), lo=1, hi=60,
                 in_title=True, in_invoice=True, rank=12),
            Slot("Conductor Material", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Copper", "Aluminum", "Copper Clad Aluminum", "Tinned Copper"),
                 in_title=True, in_invoice=True, rank=16),
            Slot("Insulation Type", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("THHN", "THWN", "THWN-2", "XHHW", "NM-B", "UF-B", "SO",
                      "SOOW", "SJOOW", "SJTW", "SE", "USE", "URD", "MC", "PVC"),
                 in_title=True, rank=20),
            _voltage(),
            Slot("Amperage Rating", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="current", lo=1, hi=2000, rank=24),
            Slot("Conductor Strand", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Solid", "Stranded"), rank=28),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=1, hi=5000, in_invoice=True, rank=40),
            _lov_color(),
            _additional(),
        ),
    )
)

_register(
    Contract(
        "wiring_device",
        (
            _series(),
            Slot("Amperage Rating", Kind.MEASURE, Style.VALUE_ONLY_UOM,
                 family="current", lo=1, hi=200,
                 in_title=True, in_invoice=True, rank=10),
            _voltage(),
            Slot("Number of Gangs", Kind.COUNT, Style.COUNT_HYPHEN, title_rank=20,
                 cues=("gang", "g"), lo=1, hi=8, in_title=True,
                 in_invoice=True, rank=14),
            Slot("Configuration", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Single Pole", "Three Way", "Four Way", "Duplex", "Simplex",
                      "GFCI", "AFCI", "Combination", "Decora", "Toggle", "Rocker"),
                 lov_aliases=(("GFI", "GFCI"), ("AFI", "AFCI"),
                              ("3-Way", "Three Way"), ("4-Way", "Four Way")),
                 in_title=True, rank=18),
            _mounting(("Flush Mount", "Surface Mount", "Wall Mount", "Recessed")),
            Slot("Grade", Kind.LOV, Style.VALUE_LABEL_HEAD,
                 lov=("Residential", "Commercial", "Industrial", "Hospital", "Specification"),
                 rank=30),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)

#: Fallback for a category with no bespoke contract yet. Deliberately short:
#: an over-long generic contract would emit a wall of blank labels and make the
#: output look worse than it is.
_register(
    Contract(
        "generic",
        (
            _series(),
            _model(),
            Slot("Width", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=200, in_title=True, in_invoice=True, rank=12),
            Slot("Length", Kind.MEASURE, Style.VALUE_UOM_LABEL, family="length",
                 lo=0.1, hi=1000, in_title=True, in_invoice=True, rank=14),
            Slot("Size", Kind.DIMENSION, Style.VALUE_ONLY_UOM, rank=16),
            _voltage(),
            _amperage(),
            Slot("Wattage", Kind.MEASURE, Style.VALUE_ONLY_UOM, family="power",
                 lo=0.1, hi=5000, in_invoice=True, rank=24),
            _mounting(),
            Slot("Pack Quantity", Kind.MEASURE, Style.VALUE_UOM_LABEL,
                 family="count", lo=1, hi=1000, in_invoice=True, rank=90),
            _lov_material(),
            _lov_color(),
            _additional(),
        ),
    )
)


def contract_for(name: str) -> Contract:
    return CONTRACTS.get(name) or CONTRACTS["generic"]


def all_lov_values() -> dict[str, tuple[str, ...]]:
    """Every controlled vocabulary in the system, for the compliance report."""
    out: dict[str, tuple[str, ...]] = {}
    for contract in CONTRACTS.values():
        for slot in contract.slots:
            if slot.lov:
                existing = set(out.get(slot.label, ()))
                out[slot.label] = tuple(sorted(existing | set(slot.lov)))
    return out


# --- dimension conventions --------------------------------------------------
# A run of lengths is not a set of interchangeable numbers. Trade order is
# fixed and category-specific, and getting it wrong silently swaps values:
#
#   49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc
#       -> 14 in diameter, 1/8 in thickness, 1 in arbor   (never the reverse)
#   1nx6-20' Pebble Beach Grooved - Trex Enhance Basics Decking
#       -> 1 in nominal thickness, 6 in nominal width, 20 ft length
#   4'x10' HardiePanel Smooth - Primed
#       -> 4 ft width, 10 ft length
#
# Keyed by arity, because the same category writes 2- and 3-part runs and the
# meaning of the first number changes with the count.
DIM_ORDERS: dict[str, dict[int, tuple[str, ...]]] = {
    "abrasive_wheel": {
        2: ("Diameter", "Thickness"),
        3: ("Diameter", "Thickness", "Arbor Size"),
    },
    "abrasive_sheet": {
        2: ("Width", "Length"),
        3: ("Width", "Length", "Thickness"),
    },
    "saw_blade": {
        2: ("Diameter", "Arbor Size"),
        3: ("Diameter", "Kerf", "Arbor Size"),
    },
    "decking": {
        2: ("Nominal Width", "Length"),
        3: ("Nominal Thickness", "Nominal Width", "Length"),
    },
    "sheet_good": {
        2: ("Nominal Width", "Length"),
        3: ("Nominal Thickness", "Nominal Width", "Length"),
    },
    "lumber": {
        2: ("Nominal Width", "Length"),
        3: ("Nominal Thickness", "Nominal Width", "Length"),
    },
    "bit": {
        2: ("Diameter", "Length"),
    },
    "generic": {
        2: ("Width", "Length"),
        3: ("Width", "Length", "Height"),
    },
}


def dim_order(contract_name: str, arity: int) -> tuple[str, ...]:
    """Which slots a dimension run of ``arity`` parts maps onto."""
    table = DIM_ORDERS.get(contract_name) or DIM_ORDERS["generic"]
    if arity in table:
        return table[arity]
    # fall back to the longest defined order, truncated
    if not table:
        return ()
    longest = table[max(table)]
    return longest[:arity]
