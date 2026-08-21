"""Taxonomy: `Classpath`, `Dept`/`Class`/`Fine`, and the product noun.

The gold rows require **two** taxonomies to be produced, and to agree:

    Classpath : Appliances & Consumer Electronics>Kitchen Appliances>Built-In Dishwashers
    Dept/Class/Fine : Appliances / Large Appliances / Dishwashers

So we classify once into a leaf :class:`Category` and read every taxonomy field
off that leaf. The two can then never disagree, which a pipeline that predicts
them independently cannot guarantee.

## Scope

The seed taxonomy below covers the categories that are *actually in the working
dataset*, which is not what the Solution Guide suggests. The guide recommends
going deep on Faucets or Fittings because those two reference specs are fully
worked -- but there is not one faucet or fitting in the 1,000 rows. The real
distribution is lighting, abrasives and power-tool accessories, composite
decking, appliances, wire and electrical, power tools, and building materials.
See docs/DERIVED_RULES.md section 10.

## Classification

Three layers, tried in order, each recording its own provenance:

1. **Lexical contract** -- weighted keyword evidence with vetoes. High
   precision, and fully explainable: the audit trail names the phrases that
   fired.
2. **Group prior** -- `Part_Manuf` is highly predictive (Kichler Lighting sells
   light fixtures), so the corpus-derived prior breaks ties. Learned, not typed.
3. **Optional model** -- a fine-tuned local classifier or the NVIDIA-hosted
   teacher, used only when the first two are inconclusive. Never allowed to
   invent a classpath outside the taxonomy.

Layer 1 alone is deliberately strong enough to run the whole catalogue with no
model and no network, which is what makes the prototype safe to hand to a judge.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from . import textnorm as T
from .provenance import Cell, Evidence, Provenance, Source, unresolved


@dataclass(frozen=True, slots=True)
class Category:
    """One leaf of the taxonomy, plus the evidence that selects it."""

    classpath: str  # "A>B>C", exactly as the delivery format wants it
    dept: str
    klass: str
    fine: str
    product_name: str  # the noun that heads every description
    contract: str = "generic"  # key into attributes.CONTRACTS
    #: At least one of these must appear. Phrases are matched on word
    #: boundaries, case-insensitively, after abbreviation expansion.
    must: tuple[str, ...] = ()
    #: Supporting evidence; each hit adds to the score.
    boost: tuple[str, ...] = ()
    #: Disqualifying evidence. A veto is absolute -- it is how "Dishwasher" is
    #: kept from claiming "Dishwasher Detergent".
    veto: tuple[str, ...] = ()

    @property
    def leaf(self) -> str:
        return self.classpath.rsplit(">", 1)[-1]


# --- the seed taxonomy ------------------------------------------------------
# Ordered by department for readability only; scoring is order-independent
# except for a deterministic tie-break on specificity.

APPLIANCES = "Appliances & Consumer Electronics"
LIGHTING = "Electrical & Lighting"
TOOLS = "Tools & Equipment"
BUILDING = "Building Materials & Millwork"
ELECTRICAL = "Electrical & Lighting"
SAFETY = "Safety & Personal Protection"

CATEGORIES: tuple[Category, ...] = (
    # ---------------- appliances ----------------
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Built-In Dishwashers",
        "Appliances", "Large Appliances", "Dishwashers",
        "Dishwasher", contract="dishwasher",
        must=("dishwasher", "dish washer"),
        veto=("detergent", "pod", "rinse aid", "cleaner", "rack only"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Refrigerators",
        "Appliances", "Large Appliances", "Refrigerators",
        "Refrigerator", contract="refrigerator",
        must=("refrigerator", "fridge", "refrig"),
        boost=("french door", "side by side", "top freezer", "bottom freezer", "counter depth"),
        veto=("water filter", "filter only", "ice maker only"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Freezers",
        "Appliances", "Large Appliances", "Freezers",
        "Freezer", contract="refrigerator",
        must=("freezer", "frzr"),
        boost=("upright", "chest"),
        veto=("refrigerator", "fridge"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Ranges",
        "Appliances", "Large Appliances", "Ranges",
        "Range", contract="range",
        must=("range", "rng"),
        boost=("electric", "gas", "slide-in", "free standing", "convection", "induction"),
        veto=("range hood", "hood", "extension", "range extender"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Cooktops",
        "Appliances", "Large Appliances", "Cooktops",
        "Cooktop", contract="range",
        must=("cooktop", "cook top", "ckt"),
        boost=("induction", "radiant", "gas", "electric"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Wall Ovens",
        "Appliances", "Large Appliances", "Wall Ovens",
        "Wall Oven", contract="range",
        must=("wall oven", "wall ov", "double oven", "single oven"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Microwave Ovens",
        "Appliances", "Large Appliances", "Microwaves",
        "Microwave Oven", contract="generic",
        must=("microwave", "mwo"),
        boost=("over the range", "countertop", "built-in", "drawer"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Range Hoods",
        "Appliances", "Large Appliances", "Range Hoods",
        "Range Hood", contract="generic",
        must=("range hood", "vent hood", "hood insert", "chimney hood"),
    ),
    Category(
        f"{APPLIANCES}>Laundry Appliances>Washers",
        "Appliances", "Large Appliances", "Washers",
        "Washer", contract="laundry",
        must=("washer", "wshr", "washing machine"),
        boost=("top load", "front load", "agitator", "impeller"),
        veto=("dishwasher", "pressure washer", "washer set", "flat washer", "lock washer"),
    ),
    Category(
        f"{APPLIANCES}>Laundry Appliances>Dryers",
        "Appliances", "Large Appliances", "Dryers",
        "Dryer", contract="laundry",
        must=("dryer", "dryr"),
        boost=("electric", "gas", "stackable", "vented", "ventless"),
        veto=("hair dryer", "blow dryer", "air dryer"),
    ),
    Category(
        f"{APPLIANCES}>Kitchen Appliances>Food Waste Disposers",
        "Appliances", "Small Appliances", "Disposers",
        "Disposer", contract="generic",
        must=("disposer", "disposal", "dsps"),
        boost=("continuous feed", "batch feed", "garbage"),
    ),
    # ---------------- lighting ----------------
    Category(
        f"{LIGHTING}>Lamps & Light Bulbs>LED Lamps",
        "Electrical", "Lighting", "Lamps",
        "LED Lamp", contract="lamp",
        must=("led",),
        boost=("bulb", "lamp", "a19", "br30", "br40", "par20", "par30", "par38",
               "candelabra", "cand", "ba11", "b11", "st19", "g25", "mr16",
               "medium", "med", "e26", "e12", "gu10", "dimmable"),
        veto=("led light", "downlight", "troffer", "fixture", "wall pack",
              "strip light", "tape light", "flood light", "ceiling", "pendant",
              "chandelier", "sconce", "driver", "power supply"),
    ),
    Category(
        f"{LIGHTING}>Lamps & Light Bulbs>Incandescent Lamps",
        "Electrical", "Lighting", "Lamps",
        "Incandescent Lamp", contract="lamp",
        must=("incandescent", "incan", "halogen", "hal"),
        boost=("bulb", "lamp"),
        veto=("fixture",),
    ),
    Category(
        f"{LIGHTING}>Lamps & Light Bulbs>Fluorescent Lamps",
        "Electrical", "Lighting", "Lamps",
        "Fluorescent Lamp", contract="lamp",
        must=("fluorescent", "fluor", "flor", "cfl"),
        boost=("t8", "t5", "t12", "tube", "bulb", "lamp"),
        veto=("fixture", "ballast", "troffer"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Chandeliers",
        "Electrical", "Lighting", "Fixtures",
        "Chandelier", contract="fixture",
        must=("chandelier", "chand"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Pendant Lights",
        "Electrical", "Lighting", "Fixtures",
        "Pendant Light", contract="fixture",
        must=("pendant", "pend"),
        boost=("mini", "linear", "island"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Wall Sconces",
        "Electrical", "Lighting", "Fixtures",
        "Wall Light", contract="fixture",
        must=("sconce", "wall light", "wall lt", "wall lantern", "wall mount light"),
        boost=("bath", "vanity", "exterior", "ext"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Bath & Vanity Lights",
        "Electrical", "Lighting", "Fixtures",
        "Bath Light", contract="fixture",
        must=("bath light", "bath lt", "vanity light", "vanity lt", "bath bar"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Flush Mount Ceiling Lights",
        "Electrical", "Lighting", "Fixtures",
        "Ceiling Light", contract="fixture",
        must=("ceiling light", "ceiling lt", "flush mount", "semi-flush",
              "semi flush", "ceiling fixture"),
        veto=("fan",),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Recessed Downlights",
        "Electrical", "Lighting", "Fixtures",
        "Downlight", contract="fixture",
        must=("downlight", "down light", "recessed", "recsd", "retrofit trim",
              "canless", "wafer"),
        boost=("trim", "housing", "baffle", "gimbal"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Exterior Wall Lights",
        "Electrical", "Lighting", "Fixtures",
        "Exterior Wall Light", contract="fixture",
        must=("ext wall lt", "exterior wall light", "outdoor wall light",
              "outdoor lantern", "porch light"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Ceiling Fans",
        "Electrical", "Lighting", "Fans",
        "Ceiling Fan", contract="fixture",
        must=("ceiling fan", "fan"),
        boost=("blade", "downrod", "remote", "reversible"),
        veto=("bath fan", "exhaust fan", "inline fan", "fan blade only"),
    ),
    # ---------------- abrasives & power-tool accessories ----------------
    Category(
        f"{TOOLS}>Power Tool Accessories>Cut-Off Wheels",
        "Tools", "Abrasives", "Cutting Wheels",
        "Cut Off Wheel", contract="abrasive_wheel",
        must=("cut off", "cutoff", "cut-off", "cut and grind", "cut n grind",
              "cut & grind"),
        boost=("disc", "wheel", "masonry", "metal", "type 1", "type 27"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Grinding Wheels",
        "Tools", "Abrasives", "Grinding Wheels",
        "Grinding Wheel", contract="abrasive_wheel",
        must=("grinding wheel", "grinding disc", "grind wheel"),
        boost=("metal", "masonry", "type 27", "depressed center"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Sanding Belts",
        "Tools", "Abrasives", "Sanding Belts",
        "Sanding Belt", contract="abrasive_sheet",
        must=("sanding belt", "sand belt"),
        boost=("grit", "aluminum oxide", "zirconia", "ceramic"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Sanding Discs",
        "Tools", "Abrasives", "Sanding Discs",
        "Sanding Disc", contract="abrasive_sheet",
        must=("sanding disc", "sand disc", "stikit", "hook and loop disc",
              "abrasive disc", "film disc"),
        boost=("grit", "p80", "p100", "p120", "p150", "p180", "p220", "p320",
               "cubitron", "net disc"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Sandpaper & Sheets",
        "Tools", "Abrasives", "Coated Abrasives",
        "Sandpaper Sheet", contract="abrasive_sheet",
        must=("sandpaper", "sanding sheet", "abrasive sheet", "sanding roll",
              "sanding sponge", "abrasive sponge", "sanding pad"),
        boost=("grit",),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Saw Blades",
        "Tools", "Power Tool Accessories", "Saw Blades",
        "Saw Blade", contract="saw_blade",
        must=("saw blade", "circular saw blade", "blade"),
        boost=("tooth", "teeth", "tpi", "carbide", "framing", "finish",
               "ripping", "combination", "diamond blade"),
        veto=("blade guard", "fan blade", "wiper blade", "blade only kit"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Reciprocating Saw Blades",
        "Tools", "Power Tool Accessories", "Saw Blades",
        "Reciprocating Saw Blade", contract="saw_blade",
        must=("recip blade", "reciprocating blade", "reciprocating saw blade",
              "sawzall", "jigsaw blade", "jig saw blade"),
        boost=("tpi", "bi-metal", "carbide"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Hole Saws",
        "Tools", "Power Tool Accessories", "Hole Saws",
        "Hole Saw", contract="generic",
        must=("hole saw", "holesaw", "hole cutter"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Drill Bits",
        "Tools", "Power Tool Accessories", "Drill Bits",
        "Drill Bit", contract="bit",
        must=("drill bit", "twist bit", "auger bit", "spade bit", "masonry bit",
              "step bit", "brad point"),
        boost=("cobalt", "titanium", "carbide", "sds"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Driver Bits",
        "Tools", "Power Tool Accessories", "Driver Bits",
        "Driver Bit", contract="bit",
        must=("drive bit", "driver bit", "phillips drive", "square drive",
              "torx bit", "hex bit", "nut driver", "insert bit", "power bit"),
        boost=("impact", "#1", "#2", "#3", "t15", "t20", "t25"),
    ),
    Category(
        f"{TOOLS}>Power Tool Accessories>Router Bits",
        "Tools", "Power Tool Accessories", "Router Bits",
        "Router Bit", contract="bit",
        must=("router bit", "roundover bit", "rabbeting bit", "flush trim bit"),
    ),
    # ---------------- power tools ----------------
    Category(
        f"{TOOLS}>Power Tools>Circular Saws",
        "Tools", "Power Tools", "Saws",
        "Circular Saw", contract="power_tool",
        must=("circular saw", "circ saw", "worm drive saw", "track saw"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Band Saws",
        "Tools", "Power Tools", "Saws",
        "Bandsaw", contract="power_tool",
        must=("bandsaw", "band saw"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Table Saws",
        "Tools", "Power Tools", "Saws",
        "Table Saw", contract="power_tool",
        must=("table saw", "cabinet saw", "jobsite saw"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Miter Saws",
        "Tools", "Power Tools", "Saws",
        "Miter Saw", contract="power_tool",
        must=("miter saw", "mitre saw", "chop saw"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Sanders",
        "Tools", "Power Tools", "Sanders",
        "Sander", contract="power_tool",
        must=("sander", "sndr"),
        boost=("orbital", "random orbit", "belt sander", "spindle", "edge",
               "oscillating", "drum", "detail"),
        veto=("sanding belt", "sanding disc", "sandpaper", "sanding sheet"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Drills & Drivers",
        "Tools", "Power Tools", "Drills",
        "Drill", contract="power_tool",
        must=("drill driver", "hammer drill", "impact driver", "impact wrench",
              "right angle drill", "rotary hammer"),
        boost=("brushless", "cordless", "kit", "bare tool"),
        veto=("drill bit", "driver bit"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Grinders",
        "Tools", "Power Tools", "Grinders",
        "Grinder", contract="power_tool",
        must=("angle grinder", "grinder", "grndr"),
        boost=("cordless", "paddle switch"),
        veto=("grinding wheel", "grinding disc", "coffee grinder"),
    ),
    Category(
        f"{TOOLS}>Power Tools>Batteries & Chargers",
        "Tools", "Power Tools", "Batteries & Chargers",
        "Charger", contract="power_tool",
        must=("charger", "chgr", "battery", "batt pack", "starter kit"),
        boost=("fast charger", "dual port", "ah", "amp hour", "12v", "18v", "20v", "60v"),
        veto=("battery light", "flashlight"),
    ),
    Category(
        f"{TOOLS}>Hand Tools>Layout & Measuring",
        "Tools", "Hand Tools", "Measuring",
        "Measuring Tool", contract="generic",
        must=("tape measure", "measuring tape", "square", "level", "chalk line",
              "clamp", "jig"),
        boost=("speed square", "framing square", "torpedo"),
        veto=("square drive", "square edge", "square d", "t-square drive"),
    ),
    # ---------------- decking, railing, siding ----------------
    Category(
        f"{BUILDING}>Decking & Railing>Composite Decking Boards",
        "Building Materials", "Decking", "Composite Decking",
        "Composite Decking Board", contract="decking",
        must=("decking", "deck board"),
        boost=("grooved", "square edge", "sq edge", "grvd", "capped", "composite"),
        veto=("fascia", "railing", "baluster", "post", "stair", "riser",
              "cleaning", "cleaner", "screw", "clip", "fastener", "pvc decking"),
    ),
    Category(
        f"{BUILDING}>Decking & Railing>PVC Decking Boards",
        "Building Materials", "Decking", "PVC Decking",
        "PVC Decking Board", contract="decking",
        must=("pvc decking", "pvc deck board", "cellular pvc decking"),
        veto=("fascia", "railing", "trim"),
    ),
    Category(
        f"{BUILDING}>Decking & Railing>Deck Fascia",
        "Building Materials", "Decking", "Fascia",
        "Deck Fascia Board", contract="decking",
        must=("fascia",),
    ),
    Category(
        f"{BUILDING}>Decking & Railing>Railing Systems",
        "Building Materials", "Decking", "Railing",
        "Railing", contract="decking",
        must=("railing", "rail kit", "hand rail", "handrail", "guard rail",
              "gate kit", "rd gate", "gate rd"),
        boost=("top rail", "bottom rail", "level", "stair"),
    ),
    Category(
        f"{BUILDING}>Decking & Railing>Balusters",
        "Building Materials", "Decking", "Balusters",
        "Baluster", contract="decking",
        must=("baluster", "spindle kit", "picket"),
    ),
    Category(
        f"{BUILDING}>Decking & Railing>Posts & Post Caps",
        "Building Materials", "Decking", "Posts",
        "Post", contract="decking",
        must=("post sleeve", "post cap", "post skirt", "newel", "post kit",
              "post trim", "support post", "blank post", "post base",
              "post mount", "deco post"),
    ),
    Category(
        f"{BUILDING}>Siding & Trim>Fiber Cement Siding",
        "Building Materials", "Siding", "Fiber Cement",
        "Fiber Cement Panel", contract="sheet_good",
        must=("hardiepanel", "hardieplank", "hardietrim", "hardiebacker",
              "fiber cement"),
        boost=("smooth", "cedarmill", "primed", "prmd", "select cedarmill"),
    ),
    Category(
        f"{BUILDING}>Siding & Trim>Engineered Wood Siding",
        "Building Materials", "Siding", "Engineered Wood",
        "Engineered Wood Siding", contract="sheet_good",
        must=("smartside", "smart side", "engineered wood siding", "lap siding"),
    ),
    Category(
        f"{BUILDING}>Panels & Sheet Goods>Sheathing & Panels",
        "Building Materials", "Panels", "Sheathing",
        "Sheathing Panel", contract="sheet_good",
        must=("osb", "plywood", "sheathing", "zip system", "subfloor panel",
              "sub floor", "plusosb", "advantech", "t&g"),
        boost=("tongue and groove", "radiant barrier", "huber"),
    ),
    Category(
        f"{BUILDING}>Lumber>Dimensional Lumber",
        "Building Materials", "Lumber", "Dimensional",
        "Lumber", contract="lumber",
        must=("lumber", "stud", "pressure treated", "kiln dried", "s4s", "timber"),
        veto=("decking", "siding", "panel"),
    ),
    Category(
        f"{BUILDING}>Concrete & Masonry>Mortar & Grout",
        "Building Materials", "Masonry", "Mortar",
        "Mortar", contract="generic",
        must=("mortar", "grout", "thinset", "type n", "type s"),
    ),
    Category(
        f"{BUILDING}>Sealants & Adhesives>Joint Tape & Sealant",
        "Building Materials", "Sealants", "Joint Sealant",
        "Joint Sealant", contract="generic",
        must=("emseal", "joint tape", "expansion joint", "sealant tape",
              "backer rod", "flashing tape"),
    ),
    Category(
        f"{BUILDING}>Windows & Doors>Windows",
        "Building Materials", "Windows & Doors", "Windows",
        "Window", contract="generic",
        must=("window", "low-e", "lowe-2", "lowe2", "low e"),
        boost=("double hung", "single hung", "casement", "awning", "slider",
               "picture", "vinyl window", "half scrn", "argon"),
        veto=("window film", "window cleaner", "windows 10", "windows 11"),
    ),
    Category(
        f"{BUILDING}>Windows & Doors>Skylights",
        "Building Materials", "Windows & Doors", "Skylights",
        "Skylight", contract="generic",
        must=("skylight", "skylt", "sky light", "sun tunnel", "roof window"),
    ),
    Category(
        f"{BUILDING}>Windows & Doors>Exterior Doors",
        "Building Materials", "Windows & Doors", "Doors",
        "Door", contract="generic",
        must=("entry door", "patio door", "patio dr", "storm door",
              "exterior door", "fiberglass door", "steel door",
              "gliding door", "gliding patio", "access door", "attic access"),
        veto=("door hardware", "door knob", "door stop", "door sweep"),
    ),
    # ---------------- electrical ----------------
    Category(
        f"{ELECTRICAL}>Wire & Cable>Building Wire",
        "Electrical", "Wire & Cable", "Building Wire",
        "Building Wire", contract="wire",
        must=("thhn", "thwn", "romex", "nm-b", "uf-b", "mc cable", "building wire"),
        boost=("awg", "copper", "aluminum", "stranded", "solid"),
    ),
    Category(
        f"{ELECTRICAL}>Wire & Cable>Portable Cord",
        "Electrical", "Wire & Cable", "Cord",
        "Portable Cord", contract="wire",
        must=("so cord", "sjoow", "sjoo", "soow", "sjtw", "portable cord",
              "extension cord", "cord reel"),
        boost=("linear foot", "per foot"),
    ),
    Category(
        f"{ELECTRICAL}>Wire & Cable>Service Entrance Cable",
        "Electrical", "Wire & Cable", "Service Cable",
        "Service Entrance Cable", contract="wire",
        must=("triplex", "quadruplex", "service entrance", "se cable",
              "ud cable", "urd"),
        boost=("aluminum", "overhead", "direct burial"),
    ),
    Category(
        f"{ELECTRICAL}>Power Distribution>Load Centers",
        "Electrical", "Power Distribution", "Load Centers",
        "Load Center", contract="generic",
        must=("load center", "loadcenter", "panelboard", "main breaker panel"),
        boost=("amp", "space", "circuit"),
    ),
    Category(
        f"{ELECTRICAL}>Power Distribution>Circuit Breakers",
        "Electrical", "Power Distribution", "Breakers",
        "Circuit Breaker", contract="generic",
        must=("circuit breaker", "breaker", "brkr", "gfci breaker", "afci breaker"),
        veto=("load center", "panel"),
    ),
    Category(
        f"{ELECTRICAL}>Wiring Devices>Receptacles",
        "Electrical", "Wiring Devices", "Receptacles",
        "Receptacle", contract="wiring_device",
        must=("receptacle", "recep", "outlet", "gfci"),
        boost=("duplex", "tamper resistant", "usb", "15a", "20a", "decora"),
        veto=("outlet box", "cover plate only"),
    ),
    Category(
        f"{ELECTRICAL}>Wiring Devices>Switches & Dimmers",
        "Electrical", "Wiring Devices", "Switches",
        "Switch", contract="wiring_device",
        must=("wall switch", "toggle switch", "dimmer", "rocker switch",
              "three way switch", "3-way switch", "occupancy sensor"),
        boost=("decora", "single pole", "smart switch"),
        veto=("switch plate only", "disconnect switch", "paddle switch"),
    ),
    Category(
        f"{ELECTRICAL}>Wire & Cable>Electrical Tape",
        "Electrical", "Wire & Cable", "Tape",
        "Electrical Tape", contract="generic",
        must=("electrical tape", "elect tape", "vinyl tape", "friction tape",
              "rubber splicing tape"),
    ),
    Category(
        f"{APPLIANCES}>Laundry Appliances>Laundry Centers",
        "Appliances", "Large Appliances", "Laundry Centers",
        "Laundry Center", contract="laundry",
        must=("laundry center", "laundry centre", "stacked laundry", "washer dryer combo"),
    ),
    Category(
        f"{BUILDING}>Sealants & Adhesives>Deck Joist Tape",
        "Building Materials", "Sealants", "Joist Tape",
        "Deck Joist Tape", contract="generic",
        must=("joist tape", "deck tape", "protecto wrap", "butyl tape"),
    ),
    Category(
        f"{BUILDING}>Panels & Sheet Goods>Drywall & Gypsum Board",
        "Building Materials", "Panels", "Drywall",
        "Drywall Panel", contract="sheet_good",
        must=("drywall", "gypsum", "sheetrock", "easi-lite", "firelite",
              "wallboard", "cement board", "backer board"),
        boost=("type x", "mold resistant", "moisture resistant", "lightweight"),
    ),
    Category(
        f"{BUILDING}>Roofing>Asphalt Shingles",
        "Building Materials", "Roofing", "Shingles",
        "Asphalt Shingle", contract="sheet_good",
        must=("shingle", "duration", "trudef", "landmark", "timberline",
              "architectural shingle"),
        boost=("bdl", "bundle", "sq", "laminated", "hip and ridge", "starter"),
    ),
    Category(
        f"{BUILDING}>Roofing>Metal Roofing Panels",
        "Building Materials", "Roofing", "Metal Panels",
        "Metal Roofing Panel", contract="sheet_good",
        must=("rib xl", "premier rib", "metal roof", "ribbed panel",
              "standing seam", "corrugated panel", "r panel"),
    ),
    Category(
        f"{BUILDING}>Roofing>Underlayment & Ice Barrier",
        "Building Materials", "Roofing", "Underlayment",
        "Roofing Underlayment", contract="sheet_good",
        must=("ice guard", "eaveguard", "ice and water", "ice & water",
              "underlayment", "roof felt", "synthetic felt", "drip edge"),
    ),
    Category(
        f"{BUILDING}>Weather Barriers>Housewrap & Rainscreen",
        "Building Materials", "Weather Barriers", "Housewrap",
        "Weather Barrier", contract="sheet_good",
        must=("rainscreen", "rain screen", "housewrap", "house wrap",
              "weather barrier", "weather resistive", "tyvek"),
    ),
    Category(
        f"{BUILDING}>Lumber>Softwood Boards",
        "Building Materials", "Lumber", "Boards",
        "Softwood Board", contract="lumber",
        must=("doug fir", "douglas fir", "cedar board", "white pine",
              "yellow pine", "spruce", "hemlock", "redwood", "poplar board"),
        boost=("stk", "s4s", "1s2e", "smooth", "rough sawn", "clear"),
        veto=("decking", "siding", "plywood", "shingle"),
    ),
    Category(
        f"{BUILDING}>Windows & Doors>Thresholds & Sills",
        "Building Materials", "Windows & Doors", "Thresholds",
        "Threshold", contract="generic",
        must=("threshold", "door sill", "saddle threshold"),
    ),
    Category(
        f"{BUILDING}>Windows & Doors>Basement & Hopper Windows",
        "Building Materials", "Windows & Doors", "Windows",
        "Basement Window", contract="generic",
        must=("hopper", "bsmt window", "basement window", "ecolite"),
        boost=("dla", "screen", "wh"),
    ),
    Category(
        f"{ELECTRICAL}>Boxes & Enclosures>Electrical Boxes",
        "Electrical", "Boxes & Enclosures", "Boxes",
        "Electrical Box", contract="generic",
        must=("oct box", "octagon box", "junction box", "device box",
              "outlet box", "gang box", "square box", "handy box", "1g box",
              "2g box", "3g box", "4g box", "masonry box", "old work box",
              "new work box"),
        boost=("bracket", "hanger", "nail on", "pvc", "metallic"),
    ),
    Category(
        f"{ELECTRICAL}>Wiring Devices>Wall Plates & Box Covers",
        "Electrical", "Wiring Devices", "Wall Plates",
        "Wall Plate", contract="wiring_device",
        must=("wall plate", "box cover", "cover plate", "decor plate",
              "faceplate", "face plate", "blank plate", "weatherproof cover"),
        boost=("1g", "2g", "3g", "gfi", "decora", "screwless"),
    ),
    Category(
        f"{ELECTRICAL}>Controls>Timers & Photocontrols",
        "Electrical", "Controls", "Timers",
        "Timer", contract="wiring_device",
        must=("timer", "time switch", "photocell", "photocontrol",
              "photo control", "astronomic"),
        boost=("outdoor", "indoor", "digital", "mechanical", "7 day", "24 hour"),
    ),
    Category(
        f"{ELECTRICAL}>Power Supplies>LED Drivers & Power Supplies",
        "Electrical", "Power Supplies", "Drivers",
        "LED Driver", contract="generic",
        must=("led driver", "power supply", "jumpstart", "transformer",
              "electronic driver"),
        boost=("constant current", "constant voltage", "dimmable", "class 2"),
    ),
    Category(
        f"{ELECTRICAL}>Boxes & Enclosures>Hangers & Brackets",
        "Electrical", "Boxes & Enclosures", "Hangers",
        "Box Hanger", contract="generic",
        must=("adjust hanger", "bar hanger", "box hanger", "fixture hanger",
              "adjustable hanger"),
    ),
    Category(
        f"{BUILDING}>Ceilings>Acoustical Ceiling Tile",
        "Building Materials", "Ceilings", "Ceiling Tile",
        "Ceiling Tile", contract="sheet_good",
        must=("ceiling tile", "ceiling panel", "fissured", "acoustical tile",
              "lay-in panel", "drop ceiling"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Strip & Tape Lighting",
        "Electrical", "Lighting", "Fixtures",
        "Strip Light", contract="fixture",
        must=("strip light", "tape light", "led strip", "under cabinet light",
              "linear light", "cove light", "shop light"),
        boost=("cct", "selectable", "plug in", "linkable"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>High Bay & Area Lights",
        "Electrical", "Lighting", "Fixtures",
        "High Bay Light", contract="fixture",
        must=("highbay", "high bay", "low bay", "area light", "wall pack",
              "canopy light", "parking lot light"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>Security & Flood Lights",
        "Electrical", "Lighting", "Fixtures",
        "Security Light", contract="fixture",
        must=("motion lt", "motion light", "security light", "flood light",
              "floodlight", "motion sensor light", "dusk to dawn"),
        boost=("pir", "adjustable head", "photocell"),
    ),
    Category(
        f"{LIGHTING}>Light Fixtures>LED Fixtures",
        "Electrical", "Lighting", "Fixtures",
        "LED Light Fixture", contract="fixture",
        must=("led lt", "led light", "led fixture", "led luminaire"),
        boost=("multi cct", "cct", "selectable", "integrated"),
        veto=("bulb", "lamp", "a19", "br30", "par38", "candelabra"),
    ),
    # ---------------- safety ----------------
    Category(
        f"{SAFETY}>Eye Protection>Safety Glasses",
        "Safety", "PPE", "Eye Protection",
        "Safety Glasses", contract="generic",
        must=("safety glass", "safety glasses", "safety eyewear", "goggle",
              "protective eyewear"),
        boost=("anti-fog", "polarized", "z87"),
    ),
    Category(
        f"{SAFETY}>Hand Protection>Work Gloves",
        "Safety", "PPE", "Hand Protection",
        "Work Gloves", contract="generic",
        must=("glove", "gloves"),
        boost=("cut resistant", "nitrile", "leather", "heated", "impact"),
    ),
)


# --- lexical matching -------------------------------------------------------


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary matcher tolerant of punctuation, spacing and plurals.

    The plural tolerance is not cosmetic. Without it a strict trailing guard
    rejects ``Balusters`` for the phrase ``baluster``, ``Post Caps`` for
    ``post cap`` and ``Discs`` for ``disc`` -- which silently cost real
    coverage on the working dataset before it was fixed.
    """
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", phrase.strip()) if p]
    body = r"[\s\-]*".join(parts)
    lead = r"(?<![A-Za-z0-9])"
    trail = r"(?:e?s)?(?![A-Za-z0-9])"
    return re.compile(lead + body + trail, re.IGNORECASE)


@lru_cache(maxsize=4096)
def _compiled(phrase: str) -> re.Pattern[str]:
    return _phrase_pattern(phrase)


@dataclass
class Match:
    """A scored category hypothesis with the evidence behind it."""

    category: Category
    score: float
    hits: list[tuple[str, int, int]] = field(default_factory=list)  # phrase, start, end
    prior: float = 0.0
    vetoed_by: str = ""

    @property
    def explanation(self) -> str:
        phrases = ", ".join(sorted({h[0] for h in self.hits}))
        parts = [f"matched [{phrases}]"] if phrases else []
        if self.prior:
            parts.append(f"group prior +{self.prior:.2f}")
        return "; ".join(parts)


class GroupPrior:
    """P(department | Part_Manuf), learned from the corpus in one pass.

    `Kichler Lighting (KICLI)` sells light fixtures and nothing else, so once a
    handful of its rows classify confidently by keyword, the rest of the group
    inherits a prior that resolves the ambiguous ones. Nothing here is typed in
    by hand -- it is fitted from whatever catalogue is loaded.
    """

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()

    def observe(self, group: str, classpath: str, weight: float = 1.0) -> None:
        if not group or not classpath:
            return
        self.counts[group][classpath] += weight
        self.totals[group] += weight

    def score(self, group: str, classpath: str) -> float:
        total = self.totals.get(group, 0)
        if not total:
            return 0.0
        return self.counts[group][classpath] / total

    def top(self, group: str) -> tuple[str, float] | None:
        if not self.counts.get(group):
            return None
        classpath, hits = self.counts[group].most_common(1)[0]
        return classpath, hits / max(self.totals[group], 1)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            group: {cp: round(n / max(self.totals[group], 1), 4) for cp, n in c.most_common(6)}
            for group, c in self.counts.items()
        }


#: Weight of one supporting-keyword hit relative to a required-keyword hit.
BOOST_WEIGHT = 0.18
#: Weight applied to the group prior. Deliberately small: the prior breaks ties,
#: it does not override lexical evidence.
PRIOR_WEIGHT = 0.35
#: Below this score we decline to classify rather than guess.
MIN_ACCEPT = 0.55


def score_categories(
    text: str,
    *,
    group: str = "",
    prior: GroupPrior | None = None,
    categories: Sequence[Category] = CATEGORIES,
) -> list[Match]:
    """Score every category against one description. Highest first."""
    matches: list[Match] = []
    for category in categories:
        veto_hit = ""
        for phrase in category.veto:
            if _compiled(phrase).search(text):
                veto_hit = phrase
                break
        if veto_hit:
            matches.append(Match(category, 0.0, vetoed_by=veto_hit))
            continue

        hits: list[tuple[str, int, int]] = []
        must_hit = False
        best_must_len = 0
        for phrase in category.must:
            m = _compiled(phrase).search(text)
            if m:
                must_hit = True
                hits.append((phrase, m.start(), m.end()))
                best_must_len = max(best_must_len, len(phrase))
        if not must_hit:
            continue

        for phrase in category.boost:
            m = _compiled(phrase).search(text)
            if m:
                hits.append((phrase, m.start(), m.end()))

        n_boost = len(hits) - sum(1 for h in hits if h[0] in category.must)
        # Specificity: a longer required phrase is stronger evidence, so
        # "sanding disc" outranks a bare "disc" and "range hood" beats "range".
        specificity = min(best_must_len / 12.0, 1.0)
        score = 0.6 + 0.25 * specificity + BOOST_WEIGHT * min(n_boost, 4)

        prior_score = 0.0
        if prior is not None and group:
            prior_score = prior.score(group, category.classpath)
            score += PRIOR_WEIGHT * prior_score

        matches.append(Match(category, score, hits=hits, prior=prior_score))

    matches.sort(key=lambda m: (-m.score, m.category.classpath))
    return matches


@dataclass
class Classification:
    category: Category | None
    confidence: float
    provenance: Provenance
    runner_up: Category | None = None
    margin: float = 0.0

    @property
    def ok(self) -> bool:
        return self.category is not None


def classify(
    text: str,
    *,
    group: str = "",
    prior: GroupPrior | None = None,
    evidence_text: str | None = None,
    categories: Sequence[Category] = CATEGORIES,
) -> Classification:
    """Classify one description into the taxonomy, with full provenance."""
    source_text = evidence_text if evidence_text is not None else text
    ranked = [m for m in score_categories(text, group=group, prior=prior, categories=categories) if m.score > 0]
    if not ranked:
        return _prior_only(group, prior) or Classification(
            None,
            0.0,
            Provenance(
                Source.UNRESOLVED,
                rule="taxonomy:no-lexical-evidence",
                confidence=0.0,
                detail="no category keyword matched; row routed to review",
            ),
        )

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = best.score - (runner_up.score if runner_up else 0.0)

    if best.score < MIN_ACCEPT:
        return Classification(
            None,
            best.score,
            Provenance(
                Source.UNRESOLVED,
                rule="taxonomy:below-threshold",
                confidence=best.score,
                detail=(
                    f"best candidate {best.category.leaf} scored {best.score:.2f} "
                    f"< {MIN_ACCEPT}; declining to guess"
                ),
            ),
            runner_up=best.category,
            margin=margin,
        )

    # Confidence blends absolute score with how clearly it beat the runner-up.
    confidence = min(0.99, 0.55 * min(best.score, 1.0) + 0.45 * min(margin / 0.4, 1.0))
    evidence = tuple(
        Evidence(source_text, start, end)
        for _phrase, start, end in best.hits
        if 0 <= start <= end <= len(source_text)
    )
    return Classification(
        best.category,
        confidence,
        Provenance(
            Source.LEXICON_EXACT,
            rule=f"taxonomy:lexical:{best.category.leaf}",
            confidence=confidence,
            evidence=evidence,
            detail=best.explanation
            + (f"; runner-up {runner_up.category.leaf} at {runner_up.score:.2f}" if runner_up else ""),
        ),
        runner_up=runner_up.category if runner_up else None,
        margin=margin,
    )


#: A distributor group must be this pure, and this well-evidenced, before its
#: prior is allowed to classify a row on its own.
PRIOR_ONLY_PURITY = 0.60
PRIOR_ONLY_MIN_ROWS = 5.0


def _prior_only(group: str, prior: GroupPrior | None) -> Classification | None:
    """Classify from the distributor prior when no keyword matched at all.

    Justified by how these catalogues are actually built: a `Part_Manuf` account
    is usually a single product line. Mirka Abrasives sells abrasives, and the
    rows that read only ``9A-570-240 Abranet 2.75x30`` are abrasive discs even
    though the description never says so -- the product line name carries the
    category and the account confirms it.

    Deliberately conservative. It fires only for a group that is at least 60%
    one category across 5+ confidently-classified rows, it caps confidence
    below the review threshold, and it says plainly in the provenance that
    there was no lexical evidence. It raises coverage without ever claiming
    more certainty than it has.
    """
    if prior is None or not group:
        return None
    top = prior.top(group)
    if top is None:
        return None
    classpath, purity = top
    if purity < PRIOR_ONLY_PURITY or prior.totals.get(group, 0) < PRIOR_ONLY_MIN_ROWS:
        return None
    category = by_classpath().get(classpath)
    if category is None:
        return None
    confidence = round(0.45 * purity, 4)
    return Classification(
        category,
        confidence,
        Provenance(
            Source.DERIVED,
            rule=f"taxonomy:group-prior:{category.leaf}",
            confidence=confidence,
            detail=(
                f"no lexical evidence; assigned from distributor prior "
                f"'{group}' -> {category.leaf} at {purity:.0%} purity over "
                f"{prior.totals[group]:.0f} classified rows"
            ),
        ),
        margin=0.0,
    )


def fit_group_prior(
    observations: Iterable[tuple[str, str, float]],
) -> GroupPrior:
    """Build a prior from (group, classpath, confidence) triples."""
    prior = GroupPrior()
    for group, classpath, weight in observations:
        prior.observe(group, classpath, weight)
    return prior


@lru_cache(maxsize=1)
def by_classpath() -> dict[str, Category]:
    return {c.classpath: c for c in CATEGORIES}


@lru_cache(maxsize=1)
def all_classpaths() -> tuple[str, ...]:
    """The controlled vocabulary for `Classpath`. Nothing else may be emitted."""
    return tuple(c.classpath for c in CATEGORIES)


def prepare_text(desc: str, mpn: str) -> str:
    """Normalise a raw description into the form the matchers expect."""
    body, _ = T.strip_leading_mpn(T.clean(desc), mpn)
    body, _ = T.strip_noise(body)
    expanded = T.expand(body)
    # keep both forms visible to the matcher: "Ceiling Lt" and "Ceiling Light"
    return f"{body} | {expanded}" if expanded.lower() != body.lower() else body
