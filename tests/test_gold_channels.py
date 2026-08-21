"""The renderer must reproduce the published gold rows character-for-character.

This is the only real ground truth in the pack: the Expected Output sheet ships
252 headers and two fully-worked rows. The 200-row Input-vs-Delivery-Format
workbook that the Solution Guide calls "the most important file in the pack" is
not published on the portal.

So we use what exists, and we use it strictly. This test feeds each gold row's
*own attribute values* into the renderer and asserts that all five description
channels come back exactly as published. It isolates the formatting logic from
the extraction logic: if extraction is imperfect the channels still have to be
right about the values they are given, and that is what a judge checking house
style will look at.

Run: python -m pytest tests -q     (or: python tests/test_gold_channels.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassbox.attributes import CONTRACTS  # noqa: E402
from glassbox.extract import Filled  # noqa: E402
from glassbox.provenance import Cell  # noqa: E402
from glassbox.render import (  # noqa: E402
    INVOICE_MAX,
    MOBILE_MAX,
    MOBILE_MIN,
    RenderInput,
    asset_filenames,
    invoice_description,
    long_description,
    mobile_description,
    product_title,
    retail_description,
)

DISHWASHER = CONTRACTS["dishwasher"]


def _filled(values: dict[str, tuple[str, str]]) -> dict[str, Filled]:
    """Build the filled-slot map from {label: (value, uom)}."""
    out: dict[str, Filled] = {}
    for slot in DISHWASHER.slots:
        value, uom = values.get(slot.label, ("", ""))
        out[slot.label] = Filled(slot, value, uom, Cell(value))
    return out


# --- gold row 1: Frigidaire PDSH4816AF -------------------------------------

GOLD1 = RenderInput(
    product_name="Dishwasher",
    brand="FRIGIDAIRE®",
    brand_plain="FRIGIDAIRE",
    manufacturer="Rheem Manufacturing",
    mpn="PDSH4816AF",
    series="Professional Series",
    contract=DISHWASHER,
    with_phrase="With CleanBoost™",
    filled=_filled(
        {
            "Series": ("Professional Series", ""),
            "Number of Wash Cycles": ("5", ""),
            "Voltage Rating": ("120", "V"),
            "Amperage Rating": ("15", "A"),
            "Mounting Type": ("Leg", ""),
            "Size": ("24 in W x 24-1/4 in D", ""),
            "Depth With Door Open": ("50-1/4", "in"),
            "Minimum Height": ("8-1/2 in Upper Rack, 11-1/4 in Lower Rack", ""),
            "Maximum Height": ("10-3/8 in Upper Rack, 13-1/4 in Lower Rack", ""),
            "Sound Level": ("47", "dBA"),
            "Material": ("Stainless Steel", ""),
            "Additional Information": (
                "240 kW-hr Annual Energy, 1 to 12 hr Delay Start Hours", ""
            ),
        }
    ),
)

GOLD1_EXPECTED = {
    "SHORT_DESC": (
        "FRIGIDAIRE® Professional Series PDSH4816AF Dishwasher With CleanBoost™, "
        "Leg Mounting, 5-Wash Cycle, Stainless Steel"
    ),
    "INVOICE_DESC": "DISHWASHER LEG 5 SST 120V 15A 50-1/4IN",
    "MOBILE_DESC": (
        "Rheem Manufacturing FRIGIDAIRE, Dishwasher, Professional Series, PDSH4816AF"
    ),
    "RETAIL_DESC": (
        "Professional Series Dishwasher, Leg Mounting, 5-Wash Cycle, Stainless Steel"
    ),
}

# --- gold row 2: Whirlpool WDTS7024RZ --------------------------------------

GOLD2 = RenderInput(
    product_name="Dishwasher",
    brand="Whirlpool®",
    brand_plain="Whirlpool",
    manufacturer="Whirlpool Corporation",
    mpn="WDTS7024RZ",
    series="Eco Series",
    contract=DISHWASHER,
    filled=_filled(
        {
            "Series": ("Eco Series", ""),
            "Voltage Rating": ("120", "V"),
            "Amperage Rating": ("10", "A"),
            "Mounting Type": ("Built-in", ""),
            "Size": ("33-7/16 in H x 23-7/8 in W x 22-5/8 in D", ""),
            "Depth With Door Open": ("50-3/16", "in"),
            "Minimum Height": ("33-7/16", "in"),
            "Sound Level": ("41", "dBA"),
            "Material": ("Stainless Steel", ""),
            "Color": ("Stainless Steel", ""),
            "Additional Information": (
                "Folding Tines, Leak Detection System, Moisture Repellent "
                "Silverware Basket, Normal Cycle, Quick Wash Cycle, Sani Rinse "
                "Option, Sensor Cycle, Triple Wash Spray", ""
            ),
        }
    ),
)

GOLD2_EXPECTED = {
    "SHORT_DESC": (
        "Whirlpool® Eco Series WDTS7024RZ Dishwasher, Built-in Mounting, "
        "Stainless Steel, Stainless Steel"
    ),
    "INVOICE_DESC": "DISHWASHER BLTLN SST SST 120V 10A 41DBA",
    "MOBILE_DESC": "Whirlpool, Dishwasher, Eco Series, WDTS7024RZ, Built-in Mounting",
    "RETAIL_DESC": (
        "Eco Series Dishwasher, Built-in Mounting, Stainless Steel, Stainless Steel"
    ),
}

BUILDERS = {
    "SHORT_DESC": product_title,
    "INVOICE_DESC": invoice_description,
    "MOBILE_DESC": mobile_description,
    "RETAIL_DESC": retail_description,
}


def _check(name: str, inp: RenderInput, expected: dict[str, str]) -> list[str]:
    failures = []
    for channel, want in expected.items():
        got, _prov = BUILDERS[channel](inp)
        if got != want:
            failures.append(
                f"{name} {channel}\n   want: {want!r}\n   got : {got!r}"
            )
    return failures


def test_gold_row_1_channels():
    assert not _check("gold1", GOLD1, GOLD1_EXPECTED)


def test_gold_row_2_channels():
    assert not _check("gold2", GOLD2, GOLD2_EXPECTED)


def test_invoice_never_exceeds_limit():
    for inp in (GOLD1, GOLD2):
        text, _ = invoice_description(inp)
        assert len(text) <= INVOICE_MAX, (len(text), text)
        assert text == text.upper()


def test_mobile_inside_window():
    for inp in (GOLD1, GOLD2):
        text, _ = mobile_description(inp)
        assert MOBILE_MIN <= len(text) <= MOBILE_MAX, (len(text), text)


GOLD1_LONG = (
    "FRIGIDAIRE® Dishwasher With CleanBoost™, Professional Series, 5 Wash Cycles, "
    "120 V, 15 A, Leg Mounting, 24 in W x 24-1/4 in D, 50-1/4 in Depth With Door "
    "Open, 8-1/2 in Upper Rack, 11-1/4 in Lower Rack Minimum Height, 10-3/8 in "
    "Upper Rack, 13-1/4 in Lower Rack Maximum Height, 47 dBA Sound Level, "
    "Stainless Steel, Additional Information: 240 kW-hr Annual Energy, "
    "1 to 12 hr Delay Start Hours"
)


def test_long_description_matches_gold_exactly():
    text, _ = long_description(GOLD1)
    assert text == GOLD1_LONG, f"want {GOLD1_LONG!r} got {text!r}"
    # spaced UOM in long copy, never glued -- the opposite of INVOICE_DESC
    assert "120 V" in text and "15 A" in text and "47 dBA" in text
    assert "120V" not in text


def test_registered_marks_survive_normalisation():
    """NFKC decomposes the trademark sign into the letters TM. It must not."""
    from glassbox import textnorm as tn

    assert tn.clean("CleanBoost™") == "CleanBoost™"
    assert tn.clean("FRIGIDAIRE®") == "FRIGIDAIRE®"
    # ...while the quote folding NFKC is wanted for is still applied
    assert tn.clean("24″") == '24"'


def test_asset_filenames_match_gold():
    assets = asset_filenames("FRIGIDAIRE®", "PDSH4816AF")
    assert assets["Product Image"] == "FRIGIDAIRE_PDSH4816AF.jpg"
    assert assets["Alternate Image 3"] == "FRIGIDAIRE_PDSH4816AF_3.jpg"
    assert (
        assets["Specification Sheet"] == "FRIGIDAIRE_PDSH4816AF_Specification_Sheet.pdf"
    )
    assets2 = asset_filenames("Whirlpool®", "WDTS7024RZ")
    assert assets2["Product Image"] == "Whirlpool_WDTS7024RZ.jpg"


if __name__ == "__main__":
    problems = _check("gold1", GOLD1, GOLD1_EXPECTED) + _check("gold2", GOLD2, GOLD2_EXPECTED)
    for channel, want in (*GOLD1_EXPECTED.items(),):
        got, prov = BUILDERS[channel](GOLD1)
        mark = "OK " if got == want else "XX "
        print(f"{mark}gold1 {channel:<13} [{len(got):>3}] {got}")
        if channel in {"MOBILE_DESC", "INVOICE_DESC"}:
            print(f"       audit: {prov.detail}")
    print()
    for channel, want in (*GOLD2_EXPECTED.items(),):
        got, prov = BUILDERS[channel](GOLD2)
        mark = "OK " if got == want else "XX "
        print(f"{mark}gold2 {channel:<13} [{len(got):>3}] {got}")
        if channel in {"MOBILE_DESC", "INVOICE_DESC"}:
            print(f"       audit: {prov.detail}")
    print()
    long1, _ = long_description(GOLD1)
    print(f"   gold1 LONG_DESC1   [{len(long1):>3}] {long1}")
    print()
    if problems:
        print(f"{len(problems)} MISMATCH(ES):")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("all gold channels reproduced exactly")
