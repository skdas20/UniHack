# Content contracts reverse-engineered from the gold rows

The Expected Output sheet ships **252 headers and 2 fully-worked rows**. Both
gold rows are `Built-In Dishwashers`. Every rule below was derived by *diffing
the two rows against each other and against their raw input*, not assumed.

Raw input for both is a single abbreviated string plus 5 mostly-placeholder
columns:

```
Mfg_Part_Num = PDSH4816AF
Part_Desc    = "PDSH4816AF Dishwasher SS - Display Only"
E1_Brand     = "-- Unbranded --"        <- placeholder, means empty
Unilog_Brand = "-- No Unilog Brand --"  <- placeholder, means empty
DIB_Brand    = "-- No DIB Brand --"     <- placeholder, means empty
Part_Manuf   = "Appliance Dealers Cooperative (APPDE)"   <- a co-op, NOT the mfr
```

Note `Part_Manuf` is a *distributor cooperative*, and the correct
`MANUFACTURER_NAME` is `Rheem Manufacturing` with `BRAND_NAME = FRIGIDAIRE®`.
The manufacturer is **not** recoverable from `Part_Manuf` — it has to come from
resolving the MPN/brand. This is the single hardest entity-resolution case in
the pack and it is row 1.

---

## 1. The attribute block is a per-category ORDERED TEMPLATE

`Built-In Dishwashers` emits exactly 15 slots, in this fixed order, **and emits
the LABEL even when the VALUE is blank**:

```
 1 Series               6 Mounting Type        11 Maximum Height
 2 Model                7 Plug Type            12 Sound Level
 3 No. of Wash Cycles   8 Size                 13 Material
 4 Voltage Rating       9 Depth With Door Open 14 Color
 5 Amperage Rating     10 Minimum Height       15 Additional Information
```

Row 1 leaves `Model`, `Plug Type`, `Color` blank; row 2 leaves `Model`,
`No. of Wash Cycles`, `Plug Type`, `Maximum Height` blank. **Both still emit
all 15 labels.**

=> Attribute output is *schema-driven*, not free extraction. A pipeline that
   emits "whatever it found" produces a structurally wrong row even when every
   value it found is correct. We therefore carry an explicit
   `AttributeContract` per classpath and always render the full slot list.

## 2. Five descriptions, five different contracts

| Field | Contract | Row 1 | Row 2 |
|---|---|---|---|
| `INVOICE_DESC` | `PRODUCT_NAME + abbreviated values in slot order`, ALL CAPS, **unit glued to number**, greedy-pack to ≤40 | `DISHWASHER LEG 5 SST 120V 15A 50-1/4IN` (38) | `DISHWASHER BLTLN SST SST 120V 10A 41DBA` (39) |
| `MOBILE_DESC` | comma-joined `[Manufacturer, Brand(no symbol), ProductName, Series, MPN, +filler attrs]`, packed into **60–80** | 75 ch, kept manufacturer | 63 ch, **dropped** manufacturer, **added** Mounting to clear 60 |
| `SHORT_DESC` | `BRAND® + Series + MPN + ProductName + [With X] + , attr phrases` | `FRIGIDAIRE® Professional Series PDSH4816AF Dishwasher With CleanBoost™, Leg Mounting, 5-Wash Cycle, Stainless Steel` | `Whirlpool® Eco Series WDTS7024RZ Dishwasher, Built-in Mounting, Stainless Steel, Stainless Steel` |
| `LONG_DESC1` | `BRAND® + ProductName + [With X] + , Series, <all attr phrases, UOM SPACED>, Additional Information: …` | `120 V`, `15 A`, `47 dBA` — spaced | same |
| `RETAIL_DESC` | `Series + ProductName + , <key attrs>` — **no brand at all** | `Professional Series Dishwasher, Leg Mounting, 5-Wash Cycle, Stainless Steel` | `Eco Series Dishwasher, Built-in Mounting, Stainless Steel, Stainless Steel` |

Observe the deliberate inconsistency between channels: **`120V` in
`INVOICE_DESC` but `120 V` in `LONG_DESC1`.** Two different UOM rendering modes
for the same underlying value. One LLM prompt asked to "follow the house style"
will not reliably produce both; a contract-driven renderer with two rendering
modes does it by construction.

Also note `Stainless Steel, Stainless Steel` in row 2 — Material *and* Color
both resolve to Stainless Steel and the duplication is **preserved, not
de-duplicated**. Deduping here would be a plausible-looking bug.

## 3. `MOBILE_DESC` proves a constraint solver is required

Row 1 keeps `Rheem Manufacturing` and lands at 75 chars.
Row 2 **drops** `Whirlpool Corporation` (which would have overflowed 80) and
**appends** `Built-in Mounting` to climb back over the 60-char floor.

That is not prompt behaviour. It is a search over an ordered candidate list with
*drop* and *append* moves, subject to a two-sided length constraint. We
implement `pack_to_window()` as exactly that, and it is deterministic and
explainable — we can show the judges which candidates were dropped and why.

## 4. Attribute phrase rendering (structured value -> prose)

| Slot | Value | Rendered |
|---|---|---|
| `Mounting Type` | `Leg` | `Leg Mounting`  (value + head-noun of label) |
| `No. of Wash Cycles` | `5` | `5-Wash Cycle` (SHORT) / `5 Wash Cycles` (LONG) |
| `Sound Level` | `47` + `dBA` | `47 dBA Sound Level` |
| `Depth With Door Open` | `50-1/4` + `in` | `50-1/4 in Depth With Door Open` |
| `Material` | `Stainless Steel` | `Stainless Steel` (bare value, no label) |
| `Additional Information` | list | `Additional Information: a, b, c` (suffixed, last) |

So each slot carries a *render style*, not just a value. Encoded as
`PhraseStyle` in `attributes.py`.

## 5. Digital assets are fully deterministic

```
Product Image         = {BRAND_no_symbol}_{MPN}.jpg
Alternate Image N     = {BRAND_no_symbol}_{MPN}_{N}.jpg
Specification Sheet   = {BRAND_no_symbol}_{MPN}_Specification_Sheet.pdf
Actual Image (Yes/No) = Yes
```
`FRIGIDAIRE_PDSH4816AF.jpg`, `Whirlpool_WDTS7024RZ_Specification_Sheet.pdf`.
Brand casing is preserved from `BRAND_NAME`, only `®`/`™` are stripped. Free
accuracy on ~25 columns.

## 6. `Standard/Approvals` is pipe-delimited, ASCII-alphabetical

`ASSE 1006|CEE Tier 2 Qualified|cUL Listed|ENERGY STAR Certified|NSF Certified|UL Listed`

Sorted by ASCII, which is why lowercase-initial `cUL Listed` sorts *before*
`ENERGY STAR` — a plain `sorted()` reproduces it, a "smart" case-insensitive
sort does not.

## 7. Honest non-derivables

`PART_NUMBER` (`20887830`) and `SKU - MY_PART_NUMBER` (`1515863`) are the
distributor's internal ERP identifiers. They are **not** a function of any
input column — no amount of AI recovers them from a 6-column row. Same for
`List Price`, `UPC`/`EAN`/`GTIN`, and `Country Of Origin`.

We emit these blank and attach an explicit `NOT_DERIVABLE` provenance reason
naming the system of record that would supply them. The Solution Guide
explicitly rewards this:

> *"Real data is imperfect — say so. […] Noticing and reporting such gaps is a
> strength, not a failure."*

Inventing a plausible 8-digit part number would be the single worst thing this
pipeline could do, and it is exactly what an unconstrained generator does.

## 8. `Dept/Class/Fine` is a SECOND taxonomy, distinct from `Classpath`

```
Dept/Class/Fine : Appliances / Large Appliances / Dishwashers
Classpath       : Appliances & Consumer Electronics>Kitchen Appliances>Built-In Dishwashers
```
Both must be produced and they must be *mutually consistent*. We classify once
into `Classpath` and derive `Dept/Class/Fine` through a crosswalk, so the two
can never disagree.

## 9. Placeholder inventory (measured on the 1000-row input)

| Column | Placeholder | Count |
|---|---|---|
| `E1_Brand` | `-- Unbranded --` | 799 / 1000 |
| `E1_Brand` | `COMMODITY - UNBRANDED` | 4 |
| `Unilog_Brand` | `-- No Unilog Brand --` | **1000 / 1000** |
| `DIB_Brand` | `-- No DIB Brand --` | 755 / 1000 |
| `Part_Manuf` | `-` | 41 |

`Unilog_Brand` carries **zero** information across the entire dataset. Any
pipeline that treats it as a feature is fitting noise. `DIB_Brand` is the most
useful brand column when present (245 rows), and it is clean (`Philips`,
`DEWALT`, `Diablo`, `Leviton`, `Satco`, `Southwire`, `Milwaukee`).

## 10. Category reality check

The Solution Guide advises going deep on Faucets or Fittings because those two
reference specs are fully worked. **There is not one faucet or fitting in the
1000-row input.** The actual distribution:

- Lighting — Philips (111), Satco (41), Kichler (56)
- Power-tool accessories / abrasives — Milwaukee (108), Diablo/Freud (46)
- Composite decking — Trex, TimberTech, Azek (via Boise Cascade 85, Parksite 55, US Lumber 43)
- Appliances — via Appliance Dealers Cooperative (84): Frigidaire, Whirlpool, Speed Queen, LG, KitchenAid
- Wire & electrical — Southwire (19), Leviton (17), Square D
- Power tools — DEWALT (55), Makita (23), Festool (16), Jet, Grizzly
- Building materials — James Hardie, LP SmartSide, lumber

Depth investment therefore goes to **Lighting, Abrasives/Cutting, Decking and
Appliances**, which together cover the large majority of rows. Following the
guide's faucet advice literally would produce a demo that cannot process a
single row of the actual dataset.
