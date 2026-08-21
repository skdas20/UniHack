# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from glassbox.units import (to_fraction, to_decimal, find_measurements, RenderMode,
                            find_dimension_group, render_dimension_group, enforce_spacing,
                            canonicalise, DECIMAL_TO_FRACTION)

print("== fractions ==")
for s in ["50.25","0.5","0.125","0.015625","3.7","24.0","0.984375","33.4375","50.1875"]:
    print(f"  {s:>10} -> {to_fraction(s)}")
print("  table size:", len(set(DECIMAL_TO_FRACTION.values())))
print("  50-1/4 ->", to_decimal("50-1/4"), " 1/8 ->", to_decimal("1/8"))

print("\n== canonicalise ==")
for s in ['"','in.','INCHES','amps','V.','Volts','dba','ft','pcs','HP','kwh','#']:
    print(f"  {s:>8} -> {canonicalise(s)}")

print("\n== real descriptions ==")
samples = [
 'DCB518ASTS06G Diablo 1/2"x18" - Sanding Belt 6pc',
 '49-94-1940 Milw 14"x1/8"x1" Masonry Cut Off Disc',
 "4'x10' HardiePanel Smooth - Primed",
 '564922 60W Led BA11 50k 3pk',
 'JT1-549 JWBS18SFX 18" Bandsaw - 1.75HP 1PH 115V',
 "1nx6-20' Pebble Beach Grooved - Trex Enhance Basics Decking",
 '3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box',
 '10-4 SO Cord (Linear Foot)',
 '49-94-0501 Milw 4"x1/4"x5/8" Metal Grinding Wheel',
]
for s in samples:
    ms = find_measurements(s)
    print(f"\n  {s}")
    for m in ms:
        print(f"      [{m.start:>3}:{m.end:<3}] {m.raw!r:<12} -> spaced={m.render()!r:<12} glued={m.render(RenderMode.GLUED)!r:<12} ({m.family})")
    g = find_dimension_group(s)
    if g:
        parts, a, b = g
        print(f"      DIM GROUP [{a}:{b}] -> {render_dimension_group(parts)!r}")

print("\n== enforce_spacing (identifier safety) ==")
for s in ['DISHWASHER 50-1/4IN 120V', 'DCB518ASTS06G Diablo 1/2"x18" Belt',
          'Size 24in W x 24-1/4in D', 'PDSH4816AF Dishwasher 15A']:
    print(f"  {s!r}\n    -> {enforce_spacing(s, protect=('DCB518ASTS06G','PDSH4816AF'))!r}")
