"""
Step 1: Inventory DHS zip files and identify Individual Recode (IR) datasets.

DHS file naming convention (XX = country code, YY = recode type, nn = phase):
    XXYYnn[Z]<FMT>.zip
    IR  = Individual Recode (women 15-49, one row per interviewed woman)
    BR  = Birth Recode        (one row per child ever born)
    KR  = Children's Recode   (one row per child under 5 in household)
    MR  = Men's Recode
    PR  = Household Member Recode
    FMT = DT (Stata .dta) | FL (flat ASCII) | SD (SAS) | SV (SPSS .sav)
"""
import os
import re
import glob
from collections import defaultdict

ROOTS = [".", "Child-Mortality-Analysis-Data-main 2"]
FMT_SUFFIX = {"DT": "Stata", "FL": "Flat ASCII", "SD": "SAS", "SV": "SPSS"}

# DHS recode types: 2-letter code appearing at positions 2:4 of the filename
RECODE_TYPES = ["IR", "BR", "KR", "MR", "PR", "HR", "HW", "NR", "GR", "GC",
                "GE", "IQ", "WI", "SQ", "AI", "BQ", "CR", "AR", "HR", "FR"]

def classify(fname):
    """Return (country, recode_type, phase, fmt) parsed from a DHS filename."""
    base = os.path.basename(fname)
    stem = base[:-4]  # strip .zip
    m = re.match(r"^([A-Z]{2})([A-Z]{2})([0-9A-Z]{1,3})(DT|FL|SD|SV)$", stem)
    if not m:
        return None
    country, rtype, phase, fmt = m.group(1), m.group(2), m.group(3), m.group(4)
    return country, rtype, phase, fmt

zips = []
for root in ROOTS:
    for p in glob.glob(os.path.join(root, "*.zip")):
        c = classify(p)
        if c:
            zips.append((p, *c))

print(f"Total classified DHS zips: {len(zips)}\n")

# Group by recode type
by_type = defaultdict(list)
for p, country, rtype, phase, fmt in zips:
    by_type[rtype].append((country, phase, fmt, p))

print("=== Distribution by recode type (country x phase x format counts) ===")
for rtype in sorted(by_type):
    entries = by_type[rtype]
    countries = sorted({c for c, _, _, _ in entries})
    formats = defaultdict(int)
    for _, _, f, _ in entries:
        formats[f] += 1
    print(f"{rtype:3s}: {len(entries):4d} files | {len(countries):3d} countries | "
          f"fmt {dict(formats)} | countries: {''.join(countries)}")

# Focus on Individual Recode (IR)
ir = by_type.get("IR", [])
print("\n=== Individual Recode (IR) datasets ===")
print(f"Number of IR zip files: {len(ir)}")
ir_countries = sorted({c for c, _, _, _ in ir})
print(f"Number of distinct countries with IR: {len(ir_countries)}")
print(f"Countries: {', '.join(ir_countries)}")

# Show format availability per country (which formats exist)
print("\n--- IR format availability per country ---")
country_fmts = defaultdict(set)
country_phase = {}
for c, phase, fmt, _ in ir:
    country_fmts[c].add(fmt)
    country_phase.setdefault(c, set()).add(phase)
for c in ir_countries:
    fmts = "".join(sorted(country_fmts[c]))
    phases = ",".join(sorted(country_phase[c]))
    has_dt = "DT" in country_fmts[c]
    print(f"  {c}: phases[{phases}] formats[{fmts}] {'* DT available' if has_dt else '  (no DT!)'}")

# How many have Stata (DT) format available (easiest to load with pandas)?
ir_dt = [x for x in ir if x[2] == "DT"]
print(f"\nIR files in Stata (DT) format: {len(ir_dt)}")
