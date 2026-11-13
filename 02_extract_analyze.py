"""
Step 2 (fixed for pandas 3.x): Extract a representative sample of Individual
Recode (IR) Stata files, analyze schema (via StataReader metadata), then load
only key columns for distribution analysis.
"""
import os
import re
import glob
import zipfile
import gc
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

ROOTS = [".", "Child-Mortality-Analysis-Data-main 2"]
EXTRACT_DIR = "extracted_ir"
os.makedirs(EXTRACT_DIR, exist_ok=True)

def classify(fname):
    stem = os.path.basename(fname)[:-4]
    m = re.match(r"^([A-Z]{2})([A-Z]{2})([0-9A-Z]{1,3})(DT|FL|SD|SV)$", stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)

ir_dt = {}
for root in ROOTS:
    for p in glob.glob(os.path.join(root, "*.zip")):
        c = classify(p)
        if c and c[1] == "IR" and c[3] == "DT":
            ir_dt[(c[0], c[2])] = p

SAMPLE = [
    ("BR", "31", "Latin America (oldest, DHS-III 1996)"),
    ("TN", "02", "North Africa (old, DHS-II)"),
    ("LK", "02", "South Asia (old, DHS-II)"),
    ("BO", "51", "Latin America (DHS-V 2008)"),
    ("EG", "61", "Middle East (DHS-VI 2014)"),
    ("KY", "61", "Central Asia (DHS-VI)"),
    ("ZA", "71", "Southern Africa (DHS-VII 2016)"),
    ("ID", "71", "SE Asia (DHS-VII 2017)"),
    ("HT", "71", "Caribbean (DHS-VII 2016-17)"),
    ("TR", "71", "Middle East/Europe (DHS-VII 2018)"),
    ("PH", "82", "SE Asia (DHS-VIII 2017)"),
    ("NP", "82", "South Asia (DHS-VIII 2016)"),
    ("NG", "8B", "West Africa (DHS-VIII 2018, largest)"),
    ("RW", "91", "East Africa (DHS-IX 2019)"),
    ("KE", "91", "East Africa (DHS-IX 2022, latest)"),
]

KEY_VARS = {
    "weights": ["v005", "v021", "v022", "v023", "v024", "v025"],
    "demographics": ["v012", "v013", "v106", "v107", "v133", "v130", "v131",
                     "v151", "v152", "v155", "v190", "v191"],
    "residence/wealth": ["v025", "v190", "v191", "v113", "v116", "v119", "v161"],
    "fertility": ["v201", "v206", "v207", "v208", "v209", "v213", "v214",
                  "v218", "v219", "v220", "v224"],
    "contraception": ["v301", "v302", "v312", "v313", "v323", "v326", "v361",
                      "v362", "v363", "v364"],
    "marriage": ["v501", "v502", "v503", "v511", "v512", "v525"],
    "media/exposure": ["v157", "v158", "v159", "v384a", "v384b", "v384c"],
    "maternal-health(IR)": ["v404", "v405", "v406", "v407", "v418", "v419",
                            "v445", "v446", "v452", "v454", "v457", "v463"],
}

def extract_dta(country, phase):
    key = (country, phase)
    if key not in ir_dt:
        return None
    zp = ir_dt[key]
    with zipfile.ZipFile(zp) as z:
        dta = [n for n in z.namelist() if n.upper().endswith(".DTA")][0]
        out = os.path.join(EXTRACT_DIR, f"{country}{phase}.dta")
        if not os.path.exists(out):
            with z.open(dta) as src, open(out, "wb") as dst:
                dst.write(src.read())
        return out

report = {}
prefix_totals = Counter()
print("Extracting + analyzing representative IR datasets...\n")
for country, phase, region in SAMPLE:
    dta = extract_dta(country, phase)
    if dta is None:
        print(f"  [SKIP] no DT for {country}{phase}")
        continue
    try:
        rdr = pd.io.stata.StataReader(dta)
        vlabels = rdr.variable_labels()
        cols = list(vlabels.keys())
        ncols = len(cols)
        val_labels = rdr.value_labels()
    except Exception as e:
        print(f"  [ERROR schema] {country}{phase}: {e}")
        continue

    prefixes = Counter(c[:1].lower() for c in cols)
    for k, v in prefixes.items():
        prefix_totals[k] += v

    # only load the columns we actually care about
    load_cols = [c for c in cols if c in set(v for vs in KEY_VARS.values() for v in vs)]
    try:
        df = pd.read_stata(dta, columns=load_cols, convert_categoricals=False)
        nrows = len(df)
        present = {g: [v for v in vs if v in cols] for g, vs in KEY_VARS.items()}

        dist = {}
        def pr(col, cond):
            if col in df:
                s = df[col].dropna()
                return round(float((s == cond).mean()), 3) if len(s) else None
            return None
        dist["urban"] = pr("v025", 1)
        dist["no_education"] = pr("v106", 0)
        dist["higher_edu"] = pr("v106", 3)
        dist["richest_q5"] = pr("v190", 5)
        dist["any_contraceptive"] = pr("v312", 3)  # v312==3 means 'using' codes >=? -> use v313 instead
        dist["modern_method(v313==3)"] = pr("v313", 3)
        dist["traditional(v313==2)"] = pr("v313", 2)
        if "v201" in df:
            dist["mean_CEB"] = round(float(df["v201"].mean()), 2)
        if "v012" in df:
            dist["age_mean"] = round(float(df["v012"].mean()), 1)

        report[(country, phase)] = dict(region=region, nrows=nrows, ncols=ncols,
                                        prefixes=dict(prefixes), present=present, dist=dist)
        print(f"{country}{phase:3s} [{region}]: rows={nrows:>7,}  cols={ncols:>4}  "
              f"prefixes={dict(prefixes)}")
        print(f"     {dist}")
        del df
    except Exception as e:
        print(f"  [ERROR load] {country}{phase}: {e}")
    gc.collect()

print("\n\n=== COLUMN PREFIX TOTALS ACROSS SAMPLE ===")
print(dict(prefix_totals))

print("\n=== HEADLINE CROSS-COUNTRY VARIATION (target for fairness/shift) ===")
print(f"{'ctry':>6} {'nrows':>9} {'urban':>6} {'noEdu':>6} {'richQ5':>6} "
      f"{'modern':>7} {'trad':>6} {'meanCEB':>8} {'age':>5}")
for (c, p) in report:
    d = report[(c, p)]["dist"]
    row = f"{c+p:>6} {report[(c,p)]['nrows']:>9,} "
    for k in ["urban", "no_education", "richest_q5", "modern_method(v313==3)",
              "traditional(v313==2)", "mean_CEB", "age_mean"]:
        v = d.get(k)
        row += f"{'' if v is None else v:>6} "
    print(row)

print("\n=== FEATURE PRESENCE MATRIX (Y = present in that dataset) ===")
allvars = [v for vs in KEY_VARS.values() for v in vs]
order = list(report.keys())
print("var".ljust(9) + "".join(f"{c+p:>6}" for c, p in order))
for v in allvars:
    row = v.ljust(9)
    for c, p in order:
        present = any(v in report[(c, p)]["present"][g] for g in KEY_VARS)
        row += f"{'Y' if present else '.':>6}"
    print(row)

# Save a compact summary for later use
import json
summary = {f"{c}{p}": {**v, "present": {k: vv for k, vv in v["present"].items() if vv}}
           for (c, p), v in report.items()}
with open("ir_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSaved ir_summary.json")
