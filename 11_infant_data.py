"""
Project 1 (revision) — censoring-correct outcome.

The reviewer's key criticism: labeling births in the last 5 years by "dead at
interview" is wrong, because a child alive at 6 months is not a known survivor
to age 5. Fix: use infant mortality (death before 12 months) with a fixed,
fully observed horizon. We restrict to children born >= 12 months before the
interview (full exposure) and label infant_death = (b5==0) AND (b7 < 12), where
b7 is age at death in completed months (imputed). Every included child is either
observed to die before 12 months or observed to survive to >= 12 months.
"""
import os, re, glob, zipfile
import numpy as np
import pandas as pd

ROOTS = [".", "Child-Mortality-Analysis-Data-main 2"]
EXTRACT = "extracted_ir"
CACHE = "cache_infant"
os.makedirs(CACHE, exist_ok=True)

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
MOTHER_COLS = ["v005", "v007", "v008", "v012", "v025", "v106", "v133", "v190"]
CHILD_WIDE = {"b3": "birth_cmc", "b5": "alive", "b7": "age_death_months",
              "b4": "sex", "bord": "birth_order", "b11": "prev_interval"}

def classify(fname):
    stem = os.path.basename(fname)[:-4]
    m = re.match(r"^([A-Z]{2})([A-Z]{2})([0-9A-Z]{1,3})(DT|FL|SD|SV)$", stem)
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None

def find_ir_dt():
    found = {}
    for root in ROOTS:
        for p in glob.glob(os.path.join(root, "*.zip")):
            c = classify(p)
            if c and c[1] == "IR" and c[3] == "DT":
                found[(c[0], c[2])] = p
    return found

def extract_dta(country, phase, zp):
    out = os.path.join(EXTRACT, f"{country}{phase}.dta")
    if os.path.exists(out):
        return out
    with zipfile.ZipFile(zp) as z:
        dta = [n for n in z.namelist() if n.upper().endswith(".DTA")][0]
        with z.open(dta) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out

ir_dt = find_ir_dt()
rows = []
for (c, p), zp in sorted(ir_dt.items()):
    dta = extract_dta(c, p, zp)
    r = pd.io.stata.StataReader(dta)
    cols = list(r.variable_labels().keys())
    colset = set(cols)
    b5cols = [x for x in cols if re.match(r"^b5_\d+$", x)]
    if not b5cols:
        print(f"  {c}{p}: SKIP (no birth history)"); continue
    maxn = max(int(x.split("_")[1]) for x in b5cols)
    frames = []
    for i in range(1, maxn + 1):
        sfx = f"{i:02d}"
        keep = [x for x in MOTHER_COLS if x in colset]
        rename = {}
        for w, lab in CHILD_WIDE.items():
            x = f"{w}_{sfx}"
            if x in colset:
                keep.append(x); rename[x] = lab
        if f"b5_{sfx}" not in colset:
            continue
        sub = pd.read_stata(dta, columns=list(dict.fromkeys(keep)), convert_categoricals=False)
        sub = sub.rename(columns=rename)
        frames.append(sub)
    ch = pd.concat(frames, ignore_index=True)
    ch = ch[ch["alive"].notna()].copy()
    ch["age_months"] = ch["v008"] - ch["birth_cmc"]
    ch = ch[(ch["birth_cmc"] > 0) & (ch["age_months"] >= 0)].copy()
    # full 12-month exposure, within the recent 5-year window (age 12-59 months)
    ch = ch[(ch["age_months"] >= 12) & (ch["age_months"] < 60)].copy()
    # infant death = died before 12 completed months
    ch["y"] = ((ch["alive"] == 0) & (ch["age_death_months"] < 12)).astype(int)
    ch["child_male"] = (ch["sex"] == 1).astype(int)
    ch["birth_order"] = ch["birth_order"].astype(float)
    ch["prev_interval"] = ch["prev_interval"].astype(float)
    ch["mother_age_birth"] = ch["v012"] - ch["age_months"] / 12.0
    ch["edu"] = ch["v106"].astype(float) if "v106" in ch else np.nan
    ch["edu_years"] = ch["v133"].astype(float) if "v133" in ch else np.nan
    ch["year"] = ch["v007"].astype(float)
    ch["weight"] = ch["v005"].astype(float) / 1e6
    keep_cols = UNIVERSAL + ["y", "year", "weight"]
    ch = ch[keep_cols]
    ch.to_parquet(f"{CACHE}/{c}{p}.parquet", index=False)
    n_deaths = int(ch["y"].sum())
    rows.append({"key": f"{c}{p}", "n": len(ch), "n_deaths": n_deaths,
                 "imr": float(ch["y"].mean() * 1000)})
    print(f"  {c}{p}: n={len(ch):>6,} infant_deaths={n_deaths:>4}  IMR={ch['y'].mean()*1000:5.1f}/1000")

res = pd.DataFrame(rows).sort_values("key")
res.to_parquet("infant_summary.parquet")
print(f"\nTotal surveys built: {len(res)}")
print(f"Surveys with >=30 infant deaths: {(res.n_deaths >= 30).sum()}")
print(f"Surveys with >=50 infant deaths: {(res.n_deaths >= 50).sum()}")
print(f"IMR range: {res.imr.min():.1f} to {res.imr.max():.1f} per 1000, mean {res.imr.mean():.1f}")
