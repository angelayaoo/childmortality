"""
Step 3 (feasibility proof): Build a child-level dataset directly from the IR
birth history (melt b5/b6/b7/b4/b8/b3/bord/b11 wide arrays) and compute the
under-five mortality rate (and its variation across wealth/residence/region)
for a few countries. This proves the pipeline for the recommended projects.
"""
import pandas as pd
import numpy as np
import re

def load_child_level(dta, country):
    r = pd.io.stata.StataReader(dta)
    cols = list(r.variable_labels().keys())

    # find max birth index
    bidx = sorted([c for c in cols if re.match(r"^b5_\d+$", c)],
                  key=lambda x: int(x.split("_")[1]))
    maxn = max(int(c.split("_")[1]) for c in bidx)

    mother_cols = ["v005", "v008", "v012", "v013", "v025", "v024", "v106", "v133", "v190"]
    mother_cols = [c for c in mother_cols if c in cols]

    # child-level columns per birth
    child_vars = {
        "b5": "alive", "b6": "age_death_months", "b7": "age_death_months_imp",
        "b8": "age_months", "b4": "sex", "b3": "birth_cmc", "bord": "birth_order",
        "b11": "prev_interval", "b2": "birth_order2",
    }
    frames = []
    for i in range(1, maxn + 1):
        sfx = f"{i:02d}"
        keep = mother_cols[:]
        colmap = {}
        for var, label in child_vars.items():
            cname = f"{var}_{sfx}"
            if cname in cols:
                keep.append(cname)
                colmap[cname] = label
        if f"b5_{sfx}" not in cols:
            continue
        keep = list(dict.fromkeys(keep))
        sub = pd.read_stata(dta, columns=keep, convert_categoricals=False)
        sub = sub.rename(columns=colmap)
        sub["birth_idx"] = i
        sub["country"] = country
        frames.append(sub)
    child = pd.concat(frames, ignore_index=True)
    # drop children never born (b5 missing)
    child = child[child["alive"].notna()].copy()
    return child

for country, phase in [("KE", "91"), ("NG", "8B"), ("RW", "91"), ("BO", "51")]:
    dta = f"extracted_ir/{country}{phase}.dta"
    ch = load_child_level(dta, country)
    ch["born_last5y"] = (ch["v008"] - ch["birth_cmc"]) < 60
    recent = ch[ch["born_last5y"] & (ch["birth_cmc"] > 0)]
    dead = (recent["alive"] == 0)
    u5 = float(dead.mean())
    n = len(recent)
    # wealth gradient
    grad = {}
    if "v190" in recent:
        for q in [1, 5]:
            sub = recent[recent["v190"] == q]
            if len(sub):
                grad[f"Q{q}"] = round(float((sub["alive"] == 0).mean()) * 1000, 1)
    print(f"{country}{phase}: n_recent_births={n:>6,}  U5MR={u5*1000:5.1f}/1000  "
          f"wealth_gradient(per1000)={grad}")
