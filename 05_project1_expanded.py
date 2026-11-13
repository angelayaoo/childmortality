"""
Project 1 (expanded): full 82-country cross-survey transfer analysis + OOD-detector limits.

Stages:
  A. Extract all 82 IR Stata files.
  B. Build child-level parquet for every country using the UNIVERSAL feature set
     (birth history + education + mother age) that exists in every DHS phase.
     Also record per-country feature coverage (schema drift) and survey year.
  C. 82x82 transfer matrix (rows=source, cols=target): diagonal = 3-fold CV AUC,
     off-diagonal = source model applied to target.
  D. OOD-detector limit analysis: for a set of source surveys, scatter
        x = domain-classifier AUC (how "OOD" the target looks in feature space)
        y = concept-drift gap = oracle(target) - transfer(source->target)
     and test whether feature-space OOD detectability predicts model failure.

Set LIMIT=<n> to run a quick subset (smoke test).
"""
import os, re, json, sys, time, glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RNG = 0
LIMIT = int(os.environ.get("LIMIT", "0")) or None   # None => all countries
CACHE = "cache_univ"
EXTRACT = "extracted_ir"
FIG = "figs"
os.makedirs(CACHE, exist_ok=True); os.makedirs(EXTRACT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# universal features (present in ALL DHS phases)
UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
# optional features (phase IV+); recorded but not used in the matrix
OPTIONAL = ["urban", "wealth_q"]

MOTHER_COLS = ["v005", "v007", "v008", "v012", "v025", "v106", "v133", "v190"]
CHILD_WIDE = {"b3": "birth_cmc", "b5": "alive", "b4": "sex",
              "bord": "birth_order", "b11": "prev_interval"}

def classify(fname):
    stem = os.path.basename(fname)[:-4]
    m = re.match(r"^([A-Z]{2})([A-Z]{2})([0-9A-Z]{1,3})(DT|FL|SD|SV)$", stem)
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None

def find_ir_dt():
    roots = [".", "Child-Mortality-Analysis-Data-main 2"]
    found = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "*.zip")):
            c = classify(p)
            if c and c[1] == "IR" and c[3] == "DT":
                found[(c[0], c[2])] = p
    return found

def extract_dta(country, phase, zp):
    out = os.path.join(EXTRACT, f"{country}{phase}.dta")
    if os.path.exists(out):
        return out
    import zipfile
    with zipfile.ZipFile(zp) as z:
        dta = [n for n in z.namelist() if n.upper().endswith(".DTA")][0]
        with z.open(dta) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out

def build_child(country, phase, dta):
    """Return (child_df, meta). child_df has UNIVERSAL features + y + weight + year."""
    cache = f"{CACHE}/{country}{phase}.parquet"
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        meta = {"n": len(df), "u5mr": float(df["y"].mean()),
                "year": float(df["year"].mean()), "n_deaths": int(df["y"].sum()),
                "has_urban": "urban" in df.columns, "has_wealth": "wealth_q" in df.columns}
        return df, meta
    r = pd.io.stata.StataReader(dta)
    cols = list(r.variable_labels().keys())
    colset = set(cols)
    b5cols = [c for c in cols if re.match(r"^b5_\d+$", c)]
    if not b5cols:
        return None, None   # non-standard IR without birth history (e.g., VN52)
    maxn = max(int(c.split("_")[1]) for c in b5cols)
    frames = []
    for i in range(1, maxn + 1):
        sfx = f"{i:02d}"
        keep = [c for c in MOTHER_COLS if c in colset]
        rename = {}
        for w, lab in CHILD_WIDE.items():
            c = f"{w}_{sfx}"
            if c in colset:
                keep.append(c); rename[c] = lab
        if f"b5_{sfx}" not in colset:
            continue
        sub = pd.read_stata(dta, columns=list(dict.fromkeys(keep)), convert_categoricals=False)
        sub = sub.rename(columns=rename)
        frames.append(sub)
    ch = pd.concat(frames, ignore_index=True)
    ch = ch[ch["alive"].notna()].copy()
    ch["age_months"] = ch["v008"] - ch["birth_cmc"]
    ch = ch[(ch["birth_cmc"] > 0) & (ch["age_months"] >= 0) & (ch["age_months"] < 60)].copy()
    ch["y"] = (ch["alive"] == 0).astype(int)
    ch["child_male"] = (ch["sex"] == 1).astype(int)
    ch["birth_order"] = ch["birth_order"].astype(float)
    ch["prev_interval"] = ch["prev_interval"].astype(float)
    ch["mother_age_birth"] = ch["v012"] - ch["age_months"] / 12.0
    ch["edu"] = ch["v106"].astype(float) if "v106" in ch else np.nan
    ch["edu_years"] = ch["v133"].astype(float) if "v133" in ch else np.nan
    if "v025" in ch:
        ch["urban"] = (ch["v025"] == 1).astype(int)
    if "v190" in ch:
        ch["wealth_q"] = ch["v190"].astype(float)
    ch["year"] = ch["v007"].astype(float)
    ch["weight"] = ch["v005"].astype(float) / 1e6
    keep_cols = UNIVERSAL + ["y", "year", "weight"] + \
                [c for c in OPTIONAL if c in ch.columns]
    ch = ch[keep_cols]
    ch.to_parquet(cache, index=False)
    meta = {"n": len(ch), "u5mr": float(ch["y"].mean()), "year": float(ch["year"].mean()),
            "n_deaths": int(ch["y"].sum()),
            "has_urban": "urban" in ch.columns, "has_wealth": "wealth_q" in ch.columns}
    return ch, meta

def make_model():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

def cv_auc(df, n_splits=3):
    X, y = df[UNIVERSAL].values, df["y"].values
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

# ---------------- A. extract ----------------
print("=== Stage A: extract all IR Stata files ===", flush=True)
ir_dt = find_ir_dt()
pairs = sorted(ir_dt.keys())
if LIMIT:
    pairs = pairs[:LIMIT]
print(f"  {len(pairs)} countries")
for i, (c, p) in enumerate(pairs):
    extract_dta(c, p, ir_dt[(c, p)])

# ---------------- B. build child data ----------------
print("\n=== Stage B: build child-level datasets ===", flush=True)
DATA, META = {}, {}
for c, p in pairs:
    t0 = time.time()
    df, meta = build_child(c, p, os.path.join(EXTRACT, f"{c}{p}.dta"))
    if df is None:
        print(f"  {c}{p}: SKIP (no birth history in IR)")
        continue
    if meta["n_deaths"] < 50:
        print(f"  {c}{p}: SKIP (n_deaths={meta['n_deaths']} < 50, unstable AUC)")
        continue
    DATA[(c, p)] = df
    META[(c, p)] = meta
    print(f"  {c}{p}: n={meta['n']:>6,}  U5MR={meta['u5mr']*1000:5.1f}/1000  "
          f"year={meta['year']:.0f}  urban={meta['has_urban']} wealth={meta['has_wealth']} "
          f"({time.time()-t0:.1f}s)", flush=True)

keys = list(DATA.keys())
print(f"\n  usable countries: {len(keys)}")
# schema drift summary
n_wealth = sum(1 for k in keys if META[k]["has_wealth"])
n_urban = sum(1 for k in keys if META[k]["has_urban"])
print(f"  with wealth index (phase IV+): {n_wealth}/{len(keys)}")
print(f"  with residence (phase III+):   {n_urban}/{len(keys)}")

# ---------------- C. transfer matrix ----------------
print("\n=== Stage C: 82x82 transfer matrix ===", flush=True)
key_by_country = {}
for c, p in keys:
    key_by_country.setdefault(c, []).append(p)

N = len(keys)
matrix = np.full((N, N), np.nan)
diag = {}
models = {}

# diagonal (CV) first
t0 = time.time()
for i, k in enumerate(keys):
    diag[k] = cv_auc(DATA[k])
    if (i + 1) % 20 == 0:
        print(f"  diagonal CV {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

# off-diagonal: train source model, predict on all targets
for i, s in enumerate(keys):
    Xs, ys = DATA[s][UNIVERSAL].values, DATA[s]["y"].values
    m = make_model(); m.fit(Xs, ys)
    models[s] = m
    for j, t in enumerate(keys):
        if s == t:
            matrix[i, j] = diag[s]
        else:
            Xt = DATA[t][UNIVERSAL].values
            matrix[i, j] = roc_auc_score(DATA[t]["y"].values,
                                         m.predict_proba(Xt)[:, 1])
    if (i + 1) % 10 == 0:
        print(f"  transfer row {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

# save matrix
idx = [f"{c}{p}" for c, p in keys]
np.save("matrix_transfer.npy", matrix)
pd.DataFrame(matrix, index=idx, columns=idx).to_parquet("matrix_transfer.parquet")
pd.Series(diag).rename("in_domain_auc").to_frame().to_parquet("diag_auc.parquet")

print("\n=== Donor-quality summary (mean off-diagonal AUC per source) ===", flush=True)
offdiag_mean = {k: float(np.nanmean([matrix[i, j] for j in range(N) if j != i]))
                for i, k in enumerate(keys)}
offdiag_mean_sorted = sorted(offdiag_mean.items(), key=lambda x: -x[1])
for k, v in offdiag_mean_sorted:
    print(f"  {k[0]}{k[1]:6s} in-domain={diag[k]:.3f}  mean-transfer={v:.3f}")

# concept-drift gap for every pair
gap = np.full((N, N), np.nan)
for i, s in enumerate(keys):
    for j, t in enumerate(keys):
        if s != t:
            gap[i, j] = diag[t] - matrix[i, j]   # oracle(target) - transfer(s->t)
pd.DataFrame(gap, index=idx, columns=idx).to_parquet("concept_drift_gap.parquet")

# ---------------- D. OOD-detector limit analysis ----------------
print("\n=== Stage D: OOD-detector limits (covariate-shift detectability vs failure) ===", flush=True)

def domain_auc(s, t):
    """AUROC of a classifier separating source vs target in feature space."""
    X = np.vstack([DATA[s][UNIVERSAL].values, DATA[t][UNIVERSAL].values])
    y = np.array([1] * len(DATA[s]) + [0] * len(DATA[t]))
    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
    for tr, te in skf.split(X, y):
        m = make_model(); m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

# diverse sources spanning phases/regions (use the first LIMIT-many that exist)
candidate_sources = [("KE", "91"), ("NG", "8B"), ("BO", "51"), ("EG", "61"),
                     ("ID", "71"), ("PH", "82"), ("RW", "91"), ("HT", "71")]
sources = [s for s in candidate_sources if s in DATA]
if not sources:
    sources = keys[:min(8, len(keys))]
print(f"  sources: {[f'{c}{p}' for c,p in sources]}")
rows = []
for s in sources:
    si = keys.index(s)
    for t in keys:
        if s == t:
            continue
        ti = keys.index(t)
        da = domain_auc(s, t)
        rows.append({"source": f"{s[0]}{s[1]}", "target": f"{t[0]}{t[1]}",
                     "domain_auc": da, "concept_drift_gap": gap[si, ti],
                     "oracle_auc": diag[t], "transfer_auc": matrix[si, ti]})
        print(f"    {s[0]}{s[1]}->{t[0]}{t[1]}: domain_auc={da:.2f}  gap={gap[si,ti]:+.3f}", flush=True)

ood = pd.DataFrame(rows)
ood.to_parquet("ood_detector_limits.parquet")

if len(ood) == 0:
    r = float("nan")
    print("\n  (no OOD pairs computed)")
else:
    # correlation: does feature-space OOD detectability predict failure?
    r = ood[["domain_auc", "concept_drift_gap"]].corr().iloc[0, 1]
    print(f"\n  Pearson r(domain_AUC, concept_drift_gap) = {r:.3f}  (n={len(ood)})")

# ---------------- figures ----------------
# Fig: transfer heatmap (if N <= 40, full; else use first 40)
sub = pd.DataFrame(matrix, index=idx, columns=idx)
if N > 40:
    sub = sub.iloc[:40, :40]
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(sub, cmap="viridis", vmin=0.5, vmax=0.75, ax=ax,
            cbar_kws={"label": "AUC"})
ax.set_title(f"Cross-survey transfer AUC (source rows -> target cols), n={N} countries")
ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_full_transfer_matrix.png", dpi=130); plt.close()

# Fig: donor quality (in-domain vs mean-transfer)
fig, ax = plt.subplots(figsize=(8, 6))
x = [diag[k] for k in keys]; y = [offdiag_mean[k] for k in keys]
ax.scatter(x, y, s=30, alpha=0.7)
for k, xi, yi in zip(keys, x, y):
    ax.annotate(f"{k[0]}{k[1]}", (xi, yi), fontsize=6, alpha=0.6)
ax.plot([0.45, 0.75], [0.45, 0.75], "k--", lw=1)
ax.set_xlabel("In-domain AUC (own CV)")
ax.set_ylabel("Mean transfer AUC (to all others)")
ax.set_title("Donor quality: how well each survey's model transfers")
plt.tight_layout(); plt.savefig(f"{FIG}/fig6_donor_quality.png", dpi=130); plt.close()

# Fig: OOD detector limits (the key DAI figure)
if len(ood) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ood["domain_auc"], ood["concept_drift_gap"], s=12, alpha=0.5, color="#4C72B0")
    ax.set_xlabel("Feature-space OOD detectability (domain-classifier AUC)")
    ax.set_ylabel("Concept-drift gap (oracle - transfer AUC)")
    ax.set_title(f"OOD detectors do not predict model failure\nr = {r:.2f}, n = {len(ood)} pairs")
    ax.axhline(0, ls="--", c="gray", lw=1)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig7_ood_detector_limits.png", dpi=130); plt.close()

# summary json
summary = {
    "n_countries": N,
    "universal_features": UNIVERSAL,
    "n_with_wealth": n_wealth, "n_with_urban": n_urban,
    "ood_corr_domain_vs_gap": round(float(r), 3),
    "ood_n_pairs": int(len(ood)),
    "donor_quality": {f"{k[0]}{k[1]}": {"in_domain": round(diag[k], 3),
                                     "mean_transfer": round(offdiag_mean[k], 3)}
                      for k in keys},
}
with open("results_project1_expanded.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDone. Saved results_project1_expanded.json, parquet matrices, and figures.")
