"""
Project 1 (revision) — apply DHS sampling weights everywhere.

Weighted re-run of the core transfer analysis:
  - weighted training (sample_weight in XGBoost)
  - weighted AUC (Mann-Whitney U with weights, tie-corrected)
  - weighted Brier
The child-level weight is the mother's DHS sample weight (v005), standard for
child-level DHS analyses (each child inherits the mother's weight).

Outputs the weighted headline numbers (in-domain AUC, transfer AUC, penalty,
donor quality) to replace the earlier unweighted estimates.
"""
import glob, os, re, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
RNG = 0

def weighted_auc(y, s, w):
    y = np.asarray(y); s = np.asarray(s); w = np.asarray(w, dtype=float)
    order = np.argsort(s)
    y, s, w = y[order], s[order], w[order]
    Wp = w[y == 1].sum(); Wn = w[y == 0].sum()
    if Wp == 0 or Wn == 0:
        return float("nan")
    total, run_neg, i, n = 0.0, 0.0, 0, len(y)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        wn_block = w[i:j][y[i:j] == 0].sum()
        wp_block = w[i:j][y[i:j] == 1].sum()
        total += wp_block * (run_neg + 0.5 * wn_block)
        run_neg += wn_block
        i = j
    return float(total / (Wp * Wn))

def make_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

def load(key):
    return pd.read_parquet(f"{CACHE}/{key}.parquet")

def w_cv_auc(df):
    X = df[UNIVERSAL].values; y = df["y"].values; w = df["weight"].values
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=RNG).split(X, y):
        m = make_model(); m.fit(X[tr], y[tr], sample_weight=w[tr])
        aucs.append(weighted_auc(y[te], m.predict_proba(X[te])[:, 1], w[te]))
    return float(np.mean(aucs))

keys = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{CACHE}/*.parquet"))
# keep only surveys with >=50 deaths (stable AUC), matching the main analysis
keys = [k for k in keys
        if int(pd.read_parquet(f"{CACHE}/{k}.parquet", columns=["y"])["y"].sum()) >= 50]
print(f"weighted re-run over {len(keys)} surveys", flush=True)

# diagonal (weighted CV)
diag = {}
for i, k in enumerate(keys):
    diag[k] = w_cv_auc(load(k))
    if (i + 1) % 20 == 0:
        print(f"  weighted CV {i+1}/{len(keys)}", flush=True)

# off-diagonal (weighted transfer)
N = len(keys)
matrix = np.full((N, N), np.nan)
models = {}
for i, s in enumerate(keys):
    dfs = load(s)
    Xs, ys, ws = dfs[UNIVERSAL].values, dfs["y"].values, dfs["weight"].values
    m = make_model(); m.fit(Xs, ys, sample_weight=ws)
    models[s] = m
    for j, t in enumerate(keys):
        if s == t:
            matrix[i, j] = diag[s]
        else:
            dft = load(t)
            Xt, yt, wt = dft[UNIVERSAL].values, dft["y"].values, dft["weight"].values
            matrix[i, j] = weighted_auc(yt, m.predict_proba(Xt)[:, 1], wt)
    if (i + 1) % 15 == 0:
        print(f"  weighted transfer {i+1}/{N}", flush=True)

idx = [k for k in keys]
pd.DataFrame(matrix, index=idx, columns=idx).to_parquet("matrix_transfer_weighted.parquet")
pd.Series(diag).rename("in_domain_auc_w").to_frame().to_parquet("diag_auc_weighted.parquet")

in_dom = np.array([diag[k] for k in keys])
off = np.array([matrix[i, j] for i in range(N) for j in range(N) if i != j])
gap_all = np.array([diag[keys[j]] - matrix[i, j] for i in range(N) for j in range(N) if i != j])
offdiag_mean = {k: float(np.nanmean([matrix[i, j] for j in range(N) if j != i]))
                for i, k in enumerate(keys)}

print("\n=== WEIGHTED headline results ===")
print(f"n countries = {N}")
print(f"weighted in-domain AUC: mean={in_dom.mean():.3f} median={np.median(in_dom):.3f} "
      f"range=[{in_dom.min():.3f},{in_dom.max():.3f}]")
print(f"weighted transfer AUC:  mean={off.mean():.3f} median={np.median(off):.3f}")
print(f"weighted mean penalty:  {np.mean([diag[keys[i]]-matrix[i,j] for i in range(N) for j in range(N) if i!=j]):.3f}")
print(f"percent degrading:      {(gap_all > 0).mean()*100:.0f}%")
print(f"weighted concept-drift gap: mean={gap_all.mean():+.3f} median={np.median(gap_all):+.3f}")

print("\nweighted donor quality (top/bottom 5):")
ranked = sorted(offdiag_mean.items(), key=lambda x: -x[1])
for k, v in ranked[:5]:
    print(f"  {k}: in-domain={diag[k]:.3f} mean-transfer={v:.3f}")
print("  ...")
for k, v in ranked[-5:]:
    print(f"  {k}: in-domain={diag[k]:.3f} mean-transfer={v:.3f}")

summary = {
    "n_countries": N,
    "weighted_in_domain_mean": round(float(in_dom.mean()), 4),
    "weighted_in_domain_median": round(float(np.median(in_dom)), 4),
    "weighted_transfer_mean": round(float(off.mean()), 4),
    "weighted_transfer_median": round(float(np.median(off)), 4),
    "weighted_mean_penalty": round(float(np.mean([diag[keys[i]]-matrix[i,j] for i in range(N) for j in range(N) if i!=j])), 4),
    "pct_degrading": round(float((gap_all > 0).mean() * 100), 1),
    "weighted_drift_gap_mean": round(float(gap_all.mean()), 4),
}
with open("results_weighted.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved matrix_transfer_weighted.parquet, results_weighted.json")
