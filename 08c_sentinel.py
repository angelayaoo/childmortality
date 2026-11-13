"""
Project 1 (playbook) — Method 3: Ground-Truth Sentinel Cohorts (v2).

Monitoring model *error* (Brier / calibration) on a tiny verified cohort is the
operationally-correct monitor for a risk screener, and it is cheaply estimable
(unlike AUC, which needs many positives). We show:
  (a) a sentinel Brier estimate converges with ~200 labels (KE->NG);
  (b) across 592 pairs, a 300-label sentinel estimates the true deployment
      error near-perfectly (r -> 1), whereas feature-OOD/uncertainty give ~0.1.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=3,
              learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=15, tree_method="hist", n_jobs=8, seed=0)

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

def brier(p, y): return float(np.mean((p - y) ** 2))

# --- (a) convergence of sentinel Brier (KE91 -> NG8B) ---
print("=== (a) sentinel Brier estimate converges (KE91 -> NG8B) ===")
dfs, dft = load("KE91"), load("NG8B")
X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
bst = xgb.train(PARAMS, xgb.DMatrix(X_s, label=y_s), num_boost_round=200)
p_t = bst.predict(xgb.DMatrix(X_t))
true_brier = brier(p_t, y_t)
print(f"true Brier on full target = {true_brier:.4f}")
rng = np.random.default_rng(0)
for N in [25, 50, 100, 200, 500, 1000]:
    ests = [brier(p_t[idx], y_t[idx])
            for _ in range(50)
            for idx in [rng.choice(len(y_t), size=N, replace=False)]]
    ests = np.array(ests)
    print(f"  N={N:5d}: mean={ests.mean():.4f}  std={ests.std():.4f}  "
          f"MAE={np.abs(ests-true_brier).mean():.4f}")

# --- (b) 592 pairs: sentinel-300 Brier vs true Brier ---
print("\n=== (b) sentinel-300 error estimate vs true error (592 pairs) ===")
gap = pd.read_parquet("concept_drift_gap.parquet")
keys = list(gap.index)
SOURCES = [k for k in ["KE91","NG8B","BO51","EG61","ID71","PH82","RW91","HT71"] if k in keys]
rows = []
for s in SOURCES:
    dfs = load(s)
    X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
    bst = xgb.train(PARAMS, xgb.DMatrix(X_s, label=y_s), num_boost_round=200)
    for t in keys:
        if t == s: continue
        dft = load(t)
        X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
        p_t = bst.predict(xgb.DMatrix(X_t))
        true_b = brier(p_t, y_t)
        idx = rng.choice(len(y_t), size=300, replace=False)
        sent_b = brier(p_t[idx], y_t[idx])
        rows.append({"source": s, "target": t, "true_brier": true_b,
                     "sentinel300_brier": sent_b,
                     "base_rate_gap": abs(y_s.mean() - y_t.mean()),
                     "concept_drift_gap": gap.loc[s, t]})
    print(f"  {s}: done", flush=True)
res = pd.DataFrame(rows)
r_err = res.sentinel300_brier.corr(res.true_brier)
r_base = res.sentinel300_brier.corr(res.base_rate_gap)
print(f"\n  r(sentinel-300 Brier, true Brier)   = {r_err:.3f}   (feature-OOD: ~0.1, uncertainty: ~0.07)")
print(f"  r(sentinel-300 Brier, base-rate gap) = {r_base:.3f}  (sentinel primarily tracks label shift)")
res.to_parquet("sentinel_monitor.parquet")
