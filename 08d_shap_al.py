"""
Project 1 (playbook) — Method 4: SHAP-Guided Active Learning.

For a fixed local label budget N, compare held-out AUC recovered by
  - random / uniform sampling
  - SHAP-guided sampling (oversample the wealth quintiles whose gradient
    diverges most between source and target)
using few-shot warm-start fine-tuning on KE -> NG.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score

RICH = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
        "edu", "urban", "wealth_q"]

def load_rich(c):
    df = pd.read_parquet(f"cache_children/{c}.parquet")[RICH + ["y"]].copy()
    df["prev_interval"] = df["prev_interval"].fillna(df["prev_interval"].median())
    return df

src, tgt = load_rich("KE91"), load_rich("NG8B")
X_s, y_s = src[RICH].values, src["y"].values
X_t, y_t = tgt[RICH].values, tgt["y"].values
n_t = len(y_t)
wealth = tgt["wealth_q"].values.astype(int)

PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=4,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=15, tree_method="hist", n_jobs=8, seed=0)

# source model (sklearn) for SHAP + get_booster for warm-start
m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
                      tree_method="hist", n_jobs=8, random_state=0, eval_metric="auc")
m.fit(X_s, y_s)
bst = m.get_booster()

# per-quintile SHAP divergence (source model on source vs on target features)
ex = shap.TreeExplainer(m)
shap_s = ex.shap_values(X_s)
shap_t = ex.shap_values(X_t)
w = RICH.index("wealth_q")
div = {q: abs(shap_s[src["wealth_q"].values == q, w].mean() -
              shap_t[tgt["wealth_q"].values == q, w].mean()) for q in range(1, 6)}
print("per-quintile SHAP divergence (wealth):", {q: round(v, 3) for q, v in div.items()})

def sample_uniform(N, rng):
    return rng.choice(n_t, size=N, replace=False)

def sample_shap_guided(N, rng):
    d = np.array([div.get(q, 1.0) for q in wealth])
    p = d / d.sum()
    return rng.choice(n_t, size=N, replace=False, p=p)

def fine_tune_and_eval(idx):
    Xn, yn = X_t[idx], y_t[idx]
    bst_n = xgb.train(PARAMS, xgb.DMatrix(Xn, label=yn), num_boost_round=60, xgb_model=bst)
    mask = np.ones(n_t, bool); mask[idx] = False
    return roc_auc_score(y_t[mask], bst_n.predict(xgb.DMatrix(X_t[mask])))

print(f"\n{'N':>6s} {'random':>8s} {'SHAP-guided':>12s}   (held-out AUC, mean over 8 seeds)")
for N in [100, 250, 500, 1000]:
    r_rand, r_guided = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        r_rand.append(fine_tune_and_eval(sample_uniform(N, rng)))
        r_guided.append(fine_tune_and_eval(sample_shap_guided(N, rng)))
    print(f"{N:6d} {np.mean(r_rand):8.3f} {np.mean(r_guided):12.3f}   "
          f"(+{np.mean(r_guided)-np.mean(r_rand):+.3f})")
