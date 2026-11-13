"""
Project 1 (playbook) — Method 1: Invariant Feature Pruning.

Hypothesis: a pre-deployment SHAP audit can identify features whose relationship
to the outcome is volatile across domains; pruning them lowers in-domain AUC a
little but raises the transfer floor (shrinks the concept-drift gap).

Volatility(f) = |mean|SHAP| of source model on source  -  mean|SHAP| of source
model on target *features* |  (needs source labels + target features only).
We then greedily prune the k most-volatile features and measure the trade-off.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RICH = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
        "edu", "urban", "wealth_q"]

def load_rich(c):
    df = pd.read_parquet(f"cache_children/{c}.parquet")[RICH + ["y"]].copy()
    df["prev_interval"] = df["prev_interval"].fillna(df["prev_interval"].median())
    return df

def make_model(seed=0):
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=seed, eval_metric="auc")

def cv_auc(df, feats):
    X, y = df[feats].values, df["y"].values
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        m = make_model(); m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

def transfer_auc(src, tgt, feats):
    m = make_model(); m.fit(src[feats].values, src["y"].values)
    return roc_auc_score(tgt["y"].values, m.predict_proba(tgt[feats].values)[:, 1])

for s, t in [("KE91", "NG8B"), ("KE91", "BO51")]:
    src, tgt = load_rich(s), load_rich(t)
    # volatility via source-model SHAP on source vs target features
    m = make_model(); m.fit(src[RICH].values, src["y"].values)
    ex = shap.TreeExplainer(m)
    shap_s = np.abs(ex.shap_values(src[RICH].values)).mean(0)
    shap_t = np.abs(ex.shap_values(tgt[RICH].values)).mean(0)
    volatility = np.abs(shap_s - shap_t)
    order = np.argsort(-volatility)   # most volatile first
    print(f"\n=== {s} -> {t} : invariant feature pruning ===")
    print(f"{'features':30s} in-domain  transfer  drift_gap")
    feats = RICH[:]
    for k in range(len(RICH)):   # prune 0..6 features, always leaving >=1
        if k > 0:
            feats = [f for f in feats if f != RICH[order[k-1]]]
        in_dom = cv_auc(src, feats)
        trans = transfer_auc(src, tgt, feats)
        print(f"{'+'.join(feats):34s} {in_dom:.3f}      {trans:.3f}    {trans-in_dom:+.3f}")
    print("volatility ranking (most volatile first):",
          [f"{RICH[i]}({volatility[i]:.3f})" for i in order])
