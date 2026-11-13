"""
Project 1 (fix) — Do predictive-uncertainty monitors beat feature-space OOD?

Feature-space detectors (Mahalanobis/IF/kNN) see covariate shift but miss concept
drift (r ~ 0.1-0.2). Here we test two *uncertainty* monitors that react to the
model's own (in)ability rather than the inputs:

  1. Conformal prediction coverage shortfall — a distribution-free coverage
     guarantee that specifically breaks under label/concept shift (not covariate
     shift), so it should track deployment failure better.
  2. Ensemble epistemic uncertainty — variance across a bagged XGBoost ensemble.

For 592 source->target pairs we correlate each monitor's "shift signal" with the
concept-drift gap (oracle - transfer) and compare against Mahalanobis (r=0.12).
"""
import os, glob, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
ALPHA = 0.1          # 90% target coverage
K_ENS = 10           # ensemble size

def make_model(seed=0):
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=seed, eval_metric="auc")

def load(key):
    return pd.read_parquet(f"{CACHE}/{key}.parquet")

def predict_ensemble(ens, X):
    S = np.zeros(len(X)); S2 = np.zeros(len(X))
    for m in ens:
        p = m.predict_proba(X)[:, 1]
        S += p; S2 += p * p
    mu = S / K_ENS
    var = np.maximum(S2 / K_ENS - mu ** 2, 0)   # epistemic (across-model) variance
    return mu, var

def mahalanobis_auc(X_s, X_t):
    from sklearn.covariance import LedoitWolf as LW
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit(X_s)
    lw = LW().fit(imp.transform(X_s))
    mu = lw.location_; prec = lw.precision_
    def d(X):
        z = imp.transform(X) - mu
        return np.einsum("ij,jk,ik->i", z, prec, z)
    s = np.concatenate([d(X_s), d(X_t)])
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))])
    return roc_auc_score(y, s)

gap = pd.read_parquet("concept_drift_gap.parquet")
keys = list(gap.index)
SOURCES = [k for k in ["KE91", "NG8B", "BO51", "EG61", "ID71", "PH82", "RW91", "HT71"]
           if k in keys]

rows = []
print("Computing uncertainty monitors over source->target pairs...", flush=True)
for s in SOURCES:
    dfs = load(s)
    X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
    Xtr, Xcal, ytr, ycal = train_test_split(X_s, y_s, test_size=0.3,
                                            stratify=y_s, random_state=0)
    # ensemble of K models on source-train
    ens = [make_model(seed=k) for k in range(K_ENS)]
    for m in ens:
        m.fit(Xtr, ytr)
    # conformal calibration on Xcal
    p_cal, _ = predict_ensemble(ens, Xcal)
    s_cal = np.where(ycal == 1, 1 - p_cal, p_cal)
    n = len(s_cal)
    q = np.sort(s_cal)[int(np.ceil((n + 1) * (1 - ALPHA))) - 1]
    # ID epistemic baseline (on held-out source)
    _, epi_id = predict_ensemble(ens, Xcal)
    epi_id_mean = epi_id.mean()
    base_s = y_s.mean()

    for t in keys:
        if t == s:
            continue
        dft = load(t)
        X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
        p_t, epi_t = predict_ensemble(ens, X_t)
        s_t = np.where(y_t == 1, 1 - p_t, p_t)
        coverage = float((s_t <= q).mean())
        coverage_shortfall = (1 - ALPHA) - coverage
        nonconf_mean = float(s_t.mean())
        epi_ratio = float(epi_t.mean() / max(epi_id_mean, 1e-12))
        rows.append({
            "source": s, "target": t,
            "concept_drift_gap": gap.loc[s, t],
            "base_rate_gap": abs(base_s - y_t.mean()),
            "mahalanobis_auc": mahalanobis_auc(Xtr, X_t),
            "coverage_shortfall": coverage_shortfall,
            "mean_nonconformity": nonconf_mean,
            "epistemic_ratio": epi_ratio,
        })
    print(f"  source {s}: done", flush=True)

res = pd.DataFrame(rows)
res.to_parquet("uncertainty_monitors.parquet")

# correlations
def corr(a, b):
    return res[a].corr(res[b])
print("\n=== Correlations with concept-drift gap (n=%d pairs) ===" % len(res))
print(f"  Mahalanobis detect_AUC   : r = {corr('mahalanobis_auc','concept_drift_gap'):+.3f}")
print(f"  Conformal coverage short : r = {corr('coverage_shortfall','concept_drift_gap'):+.3f}")
print(f"  Mean nonconformity       : r = {corr('mean_nonconformity','concept_drift_gap'):+.3f}")
print(f"  Epistemic uncertainty    : r = {corr('epistemic_ratio','concept_drift_gap'):+.3f}")
print("\n=== Correlations with base-rate gap (what uncertainty actually tracks) ===")
print(f"  Conformal coverage short : r = {corr('coverage_shortfall','base_rate_gap'):+.3f}")
print(f"  Epistemic uncertainty    : r = {corr('epistemic_ratio','base_rate_gap'):+.3f}")

# figure: scatter of each monitor vs concept-drift gap
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, name in zip(axes,
        ["mahalanobis_auc", "coverage_shortfall", "epistemic_ratio"],
        ["Mahalanobis (feature-space OOD)", "Conformal coverage shortfall", "Epistemic uncertainty"]):
    r = corr(col, "concept_drift_gap")
    ax.scatter(res[col], res["concept_drift_gap"], s=10, alpha=0.4, color="#4C72B0")
    ax.set_xlabel(col)
    ax.set_title(f"{name}\nr = {r:+.2f}")
    ax.axhline(0, ls="--", c="gray", lw=1)
axes[0].set_ylabel("Concept-drift gap (oracle - transfer)")
fig.suptitle("Uncertainty monitors vs feature-space OOD for predicting failure", y=1.02)
plt.tight_layout(); plt.savefig("figs/fig10_uncertainty_monitors.png", dpi=130); plt.close()

summary = {
    "n_pairs": int(len(res)),
    "r_mahalanobis": round(corr("mahalanobis_auc", "concept_drift_gap"), 3),
    "r_conformal_coverage": round(corr("coverage_shortfall", "concept_drift_gap"), 3),
    "r_nonconformity": round(corr("mean_nonconformity", "concept_drift_gap"), 3),
    "r_epistemic": round(corr("epistemic_ratio", "concept_drift_gap"), 3),
    "r_coverage_vs_baserate": round(corr("coverage_shortfall", "base_rate_gap"), 3),
    "r_epistemic_vs_baserate": round(corr("epistemic_ratio", "base_rate_gap"), 3),
}
with open("results_uncertainty.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved uncertainty_monitors.parquet, results_uncertainty.json, fig10.")
