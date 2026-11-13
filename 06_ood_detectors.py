"""
Project 1 (c) — OOD-detector suite.

Question: do *actual* OOD detectors (Mahalanobis, Isolation Forest, k-NN,
GBM-confidence) (i) flag a shifted target as OOD (they should, since covariate
shift is large), but (ii) fail to localize/predict the *harmful* drift?

Part A (flagship pairs): deep-dive. For Kenya->Nigeria (and 2 others), report
  - detector AUROC for "ID (source) vs OOD (target)"  -> detection works
  - per-sample Spearman rho(OOD score, |y - p|) on target -> localization fails
Part B (592 pairs): for 8 sources x all targets, correlate each detector's
  OOD-detectability (AUROC) with the concept-drift gap (oracle - transfer).
"""
import os, glob, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
RNG = 0

def make_model():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

def load(key):
    return pd.read_parquet(f"{CACHE}/{key}.parquet")

def fit_imputer(X):
    return SimpleImputer(strategy="median").fit(X)

def mahalanobis_fit(X):
    lw = LedoitWolf().fit(X)
    return lw.location_, lw.precision_
def mahalanobis_score(mu, prec, X):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)

class Detectors:
    def __init__(self, X_fit):
        self.imp = fit_imputer(X_fit)
        Xf = self.imp.transform(X_fit)
        self.mu, self.prec = mahalanobis_fit(Xf)
        self.ifor = IsolationForest(n_estimators=200, random_state=RNG, n_jobs=8).fit(Xf)
        self.knn = NearestNeighbors(n_neighbors=5, n_jobs=8).fit(Xf)
    def scores(self, X):
        Xf = self.imp.transform(X)
        out = {}
        out["mahalanobis"] = mahalanobis_score(self.mu, self.prec, Xf)
        out["isolation_forest"] = -self.ifor.score_samples(Xf)
        d, _ = self.knn.kneighbors(Xf)
        out["knn"] = d[:, -1]
        return out

def gbm_conf(model, X):
    p = model.predict_proba(X)[:, 1]
    return 1.0 - np.maximum(p, 1 - p)   # higher = less confident = OOD

def det_auc(s_id, s_ood):
    y = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
    s = np.concatenate([s_id, s_ood])
    return roc_auc_score(y, s)

# ---------------- Part A: flagship pairs ----------------
print("=== Part A: flagship pairs (deep-dive) ===")
FLAGSHIP = [("KE", "91", "NG", "8B"), ("NG", "8B", "KE", "91"),
            ("KE", "91", "BO", "51"), ("KE", "91", "RW", "91")]
partA_rows = []
for s, sp, t, tp in FLAGSHIP:
    dfs, dft = load(f"{s}{sp}"), load(f"{t}{tp}")
    X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
    X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
    # train mortality model on source
    model = make_model(); model.fit(X_s, y_s)
    # split source: fit detectors on 70%, ID scores on held-out 30%
    Xf, Xid, _, _ = train_test_split(X_s, y_s, test_size=0.3,
                                     stratify=y_s, random_state=RNG)
    det = Detectors(Xf)
    s_id = det.scores(Xid)
    s_ood = det.scores(X_t)
    s_ood["gbm_confidence"] = gbm_conf(model, X_t)
    s_id["gbm_confidence"] = gbm_conf(model, Xid)
    p_t = model.predict_proba(X_t)[:, 1]
    err = np.abs(y_t - p_t)
    print(f"\n  {s}{sp} -> {t}{tp}  (n_id={len(Xid)}, n_ood={len(X_t)})")
    for name in ["mahalanobis", "isolation_forest", "knn", "gbm_confidence"]:
        auc = det_auc(s_id[name], s_ood[name])
        rho, _ = spearmanr(s_ood[name], err)
        partA_rows.append({"pair": f"{s}{sp}->{t}{tp}", "detector": name,
                           "detect_auc": auc, "err_spearman": rho})
        print(f"    {name:18s} detect_AUC={auc:.3f}  "
              f"Spearman(OODscore, |y-p|)={rho:+.3f}")
partA = pd.DataFrame(partA_rows)
partA.to_parquet("ood_detectors_partA.parquet")

# ---------------- Part B: 592-pair replication ----------------
print("\n=== Part B: 592-pair replication (OOD-detectability vs concept drift) ===")
gap = pd.read_parquet("concept_drift_gap.parquet")
keys = list(gap.index)
sources = [("KE","91"), ("NG","8B"), ("BO","51"), ("EG","61"),
           ("ID","71"), ("PH","82"), ("RW","91"), ("HT","71")]
sources = [f"{c}{p}" for c, p in sources if f"{c}{p}" in keys]

rows = []
for s in sources:
    dfs = load(s)
    X_s = dfs[UNIVERSAL].values
    y_s = dfs["y"].values
    Xf, Xid = train_test_split(X_s, test_size=0.3, stratify=y_s, random_state=RNG)
    det = Detectors(Xf)
    s_id = det.scores(Xid)
    for t in keys:
        if t == s:
            continue
        X_t = load(t)[UNIVERSAL].values
        s_ood = det.scores(X_t)
        for name in ["mahalanobis", "isolation_forest", "knn"]:
            auc = det_auc(s_id[name], s_ood[name])
            rows.append({"source": s, "target": t, "detector": name,
                         "detect_auc": auc,
                         "concept_drift_gap": gap.loc[s, t]})
    print(f"  source {s}: done ({len(keys)-1} targets)", flush=True)

ood = pd.DataFrame(rows)
ood.to_parquet("ood_detectors_partB.parquet")

# correlation per detector
print("\n  Correlation  r(OOD-detect_AUC, concept_drift_gap)  per detector:")
for name in ["mahalanobis", "isolation_forest", "knn"]:
    sub = ood[ood.detector == name]
    r = sub.detect_auc.corr(sub.concept_drift_gap)
    print(f"    {name:18s} r={r:.3f}  (n={len(sub)})")

# ---------------- figures ----------------
# Fig 8: detector AUROCs on flagship pairs
fig, ax = plt.subplots(figsize=(7, 4.5))
piv = partA.pivot_table(index="pair", columns="detector", values="detect_auc")
piv = piv[["mahalanobis", "isolation_forest", "knn", "gbm_confidence"]]
piv.plot(kind="bar", ax=ax, width=0.8)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_ylabel("AUROC (ID vs OOD)")
ax.set_ylim(0, 1)
ax.set_title("OOD detectors DO flag the shifted target (covariate shift)")
ax.legend(title="detector", fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig8_detectors_detect_shift.png", dpi=130); plt.close()

# Fig 9: detector OOD-detectability vs concept-drift gap (all detectors)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, name in zip(axes, ["mahalanobis", "isolation_forest", "knn"]):
    sub = ood[ood.detector == name]
    r = sub.detect_auc.corr(sub.concept_drift_gap)
    ax.scatter(sub.detect_auc, sub.concept_drift_gap, s=10, alpha=0.4, color="#4C72B0")
    ax.set_xlabel("OOD-detectability (AUROC)")
    ax.set_title(f"{name}\nr = {r:.2f}")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xlim(0.4, 1.0)
axes[0].set_ylabel("Concept-drift gap (oracle - transfer)")
fig.suptitle("OOD detectors fail to predict model failure (concept drift)", y=1.02)
plt.tight_layout(); plt.savefig("figs/fig9_detectors_miss_drift.png", dpi=130); plt.close()

summary = {
    "flagship": partA.to_dict(orient="records"),
    "partB_corr": {
        name: round(float(ood[ood.detector == name].detect_auc.corr(
            ood[ood.detector == name].concept_drift_gap)), 3)
        for name in ["mahalanobis", "isolation_forest", "knn"]},
}
with open("results_ood_detectors.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDone. Saved ood_detectors_partA/B.parquet, results_ood_detectors.json, figs 8-9.")
