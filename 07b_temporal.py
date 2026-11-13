"""
Project 1 (b) — Temporal axis: does the same OOD-detector blindness hold for
WITHIN-COUNTRY time shift (not just cross-border)?

Bangladesh has 3 usable rounds: BD72 (2014) -> BD7R (2017-18) -> BD81 (2022).
For each ordered (earlier -> later) pair we measure:
  - label shift (U5MR), transfer AUC, oracle AUC, concept-drift gap
  - OOD detectors (Mahalanobis / IsolationForest / kNN) detect_AUC
  - conformal coverage shortfall
and ask: do the detectors flag the later round as OOD, and does that track the
actual concept drift?
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

def make_model(seed=0):
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=seed, eval_metric="auc")

def cv_auc(df):
    X, y = df[UNIVERSAL].values, df["y"].values
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        m = make_model(); m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

def detect_aucs(X_id_fit, X_id_eval, X_ood):
    imp = SimpleImputer(strategy="median").fit(X_id_fit)
    A = imp.transform(X_id_eval); B = imp.transform(X_ood)
    F = imp.transform(X_id_fit)
    out = {}
    lw = LedoitWolf().fit(F); d = lambda Z: np.einsum("ij,jk,ik->i", Z - lw.location_, lw.precision_, Z - lw.location_)
    out["mahalanobis"] = _auc(d(A), d(B))
    iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=8).fit(F)
    out["isolation_forest"] = _auc(-iso.score_samples(A), -iso.score_samples(B))
    nn = NearestNeighbors(n_neighbors=5, n_jobs=8).fit(F)
    out["knn"] = _auc(nn.kneighbors(A)[0][:, -1], nn.kneighbors(B)[0][:, -1])
    return out

def _auc(a, b):
    s = np.concatenate([a, b]); y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    return roc_auc_score(y, s)

ROUNDS = [("BD72", 2014), ("BD7R", 2017), ("BD81", 2022)]
print("Bangladesh temporal rounds: BD72(2014) -> BD7R(2017) -> BD81(2022)\n")

for i in range(len(ROUNDS) - 1):
    for j in range(i + 1, len(ROUNDS)):
        e, ey = ROUNDS[i]; l, ly = ROUNDS[j]
        dfe, dfl = load(e), load(l)
        Xe, ye = dfe[UNIVERSAL].values, dfe["y"].values
        Xl, yl = dfl[UNIVERSAL].values, dfl["y"].values
        # transfer + oracle
        m = make_model(); m.fit(Xe, ye)
        transfer = roc_auc_score(yl, m.predict_proba(Xl)[:, 1])
        oracle = cv_auc(dfl)
        gap = oracle - transfer
        # detectors
        Xf, Xid = train_test_split(Xe, test_size=0.3, stratify=ye, random_state=0)
        det = detect_aucs(Xf, Xid, Xl)
        # conformal coverage shortfall
        Xtr, Xcal, ytr, ycal = train_test_split(Xe, ye, test_size=0.3, stratify=ye, random_state=0)
        mm = make_model(); mm.fit(Xtr, ytr)
        p_cal = mm.predict_proba(Xcal)[:, 1]
        s_cal = np.where(ycal == 1, 1 - p_cal, p_cal)
        q = np.sort(s_cal)[int(np.ceil((len(s_cal) + 1) * 0.9)) - 1]
        p_l = mm.predict_proba(Xl)[:, 1]
        cov = float((np.where(yl == 1, 1 - p_l, p_l) <= q).mean())
        short = 0.9 - cov
        print(f"{e}({ey}) -> {l}({ly}):")
        print(f"    U5MR {ye.mean()*1000:.0f}->{yl.mean()*1000:.0f}/1000 | "
              f"transfer AUC={transfer:.3f}  oracle AUC={oracle:.3f}  "
              f"concept-drift gap={gap:+.3f}")
        print(f"    detectors: " + "  ".join(f"{k}={v:.2f}" for k, v in det.items()) +
              f"  | conformal coverage shortfall={short:+.3f}")
