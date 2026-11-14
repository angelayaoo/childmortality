"""
Project 1 (revision) — diversify architecture: XGBoost vs logistic regression vs MLP.

Tests whether the "OOD detectors miss concept drift" finding is fundamental to
tabular health data or an artifact of gradient-boosted trees. For each of three
architectures we compute (weighted) transfer AUC and the correlation between
feature-space OOD detectability (Mahalanobis) and the concept-drift gap.
"""
import glob, os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
SOURCES = ["KE91", "NG8B", "BO51", "EG61", "ID71", "PH82", "RW91", "HT71"]
RNG = 0

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

def weighted_auc(y, s, w):
    y = np.asarray(y); s = np.asarray(s); w = np.asarray(w, dtype=float)
    order = np.argsort(s); y, s, w = y[order], s[order], w[order]
    Wp = w[y == 1].sum(); Wn = w[y == 0].sum()
    if Wp == 0 or Wn == 0: return float("nan")
    total, run_neg, i, n = 0.0, 0.0, 0, len(y)
    while i < n:
        j = i
        while j < n and s[j] == s[i]: j += 1
        wn = w[i:j][y[i:j] == 0].sum(); wp = w[i:j][y[i:j] == 1].sum()
        total += wp * (run_neg + 0.5 * wn); run_neg += wn; i = j
    return float(total / (Wp * Wn))

def make_xgb():
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

def make_lr():
    return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RNG)

def make_mlp():
    # sklearn MLPClassifier does not accept sample_weight in this version; trained unweighted
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         solver="adam", max_iter=200, random_state=RNG, early_stopping=False)

def fit_predict(name, model, Xtr, ytr, wtr, Xte):
    if name == "xgb":
        model.fit(Xtr, ytr, sample_weight=wtr)
    elif name == "lr":
        model.fit(Xtr, ytr, sample_weight=wtr)
    else:  # mlp (unweighted)
        model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]

def mahalanobis_auc(X_s, X_t):
    imp = SimpleImputer(strategy="median").fit(X_s)
    lw = LedoitWolf().fit(imp.transform(X_s))
    mu, prec = lw.location_, lw.precision_
    def d(X): z = imp.transform(X) - mu; return np.einsum("ij,jk,ik->i", z, prec, z)
    s = np.concatenate([d(X_s), d(X_t)])
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))])
    return roc_auc_score(y, s)

keys = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{CACHE}/*.parquet"))
keys = [k for k in keys if int(pd.read_parquet(f"{CACHE}/{k}.parquet", columns=["y"])["y"].sum()) >= 50]

results = {}
for name, mk in [("xgb", make_xgb), ("lr", make_lr), ("mlp", make_mlp)]:
    rows = []
    for s in SOURCES:
        dfs = load(s)
        X_s, y_s, w_s = dfs[UNIVERSAL].values, dfs["y"].values, dfs["weight"].values
        # impute + scale for lr/mlp (XGBoost handles NaN natively)
        if name in ("lr", "mlp"):
            imp = SimpleImputer(strategy="median").fit(X_s)
            sc = StandardScaler().fit(imp.transform(X_s))
            def prep(X): return sc.transform(imp.transform(X))
            X_s_s = prep(X_s)
        else:
            X_s_s = X_s
            def prep(X): return X
        for t in keys:
            if t == s: continue
            dft = load(t)
            X_t, y_t, w_t = dft[UNIVERSAL].values, dft["y"].values, dft["weight"].values
            X_t_s = prep(X_t)
            model = mk()
            p = fit_predict(name, model, X_s_s, y_s, w_s, X_t_s)
            transfer = weighted_auc(y_t, p, w_t)
            gap = None  # filled below once diag known
            da = mahalanobis_auc(X_s, X_t)
            rows.append({"source": s, "target": t, "transfer": transfer,
                         "detect_auc": da, "diag_target": None})
        print(f"  [{name}] {s}: done", flush=True)
    # weighted in-domain CV for targets (3-fold) to get oracle -> gap
    diag = {}
    for t in keys:
        dft = load(t)
        X, y, w = dft[UNIVERSAL].values, dft["y"].values, dft["weight"].values
        aucs = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=RNG).split(X, y):
            if name in ("lr", "mlp"):
                imp = SimpleImputer(strategy="median").fit(X[tr])
                sc = StandardScaler().fit(imp.transform(X[tr]))
                Xtr = sc.transform(imp.transform(X[tr]))
                Xte = sc.transform(imp.transform(X[te]))
            else:
                Xtr, Xte = X[tr], X[te]
            m = mk()
            if name == "xgb": m.fit(Xtr, y[tr], sample_weight=w[tr])
            elif name == "lr": m.fit(Xtr, y[tr], sample_weight=w[tr])
            else: m.fit(Xtr, y[tr])
            aucs.append(weighted_auc(y[te], m.predict_proba(Xte)[:, 1], w[te]))
        diag[t] = float(np.mean(aucs))
    for r in rows:
        r["gap"] = diag[r["target"]] - r["transfer"]
    res = pd.DataFrame(rows)
    res.to_parquet(f"arch_{name}.parquet")
    r_auc = res["detect_auc"].corr(res["gap"])
    results[name] = {
        "n_pairs": int(len(res)),
        "mean_transfer": round(float(res["transfer"].mean()), 4),
        "mean_gap": round(float(res["gap"].mean()), 4),
        "r_detect_vs_gap": round(float(r_auc), 3),
    }
    print(f"\n=== {name}: mean transfer={res['transfer'].mean():.3f}  "
          f"mean gap={res['gap'].mean():+.3f}  "
          f"r(Mahalanobis detect_AUC, gap)={r_auc:.3f}\n", flush=True)

with open("results_architectures.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results_architectures.json")
