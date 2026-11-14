"""
Project 1 (revision) — expanded longitudinal analysis (weighted).

Two multi-round countries exist in the data: Bangladesh (2007-2022) and Armenia
(2000-2016). For each ordered (earlier -> later) pair we measure weighted label
shift (U5MR), weighted transfer AUC, concept-drift gap, covariate-shift
detectability (Mahalanobis), and conformal coverage shortfall.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=3,
              learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=15, tree_method="hist", n_jobs=8, seed=0)

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

def weighted_auc(y, s, w):
    y = np.asarray(y); s = np.asarray(s); w = np.asarray(w, dtype=float)
    o = np.argsort(s); y, s, w = y[o], s[o], w[o]
    Wp = w[y == 1].sum(); Wn = w[y == 0].sum()
    if Wp == 0 or Wn == 0: return float("nan")
    tot, rn, i, n = 0.0, 0.0, 0, len(y)
    while i < n:
        j = i
        while j < n and s[j] == s[i]: j += 1
        wn = w[i:j][y[i:j] == 0].sum(); wp = w[i:j][y[i:j] == 1].sum()
        tot += wp * (rn + 0.5 * wn); rn += wn; i = j
    return float(tot / (Wp * Wn))

def w_u5mr(df):
    y, w = df["y"].values, df["weight"].values
    return float(np.sum(w * y) / np.sum(w) * 1000)

def w_cv_auc(df):
    X, y, w = df[UNIVERSAL].values, df["y"].values, df["weight"].values
    a = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        m = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=y[tr], weight=w[tr]), num_boost_round=200)
        a.append(weighted_auc(y[te], m.predict(xgb.DMatrix(X[te])), w[te]))
    return float(np.mean(a))

def mahalanobis_auc(X_s, X_t):
    imp = SimpleImputer(strategy="median").fit(X_s)
    lw = LedoitWolf().fit(imp.transform(X_s))
    mu, prec = lw.location_, lw.precision_
    def d(X): z = imp.transform(X) - mu; return np.einsum("ij,jk,ik->i", z, prec, z)
    s = np.concatenate([d(X_s), d(X_t)])
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))])
    return roc_auc_score(y, s)

ROUNDS = [("BD72", 2014), ("BD7R", 2017), ("BD81", 2022),
          ("AM42", 2000), ("AM54", 2005), ("AM61", 2010), ("AM72", 2016)]
GROUPS = {"Bangladesh": ["BD72", "BD7R", "BD81"],
          "Armenia": ["AM42", "AM54", "AM61", "AM72"]}
YEAR = {k: y for k, y in ROUNDS}

for country, keys in GROUPS.items():
    print(f"\n=== {country} ===")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            e, l = keys[i], keys[j]
            dfe, dfl = load(e), load(l)
            Xe, ye, we = dfe[UNIVERSAL].values, dfe["y"].values, dfe["weight"].values
            Xl, yl, wl = dfl[UNIVERSAL].values, dfl["y"].values, dfl["weight"].values
            m = xgb.train(PARAMS, xgb.DMatrix(Xe, label=ye, weight=we), num_boost_round=200)
            transfer = weighted_auc(yl, m.predict(xgb.DMatrix(Xl)), wl)
            oracle = w_cv_auc(dfl)
            gap = oracle - transfer
            da = mahalanobis_auc(Xe, Xl)
            # conformal coverage shortfall (weighted)
            Xtr, Xcal, ytr, ycal, wtr, wcal = train_test_split(
                Xe, ye, we, test_size=0.3, stratify=ye, random_state=0)
            mc = xgb.train(PARAMS, xgb.DMatrix(Xtr, label=ytr, weight=wtr), num_boost_round=200)
            p_cal = mc.predict(xgb.DMatrix(Xcal))
            s_cal = np.where(ycal == 1, 1 - p_cal, p_cal)
            q = np.sort(s_cal)[int(np.ceil((len(s_cal) + 1) * 0.9)) - 1]
            p_l = mc.predict(xgb.DMatrix(Xl))
            cov = float(np.sum(wl * (np.where(yl == 1, 1 - p_l, p_l) <= q)) / np.sum(wl))
            short = 0.9 - cov
            print(f"  {e}({YEAR[e]}) -> {l}({YEAR[l]}): U5MR {w_u5mr(dfe):.0f}->{w_u5mr(dfl):.0f}/1000 | "
                  f"transfer={transfer:.3f} oracle={oracle:.3f} gap={gap:+.3f} | "
                  f"covariate-shift detect={da:.2f} conformal-short={short:+.3f}")
