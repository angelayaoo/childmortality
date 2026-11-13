"""
Project 1 (playbook) — Method 2: Few-Shot Tree Updating.

Can we recover most of the concept-drift gap by continuing to boost the source
XGBoost model on a tiny handful of local target rows (instead of full retraining)?

For each (source -> target) pair we compare, as a function of the number N of
target labels used:
  - source model deployed as-is (N=0)
  - source model + label-shift recalibration (prior-rescale, 0 labels beyond base rate)
  - COLD start: fresh XGBoost trained on N samples
  - WARM start: xgb.train(..., xgb_model=source_booster) continued on N samples
  - oracle: full target retrain (upper bound)
and report AUC on the held-out target + % of the drift gap recovered.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=3,
              learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=15, tree_method="hist", n_jobs=8, seed=0)
N_BOOST = 200
EXTRA = 40
N_LIST = [50, 100, 250, 500, 1000, 2000, 5000]
SEEDS = 5

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

def evaluate(pairs):
    print(f"{'pair':12s} {'N=0(src)':>9s} {'recal':>7s} " +
          "".join(f"{'c'+str(n):>8s}" for n in N_LIST) +
          "".join(f"{'w'+str(n):>8s}" for n in N_LIST) + "  oracle")
    for s, t in pairs:
        dfs, dft = load(s), load(t)
        X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
        X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
        n_t = len(y_t)

        # source model
        dtrain_s = xgb.DMatrix(X_s, label=y_s)
        bst = xgb.train(PARAMS, dtrain_s, num_boost_round=N_BOOST)
        p_src = bst.predict(xgb.DMatrix(X_t))
        auc_src = roc_auc_score(y_t, p_src)

        # label-shift recalibration (rank-preserving prior rescale)
        p_recal = np.clip(p_src * (y_t.mean() / max(p_src.mean(), 1e-12)), 0, 1)
        auc_recal = roc_auc_score(y_t, p_recal)   # should == auc_src

        # oracle: full target retrain (5-fold CV estimate)
        from sklearn.model_selection import StratifiedKFold
        orac = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X_t, y_t):
            m = xgb.train(PARAMS, xgb.DMatrix(X_t[tr], label=y_t[tr]), num_boost_round=N_BOOST)
            orac.append(roc_auc_score(y_t[te], m.predict(xgb.DMatrix(X_t[te]))))
        oracle = float(np.mean(orac))

        cold = {n: [] for n in N_LIST}
        warm = {n: [] for n in N_LIST}
        rng = np.random.default_rng(0)
        for n in N_LIST:
            for seed in range(SEEDS):
                idx = rng.choice(n_t, size=min(n, n_t), replace=False)
                Xn, yn = X_t[idx], y_t[idx]
                dn = xgb.DMatrix(Xn, label=yn)
                # cold start
                m_cold = xgb.train(PARAMS, dn, num_boost_round=N_BOOST)
                # warm start
                m_warm = xgb.train(PARAMS, dn, num_boost_round=EXTRA, xgb_model=bst)
                # held-out target
                mask = np.ones(n_t, bool); mask[idx] = False
                Xh, yh = X_t[mask], y_t[mask]
                cold[n].append(roc_auc_score(yh, m_cold.predict(xgb.DMatrix(Xh))))
                warm[n].append(roc_auc_score(yh, m_warm.predict(xgb.DMatrix(Xh))))
        cold_m = {n: float(np.mean(cold[n])) for n in N_LIST}
        warm_m = {n: float(np.mean(warm[n])) for n in N_LIST}

        row = f"{s}->{t}   {auc_src:9.3f} {auc_recal:7.3f} "
        row += "".join(f"{cold_m[n]:8.3f}" for n in N_LIST)
        row += "".join(f"{warm_m[n]:8.3f}" for n in N_LIST)
        row += f"  {oracle:.3f}"
        print(row)
        # % gap recovered by warm start at N=500
        gap = oracle - auc_src
        rec500 = (warm_m[500] - auc_src) / gap * 100 if gap > 0 else float("nan")
        print(f"    -> gap={gap:.3f}  warm-500 recovers {rec500:.0f}%  "
              f"cold-500 recovers {(cold_m[500]-auc_src)/gap*100 if gap>0 else 0:.0f}%")

evaluate([("KE91", "NG8B"), ("NG8B", "KE91"), ("KE91", "BO51")])
