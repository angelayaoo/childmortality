"""Figures for the two winning playbook methods: few-shot recovery + sentinel."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_univ"
PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=3,
              learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=15, tree_method="hist", n_jobs=8, seed=0)

def load(k): return pd.read_parquet(f"{CACHE}/{k}.parquet")

# ---- Fig 14: few-shot recovery curve (KE -> NG) ----
dfs, dft = load("KE91"), load("NG8B")
X_s, y_s = dfs[UNIVERSAL].values, dfs["y"].values
X_t, y_t = dft[UNIVERSAL].values, dft["y"].values
n_t = len(y_t)
bst = xgb.train(PARAMS, xgb.DMatrix(X_s, label=y_s), num_boost_round=200)
auc_src = roc_auc_score(y_t, bst.predict(xgb.DMatrix(X_t)))
orac = []
for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X_t, y_t):
    m = xgb.train(PARAMS, xgb.DMatrix(X_t[tr], label=y_t[tr]), num_boost_round=200)
    orac.append(roc_auc_score(y_t[te], m.predict(xgb.DMatrix(X_t[te]))))
oracle = float(np.mean(orac))

N_LIST = [50, 100, 250, 500, 1000, 2000, 5000]
warm, cold = [], []
rng = np.random.default_rng(0)
for n in N_LIST:
    w_, c_ = [], []
    for seed in range(5):
        idx = rng.choice(n_t, size=n, replace=False)
        dn = xgb.DMatrix(X_t[idx], label=y_t[idx])
        mw = xgb.train(PARAMS, dn, num_boost_round=40, xgb_model=bst)
        mc = xgb.train(PARAMS, dn, num_boost_round=200)
        mask = np.ones(n_t, bool); mask[idx] = False
        w_.append(roc_auc_score(y_t[mask], mw.predict(xgb.DMatrix(X_t[mask]))))
        c_.append(roc_auc_score(y_t[mask], mc.predict(xgb.DMatrix(X_t[mask]))))
    warm.append(np.mean(w_)); cold.append(np.mean(c_))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(N_LIST, warm, "o-", color="#2C7BB6", label="few-shot warm-start (tree continuation)")
ax.plot(N_LIST, cold, "s--", color="#999", label="cold start (fresh on N)")
ax.axhline(auc_src, ls=":", c="gray", label=f"source as-is (AUC={auc_src:.3f})")
ax.axhline(oracle, ls="--", c="#D7191C", label=f"full retrain oracle (AUC={oracle:.3f})")
ax.set_xscale("log"); ax.set_xticks(N_LIST); ax.set_xticklabels([str(n) for n in N_LIST])
ax.set_xlabel("Number of local target labels (log scale)")
ax.set_ylabel("AUC on held-out target")
ax.set_title("Few-shot adaptation recovers ~80% of the drift gap with ~2000 labels")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig14_fewshot_recovery.png", dpi=140); plt.close()

# ---- Fig 15: sentinel convergence (KE -> NG) ----
p_t = bst.predict(xgb.DMatrix(X_t))
true_b = float(np.mean((p_t - y_t) ** 2))
Ns = [25, 50, 100, 200, 500, 1000]
maes = []
rng = np.random.default_rng(0)
for N in Ns:
    ests = []
    for _ in range(50):
        idx = rng.choice(n_t, size=N, replace=False)
        ests.append(float(np.mean((p_t[idx] - y_t[idx]) ** 2)))
    maes.append(np.mean(np.abs(np.array(ests) - true_b)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns, maes, "o-", color="#4C72B0")
ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("Sentinel cohort size (labels)")
ax.set_ylabel("Mean absolute error of Brier estimate")
ax.set_title("~200 verified labels suffice to monitor deployment error")
plt.tight_layout(); plt.savefig("figs/fig15_sentinel_convergence.png", dpi=140); plt.close()
print("saved fig14, fig15")
