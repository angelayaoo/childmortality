"""
Project 1 (revision) — censoring-correct re-run, reframed question:
"Do input-shift alarms predict cross-survey performance loss?"

Outcome: infant mortality (death before 12 months) among children with full
12-month exposure. Weighted throughout (sample weight + weighted AUC).

Reports:
  1. weighted transfer matrix (75 surveys, >=30 infant deaths)
  2. for 8 source surveys x all targets: input-shift detectability (Mahalanobis,
     domain classifier) vs. transfer loss (oracle - transfer)
  3. does input-shift detectability identify *high-loss* transfers? (correlation
     + AUROC of a high-loss flag + precision at top-k)
"""
import glob, os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer

UNIVERSAL = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
             "edu", "edu_years"]
CACHE = "cache_infant"
SOURCES = ["KE91", "NG8B", "BO51", "EG61", "ID71", "PH82", "RW91", "HT71"]
RNG = 0

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

def make_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

keys = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{CACHE}/*.parquet"))
keys = [k for k in keys if int(load(k)["y"].sum()) >= 30]
print(f"{len(keys)} surveys with >=30 infant deaths", flush=True)

def w_cv_auc(k):
    d = load(k); X, y, w = d[UNIVERSAL].values, d["y"].values, d["weight"].values
    a = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=RNG).split(X, y):
        m = make_model(); m.fit(X[tr], y[tr], sample_weight=w[tr])
        a.append(weighted_auc(y[te], m.predict_proba(X[te])[:, 1], w[te]))
    return float(np.mean(a))

# weighted in-domain CV
diag = {}
for i, k in enumerate(keys):
    diag[k] = w_cv_auc(k)
    if (i + 1) % 25 == 0: print(f"  CV {i+1}/{len(keys)}", flush=True)

# transfer matrix
N = len(keys); matrix = np.full((N, N), np.nan); models = {}
for i, s in enumerate(keys):
    ds = load(s); Xs, ys, ws = ds[UNIVERSAL].values, ds["y"].values, ds["weight"].values
    m = make_model(); m.fit(Xs, ys, sample_weight=ws); models[s] = m
    for j, t in enumerate(keys):
        if s == t:
            matrix[i, j] = diag[s]
        else:
            dt = load(t); Xt, yt, wt = dt[UNIVERSAL].values, dt["y"].values, dt["weight"].values
            matrix[i, j] = weighted_auc(yt, m.predict_proba(Xt)[:, 1], wt)
    if (i + 1) % 25 == 0: print(f"  transfer {i+1}/{N}", flush=True)

in_dom = np.array([diag[k] for k in keys])
off = np.array([matrix[i, j] for i in range(N) for j in range(N) if i != j])
loss = np.array([diag[keys[j]] - matrix[i, j] for i in range(N) for j in range(N) if i != j])
print("\n=== weighted infant-mortality transfer ===")
print(f"in-domain AUC: mean={in_dom.mean():.3f} median={np.median(in_dom):.3f}")
print(f"transfer AUC:  mean={off.mean():.3f} median={np.median(off):.3f}")
print(f"transfer loss: mean={loss.mean():.3f} median={np.median(loss):.3f}  "
      f"pct>0={(loss>0).mean()*100:.0f}%  pct>0.05={(loss>0.05).mean()*100:.1f}%")

# ---- detector analysis: 8 sources x all targets ----
def mahalanobis_auc(X_s, X_t):
    imp = SimpleImputer(strategy="median").fit(X_s)
    lw = LedoitWolf().fit(imp.transform(X_s))
    mu, prec = lw.location_, lw.precision_
    def d(X): z = imp.transform(X) - mu; return np.einsum("ij,jk,ik->i", z, prec, z)
    s = np.concatenate([d(X_s), d(X_t)])
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))])
    return roc_auc_score(y, s)

def domain_auc(X_s, X_t):
    imp = SimpleImputer(strategy="median").fit(X_s)
    X = imp.transform(np.vstack([X_s, X_t]))
    y = np.concatenate([np.ones(len(X_s)), np.zeros(len(X_t))])
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=RNG).split(X, y):
        m = xgb.XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist",
                              n_jobs=8, random_state=RNG, eval_metric="auc")
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

rows = []
for s in SOURCES:
    ds = load(s); Xs = ds[UNIVERSAL].values
    for t in keys:
        if t == s: continue
        dt = load(t); Xt = dt[UNIVERSAL].values
        ma = mahalanobis_auc(Xs, Xt)
        da = domain_auc(Xs, Xt)
        rows.append({"source": s, "target": t, "mahal": ma, "domain": da,
                     "loss": diag[t] - matrix[keys.index(s), keys.index(t)]})
    print(f"  detectors {s}: done", flush=True)

res = pd.DataFrame(rows)
res.to_parquet("detector_loss_infant.parquet")
r_mahal = res["mahal"].corr(res["loss"])
r_domain = res["domain"].corr(res["loss"])
print(f"\n=== input-shift vs transfer loss (n={len(res)} pairs) ===")
print(f"r(Mahalanobis detect, loss) = {r_mahal:.3f}")
print(f"r(domain-classifier detect, loss) = {r_domain:.3f}")

# high-loss detection: can the alarm flag transfers with loss > threshold?
for thresh in [0.03, 0.05]:
    hl = (res["loss"] > thresh).astype(int)
    if hl.sum() < 5 or hl.sum() > len(hl) - 5:
        print(f"  loss>{thresh}: {hl.sum()} positives, skipping AUROC")
        continue
    a_m = roc_auc_score(hl, res["mahal"])
    a_d = roc_auc_score(hl, res["domain"])
    print(f"  loss>{thresh}: n={hl.sum()}  AUROC(Mahal)={a_m:.2f}  AUROC(domain)={a_d:.2f}")

# precision@top-20% of alarm
k = max(1, int(0.2 * len(res)))
top = res.nlargest(k, "mahal")
hl_top = (top["loss"] > 0.05).mean()
print(f"  precision@top-20% Mahalanobis alarm (loss>0.05): {hl_top:.2f}  (baseline {(res['loss']>0.05).mean():.2f})")

summary = {
    "n_surveys": N, "n_pairs": int(len(res)),
    "in_domain_mean": round(float(in_dom.mean()), 4),
    "transfer_mean": round(float(off.mean()), 4),
    "loss_mean": round(float(loss.mean()), 4),
    "pct_loss_gt_005": round(float((loss > 0.05).mean() * 100), 1),
    "r_mahal_vs_loss": round(float(r_mahal), 3),
    "r_domain_vs_loss": round(float(r_domain), 3),
}
with open("results_infant_transfer.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved results_infant_transfer.json")
