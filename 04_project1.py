"""
Project 1 — "The Generalization Cliff"
Cross-survey concept shift in child-mortality risk models.

Pipeline:
  1. Melt the IR birth-history arrays into child-level rows (one row per birth
     in the last 5 years), build a transferable feature set, target = under-5 death.
  2. Train a 4x4 transfer matrix: source model -> target AUC (diagonal = in-domain CV).
  3. Decompose the Kenya->Nigeria gap into:
       - covariate shift (per-feature PSI + domain-classifier AUC)
       - concept drift  (feature-importance drift + wealth partial-dependence flip
                         + retrain-gap = oracle - transfer)
       - label/calibration shift (Brier before/after isotonic recalibration)
"""
import os, re, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 0
COUNTRIES = {"KE": "91", "NG": "8B", "RW": "91", "BO": "51"}
CACHE = "cache_children"
FIG = "figs"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

FEATURES = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
            "edu", "urban", "wealth_q"]
MOTHER_COLS = ["v005", "v008", "v012", "v025", "v106", "v190"]
CHILD_WIDE = {"b3": "birth_cmc", "b5": "alive", "b4": "sex",
              "bord": "birth_order", "b11": "prev_interval"}

def melt_births(country, phase):
    cache = f"{CACHE}/{country}{phase}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    dta = f"extracted_ir/{country}{phase}.dta"
    r = pd.io.stata.StataReader(dta)
    cols = list(r.variable_labels().keys())
    maxn = max(int(c.split("_")[1]) for c in cols if re.match(r"^b5_\d+$", c))
    frames = []
    for i in range(1, maxn + 1):
        sfx = f"{i:02d}"
        keep = list(MOTHER_COLS)
        rename = {}
        for w, lab in CHILD_WIDE.items():
            c = f"{w}_{sfx}"
            if c in cols:
                keep.append(c)
                rename[c] = lab
        if f"b5_{sfx}" not in cols:
            continue
        sub = pd.read_stata(dta, columns=list(dict.fromkeys(keep)),
                            convert_categoricals=False)
        sub = sub.rename(columns=rename)
        frames.append(sub)
    ch = pd.concat(frames, ignore_index=True)
    ch = ch[ch["alive"].notna()].copy()
    ch["age_months"] = ch["v008"] - ch["birth_cmc"]
    ch = ch[(ch["birth_cmc"] > 0) & (ch["age_months"] >= 0) & (ch["age_months"] < 60)]
    ch = ch.copy()
    ch["y"] = (ch["alive"] == 0).astype(int)
    ch["child_male"] = (ch["sex"] == 1).astype(int)
    ch["birth_order"] = ch["birth_order"].astype(float)
    ch["prev_interval"] = ch["prev_interval"].astype(float)
    ch["mother_age_birth"] = ch["v012"] - ch["age_months"] / 12.0
    ch["edu"] = ch["v106"].astype(float)
    ch["urban"] = (ch["v025"] == 1).astype(int)
    ch["wealth_q"] = ch["v190"].astype(float)
    ch["weight"] = ch["v005"].astype(float) / 1e6
    ch["country"] = country
    ch = ch[FEATURES + ["y", "weight", "country"]]
    ch.to_parquet(cache, index=False)
    return ch

def make_model():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        tree_method="hist", n_jobs=8, random_state=RNG, eval_metric="auc")

# ---------------- 1. load all data ----------------
print("Building child-level datasets...")
DATA = {}
for c, p in COUNTRIES.items():
    df = melt_births(c, p)
    DATA[c] = df
    print(f"  {c}{p}: n={len(df):>6,}  U5MR={df['y'].mean()*1000:5.1f}/1000  "
          f"prev_int NaN%={(df['prev_interval'].isna().mean()*100):.1f}")

# ---------------- 2. transfer matrix ----------------
def in_domain_auc(df):
    X, y = df[FEATURES].values, df["y"].values
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))

print("\nComputing 4x4 transfer matrix (source model -> target AUC)...")
keys = list(COUNTRIES)
models = {}       # (source) -> fitted model
matrix = {}
for s in keys:
    Xs = DATA[s][FEATURES].values
    ys = DATA[s]["y"].values
    for t in keys:
        if s == t:
            auc = in_domain_auc(DATA[s])
        else:
            if s not in models:
                m = make_model(); m.fit(Xs, ys); models[s] = m
            Xt = DATA[t][FEATURES].values
            auc = roc_auc_score(DATA[t]["y"].values,
                                models[s].predict_proba(Xt)[:, 1])
        matrix[(s, t)] = auc
        print(f"  {s}->{t}: AUC={auc:.3f}")

M = pd.DataFrame([[matrix[(s, t)] for t in keys] for s in keys],
                 index=[f"{s} (train)" for s in keys],
                 columns=[f"{t} (test)" for t in keys])
print("\nTransfer matrix (rows=train source, cols=test target):")
print(M.round(3).to_string())

# ---------------- 3. deep decomposition: Kenya -> Nigeria ----------------
S, T = "KE", "NG"
print(f"\n=== Deep decomposition: {S} -> {T} ===")
X_s, y_s = DATA[S][FEATURES].values, DATA[S]["y"].values
X_t, y_t = DATA[T][FEATURES].values, DATA[T]["y"].values
auc_ss = in_domain_auc(DATA[S])
auc_tt = in_domain_auc(DATA[T])
m_s = models[S]
p_s_on_t = m_s.predict_proba(X_t)[:, 1]
auc_s_on_t = roc_auc_score(y_t, p_s_on_t)
concept_drift = auc_tt - auc_s_on_t   # ranking info recoverable ONLY by retraining on target
# symmetric direction
m_t_rev = make_model(); m_t_rev.fit(X_t, y_t)
auc_t_on_s = roc_auc_score(y_s, m_t_rev.predict_proba(X_s)[:, 1])
concept_drift_rev = auc_ss - auc_t_on_s
print(f"  Kenya  in-domain AUC (CV)              : {auc_ss:.3f}")
print(f"  Kenya -> Nigeria transfer AUC          : {auc_s_on_t:.3f}")
print(f"  Nigeria in-domain AUC (CV = oracle)    : {auc_tt:.3f}")
print(f"  >> CONCEPT-DRIFT GAP (oracle - transfer): {concept_drift:+.3f} AUC")
print(f"  reverse: Nigeria -> Kenya transfer AUC : {auc_t_on_s:.3f}")
print(f"  >> reverse concept-drift gap           : {concept_drift_rev:+.3f} AUC")

# --- covariate shift: per-feature PSI + domain classifier ---
def psi(src, tgt):
    """Population Stability Index between two series (10 equal-frequency bins + missing bin)."""
    def hist(s):
        s = s.copy()
        miss = s.isna()
        if miss.any():
            s = s[~miss]
            h, edges = np.histogram(s, bins=10)
        else:
            h, edges = np.histogram(s, bins=10)
        return h / h.sum(), (miss.mean(), edges)
    ps, (ms, _) = hist(src)
    pt, (mt, _) = hist(tgt)
    # missing bin
    eps = 1e-4
    ps = np.append(ps, max(ms, eps)); pt = np.append(pt, max(mt, eps))
    ps = np.clip(ps, eps, None); pt = np.clip(pt, eps, None)
    return float(np.sum((ps - pt) * np.log(ps / pt)))

print("\n  Covariate shift (Kenya vs Nigeria):")
psi_vals = {}
for f in FEATURES:
    v = psi(DATA[S][f], DATA[T][f])
    psi_vals[f] = v
    print(f"    {f:16s} PSI={v:.3f}")

# domain classifier AUC (can features tell source from target?)
dom = pd.concat([DATA[S][FEATURES].assign(src=1), DATA[T][FEATURES].assign(src=0)])
dom_auc = in_domain_auc(dom.rename(columns={"src": "y"})[FEATURES + ["y"]])
print(f"    domain-classifier AUC (covariate shift): {dom_auc:.3f}")

# --- concept drift: feature importance + wealth partial dependence ---
m_t = make_model(); m_t.fit(X_t, y_t)
imp_s = m_s.feature_importances_
imp_t = m_t.feature_importances_
print("\n  Feature-importance drift (gain):")
for f, a, b in zip(FEATURES, imp_s, imp_t):
    print(f"    {f:16s} Kenya={a:.3f}  Nigeria={b:.3f}  drift={b-a:+.3f}")

def wealth_pd(model, df):
    X = df[FEATURES].copy()
    out = {}
    for q in [1, 2, 3, 4, 5]:
        Xq = X.copy()
        Xq["wealth_q"] = q
        out[q] = float(model.predict_proba(Xq.values)[:, 1].mean())
    return out

pd_s = wealth_pd(m_s, DATA[S])
pd_t = wealth_pd(m_t, DATA[T])
print("\n  Wealth partial dependence  P(y=1 | wealth_q):")
print(f"    Kenya  : { {q: round(v*1000,1) for q,v in pd_s.items()} } (per 1000)")
print(f"    Nigeria: { {q: round(v*1000,1) for q,v in pd_t.items()} } (per 1000)")

# --- label/base-rate shift: model under-predicts target risk ---
mean_pred = float(np.mean(p_s_on_t))
label_mult = float(y_t.mean()) / mean_pred if mean_pred > 0 else float('nan')
# monotonic prior-rescaling (rank-preserving) corrects the base rate
p_rescaled = np.clip(p_s_on_t * (y_t.mean() / mean_pred), 0, 1)
brier_raw = brier_score_loss(y_t, p_s_on_t)
brier_rescaled = brier_score_loss(y_t, p_rescaled)
auc_after_rescale = roc_auc_score(y_t, p_rescaled)  # must equal auc_s_on_t (rank-invariant)
print(f"\n  Label/base-rate shift on Nigeria:")
print(f"    source base rate (Kenya)       = {y_s.mean()*1000:.1f}/1000")
print(f"    target base rate (Nigeria)     = {y_t.mean()*1000:.1f}/1000")
print(f"    mean model prediction on target= {mean_pred*1000:.1f}/1000  (under-predicts by {label_mult:.2f}x)")
print(f"    Brier (raw)                    = {brier_raw:.4f}")
print(f"    Brier (after prior-rescale)    = {brier_rescaled:.4f}")
print(f"    AUC after prior-rescale        = {auc_after_rescale:.3f} (== raw {auc_s_on_t:.3f}: ranking unchanged)")

# ---------------- figures ----------------
sns_style_ok = True
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception:
    sns_style_ok = False

# Fig 1: transfer heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(M, annot=True, fmt=".3f", cmap="viridis", vmin=0.5, vmax=0.75,
            ax=ax, cbar_kws={"label": "AUC"})
ax.set_title("Child-mortality risk model: cross-survey AUC (generalization cliff)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_transfer_matrix.png", dpi=150); plt.close()

# Fig 2: covariate shift (PSI)
fig, ax = plt.subplots(figsize=(6, 4))
names = list(psi_vals)
vals = [psi_vals[f] for f in names]
ax.bar(names, vals, color="#4C72B0")
ax.axhline(0.1, ls="--", c="gray", lw=1, label="PSI=0.1 (minor shift)")
ax.axhline(0.25, ls="--", c="red", lw=1, label="PSI=0.25 (major shift)")
ax.set_ylabel("PSI (Kenya vs Nigeria)")
ax.set_title(f"Covariate shift  |  domain-classifier AUC={dom_auc:.2f}")
ax.tick_params(axis="x", rotation=30)
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_covariate_shift.png", dpi=150); plt.close()

# Fig 3: concept drift — wealth partial dependence flip
fig, ax = plt.subplots(figsize=(6, 4))
qs = [1, 2, 3, 4, 5]
ax.plot(qs, [pd_s[q]*1000 for q in qs], "o-", label="Kenya model", color="#2C7BB6")
ax.plot(qs, [pd_t[q]*1000 for q in qs], "s-", label="Nigeria model", color="#D7191C")
ax.set_xlabel("Wealth quintile (1=poorest, 5=richest)")
ax.set_ylabel("Predicted under-5 mortality (/1000)")
ax.set_title("Concept drift: the wealth->mortality relationship changes")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_concept_drift.png", dpi=150); plt.close()

# Fig 4: feature importance drift
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(FEATURES)); w = 0.35
ax.bar(x - w/2, imp_s, w, label="Kenya", color="#2C7BB6")
ax.bar(x + w/2, imp_t, w, label="Nigeria", color="#D7191C")
ax.set_xticks(x); ax.set_xticklabels(FEATURES, rotation=30, ha="right")
ax.set_ylabel("Feature importance (gain)")
ax.set_title("Feature-importance drift")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_importance_drift.png", dpi=150); plt.close()

# Fig 5: calibration / label shift
fig, ax = plt.subplots(figsize=(6, 4))
def cal_curve(p, y, label):
    order = np.argsort(p)
    p_, y_ = p[order], y[order]
    b = np.array_split(np.arange(len(p_)), 10)
    x = [p_[i].mean() for i in b]; yy = [y_[i].mean() for i in b]
    ax.plot(x, yy, "o-", label=label)
cal_curve(p_s_on_t, y_t, "raw (Kenya model on Nigeria)")
cal_curve(p_rescaled, y_t, "prior-rescaled (rank-preserving)")
ax.plot([0, p_s_on_t.max()], [0, p_s_on_t.max()], "k--", lw=1)
ax.set_xlabel("Mean predicted P(death)")
ax.set_ylabel("Observed P(death)")
ax.set_title("Label shift: a rank-preserving rescale fixes calibration, not ranking")
ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/fig5_calibration.png", dpi=150); plt.close()

# save results
results = {
    "matrix": {f"{s}->{t}": round(matrix[(s, t)], 4) for s in keys for t in keys},
    "decomposition_KE_NG": {
        "auc_ke_cv": round(auc_ss, 4), "auc_ke_on_ng": round(auc_s_on_t, 4),
        "auc_ng_cv": round(auc_tt, 4),
        "concept_drift_gap": round(concept_drift, 4),
        "reverse_ng_on_ke": round(auc_t_on_s, 4),
        "reverse_concept_drift_gap": round(concept_drift_rev, 4),
        "psi": {f: round(v, 4) for f, v in psi_vals.items()},
        "domain_auc": round(dom_auc, 4),
        "wealth_pd_ke": {str(q): round(v, 4) for q, v in pd_s.items()},
        "wealth_pd_ng": {str(q): round(v, 4) for q, v in pd_t.items()},
        "mean_pred_on_target": round(mean_pred, 5),
        "label_multiplier": round(label_mult, 3),
        "brier_raw": round(brier_raw, 4),
        "brier_rescaled": round(brier_rescaled, 4),
        "auc_after_rescale": round(auc_after_rescale, 4),
        "base_rate_ke": round(float(y_s.mean()), 5),
        "base_rate_ng": round(float(y_t.mean()), 5),
    },
    "features": FEATURES,
}
with open("results_project1.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results_project1.json and figures to ./figs/")
