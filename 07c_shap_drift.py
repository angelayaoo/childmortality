"""
Project 1 (c) — Feature-level concept-drift isolation with SHAP.

Instead of "concept drift happened", isolate WHICH relationship broke.
Compare SHAP values between source (Kenya) and target (Nigeria) models on the
RICH feature set (incl. wealth + residence) to show the wealth->mortality
relationship flips from flat (Kenya) to strongly protective (Nigeria), and
education's importance rises.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RICH = ["child_male", "birth_order", "prev_interval", "mother_age_birth",
        "edu", "urban", "wealth_q"]

def load_rich(country):
    # use the rich-feature cache from 04_project1.py (has wealth_q, urban)
    df = pd.read_parquet(f"cache_children/{country}.parquet")
    df = df[RICH + ["y"]].copy()
    df["prev_interval"] = df["prev_interval"].fillna(df["prev_interval"].median())
    return df

def make_model(seed=0):
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        tree_method="hist", n_jobs=8, random_state=seed, eval_metric="auc")

ke = load_rich("KE91")
ng = load_rich("NG8B")

m_ke = make_model(); m_ke.fit(ke[RICH].values, ke["y"].values)
m_ng = make_model(); m_ng.fit(ng[RICH].values, ng["y"].values)

# SHAP values: each model explained on its OWN data
ex_ke = shap.TreeExplainer(m_ke)
ex_ng = shap.TreeExplainer(m_ng)
shap_ke = ex_ke.shap_values(ke[RICH].values)   # (n, d)
shap_ng = ex_ng.shap_values(ng[RICH].values)

# ---- 1. wealth dependence flip (the key visual) ----
w = RICH.index("wealth_q")
fig, ax = plt.subplots(figsize=(6, 4.5))
for shap_vals, df, color, lab in [(shap_ke, ke, "#2C7BB6", "Kenya model (on Kenya)"),
                                   (shap_ng, ng, "#D7191C", "Nigeria model (on Nigeria)")]:
    qs = np.arange(1, 6)
    means = [shap_vals[df["wealth_q"].values == q, w].mean() for q in qs]
    ax.plot(qs, means, "o-", color=color, label=lab)
ax.axhline(0, ls="--", c="gray", lw=1)
ax.set_xlabel("Wealth quintile (1=poorest, 5=richest)")
ax.set_ylabel("Mean SHAP value of wealth  (log-odds of death)")
ax.set_title("Concept drift: wealth flips from flat to strongly protective")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig11_shap_wealth_flip.png", dpi=140); plt.close()

# ---- 2. SHAP feature-importance divergence ----
imp_ke = np.abs(shap_ke).mean(axis=0)
imp_ng = np.abs(shap_ng).mean(axis=0)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(RICH)); wbar = 0.35
ax.bar(x - wbar/2, imp_ke, wbar, label="Kenya", color="#2C7BB6")
ax.bar(x + wbar/2, imp_ng, wbar, label="Nigeria", color="#D7191C")
ax.set_xticks(x); ax.set_xticklabels(RICH, rotation=30, ha="right")
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("Feature-importance drift (SHAP)")
ax.legend()
plt.tight_layout(); plt.savefig("figs/fig12_shap_importance.png", dpi=140); plt.close()

# ---- 3. SHAP summary (beeswarm) for Nigeria, colored by wealth ----
fig = plt.figure(figsize=(7, 5))
shap.summary_plot(shap_ng, ng[RICH].values, feature_names=RICH, show=False,
                  plot_size=(7, 5), max_display=7)
plt.title("Nigeria model: SHAP summary", fontsize=11)
plt.tight_layout(); plt.savefig("figs/fig13_shap_summary_ng.png", dpi=140, bbox_inches="tight"); plt.close()

# ---- report ----
print("SHAP feature importance (mean |SHAP|):")
print(f"{'feature':18s} {'Kenya':>8s} {'Nigeria':>8s} {'drift':>8s}")
for f, a, b in zip(RICH, imp_ke, imp_ng):
    print(f"{f:18s} {a:8.4f} {b:8.4f} {b-a:+8.4f}")
print("\nWealth SHAP (log-odds) by quintile:")
for q in range(1, 6):
    a = shap_ke[ke["wealth_q"].values == q, w].mean()
    b = shap_ng[ng["wealth_q"].values == q, w].mean()
    print(f"  Q{q}: Kenya={a:+.4f}  Nigeria={b:+.4f}")
print("\nSaved figs/fig11-13.")
