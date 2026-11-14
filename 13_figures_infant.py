"""Regenerate figures for the reframed (censoring-correct) paper."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = pd.read_parquet("detector_loss_infant.parquet")

# Figure 1: input-shift alarm vs transfer loss (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col, name in zip(axes, ["mahal", "domain"],
                         ["Mahalanobis distance", "Domain classifier"]):
    r = res[col].corr(res["loss"])
    ax.scatter(res[col], res["loss"], s=8, alpha=0.4, color="#4C72B0")
    ax.set_xlabel(f"{name} detectability (AUC)")
    ax.set_title(f"{name}\nr = {r:.2f}")
    ax.axhline(0, ls="--", c="gray", lw=0.8)
axes[0].set_ylabel("Transfer loss (oracle AUC - transferred AUC)")
fig.suptitle("Input-shift alarms do not reliably predict transfer loss", y=1.02)
plt.tight_layout()
plt.savefig("paper/Figures/fig_alarm_vs_loss.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: AUROC of each alarm for flagging high-loss transfers
from sklearn.metrics import roc_auc_score
fig, ax = plt.subplots(figsize=(5, 3.6))
thr = 0.05
hl = (res["loss"] > thr).astype(int)
aucs = [roc_auc_score(hl, res[c]) for c in ["mahal", "domain"]]
bars = ax.bar(["Mahalanobis", "Domain\nclassifier"], aucs, color=["#4C72B0", "#D7191C"], width=0.5)
ax.axhline(0.5, ls="--", c="gray", lw=0.8)
for b, a in zip(bars, aucs):
    ax.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.2f}", ha="center")
ax.set_ylabel("AUROC for flagging loss > 0.05")
ax.set_ylim(0.4, 0.85)
ax.set_title(f"Alarms flag high-loss transfers\nonly slightly above chance (n={int(hl.sum())} high-loss)")
plt.tight_layout()
plt.savefig("paper/Figures/fig_highloss_auroc.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: transfer loss distribution
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.hist(res["loss"], bins=40, color="#4C72B0", alpha=0.85)
ax.axvline(0, ls="--", c="gray", lw=0.8)
ax.axvline(res["loss"].mean(), ls="-", c="#D7191C", lw=1.2, label=f"mean = {res['loss'].mean():.3f}")
ax.set_xlabel("Transfer loss (oracle AUC - transferred AUC)")
ax.set_ylabel("Source-target pairs")
ax.set_title(f"Transfer loss across {len(res)} pairs (weighted, infant mortality)")
ax.legend()
plt.tight_layout()
plt.savefig("paper/Figures/fig_loss_dist.png", dpi=150, bbox_inches="tight")
plt.close()

print("saved 3 figures")
