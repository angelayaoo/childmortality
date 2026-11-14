# Child-Mortality ML — Concept Drift & the Limits of OOD Detection

**A study of whether machine-learning models for under-five child mortality can
be safely deployed across DHS surveys — and whether out-of-distribution (OOD)
detectors can catch the drift that breaks them.**

*Target venue: AAAI Deployable AI (DAI) workshop.*

---

## Research question

> When a child-mortality risk model trained on one Demographic and Health Survey
> (DHS) is deployed to a new population — a different country, or the same
> country years later — **(1)** how much does it degrade, **(2)** what kind of
> shift causes the degradation, and **(3)** can standard OOD / uncertainty
> monitors detect it before it causes harm?

## Headline findings

1. **Transfer degrades.** Across **75 DHS surveys / 5,550 transfer pairs**,
   in-domain AUC is **0.585** vs. **0.556** when transferred (mean −0.029; 76%
   of pairs degrade). "Donor quality" spans **0.50 → 0.58** and scales with the
   base mortality rate (`r = 0.29`).

2. **The degradation is concept drift, and it is undetectable.**
   - Feature-space OOD detectors (Mahalanobis, Isolation Forest, k-NN, domain
     classifier) *do* see covariate shift (AUC 0.68–0.90) but explain **≤4%** of
     actual model failure (`r = 0.11–0.19`).
   - Uncertainty monitors (conformal coverage, ensemble epistemic) explain
     **≤1%** (`r ≈ 0.07`).
   - Within-country temporal drift (Bangladesh 2014→2022) degrades the model
     (+0.028 AUC) while every detector stays at **chance (≈0.50)**.

3. **The broken mechanism is localizable (SHAP).** Kenya→Nigeria: education and
   urban residence gain importance; maternal age and child sex lose importance;
   wealth's protective gradient sharpens from weak to strongly protective.

## The validated framework — *Deploy → Monitor → Adapt*

We tested four candidate "fixes" and falsified half of them:

| method | verdict | evidence |
|---|---|---|
| Invariant feature pruning | ❌ fails | pruning volatile features *lowers* transfer AUC |
| **Few-shot tree updating** | ✅ works | ~2,000 labels recover 60–80% of the drift gap |
| **Sentinel-cohort monitoring** | ✅ best monitor | 300 labels estimate true error at `r = 0.87` |
| SHAP-guided active learning | ❌ fails | divergence signal too weak; ≈ random |

**Protocol:** monitor true error (Brier) on a ~300-label sentinel cohort
(`r = 0.87`), adapt with few-shot fine-tuning (~2,000 labels), and fix label
shift for free via prior-rescale recalibration — rather than relying on
input- or confidence-based OOD detectors.

---

## Repository layout

```
01_inventory.py            # enumerate DHS zips, identify Individual Recode files
02_extract_analyze.py      # extract + schema/feature audit (representative sample)
03_child_mortality_proof.py# build child-level data, prove U5MR pipeline
04_project1.py             # 4-country transfer + drift decomposition (rich features)
05_project1_expanded.py    # 75-country transfer matrix + OOD-detector limits
06_ood_detectors.py        # Mahalanobis / IsolationForest / k-NN detector suite
07a_conformal_uncertainty.py # conformal + epistemic uncertainty monitors
07b_temporal.py            # within-country temporal drift (Bangladesh 2014→2022)
07c_shap_drift.py          # SHAP feature-level concept-drift isolation
08a_invariant_pruning.py   # playbook method 1 (fails)
08b_fewshot.py             # playbook method 2 (few-shot tree updating)
08c_sentinel.py            # playbook method 3 (sentinel cohorts)
08d_shap_al.py             # playbook method 4 (SHAP-guided active learning)
08e_figures.py             # recovery-curve + sentinel-convergence figures

paper/                     # DAI workshop paper (AAAI-2027 format): source, PDF, bib, figures
figs/                      # all result figures (fig1–fig15)
results_*.json             # aggregate result summaries
```

## Data note

The raw **DHS survey files are NOT included** in this repository (they require
registration with the DHS Program and are not redistributable). To reproduce,
place the DHS Individual Recode `.zip` files (and/or `*DT.zip` Stata variants)
in the repo root; the scripts extract and process them automatically. The
Individual Recode (IR) file is woman-level but embeds the full birth history as
wide arrays, which the scripts melt into child-level rows (one row per birth in
the last 5 years; target = under-five death, `b5 == 0`).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 05_project1_expanded.py   # 75-country transfer matrix + OOD limits
python 06_ood_detectors.py       # real OOD detector suite
python 07a_conformal_uncertainty.py
python 07b_temporal.py
python 07c_shap_drift.py
python 08b_fewshot.py            # few-shot adaptation
python 08c_sentinel.py           # sentinel monitoring
```

*All experiments run locally on CPU (pandas / scikit-learn / xgboost only).*
