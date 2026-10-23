# Credit Scoring Lab: Project Plan

**Goal:** an end-to-end, bank-grade probability-of-default (PD) model on real consumer-credit data. It should combine the scorecard regulators expect, modern machine learning, and the governance the EU AI Act asks of high-risk credit systems. Every number in the README comes from code in this repo.

**Dataset:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), from the UCI Machine Learning Repository, licensed CC BY 4.0.
- **Contents:** 30,000 real credit-card holders from a Taiwanese bank (2005). For each one there are 6 months of repayment status, bill amounts and payments, plus credit limit and demographics. The label is default in the following month (about 22% default).
- **Reference:** Yeh & Lien (2009), *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*, Expert Systems with Applications 36(2).
- **Why this dataset:**
  - Open license and direct download, so the data can be shared with the repo.
  - Its monthly history suits SQL feature engineering.
  - At 30k rows it's within TabPFN-2.5's limit (50k), so the foundation model can use all the data.
  - The TabPFN *Nature* paper analysed this same dataset ([Hollmann et al., 2025](https://www.nature.com/articles/s41586-024-08328-6)).
- *(Home Credit was the first choice, but Kaggle's phone verification blocked the download.)*

---

## Pipeline

| # | Step | What it shows a bank | Method / reference |
|---|---|---|---|
| 1 | **SQL feature engineering** (DuckDB) | Real SQL on banking data | Reshape the data into a monthly table (one row per customer per month), then use SQL window functions for utilisation trends, worst and recent delinquency, payment-to-bill ratios and months since last delay |
| 2 | **Regulatory baseline scorecard** | The model risk teams actually deploy | Optimal binning + WoE/IV + logistic regression, then scaled to points (PDO). Navas-Palencia (2020), *Optimal binning: mathematical programming formulation*, [arXiv:2001.08025](https://arxiv.org/abs/2001.08025) |
| 3 | **Glass-box machine learning** | Accuracy without losing interpretability | Explainable Boosting Machine. Nori et al. (2019), *InterpretML*, [arXiv:1909.09223](https://arxiv.org/abs/1909.09223) |
| 4 | **Gradient boosting with monotonic constraints** | A strong challenger that still respects business logic (e.g. more late payments never lowers risk) | LightGBM + Optuna tuning |
| 5 | **Research extension: tabular foundation model** | Familiarity with 2025–2026 research | TabPFN ([Hollmann et al., *Nature* 2025](https://www.nature.com/articles/s41586-024-08328-6); TabPFN-2.5, [arXiv:2511.08667](https://arxiv.org/abs/2511.08667)). We test whether **balanced context sampling**, reported by Tanna et al. (May 2026, [arXiv:2605.18635](https://arxiv.org/abs/2605.18635)) to beat uniform sampling on Home Credit and Lending Club, also holds on this dataset |
| 6 | **Evaluation the way banks report it** | Speaks the risk team's language | AUC, **Gini, KS**, Brier score, calibration plots, bootstrap confidence intervals |
| 7 | **Profit-based cut-off** | Ties the model to money, not just accuracy | Expected Maximum Profit. Verbraken et al. (2014), *EJOR* |
| 8 | **Rating grades and expected loss** | Turns PD into a credit-risk tool | PD calibration, then a master scale (rating grades), then EL = PD × LGD × EAD. EAD is the latest bill amount (an assumption) and LGD is 45% (the Basel foundation-IRB value for senior unsecured exposures); both are labelled as assumptions |
| 9 | **Uncertainty: "refer to analyst" zone** | Human oversight of uncertain cases (AI Act Art. 14) | Class-conditional (Mondrian) split conformal prediction |
| 10 | **Explanations** | The right to an explanation | SHAP (global and per-applicant), scorecard points, and reason codes written in plain French |
| 11 | **Fairness audit** | Bias checks (AI Act Art. 10) | Fairlearn: approval rates and error rates by sex, age group and marital status. Sex and marital status are never model inputs; they're audited only |
| 12 | **Stability monitoring** | How a model is watched in production | PSI and CSI (stability indices for scores and features) |
| 13 | **Model card** | AI Act-style documentation | Intended use, data, metrics, limits, fairness, monitoring |
| 14 | **Streamlit demo** | A loan-officer interface | PD, grade, points breakdown, French reason codes, confidence flag |

## Repo layout
```
credit-scoring/
  data/raw/            # UCI .xls (downloaded by src/download.py)
  data/processed/      # feature table (parquet)
  src/                 # features.py, scorecard.py, models.py, evaluate.py, conformal.py, fairness.py
  notebooks/           # 01_eda, 02_scorecard, 03_models, 04_governance
  reports/figures/     # charts for README / LinkedIn
  models/              # saved models
  app/                 # streamlit app
  README.md, MODEL_CARD.md, requirements.txt
```

## Rules
- No number goes in the README, CV or LinkedIn unless a script in this repo produced it.
- Fixed random seeds, with train / validation / test split once, stratified (60/20/20). Thresholds and calibration are set on validation only; test is touched once, at the end.
- Results are compared with the published benchmarks on the same dataset where possible (Yeh & Lien 2009; Hollmann et al. 2025).
