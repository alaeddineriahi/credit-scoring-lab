# Model Card: Credit-card PD model (LightGBM with monotonic constraints)

Structured after Mitchell et al. (2019), *Model Cards for Model Reporting*, and the documentation themes of the EU AI Act for high-risk systems (Annex III, credit scoring).

## Intended use
- **Purpose:** estimate the 1-month probability of default (PD) of credit-card holders to support credit decisions. The model is an **aid to a human credit analyst, not an autonomous decision-maker**.
- **When decisions are automatic:** only when the conformal prediction set is a single label that agrees with the profit-based decision. Every other case is referred to an analyst.
- **Out of scope:** other products (mortgages, SME loans), other countries or periods without re-validation, and any use as a sole basis for a decision about an individual.

## Data
- **Source:** UCI *Default of Credit Card Clients* (CC BY 4.0). 30,000 clients of a Taiwanese bank in 2005, with a 22.1% default rate.
- **Split:** stratified 60/20/20. Calibration and the cut-off are set on validation, and the test set is scored once.
- **Features:** 42 behavioural and account features built with SQL (delinquency, utilisation, payment ratios).
- **Excluded:** sex, marital status and age are **not used as inputs**; they're used for the fairness audit only.

## Performance (test, n = 6,000)
| Metric | Value |
|---|---|
| AUC [95% bootstrap CI] | 0.792 [0.778–0.806] |
| Gini / KS | 0.583 / 0.450 |
| Brier score | 0.131 |
| Gain over the regulatory scorecard (AUC) | +0.021 [+0.014, +0.028] |
| Calibration | Platt scaling. Observed and predicted default rates agree within each rating grade |

## Decision policy
- **Profit-based cut-off:** approve if PD < 0.594, which maximises expected profit under Verbraken et al. (2014) assumptions.
- **Conformal triage** (α = 0.10, class-conditional):
  - 34.6% approved automatically (6.7% default rate)
  - 9.4% declined automatically (72.8%)
  - 56.0% sent to an analyst
  - Measured coverage: 89.9% (good payers) and 89.5% (defaulters)

## Fairness
At the decision cut-off:

| Attribute | Approval-rate gap | Equalized-odds gap |
|---|---|---|
| Sex | 0.7 pts | 0.033 |
| Age band | 3.4 pts | 0.053 |
| Marital status | 3.0 pts | 0.099 (driven by a group of n = 73) |

The AUC is similar across groups (0.766–0.796). These gaps should be reviewed against the institution's own fairness policy before any deployment.

## Explainability
- **Global:** SHAP importance for the whole model.
- **Per applicant:** the top risk and protective factors, given as French reason codes.
- **Monotonicity:** constraints on the delinquency, payment and limit features. A test on 200 applicants found 0 violations when only the delay changes.

## Monitoring
- **Score stability:** PSI between train and test = 0.0007. Recommended alert thresholds: PSI > 0.10 (investigate), > 0.25 (re-develop).
- **Recommended ongoing checks:** monthly PSI for the score and the main features, observed versus predicted default rate by grade, the conformal coverage rate, and fairness gaps.

## Limitations and risks
- **Validation:** the split is random, not out-of-time (the data has a single observation date).
- **Portfolio:** the data is a 2005 portfolio, so PD levels won't transfer.
- **Expected-loss assumptions:** EL uses an assumed LGD (45%) and EAD (latest bill).
- **Profit assumptions:** the profit cut-off depends on the assumed ROI and LGD distribution.
