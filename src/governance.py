"""Turning PDs into decisions, and checking them the way a model validator and the EU AI Act expect.

1. Champion selection on the validation set (never on test).
2. Profit-based cut-off with the Expected Maximum Profit measure (Verbraken et al. 2014, EJOR 238(2)),
   using the EMP package defaults: LGD point masses p0 = 0.55 (no loss), p1 = 0.10 (total loss),
   ROI = 0.2644.
3. Rating master scale and expected loss (EL = PD x LGD x EAD; LGD = 45% and EAD = latest bill are
   stated assumptions).
4. "Refer to analyst" zone with Mondrian conformal prediction (alpha = 0.10).
5. Fairness audit by sex, age band and marital status (none of them are model inputs).
6. Stability: Population Stability Index of the score, train vs test.
7. Explanations: SHAP for the champion and French reason codes.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import (MetricFrame, demographic_parity_difference, equalized_odds_difference,
                               false_negative_rate, false_positive_rate, selection_rate)
from sklearn.metrics import roc_auc_score

from conformal import mondrian_thresholds, prediction_sets, summarize
from config import FIG_DIR, MODEL_DIR, PROCESSED_DIR, ROOT, TARGET
from evaluate import save_json

P0, P1, ROI = 0.55, 0.10, 0.2644
LGD, ALPHA = 0.45, 0.10
GRADES = [("A", 0.00, 0.03), ("B", 0.03, 0.06), ("C", 0.06, 0.10), ("D", 0.10, 0.20),
          ("E", 0.20, 0.35), ("F", 0.35, 0.55), ("G", 0.55, 1.01)]

REASONS_FR = {
    "delay_1": "Retard de paiement sur le dernier mois",
    "delay_2": "Retard de paiement il y a deux mois",
    "delay_3": "Retard de paiement il y a trois mois",
    "delay_4": "Retard de paiement il y a quatre mois",
    "delay_5": "Retard de paiement il y a cinq mois",
    "delay_6": "Retard de paiement il y a six mois",
    "code_1": "Mode de remboursement du dernier mois",
    "code_2": "Mode de remboursement d'il y a deux mois",
    "code_3": "Mode de remboursement d'il y a trois mois",
    "bill_1": "Montant de la dernière facture", "n_paid_in_full": "Peu de mois remboursés intégralement",
    "n_no_usage": "Profil d'utilisation de la carte",
    "util_slope": "Utilisation du plafond en hausse", "bill_growth": "Hausse récente de la facture",
    "education": "Niveau d'études", "code_4": "Mode de remboursement d'il y a quatre mois",
    "code_5": "Mode de remboursement d'il y a cinq mois", "code_6": "Mode de remboursement d'il y a six mois",
    "paid_4": "Paiement d'il y a quatre mois faible", "paid_5": "Paiement d'il y a cinq mois faible",
    "paid_6": "Paiement d'il y a six mois faible",
    **{f"bill_{m}": f"Montant de la facture d'il y a {m} mois" for m in range(2, 7)},
    "max_delay_3m": "Retard maximal élevé sur les 3 derniers mois",
    "max_delay_6m": "Retard maximal élevé sur les 6 derniers mois",
    "n_delay_months": "Nombre élevé de mois avec retard",
    "months_since_delay": "Retard de paiement récent",
    "n_missed_payments": "Mensualités non réglées alors qu'un solde était dû",
    "limit_bal": "Plafond de crédit faible",
    "util_avg": "Utilisation moyenne du plafond élevée",
    "util_last": "Utilisation du plafond élevée ce mois-ci",
    "util_max": "Pic d'utilisation du plafond élevé",
    "pay_ratio_avg": "Faible part des factures remboursée",
    "pay_ratio_min_3m": "Remboursement partiel récent",
    "paid_to_limit_6m": "Faibles remboursements au regard du plafond",
    "delay_trend": "Dégradation récente du comportement de paiement",
    "paid_1": "Paiement du dernier mois faible", "paid_2": "Paiement d'il y a deux mois faible",
    "paid_3": "Paiement d'il y a trois mois faible",
}


# --- 2. Expected Maximum Profit ---------------------------------------------------------------
def emp(y, pd_, n_lambda: int = 201):
    """EMP and the expected share of rejected applicants at the profit-maximising cut-off."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    pi_def, pi_good = y.mean(), 1 - y.mean()
    order = np.argsort(-pd_)                       # reject the riskiest first
    y_sorted = y[order]
    f_def = np.concatenate([[0], np.cumsum(y_sorted) / y.sum()])          # share of defaulters rejected
    f_good = np.concatenate([[0], np.cumsum(1 - y_sorted) / (1 - y).sum()])
    lambdas = np.linspace(0, 1, n_lambda)
    best_profit, best_idx = [], []
    for lam in lambdas:
        profit = lam * pi_def * f_def - ROI * pi_good * f_good
        best_idx.append(int(profit.argmax()))
        best_profit.append(profit.max())
    best_profit, best_idx = np.array(best_profit), np.array(best_idx)
    reject_share = (pi_def * f_def[best_idx] + pi_good * f_good[best_idx])
    interior = slice(1, -1)
    w_cont = (1 - P0 - P1)
    emp_value = (P0 * best_profit[0] + P1 * best_profit[-1]
                 + w_cont * np.trapezoid(best_profit[interior], lambdas[interior]) / (lambdas[-2] - lambdas[1]))
    emp_reject = (P0 * reject_share[0] + P1 * reject_share[-1]
                  + w_cont * np.trapezoid(reject_share[interior], lambdas[interior]) / (lambdas[-2] - lambdas[1]))
    return float(emp_value), float(emp_reject)


def cutoff_for_reject_share(pd_valid, reject_share: float) -> float:
    return float(np.quantile(pd_valid, 1 - reject_share))


# --- 6. PSI -------------------------------------------------------------------------------------
def psi(expected, actual, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def age_band(age):
    return pd.cut(age, [0, 29, 44, 200], labels=["<30", "30-44", "45+"])


def main() -> None:
    feats_df = pd.read_parquet(PROCESSED_DIR / "features.parquet").set_index("id")
    va = pd.read_parquet(ROOT / "reports" / "valid_predictions.parquet")
    te = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    names = [c[3:] for c in va.columns if c.startswith("pd_")]
    report = {}

    # 1. Champion chosen on validation AUC
    valid_auc = {n: roc_auc_score(va[TARGET], va[f"pd_{n}"]) for n in names}
    champion = max(valid_auc, key=valid_auc.get)
    report["champion"] = champion
    report["valid_auc"] = {k: round(v, 4) for k, v in valid_auc.items()}
    print("Champion (validation AUC):", champion, report["valid_auc"])
    pv, pt = va[f"pd_{champion}"].values, te[f"pd_{champion}"].values

    # 2. EMP and cut-off
    emp_rows = {}
    for n in names:
        e, r = emp(te[TARGET], te[f"pd_{n}"])
        emp_rows[n] = {"emp_per_unit_lent": round(e, 4), "expected_reject_share": round(r, 4)}
    report["emp_test"] = emp_rows
    _, reject_valid = emp(va[TARGET], pv)
    cutoff = cutoff_for_reject_share(pv, reject_valid)
    approve = pt < cutoff
    report["decision"] = {"cutoff_pd": round(cutoff, 4), "approval_rate_test": round(float(approve.mean()), 4),
                          "default_rate_approved": round(float(te[TARGET][approve].mean()), 4),
                          "default_rate_rejected": round(float(te[TARGET][~approve].mean()), 4)}
    print("EMP (test):", emp_rows)
    print("Decision:", report["decision"])

    # 3. Rating grades and expected loss
    te = te.join(feats_df[["bill_1"]], on="id")
    te["grade"] = pd.cut(pt, [g[1] for g in GRADES] + [GRADES[-1][2]], labels=[g[0] for g in GRADES], right=False)
    te["ead"] = te["bill_1"].clip(lower=0)
    te["el"] = pt * LGD * te["ead"]
    grade_tab = te.groupby("grade", observed=False).agg(
        n=("id", "size"), pd_mean=(f"pd_{champion}", "mean"), observed_dr=(TARGET, "mean"),
        ead_sum=("ead", "sum"), el_sum=("el", "sum")).round(4)
    report["master_scale"] = grade_tab.reset_index().to_dict(orient="records")
    print(grade_tab)

    # 4. Conformal "refer to analyst" zone
    raw_v, raw_t = va[f"raw_{champion}"].values, te[f"raw_{champion}"].values
    q = mondrian_thresholds(va[TARGET], raw_v, alpha=ALPHA)
    labels = prediction_sets(raw_t, q)
    report["conformal"] = {"alpha": ALPHA, "thresholds": q, **summarize(te[TARGET], labels)}
    print("Conformal:", report["conformal"])

    # Triage: automatic only when the conformal label agrees with the profit decision
    auto_approve = (labels == "good") & approve
    auto_decline = (labels == "default") & ~approve
    refer = ~(auto_approve | auto_decline)
    yt = te[TARGET].values
    report["triage"] = {
        name: {"share": round(float(m.mean()), 4), "default_rate": round(float(yt[m].mean()), 4) if m.any() else None}
        for name, m in (("auto_approve", auto_approve), ("auto_decline", auto_decline), ("analyst_review", refer))
    }
    print("Triage:", report["triage"])

    # 5. Fairness audit on the approve/decline decision (decline = predicted 1)
    y_pred = (~approve).astype(int)
    sens = {"sex": te["sex"].map({1: "homme", 2: "femme"}), "age": age_band(te["age"]),
            "marriage": te["marriage"].map({1: "marié(e)", 2: "célibataire", 3: "autre"})}
    fairness = {}
    for attr, groups in sens.items():
        mf = MetricFrame(metrics={"approval_rate": lambda yt, yp: 1 - selection_rate(yt, yp),
                                  "false_decline_rate": false_positive_rate,
                                  "missed_default_rate": false_negative_rate},
                         y_true=te[TARGET], y_pred=y_pred, sensitive_features=groups)
        fairness[attr] = {
            "by_group": mf.by_group.round(4).reset_index().to_dict(orient="records"),
            "demographic_parity_diff": round(float(demographic_parity_difference(te[TARGET], y_pred, sensitive_features=groups)), 4),
            "equalized_odds_diff": round(float(equalized_odds_difference(te[TARGET], y_pred, sensitive_features=groups)), 4),
            "auc_by_group": {str(g): round(float(roc_auc_score(te[TARGET][groups == g], pt[groups == g])), 4)
                             for g in groups.dropna().unique()},
        }
    report["fairness"] = fairness
    print("Fairness:", {k: (v["demographic_parity_diff"], v["equalized_odds_diff"]) for k, v in fairness.items()})

    # 6. Stability (train vs test score distribution)
    ch = joblib.load(MODEL_DIR / "challengers.joblib")
    feats = ch["features"]
    train = feats_df[feats_df.split == "train"]
    if champion == "LightGBM monotone":
        train_raw = ch["lgbm"].predict_proba(train[feats])[:, 1]
    elif champion == "EBM":
        train_raw = ch["ebm"].predict_proba(train[feats])[:, 1]
    else:
        sc = joblib.load(MODEL_DIR / "scorecard.joblib")
        train_raw = sc["scorecard"].predict_proba(train[sc["variables"]])[:, 1]
    report["psi_train_vs_test"] = round(psi(train_raw, raw_t), 4)
    print("PSI:", report["psi_train_vs_test"])

    # 7. SHAP explanations for the champion (tree models) and French reason codes
    model = ch["lgbm"] if champion == "LightGBM monotone" else None
    if model is not None:
        X_te = feats_df.loc[te["id"], feats]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
        report["shap_top10"] = imp.head(10).round(4).to_dict()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        imp.head(12)[::-1].plot.barh(ax=ax, color="#2b6cb0")
        ax.set_xlabel("Impact moyen |SHAP| sur le log-odds de défaut")
        ax.set_title("Variables les plus influentes (LightGBM)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "shap_importance.png", dpi=160)
        plt.close(fig)
        # Example reason codes for the riskiest declined applicant
        i = int(np.argmax(pt))
        top = pd.Series(sv[i], index=feats).sort_values(ascending=False).head(3)
        report["example_reason_codes"] = {"pd": round(float(pt[i]), 4),
                                          "reasons": [REASONS_FR.get(f, f) for f in top.index]}
        print("Example reason codes:", report["example_reason_codes"])

    save_json(report, "governance.json")


if __name__ == "__main__":
    main()
