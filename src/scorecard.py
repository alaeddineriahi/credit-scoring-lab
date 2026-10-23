"""Regulatory baseline: optimal binning + WoE + logistic regression, scaled to scorecard points.

Reference: Navas-Palencia (2020), "Optimal binning: mathematical programming formulation",
arXiv:2001.08025 (the optbinning library).
"""
import joblib
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from config import MODEL_DIR, PROCESSED_DIR, TARGET
from evaluate import bootstrap_auc_ci, metrics
from features import feature_columns

# Points to double the odds = 20, 600 points at odds 50:1 (a common industry convention).
SCALING = {"pdo": 20, "odds": 50, "scorecard_points": 600}


MIN_IV, MAX_CORR, MAX_VARS = 0.02, 0.6, 12


def _binning(feature_names: list[str]) -> BinningProcess:
    return BinningProcess(
        variable_names=feature_names,
        categorical_variables=[c for c in ["education"] if c in feature_names],
        binning_fit_params={name: {"monotonic_trend": "auto_asc_desc"}
                            for name in feature_names if name != "education"},
    )


def select_variables(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Greedy IV ranking with a WoE-correlation filter, as done in bank scorecard development:
    take variables by decreasing IV and drop any whose WoE correlates > MAX_CORR with one already kept."""
    bp = _binning(list(X.columns)).fit(X, y)
    summary = bp.summary().set_index("name")["iv"].astype(float).sort_values(ascending=False)
    woe = bp.transform(X, metric="woe")
    corr = woe.corr().abs()
    kept = []
    for name, iv in summary.items():
        if iv < MIN_IV or len(kept) >= MAX_VARS:
            continue
        if all(corr.loc[name, k] <= MAX_CORR for k in kept):
            kept.append(name)
    return summary.loc[kept].rename("iv").reset_index()


def build_scorecard(feature_names: list[str]) -> Scorecard:
    binning = _binning(feature_names)
    return Scorecard(
        binning_process=binning,
        estimator=LogisticRegression(max_iter=1000, C=1.0),
        scaling_method="pdo_odds",
        scaling_method_params=SCALING,
        reverse_scorecard=True,  # higher score = lower risk, as in bank scorecards
    )


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feats = feature_columns(df)
    train, valid = df[df.split == "train"], df[df.split == "valid"]

    selected = select_variables(train[feats], train[TARGET])
    print("Selected variables (IV, after correlation filter):")
    print(selected.to_string(index=False))
    sc_vars = list(selected["name"])

    # With optbinning's WoE convention every coefficient must be negative; a positive one signals
    # multicollinearity, so the offending variable is dropped and the model refitted.
    while True:
        sc = build_scorecard(sc_vars)
        sc.fit(train[sc_vars], train[TARGET])
        coefs = dict(zip(sc_vars, sc.estimator_.coef_[0].round(3)))
        wrong = [v for v, c in coefs.items() if c > 0]
        if not wrong:
            break
        print(f"Dropping {wrong} (coefficient sign reversal)")
        sc_vars = [v for v in sc_vars if v not in wrong]
    print("Logistic coefficients on WoE:", {k: float(v) for k, v in coefs.items()})

    p_valid = sc.predict_proba(valid[sc_vars])[:, 1]
    m = metrics(valid[TARGET], p_valid)
    m["auc_ci95"] = bootstrap_auc_ci(valid[TARGET], p_valid)
    print("Validation:", m)

    table = sc.table(style="detailed")
    table.to_csv(MODEL_DIR / "scorecard_table.csv", index=False)
    joblib.dump({"scorecard": sc, "variables": sc_vars}, MODEL_DIR / "scorecard.joblib")
    pts = sc.score(valid[sc_vars])
    print(f"Points range on validation: {pts.min():.0f} - {pts.max():.0f}")


if __name__ == "__main__":
    main()
