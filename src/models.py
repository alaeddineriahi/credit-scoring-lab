"""Challenger models, calibration and the final (one-time) test-set comparison.

- Explainable Boosting Machine: glass-box GAM with pairwise interactions
  (Nori et al. 2019, "InterpretML", arXiv:1909.09223).
- LightGBM with monotonic constraints: risk can only move in the business-logical direction.
Hyper-parameters are tuned with 5-fold CV on the training set only; the validation set is used
for probability calibration; the test set is scored once, at the end.
"""
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import MODEL_DIR, PROCESSED_DIR, ROOT, SEED, TARGET
from evaluate import bootstrap_auc_ci, metrics, plot_calibration, save_json
from features import feature_columns

# +1: higher value can only increase PD; -1: can only decrease PD.
MONOTONE = {
    "max_delay_6m": 1, "max_delay_3m": 1, "n_delay_months": 1, "n_missed_payments": 1,
    "months_since_delay": -1, "n_paid_in_full": -1, "pay_ratio_avg": -1, "pay_ratio_min_3m": -1,
    "paid_to_limit_6m": -1, "limit_bal": -1, "delay_trend": 1,
    **{f"delay_{m}": 1 for m in range(1, 7)},
}


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    constraints = [MONOTONE.get(c, 0) for c in X.columns]

    def objective(trial):
        params = {
            "objective": "binary", "verbosity": -1, "seed": SEED, "n_jobs": 8,
            "monotone_constraints": constraints, "monotone_constraints_method": "advanced",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
        }
        aucs, iters = [], []
        for tr, va in cv.split(X, y):
            dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
            dva = lgb.Dataset(X.iloc[va], y.iloc[va])
            booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(roc_auc_score(y.iloc[va], booster.predict(X.iloc[va], num_iteration=booster.best_iteration)))
            iters.append(booster.best_iteration)
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    best["cv_auc"] = round(study.best_value, 4)
    return best


class PlattCalibrator:
    """Platt scaling on the log-odds of a frozen model's PD, fitted on the validation set."""

    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.lr = LogisticRegression()

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, X, y):
        self.lr.fit(self._logit(self.predict_fn(X)), y)
        return self

    def predict_pd(self, X):
        return self.lr.predict_proba(self._logit(self.predict_fn(X)))[:, 1]


def paired_auc_diff_ci(y, p_a, p_b, n_boot: int = 1000) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feats = feature_columns(df)
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    Xtr, ytr = tr[feats], tr[TARGET]

    # 1) Scorecard (already fitted by scorecard.py)
    sc_bundle = joblib.load(MODEL_DIR / "scorecard.joblib")
    scorecard, sc_vars = sc_bundle["scorecard"], sc_bundle["variables"]

    # 2) Explainable Boosting Machine
    ebm = ExplainableBoostingClassifier(interactions=10, random_state=SEED, n_jobs=8)
    ebm.fit(Xtr, ytr)

    # 3) LightGBM with monotonic constraints, tuned on train CV
    best = tune_lgbm(Xtr, ytr)
    print("LightGBM best params:", best)
    lgbm = lgb.LGBMClassifier(
        objective="binary", verbosity=-1, random_state=SEED, n_jobs=8,
        monotone_constraints=[MONOTONE.get(c, 0) for c in feats], monotone_constraints_method="advanced",
        bagging_freq=1, **{k: v for k, v in best.items() if k != "cv_auc"},
    )
    lgbm.fit(Xtr, ytr)

    # Raw PD functions (models trained on train only) and Platt calibration on the validation set
    raw = {
        "Scorecard (WoE + logit)": lambda X: scorecard.predict_proba(X[sc_vars])[:, 1],
        "EBM": lambda X: ebm.predict_proba(X[feats])[:, 1],
        "LightGBM monotone": lambda X: lgbm.predict_proba(X[feats])[:, 1],
    }
    calibrated = {name: PlattCalibrator(fn).fit(va, va[TARGET]) for name, fn in raw.items()}

    # Final, one-time evaluation on the test set
    results, preds, curves = {}, {}, {}
    for name, cal in calibrated.items():
        p = cal.predict_pd(te)
        m = metrics(te[TARGET], p)
        m["auc_ci95"] = bootstrap_auc_ci(te[TARGET], p)
        results[name] = m
        preds[name] = p
        curves[name] = (te[TARGET].values, p)
        print(f"{name:28s} {m}")

    base = "Scorecard (WoE + logit)"
    for name in preds:
        if name != base:
            results[name]["auc_gain_vs_scorecard"] = paired_auc_diff_ci(te[TARGET], preds[name], preds[base])

    results["_meta"] = {"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                        "default_rate_test": round(float(te[TARGET].mean()), 4),
                        "lgbm_params": best, "n_features": len(feats)}
    save_json(results, "test_metrics.json")
    plot_calibration(curves, "calibration_test.png")

    # Keep raw (pre-Platt) validation and test scores for conformal prediction, plus calibrated PDs.
    for split_df, split_name in ((va, "valid"), (te, "test")):
        out = split_df[["id", TARGET, "sex", "marriage", "age"]].copy()
        for name, fn in raw.items():
            out[f"raw_{name}"] = fn(split_df)
            out[f"pd_{name}"] = calibrated[name].predict_pd(split_df)
        out.to_parquet(ROOT / "reports" / f"{split_name}_predictions.parquet", index=False)
    joblib.dump({"ebm": ebm, "lgbm": lgbm, "features": feats,
                 "platt": {n: c.lr for n, c in calibrated.items()}},
                MODEL_DIR / "challengers.joblib")


if __name__ == "__main__":
    main()
