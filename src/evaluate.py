"""Metrics in the language of credit-risk teams: AUC, Gini, KS, Brier, calibration, bootstrap CIs."""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import FIG_DIR, SEED


def ks_stat(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_stat(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
    }


def bootstrap_auc_ci(y, p, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def plot_calibration(results: dict[str, tuple], fname: str, n_bins: int = 10) -> None:
    """results: {model_name: (y_true, p_pred)}"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Calibration parfaite")
    for name, (y, p) in results.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.5, label=name)
    ax.set_xlabel("PD prédite")
    ax.set_ylabel("Taux de défaut observé")
    ax.set_title("Courbe de calibration (jeu de test)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=160)
    plt.close(fig)


def save_json(obj, fname: str) -> None:
    from config import ROOT
    path = ROOT / "reports" / fname
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
