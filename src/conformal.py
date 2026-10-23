"""Class-conditional (Mondrian) split conformal prediction: a "refer to analyst" zone.

For each applicant the model returns a prediction set: {good}, {default}, or {good, default}.
Ambiguous or empty sets are referred to a human analyst (human oversight, EU AI Act Art. 14).
Calibration uses the validation set and model outputs from models trained on the train set only,
so validation and test are exchangeable and the per-class coverage guarantee holds.
Reference: Vovk et al. (2005), Algorithmic Learning in a Random World; Angelopoulos & Bates (2021),
"A Gentle Introduction to Conformal Prediction", arXiv:2107.07511.
"""
import numpy as np


def mondrian_thresholds(y_cal, p_cal, alpha: float = 0.10) -> dict[int, float]:
    """Per-class conformal quantile of the non-conformity score s = 1 - p(true class)."""
    y_cal, p_cal = np.asarray(y_cal), np.asarray(p_cal)
    q = {}
    for c in (0, 1):
        p_true = p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c]
        scores = np.sort(1 - p_true)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        q[c] = float(scores[min(k, n - 1)])
    return q


def prediction_sets(p, q: dict[int, float]) -> np.ndarray:
    """Return an array of labels: 'good', 'default', or 'refer' (ambiguous or empty set)."""
    p = np.asarray(p)
    in_good = p <= q[0]          # non-conformity of "good" is 1 - (1 - p) = p
    in_default = 1 - p <= q[1]   # non-conformity of "default" is 1 - p
    out = np.where(in_good & ~in_default, "good",
          np.where(in_default & ~in_good, "default", "refer"))
    return out


def summarize(y, labels) -> dict:
    y, labels = np.asarray(y), np.asarray(labels)
    res = {}
    for c, name in ((0, "good"), (1, "default")):
        mask = y == c
        covered = (labels[mask] == name) | (labels[mask] == "refer")
        res[f"coverage_{name}"] = round(float(covered.mean()), 4)
    for lab in ("good", "default", "refer"):
        m = labels == lab
        res[f"share_{lab}"] = round(float(m.mean()), 4)
        res[f"default_rate_{lab}"] = round(float(y[m].mean()), 4) if m.any() else None
    return res
