"""Figures for the README and LinkedIn post (all computed from the saved test predictions)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from config import FIG_DIR, ROOT, TARGET

COLORS = {"Scorecard (WoE + logit)": "#718096", "EBM": "#38a169", "LightGBM monotone": "#2b6cb0"}


def main() -> None:
    te = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    gov = json.loads((ROOT / "reports" / "governance.json").read_text(encoding="utf-8"))
    names = [c[3:] for c in te.columns if c.startswith("pd_")]

    # ROC curves
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(te[TARGET], te[f"pd_{n}"])
        auc = roc_auc_score(te[TARGET], te[f"pd_{n}"])
        ax.plot(fpr, tpr, lw=1.8, color=COLORS.get(n), label=f"{n} (AUC {auc:.3f}, Gini {2 * auc - 1:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbes ROC — jeu de test (n = 6 000)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_test.png", dpi=160)
    plt.close(fig)

    # PD distributions by outcome (champion)
    champ = gov["champion"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(te.loc[te[TARGET] == 0, f"pd_{champ}"], bins=bins, alpha=0.6, density=True, label="Bons payeurs", color="#38a169")
    ax.hist(te.loc[te[TARGET] == 1, f"pd_{champ}"], bins=bins, alpha=0.6, density=True, label="Défauts", color="#e53e3e")
    ax.axvline(gov["decision"]["cutoff_pd"], color="black", ls="--", lw=1, label="Seuil (profit maximal, EMP)")
    ax.set_xlabel("PD calibrée")
    ax.set_ylabel("Densité")
    ax.set_title(f"Séparation des classes — {champ}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_distribution.png", dpi=160)
    plt.close(fig)

    # Master scale: predicted vs observed default rate by grade
    ms = pd.DataFrame(gov["master_scale"])
    ms = ms[ms["n"] >= 30]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, ms["pd_mean"], 0.4, label="PD moyenne prédite", color="#2b6cb0")
    ax.bar(x + 0.2, ms["observed_dr"], 0.4, label="Taux de défaut observé", color="#ed8936")
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(ms["grade"], ms["n"])])
    ax.set_ylabel("Taux")
    ax.set_title("Échelle de notation — prédit vs observé (test)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "master_scale.png", dpi=160)
    plt.close(fig)
    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
