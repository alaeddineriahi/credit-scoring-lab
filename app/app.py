"""Loan-officer demo: PD, rating grade, decision, confidence flag and reasons in French.

Run from the project root:  .venv/Scripts/streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features  # noqa: E402
from governance import GRADES, LGD, REASONS_FR  # noqa: E402

STATUS_LABELS = {-2: "Pas d'utilisation", -1: "Payé intégralement", 0: "Paiement minimum",
                 1: "1 mois de retard", 2: "2 mois de retard", 3: "3 mois de retard",
                 4: "4 mois de retard", 5: "5 mois de retard", 6: "6 mois de retard",
                 7: "7 mois de retard", 8: "8 mois et plus"}
MONTHS = ["Sept.", "Août", "Juil.", "Juin", "Mai", "Avril"]
FAVORABLE_FR = {
    "delay_1": "Paiement à jour le dernier mois", "delay_2": "Paiement à jour il y a deux mois",
    "delay_3": "Paiement à jour il y a trois mois", "max_delay_3m": "Aucun retard important sur 3 mois",
    "code_1": "Mode de remboursement favorable le dernier mois",
    "code_2": "Mode de remboursement favorable il y a deux mois",
    "code_3": "Mode de remboursement favorable il y a trois mois",
    "bill_1": "Montant de la dernière facture maîtrisé", "n_paid_in_full": "Factures souvent remboursées intégralement",
    "util_slope": "Utilisation du plafond stable ou en baisse", "bill_growth": "Facture stable",
    "education": "Niveau d'études", "n_no_usage": "Profil d'utilisation de la carte",
    "delay_4": "Paiement à jour il y a quatre mois", "delay_5": "Paiement à jour il y a cinq mois",
    "delay_6": "Paiement à jour il y a six mois", "code_4": "Mode de remboursement favorable il y a quatre mois",
    "code_5": "Mode de remboursement favorable il y a cinq mois", "code_6": "Mode de remboursement favorable il y a six mois",
    "paid_4": "Paiement d'il y a quatre mois significatif", "paid_5": "Paiement d'il y a cinq mois significatif",
    "paid_6": "Paiement d'il y a six mois significatif",
    **{f"bill_{m}": f"Montant de la facture d'il y a {m} mois maîtrisé" for m in range(2, 7)},
    "max_delay_6m": "Aucun retard important sur 6 mois", "n_delay_months": "Peu de mois avec retard",
    "months_since_delay": "Pas de retard récent", "n_missed_payments": "Mensualités réglées régulièrement",
    "limit_bal": "Plafond de crédit élevé", "util_avg": "Utilisation modérée du plafond",
    "util_last": "Utilisation modérée du plafond ce mois-ci", "util_max": "Pas de pic d'utilisation",
    "pay_ratio_avg": "Bonne part des factures remboursée", "pay_ratio_min_3m": "Remboursements réguliers récents",
    "paid_to_limit_6m": "Remboursements importants au regard du plafond",
    "delay_trend": "Comportement de paiement stable ou en amélioration",
    "paid_1": "Paiement du dernier mois significatif", "paid_2": "Paiement d'il y a deux mois significatif",
    "paid_3": "Paiement d'il y a trois mois significatif",
}


@st.cache_resource
def load():
    ch = joblib.load(ROOT / "models" / "challengers.joblib")
    sc = joblib.load(ROOT / "models" / "scorecard.joblib")
    gov = json.loads((ROOT / "reports" / "governance.json").read_text(encoding="utf-8"))
    feats_df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    return ch, sc, gov, feats_df, shap.TreeExplainer(ch["lgbm"])


def platt(lr, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


ch, sc, gov, feats_df, explainer = load()
feats, lgbm = ch["features"], ch["lgbm"]
champ = gov["champion"]

st.set_page_config(page_title="Scoring crédit", page_icon="📊", layout="wide")
st.title("Scoring crédit — aide à la décision")
st.caption("Démonstration pédagogique sur données publiques (UCI, CC BY 4.0). "
           f"Modèle : {champ} calibré · seuil fixé par profit maximal (EMP).")

test_ids = feats_df.loc[feats_df.split == "test", "id"].tolist()
with st.sidebar:
    st.header("Dossier")
    pick = st.selectbox("Partir d'un client du jeu de test", test_ids[:200], index=0)
    base = feats_df.set_index("id").loc[pick]
    limit_bal = st.number_input("Plafond de crédit (NT$)", 10_000, 1_000_000, int(base["limit_bal"]), step=10_000)
    education = st.selectbox("Niveau d'études", [1, 2, 3, 4], index=[1, 2, 3, 4].index(int(base["education"])),
                             format_func={1: "Master/doctorat", 2: "Université", 3: "Lycée", 4: "Autre"}.get)
    st.subheader("Historique (6 derniers mois)")
    rows = []
    for m in range(1, 7):
        with st.expander(f"{MONTHS[m - 1]} 2005", expanded=(m == 1)):
            status = st.selectbox("Statut", list(STATUS_LABELS), key=f"s{m}",
                                  index=list(STATUS_LABELS).index(int(base[f"pay_status_{m}"])),
                                  format_func=STATUS_LABELS.get)
            bill = st.number_input("Facture", -200_000, 1_000_000, int(base[f"bill_{m}"]), key=f"b{m}")
            paid = st.number_input("Paiement", 0, 1_000_000, int(base[f"paid_{m}"]), key=f"p{m}")
        rows.append((status, bill, paid))

raw = {"id": 0, "limit_bal": limit_bal, "education": education, "sex": 1, "marriage": 1, "age": 30}
for m, (s, b, p) in enumerate(rows, start=1):
    raw.update({f"pay_status_{m}": s, f"bill_{m}": b, f"paid_{m}": p})
x = build_features(pd.DataFrame([raw]))[feats]

raw_pd = float(lgbm.predict_proba(x)[:, 1][0])
pd_cal = float(platt(ch["platt"][champ], np.array([raw_pd]))[0])
grade = next(g for g, lo, hi in GRADES if lo <= pd_cal < hi)
cutoff = gov["decision"]["cutoff_pd"]
q = {int(k): v for k, v in gov["conformal"]["thresholds"].items()}
conf = ("good" if raw_pd <= q[0] and not (1 - raw_pd <= q[1]) else
        "default" if (1 - raw_pd <= q[1]) and not raw_pd <= q[0] else "refer")
points = float(sc["scorecard"].score(x[sc["variables"]])[0])
ead = max(raw["bill_1"], 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Probabilité de défaut", f"{pd_cal:.1%}")
c2.metric("Note", grade)
c3.metric("Score (grille)", f"{points:.0f} pts")
c4.metric("Perte attendue", f"{pd_cal * LGD * ead:,.0f} NT$", help="PD × LGD (45 %, hypothèse) × EAD (dernière facture, hypothèse)")

approve = pd_cal < cutoff
decision = "✅ Accord" if approve else "❌ Refus"
# Automatic only when the conformal prediction is a single label that agrees with the profit rule.
agrees = (conf == "good" and approve) or (conf == "default" and not approve)
if not agrees:
    why = ("la prédiction conforme est ambiguë" if conf == "refer"
           else "le modèle classe ce dossier en défaut avec confiance, alors que la règle de profit accorde"
           if approve else "le modèle classe ce dossier en bon payeur avec confiance, alors que la règle de profit refuse")
    st.warning(f"**{decision} proposé — à valider par un analyste** ({why}).")
elif approve:
    st.success(f"**{decision} automatique** — décision à confiance élevée.")
else:
    st.error(f"**{decision} automatique** — décision à confiance élevée.")

sv = explainer.shap_values(x)
sv = sv[1] if isinstance(sv, list) else sv
contrib = pd.Series(sv[0], index=feats).sort_values(ascending=False)
st.subheader("Principaux facteurs de risque")
for f, v in contrib.head(3).items():
    if v > 0:
        st.write(f"• {REASONS_FR.get(f, f)}  (+{v:.2f} log-odds)")
st.subheader("Principaux facteurs favorables")
for f, v in contrib.tail(3)[::-1].items():
    if v < 0:
        st.write(f"• {FAVORABLE_FR.get(f, f)}  ({v:.2f} log-odds)")
st.caption("Explications SHAP du modèle LightGBM à contraintes monotones. "
           "Le sexe, l'âge et la situation familiale ne sont pas utilisés par le modèle.")
