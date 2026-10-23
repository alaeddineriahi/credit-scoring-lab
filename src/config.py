"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

RAW_XLS = RAW_DIR / "default of credit card clients.xls"
DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

SEED = 42
TARGET = "default"
ID_COL = "id"

# Protected attributes: never model inputs, used only for the fairness audit.
PROTECTED = ["sex", "marriage", "age"]

# Split proportions (train / validation / test), stratified on the target.
SPLIT = (0.6, 0.2, 0.2)

for d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
