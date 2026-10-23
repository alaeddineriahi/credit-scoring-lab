"""Reproduce every number in the README from scratch: python run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["download.py", "features.py", "scorecard.py", "models.py", "governance.py", "figures.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
print("\nOptional research extension (slow on CPU): python src/tabpfn_experiment.py")
