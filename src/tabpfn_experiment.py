"""Research extension: tabular foundation model (TabPFN) with different context-construction strategies.

TabPFN does not train on the data: it reads a "context" of labelled examples and predicts in a single
forward pass (Hollmann et al., Nature 2025; TabPFN-2.5, arXiv:2511.08667). Tanna et al. (May 2026,
arXiv:2605.18635) report that on Home Credit and Lending Club, *how the context is sampled* matters
more than *which* foundation model is used, with balanced sampling beating uniform sampling.
This script checks whether that finding transfers to the UCI credit-card data.

Protocol: strategies x context sizes are compared on the validation set (repeated with 3 seeds);
only the best configuration is scored on the test set, once.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import PROCESSED_DIR, ROOT, SEED, TARGET
from evaluate import bootstrap_auc_ci, metrics
from features import feature_columns

# CPU-only budget: contexts up to 2,000 examples, 2 seeds, validation sweep on a fixed stratified
# subsample of 1,500 rows. The final test evaluation uses the full test set.
SIZES = [500, 1000, 2000]
STRATEGIES = ["uniform", "balanced"]
SEEDS = [SEED, SEED + 1]
VALID_SUBSAMPLE = 1500


def sample_context(train: pd.DataFrame, size: int, strategy: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        idx = rng.choice(train.index, size=min(size, len(train)), replace=False)
        return train.loc[idx]
    pos, neg = train[train[TARGET] == 1], train[train[TARGET] == 0]
    half = size // 2
    idx = np.concatenate([rng.choice(pos.index, size=min(half, len(pos)), replace=False),
                          rng.choice(neg.index, size=size - min(half, len(pos)), replace=False)])
    return train.loc[idx]


def make_model():
    # TabPFN v2 (the Nature 2025 model): its weights are openly downloadable. Newer versions
    # (2.5+) require a Prior Labs account and licence token.
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=SEED, ignore_pretraining_limits=True)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feats = feature_columns(df)
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    va = va.groupby(TARGET, group_keys=False).sample(frac=VALID_SUBSAMPLE / len(va), random_state=SEED)

    rows = []
    for size in SIZES:
        for strategy in STRATEGIES:
            aucs = []
            for seed in SEEDS:
                ctx = sample_context(tr, size, strategy, seed)
                t0 = time.time()
                model = make_model().fit(ctx[feats], ctx[TARGET])
                p = model.predict_proba(va[feats])[:, 1]
                aucs.append(roc_auc_score(va[TARGET], p))
                print(f"size={size:5d} {strategy:8s} seed={seed} AUC={aucs[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
            rows.append({"context_size": size, "strategy": strategy,
                         "valid_auc_mean": round(float(np.mean(aucs)), 4),
                         "valid_auc_std": round(float(np.std(aucs)), 4)})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))

    best = sweep.sort_values("valid_auc_mean", ascending=False).iloc[0]
    ctx = sample_context(tr, int(best.context_size), best.strategy, SEED)
    p_test = make_model().fit(ctx[feats], ctx[TARGET]).predict_proba(te[feats])[:, 1]
    m = metrics(te[TARGET], p_test)
    m["auc_ci95"] = bootstrap_auc_ci(te[TARGET], p_test)
    print("Best config on test:", dict(best), m)

    out = {"sweep_validation": rows, "best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
           "test_metrics_best": m,
           "note": "Brier/log-loss are not comparable for balanced contexts (shifted base rate); compare AUC/Gini/KS."}
    (ROOT / "reports" / "tabpfn_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
