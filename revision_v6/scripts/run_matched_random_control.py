"""Five bounded RF fits for the training-size-matched random control.

This script never invokes the historical benchmark. Completed checkpoints are
validated and reused. Run sequentially to keep memory bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


V6 = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOF_V6_SOURCE_ROOT", V6.parent)).resolve()
OUT = V6 / "results" / "matched_random_control"
CHECKPOINTS = OUT / "checkpoints"
MASTER = ROOT / "results/global_processed/assembled_master_table.csv"
METRICS = ROOT / "results/metrics/all_fold_metrics_long.csv"
TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
FEATURES = [
    "ASA", "AVA", "AVAf", "Density", "Df", "Di", "Dif", "NASA", "POAVA", "POAVAf",
    "UC_volume", "cavity_window_gap", "density_pld_interaction", "lcd_pld_ratio",
    "log_lcd_plus1", "log_pld_plus1", "porosity_window_product", "sa_pv_ratio", "vASA",
    "vf_density_ratio",
]


def digest_ids(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str)).encode("utf-8")).hexdigest()


def top5_recovery(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = max(1, int(np.ceil(0.05 * len(y_true))))
    true_top = set(np.argsort(-y_true, kind="stable")[:k])
    pred_top = set(np.argsort(-y_pred, kind="stable")[:k])
    return len(true_top & pred_top) / k


def references() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    ref = metrics[
        (metrics["target_slug"] == "co2_015bar")
        & (metrics["descriptor_family"] == "enriched_interpretable")
        & (metrics["split_family"] == "topology_grouped")
        & (metrics["model"] == "rf")
    ].sort_values("fold_id")
    if len(ref) != 5:
        raise RuntimeError(f"Expected five topology comparator folds, found {len(ref)}")
    return ref


def load_data() -> pd.DataFrame:
    usecols = ["filename_norm", TARGET, *FEATURES]
    df = pd.read_csv(MASTER, usecols=usecols, low_memory=False)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if len(df) != 263744 or not df["filename_norm"].is_unique:
        raise RuntimeError(f"Universe gate failed: rows={len(df)}, unique={df['filename_norm'].nunique()}")
    return df


def validate_checkpoint(data: dict, ref: pd.Series) -> None:
    required = {"fold_id", "n_train", "n_test", "train_ids_sha256", "test_ids_sha256", "r2", "mae"}
    if not required <= data.keys():
        raise RuntimeError("Checkpoint missing required fields")
    if int(data["fold_id"]) != int(ref["fold_id"]):
        raise RuntimeError("Checkpoint fold mismatch")
    if int(data["n_train"]) != int(ref["n_train"]) or int(data["n_test"]) != int(ref["n_test"]):
        raise RuntimeError("Checkpoint population mismatch")


def run_fold(df: pd.DataFrame, ref: pd.Series) -> dict:
    fold = int(ref["fold_id"])
    checkpoint = CHECKPOINTS / f"matched_random__fold{fold}.json"
    if checkpoint.exists():
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        validate_checkpoint(data, ref)
        print(f"reuse fold {fold}")
        return data

    seed = 20261020 + fold
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_test = int(ref["n_test"])
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    if len(train_idx) != int(ref["n_train"]) or len(set(train_idx) & set(test_idx)):
        raise RuntimeError("Strict OOS or size gate failed")
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    params = {k.replace("model__", ""): v for k, v in json.loads(ref["best_params"]).items()}
    pipe = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), FEATURES),
        ], remainder="drop", sparse_threshold=0.3)),
        ("model", RandomForestRegressor(
            random_state=42, n_jobs=1, bootstrap=True, max_samples=0.70, max_features=0.70,
            **params,
        )),
    ])
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(train[FEATURES], train[TARGET].to_numpy())
    pred = pipe.predict(test[FEATURES])
    y = test[TARGET].to_numpy()
    recovery = top5_recovery(y, pred)
    result = {
        "analysis": "training_size_matched_random_control",
        "target_slug": "co2_015bar",
        "descriptor_family": "enriched_interpretable",
        "model": "rf",
        "fold_id": fold,
        "split_seed": seed,
        "universe_n": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "strict_oos_overlap_n": 0,
        "train_ids_sha256": digest_ids(train["filename_norm"]),
        "test_ids_sha256": digest_ids(test["filename_norm"]),
        "best_params_source": "paired topology fold; reused without retuning",
        "best_params": ref["best_params"],
        "r2": r2_score(y, pred),
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "spearman_rho": stats.spearmanr(y, pred).statistic,
        "kendall_tau": stats.kendalltau(y, pred).statistic,
        "top_5pct_recovery": recovery,
        "fit_seconds": time.time() - started,
        "python_pid": os.getpid(),
    }
    temp = checkpoint.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(checkpoint)
    print(f"completed fold {fold} in {result['fit_seconds']:.1f}s")
    return result


def write_summary(results: list[dict], ref: pd.DataFrame) -> None:
    controls = pd.DataFrame(results).sort_values("fold_id")
    compare = controls.merge(
        ref[["fold_id", "r2", "mae", "rmse", "spearman_rho", "kendall_tau", "top_5pct_recall"]],
        on="fold_id", suffixes=("_matched_random", "_topology"), validate="one_to_one",
    )
    compare["r2_advantage_random_minus_topology"] = compare["r2_matched_random"] - compare["r2_topology"]
    compare["mae_advantage_topology_minus_random"] = compare["mae_topology"] - compare["mae_matched_random"]
    compare["top5_advantage_random_minus_topology"] = compare["top_5pct_recovery"] - compare["top_5pct_recall"]
    compare.to_csv(OUT / "matched_random_vs_topology_fold_metrics.csv", index=False)
    numeric = ["r2_advantage_random_minus_topology", "mae_advantage_topology_minus_random", "top5_advantage_random_minus_topology"]
    summary = compare[numeric].agg(["mean", "std", "min", "max"]).T.reset_index(names="contrast")
    summary.to_csv(OUT / "matched_random_vs_topology_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    x = np.arange(len(compare))
    width = 0.36
    axes[0].bar(x - width / 2, compare["r2_matched_random"], width, label="Matched random", color="#4C78A8")
    axes[0].bar(x + width / 2, compare["r2_topology"], width, label="Topology", color="#E45756")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="$R^2$", xlabel="Paired fold", xticks=x, xticklabels=compare["fold_id"], title="Held-out fit")
    axes[1].bar(x - width / 2, compare["top_5pct_recovery"], width, color="#4C78A8")
    axes[1].bar(x + width / 2, compare["top_5pct_recall"], width, color="#E45756")
    axes[1].set(ylabel="Top-5% recovery", xlabel="Paired fold", xticks=x, xticklabels=compare["fold_id"], title="Screening recovery")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Training-size-matched random controls vs topology holdout")
    fig.tight_layout()
    fig.savefig(V6 / "figures/matched_random_vs_topology.png", dpi=200)
    fig.savefig(V6 / "figures/matched_random_vs_topology.pdf")
    plt.close(fig)


def main(folds: list[int] | None) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ref = references()
    df = load_data()
    selected = list(range(5)) if folds is None else folds
    results = []
    for fold in range(5):
        row = ref[ref["fold_id"] == fold].iloc[0]
        if fold in selected:
            results.append(run_fold(df, row))
        else:
            cp = CHECKPOINTS / f"matched_random__fold{fold}.json"
            if cp.exists():
                data = json.loads(cp.read_text(encoding="utf-8"))
                validate_checkpoint(data, row)
                results.append(data)
    if len(results) == 5:
        write_summary(results, ref)
    print(f"valid checkpoints: {len(results)}/5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, action="append", choices=range(5))
    args = parser.parse_args()
    main(args.fold)
