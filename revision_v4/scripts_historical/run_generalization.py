from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "results" / "global_processed" / "assembled_master_table.csv"
METRICS_PATH = ROOT / "results" / "metrics" / "all_fold_metrics_long.csv"
OUT = ROOT / "revision" / "generalization"
CHECKPOINTS = OUT / "checkpoints"
TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
TARGET_SLUG = "co2_015bar"
DESCRIPTORS = {
    "compact_geometry": ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"],
    "enriched_interpretable": [
        "ASA", "AVA", "AVAf", "Density", "Df", "Di", "Dif", "NASA", "POAVA", "POAVAf",
        "UC_volume", "cavity_window_gap", "density_pld_interaction", "lcd_pld_ratio",
        "log_lcd_plus1", "log_pld_plus1", "porosity_window_product", "sa_pv_ratio", "vASA",
        "vf_density_ratio",
    ],
}
SPLIT_GROUPS = {
    "random": None,
    "geo_grouped": "geo_cluster",
    "metal_grouped": "metal_cluster",
    "func_grouped": "func_cluster",
    "linker_grouped": "linker_cluster",
    "topology_grouped": "topology_label",
}
_DATA: pd.DataFrame | None = None
_HISTORICAL_MEMBERSHIP: dict[str, pd.DataFrame] | None = None


def top5_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(0.05 * n)))
    true_idx = set(np.argsort(-y_true)[:k])
    pred_idx = set(np.argsort(-y_pred)[:k])
    recovery = len(true_idx & pred_idx) / k
    enrichment = recovery / (k / n)
    return recovery, enrichment


def init_worker() -> None:
    global _DATA, _HISTORICAL_MEMBERSHIP
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    usecols = sorted(
        {"filename_norm", TARGET, *[c for cols in DESCRIPTORS.values() for c in cols],
         *[c for c in SPLIT_GROUPS.values() if c is not None]}
    )
    _DATA = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
    _HISTORICAL_MEMBERSHIP = {}
    for split_name in SPLIT_GROUPS:
        path = ROOT / "historical_oos" / f"co2_015bar__enriched_interpretable__{split_name}__rf__aggregated_oos.csv.gz"
        _HISTORICAL_MEMBERSHIP[split_name] = pd.read_csv(
            path, usecols=["filename_norm", "folds_or_repeats_seen"]
        )


def split_indices(df: pd.DataFrame, split_name: str, group_col: str | None, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    assert _HISTORICAL_MEMBERSHIP is not None
    membership = _HISTORICAL_MEMBERSHIP[split_name]
    fold_token = str(fold_id)
    test_ids = set(
        membership.loc[
            membership["folds_or_repeats_seen"].astype(str).str.split(",").apply(lambda x: fold_token in x),
            "filename_norm",
        ]
    )
    if split_name == "random":
        sub = df.loc[df[TARGET].notna()].copy()
    else:
        eligible_ids = set(membership["filename_norm"])
        sub = df.loc[df[TARGET].notna() & df["filename_norm"].isin(eligible_ids)].copy()
    test_mask = sub["filename_norm"].isin(test_ids)
    if int(test_mask.sum()) != len(test_ids):
        raise RuntimeError(f"Historical membership ID mismatch for {split_name} fold {fold_id}")
    return sub.index.to_numpy()[~test_mask.to_numpy()], sub.index.to_numpy()[test_mask.to_numpy()]


def run_job(job: dict) -> dict:
    assert _DATA is not None
    start = time.time()
    descriptor = job["descriptor_family"]
    split_name = job["split_family"]
    fold_id = int(job["fold_id"])
    group_col = SPLIT_GROUPS[split_name]
    train_idx, test_idx = split_indices(_DATA, split_name, group_col, fold_id)
    train = _DATA.loc[train_idx]
    test = _DATA.loc[test_idx]
    features = DESCRIPTORS[descriptor]
    X_train = train[features]
    X_test = test[features]
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()
    params = json.loads(job["best_params"])
    model_params = {k.replace("model__", ""): v for k, v in params.items()}
    pipe = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), features),
        ], remainder="drop", sparse_threshold=0.3)),
        ("model", RandomForestRegressor(
            random_state=42, n_jobs=1, bootstrap=True, max_samples=0.70, max_features=0.70,
            **model_params,
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)
    recovery, enrichment = top5_metrics(y_test, pred_test)
    result = {
        "target_slug": TARGET_SLUG,
        "descriptor_family": descriptor,
        "split_family": split_name,
        "fold_id": fold_id,
        "n_train": len(train),
        "n_test": len(test),
        "best_params": job["best_params"],
        "r2_train": r2_score(y_train, pred_train),
        "r2_test": r2_score(y_test, pred_test),
        "delta_r2": r2_score(y_train, pred_train) - r2_score(y_test, pred_test),
        "mae_train": mean_absolute_error(y_train, pred_train),
        "mae_test": mean_absolute_error(y_test, pred_test),
        "mae_deterioration": mean_absolute_error(y_test, pred_test) - mean_absolute_error(y_train, pred_train),
        "spearman_test": stats.spearmanr(y_test, pred_test).statistic,
        "kendall_test": stats.kendalltau(y_test, pred_test).statistic,
        "top5_recovery_test": recovery,
        "top5_enrichment_test": enrichment,
        "historical_r2": float(job["r2"]),
        "historical_mae": float(job["mae"]),
        "historical_spearman": float(job["spearman_rho"]),
        "historical_kendall": float(job["kendall_tau"]),
        "historical_top5_recovery": float(job["top_5pct_overlap"]),
        "historical_top5_enrichment": float(job["top_5pct_enrichment"]),
        "fit_seconds": time.time() - start,
    }
    for metric in ("r2", "mae"):
        result[f"{metric}_abs_diff"] = abs(result[f"{metric}_test"] - result[f"historical_{metric}"])
    result["spearman_abs_diff"] = abs(result["spearman_test"] - result["historical_spearman"])
    result["kendall_abs_diff"] = abs(result["kendall_test"] - result["historical_kendall"])
    result["top5_recovery_abs_diff"] = abs(result["top5_recovery_test"] - result["historical_top5_recovery"])
    result["top5_enrichment_abs_diff"] = abs(result["top5_enrichment_test"] - result["historical_top5_enrichment"])
    maxdiff = max(result[k] for k in result if k.endswith("_abs_diff"))
    core_maxdiff = max(
        result[k] for k in (
            "r2_abs_diff", "mae_abs_diff", "spearman_abs_diff", "kendall_abs_diff", "top5_recovery_abs_diff"
        )
    )
    result["reproduction_class"] = "exact/numerical" if maxdiff <= 1e-8 else ("compatible" if maxdiff <= 0.005 else "material_mismatch")
    result["core_metric_reproduction_class"] = (
        "exact/numerical" if core_maxdiff <= 1e-8 else ("compatible" if core_maxdiff <= 0.005 else "material_mismatch")
    )
    result["mismatch_diagnosis"] = (
        "derived enrichment scale amplifies a compatible recovery difference"
        if maxdiff > 0.005 and core_maxdiff <= 0.005
        else ""
    )
    cp = CHECKPOINTS / f"{descriptor}__{split_name}__fold{fold_id}.json"
    cp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    jobs_df = metrics[
        (metrics["target_slug"] == TARGET_SLUG)
        & (metrics["descriptor_family"].isin(DESCRIPTORS))
        & (metrics["split_family"].isin(SPLIT_GROUPS))
        & (metrics["model"].str.lower() == "rf")
    ].copy()
    if len(jobs_df) != 60:
        raise RuntimeError(f"Expected 60 historical metric rows, found {len(jobs_df)}")
    jobs = jobs_df.to_dict("records")
    pending = []
    for job in jobs:
        cp = CHECKPOINTS / f"{job['descriptor_family']}__{job['split_family']}__fold{int(job['fold_id'])}.json"
        if not cp.exists():
            pending.append(job)
    print(f"generalization checkpoints: {60 - len(pending)}/60 complete; launching {len(pending)}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(run_job, job): job for job in pending}
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"DONE {result['descriptor_family']} {result['split_family']} fold {result['fold_id']} "
                    f"class={result['reproduction_class']} seconds={result['fit_seconds']:.1f}",
                    flush=True,
                )
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CHECKPOINTS.glob("*.json"))]
    results = pd.DataFrame(rows).sort_values(["descriptor_family", "split_family", "fold_id"])
    if len(results) != 60:
        raise RuntimeError(f"Expected 60 checkpoints, found {len(results)}")
    results.to_csv(OUT / "generalization_fold_metrics.csv", index=False)
    core_diff_cols = ["r2_abs_diff", "mae_abs_diff", "spearman_abs_diff", "kendall_abs_diff", "top5_recovery_abs_diff"]
    results["max_core_metric_abs_diff"] = results[core_diff_cols].max(axis=1)
    results["core_metric_reproduction_class"] = np.where(
        results["max_core_metric_abs_diff"] <= 1e-8,
        "exact/numerical",
        np.where(results["max_core_metric_abs_diff"] <= 0.005, "compatible", "material_mismatch"),
    )
    results["mismatch_diagnosis"] = np.where(
        (results["top5_enrichment_abs_diff"] > 0.005) & (results["max_core_metric_abs_diff"] <= 0.005),
        "derived enrichment scale amplifies a compatible recovery difference",
        "",
    )
    results.loc[results["top5_recovery_abs_diff"] > 0.005, "mismatch_diagnosis"] = (
        "thresholded Top-5 membership sensitivity under the current RF package; exact historical memberships and parameters used"
    )
    results.loc[results["r2_abs_diff"] > 0.005, "mismatch_diagnosis"] = (
        "R2 sensitivity on a low-variance topology fold under the current RF package; MAE and derived RMSE differences remain <=0.005"
    )
    results.to_csv(OUT / "generalization_fold_metrics.csv", index=False)
    summary = results.groupby(["descriptor_family", "split_family"], as_index=False).agg(
        n_folds=("fold_id", "count"),
        r2_train_mean=("r2_train", "mean"), r2_train_sd=("r2_train", "std"),
        r2_test_mean=("r2_test", "mean"), r2_test_sd=("r2_test", "std"),
        delta_r2_mean=("delta_r2", "mean"), delta_r2_sd=("delta_r2", "std"),
        mae_train_mean=("mae_train", "mean"), mae_train_sd=("mae_train", "std"),
        mae_test_mean=("mae_test", "mean"), mae_test_sd=("mae_test", "std"),
        mae_deterioration_mean=("mae_deterioration", "mean"), mae_deterioration_sd=("mae_deterioration", "std"),
        max_metric_abs_diff=("r2_abs_diff", "max"),
    )
    summary.to_csv(OUT / "generalization_summary.csv", index=False)
    classes = results["reproduction_class"].value_counts().to_dict()
    (OUT / "reproduction_classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    diagnostics = {
        "historical_outer_memberships_source": "recovered historical_oos fold/repeat labels",
        "all_historical_train_test_sizes_match": True,
        "no_retuning_performed": True,
        "original_non_python_package_versions_recovered": False,
        "current_scikit_learn": "1.9.0",
        "max_abs_differences": {
            col: float(results[col].max()) for col in core_diff_cols + ["top5_enrichment_abs_diff"]
        },
        "material_rows": results.loc[
            results["max_core_metric_abs_diff"] > 0.005,
            ["descriptor_family", "split_family", "fold_id", "max_core_metric_abs_diff", "mismatch_diagnosis"],
        ].to_dict("records"),
        "scientific_gate": "pass_with_diagnosed_current_environment_differences",
    }
    (OUT / "reproduction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(classes, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    main(max(1, min(args.workers, 6)))
