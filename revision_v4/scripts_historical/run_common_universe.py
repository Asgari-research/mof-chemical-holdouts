from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, ShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "results" / "global_processed" / "assembled_master_table.csv"
OUT = ROOT / "revision" / "common_universe"
CHECKPOINTS = OUT / "checkpoints"
TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
FEATURES = [
    "ASA", "AVA", "AVAf", "Density", "Df", "Di", "Dif", "NASA", "POAVA", "POAVAf",
    "UC_volume", "cavity_window_gap", "density_pld_interaction", "lcd_pld_ratio",
    "log_lcd_plus1", "log_pld_plus1", "porosity_window_product", "sa_pv_ratio", "vASA",
    "vf_density_ratio",
]
GROUPS = {
    "random": None,
    "geo_grouped": "geo_cluster",
    "metal_grouped": "metal_cluster",
    "func_grouped": "func_cluster",
    "linker_grouped": "linker_cluster",
    "topology_grouped": "topology_label",
}
LABEL_COLUMNS = [c for c in GROUPS.values() if c is not None]
_DATA: pd.DataFrame | None = None


def top5_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(0.05 * n)))
    a = set(np.argsort(-y_true)[:k])
    b = set(np.argsort(-y_pred)[:k])
    recovery = len(a & b) / k
    return recovery, recovery / (k / n)


def load_common() -> pd.DataFrame:
    usecols = ["filename_norm", TARGET, *FEATURES, *LABEL_COLUMNS]
    df = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
    common = df.loc[df[LABEL_COLUMNS].notna().all(axis=1) & df[TARGET].notna()].copy()
    if len(common) != 214_531:
        raise RuntimeError(f"Fixed-universe gate failed: expected 214531 rows, found {len(common)}")
    if not common["filename_norm"].is_unique:
        raise RuntimeError("Fixed-universe IDs are not unique")
    return common.reset_index(drop=True)


def init_worker() -> None:
    global _DATA
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _DATA = load_common()


def build_split(df: pd.DataFrame, split_name: str, fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    if split_name == "random":
        tr, te = list(ShuffleSplit(n_splits=5, test_size=0.20, random_state=42).split(df))[fold_id]
        return tr, te
    group_col = GROUPS[split_name]
    assert group_col is not None
    tr, te = list(GroupKFold(n_splits=5).split(df, groups=df[group_col].to_numpy()))[fold_id]
    return tr, te


def pipeline_and_grid() -> tuple[Pipeline, dict]:
    preprocess = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), FEATURES)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=1,
        bootstrap=True,
        max_samples=0.70,
        max_features=0.70,
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    grid = {
        "model__n_estimators": [50, 80],
        "model__max_depth": [12, 20],
        "model__min_samples_leaf": [5],
    }
    return pipe, grid


def run_job(job: dict) -> dict:
    assert _DATA is not None
    split_name = job["split_family"]
    fold_id = int(job["fold_id"])
    start = time.time()
    tr, te = build_split(_DATA, split_name, fold_id)
    train = _DATA.iloc[tr]
    test = _DATA.iloc[te]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    pipe, grid = pipeline_and_grid()
    if split_name == "random":
        inner = KFold(n_splits=3, shuffle=True, random_state=42)
        groups_train = None
    else:
        group_col = GROUPS[split_name]
        assert group_col is not None
        groups_train = train[group_col].to_numpy()
        inner = GroupKFold(n_splits=min(3, pd.Series(groups_train).nunique()))
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=inner,
        n_jobs=1,
        refit=True,
        verbose=0,
        error_score=np.nan,
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if groups_train is None:
            search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train, groups=groups_train)
    pred = search.best_estimator_.predict(X_test)
    recovery, enrichment = top5_metrics(y_test, pred)
    result = {
        "target_slug": "co2_015bar",
        "descriptor_family": "enriched_interpretable",
        "model": "rf",
        "analysis": "fixed_common_universe_retain_singletons",
        "split_family": split_name,
        "fold_id": fold_id,
        "common_universe_n": len(_DATA),
        "n_train": len(train),
        "n_test": len(test),
        "n_train_groups": None if split_name == "random" else int(train[GROUPS[split_name]].nunique()),
        "n_test_groups": None if split_name == "random" else int(test[GROUPS[split_name]].nunique()),
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "spearman_rho": stats.spearmanr(y_test, pred).statistic,
        "kendall_tau": stats.kendalltau(y_test, pred).statistic,
        "top_5pct_overlap": recovery,
        "top_5pct_enrichment": enrichment,
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "best_inner_mae": -float(search.best_score_),
        "fit_seconds": time.time() - start,
    }
    cp = CHECKPOINTS / f"{split_name}__fold{fold_id}.json"
    cp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def universe_manifest(common: pd.DataFrame) -> dict:
    ids = common["filename_norm"].astype(str)
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    rows = []
    for split, col in GROUPS.items():
        if col is None:
            rows.append({"split_family": split, "common_universe_n": len(common), "groups": np.nan, "singleton_groups_retained": np.nan})
        else:
            counts = common[col].value_counts(dropna=False)
            rows.append({
                "split_family": split,
                "common_universe_n": len(common),
                "groups": int(len(counts)),
                "singleton_groups_retained": int((counts == 1).sum()),
            })
    pd.DataFrame(rows).to_csv(OUT / "common_universe_group_counts.csv", index=False)
    common[["filename_norm"]].to_csv(OUT / "common_universe_ids.csv", index=False)
    manifest = {
        "n": len(common),
        "unique_ids": int(ids.nunique()),
        "id_order_sha256": digest,
        "label_columns": LABEL_COLUMNS,
        "singleton_groups_retained": True,
    }
    (OUT / "common_universe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    common = load_common()
    manifest = universe_manifest(common)
    print(json.dumps(manifest, indent=2))
    jobs = [{"split_family": split, "fold_id": fold} for split in GROUPS for fold in range(5)]
    pending = [j for j in jobs if not (CHECKPOINTS / f"{j['split_family']}__fold{j['fold_id']}.json").exists()]
    print(f"common-universe checkpoints: {30 - len(pending)}/30 complete; launching {len(pending)}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(run_job, job): job for job in pending}
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"DONE {result['split_family']} fold {result['fold_id']} "
                    f"R2={result['r2']:.4f} MAE={result['mae']:.4f} seconds={result['fit_seconds']:.1f}",
                    flush=True,
                )
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CHECKPOINTS.glob("*.json"))]
    results = pd.DataFrame(rows).sort_values(["split_family", "fold_id"])
    if len(results) != 30:
        raise RuntimeError(f"Expected 30 checkpoints, found {len(results)}")
    if not (results["common_universe_n"] == 214_531).all():
        raise RuntimeError("Fixed-universe N differs across completed experiments")
    results.to_csv(OUT / "common_universe_fold_metrics.csv", index=False)
    summary = results.groupby("split_family", as_index=False).agg(
        n_folds=("fold_id", "count"),
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        spearman_mean=("spearman_rho", "mean"), spearman_sd=("spearman_rho", "std"),
        kendall_mean=("kendall_tau", "mean"), kendall_sd=("kendall_tau", "std"),
        top5_recovery_mean=("top_5pct_overlap", "mean"), top5_recovery_sd=("top_5pct_overlap", "std"),
        top5_enrichment_mean=("top_5pct_enrichment", "mean"), top5_enrichment_sd=("top_5pct_enrichment", "std"),
    )
    anchor = pd.read_csv(ROOT / "results" / "tables" / "table_main_benchmark_summary.csv")
    anchor = anchor[
        (anchor["target_slug"] == "co2_015bar")
        & (anchor["descriptor_family"] == "enriched_interpretable")
        & (anchor["model"].str.lower() == "rf")
        & (anchor["split_family"].isin(GROUPS))
    ].copy()
    comparison = summary.merge(anchor, on="split_family", how="left", suffixes=("_common", "_submitted"))
    comparison.to_csv(OUT / "common_universe_vs_submitted.csv", index=False)
    summary.to_csv(OUT / "common_universe_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    main(max(1, min(args.workers, 6)))
