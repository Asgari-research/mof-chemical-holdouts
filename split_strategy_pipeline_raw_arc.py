#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 1 pipeline: Split Strategy Matters in MOF Machine Learning
================================================================

Purpose
-------
This script is a more manuscript-facing and visually stronger revision of the
original split-strategy benchmark pipeline. It is designed to run in VS Code,
Spyder, or a standard Python environment, and to save intermediate outputs so
the project can be resumed after interruption.

Scientific scope
----------------
The central question is not which descriptor family "wins" in an absolute sense,
but how the scientific conclusion of a MOF benchmark changes when the evaluation
protocol changes from interpolation-like random splits to deployment-relevant
grouped extrapolation splits. The pipeline therefore keeps models and descriptor
families matched while varying the split family.

Expected input files
--------------------
Required:
- clean_data.csv
- geo-clusters.csv
- mc-clusters.csv
- func-clusters.csv
- flig-clusters.csv

Optional:
- geometric_properties.csv
- all_topology_lists.csv

The script does not require internet access once the CSV files are present.
"""

from __future__ import annotations

import json
import math
import os
import gc
import pickle
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import combinations, permutations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# SECTION 1. USER CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

PIPELINE_VARIANT = "Version_2"
OUTPUT_DIR = BASE_DIR / f"paper1_split_strategy_outputs_{PIPELINE_VARIANT}"
MEMORY_SAFE_PATCH = "v7_polished_split_consequence_figures"

RANDOM_SEED = 42
OUTER_RANDOM_SPLITS = 5
OUTER_GROUP_FOLDS = 5
TEST_SIZE = 0.20
INNER_CV_RANDOM = 3
INNER_CV_GROUP = 3
TOP_K_FRACTIONS = [0.01, 0.02, 0.05, 0.10]

RESUME_IF_AVAILABLE = True
OVERWRITE_FIGURES = True
# If True, tables/figures/LaTeX are regenerated even when the manifest says those stages are complete.
# This is useful for post-hoc analysis/plotting revisions without rerunning expensive ML fits.
FORCE_REBUILD_TABLES = False
# Publication-table exports can be regenerated from existing tables without rebuilding all analysis tables.
FORCE_REBUILD_PUBLICATION_TABLES = True
FORCE_REBUILD_FIGURES = True
FORCE_REBUILD_LATEX = True
MIN_GROUP_SIZE_FOR_ANALYSIS = 5
N_BOOTSTRAP = 1000
MAX_ROWS_FOR_VISUAL_PCA = 20000

# MAX_CATEGORICAL_LEVELS_FOR_ONEHOT caps rare topology/category expansion and
# prevents dense HGB/MLP design matrices from becoming unexpectedly huge.
MAX_CATEGORICAL_LEVELS_FOR_ONEHOT = 128
PERMUTATION_IMPORTANCE_REPEATS = 1
MAX_ROWS_FOR_PERMUTATION_IMPORTANCE = 3000

# Full out-of-fold predictions are saved per experiment. To avoid MemoryError,
# the script does NOT concatenate all prediction files into one giant table.
# Instead, it keeps a registry plus a reproducible sample for figures/tables that
# need point-level predictions.
MAX_ROWS_PER_PREDICTION_FILE_FOR_FIGURES = 500

# Exact post-hoc analyses use the per-experiment prediction CSV files and stream/aggregate
# them in small pieces. These analyses do not rerun the ML models.
EXACT_POSTHOC_CHUNKSIZE = 200000
EXACT_GROUP_MIN_N = 20
EXACT_GROUP_TARGET_SLUGS = ["co2_015bar"]
EXACT_GROUP_DESCRIPTOR_FAMILIES = ["enriched_interpretable", "geometry_plus_topology"]
EXACT_GROUP_MODELS = ["rf", "hgb"]
EXACT_ELITE_TARGET_SLUGS = ["co2_0015bar", "co2_015bar", "ch4_58bar", "ch4_65bar"]
EXACT_ELITE_DESCRIPTOR_FAMILIES = ["compact_geometry", "enriched_interpretable", "geometry_plus_topology"]
EXACT_ELITE_MODELS = ["rf", "hgb"]
CASE_STUDY_TARGET_SLUG = "co2_015bar"
CASE_STUDY_DESCRIPTOR_FAMILY = "enriched_interpretable"
CASE_STUDY_MODEL = "rf"

# CPU / parallelism control.
# Set N_JOBS to the number of processors you want to use.
N_JOBS = 1

# Anchor-target paper framing:
TARGET_COLUMNS_ALL = [
    "uptake(mmol/g) CO2 at 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar",
    "uptake(mmol/g) methane at 65 bar",
]
ANCHOR_TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
SECONDARY_TARGET = "uptake(mmol/g) methane at 5.8 bar"
RUN_ALL_TARGETS = True
TARGET_COLUMNS = TARGET_COLUMNS_ALL if RUN_ALL_TARGETS else [ANCHOR_TARGET, SECONDARY_TARGET]

TARGET_SLUGS = {
    "uptake(mmol/g) CO2 at 0.015 bar": "co2_0015bar",
    "uptake(mmol/g) CO2 at 0.15 bar": "co2_015bar",
    "uptake(mmol/g) methane at 5.8 bar": "ch4_58bar",
    "uptake(mmol/g) methane at 65 bar": "ch4_65bar",
}

REQUIRED_CLEAN_COLUMNS = [
    "filename", "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
    "uptake(mmol/g) CO2 at 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar",
    "uptake(mmol/g) methane at 65 bar",
]
REQUIRED_CLUSTER_COLUMNS = ["filename", "cluster_id"]

ID_COL = "filename"
TOPOLOGY_COLUMN_RAW = "Crystalnet"
TOPOLOGY_COLUMN_GROUPED = "topology_label"

BASE_GEOMETRY_COLUMNS = [
    "UC_volume", "Density", "ASA", "vASA", "gASA", "NASA", "gNASA", "vNASA",
    "AVA", "AVAf", "AVAg", "NAVA", "NAVAf", "NAVAg", "POAVA", "POAVAf",
    "POAVAg", "NPOAVA", "NPOAVAf", "NPOAVAg", "Di", "Df", "Dif"
]


COMPACT_GEOMETRY_COLUMNS = ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"]

ENGINEERED_COLUMNS = [
    "lcd_pld_ratio",
    "cavity_window_gap",
    "sa_pv_ratio",
    "vf_density_ratio",
    "log_pld_plus1",
    "log_lcd_plus1",
    "porosity_window_product",
    "density_pld_interaction",
]

DISPLAY_SPLIT_NAMES = {
    "random": "Random",
    "geo_grouped": "Geometry-grouped",
    "metal_grouped": "Metal-grouped",
    "func_grouped": "Functional-group-grouped",
    "linker_grouped": "Linker-grouped",
    "topology_grouped": "Topology-grouped",
    "chemistry_ensemble": "Chemistry ensemble",
}
DISPLAY_DESCRIPTOR_NAMES = {
    "compact_geometry": "Compact geometry",
    "enriched_interpretable": "Enriched interpretable",
    "topology_only": "Topology only",
    "geometry_plus_topology": "Geometry + topology",
}


# =============================================================================
# SECTION 2. SMALL UTILITIES
# =============================================================================

class DualLogger:
    """Simple logger writing to terminal and file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_filename_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".cif", "", regex=False).str.strip()


def check_required_columns(df: pd.DataFrame, required: Sequence[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def resolve_topology_identifier_column(df: pd.DataFrame) -> str:
    for candidate in ["Name", "filename"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("all_topology_lists.csv must contain either 'Name' or 'filename' for identifier normalization.")


def safe_pickle_dump(obj, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def safe_json_dump(obj, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def read_json_if_exists(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def copy_log_snapshot(log_file: Path, snapshot_name: str) -> None:
    if log_file.exists():
        shutil.copy2(log_file, log_file.parent / snapshot_name)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def kendall_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    tau, _ = stats.kendalltau(y_true, y_pred)
    return float(tau)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute a simple NDCG@k for ranking-oriented screening quality."""
    if len(y_true) == 0:
        return np.nan
    k = max(1, min(k, len(y_true)))
    pred_order = np.argsort(-y_pred)[:k]
    ideal_order = np.argsort(-y_true)[:k]
    gains = np.maximum(y_true[pred_order], 0.0)
    ideal_gains = np.maximum(y_true[ideal_order], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts)
    idcg = np.sum(ideal_gains * discounts)
    return float(dcg / idcg) if idcg > 0 else np.nan


def top_k_metrics(y_true: np.ndarray, y_pred: np.ndarray, fractions: Sequence[float]) -> Dict[str, float]:
    """
    Screening-oriented metrics.

    overlap: fraction of the true elite recovered in predicted top-k
    precision: same as overlap here because predicted set size = true set size = k
    recall: same value in same-size setting
    enrichment: precision divided by random-hit-rate baseline (k / n)
    ndcg: ranking quality inside the predicted top-k list
    """
    n = len(y_true)
    if n == 0:
        return {}
    order_true = np.argsort(-y_true)
    order_pred = np.argsort(-y_pred)
    metrics = {}
    for frac in fractions:
        k = max(1, int(math.ceil(frac * n)))
        true_top = set(order_true[:k])
        pred_top = set(order_pred[:k])
        overlap_count = len(true_top & pred_top)
        overlap = overlap_count / k
        random_baseline_precision = k / n
        enrichment = overlap / random_baseline_precision if random_baseline_precision > 0 else np.nan
        metrics[f"top_{int(frac*100)}pct_overlap"] = overlap
        metrics[f"top_{int(frac*100)}pct_precision"] = overlap
        metrics[f"top_{int(frac*100)}pct_recall"] = overlap
        metrics[f"top_{int(frac*100)}pct_enrichment"] = enrichment
        metrics[f"top_{int(frac*100)}pct_ndcg"] = ndcg_at_k(y_true, y_pred, k)
    return metrics


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = N_BOOTSTRAP, alpha: float = 0.05, seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = np.mean(sample)
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lower), float(upper)


def paired_bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOTSTRAP, alpha: float = 0.05, seed: int = RANDOM_SEED) -> Tuple[float, float, float, float]:
    """
    Returns mean difference a-b, CI low/high, and two-sided bootstrap sign p-value.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]
    if len(arr_a) == 0:
        return np.nan, np.nan, np.nan, np.nan
    diffs = arr_a - arr_b
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    idx = np.arange(len(diffs))
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        boot[i] = np.mean(diffs[sample_idx])
    low = np.percentile(boot, 100 * alpha / 2)
    high = np.percentile(boot, 100 * (1 - alpha / 2))
    p_val = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(np.mean(diffs)), float(low), float(high), float(min(p_val, 1.0))



def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return np.nan
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else np.nan


def exact_rank_permutation_pvalue(observed_corr: float, n_items: int, method: str = "spearman") -> float:
    if not np.isfinite(observed_corr) or n_items < 2:
        return np.nan
    base = np.arange(1, n_items + 1)
    vals = []
    for perm in permutations(base):
        if method == "kendall":
            vals.append(stats.kendalltau(base, perm).correlation)
        else:
            vals.append(stats.spearmanr(base, perm).correlation)
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals <= observed_corr))


def one_way_eta_squared(df: pd.DataFrame, response: str, factor: str) -> float:
    work = df[[response, factor]].dropna().copy()
    if work.empty or work[factor].nunique() < 2:
        return np.nan
    y = work[response].astype(float).values
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total <= 0:
        return np.nan
    ss_between = 0.0
    for _, sub in work.groupby(factor):
        ss_between += len(sub) * (sub[response].mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def fit_additive_factor_model(df: pd.DataFrame, response: str, factors: Sequence[str]) -> pd.DataFrame:
    work = df[[response] + list(factors)].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    X = pd.get_dummies(work[list(factors)], drop_first=True)
    X.insert(0, "Intercept", 1.0)
    y = work[response].astype(float).values
    Xv = X.astype(float).values
    beta, _, _, _ = np.linalg.lstsq(Xv, y, rcond=None)
    yhat = Xv @ beta
    resid = y - yhat
    n, p = Xv.shape
    dof = max(n - p, 1)
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, a_min=0, a_max=None))
    tvals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2 * stats.t.sf(np.abs(tvals), df=dof)
    return pd.DataFrame({
        "term": X.columns,
        "coefficient": beta,
        "std_error": se,
        "t_value": tvals,
        "p_value": pvals,
        "response": response,
        "model_r2": 1.0 - (rss / tss if tss > 0 else np.nan),
        "n_rows": n,
    })


def average_standardized_shift(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: Sequence[str]) -> Dict[str, float]:
    cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]
    if not cols:
        return {
            "shift_centroid_distance": np.nan,
            "shift_mean_abs_z": np.nan,
            "shift_avg_wasserstein": np.nan,
        }
    train_num = train_df[cols].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[cols].apply(pd.to_numeric, errors="coerce")
    train_means = train_num.mean(axis=0)
    train_stds = train_num.std(axis=0).replace(0, np.nan)
    z_train_mean = ((train_num.mean(axis=0) - train_means) / train_stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z_test_mean = ((test_num.mean(axis=0) - train_means) / train_stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    centroid_distance = float(np.linalg.norm((z_test_mean - z_train_mean).values))
    mean_abs_z = float(np.mean(np.abs((z_test_mean - z_train_mean).values)))
    wass = []
    for c in cols:
        tr = train_num[c].dropna().values
        te = test_num[c].dropna().values
        if len(tr) == 0 or len(te) == 0:
            continue
        scale = float(train_stds[c]) if pd.notna(train_stds[c]) and float(train_stds[c]) > 0 else 1.0
        wass.append(float(stats.wasserstein_distance(tr / scale, te / scale)))
    return {
        "shift_centroid_distance": centroid_distance,
        "shift_mean_abs_z": mean_abs_z,
        "shift_avg_wasserstein": float(np.mean(wass)) if wass else np.nan,
    }


def mean_prediction_by_filename(preds_df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["target_slug", "descriptor_family", "split_family", "model", "filename_norm"]
    return preds_df.groupby(grp_cols, as_index=False).agg(
        mean_prediction=("prediction", "mean"),
        mean_target=("target", "mean"),
    )

def format_display_name(mapping: Dict[str, str], key: str) -> str:
    return mapping.get(key, key)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })



class ProjectManager:
    def __init__(self, base_output_dir: Path, logger: DualLogger):
        self.base_output_dir = base_output_dir
        self.logger = logger
        ensure_dir(self.base_output_dir)
        self.manifest_path = self.base_output_dir / "run_manifest.json"
        self.manifest = read_json_if_exists(
            self.manifest_path,
            default={"stages_completed": [], "experiments_completed": [], "notes": []}
        )

    def save(self) -> None:
        safe_json_dump(self.manifest, self.manifest_path)

    def stage_done(self, stage_name: str) -> bool:
        return stage_name in self.manifest["stages_completed"]

    def mark_stage_done(self, stage_name: str) -> None:
        if stage_name not in self.manifest["stages_completed"]:
            self.manifest["stages_completed"].append(stage_name)
            self.save()

    def experiment_done(self, experiment_key: str) -> bool:
        return experiment_key in self.manifest["experiments_completed"]

    def mark_experiment_done(self, experiment_key: str) -> None:
        if experiment_key not in self.manifest["experiments_completed"]:
            self.manifest["experiments_completed"].append(experiment_key)
            self.save()

    def note(self, message: str) -> None:
        self.manifest["notes"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
        })
        self.save()



@dataclass
class LoadedData:
    data: pd.DataFrame
    descriptor_families: Dict[str, List[str]]
    split_group_columns: Dict[str, Optional[str]]
    merge_diagnostics: pd.DataFrame


class DataAssembler:
    def __init__(self, data_dir: Path, output_dir: Path, logger: DualLogger):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = logger
        self.processed_dir = output_dir / "global_processed"
        ensure_dir(self.processed_dir)

    def _try_load_csv(self, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
        path = self.data_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required input file not found: {path}")
            self.logger.write(f"Optional file not found and will be skipped: {filename}")
            return None
        self.logger.write(f"Reading {filename}")
        return pd.read_csv(path, low_memory=False)

    def load_and_prepare(self, resume: bool = True) -> LoadedData:
        assembled_path = self.processed_dir / "assembled_master_table.pkl"
        families_path = self.processed_dir / "descriptor_families.json"
        splits_path = self.processed_dir / "split_group_columns.json"
        merge_diag_path = self.processed_dir / "merge_diagnostics.csv"

        if resume and assembled_path.exists() and families_path.exists() and splits_path.exists() and merge_diag_path.exists():
            self.logger.write("Loading processed master table and configuration from disk.")
            data = pd.read_pickle(assembled_path)
            descriptor_families = read_json_if_exists(families_path, default={})
            split_group_columns = read_json_if_exists(splits_path, default={})
            merge_diagnostics = pd.read_csv(merge_diag_path)
            return LoadedData(data, descriptor_families, split_group_columns, merge_diagnostics)

        clean_data = self._try_load_csv("clean_data.csv", required=True)
        geometry = self._try_load_csv("geometric_properties.csv", required=False)
        topology = self._try_load_csv("all_topology_lists.csv", required=False)
        geo_clusters = self._try_load_csv("geo-clusters.csv", required=True)
        mc_clusters = self._try_load_csv("mc-clusters.csv", required=True)
        func_clusters = self._try_load_csv("func-clusters.csv", required=True)
        flig_clusters = self._try_load_csv("flig-clusters.csv", required=True)

        check_required_columns(clean_data, REQUIRED_CLEAN_COLUMNS, "clean_data.csv")
        check_required_columns(geo_clusters, REQUIRED_CLUSTER_COLUMNS, "geo-clusters.csv")
        check_required_columns(mc_clusters, REQUIRED_CLUSTER_COLUMNS, "mc-clusters.csv")
        check_required_columns(func_clusters, REQUIRED_CLUSTER_COLUMNS, "func-clusters.csv")
        check_required_columns(flig_clusters, REQUIRED_CLUSTER_COLUMNS, "flig-clusters.csv")

        merge_rows = []

        clean_data["filename_norm"] = normalize_filename_series(clean_data[ID_COL])
        merge_rows.append({"step": "clean_data_loaded", "rows_before": np.nan, "rows_after": len(clean_data), "notes": "clean_data master table"})

        if geometry is not None:
            geometry["filename_norm"] = normalize_filename_series(geometry[ID_COL])
        if topology is not None:
            topology_id_col = resolve_topology_identifier_column(topology)
            topology["filename_norm"] = normalize_filename_series(topology[topology_id_col])
        geo_clusters["filename_norm"] = normalize_filename_series(geo_clusters[ID_COL])
        mc_clusters["filename_norm"] = normalize_filename_series(mc_clusters[ID_COL])
        func_clusters["filename_norm"] = normalize_filename_series(func_clusters[ID_COL])
        flig_clusters["filename_norm"] = normalize_filename_series(flig_clusters[ID_COL])

        df = clean_data.copy().drop_duplicates(subset=["filename_norm"]).reset_index(drop=True)

        if geometry is not None:
            keep_cols = [c for c in ["filename_norm", "ARC-MOF", "DB_num", "order_geo", "bool_geo"] if c in geometry.columns]
            geom_meta = geometry[keep_cols].drop_duplicates(subset=["filename_norm"])
            before = len(df)
            df = df.merge(geom_meta, on="filename_norm", how="left")
            merge_rows.append({"step": "geometry_meta_merge", "rows_before": before, "rows_after": len(df), "notes": "optional geometric metadata"})
        else:
            merge_rows.append({"step": "geometry_meta_merge", "rows_before": len(df), "rows_after": len(df), "notes": "skipped; optional file unavailable"})

        if topology is not None:
            top_keep_cols = [c for c in ["filename_norm", "likely topology", "Crystalnet"] if c in topology.columns]
            topo_meta = topology[top_keep_cols].drop_duplicates(subset=["filename_norm"])
            before = len(df)
            df = df.merge(topo_meta, on="filename_norm", how="left", suffixes=("", "_topfile"))
            merge_rows.append({"step": "topology_meta_merge", "rows_before": before, "rows_after": len(df), "notes": "optional topology metadata"})
        else:
            merge_rows.append({"step": "topology_meta_merge", "rows_before": len(df), "rows_after": len(df), "notes": "skipped; optional file unavailable"})

        # Rename cluster ids.
        geo_clusters = geo_clusters[["filename_norm", "cluster_id"]].rename(columns={"cluster_id": "geo_cluster"})
        mc_clusters = mc_clusters[["filename_norm", "cluster_id"]].rename(columns={"cluster_id": "metal_cluster"})
        func_clusters = func_clusters[["filename_norm", "cluster_id"]].rename(columns={"cluster_id": "func_cluster"})
        flig_clusters = flig_clusters[["filename_norm", "cluster_id"]].rename(columns={"cluster_id": "linker_cluster"})

        for name, part in [
            ("geo_cluster_merge", geo_clusters),
            ("metal_cluster_merge", mc_clusters),
            ("func_cluster_merge", func_clusters),
            ("linker_cluster_merge", flig_clusters),
        ]:
            before = len(df)
            df = df.merge(part, on="filename_norm", how="left")
            merge_rows.append({"step": name, "rows_before": before, "rows_after": len(df), "notes": "group labels merged"})

        df = df.drop_duplicates(subset=["filename_norm"]).reset_index(drop=True)

        # If the optional topology file provides a cleaner Crystalnet than clean_data, fill missing values.
        if "Crystalnet_topfile" in df.columns:
            df[TOPOLOGY_COLUMN_RAW] = df[TOPOLOGY_COLUMN_RAW].fillna(df["Crystalnet_topfile"])
            df = df.drop(columns=["Crystalnet_topfile"])

        # Feature engineering.
        eps = 1e-9
        df["lcd_pld_ratio"] = df["Di"] / (df["Df"] + eps)
        df["cavity_window_gap"] = df["Dif"]
        df["sa_pv_ratio"] = df["ASA"] / (df["AVA"] + eps)
        df["vf_density_ratio"] = df["AVAf"] / (df["Density"] + eps)
        df["log_pld_plus1"] = np.log1p(np.clip(df["Df"], a_min=0, a_max=None))
        df["log_lcd_plus1"] = np.log1p(np.clip(df["Di"], a_min=0, a_max=None))
        df["porosity_window_product"] = df["POAVA"] * df["Df"]
        df["density_pld_interaction"] = df["Density"] * df["Df"]

        # Group rare topology labels.
        if TOPOLOGY_COLUMN_RAW in df.columns:
            raw_top = df[TOPOLOGY_COLUMN_RAW].fillna("missing").astype(str)
            counts = raw_top.value_counts(dropna=False)
            keep = set(counts[counts >= 50].index.tolist())
            df[TOPOLOGY_COLUMN_GROUPED] = raw_top.where(raw_top.isin(keep), "other")
        else:
            df[TOPOLOGY_COLUMN_GROUPED] = "missing"

        df["all_targets_present"] = df[TARGET_COLUMNS_ALL].notna().all(axis=1)

        enriched_geometry = sorted(set(COMPACT_GEOMETRY_COLUMNS + [
            "UC_volume", "vASA", "NASA", "POAVAf", "Dif"
        ] + ENGINEERED_COLUMNS))
        enriched_geometry = [c for c in enriched_geometry if c in df.columns]

        topology_only = [TOPOLOGY_COLUMN_GROUPED]
        geometry_plus_topology = enriched_geometry + topology_only

        descriptor_families = {
            "compact_geometry": [c for c in COMPACT_GEOMETRY_COLUMNS if c in df.columns],
            "enriched_interpretable": enriched_geometry,
            "topology_only": topology_only,
            "geometry_plus_topology": geometry_plus_topology,
        }

        split_group_columns = {
            "random": None,
            "geo_grouped": "geo_cluster",
            "metal_grouped": "metal_cluster",
            "func_grouped": "func_cluster",
            "linker_grouped": "linker_cluster",
            "topology_grouped": TOPOLOGY_COLUMN_GROUPED,
        }

        merge_diagnostics = pd.DataFrame(merge_rows)

        df.to_pickle(assembled_path)
        df.to_csv(self.processed_dir / "assembled_master_table.csv", index=False)
        safe_json_dump(descriptor_families, families_path)
        safe_json_dump(split_group_columns, splits_path)
        merge_diagnostics.to_csv(merge_diag_path, index=False)

        self.logger.write(f"Processed master table saved with shape {df.shape}")
        return LoadedData(df, descriptor_families, split_group_columns, merge_diagnostics)



def make_one_hot_encoder(force_dense: bool) -> OneHotEncoder:
    """
    Version-tolerant OneHotEncoder factory.

    Why this helper exists
    ----------------------
    The original script used the sklearn default OneHotEncoder, which can return
    a sparse matrix. That is fine for some estimators, but
    HistGradientBoostingRegressor requires a dense matrix and fails with:

        TypeError: Sparse data was passed for X, but dense data is required.

    For HGB and MLP we therefore force dense preprocessing. To keep this safe on
    topology-only and geometry+topology runs, max_categories caps the one-hot
    expansion and pools less frequent categories into an infrequent bin when the
    installed sklearn version supports it.
    """
    sparse_value = not force_dense
    candidate_kwargs = [
        {
            "handle_unknown": "infrequent_if_exist",
            "max_categories": MAX_CATEGORICAL_LEVELS_FOR_ONEHOT,
        },
        {
            "handle_unknown": "ignore",
            "max_categories": MAX_CATEGORICAL_LEVELS_FOR_ONEHOT,
        },
        {
            "handle_unknown": "ignore",
        },
    ]

    for base_kwargs in candidate_kwargs:
        for sparse_key in ("sparse_output", "sparse"):
            kwargs = dict(base_kwargs)
            kwargs[sparse_key] = sparse_value
            try:
                return OneHotEncoder(**kwargs)
            except TypeError:
                continue

    # Very old sklearn fallback.
    return OneHotEncoder(handle_unknown="ignore")


class ModelFactory:
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed

    def build_pipeline_and_grid(self, numeric_cols: List[str], categorical_cols: List[str], model_name: str):
        force_dense_preprocess = model_name in {"hgb", "mlp"}
        scale_numeric = model_name in {"ridge", "mlp"}

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=num_steps)

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(force_dense=force_dense_preprocess)),
        ])

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
            # 0.0 means "always return dense" when dense preprocessing is needed.
            # This is essential for HGB/MLP; Ridge/RF may keep sparse matrices.
            sparse_threshold=0.0 if force_dense_preprocess else 0.3,
        )

        if model_name == "ridge":
            model = Ridge()
            param_grid = {
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            }

        elif model_name == "rf":

            


            model = RandomForestRegressor(
                random_state=self.random_seed,
                n_jobs=1,
                bootstrap=True,
                max_samples=0.70,
                max_features=0.70,
            )
            param_grid = {
                "model__n_estimators": [50, 80],
                "model__max_depth": [12, 20],
                "model__min_samples_leaf": [5],
            }

        elif model_name == "hgb":

            


            model = HistGradientBoostingRegressor(
                random_state=self.random_seed,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=15,
            )
            param_grid = {
                "model__learning_rate": [0.05, 0.10],
                "model__max_iter": [80, 150],
                "model__max_leaf_nodes": [31],
                "model__l2_regularization": [0.0, 0.01],
            }

        elif model_name == "mlp":
            model = MLPRegressor(
                random_state=self.random_seed,
                max_iter=220,
                early_stopping=True,
                n_iter_no_change=10,
                tol=1e-4,
            )
            param_grid = {
                "model__hidden_layer_sizes": [(32,), (64,)],
                "model__alpha": [1e-3, 1e-2],
                "model__learning_rate_init": [1e-3],
            }

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        return pipe, param_grid



# =============================================================================
# SECTION 6. BENCHMARK ENGINE
# =============================================================================

class BenchmarkEngine:
    def __init__(self, output_dir: Path, logger: DualLogger, project_manager: ProjectManager):
        self.output_dir = output_dir
        self.logger = logger
        self.pm = project_manager
        self.model_factory = ModelFactory(random_seed=RANDOM_SEED)

        self.metrics_dir = output_dir / "global_results" / "metrics"
        self.preds_dir = output_dir / "global_results" / "predictions"
        self.models_dir = output_dir / "global_results" / "models"
        self.tables_dir = output_dir / "global_results" / "tables"
        self.fig_main_dir = output_dir / "global_results" / "figures" / "main"
        self.fig_si_dir = output_dir / "global_results" / "figures" / "si"
        self.importance_dir = output_dir / "global_results" / "feature_importance"
        for d in [self.metrics_dir, self.preds_dir, self.models_dir, self.tables_dir,
                  self.fig_main_dir, self.fig_si_dir, self.importance_dir]:
            ensure_dir(d)

    def _build_splits(self, df: pd.DataFrame, split_name: str, group_col: Optional[str]):
        y_valid = df["target"].notna()
        sub = df.loc[y_valid].copy()
        idx = np.arange(len(sub))

        if split_name == "random":
            splitter = ShuffleSplit(
                n_splits=OUTER_RANDOM_SPLITS,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
            )
            splits = []
            for fold_id, (tr, te) in enumerate(splitter.split(idx)):
                splits.append({
                    "fold_id": fold_id,
                    "train_idx": sub.index.values[tr],
                    "test_idx": sub.index.values[te],
                    "split_name": split_name,
                })
            return splits

        if group_col is None or group_col not in sub.columns:
            return []

        sub = sub.loc[sub[group_col].notna()].copy()
        if sub.empty:
            return []

        counts = sub[group_col].value_counts()
        valid_groups = counts[counts >= 2].index
        sub = sub.loc[sub[group_col].isin(valid_groups)].copy()
        if sub.empty:
            return []

        n_groups = sub[group_col].nunique()
        n_splits = min(OUTER_GROUP_FOLDS, n_groups)
        if n_splits < 2:
            return []

        gkf = GroupKFold(n_splits=n_splits)
        groups = sub[group_col].values
        splits = []
        for fold_id, (tr, te) in enumerate(gkf.split(sub, groups=groups)):
            splits.append({
                "fold_id": fold_id,
                "train_idx": sub.index.values[tr],
                "test_idx": sub.index.values[te],
                "split_name": split_name,
            })
        return splits

    def _inner_cv(self, split_name: str, groups_train: Optional[np.ndarray]):
        if split_name == "random":
            return KFold(n_splits=INNER_CV_RANDOM, shuffle=True, random_state=RANDOM_SEED)
        n_groups = len(pd.Series(groups_train).dropna().unique()) if groups_train is not None else 0
        n_splits = min(INNER_CV_GROUP, n_groups)
        if n_splits < 2:
            return KFold(n_splits=INNER_CV_RANDOM, shuffle=True, random_state=RANDOM_SEED)
        return GroupKFold(n_splits=n_splits)

    def _prepare_X(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        X = df[feature_cols].copy()
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]
        return X, numeric_cols, categorical_cols

    def _experiment_key(self, target_slug: str, descriptor_name: str, split_name: str, model_name: str, fold_id: int) -> str:
        return f"{target_slug}__{descriptor_name}__{split_name}__{model_name}__fold{fold_id}"

    def run_all_experiments(self, df: pd.DataFrame, descriptor_families: Dict[str, List[str]], split_group_columns: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        model_names = ["ridge", "rf", "hgb", "mlp"]

        total_experiments = 0
        for target_col in TARGET_COLUMNS:
            for descriptor_name in descriptor_families.keys():
                for split_name in split_group_columns.keys():
                    n_folds = OUTER_RANDOM_SPLITS if split_name == "random" else OUTER_GROUP_FOLDS
                    total_experiments += len(model_names) * n_folds

        self.logger.write(f"Total planned experiments (upper bound): {total_experiments}")
        self.logger.write(f"Already completed experiments in manifest: {len(self.pm.manifest.get('experiments_completed', []))}")

        all_metrics = []
        prediction_registry = []

        for target_col in TARGET_COLUMNS:
            target_slug = TARGET_SLUGS[target_col]
            self.logger.write(f"Starting target: {target_col} ({target_slug})")
            df_target = df.copy()
            df_target["target"] = df_target[target_col]

            for descriptor_name, feature_cols in descriptor_families.items():
                self.logger.write(f"  Descriptor family: {descriptor_name} ({len(feature_cols)} columns)")
                if not feature_cols:
                    self.logger.write("  Skipping empty descriptor family.")
                    continue

                for split_name, group_col in split_group_columns.items():
                    self.logger.write(f"    Split family: {split_name}")
                    splits = self._build_splits(df_target, split_name, group_col)
                    if len(splits) == 0:
                        self.logger.write(f"    Skipping {split_name}; no valid splits formed.")
                        continue

                    for model_name in model_names:
                        self.logger.write(f"      Model: {model_name}")
                        for split in splits:
                            fold_id = split["fold_id"]
                            exp_key = self._experiment_key(target_slug, descriptor_name, split_name, model_name, fold_id)

                            metrics_path = self.metrics_dir / f"{exp_key}_metrics.csv"
                            preds_path = self.preds_dir / f"{exp_key}_predictions.csv"
                            model_path = self.models_dir / f"{exp_key}_best_model.joblib"
                            importance_path = self.importance_dir / f"{exp_key}_permutation_importance.csv"

                            if RESUME_IF_AVAILABLE and self.pm.experiment_done(exp_key) and metrics_path.exists() and preds_path.exists():
                                all_metrics.append(pd.read_csv(metrics_path))
                                prediction_registry.append({
                                    "experiment_key": exp_key,
                                    "predictions_path": str(preds_path),
                                    "target_slug": target_slug,
                                    "descriptor_family": descriptor_name,
                                    "split_family": split_name,
                                    "model": model_name,
                                    "fold_id": fold_id,
                                })
                                continue

                            train_idx = split["train_idx"]
                            test_idx = split["test_idx"]
                            train_df = df_target.loc[train_idx].copy()
                            test_df = df_target.loc[test_idx].copy()

                            train_df = train_df.loc[train_df["target"].notna()].copy()
                            test_df = test_df.loc[test_df["target"].notna()].copy()

                            if group_col is not None and group_col in train_df.columns:
                                train_df = train_df.loc[train_df[group_col].notna()].copy()
                                test_df = test_df.loc[test_df[group_col].notna()].copy()

                            if train_df.empty or test_df.empty:
                                self.logger.write(f"        Skipping empty train/test subset for {exp_key}")
                                continue

                            X_train, numeric_cols, categorical_cols = self._prepare_X(train_df, feature_cols)
                            X_test = test_df[feature_cols].copy()
                            y_train = train_df["target"].values
                            y_test = test_df["target"].values
                            groups_train = train_df[group_col].values if (group_col is not None and group_col in train_df.columns) else None

                            pipe, param_grid = self.model_factory.build_pipeline_and_grid(numeric_cols, categorical_cols, model_name)
                            inner_cv = self._inner_cv(split_name, groups_train)

                            search = GridSearchCV(
                                estimator=pipe,
                                param_grid=param_grid,
                                scoring="neg_mean_absolute_error",
                                cv=inner_cv,
                                n_jobs=N_JOBS,
                                refit=True,
                                verbose=0,
                                error_score=np.nan,
                                return_train_score=False,
                            )

                            fit_start = time.time()
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    if split_name == "random" or groups_train is None or isinstance(inner_cv, KFold):
                                        search.fit(X_train, y_train)
                                    else:
                                        search.fit(X_train, y_train, groups=groups_train)
                            except Exception as exc:
                                error_message = repr(exc)
                                self.logger.write(
                                    f"        Fit failed for {exp_key}; recording failure and continuing. "
                                    f"Error: {error_message[:600]}"
                                )
                                failure_metrics = pd.DataFrame([{
                                    "target_name": target_col,
                                    "target_slug": target_slug,
                                    "descriptor_family": descriptor_name,
                                    "split_family": split_name,
                                    "group_column": group_col,
                                    "model": model_name,
                                    "fold_id": fold_id,
                                    "n_train": len(train_df),
                                    "n_test": len(test_df),
                                    "r2": np.nan,
                                    "mae": np.nan,
                                    "rmse": np.nan,
                                    "spearman_rho": np.nan,
                                    "kendall_tau": np.nan,
                                    "best_params": "{}",
                                    "fit_status": "failed",
                                    "fit_seconds": time.time() - fit_start,
                                    "fit_error": error_message,
                                }])
                                failure_metrics.to_csv(metrics_path, index=False)
                                all_metrics.append(failure_metrics)
                                self.pm.mark_experiment_done(exp_key)
                                del failure_metrics
                                gc.collect()
                                continue

                            best_model = search.best_estimator_
                            y_pred = best_model.predict(X_test)
                            shift_metrics = average_standardized_shift(train_df, test_df, numeric_cols)

                            metrics = {
                                "target_name": target_col,
                                "target_slug": target_slug,
                                "descriptor_family": descriptor_name,
                                "split_family": split_name,
                                "group_column": group_col,
                                "model": model_name,
                                "fold_id": fold_id,
                                "n_train": len(train_df),
                                "n_test": len(test_df),
                                "r2": r2_score(y_test, y_pred),
                                "mae": mae(y_test, y_pred),
                                "rmse": rmse(y_test, y_pred),
                                "spearman_rho": spearman_safe(y_test, y_pred),
                                "kendall_tau": kendall_safe(y_test, y_pred),
                                "best_params": json.dumps(search.best_params_),
                                "fit_status": "ok",
                                "fit_seconds": time.time() - fit_start,
                                "fit_error": "",
                            }
                            metrics.update(shift_metrics)
                            metrics.update(top_k_metrics(y_test, y_pred, TOP_K_FRACTIONS))
                            fold_metrics = pd.DataFrame([metrics])

                            fold_preds = pd.DataFrame({
                                "filename_norm": test_df["filename_norm"].values,
                                "target": y_test,
                                "prediction": y_pred,
                                "absolute_error": np.abs(y_test - y_pred),
                                "residual": y_test - y_pred,
                                "target_name": target_col,
                                "target_slug": target_slug,
                                "descriptor_family": descriptor_name,
                                "split_family": split_name,
                                "group_column": group_col,
                                "model": model_name,
                                "fold_id": fold_id,
                                "geo_cluster": test_df.get("geo_cluster", pd.Series(index=test_df.index, dtype=float)).values,
                                "metal_cluster": test_df.get("metal_cluster", pd.Series(index=test_df.index, dtype=float)).values,
                                "func_cluster": test_df.get("func_cluster", pd.Series(index=test_df.index, dtype=float)).values,
                                "linker_cluster": test_df.get("linker_cluster", pd.Series(index=test_df.index, dtype=float)).values,
                                "Density": test_df.get("Density", pd.Series(index=test_df.index, dtype=float)).values,
                                "ASA": test_df.get("ASA", pd.Series(index=test_df.index, dtype=float)).values,
                                "AVA": test_df.get("AVA", pd.Series(index=test_df.index, dtype=float)).values,
                                "AVAf": test_df.get("AVAf", pd.Series(index=test_df.index, dtype=float)).values,
                                "POAVA": test_df.get("POAVA", pd.Series(index=test_df.index, dtype=float)).values,
                                "Di": test_df.get("Di", pd.Series(index=test_df.index, dtype=float)).values,
                                "Df": test_df.get("Df", pd.Series(index=test_df.index, dtype=float)).values,
                                "Dif": test_df.get("Dif", pd.Series(index=test_df.index, dtype=float)).values,
                                TOPOLOGY_COLUMN_GROUPED: test_df.get(TOPOLOGY_COLUMN_GROUPED, pd.Series(index=test_df.index, dtype=object)).values,
                            })

                            fold_metrics.to_csv(metrics_path, index=False)
                            fold_preds.to_csv(preds_path, index=False)
                            joblib.dump(best_model, model_path)

                            try:
                                if len(X_test) > MAX_ROWS_FOR_PERMUTATION_IMPORTANCE:
                                    rng_pi = np.random.default_rng(RANDOM_SEED)
                                    pi_positions = rng_pi.choice(
                                        np.arange(len(X_test)),
                                        size=MAX_ROWS_FOR_PERMUTATION_IMPORTANCE,
                                        replace=False,
                                    )
                                    X_importance = X_test.iloc[pi_positions].copy()
                                    y_importance = y_test[pi_positions]
                                else:
                                    X_importance = X_test
                                    y_importance = y_test

                                perm = permutation_importance(
                                    best_model,
                                    X_importance,
                                    y_importance,
                                    n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
                                    random_state=RANDOM_SEED,
                                    n_jobs=N_JOBS,
                                    scoring="neg_mean_absolute_error",
                                )
                                importance_df = pd.DataFrame({
                                    "feature": feature_cols,
                                    "importance_mean": perm.importances_mean,
                                    "importance_std": perm.importances_std,
                                    "target_slug": target_slug,
                                    "descriptor_family": descriptor_name,
                                    "split_family": split_name,
                                    "model": model_name,
                                    "fold_id": fold_id,
                                }).sort_values("importance_mean", ascending=False)
                                importance_df.to_csv(importance_path, index=False)
                            except Exception as exc:
                                self.logger.write(f"Permutation importance failed for {exp_key}: {exc}")

                            all_metrics.append(fold_metrics)
                            prediction_registry.append({
                                "experiment_key": exp_key,
                                "predictions_path": str(preds_path),
                                "target_slug": target_slug,
                                "descriptor_family": descriptor_name,
                                "split_family": split_name,
                                "model": model_name,
                                "fold_id": fold_id,
                            })
                            self.pm.mark_experiment_done(exp_key)

                            # Keep the long prediction files on disk, but release fold-level
                            # arrays/dataframes immediately to prevent cumulative RAM growth.
                            del fold_preds, X_train, X_test, y_train, y_test, y_pred, best_model, search
                            gc.collect()

        metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

        metrics_df.to_csv(self.metrics_dir / "all_fold_metrics_long.csv", index=False)
        metrics_df.to_pickle(self.metrics_dir / "all_fold_metrics_long.pkl")

        prediction_registry_df = pd.DataFrame(prediction_registry).drop_duplicates()
        registry_path = self.preds_dir / "prediction_file_registry.csv"
        prediction_registry_df.to_csv(registry_path, index=False)


        sample_parts = []
        for _, row in prediction_registry_df.iterrows():
            path = Path(row["predictions_path"])
            if not path.exists():
                continue
            try:
                temp = pd.read_csv(path)
                if len(temp) > MAX_ROWS_PER_PREDICTION_FILE_FOR_FIGURES:
                    temp = temp.sample(MAX_ROWS_PER_PREDICTION_FILE_FOR_FIGURES, random_state=RANDOM_SEED)
                sample_parts.append(temp)
            except Exception as exc:
                self.logger.write(f"Could not sample predictions from {path.name}: {exc}")

        preds_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
        sampled_csv = self.preds_dir / "sampled_fold_predictions_for_figures.csv"
        sampled_pkl = self.preds_dir / "sampled_fold_predictions_for_figures.pkl"
        preds_df.to_csv(sampled_csv, index=False)
        preds_df.to_pickle(sampled_pkl)

        self.logger.write(
            "Saved memory-safe prediction registry and sampled prediction table. "
            f"Registry: {registry_path.name}; sampled rows: {len(preds_df)}. "
            "Full predictions remain in per-experiment CSV files."
        )

        return metrics_df, preds_df




# =============================================================================
# SECTION 6B. EXACT DISK-BASED POST-HOC ANALYSES
# =============================================================================

class PredictionPostProcessor:
    """Memory-safe exact analyses from per-experiment prediction CSV files.

    The benchmark intentionally avoids concatenating all prediction files into one
    huge DataFrame. This class recovers exact post-hoc statistics by reading only
    the specific prediction files/columns needed for each analysis and reducing
    them to compact tables.
    """

    def __init__(self, preds_dir: Path, tables_dir: Path, logger: DualLogger):
        self.preds_dir = preds_dir
        self.tables_dir = tables_dir
        self.logger = logger
        ensure_dir(self.tables_dir)

    def _load_registry(self) -> pd.DataFrame:
        registry_path = self.preds_dir / "prediction_file_registry.csv"
        if registry_path.exists():
            reg = pd.read_csv(registry_path)
            reg["predictions_path"] = reg["predictions_path"].astype(str)
            return reg

        # Fallback for older outputs: infer the registry from file names.
        rows = []
        for path in sorted(self.preds_dir.glob("*_predictions.csv")):
            stem = path.name.replace("_predictions.csv", "")
            parts = stem.split("__")
            if len(parts) >= 5:
                rows.append({
                    "experiment_key": stem,
                    "predictions_path": str(path),
                    "target_slug": parts[0],
                    "descriptor_family": parts[1],
                    "split_family": parts[2],
                    "model": parts[3],
                    "fold_id": parts[4].replace("fold", ""),
                })
        reg = pd.DataFrame(rows)
        if not reg.empty:
            reg.to_csv(registry_path, index=False)
        return reg

    def _safe_read_csv_usecols(self, path: Path, desired_cols: Sequence[str], chunksize: Optional[int] = None):
        """Read a CSV with a desired usecols list while tolerating missing columns."""
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in desired_cols if c in header.columns]
        if chunksize is None:
            return pd.read_csv(path, usecols=usecols)
        return pd.read_csv(path, usecols=usecols, chunksize=chunksize)

    def _filter_registry(
        self,
        reg: pd.DataFrame,
        target_slugs: Optional[Sequence[str]] = None,
        descriptor_families: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        split_families: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        sub = reg.copy()
        if target_slugs is not None:
            sub = sub[sub["target_slug"].isin(target_slugs)]
        if descriptor_families is not None:
            sub = sub[sub["descriptor_family"].isin(descriptor_families)]
        if models is not None:
            sub = sub[sub["model"].isin(models)]
        if split_families is not None:
            sub = sub[sub["split_family"].isin(split_families)]
        return sub.reset_index(drop=True)

    def _mean_predictions_for_combo(
        self,
        reg: pd.DataFrame,
        target_slug: str,
        descriptor_family: str,
        model: str,
        split_family: str,
    ) -> pd.DataFrame:
        files = self._filter_registry(
            reg,
            target_slugs=[target_slug],
            descriptor_families=[descriptor_family],
            models=[model],
            split_families=[split_family],
        )
        parts = []
        desired_cols = ["filename_norm", "target", "prediction"]
        for _, row in files.iterrows():
            path = Path(row["predictions_path"])
            if not path.exists():
                continue
            try:
                for chunk in self._safe_read_csv_usecols(path, desired_cols, chunksize=EXACT_POSTHOC_CHUNKSIZE):
                    if chunk.empty:
                        continue
                    g = chunk.groupby("filename_norm", as_index=False).agg(
                        sum_prediction=("prediction", "sum"),
                        sum_target=("target", "sum"),
                        n_pred=("prediction", "size"),
                    )
                    parts.append(g)
            except Exception as exc:
                self.logger.write(f"Exact mean-prediction aggregation failed for {path.name}: {exc}")

        if not parts:
            return pd.DataFrame(columns=["filename_norm", "mean_prediction", "mean_target", "n_pred"])

        tmp = pd.concat(parts, ignore_index=True)
        out = tmp.groupby("filename_norm", as_index=False).agg(
            sum_prediction=("sum_prediction", "sum"),
            sum_target=("sum_target", "sum"),
            n_pred=("n_pred", "sum"),
        )
        out["mean_prediction"] = out["sum_prediction"] / out["n_pred"].replace(0, np.nan)
        out["mean_target"] = out["sum_target"] / out["n_pred"].replace(0, np.nan)
        return out[["filename_norm", "mean_prediction", "mean_target", "n_pred"]]

    def build_exact_group_resolved_errors(self, reg: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED]
        desired_cols = [
            "target_slug", "descriptor_family", "split_family", "model", "fold_id",
            "absolute_error", "residual", "target", "prediction",
            *group_cols,
        ]

        filtered = self._filter_registry(
            reg,
            target_slugs=EXACT_GROUP_TARGET_SLUGS,
            descriptor_families=EXACT_GROUP_DESCRIPTOR_FAMILIES,
            models=EXACT_GROUP_MODELS,
        )
        parts = []

        for _, row in filtered.iterrows():
            path = Path(row["predictions_path"])
            if not path.exists():
                continue
            try:
                for chunk in self._safe_read_csv_usecols(path, desired_cols, chunksize=EXACT_POSTHOC_CHUNKSIZE):
                    if chunk.empty:
                        continue

                    # Some older prediction files may not have all metadata columns;
                    # registry metadata is authoritative and stable.
                    for meta_col in ["target_slug", "descriptor_family", "split_family", "model", "fold_id"]:
                        if meta_col not in chunk.columns:
                            chunk[meta_col] = row.get(meta_col, np.nan)
                    chunk["abs_error_sq"] = pd.to_numeric(chunk["absolute_error"], errors="coerce") ** 2

                    for group_col in group_cols:
                        if group_col not in chunk.columns:
                            continue
                        sub = chunk.loc[chunk[group_col].notna(), [
                            "target_slug", "descriptor_family", "split_family", "model",
                            group_col, "absolute_error", "abs_error_sq", "residual", "target", "prediction"
                        ]].copy()
                        if sub.empty:
                            continue
                        sub["group_column"] = group_col
                        sub["group_id"] = sub[group_col].astype(str)
                        g = sub.groupby(
                            ["target_slug", "descriptor_family", "split_family", "model", "group_column", "group_id"],
                            as_index=False,
                            observed=False,
                        ).agg(
                            n=("absolute_error", "size"),
                            sum_abs_error=("absolute_error", "sum"),
                            sum_abs_error_sq=("abs_error_sq", "sum"),
                            sum_signed_residual=("residual", "sum"),
                            sum_target=("target", "sum"),
                            sum_prediction=("prediction", "sum"),
                        )
                        parts.append(g)
            except Exception as exc:
                self.logger.write(f"Exact group aggregation failed for {path.name}: {exc}")

        if not parts:
            return pd.DataFrame()

        agg = pd.concat(parts, ignore_index=True)
        keys = ["target_slug", "descriptor_family", "split_family", "model", "group_column", "group_id"]
        out = agg.groupby(keys, as_index=False, observed=False).agg(
            n=("n", "sum"),
            sum_abs_error=("sum_abs_error", "sum"),
            sum_abs_error_sq=("sum_abs_error_sq", "sum"),
            sum_signed_residual=("sum_signed_residual", "sum"),
            sum_target=("sum_target", "sum"),
            sum_prediction=("sum_prediction", "sum"),
        )
        out = out[out["n"] >= EXACT_GROUP_MIN_N].copy()
        if out.empty:
            return out

        out["mean_abs_error"] = out["sum_abs_error"] / out["n"]
        var_num = out["sum_abs_error_sq"] - (out["sum_abs_error"] ** 2) / out["n"].replace(0, np.nan)
        out["std_abs_error"] = np.sqrt(np.maximum(var_num / (out["n"] - 1).replace(0, np.nan), 0))
        out["median_abs_error"] = np.nan
        out["mean_signed_residual"] = out["sum_signed_residual"] / out["n"]
        out["target_mean"] = out["sum_target"] / out["n"]
        out["pred_mean"] = out["sum_prediction"] / out["n"]
        out["group_label"] = out["group_column"].str.replace("_cluster", "", regex=False).str.replace("topology_label", "topology", regex=False) + ":" + out["group_id"].astype(str)
        out["source_note"] = "Exact disk-based aggregation from full per-experiment prediction CSV files."
        out = out.sort_values(["target_slug", "descriptor_family", "model", "split_family", "mean_abs_error"], ascending=[True, True, True, True, False])
        path = self.tables_dir / "table_exact_group_resolved_errors.csv"
        out.to_csv(path, index=False)
        self.logger.write(f"Exact group-resolved errors saved: {path.name} ({len(out)} rows)")
        return out

    def build_exact_elite_list_stability(self, reg: pd.DataFrame) -> pd.DataFrame:
        split_order = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]
        rows = []

        for target_slug in EXACT_ELITE_TARGET_SLUGS:
            for descriptor_family in EXACT_ELITE_DESCRIPTOR_FAMILIES:
                for model in EXACT_ELITE_MODELS:
                    by_split = {}
                    for split_family in split_order:
                        mp = self._mean_predictions_for_combo(reg, target_slug, descriptor_family, model, split_family)
                        if not mp.empty:
                            by_split[split_family] = mp

                    for s1, s2 in combinations([s for s in split_order if s in by_split], 2):
                        g1 = by_split[s1][["filename_norm", "mean_prediction", "mean_target"]].rename(
                            columns={"mean_prediction": "p1", "mean_target": "target_1"}
                        )
                        g2 = by_split[s2][["filename_norm", "mean_prediction", "mean_target"]].rename(
                            columns={"mean_prediction": "p2", "mean_target": "target_2"}
                        )
                        merged = g1.merge(g2, on="filename_norm", how="inner")
                        if len(merged) < 100:
                            continue
                        if merged["p1"].nunique() > 1 and merged["p2"].nunique() > 1:
                            rank_rho = stats.spearmanr(merged["p1"], merged["p2"]).correlation
                        else:
                            rank_rho = np.nan

                        for frac in TOP_K_FRACTIONS:
                            k = max(1, int(math.ceil(frac * len(merged))))
                            top1 = merged.nlargest(k, "p1")["filename_norm"].tolist()
                            top2 = merged.nlargest(k, "p2")["filename_norm"].tolist()
                            rows.append({
                                "target_slug": target_slug,
                                "descriptor_family": descriptor_family,
                                "model": model,
                                "split_family_1": s1,
                                "split_family_2": s2,
                                "top_fraction": frac,
                                "n_common_items": int(len(merged)),
                                "top_k": int(k),
                                "elite_jaccard": jaccard_similarity(top1, top2),
                                "prediction_spearman": float(rank_rho) if pd.notna(rank_rho) else np.nan,
                                "source_note": "Exact disk-based aggregation from full per-experiment prediction CSV files.",
                            })

        out = pd.DataFrame(rows)
        path = self.tables_dir / "table_exact_elite_list_stability.csv"
        out.to_csv(path, index=False)
        self.logger.write(f"Exact elite-list stability saved: {path.name} ({len(out)} rows)")
        return out

    def build_case_study_candidates(self, reg: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if master_df is None or master_df.empty:
            return pd.DataFrame()

        random_pred = self._mean_predictions_for_combo(
            reg, CASE_STUDY_TARGET_SLUG, CASE_STUDY_DESCRIPTOR_FAMILY, CASE_STUDY_MODEL, "random"
        )
        if random_pred.empty:
            return pd.DataFrame()

        candidate_parts = []
        for split_family in ["geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]:
            grouped_pred = self._mean_predictions_for_combo(
                reg, CASE_STUDY_TARGET_SLUG, CASE_STUDY_DESCRIPTOR_FAMILY, CASE_STUDY_MODEL, split_family
            )
            if grouped_pred.empty:
                continue

            merged = random_pred.rename(columns={
                "mean_prediction": "random_prediction",
                "mean_target": "target_value",
                "n_pred": "random_n_pred",
            }).merge(
                grouped_pred.rename(columns={
                    "mean_prediction": "grouped_prediction",
                    "mean_target": "target_value_grouped",
                    "n_pred": "grouped_n_pred",
                }),
                on="filename_norm",
                how="inner",
            )
            if merged.empty:
                continue
            merged["target_value"] = merged["target_value"].fillna(merged["target_value_grouped"])
            merged["random_abs_error"] = np.abs(merged["target_value"] - merged["random_prediction"])
            merged["grouped_abs_error"] = np.abs(merged["target_value"] - merged["grouped_prediction"])
            merged["error_increase_grouped_minus_random"] = merged["grouped_abs_error"] - merged["random_abs_error"]
            merged["split_family"] = split_family
            candidate_parts.append(merged)

        if not candidate_parts:
            return pd.DataFrame()

        candidates = pd.concat(candidate_parts, ignore_index=True)

        meta_cols = [
            "filename_norm", "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
            "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED,
        ]
        meta_cols = [c for c in meta_cols if c in master_df.columns]
        meta = master_df[meta_cols].drop_duplicates("filename_norm")
        candidates = candidates.merge(meta, on="filename_norm", how="left")

        sort_cols = ["error_increase_grouped_minus_random", "grouped_abs_error"]
        candidates = candidates.sort_values(sort_cols, ascending=[False, False]).head(250).reset_index(drop=True)
        candidates["case_study_rank"] = np.arange(1, len(candidates) + 1)
        candidates["source_note"] = (
            "Candidate structures where grouped-split prediction is much worse than random-split prediction; "
            "use these rows to select CIFs for manual structural visualisation."
        )

        keep_cols = [
            "case_study_rank", "filename_norm", "split_family", "target_value",
            "random_prediction", "grouped_prediction", "random_abs_error",
            "grouped_abs_error", "error_increase_grouped_minus_random",
            "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
            "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED,
            "source_note",
        ]
        keep_cols = [c for c in keep_cols if c in candidates.columns]
        out = candidates[keep_cols].copy()
        path = self.tables_dir / "table_case_study_candidates.csv"
        out.to_csv(path, index=False)
        self.logger.write(f"Case-study candidate table saved: {path.name} ({len(out)} rows)")
        return out

    def build_exact_posthoc_tables(self, metrics_df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        reg = self._load_registry()
        if reg.empty:
            self.logger.write("Prediction registry is empty; exact post-hoc analyses skipped.")
            return {}

        tables: Dict[str, pd.DataFrame] = {}
        group_resolved = self.build_exact_group_resolved_errors(reg)
        if not group_resolved.empty:
            tables["group_resolved_errors"] = group_resolved
            hardest = (
                group_resolved.sort_values(["mean_abs_error", "n"], ascending=[False, False])
                .groupby(["target_slug", "group_column"], observed=False)
                .head(30)
                .reset_index(drop=True)
            )
            hardest.to_csv(self.tables_dir / "table_hardest_heldout_groups.csv", index=False)
            tables["hardest_heldout_groups"] = hardest

        elite = self.build_exact_elite_list_stability(reg)
        if not elite.empty:
            tables["elite_list_stability"] = elite

        case_candidates = self.build_case_study_candidates(reg, master_df)
        if not case_candidates.empty:
            tables["case_study_candidates"] = case_candidates

        # Compact split-severity table for main-text interpretation.
        if not metrics_df.empty:
            severity = metrics_df.groupby(["target_slug", "descriptor_family", "model", "split_family"], as_index=False, observed=False).agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                top5_overlap_mean=("top_5pct_overlap", "mean") if "top_5pct_overlap" in metrics_df.columns else ("r2", "count"),
                shift_mean=("shift_avg_wasserstein", "mean") if "shift_avg_wasserstein" in metrics_df.columns else ("r2", "count"),
            )
            severity.to_csv(self.tables_dir / "table_split_family_severity_summary.csv", index=False)
            tables["split_family_severity_summary"] = severity

        return tables




class TableBuilder:
    def __init__(self, tables_dir: Path, logger: DualLogger):
        self.tables_dir = tables_dir
        self.logger = logger
        ensure_dir(self.tables_dir)

    def _chemistry_ensemble_name(self, split_name: str) -> str:
        return "chemistry_ensemble" if split_name in {"metal_grouped", "func_grouped", "linker_grouped"} else split_name

    def _mean_ci_table(self, df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
        rows = []
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: val for col, val in zip(group_cols, keys)}
            for metric in metric_cols:
                mean_val, ci_low, ci_high = bootstrap_mean_ci(sub[metric].dropna().tolist())
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_ci_low"] = ci_low
                row[f"{metric}_ci_high"] = ci_high
                row[f"{metric}_std"] = float(sub[metric].std()) if metric in sub.columns else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def build_all_tables(
        self,
        metrics_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        master_df: Optional[pd.DataFrame] = None,
        split_group_columns: Optional[Dict[str, Optional[str]]] = None,
        merge_diagnostics: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        metric_cols = ["r2", "mae", "rmse", "spearman_rho", "kendall_tau"] + [c for c in metrics_df.columns if c.startswith("top_")]
        core_group_cols = ["target_slug", "descriptor_family", "split_family", "model"]

        if merge_diagnostics is not None:
            merge_diagnostics.to_csv(self.tables_dir / "table_merge_diagnostics.csv", index=False)
            tables["merge_diagnostics"] = merge_diagnostics

        # Dataset summary and merged coverage.
        if master_df is not None:
            rows = []
            n_total = len(master_df)
            for component, cols in {
                "clean_data_master": ["filename_norm"],
                "geometry_groups": ["geo_cluster"],
                "metal_groups": ["metal_cluster"],
                "functional_groups": ["func_cluster"],
                "linker_groups": ["linker_cluster"],
                "topology_labels": [TOPOLOGY_COLUMN_GROUPED],
            }.items():
                present = master_df[cols].notna().all(axis=1).sum()
                rows.append({
                    "component": component,
                    "n_rows_with_component": int(present),
                    "coverage_fraction": float(present / n_total) if n_total else np.nan,
                    "n_total_rows": int(n_total),
                })
            for target in TARGET_COLUMNS_ALL:
                present = master_df[target].notna().sum()
                rows.append({
                    "component": f"target::{target}",
                    "n_rows_with_component": int(present),
                    "coverage_fraction": float(present / n_total) if n_total else np.nan,
                    "n_total_rows": int(n_total),
                })
            dataset_summary = pd.DataFrame(rows)
            dataset_summary.to_csv(self.tables_dir / "table_dataset_summary_and_coverage.csv", index=False)
            tables["dataset_summary_and_coverage"] = dataset_summary

        benchmark_summary = self._mean_ci_table(metrics_df, core_group_cols, metric_cols)
        benchmark_summary.to_csv(self.tables_dir / "table_main_benchmark_summary.csv", index=False)
        benchmark_summary.to_pickle(self.tables_dir / "table_main_benchmark_summary.pkl")
        tables["main_benchmark_summary"] = benchmark_summary

        metrics_ensemble = metrics_df.copy()
        metrics_ensemble["split_superfamily"] = metrics_ensemble["split_family"].map(self._chemistry_ensemble_name)
        split_superfamily_summary = self._mean_ci_table(
            metrics_ensemble,
            ["target_slug", "descriptor_family", "split_superfamily", "model"],
            metric_cols,
        )
        split_superfamily_summary.to_csv(self.tables_dir / "table_split_superfamily_summary.csv", index=False)
        tables["split_superfamily_summary"] = split_superfamily_summary

        # Optimism gap relative to random.
        raw_gap_rows = []
        for (target_slug, descriptor_family, model, fold_id), sub in metrics_df.groupby(["target_slug", "descriptor_family", "model", "fold_id"]):
            rand = sub.loc[sub["split_family"] == "random"]
            if rand.empty:
                continue
            for split_name in sorted([s for s in sub["split_family"].unique() if s != "random"]):
                grp = sub.loc[sub["split_family"] == split_name]
                if grp.empty:
                    continue
                row = {
                    "target_slug": target_slug,
                    "descriptor_family": descriptor_family,
                    "model": model,
                    "fold_id": fold_id,
                    "split_family": split_name,
                }
                for c in metric_cols:
                    row[f"delta_from_random__{c}"] = float(rand.iloc[0][c] - grp.iloc[0][c])
                raw_gap_rows.append(row)
        optimism_gap_raw = pd.DataFrame(raw_gap_rows)
        optimism_gap_raw.to_csv(self.tables_dir / "table_optimism_gap_raw.csv", index=False)
        tables["optimism_gap_raw"] = optimism_gap_raw

        gap_metric_cols = [c for c in optimism_gap_raw.columns if c.startswith("delta_from_random__")]
        optimism_gap = self._mean_ci_table(
            optimism_gap_raw,
            ["target_slug", "descriptor_family", "split_family", "model"],
            gap_metric_cols,
        ) if not optimism_gap_raw.empty else pd.DataFrame()
        optimism_gap.to_csv(self.tables_dir / "table_optimism_gap.csv", index=False)
        tables["optimism_gap"] = optimism_gap

        # Paired significance summary for random vs grouped.
        sig_rows = []
        for (target_slug, descriptor_family, model), sub in metrics_df.groupby(["target_slug", "descriptor_family", "model"]):
            rand_sub = sub[sub["split_family"] == "random"].sort_values("fold_id")
            for split_name in sorted([s for s in sub["split_family"].unique() if s != "random"]):
                grp_sub = sub[sub["split_family"] == split_name].sort_values("fold_id")
                if rand_sub.empty or grp_sub.empty:
                    continue
                for metric in ["r2", "mae", "rmse", "spearman_rho", "top_5pct_overlap", "top_5pct_enrichment"]:
                    if metric not in rand_sub.columns or metric not in grp_sub.columns:
                        continue
                    n_match = min(len(rand_sub), len(grp_sub))
                    diff_mean, low, high, p_val = paired_bootstrap_diff_ci(
                        rand_sub[metric].values[:n_match],
                        grp_sub[metric].values[:n_match],
                    )
                    sig_rows.append({
                        "target_slug": target_slug,
                        "descriptor_family": descriptor_family,
                        "model": model,
                        "split_family": split_name,
                        "metric": metric,
                        "mean_diff_random_minus_grouped": diff_mean,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_p_value": p_val,
                    })
        significance_table = pd.DataFrame(sig_rows)
        significance_table.to_csv(self.tables_dir / "table_paired_significance_summary.csv", index=False)
        tables["paired_significance_summary"] = significance_table

        # Group-resolved error anatomy.
        group_error_parts = []
        for group_col in ["geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED]:
            if group_col not in preds_df.columns:
                continue
            sub = preds_df.loc[preds_df[group_col].notna()].copy()
            if sub.empty:
                continue
            g = (
                sub.groupby(["target_slug", "descriptor_family", "split_family", "model", group_col])
                .agg(
                    n=("absolute_error", "size"),
                    mean_abs_error=("absolute_error", "mean"),
                    median_abs_error=("absolute_error", "median"),
                    std_abs_error=("absolute_error", "std"),
                    mean_signed_residual=("residual", "mean"),
                    target_mean=("target", "mean"),
                    pred_mean=("prediction", "mean"),
                )
                .reset_index()
            )
            g = g.loc[g["n"] >= MIN_GROUP_SIZE_FOR_ANALYSIS].copy()
            g["group_column"] = group_col
            group_error_parts.append(g)
        group_resolved_errors = pd.concat(group_error_parts, ignore_index=True) if group_error_parts else pd.DataFrame()
        if not group_resolved_errors.empty:
            group_resolved_errors["source_note"] = (
                "Computed from the sampled prediction table for memory-safe point-level diagnostics; "
                "full per-experiment prediction CSVs are retained in the predictions folder."
            )
        group_resolved_errors.to_csv(self.tables_dir / "table_group_resolved_errors.csv", index=False)
        tables["group_resolved_errors"] = group_resolved_errors

        hardest_groups = (
            group_resolved_errors.sort_values(["mean_abs_error", "n"], ascending=[False, False])
            .groupby(["target_slug", "group_column"])
            .head(20)
            .reset_index(drop=True)
        ) if not group_resolved_errors.empty else pd.DataFrame()
        hardest_groups.to_csv(self.tables_dir / "table_hardest_heldout_groups.csv", index=False)
        tables["hardest_heldout_groups"] = hardest_groups

        # Ranking inversion: descriptors within each split and models within each descriptor.
        rank_rows = []
        corr_rows = []

        # Model ranks within descriptor family.
        for (target_slug, descriptor_family), sub in metrics_df.groupby(["target_slug", "descriptor_family"]):
            means = sub.groupby(["split_family", "model"])["r2"].mean().reset_index()
            ranking_map = {}
            for split_family, temp in means.groupby("split_family"):
                temp = temp.sort_values("r2", ascending=False).copy()
                temp["rank"] = np.arange(1, len(temp) + 1)
                temp["rank_type"] = "model_within_descriptor"
                temp["target_slug"] = target_slug
                temp["descriptor_family"] = descriptor_family
                rank_rows.append(temp)
                ranking_map[split_family] = temp.set_index("model")["rank"]
            split_names = sorted(ranking_map.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    s1, s2 = split_names[i], split_names[j]
                    aligned = pd.concat([ranking_map[s1], ranking_map[s2]], axis=1, join="inner")
                    aligned.columns = ["rank_1", "rank_2"]
                    corr_rows.append({
                        "target_slug": target_slug,
                        "comparison_scope": "model_within_descriptor",
                        "scope_value": descriptor_family,
                        "split_family_1": s1,
                        "split_family_2": s2,
                        "spearman_rank_correlation": stats.spearmanr(aligned["rank_1"], aligned["rank_2"]).correlation if len(aligned) >= 2 else np.nan,
                        "kendall_rank_correlation": stats.kendalltau(aligned["rank_1"], aligned["rank_2"]).correlation if len(aligned) >= 2 else np.nan,
                    })

        # Descriptor ranks within model.
        for (target_slug, model), sub in metrics_df.groupby(["target_slug", "model"]):
            means = sub.groupby(["split_family", "descriptor_family"])["r2"].mean().reset_index()
            ranking_map = {}
            for split_family, temp in means.groupby("split_family"):
                temp = temp.sort_values("r2", ascending=False).copy()
                temp["rank"] = np.arange(1, len(temp) + 1)
                temp["rank_type"] = "descriptor_within_model"
                temp["target_slug"] = target_slug
                temp["model"] = model
                rank_rows.append(temp.rename(columns={"descriptor_family": "entity"}))
                ranking_map[split_family] = temp.set_index("descriptor_family")["rank"]
            split_names = sorted(ranking_map.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    s1, s2 = split_names[i], split_names[j]
                    aligned = pd.concat([ranking_map[s1], ranking_map[s2]], axis=1, join="inner")
                    aligned.columns = ["rank_1", "rank_2"]
                    corr_rows.append({
                        "target_slug": target_slug,
                        "comparison_scope": "descriptor_within_model",
                        "scope_value": model,
                        "split_family_1": s1,
                        "split_family_2": s2,
                        "spearman_rank_correlation": stats.spearmanr(aligned["rank_1"], aligned["rank_2"]).correlation if len(aligned) >= 2 else np.nan,
                        "kendall_rank_correlation": stats.kendalltau(aligned["rank_1"], aligned["rank_2"]).correlation if len(aligned) >= 2 else np.nan,
                    })

        ranking_table = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
        ranking_corr_table = pd.DataFrame(corr_rows)
        ranking_table.to_csv(self.tables_dir / "table_ranks_by_split.csv", index=False)
        ranking_corr_table.to_csv(self.tables_dir / "table_ranking_inversion_correlations.csv", index=False)
        tables["ranks_by_split"] = ranking_table
        tables["ranking_inversion_correlations"] = ranking_corr_table

        # Group-size diagnostic summary.
        if master_df is not None and split_group_columns is not None:
            rows = []
            for split_name, group_col in split_group_columns.items():
                if group_col is None or group_col not in master_df.columns:
                    continue
                counts = master_df[group_col].dropna().value_counts()
                if len(counts) == 0:
                    continue
                rows.append({
                    "split_family": split_name,
                    "group_column": group_col,
                    "n_groups": int(len(counts)),
                    "min_group_size": int(counts.min()),
                    "median_group_size": float(counts.median()),
                    "mean_group_size": float(counts.mean()),
                    "max_group_size": int(counts.max()),
                })
            split_group_size_summary = pd.DataFrame(rows)
        else:
            split_group_size_summary = pd.DataFrame()
        split_group_size_summary.to_csv(self.tables_dir / "table_split_group_size_summary.csv", index=False)
        tables["split_group_size_summary"] = split_group_size_summary

        # Target overview.
        target_rows = []
        for target_slug in sorted(preds_df["target_slug"].dropna().unique()):
            sub = preds_df[preds_df["target_slug"] == target_slug]
            target_rows.append({
                "target_slug": target_slug,
                "target_name": sub["target_name"].iloc[0],
                "n_prediction_rows": int(len(sub)),
                "mean": float(sub["target"].mean()),
                "median": float(sub["target"].median()),
                "std": float(sub["target"].std()),
                "min": float(sub["target"].min()),
                "max": float(sub["target"].max()),
            })
        target_overview = pd.DataFrame(target_rows)
        if not target_overview.empty:
            target_overview["source_note"] = (
                "Computed from the memory-safe sampled prediction table used for point-level figures. "
                "Exact fold-level metrics are in all_fold_metrics_long.csv; full predictions remain as per-experiment CSV files."
            )
        target_overview.to_csv(self.tables_dir / "table_target_overview.csv", index=False)
        tables["target_overview"] = target_overview

        # Distribution-shift summary.
        shift_cols = [c for c in ["shift_centroid_distance", "shift_mean_abs_z", "shift_avg_wasserstein"] if c in metrics_df.columns]
        distribution_shift_summary = self._mean_ci_table(metrics_df, core_group_cols, shift_cols) if shift_cols else pd.DataFrame()
        distribution_shift_summary.to_csv(self.tables_dir / "table_distribution_shift_summary.csv", index=False)
        tables["distribution_shift_summary"] = distribution_shift_summary

        # Exact significance of ranking inversion severity.
        rank_sig_rows = []
        if not ranking_corr_table.empty:
            for _, row in ranking_corr_table.iterrows():
                n_items = 4
                rank_sig_rows.append({
                    **row.to_dict(),
                    "spearman_left_p": exact_rank_permutation_pvalue(row["spearman_rank_correlation"], n_items, method="spearman"),
                    "kendall_left_p": exact_rank_permutation_pvalue(row["kendall_rank_correlation"], n_items, method="kendall"),
                })
        rank_inversion_significance = pd.DataFrame(rank_sig_rows)
        rank_inversion_significance.to_csv(self.tables_dir / "table_rank_inversion_significance.csv", index=False)
        tables["rank_inversion_significance"] = rank_inversion_significance

        # Elite-list stability across split families using mean out-of-fold predictions.
        elite_rows = []
        mean_preds = mean_prediction_by_filename(preds_df) if not preds_df.empty else pd.DataFrame()
        if not mean_preds.empty:
            pair_splits = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped"]
            for (target_slug, descriptor_family, model), sub in mean_preds.groupby(["target_slug", "descriptor_family", "model"]):
                by_split = {s: g.copy() for s, g in sub.groupby("split_family")}
                for s1, s2 in combinations([s for s in pair_splits if s in by_split], 2):
                    g1 = by_split[s1][["filename_norm", "mean_prediction"]].rename(columns={"mean_prediction": "p1"})
                    g2 = by_split[s2][["filename_norm", "mean_prediction"]].rename(columns={"mean_prediction": "p2"})
                    merged = g1.merge(g2, on="filename_norm", how="inner")
                    if len(merged) < 20:
                        continue
                    for frac in TOP_K_FRACTIONS:
                        k = max(1, int(math.ceil(frac * len(merged))))
                        top1 = merged.nlargest(k, "p1")["filename_norm"].tolist()
                        top2 = merged.nlargest(k, "p2")["filename_norm"].tolist()
                        elite_rows.append({
                            "target_slug": target_slug,
                            "descriptor_family": descriptor_family,
                            "model": model,
                            "split_family_1": s1,
                            "split_family_2": s2,
                            "top_fraction": frac,
                            "n_common_items": int(len(merged)),
                            "elite_jaccard": jaccard_similarity(top1, top2),
                        })
        elite_list_stability = pd.DataFrame(elite_rows)
        if not elite_list_stability.empty:
            elite_list_stability["source_note"] = (
                "Computed from the sampled prediction table for memory-safe shortlist-stability diagnostics; "
                "full per-experiment prediction CSVs are retained in the predictions folder."
            )
        elite_list_stability.to_csv(self.tables_dir / "table_elite_list_stability.csv", index=False)
        tables["elite_list_stability"] = elite_list_stability

        # Additive factor model and marginal variance decomposition.
        factor_effect_parts = []
        for response in ["r2", "mae", "rmse", "top_5pct_overlap", "top_5pct_enrichment"]:
            if response in metrics_df.columns:
                factor_effect_parts.append(
                    fit_additive_factor_model(metrics_df, response, ["split_family", "descriptor_family", "model", "target_slug"])
                )
        factor_effects = pd.concat([x for x in factor_effect_parts if not x.empty], ignore_index=True) if factor_effect_parts else pd.DataFrame()
        factor_effects.to_csv(self.tables_dir / "table_additive_factor_effects.csv", index=False)
        tables["additive_factor_effects"] = factor_effects

        vd_rows = []
        for response in ["r2", "mae", "rmse", "top_5pct_overlap", "top_5pct_enrichment"]:
            if response not in metrics_df.columns:
                continue
            for factor in ["split_family", "descriptor_family", "model", "target_slug"]:
                vd_rows.append({
                    "response": response,
                    "factor": factor,
                    "eta_squared": one_way_eta_squared(metrics_df, response, factor),
                })
        variance_decomposition = pd.DataFrame(vd_rows)
        variance_decomposition.to_csv(self.tables_dir / "table_variance_decomposition.csv", index=False)
        tables["variance_decomposition"] = variance_decomposition

        # Main-paper compact tables restricted to anchor + secondary and headline descriptor families.
        main_text_targets = {TARGET_SLUGS[ANCHOR_TARGET], TARGET_SLUGS[SECONDARY_TARGET]}
        headline_families = {"compact_geometry", "enriched_interpretable"}
        main_paper_compact = benchmark_summary[
            benchmark_summary["target_slug"].isin(main_text_targets) &
            benchmark_summary["descriptor_family"].isin(headline_families)
        ].copy()
        main_paper_compact.to_csv(self.tables_dir / "table_main_paper_compact.csv", index=False)
        tables["main_paper_compact"] = main_paper_compact

        self.logger.write("All major tables saved.")
        return tables



class FigureBuilder:
    def __init__(self, fig_main_dir: Path, fig_si_dir: Path, logger: DualLogger):
        self.fig_main_dir = fig_main_dir
        self.fig_si_dir = fig_si_dir
        self.logger = logger
        self.fig_main_data_dir = self.fig_main_dir / "figure_data_csv"
        self.fig_si_data_dir = self.fig_si_dir / "figure_data_csv"
        ensure_dir(self.fig_main_dir)
        ensure_dir(self.fig_si_dir)
        ensure_dir(self.fig_main_data_dir)
        ensure_dir(self.fig_si_data_dir)
        configure_matplotlib()

    def _savefig(self, fig: plt.Figure, path: Path) -> None:
        ensure_dir(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _figure_data_path(self, figure_path: Path) -> Path:
        stem = figure_path.stem
        if figure_path.parent == self.fig_main_dir:
            return self.fig_main_data_dir / f"{stem}_plot_data.csv"
        return self.fig_si_data_dir / f"{stem}_plot_data.csv"

    def _save_figure_data_csv(self, figure_path: Path, data, data_name: str = "plot_data") -> None:
        csv_path = self._figure_data_path(figure_path)
        if isinstance(data, pd.DataFrame):
            out = data.copy()
        elif isinstance(data, dict):
            frames = []
            for key, value in data.items():
                if value is None:
                    continue
                if isinstance(value, pd.Series):
                    df = value.reset_index()
                    value_col = value.name if value.name is not None else "value"
                    if len(df.columns) == 2:
                        df.columns = ["x", value_col]
                    df.insert(0, "source", key)
                    frames.append(df)
                elif isinstance(value, pd.DataFrame):
                    df = value.copy()
                    df.insert(0, "source", key)
                    frames.append(df)
                else:
                    try:
                        df = pd.DataFrame(value)
                        df.insert(0, "source", key)
                        frames.append(df)
                    except Exception:
                        continue
            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            out = pd.DataFrame(data)
        ensure_dir(csv_path.parent)
        out.to_csv(csv_path, index=False)
        self.logger.write(f"Saved figure data CSV for {figure_path.name}: {csv_path.name}")


    def _plot_wide_bar_with_yerr(self, ax, pivot: pd.DataFrame, err: Optional[pd.DataFrame] = None) -> None:
        """Draw grouped bars with correctly shaped per-series error bars.

        This avoids a pandas/matplotlib yerr shape mismatch that can occur when
        plotting wide DataFrames with MultiIndex columns.
        """
        if pivot is None or pivot.empty:
            ax.axis("off")
            return

        pivot = pivot.copy()
        pivot = pivot.apply(pd.to_numeric, errors="coerce")
        if err is not None and not err.empty:
            err = err.reindex(index=pivot.index, columns=pivot.columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            err = None

        n_groups = len(pivot.index)
        n_series = len(pivot.columns)
        x = np.arange(n_groups, dtype=float)
        total_width = 0.82
        bar_width = total_width / max(n_series, 1)

        for j, col in enumerate(pivot.columns):
            offset = -total_width / 2.0 + bar_width * (j + 0.5)
            y = pivot[col].values.astype(float)
            if err is not None:
                yerr = err[col].values.astype(float)
            else:
                yerr = None

            if isinstance(col, tuple):
                label = " | ".join(str(c) for c in col)
            else:
                label = str(col)

            ax.bar(
                x + offset,
                y,
                width=bar_width * 0.92,
                yerr=yerr,
                capsize=3 if yerr is not None else 0,
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pivot.index])


    def _ordered_splits(self, values: Sequence[str]) -> List[str]:
        order = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped", "chemistry_ensemble"]
        present = set(str(v) for v in values)
        return [s for s in order if s in present] + sorted([s for s in present if s not in order])

    def _choose_available(self, df: pd.DataFrame, column: str, preferred: Sequence[str]) -> Optional[str]:
        if df.empty or column not in df.columns:
            return None
        vals = set(df[column].dropna().astype(str))
        for p in preferred:
            if p in vals:
                return p
        return next(iter(vals), None)

    def figure_2_v6_split_family_severity(self, metrics_df: pd.DataFrame) -> None:
        """Anchor-target split-family severity without hiding metal/functional/linker/topology in an ensemble."""
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = metrics_df[
            (metrics_df["target_slug"] == anchor_slug) &
            (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"])) &
            (metrics_df["model"].isin(["rf", "hgb"]))
        ].copy()
        if sub.empty:
            return

        split_order = self._ordered_splits(sub["split_family"].unique())
        sub["split_family"] = pd.Categorical(sub["split_family"], categories=split_order, ordered=True)

        grp = (
            sub.groupby(["split_family", "descriptor_family", "model"], observed=False)
            .agg(
                r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            )
            .reset_index()
            .dropna(subset=["split_family"])
        )

        fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
        for ax, metric_mean, metric_std, ylabel, title in [
            (axes[0], "r2_mean", "r2_std", "Mean R$^2$", "Figure 2a. Split-family severity in predictive accuracy"),
            (axes[1], "mae_mean", "mae_std", "Mean MAE", "Figure 2b. Error increase under grouped extrapolation"),
        ]:
            pivot = grp.pivot_table(
                index="split_family",
                columns=["descriptor_family", "model"],
                values=metric_mean,
                observed=False,
            ).reindex(split_order)
            err = grp.pivot_table(
                index="split_family",
                columns=["descriptor_family", "model"],
                values=metric_std,
                observed=False,
            ).reindex_like(pivot)
            self._plot_wide_bar_with_yerr(ax, pivot, err)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in ax.get_xticklabels()], rotation=25, ha="right")
            ax.legend(frameon=False, ncols=2, fontsize=8)

        figure_path = self.fig_main_dir / "figure2_v6_split_family_severity_anchor.png"
        self._save_figure_data_csv(figure_path, {"split_family_severity_anchor": grp})
        self._savefig(fig, figure_path)

    def figure_3_v6_target_split_sensitivity(self, metrics_df: pd.DataFrame) -> None:
        """Target-by-split sensitivity map for the strongest interpretable benchmark setting."""
        preferred_model = "rf" if "rf" in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "mlp", "ridge"])
        descriptor = "enriched_interpretable" if "enriched_interpretable" in set(metrics_df["descriptor_family"].astype(str)) else self._choose_available(metrics_df, "descriptor_family", ["geometry_plus_topology", "compact_geometry"])
        if preferred_model is None or descriptor is None:
            return

        sub = metrics_df[(metrics_df["model"] == preferred_model) & (metrics_df["descriptor_family"] == descriptor)].copy()
        if sub.empty:
            return
        split_order = self._ordered_splits(sub["split_family"].unique())
        target_order = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(sub["target_slug"])]

        r2 = sub.groupby(["target_slug", "split_family"], observed=False)["r2"].mean().reset_index()
        mae_df = sub.groupby(["target_slug", "split_family"], observed=False)["mae"].mean().reset_index()

        r2_piv = r2.pivot_table(index="target_slug", columns="split_family", values="r2", observed=False).reindex(index=target_order, columns=split_order)
        mae_piv = mae_df.pivot_table(index="target_slug", columns="split_family", values="mae", observed=False).reindex(index=target_order, columns=split_order)

        # Delta R2 relative to random makes split severity immediately visible.
        if "random" in r2_piv.columns:
            delta = r2_piv.sub(r2_piv["random"], axis=0)
        else:
            delta = r2_piv * np.nan

        fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
        for ax, pivot, title, fmt in [
            (axes[0], r2_piv, f"Figure 3a. Mean R$^2$ by target and split ({descriptor}, {preferred_model})", ".2f"),
            (axes[1], delta, "Figure 3b. R$^2$ change relative to random split", ".2f"),
        ]:
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, c) for c in pivot.columns], rotation=30, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(title)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:{fmt}}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        figure_path = self.fig_main_dir / "figure3_v6_target_split_sensitivity_map.png"
        plot_data = {
            "r2_by_target_split": r2_piv.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="mean_r2"),
            "delta_r2_vs_random": delta.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="delta_r2_vs_random"),
            "mae_by_target_split": mae_piv.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="mean_mae"),
        }
        self._save_figure_data_csv(figure_path, plot_data)
        self._savefig(fig, figure_path)

    def figure_4_v6_screening_stability(self, metrics_df: pd.DataFrame) -> None:
        """Screening stability across split families, avoiding overclaiming a collapse."""
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        model = "rf" if "rf" in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "mlp", "ridge"])
        if model is None:
            return
        sub = metrics_df[
            (metrics_df["target_slug"] == anchor_slug) &
            (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable", "geometry_plus_topology"])) &
            (metrics_df["model"] == model)
        ].copy()
        if sub.empty:
            return
        split_order = self._ordered_splits(sub["split_family"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, metric, title in [
            (axes[0], "top_5pct_overlap", "Figure 4a. Elite-retrieval overlap across split families"),
            (axes[1], "top_5pct_enrichment", "Figure 4b. Elite-retrieval enrichment across split families"),
        ]:
            if metric not in sub.columns:
                ax.axis("off")
                continue
            pivot = (
                sub.groupby(["split_family", "descriptor_family"], observed=False)[metric]
                .mean()
                .unstack("descriptor_family")
                .reindex(split_order)
            )
            err = (
                sub.groupby(["split_family", "descriptor_family"], observed=False)[metric]
                .std()
                .unstack("descriptor_family")
                .reindex_like(pivot)
            )
            self._plot_wide_bar_with_yerr(ax, pivot, err)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in ax.get_xticklabels()], rotation=25, ha="right")
            ax.legend(frameon=False, fontsize=8)

        figure_path = self.fig_main_dir / "figure4_v6_screening_stability_by_split.png"
        self._save_figure_data_csv(figure_path, {"screening_stability": sub})
        self._savefig(fig, figure_path)

    def figure_5_v6_exact_group_error_anatomy(self, group_error_df: pd.DataFrame) -> None:
        """Hardest exact held-out groups from full per-experiment predictions."""
        if group_error_df is None or group_error_df.empty:
            self.logger.write("Exact group error table is empty; Figure 5 skipped.")
            return

        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = group_error_df[group_error_df["target_slug"] == anchor_slug].copy()
        if sub.empty:
            sub = group_error_df.copy()

        descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "geometry_plus_topology", "compact_geometry"])
        model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", ["rf", "hgb", "mlp", "ridge"])
        if descriptor is not None:
            sub = sub[sub["descriptor_family"] == descriptor].copy()
        if model is not None:
            sub = sub[sub["model"] == model].copy()

        # Geometry panel should reflect geometry-grouped extrapolation. Chemistry/topology panel
        # uses the matching grouped split for each group type.
        geo = sub[(sub["group_column"] == "geo_cluster") & (sub["split_family"] == "geo_grouped")].copy()
        if geo.empty:
            geo = sub[sub["group_column"] == "geo_cluster"].copy()
        geo = geo.sort_values("mean_abs_error", ascending=False).head(12)

        chem_masks = []
        mapping = {
            "metal_cluster": "metal_grouped",
            "func_cluster": "func_grouped",
            "linker_cluster": "linker_grouped",
            TOPOLOGY_COLUMN_GROUPED: "topology_grouped",
        }
        for group_col, split_name in mapping.items():
            chem_masks.append((sub["group_column"] == group_col) & (sub["split_family"] == split_name))
        chem_mask = np.logical_or.reduce(chem_masks) if chem_masks else np.array([False] * len(sub))
        chem = sub[chem_mask].copy()
        if chem.empty:
            chem = sub[sub["group_column"].isin(list(mapping.keys()))].copy()
        chem = chem.sort_values("mean_abs_error", ascending=False).head(12)

        fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
        for ax, temp, title in [
            (axes[0], geo, "Figure 5a. Hardest geometry groups under geometry hold-out"),
            (axes[1], chem, "Figure 5b. Hardest chemistry/topology groups under matched hold-out"),
        ]:
            if temp.empty:
                ax.axis("off")
                ax.set_title(title + " (no groups passed filter)")
                continue
            labels = temp["group_label"].astype(str).tolist() if "group_label" in temp.columns else temp["group_id"].astype(str).tolist()
            sem = temp["std_abs_error"].fillna(0).values / np.sqrt(temp["n"].replace(0, np.nan).values)
            ax.bar(range(len(temp)), temp["mean_abs_error"].values, yerr=sem, capsize=3)
            ax.set_xticks(range(len(temp)))
            ax.set_xticklabels(labels, rotation=70, ha="right")
            ax.set_ylabel("Mean absolute error")
            ax.set_title(title)
            ax.text(0.01, 0.98, f"{descriptor} | {model}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

        figure_path = self.fig_main_dir / "figure5_v6_exact_group_resolved_error_anatomy.png"
        self._save_figure_data_csv(figure_path, {"hardest_geometry_groups_exact": geo, "hardest_chemistry_topology_groups_exact": chem})
        self._savefig(fig, figure_path)

    def figure_7_v6_distribution_shift_vs_performance(self, metrics_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = metrics_df[metrics_df["target_slug"] == anchor_slug].copy()
        needed = ["shift_centroid_distance", "shift_avg_wasserstein", "r2"]
        if sub.empty or not all(c in sub.columns for c in needed):
            return

        split_order = self._ordered_splits(sub["split_family"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for split_family, temp in sub.groupby("split_family", observed=False):
            axes[0].scatter(
                temp["shift_centroid_distance"],
                temp["r2"],
                s=48,
                alpha=0.78,
                label=format_display_name(DISPLAY_SPLIT_NAMES, split_family),
            )
        valid = sub[["shift_centroid_distance", "r2"]].dropna()
        corr = np.nan
        if len(valid) >= 3 and valid["shift_centroid_distance"].nunique() > 1:
            corr = stats.spearmanr(valid["shift_centroid_distance"], valid["r2"]).correlation
            x = np.linspace(valid["shift_centroid_distance"].min(), valid["shift_centroid_distance"].max(), 100)
            slope, intercept, *_ = stats.linregress(valid["shift_centroid_distance"], valid["r2"])
            axes[0].plot(x, intercept + slope * x, linestyle="--", linewidth=1.5)
        axes[0].set_title(f"Figure 7a. Distribution shift versus performance (Spearman = {corr:.2f})")
        axes[0].set_xlabel("Centroid shift distance")
        axes[0].set_ylabel("R$^2$")
        axes[0].legend(frameon=False, fontsize=8)

        shift_bar = sub.groupby("split_family", observed=False)["shift_avg_wasserstein"].mean().reindex(split_order)
        shift_bar.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Figure 7b. Average descriptor-space shift by split family")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Mean standardized Wasserstein shift")
        axes[1].set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in axes[1].get_xticklabels()], rotation=25, ha="right")

        figure_path = self.fig_main_dir / "figure7_v6_distribution_shift_vs_performance.png"
        corr_df = pd.DataFrame([{"anchor_target": anchor_slug, "spearman_shift_r2": corr}])
        self._save_figure_data_csv(figure_path, {"shift_vs_performance_points": sub, "shift_bar_summary": shift_bar.reset_index().rename(columns={"split_family": "split_family", "shift_avg_wasserstein": "mean_shift_avg_wasserstein"}), "shift_performance_correlation": corr_df})
        self._savefig(fig, figure_path)

    def figure_8_v6_exact_elite_list_stability(self, elite_df: pd.DataFrame) -> None:
        if elite_df is None or elite_df.empty:
            self.logger.write("Elite-list stability table is empty; Figure 8 skipped.")
            return
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = elite_df[(elite_df["target_slug"] == anchor_slug) & (elite_df["top_fraction"] == 0.05)].copy()
        if sub.empty:
            sub = elite_df[elite_df["top_fraction"] == 0.05].copy()
        if sub.empty:
            return

        descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "geometry_plus_topology", "compact_geometry"])
        model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", ["rf", "hgb", "mlp", "ridge"])
        if descriptor is not None:
            sub = sub[sub["descriptor_family"] == descriptor].copy()
        if model is not None:
            sub = sub[sub["model"] == model].copy()
        if sub.empty:
            return

        sub["pair"] = sub["split_family_1"] + " vs " + sub["split_family_2"]
        top = sub.sort_values("elite_jaccard", ascending=True).head(12).copy()
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        axes[0].barh(range(len(top)), top["elite_jaccard"].values)
        axes[0].set_yticks(range(len(top)))
        axes[0].set_yticklabels(top["pair"].str.replace("_", " "))
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Top-5% Jaccard overlap")
        axes[0].set_title("Figure 8a. Least stable elite-list comparisons")

        axes[1].barh(range(len(top)), top["prediction_spearman"].values)
        axes[1].set_yticks(range(len(top)))
        axes[1].set_yticklabels(top["pair"].str.replace("_", " "))
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Prediction-rank Spearman correlation")
        axes[1].set_title("Figure 8b. Overall ranking agreement")

        for ax in axes:
            ax.text(0.01, 0.02, f"{anchor_slug} | {descriptor} | {model} | exact full predictions", transform=ax.transAxes, fontsize=9, ha="left", va="bottom")

        figure_path = self.fig_main_dir / "figure8_v6_exact_elite_list_stability.png"
        self._save_figure_data_csv(figure_path, {"elite_stability_exact_top5": sub, "least_stable_pairs": top})
        self._savefig(fig, figure_path)


    def build_all_figures(self, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], master_df: Optional[pd.DataFrame] = None) -> None:
        if metrics_df.empty:
            self.logger.write("No metrics available; skipping figure generation.")
            return


        self.figure_1_study_design_overview()
        self.figure_2_v6_split_family_severity(metrics_df)
        self.figure_3_v6_target_split_sensitivity(metrics_df)
        self.figure_4_v6_screening_stability(metrics_df)
        self.figure_5_v6_exact_group_error_anatomy(tables.get("group_resolved_errors", pd.DataFrame()))
        self.figure_6_descriptor_space_and_example_split(master_df if master_df is not None else preds_df)
        self.figure_7_v6_distribution_shift_vs_performance(metrics_df)
        self.figure_8_v6_exact_elite_list_stability(tables.get("elite_list_stability", pd.DataFrame()))

        if not preds_df.empty:
            self.si_figure_target_distributions(preds_df)
            self.si_figure_pld_decile_errors(preds_df)
            self.si_figure_prediction_scatter_panels(preds_df)
        else:
            self.logger.write("Sampled prediction table empty; point-level SI figures skipped.")
        self.si_figure_variance_decomposition(tables.get("variance_decomposition", pd.DataFrame()))
        self.si_figure_splitwise_metric_heatmaps(metrics_df)
        self.si_figure_group_size_distributions(tables.get("split_group_size_summary", pd.DataFrame()))
        self.si_figure_error_vs_rank(metrics_df)

    def figure_1_study_design_overview(self) -> None:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        boxes = [
            (0.03, 0.70, 0.24, 0.18, "Input tables\nclean_data\ngeo / mc / func / flig clusters\noptional geometry + topology"),
            (0.31, 0.70, 0.18, 0.18, "Merge + normalize IDs\nfeature engineering\nanalysis-ready master table"),
            (0.54, 0.70, 0.18, 0.18, "Descriptor families\ncompact geometry\nenriched interpretable\ntopology variants"),
            (0.76, 0.70, 0.20, 0.18, "Split families\nrandom\ngeometry-grouped\nchemistry-grouped\noptional topology-grouped"),
            (0.15, 0.33, 0.20, 0.18, "Matched models\nRidge\nRF\nHGB\nMLP"),
            (0.40, 0.33, 0.24, 0.18, "Metrics\nR² / MAE / RMSE\nSpearman / Kendall\ntop-k overlap\nenrichment / NDCG"),
            (0.71, 0.33, 0.23, 0.18, "Outputs\nmain-text figures\nSI figures\nCSV / PKL / LaTeX\nsaved models + logs"),
        ]
        for x, y, w, h, txt in boxes:
            rect = plt.Rectangle((x, y), w, h, linewidth=1.6, fill=False, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, txt, transform=ax.transAxes, ha="center", va="center", fontsize=11)

        arrowprops = dict(arrowstyle="->", lw=1.6)
        coords = [
            ((0.27, 0.79), (0.31, 0.79)),
            ((0.49, 0.79), (0.54, 0.79)),
            ((0.72, 0.79), (0.76, 0.79)),
            ((0.22, 0.66), (0.24, 0.51)),
            ((0.52, 0.66), (0.52, 0.51)),
            ((0.86, 0.66), (0.82, 0.51)),
            ((0.35, 0.42), (0.40, 0.42)),
            ((0.64, 0.42), (0.71, 0.42)),
        ]
        for (x1, y1), (x2, y2) in coords:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrowprops)

        ax.set_title("Figure 1. Study design overview: matched models and descriptors under alternative split families", pad=12)
        figure_path = self.fig_main_dir / "figure1_study_design_overview.png"
        figure_data = pd.DataFrame(boxes, columns=["x", "y", "width", "height", "label"])
        self._save_figure_data_csv(figure_path, figure_data)
        self._savefig(fig, figure_path)

    def figure_2_headline_optimism_gap(self, metrics_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        headline = metrics_df[
            (metrics_df["target_slug"] == anchor_slug) &
            (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"])) &
            (metrics_df["split_family"].isin(["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped"]))
        ].copy()

        # Chemistry ensemble for cleaner presentation.
        headline["split_panel"] = headline["split_family"].replace({
            "metal_grouped": "chemistry_ensemble",
            "func_grouped": "chemistry_ensemble",
            "linker_grouped": "chemistry_ensemble",
        })
        grp = (
            headline.groupby(["split_panel", "descriptor_family", "model"])
            .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                 mae_mean=("mae", "mean"), mae_std=("mae", "std"))
            .reset_index()
        )

        order = ["random", "geo_grouped", "chemistry_ensemble"]
        grp["split_panel"] = pd.Categorical(grp["split_panel"], categories=order, ordered=True)
        grp = grp.sort_values("split_panel")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, metric_mean, metric_std, ylabel in [
            (axes[0], "r2_mean", "r2_std", "Mean R$^2$"),
            (axes[1], "mae_mean", "mae_std", "Mean MAE"),
        ]:
            pivot = grp.pivot_table(
                index="split_panel",
                columns=["descriptor_family", "model"],
                values=metric_mean,
                observed=False,
            )
            err = grp.pivot_table(
                index="split_panel",
                columns=["descriptor_family", "model"],
                values=metric_std,
                observed=False,
            ).reindex_like(pivot)
            self._plot_wide_bar_with_yerr(ax, pivot, err)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in ax.get_xticklabels()], rotation=20)
            ax.legend(loc="best", frameon=False)
        axes[0].set_title(f"Figure 2a. Anchor-target optimism gap ({anchor_slug})")
        axes[1].set_title("Figure 2b. Error inflation under grouped extrapolation")
        figure_path = self.fig_main_dir / "figure2_headline_optimism_gap_composite.png"
        self._save_figure_data_csv(figure_path, {"figure2_group_summary": grp})
        self._savefig(fig, figure_path)

    def figure_3_ranking_inversion_map(self, metrics_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = metrics_df[metrics_df["target_slug"] == anchor_slug].copy()

        # Panel A: descriptor ranks within each model.
        desc_means = sub.groupby(["split_family", "model", "descriptor_family"])["r2"].mean().reset_index()
        desc_rows = []
        for (split_family, model), temp in desc_means.groupby(["split_family", "model"]):
            temp = temp.sort_values("r2", ascending=False).copy()
            temp["rank"] = np.arange(1, len(temp) + 1)
            desc_rows.append(temp)
        desc_rank = pd.concat(desc_rows, ignore_index=True) if desc_rows else pd.DataFrame()

        # Panel B: model ranks within each descriptor family.
        model_means = sub.groupby(["split_family", "descriptor_family", "model"])["r2"].mean().reset_index()
        model_rows = []
        for (split_family, descriptor_family), temp in model_means.groupby(["split_family", "descriptor_family"]):
            temp = temp.sort_values("r2", ascending=False).copy()
            temp["rank"] = np.arange(1, len(temp) + 1)
            model_rows.append(temp)
        model_rank = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        if not desc_rank.empty:
            desc_rank["row_label"] = desc_rank["split_family"] + " | " + desc_rank["model"]
            pivot = desc_rank.pivot_table(index="row_label", columns="descriptor_family", values="rank", observed=False)
            im = axes[0].imshow(pivot.values, aspect="auto")
            axes[0].set_xticks(np.arange(len(pivot.columns)))
            axes[0].set_xticklabels([format_display_name(DISPLAY_DESCRIPTOR_NAMES, c) for c in pivot.columns], rotation=35, ha="right")
            axes[0].set_yticks(np.arange(len(pivot.index)))
            axes[0].set_yticklabels([r.replace("_", " ") for r in pivot.index])
            axes[0].set_title("Figure 3a. Descriptor rank inversion across split families")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    axes[0].text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        if not model_rank.empty:
            model_rank["row_label"] = model_rank["split_family"] + " | " + model_rank["descriptor_family"]
            pivot = model_rank.pivot_table(index="row_label", columns="model", values="rank", observed=False)
            im = axes[1].imshow(pivot.values, aspect="auto")
            axes[1].set_xticks(np.arange(len(pivot.columns)))
            axes[1].set_xticklabels(pivot.columns)
            axes[1].set_yticks(np.arange(len(pivot.index)))
            axes[1].set_yticklabels([r.replace("_", " ") for r in pivot.index])
            axes[1].set_title("Figure 3b. Model rank inversion across split families")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    axes[1].text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        figure_path = self.fig_main_dir / "figure3_ranking_inversion_map.png"
        self._save_figure_data_csv(figure_path, {"descriptor_rank_table": desc_rank, "model_rank_table": model_rank})
        self._savefig(fig, figure_path)

    def figure_4_screening_stability_collapse(self, metrics_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = metrics_df[
            (metrics_df["target_slug"] == anchor_slug) &
            (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"]))
        ].copy()
        sub["split_panel"] = sub["split_family"].replace({
            "metal_grouped": "chemistry_ensemble",
            "func_grouped": "chemistry_ensemble",
            "linker_grouped": "chemistry_ensemble",
        })
        sub = sub[sub["split_panel"].isin(["random", "geo_grouped", "chemistry_ensemble"])]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, metric, title in [
            (axes[0], "top_5pct_overlap", "Figure 4a. Top-5% overlap collapse"),
            (axes[1], "top_5pct_enrichment", "Figure 4b. Top-5% enrichment collapse"),
        ]:
            if metric not in sub.columns:
                continue
            pivot = sub.groupby(["split_panel", "descriptor_family"], observed=False)[metric].mean().unstack("descriptor_family")
            err = sub.groupby(["split_panel", "descriptor_family"], observed=False)[metric].std().unstack("descriptor_family").reindex_like(pivot)
            pivot = pivot.reindex(["random", "geo_grouped", "chemistry_ensemble"])
            err = err.reindex_like(pivot)
            self._plot_wide_bar_with_yerr(ax, pivot, err)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in ax.get_xticklabels()], rotation=20)
            ax.legend(frameon=False)
        figure_path = self.fig_main_dir / "figure4_screening_stability_collapse.png"
        self._save_figure_data_csv(figure_path, {"screening_metrics_summary": sub})
        self._savefig(fig, figure_path)

    def figure_5_group_error_anatomy(self, group_error_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        if group_error_df.empty:
            return

        sub = group_error_df[
            (group_error_df["target_slug"] == anchor_slug) &
            (group_error_df["descriptor_family"] == "enriched_interpretable") &
            (group_error_df["model"] == "hgb")
        ].copy()
        if sub.empty:
            sub = group_error_df.copy()

        geo = sub[sub["group_column"] == "geo_cluster"].sort_values("mean_abs_error", ascending=False).head(15)
        chem = sub[sub["group_column"].isin(["metal_cluster", "func_cluster", "linker_cluster"])].copy()
        chem = chem.sort_values("mean_abs_error", ascending=False).head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, temp, title, group_col in [
            (axes[0], geo, "Figure 5a. Hardest held-out geometry groups", "geo_cluster"),
            (axes[1], chem, "Figure 5b. Hardest held-out chemistry groups", "group_column"),
        ]:
            if temp.empty:
                ax.axis("off")
                continue
            labels = temp[group_col].astype(str).tolist()
            ax.bar(range(len(temp)), temp["mean_abs_error"].values, yerr=temp["std_abs_error"].fillna(0).values, capsize=3)
            ax.set_xticks(range(len(temp)))
            ax.set_xticklabels(labels, rotation=85)
            ax.set_ylabel("Mean absolute error")
            ax.set_title(title)
        figure_path = self.fig_main_dir / "figure5_group_resolved_error_anatomy.png"
        self._save_figure_data_csv(figure_path, {"hardest_geometry_groups": geo, "hardest_chemistry_groups": chem})
        self._savefig(fig, figure_path)

    def figure_6_descriptor_space_and_example_split(self, master_df: pd.DataFrame) -> None:
        needed = [c for c in ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"] if c in master_df.columns]
        if len(needed) < 2:
            return

        sub = master_df[needed + ["geo_cluster", "metal_cluster"]].dropna(subset=needed).copy()
        if len(sub) > MAX_ROWS_FOR_VISUAL_PCA:
            sub = sub.sample(MAX_ROWS_FOR_VISUAL_PCA, random_state=RANDOM_SEED)

        X = StandardScaler().fit_transform(sub[needed].values)
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

        # Simulate one geometry-grouped split to show why the task is harder.
        geo_values = sub["geo_cluster"].dropna().astype(str)
        if len(geo_values) > 0:
            counts = geo_values.value_counts()
            held_out_groups = set(counts.head(10).index.tolist())
            membership = sub["geo_cluster"].astype(str).isin(held_out_groups).map({True: "example_test_group", False: "example_train_group"})
        else:
            membership = pd.Series(["all_points"] * len(sub), index=sub.index)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        geo_codes = pd.factorize(sub["geo_cluster"].astype(str))[0]
        axes[0].scatter(coords[:, 0], coords[:, 1], c=geo_codes, s=8, alpha=0.75)
        axes[0].set_title("Figure 6a. Descriptor-space PCA colored by geometry cluster")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        membership_codes = pd.factorize(membership.astype(str))[0]
        axes[1].scatter(coords[:, 0], coords[:, 1], c=membership_codes, s=8, alpha=0.75)
        axes[1].set_title("Figure 6b. Example grouped hold-out in descriptor space")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")

        figure_path = self.fig_main_dir / "figure6_descriptor_space_and_example_split.png"
        figure_data = sub.copy()
        figure_data["pca_pc1"] = coords[:, 0]
        figure_data["pca_pc2"] = coords[:, 1]
        figure_data["example_split_membership"] = membership.values
        self._save_figure_data_csv(figure_path, figure_data)
        self._savefig(fig, figure_path)


    def figure_7_distribution_shift_vs_performance(self, metrics_df: pd.DataFrame) -> None:
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = metrics_df[metrics_df["target_slug"] == anchor_slug].copy()
        needed = ["shift_centroid_distance", "shift_avg_wasserstein", "r2"]
        if sub.empty or not all(c in sub.columns for c in needed):
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for split_family, temp in sub.groupby("split_family"):
            axes[0].scatter(
                temp["shift_centroid_distance"],
                temp["r2"],
                s=50,
                alpha=0.8,
                label=format_display_name(DISPLAY_SPLIT_NAMES, split_family),
            )
        axes[0].set_title("Figure 7a. Train-test shift versus performance")
        axes[0].set_xlabel("Centroid shift distance")
        axes[0].set_ylabel("R$^2$")
        axes[0].legend(frameon=False)

        panel = sub.copy()
        panel["split_panel"] = panel["split_family"].replace({
            "metal_grouped": "chemistry_ensemble",
            "func_grouped": "chemistry_ensemble",
            "linker_grouped": "chemistry_ensemble",
        })
        shift_bar = panel.groupby("split_panel")["shift_avg_wasserstein"].mean().reindex(
            ["random", "geo_grouped", "chemistry_ensemble"]
        )
        shift_bar.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Figure 7b. Average descriptor-space shift by split family")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Mean standardized Wasserstein shift")
        axes[1].set_xticklabels(
            [format_display_name(DISPLAY_SPLIT_NAMES, x.get_text()) for x in axes[1].get_xticklabels()],
            rotation=20,
        )
        figure_path = self.fig_main_dir / "figure7_distribution_shift_vs_performance.png"
        self._save_figure_data_csv(figure_path, {"shift_vs_performance_points": sub, "shift_bar_summary": shift_bar.reset_index().rename(columns={"split_panel": "split_panel", "shift_avg_wasserstein": "mean_shift_avg_wasserstein"})})
        self._savefig(fig, figure_path)

    def figure_8_elite_list_stability(self, elite_df: pd.DataFrame) -> None:
        if elite_df.empty:
            return
        anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
        sub = elite_df[(elite_df["target_slug"] == anchor_slug) & (elite_df["top_fraction"] == 0.05)].copy()
        if sub.empty:
            sub = elite_df[elite_df["top_fraction"] == 0.05].copy()
        if sub.empty:
            return
        sub["pair"] = sub["split_family_1"] + " vs " + sub["split_family_2"]
        pivot = sub.groupby("pair")["elite_jaccard"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Figure 8. Elite-list stability across split families (top 5%)")
        ax.set_xlabel("")
        ax.set_ylabel("Mean Jaccard overlap")
        ax.tick_params(axis="x", rotation=35)
        figure_path = self.fig_main_dir / "figure8_elite_list_stability.png"
        self._save_figure_data_csv(figure_path, {"elite_stability_raw": sub, "elite_stability_bar_summary": pivot.reset_index().rename(columns={"index": "pair", "elite_jaccard": "mean_elite_jaccard"})})
        self._savefig(fig, figure_path)

    def si_figure_variance_decomposition(self, variance_df: pd.DataFrame) -> None:
        if variance_df.empty:
            return
        pivot = variance_df.pivot_table(index="factor", columns="response", values="eta_squared", observed=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("SI variance decomposition across benchmark factors")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        figure_path = self.fig_si_dir / "si_variance_decomposition.png"
        self._save_figure_data_csv(figure_path, variance_df)
        self._savefig(fig, figure_path)

    def si_figure_target_distributions(self, preds_df: pd.DataFrame) -> None:
        target_slugs = sorted(preds_df["target_slug"].dropna().unique())
        if not target_slugs:
            return
        n = len(target_slugs)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, target_slug in zip(axes, target_slugs):
            sub = preds_df[preds_df["target_slug"] == target_slug]
            ax.hist(sub["target"].dropna().values, bins=40)
            ax.set_title(f"Target distribution: {target_slug}")
            ax.set_xlabel("Target value")
            ax.set_ylabel("Count")
        for ax in axes[len(target_slugs):]:
            ax.axis("off")
        figure_path = self.fig_si_dir / "si_target_distributions.png"
        self._save_figure_data_csv(figure_path, preds_df[["target_slug", "target_name", "target"]].copy())
        self._savefig(fig, figure_path)

    def si_figure_splitwise_metric_heatmaps(self, metrics_df: pd.DataFrame) -> None:
        for metric in ["r2", "mae", "spearman_rho", "kendall_tau", "top_5pct_overlap", "top_5pct_enrichment"]:
            if metric not in metrics_df.columns:
                continue
            pivot = metrics_df.groupby(["split_family", "model"])[metric].mean().unstack("model")
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([format_display_name(DISPLAY_SPLIT_NAMES, idx) for idx in pivot.index])
            ax.set_title(f"SI heatmap of mean {metric}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
            figure_path = self.fig_si_dir / f"si_heatmap_{metric}.png"
            heatmap_df = pivot.reset_index().melt(id_vars="split_family", var_name="model", value_name=f"mean_{metric}")
            self._save_figure_data_csv(figure_path, heatmap_df)
            self._savefig(fig, figure_path)

    def si_figure_pld_decile_errors(self, preds_df: pd.DataFrame) -> None:
        sub = preds_df.dropna(subset=["Df", "absolute_error"]).copy()
        if sub.empty:
            return
        sub["pld_decile"] = pd.qcut(sub["Df"], q=10, duplicates="drop")
        dec = (
            sub.groupby(
                ["target_slug", "descriptor_family", "model", "pld_decile"],
                observed=False,
            )["absolute_error"]
            .mean()
            .reset_index()
        )
        target_slugs = sorted(dec["target_slug"].unique())
        n = len(target_slugs)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, target_slug in zip(axes, target_slugs):
            temp = dec[dec["target_slug"] == target_slug]
            plot_df = temp.groupby("pld_decile", observed=False)["absolute_error"].mean().reset_index()
            ax.plot(range(len(plot_df)), plot_df["absolute_error"].values, marker="o")
            ax.set_title(f"PLD-decile-resolved MAE: {target_slug}")
            ax.set_xlabel("PLD decile")
            ax.set_ylabel("Mean absolute error")
        for ax in axes[len(target_slugs):]:
            ax.axis("off")
        figure_path = self.fig_si_dir / "si_pld_decile_errors.png"
        self._save_figure_data_csv(figure_path, dec)
        self._savefig(fig, figure_path)
        dec.to_csv(self.fig_si_dir / "si_pld_decile_errors_table.csv", index=False)

    def si_figure_prediction_scatter_panels(self, preds_df: pd.DataFrame) -> None:
        summary = (
            preds_df.groupby(["target_slug", "descriptor_family", "split_family", "model"])
            .agg(mae_mean=("absolute_error", "mean"), n=("prediction", "size"))
            .reset_index()
        )
        best_keys = (
            summary.sort_values("mae_mean", ascending=True)
            .groupby("target_slug")
            .head(1)[["target_slug", "descriptor_family", "split_family", "model"]]
        )
        merged = preds_df.merge(best_keys, on=["target_slug", "descriptor_family", "split_family", "model"], how="inner")
        target_slugs = sorted(merged["target_slug"].unique())
        if not target_slugs:
            return
        n = len(target_slugs)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, target_slug in zip(axes, target_slugs):
            sub = merged[merged["target_slug"] == target_slug]
            ax.scatter(sub["target"], sub["prediction"], s=8, alpha=0.5)
            min_val = min(sub["target"].min(), sub["prediction"].min())
            max_val = max(sub["target"].max(), sub["prediction"].max())
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax.set_title(f"Best prediction scatter: {target_slug}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
        for ax in axes[len(target_slugs):]:
            ax.axis("off")
        figure_path = self.fig_si_dir / "si_prediction_scatter_panels.png"
        self._save_figure_data_csv(figure_path, merged)
        self._savefig(fig, figure_path)

    def si_figure_group_size_distributions(self, group_size_table: pd.DataFrame) -> None:
        if group_size_table.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(group_size_table))
        ax.bar(x, group_size_table["mean_group_size"].values)
        ax.errorbar(x, group_size_table["median_group_size"].values, fmt="o")
        ax.set_xticks(x)
        ax.set_xticklabels([format_display_name(DISPLAY_SPLIT_NAMES, s) for s in group_size_table["split_family"]], rotation=20)
        ax.set_ylabel("Group size")
        ax.set_title("SI group-size diagnostic summary")
        figure_path = self.fig_si_dir / "si_group_size_summary.png"
        self._save_figure_data_csv(figure_path, group_size_table)
        self._savefig(fig, figure_path)

    def si_figure_error_vs_rank(self, metrics_df: pd.DataFrame) -> None:
        screening_metric = "top_5pct_overlap" if "top_5pct_overlap" in metrics_df.columns else None
        if screening_metric is None:
            return
        summary = metrics_df.groupby(["split_family", "descriptor_family", "model"]).agg(
            mae_mean=("mae", "mean"),
            overlap_mean=(screening_metric, "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(10, 7))
        for split_family, sub in summary.groupby("split_family"):
            ax.scatter(sub["mae_mean"], sub["overlap_mean"], s=60, alpha=0.8, label=format_display_name(DISPLAY_SPLIT_NAMES, split_family))
            for _, r in sub.iterrows():
                ax.text(r["mae_mean"], r["overlap_mean"], f"{r['model']}|{r['descriptor_family']}", fontsize=8)
        ax.set_title("SI calibration-style error-versus-screening plot")
        ax.set_xlabel("Mean MAE")
        ax.set_ylabel(screening_metric)
        ax.legend(frameon=False)
        figure_path = self.fig_si_dir / "si_error_vs_screening_rank.png"
        self._save_figure_data_csv(figure_path, summary)
        self._savefig(fig, figure_path)




class ManuscriptExporter:
    def __init__(self, output_dir: Path, logger: DualLogger):
        self.output_dir = output_dir
        self.logger = logger
        self.latex_dir = output_dir / "global_results" / "latex_exports"
        ensure_dir(self.latex_dir)

    def dataframe_to_latex(self, df: pd.DataFrame, path: Path, caption: str, label: str, index: bool = False, float_format: str = "%.3f") -> None:
        latex = df.to_latex(index=index, escape=False, caption=caption, label=label, float_format=float_format)
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex)

    def export_key_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        mapping = {
            "merge_diagnostics": ("latex_merge_diagnostics.tex", "Merge diagnostics and retained-row counts.", "tab:merge_diagnostics"),
            "dataset_summary_and_coverage": ("latex_dataset_summary_and_coverage.tex", "Dataset summary and merged file coverage.", "tab:dataset_summary_and_coverage"),
            "main_benchmark_summary": ("latex_main_benchmark_summary.tex", "Main benchmark summary across targets, descriptor families, split families, and models.", "tab:main_benchmark_summary"),
            "split_superfamily_summary": ("latex_split_superfamily_summary.tex", "Benchmark summary collapsed to split superfamilies, including the chemistry ensemble.", "tab:split_superfamily_summary"),
            "optimism_gap": ("latex_optimism_gap.tex", "Optimism-gap table comparing grouped splits against random evaluation.", "tab:optimism_gap"),
            "paired_significance_summary": ("latex_paired_significance_summary.tex", "Paired bootstrap significance summary for random versus grouped splits.", "tab:paired_significance_summary"),
            "hardest_heldout_groups": ("latex_hardest_heldout_groups.tex", "Hardest held-out geometry and chemistry groups.", "tab:hardest_heldout_groups"),
            "group_resolved_errors": ("latex_group_resolved_errors.tex", "Group-resolved error anatomy table.", "tab:group_resolved_errors"),
            "ranking_inversion_correlations": ("latex_ranking_inversion_correlations.tex", "Rank-correlation summary across split families.", "tab:ranking_inversion_correlations"),
            "split_group_size_summary": ("latex_split_group_size_summary.tex", "Split-diagnostic summary of group sizes.", "tab:split_group_size_summary"),
            "target_overview": ("latex_target_overview.tex", "Target-distribution overview.", "tab:target_overview"),
            "main_paper_compact": ("latex_main_paper_compact.tex", "Compact main-text benchmark table for the anchor and secondary targets.", "tab:main_paper_compact"),
            "distribution_shift_summary": ("latex_distribution_shift_summary.tex", "Summary of train-test distribution shift across split families.", "tab:distribution_shift_summary"),
            "rank_inversion_significance": ("latex_rank_inversion_significance.tex", "Exact significance summary for ranking inversion severity across split families.", "tab:rank_inversion_significance"),
            "elite_list_stability": ("latex_elite_list_stability.tex", "Elite-list stability across split families.", "tab:elite_list_stability"),
            "additive_factor_effects": ("latex_additive_factor_effects.tex", "Additive factor-model summary across benchmark outputs.", "tab:additive_factor_effects"),
            "variance_decomposition": ("latex_variance_decomposition.tex", "Marginal variance decomposition across split family, descriptor family, model, and target.", "tab:variance_decomposition"),
            "split_family_severity_summary": ("latex_split_family_severity_summary.tex", "Split-family severity summary across targets, descriptor families, and models.", "tab:split_family_severity_summary"),
            "case_study_candidates": ("latex_case_study_candidates.tex", "Candidate MOFs for manual structural visualisation and grouped-extrapolation case studies.", "tab:case_study_candidates"),
        }
        for key, (filename, caption, label) in mapping.items():
            if key in tables and not tables[key].empty:
                self.dataframe_to_latex(tables[key], self.latex_dir / filename, caption=caption, label=label, index=False)
        self.logger.write("LaTeX table exports saved.")


class ProgressReporter:
    def __init__(self, output_dir: Path, logger: DualLogger, project_manager: ProjectManager):
        self.output_dir = output_dir
        self.logger = logger
        self.pm = project_manager
        self.progress_path = output_dir / "progress_report.json"

    def update(self) -> None:
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages_completed": self.pm.manifest.get("stages_completed", []),
            "n_stages_completed": len(self.pm.manifest.get("stages_completed", [])),
            "n_experiments_completed": len(self.pm.manifest.get("experiments_completed", [])),
            "notes": self.pm.manifest.get("notes", []),
            "anchor_target": ANCHOR_TARGET,
            "secondary_target": SECONDARY_TARGET,
            "run_all_targets": RUN_ALL_TARGETS,
        }
        safe_json_dump(payload, self.progress_path)


def main() -> None:
    start_time = time.time()

    ensure_dir(OUTPUT_DIR)
    logs_dir = OUTPUT_DIR / "logs"
    ensure_dir(logs_dir)
    log_file = logs_dir / "run_log.txt"

    logger = DualLogger(log_file)
    pm = ProjectManager(OUTPUT_DIR, logger)
    reporter = ProgressReporter(OUTPUT_DIR, logger, pm)

    logger.write("=" * 88)
    logger.write("Starting revised Paper 1 MOF split-strategy benchmark pipeline")
    logger.write(f"Base directory: {BASE_DIR}")
    logger.write(f"Output directory: {OUTPUT_DIR}")
    logger.write(f"Anchor target: {ANCHOR_TARGET}")
    logger.write(f"Secondary target: {SECONDARY_TARGET}")
    logger.write(f"Run all targets: {RUN_ALL_TARGETS}")
    logger.write(f"N_JOBS: {N_JOBS}")
    logger.write(f"Pipeline variant: {PIPELINE_VARIANT}")
    logger.write(f"Memory-safe patch: {MEMORY_SAFE_PATCH}")
    logger.write(f"Max one-hot categorical levels: {MAX_CATEGORICAL_LEVELS_FOR_ONEHOT}")
    logger.write(f"Permutation importance repeats: {PERMUTATION_IMPORTANCE_REPEATS}")
    logger.write(f"Permutation importance max rows: {MAX_ROWS_FOR_PERMUTATION_IMPORTANCE}")
    logger.write(f"Python version: {sys.version}")
    logger.write("=" * 88)
    reporter.update()

    try:
        stage_name = "data_assembly"
        assembler = DataAssembler(DATA_DIR, OUTPUT_DIR, logger)
        if not (RESUME_IF_AVAILABLE and pm.stage_done(stage_name)):
            loaded = assembler.load_and_prepare(resume=RESUME_IF_AVAILABLE)
            pm.mark_stage_done(stage_name)
            pm.note("Data assembly finished.")
            reporter.update()
        else:
            loaded = assembler.load_and_prepare(resume=True)
            logger.write("Stage already complete: data assembly")

        copy_log_snapshot(log_file, "mid_project_log_after_data_assembly.txt")

        stage_name = "benchmark_experiments"
        engine = BenchmarkEngine(OUTPUT_DIR, logger, pm)
        if not (RESUME_IF_AVAILABLE and pm.stage_done(stage_name)):
            metrics_df, preds_df = engine.run_all_experiments(
                loaded.data,
                loaded.descriptor_families,
                loaded.split_group_columns,
            )
            pm.mark_stage_done(stage_name)
            pm.note("Benchmark experiments finished.")
            reporter.update()
        else:
            logger.write("Stage already complete: benchmark experiments")
            metrics_df = pd.read_pickle(engine.metrics_dir / "all_fold_metrics_long.pkl")
            sampled_preds_path = engine.preds_dir / "sampled_fold_predictions_for_figures.pkl"
            legacy_preds_path = engine.preds_dir / "all_fold_predictions_long.pkl"
            if sampled_preds_path.exists():
                preds_df = pd.read_pickle(sampled_preds_path)
                logger.write(f"Loaded sampled prediction table: {sampled_preds_path.name}")
            elif legacy_preds_path.exists():
                # Backward-compatible fallback for small/older runs. Large runs should not
                # produce this file because it can exceed available RAM.
                preds_df = pd.read_pickle(legacy_preds_path)
                logger.write(f"Loaded legacy full prediction table: {legacy_preds_path.name}")
            else:
                preds_df = pd.DataFrame()
                logger.write("No sampled prediction table found; point-level figures/tables may be skipped.")

        copy_log_snapshot(log_file, "mid_project_log_after_benchmarks.txt")

        stage_name = "table_building"
        table_builder = TableBuilder(engine.tables_dir, logger)
        existing_tables_pickle = engine.tables_dir / "all_tables_dict.pkl"
        existing_table_csvs = list(engine.tables_dir.glob("table_*.csv")) if engine.tables_dir.exists() else []
        existing_tables_available = existing_tables_pickle.exists() or len(existing_table_csvs) > 0




        should_build_tables = FORCE_REBUILD_TABLES or not (RESUME_IF_AVAILABLE and (pm.stage_done(stage_name) or existing_tables_available))
        if should_build_tables:
            if FORCE_REBUILD_TABLES and pm.stage_done(stage_name):
                logger.write("Rebuilding tables because FORCE_REBUILD_TABLES=True")
            tables = table_builder.build_all_tables(
                metrics_df,
                preds_df,
                master_df=loaded.data,
                split_group_columns=loaded.split_group_columns,
                merge_diagnostics=loaded.merge_diagnostics,
            )
            post_processor = PredictionPostProcessor(engine.preds_dir, engine.tables_dir, logger)
            exact_tables = post_processor.build_exact_posthoc_tables(metrics_df, master_df=loaded.data)
            if exact_tables:
                tables.update(exact_tables)
                logger.write(f"Integrated exact post-hoc tables: {sorted(exact_tables.keys())}")
            safe_pickle_dump(tables, engine.tables_dir / "all_tables_dict.pkl")
            pm.mark_stage_done(stage_name)
            pm.note("Table building and exact post-hoc analyses finished.")
            reporter.update()
        else:
            logger.write("Stage already complete or existing table files detected: table building")
            tables_pickle = engine.tables_dir / "all_tables_dict.pkl"
            if tables_pickle.exists():
                with open(tables_pickle, "rb") as f:
                    tables = pickle.load(f)
            else:
                # Reconstruct a lightweight tables dictionary from existing CSV exports.
                # This enables publication-table regeneration without recomputing analysis tables.
                tables = {}
                for csv_path in sorted(engine.tables_dir.glob("table_*.csv")):
                    key = csv_path.stem
                    if key.startswith("table_exact_"):
                        key = key.replace("table_exact_", "")
                    elif key.startswith("table_"):
                        key = key.replace("table_", "")
                    try:
                        df_loaded = pd.read_csv(csv_path, low_memory=False)
                        if not df_loaded.empty:
                            tables[key] = df_loaded
                    except Exception as exc:
                        logger.write(f"Could not load existing table CSV {csv_path.name}: {exc}")
                safe_pickle_dump(tables, tables_pickle)
                logger.write(f"Reconstructed tables dictionary from existing CSVs: {len(tables)} tables")

        stage_name = "figure_building"
        figure_builder = FigureBuilder(engine.fig_main_dir, engine.fig_si_dir, logger)
        should_build_figures = FORCE_REBUILD_FIGURES or not (RESUME_IF_AVAILABLE and pm.stage_done(stage_name))
        if should_build_figures:
            if FORCE_REBUILD_FIGURES and pm.stage_done(stage_name):
                logger.write("Regenerating figures because FORCE_REBUILD_FIGURES=True")
            figure_builder.build_all_figures(metrics_df, preds_df, tables, master_df=loaded.data)
            pm.mark_stage_done(stage_name)
            pm.note("Figure building finished.")
            reporter.update()
        else:
            logger.write("Stage already complete: figure building")

        stage_name = "latex_exports"
        exporter = ManuscriptExporter(OUTPUT_DIR, logger)
        should_export_latex = FORCE_REBUILD_LATEX or not (RESUME_IF_AVAILABLE and pm.stage_done(stage_name))
        if should_export_latex:
            if FORCE_REBUILD_LATEX and pm.stage_done(stage_name):
                logger.write("Regenerating LaTeX exports because FORCE_REBUILD_LATEX=True")
            exporter.export_key_tables(tables)
            pm.mark_stage_done(stage_name)
            pm.note("LaTeX exports finished.")
            reporter.update()
        else:
            logger.write("Stage already complete: LaTeX exports")

        stage_name = "final_summary"
        runtime_seconds = time.time() - start_time
        final_summary = {
            "runtime_seconds": runtime_seconds,
            "runtime_hours": runtime_seconds / 3600,
            "n_metrics_rows": int(len(metrics_df)),
            "n_prediction_rows": int(len(preds_df)),
            "targets_run": TARGET_COLUMNS,
            "anchor_target": ANCHOR_TARGET,
            "secondary_target": SECONDARY_TARGET,
            "descriptor_families": list(loaded.descriptor_families.keys()),
            "split_families": list(loaded.split_group_columns.keys()),
            "models": ["ridge", "rf", "hgb", "mlp"],
            "completed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_variant": PIPELINE_VARIANT,
            "memory_safe_patch": MEMORY_SAFE_PATCH,
            "force_rebuild_tables": FORCE_REBUILD_TABLES,
            "force_rebuild_figures": FORCE_REBUILD_FIGURES,
            "force_rebuild_latex": FORCE_REBUILD_LATEX,
            "max_onehot_categorical_levels": MAX_CATEGORICAL_LEVELS_FOR_ONEHOT,
            "permutation_importance_repeats": PERMUTATION_IMPORTANCE_REPEATS,
            "permutation_importance_max_rows": MAX_ROWS_FOR_PERMUTATION_IMPORTANCE,
            "model_grid_note": "RF/HGB/MLP grids lightened; dense preprocessing enforced for HGB/MLP; full predictions kept as per-experiment CSV files with memory-safe sampled table; exact post-hoc analyses stream per-experiment prediction files without RAM-heavy concatenation.",
        }
        safe_json_dump(final_summary, OUTPUT_DIR / "final_run_summary.json")
        pm.mark_stage_done(stage_name)
        pm.note("Final summary written.")
        reporter.update()

        copy_log_snapshot(log_file, "final_project_log.txt")

        logger.write("=" * 88)
        logger.write("Pipeline completed successfully.")
        logger.write(f"Total runtime: {runtime_seconds / 60:.2f} minutes")
        logger.write(f"Metrics rows saved: {len(metrics_df)}")
        logger.write(f"Prediction rows saved: {len(preds_df)}")
        logger.write(f"Metrics folder: {engine.metrics_dir}")
        logger.write(f"Predictions folder: {engine.preds_dir}")
        logger.write(f"Tables folder: {engine.tables_dir}")
        logger.write(f"Main figures folder: {engine.fig_main_dir}")
        logger.write(f"SI figures folder: {engine.fig_si_dir}")
        logger.write("=" * 88)

    except Exception as exc:
        logger.write(f"Pipeline failed with an exception: {repr(exc)}")
        pm.note(f"Pipeline failed: {repr(exc)}")
        reporter.update()
        copy_log_snapshot(log_file, "crash_log_snapshot.txt")
        raise





from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

V7_SPLIT_ORDER = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]
V7_SPLIT_COLORS = {
    "random": "#4E79A7",
    "geo_grouped": "#59A14F",
    "metal_grouped": "#E15759",
    "func_grouped": "#F28E2B",
    "linker_grouped": "#76B7B2",
    "topology_grouped": "#B07AA1",
    "chemistry_ensemble": "#9C755F",
}
V7_DESCRIPTOR_COLORS = {
    "compact_geometry": "#5F6F7F",
    "enriched_interpretable": "#D9A441",
    "geometry_plus_topology": "#59A14F",
    "topology_only": "#8E6C8A",
}
V7_GROUP_COLORS = {
    "geo_cluster": "#59A14F",
    "metal_cluster": "#E15759",
    "func_cluster": "#F28E2B",
    "linker_cluster": "#76B7B2",
    TOPOLOGY_COLUMN_GROUPED: "#B07AA1",
}
V7_SHORT_SPLIT_LABELS = {
    "random": "Random",
    "geo_grouped": "Geometry",
    "metal_grouped": "Metal",
    "func_grouped": "Functional",
    "linker_grouped": "Linker",
    "topology_grouped": "Topology",
    "chemistry_ensemble": "Chemistry",
}
V7_SHORT_DESCRIPTOR_LABELS = {
    "compact_geometry": "Compact geometry",
    "enriched_interpretable": "Enriched descriptors",
    "geometry_plus_topology": "Geometry + topology",
    "topology_only": "Topology only",
}


def _v7_configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 320,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlepad": 9,
    })


def _v7_split_label(split: str) -> str:
    return V7_SHORT_SPLIT_LABELS.get(str(split), str(split).replace("_", " ").title())


def _v7_descriptor_label(desc: str) -> str:
    return V7_SHORT_DESCRIPTOR_LABELS.get(str(desc), str(desc).replace("_", " ").title())


def _v7_group_label(group_col: str, group_id: str) -> str:
    prefix = {
        "geo_cluster": "geo",
        "metal_cluster": "metal",
        "func_cluster": "func",
        "linker_cluster": "linker",
        TOPOLOGY_COLUMN_GROUPED: "topology",
    }.get(str(group_col), str(group_col).replace("_cluster", ""))
    gid = str(group_id).replace(".0", "")
    return f"{prefix}:{gid}"


def _v7_pair_label(pair: str) -> str:
    pair = str(pair)
    replacements = {
        "random": "Random",
        "geo_grouped": "Geo",
        "metal_grouped": "Metal",
        "func_grouped": "Func",
        "linker_grouped": "Linker",
        "topology_grouped": "Topology",
    }
    for old, new in replacements.items():
        pair = pair.replace(old, new)
    return pair.replace("_", " ")


def _v7_ordered_splits(values: Sequence[str]) -> List[str]:
    present = set(str(v) for v in values if pd.notna(v))
    return [s for s in V7_SPLIT_ORDER if s in present] + sorted([s for s in present if s not in V7_SPLIT_ORDER])


def _v7_apply_axis_style(ax, grid_axis: str = "y") -> None:
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _v7_sem(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().values
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _v7_summary_sem(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(list(group_cols), observed=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{m}_sem"] = _v7_sem(vals)
            row[f"{m}_std"] = float(vals.std()) if vals.notna().sum() > 1 else 0.0
            row[f"{m}_n"] = int(vals.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


_V6_EXACT_POSTHOC_BUILD = PredictionPostProcessor.build_exact_posthoc_tables


def _v7_build_split_severity_decomposition(metrics_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    required = {"target_slug", "descriptor_family", "model", "fold_id", "split_family", "r2", "mae", "shift_centroid_distance", "shift_avg_wasserstein"}
    if not required.issubset(set(metrics_df.columns)):
        return pd.DataFrame()

    rand = metrics_df[metrics_df["split_family"] == "random"][["target_slug", "descriptor_family", "model", "fold_id", "r2", "mae"]].rename(
        columns={"r2": "random_r2", "mae": "random_mae"}
    )
    grouped = metrics_df[metrics_df["split_family"] != "random"].copy()
    merged = grouped.merge(rand, on=["target_slug", "descriptor_family", "model", "fold_id"], how="left")
    merged["r2_loss_vs_random"] = merged["random_r2"] - merged["r2"]
    merged["mae_increase_vs_random"] = merged["mae"] - merged["random_mae"]

    rows = []
    for split_family, sub in merged.groupby("split_family", observed=False):
        row = {
            "split_family": split_family,
            "n_fold_comparisons": int(len(sub)),
            "mean_r2_loss_vs_random": float(sub["r2_loss_vs_random"].mean()),
            "mean_mae_increase_vs_random": float(sub["mae_increase_vs_random"].mean()),
            "mean_centroid_shift": float(sub["shift_centroid_distance"].mean()),
            "mean_wasserstein_shift": float(sub["shift_avg_wasserstein"].mean()),
        }
        valid = sub[["shift_centroid_distance", "r2_loss_vs_random"]].dropna()
        row["spearman_shift_vs_r2_loss"] = (
            float(stats.spearmanr(valid["shift_centroid_distance"], valid["r2_loss_vs_random"]).correlation)
            if len(valid) >= 3 and valid["shift_centroid_distance"].nunique() > 1 else np.nan
        )
        valid = sub[["shift_avg_wasserstein", "mae_increase_vs_random"]].dropna()
        row["spearman_wasserstein_vs_mae_increase"] = (
            float(stats.spearmanr(valid["shift_avg_wasserstein"], valid["mae_increase_vs_random"]).correlation)
            if len(valid) >= 3 and valid["shift_avg_wasserstein"].nunique() > 1 else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_r2_loss_vs_random", ascending=False)
    out["scope_note"] = "Split-consequence decomposition only; not a topology-residual or local trust-map analysis."
    out.to_csv(tables_dir / "table_split_severity_decomposition.csv", index=False)
    merged.to_csv(tables_dir / "table_split_severity_fold_level_losses.csv", index=False)
    return out


def _v7_build_elite_dropout_candidates(self: PredictionPostProcessor, reg: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if reg is None or reg.empty:
        return pd.DataFrame()
    target_slug = CASE_STUDY_TARGET_SLUG
    descriptor = CASE_STUDY_DESCRIPTOR_FAMILY
    model = CASE_STUDY_MODEL
    random_pred = self._mean_predictions_for_combo(reg, target_slug, descriptor, model, "random")
    if random_pred.empty:
        return pd.DataFrame()

    rows = []
    for split_family in ["geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]:
        grouped_pred = self._mean_predictions_for_combo(reg, target_slug, descriptor, model, split_family)
        if grouped_pred.empty:
            continue
        merged = random_pred.rename(columns={"mean_prediction": "random_prediction", "mean_target": "target_value"}).merge(
            grouped_pred.rename(columns={"mean_prediction": "grouped_prediction", "mean_target": "target_value_grouped"}),
            on="filename_norm",
            how="inner",
        )
        if len(merged) < 100:
            continue
        merged["target_value"] = merged["target_value"].fillna(merged["target_value_grouped"])
        merged["random_rank"] = merged["random_prediction"].rank(ascending=False, method="first")
        merged["grouped_rank"] = merged["grouped_prediction"].rank(ascending=False, method="first")
        merged["rank_shift_grouped_minus_random"] = merged["grouped_rank"] - merged["random_rank"]
        k = max(1, int(math.ceil(0.05 * len(merged))))
        merged["random_top5"] = merged["random_rank"] <= k
        merged["grouped_top5"] = merged["grouped_rank"] <= k
        moved = merged[merged["random_top5"] != merged["grouped_top5"]].copy()
        if moved.empty:
            continue
        moved["elite_transition"] = np.where(moved["random_top5"], "Dropped from random elite", "Entered grouped elite")
        moved["split_family"] = split_family
        moved["top_k"] = k
        rows.append(moved)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if master_df is not None and not master_df.empty:
        meta_cols = [
            "filename_norm", "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
            "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED,
        ]
        meta_cols = [c for c in meta_cols if c in master_df.columns]
        out = out.merge(master_df[meta_cols].drop_duplicates("filename_norm"), on="filename_norm", how="left")
    out["abs_rank_shift"] = out["rank_shift_grouped_minus_random"].abs()
    out = out.sort_values(["abs_rank_shift", "split_family"], ascending=[False, True]).head(500).reset_index(drop=True)
    out["scope_note"] = "Split-induced elite-list transition; use for screening-consequence case studies, not topology-residual claims."
    keep = [
        "filename_norm", "split_family", "elite_transition", "target_value", "random_prediction", "grouped_prediction",
        "random_rank", "grouped_rank", "rank_shift_grouped_minus_random", "top_k",
        "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
        "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED, "scope_note",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    out.to_csv(self.tables_dir / "table_elite_dropout_candidates.csv", index=False)
    return out


def _v7_build_exact_posthoc_tables(self: PredictionPostProcessor, metrics_df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    tables = _V6_EXACT_POSTHOC_BUILD(self, metrics_df, master_df=master_df)
    severity = _v7_build_split_severity_decomposition(metrics_df, self.tables_dir)
    if not severity.empty:
        tables["split_severity_decomposition"] = severity
    reg = self._load_registry()
    dropout = _v7_build_elite_dropout_candidates(self, reg, master_df)
    if not dropout.empty:
        tables["elite_dropout_candidates"] = dropout
    return tables


PredictionPostProcessor.build_exact_posthoc_tables = _v7_build_exact_posthoc_tables


_V6_EXPORT_KEY_TABLES = ManuscriptExporter.export_key_tables


def _v7_export_key_tables(self: ManuscriptExporter, tables: Dict[str, pd.DataFrame]) -> None:
    _V6_EXPORT_KEY_TABLES(self, tables)
    extra_mapping = {
        "split_severity_decomposition": ("latex_split_severity_decomposition.tex", "Split-severity decomposition linking grouped-holdout performance loss to distribution shift.", "tab:split_severity_decomposition"),
        "elite_dropout_candidates": ("latex_elite_dropout_candidates.tex", "Candidate MOFs whose top-5\\% elite status changes between random and grouped evaluation.", "tab:elite_dropout_candidates"),
    }
    for key, (filename, caption, label) in extra_mapping.items():
        if key in tables and not tables[key].empty:
            self.dataframe_to_latex(tables[key], self.latex_dir / filename, caption=caption, label=label, index=False)
    self.logger.write("V7 extra LaTeX exports saved.")


ManuscriptExporter.export_key_tables = _v7_export_key_tables


def _v7_savefig(self: FigureBuilder, fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout(pad=1.15)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


FigureBuilder._savefig = _v7_savefig


def _v7_figure_1_workflow(self: FigureBuilder) -> None:
    _v7_configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    ax.set_axis_off()
    boxes = [
        (0.06, 0.66, 0.22, 0.17, "Data", "clean adsorption table\ncluster labels\noptional topology metadata", "#EAF2F8"),
        (0.39, 0.66, 0.22, 0.17, "Matched design", "same targets\nsame descriptors\nsame model grid", "#EEF6EA"),
        (0.72, 0.66, 0.22, 0.17, "Split families", "random\ngeometry / chemistry\ntopology stress test", "#F7EEF6"),
        (0.06, 0.30, 0.22, 0.17, "Models", "Ridge, RF, HGB, MLP\nmain text: RF-centered\nSI: all models", "#FDF3E7"),
        (0.39, 0.30, 0.22, 0.17, "Consequences", "accuracy loss\nrank inversion\nelite-list instability", "#FFF8E6"),
        (0.72, 0.30, 0.22, 0.17, "Outputs", "tables + figure data\nPNG/PDF figures\ncase-study candidates", "#EEF1F5"),
    ]
    for x, y, w, h, head, body, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.018",
                             linewidth=1.25, edgecolor="#3A3A3A", facecolor=color, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.02, y + h - 0.045, head, transform=ax.transAxes, ha="left", va="center",
                fontsize=12.5, fontweight="bold", color="#222222")
        ax.text(x + 0.02, y + h - 0.085, body, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, color="#333333", linespacing=1.35)
    arrows = [((0.28, 0.745), (0.39, 0.745)), ((0.61, 0.745), (0.72, 0.745)),
              ((0.17, 0.66), (0.17, 0.47)), ((0.50, 0.66), (0.50, 0.47)),
              ((0.83, 0.66), (0.83, 0.47)), ((0.28, 0.385), (0.39, 0.385)), ((0.61, 0.385), (0.72, 0.385))]
    for start, end in arrows:
        arr = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.2, color="#303030")
        ax.add_patch(arr)
    ax.text(0.5, 0.92, "Split-strategy benchmark: matched models, changing generalization regime",
            transform=ax.transAxes, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.08, "Scope boundary: the main claim is split-induced change in benchmark conclusions, not topology-residual mechanism.",
            transform=ax.transAxes, ha="center", va="center", fontsize=10.5, color="#555555")
    figure_path = self.fig_main_dir / "figure1_v7_polished_workflow.png"
    figure_data = pd.DataFrame(boxes, columns=["x", "y", "width", "height", "heading", "body", "fill_color"])
    self._save_figure_data_csv(figure_path, figure_data)
    self._savefig(fig, figure_path)


def _v7_figure_2_split_severity(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    model = "rf" if "rf" in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "mlp", "ridge"])
    sub = metrics_df[(metrics_df["target_slug"] == anchor_slug) &
                     (metrics_df["model"] == model) &
                     (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"]))].copy()
    if sub.empty:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    summary = _v7_summary_sem(sub, ["split_family", "descriptor_family"], ["r2", "mae"])
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), sharex=True)
    offsets = {"compact_geometry": -0.08, "enriched_interpretable": 0.08}
    markers = {"compact_geometry": "o", "enriched_interpretable": "s"}
    for ax, metric, ylabel, title in [
        (axes[0], "r2", "Mean R$^2$", "Predictive accuracy"),
        (axes[1], "mae", "Mean MAE", "Error under grouped extrapolation"),
    ]:
        x = np.arange(len(split_order), dtype=float)
        for desc in ["compact_geometry", "enriched_interpretable"]:
            temp = summary[summary["descriptor_family"] == desc].set_index("split_family").reindex(split_order)
            y = temp[f"{metric}_mean"].values.astype(float)
            yerr = temp[f"{metric}_sem"].fillna(0).values.astype(float)
            ax.errorbar(x + offsets[desc], y, yerr=yerr, marker=markers[desc], linewidth=1.8,
                        markersize=6.5, capsize=3, color=V7_DESCRIPTOR_COLORS[desc], label=_v7_descriptor_label(desc))
        for i, sp in enumerate(split_order):
            ax.axvspan(i - 0.45, i + 0.45, color=V7_SPLIT_COLORS.get(sp, "#DDDDDD"), alpha=0.055, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([_v7_split_label(s) for s in split_order], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _v7_apply_axis_style(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Figure 2. Split-family severity for the anchor target ({anchor_slug}; model = {model.upper()})", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure2_v7_split_family_severity_polished.png"
    self._save_figure_data_csv(figure_path, {"plot_summary": summary, "raw_fold_metrics": sub})
    self._savefig(fig, figure_path)


def _v7_figure_3_target_sensitivity(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    model = "rf" if "rf" in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "mlp", "ridge"])
    descriptor = "enriched_interpretable"
    sub = metrics_df[(metrics_df["model"] == model) & (metrics_df["descriptor_family"] == descriptor)].copy()
    if sub.empty:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    target_order = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(sub["target_slug"])]
    r2 = sub.groupby(["target_slug", "split_family"], observed=False)["r2"].mean().reset_index()
    r2_piv = r2.pivot_table(index="target_slug", columns="split_family", values="r2", observed=False).reindex(index=target_order, columns=split_order)
    delta = r2_piv.sub(r2_piv["random"], axis=0) if "random" in r2_piv.columns else r2_piv * np.nan
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.7))
    im0 = axes[0].imshow(r2_piv.values, aspect="auto", cmap="viridis", vmin=np.nanmin(r2_piv.values), vmax=np.nanmax(r2_piv.values))
    max_abs = np.nanmax(np.abs(delta.values)) if np.isfinite(delta.values).any() else 1.0
    im1 = axes[1].imshow(delta.values, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    for ax, pivot, title, im, fmt in [
        (axes[0], r2_piv, "Mean R$^2$", im0, ".2f"),
        (axes[1], delta, "R$^2$ change relative to random", im1, ".2f"),
    ]:
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_v7_split_label(c) for c in pivot.columns], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                label = f"{val:{fmt}}" if np.isfinite(val) else "NA"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color="#111111")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
        _v7_apply_axis_style(ax, grid_axis="")
    fig.suptitle(f"Figure 3. Target-dependent sensitivity to split family ({descriptor.replace('_', ' ')}, {model.upper()})", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure3_v7_target_split_sensitivity_polished.png"
    self._save_figure_data_csv(figure_path, {"mean_r2": r2_piv.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="mean_r2"),
                                             "delta_r2_vs_random": delta.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="delta_r2")})
    self._savefig(fig, figure_path)


def _v7_figure_4_screening(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    model = "rf" if "rf" in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "mlp", "ridge"])
    sub = metrics_df[(metrics_df["target_slug"] == anchor_slug) &
                     (metrics_df["model"] == model) &
                     (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"]))].copy()
    if sub.empty or "top_5pct_overlap" not in sub.columns:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    summary = _v7_summary_sem(sub, ["split_family", "descriptor_family"], ["top_5pct_overlap", "top_5pct_enrichment"])
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.7), sharex=True)
    offsets = {"compact_geometry": -0.08, "enriched_interpretable": 0.08}
    for ax, metric, ylabel, title in [
        (axes[0], "top_5pct_overlap", "Top-5% overlap", "Elite recovery"),
        (axes[1], "top_5pct_enrichment", "Top-5% enrichment", "Enrichment over random selection"),
    ]:
        x = np.arange(len(split_order), dtype=float)
        for desc in ["compact_geometry", "enriched_interpretable"]:
            temp = summary[summary["descriptor_family"] == desc].set_index("split_family").reindex(split_order)
            ax.errorbar(x + offsets[desc], temp[f"{metric}_mean"], yerr=temp[f"{metric}_sem"].fillna(0),
                        marker="o" if desc == "compact_geometry" else "s", linewidth=1.8, markersize=6.5,
                        capsize=3, color=V7_DESCRIPTOR_COLORS[desc], label=_v7_descriptor_label(desc))
        for i, sp in enumerate(split_order):
            ax.axvspan(i - 0.45, i + 0.45, color=V7_SPLIT_COLORS.get(sp, "#DDDDDD"), alpha=0.055, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([_v7_split_label(s) for s in split_order], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _v7_apply_axis_style(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Figure 4. Screening consequences of split choice ({anchor_slug}; model = {model.upper()})", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure4_v7_screening_consequences_polished.png"
    self._save_figure_data_csv(figure_path, {"plot_summary": summary, "raw_fold_metrics": sub})
    self._savefig(fig, figure_path)


def _v7_figure_5_group_lollipop(self: FigureBuilder, group_error_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if group_error_df is None or group_error_df.empty:
        self.logger.write("Exact group error table is empty; V7 Figure 5 skipped.")
        return
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = group_error_df[group_error_df["target_slug"] == anchor_slug].copy()
    if sub.empty:
        sub = group_error_df.copy()
    descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "compact_geometry", "geometry_plus_topology"])
    model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", ["rf", "hgb", "mlp", "ridge"])
    if descriptor is not None:
        sub = sub[sub["descriptor_family"] == descriptor].copy()
    if model is not None:
        sub = sub[sub["model"] == model].copy()
    geo = sub[(sub["group_column"] == "geo_cluster") & (sub["split_family"] == "geo_grouped")].copy()
    if geo.empty:
        geo = sub[sub["group_column"] == "geo_cluster"].copy()
    geo = geo.sort_values("mean_abs_error", ascending=False).head(12)
    chem_parts = []
    for group_col, split_name in [("metal_cluster", "metal_grouped"), ("func_cluster", "func_grouped"), ("linker_cluster", "linker_grouped"), (TOPOLOGY_COLUMN_GROUPED, "topology_grouped")]:
        chem_parts.append(sub[(sub["group_column"] == group_col) & (sub["split_family"] == split_name)].copy())
    chem = pd.concat([p for p in chem_parts if not p.empty], ignore_index=True) if chem_parts else pd.DataFrame()
    if chem.empty:
        chem = sub[sub["group_column"].isin(["metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED])].copy()
    chem = chem.sort_values("mean_abs_error", ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6))
    for ax, temp, title in [(axes[0], geo, "Hardest geometry hold-outs"), (axes[1], chem, "Hardest chemistry/topology hold-outs")]:
        if temp.empty:
            ax.axis("off")
            ax.set_title(title + " (no eligible groups)")
            continue
        temp = temp.sort_values("mean_abs_error", ascending=True).reset_index(drop=True)
        y = np.arange(len(temp))
        colors = [V7_GROUP_COLORS.get(gc, "#4E79A7") for gc in temp["group_column"]]
        labels = [_v7_group_label(r["group_column"], r.get("group_id", r.get("group_label", ""))) for _, r in temp.iterrows()]
        sem = temp["std_abs_error"].fillna(0).values / np.sqrt(temp["n"].replace(0, np.nan).values)
        ax.hlines(y, 0, temp["mean_abs_error"].values, color="#CCCCCC", linewidth=1.6)
        ax.errorbar(temp["mean_abs_error"].values, y, xerr=sem, fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2)
        ax.scatter(temp["mean_abs_error"].values, y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Mean absolute error")
        ax.set_title(title)
        _v7_apply_axis_style(ax, grid_axis="x")
    axes[0].text(0.02, 0.04, f"{anchor_slug}; {descriptor}; {model}", transform=axes[0].transAxes, fontsize=9.5, color="#555555")
    fig.suptitle("Figure 5. Concrete held-out groups behind split-severity", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure5_v7_hard_group_lollipop_polished.png"
    self._save_figure_data_csv(figure_path, {"geometry_groups": geo, "chemistry_topology_groups": chem})
    self._savefig(fig, figure_path)


def _v7_figure_6_shift_and_space(self: FigureBuilder, metrics_df: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = metrics_df[metrics_df["target_slug"] == anchor_slug].copy()
    if sub.empty or not {"shift_centroid_distance", "shift_avg_wasserstein", "r2"}.issubset(sub.columns):
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    for split_family, temp in sub.groupby("split_family", observed=False):
        axes[0].scatter(temp["shift_centroid_distance"], temp["r2"], s=46, alpha=0.72,
                        color=V7_SPLIT_COLORS.get(split_family, "#999999"), label=_v7_split_label(split_family),
                        edgecolor="white", linewidth=0.35)
    valid = sub[["shift_centroid_distance", "r2"]].dropna()
    corr = np.nan
    if len(valid) >= 3 and valid["shift_centroid_distance"].nunique() > 1:
        corr = stats.spearmanr(valid["shift_centroid_distance"], valid["r2"]).correlation
        slope, intercept, *_ = stats.linregress(valid["shift_centroid_distance"], valid["r2"])
        xline = np.linspace(valid["shift_centroid_distance"].min(), valid["shift_centroid_distance"].max(), 120)
        axes[0].plot(xline, intercept + slope * xline, color="#333333", linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Centroid shift distance")
    axes[0].set_ylabel("R$^2$")
    axes[0].set_title(f"Shift vs accuracy (ρ = {corr:.2f})")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    _v7_apply_axis_style(axes[0])

    shift_bar = sub.groupby("split_family", observed=False)["shift_avg_wasserstein"].mean().reindex(split_order)
    y = np.arange(len(shift_bar))
    axes[1].barh(y, shift_bar.values, color=[V7_SPLIT_COLORS.get(s, "#999999") for s in shift_bar.index], alpha=0.9)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([_v7_split_label(s) for s in shift_bar.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean standardized Wasserstein shift")
    axes[1].set_title("Shift severity by split family")
    _v7_apply_axis_style(axes[1], grid_axis="x")

    needed = [c for c in ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"] if master_df is not None and c in master_df.columns]
    if master_df is not None and len(needed) >= 2 and "geo_cluster" in master_df.columns:
        pca_sub = master_df[needed + ["geo_cluster"]].dropna(subset=needed).copy()
        if len(pca_sub) > MAX_ROWS_FOR_VISUAL_PCA:
            pca_sub = pca_sub.sample(MAX_ROWS_FOR_VISUAL_PCA, random_state=RANDOM_SEED)
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(pca_sub[needed].values))
        counts = pca_sub["geo_cluster"].astype(str).value_counts()
        highlight_groups = set(counts.head(5).index.tolist())
        highlight = pca_sub["geo_cluster"].astype(str).isin(highlight_groups).values
        axes[2].scatter(coords[~highlight, 0], coords[~highlight, 1], s=5, alpha=0.16, color="#9E9E9E", linewidths=0)
        axes[2].scatter(coords[highlight, 0], coords[highlight, 1], s=13, alpha=0.75, color=V7_SPLIT_COLORS["geo_grouped"], edgecolor="white", linewidth=0.15)
        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        axes[2].set_title("Descriptor-space example\n(top geometry groups highlighted)")
        pca_data = pca_sub.copy()
        pca_data["pca_pc1"] = coords[:, 0]
        pca_data["pca_pc2"] = coords[:, 1]
        pca_data["highlighted_geometry_group"] = highlight
    else:
        axes[2].axis("off")
        pca_data = pd.DataFrame()
    _v7_apply_axis_style(axes[2])
    fig.suptitle("Figure 6. Distribution shift explains why split families are not interchangeable", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure6_v7_shift_and_descriptor_space_polished.png"
    self._save_figure_data_csv(figure_path, {"shift_points": sub, "shift_bar": shift_bar.reset_index().rename(columns={"shift_avg_wasserstein": "mean_shift_avg_wasserstein"}), "pca_example": pca_data})
    self._savefig(fig, figure_path)


def _v7_figure_7_elite_stability(self: FigureBuilder, elite_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if elite_df is None or elite_df.empty:
        self.logger.write("Elite-list stability table is empty; V7 Figure 7 skipped.")
        return
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = elite_df[(elite_df["target_slug"] == anchor_slug) & (elite_df["top_fraction"] == 0.05)].copy()
    if sub.empty:
        sub = elite_df[elite_df["top_fraction"] == 0.05].copy()
    descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "compact_geometry", "geometry_plus_topology"])
    model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", ["rf", "hgb", "mlp", "ridge"])
    if descriptor is not None:
        sub = sub[sub["descriptor_family"] == descriptor].copy()
    if model is not None:
        sub = sub[sub["model"] == model].copy()
    if sub.empty:
        return
    sub["pair"] = sub["split_family_1"].astype(str) + " vs " + sub["split_family_2"].astype(str)
    top = sub.sort_values("elite_jaccard", ascending=True).head(12).copy()
    colors = [V7_SPLIT_COLORS["topology_grouped"] if "topology_grouped" in p else "#7F8C8D" for p in top["pair"]]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2))
    y = np.arange(len(top))
    axes[0].barh(y, top["elite_jaccard"].values, color=colors, alpha=0.92)
    axes[0].axvline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_v7_pair_label(p) for p in top["pair"]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Top-5% Jaccard overlap")
    axes[0].set_title("Least stable elite shortlists")
    _v7_apply_axis_style(axes[0], grid_axis="x")
    axes[1].barh(y, top["prediction_spearman"].values, color=colors, alpha=0.92)
    axes[1].axvline(0.9, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([_v7_pair_label(p) for p in top["pair"]])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Overall prediction-rank Spearman")
    axes[1].set_title("Global rank agreement can remain high")
    _v7_apply_axis_style(axes[1], grid_axis="x")
    fig.suptitle(f"Figure 7. Elite-list instability despite broad rank agreement ({anchor_slug}; {descriptor}; {model.upper()})", fontsize=14, fontweight="bold")
    figure_path = self.fig_main_dir / "figure7_v7_exact_elite_instability_polished.png"
    self._save_figure_data_csv(figure_path, {"elite_stability_exact_top5": sub, "least_stable_pairs": top})
    self._savefig(fig, figure_path)


def _v7_si_error_vs_rank(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    metric = "top_5pct_overlap"
    if metric not in metrics_df.columns:
        return
    summary = metrics_df.groupby(["split_family", "descriptor_family", "model"], observed=False).agg(
        mae_mean=("mae", "mean"), overlap_mean=(metric, "mean"), r2_mean=("r2", "mean")
    ).reset_index()
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for split_family, sub in summary.groupby("split_family", observed=False):
        ax.scatter(sub["mae_mean"], sub["overlap_mean"], s=66, alpha=0.82,
                   color=V7_SPLIT_COLORS.get(split_family, "#888888"), label=_v7_split_label(split_family),
                   edgecolor="white", linewidth=0.4)
    label_df = pd.concat([
        summary.nsmallest(4, "overlap_mean"),
        summary.nlargest(3, "mae_mean"),
        summary.nlargest(3, "overlap_mean"),
    ]).drop_duplicates()
    for _, r in label_df.iterrows():
        ax.text(r["mae_mean"], r["overlap_mean"], f" {r['model']} | {_v7_descriptor_label(r['descriptor_family'])}", fontsize=8.3, va="center")
    ax.set_xlabel("Mean MAE")
    ax.set_ylabel("Top-5% overlap")
    ax.set_title("SI. Error versus screening quality")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    _v7_apply_axis_style(ax)
    figure_path = self.fig_si_dir / "si_v7_error_vs_screening_rank_clean.png"
    self._save_figure_data_csv(figure_path, summary)
    self._savefig(fig, figure_path)


def _v7_si_group_size(self: FigureBuilder, group_size_table: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if group_size_table is None or group_size_table.empty:
        return
    table = group_size_table.copy()
    table["split_family"] = pd.Categorical(table["split_family"], categories=_v7_ordered_splits(table["split_family"]), ordered=True)
    table = table.sort_values("split_family")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(table))
    ax.bar(x, table["mean_group_size"].values, color=[V7_SPLIT_COLORS.get(str(s), "#888888") for s in table["split_family"]], alpha=0.85, label="Mean")
    ax.scatter(x, table["median_group_size"].values, color="#222222", s=42, label="Median", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_v7_split_label(str(s)) for s in table["split_family"]], rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Group size (log scale)")
    ax.set_title("SI. Group-size diagnostic summary")
    ax.legend(frameon=False)
    _v7_apply_axis_style(ax)
    figure_path = self.fig_si_dir / "si_v7_group_size_summary_logscale.png"
    self._save_figure_data_csv(figure_path, table)
    self._savefig(fig, figure_path)


def _v7_build_all_figures(self: FigureBuilder, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], master_df: Optional[pd.DataFrame] = None) -> None:
    if metrics_df.empty:
        self.logger.write("No metrics available; skipping figure generation.")
        return
    _v7_configure_matplotlib()
    self.figure_1_study_design_overview()
    _v7_figure_2_split_severity(self, metrics_df)
    _v7_figure_3_target_sensitivity(self, metrics_df)
    _v7_figure_4_screening(self, metrics_df)
    _v7_figure_5_group_lollipop(self, tables.get("group_resolved_errors", pd.DataFrame()))
    _v7_figure_6_shift_and_space(self, metrics_df, master_df)
    _v7_figure_7_elite_stability(self, tables.get("elite_list_stability", pd.DataFrame()))

    if not preds_df.empty:
        self.si_figure_target_distributions(preds_df)
        self.si_figure_pld_decile_errors(preds_df)
        self.si_figure_prediction_scatter_panels(preds_df)
    else:
        self.logger.write("Sampled prediction table empty; point-level SI figures skipped.")
    self.si_figure_variance_decomposition(tables.get("variance_decomposition", pd.DataFrame()))
    self.si_figure_splitwise_metric_heatmaps(metrics_df)
    _v7_si_group_size(self, tables.get("split_group_size_summary", pd.DataFrame()))
    _v7_si_error_vs_rank(self, metrics_df)


FigureBuilder.figure_1_study_design_overview = _v7_figure_1_workflow
FigureBuilder.build_all_figures = _v7_build_all_figures






V8_PATCH_NOTE = "v8_clear_target_labels_panel_letters_table_manifests"

V8_TARGET_LABELS_MPL = {
    "co2_0015bar": r"CO$_2$ at 0.015 bar",
    "co2_015bar": r"CO$_2$ at 0.15 bar",
    "ch4_58bar": r"CH$_4$ at 5.8 bar",
    "ch4_65bar": r"CH$_4$ at 65 bar",
}
V8_TARGET_LABELS_TEX = {
    "co2_0015bar": r"CO$_2$ at 0.015 bar",
    "co2_015bar": r"CO$_2$ at 0.15 bar",
    "ch4_58bar": r"CH$_4$ at 5.8 bar",
    "ch4_65bar": r"CH$_4$ at 65 bar",
}
V8_TARGET_LABELS_PLAIN = {
    "co2_0015bar": "CO2 at 0.015 bar",
    "co2_015bar": "CO2 at 0.15 bar",
    "ch4_58bar": "CH4 at 5.8 bar",
    "ch4_65bar": "CH4 at 65 bar",
}


def _v8_target_label(target_slug: str, mode: str = "mpl") -> str:
    mapping = V8_TARGET_LABELS_TEX if mode == "tex" else V8_TARGET_LABELS_MPL if mode == "mpl" else V8_TARGET_LABELS_PLAIN
    return mapping.get(str(target_slug), str(target_slug).replace("_", " "))


def _v8_panel_label(ax, letter: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, fontweight="bold", color="#222222")


def _v8_slug_columns_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["target_slug", "target_name"]:
        if col in out.columns:
            out[col + "_display"] = out[col].astype(str).map(lambda x: _v8_target_label(x, mode="plain"))
    return out


def _v8_figure_1_workflow(self: FigureBuilder) -> None:
    _v7_configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14.8, 7.9))
    ax.set_axis_off()

    boxes = [
        (0.055, 0.665, 0.245, 0.180, "Data", "clean adsorption table\ncluster labels\noptional topology metadata", "#EAF2F8"),
        (0.382, 0.665, 0.245, 0.180, "Matched design", "same targets\nsame descriptors\nsame model grid", "#EEF6EA"),
        (0.710, 0.665, 0.245, 0.180, "Split families", "random\ngeometry / chemistry\ntopology stress test", "#F7EEF6"),
        (0.055, 0.315, 0.245, 0.180, "Models", "Ridge, RF, HGB, MLP\nmain text: RF-centred\nSI: all models", "#FDF3E7"),
        (0.382, 0.315, 0.245, 0.180, "Consequences", "accuracy loss\nrank inversion\nelite-list instability", "#FFF8E6"),
        (0.710, 0.315, 0.245, 0.180, "Outputs", "tables + figure data\nPNG/PDF figures\ncase-study candidates", "#EEF1F5"),
    ]
    for x, y, w, h, head, body, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.018",
                             linewidth=1.25, edgecolor="#3A3A3A", facecolor=color, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + 0.038, y + h - 0.050, head, transform=ax.transAxes, ha="left", va="center",
                fontsize=12.4, fontweight="bold", color="#222222")
        ax.text(x + 0.038, y + h - 0.092, body, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.4, color="#333333", linespacing=1.32)


    arrows = [
        ((0.300, 0.755), (0.382, 0.755)), 
        ((0.627, 0.755), (0.710, 0.755)),  
        
        ((0.177, 0.665), (0.177, 0.495)),  
        ((0.505, 0.665), (0.505, 0.495)),  
        ((0.833, 0.665), (0.833, 0.495)),  
        ((0.300, 0.405), (0.382, 0.405)), 
        ((0.627, 0.405), (0.710, 0.405)),  
    ]
    for start, end in arrows:
        arr = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=15,
                              linewidth=1.25, color="#303030", shrinkA=2.0, shrinkB=2.0)
        ax.add_patch(arr)

    ax.text(0.5, 0.920, "Split-strategy benchmark: matched models, changing generalization regime",
            transform=ax.transAxes, ha="center", va="center", fontsize=15.2, fontweight="bold")
    ax.text(0.5, 0.085, "Scope boundary: the main claim is split-induced change in benchmark conclusions, not topology-residual mechanism.",
            transform=ax.transAxes, ha="center", va="center", fontsize=10.4, color="#555555")
    figure_path = self.fig_main_dir / "figure1_v8_polished_workflow.png"
    figure_data = pd.DataFrame(boxes, columns=["x", "y", "width", "height", "heading", "body", "color"])
    self._save_figure_data_csv(figure_path, figure_data)
    self._savefig(fig, figure_path)


def _v8_figure_2_split_severity(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = metrics_df[(metrics_df["target_slug"] == anchor_slug) &
                     (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"])) &
                     (metrics_df["model"] == CASE_STUDY_MODEL)].copy()
    if sub.empty:
        sub = metrics_df[(metrics_df["target_slug"] == anchor_slug) &
                         (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"]))].copy()
    if sub.empty:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    summary = _v7_summary_sem(sub, ["split_family", "descriptor_family"], ["r2", "mae"])
    summary["split_family"] = pd.Categorical(summary["split_family"], categories=split_order, ordered=True)
    summary = summary.sort_values(["split_family", "descriptor_family"])
    x = np.arange(len(split_order))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.4))
    for ax, metric, ylabel, title, letter in [
        (axes[0], "r2", r"Mean R$^2$", "Predictive accuracy", "a"),
        (axes[1], "mae", "Mean MAE", "Error under grouped extrapolation", "b"),
    ]:
        for i, desc in enumerate(["compact_geometry", "enriched_interpretable"]):
            temp = summary[summary["descriptor_family"] == desc].set_index("split_family").reindex(split_order)
            offset = (i - 0.5) * width
            ax.errorbar(x + offset, temp[f"{metric}_mean"].values, yerr=temp[f"{metric}_sem"].values,
                        marker="o" if desc == "compact_geometry" else "s", linewidth=2.0, markersize=6,
                        capsize=3, color=V7_DESCRIPTOR_COLORS.get(desc, "#777777"), label=_v7_descriptor_label(desc))
        for j, split in enumerate(split_order):
            ax.axvspan(j - 0.45, j + 0.45, color=V7_SPLIT_COLORS.get(split, "#CCCCCC"), alpha=0.07, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([_v7_split_label(s) for s in split_order], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _v8_panel_label(ax, letter)
        _v7_apply_axis_style(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Figure 2. Split-family severity for the anchor target ({_v8_target_label(anchor_slug)}; model = {CASE_STUDY_MODEL.upper()})",
                 fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure2_v8_split_family_severity_polished.png"
    self._save_figure_data_csv(figure_path, summary)
    self._savefig(fig, figure_path)


def _v8_figure_3_target_sensitivity(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    descriptor = "enriched_interpretable"
    model = CASE_STUDY_MODEL if CASE_STUDY_MODEL in set(metrics_df["model"].astype(str)) else self._choose_available(metrics_df, "model", ["hgb", "rf", "ridge", "mlp"])
    sub = metrics_df[(metrics_df["descriptor_family"] == descriptor) & (metrics_df["model"] == model)].copy()
    if sub.empty:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    target_order = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(sub["target_slug"])]
    r2 = sub.groupby(["target_slug", "split_family"], observed=False)["r2"].mean().reset_index()
    r2_piv = r2.pivot_table(index="target_slug", columns="split_family", values="r2", observed=False).reindex(index=target_order, columns=split_order)
    delta = r2_piv.sub(r2_piv["random"], axis=0) if "random" in r2_piv.columns else r2_piv * np.nan
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.9))
    for ax, pivot, title, letter, cmap in [
        (axes[0], r2_piv, r"Mean R$^2$", "a", "viridis"),
        (axes[1], delta, r"R$^2$ change relative to random", "b", "RdBu_r"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_v7_split_label(c) for c in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([_v8_target_label(t) for t in pivot.index])
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=8.5, color="#111111")
        _v8_panel_label(ax, letter)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        _v7_apply_axis_style(ax, grid_axis=None)
    fig.suptitle(f"Figure 3. Target-dependent sensitivity to split family ({_v7_descriptor_label(descriptor)}, {str(model).upper()})",
                 fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure3_v8_target_split_sensitivity_polished.png"
    plot_data = {"mean_r2": r2_piv.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="mean_r2"),
                 "delta_r2_vs_random": delta.reset_index().melt(id_vars="target_slug", var_name="split_family", value_name="delta_r2_vs_random")}
    self._save_figure_data_csv(figure_path, plot_data)
    self._savefig(fig, figure_path)


def _v8_figure_4_screening(self: FigureBuilder, metrics_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = metrics_df[(metrics_df["target_slug"] == anchor_slug) &
                     (metrics_df["descriptor_family"].isin(["compact_geometry", "enriched_interpretable"])) &
                     (metrics_df["model"] == CASE_STUDY_MODEL)].copy()
    if sub.empty:
        return
    split_order = _v7_ordered_splits(sub["split_family"].unique())
    summary = _v7_summary_sem(sub, ["split_family", "descriptor_family"], ["top_5pct_overlap", "top_5pct_enrichment"])
    summary["split_family"] = pd.Categorical(summary["split_family"], categories=split_order, ordered=True)
    summary = summary.sort_values(["split_family", "descriptor_family"])
    x = np.arange(len(split_order))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.2))
    for ax, metric, ylabel, title, letter in [
        (axes[0], "top_5pct_overlap", "Top-5% overlap", "Elite recovery", "a"),
        (axes[1], "top_5pct_enrichment", "Top-5% enrichment", "Enrichment over random selection", "b"),
    ]:
        for i, desc in enumerate(["compact_geometry", "enriched_interpretable"]):
            temp = summary[summary["descriptor_family"] == desc].set_index("split_family").reindex(split_order)
            offset = (i - 0.5) * width
            ax.errorbar(x + offset, temp[f"{metric}_mean"].values, yerr=temp[f"{metric}_sem"].values,
                        marker="o" if desc == "compact_geometry" else "s", linewidth=2.0, markersize=6,
                        capsize=3, color=V7_DESCRIPTOR_COLORS.get(desc, "#777777"), label=_v7_descriptor_label(desc))
        for j, split in enumerate(split_order):
            ax.axvspan(j - 0.45, j + 0.45, color=V7_SPLIT_COLORS.get(split, "#CCCCCC"), alpha=0.07, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([_v7_split_label(s) for s in split_order], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _v8_panel_label(ax, letter)
        _v7_apply_axis_style(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Figure 4. Screening consequences of split choice ({_v8_target_label(anchor_slug)}; model = {CASE_STUDY_MODEL.upper()})",
                 fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure4_v8_screening_consequences_polished.png"
    self._save_figure_data_csv(figure_path, summary)
    self._savefig(fig, figure_path)


def _v8_figure_5_group_lollipop(self: FigureBuilder, group_error_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if group_error_df is None or group_error_df.empty:
        self.logger.write("Exact group error table is empty; V8 Figure 5 skipped.")
        return
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = group_error_df[group_error_df["target_slug"] == anchor_slug].copy()
    if sub.empty:
        sub = group_error_df.copy()
    descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "geometry_plus_topology", "compact_geometry"])
    model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", [CASE_STUDY_MODEL, "hgb", "rf", "ridge", "mlp"])
    if descriptor is not None:
        sub = sub[sub["descriptor_family"] == descriptor].copy()
    if model is not None:
        sub = sub[sub["model"] == model].copy()
    if sub.empty:
        return
    sub["group_label_v8"] = sub.apply(lambda r: _v7_group_label(r.get("group_column", ""), r.get("group_id", "")), axis=1)
    geo = sub[(sub["group_column"] == "geo_cluster") & (sub["split_family"] == "geo_grouped")].sort_values("mean_abs_error", ascending=False).head(12)
    mapping = {"metal_cluster": "metal_grouped", "func_cluster": "func_grouped", "linker_cluster": "linker_grouped", TOPOLOGY_COLUMN_GROUPED: "topology_grouped"}
    chem = pd.concat([sub[(sub["group_column"] == g) & (sub["split_family"] == sf)] for g, sf in mapping.items()], ignore_index=True)
    if chem.empty:
        chem = sub[sub["group_column"].isin(mapping.keys())].copy()
    chem = chem.sort_values("mean_abs_error", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.5))
    for ax, temp, title, letter in [
        (axes[0], geo, "Hardest geometry hold-outs", "a"),
        (axes[1], chem, "Hardest chemistry/topology hold-outs", "b"),
    ]:
        if temp.empty:
            ax.axis("off")
            continue
        temp = temp.sort_values("mean_abs_error", ascending=True).reset_index(drop=True)
        y = np.arange(len(temp))
        colors = [V7_GROUP_COLORS.get(gc, "#777777") for gc in temp["group_column"].astype(str)]
        sem = temp["std_abs_error"].fillna(0).values / np.sqrt(temp["n"].replace(0, np.nan).values)
        ax.hlines(y, xmin=0, xmax=temp["mean_abs_error"].values, color="#CCCCCC", linewidth=1.4)
        ax.errorbar(temp["mean_abs_error"].values, y, xerr=sem, fmt="o", color="#333333", ecolor="#666666", capsize=2, markersize=0)
        ax.scatter(temp["mean_abs_error"].values, y, s=54, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(temp["group_label_v8"].astype(str).tolist())
        ax.set_xlabel("Mean absolute error")
        ax.set_title(title)
        _v8_panel_label(ax, letter)
        _v7_apply_axis_style(ax, grid_axis="x")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=V7_GROUP_COLORS[g], label=lab) for g, lab in [
        ("geo_cluster", "Geometry groups"), ("metal_cluster", "Metal groups"), ("func_cluster", "Functional groups"),
        ("linker_cluster", "Linker groups"), (TOPOLOGY_COLUMN_GROUPED, "Topology groups")
    ]]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Figure 5. Concrete held-out groups behind split severity ({_v8_target_label(anchor_slug)}; {_v7_descriptor_label(descriptor)}; {str(model).upper()})",
                 fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure5_v8_hard_group_lollipop_polished.png"
    self._save_figure_data_csv(figure_path, {"hardest_geometry_groups_exact": geo, "hardest_chemistry_topology_groups_exact": chem})
    self._savefig(fig, figure_path)


def _v8_figure_6_shift_and_space(self: FigureBuilder, metrics_df: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> None:
    _v7_configure_matplotlib()
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = metrics_df[metrics_df["target_slug"] == anchor_slug].copy()
    if sub.empty or not {"shift_centroid_distance", "shift_avg_wasserstein", "r2"}.issubset(sub.columns):
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.9), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    for split_family, temp in sub.groupby("split_family", observed=False):
        axes[0].scatter(temp["shift_centroid_distance"], temp["r2"], s=45, alpha=0.72,
                        color=V7_SPLIT_COLORS.get(split_family, "#888888"), label=_v7_split_label(split_family),
                        edgecolor="white", linewidth=0.35)
    valid = sub[["shift_centroid_distance", "r2"]].dropna()
    corr = np.nan
    if len(valid) >= 3 and valid["shift_centroid_distance"].nunique() > 1:
        corr = stats.spearmanr(valid["shift_centroid_distance"], valid["r2"]).correlation
        slope, intercept, *_ = stats.linregress(valid["shift_centroid_distance"], valid["r2"])
        xline = np.linspace(valid["shift_centroid_distance"].min(), valid["shift_centroid_distance"].max(), 120)
        axes[0].plot(xline, intercept + slope * xline, color="#333333", linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Centroid shift distance")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title(f"Shift vs accuracy (ρ = {corr:.2f})")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    _v8_panel_label(axes[0], "a")
    _v7_apply_axis_style(axes[0])

    split_order = _v7_ordered_splits(sub["split_family"].unique())
    shift_bar = sub.groupby("split_family", observed=False)["shift_avg_wasserstein"].mean().reindex(split_order)
    y = np.arange(len(shift_bar))
    axes[1].barh(y, shift_bar.values, color=[V7_SPLIT_COLORS.get(s, "#999999") for s in shift_bar.index], alpha=0.9)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([_v7_split_label(s) for s in shift_bar.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean standardized Wasserstein shift")
    axes[1].set_title("Shift severity by split family")
    _v8_panel_label(axes[1], "b")
    _v7_apply_axis_style(axes[1], grid_axis="x")

    needed = [c for c in ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"] if master_df is not None and c in master_df.columns]
    if master_df is not None and len(needed) >= 2 and "geo_cluster" in master_df.columns:
        pca_sub = master_df[needed + ["geo_cluster"]].dropna(subset=needed).copy()
        if len(pca_sub) > MAX_ROWS_FOR_VISUAL_PCA:
            pca_sub = pca_sub.sample(MAX_ROWS_FOR_VISUAL_PCA, random_state=RANDOM_SEED)
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(pca_sub[needed].values))
        counts = pca_sub["geo_cluster"].astype(str).value_counts()
        highlight_groups = set(counts.head(5).index.tolist())
        highlight = pca_sub["geo_cluster"].astype(str).isin(highlight_groups).values
        axes[2].scatter(coords[~highlight, 0], coords[~highlight, 1], s=5, alpha=0.16, color="#9E9E9E", linewidths=0)
        axes[2].scatter(coords[highlight, 0], coords[highlight, 1], s=13, alpha=0.75, color=V7_SPLIT_COLORS["geo_grouped"], edgecolor="white", linewidth=0.15)
        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        axes[2].set_title("Descriptor-space example\n(top geometry groups highlighted)")
        pca_data = pca_sub.copy()
        pca_data["pca_pc1"] = coords[:, 0]
        pca_data["pca_pc2"] = coords[:, 1]
        pca_data["highlighted_geometry_group"] = highlight
    else:
        axes[2].axis("off")
        pca_data = pd.DataFrame()
    _v8_panel_label(axes[2], "c")
    _v7_apply_axis_style(axes[2])
    fig.suptitle("Figure 6. Distribution shift explains why split families are not interchangeable", fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure6_v8_shift_and_descriptor_space_polished.png"
    self._save_figure_data_csv(figure_path, {"shift_points": sub, "shift_bar": shift_bar.reset_index().rename(columns={"shift_avg_wasserstein": "mean_shift_avg_wasserstein"}), "pca_example": pca_data})
    self._savefig(fig, figure_path)


def _v8_figure_7_elite_stability(self: FigureBuilder, elite_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if elite_df is None or elite_df.empty:
        self.logger.write("Elite-list stability table is empty; V8 Figure 7 skipped.")
        return
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = elite_df[(elite_df["target_slug"] == anchor_slug) & (elite_df["top_fraction"] == 0.05)].copy()
    if sub.empty:
        sub = elite_df[elite_df["top_fraction"] == 0.05].copy()
    descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "compact_geometry", "geometry_plus_topology"])
    model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", [CASE_STUDY_MODEL, "hgb", "rf", "ridge", "mlp"])
    if descriptor is not None:
        sub = sub[sub["descriptor_family"] == descriptor].copy()
    if model is not None:
        sub = sub[sub["model"] == model].copy()
    if sub.empty:
        return
    sub["pair"] = sub["split_family_1"].astype(str) + " vs " + sub["split_family_2"].astype(str)
    top = sub.sort_values("elite_jaccard", ascending=True).head(12).copy()
    colors = [V7_SPLIT_COLORS["topology_grouped"] if "topology_grouped" in p else "#7F8C8D" for p in top["pair"]]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2))
    y = np.arange(len(top))
    axes[0].barh(y, top["elite_jaccard"].values, color=colors, alpha=0.92)
    axes[0].axvline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_v7_pair_label(p) for p in top["pair"]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Top-5% Jaccard overlap")
    axes[0].set_title("Least stable elite shortlists")
    _v8_panel_label(axes[0], "a")
    _v7_apply_axis_style(axes[0], grid_axis="x")
    axes[1].barh(y, top["prediction_spearman"].values, color=colors, alpha=0.92)
    axes[1].axvline(0.9, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([_v7_pair_label(p) for p in top["pair"]])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Overall prediction-rank Spearman")
    axes[1].set_title("Global rank agreement can remain high")
    _v8_panel_label(axes[1], "b")
    _v7_apply_axis_style(axes[1], grid_axis="x")
    fig.suptitle(f"Figure 7. Elite-list instability despite broad rank agreement ({_v8_target_label(anchor_slug)}; {_v7_descriptor_label(descriptor)}; {str(model).upper()})",
                 fontsize=14.3, fontweight="bold")
    figure_path = self.fig_main_dir / "figure7_v8_exact_elite_instability_polished.png"
    self._save_figure_data_csv(figure_path, {"elite_stability_exact_top5": sub, "least_stable_pairs": top})
    self._savefig(fig, figure_path)


def _v8_si_target_distributions(self: FigureBuilder, preds_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    target_slugs = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(preds_df["target_slug"].dropna())]
    if not target_slugs:
        return
    ncols = 2
    nrows = int(math.ceil(len(target_slugs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.3 * nrows))
    axes = np.array(axes).reshape(-1)
    for idx, (ax, target_slug) in enumerate(zip(axes, target_slugs)):
        sub = preds_df[preds_df["target_slug"] == target_slug]
        ax.hist(sub["target"].dropna().values, bins=40, color="#6C7A89", alpha=0.85)
        ax.set_title(_v8_target_label(target_slug))
        ax.set_xlabel(r"Target uptake (mmol g$^{-1}$)")
        ax.set_ylabel("Count")
        _v8_panel_label(ax, chr(ord("a") + idx))
        _v7_apply_axis_style(ax)
    for ax in axes[len(target_slugs):]:
        ax.axis("off")
    fig.suptitle("SI Figure. Target-distribution diagnostics", fontsize=14, fontweight="bold")
    figure_path = self.fig_si_dir / "si_v8_target_distributions_clear_labels.png"
    self._save_figure_data_csv(figure_path, _v8_slug_columns_to_labels(preds_df[["target_slug", "target_name", "target"]].copy()))
    self._savefig(fig, figure_path)


def _v8_si_pld_decile_errors(self: FigureBuilder, preds_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    sub = preds_df.dropna(subset=["Df", "absolute_error"]).copy()
    if sub.empty:
        return
    sub["pld_decile"] = pd.qcut(sub["Df"], q=10, duplicates="drop")
    dec = sub.groupby(["target_slug", "pld_decile"], observed=False)["absolute_error"].mean().reset_index()
    target_slugs = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(dec["target_slug"].dropna())]
    if not target_slugs:
        return
    ncols = 2
    nrows = int(math.ceil(len(target_slugs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for idx, (ax, target_slug) in enumerate(zip(axes, target_slugs)):
        temp = dec[dec["target_slug"] == target_slug].copy()
        ax.plot(range(len(temp)), temp["absolute_error"].values, marker="o", linewidth=1.8, color="#4E79A7")
        ax.set_title(_v8_target_label(target_slug))
        ax.set_xlabel("PLD decile")
        ax.set_ylabel("Mean absolute error")
        _v8_panel_label(ax, chr(ord("a") + idx))
        _v7_apply_axis_style(ax)
    for ax in axes[len(target_slugs):]:
        ax.axis("off")
    fig.suptitle("SI Figure. PLD-decile-resolved error", fontsize=14, fontweight="bold")
    figure_path = self.fig_si_dir / "si_v8_pld_decile_errors_clear_labels.png"
    dec_export = dec.copy()
    dec_export["target_display"] = dec_export["target_slug"].map(lambda x: _v8_target_label(x, mode="plain"))
    self._save_figure_data_csv(figure_path, dec_export)
    self._savefig(fig, figure_path)
    dec_export.to_csv(self.fig_si_dir / "si_v8_pld_decile_errors_table.csv", index=False)


def _v8_si_prediction_scatter_panels(self: FigureBuilder, preds_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    summary = preds_df.groupby(["target_slug", "descriptor_family", "split_family", "model"], observed=False).agg(
        mae_mean=("absolute_error", "mean"), n=("prediction", "size")
    ).reset_index()
    if summary.empty:
        return
    best_keys = summary.sort_values("mae_mean", ascending=True).groupby("target_slug", observed=False).head(1)[["target_slug", "descriptor_family", "split_family", "model"]]
    merged = preds_df.merge(best_keys, on=["target_slug", "descriptor_family", "split_family", "model"], how="inner")
    target_slugs = [TARGET_SLUGS[t] for t in TARGET_COLUMNS if TARGET_SLUGS[t] in set(merged["target_slug"].dropna())]
    if not target_slugs:
        return
    ncols = 2
    nrows = int(math.ceil(len(target_slugs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.9 * nrows))
    axes = np.array(axes).reshape(-1)
    for idx, (ax, target_slug) in enumerate(zip(axes, target_slugs)):
        temp = merged[merged["target_slug"] == target_slug]
        ax.scatter(temp["target"], temp["prediction"], s=9, alpha=0.42, color="#4E79A7", linewidths=0)
        min_val = min(temp["target"].min(), temp["prediction"].min())
        max_val = max(temp["target"].max(), temp["prediction"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#333333", linewidth=1.0)
        ax.set_title(_v8_target_label(target_slug))
        ax.set_xlabel(r"True uptake (mmol g$^{-1}$)")
        ax.set_ylabel(r"Predicted uptake (mmol g$^{-1}$)")
        _v8_panel_label(ax, chr(ord("a") + idx))
        _v7_apply_axis_style(ax)
    for ax in axes[len(target_slugs):]:
        ax.axis("off")
    fig.suptitle("SI Figure. Representative prediction scatter panels", fontsize=14, fontweight="bold")
    figure_path = self.fig_si_dir / "si_v8_prediction_scatter_panels_clear_labels.png"
    self._save_figure_data_csv(figure_path, _v8_slug_columns_to_labels(merged))
    self._savefig(fig, figure_path)


_V7_EXPORT_KEY_TABLES_AFTER_PATCH = ManuscriptExporter.export_key_tables


def _v8_export_key_tables(self: ManuscriptExporter, tables: Dict[str, pd.DataFrame]) -> None:
    _V7_EXPORT_KEY_TABLES_AFTER_PATCH(self, tables)
    main_keys = [
        "dataset_summary_and_coverage", "split_family_severity_summary", "split_severity_decomposition",
        "main_paper_compact", "elite_list_stability", "hardest_heldout_groups", "elite_dropout_candidates",
    ]
    si_keys = sorted([k for k, v in tables.items() if isinstance(v, pd.DataFrame) and not v.empty])
    main_manifest = pd.DataFrame({
        "table_key": [k for k in main_keys if k in tables and not tables[k].empty],
        "csv_file": [f"table_{k}.csv" if not k.startswith("table_") else f"{k}.csv" for k in main_keys if k in tables and not tables[k].empty],
        "recommended_location": ["main text"] * len([k for k in main_keys if k in tables and not tables[k].empty]),
    })
    si_manifest = pd.DataFrame({
        "table_key": si_keys,
        "csv_file": [f"table_{k}.csv" if not k.startswith("table_") else f"{k}.csv" for k in si_keys],
        "recommended_location": ["supplementary information"] * len(si_keys),
    })
    for name, df, caption, label in [
        ("main_text_table_manifest", main_manifest, "Main-text table manifest generated by the v8 script.", "tab:main_text_table_manifest"),
        ("si_table_manifest", si_manifest, "Supplementary table manifest generated by the v8 script.", "tab:si_table_manifest"),
    ]:
        if df.empty:
            continue
        df.to_csv(self.output_dir / "global_results" / "tables" / f"table_{name}.csv", index=False)
        self.dataframe_to_latex(df, self.latex_dir / f"latex_{name}.tex", caption=caption, label=label, index=False)
    self.logger.write("V8 table manifests exported as CSV and LaTeX.")


def _v8_build_all_figures(self: FigureBuilder, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], master_df: Optional[pd.DataFrame] = None) -> None:
    if metrics_df.empty:
        self.logger.write("No metrics available; skipping figure generation.")
        return
    _v7_configure_matplotlib()
    _v8_figure_1_workflow(self)
    _v8_figure_2_split_severity(self, metrics_df)
    _v8_figure_3_target_sensitivity(self, metrics_df)
    _v8_figure_4_screening(self, metrics_df)
    _v8_figure_5_group_lollipop(self, tables.get("group_resolved_errors", pd.DataFrame()))
    _v8_figure_6_shift_and_space(self, metrics_df, master_df)
    _v8_figure_7_elite_stability(self, tables.get("elite_list_stability", pd.DataFrame()))

    if not preds_df.empty:
        _v8_si_target_distributions(self, preds_df)
        _v8_si_pld_decile_errors(self, preds_df)
        _v8_si_prediction_scatter_panels(self, preds_df)
    else:
        self.logger.write("Sampled prediction table empty; point-level SI figures skipped.")
    self.si_figure_variance_decomposition(tables.get("variance_decomposition", pd.DataFrame()))
    self.si_figure_splitwise_metric_heatmaps(metrics_df)
    _v7_si_group_size(self, tables.get("split_group_size_summary", pd.DataFrame()))
    _v7_si_error_vs_rank(self, metrics_df)


ManuscriptExporter.export_key_tables = _v8_export_key_tables
FigureBuilder.figure_1_study_design_overview = _v8_figure_1_workflow
FigureBuilder.build_all_figures = _v8_build_all_figures





V9_PATCH_NOTE = "v9_publication_tables_from_existing_and_figure5_legend_fix"


FORCE_REBUILD_TABLES = False
if "FORCE_REBUILD_PUBLICATION_TABLES" not in globals():
    FORCE_REBUILD_PUBLICATION_TABLES = True

PUBLICATION_MAIN_TABLE_KEYS = [
    "dataset_summary_and_coverage",
    "split_family_severity_summary",
    "split_severity_decomposition",
    "main_paper_compact",
    "elite_list_stability",
    "hardest_heldout_groups",
]
PUBLICATION_SI_TABLE_KEYS = [
    "main_benchmark_summary",
    "optimism_gap",
    "paired_significance_summary",
    "distribution_shift_summary",
    "ranking_inversion_correlations",
    "rank_inversion_significance",
    "group_resolved_errors",
    "split_group_size_summary",
    "target_overview",
    "variance_decomposition",
    "additive_factor_effects",
    "elite_dropout_candidates",
    "case_study_candidates",
]


def _v9_table_key_to_csv_name(key: str) -> str:
    if key.startswith("table_"):
        return f"{key}.csv"
    return f"table_{key}.csv"


def _v9_find_existing_table_csv(tables_dir: Path, key: str) -> Optional[Path]:
    candidates = [
        tables_dir / _v9_table_key_to_csv_name(key),
        tables_dir / f"{key}.csv",
        tables_dir / f"table_exact_{key}.csv",
    ]
    # Some table keys are aliases for exact post-hoc files.
    alias = {
        "elite_list_stability": "table_exact_elite_list_stability.csv",
        "group_resolved_errors": "table_exact_group_resolved_errors.csv",
    }.get(key)
    if alias:
        candidates.insert(0, tables_dir / alias)
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _v9_latex_safe_caption(key: str, location: str) -> str:
    title = key.replace("_", " ").capitalize()
    return f"Publication-ready {location} table export: {title}."


def _v9_load_or_copy_publication_table(tables: Dict[str, pd.DataFrame], tables_dir: Path, key: str) -> Optional[pd.DataFrame]:
    if key in tables and isinstance(tables[key], pd.DataFrame) and not tables[key].empty:
        return tables[key].copy()
    csv_path = _v9_find_existing_table_csv(tables_dir, key)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        return df if not df.empty else None
    except Exception:
        return None


_V8_EXPORT_KEY_TABLES_FOR_V9 = ManuscriptExporter.export_key_tables


def _v9_export_publication_tables(self: ManuscriptExporter, tables: Dict[str, pd.DataFrame]) -> None:
    # Preserve all previous LaTeX exports first.
    _V8_EXPORT_KEY_TABLES_FOR_V9(self, tables)

    if not FORCE_REBUILD_PUBLICATION_TABLES:
        self.logger.write("Publication-table rebuild skipped because FORCE_REBUILD_PUBLICATION_TABLES=False.")
        return

    tables_dir = self.output_dir / "global_results" / "tables"
    pub_dir = self.output_dir / "global_results" / "publication_tables"
    pub_main_dir = pub_dir / "main_text"
    pub_si_dir = pub_dir / "si"
    for d in [pub_dir, pub_main_dir, pub_si_dir]:
        ensure_dir(d)

    manifest_rows = []
    for location, keys, out_dir in [
        ("main text", PUBLICATION_MAIN_TABLE_KEYS, pub_main_dir),
        ("supplementary information", PUBLICATION_SI_TABLE_KEYS, pub_si_dir),
    ]:
        for key in keys:
            df = _v9_load_or_copy_publication_table(tables, tables_dir, key)
            if df is None or df.empty:
                continue
            df = _v8_slug_columns_to_labels(df)
            out_csv = out_dir / _v9_table_key_to_csv_name(key)
            out_tex = out_dir / f"latex_{key}.tex"
            df.to_csv(out_csv, index=False)
            self.dataframe_to_latex(
                df,
                out_tex,
                caption=_v9_latex_safe_caption(key, location),
                label=f"tab:{key}",
                index=False,
            )
            manifest_rows.append({
                "table_key": key,
                "recommended_location": location,
                "csv_file": str(out_csv.relative_to(self.output_dir / "global_results")),
                "latex_file": str(out_tex.relative_to(self.output_dir / "global_results")),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
            })

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        manifest_csv = pub_dir / "publication_table_manifest.csv"
        manifest_tex = pub_dir / "latex_publication_table_manifest.tex"
        manifest.to_csv(manifest_csv, index=False)
        self.dataframe_to_latex(
            manifest,
            manifest_tex,
            caption="Manifest of publication-ready CSV and LaTeX tables regenerated from existing analysis tables.",
            label="tab:publication_table_manifest",
            index=False,
        )
    self.logger.write(f"V9 publication table exports regenerated from existing tables: {len(manifest_rows)} tables.")


ManuscriptExporter.export_key_tables = _v9_export_publication_tables


def _v9_figure_5_group_lollipop(self: FigureBuilder, group_error_df: pd.DataFrame) -> None:
    _v7_configure_matplotlib()
    if group_error_df is None or group_error_df.empty:
        self.logger.write("Exact group error table is empty; V9 Figure 5 skipped.")
        return
    anchor_slug = TARGET_SLUGS[ANCHOR_TARGET]
    sub = group_error_df[group_error_df["target_slug"] == anchor_slug].copy()
    if sub.empty:
        sub = group_error_df.copy()
    descriptor = self._choose_available(sub, "descriptor_family", ["enriched_interpretable", "geometry_plus_topology", "compact_geometry"])
    model = self._choose_available(sub[sub["descriptor_family"] == descriptor] if descriptor else sub, "model", [CASE_STUDY_MODEL, "hgb", "rf", "ridge", "mlp"])
    if descriptor is not None:
        sub = sub[sub["descriptor_family"] == descriptor].copy()
    if model is not None:
        sub = sub[sub["model"] == model].copy()
    if sub.empty:
        return

    sub["group_label_v9"] = sub.apply(lambda r: _v7_group_label(r.get("group_column", ""), r.get("group_id", "")), axis=1)
    geo = sub[(sub["group_column"] == "geo_cluster") & (sub["split_family"] == "geo_grouped")].sort_values("mean_abs_error", ascending=False).head(12)
    mapping = {
        "metal_cluster": "metal_grouped",
        "func_cluster": "func_grouped",
        "linker_cluster": "linker_grouped",
        TOPOLOGY_COLUMN_GROUPED: "topology_grouped",
    }
    chem_parts = [sub[(sub["group_column"] == g) & (sub["split_family"] == sf)] for g, sf in mapping.items()]
    chem = pd.concat([p for p in chem_parts if not p.empty], ignore_index=True) if chem_parts else pd.DataFrame()
    if chem.empty:
        chem = sub[sub["group_column"].isin(mapping.keys())].copy()
    chem = chem.sort_values("mean_abs_error", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.8))
    for ax, temp, title, letter in [
        (axes[0], geo, "Hardest geometry hold-outs", "a"),
        (axes[1], chem, "Hardest chemistry/topology hold-outs", "b"),
    ]:
        if temp.empty:
            ax.axis("off")
            ax.set_title(title + " (no eligible groups)")
            continue
        temp = temp.sort_values("mean_abs_error", ascending=True).reset_index(drop=True)
        y = np.arange(len(temp))
        colors = [V7_GROUP_COLORS.get(gc, "#777777") for gc in temp["group_column"].astype(str)]
        sem = temp["std_abs_error"].fillna(0).values / np.sqrt(temp["n"].replace(0, np.nan).values)
        ax.hlines(y, xmin=0, xmax=temp["mean_abs_error"].values, color="#CCCCCC", linewidth=1.45)
        ax.errorbar(temp["mean_abs_error"].values, y, xerr=sem, fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2)
        ax.scatter(temp["mean_abs_error"].values, y, s=58, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(temp["group_label_v9"].astype(str).tolist())
        ax.set_xlabel("Mean absolute error", labelpad=8)
        ax.set_title(title, pad=10)
        _v8_panel_label(ax, letter, x=-0.10, y=1.08)
        _v7_apply_axis_style(ax, grid_axis="x")

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=V7_GROUP_COLORS[g], label=lab) for g, lab in [
        ("geo_cluster", "Geometry groups"),
        ("metal_cluster", "Metal groups"),
        ("func_cluster", "Functional groups"),
        ("linker_cluster", "Linker groups"),
        (TOPOLOGY_COLUMN_GROUPED, "Topology groups"),
    ]]

    # Legend moved above the plotting area; no overlap with the x-axis labels.
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.905), borderaxespad=0.0)
    fig.suptitle(
        f"Figure 5. Concrete held-out groups behind split severity ({_v8_target_label(anchor_slug)}; {_v7_descriptor_label(descriptor)}; {str(model).upper()})",
        fontsize=14.3,
        fontweight="bold",
        y=0.985,
    )
    # Manual margins are used instead of tight_layout to keep the legend clear of both panels and x-axis labels.
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.125, top=0.805, wspace=0.16)
    figure_path = self.fig_main_dir / "figure5_v9_hard_group_lollipop_polished.png"
    self._save_figure_data_csv(figure_path, {"hardest_geometry_groups_exact": geo, "hardest_chemistry_topology_groups_exact": chem})
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, bbox_inches="tight", dpi=320)
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _v9_build_all_figures(self: FigureBuilder, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, tables: Dict[str, pd.DataFrame], master_df: Optional[pd.DataFrame] = None) -> None:
    if metrics_df.empty:
        self.logger.write("No metrics available; skipping figure generation.")
        return
    _v7_configure_matplotlib()
    _v8_figure_1_workflow(self)
    _v8_figure_2_split_severity(self, metrics_df)
    _v8_figure_3_target_sensitivity(self, metrics_df)
    _v8_figure_4_screening(self, metrics_df)
    _v9_figure_5_group_lollipop(self, tables.get("group_resolved_errors", pd.DataFrame()))
    _v8_figure_6_shift_and_space(self, metrics_df, master_df)
    _v8_figure_7_elite_stability(self, tables.get("elite_list_stability", pd.DataFrame()))

    if not preds_df.empty:
        _v8_si_target_distributions(self, preds_df)
        _v8_si_pld_decile_errors(self, preds_df)
        _v8_si_prediction_scatter_panels(self, preds_df)
    else:
        self.logger.write("Sampled prediction table empty; point-level SI figures skipped.")
    self.si_figure_variance_decomposition(tables.get("variance_decomposition", pd.DataFrame()))
    self.si_figure_splitwise_metric_heatmaps(metrics_df)
    _v7_si_group_size(self, tables.get("split_group_size_summary", pd.DataFrame()))
    _v7_si_error_vs_rank(self, metrics_df)


FigureBuilder.build_all_figures = _v9_build_all_figures






V10_PATCH_NOTE = "v10_robust_manual_latex_export_no_pandas_styler"


def _v10_latex_escape(value) -> str:
    """Convert a scalar to a compact, LaTeX-safe string for table cells."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        text = f"{float(value):.3g}"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > 180:
        text = text[:177] + "..."
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _v10_flatten_columns(columns) -> List[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            label = " | ".join(str(c) for c in col if str(c) != "")
        else:
            label = str(col)
        flat.append(label)
    # Ensure unique column labels after flattening.
    seen = {}
    unique = []
    for label in flat:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label}_{count+1}")
    return unique


def _v10_dataframe_to_latex(self: ManuscriptExporter, df: pd.DataFrame, path: Path, caption: str, label: str, index: bool = False, float_format: str = "%.3f") -> None:
    """Robust LaTeX export that avoids pandas Styler.to_latex entirely.

    The full tables remain available as CSV.  For LaTeX, very large tables are
    capped to a preview-sized number of rows to keep TeX compilation realistic;
    the caption records when this happens.
    """
    ensure_dir(path.parent)
    if df is None or df.empty:
        path.write_text("% Empty table; no LaTeX tabular emitted.\n", encoding="utf-8")
        return

    work = df.copy()
    if index:
        work = work.reset_index()
    else:
        work = work.reset_index(drop=True)
    work.columns = _v10_flatten_columns(work.columns)

    # TeX is not an appropriate container for thousands of rows.  Keep CSVs full;
    # write compact LaTeX previews for inclusion/review.
    max_rows = 80
    was_truncated = len(work) > max_rows
    if was_truncated:
        work = work.head(max_rows).copy()

    n_cols = len(work.columns)
    col_spec = "l" * max(n_cols, 1)
    safe_caption = _v10_latex_escape(caption)
    if was_truncated:
        safe_caption += f" (First {max_rows} rows shown; full table is available in the matching CSV export.)"
    safe_label = _v10_latex_escape(label).replace(r"\_", "_")  # labels can safely keep underscores

    header = " & ".join(_v10_latex_escape(c) for c in work.columns) + r" \\"
    body_lines = []
    for _, row in work.iterrows():
        body_lines.append(" & ".join(_v10_latex_escape(v) for v in row.tolist()) + r" \\")

    # resizebox keeps wide CSV-derived tables compilable in Overleaf/manuscripts.
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(f"\\caption{{{safe_caption}}}")
    latex.append(f"\\label{{{safe_label}}}")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")
    latex.append(header)
    latex.append(r"\midrule")
    latex.extend(body_lines)
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\end{table}")
    path.write_text("\n".join(latex) + "\n", encoding="utf-8")


ManuscriptExporter.dataframe_to_latex = _v10_dataframe_to_latex



import re

V11_PATCH_NOTE = "v11_paperA_raw_arcmof_candidate_audit_and_plausibility_overlay"

# Prefer a reproducible project layout, but keep the user's current root-folder
# workflow valid through the search-path helper below.
DATA_DIR = BASE_DIR / "data" / "raw" / "arcmof"

V11_ARCMOF_REQUIRED_FILES = [
    "geometric_properties.csv",
    "post_comb_vsa-CO2.csv",
    "methane.csv",
    "geo-clusters.csv",
    "mc-clusters.csv",
    "func-clusters.csv",
    "flig-clusters.csv",
    "all_topology_lists.csv",
]

V11_ARCMOF_RECOMMENDED_FILES = [
    "ARC-MOF_Dim.csv",
    "overall_process.csv",
    "ARCMOF_20241004.tar.gz",
]

V11_CORE_RECOMMENDED_FILES = [
    "ASR_data_SI_20250204.csv",
    "FSR_data_SI_20250204.csv",
    "ION_data_SI_20250204.csv",
    "12089-recommended-screening-list.csv",
    "ASR_FSR_check.csv",
    "CoREMOF2024DB_SI_20250204.zip",
    "mofid-v2.zip",
    "water.zip",
]


def _v11_norm_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _v11_key_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".cif", "", regex=False).str.strip().map(_v11_norm_text)


def _v11_search_roots(data_dir: Path) -> List[Path]:
    roots = [
        data_dir,
        BASE_DIR / "data" / "raw" / "arcmof",
        BASE_DIR / "data" / "raw" / "arcmof" / "descriptors",
        BASE_DIR / "data" / "raw" / "arcmof" / "adsorption",
        BASE_DIR / "data" / "raw" / "arcmof" / "clusters",
        BASE_DIR / "data" / "raw" / "arcmof" / "topology",
        BASE_DIR / "data" / "raw" / "arcmof" / "process",
        BASE_DIR / "data" / "raw" / "core_mof_2024",
        BASE_DIR / "data" / "raw" / "mosaec_db",
        BASE_DIR / "data" / "overlays",
        BASE_DIR,
    ]
    out = []
    seen = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if str(rp) not in seen:
            out.append(r)
            seen.add(str(rp))
    return out


def _v11_find_file(filename: str, data_dir: Optional[Path] = None) -> Optional[Path]:
    data_dir = data_dir or DATA_DIR
    for root in _v11_search_roots(data_dir):
        path = root / filename
        if path.exists():
            return path
    # One shallow recursive fallback for users who made slightly different folders.
    for root in [BASE_DIR / "data", BASE_DIR]:
        if not root.exists():
            continue
        try:
            matches = list(root.glob(f"**/{filename}"))
        except Exception:
            matches = []
        if matches:
            # Prefer the data/raw tree over old output directories.
            matches = sorted(matches, key=lambda p: ("paper1_split_strategy_outputs" in str(p), len(str(p))))
            return matches[0]
    return None


def _v11_read_csv(filename: str, data_dir: Optional[Path] = None, **kwargs) -> Optional[pd.DataFrame]:
    path = _v11_find_file(filename, data_dir=data_dir)
    if path is None:
        return None
    return pd.read_csv(path, low_memory=False, **kwargs)


def _v11_identifier_column(df: pd.DataFrame) -> str:
    candidates = [
        "filename", "Filename", "Name", "name", "MOF", "mof", "ARC-MOF", "ARCMOF",
        "mof_name", "MOF_name", "structure", "Structure", "refcode", "Refcode", "REFCODE",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return str(df.columns[0])


def _v11_find_numeric_target_column(df: pd.DataFrame, exact_name: str, gas: str, pressure_bar: str, source_hint: str) -> Optional[str]:
    if exact_name in df.columns:
        return exact_name
    norm_exact = _v11_norm_text(exact_name)
    for c in df.columns:
        if _v11_norm_text(c) == norm_exact:
            return c

    pressure_aliases = {
        "0.015": ["0015", "00150", "15mbar", "0015bar", "p0015", "0p015"],
        "0.15": ["015", "0150", "150mbar", "015bar", "p015", "0p15"],
        "5.8": ["58", "580", "58bar", "5p8"],
        "65": ["65", "650", "65bar"],
    }.get(str(pressure_bar), [_v11_norm_text(pressure_bar)])
    gas_aliases = {
        "CO2": ["co2", "carbondioxide"],
        "CH4": ["ch4", "methane"],
    }.get(gas, [_v11_norm_text(gas)])

    source_hint_norm = _v11_norm_text(source_hint)

    file_is_gas_specific = any(g in source_hint_norm for g in gas_aliases)

    candidates = []
    for c in df.columns:
        nc = _v11_norm_text(c)
        if c == _v11_identifier_column(df):
            continue
        if any(bad in nc for bad in ["std", "stdev", "stderr", "error", "err", "sigma"]):
            continue
        gas_ok = file_is_gas_specific or any(g in nc for g in gas_aliases)
        pressure_ok = any(p in nc for p in pressure_aliases)
        uptake_ok = any(tok in nc for tok in ["uptake", "loading", "mmolg", "ads", "excess", "absolute"])
        if gas_ok and pressure_ok:
            numeric_fraction = pd.to_numeric(df[c], errors="coerce").notna().mean()
            score = 10 * numeric_fraction + (3 if uptake_ok else 0) + len(nc) / 10000
            candidates.append((score, c))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def _v11_try_make_long_format_target(df: pd.DataFrame, exact_name: str, gas: str, pressure_bar: str, source_hint: str) -> Optional[pd.DataFrame]:
    """Convert a long adsorption table to filename_norm + exact target if needed."""
    id_col = _v11_identifier_column(df)
    norm_cols = {_v11_norm_text(c): c for c in df.columns}
    pressure_cols = [c for c in df.columns if any(tok in _v11_norm_text(c) for tok in ["pressure", "press", "pbar", "bar"])]
    uptake_cols = [c for c in df.columns if any(tok in _v11_norm_text(c) for tok in ["uptake", "loading", "mmolg", "adsorption"])]
    if not pressure_cols or not uptake_cols:
        return None
    pcol = pressure_cols[0]
    # Prefer gas-specific uptake if available, else first numeric uptake column.
    gas_aliases = {"CO2": ["co2", "carbondioxide"], "CH4": ["ch4", "methane"]}.get(gas, [_v11_norm_text(gas)])
    ucol = None
    for c in uptake_cols:
        nc = _v11_norm_text(c)
        if any(g in nc for g in gas_aliases):
            ucol = c
            break
    if ucol is None:
        ucol = uptake_cols[0]
    temp = df[[id_col, pcol, ucol]].copy()
    temp[pcol] = pd.to_numeric(temp[pcol], errors="coerce")
    temp[ucol] = pd.to_numeric(temp[ucol], errors="coerce")
    target_pressure = float(pressure_bar)
    temp = temp[np.isclose(temp[pcol], target_pressure, rtol=1e-4, atol=max(1e-5, target_pressure * 1e-4))].copy()
    if temp.empty:
        return None
    temp["filename_norm"] = normalize_filename_series(temp[id_col])
    out = temp.groupby("filename_norm", as_index=False)[ucol].median().rename(columns={ucol: exact_name})
    return out


def _v11_target_table_from_raw(raw_df: pd.DataFrame, exact_name: str, gas: str, pressure_bar: str, source_hint: str) -> Optional[pd.DataFrame]:
    id_col = _v11_identifier_column(raw_df)
    col = _v11_find_numeric_target_column(raw_df, exact_name, gas, pressure_bar, source_hint)
    if col is not None:
        out = raw_df[[id_col, col]].copy()
        out["filename_norm"] = normalize_filename_series(out[id_col])
        out[exact_name] = pd.to_numeric(out[col], errors="coerce")
        out = out[["filename_norm", exact_name]].dropna(subset=[exact_name])
        out = out.groupby("filename_norm", as_index=False)[exact_name].median()
        return out
    return _v11_try_make_long_format_target(raw_df, exact_name, gas, pressure_bar, source_hint)


def _v11_standardize_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    id_col = _v11_identifier_column(out)
    if "filename" not in out.columns:
        out["filename"] = out[id_col].astype(str)
    out["filename_norm"] = normalize_filename_series(out["filename"])


    synonym_map = {
        "Density": ["density", "crystaldensity", "rho"],
        "ASA": ["asa", "accessiblesurfacearea"],
        "AVA": ["ava", "accessiblevolume"],
        "AVAf": ["avaf", "voidfraction", "accessiblevolumefraction"],
        "POAVA": ["poava", "probeoccupiableaccessiblevolume"],
        "Di": ["di", "lcd", "largestcavitydiameter"],
        "Df": ["df", "pld", "porelimitingdiameter"],
        "Dif": ["dif", "lcddifference", "lcdpld", "cavitywindowgap"],
        "UC_volume": ["ucvolume", "unitcellvolume"],
    }
    norm_to_col = {_v11_norm_text(c): c for c in out.columns}
    for canonical, aliases in synonym_map.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in norm_to_col:
                out[canonical] = out[norm_to_col[alias]]
                break
    for c in ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _v11_build_clean_data_from_raw_arcmof(data_dir: Path, logger: Optional[DualLogger] = None) -> Optional[pd.DataFrame]:
    clean_path = _v11_find_file("clean_data.csv", data_dir=data_dir)
    if clean_path is not None:
        return pd.read_csv(clean_path, low_memory=False)

    geom_path = _v11_find_file("geometric_properties.csv", data_dir=data_dir)
    co2_path = _v11_find_file("post_comb_vsa-CO2.csv", data_dir=data_dir)
    methane_path = _v11_find_file("methane.csv", data_dir=data_dir)
    if geom_path is None or co2_path is None or methane_path is None:
        return None

    if logger:
        logger.write("clean_data.csv was not found; building it from raw ARC-MOF geometric_properties.csv, post_comb_vsa-CO2.csv, and methane.csv.")
    geom = pd.read_csv(geom_path, low_memory=False)
    df = _v11_standardize_geometry_columns(geom)

    targets = [
        (co2_path, "uptake(mmol/g) CO2 at 0.015 bar", "CO2", "0.015"),
        (co2_path, "uptake(mmol/g) CO2 at 0.15 bar", "CO2", "0.15"),
        (methane_path, "uptake(mmol/g) methane at 5.8 bar", "CH4", "5.8"),
        (methane_path, "uptake(mmol/g) methane at 65 bar", "CH4", "65"),
    ]
    raw_cache: Dict[str, pd.DataFrame] = {}
    for path, exact_name, gas, pressure in targets:
        key = str(path)
        if key not in raw_cache:
            raw_cache[key] = pd.read_csv(path, low_memory=False)
        target_table = _v11_target_table_from_raw(raw_cache[key], exact_name, gas, pressure, path.name)
        if target_table is None or target_table.empty:
            if logger:
                logger.write(f"WARNING: Could not infer target column for {exact_name} from {path.name}; filling with NaN.")
            df[exact_name] = np.nan
        else:
            df = df.merge(target_table, on="filename_norm", how="left")
            if logger:
                logger.write(f"Added target {exact_name}: {int(df[exact_name].notna().sum())} matched rows.")

    # Keep a manuscript-facing clean table where future reruns can start directly.
    out_dir = BASE_DIR / "data" / "raw" / "arcmof"
    ensure_dir(out_dir)
    out_path = out_dir / "clean_data.csv"
    df.to_csv(out_path, index=False)
    if logger:
        logger.write(f"Generated clean_data.csv at {out_path}")
    return df


_V10_TRY_LOAD_CSV = DataAssembler._try_load_csv


def _v11_try_load_csv(self: DataAssembler, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    if filename == "clean_data.csv":
        built = _v11_build_clean_data_from_raw_arcmof(self.data_dir, logger=self.logger)
        if built is not None:
            return built
    path = _v11_find_file(filename, data_dir=self.data_dir)
    if path is None:
        if required:
            raise FileNotFoundError(
                f"Required input file not found: {filename}. Searched project root and data/raw/arcmof subfolders."
            )
        self.logger.write(f"Optional file not found and will be skipped: {filename}")
        return None
    self.logger.write(f"Reading {filename} from {path}")
    return pd.read_csv(path, low_memory=False)


DataAssembler._try_load_csv = _v11_try_load_csv


def _v11_load_overlay_table(filename: str) -> Optional[pd.DataFrame]:
    path = _v11_find_file(filename, data_dir=BASE_DIR / "data")
    if path is None:
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        id_col = _v11_identifier_column(df)
        df["overlay_key"] = _v11_key_series(df[id_col])
        return df
    except Exception:
        return None


def _v11_compact_overlay(df: pd.DataFrame, prefix: str, max_extra_cols: int = 12) -> pd.DataFrame:
    if df is None or df.empty or "overlay_key" not in df.columns:
        return pd.DataFrame()
    preferred = []
    for c in df.columns:
        nc = _v11_norm_text(c)
        if any(tok in nc for tok in ["name", "refcode", "mofid", "topology", "oms", "stability", "water", "hydro", "decomposition", "checker", "valid", "cr", "asr", "fsr", "ion", "source", "dataset", "solvent"]):
            preferred.append(c)
    preferred = [c for c in preferred if c != "overlay_key"][:max_extra_cols]
    keep = ["overlay_key"] + preferred
    out = df[keep].drop_duplicates("overlay_key").copy()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "overlay_key"})
    out[f"{prefix}_matched"] = True
    return out


def _v11_core_overlay() -> pd.DataFrame:
    parts = []
    for filename, label in [
        ("ASR_data_SI_20250204.csv", "CoRE_ASR"),
        ("FSR_data_SI_20250204.csv", "CoRE_FSR"),
        ("ION_data_SI_20250204.csv", "CoRE_ION"),
        ("12089-recommended-screening-list.csv", "CoRE_recommended"),
    ]:
        df = _v11_load_overlay_table(filename)
        if df is not None and not df.empty:
            df["core_overlay_source"] = label
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True, sort=False)
    compact = _v11_compact_overlay(raw, "core", max_extra_cols=16)
    if compact.empty:
        return compact
    sources = raw.groupby("overlay_key", as_index=False)["core_overlay_source"].agg(lambda x: ";".join(sorted(set(map(str, x)))))
    compact = compact.merge(sources, on="overlay_key", how="left")
    return compact


def _v11_annotate_with_plausibility(df: pd.DataFrame, tables_dir: Path, logger: Optional[DualLogger] = None, output_name: str = "table_candidates_with_plausibility.csv") -> pd.DataFrame:
    if df is None or df.empty or "filename_norm" not in df.columns:
        return df
    out = df.copy()
    out["overlay_key"] = _v11_key_series(out["filename_norm"])
    overlays = []
    core = _v11_core_overlay()
    if not core.empty:
        overlays.append(core)
    for filename, prefix in [
        ("mosaec_plausibility_overlay.csv", "mosaec"),
        ("mofchecker_flags.csv", "mofchecker"),
        ("manual_candidate_notes.csv", "manual"),
    ]:
        overlay = _v11_load_overlay_table(filename)
        compact = _v11_compact_overlay(overlay, prefix, max_extra_cols=16) if overlay is not None else pd.DataFrame()
        if not compact.empty:
            overlays.append(compact)
    for overlay in overlays:
        out = out.merge(overlay, on="overlay_key", how="left")
    out["any_external_plausibility_match"] = False
    for c in out.columns:
        if c.endswith("_matched"):
            out["any_external_plausibility_match"] = out["any_external_plausibility_match"] | out[c].fillna(False).astype(bool)
    out = out.drop(columns=["overlay_key"], errors="ignore")
    path = tables_dir / output_name
    out.to_csv(path, index=False)
    if logger:
        logger.write(f"Saved plausibility-annotated candidate table: {path.name} ({len(out)} rows)")
    return out


def _v11_build_robust_surviving_candidates(self: PredictionPostProcessor, reg: pd.DataFrame, master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if reg is None or reg.empty:
        return pd.DataFrame()
    target_slug = CASE_STUDY_TARGET_SLUG
    split_order = ["random", "geo_grouped", "metal_grouped", "func_grouped", "linker_grouped", "topology_grouped"]
    rows = []
    for descriptor in ["enriched_interpretable", "compact_geometry", "geometry_plus_topology"]:
        for model in ["rf", "hgb"]:
            by_split = {}
            for split_family in split_order:
                mp = self._mean_predictions_for_combo(reg, target_slug, descriptor, model, split_family)
                if mp is None or mp.empty:
                    continue
                mp = mp.copy()
                n = len(mp)
                mp[f"prediction_{split_family}"] = mp["mean_prediction"]
                mp[f"target_{split_family}"] = mp["mean_target"]
                mp[f"rank_{split_family}"] = mp["mean_prediction"].rank(ascending=False, method="min")
                mp[f"rank_percentile_{split_family}"] = mp[f"rank_{split_family}"] / max(n, 1)
                mp[f"top5_{split_family}"] = mp[f"rank_percentile_{split_family}"] <= 0.05
                mp[f"top10_{split_family}"] = mp[f"rank_percentile_{split_family}"] <= 0.10
                keep = ["filename_norm", f"prediction_{split_family}", f"target_{split_family}", f"rank_{split_family}", f"rank_percentile_{split_family}", f"top5_{split_family}", f"top10_{split_family}"]
                by_split[split_family] = mp[keep]
            if "random" not in by_split or len(by_split) < 3:
                continue
            merged = None
            for split_family in split_order:
                if split_family not in by_split:
                    continue
                merged = by_split[split_family] if merged is None else merged.merge(by_split[split_family], on="filename_norm", how="inner")
            if merged is None or merged.empty:
                continue
            top5_cols = [c for c in merged.columns if c.startswith("top5_")]
            top10_cols = [c for c in merged.columns if c.startswith("top10_")]
            pred_cols = [c for c in merged.columns if c.startswith("prediction_")]
            rankpct_cols = [c for c in merged.columns if c.startswith("rank_percentile_")]
            merged["n_split_families_present"] = len(by_split)
            merged["n_top5_splits"] = merged[top5_cols].sum(axis=1)
            merged["n_top10_splits"] = merged[top10_cols].sum(axis=1)
            merged["mean_prediction_across_splits"] = merged[pred_cols].mean(axis=1)
            merged["min_prediction_across_splits"] = merged[pred_cols].min(axis=1)
            merged["worst_rank_percentile_across_splits"] = merged[rankpct_cols].max(axis=1)
            merged["descriptor_family"] = descriptor
            merged["model"] = model
            merged["target_slug"] = target_slug
            # A conservative survivor: high in random and still high under several grouped audits.
            survivor = merged[(merged.get("top10_random", False)) & (merged["n_top10_splits"] >= max(3, int(np.ceil(0.5 * len(by_split)))))]
            if survivor.empty:
                survivor = merged.sort_values(["n_top10_splits", "worst_rank_percentile_across_splits", "mean_prediction_across_splits"], ascending=[False, True, False]).head(50)
            rows.append(survivor)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if master_df is not None and not master_df.empty:
        meta_cols = [
            "filename_norm", "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif",
            "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", TOPOLOGY_COLUMN_GROUPED,
            "Crystalnet", "likely topology", "ARC-MOF", "DB_num", "order_geo", "bool_geo",
        ]
        meta_cols = [c for c in meta_cols if c in master_df.columns]
        out = out.merge(master_df[meta_cols].drop_duplicates("filename_norm"), on="filename_norm", how="left")
    out = out.sort_values(["n_top10_splits", "n_top5_splits", "worst_rank_percentile_across_splits", "mean_prediction_across_splits"], ascending=[False, False, True, False])
    out = out.drop_duplicates(["filename_norm", "descriptor_family", "model"]).head(250).reset_index(drop=True)
    out["survivor_rank"] = np.arange(1, len(out) + 1)
    out["case_study_role"] = "split-surviving conservative candidate"
    out["scope_note"] = "High-ranked under random evaluation and retained under multiple grouped split audits; intended for Manuscript A robust-candidate section."
    out.to_csv(self.tables_dir / "table_robust_surviving_candidates.csv", index=False)
    annotated = _v11_annotate_with_plausibility(out, self.tables_dir, logger=self.logger, output_name="table_robust_surviving_candidates_with_plausibility.csv")
    return annotated


def _v11_build_case_study_selection_sheet(tables: Dict[str, pd.DataFrame], tables_dir: Path, logger: Optional[DualLogger] = None) -> pd.DataFrame:
    parts = []
    mapping = [
        ("robust_surviving_candidates", "robust survivor"),
        ("elite_dropout_candidates", "elite dropout / entrant"),
        ("case_study_candidates", "large grouped-error increase"),
        ("hardest_heldout_groups", "hard held-out group"),
    ]
    for key, role in mapping:
        df = tables.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        temp = df.copy()
        temp["recommended_case_study_role"] = role
        parts.append(temp.head(80))
    if not parts:
        return pd.DataFrame()
    # Keep all columns because the tables have different structures; CSV is the source of truth.
    out = pd.concat(parts, ignore_index=True, sort=False)
    out.to_csv(tables_dir / "table_case_study_selection_sheet.csv", index=False)
    annotated = _v11_annotate_with_plausibility(out, tables_dir, logger=logger, output_name="table_case_study_selection_sheet_with_plausibility.csv")
    if logger:
        logger.write(f"Saved case-study selection sheet: {len(annotated)} rows")
    return annotated


_V10_OR_PREVIOUS_POSTHOC = PredictionPostProcessor.build_exact_posthoc_tables


def _v11_build_exact_posthoc_tables(self: PredictionPostProcessor, metrics_df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    tables = _V10_OR_PREVIOUS_POSTHOC(self, metrics_df, master_df=master_df)
    reg = self._load_registry()
    robust = _v11_build_robust_surviving_candidates(self, reg, master_df)
    if robust is not None and not robust.empty:
        tables["robust_surviving_candidates"] = robust
    selection = _v11_build_case_study_selection_sheet(tables, self.tables_dir, logger=self.logger)
    if selection is not None and not selection.empty:
        tables["case_study_selection_sheet"] = selection
    return tables


PredictionPostProcessor.build_exact_posthoc_tables = _v11_build_exact_posthoc_tables


_V10_OR_PREVIOUS_EXPORT = ManuscriptExporter.export_key_tables


def _v11_export_key_tables(self: ManuscriptExporter, tables: Dict[str, pd.DataFrame]) -> None:
    _V10_OR_PREVIOUS_EXPORT(self, tables)
    extra_mapping = {
        "robust_surviving_candidates": ("latex_robust_surviving_candidates.tex", "Split-surviving candidates retained under multiple grouped split audits.", "tab:robust_surviving_candidates"),
        "elite_dropout_candidates": ("latex_elite_dropout_candidates.tex", "Candidate MOFs whose elite status changes between random and grouped evaluation.", "tab:elite_dropout_candidates"),
        "case_study_selection_sheet": ("latex_case_study_selection_sheet.tex", "Candidate and group examples selected for Manuscript A structural inspection.", "tab:case_study_selection_sheet"),
    }
    for key, (filename, caption, label) in extra_mapping.items():
        if key in tables and tables[key] is not None and not tables[key].empty:
            self.dataframe_to_latex(tables[key], self.latex_dir / filename, caption=caption, label=label, index=False)
    self.logger.write("V11 Paper-A candidate-audit LaTeX exports saved.")


ManuscriptExporter.export_key_tables = _v11_export_key_tables


def _v11_write_input_manifest(output_dir: Path, logger: Optional[DualLogger] = None) -> None:
    manifest_rows = []
    for category, files in [
        ("ARC-MOF required/recommended", V11_ARCMOF_REQUIRED_FILES + V11_ARCMOF_RECOMMENDED_FILES),
        ("CoRE MOF 2024 optional plausibility", V11_CORE_RECOMMENDED_FILES),
        ("Local optional overlays", ["mosaec_plausibility_overlay.csv", "mofchecker_flags.csv", "manual_candidate_notes.csv"]),
    ]:
        for filename in files:
            path = _v11_find_file(filename, data_dir=BASE_DIR / "data")
            manifest_rows.append({
                "category": category,
                "filename": filename,
                "found": bool(path is not None),
                "resolved_path": str(path) if path is not None else "",
            })
    out = pd.DataFrame(manifest_rows)
    ensure_dir(output_dir / "global_results" / "tables")
    out.to_csv(output_dir / "global_results" / "tables" / "table_input_file_manifest.csv", index=False)
    if logger:
        logger.write("V11 input file manifest saved as table_input_file_manifest.csv")


# Add the manifest to the end of the normal reporter flow by wrapping main.
_V10_OR_PREVIOUS_MAIN = main


def main() -> None:
    _V10_OR_PREVIOUS_MAIN()
    try:
        logger = DualLogger(OUTPUT_DIR / "logs" / "run_log.txt")
        _v11_write_input_manifest(OUTPUT_DIR, logger=logger)
    except Exception:
        pass


if __name__ == "__main__":
    main()
