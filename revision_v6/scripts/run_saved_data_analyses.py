"""Run V6 analyses that require no model fitting.

All inputs are read from the parent project. Outputs are compact CSV/JSON files in
this V6 workspace. The historical pipeline and V4 outputs are never modified.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


V6 = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOF_V6_SOURCE_ROOT", V6.parent)).resolve()
OUT = V6 / "results"
PROV = V6 / "provenance"
FIG = V6 / "figures"


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_top_ids(df: pd.DataFrame, score: str, fraction: float = 0.05) -> set[str]:
    n = max(1, math.ceil(len(df) * fraction))
    ordered = df.sort_values([score, "filename_norm"], ascending=[False, True], kind="mergesort")
    return set(ordered.head(n)["filename_norm"])


def run_preflight() -> dict[str, object]:
    ids = pd.read_csv(ROOT / "revision/common_universe/common_universe_ids.csv")
    common_ckpts = list((ROOT / "revision/common_universe/checkpoints").glob("*.json"))
    gen_ckpts = list((ROOT / "revision/generalization/checkpoints").glob("*.json"))
    master = pd.read_csv(
        ROOT / "results/global_processed/assembled_master_table.csv",
        usecols=["filename_norm", "Dif", "cavity_window_gap"],
    )
    gap_equal = bool(np.allclose(master["Dif"], master["cavity_window_gap"], equal_nan=True))
    pipeline_text = (ROOT / "old code/split_strategy_pipeline_raw_arc_V2.py").read_text(
        encoding="utf-8", errors="replace"
    )
    checks = {
        "common_universe_rows": int(len(ids)),
        "common_universe_unique_ids": int(ids["filename_norm"].nunique()),
        "common_universe_checkpoint_count": len(common_ckpts),
        "generalization_checkpoint_count": len(gen_ckpts),
        "cavity_window_gap_equals_Dif_all_rows": gap_equal,
        "historical_pipeline_contains_executed_assignment": 'df["cavity_window_gap"] = df["Dif"]' in pipeline_text,
    }
    checks["status"] = "PASS" if (
        checks["common_universe_rows"] == 214531
        and checks["common_universe_unique_ids"] == 214531
        and checks["common_universe_checkpoint_count"] == 30
        and checks["generalization_checkpoint_count"] == 60
        and gap_equal
        and checks["historical_pipeline_contains_executed_assignment"]
    ) else "FAIL"
    write_json(OUT / "preflight_invariants.json", checks)
    return checks


def run_repeat_availability() -> None:
    manifest = pd.read_csv(ROOT / "historical_oos/original_prediction_files_manifest.csv")
    accessible = manifest["original_path"].map(lambda p: Path(p).exists())
    random_path = ROOT / "historical_oos/co2_015bar__enriched_interpretable__random__rf__aggregated_oos.csv.gz"
    random = pd.read_csv(random_path)
    mult = (
        random.groupby("n_oos_predictions", as_index=False)
        .agg(structures=("filename_norm", "size"))
        .sort_values("n_oos_predictions")
    )
    mult["fraction"] = mult["structures"] / len(random)
    mult.to_csv(OUT / "random_oos_multiplicity.csv", index=False)
    finding = {
        "status": "BLOCKED_MISSING_REPEAT_LEVEL_PREDICTIONS",
        "manifest_rows": int(len(manifest)),
        "accessible_original_files": int(accessible.sum()),
        "aggregate_rows": int(len(random)),
        "aggregate_has_repeat_count": True,
        "aggregate_has_individual_repeat_predictions": False,
        "reason": "The compact random archive stores one mean prediction per structure. Expanding it would fabricate repeat-level independence.",
    }
    write_json(OUT / "repeat_level_prediction_availability.json", finding)


def run_pairwise_shortlists() -> None:
    base = ROOT / "historical_oos"
    rp = base / "co2_015bar__enriched_interpretable__random__rf__aggregated_oos.csv.gz"
    tp = base / "co2_015bar__enriched_interpretable__topology_grouped__rf__aggregated_oos.csv.gz"
    random = pd.read_csv(rp).rename(columns={"y_true": "y_true_random", "y_pred_mean": "random_prediction"})
    topology = pd.read_csv(tp).rename(columns={"y_true": "y_true_topology", "y_pred_mean": "topology_prediction"})
    common = random.merge(topology, on="filename_norm", how="inner", validate="one_to_one", suffixes=("_random", "_topology"))
    if not np.allclose(common["y_true_random"], common["y_true_topology"], equal_nan=False):
        raise ValueError("True targets disagree between random and topology archives")
    common["y_true"] = common["y_true_random"]
    random_top = deterministic_top_ids(common, "random_prediction")
    topology_top = deterministic_top_ids(common, "topology_prediction")
    reference_top = deterministic_top_ids(common, "y_true")
    common["shortlist_group"] = np.select(
        [
            common["filename_norm"].isin(random_top & topology_top),
            common["filename_norm"].isin(random_top - topology_top),
            common["filename_norm"].isin(topology_top - random_top),
        ],
        ["common", "random_only", "topology_only"],
        default="neither",
    )
    membership = common[["filename_norm", "y_true", "random_prediction", "topology_prediction", "shortlist_group"]]
    membership.to_csv(OUT / "pairwise_shortlist_membership.csv", index=False)

    def recovery(selected: set[str]) -> float:
        return len(selected & reference_top) / len(reference_top)

    summary = pd.DataFrame([
        {
            "common_population_n": len(common),
            "top_fraction": 0.05,
            "top_k": len(reference_top),
            "random_topology_jaccard": len(random_top & topology_top) / len(random_top | topology_top),
            "random_reference_recovery": recovery(random_top),
            "topology_reference_recovery": recovery(topology_top),
            "common_shortlist_n": len(random_top & topology_top),
            "random_only_n": len(random_top - topology_top),
            "topology_only_n": len(topology_top - random_top),
            "tie_policy": "ceil(0.05*N); descending score then ascending filename_norm",
        }
    ])
    summary.to_csv(OUT / "pairwise_top5_summary.csv", index=False)


def run_group_and_fold_summaries() -> None:
    exact = pd.read_csv(ROOT / "results/tables/table_exact_group_resolved_errors.csv")
    subset = exact[
        (exact["target_slug"] == "co2_015bar")
        & (exact["descriptor_family"] == "enriched_interpretable")
        & (exact["split_family"] == "topology_grouped")
        & (exact["model"] == "rf")
        & (exact["group_column"] == "topology_label")
    ].copy()
    subset["absolute_error_contribution"] = subset["sum_abs_error"] / subset["sum_abs_error"].sum()
    subset = subset.sort_values("absolute_error_contribution", ascending=False)
    subset["cumulative_absolute_error_contribution"] = subset["absolute_error_contribution"].cumsum()
    subset.to_csv(OUT / "topology_group_error_concentration.csv", index=False)

    other = subset[subset["group_id"].astype(str) == "other"]
    if len(other) != 1:
        raise ValueError(f"Expected one topology 'other' row, found {len(other)}")
    all_n = int(subset["n"].sum())
    without_n = all_n - int(other.iloc[0]["n"])
    all_abs = float(subset["sum_abs_error"].sum())
    without_abs = all_abs - float(other.iloc[0]["sum_abs_error"])
    ablation = pd.DataFrame([
        {"population": "original_topology_oos", "n": all_n, "pooled_mae": all_abs / all_n},
        {"population": "exclude_rare_topology_other", "n": without_n, "pooled_mae": without_abs / without_n},
    ])
    ablation["population_change_vs_original"] = ablation["n"] - all_n
    ablation["interpretation"] = "evaluation-only population ablation; predictions and training fits unchanged"
    ablation.to_csv(OUT / "rare_topology_other_evaluation_ablation.csv", index=False)

    folds = pd.read_csv(ROOT / "results/metrics/all_fold_metrics_long.csv")
    topo_folds = folds[
        (folds["target_slug"] == "co2_015bar")
        & (folds["descriptor_family"] == "enriched_interpretable")
        & (folds["split_family"] == "topology_grouped")
        & (folds["model"] == "rf")
    ].copy()
    topo_folds.to_csv(OUT / "topology_fold_heterogeneity.csv", index=False)

    target_order = ["co2_0015bar", "co2_015bar", "ch4_58bar", "ch4_65bar"]
    pressure = folds[
        folds["target_slug"].isin(target_order)
        & (folds["descriptor_family"] == "enriched_interpretable")
        & (folds["model"] == "rf")
    ]
    pressure_summary = pressure.groupby(["target_slug", "split_family"], as_index=False).agg(
        folds=("fold_id", "nunique"),
        n_test_sum=("n_test", "sum"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        spearman_mean=("spearman_rho", "mean"),
        top5_fold_local_recovery_mean=("top_5pct_recall", "mean"),
    )
    pressure_summary["granularity"] = "fold-level metrics; not paired per-MOF residuals"
    pressure_summary.to_csv(OUT / "pressure_target_fold_summary.csv", index=False)


def make_figures() -> None:
    summary = pd.read_csv(OUT / "pairwise_top5_summary.csv").iloc[0]
    labels = ["Shared", "Random only", "Topology only"]
    values = [summary["common_shortlist_n"], summary["random_only_n"], summary["topology_only_n"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Structures in top-5% list")
    ax.set_title("Random vs topology shortlist membership\ncommon OOS population")
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(values):
        ax.text(i, value, f"{int(value):,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "pairwise_top5_membership.png", dpi=200)
    fig.savefig(FIG / "pairwise_top5_membership.pdf")
    plt.close(fig)

    folds = pd.read_csv(OUT / "topology_fold_heterogeneity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].bar(folds["fold_id"].astype(str), folds["r2"], color="#4C78A8")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Topology fold", ylabel="$R^2$", title="Held-out fit")
    axes[1].bar(folds["fold_id"].astype(str), folds["top_5pct_recall"], color="#E45756")
    axes[1].set(xlabel="Topology fold", ylabel="Top-5% recovery", title="Fold-local screening")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CO$_2$ 0.15 bar enriched-RF topology heterogeneity")
    fig.tight_layout()
    fig.savefig(FIG / "topology_fold_heterogeneity.png", dpi=200)
    fig.savefig(FIG / "topology_fold_heterogeneity.pdf")
    plt.close(fig)


def run_asa_evidence() -> None:
    cols = ["filename_norm", "UC_volume", "Density", "ASA", "vASA", "gASA"]
    master = pd.read_csv(ROOT / "results/global_processed/assembled_master_table.csv", usecols=cols)
    sample = master.sort_values("filename_norm").head(20).copy()
    sample["ASA_interpretation"] = "angstrom^2 per unit cell"
    sample["gASA_interpretation"] = "m^2 per g"
    sample.to_csv(OUT / "asa_unit_evidence.csv", index=False)
    write_json(OUT / "asa_unit_finding.json", {
        "status": "UNIT_LABEL_CORRECTION_REQUIRED",
        "historical_feature_used": "ASA",
        "ASA_unit": "angstrom^2 per unit cell",
        "mass_specific_surface_area_column": "gASA",
        "gASA_unit": "m^2/g",
        "conversion_performed": False,
        "reason": "Both columns are present in the source data and contain different values; relabeling is required rather than blind conversion.",
    })


def record_environment() -> None:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(PROV / "environment_current.json", {
        "provenance_scope": "current V6 saved-data analysis environment; not claimed as historical benchmark environment",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    })


def write_manifest() -> None:
    rows = []
    for directory in [OUT, FIG, PROV]:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.csv":
                rows.append({"path": path.relative_to(V6).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(PROV / "SHA256SUMS.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight()
    if preflight["status"] != "PASS":
        raise RuntimeError(f"Preflight failed: {preflight}")
    run_repeat_availability()
    run_pairwise_shortlists()
    run_group_and_fold_summaries()
    run_asa_evidence()
    make_figures()
    record_environment()
    write_manifest()
    print("Saved-data V6 analyses completed; model fits performed: 0")


if __name__ == "__main__":
    main()
