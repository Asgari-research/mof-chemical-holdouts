"""Bounded saved-data analyses for geometry protocol, pressure groups and support."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_MOF_SOURCE_ROOT",str(OUT.parents[1]))).resolve()
RESULTS = OUT / "results"
V6 = ROOT / "V6_AdvancedScience"
MASTER = ROOT / "results/global_processed/assembled_master_table.csv"
TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
GEO = ["Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"]
TARGETS = {
    "co2_0015bar": "uptake(mmol/g) CO2 at 0.015 bar",
    "co2_015bar": TARGET,
    "ch4_58bar": "uptake(mmol/g) methane at 5.8 bar",
    "ch4_65bar": "uptake(mmol/g) methane at 65 bar",
}


def read_master(columns):
    df = pd.read_csv(MASTER, usecols=list(dict.fromkeys(columns)), low_memory=False)
    assert df.filename_norm.is_unique
    return df


def task_a():
    sizes = pd.read_csv(ROOT / "revision/group_audit/group_size_distributions.csv",low_memory=False)
    audit = pd.read_csv(ROOT / "revision/group_audit/group_construction_summary.csv")
    metrics = pd.read_csv(ROOT / "results/metrics/all_fold_metrics_long.csv")
    metrics = metrics[(metrics.target_slug == "co2_015bar") & (metrics.descriptor_family == "enriched_interpretable") & (metrics.model == "rf")]
    rows = []
    for _, a in audit.iterrows():
        sub = sizes[sizes.group_family == a.group_family]
        rows.append({"group_family": a.group_family, "source_column": a.source_column,
                     "total_rows": int(a.total_rows), "missing_rows": int(a.missing_rows),
                     "raw_groups": int(a.raw_groups), "singleton_groups_excluded": int(a.singleton_groups_removed_original),
                     "singleton_rows_excluded": int(a.singleton_rows_removed_original),
                     "eligible_rows": int(a.eligible_rows_original), "eligible_groups": int(a.eligible_groups_original),
                     "eligible_size_median": float(a.eligible_group_size_median),
                     "eligible_size_max": int(a.eligible_group_size_max),
                     "fraction_eligible_groups_size_2_or_3": float((sub.loc[sub.originally_eligible, "size"] <= 3).mean()) if len(sub) else np.nan,
                     "fraction_eligible_rows_in_size_2_or_3": float(sub.loc[sub.originally_eligible & (sub["size"] <= 3), "size"].sum()/a.eligible_rows_original) if len(sub) else np.nan})
    pd.DataFrame(rows).to_csv(RESULTS / "group_protocol_sizes.csv", index=False)
    metrics.groupby("split_family", as_index=False).agg(folds=("fold_id", "nunique"), r2_mean=("r2", "mean"), r2_min=("r2", "min"), r2_max=("r2", "max"), mae_mean=("mae", "mean"), top5_fold_recovery_mean=("top_5pct_recall", "mean"), n_test_sum=("n_test", "sum")).to_csv(RESULTS / "anchor_split_fold_summary.csv", index=False)


def task_e(master):
    all_metrics=pd.read_csv(ROOT / "results/metrics/all_fold_metrics_long.csv")
    pressure_folds=all_metrics[(all_metrics.target_slug.isin(TARGETS)) & (all_metrics.descriptor_family=="enriched_interpretable") & (all_metrics.split_family=="topology_grouped") & (all_metrics.model=="rf")].copy()
    assert len(pressure_folds)==20
    pressure_folds[["target_slug","fold_id","n_train","n_test","r2","mae","top_5pct_recall"]].to_csv(RESULTS / "topology_pressure_fold_metrics.csv",index=False)
    exact = pd.read_csv(ROOT / "results/tables/table_exact_group_resolved_errors.csv")
    exact = exact[(exact.target_slug == "co2_015bar") & (exact.descriptor_family == "enriched_interpretable") & (exact.split_family == "topology_grouped") & (exact.model == "rf") & (exact.group_column == "topology_label")].copy()
    assert len(exact) == 47 and exact.n.sum() == 263744
    exact[["target_slug", "group_id", "n", "mean_abs_error", "mean_signed_residual", "source_note"]].to_csv(RESULTS / "topology_exact_co2_015bar.csv", index=False)
    sample = pd.read_csv(ROOT / "results/tables/table_group_resolved_errors.csv")
    sample = sample[(sample.descriptor_family == "enriched_interpretable") & (sample.split_family == "topology_grouped") & (sample.model == "rf") & (sample.group_column == "topology_label")].copy()
    assert set(sample.target_slug) == set(TARGETS)
    denoms = []
    for slug, col in TARGETS.items():
        y = master[col].dropna()
        assert len(y) == 263744
        denoms.append({"target_slug": slug, "target_n": len(y), "q25": y.quantile(.25), "q75": y.quantile(.75), "iqr_mmol_g": y.quantile(.75)-y.quantile(.25)})
    denoms = pd.DataFrame(denoms)
    denoms.to_csv(RESULTS / "target_iqr_denominators.csv", index=False)
    sample = sample.merge(denoms[["target_slug", "iqr_mmol_g"]], on="target_slug", validate="many_to_one")
    sample["mae_over_target_iqr"] = sample.mean_abs_error / sample.iqr_mmol_g
    sample["truth_minus_prediction_over_target_iqr"] = sample.mean_signed_residual / sample.iqr_mmol_g
    sample = sample[["target_slug", "topology_label", "n", "mean_abs_error", "mean_signed_residual", "iqr_mmol_g", "mae_over_target_iqr", "truth_minus_prediction_over_target_iqr", "source_note"]]
    sample.to_csv(RESULTS / "topology_pressure_sampled_families.csv", index=False)
    eligible = sample[sample.n >= 50]
    ranks = eligible.groupby("target_slug", group_keys=False).apply(lambda x: x.assign(within_target_rank=x.mae_over_target_iqr.rank(ascending=False, method="min")), include_groups=False).reset_index(drop=True)
    # pandas groupby apply may omit the grouping key; merge by original keys.
    if "target_slug" not in ranks.columns:
        ranks = eligible.copy()
        ranks["within_target_rank"] = ranks.groupby("target_slug").mae_over_target_iqr.rank(ascending=False, method="min")
    persistence = ranks.groupby("topology_label", as_index=False).agg(targets=("target_slug", "nunique"), min_sample_n=("n", "min"), mean_normalized_mae=("mae_over_target_iqr", "mean"), worst_rank=("within_target_rank", "max"), best_rank=("within_target_rank", "min"))
    persistence.sort_values(["targets", "mean_normalized_mae"], ascending=[False, False]).to_csv(RESULTS / "topology_pressure_persistence_sampled.csv", index=False)


def task_d(master):
    start = time.monotonic()
    pair = pd.read_csv(V6 / "postprocessing_closure/results/same_mof_residual_comparison.csv", usecols=["filename_norm", "topology_absolute_error", "paired_error_difference", "topology_label"])
    assert len(pair) == 177411 and pair.filename_norm.is_unique
    topo = pd.read_csv(ROOT / "historical_oos/co2_015bar__enriched_interpretable__topology_grouped__rf__aggregated_oos.csv.gz", usecols=["filename_norm", "folds_or_repeats_seen"])
    assert topo.filename_norm.is_unique and len(topo) == 263744
    counts = topo.folds_or_repeats_seen.astype(str).value_counts()
    assert set(counts.index) == set(map(str, range(5)))
    pair = pair.merge(topo.rename(columns={"folds_or_repeats_seen":"topology_fold"}), on="filename_norm", validate="one_to_one")
    labelcheck=pair[["filename_norm","topology_label"]].merge(master[["filename_norm","topology_label"]],on="filename_norm",validate="one_to_one",suffixes=("_pair","_master"))
    assert len(labelcheck)==len(pair) and (labelcheck.topology_label_pair==labelcheck.topology_label_master).all()
    base = master[["filename_norm", *GEO]].copy()
    assert base.filename_norm.is_unique and len(base) == 263744
    joined = pair.merge(base, on="filename_norm", validate="one_to_one")
    chunks=[]
    for fold in range(5):
        test_ids = set(topo.loc[topo.folds_or_repeats_seen.astype(str) == str(fold), "filename_norm"])
        train = base.loc[~base.filename_norm.isin(test_ids)]
        query = joined.loc[joined.topology_fold.astype(str) == str(fold)]
        assert len(train) + len(test_ids) == len(base)
        assert not set(train.filename_norm).intersection(query.filename_norm)
        assert not set(master.loc[~master.filename_norm.isin(test_ids),"topology_label"]).intersection(set(master.loc[master.filename_norm.isin(test_ids),"topology_label"]))
        imputer = SimpleImputer(strategy="median")
        scale = StandardScaler()
        tr = scale.fit_transform(imputer.fit_transform(train[GEO]))
        te = scale.transform(imputer.transform(query[GEO]))
        tree = cKDTree(tr)
        distance,_ = tree.query(te, k=1, workers=1)
        chunks.append(pd.DataFrame({"filename_norm":query.filename_norm.to_numpy(), "topology_fold":fold, "nearest_training_geometry_z_distance":distance, "topology_absolute_error":query.topology_absolute_error.to_numpy(), "topology_minus_random_absolute_error":query.paired_error_difference.to_numpy(), "topology_label":query.topology_label.to_numpy()}))
        print(f"support fold {fold}: {len(query)} queries, {time.monotonic()-start:.1f}s", flush=True)
    out = pd.concat(chunks, ignore_index=True)
    assert len(out) == len(pair) and out.filename_norm.is_unique
    out.to_csv(RESULTS / "paired_geometry_support.csv", index=False)
    out["support_quartile"] = pd.qcut(out.nearest_training_geometry_z_distance.rank(method="first"), 4, labels=[1,2,3,4])
    summary = out.groupby("support_quartile", observed=True).agg(n=("filename_norm", "size"), distance_median=("nearest_training_geometry_z_distance", "median"), topology_mae=("topology_absolute_error", "mean"), paired_error_mean=("topology_minus_random_absolute_error", "mean"), paired_error_median=("topology_minus_random_absolute_error", "median"), fraction_topology_error_gt_1_mmol_g=("topology_absolute_error",lambda x:float((x>1).mean()))).reset_index()
    summary.to_csv(RESULTS / "geometry_support_quartiles.csv", index=False)
    family = out.groupby("topology_label").agg(n=("filename_norm", "size"), distance_median=("nearest_training_geometry_z_distance", "median"), topology_mae=("topology_absolute_error", "mean"), paired_error_mean=("topology_minus_random_absolute_error", "mean")).reset_index()
    family.loc[family.n >= 1000].sort_values("n", ascending=False).to_csv(RESULTS / "geometry_support_major_families.csv", index=False)
    within=[]
    for label,g in out.groupby("topology_label"):
        if len(g) >= 1000:
            within.append({"topology_label":label,"n":len(g),
                "rho_support_vs_topo_error":float(spearmanr(g.nearest_training_geometry_z_distance,g.topology_absolute_error).statistic),
                "rho_support_vs_paired_error":float(spearmanr(g.nearest_training_geometry_z_distance,g.topology_minus_random_absolute_error).statistic)})
    pd.DataFrame(within).sort_values("n",ascending=False).to_csv(RESULTS / "geometry_support_within_family.csv",index=False)
    rho1 = spearmanr(out.nearest_training_geometry_z_distance, out.topology_absolute_error)
    rho2 = spearmanr(out.nearest_training_geometry_z_distance, out.topology_minus_random_absolute_error)
    (RESULTS / "geometry_support_association.json").write_text(json.dumps({"n":len(out), "features":GEO, "scaling":"train-only median imputation and standard scaling per topology fold", "spearman_distance_vs_topology_absolute_error":float(rho1.statistic), "spearman_distance_vs_topology_minus_random_error":float(rho2.statistic), "elapsed_seconds":time.monotonic()-start}, indent=2), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    task_a()
    master=read_master(["filename_norm", "topology_label", *GEO, *TARGETS.values()])
    task_e(master)
    task_d(master)

if __name__ == "__main__":
    main()
