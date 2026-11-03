"""Recalculate reviewer evidence from recovered ORIGINAL ARC-MOF prediction CSVs.

No fitting or simulation. Outputs are compact summaries; source archives and
historical tables are read-only. Set ARC_MOF_SOURCE_ROOT to the full local
ChemSci_revision source; set ARCMOF_RECOVERED_ROOT to the extracted archives.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

OUT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARC_MOF_SOURCE_ROOT", OUT.parents[1]))
REC = Path(os.environ.get("ARCMOF_RECOVERED_ROOT", ROOT.parent / "transfer" / "ARCMOF_recovered_working"))
CORE = REC / "ARCMOF_Core"
PRESSURE = REC / "ARCMOF_Pressure"
CORE_RUN = CORE / "paper1_split_strategy_outputs_Version_2"
PRESSURE_PRED = PRESSURE / "paper1_split_strategy_outputs_Version_2" / "global_results" / "predictions"
R = OUT / "results"
TARGETS = {
    "co2_0015bar": "uptake(mmol/g) CO2 at 0.015 bar",
    "co2_015bar": "uptake(mmol/g) CO2 at 0.15 bar",
    "ch4_58bar": "uptake(mmol/g) methane at 5.8 bar",
    "ch4_65bar": "uptake(mmol/g) methane at 65 bar",
}
GROUPS = {"geo": "geo_cluster", "mc": "metal_cluster", "func": "func_cluster", "flig": "linker_cluster"}
PRED_COLS = ["filename_norm", "target", "prediction", "absolute_error", "residual", "target_slug", "descriptor_family", "split_family", "model", "fold_id", "topology_label"]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(values: pd.Series) -> pd.Series:
    # Exact historical implementation, not a suffix-only replacement.
    return values.astype(str).str.replace(".cif", "", regex=False).str.strip()


def top_ids(frame: pd.DataFrame, score: str, k: int) -> set[str]:
    return set(frame.sort_values([score, "filename_norm"], ascending=[False, True], kind="mergesort").head(k).filename_norm)


def verify_archives() -> None:
    rows = []
    for directory in (CORE, PRESSURE):
        checks = pd.read_csv(directory / "CHECKSUMS.csv")
        for item in checks.itertuples(index=False):
            path = directory / str(item.RelativePath).replace("\\", "/")
            assert path.is_file(), path
            digest = sha(path)
            assert path.stat().st_size == int(item.SizeBytes) and digest.upper() == str(item.SHA256).upper(), path
            rows.append({"archive_root": directory.name, "relative_path": str(item.RelativePath).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": digest, "status": "PASS"})
    pd.DataFrame(rows).to_csv(R / "recovered_file_checksums.csv", index=False)
    assert len(rows) == 43
    recovered_pipeline = CORE / "split_strategy_pipeline_raw_arc_V2.py"
    historical_pipeline = ROOT / "old code" / "split_strategy_pipeline_raw_arc_V2.py"
    assert sha(recovered_pipeline) == sha(historical_pipeline)
    archives = []
    for name in ("ARCMOF_Core_Evidence.zip", "ARCMOF_Pressure_Evidence.zip"):
        archive = REC.parent / name
        archives.append({"archive": name, "size_bytes": archive.stat().st_size, "sha256": sha(archive)})
    pd.DataFrame(archives).to_csv(R / "source_archive_sha256.csv", index=False)


def geometry_audit(master: pd.DataFrame, metrics: pd.DataFrame) -> None:
    alignment = []
    for stem, column in GROUPS.items():
        source = pd.read_csv(CORE / f"{stem}-clusters.csv", usecols=["filename", "cluster_id"])
        source["filename_norm"] = normalize(source.filename)
        assert source.filename_norm.is_unique and not source.cluster_id.isna().any()
        join = master[["filename_norm", column]].merge(source[["filename_norm", "cluster_id"]], on="filename_norm", how="left", validate="one_to_one", indicator=True)
        expected = pd.to_numeric(join[column], errors="coerce")
        actual = pd.to_numeric(join.cluster_id, errors="coerce")
        equal = np.isclose(expected, actual, rtol=0, atol=0, equal_nan=True)
        assert equal.all(), stem
        alignment.append({"group_source": f"{stem}-clusters.csv", "master_column": column, "source_rows": len(source), "unique_source_ids": source.filename_norm.nunique(), "evaluated_rows": len(join), "matched_ids": int((join._merge == "both").sum()), "label_mismatches": int((~equal).sum())})
    pd.DataFrame(alignment).to_csv(R / "group_source_alignment.csv", index=False)
    counts = master.geo_cluster.value_counts(dropna=False)
    eligible = counts[counts >= 2]
    size_bins = pd.DataFrame({"group_size": counts.value_counts().index.astype(int), "group_count": counts.value_counts().values}).sort_values("group_size")
    size_bins["structure_count"] = size_bins.group_size * size_bins.group_count
    size_bins.to_csv(R / "geometry_group_size_distribution.csv", index=False)
    summary = {"evaluated_master_rows": len(master), "raw_groups": len(counts), "missing_rows": int(master.geo_cluster.isna().sum()), "singleton_groups_excluded": int((counts == 1).sum()), "singleton_rows_excluded": int(counts[counts == 1].sum()), "eligible_groups": len(eligible), "eligible_rows": int(eligible.sum()), "eligible_group_median_size": float(eligible.median()), "eligible_group_max_size": int(eligible.max()), "eligible_groups_size_2_or_3_fraction": float((eligible <= 3).mean()), "eligible_rows_in_size_2_or_3_fraction": float(eligible[eligible <= 3].sum() / eligible.sum())}
    assert (summary["raw_groups"], summary["singleton_rows_excluded"], summary["eligible_groups"], summary["eligible_rows"]) == (146411, 78631, 67780, 185113)
    (R / "geometry_group_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    singleton_rows = master.loc[master.geo_cluster.map(counts).eq(1), ["filename_norm", "geo_cluster"]].sort_values("filename_norm")
    assert len(singleton_rows) == summary["singleton_rows_excluded"] and singleton_rows.filename_norm.is_unique
    singleton_rows.to_csv(R / "geometry_singleton_excluded_ids.csv.gz", index=False, compression="gzip")
    sub = master.loc[master.geo_cluster.map(counts).ge(2)].copy().reset_index(drop=True)
    reference = metrics[(metrics.target_slug == "co2_015bar") & (metrics.descriptor_family == "enriched_interpretable") & (metrics.split_family == "geo_grouped") & (metrics.model == "rf")].set_index("fold_id")
    assert len(reference) == 5
    folds = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(sub, groups=sub.geo_cluster)):
        train_groups, test_groups = set(sub.iloc[train].geo_cluster), set(sub.iloc[test].geo_cluster)
        overlap = len(train_groups & test_groups)
        row = {"fold": fold, "train_n": len(train), "test_n": len(test), "train_groups": len(train_groups), "test_groups": len(test_groups), "group_overlap": overlap, "saved_metric_test_n": int(reference.loc[fold, "n_test"])}
        assert row["test_n"] == row["saved_metric_test_n"] and overlap == 0
        folds.append(row)
    pd.DataFrame(folds).to_csv(R / "geometry_fold_integrity.csv", index=False)


def read_prediction(path: Path, *, slug: str, split: str, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=PRED_COLS, low_memory=False)
    assert frame.filename_norm.is_unique
    assert (frame.target_slug == slug).all() and (frame.split_family == split).all()
    assert (frame.descriptor_family == "enriched_interpretable").all() and (frame.model == "rf").all()
    assert (frame.fold_id == fold).all()
    assert np.allclose(frame.residual, frame.target - frame.prediction, atol=1e-10, rtol=0)
    assert np.allclose(frame.absolute_error, np.abs(frame.residual), atol=1e-10, rtol=0)
    frame["fold"] = fold
    return frame


def fold_metrics(frame: pd.DataFrame, reference: pd.Series, slug: str, split: str, fold: int) -> dict:
    y, pred = frame.target.to_numpy(), frame.prediction.to_numpy()
    k = math.ceil(.05 * len(frame))
    true_top, pred_top = set(np.argsort(-y)[:k]), set(np.argsort(-pred)[:k])
    recovery = len(true_top & pred_top) / k
    result = {"target_slug": slug, "split_family": split, "fold": fold, "n": len(frame), "saved_n": int(reference.n_test), "r2": r2_score(y, pred), "saved_r2": float(reference.r2), "mae": float(np.mean(np.abs(y - pred))), "saved_mae": float(reference.mae), "fold_local_top5_recovery": recovery, "saved_fold_local_top5_recovery": float(reference.top_5pct_recall)}
    result["max_metric_abs_difference"] = max(abs(result["r2"] - result["saved_r2"]), abs(result["mae"] - result["saved_mae"]), abs(result["fold_local_top5_recovery"] - result["saved_fold_local_top5_recovery"]))
    assert result["n"] == result["saved_n"]
    return result


def random_analysis(master: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict[int, set[str]]]:
    directory = CORE_RUN / "global_results" / "predictions"
    reference = metrics[(metrics.target_slug == "co2_015bar") & (metrics.descriptor_family == "enriched_interpretable") & (metrics.split_family == "random") & (metrics.model == "rf")].set_index("fold_id")
    assert len(reference) == 5
    frames, fold_results = [], []
    for fold in range(5):
        name = f"co2_015bar__enriched_interpretable__random__rf__fold{fold}_predictions.csv"
        frame = read_prediction(directory / name, slug="co2_015bar", split="random", fold=fold)
        frames.append(frame[["filename_norm", "target", "prediction", "fold"]])
        fold_results.append(fold_metrics(frame, reference.loc[fold], "co2_015bar", "random", fold))
    pd.DataFrame(fold_results).to_csv(R / "original_random_fold_reconciliation.csv", index=False)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 263745
    grouped = raw.groupby("filename_norm", sort=False)
    agg = grouped.agg(y_true=("target", "first"), y_min=("target", "min"), y_max=("target", "max"), y_pred_mean=("prediction", "mean"), n_oos_predictions=("prediction", "size"), folds_or_repeats_seen=("fold", lambda x: ",".join(map(str, sorted(x))))).reset_index()
    assert np.allclose(agg.y_min, agg.y_max, atol=1e-12, rtol=0)
    agg = agg.drop(columns=["y_min", "y_max"])
    assert len(agg) == 177411 and agg.filename_norm.is_unique
    target_check = agg.merge(master[["filename_norm", TARGETS["co2_015bar"]]], on="filename_norm", validate="one_to_one")
    assert len(target_check) == len(agg) and np.allclose(target_check.y_true, target_check[TARGETS["co2_015bar"]], atol=1e-10, rtol=0)
    archive = pd.read_csv(ROOT / "historical_oos" / "co2_015bar__enriched_interpretable__random__rf__aggregated_oos.csv.gz")
    assert archive.filename_norm.is_unique
    check = agg.merge(archive, on="filename_norm", validate="one_to_one", suffixes=("_original", "_archive"))
    assert len(check) == len(agg) == len(archive)
    assert (check.n_oos_predictions_original == check.n_oos_predictions_archive).all()
    assert (check.folds_or_repeats_seen_original.astype(str) == check.folds_or_repeats_seen_archive.astype(str)).all()
    target_max = float(np.max(np.abs(check.y_true_original - check.y_true_archive)))
    pred_diff = np.abs(check.y_pred_mean_original - check.y_pred_mean_archive)
    multiplicity = agg.n_oos_predictions.value_counts().sort_index()
    assert list(multiplicity.index) == [1, 2, 3, 4, 5]
    multiplicity.rename_axis("oos_predictions_per_mof").reset_index(name="structures").to_csv(R / "original_random_multiplicity.csv", index=False)
    previous = pd.read_csv(ROOT / "V6_AdvancedScience" / "final_reviewer_computation" / "results" / "random_aggregate_reconstruction.csv").iloc[0]
    reconciliation = {"original_prediction_rows": len(raw), "original_unique_oos_mofs": len(agg), "archive_unique_oos_mofs": len(archive), "multiplicity_mismatches": 0, "fold_token_mismatches": 0, "target_max_absolute_difference": target_max, "original_vs_archive_mean_prediction_mae": float(pred_diff.mean()), "original_vs_archive_max_prediction_abs_difference": float(pred_diff.max()), "original_vs_archive_within_1e_10_fraction": float((pred_diff < 1e-10).mean()), "prior_new_fit_vs_archive_mean_prediction_mae": float(previous.mean_absolute_prediction_difference)}
    pd.DataFrame([reconciliation]).to_csv(R / "original_random_archive_reconciliation.csv", index=False)

    k = math.ceil(.05 * len(agg))
    mean_ids = top_ids(agg, "y_pred_mean", k)
    truth_ids = top_ids(agg, "y_true", k)
    average_recovery = len(mean_ids & truth_ids) / k
    sampled = []
    sampled_lists: dict[int, set[str]] = {}
    for seed in range(20260, 20265):
        rng = np.random.default_rng(seed)
        draw = raw.assign(draw=rng.random(len(raw)))
        selected = draw.loc[draw.groupby("filename_norm").draw.idxmin(), ["filename_norm", "prediction"]]
        one = selected.merge(agg[["filename_norm", "y_true", "y_pred_mean", "n_oos_predictions"]], on="filename_norm", validate="one_to_one")
        assert len(one) == len(agg) and one.filename_norm.is_unique
        ids = top_ids(one, "prediction", k)
        sampled_lists[seed] = ids
        sampled.append({"seed": seed, "population_n": len(one), "k": k, "singletons": int(multiplicity.loc[1]), "one_vs_average_jaccard": len(ids & mean_ids) / len(ids | mean_ids), "changed_shortlist_memberships_symmetric_difference": len(ids ^ mean_ids), "one_prediction_reference_recovery": len(ids & truth_ids) / k, "average_reference_recovery": average_recovery, "one_minus_average_recovery": len(ids & truth_ids) / k - average_recovery})
    pd.DataFrame(sampled).to_csv(R / "original_random_sampling_sensitivity.csv", index=False)
    return agg, fold_results, sampled_lists


def pressure_analysis(master: pd.DataFrame, metrics: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    fold_results: list[dict] = []
    family_rows, iqr_rows = [], []
    for slug, target_col in TARGETS.items():
        reference = metrics[(metrics.target_slug == slug) & (metrics.descriptor_family == "enriched_interpretable") & (metrics.split_family == "topology_grouped") & (metrics.model == "rf")].set_index("fold_id")
        assert len(reference) == 5
        parts = []
        for fold in range(5):
            name = f"{slug}__enriched_interpretable__topology_grouped__rf__fold{fold}_predictions.csv"
            part = read_prediction(PRESSURE_PRED / name, slug=slug, split="topology_grouped", fold=fold)
            fold_results.append(fold_metrics(part, reference.loc[fold], slug, "topology_grouped", fold))
            parts.append(part[["filename_norm", "target", "prediction", "absolute_error", "residual", "topology_label", "fold"]])
        full = pd.concat(parts, ignore_index=True)
        assert len(full) == len(master) == 263744 and full.filename_norm.is_unique
        assert full.topology_label.notna().all() and full.topology_label.nunique() == 47
        fold_counts = full.groupby("topology_label").fold.nunique()
        assert (fold_counts == 1).all(), slug
        check = full.merge(master[["filename_norm", target_col, "topology_label"]], on="filename_norm", validate="one_to_one", suffixes=("_pred", "_master"))
        assert len(check) == len(full)
        assert (check.topology_label_pred == check.topology_label_master).all()
        assert np.allclose(check.target, check[target_col], atol=1e-10, rtol=0)
        q25, q75 = full.target.quantile([.25, .75])
        iqr = q75 - q25
        assert iqr > 0
        iqr_rows.append({"target_slug": slug, "target_column": target_col, "population_n": len(full), "q25_mmol_g": q25, "q75_mmol_g": q75, "iqr_mmol_g": iqr})
        family = full.groupby("topology_label", as_index=False).agg(n=("filename_norm", "size"), mae_mmol_g=("absolute_error", "mean"), mean_signed_residual_truth_minus_prediction_mmol_g=("residual", "mean"))
        family["target_slug"] = slug
        family["target_iqr_mmol_g"] = iqr
        family["mae_over_target_iqr"] = family.mae_mmol_g / iqr
        family["signed_residual_over_target_iqr"] = family.mean_signed_residual_truth_minus_prediction_mmol_g / iqr
        family_rows.append(family)
        frames[slug] = full
    pd.DataFrame(fold_results).to_csv(R / "original_topology_fold_reconciliation.csv", index=False)
    pd.DataFrame(iqr_rows).to_csv(R / "pressure_target_iqr_exact.csv", index=False)
    families = pd.concat(family_rows, ignore_index=True)
    assert len(families) == 188
    families.to_csv(R / "pressure_family_exact.csv", index=False)
    # The previously saved exact group table covered only CO2 0.15 bar.
    old = pd.read_csv(ROOT / "results" / "tables" / "table_exact_group_resolved_errors.csv")
    old = old[(old.target_slug == "co2_015bar") & (old.descriptor_family == "enriched_interpretable") & (old.split_family == "topology_grouped") & (old.model == "rf") & (old.group_column == "topology_label")]
    previous = families[families.target_slug == "co2_015bar"].merge(old[["group_id", "n", "mean_abs_error", "mean_signed_residual"]], left_on="topology_label", right_on="group_id", validate="one_to_one", suffixes=("_recovered", "_saved"))
    assert len(previous) == 47 and (previous.n_recovered == previous.n_saved).all()
    diff_mae = np.abs(previous.mae_mmol_g - previous.mean_abs_error)
    diff_resid = np.abs(previous.mean_signed_residual_truth_minus_prediction_mmol_g - previous.mean_signed_residual)
    pd.DataFrame([{"groups": len(previous), "n": int(previous.n_recovered.sum()), "max_mae_abs_difference": float(diff_mae.max()), "max_signed_residual_abs_difference": float(diff_resid.max())}]).to_csv(R / "prior_exact_group_reconciliation.csv", index=False)
    return frames, fold_results


def paired_pressure_analysis(master: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    wide = master[["filename_norm", "topology_label"]].copy()
    coverage = []
    for slug, frame in frames.items():
        coverage.append({"target_slug": slug, "rows": len(frame), "unique_ids": frame.filename_norm.nunique(), "unique_topology_labels": frame.topology_label.nunique()})
        data = frame[["filename_norm", "target", "prediction", "absolute_error", "residual", "fold"]].rename(columns={c: f"{slug}__{c}" for c in ("target", "prediction", "absolute_error", "residual", "fold")})
        wide = wide.merge(data, on="filename_norm", how="inner", validate="one_to_one")
    coverage.append({"target_slug": "four_target_intersection", "rows": len(wide), "unique_ids": wide.filename_norm.nunique(), "unique_topology_labels": wide.topology_label.nunique()})
    pd.DataFrame(coverage).to_csv(R / "pressure_id_coverage.csv", index=False)
    assert len(wide) == len(master) == 263744 and wide.filename_norm.is_unique
    for slug in list(TARGETS)[1:]:
        assert (wide[f"{slug}__fold"] == wide["co2_0015bar__fold"]).all()
    iqrs = pd.read_csv(R / "pressure_target_iqr_exact.csv").set_index("target_slug").iqr_mmol_g.to_dict()
    top_sets = {}
    k = math.ceil(.10 * len(wide))
    for slug in TARGETS:
        col = f"{slug}__norm_absolute_error"
        wide[col] = wide[f"{slug}__absolute_error"] / iqrs[slug]
        top_sets[slug] = top_ids(wide, col, k)
    ids = wide.filename_norm
    number_high = sum(ids.isin(top_sets[slug]).astype(int) for slug in TARGETS)
    pd.DataFrame({"targets_in_global_top_error_decile": range(5)}).merge(number_high.value_counts().rename_axis("targets_in_global_top_error_decile").reset_index(name="structures"), how="left", on="targets_in_global_top_error_decile").fillna({"structures": 0}).astype(int).to_csv(R / "pressure_high_error_multiplicity.csv", index=False)
    pairs = []
    for left, right in combinations(TARGETS, 2):
        a, b = top_sets[left], top_sets[right]
        intersection = len(a & b)
        rho = wide[[f"{left}__norm_absolute_error", f"{right}__norm_absolute_error"]].corr(method="spearman").iloc[0, 1]
        pairs.append({"target_left": left, "target_right": right, "paired_mofs": len(wide), "high_error_top10_k_each": k, "spearman_normalized_absolute_error": rho, "both_global_top10_error_n": intersection, "global_top10_error_jaccard": intersection / len(a | b), "both_top10_vs_independence_lift": intersection / (k * k / len(wide)), "p_right_top10_given_left_top10": intersection / k})
    pd.DataFrame(pairs).to_csv(R / "pressure_pairwise_error_persistence.csv", index=False)
    within = []
    family_sizes = wide.topology_label.value_counts()
    for label, n in family_sizes.items():
        if n < 500:
            continue
        sub = wide.loc[wide.topology_label == label]
        local_k = math.ceil(.10 * n)
        family_top = {slug: top_ids(sub, f"{slug}__norm_absolute_error", local_k) for slug in TARGETS}
        for left, right in combinations(TARGETS, 2):
            overlap = len(family_top[left] & family_top[right])
            rho = sub[[f"{left}__norm_absolute_error", f"{right}__norm_absolute_error"]].corr(method="spearman").iloc[0, 1]
            within.append({"topology_label": label, "family_n": n, "target_left": left, "target_right": right, "family_top10_k_each": local_k, "spearman_normalized_absolute_error": rho, "both_family_top10_error_n": overlap, "within_family_top10_overlap_fraction": overlap / local_k, "within_family_overlap_vs_independence_lift": overlap / (local_k * local_k / n)})
    pd.DataFrame(within).sort_values(["family_n", "topology_label", "target_left", "target_right"], ascending=[False, True, True, True]).to_csv(R / "within_family_pressure_persistence.csv", index=False)
    family = pd.read_csv(R / "pressure_family_exact.csv")
    eligible = family[family.n >= 500].copy()
    count_eligible = eligible.groupby("target_slug").topology_label.nunique()
    assert count_eligible.nunique() == 1
    top_quartile_cutoff = math.ceil(.25 * count_eligible.iloc[0])
    eligible["normalized_mae_rank"] = eligible.groupby("target_slug").mae_over_target_iqr.rank(ascending=False, method="min")
    eligible["top_quartile_difficulty"] = eligible.normalized_mae_rank <= top_quartile_cutoff
    ranks = []
    for label, sub in eligible.groupby("topology_label"):
        row = {"topology_label": label, "n_each_target": int(sub.n.iloc[0]), "eligible_families_each_target": int(count_eligible.iloc[0]), "top_quartile_rank_cutoff": top_quartile_cutoff, "targets_in_top_quartile_difficulty": int(sub.top_quartile_difficulty.sum())}
        for item in sub.itertuples():
            row[f"{item.target_slug}__mae_over_iqr"] = item.mae_over_target_iqr
            row[f"{item.target_slug}__signed_residual_over_iqr"] = item.signed_residual_over_target_iqr
            row[f"{item.target_slug}__difficulty_rank"] = item.normalized_mae_rank
        ranks.append(row)
    pd.DataFrame(ranks).sort_values(["targets_in_top_quartile_difficulty", "co2_015bar__mae_over_iqr"], ascending=[False, False]).to_csv(R / "pressure_family_rank_summary.csv", index=False)
    sampled = pd.read_csv(ROOT / "V6_AdvancedScience" / "final_reviewer_computation" / "results" / "topology_pressure_sampled_families.csv")
    comparison = sampled.merge(family[["target_slug", "topology_label", "n", "mae_mmol_g", "mean_signed_residual_truth_minus_prediction_mmol_g", "mae_over_target_iqr"]], on=["target_slug", "topology_label"], validate="one_to_one", suffixes=("_sampled", "_full"))
    comparison["sample_minus_full_normalized_mae"] = comparison.mae_over_target_iqr_sampled - comparison.mae_over_target_iqr_full
    comparison[["target_slug", "topology_label", "n_sampled", "n_full", "mae_over_target_iqr_sampled", "mae_over_target_iqr_full", "sample_minus_full_normalized_mae"]].to_csv(R / "sampled_vs_full_pressure_families.csv", index=False)


def shortlist_reconciliation(random_agg: pd.DataFrame, sample_lists: dict[int, set[str]], topology: pd.DataFrame) -> None:
    archive = pd.read_csv(ROOT / "historical_oos" / "co2_015bar__enriched_interpretable__topology_grouped__rf__aggregated_oos.csv.gz")
    topo_check = topology[["filename_norm", "target", "prediction"]].merge(archive, on="filename_norm", validate="one_to_one")
    assert len(topo_check) == 263744 and (archive.n_oos_predictions == 1).all()
    top_diff = np.abs(topo_check.prediction - topo_check.y_pred_mean)
    pd.DataFrame([{"n": len(topo_check), "topology_original_vs_archive_prediction_mae": float(top_diff.mean()), "topology_original_vs_archive_max_abs_difference": float(top_diff.max())}]).to_csv(R / "original_topology_archive_reconciliation.csv", index=False)
    common = random_agg.merge(topology[["filename_norm", "target", "prediction"]].rename(columns={"target": "topology_target", "prediction": "topology_prediction"}), on="filename_norm", validate="one_to_one")
    assert len(common) == 177411 and np.allclose(common.y_true, common.topology_target, atol=1e-10, rtol=0)
    k = math.ceil(.05 * len(common))
    true_ids = top_ids(common, "y_true", k)
    random_ids = top_ids(common, "y_pred_mean", k)
    topo_ids = top_ids(common, "topology_prediction", k)
    intersection = len(random_ids & topo_ids)
    summary = {"common_oos_population_n": len(common), "k": k, "common_shortlist_n": intersection, "random_only_n": len(random_ids - topo_ids), "topology_only_n": len(topo_ids - random_ids), "random_topology_jaccard": intersection / len(random_ids | topo_ids), "random_true_top5_recovery": len(random_ids & true_ids) / k, "topology_true_top5_recovery": len(topo_ids & true_ids) / k, "tie_policy": "ceil(0.05*N); descending score then ascending filename_norm"}
    pd.DataFrame([summary]).to_csv(R / "original_random_topology_shortlist.csv", index=False)
    prior = pd.read_csv(ROOT / "V6_AdvancedScience" / "results" / "pairwise_top5_summary.csv").iloc[0]
    fields = [("common_oos_population_n", "common_population_n"), ("k", "top_k"), ("common_shortlist_n", "common_shortlist_n"), ("random_only_n", "random_only_n"), ("topology_only_n", "topology_only_n"), ("random_topology_jaccard", "random_topology_jaccard"), ("random_true_top5_recovery", "random_reference_recovery"), ("topology_true_top5_recovery", "topology_reference_recovery")]
    reconcile = []
    for recovered_name, prior_name in fields:
        delta = float(summary[recovered_name]) - float(prior[prior_name])
        reconcile.append({"metric": recovered_name, "recovered_original": summary[recovered_name], "prior_v6": prior[prior_name], "difference": delta, "status": "MATCH" if abs(delta) < 1e-10 else "DIFFERENCE"})
    pd.DataFrame(reconcile).to_csv(R / "v6_shortlist_reconciliation.csv", index=False)
    sampled = []
    for seed, ids in sample_lists.items():
        overlap = len(ids & topo_ids)
        sampled.append({"seed": seed, "population_n": len(common), "k": k, "one_random_vs_topology_jaccard": overlap / len(ids | topo_ids), "one_random_topology_common_n": overlap, "mean_random_vs_topology_jaccard": summary["random_topology_jaccard"], "one_random_true_top5_recovery": len(ids & true_ids) / k, "mean_random_true_top5_recovery": summary["random_true_top5_recovery"], "topology_true_top5_recovery": summary["topology_true_top5_recovery"]})
    pd.DataFrame(sampled).to_csv(R / "original_sampling_vs_topology_shortlist.csv", index=False)


def main() -> None:
    R.mkdir(parents=True, exist_ok=True)
    verify_archives()
    columns = ["filename_norm", "geo_cluster", "metal_cluster", "func_cluster", "linker_cluster", "topology_label", *TARGETS.values()]
    master = pd.read_csv(ROOT / "results" / "global_processed" / "assembled_master_table.csv", usecols=columns, low_memory=False)
    assert len(master) == 263744 and master.filename_norm.is_unique
    metrics = pd.read_csv(CORE_RUN / "global_results" / "metrics" / "all_fold_metrics_long.csv")
    local_metrics = pd.read_csv(ROOT / "results" / "metrics" / "all_fold_metrics_long.csv")
    assert len(metrics) == len(local_metrics) == 1920
    assert metrics.equals(local_metrics)
    geometry_audit(master, metrics)
    random_agg, _, sample_lists = random_analysis(master, metrics)
    frames, _ = pressure_analysis(master, metrics)
    paired_pressure_analysis(master, frames)
    shortlist_reconciliation(random_agg, sample_lists, frames["co2_015bar"])
    print("Historical audit complete; no models were fitted.")


if __name__ == "__main__":
    main()
