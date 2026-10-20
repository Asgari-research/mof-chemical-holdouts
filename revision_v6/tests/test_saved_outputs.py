from pathlib import Path

import json
import pandas as pd


V6 = Path(__file__).resolve().parents[1]
RESULTS = V6 / "results"


def test_preflight_invariants():
    data = json.loads((RESULTS / "preflight_invariants.json").read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["common_universe_rows"] == 214531
    assert data["common_universe_unique_ids"] == 214531
    assert data["common_universe_checkpoint_count"] == 30
    assert data["generalization_checkpoint_count"] == 60


def test_pairwise_population_and_membership_are_unique():
    membership = pd.read_csv(RESULTS / "pairwise_shortlist_membership.csv")
    summary = pd.read_csv(RESULTS / "pairwise_top5_summary.csv").iloc[0]
    assert membership["filename_norm"].is_unique
    assert len(membership) == summary["common_population_n"]
    assert set(membership["shortlist_group"]) <= {"common", "random_only", "topology_only", "neither"}
    assert (membership["y_true"].notna()).all()


def test_top_k_counts_and_metrics():
    membership = pd.read_csv(RESULTS / "pairwise_shortlist_membership.csv")
    summary = pd.read_csv(RESULTS / "pairwise_top5_summary.csv").iloc[0]
    selected = membership["shortlist_group"].isin(["common", "random_only"])
    selected_topology = membership["shortlist_group"].isin(["common", "topology_only"])
    assert selected.sum() == summary["top_k"]
    assert selected_topology.sum() == summary["top_k"]
    assert 0 <= summary["random_topology_jaccard"] <= 1
    assert 0 <= summary["random_reference_recovery"] <= 1
    assert 0 <= summary["topology_reference_recovery"] <= 1


def test_repeat_level_limitation_is_explicit():
    data = json.loads((RESULTS / "repeat_level_prediction_availability.json").read_text(encoding="utf-8"))
    assert data["status"] == "BLOCKED_MISSING_REPEAT_LEVEL_PREDICTIONS"
    assert data["accessible_original_files"] == 0
    assert data["aggregate_has_individual_repeat_predictions"] is False


def test_asa_columns_are_not_conflated():
    evidence = pd.read_csv(RESULTS / "asa_unit_evidence.csv")
    assert {"ASA", "gASA"}.issubset(evidence.columns)
    assert (evidence["ASA"] != evidence["gASA"]).any()


def test_topology_error_contributions_are_single_lens():
    groups = pd.read_csv(RESULTS / "topology_group_error_concentration.csv")
    assert set(groups["group_column"]) == {"topology_label"}
    assert abs(groups["absolute_error_contribution"].sum() - 1.0) < 1e-10


def test_rare_other_ablation_declares_population_change():
    ablation = pd.read_csv(RESULTS / "rare_topology_other_evaluation_ablation.csv")
    assert list(ablation["population"]) == ["original_topology_oos", "exclude_rare_topology_other"]
    assert ablation.iloc[1]["n"] < ablation.iloc[0]["n"]
    assert "evaluation-only" in ablation.iloc[1]["interpretation"]
