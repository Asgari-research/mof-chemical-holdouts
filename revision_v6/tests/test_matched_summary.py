from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_five_paired_folds_and_exact_sizes():
    table = pd.read_csv(ROOT / "results/matched_random_control/matched_random_vs_topology_fold_metrics.csv")
    assert len(table) == 5
    assert table["fold_id"].nunique() == 5
    assert (table["n_train"] + table["n_test"] == 263744).all()
    assert (table["r2_advantage_random_minus_topology"] > 0).all()
    assert (table["top5_advantage_random_minus_topology"] > 0).all()


def test_reported_summary_matches_fold_table():
    table = pd.read_csv(ROOT / "results/matched_random_control/matched_random_vs_topology_fold_metrics.csv")
    summary = pd.read_csv(ROOT / "results/matched_random_control/matched_random_vs_topology_summary.csv").set_index("contrast")
    for name in [
        "r2_advantage_random_minus_topology",
        "mae_advantage_topology_minus_random",
        "top5_advantage_random_minus_topology",
    ]:
        assert abs(table[name].mean() - summary.loc[name, "mean"]) < 1e-12
