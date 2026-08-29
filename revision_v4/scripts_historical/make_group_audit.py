from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "group_audit"
GROUPS = {
    "geometry": "geo_cluster",
    "metal": "metal_cluster",
    "functional": "func_cluster",
    "linker": "linker_cluster",
    "topology": "topology_label",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(
        ROOT / "results" / "global_processed" / "assembled_master_table.csv",
        usecols=["filename_norm", "Crystalnet", *GROUPS.values()],
        low_memory=False,
    )
    rows = []
    dist_rows = []
    for family, col in GROUPS.items():
        labelled = master.loc[master[col].notna(), col]
        counts = labelled.value_counts()
        eligible = counts[counts >= 2]
        rows.append({
            "group_family": family,
            "source_column": col,
            "total_rows": len(master),
            "non_null_rows": int(len(labelled)),
            "missing_rows": int(master[col].isna().sum()),
            "raw_groups": int(len(counts)),
            "singleton_groups_removed_original": int((counts == 1).sum()),
            "singleton_rows_removed_original": int(counts[counts == 1].sum()),
            "eligible_rows_original": int(eligible.sum()),
            "eligible_groups_original": int(len(eligible)),
            "eligible_group_size_min": int(eligible.min()),
            "eligible_group_size_q25": float(eligible.quantile(0.25)),
            "eligible_group_size_median": float(eligible.median()),
            "eligible_group_size_q75": float(eligible.quantile(0.75)),
            "eligible_group_size_max": int(eligible.max()),
            "eligible_group_size_mean": float(eligible.mean()),
        })
        for group_id, size in counts.items():
            dist_rows.append({"group_family": family, "group_id": group_id, "size": int(size), "originally_eligible": bool(size >= 2)})
    pd.DataFrame(rows).to_csv(OUT / "group_construction_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT / "group_size_distributions.csv", index=False)
    raw_top = master["Crystalnet"].fillna("missing").astype(str)
    raw_counts = raw_top.value_counts(dropna=False)
    topology = pd.DataFrame({
        "rule": ["raw Crystalnet categories with count >=50 retained; categories with count <50 mapped to other"],
        "raw_categories": [int(len(raw_counts))],
        "raw_categories_below_50": [int((raw_counts < 50).sum())],
        "rows_mapped_to_other": [int(raw_counts[raw_counts < 50].sum())],
        "final_topology_groups": [int(master["topology_label"].nunique(dropna=True))],
        "final_other_rows": [int((master["topology_label"] == "other").sum())],
    })
    topology.to_csv(OUT / "topology_rare_category_rule.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(topology.to_string(index=False))


if __name__ == "__main__":
    main()
