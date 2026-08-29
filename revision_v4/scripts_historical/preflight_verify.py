from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "preflight"
HIST = ROOT / "historical_oos"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(ROOT / "clean_data.csv", nrows=5, low_memory=False)
    master = pd.read_csv(
        ROOT / "results" / "global_processed" / "assembled_master_table.csv",
        usecols=["filename_norm", "Dif", "cavity_window_gap"],
        low_memory=False,
    )
    cavity_max_abs_diff = float(
        np.nanmax(np.abs(master["Dif"].to_numpy() - master["cavity_window_gap"].to_numpy()))
    )

    splits = [
        "random",
        "geo_grouped",
        "metal_grouped",
        "func_grouped",
        "linker_grouped",
        "topology_grouped",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for split in splits:
        path = HIST / f"co2_015bar__enriched_interpretable__{split}__rf__aggregated_oos.csv.gz"
        df = pd.read_csv(path)
        if not df["filename_norm"].is_unique:
            raise RuntimeError(f"Non-unique filename_norm in {path.name}")
        tables[split] = df

    rows = []
    random = tables["random"][["filename_norm", "y_pred_mean"]]
    for split in splits[1:]:
        common = random.merge(
            tables[split][["filename_norm", "y_pred_mean"]],
            on="filename_norm",
            how="inner",
            suffixes=("_random", "_grouped"),
            validate="one_to_one",
        )
        rho = float(spearmanr(common["y_pred_mean_random"], common["y_pred_mean_grouped"]).statistic)
        n = len(common)
        for frac in (0.05, 0.01):
            k = max(1, int(np.ceil(frac * n)))
            a = set(common.nlargest(k, "y_pred_mean_random")["filename_norm"])
            b = set(common.nlargest(k, "y_pred_mean_grouped")["filename_norm"])
            intersection_n = len(a & b)
            union_n = len(a | b)
            rows.append(
                {
                    "split_a": "random",
                    "split_b": split,
                    "common_n": n,
                    "elite_fraction": frac,
                    "k": k,
                    "intersection_n": intersection_n,
                    "union_n": union_n,
                    "jaccard": intersection_n / union_n,
                    "spearman": rho,
                }
            )
    reconstructed = pd.DataFrame(rows)
    expected = pd.read_csv(HIST / "reconstructed_exact_elite_stability.csv")
    merged = reconstructed.merge(
        expected,
        on=["split_a", "split_b", "common_n", "elite_fraction", "k", "intersection_n", "union_n"],
        suffixes=("_new", "_expected"),
        validate="one_to_one",
    )
    merged["jaccard_abs_diff"] = (merged["jaccard_new"] - merged["jaccard_expected"]).abs()
    merged["spearman_abs_diff"] = (merged["spearman_new"] - merged["spearman_expected"]).abs()
    historical_match = len(merged) == len(expected) == len(reconstructed) and float(
        merged[["jaccard_abs_diff", "spearman_abs_diff"]].to_numpy().max()
    ) <= 1e-12
    reconstructed.to_csv(OUT / "historical_elite_recomputed.csv", index=False)
    merged.to_csv(OUT / "historical_elite_comparison.csv", index=False)

    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (OUT / "revision_environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    summary = {
        "clean_data_readable": True,
        "clean_data_sample_rows": len(clean),
        "clean_data_columns": list(clean.columns),
        "historical_oos_tables_readable": len(tables),
        "historical_oos_row_counts": {k: len(v) for k, v in tables.items()},
        "historical_elite_exact_match": historical_match,
        "historical_elite_max_jaccard_abs_diff": float(merged["jaccard_abs_diff"].max()),
        "historical_elite_max_spearman_abs_diff": float(merged["spearman_abs_diff"].max()),
        "cavity_window_gap_equals_Dif": cavity_max_abs_diff == 0.0,
        "cavity_window_gap_max_abs_diff": cavity_max_abs_diff,
        "versions": versions,
    }
    (OUT / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not historical_match:
        raise RuntimeError("Historical OOS elite reconstruction mismatch")
    if cavity_max_abs_diff != 0.0:
        raise RuntimeError("cavity_window_gap does not reproduce Dif")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
