from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision"
FILES = {
    "main": REV / "main_manuscript_Chemical_Science_revised.tex",
    "esi": REV / "electronic_supplementary_information_Chemical_Science_revised.tex",
    "response_md": REV / "response_to_reviewers.md",
    "response_tex": REV / "response_to_reviewers.tex",
}
TEXT = {k: p.read_text(encoding="utf-8") for k, p in FILES.items()}
TRACE: list[dict] = []


def check(claim: str, source: str, token: str, locations: list[str]) -> None:
    missing = [loc for loc in locations if token not in TEXT[loc]]
    TRACE.append({
        "claim": claim,
        "source": source,
        "expected_text": token,
        "locations": ";".join(locations),
        "status": "PASS" if not missing else "FAIL",
        "missing_from": ";".join(missing),
    })


def main() -> None:
    g = pd.read_csv(REV / "generalization" / "generalization_summary.csv").set_index(["descriptor_family", "split_family"])
    c = pd.read_csv(REV / "common_universe" / "common_universe_summary.csv").set_index("split_family")
    group = pd.read_csv(REV / "group_audit" / "group_construction_summary.csv").set_index("group_family")
    pre = json.loads((REV / "preflight" / "preflight_summary.json").read_text(encoding="utf-8"))
    diag = json.loads((REV / "generalization" / "reproduction_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((REV / "common_universe" / "common_universe_manifest.json").read_text(encoding="utf-8"))

    for descriptor, label in [("compact_geometry", "Compact"), ("enriched_interpretable", "Enriched")]:
        for split, split_label in [
            ("random", "Random"), ("geo_grouped", "Geometry"), ("metal_grouped", "Metal"),
            ("func_grouped", "Functional"), ("linker_grouped", "Linker"), ("topology_grouped", "Topology"),
        ]:
            r = g.loc[(descriptor, split)]
            row = (
                f"{label} & {split_label} & {r.r2_train_mean:.3f} & {r.r2_test_mean:.3f} & "
                f"{r.delta_r2_mean:.3f} & {r.mae_train_mean:.3f} & {r.mae_test_mean:.3f} & "
                f"{r.mae_deterioration_mean:.3f}"
            )
            check(f"generalization table {descriptor}/{split}", "generalization_summary.csv", row, ["esi"])

    for split, label in [
        ("random", "Random"), ("geo_grouped", "Geometry"), ("func_grouped", "Functional"),
        ("linker_grouped", "Linker"), ("metal_grouped", "Metal"), ("topology_grouped", "Topology"),
    ]:
        r = c.loc[split]
        row = (
            f"{label} & ${r.r2_mean:.3f}\\pm{r.r2_sd:.3f}$ & "
            f"${r.mae_mean:.3f}\\pm{r.mae_sd:.3f}$ & "
            f"${r.spearman_mean:.3f}\\pm{r.spearman_sd:.3f}$ & "
            f"${100*r.top5_recovery_mean:.1f}\\pm{100*r.top5_recovery_sd:.1f}\\%$"
        )
        check(f"fixed-universe table {split}", "common_universe_summary.csv", row, ["esi"])

    check("common-universe N", "common_universe_manifest.json", f"{manifest['n']:,}", ["main", "esi", "response_md", "response_tex"])
    check("historical random-topology common N prose", "preflight_summary.json", "177,411", ["main", "esi", "response_md"])
    check("historical random-topology common N TeX", "preflight_summary.json", "177{,}411", ["response_tex"])
    for value in ["0.917090", "0.199675", "0.066066"]:
        check(f"historical exact elite value {value}", "historical_elite_recomputed.csv", value, ["main", "esi", "response_md", "response_tex"])
    check("cavity semantics main", "preflight_summary.json", "cavity\\_window\\_gap} is exactly Dif", ["main"])
    check("cavity semantics ESI", "preflight_summary.json", "cavity\\_window\\_gap}=Dif", ["esi"])
    for dim in ["eight", "20", "one raw categorical", "21 raw"]:
        check(f"descriptor dimension phrase {dim}", "descriptor_families.json", dim, ["main"])

    for family, label in [("geometry", "Geometry"), ("metal", "Metal"), ("functional", "Functional"), ("linker", "Linker"), ("topology", "Topology")]:
        r = group.loc[family]
        token = f"{label} & {int(r.non_null_rows):,} & {int(r.missing_rows):,} & {int(r.raw_groups):,} & {int(r.singleton_groups_removed_original):,} & {int(r.eligible_rows_original):,} & {int(r.eligible_groups_original):,}"
        check(f"group construction {family}", "group_construction_summary.csv", token, ["esi"])

    check("maximum R2 reproduction difference", "reproduction_diagnostics.json", f"{diag['max_abs_differences']['r2_abs_diff']:.6f}", ["main", "esi", "response_md", "response_tex"])
    check("fold-2 MAE reproduction difference", "generalization_fold_metrics.csv", "0.000126", ["main", "esi", "response_md", "response_tex"])
    for version in [pre["versions"]["numpy"], pre["versions"]["pandas"], pre["versions"]["scikit_learn"]]:
        check(f"revision version {version}", "preflight_summary.json", version, ["main", "esi", "response_md", "response_tex"])
    check("revision Python version", "preflight_summary.json", "3.13.13", ["main", "esi", "response_md", "response_tex"])
    check("historical Python version", "historical run log", "3.11.14", ["main", "esi", "response_md", "response_tex"])
    check("random seed main/response", "historical code", "seed 42", ["main", "response_md", "response_tex"])
    check("random seed ESI", "historical code", "seed was 42", ["esi"])

    for fig in [
        "figure3_revised_single_panel.pdf", "figure4_revised_percent.pdf",
        "figure5_revised_uniform_markers.pdf", "figure7_revised_spearman_0_1.pdf",
    ]:
        check(f"figure mapping {fig}", "revision/figures", fig, ["main"])
        if not (REV / "figures" / fig).exists():
            TRACE[-1]["status"] = "FAIL"
            TRACE[-1]["missing_from"] = "filesystem"

    out = pd.DataFrame(TRACE)
    out.to_csv(REV / "NUMERICAL_TRACEABILITY.csv", index=False)
    failures = out[out.status != "PASS"]
    summary = {
        "checks": len(out),
        "passed": int((out.status == "PASS").sum()),
        "failed": len(failures),
        "status": "PASS" if failures.empty else "FAIL",
        "failures": failures.to_dict("records"),
    }
    (REV / "NUMERICAL_AUDIT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
