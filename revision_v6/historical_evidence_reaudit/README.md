# Recovered original ARC-MOF evidence audit

This is a read-only numerical audit of two original-output archives. It made **zero RF fits** and did not alter historical files or manuscript text. The local audit verified all 43 inventoried extracted files by size and SHA256; the hashes are released in `results/`. The original archives, original ARC-MOF inputs, full random/topology prediction rows, joined RAC features, model checkpoints, and the local list of individual geometry-singleton IDs are **not** included here. The compact CSV/JSON files and tests can be inspected without archive access. This compact release does not support a fresh RF refit by itself.

To check the released compact results without any original archives, run from the repository root:

```powershell
python -B -m unittest discover -s revision_v6/historical_evidence_reaudit/tests -v
```

To **recompute** the audit from original rows, obtain authorized local access to the recovered `ARCMOF_Core/` and `ARCMOF_Pressure/` archive contents and the full `ChemSci_revision` source (processed master, historical aggregate archive, saved V6 outputs and original metrics). Then set both paths and run:

```powershell
$env:ARC_MOF_SOURCE_ROOT = 'C:\path\to\ChemSci_revision'
$env:ARCMOF_RECOVERED_ROOT = 'C:\path\to\ARCMOF_recovered_working'
python revision_v6/historical_evidence_reaudit/scripts/audit_original_predictions.py
```

`ARCMOF_RECOVERED_ROOT` must point to the parent directory containing `ARCMOF_Core/` and `ARCMOF_Pressure/`. The script reads original fold-prediction CSVs and the local processed master; recomputation can also create a **local-only** `geometry_singleton_excluded_ids.csv.gz`, which must not be published. The main interpretation and corrections to earlier V6 evidence statements are in `FINAL_HISTORICAL_EVIDENCE_AUDIT.md`.

Top-5% lists use `ceil(0.05*N)`, descending score and ascending normalized filename for ties. The five one-prediction-per-MOF samples use NumPy generator seeds 20260–20264, draw uniformly among each MOF's actual available historical OOS appearances, and retain the same 177,411-MOF eligible population. Cross-pressure high-error lists analogously use `ceil(0.10*N)` and the same tie rule. Signed residual is **truth minus prediction** throughout.

Geometry provenance: the V2 source inventory includes the executed `split_strategy_pipeline_raw_arc_V2.py` and `build_compact_historical_oos.py`, but no located upstream generator for `geo-clusters.csv`. The recovered labels verify assignment and held-out grouping, not their clustering features, algorithm, distance metric or threshold.

Original Figure 2 (`figure2_v8_split_family_severity_polished`) uses **mean ± standard error of the mean across five outer-fold metric values**, not standard deviation. In the executed V2 pipeline, `_v7_summary_sem` (around lines 3344–3363) computes sample SD (`ddof=1`) divided by √5, and `_v8_figure_2_split_severity` (around lines 4008–4050) passes `r2_sem` and `mae_sem` as `yerr`. The existing repository file `figure_redraw_package_v2/data/main/figure_data_csv/figure2_v8_split_family_severity_polished_plot_data.csv` stores both `_sem` and `_std`; all 12 plotted split/descriptor rows have `n=5` and satisfy SEM=SD/√5. A precise caption is: “Points show the mean across five outer folds; error bars show ±1 standard error of those five fold metrics.” This is a caption correction; the existing plotted PDF needs no redraw for that wording. Fold errors are descriptive variation across validation folds, not independent-sample confidence intervals.
