# Figure inventory and evidence boundaries

**Scope:** V6 Advanced Science revision source release, 2026-11-03. This is a map of the 22 approved figures. The PDFs in `../../manuscript/v6/figures/` are locked; regeneration scripts produce **previews**, not replacements.

The figure source filenames are historical and are **not** equivalent to displayed manuscript numbering. Use the map below, not the number inside the asset name.

| Manuscript figure | Locked PDF asset | Evidence / available source | What is *not* guaranteed |
|---|---|---|---|
| Main 1 | `Figure1_Main.pdf` | TikZ vector workflow; matching editable source supplied | Exact editable source; compiled approved PDF frozen |
| Main 2 | `figure2_v8_split_family_severity_polished.pdf` | Historical split-severity plotting CSV in public figure_redraw_package_v2 | Numerical plot-data available; final typography/layout may differ |
| Main 3 | `v6_matched_random_vs_topology.pdf` | Matched-control five-fold CSV and plotting/replot_v6_controls.py | Five-fold summary regeneration; not a fresh model fit |
| Main 4 | `figure3_revised_single_panel.pdf` | Historical target-by-split CSV in figure_redraw_package_v2 | Numerical plot-data available; final formatting may differ |
| Main 5 | `figure6_v8_shift_and_descriptor_space_polished.pdf` | Pinned public descriptor-shift CSV in figure_redraw_package_v2 | Historical numeric source; exact final hybrid-artwork regeneration not certified |
| Main 6 | `figure5_revised_uniform_markers.pdf` | Historical difficult-group CSV plus original per-group aggregation | Source table available; display says Pore-regime, upstream boundaries not verified |
| Main 7 | `Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf` | Original artwork and numeric table excerpts; original CIF/render settings not recovered | Approved unchanged original artwork; no independent structural redraw |
| Main 8 | `figure4_revised_percent.pdf` | Historical screening metrics plotting CSV | Numerical source available; percentages/caption definitions must stay matched |
| Main 9 | `figure7_revised_spearman_0_1.pdf` | Historical top-5% shortlist and rank plotting CSV | Numerical source available; exact final style may differ |
| SI S1 | `si_v7_group_size_summary_logscale.pdf` | Historical group-size summary CSV | Source available |
| SI S2 | `si_v8_target_distributions_clear_labels.pdf` | Raw target-distribution plot-data CSV not recovered in V6 handoff | Exact redraw from original target rows not certified |
| SI S3 | `si_v8_prediction_scatter_panels_clear_labels.pdf` | Large plotted-prediction CSV at pinned public Git path | Source linked; source inputs not bundled in the handoff |
| SI S4 | `si_v8_pld_decile_errors_clear_labels.pdf` | Historical PLD-decile CSV | Source available |
| SI S5 | `si_heatmap_r2.pdf` | Historical mean R2 heatmap CSV | Source available |
| SI S6 | `si_heatmap_mae.pdf` | Historical mean MAE heatmap CSV | Source available |
| SI S7 | `si_heatmap_spearman_rho.pdf` | Historical mean Spearman rho heatmap CSV | Source available |
| SI S8 | `si_heatmap_kendall_tau.pdf` | Historical mean Kendall tau heatmap CSV | Source available |
| SI S9 | `si_heatmap_top_5pct_overlap.pdf` | Historical top-5% true-reference recovery heatmap CSV | Source available; overlap is not predicted-list Jaccard |
| SI S10 | `si_heatmap_top_5pct_enrichment.pdf` | Historical top-5% enrichment heatmap CSV | Source available |
| SI S11 | `si_v7_error_vs_screening_rank_clean.pdf` | Historical error-vs-rank CSV | Source available |
| SI S12 | `si_variance_decomposition.pdf` | Historical metric variance decomposition CSV | Source available |
| SI S13 | `v6_topology_fold_heterogeneity.pdf` | Five saved topology-fold scores and plotting/replot_v6_controls.py | Aggregate plot; not refitting |

## Scientific interpretation that remains open

The locked *Main 6* difficult-group artwork uses the label **Pore-regime NNNNN**. This is a display label for an imported geometry group ID, **not an independently established physical pore regime, threshold, or pore-shape mechanism**. The geometry-cluster upstream generator and numeric thresholds were not recovered in the V6 evidence. The author's earlier V6 wording used `Geometry-ID` intentionally. The approved artwork is retained byte-identically here; the discrepancy must be resolved through caption wording, response-letter wording, and author sign-off before journal submission. This documentation does not claim that the underlying physical group boundaries were verified.

The structural-audit figure is original preserved artwork, not an independently regenerated molecular structure. The underlying CIF/view settings are not recovered in the handoff.

## Replotting

- `manuscript/v6/plotting/replot_v6_controls.py` regenerates **two preview charts** in `manuscript/v6/generated_previews/`, never the locked final PDFs.
- Existing `figure_redraw_package_v2/code/make_publication_figures.py` provides historical analytical replotting from exported plot CSVs. It may produce distinct styling/label positions from the approved PDFs. Run with its own output directory.
- Two large plotted-data CSVs are pinned at [Git commit ff83785](https://github.com/Asgari-research/mof-chemical-holdouts/tree/ff83785a76f14fb18f2dac455105a892bf29452a/figure_redraw_package_v2/data), and the target-distribution raw input is not certified recovered.

Never call a regenerated preview byte-identical without comparing its content against the locked reference.
