# Output guide

## Generated outputs

The pipeline generates many outputs. Most generated outputs should not be committed directly.

Generated output folder:

```text
paper1_split_strategy_outputs_safe_lighter_v2/
```

Key generated files include:

- `global_results/metrics/all_fold_metrics_long.csv`
- `global_results/predictions/prediction_file_registry.csv`
- `global_results/predictions/*_predictions.csv`
- `global_results/predictions/sampled_fold_predictions_for_figures.csv`
- `global_results/tables/*.csv`
- `global_results/figures/main/*.png` and `.pdf`
- `global_results/figures/si/*.png` and `.pdf`
- `global_results/figures/main/figure_data_csv/*.csv`
- `global_results/figures/si/figure_data_csv/*.csv`

The folder `figure_redraw_package_v2/` contains a cleaned publication-facing subset of figure outputs and plot-data CSV files. It is intended for manuscript figure regeneration and source-data inspection. It should not be confused with the full generated pipeline output folder.
