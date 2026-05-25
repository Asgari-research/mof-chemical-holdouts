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

## What to commit

Commit compact and publication-facing files only:

- figure/table source-data CSVs;
- selected summary tables used in main text/SI;
- final publication figures if not too large;
- code and documentation.

## What not to commit

Do not commit:

- raw ARC--MOF source data;
- `clean_data.csv` or cluster-label input files;
- full per-experiment prediction files if very large;
- `.joblib` model files;
- `.pkl` files;
- generated output folders.
