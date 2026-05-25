# Reproducibility guide

## Environment

Use either `environment.yml` or `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate mof-chemical-holdouts
```

or:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

Place required input CSV files next to `split_strategy_pipeline_raw_arc.py`, then run:

```bash
python split_strategy_pipeline_raw_arc.py
```

The pipeline is designed to be restartable. It writes a run manifest and skips completed experiments when `RESUME_IF_AVAILABLE = True`.

## Expected output folder

The script writes a generated folder such as:

```text
paper1_split_strategy_outputs_safe_lighter_v2/
```

Important subfolders include:

```text
global_processed/
global_results/metrics/
global_results/predictions/
global_results/tables/
global_results/figures/main/
global_results/figures/si/
global_results/feature_importance/
```

## Memory and CPU settings

The script is configured for conservative execution:

- `N_JOBS = 1`
- memory-safe sampled prediction table for figures
- exact post-hoc analyses stream full prediction files from disk

Increase `N_JOBS` only if RAM is sufficient.

## Exact post-processing

The pipeline writes per-experiment prediction CSVs and a prediction registry. Exact group-resolved errors and exact elite-list stability are computed from these per-experiment files without concatenating all predictions into memory.
