# MOF Chemical Holdouts Benchmark

Reproducibility package for the manuscript:

**Beyond Random Splits: Domain Shift and Shortlist Instability in Machine-Learning-Guided MOF Discovery**

This repository supports a split-strategy benchmark for machine-learning-guided MOF adsorption screening. The workflow compares random train/test splits with chemically and structurally grouped holdouts while keeping the targets, descriptor families, model classes, hyperparameter logic, and metrics matched.

## What this repository contains

- `split_strategy_pipeline_raw_arc.py` — single-file Python pipeline used to assemble the benchmark table, run split-strategy ML experiments, generate figures/tables, and perform exact post-processing.
- `docs/` — data-access, output, reproducibility, and GitHub setup notes.
- `requirements.txt` and `environment.yml` — minimal Python environment specifications.

## What this repository does **not** contain

This repository intentionally does **not** redistribute the ARC--MOF source database or derived large raw/working data files. Users must obtain the source data from the original ARC--MOF data source and comply with the original license and citation requirements.

The required local input files are:

- `clean_data.csv`
- `geo-clusters.csv`
- `mc-clusters.csv`
- `func-clusters.csv`
- `flig-clusters.csv`

Optional local input files are:

- `geometric_properties.csv`
- `all_topology_lists.csv`

These files are ignored by Git to avoid redistributing the database. Place them next to `split_strategy_pipeline_raw_arc.py` before running the pipeline, or edit `DATA_DIR` in the script.

## Scientific scope

The central question is not which model or descriptor family is universally best. The goal is to test how the scientific conclusion of a MOF adsorption ML benchmark changes when the validation regime changes from interpolation-like random splits to deployment-relevant grouped extrapolation splits.

The main benchmark targets are:

- CO2 uptake at 0.015 bar
- CO2 uptake at 0.15 bar
- CH4 uptake at 5.8 bar
- CH4 uptake at 65 bar

The anchor target in the manuscript is CO2 uptake at 0.15 bar. This target is used as an adsorption-relevant benchmark target, not as a full process-optimality metric.

## Quick start

```bash
# 1. Create and activate an environment
conda env create -f environment.yml
conda activate mof-chemical-holdouts

# or with pip
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

# 2. Place required local input CSV files next to the pipeline script
# clean_data.csv, geo-clusters.csv, mc-clusters.csv, func-clusters.csv, flig-clusters.csv

# 3. Run the pipeline
python split_strategy_pipeline_raw_arc.py
```

The script writes outputs to a generated folder named similar to:

```text
paper1_split_strategy_outputs_safe_lighter_v2/
```

That generated folder is ignored by Git. Copy only compact, publication-facing outputs into `source_data/`, `figures/`, or `manuscript/` when preparing a release.

## Recommended release contents

Track these files in Git:

- source code
- README and documentation
- environment files
- compact source-data CSVs for figures and tables
- final manuscript/SI figures if not too large
- final manuscript/SI LaTeX files if desired

Do not track:

- ARC--MOF raw data
- local full working tables
- full per-experiment prediction CSVs if very large
- trained model binaries
- logs, caches, temporary outputs, and generated folders

## Citation

If you use this repository, please cite:

1. The associated manuscript, once available.
2. The ARC--MOF dataset and ARC--MOF data paper.

See `CITATION.cff` and `docs/DATA_ACCESS.md` for details.
