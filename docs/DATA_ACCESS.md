# Data access and non-redistribution policy

This repository intentionally does **not** redistribute ARC--MOF source data or local derived working tables.

## Required local files

The pipeline expects the following required files to be present locally:

- `clean_data.csv`
- `geo-clusters.csv`
- `mc-clusters.csv`
- `func-clusters.csv`
- `flig-clusters.csv`

Optional files:

- `geometric_properties.csv`
- `all_topology_lists.csv`

By default, `split_strategy_pipeline_raw_arc.py` sets `BASE_DIR = Path(__file__).resolve().parent` and `DATA_DIR = BASE_DIR`, so these files should be placed next to the script unless `DATA_DIR` is edited.

## Why the database is not included

The ARC--MOF database and large derived prediction files are not included to avoid redistributing third-party data and to keep the repository lightweight. Users should obtain source files from the original ARC--MOF source and follow its license and attribution requirements.

## Citation requirements

When using this workflow, cite:

1. The associated manuscript.
2. The ARC--MOF data paper.
3. The ARC--MOF dataset/data record used to obtain the raw files.

## Manuscript data-availability text

A safe data-availability statement should include:

- repository/archive URL for this code package;
- exact ARC--MOF release/version;
- download date;
- DOI or source record URL;
- raw filenames used to build `clean_data.csv`;
- local preprocessing script or README used to generate the working table.

Do not submit the manuscript with placeholders such as `INSERT_ARC--MOF_RELEASE_OR_VERSION`.
