# Data

This folder contains the processed benchmark input table used to support the chemically auditable MOF holdout-validation study.

## Included file

* `clean_data.csv`

`clean_data.csv` is a processed ARC--MOF-derived benchmark table prepared for this study. It was generated from ARC--MOF source data through data cleaning, identifier normalization, adsorption-target organization, descriptor preparation, and construction of machine-learning-ready inputs. It should therefore be treated as a derived benchmark table, not as the original ARC--MOF release.

The file is included to support transparency, reproducibility, and reuse of the reported split-strategy benchmark. It allows users to inspect and rerun the benchmark workflow without reconstructing the main working table from raw ARC--MOF source files.

## Original data source

The underlying source data are derived from ARC--MOF. Users should cite the original ARC--MOF dataset and associated publication when using `clean_data.csv`.

Original ARC--MOF data record:

```text
https://doi.org/10.5281/zenodo.6908728
```

Associated ARC--MOF publication:

```bibtex
@article{burner2023arcmof,
  title   = {ARC--MOF: A Diverse Database of Metal--Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning},
  author  = {Burner, Jake and Schwiedrzik, Luca and Krykunov, Mykhaylo and Luo, Jun and Boyd, Peter G. and Woo, Tom K.},
  journal = {Chemistry of Materials},
  volume  = {35},
  number  = {3},
  pages   = {900--916},
  year    = {2023},
  doi     = {10.1021/acs.chemmater.2c02485}
}
```

ARC--MOF is distributed under the Creative Commons Attribution 4.0 International license. Because `clean_data.csv` is derived from ARC--MOF, users should follow the original ARC--MOF license and citation requirements.

## Files not included

This folder does not redistribute the full ARC--MOF database or raw source files.

The following files are not included and should be obtained from the original ARC--MOF source records or generated locally when needed:

* `geometric_properties.csv`
* `post_comb_vsa-CO2.csv`
* `methane.csv`
* raw ARC--MOF CIF archives
* full ARC--MOF structural/database files
* large generated prediction files
* trained model objects
* local output folders

For the full split-strategy pipeline, additional grouped metadata files may be required locally, depending on the workflow stage being rerun:

* `geo-clusters.csv`
* `mc-clusters.csv`
* `func-clusters.csv`
* `flig-clusters.csv`
* `all_topology_lists.csv`

These files are not redistributed here unless explicitly permitted and documented.

## How to use

To run the benchmark pipeline using the included processed table, either:

1. keep `clean_data.csv` in this `data/` folder and set the pipeline data path to `data/`, or
2. copy `clean_data.csv` next to `split_strategy_pipeline_raw_arc.py`, depending on the current pipeline configuration.

The current pipeline documentation should be checked for the expected `DATA_DIR` setting before running.

## Important scope note

`clean_data.csv` is provided as a derived benchmark table for this specific study. It should not be interpreted as an independent replacement for the original ARC--MOF database. Users who need the raw source data, full structural database, or additional descriptor files should obtain them directly from the original ARC--MOF records and comply with the original license and attribution requirements.
