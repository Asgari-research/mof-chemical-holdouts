# Data access and redistribution policy

This repository includes a processed ARC--MOF-derived benchmark table for reproducibility:

```text
data/clean_data.csv
```

The included `clean_data.csv` file is a derived, machine-learning-ready benchmark table prepared for the present study. It was generated from ARC--MOF source data through data cleaning, identifier normalization, adsorption-target organization, descriptor preparation, and construction of benchmark-ready inputs. It should not be interpreted as the original ARC--MOF release.

## Original ARC--MOF source

The underlying source data are derived from ARC--MOF.

Original ARC--MOF data record:

```text
https://doi.org/10.5281/zenodo.6908728
```

Users should cite the ARC--MOF dataset and associated publication when using this repository or the included `clean_data.csv`.

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

ARC--MOF is distributed under the Creative Commons Attribution 4.0 International license. Users should follow the original ARC--MOF license and attribution requirements.

## Included data

Included in this repository:

```text
data/clean_data.csv
```

This file is included to support transparency, reproducibility, and reuse of the reported benchmark analysis.

## Files not redistributed

This repository does not redistribute the full ARC--MOF database or raw ARC--MOF source files.

The following files are not included unless explicitly permitted and documented:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
raw ARC--MOF CIF archives
complete ARC--MOF structural/database files
large full prediction files
trained model binaries
large generated output folders
```

Depending on the workflow stage, the following grouped metadata files may also be required locally:

```text
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

If these files are not included in the repository, users should obtain or generate them according to the project instructions and the original data-access conditions.

## Pipeline data paths

The main pipeline is:

```text
split_strategy_pipeline_raw_arc.py
```

Depending on the current configuration, the pipeline may expect `clean_data.csv` either:

1. in the repository `data/` folder, or
2. next to `split_strategy_pipeline_raw_arc.py`.

Before running the workflow, check the `DATA_DIR` or equivalent path setting inside the pipeline script.

## Citation requirements

When using this workflow or the included derived data file, cite:

1. the associated manuscript,
2. the ARC--MOF data paper,
3. the ARC--MOF dataset/data record,
4. this software repository, if used directly.
