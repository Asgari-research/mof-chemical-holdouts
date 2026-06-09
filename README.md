# MOF Chemical Holdouts Benchmark

Reproducibility package for the manuscript:

**The Split Defines the Claim: A Validation Grammar for Chemically Auditable MOF Discovery**

This repository supports a chemically auditable split-strategy benchmark for machine-learning-guided metal--organic framework (MOF) adsorption screening. The workflow compares random train/test splits with chemically and structurally grouped holdouts while keeping targets, descriptor families, model classes, hyperparameter logic, and metrics matched.

The central purpose is not to identify a universally best model or descriptor family. The purpose is to test how the scientific meaning of a MOF ML screening claim changes when validation shifts from interpolation-like random splitting to deployment-relevant grouped extrapolation across geometry, metal-cluster, functional-group, linker, and topology families.

In this project, the validation split is treated as a scientific variable:

> **The split defines the claim.**

---

## Repository overview

This repository contains code, documentation, a processed ARC--MOF-derived benchmark table, compact source-data files, and figure-regeneration materials for the split-strategy MOF adsorption benchmark.

Main components:

```text
mof-chemical-holdouts/
├── split_strategy_pipeline_raw_arc.py
├── README.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
├── data/
│   ├── README.md
│   └── clean_data.csv
├── docs/
├── figure_redraw_package_v2/
└── source_data/                  # optional, if compact source-data tables are unpacked here
```

The repository supports three levels of reproducibility:

1. **Benchmark rerun** from the included processed benchmark table and/or locally supplied ARC--MOF-derived input files.
2. **Exact post-processing audit** from regenerated prediction outputs.
3. **Publication figure regeneration** from compact plot-data CSV files in `figure_redraw_package_v2/`.

---

## What this repository contains

### Main pipeline

- `split_strategy_pipeline_raw_arc.py`  
  Main Python pipeline used to assemble the benchmark table, run split-strategy ML experiments, generate metrics, and perform exact post-processing.

The pipeline supports matched comparison across:

- random splitting,
- geometry-grouped splitting,
- metal-cluster-grouped splitting,
- functional-group-grouped splitting,
- linker-grouped splitting,
- topology-grouped splitting.

The main benchmark targets are:

- CO2 uptake at 0.015 bar,
- CO2 uptake at 0.15 bar,
- CH4 uptake at 5.8 bar,
- CH4 uptake at 65 bar.

The manuscript uses **CO2 uptake at 0.15 bar** as the anchor adsorption target. This target is used as an adsorption-relevant validation target, not as a complete process-optimality metric.

---

## Data included in this repository

This repository includes one processed ARC--MOF-derived benchmark table:

```text
data/clean_data.csv
```

`clean_data.csv` is a derived, machine-learning-ready benchmark table prepared for the present split-strategy study. It was generated from ARC--MOF source data through data cleaning, identifier normalization, adsorption-target organization, descriptor preparation, and construction of benchmark-ready inputs.

It should be treated as a **processed benchmark table**, not as the original ARC--MOF release.

The file is included to support transparency, reproducibility, and reuse of the reported benchmark analysis. It allows users to inspect and rerun the benchmark workflow without reconstructing the main working table from raw ARC--MOF source files.

---

## Original data source

The underlying source data are derived from ARC--MOF. Users should cite the original ARC--MOF dataset and associated publication when using `clean_data.csv` or this repository.

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

Users should follow the original ARC--MOF license and attribution requirements. This repository does not grant any additional rights to redistribute raw third-party ARC--MOF database files beyond the rights provided by the original data source.

---

## Files not included

This repository does **not** redistribute the full ARC--MOF database or raw ARC--MOF source files.

The following files are intentionally not included unless explicitly permitted and documented:

- raw ARC--MOF database files,
- raw third-party MOF structural databases,
- complete ARC--MOF CIF archives,
- `geometric_properties.csv`,
- `post_comb_vsa-CO2.csv`,
- `methane.csv`,
- large full per-experiment prediction CSV files,
- trained model binaries,
- large local working tables,
- local cache folders,
- temporary logs,
- private local paths,
- large generated output folders.

The repository remains a compact reproducibility, source-data, and figure-generation package rather than a redistribution of ARC--MOF.

---

## Additional local files for full regeneration

The included `data/clean_data.csv` supports rerunning the benchmark from the processed working table. Some full-regeneration or consistency-check workflows may require additional local grouped metadata files.

Depending on the workflow stage, the following additional files may be required locally:

```text
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
```

Optional local inputs:

```text
geometric_properties.csv
all_topology_lists.csv
```

If `clean_data.csv` is not used directly, the pipeline can build it from ARC--MOF-derived raw files, including:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
```

together with the grouped metadata files.

These raw/source files should be obtained from the original ARC--MOF records and placed locally according to `docs/DATA_ACCESS.md`.

---

## Documentation

The `docs/` folder contains documentation for data access, expected local file layout, output interpretation, source-data organization, GitHub/release preparation, and submission checks.

Recommended documentation files include:

```text
docs/
├── DATA_ACCESS.md
├── OUTPUTS.md
├── SOURCE_DATA_MANIFEST.md
├── REPRODUCIBILITY.md
├── GITHUB_SETUP.md
└── SUBMISSION_CHECKLIST.md
```

---

## Figure-redraw package

The folder:

```text
figure_redraw_package_v2/
```

contains a publication-facing figure-regeneration package with the cleaned plotting script, plot-data CSV files, regenerated main/SI figures, a figure manifest, and a QC report.

Expected structure:

```text
figure_redraw_package_v2/
├── README.md
├── requirements.txt
├── code/
│   └── make_publication_figures.py
├── data/
│   ├── main/
│   │   └── figure_data_csv/
│   ├── si/
│   │   └── figure_data_csv/
│   └── structural_audit/
└── figures/
    ├── main/
    ├── si/
    ├── figure_manifest.csv
    └── QC_REPORT.txt
```

The analytical main-text and SI figures are redrawn from supplied plot-data CSV files. The structural-audit figure is preserved as finalized manuscript artwork unless the underlying CIF files, rendering settings, and composite-figure script are also supplied.

---

## Scientific scope

This repository supports a validation-regime benchmark, not a complete materials-selection or process-optimization study.

The manuscript asks:

> When does a MOF ML validation score preserve the chemical meaning of the intended screening decision?

A random split asks whether a model interpolates within a familiar database neighbourhood. Grouped splits ask different chemical transfer questions:

| Split family | Chemistry-facing question | Interpretation |
|---|---|---|
| Random | Can the model interpolate within the same database neighbourhood? | Baseline interpolation regime. Useful, but insufficient by itself for discovery claims involving unfamiliar chemistry. |
| Geometry-grouped | Can the model transfer across pore-size or pore-shape regimes? | Pore-regime holdout, interpreted through PLD, LCD, density, void fraction, accessible surface area, and pore-volume descriptors. |
| Metal-cluster-grouped | Can the model transfer to unfamiliar node chemistry or coordination environments? | Node-family holdout. Relevant to metal identity, local coordination, charge distribution, and node environment. |
| Functional-group-grouped | Can the model transfer to unfamiliar local chemical motifs? | Local-motif holdout. Relevant to polar groups, heteroatoms, electrostatics, and low-pressure adsorption environments. |
| Linker-grouped | Can the model transfer to unfamiliar organic scaffold families? | Linker-family holdout, analogous in spirit to scaffold splits in molecular ML but adapted to framework solids. |
| Topology-grouped | Can the model transfer to unfamiliar framework connectivity and pore-network architecture? | Severe structural-extrapolation holdout. This is not a claim that topology alone controls uptake. |

The study reports regression metrics, rank metrics, top-5% screening metrics, exact elite-list stability, held-out-group diagnostics, and structural audit cases.

---

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate mof-chemical-holdouts
```

### Option 2: pip / virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The figure-redraw package has a smaller plotting-only environment. To use it independently:

```bash
cd figure_redraw_package_v2
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick start: run the main benchmark pipeline

After placing the required local input files in the expected location, run:

```bash
python split_strategy_pipeline_raw_arc.py
```

The included processed table is:

```text
data/clean_data.csv
```

Depending on the current configuration, the pipeline may expect `clean_data.csv` either:

1. in the repository `data/` folder, or
2. next to `split_strategy_pipeline_raw_arc.py`.

Before running the workflow, check the `DATA_DIR` or equivalent path setting inside the pipeline script.

The script writes generated outputs to a local output directory. Depending on the script version, this folder may have a name similar to:

```text
paper1_split_strategy_outputs_safe_lighter_v2/
```

Generated folders are ignored by Git. Do not commit large local generated outputs unless they have been curated into compact source-data or publication-facing files.

---

## Quick start: regenerate publication figures

The figure-redraw package can be run independently from the benchmark pipeline.

```bash
cd figure_redraw_package_v2
python code/make_publication_figures.py
```

For a clean conda environment:

```bash
conda create -n splitfigs python=3.11 -y
conda activate splitfigs
pip install -r requirements.txt
python code/make_publication_figures.py
```

To render one figure only:

```bash
python code/make_publication_figures.py --only figure3
python code/make_publication_figures.py --only si_prediction_scatter
```

Default outputs are written to:

```text
figure_redraw_package_v2/figures/main/
figure_redraw_package_v2/figures/si/
```

The package also contains:

```text
figure_redraw_package_v2/figures/figure_manifest.csv
figure_redraw_package_v2/figures/QC_REPORT.txt
```

The manifest links figure outputs to source CSV files. The QC report records figure-generation checks.

---

## Large-file note

One SI source-data file may be large:

```text
figure_redraw_package_v2/data/si/figure_data_csv/si_v8_target_distributions_clear_labels_plot_data.csv
```

If GitHub warns about file size, use Git LFS:

```bash
git lfs install
git lfs track "figure_redraw_package_v2/data/si/figure_data_csv/si_v8_target_distributions_clear_labels_plot_data.csv"
git add .gitattributes
```

If `data/clean_data.csv` is large, track it with Git LFS as well:

```bash
git lfs install
git lfs track "data/clean_data.csv"
git add .gitattributes
```

Alternatively, replace very large point-level plotting CSVs with compact binned source-data files and update the plotting script accordingly.

---

## Expected outputs

The full benchmark pipeline can generate:

- fold-level metrics,
- split-family benchmark summaries,
- target-sensitivity summaries,
- screening recovery metrics,
- top-5% enrichment metrics,
- exact elite-list stability outputs,
- held-out-group error summaries,
- structural-audit candidate tables,
- source-data tables for manuscript and SI figures.

Large generated outputs should usually remain local and ignored by Git.

---

## Source-data and figure traceability

The manuscript and SI are intended to be traceable through compact source-data files.

A recommended source-data organization is:

```text
source_data/
├── main_figures/
├── main_tables/
├── si_figures/
└── si_tables/
```

The current figure-redraw package stores figure plot data under:

```text
figure_redraw_package_v2/data/main/figure_data_csv/
figure_redraw_package_v2/data/si/figure_data_csv/
```

A source-data manifest should explain how each manuscript/SI figure or table maps to its corresponding CSV file.

Recommended manifest location:

```text
docs/SOURCE_DATA_MANIFEST.md
```

---

## Structural audit cases

The manuscript includes CIF-based structural audit examples to connect split-induced high-error regimes and shortlist transitions to inspectable MOF structures.

These cases are diagnostic examples. They are not claimed to be:

- best MOFs,
- process-optimal sorbents,
- mechanistic proof that topology alone controls adsorption,
- independent validation sets.

Selected structural-audit artwork and compact source-data records may be included in the repository. The full ARC--MOF CIF database should not be redistributed unless permitted by the original data license.

---

## Manual audit protocol

The SI includes a manual audit protocol for interpreting high-error grouped-split cases.

The protocol traces:

```text
split family
→ machine-readable held-out group label
→ source-data row
→ descriptor values
→ prediction audit values
→ optional CIF/structure inspection
→ conservative chemical interpretation
```

Recommended language for manual interpretation:

- use: “illustrates,” “is consistent with,” “suggests,” “diagnostic,” “high-error regime,” “shortlist transition”;
- avoid: “proves,” “demonstrates the mechanism,” “topology controls uptake,” “best MOF,” “process-optimal sorbent.”

---

## Recommended files to track in Git

Track:

```text
README.md
LICENSE
CITATION.cff
requirements.txt
environment.yml
split_strategy_pipeline_raw_arc.py
data/README.md
data/clean_data.csv
docs/
figure_redraw_package_v2/README.md
figure_redraw_package_v2/requirements.txt
figure_redraw_package_v2/code/make_publication_figures.py
figure_redraw_package_v2/data/main/figure_data_csv/
figure_redraw_package_v2/data/si/figure_data_csv/
figure_redraw_package_v2/data/structural_audit/
figure_redraw_package_v2/figures/main/
figure_redraw_package_v2/figures/si/
figure_redraw_package_v2/figures/figure_manifest.csv
figure_redraw_package_v2/figures/QC_REPORT.txt
compact source-data CSVs
manuscript/SI LaTeX files, if desired
final manuscript/SI figure PDFs, if desired
```

Do not track:

```text
raw ARC--MOF database files
full ARC--MOF CIF archive
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
large full per-experiment prediction CSVs
trained model binaries
temporary output folders
cache folders
__pycache__/
*.pyc
logs
local notebook checkpoints
private local paths
credentials or tokens
```

---

## Recommended `.gitignore` policy

Because this repository intentionally includes `data/clean_data.csv` but excludes raw ARC--MOF source files and large generated outputs, the `.gitignore` should allow only the curated processed data file inside `data/`.

Recommended data section:

```gitignore
# Data policy
data/*
!data/
!data/README.md
!data/clean_data.csv
```

Recommended full additions:

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Local environments
.venv/
env/
*.egg-info/

# Logs and temporary outputs
*.log
*.tmp
temp/
tmp/

# Large generated benchmark outputs
paper1_split_strategy_outputs*/
outputs/
generated_outputs/
full_predictions/
models/
model_binaries/

# Raw ARC--MOF and local source files
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv

# Archives
*.zip
*.tar
*.tar.gz
*.7z

# OS/editor
.DS_Store
Thumbs.db
.vscode/
```

---

## Reproducing manuscript figures

To regenerate publication figures:

```bash
cd figure_redraw_package_v2
pip install -r requirements.txt
python code/make_publication_figures.py
```

Then use:

```text
figure_redraw_package_v2/figures/main/
figure_redraw_package_v2/figures/si/
```

for manuscript and SI figure files.

The structural-audit figure is included as finalized artwork in the package. It is not fully regenerated from code unless the corresponding CIF files, rendering settings, and composite-figure script are also provided.

---

## Data availability statement

This repository includes:

- code for the split-strategy benchmark,
- `data/clean_data.csv`, a processed ARC--MOF-derived benchmark table,
- compact source-data and plot-data files,
- regenerated manuscript and SI figures,
- figure-regeneration scripts,
- documentation for required local inputs,
- reproducibility and submission-check guidance.

This repository does not redistribute raw ARC--MOF database files, raw third-party structural databases, full ARC--MOF CIF archives, large full-prediction files, or fitted model objects. Users who need the original ARC--MOF source data should obtain them from the original data source and follow the original license and citation requirements.

Large full-prediction files and fitted models are intentionally excluded from version control and should be regenerated locally.

---

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{DiBellaAbaeiHadjiThomasAsgari2026SplitDefinesClaim,
  title   = {The Split Defines the Claim: A Validation Grammar for Chemically Auditable MOF Discovery},
  author  = {Di Bella, Angelo and Abaei, Shayan and Hadji-Thomas, Andre and Asgari, Mehrdad},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

Please also cite the ARC--MOF database:

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

Update the manuscript citation once DOI, journal, preprint, or submission details are available.

---

## License

This repository is distributed under the MIT License unless otherwise noted.

The license applies to code and repository documentation. It does not grant redistribution rights for raw third-party database files, including ARC--MOF data, unless those rights are provided by the original data source.

The included `data/clean_data.csv` is a processed benchmark table derived from ARC--MOF. Users should follow the original ARC--MOF license and citation requirements when using this derived data file.

---

## Contact

For questions about the benchmark, repository, or manuscript reproducibility package, please use the GitHub issue tracker or contact the corresponding author listed in the manuscript.

---

## Short summary

This repository supports a chemically auditable MOF ML validation benchmark. It asks whether model conclusions survive the validation regime relevant to the intended discovery claim. Random splits test interpolation; grouped holdouts test transfer to unfamiliar pore regimes, node families, linkers, motifs, or topologies. Shortlist stability and high-error groups are reported so that screening claims can be inspected chemically, not only statistically.
