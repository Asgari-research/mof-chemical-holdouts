# MOF Chemical Holdouts Benchmark

Reproducibility package for the manuscript:

**The Split Defines the Claim: A Validation Grammar for Chemically Auditable MOF Discovery**

This repository supports a chemically auditable split-strategy benchmark for machine-learning-guided metal--organic framework (MOF) adsorption screening. The workflow compares random train/test splits with chemically and structurally grouped holdouts while keeping targets, descriptor families, model classes, hyperparameter logic, and metrics matched.

The central purpose is not to identify a universally best model or descriptor family. The purpose is to test how the scientific meaning of a MOF ML screening claim changes when validation shifts from interpolation-like random splitting to deployment-relevant grouped extrapolation across geometry, metal-cluster, functional-group, linker, and topology families.

In this project, the validation split is treated as a scientific variable: **the split defines the claim**.

---

## Repository overview

This repository contains code, documentation, compact source-data files, and figure-regeneration materials for the split-strategy MOF adsorption benchmark.

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
├── docs/
├── figure_redraw_package_v2/
```

The repository is designed to support three levels of reproducibility:

1. **Benchmark rerun** from locally supplied ARC--MOF-derived input files.
2. **Exact post-processing audit** from regenerated prediction outputs.
3. **Publication figure regeneration** from compact plot-data CSV files in `figure_redraw_package_v2/`.

---

## What this repository contains

### Main pipeline

* `split_strategy_pipeline_raw_arc.py`
  Main Python pipeline used to assemble the benchmark table, run split-strategy ML experiments, generate metrics, and perform exact post-processing.

The pipeline supports matched comparison across:

* random splitting,
* geometry-grouped splitting,
* metal-cluster-grouped splitting,
* functional-group-grouped splitting,
* linker-grouped splitting,
* topology-grouped splitting.

The main benchmark targets are:

* CO2 uptake at 0.015 bar,
* CO2 uptake at 0.15 bar,
* CH4 uptake at 5.8 bar,
* CH4 uptake at 65 bar.

The manuscript uses **CO2 uptake at 0.15 bar** as the anchor adsorption target. This target is used as an adsorption-relevant validation target, not as a complete process-optimality metric.

---

### Documentation

* `docs/`
  Documentation for data access, expected local file layout, output interpretation, source-data organization, GitHub/release preparation, and submission checks.

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

### Figure-redraw package

* `figure_redraw_package_v2/`
  Publication-facing figure-regeneration package containing the cleaned plotting script, plot-data CSV files, regenerated main/SI figures, a figure manifest, and a QC report.

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

The analytical main-text and SI figures are redrawn from supplied plot-data CSV files. The structural-audit figure is preserved as finalized manuscript artwork unless the underlying CIF/rendering assets are also supplied.

---

## Scientific scope

This repository supports a validation-regime benchmark, not a complete materials-selection or process-optimization study.

The manuscript asks:

> When does a MOF ML validation score preserve the chemical meaning of the intended screening decision?

A random split asks whether a model interpolates within a familiar database neighbourhood. Grouped splits ask different chemical transfer questions:

| Split family             | Chemistry-facing question                                                                  | Interpretation                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Random                   | Can the model interpolate within the same database neighbourhood?                          | Baseline interpolation regime. Useful, but insufficient by itself for discovery claims involving unfamiliar chemistry.           |
| Geometry-grouped         | Can the model transfer across pore-size or pore-shape regimes?                             | Pore-regime holdout, interpreted through PLD, LCD, density, void fraction, accessible surface area, and pore-volume descriptors. |
| Metal-cluster-grouped    | Can the model transfer to unfamiliar node chemistry or coordination environments?          | Node-family holdout. Relevant to metal identity, local coordination, charge distribution, and node environment.                  |
| Functional-group-grouped | Can the model transfer to unfamiliar local chemical motifs?                                | Local-motif holdout. Relevant to polar groups, heteroatoms, electrostatics, and low-pressure adsorption environments.            |
| Linker-grouped           | Can the model transfer to unfamiliar organic scaffold families?                            | Linker-family holdout, analogous in spirit to scaffold splits in molecular ML but adapted to framework solids.                   |
| Topology-grouped         | Can the model transfer to unfamiliar framework connectivity and pore-network architecture? | Severe structural-extrapolation holdout. This is not a claim that topology alone controls uptake.                                |

The study reports regression metrics, rank metrics, top-5% screening metrics, exact elite-list stability, held-out-group diagnostics, and structural audit cases.

---

## What this repository does not contain

This repository intentionally does **not** redistribute:

* raw ARC--MOF database files,
* raw third-party MOF structural databases,
* full local working tables,
* large full per-experiment prediction CSVs,
* trained model binaries,
* local cache folders,
* temporary logs,
* private local paths,
* complete ARC--MOF CIF archives.

Users must obtain ARC--MOF source data from the original providers and comply with the original license and citation requirements.

Raw third-party database files are intentionally excluded so that this repository remains a compact reproducibility and source-data package rather than a redistribution of ARC--MOF.

---

## Required local input files

To rerun the full benchmark, place the required local input files next to `split_strategy_pipeline_raw_arc.py`, or edit the data path inside the script.

Required local inputs:

```text
clean_data.csv
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

If `clean_data.csv` is not already available, the pipeline can build it from ARC--MOF-derived raw files, including:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
```

The exact required file names and expected layout should also be documented in `docs/DATA_ACCESS.md`.

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

After placing the required local input files in the expected location:

```bash
python split_strategy_pipeline_raw_arc.py
```

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

## Figure-redraw package contents

The expected figure-redraw package includes:

```text
figure_redraw_package_v2/
├── README.md
├── requirements.txt
├── code/
│   └── make_publication_figures.py
├── data/
│   ├── main/
│   │   └── figure_data_csv/
│   │       ├── figure1_v8_polished_workflow_plot_data.csv
│   │       ├── figure2_v8_split_family_severity_polished_plot_data.csv
│   │       ├── figure3_v8_target_split_sensitivity_polished_plot_data.csv
│   │       ├── figure4_v8_screening_consequences_polished_plot_data.csv
│   │       ├── figure5_v9_hard_group_lollipop_polished_plot_data.csv
│   │       ├── figure6_v8_shift_and_descriptor_space_polished_plot_data.csv
│   │       └── figure7_v8_exact_elite_instability_polished_plot_data.csv
│   ├── si/
│   │   └── figure_data_csv/
│   │       ├── si_v7_group_size_summary_logscale_plot_data.csv
│   │       ├── si_v8_target_distributions_clear_labels_plot_data.csv
│   │       ├── si_v8_prediction_scatter_panels_clear_labels_plot_data.csv
│   │       ├── si_v8_pld_decile_errors_clear_labels_plot_data.csv
│   │       ├── si_heatmap_r2_plot_data.csv
│   │       ├── si_heatmap_mae_plot_data.csv
│   │       ├── si_heatmap_spearman_rho_plot_data.csv
│   │       ├── si_heatmap_kendall_tau_plot_data.csv
│   │       ├── si_heatmap_top_5pct_overlap_plot_data.csv
│   │       ├── si_heatmap_top_5pct_enrichment_plot_data.csv
│   │       ├── si_v7_error_vs_screening_rank_clean_plot_data.csv
│   │       └── si_variance_decomposition_plot_data.csv
│   └── structural_audit/
│       └── Figure_main_structural_audit_grouped_extrapolation_v7_final.pdf
└── figures/
    ├── main/
    ├── si/
    ├── figure_manifest.csv
    └── QC_REPORT.txt
```

Notes:

* Analytical figures are regenerated from CSV files.
* The structural-audit figure is included as finalized artwork unless the underlying CIF/rendering workflow is also provided.
* Large point-level SI CSV files may require Git LFS or replacement with compact binned source-data files.

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

Alternatively, replace the raw point-level plotting CSV with a compact binned histogram source-data CSV and update the plotting script accordingly.

---

## Expected outputs

The full benchmark pipeline can generate:

* fold-level metrics,
* split-family benchmark summaries,
* target-sensitivity summaries,
* screening recovery metrics,
* top-5% enrichment metrics,
* exact elite-list stability outputs,
* held-out-group error summaries,
* structural-audit candidate tables,
* source-data tables for manuscript and SI figures.

The exact output folder name may depend on the pipeline version. Large generated outputs should usually remain local and ignored by Git.

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

* best MOFs,
* process-optimal sorbents,
* mechanistic proof that topology alone controls adsorption,
* independent validation sets.

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

* use: “illustrates,” “is consistent with,” “suggests,” “diagnostic,” “high-error regime,” “shortlist transition”;
* avoid: “proves,” “demonstrates the mechanism,” “topology controls uptake,” “best MOF,” “process-optimal sorbent.”

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
local full working tables
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

## Recommended `.gitignore` additions

The repository should ignore raw and generated local data:

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

# Raw/local data
data/raw/
data/local/
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
geometric_properties.csv
all_topology_lists.csv
post_comb_vsa-CO2.csv
methane.csv

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

If a required release artifact is a zip archive, document why it is included. In general, prefer unpacked folders over zip files for source-data and reproducibility materials.

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

This repository does not redistribute raw ARC--MOF database files. Users should obtain ARC--MOF from the original data source and follow the original license and citation requirements.

The repository provides:

* code for the split-strategy benchmark,
* compact source-data and plot-data files,
* figure-regeneration scripts,
* documentation for required local inputs,
* reproducibility and submission-check guidance.

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
@article{Burner2023ARCMOF,
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

---

## Contact

For questions about the benchmark, repository, or manuscript reproducibility package, please use the GitHub issue tracker or contact the corresponding author listed in the manuscript.

---

## Short summary

This repository supports a chemically auditable MOF ML validation benchmark. It asks whether model conclusions survive the validation regime relevant to the intended discovery claim. Random splits test interpolation; grouped holdouts test transfer to unfamiliar pore regimes, node families, linkers, motifs, or topologies. Shortlist stability and high-error groups are reported so that screening claims can be inspected chemically, not only statistically.
