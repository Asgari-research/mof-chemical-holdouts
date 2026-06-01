# Split Strategy manuscript figure-redraw package

This package contains a clean Python plotting script, the supplied plot-data CSV files, and regenerated manuscript-ready figures in both PNG and PDF format.

## Contents

- `code/make_publication_figures.py` — complete Python script used to regenerate the figures.
- `data/main/figure_data_csv/` — source CSV files for main Figures 1–7.
- `data/si/figure_data_csv/` — source CSV files for the SI figures.
- `data/structural_audit/` — finalized structural-audit artwork copied from the manuscript package.
- `figures/main/` — regenerated main-text figures, using manuscript-compatible filenames.
- `figures/si/` — regenerated SI figures, using manuscript-compatible filenames.
- `figures/figure_manifest.csv` — file-by-file manifest linking outputs to source CSVs.
- `figures/QC_REPORT.txt` — short quality-control note.

## How to run

From the package root:

```bash
python code/make_publication_figures.py
```

For a clean conda environment:

```bash
conda create -n splitfigs python=3.11 -y
conda activate splitfigs
pip install -r requirements.txt
python code/make_publication_figures.py
```

The default script writes outputs to `figures/main` and `figures/si`. To render one figure only:

```bash
python code/make_publication_figures.py --only figure3
python code/make_publication_figures.py --only si_prediction_scatter
```

## Notes

The analytical figures are redrawn directly from the supplied plot-data CSVs. The structural-audit figure in the manuscript is preserved as the existing PDF because the archive contained the finalized artwork but not the underlying CIF/rendering assets needed for a faithful programmatic redraw.

The generated filenames are kept compatible with the current LaTeX package, so the folders `figures/main` and `figures/si` can be copied over the corresponding manuscript folders.
