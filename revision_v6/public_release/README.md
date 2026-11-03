# V6 public scientific release (Advanced Science major revision)

**Study:** *The Validation Split Defines the Discovery Claim in AI-Guided MOF Screening*  
**Status:** manuscript revision source and compact reproducibility evidence, **not a journal acceptance notice**.  
**Source basis:** Coauthor handoff (2026-10-21), publicly accessible computational V6 branch at Git commit `ff83785a76f14fb18f2dac455105a892bf29452a`, and locked final figure set (2026-11-03).

## What to use

| Need | File or location |
|---|---|
| Read revised scientific text | `manuscript/v6/Main_Manuscript_Advanced_Science_V6.tex` |
| Read Supporting Information | `manuscript/v6/Supporting_Information_Advanced_Science_V6.tex` |
| Current approved vector figure PDFs | `manuscript/v6/figures/` |
| Check frozen figure integrity | `python revision_v6/public_release/verify_public_release.py` |
| Audit table of 22 figure sources | `revision_v6/public_release/FIGURE_SOURCES.md` |
| Locate computation/results | `revision_v6/README.md`, `revision_v6/scripts/` and `revision_v6/final_reviewer_computation/` (existing branch) |
| Historical original plots | `figure_redraw_package_v2/` (existing branch) |
| Processed benchmark data | `data/clean_data.zip` (existing branch) |

## Minimal verification (Python standard library only)

From the repository root:

```bash
python revision_v6/public_release/verify_public_release.py
```

This command independently checks: (i) the 22 approved figure checksums, (ii) all current manuscript/SI figure include targets, (iii) aggregate arithmetic for the five-fold matched random/topology control, and (iv) the stored top-5%-shortlist Jaccard counts. A passing report is **not** evidence of fresh RF refitting or re-running original GCMC calculations.

For independent plotting previews (requires pandas, numpy, matplotlib):

```bash
python -m pip install numpy pandas matplotlib
python manuscript/v6/plotting/replot_v6_controls.py
```

Outputs are written to `manuscript/v6/generated_previews/`; the approved PDFs remain unchanged. The saved fold metrics are sources for plot recreation, not full training inputs.

## Reproducibility levels

1. **Exact integrity of distributed artwork:** SHA-256 verification for all 22 PDFs; available without external data.
2. **Exact checks of compact reported aggregates:** five-fold controls and shortlist Jaccard arithmetic from supplied CSV/JSON; available without fitting models.
3. **Historical figure reproduction:** many exported plot-data CSVs and plotting scripts exist, but some published layouts and rare inputs are not fully recoverable from this release.
4. **Original computational reproduction:** a fresh full benchmark run requires authorized access to the original processed data, all grouping label inputs, implementation environment and historic training/prediction inputs. These are **not all provided** by this compact public release. The repository must not be described as proving a byte-identical reproduction of every original experiment.

The original ARC–MOF source should be cited and obtained in accordance with its rights and dataset documentation. See the repository's `README.md` and `docs/DATA_ACCESS.md`.

## Source code and paper usage

- The V6 manuscript and SI share sources with clean/marked TeX wrappers. The reviewer-response letter and editorial correspondence remain in the private coauthor handoff and are **not** part of the public release.
- The user-approved figures in `manuscript/v6/figures/` are fixed. The complete SHA-256 manifest is provided. All other source-code examples should output to new preview/output folders.
- **Important unresolved reader warning:** *Main Figure 6* labels imported geometry IDs as “Pore-regime” even though the physical clustering thresholds were not independently verified. These IDs should not be interpreted as validated physical pore regimes. See `FIGURE_SOURCES.md` before using them in new science.
- The previous V4 release, `manuscript/v4/`, `revision_v4/`, and root data/code are preserved.

## Citation, licensing and support

Cite the associated manuscript, the original ARC–MOF dataset/data paper, and the software as directed by `CITATION.cff`. An MIT software license does not override any license or reuse conditions applying to source datasets or third-party structural illustrations. Check figure and source-data rights before further redistribution.

If reproducibility inputs are unavailable or a figure label is ambiguous, report the limitation instead of inferring missing details.
