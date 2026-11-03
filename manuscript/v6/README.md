# V6 manuscript + Supporting Information source

The LaTeX source in this folder corresponds to the Advanced Science revision handoff of October 21, 2026 with **only the Figure 1 include path updated to the 2026-11-03 approved vector PDF**. Other scientific wording, captions, references, and SI content have not yet been re-edited as part of the GitHub packaging step.

## Entry points

- `Main_Manuscript_Advanced_Science_V6.tex`: highlighted main version
- `Main_Manuscript_Advanced_Science_V6_Clean.tex`: clean wrapper
- `Supporting_Information_Advanced_Science_V6.tex`: highlighted SI
- `Supporting_Information_Advanced_Science_V6_Clean.tex`: clean SI wrapper
- `references.bib`: manuscript bibliography
- `figures/`: exactly 22 approved one-page PDF figures. **Do not edit or re-export.**
- `figure_sources/Figure1_Main_TikZ.tex`: editable source for current Figure 1
- `plotting/replot_v6_controls.py`: preview-only generation of stored control figures.

## Compilation

Use a reasonably complete TeX Live/MiKTeX installation with pdfLaTeX and BibTeX. The original coauthor handoff builds the clean and marked variants as separate main files. A normal local sequence is:

```bash
cd manuscript/v6
pdflatex -interaction=nonstopmode -halt-on-error Main_Manuscript_Advanced_Science_V6_Clean.tex
bibtex Main_Manuscript_Advanced_Science_V6_Clean
pdflatex -interaction=nonstopmode -halt-on-error Main_Manuscript_Advanced_Science_V6_Clean.tex
pdflatex -interaction=nonstopmode -halt-on-error Main_Manuscript_Advanced_Science_V6_Clean.tex
```

Check the exact project-specific BibTeX job names and whether the SI uses a bibliography when building on your machine. **This public handoff has not been certified as a new, final PDF compilation**; the last coauthor-generated PDFs were produced before the new figures were locked. Rebuild and QA all output PDFs during the Overleaf phase.

## Critical scientific caveat

The frozen *Main Figure 6* displays `Pore-regime` for imported geometry IDs; V6 Methods and handoff notes make clear that the upstream geometry grouping thresholds could not be recovered. This is a labeling discrepancy, not evidence that the numeric MAE data are wrong. The figures remain untouched; the author/supervisor must approve the final textual explanation before submission.

## Files intentionally absent

The point-by-point reviewer response, original reviewer reports, editor correspondence, internal coauthor guide, and compiled submission PDFs are not part of this *public* scientific repository. They are retained separately for the private journal revision workflow.
