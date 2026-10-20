# V6 computational revision

This directory contains the compact, public portion of the October 2026 Advanced Science computational revision. It preserves `revision_v4/` and `manuscript/v4/` unchanged.

## Main results

- Five deterministic random controls use exactly the train/test counts of the five CO2 0.15-bar enriched-RF topology folds. Random-minus-topology R² is positive in every pair and averages 0.4905. The mean top-5% recovery advantage is 0.2460.
- On 177,411 structures in the pairwise common OOS population, the random/topology predicted top-5% Jaccard is 0.199675. Recovery of the global true-uptake top 5% is 0.620223 for random and 0.248337 for topology.
- Removing the 3,898 rows assigned to the collapsed rare-topology `other` category changes pooled topology MAE from 0.45712 to 0.45706 mmol/g. This is an evaluation-population ablation, not a retraining experiment.
- The historical model feature `ASA` is the Å²-per-unit-cell column; `gASA` is the separate m²/g column.

Fold-level MAE remains heterogeneous: topology has lower MAE than matched random in two of five folds. The consistent findings are the paired R² and top-5% recovery advantages, and all fold results are retained in the CSV output.

## Input boundary

The public package includes compact outputs, including the 15 MB pairwise shortlist membership table, but does not duplicate the local joined master table, full historical predictions, model objects, or local fit checkpoints. The six recovered historical OOS aggregates remain external to normal Git history. Individual random-repeat predictions were unavailable, so this release does not claim repeat-level sampling sensitivity.

Set `MOF_V6_SOURCE_ROOT` to a local directory containing the full `revision/`, `historical_oos/`, and `results/` evidence tree before rerunning:

```powershell
$env:MOF_V6_SOURCE_ROOT = 'path\to\full-research-inputs'
python revision_v6/scripts/run_saved_data_analyses.py
python revision_v6/scripts/run_matched_random_control.py
python -m pytest revision_v6/tests -q
```

The matched-control script writes one validated JSON checkpoint per fold and reuses successful checkpoints. It performs five RF fits with sequential outer execution and `n_jobs=1`; it never invokes the historical 1,920-job benchmark.

## Contents

- `scripts/`: saved-data workflow and bounded matched-random control.
- `results/`: compact tables and explicit input-availability findings.
- `figures/`: PDF and PNG diagnostic figures.
- `tests/`: output invariants runnable without the private/full input tree.
- `provenance/`: current environment and SHA256 manifest.

Historical Python 3.11.14 was recorded by V4. Exact historical NumPy, pandas, and scikit-learn versions remain unknown. The V6 environment record describes only the current revision environment.
