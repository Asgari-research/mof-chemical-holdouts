# Final bounded ARC-MOF reviewer computation

This directory contains a saved-data geometry/group audit, an exact-membership five-run random-RF ranking sensitivity, and a paired topology-fold chemistry-proxy RF control. The complete ARC-MOF source tree is required to rerun it; large per-MOF outputs and model checkpoints remain local. No manuscript, SI, or response-letter source is modified.

Run in the complete local source root with the installed Python environment. In the public repository copy, replace the `V6_AdvancedScience/` prefix below with `revision_v6/` and set `ARC_MOF_SOURCE_ROOT` to the complete local source root; the public compact package does not contain all historical inputs or the frozen local V6 paired-residual table.

```text
python V6_AdvancedScience/final_reviewer_computation/scripts/saved_data.py
python V6_AdvancedScience/final_reviewer_computation/scripts/random_reconstruction.py
python V6_AdvancedScience/final_reviewer_computation/scripts/chemistry_control.py
$env:CLOSURE_TARGET='co2_0015bar'; python V6_AdvancedScience/final_reviewer_computation/scripts/chemistry_control.py
python V6_AdvancedScience/final_reviewer_computation/scripts/fit_inventory.py
python -m unittest discover -s V6_AdvancedScience/final_reviewer_computation/tests -v
```

The random and chemistry scripts resume from per-fold JSON and compressed prediction checkpoints. Do not remove checkpoints to reproduce summaries; doing so would refit models. The random script checks original fold membership exactly and uses the mean of its *new* five predictions for each run-versus-average ranking comparison. Its predictions are not asserted to be byte-identical to the historical archive. The RAC control uses six predeclared chemistry proxies, complete-case RAC coverage, original topology fold IDs, original 20-feature RF baseline, unchanged saved RF hyperparameters, and training-only median imputation. RACs are chemistry proxies, not electrostatic charges.

`results/paired_geometry_support.csv` and `checkpoints/` are local audit artifacts, excluded from public Git. Compact tables and the reviewer evidence file contain the publishable results.
