# V4 validation and reproducibility evidence

This directory contains the bounded validation and reproducibility analyses supporting **The Validation Split Defines the Discovery Claim in AI-Guided MOF Screening**.

## What is included

- `generalization/`: final-estimator RF refits using recovered historical outer memberships and saved fold-specific RF settings, with fold-level metrics, summaries, and reproduction diagnostics.
- `common_universe/`: fixed-universe sensitivity analysis on 214,531 label-complete structures across all six split strategies.
- `group_audit/`: compact group-construction summaries, eligibility counts, and the topology rare-category rule.
- `historical_oos/`: compact outer-OOS summary tables used to verify exact shortlist comparisons.
- `preflight/`: historical-versus-recomputed checks and revision-environment metadata.
- `scripts_historical/`: scripts used to generate and audit the V4 validation analyses.

## Provenance limitations

This is a portable reproducibility-evidence package, not a claim of bit-for-bit historical end-to-end reproduction. The original run logged Python 3.11.14, but exact historical non-Python package versions were not recovered. The V4 analysis environment is recorded separately. The complete historical fitted-model and prediction archives are not distributed in the normal Git history.

One enriched/topology held-out fold differs slightly under the recovered-current refit environment (largest absolute held-out R2 difference 0.011318; corresponding MAE difference 0.000126 mmol g^-1). No retuning was performed to improve agreement.

The historically executed engineered descriptor `cavity_window_gap` is preserved exactly as `Dif`; it duplicates an existing raw input and is not retrospectively reinterpreted as `Di-Df`.

## Relationship to the root pipeline

The archived V4 workspace contained `split_strategy_pipeline_raw_arc_V2.py`, but its Git object is byte-identical to the existing repository-root `split_strategy_pipeline_raw_arc.py`. It is therefore intentionally not duplicated here.

The scripts under `scripts_historical/` retain assumptions from the V4 analysis workspace and are provided for provenance and auditability. They should not be described as a one-command clean-room reproduction unless their documented input layout is reconstructed.
