# V4 reproducibility note

The V4 release adds targeted validation and reproducibility evidence without introducing new targets, split families, descriptor families, or model families.

## Added validation analyses

1. **Training–held-out generalization gaps**: 60 RF final-estimator refits (2 descriptor families × 6 split families × 5 outer folds/repeats) using recovered historical outer memberships and saved fold-specific RF settings, with no retuning.
2. **Fixed-universe sensitivity**: 30 enriched-RF experiments (6 split families × 5 folds/repeats) on a common 214,531-structure label-complete universe. Singleton groups are deliberately retained only in this sensitivity analysis so that the candidate universe remains fixed.
3. **Group-construction audit**: labelled/eligible counts, raw/eligible group counts, singleton treatment, missing labels, and the topology rare-category rule.
4. **Exact OOS shortlist audit**: recovered outer-OOS predictions are aggregated by structure before pairwise intersection and ranking; elite sets are not unioned or voted.

## Key verification values

- Enriched RF, topology grouping: mean train R2 ≈ 0.855; held-out R2 ≈ 0.265; delta R2 ≈ 0.590.
- Fixed universe: random mean R2 ≈ 0.764; topology mean R2 ≈ 0.227.
- Historical exact random-vs-topology comparison: common N = 177,411; Spearman = 0.917090; top-5% Jaccard = 0.199675; top-1% Jaccard = 0.066066.

## Environment caveat

Historical Python 3.11.14 was recorded. Exact historical NumPy/pandas/scikit-learn versions were not recovered. The V4 analysis environment is stored in `revision_v4/preflight/revision_environment.json`; it must not be presented as the historical environment.

## Processed input table

The public repository stores the processed benchmark table in compressed form as `data/clean_data.zip`. Extract it to obtain `clean_data.csv` when a workflow requires the uncompressed file.
