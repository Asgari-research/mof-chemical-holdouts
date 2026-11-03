# Scientific results and limitations

| Analysis | Population | Result | Interpretation |
|---|---:|---|---|
| Training-size-matched control | 5 paired folds; 263,744-row universe | mean ΔR² (random − topology) = 0.4905; positive in 5/5 | Sample count alone does not explain the topology penalty. |
| Matched top-5% recovery | same five pairs | mean Δrecovery = 0.2460; positive in 5/5 | Topology transfer reduces screening recovery beyond the size effect. |
| Matched MAE | same five pairs | mean topology − random = +0.1841 mmol/g; range −0.1187 to +0.6196 | MAE is fold-heterogeneous and must be shown fold by fold. |
| Aggregate shortlist stability | 177,411 common OOS structures | Jaccard = 0.199675 | Prediction-list agreement is low despite high global rank correlation reported in V4. |
| Global-reference top-5% recovery | 8,871 reference structures | random = 0.620223; topology = 0.248337 | Recovery is distinct from Jaccard and from mean fold-local recovery. |
| Rare-topology evaluation ablation | 263,744 vs 259,846 OOS rows | MAE 0.45712 vs 0.45706 mmol/g | The collapsed `other` rows do not drive the pooled mean; no causal retraining claim. |

Unsupported with the available archive: individual repeat-level random sensitivity, per-MOF cross-pressure residual pairing, and independent cross-database validation. The compact random archive stores the number of OOS predictions and their mean, not the separate repeat predictions.
