"""Summarize the 25 checkpointed RF fits without loading prediction rows."""
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parents[1]/'results'
random=pd.read_csv(R/'random_reconstruction_fit_summary.csv')[['fold','elapsed_seconds']].assign(target_slug='co2_015bar',study='random_protocol_sensitivity',variant='random_rf')
primary=pd.read_csv(R/'rac_paired_fold_metrics.csv')[['fold','variant','elapsed_seconds']].assign(target_slug='co2_015bar',study='paired_rac_topology')
secondary=pd.read_csv(R/'co2_0015bar_rac_paired_fold_metrics.csv')[['fold','variant','elapsed_seconds']].assign(target_slug='co2_0015bar',study='paired_rac_topology')
out=pd.concat([random,primary,secondary],ignore_index=True)[['study','target_slug','fold','variant','elapsed_seconds']]
assert len(out)==25 and len(out.drop_duplicates(['study','target_slug','fold','variant']))==25
assert out.elapsed_seconds.sum()<3*60*60
out.to_csv(R/'fit_inventory.csv',index=False)
print(f'{len(out)} fits, {out.elapsed_seconds.sum():.1f} fit-seconds')
