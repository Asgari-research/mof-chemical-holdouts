"""Paired, complete-case topology holdout using six predeclared RAC chemistry proxies.

The original fold assignment, RF hyperparameters and median imputation are fixed.
No model selection or test-label feature selection is performed. Checkpoints prevent
repeat fits; the target is CO2 at 0.15 bar only unless explicitly requested below.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('ARC_MOF_SOURCE_ROOT',str(OUT.parents[1]))).resolve()
RES=OUT/'results'
SLUG=os.environ.get('CLOSURE_TARGET','co2_015bar')
assert SLUG in {'co2_015bar','co2_0015bar'}
TARGET={'co2_015bar':'uptake(mmol/g) CO2 at 0.15 bar','co2_0015bar':'uptake(mmol/g) CO2 at 0.015 bar'}[SLUG]
CP=OUT/'checkpoints'/('chemistry_control' if SLUG=='co2_015bar' else f'chemistry_control_{SLUG}')
PREFIX='' if SLUG=='co2_015bar' else f'{SLUG}_'
RACS=['f-chi-0-all','f-Z-0-all','mc-chi-0-all','mc-Z-0-all','f-lig-chi-0','f-lig-Z-0']

def top_ids(frame:pd.DataFrame,col:str,k:int)->set[str]:
    return set(frame.sort_values([col,'filename_norm'],ascending=[False,True],kind='mergesort').head(k).filename_norm)

def main():
    RES.mkdir(parents=True,exist_ok=True); CP.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'results/global_processed/descriptor_families.json').read_text())['enriched_interpretable']
    assert not set(base)&set(RACS)
    m=pd.read_csv(ROOT/'results/global_processed/assembled_master_table.csv',usecols=['filename_norm','topology_label',TARGET,*base],low_memory=False)
    assert m.filename_norm.is_unique and m[TARGET].notna().all()
    r=pd.read_csv(ROOT/'data/RACs.csv',usecols=['filename',*RACS],low_memory=False)
    r['filename_norm']=r.filename.str.replace(r'\.cif$','',regex=True)
    assert r.filename_norm.is_unique and not r[RACS].isna().any().any()
    m=m.merge(r.drop(columns='filename'),on='filename_norm',how='left',validate='one_to_one',indicator=True)
    covered=m.loc[m._merge=='both',RACS]
    for i,left in enumerate(RACS):
        for right in RACS[i+1:]:
            assert not np.array_equal(covered[left].to_numpy(),covered[right].to_numpy())
    coverage=m.groupby(['topology_label','_merge'],observed=True).size().unstack(fill_value=0).reset_index()
    coverage.to_csv(RES/f'{PREFIX}rac_topology_coverage.csv',index=False)
    audit={'target_slug':SLUG,'target_column':TARGET,'master_n':len(m),'rac_unique_ids':len(r),'covered_n':int((m._merge=='both').sum()),'missing_n':int((m._merge!='both').sum()),'coverage_fraction':float((m._merge=='both').mean()),'features_baseline':base,'rac_features_predeclared':RACS,'rac_features_overlap_baseline':[], 'rac_exact_duplicate_columns':[], 'rac_nonnull_covered':bool(m.loc[m._merge=='both',RACS].notna().all().all())}
    (RES/f'{PREFIX}rac_join_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    m=m.loc[m._merge=='both'].drop(columns='_merge').reset_index(drop=True)
    archive=pd.read_csv(ROOT/'historical_oos/co2_015bar__enriched_interpretable__topology_grouped__rf__aggregated_oos.csv.gz')
    assert archive.filename_norm.is_unique and len(archive)==263744 and (archive.n_oos_predictions==1).all()
    archive_cols=['filename_norm','folds_or_repeats_seen']+(['y_true'] if SLUG=='co2_015bar' else [])
    m=m.merge(archive[archive_cols],on='filename_norm',validate='one_to_one')
    assert len(m)==audit['covered_n']
    if SLUG=='co2_015bar':
        assert np.allclose(m[TARGET],m.y_true,rtol=0,atol=1e-10)
    else:
        assert m[TARGET].notna().all()
    m['fold']=m.folds_or_repeats_seen.astype(int)
    assert set(m.fold)==set(range(5))
    metric=pd.read_csv(ROOT/'results/metrics/all_fold_metrics_long.csv')
    ref=metric[(metric.target_slug==SLUG)&(metric.descriptor_family=='enriched_interpretable')&(metric.split_family=='topology_grouped')&(metric.model=='rf')].set_index('fold_id')
    assert len(ref)==5
    for fold in range(5):
        assert int((archive.folds_or_repeats_seen.astype(int)==fold).sum())==int(ref.loc[fold,'n_test'])
    rows=[]; group_rows=[]; split_rows=[]
    for fold in range(5):
        train=m[m.fold!=fold]; test=m[m.fold==fold]
        assert len(set(train.filename_norm)&set(test.filename_norm))==0
        assert len(set(train.topology_label)&set(test.topology_label))==0
        split_rows.append({'fold':fold,'n_train':len(train),'n_test':len(test),'train_groups':sorted(train.topology_label.unique().tolist()),'test_groups':sorted(test.topology_label.unique().tolist())})
        params={k.removeprefix('model__'):v for k,v in json.loads(ref.loc[fold,'best_params']).items()}
        k=int(np.ceil(.05*len(test)))
        truth=top_ids(test,TARGET,k)
        for variant,features in [('original',base),('original_plus_rac',base+RACS)]:
            cp=CP/f'fold{fold}_{variant}.json'; predfile=CP/f'fold{fold}_{variant}_predictions.csv.gz'
            if cp.exists() and predfile.exists():
                info=json.loads(cp.read_text(encoding='utf-8'))
                assert info['fold']==fold and info['variant']==variant and info['n_train']==len(train) and info['n_test']==len(test)
                p=pd.read_csv(predfile)
                print(f'reuse {variant} fold {fold}',flush=True)
            else:
                pipe=Pipeline([('preprocess',ColumnTransformer([('num',SimpleImputer(strategy='median'),features)],remainder='drop',sparse_threshold=.3)),('model',RandomForestRegressor(random_state=42,n_jobs=1,bootstrap=True,max_samples=.70,max_features=.70,**params))])
                start=time.monotonic()
                pipe.fit(train[features],train[TARGET])
                pred=pipe.predict(test[features])
                p=pd.DataFrame({'filename_norm':test.filename_norm.to_numpy(),'y_true':test[TARGET].to_numpy(),'prediction':pred,'topology_label':test.topology_label.to_numpy()})
                p.to_csv(predfile,index=False,compression='gzip')
                info={'fold':fold,'variant':variant,'n_train':len(train),'n_test':len(test),'train_groups':sorted(train.topology_label.unique().tolist()),'test_groups':sorted(test.topology_label.unique().tolist()),'features':features,'params':params,'seed':42,'elapsed_seconds':time.monotonic()-start}
                cp.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
                print(f'fit {variant} fold {fold}: {info["elapsed_seconds"]:.1f}s',flush=True)
            assert p.filename_norm.is_unique and set(p.filename_norm)==set(test.filename_norm)
            assert np.allclose(p.set_index('filename_norm').loc[test.filename_norm,'y_true'].to_numpy(),test[TARGET].to_numpy(),rtol=0,atol=1e-10)
            selected=top_ids(p,'prediction',k)
            rows.append({'fold':fold,'variant':variant,'n_train':len(train),'n_test':len(test),'n_train_groups':train.topology_label.nunique(),'n_test_groups':test.topology_label.nunique(),'k':k,'r2':r2_score(p.y_true,p.prediction),'mae':mean_absolute_error(p.y_true,p.prediction),'top5_fold_local_recovery':len(selected&truth)/k,'elapsed_seconds':info['elapsed_seconds']})
            p['absolute_error']=np.abs(p.y_true-p.prediction)
            for label,g in p.groupby('topology_label'):
                group_rows.append({'fold':fold,'variant':variant,'topology_label':label,'n':len(g),'mae':g.absolute_error.mean(),'mean_signed_residual':(g.y_true-g.prediction).mean()})
    tab=pd.DataFrame(rows);tab.to_csv(RES/f'{PREFIX}rac_paired_fold_metrics.csv',index=False)
    family=pd.DataFrame(group_rows)
    family.to_csv(RES/f'{PREFIX}rac_paired_topology_errors.csv',index=False)
    fp=family.pivot(index=['fold','topology_label','n'],columns='variant',values='mae').reset_index()
    fp['delta_mae']=fp['original_plus_rac']-fp['original']
    fp.sort_values('n',ascending=False).to_csv(RES/f'{PREFIX}rac_paired_topology_deltas.csv',index=False)
    a=tab.pivot(index='fold',columns='variant',values=['r2','mae','top5_fold_local_recovery'])
    delta=pd.DataFrame({'fold':a.index,'delta_r2':a['r2']['original_plus_rac']-a['r2']['original'],'delta_mae':a['mae']['original_plus_rac']-a['mae']['original'],'delta_recovery':a['top5_fold_local_recovery']['original_plus_rac']-a['top5_fold_local_recovery']['original']})
    delta.to_csv(RES/f'{PREFIX}rac_paired_fold_deltas.csv',index=False)
    (RES/f'{PREFIX}rac_fold_groups.json').write_text(json.dumps(split_rows,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
