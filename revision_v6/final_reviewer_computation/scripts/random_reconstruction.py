"""At most five checkpointed historical-protocol random RF fits; no search."""
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
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('ARC_MOF_SOURCE_ROOT',str(OUT.parents[1]))).resolve()
RES=OUT/'results'
CP=OUT/'checkpoints'/'random_reconstruction'
TARGET='uptake(mmol/g) CO2 at 0.15 bar'

def top_ids(frame, score):
    k=int(np.ceil(.05*len(frame)))
    return set(frame.sort_values([score,'filename_norm'],ascending=[False,True],kind='mergesort').head(k).filename_norm),k

def main():
    CP.mkdir(parents=True,exist_ok=True)
    RES.mkdir(parents=True,exist_ok=True)
    features=json.loads((ROOT/'results/global_processed/descriptor_families.json').read_text())['enriched_interpretable']
    df=pd.read_csv(ROOT/'results/global_processed/assembled_master_table.csv',usecols=['filename_norm',TARGET,*features],low_memory=False)
    df=df[df[TARGET].notna()].reset_index(drop=True)
    assert len(df)==263744 and df.filename_norm.is_unique
    archive=pd.read_csv(ROOT/'historical_oos/co2_015bar__enriched_interpretable__random__rf__aggregated_oos.csv.gz')
    assert archive.filename_norm.is_unique
    tokens=archive.folds_or_repeats_seen.astype(str).str.split(',')
    archive_sets={fold:set(archive.loc[tokens.map(lambda t:str(fold) in t),'filename_norm']) for fold in range(5)}
    metrics=pd.read_csv(ROOT/'results/metrics/all_fold_metrics_long.csv')
    ref=metrics[(metrics.target_slug=='co2_015bar')&(metrics.descriptor_family=='enriched_interpretable')&(metrics.split_family=='random')&(metrics.model=='rf')].set_index('fold_id')
    assert len(ref)==5
    splits=list(ShuffleSplit(n_splits=5,test_size=.2,random_state=42).split(df))
    membership=[]
    for fold,(tr,te) in enumerate(splits):
        ids=set(df.iloc[te].filename_norm)
        membership.append({'fold':fold,'n_test':len(te),'archive_test_n':len(archive_sets[fold]),'symmetric_difference_n':len(ids^archive_sets[fold])})
        if ids!=archive_sets[fold]: raise RuntimeError(f'Fold {fold} membership mismatch')
    pd.DataFrame(membership).to_csv(RES/'random_membership_reconstruction.csv',index=False)
    summaries=[]
    for fold,(tr,te) in enumerate(splits):
        cp=CP/f'fold{fold}.json'
        predfile=CP/f'fold{fold}_predictions.csv.gz'
        if cp.exists() and predfile.exists():
            result=json.loads(cp.read_text(encoding='utf-8'))
            if result.get('fold')!=fold or result.get('n_test')!=len(te): raise RuntimeError('Invalid checkpoint')
            print(f'reuse fold {fold}',flush=True)
            summaries.append(result)
            continue
        params={k.removeprefix('model__'):v for k,v in json.loads(ref.loc[fold,'best_params']).items()}
        model=Pipeline([('preprocess',ColumnTransformer([('num',SimpleImputer(strategy='median'),features)],remainder='drop',sparse_threshold=.3)),('model',RandomForestRegressor(random_state=42,n_jobs=1,bootstrap=True,max_samples=.70,max_features=.70,**params))])
        start=time.monotonic()
        model.fit(df.iloc[tr][features],df.iloc[tr][TARGET])
        pred=model.predict(df.iloc[te][features])
        out=pd.DataFrame({'filename_norm':df.iloc[te].filename_norm.to_numpy(),'y_true':df.iloc[te][TARGET].to_numpy(),'prediction':pred})
        out.to_csv(predfile,index=False,compression='gzip')
        old=out.merge(archive[['filename_norm','y_true','y_pred_mean','n_oos_predictions']],on='filename_norm',validate='one_to_one')
        if not np.allclose(old.y_true_x,old.y_true_y,atol=1e-10):raise RuntimeError('Target mismatch')
        diff=np.abs(old.prediction-old.y_pred_mean)
        singleton=old.n_oos_predictions==1
        result={'fold':fold,'n_train':len(tr),'n_test':len(te),'seed':42,'params':params,'features':features,'elapsed_seconds':time.monotonic()-start,'singleton_n':int(singleton.sum()),'singleton_prediction_mae_vs_archive':float(diff[singleton].mean()),'singleton_prediction_max_abs_diff':float(diff[singleton].max()),'singleton_fraction_within_1e_6':float((diff[singleton]<1e-6).mean())}
        cp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        summaries.append(result)
        print(f'fit fold {fold}: {result["elapsed_seconds"]:.1f}s, singleton MAE difference {result["singleton_prediction_mae_vs_archive"]:.6f}',flush=True)
    pd.DataFrame(summaries).drop(columns=['features','params']).to_csv(RES/'random_reconstruction_fit_summary.csv',index=False)
    # Reaggregate only the five NEW runs. Never mix new per-run predictions with
    # the historical mean when prediction-level reconstruction is not exact.
    allruns=[]
    for fold in range(5):
        x=pd.read_csv(CP/f'fold{fold}_predictions.csv.gz');x['fold']=fold;allruns.append(x)
    concat=pd.concat(allruns,ignore_index=True)
    agg=concat.groupby('filename_norm',as_index=False).agg(new_mean=('prediction','mean'),n=('prediction','size'),y_true=('y_true','first'))
    check=agg.merge(archive,on='filename_norm',validate='one_to_one')
    assert len(check)==len(archive) and (check.n==check.n_oos_predictions).all() and np.allclose(check.y_true_x,check.y_true_y)
    pd.DataFrame([{'n':len(check),'mean_absolute_prediction_difference':float(np.abs(check.new_mean-check.y_pred_mean).mean()),'max_absolute_prediction_difference':float(np.abs(check.new_mean-check.y_pred_mean).max()),'fraction_within_1e_6':float((np.abs(check.new_mean-check.y_pred_mean)<1e-6).mean()),'fits':5,'total_fit_seconds':sum(x['elapsed_seconds'] for x in summaries)}]).to_csv(RES/'random_aggregate_reconstruction.csv',index=False)
    # Each comparison uses the SAME OOS structure population and top-k denominator.
    compare=[]
    for fold in range(5):
        run=concat.loc[concat.fold==fold].copy()
        shared=run.merge(agg[['filename_norm','new_mean','n']],on='filename_norm',validate='one_to_one')
        for name,sub in [('all_run_oos',shared),('multi_oos_only',shared[shared.n>=2].copy())]:
            if len(sub)<20:continue
            this,k=top_ids(sub,'prediction'); avg,_=top_ids(sub,'new_mean'); truth,_=top_ids(sub,'y_true')
            compare.append({'fold':fold,'population':name,'n':len(sub),'k':k,'singletons':int((sub.n==1).sum()),'run_vs_new_average_jaccard':len(this&avg)/len(this|avg),'run_reference_recovery':len(this&truth)/k,'new_average_reference_recovery':len(avg&truth)/k,'list_symmetric_difference_n':len(this^avg)})
    pd.DataFrame(compare).to_csv(RES/'random_run_vs_average_rankings.csv',index=False)
    # One actual OOS prediction per MOF on the same 177,411-ID eligible cohort.
    # This is a deterministic sampling sensitivity, not five independent test sets.
    assert (concat.groupby('filename_norm').y_true.nunique()==1).all()
    global_frame=agg.rename(columns={'new_mean':'average_prediction'})
    avg_ids,k=top_ids(global_frame,'average_prediction')
    true_ids,_=top_ids(global_frame,'y_true')
    sampled=[]
    for seed in range(20260,20265):
        rng=np.random.default_rng(seed)
        chosen=concat.assign(draw=rng.random(len(concat))).loc[lambda z:z.groupby('filename_norm').draw.idxmin()]
        one=chosen[['filename_norm','prediction']].merge(global_frame,on='filename_norm',validate='one_to_one')
        assert len(one)==len(global_frame) and one.filename_norm.is_unique
        ids,_=top_ids(one,'prediction')
        sampled.append({'seed':seed,'n':len(one),'k':k,'singletons':int((agg.n==1).sum()),
            'one_vs_average_jaccard':len(ids&avg_ids)/len(ids|avg_ids),
            'one_reference_recovery':len(ids&true_ids)/k,
            'average_reference_recovery':len(avg_ids&true_ids)/k,
            'list_symmetric_difference_n':len(ids^avg_ids)})
    pd.DataFrame(sampled).to_csv(RES/'random_one_oos_sampling_sensitivity.csv',index=False)

if __name__=='__main__':main()
