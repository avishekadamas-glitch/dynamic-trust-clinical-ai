#!/usr/bin/env python3
"""Recompute all probability scores and use a common 34/34 cluster bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
SEED=202609063

def main():
    copula=pd.read_csv(ROOT/'work/corrected/dependence/copula_lopo_predictions.csv')
    full=pd.read_csv(OUT/'full_history_481_64.csv')
    multi=pd.read_csv(ROOT/'work/corrected/prediction/corrected_predictions.csv')
    multi=multi.rename(columns={'observed_rating':'observed_category','normalized_RPS':'mean_normalized_RPS',**{f'p_rating_{k}':f'p{k}' for k in range(1,8)}})
    multi['model']=['multinomial_C'+str(c)+'_'+m for c,m in zip(multi.C,multi.model)]
    frame=pd.concat([copula,full,multi],ignore_index=True)
    probs=frame[[f'p{k}' for k in range(1,8)]].to_numpy();targets=frame.observed_category.to_numpy(int)-1
    old=frame.log_loss.to_numpy().copy()
    frame['log_loss']=-np.log(probs[np.arange(len(frame)),targets])
    frame['mean_normalized_RPS']=np.mean((np.cumsum(probs,axis=1)[:,:6]-(targets[:,None]<=np.arange(6)))**2,axis=1)
    assert np.max(abs(probs.sum(axis=1)-1))<1e-10
    people=frame.groupby(['CaseSeries','ID','model'])[['log_loss','mean_normalized_RPS']].mean().reset_index()
    people.to_csv(OUT/'all_participant_scores.csv',index=False)
    rng=np.random.default_rng(SEED);ia=rng.integers(0,34,(10000,34));ib=rng.integers(0,34,(10000,34))
    pairs=[('case_margin','dynamic_one_lag'),('static_one_lag','dynamic_one_lag'),('static_full_history','dynamic_full_history'),('static_one_lag','static_full_history'),('dynamic_one_lag','dynamic_full_history'),('dynamic_full_history','multinomial_C1.0_plus_past_distribution'),('dynamic_full_history','multinomial_C1.0_plus_latest_rating')]
    for c in [.1,1.,10.]:
        prefix=f'multinomial_C{c}_'
        pairs += [(prefix+'case_series',prefix+'plus_past_distribution'),(prefix+'plus_past_distribution',prefix+'plus_latest_rating')]
    comparisons=[]
    for metric in ['log_loss','mean_normalized_RPS']:
        wide=people.pivot(index=['CaseSeries','ID'],columns='model',values=metric)
        for comparator,reference in pairs:
            delta=wide[comparator]-wide[reference];a=delta.loc['A'].to_numpy();b=delta.loc['B'].to_numpy()
            boot=(a[ia].mean(axis=1)+b[ib].mean(axis=1))/2
            comparisons.append({'metric':metric,'comparator':comparator,'reference':reference,'mean_difference':float(delta.mean()),'interval95':np.quantile(boot,[.025,.975]).tolist(),'series_A_difference':float(a.mean()),'series_B_difference':float(b.mean())})
    result={'seed':SEED,'bootstrap_replicates':10000,'strata':'CaseSeries, fixed allocation 34/34','scope':'Conditional paired participant resampling of saved out-of-fold probabilities; no population refits, fixed cases and sequences. Positive differences favor reference.','maximum_saved_score_error':float(abs(old-frame.log_loss).max()),'maximum_probability_sum_error':float(abs(probs.sum(axis=1)-1).max()),'score_means':frame.groupby('model')[['log_loss','mean_normalized_RPS']].mean().reset_index().to_dict('records'),'comparisons':comparisons}
    (OUT/'revised_scores.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(result['score_means']).to_csv(OUT/'all_forecast_scores.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'all_paired_score_contrasts.csv',index=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
