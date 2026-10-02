#!/usr/bin/env python3
"""Leave-one-person-out, one-lag conditional ordinal-copula prediction.
This is not the full-history latent-state filter. Both dependence models use
exactly the same one previous ordinal rating; each training fold re-estimates
case margins and parameters without using any held-out-person responses.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from ordinal_dependence import PairKernel,load_data,PAIR_T,PAIR_S

BASE=Path(__file__).resolve().parent
SEED=20260906

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='upload/Corrected_Clinical_Trial_Long_Data.csv')
    parser.add_argument('--output',default='work/corrected/dependence')
    args=parser.parse_args();source=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tic=time.monotonic();data=load_data(source)
    d=pd.read_csv(source)
    ids=[sorted(d.loc[d.CaseSeries==g,'ID'].unique()) for g in ['A','B']]
    adjacent=np.where(PAIR_S-PAIR_T==1)[0]
    pred=[];folds=[]
    for group in range(2):
        for person in range(34):
            training=[data[g] if g!=group else np.delete(data[g],person,axis=0) for g in range(2)]
            kernel=PairKernel(training)
            dynamic=kernel.fit(starts=((.44,.39),))
            static=kernel.fit(starts=((.47,0.),),static=True)
            d_joint=kernel.evaluate(dynamic['omega'],dynamic['phi'],gradients=False)[1]
            s_joint=kernel.evaluate(static['omega'],0.,gradients=False)[1]
            fold={'ID':int(ids[group][person]),'CaseSeries':['A','B'][group],'omega_dynamic':dynamic['omega'],'phi_dynamic':dynamic['phi'],'omega_static':static['omega'],'dynamic_converged':dynamic['optimizer_success'],'static_converged':static['optimizer_success']}
            folds.append(fold)
            for t in range(1,21):
                prev=int(data[group,person,t-1]);target=int(data[group,person,t])
                q=group*210+adjacent[t-1]
                denom=kernel.marginal_prob[group,t-1,prev]
                probs={'case_margin':kernel.marginal_prob[group,t], 'static_one_lag':s_joint[q,prev,:]/denom,'dynamic_one_lag':d_joint[q,prev,:]/denom}
                for model,p in probs.items():
                    assert np.min(p)>0 and abs(np.sum(p)-1)<1e-10
                    ll=-np.log(p[target])
                    rps=np.mean((np.cumsum(p)[:6]-(target<=np.arange(6)))**2)
                    pred.append({'ID':fold['ID'],'CaseSeries':fold['CaseSeries'],'Trial':t+1,'observed_category':target+1,'previous_category':prev+1,'model':model,'log_loss':ll,'mean_normalized_RPS':rps,**{f'p{k+1}':float(p[k]) for k in range(7)}})
            if len(folds)%10==0: print('LOPO',len(folds),'elapsed',round(time.monotonic()-tic,1),flush=True)
    frame=pd.DataFrame(pred);frame.to_csv(out/'copula_lopo_predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'copula_lopo_fits.csv',index=False)
    people=frame.groupby(['ID','CaseSeries','model'])[['log_loss','mean_normalized_RPS']].mean().reset_index()
    people.to_csv(out/'copula_lopo_participant_scores.csv',index=False)
    metrics=frame.groupby('model')[['log_loss','mean_normalized_RPS']].mean().reset_index().to_dict('records')
    rng=np.random.default_rng(SEED);comparisons=[]
    for metric in ['log_loss','mean_normalized_RPS']:
        wide=people.pivot(index=['CaseSeries','ID'],columns='model',values=metric)
        for comparator in ['case_margin','static_one_lag']:
            delta=wide[comparator]-wide['dynamic_one_lag']
            ga=delta.loc['A'].to_numpy();gb=delta.loc['B'].to_numpy()
            boot=(ga[rng.integers(0,34,(10000,34))].mean(axis=1)+gb[rng.integers(0,34,(10000,34))].mean(axis=1))/2
            comparisons.append({'metric':metric,'comparator':comparator,'reference':'dynamic_one_lag','difference_definition':'comparator loss minus dynamic one-lag loss; positive favors dynamic','mean_difference':float(delta.mean()),'percentile_95_interval':[float(x) for x in np.quantile(boot,[.025,.975])],'bootstrap_replicates':10000,'bootstrap_unit':'participant, stratified by CaseSeries','interval_scope':'conditional on the fitted out-of-fold predictions; no model refitting in this score bootstrap'})
    result={'participants':68,'predicted_trials_per_participant':20,'predictions_per_model':1360,'models':metrics,'paired_comparisons':comparisons,'all_training_optimizers_converged':all(x['dynamic_converged'] and x['static_converged'] for x in folds),'mean_normalized_RPS_definition':'mean of six squared cumulative-category errors; range 0 to 1, lower is better','fold_rule':'Each entire participant excluded from margins and dependence parameter estimation; only their immediately preceding observed rating used at prediction.','target_scope':'New participants on the same two observed fixed case sequences.','important_limit':'One-lag conditional forecasts, not full-history latent filtering; static-person comparator also uses only one prior rating.','elapsed_seconds':time.monotonic()-tic}
    (out/'copula_lopo_results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
