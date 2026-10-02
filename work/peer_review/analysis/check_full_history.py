#!/usr/bin/env python3
"""Independent static integration, grid stability and strict-prefix checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtri,roots_hermitenorm
from full_history import ROOT,load_data,margins,forecast,normal_interval
OUT=Path(__file__).resolve().parent

def static_direct(y,p,eta,nodes=601):
    x,w=roots_hermitenorm(nodes);b=x*np.sqrt(eta);w=w/np.sqrt(2*np.pi)
    edges=np.column_stack([np.full(21,-np.inf),ndtri(np.cumsum(p,axis=1)[:,:6]),np.full(21,np.inf)])
    sd=np.sqrt(1-eta);out=[]
    for t in range(21):
        cat=normal_interval((edges[t,:-1,None]-b)/sd,(edges[t,1:,None]-b)/sd)
        if t:out.append(cat@w)
        w=w*cat[y[t]];w=w/w.sum()
    return np.array(out)

def main():
    low=pd.read_csv(OUT/'full_history_241_64.csv');high=pd.read_csv(OUT/'full_history_481_64.csv');prefixbase=pd.read_csv(OUT/'full_history_61_32.csv')
    pcols=[f'p{k}' for k in range(1,8)]
    result={'grid_comparison':{'maximum_probability_difference':float(abs(low[pcols].to_numpy()-high[pcols].to_numpy()).max()),'maximum_trial_logloss_difference':float(abs(low.log_loss-high.log_loss).max()),'maximum_mean_score_difference':float(abs(low.groupby('model').log_loss.mean()-high.groupby('model').log_loss.mean()).max())}}
    data=load_data(ROOT/'upload/Corrected_Clinical_Trial_Long_Data.csv');d=pd.read_csv(ROOT/'upload/Corrected_Clinical_Trial_Long_Data.csv');fits=pd.read_csv(ROOT/'work/corrected/dependence/copula_lopo_fits.csv').set_index('ID')
    prefix_error=0.;static_error=0.;checks=0
    for g,series in enumerate(['A','B']):
        for i,ident in enumerate(sorted(d.loc[d.CaseSeries==series,'ID'].unique())):
            train=[data[h] if h!=g else np.delete(data[h],i,axis=0) for h in range(2)];p=margins(train)[g];y=data[g,i];fit=fits.loc[ident]
            ref=high[(high.ID==ident)&(high.model=='static_full_history')].sort_values('Trial')[pcols].to_numpy()
            static_error=max(static_error,float(abs(ref-static_direct(y,p,fit.omega_static)).max()))
            dynamic=prefixbase[(prefixbase.ID==ident)&(prefixbase.model=='dynamic_full_history')].sort_values('Trial')[pcols].to_numpy()
            for t in range(1,21):
                alt=y.copy();alt[t:]=(alt[t:]+3)%7
                pred,_=forecast(alt,p,fit.omega_dynamic,fit.phi_dynamic,61,32)
                prefix_error=max(prefix_error,float(abs(pred[:t]-dynamic[:t]).max()));checks+=1
    result.update({'independent_static_max_probability_difference':static_error,'strict_prefix_checks':checks,'maximum_probability_change_before_perturbed_outcomes':prefix_error})
    assert result['grid_comparison']['maximum_probability_difference']<1e-7
    assert static_error<1e-7 and prefix_error<1e-12
    (OUT/'full_history_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
