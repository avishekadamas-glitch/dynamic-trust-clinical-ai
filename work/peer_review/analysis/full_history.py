#!/usr/bin/env python3
"""Deterministic full-prefix Gaussian ordinal forecasts from saved LOPO fits.

Outer Gauss-Hermite integration over the stable person effect; inner
Gauss-Legendre forward integration over each observed residual interval.
No population parameter or case-margin estimate uses the held-out person's
responses. Completed ratings update only their prediction-time latent-state
distribution. Probabilities are saved before the current outcome updates it.
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, roots_hermitenorm
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'work/corrected'))
from ordinal_dependence import load_data

def margins(training, pseudocount=.5):
    return np.stack([(np.stack([(g==k).sum(axis=0) for k in range(7)],axis=-1)+pseudocount)/(len(g)+7*pseudocount) for g in training])

def normal_interval(lo,hi):
    return np.maximum(0.,np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo)))

def forecast(y,probs,eta,phi,nb=61,nu=32):
    hb,hw=roots_hermitenorm(nb)
    b=hb*np.sqrt(eta); bw=hw/np.sqrt(2*np.pi)
    lx,lw=leggauss(nu)
    sd=np.sqrt(1-eta); innovation=sd*np.sqrt(1-phi**2)
    edges=np.column_stack([np.full(21,-np.inf),ndtri(np.cumsum(probs,axis=1)[:,:6]),np.full(21,np.inf)])
    out=[]; update_errors=[]; joint=None; old_u=None
    for t in range(21):
        if t==0:
            q=probs[t].copy()
        else:
            mu=b[:,None]+phi*old_u
            lo=(edges[t,:7][None,None,:]-mu[:,:,None])/innovation
            hi=(edges[t,1:][None,None,:]-mu[:,:,None])/innovation
            q=np.sum(joint[:,:,None]*normal_interval(lo,hi),axis=(0,1))
            out.append(q)
        # Update only after saving the forecast. Truncation is solely numerical;
        # +/-10 stationary residual SD loses less than 1.6e-23 prior mass.
        low=np.maximum(edges[t,y[t]]-b,-10*sd)
        high=np.minimum(edges[t,y[t]+1]-b,10*sd)
        width=np.maximum(0.,high-low)
        u=low[:,None]+width[:,None]*(lx[None,:]+1)/2
        iw=width[:,None]*lw[None,:]/2
        if t==0:
            new=bw[:,None]*np.exp(-.5*(u/sd)**2)/(np.sqrt(2*np.pi)*sd)*iw
        else:
            kernel=np.exp(-.5*((u[:,:,None]-phi*old_u[:,None,:])/innovation)**2)/(np.sqrt(2*np.pi)*innovation)
            new=np.einsum('bij,bj->bi',kernel,joint,optimize=False)*iw
        mass=new.sum()
        update_errors.append(abs(mass-q[y[t]]))
        assert mass>0 and np.isfinite(mass)
        joint=new/mass;old_u=u
    return np.asarray(out),max(update_errors)

def run(nb,nu,output):
    tic=time.monotonic()
    data=load_data(ROOT/'upload/Corrected_Clinical_Trial_Long_Data.csv')
    frame=pd.read_csv(ROOT/'upload/Corrected_Clinical_Trial_Long_Data.csv')
    fits=pd.read_csv(ROOT/'work/corrected/dependence/copula_lopo_fits.csv').set_index('ID')
    ids=[sorted(frame.loc[frame.CaseSeries==g,'ID'].unique()) for g in ['A','B']]
    rows=[];errors=[]
    for g in range(2):
        for i in range(34):
            ident=ids[g][i]; fit=fits.loc[ident]
            training=[data[h] if h!=g else np.delete(data[h],i,axis=0) for h in range(2)]
            probs=margins(training)[g];y=data[g,i]
            for label,eta,phi in [('static_full_history',fit.omega_static,0.),('dynamic_full_history',fit.omega_dynamic,fit.phi_dynamic)]:
                pred,err=forecast(y,probs,eta,phi,nb,nu);errors.append(err)
                for tt,q in enumerate(pred,start=1):
                    assert np.min(q)>=0 and abs(q.sum()-1)<1e-10
                    target=y[tt]
                    rows.append({'ID':ident,'CaseSeries':['A','B'][g],'Trial':tt+1,'observed_category':target+1,'model':label,'log_loss':-np.log(q[target]),'mean_normalized_RPS':np.mean((np.cumsum(q)[:6]-(target<=np.arange(6)))**2),**{f'p{k+1}':float(q[k]) for k in range(7)}})
            if (len(rows)//40)%10==0:print('PEOPLE',len(rows)//40,'seconds',round(time.monotonic()-tic,1),flush=True)
    pred=pd.DataFrame(rows); output.mkdir(parents=True,exist_ok=True)
    pred.to_csv(output/f'full_history_{nb}_{nu}.csv',index=False)
    # At trial2 the strict prefix contains one rating, so independent bivariate
    # category rectangles must agree with this forward integration.
    saved=pd.read_csv(ROOT/'work/corrected/dependence/copula_lopo_predictions.csv')
    first=[]
    for long,short in [('static_full_history','static_one_lag'),('dynamic_full_history','dynamic_one_lag')]:
        a=pred[(pred.Trial==2)&(pred.model==long)].sort_values('ID')[[f'p{k}' for k in range(1,8)]].to_numpy()
        z=saved[(saved.Trial==2)&(saved.model==short)].sort_values('ID')[[f'p{k}' for k in range(1,8)]].to_numpy()
        first.append({'model':long,'max_probability_error':float(abs(a-z).max())})
    res={'person_nodes':nb,'residual_nodes':nu,'maximum_filter_update_mass_error':float(max(errors)),'first_lag_comparison':first,'score_means':pred.groupby('model')[['log_loss','mean_normalized_RPS']].mean().reset_index().to_dict('records'),'elapsed_seconds':time.monotonic()-tic}
    (output/f'full_history_{nb}_{nu}.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2),flush=True)
    return pred

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--person-nodes',type=int,default=481);p.add_argument('--residual-nodes',type=int,default=64);p.add_argument('--output',default=str(ROOT/'work/peer_review/analysis'))
    a=p.parse_args();run(a.person_nodes,a.residual_nodes,Path(a.output))
