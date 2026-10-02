#!/usr/bin/env python3
"""Restricted Gaussian-copula dependence feasibility analysis.

Input: corrected ordinal direct-trust ratings, one complete 21-trial series per person.
Two-stage case-specific ordinal margins use fixed Jeffreys smoothing 0.5/category.
Latent unit-variance normal correlation is omega+(1-omega)*phi**lag.
All 210 trial pairs contribute a composite likelihood per participant.
This is an adaptation of established mixed autoregressive ordinal models, not
an estimate of subjective capability probabilities or a feedback-learning gain.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

SEED = 20260905
PAIR_T, PAIR_S = np.triu_indices(21, 1)
LAGS = np.tile(PAIR_S-PAIR_T, 2).astype(float)
TWOPI = 2*np.pi

class PairKernel:
    def __init__(self, data, nodes=128):
        # data[group, subject, trial]; category coding 0..6
        self.data = [np.asarray(group,dtype=int) for group in data]
        self.n_groups = len(self.data)
        self.n_people = np.array([len(group) for group in self.data])
        self.n_trials = self.data[0].shape[1]
        assert self.n_groups == 2 and self.n_trials == 21
        assert all(group.min()>=0 and group.max()<=6 for group in self.data)
        counts = np.stack([np.stack([(group==k).sum(axis=0) for k in range(7)],axis=-1) for group in self.data])
        self.marginal_prob = (counts+.5)/(self.n_people[:,None,None]+3.5)
        self.marginal_cum = np.cumsum(self.marginal_prob, axis=-1)[...,:6]
        self.cutpoints = ndtri(self.marginal_cum)
        ct = self.cutpoints[:,PAIR_T,:].reshape(-1,6)
        cs = self.cutpoints[:,PAIR_S,:].reshape(-1,6)
        self.a = ct[:,:,None]
        self.b = cs[:,None,:]
        self.ab = self.a*self.b
        self.aa_bb = self.a*self.a+self.b*self.b
        self.base = ndtr(self.a)*ndtr(self.b)
        # Boundary values of bivariate CDF: F(infinity,b)=Phi(b).
        self.cdf_base = np.zeros((420,8,8))
        self.cdf_base[:,7,1:7] = ndtr(cs)
        self.cdf_base[:,1:7,7] = ndtr(ct)
        self.cdf_base[:,7,7] = 1.
        self.weights = np.zeros((420,7,7))
        for g in range(2):
            for q,(t,s) in enumerate(zip(PAIR_T,PAIR_S)):
                self.weights[g*210+q] = np.bincount(self.data[g][:,t]*7+self.data[g][:,s],minlength=49).reshape(7,7)
        self.count = self.weights.sum()
        self.rectangle_nodes,self.rectangle_weights = leggauss(128)
        self.rectangle_nodes = (self.rectangle_nodes+1)/2
        self.rectangle_weights = self.rectangle_weights/2
        self.nodes,self.node_weights = leggauss(nodes)
        self.quad_nodes = (self.nodes+1)/2
        self.quad_weights = self.node_weights/2
        self.calls = 0

    def evaluate(self, omega, phi, gradients=True):
        self.calls += 1
        rho = omega+(1-omega)*phi**LAGS
        # Plackett identity: Phi_2(a,b;rho)=Phi(a)Phi(b)+integral_0^rho density_2(a,b;r)dr.
        r = rho[:,None,None,None]*self.quad_nodes[None,None,None,:]
        den = 1-r*r
        density = np.exp(-(self.aa_bb[:,:,:,None]-2*r*self.ab[:,:,:,None])/(2*den))/(TWOPI*np.sqrt(den))
        integral = rho[:,None,None]*np.sum(density*self.quad_weights,axis=-1)
        cdf = self.cdf_base.copy()
        cdf[:,1:7,1:7] = self.base+integral
        prob = np.diff(np.diff(cdf,axis=1),axis=2)
        # Tiny rectangle differences suffer cancellation even when CDF error is tiny.
        # Recompute them as a positive one-dimensional conditional-normal integral.
        ii,ka,kb=np.where(prob<1e-9)
        if len(ii):
            cum_a=np.column_stack([np.zeros(420),ndtr(self.a[:,:,0]),np.ones(420)])
            edges_b=np.column_stack([np.full(420,-np.inf),self.b[:,0,:],np.full(420,np.inf)])
            ulo,uhi=cum_a[ii,ka],cum_a[ii,ka+1]
            xx=ndtri(ulo[:,None]+(uhi-ulo)[:,None]*self.rectangle_nodes)
            sr=np.sqrt(1-rho[ii]**2)[:,None]
            lower=(edges_b[ii,kb,None]-rho[ii,None]*xx)/sr
            upper=(edges_b[ii,kb+1,None]-rho[ii,None]*xx)/sr
            cond=np.where(lower>0,ndtr(-lower)-ndtr(-upper),ndtr(upper)-ndtr(lower))
            prob[ii,ka,kb]=(uhi-ulo)*np.sum(cond*self.rectangle_weights,axis=1)
        p_safe = np.maximum(prob,1e-14)
        objective = -np.sum(self.weights*np.log(p_safe))/self.count
        if not gradients:
            return objective, prob
        d_cdf = np.zeros_like(cdf)
        den_r = 1-rho[:,None,None]**2
        d_cdf[:,1:7,1:7] = np.exp(-(self.aa_bb-2*rho[:,None,None]*self.ab)/(2*den_r))/(TWOPI*np.sqrt(den_r))
        dp_drho = np.diff(np.diff(d_cdf,axis=1),axis=2)
        dloss_drho = -np.sum(self.weights*dp_drho/p_safe*(prob>1e-14),axis=(1,2))/self.count
        d_rho_omega = 1-phi**LAGS
        d_rho_phi = (1-omega)*LAGS*phi**(LAGS-1)
        gradient = np.array([np.sum(dloss_drho*d_rho_omega),np.sum(dloss_drho*d_rho_phi)])
        return objective, gradient

    def fit(self, starts=((.3,.3),), static=False):
        records=[]
        if static:
            for start in starts:
                def fun(x):
                    loss,grad=self.evaluate(float(x[0]),0.)
                    return loss,grad[:1]
                res=minimize(fun,[float(start[0])],jac=True,method='L-BFGS-B',bounds=[(0.,.95)],options={'maxiter':100,'ftol':1e-11,'gtol':1e-7,'maxls':30})
                records.append(res)
            best=min(records,key=lambda r:r.fun)
            omega,phi=float(best.x[0]),0.
        else:
            for start in starts:
                res=minimize(lambda x:self.evaluate(float(x[0]),float(x[1])),start,jac=True,method='L-BFGS-B',bounds=[(0.,.95),(0.,.95)],options={'maxiter':100,'ftol':1e-11,'gtol':1e-7,'maxls':30})
                records.append(res)
            best=min(records,key=lambda r:r.fun)
            omega,phi=map(float,best.x)
        loss,prob=self.evaluate(omega,phi,gradients=False)
        return {'omega':omega,'phi':phi,'mean_pair_negative_loglikelihood':float(loss),'composite_loglikelihood':float(-loss*self.count),'optimizer_success':bool(best.success),'message':str(best.message),'iterations':int(best.nit),'function_calls':int(best.nfev),'omega_lower_boundary':bool(omega<1e-5),'omega_upper_boundary':bool(omega>.95-1e-5),'phi_lower_boundary':bool(phi<1e-5),'phi_upper_boundary':bool(phi>.95-1e-5),'minimum_pair_probability':float(prob.min()),'pair_probability_sum_max_error':float(np.max(np.abs(prob.sum(axis=(1,2))-1)))}

def kernel_check():
    nodes,weights=leggauss(128)
    def cdf(a,b,rho):
        r=rho*(nodes+1)/2
        den=1-r*r
        return ndtr(a)*ndtr(b)+rho/2*np.sum(weights*np.exp(-(a*a-2*r*a*b+b*b)/(2*den))/(TWOPI*np.sqrt(den)))
    cases=[(-2,-1,0.),(0,0,0.),(.5,-.2,.2),(-1.5,.7,.65),(0,0,.95),(.2,.2,.9975),(-2,-2,.9975),(2,-2,.9975),(-.5,.5,.9)]
    records=[]
    for k,(a,b,rho) in enumerate(cases):
        q=cdf(a,b,rho)
        reference=float(multivariate_normal.cdf([a,b],mean=[0,0],cov=[[1,rho],[rho,1]],maxpts=3000000,abseps=1e-10,releps=1e-10,rng=np.random.default_rng(SEED+k)))
        exact_zero=(.25+np.arcsin(rho)/(2*np.pi)) if a==0 and b==0 else None
        records.append({'a':a,'b':b,'rho':rho,'quadrature':float(q),'scipy_cdf':reference,'absolute_difference':float(abs(q-reference)),'exact_zero_threshold_cdf':exact_zero})
    maxdiff=max(r['absolute_difference'] for r in records)
    if maxdiff > 2e-7:
        raise RuntimeError(f'Quadrature check failed: max CDF difference {maxdiff}')
    return {'nodes':128,'tail_rectangle_fallback':'128-node conditional-normal probability integral for rectangles below 1e-9','maximum_absolute_difference_vs_scipy':maxdiff,'cases':records}

def load_data(path):
    d=pd.read_csv(path).sort_values(['CaseSeries','ID','Trial'])
    arrays=[]
    for group,g in d.groupby('CaseSeries',sort=True):
        p=g.pivot(index='ID',columns='Trial',values='direct_trust_item')
        assert p.shape == (34,21) and not p.isna().any().any()
        assert list(p.columns)==list(range(1,22))
        arrays.append(p.to_numpy().astype(int)-1)
    return np.stack(arrays)

def simulate(rng,cutpoints,omega,phi):
    b=rng.normal(size=(2,34,1))*np.sqrt(omega)
    u=np.empty((2,34,21));u[:,:,0]=rng.normal(size=(2,34))*np.sqrt(1-omega)
    innovation=np.sqrt((1-omega)*(1-phi*phi))
    for t in range(1,21):
        u[:,:,t]=phi*u[:,:,t-1]+innovation*rng.normal(size=(2,34))
    latent=b+u
    return (latent[:,:,:,None]>cutpoints[:,None,:,:]).sum(axis=-1)

def summarized_recovery(frame):
    out=[]
    for phi,g in frame.groupby('true_phi',sort=True):
        r={'true_omega':.3,'true_phi':float(phi),'replicates':int(len(g)),'optimizer_success_fraction':float(g.optimizer_success.mean())}
        for name,truth in [('omega',.3),('phi',float(phi))]:
            values=g[name].to_numpy()
            r.update({f'{name}_mean':float(values.mean()),f'{name}_median':float(np.median(values)),f'{name}_bias':float(np.mean(values-truth)),f'{name}_rmse':float(np.sqrt(np.mean((values-truth)**2))),f'{name}_sd':float(np.std(values,ddof=1)),f'{name}_mean_monte_carlo_se':float(np.std(values,ddof=1)/np.sqrt(len(values))),f'{name}_q025':float(np.quantile(values,.025)),f'{name}_q975':float(np.quantile(values,.975)),f'{name}_lower_boundary_fraction':float(g[f'{name}_lower_boundary'].mean()),f'{name}_upper_boundary_fraction':float(g[f'{name}_upper_boundary'].mean())})
        out.append(r)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='upload/Corrected_Clinical_Trial_Long_Data.csv')
    p.add_argument('--output',default='work/corrected/dependence')
    p.add_argument('--bootstraps',type=int,default=200)
    p.add_argument('--simulation-replicates',type=int,default=100)
    p.add_argument('--seed',type=int,default=SEED)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();rng=np.random.default_rng(args.seed)
    check=kernel_check();print('KERNEL',json.dumps(check),flush=True)
    data=load_data(args.input);kernel=PairKernel(data)
    # Verify the analytic objective gradient with a central finite difference.
    loc=np.array([.3,.4]);f,g=kernel.evaluate(*loc);h=1e-5
    finite=np.array([(kernel.evaluate(*(loc+np.eye(2)[j]*h))[0]-kernel.evaluate(*(loc-np.eye(2)[j]*h))[0])/(2*h) for j in range(2)])
    grad_error=float(np.max(np.abs(g-finite)))
    assert grad_error<1e-5,(g,finite)
    dynamic=kernel.fit(starts=((.3,.3),(.6,.1),(.1,.7),(0.,.9)))
    static=kernel.fit(starts=((.3,0.),),static=True)
    # Dense grid independently checks the fitted objective is no worse than sampled alternatives.
    grid_best=(np.inf,None)
    for omega in np.linspace(0,.9,7):
        for phi in np.linspace(0,.9,7):
            f=kernel.evaluate(omega,phi)[0]
            if f<grid_best[0]:grid_best=(float(f),[float(omega),float(phi)])
    summary={'model':'two-stage case-margin Gaussian copula; latent Corr(h)=omega+(1-omega)*phi^h','participants':68,'participants_per_series':34,'trials':21,'categories':7,'pairs_per_participant':210,'pair_count':int(kernel.count),'case_margin_smoothing_per_category':.5,'parameter_bounds':[0.,.95],'seed':args.seed,'quadrature_check':check,'analytic_gradient_max_absolute_error':grad_error,'empirical_dynamic':dynamic,'empirical_static_person':static,'grid_check':{'best_grid_loss':grid_best[0],'best_grid_parameters':grid_best[1],'optimizer_no_worse':bool(dynamic['mean_pair_negative_loglikelihood']<=grid_best[0]+1e-8)},'warnings':['Pairwise composite objective is not full likelihood; no ordinary likelihood-ratio p value or AIC is reported.','Case-specific margins are estimated, and re-estimated in each bootstrap and simulation replicate.','Latent copula propensity is scale-anchored by standard-normal case margins, not a subjective capability probability.','Previous correctness is structurally confounded with unrestricted current case effects; no feedback gain is estimated.','Boundary percentile intervals are descriptive and their nominal coverage is not established.','Simulation recovery assesses this generating model, with known observed-design cutpoints; it does not validate the broader scientific theory.']}
    (out/'dependence_results.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'case_margin_estimates.npz',category_probabilities=kernel.marginal_prob,cutpoints=kernel.cutpoints)
    print('EMPIRICAL',json.dumps({'dynamic':dynamic,'static':static,'elapsed_seconds':time.monotonic()-start}),flush=True)
    rows=[]
    for rep in range(args.bootstraps):
        sample=np.stack([data[group,rng.integers(0,34,34)] for group in range(2)])
        res=PairKernel(sample).fit(starts=((dynamic['omega'],dynamic['phi']),))
        rows.append({'replicate':rep+1,**res})
        if (rep+1)%20==0:
            pd.DataFrame(rows).to_csv(out/'bootstrap_replicates.csv',index=False)
            print('BOOTSTRAP',rep+1,'seconds',round(time.monotonic()-start,1),flush=True)
    bframe=pd.DataFrame(rows);bframe.to_csv(out/'bootstrap_replicates.csv',index=False)
    if len(bframe):
        summary['participant_bootstrap']={'replicates':len(bframe),'stratified_by':'CaseSeries','margin_refit_each_replicate':True,'optimizer_success_fraction':float(bframe.optimizer_success.mean()),'omega_percentile_95_interval':[float(x) for x in bframe.omega.quantile([.025,.975])],'phi_percentile_95_interval':[float(x) for x in bframe.phi.quantile([.025,.975])],'omega_lower_boundary_fraction':float(bframe.omega_lower_boundary.mean()),'omega_upper_boundary_fraction':float(bframe.omega_upper_boundary.mean()),'phi_lower_boundary_fraction':float(bframe.phi_lower_boundary.mean()),'phi_upper_boundary_fraction':float(bframe.phi_upper_boundary.mean())}
        (out/'dependence_results.json').write_text(json.dumps(summary,indent=2))
    simrows=[]
    for true_phi in [0.,.4,.85]:
        for rep in range(args.simulation_replicates):
            generated=simulate(rng,kernel.cutpoints,.3,true_phi)
            res=PairKernel(generated).fit(starts=((.3,.3),))
            simrows.append({'true_omega':.3,'true_phi':true_phi,'replicate':rep+1,**res})
            if (rep+1)%20==0:
                pd.DataFrame(simrows).to_csv(out/'recovery_replicates.csv',index=False)
                print('RECOVERY phi',true_phi,'rep',rep+1,'seconds',round(time.monotonic()-start,1),flush=True)
    sframe=pd.DataFrame(simrows);sframe.to_csv(out/'recovery_replicates.csv',index=False)
    summary['parameter_recovery']=summarized_recovery(sframe) if len(sframe) else []
    summary['elapsed_seconds']=time.monotonic()-start
    (out/'dependence_results.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',json.dumps({'bootstrap':summary.get('participant_bootstrap'),'recovery':summary['parameter_recovery'],'elapsed_seconds':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
