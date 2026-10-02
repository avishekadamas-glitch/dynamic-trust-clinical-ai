from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.special import ndtr,ndtri
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'work/peer_review/analysis'));sys.path.insert(0,str(R/'work/corrected'))
from full_history import forecast,margins
from ordinal_dependence import load_data
D=load_data(R/'upload/Corrected_Clinical_Trial_Long_Data.csv');df=pd.read_csv(R/'upload/Corrected_Clinical_Trial_Long_Data.csv')
fits=pd.read_csv(R/'work/corrected/dependence/copula_lopo_fits.csv').set_index('ID')
ids=[sorted(df.loc[df.CaseSeries==g,'ID'].unique()) for g in ['A','B']]
L=pd.read_csv(R/'work/peer_review/analysis/full_history_61_32.csv');H=pd.read_csv(R/'work/peer_review/analysis/full_history_121_64.csv')
qcols=[f'p{k}' for k in range(1,8)]
maxdiff=abs(L[qcols].to_numpy()-H[qcols].to_numpy()).max(axis=1)
worst=L.assign(delta=maxdiff).sort_values('delta',ascending=False).ID.drop_duplicates().head(3).tolist()
selected=list(dict.fromkeys(worst+[ids[0][0],ids[0][-1],ids[1][0],ids[1][-1]]))

def interval(low,high):
 return np.maximum(1e-300,np.where(low>0,ndtr(-low)-ndtr(-high),ndtr(high)-ndtr(low)))

def static_exact(y,P,eta):
 edges=np.column_stack([np.full(21,-np.inf),ndtri(np.cumsum(P,axis=1)[:,:6]),np.full(21,np.inf)])
 sd=np.sqrt(1-eta);root=np.sqrt(eta);out=[];errs=[]
 for t in range(1,21):
  lo=edges[np.arange(t),y[:t]];hi=edges[np.arange(t),y[:t]+1]
  def lp(z):
   return -.5*z*z+np.log(interval((lo-root*z)/sd,(hi-root*z)/sd)).sum()
  mode=minimize_scalar(lambda z:-lp(z),bounds=(-12,12),method='bounded').x
  shift=lp(mode)
  def integrand(z):
   weight=np.exp(lp(z)-shift)
   current=interval((edges[t,:7]-root*z)/sd,(edges[t,1:]-root*z)/sd)
   return weight*np.r_[current,1.]
  val,err=quad_vec(integrand,-12.,12.,epsabs=1e-11,epsrel=1e-11,limit=300)
  out.append(val[:7]/val[7]);errs.append(err/val[7])
 return np.array(out),max(errs)

report={'implementation_sha256':hashlib.sha256((R/'work/peer_review/analysis/full_history.py').read_bytes()).hexdigest(),'selected_ids':[int(v) for v in selected],'static_adaptive_comparisons':[],'dynamic_rectangle_comparisons':[],'strict_prefix_checks':[]}
for ident in selected:
 group=0 if ident in ids[0] else 1;person=ids[group].index(ident)
 train=[D[g] if g!=group else np.delete(D[g],person,axis=0) for g in range(2)]
 P=margins(train)[group];y=D[group,person];fit=fits.loc[ident]
 exact,err=static_exact(y,P,fit.omega_static)
 high,diagnostic=forecast(y,P,fit.omega_static,0.,241,96)
 saved=H[(H.ID==ident)&(H.model=='static_full_history')].sort_values('Trial')[qcols].to_numpy()
 report['static_adaptive_comparisons'].append({'ID':int(ident),'quad_reported_relative_error':err,'max_error_241_96':float(abs(exact-high).max()),'max_error_121_64':float(abs(exact-saved).max())})
 # Full-prefix perturbation: alteration begins at target index 8 (trial9),
 # leaving the preceding observed sequence untouched. Earlier predictions must be equal.
 original,_=forecast(y,P,fit.omega_dynamic,fit.phi_dynamic,121,64)
 changed=y.copy();changed[8:]=(changed[8:]+1)%7
 other,_=forecast(changed,P,fit.omega_dynamic,fit.phi_dynamic,121,64)
 report['strict_prefix_checks'].append({'ID':int(ident),'targets_through_trial_9_max_difference':float(abs(original[:8]-other[:8]).max())})
 if ident in selected[:3]:
  for t in [2,3]:
   # An independent multivariate normal rectangle represents the joint prefix
   # and target directly, without a recursive latent-state filter.
   n=t+1; ix=np.arange(n);cov=fit.omega_dynamic+(1-fit.omega_dynamic)*fit.phi_dynamic**abs(ix[:,None]-ix[None,:])
   edges=np.column_stack([np.full(21,-np.inf),ndtri(np.cumsum(P,axis=1)[:,:6]),np.full(21,np.inf)])
   vals=[]
   for cat in range(7):
    cats=np.r_[y[:t],cat];lo=edges[np.arange(n),cats];hi=edges[np.arange(n),cats+1]
    vals.append(multivariate_normal.cdf(hi,mean=np.zeros(n),cov=cov,lower_limit=lo,maxpts=1000000,abseps=1e-9,releps=1e-9,rng=np.random.default_rng(202609060+int(ident)*100+t*10+cat)))
   q=np.array(vals)/sum(vals)
   report['dynamic_rectangle_comparisons'].append({'ID':int(ident),'Trial':t+1,'prefix_probability':float(sum(vals)),'max_error':float(abs(q-original[t-1]).max())})
 print('verified person',ident,flush=True)
(R/'work/peer_review/full_history_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
