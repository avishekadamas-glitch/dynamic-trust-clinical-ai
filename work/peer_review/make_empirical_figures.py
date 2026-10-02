from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B=Path(__file__).resolve().parents[2];W=B/'work/peer_review'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'dejavusans'})
checks=pd.read_csv(W/'analysis/model_checks.csv').set_index('summary')
contrasts=pd.read_csv(W/'analysis/all_paired_score_contrasts.csv')
fig,axes=plt.subplots(1,2,figsize=(8.1,3.6),layout='constrained',gridspec_kw={'width_ratios':[1.05,1.2]})
ax=axes[0]
for y,key in zip([2,1,0],['adjacent_equal','adjacent_equal_A','adjacent_equal_B']):
 r=checks.loc[key];mu=r.model_mean*100;lo=r.model_range_lower*100;hi=r.model_range_upper*100
 ax.errorbar(mu,y-.075,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#285B84',capsize=4,ms=5)
 ax.scatter(r.observed*100,y+.075,marker='D',s=26,color='#172B3A',zorder=4)
ax.set_yticks([2,1,0],['All 68','Series A','Series B']);ax.set(xlim=(18,58),ylim=(-.55,2.5),xlabel='Identical adjacent ratings (%)')
ax.set_title('A  Observed and modeled repetition',loc='left',fontweight='bold',fontsize=10)
ax.legend(handles=[Line2D([],[],color='#172B3A',marker='D',linestyle='none',label='Observed'),Line2D([],[],color='#285B84',marker='o',label='Model mean and 95% range')],frameon=False,fontsize=7.8,loc='upper left',bbox_to_anchor=(0,-.20))
ax=axes[1]
for y,mode in zip([1,0],['one_lag','full_history']):
 r=contrasts.loc[(contrasts.metric=='log_loss')&(contrasts.comparator=='static_'+mode)&(contrasts.reference=='dynamic_'+mode)].iloc[0]
 lo,hi=ast.literal_eval(r.interval95);mu=r.mean_difference
 ax.errorbar(mu,y,xerr=[[mu-lo],[hi-mu]],fmt='o',color='#247A73',capsize=4,ms=5)
 ax.text(.064,y,f'{mu:.3f}',va='center',fontsize=8.2)
ax.axvline(0,color='#8B9399',ls='--',lw=1)
ax.set_yticks([1,0],['One previous\nrating','Full preceding\nhistory']);ax.set(xlim=(-.055,.080),ylim=(-.55,1.5),xlabel='Log-loss improvement (static − dynamic)')
ax.set_title('B  Matched forecast comparisons',loc='left',fontweight='bold',fontsize=10)
ax.set_xticks([-.04,0,.04,.08]);ax.text(.5,-.28,'Points and conditional 95% paired intervals',ha='center',transform=ax.transAxes,fontsize=7.8,color='#61707B')
for ax in axes:ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
for ext in ['png','svg','pdf']:fig.savefig(W/f'figure3.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
plt.close(fig)
sim=pd.read_csv(B/'work/corrected/dependence/recovery_replicates.csv')
fig,axes=plt.subplots(1,2,figsize=(8.1,3.1),layout='constrained')
for j,key in enumerate(['omega','phi']):
 ax=axes[j]
 for i,ph in enumerate([0,.4,.85]):
  vals=sim.loc[sim.true_phi.eq(ph),key].values;lo,hi=np.quantile(vals,[.025,.975]);mu=vals.mean();truth=.3 if key=='omega' else ph
  ax.errorbar(i+.045,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',color='#285B84',capsize=4,ms=5,label='Estimate mean and range' if i==0 else None)
  ax.scatter(i-.045,truth,marker='x',color='#9D511C',s=42,label='Generating value' if i==0 else None,zorder=3)
 ax.set_xticks([0,1,2],['0','0.40','0.85']);ax.set_xlabel(r'Generating persistence $\phi$');ax.set_ylim(-.035,1);ax.set_ylabel(r'Estimated stable fraction $\eta$' if key=='omega' else r'Estimated persistence $\phi$');ax.grid(axis='y',alpha=.15)
 ax.set_title('A  Stable component' if j==0 else 'B  Serial persistence',loc='left',fontweight='bold')
axes[0].legend(frameon=False,fontsize=7.5)
for ext in ['png','svg','pdf']:fig.savefig(W/f'figure_s2_1.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
plt.close(fig)
